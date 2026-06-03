//! Windows ML runtime bootstrap for ORT-backed inference.
//!
//! Windows App SDK Runtime 2.x carries the shared ONNX Runtime and DirectML
//! binaries. We keep this initialized for the process lifetime so later ORT
//! sessions can continue resolving package-graph DLLs; do not call shutdown from
//! a model/session drop path or from realtime audio code.

use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use std::sync::OnceLock;

use anyhow::{anyhow, bail, Context, Result};
use ort::environment::Environment;
use tracing::{info, warn};

const COINIT_MULTITHREADED: u32 = 0;
const RPC_E_CHANGED_MODE: i32 = 0x80010106u32 as i32;
const WINDOWS_APP_SDK_2_1: u32 = 0x0002_0001;
const WINDOWS_ML_BOOTSTRAP_ENV: &str = "VC_RS_WINDOWSML_BOOTSTRAP_DLL";

type Hmodule = *mut c_void;
type Farproc = *mut c_void;
type Hresult = i32;

type MddBootstrapInitialize2 = unsafe extern "system" fn(u32, *const u16, u64, u32) -> Hresult;
type WinMLEpCatalogHandle = *mut c_void;
type WinMLEpHandle = *mut c_void;
type WinMLEpCatalogCreate = unsafe extern "system" fn(*mut WinMLEpCatalogHandle) -> Hresult;
type WinMLEpCatalogRelease = unsafe extern "system" fn(WinMLEpCatalogHandle);
type WinMLEpCatalogFindProvider = unsafe extern "system" fn(
    WinMLEpCatalogHandle,
    *const c_char,
    *const c_char,
    *mut WinMLEpHandle,
) -> Hresult;
type WinMLEpGetReadyState = unsafe extern "system" fn(WinMLEpHandle, *mut i32) -> Hresult;
type WinMLEpEnsureReady = unsafe extern "system" fn(WinMLEpHandle) -> Hresult;
type WinMLEpGetLibraryPathSize = unsafe extern "system" fn(WinMLEpHandle, *mut usize) -> Hresult;
type WinMLEpGetLibraryPath =
    unsafe extern "system" fn(WinMLEpHandle, usize, *mut c_char, *mut usize) -> Hresult;

#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryW(name: *const u16) -> Hmodule;
    fn GetProcAddress(module: Hmodule, name: *const c_char) -> Farproc;
}

#[link(name = "ole32")]
extern "system" {
    fn CoInitializeEx(reserved: *mut c_void, coinit: u32) -> Hresult;
    fn CoUninitialize();
}

struct BootstrapState {
    _module: Hmodule,
}

unsafe impl Send for BootstrapState {}
unsafe impl Sync for BootstrapState {}

static WINDOWS_ML_BOOTSTRAP: OnceLock<Result<BootstrapState, String>> = OnceLock::new();
static CATALOG_EP: OnceLock<Result<Option<CatalogExecutionProvider>, String>> = OnceLock::new();

#[derive(Clone, Copy, Debug)]
pub(crate) enum CatalogExecutionProvider {
    NvTensorRtRtx,
    Qnn,
    OpenVino,
    MiGraphX,
    VitisAi,
}

pub(crate) fn ensure_initialized() -> Result<()> {
    let state = WINDOWS_ML_BOOTSTRAP.get_or_init(initialize);
    state
        .as_ref()
        .map(|_| ())
        .map_err(|err| anyhow!(err.clone()))
}

pub(crate) fn try_register_best_catalog_ep() -> Result<Option<CatalogExecutionProvider>> {
    ensure_initialized()?;
    let ep = CATALOG_EP.get_or_init(register_best_catalog_ep);
    ep.as_ref().copied().map_err(|err| anyhow!(err.clone()))
}

fn initialize() -> Result<BootstrapState, String> {
    initialize_inner().map_err(|err| format!("{err:#}"))
}

fn initialize_inner() -> Result<BootstrapState> {
    let com = ComInit::new()?;
    let bootstrap_path = std::env::var(WINDOWS_ML_BOOTSTRAP_ENV)
        .unwrap_or_else(|_| "Microsoft.WindowsAppRuntime.Bootstrap.dll".to_string());
    let wide_path = wide(&bootstrap_path);
    let module = unsafe { LoadLibraryW(wide_path.as_ptr()) };
    if module.is_null() {
        bail!(
            "failed to load {bootstrap_path}; install Windows App SDK Runtime 2.x and place Microsoft.WindowsAppRuntime.Bootstrap.dll beside the executable/plugin, or set {WINDOWS_ML_BOOTSTRAP_ENV}"
        );
    }

    let initialize2: MddBootstrapInitialize2 =
        unsafe { load_symbol(module, b"MddBootstrapInitialize2\0")? };
    check_hr(
        unsafe { initialize2(WINDOWS_APP_SDK_2_1, ptr::null(), 0, 0) },
        "MddBootstrapInitialize2(Windows App SDK Runtime 2.1)",
    )
    .context(
        "failed to initialize Windows App SDK Runtime 2.1; install WindowsAppRuntime.2 2.1 or newer",
    )?;

    let committed = ort::init_from("onnxruntime.dll")
        .context("failed to load Windows ML ONNX Runtime from Windows App SDK Runtime")?
        .commit();
    if !committed {
        bail!(
            "ONNX Runtime was initialized before Windows ML; load a windowsml* provider before any non-Windows ML ORT provider in this process"
        );
    }

    drop(com);
    Ok(BootstrapState { _module: module })
}

fn register_best_catalog_ep() -> Result<Option<CatalogExecutionProvider>, String> {
    register_best_catalog_ep_inner().map_err(|err| format!("{err:#}"))
}

fn register_best_catalog_ep_inner() -> Result<Option<CatalogExecutionProvider>> {
    let catalog_api = CatalogApi::load()?;
    let mut catalog = ptr::null_mut();
    check_hr(
        unsafe { (catalog_api.create)(&mut catalog) },
        "WinMLEpCatalogCreate",
    )?;
    let _catalog = ReleaseCatalog {
        handle: catalog,
        release: catalog_api.release,
    };

    for candidate in CATALOG_PRIORITY {
        match try_register_candidate(&catalog_api, catalog, candidate) {
            Ok(true) => {
                info!(
                    "registered Windows ML catalog EP {} from {}",
                    candidate.catalog_name, candidate.registration_name
                );
                return Ok(Some(candidate.provider));
            }
            Ok(false) => {}
            Err(err) => {
                warn!(
                    "failed to register Windows ML catalog EP {}: {err:#}",
                    candidate.catalog_name
                );
            }
        }
    }
    Ok(None)
}

fn try_register_candidate(
    api: &CatalogApi,
    catalog: WinMLEpCatalogHandle,
    candidate: &CatalogCandidate,
) -> Result<bool> {
    let catalog_name = CString::new(candidate.catalog_name)?;
    let mut ep = ptr::null_mut();
    let hr = unsafe { (api.find_provider)(catalog, catalog_name.as_ptr(), ptr::null(), &mut ep) };
    if hr < 0 || ep.is_null() {
        return Ok(false);
    }

    let mut state = 0;
    check_hr(
        unsafe { (api.get_ready_state)(ep, &mut state) },
        "WinMLEpGetReadyState",
    )?;
    if state == 2 {
        return Ok(false);
    }
    check_hr(unsafe { (api.ensure_ready)(ep) }, "WinMLEpEnsureReady")?;

    let path = read_ep_library_path(api, ep)?;
    if path.is_empty() {
        return Ok(false);
    }

    let env = Environment::current()?;
    let _library = env.register_ep_library(candidate.registration_name, &path)?;
    Ok(true)
}

fn read_ep_library_path(api: &CatalogApi, ep: WinMLEpHandle) -> Result<String> {
    let mut len = 0usize;
    check_hr(
        unsafe { (api.get_library_path_size)(ep, &mut len) },
        "WinMLEpGetLibraryPathSize",
    )?;
    if len == 0 {
        return Ok(String::new());
    }
    let mut bytes = vec![0u8; len];
    let mut used = 0usize;
    check_hr(
        unsafe { (api.get_library_path)(ep, len, bytes.as_mut_ptr().cast(), &mut used) },
        "WinMLEpGetLibraryPath",
    )?;
    let path = unsafe { CStr::from_ptr(bytes.as_ptr().cast()) }
        .to_string_lossy()
        .into_owned();
    Ok(path)
}

struct CatalogApi {
    _module: Hmodule,
    create: WinMLEpCatalogCreate,
    release: WinMLEpCatalogRelease,
    find_provider: WinMLEpCatalogFindProvider,
    get_ready_state: WinMLEpGetReadyState,
    ensure_ready: WinMLEpEnsureReady,
    get_library_path_size: WinMLEpGetLibraryPathSize,
    get_library_path: WinMLEpGetLibraryPath,
}

impl CatalogApi {
    fn load() -> Result<Self> {
        let module_name = wide("Microsoft.Windows.AI.MachineLearning.dll");
        let module = unsafe { LoadLibraryW(module_name.as_ptr()) };
        if module.is_null() {
            bail!(
                "failed to load Microsoft.Windows.AI.MachineLearning.dll from Windows App SDK Runtime"
            );
        }
        Ok(Self {
            _module: module,
            create: unsafe { load_symbol(module, b"WinMLEpCatalogCreate\0")? },
            release: unsafe { load_symbol(module, b"WinMLEpCatalogRelease\0")? },
            find_provider: unsafe { load_symbol(module, b"WinMLEpCatalogFindProvider\0")? },
            get_ready_state: unsafe { load_symbol(module, b"WinMLEpGetReadyState\0")? },
            ensure_ready: unsafe { load_symbol(module, b"WinMLEpEnsureReady\0")? },
            get_library_path_size: unsafe { load_symbol(module, b"WinMLEpGetLibraryPathSize\0")? },
            get_library_path: unsafe { load_symbol(module, b"WinMLEpGetLibraryPath\0")? },
        })
    }
}

struct ReleaseCatalog {
    handle: WinMLEpCatalogHandle,
    release: WinMLEpCatalogRelease,
}

impl Drop for ReleaseCatalog {
    fn drop(&mut self) {
        unsafe { (self.release)(self.handle) };
    }
}

struct CatalogCandidate {
    catalog_name: &'static str,
    registration_name: &'static str,
    provider: CatalogExecutionProvider,
}

const CATALOG_PRIORITY: &[CatalogCandidate] = &[
    CatalogCandidate {
        catalog_name: "NvTensorRtRtxExecutionProvider",
        registration_name: "NvTensorRtRtx",
        provider: CatalogExecutionProvider::NvTensorRtRtx,
    },
    // Older catalog builds used this all-caps RTX spelling.
    CatalogCandidate {
        catalog_name: "NvTensorRTRTXExecutionProvider",
        registration_name: "NvTensorRtRtx",
        provider: CatalogExecutionProvider::NvTensorRtRtx,
    },
    CatalogCandidate {
        catalog_name: "QNNExecutionProvider",
        registration_name: "QNN",
        provider: CatalogExecutionProvider::Qnn,
    },
    CatalogCandidate {
        catalog_name: "OpenVINOExecutionProvider",
        registration_name: "OpenVINO",
        provider: CatalogExecutionProvider::OpenVino,
    },
    CatalogCandidate {
        catalog_name: "MIGraphXExecutionProvider",
        registration_name: "MIGraphX",
        provider: CatalogExecutionProvider::MiGraphX,
    },
    CatalogCandidate {
        catalog_name: "VitisAIExecutionProvider",
        registration_name: "VitisAI",
        provider: CatalogExecutionProvider::VitisAi,
    },
];

struct ComInit {
    should_uninitialize: bool,
}

impl ComInit {
    fn new() -> Result<Self> {
        let hr = unsafe { CoInitializeEx(ptr::null_mut(), COINIT_MULTITHREADED) };
        match hr {
            0 | 1 => Ok(Self {
                should_uninitialize: true,
            }),
            RPC_E_CHANGED_MODE => Ok(Self {
                should_uninitialize: false,
            }),
            hr if hr < 0 => bail!("CoInitializeEx failed: HRESULT 0x{:08X}", hr as u32),
            _ => Ok(Self {
                should_uninitialize: false,
            }),
        }
    }
}

impl Drop for ComInit {
    fn drop(&mut self) {
        if self.should_uninitialize {
            unsafe { CoUninitialize() };
        }
    }
}

unsafe fn load_symbol<T>(module: Hmodule, name: &'static [u8]) -> Result<T> {
    let symbol = GetProcAddress(module, name.as_ptr().cast());
    if symbol.is_null() {
        let name = std::str::from_utf8(&name[..name.len() - 1]).unwrap_or("<non-utf8>");
        bail!("GetProcAddress({name}) failed for Windows App SDK bootstrapper");
    }
    Ok(std::mem::transmute_copy(&symbol))
}

fn check_hr(hr: Hresult, what: &str) -> Result<()> {
    if hr < 0 {
        bail!("{what} failed: HRESULT 0x{:08X}", hr as u32);
    }
    Ok(())
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}
