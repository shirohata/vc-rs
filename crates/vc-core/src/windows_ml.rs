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
const ERROR_SUCCESS: i32 = 0;
const ERROR_INSUFFICIENT_BUFFER: i32 = 122;
const APPMODEL_ERROR_NO_PACKAGE: i32 = 15700;
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
type WinMLEpCatalogEnumProviders =
    unsafe extern "system" fn(WinMLEpCatalogHandle, WinMLEpEnumCallback, *mut c_void) -> Hresult;
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
type WinMLEpEnumCallback =
    unsafe extern "system" fn(WinMLEpHandle, *const WinMLEpInfoRaw, *mut c_void) -> i32;

#[repr(C)]
struct WinMLEpInfoRaw {
    name: *const c_char,
    version: *const c_char,
    package_family_name: *const c_char,
    library_path: *const c_char,
    package_root_path: *const c_char,
    ready_state: i32,
    certification: i32,
}

#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryW(name: *const u16) -> Hmodule;
    fn GetProcAddress(module: Hmodule, name: *const c_char) -> Farproc;
    fn GetCurrentPackageFullName(
        package_full_name_length: *mut u32,
        package_full_name: *mut u16,
    ) -> i32;
}

#[link(name = "ole32")]
extern "system" {
    fn CoInitializeEx(reserved: *mut c_void, coinit: u32) -> Hresult;
    fn CoUninitialize();
}

struct BootstrapState {
    _module: Option<Hmodule>,
}

unsafe impl Send for BootstrapState {}
unsafe impl Sync for BootstrapState {}

static WINDOWS_ML_BOOTSTRAP: OnceLock<Result<BootstrapState, String>> = OnceLock::new();
// Cache only *successful* registration. Negative results (NotPresent / errors)
// must never be memoized: an explicit windowsml-* provider can trigger an EP
// download, after which a later load in the same process must be able to
// register the now-Ready EP instead of seeing a stale failure. (A populated cell
// therefore means "already registered" and short-circuits.)
static BEST_CATALOG_EP: OnceLock<CatalogExecutionProvider> = OnceLock::new();
static CATALOG_NV_TENSORRT_RTX_EP: OnceLock<()> = OnceLock::new();
static CATALOG_QNN_EP: OnceLock<()> = OnceLock::new();
static CATALOG_OPENVINO_EP: OnceLock<()> = OnceLock::new();
static CATALOG_MIGRAPHX_EP: OnceLock<()> = OnceLock::new();
static CATALOG_VITISAI_EP: OnceLock<()> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatalogExecutionProvider {
    NvTensorRtRtx,
    Qnn,
    OpenVino,
    MiGraphX,
    VitisAi,
}

impl CatalogExecutionProvider {
    pub fn label(self) -> &'static str {
        match self {
            Self::NvTensorRtRtx => "NvTensorRtRtx",
            Self::Qnn => "QNN",
            Self::OpenVino => "OpenVINO",
            Self::MiGraphX => "MIGraphX",
            Self::VitisAi => "VitisAI",
        }
    }

    /// The `Provider` that selects this catalog EP. One half of the bijection
    /// with [`Provider::catalog_ep`]; every other `CatalogExecutionProvider` ↔
    /// `Provider` correspondence is derived from this pair.
    pub fn vc_provider(self) -> crate::Provider {
        match self {
            Self::NvTensorRtRtx => crate::Provider::WindowsMlNvTensorRtRtx,
            Self::Qnn => crate::Provider::WindowsMlQnn,
            Self::OpenVino => crate::Provider::WindowsMlOpenVino,
            Self::MiGraphX => crate::Provider::WindowsMlMiGraphX,
            Self::VitisAi => crate::Provider::WindowsMlVitisAi,
        }
    }

    pub fn vc_provider_name(self) -> &'static str {
        // Derived from the provider's own label so the two can't drift.
        self.vc_provider().label()
    }

    pub fn from_catalog_name(name: &str) -> Option<Self> {
        CATALOG_PRIORITY
            .iter()
            .find(|candidate| candidate.catalog_name == name)
            .map(|candidate| candidate.provider)
    }
}

impl crate::Provider {
    /// The Windows ML catalog EP this provider selects, if any. The single
    /// `Provider` → `CatalogExecutionProvider` map (inverse of
    /// [`CatalogExecutionProvider::vc_provider`]); `sessions::load_session` and
    /// [`provider_prepare_pending`] both read it here instead of re-deriving it.
    pub fn catalog_ep(self) -> Option<CatalogExecutionProvider> {
        match self {
            crate::Provider::WindowsMlNvTensorRtRtx => {
                Some(CatalogExecutionProvider::NvTensorRtRtx)
            }
            crate::Provider::WindowsMlQnn => Some(CatalogExecutionProvider::Qnn),
            crate::Provider::WindowsMlOpenVino => Some(CatalogExecutionProvider::OpenVino),
            crate::Provider::WindowsMlMiGraphX => Some(CatalogExecutionProvider::MiGraphX),
            crate::Provider::WindowsMlVitisAi => Some(CatalogExecutionProvider::VitisAi),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatalogReadyState {
    Ready,
    NotReady,
    NotPresent,
    Unknown(i32),
}

impl CatalogReadyState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ready => "Ready",
            Self::NotReady => "NotReady",
            Self::NotPresent => "NotPresent",
            Self::Unknown(_) => "Unknown",
        }
    }

    fn from_raw(state: i32) -> Self {
        match state {
            0 => Self::Ready,
            1 => Self::NotReady,
            2 => Self::NotPresent,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CatalogProviderInfo {
    pub name: String,
    pub version: String,
    pub package_family_name: String,
    pub library_path: String,
    pub package_root_path: String,
    pub ready_state: CatalogReadyState,
    pub certification: i32,
    pub vc_provider: Option<CatalogExecutionProvider>,
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
    if let Some(ep) = BEST_CATALOG_EP.get() {
        return Ok(Some(*ep));
    }
    // Some catalog providers report NotReady after every process start until
    // EnsureReady is called. Keep auto windowsml behavior consistent across CLI,
    // GUI, and VST3: try to prepare listed providers, but still fall back when
    // the catalog reports NotPresent or an unknown state.
    let selected = register_best_catalog_ep_inner()?;
    if let Some(ep) = selected {
        let _ = BEST_CATALOG_EP.set(ep);
        mark_catalog_ep_registered(ep);
    }
    Ok(selected)
}

pub(crate) fn try_register_catalog_ep(provider: CatalogExecutionProvider) -> Result<bool> {
    ensure_initialized()?;
    if catalog_ep_cache(provider).get().is_some() {
        return Ok(true);
    }
    // Explicit provider: download/prepare the EP when the catalog lists it but it
    // is not yet Ready. Only a successful registration is cached.
    let registered = register_catalog_ep_inner(provider)?;
    if registered {
        mark_catalog_ep_registered(provider);
    }
    Ok(registered)
}

/// The catalog EPs this device's Windows ML catalog lists, as `Provider`s, in
/// vc-rs priority order — including not-yet-installed (`NotPresent`) ones, since
/// selecting them triggers a download on load. Enumerated once and cached for
/// the process, so front-end pickers can call it freely; enumeration failure
/// (no Windows ML runtime) yields an empty list and the picker shows only the
/// build's base providers.
pub fn available_catalog_providers() -> &'static [crate::Provider] {
    static CACHE: OnceLock<Vec<crate::Provider>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let listed = list_catalog_providers().unwrap_or_default();
        CATALOG_PRIORITY
            .iter()
            .map(|candidate| candidate.provider)
            .filter(|ep| listed.iter().any(|info| info.vc_provider == Some(*ep)))
            .map(CatalogExecutionProvider::vc_provider)
            // CATALOG_PRIORITY lists TensorRT-RTX twice (two catalog spellings);
            // collapse to one entry per provider while keeping priority order.
            .fold(Vec::new(), |mut acc, provider| {
                if !acc.contains(&provider) {
                    acc.push(provider);
                }
                acc
            })
    })
}

pub fn list_catalog_providers() -> Result<Vec<CatalogProviderInfo>> {
    ensure_initialized()?;
    with_catalog(|catalog_api, catalog| {
        let mut providers = Vec::<CatalogProviderInfo>::new();
        check_hr(
            unsafe {
                (catalog_api.enum_providers)(
                    catalog,
                    enum_provider_info,
                    (&mut providers as *mut Vec<CatalogProviderInfo>).cast(),
                )
            },
            "WinMLEpCatalogEnumProviders",
        )?;
        Ok(providers)
    })
}

pub fn select_best_catalog_provider(
    providers: &[CatalogProviderInfo],
) -> Option<CatalogExecutionProvider> {
    select_best_catalog_provider_info(providers).and_then(|provider| provider.vc_provider)
}

pub fn select_best_catalog_provider_info(
    providers: &[CatalogProviderInfo],
) -> Option<&CatalogProviderInfo> {
    select_catalog_provider_info_by_candidates(providers, CATALOG_PRIORITY.iter())
}

pub fn select_catalog_provider_info(
    providers: &[CatalogProviderInfo],
    catalog_provider: CatalogExecutionProvider,
) -> Option<&CatalogProviderInfo> {
    select_catalog_provider_info_by_candidates(
        providers,
        CATALOG_PRIORITY
            .iter()
            .filter(move |candidate| candidate.provider == catalog_provider),
    )
}

fn select_catalog_provider_info_by_candidates(
    providers: &[CatalogProviderInfo],
    candidates: impl Iterator<Item = &'static CatalogCandidate>,
) -> Option<&CatalogProviderInfo> {
    let mut fallback = None;
    for candidate in candidates {
        for provider in providers
            .iter()
            .filter(|provider| provider.name == candidate.catalog_name)
        {
            if fallback.is_none() {
                fallback = Some(provider);
            }
            if provider.ready_state == CatalogReadyState::Ready
                || provider.ready_state == CatalogReadyState::NotReady
            {
                return Some(provider);
            }
        }
    }
    fallback
}

pub fn ensure_catalog_provider_ready(
    provider: CatalogExecutionProvider,
) -> Result<CatalogProviderInfo> {
    ensure_initialized()?;
    with_catalog(|catalog_api, catalog| {
        let candidate =
            find_catalog_candidate(catalog_api, catalog, provider)?.ok_or_else(|| {
                anyhow!(
                    "Windows ML catalog EP {} is not available on this device",
                    provider.label()
                )
            })?;
        check_hr(
            unsafe { (catalog_api.ensure_ready)(candidate.handle) },
            "WinMLEpEnsureReady",
        )?;
        provider_info_from_handle(catalog_api, candidate.handle, candidate.candidate)
    })
}

/// Read a catalog EP's ready state without preparing or registering anything.
///
/// Front-ends call this to decide whether a model load will block on a
/// preparation step, so they can surface a status before the blocking
/// registration. Returns `NotPresent` when the EP is not listed for this device.
pub fn catalog_provider_ready_state(
    provider: CatalogExecutionProvider,
) -> Result<CatalogReadyState> {
    ensure_initialized()?;
    with_catalog(
        |api, catalog| match find_catalog_candidate(api, catalog, provider)? {
            Some(found) => {
                let mut state = 0;
                check_hr(
                    unsafe { (api.get_ready_state)(found.handle, &mut state) },
                    "WinMLEpGetReadyState",
                )?;
                Ok(CatalogReadyState::from_raw(state))
            }
            None => Ok(CatalogReadyState::NotPresent),
        },
    )
}

pub fn catalog_provider_requires_prepare(provider: CatalogExecutionProvider) -> Result<bool> {
    Ok(catalog_provider_ready_state(provider)? == CatalogReadyState::NotReady)
}

/// Best-effort: true if loading `provider` will call `EnsureReady` for a listed
/// but not-yet-ready Windows ML catalog EP. `NotPresent` is intentionally false:
/// ordinary model load should not install absent catalog packages.
pub fn provider_prepare_pending(provider: crate::Provider) -> bool {
    let Some(ep) = provider.catalog_ep() else {
        return false;
    };
    matches!(catalog_provider_requires_prepare(ep), Ok(true))
}

fn initialize() -> Result<BootstrapState, String> {
    initialize_inner().map_err(|err| format!("{err:#}"))
}

fn initialize_inner() -> Result<BootstrapState> {
    let com = ComInit::new()?;
    let (bootstrap_path, bootstrap_path_from_env) = std::env::var(WINDOWS_ML_BOOTSTRAP_ENV)
        .map(|path| (path, true))
        .unwrap_or_else(|_| {
            (
                "Microsoft.WindowsAppRuntime.Bootstrap.dll".to_string(),
                false,
            )
        });
    let wide_path = wide(&bootstrap_path);
    let module = unsafe { LoadLibraryW(wide_path.as_ptr()) };
    let module = if module.is_null() {
        let load_error = std::io::Error::last_os_error();
        if bootstrap_path_from_env {
            bail!(
                "failed to load Windows App SDK bootstrapper from {bootstrap_path} (set by {WINDOWS_ML_BOOTSTRAP_ENV}): {load_error}"
            );
        }
        if !has_package_identity()? {
            bail!(
                "failed to load Windows App SDK bootstrapper {bootstrap_path}: {load_error}. Standalone ZIP/VST3 Windows ML builds must ship Microsoft.WindowsAppRuntime.Bootstrap.dll beside the executable or plugin DLL; run the repository packaging script or set {WINDOWS_ML_BOOTSTRAP_ENV} to the bootstrapper DLL path"
            );
        }
        // Store/MSIX packages declare Microsoft.WindowsAppRuntime.2 as a
        // framework dependency, so the runtime DLLs are already in the package
        // graph and the unpackaged-app bootstrapper is intentionally absent.
        // Only take this path after checking package identity; otherwise a
        // broken ZIP/VST3 package would fall through into ORT initialization and
        // look like a hang instead of a missing-file error.
        warn!(
            "Windows App SDK bootstrapper {bootstrap_path} was not found; trying packaged Windows App Runtime dependency"
        );
        None
    } else {
        let initialize2: MddBootstrapInitialize2 =
            unsafe { load_symbol(module, b"MddBootstrapInitialize2\0")? };
        check_hr(
            unsafe { initialize2(WINDOWS_APP_SDK_2_1, ptr::null(), 0, 0) },
            "MddBootstrapInitialize2(Windows App SDK Runtime 2.1)",
        )
        .context(
            "failed to initialize Windows App SDK Runtime 2.1; install WindowsAppRuntime.2 2.1 or newer",
        )?;
        Some(module)
    };

    let committed = ort::init_from("onnxruntime.dll")
        .context("failed to load Windows ML ONNX Runtime from Windows App Runtime")?
        .commit();
    if !committed {
        bail!(
            "ONNX Runtime was initialized before Windows ML; load a windowsml* provider before any non-Windows ML ORT provider in this process"
        );
    }

    drop(com);
    Ok(BootstrapState { _module: module })
}

fn has_package_identity() -> Result<bool> {
    let mut len = 0u32;
    let rc = unsafe { GetCurrentPackageFullName(&mut len, ptr::null_mut()) };
    match rc {
        ERROR_SUCCESS | ERROR_INSUFFICIENT_BUFFER => Ok(true),
        APPMODEL_ERROR_NO_PACKAGE => Ok(false),
        other => bail!(
            "failed to detect Windows package identity: GetCurrentPackageFullName returned 0x{other:08X}"
        ),
    }
}

fn register_best_catalog_ep_inner() -> Result<Option<CatalogExecutionProvider>> {
    with_catalog(|catalog_api, catalog| {
        for candidate in CATALOG_PRIORITY {
            if catalog_ep_cache(candidate.provider).get().is_some() {
                return Ok(Some(candidate.provider));
            }
            match try_register_candidate(catalog_api, catalog, candidate, false) {
                Ok(true) => {
                    info!(
                        "registered Windows ML catalog EP {}",
                        candidate.catalog_name
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
    })
}

fn register_catalog_ep_inner(provider: CatalogExecutionProvider) -> Result<bool> {
    with_catalog(|catalog_api, catalog| {
        let mut saw_candidate = false;
        let mut last_error = None;
        for candidate in CATALOG_PRIORITY
            .iter()
            .filter(|candidate| candidate.provider == provider)
        {
            saw_candidate = true;
            match try_register_candidate(catalog_api, catalog, candidate, true) {
                Ok(true) => {
                    info!(
                        "registered Windows ML catalog EP {}",
                        candidate.catalog_name
                    );
                    return Ok(true);
                }
                Ok(false) => {}
                Err(err) => {
                    warn!(
                        "failed to register Windows ML catalog EP {}: {err:#}",
                        candidate.catalog_name
                    );
                    last_error = Some(err.context(format!(
                        "failed to prepare/register Windows ML catalog EP {}",
                        candidate.catalog_name
                    )));
                }
            }
        }

        if !saw_candidate {
            bail!("unknown Windows ML catalog EP {}", provider.label());
        }
        // Explicit provider selection must surface the real EnsureReady or
        // registration failure. Returning false here used to replace it with a
        // generic "not present or not ready" error, which made the standalone
        // GUI's automatic preparation look as if it had never run.
        if let Some(err) = last_error {
            return Err(err);
        }
        Ok(false)
    })
}

fn with_catalog<T>(f: impl FnOnce(&CatalogApi, WinMLEpCatalogHandle) -> Result<T>) -> Result<T> {
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
    f(&catalog_api, catalog)
}

fn catalog_ep_cache(provider: CatalogExecutionProvider) -> &'static OnceLock<()> {
    match provider {
        CatalogExecutionProvider::NvTensorRtRtx => &CATALOG_NV_TENSORRT_RTX_EP,
        CatalogExecutionProvider::Qnn => &CATALOG_QNN_EP,
        CatalogExecutionProvider::OpenVino => &CATALOG_OPENVINO_EP,
        CatalogExecutionProvider::MiGraphX => &CATALOG_MIGRAPHX_EP,
        CatalogExecutionProvider::VitisAi => &CATALOG_VITISAI_EP,
    }
}

fn mark_catalog_ep_registered(provider: CatalogExecutionProvider) {
    let _ = catalog_ep_cache(provider).set(());
}

#[derive(Clone, Copy)]
struct FoundCatalogCandidate<'a> {
    handle: WinMLEpHandle,
    candidate: &'a CatalogCandidate,
}

fn find_catalog_candidate(
    api: &CatalogApi,
    catalog: WinMLEpCatalogHandle,
    provider: CatalogExecutionProvider,
) -> Result<Option<FoundCatalogCandidate<'static>>> {
    let mut fallback = None;
    for candidate in CATALOG_PRIORITY
        .iter()
        .filter(|candidate| candidate.provider == provider)
    {
        let catalog_name = CString::new(candidate.catalog_name)?;
        let mut ep = ptr::null_mut();
        let hr =
            unsafe { (api.find_provider)(catalog, catalog_name.as_ptr(), ptr::null(), &mut ep) };
        if hr >= 0 && !ep.is_null() {
            let found = FoundCatalogCandidate {
                handle: ep,
                candidate,
            };
            if fallback.is_none() {
                fallback = Some(found);
            }

            let mut state = 0;
            match check_hr(
                unsafe { (api.get_ready_state)(ep, &mut state) },
                "WinMLEpGetReadyState",
            ) {
                Ok(()) => {
                    let state = CatalogReadyState::from_raw(state);
                    if state == CatalogReadyState::Ready || state == CatalogReadyState::NotReady {
                        return Ok(Some(found));
                    }
                }
                Err(err) => {
                    warn!(
                        "failed to read Windows ML catalog EP {} ready state while selecting provider alias: {err:#}",
                        candidate.catalog_name
                    );
                }
            }
        }
    }
    Ok(fallback)
}

fn provider_info_from_handle(
    api: &CatalogApi,
    ep: WinMLEpHandle,
    candidate: &CatalogCandidate,
) -> Result<CatalogProviderInfo> {
    let mut state = 0;
    check_hr(
        unsafe { (api.get_ready_state)(ep, &mut state) },
        "WinMLEpGetReadyState",
    )?;
    let library_path = read_ep_library_path(api, ep).unwrap_or_default();
    Ok(CatalogProviderInfo {
        name: candidate.catalog_name.to_string(),
        version: String::new(),
        package_family_name: String::new(),
        library_path,
        package_root_path: String::new(),
        ready_state: CatalogReadyState::from_raw(state),
        certification: 0,
        vc_provider: Some(candidate.provider),
    })
}

unsafe extern "system" fn enum_provider_info(
    _ep: WinMLEpHandle,
    info: *const WinMLEpInfoRaw,
    context: *mut c_void,
) -> i32 {
    if info.is_null() || context.is_null() {
        return 1;
    }
    let providers = &mut *(context.cast::<Vec<CatalogProviderInfo>>());
    let info = &*info;
    let name = cstr_or_empty(info.name);
    providers.push(CatalogProviderInfo {
        vc_provider: catalog_provider_from_name(&name),
        name,
        version: cstr_or_empty(info.version),
        package_family_name: cstr_or_empty(info.package_family_name),
        library_path: cstr_or_empty(info.library_path),
        package_root_path: cstr_or_empty(info.package_root_path),
        ready_state: CatalogReadyState::from_raw(info.ready_state),
        certification: info.certification,
    });
    1
}

fn catalog_provider_from_name(name: &str) -> Option<CatalogExecutionProvider> {
    CatalogExecutionProvider::from_catalog_name(name)
}

fn try_register_candidate(
    api: &CatalogApi,
    catalog: WinMLEpCatalogHandle,
    candidate: &CatalogCandidate,
    allow_not_present: bool,
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
    match CatalogReadyState::from_raw(state) {
        CatalogReadyState::Ready | CatalogReadyState::NotReady => {}
        CatalogReadyState::NotPresent if allow_not_present => {}
        CatalogReadyState::NotPresent | CatalogReadyState::Unknown(_) => {
            return Ok(false);
        }
    }
    check_hr(unsafe { (api.ensure_ready)(ep) }, "WinMLEpEnsureReady")?;

    let path = read_ep_library_path(api, ep)?;
    if path.is_empty() {
        return Ok(false);
    }

    let env = Environment::current()?;
    match env.register_ep_library(candidate.catalog_name, &path) {
        Ok(_library) => {}
        Err(err) if library_already_registered_error(&err, candidate.catalog_name) => {}
        Err(err) => return Err(err.into()),
    }
    Ok(true)
}

fn library_already_registered_error(err: &ort::Error, catalog_name: &str) -> bool {
    let message = err.to_string();
    message.contains("already registered") && message.contains(catalog_name)
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

unsafe fn cstr_or_empty(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

struct CatalogApi {
    _module: Hmodule,
    create: WinMLEpCatalogCreate,
    release: WinMLEpCatalogRelease,
    enum_providers: WinMLEpCatalogEnumProviders,
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
            enum_providers: unsafe { load_symbol(module, b"WinMLEpCatalogEnumProviders\0")? },
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
    provider: CatalogExecutionProvider,
}

const CATALOG_PRIORITY: &[CatalogCandidate] = &[
    CatalogCandidate {
        catalog_name: "NvTensorRtRtxExecutionProvider",
        provider: CatalogExecutionProvider::NvTensorRtRtx,
    },
    // Older catalog builds used this all-caps RTX spelling.
    CatalogCandidate {
        catalog_name: "NvTensorRTRTXExecutionProvider",
        provider: CatalogExecutionProvider::NvTensorRtRtx,
    },
    CatalogCandidate {
        catalog_name: "QNNExecutionProvider",
        provider: CatalogExecutionProvider::Qnn,
    },
    CatalogCandidate {
        catalog_name: "OpenVINOExecutionProvider",
        provider: CatalogExecutionProvider::OpenVino,
    },
    CatalogCandidate {
        catalog_name: "MIGraphXExecutionProvider",
        provider: CatalogExecutionProvider::MiGraphX,
    },
    CatalogCandidate {
        catalog_name: "VitisAIExecutionProvider",
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

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(name: &str, ready_state: CatalogReadyState) -> CatalogProviderInfo {
        CatalogProviderInfo {
            name: name.to_string(),
            version: String::new(),
            package_family_name: String::new(),
            library_path: String::new(),
            package_root_path: String::new(),
            ready_state,
            certification: 0,
            vc_provider: CatalogExecutionProvider::from_catalog_name(name),
        }
    }

    #[test]
    fn catalog_ep_and_vc_provider_are_inverses() {
        // Every catalog EP round-trips through the single bijection, and its
        // provider-name string stays in step with the provider's own label.
        for ep in [
            CatalogExecutionProvider::NvTensorRtRtx,
            CatalogExecutionProvider::Qnn,
            CatalogExecutionProvider::OpenVino,
            CatalogExecutionProvider::MiGraphX,
            CatalogExecutionProvider::VitisAi,
        ] {
            let provider = ep.vc_provider();
            assert_eq!(provider.catalog_ep(), Some(ep));
            assert_eq!(ep.vc_provider_name(), provider.label());
        }
    }

    #[test]
    fn selects_ready_alias_for_same_catalog_provider() {
        let providers = vec![
            provider(
                "NvTensorRtRtxExecutionProvider",
                CatalogReadyState::NotPresent,
            ),
            provider("NvTensorRTRTXExecutionProvider", CatalogReadyState::Ready),
        ];

        let selected = select_best_catalog_provider_info(&providers).unwrap();

        assert_eq!(selected.name, "NvTensorRTRTXExecutionProvider");
        assert_eq!(
            select_best_catalog_provider(&providers),
            Some(CatalogExecutionProvider::NvTensorRtRtx)
        );
    }

    #[test]
    fn selects_ready_alias_for_explicit_catalog_provider() {
        let providers = vec![
            provider(
                "NvTensorRTRTXExecutionProvider",
                CatalogReadyState::NotReady,
            ),
            provider("NvTensorRtRtxExecutionProvider", CatalogReadyState::Ready),
        ];

        let selected =
            select_catalog_provider_info(&providers, CatalogExecutionProvider::NvTensorRtRtx)
                .unwrap();

        assert_eq!(selected.name, "NvTensorRtRtxExecutionProvider");
    }

    #[test]
    fn selects_not_ready_priority_candidate_before_later_ready_provider() {
        let providers = vec![
            provider(
                "NvTensorRtRtxExecutionProvider",
                CatalogReadyState::NotPresent,
            ),
            provider(
                "NvTensorRTRTXExecutionProvider",
                CatalogReadyState::NotReady,
            ),
            provider("QNNExecutionProvider", CatalogReadyState::Ready),
        ];

        let selected = select_best_catalog_provider_info(&providers).unwrap();

        assert_eq!(selected.name, "NvTensorRTRTXExecutionProvider");
        assert_eq!(
            select_best_catalog_provider(&providers),
            Some(CatalogExecutionProvider::NvTensorRtRtx)
        );
    }

    #[test]
    fn falls_back_to_first_priority_candidate_when_none_are_ready_or_preparable() {
        let providers = vec![
            provider("QNNExecutionProvider", CatalogReadyState::Unknown(99)),
            provider(
                "NvTensorRtRtxExecutionProvider",
                CatalogReadyState::NotPresent,
            ),
            provider(
                "NvTensorRTRTXExecutionProvider",
                CatalogReadyState::Unknown(99),
            ),
        ];

        let selected = select_best_catalog_provider_info(&providers).unwrap();

        assert_eq!(selected.name, "NvTensorRtRtxExecutionProvider");
        assert_eq!(
            select_best_catalog_provider(&providers),
            Some(CatalogExecutionProvider::NvTensorRtRtx)
        );
    }
}
