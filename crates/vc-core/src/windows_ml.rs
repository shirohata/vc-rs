//! Windows ML runtime bootstrap for ORT-backed inference.
//!
//! Windows App SDK Runtime 2.x carries the shared ONNX Runtime and DirectML
//! binaries. We keep this initialized for the process lifetime so later ORT
//! sessions can continue resolving package-graph DLLs; do not call shutdown from
//! a model/session drop path or from realtime audio code.

use std::ffi::{c_char, c_void, CStr, CString, OsString};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
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
const WINDOWS_APP_SDK_MAJOR: u32 = 0x0002_0000;
// Starting with SDK 2.x the bootstrapper ignores majorMinorVersion's minor
// bits. Enforce the C catalog API's 2.1 floor through PACKAGE_VERSION instead;
// keep this packed Major.Minor.Build.Revision in sync with the Store manifest
// default in scripts/package-store-msix.ps1. Zero here accepts older runtimes.
const WINDOWS_APP_RUNTIME_MIN_VERSION: u64 = (2u64 << 48) | (1u64 << 32);
const WINDOWS_ML_BOOTSTRAP_ENV: &str = "VC_RS_WINDOWSML_BOOTSTRAP_DLL";
const WINDOWS_APP_RUNTIME_FAMILY: &str = "Microsoft.WindowsAppRuntime.2_8wekyb3d8bbwe";

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
    fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> Hmodule;
    fn GetModuleHandleExW(flags: u32, address: *const u16, module: *mut Hmodule) -> i32;
    fn GetModuleFileNameW(module: Hmodule, filename: *mut u16, size: u32) -> u32;
    fn GetProcAddress(module: Hmodule, name: *const c_char) -> Farproc;
    fn GetCurrentPackageFullName(
        package_full_name_length: *mut u32,
        package_full_name: *mut u16,
    ) -> i32;
    fn GetCurrentPackageInfo(
        flags: u32,
        buffer_length: *mut u32,
        buffer: *mut u8,
        count: *mut u32,
    ) -> i32;
}

#[link(name = "ole32")]
extern "system" {
    fn CoInitializeEx(reserved: *mut c_void, coinit: u32) -> Hresult;
    fn CoUninitialize();
}

struct BootstrapState {
    _module: Option<Hmodule>,
    runtime_directory: PathBuf,
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
            crate::Provider::WindowsMlOpenVino
            | crate::Provider::WindowsMlOpenVinoCpu
            | crate::Provider::WindowsMlOpenVinoGpu
            | crate::Provider::WindowsMlOpenVinoNpu => Some(CatalogExecutionProvider::OpenVino),
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
    let registered = register_catalog_ep_inner(provider, true)?;
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
                    if provider == crate::Provider::WindowsMlOpenVino {
                        acc.extend([
                            crate::Provider::WindowsMlOpenVinoCpu,
                            crate::Provider::WindowsMlOpenVinoGpu,
                            crate::Provider::WindowsMlOpenVinoNpu,
                        ]);
                    }
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

/// Prepare the installed OpenVINO EP and inspect its hardware. NotReady is a
/// normal per-process state; library paths can remain empty until EnsureReady.
/// Return None for a listed NotPresent EP until the user authorizes acquisition.
/// Call on a UI-owned background thread, never on an audio callback.
pub(crate) fn probe_openvino_devices(allow_download: bool) -> Result<Option<Vec<crate::Provider>>> {
    ensure_initialized()?;
    let provider = CatalogExecutionProvider::OpenVino;
    if catalog_ep_cache(provider).get().is_none() {
        let providers = list_catalog_providers()?;
        let entry = select_catalog_provider_info(&providers, provider)
            .context("OpenVINO EP is not listed by Windows ML on this machine")?;
        if entry.ready_state == CatalogReadyState::NotPresent && !allow_download {
            return Ok(None);
        }
        anyhow::ensure!(
            register_catalog_ep_inner(provider, allow_download)?,
            "OpenVINO EP is not installed or is not listed by Windows ML"
        );
        mark_catalog_ep_registered(provider);
    }
    let env = Environment::current()?;
    let mut available = Vec::new();
    let mut enumerated = false;
    for device in env.devices() {
        enumerated = true;
        if CatalogExecutionProvider::from_catalog_name(device.ep()?)
            != Some(CatalogExecutionProvider::OpenVino)
        {
            continue;
        }
        for &candidate in crate::Provider::OPENVINO_DEVICES {
            if candidate.openvino_device_type() == Some(device.hardware_device().ty())
                && !available.contains(&candidate)
            {
                available.push(candidate);
            }
        }
    }
    // ort's devices() iterator suppresses GetEpDevices errors and becomes
    // empty. Do not turn a failed enumeration into three disabled choices.
    anyhow::ensure!(enumerated, "ONNX Runtime did not return a device list");
    Ok(Some(available))
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

/// Explicit preparation for frontends that present download/terms before model
/// loading. This shares registration and its process cache with RvcPipeline;
/// callers must obtain consent before invoking it, never from an audio callback.
pub fn prepare_provider(provider: crate::Provider) -> Result<()> {
    ensure_initialized()?;
    if let Some(ep) = provider.catalog_ep() {
        if !try_register_catalog_ep(ep)? {
            bail!("The selected execution provider is not available on this device");
        }
    } else if provider == crate::Provider::WindowsMl {
        try_register_best_catalog_ep()?;
    }
    Ok(())
}

/// Read-only readiness check for setup. It never invokes EnsureReady. Auto
/// selection may fall through multiple candidates, so every listed candidate
/// that might need acquisition must be covered by the explicit prepare action.
pub fn preparation_required(provider: crate::Provider) -> Result<bool> {
    let entries = list_catalog_providers()?;
    if let Some(ep) = provider.catalog_ep() {
        let entry = entries
            .iter()
            .find(|entry| entry.vc_provider == Some(ep))
            .ok_or_else(|| {
                anyhow!("The selected execution provider is not available on this device")
            })?;
        return Ok(entry.ready_state != CatalogReadyState::Ready);
    }
    Ok(provider == crate::Provider::WindowsMl
        && entries.iter().any(|entry| {
            entry.vc_provider.is_some() && entry.ready_state == CatalogReadyState::NotReady
        }))
}

fn initialize() -> Result<BootstrapState, String> {
    initialize_inner().map_err(|err| format!("{err:#}"))
}

fn bundled_bootstrap_path() -> Result<PathBuf> {
    // vc-core is linked into the EXE or VST3 DLL. current_exe() would locate
    // the DAW, and AddDllDirectory does not affect ordinary LoadLibraryW unless
    // the host opts into that search policy. Resolve our own module by address
    // instead; never change the DAW's process-wide DLL policy to load Windows ML.
    let mut module = ptr::null_mut();
    let flags = 0x0000_0004 | 0x0000_0002; // FROM_ADDRESS | UNCHANGED_REFCOUNT
    if unsafe {
        GetModuleHandleExW(
            flags,
            bundled_bootstrap_path as *const () as *const u16,
            &mut module,
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error()).context("failed to locate Windows ML module");
    }
    let mut buffer = vec![0u16; 260];
    loop {
        let len = unsafe { GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as u32) };
        if len == 0 {
            return Err(std::io::Error::last_os_error())
                .context("failed to resolve Windows ML module path");
        }
        if (len as usize) < buffer.len() {
            let mut path = PathBuf::from(std::ffi::OsString::from_wide(&buffer[..len as usize]));
            path.set_file_name("Microsoft.WindowsAppRuntime.Bootstrap.dll");
            return Ok(path);
        }
        buffer.resize(buffer.len() * 2, 0);
    }
}

fn initialize_inner() -> Result<BootstrapState> {
    let com = ComInit::new()?;
    let (bootstrap_path, bootstrap_path_from_env) = match std::env::var_os(WINDOWS_ML_BOOTSTRAP_ENV)
    {
        Some(path) => (PathBuf::from(path), true),
        None => (bundled_bootstrap_path()?, false),
    };
    let wide_path: Vec<u16> = bootstrap_path
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let bootstrap_path = bootstrap_path.display();
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
            unsafe {
                initialize2(
                    WINDOWS_APP_SDK_MAJOR,
                    ptr::null(),
                    WINDOWS_APP_RUNTIME_MIN_VERSION,
                    0,
                )
            },
            "MddBootstrapInitialize2(Windows App SDK Runtime 2.1)",
        )
        .context(
            "failed to initialize Windows App SDK Runtime 2.1; install WindowsAppRuntime.2 2.1 or newer",
        )?;
        Some(module)
    };

    // Windows 10's unpackaged DLL search can choose System32 before the paths
    // added by bootstrap. Use only the resolved package graph, not an installed
    // package scan (which could select a different runtime from bootstrap/MSIX).
    let runtime_directory = runtime_directory(&current_graph_packages()?)?;
    let ort_path = runtime_directory.join("onnxruntime.dll");
    // DirectML is also delay-loaded by ORT. Pin the matching copy first; merely
    // giving ORT an absolute path does not control later dependency searches.
    load_runtime_library(&runtime_directory.join("DirectML.dll"))?;
    load_runtime_library(&ort_path)?;
    let committed = ort::init_from(&ort_path)
        .with_context(|| {
            format!(
                "failed to load Windows ML ONNX Runtime from {}",
                ort_path.display()
            )
        })?
        .commit();
    if !committed {
        bail!(
            "ONNX Runtime was initialized before Windows ML; load a windowsml* provider before any non-Windows ML ORT provider in this process"
        );
    }

    drop(com);
    Ok(BootstrapState {
        _module: module,
        runtime_directory,
    })
}

// ABI layouts from appmodel.h. The buffer returned by GetCurrentPackageInfo
// contains PACKAGE_INFO[count] followed by the strings its pointers reference.
#[repr(C)]
struct PackageId {
    reserved: u32,
    processor_architecture: u32,
    version: u64,
    name: *mut u16,
    publisher: *mut u16,
    resource_id: *mut u16,
    publisher_id: *mut u16,
}

#[repr(C)]
struct PackageInfo {
    reserved: u32,
    flags: u32,
    path: *mut u16,
    full_name: *mut u16,
    family_name: *mut u16,
    id: PackageId,
}

struct GraphPackage {
    family: OsString,
    version: u64,
    path: PathBuf,
}

fn current_graph_packages() -> Result<Vec<GraphPackage>> {
    // HEAD | DIRECT | STATIC | DYNAMIC. Opt into dynamic dependencies explicitly:
    // Windows App SDK hooks this API on Windows 10; Windows 11 provides it natively.
    // STATIC also includes the declared framework dependency in Store/MSIX apps.
    const FLAGS: u32 = 0x10 | 0x20 | 0x0008_0000 | 0x0010_0000;
    let mut bytes = 0;
    let mut count = 0;
    let mut rc = unsafe { GetCurrentPackageInfo(FLAGS, &mut bytes, ptr::null_mut(), &mut count) };
    loop {
        if rc != ERROR_INSUFFICIENT_BUFFER {
            bail!("failed to read Windows App Runtime package graph: GetCurrentPackageInfo returned 0x{rc:08X}");
        }
        // u64 storage provides PACKAGE_INFO alignment, unlike Vec<u8>. A host
        // may add dependencies concurrently, so retry if the required size grows.
        let mut buffer = vec![0u64; (bytes as usize).div_ceil(8)];
        rc = unsafe {
            GetCurrentPackageInfo(FLAGS, &mut bytes, buffer.as_mut_ptr().cast(), &mut count)
        };
        if rc == ERROR_INSUFFICIENT_BUFFER {
            continue;
        }
        if rc != ERROR_SUCCESS {
            bail!("failed to read Windows App Runtime package graph: GetCurrentPackageInfo returned 0x{rc:08X}");
        }
        if count as usize > buffer.len() * 8 / std::mem::size_of::<PackageInfo>() {
            bail!("Windows App Runtime package graph returned an invalid package count");
        }
        let packages = unsafe {
            std::slice::from_raw_parts(buffer.as_ptr().cast::<PackageInfo>(), count as usize)
        };
        return packages
            .iter()
            .map(|package| {
                Ok(GraphPackage {
                    family: graph_string(&buffer, package.family_name)?,
                    version: package.id.version,
                    path: PathBuf::from(graph_string(&buffer, package.path)?),
                })
            })
            .collect();
    }
}

fn graph_string(buffer: &[u64], value: *const u16) -> Result<OsString> {
    let start = buffer.as_ptr() as usize;
    let end = start + std::mem::size_of_val(buffer);
    let address = value as usize;
    if address < start || address >= end || !address.is_multiple_of(2) {
        bail!("Windows App Runtime package graph returned an invalid string pointer");
    }
    // Bound the scan to the owning buffer, which remains alive until all graph
    // strings have been copied. No raw package pointers escape this function.
    let units = unsafe { std::slice::from_raw_parts(value, (end - address) / 2) };
    let len = units
        .iter()
        .position(|&unit| unit == 0)
        .context("Windows App Runtime package graph returned an unterminated string")?;
    Ok(OsString::from_wide(&units[..len]))
}

fn runtime_directory(packages: &[GraphPackage]) -> Result<PathBuf> {
    let package = packages
        .iter()
        .find(|package| {
            package.family == WINDOWS_APP_RUNTIME_FAMILY
                && package.version >= WINDOWS_APP_RUNTIME_MIN_VERSION
        })
        .context("Windows App Runtime 2.1 or newer was not found in the current package graph")?;
    if !package.path.is_absolute() {
        bail!("Windows App Runtime package graph returned a non-absolute runtime path");
    }
    Ok(package.path.clone())
}

fn load_runtime_library(path: &Path) -> Result<Hmodule> {
    let name: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    // DLL_LOAD_DIR | SYSTEM32: resolve immediate dependencies beside this DLL
    // and in Windows, without changing the DAW's process-wide search policy.
    let module = unsafe { LoadLibraryExW(name.as_ptr(), ptr::null_mut(), 0x100 | 0x800) };
    if module.is_null() {
        return Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "failed to load Windows ML runtime library {}",
                path.display()
            )
        });
    }
    // Deliberately retain this reference for the process lifetime, like bootstrap
    // and ORT. Sessions/catalog callbacks can outlive any one frontend worker.
    Ok(module)
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

fn register_catalog_ep_inner(
    provider: CatalogExecutionProvider,
    allow_not_present: bool,
) -> Result<bool> {
    with_catalog(|catalog_api, catalog| {
        let mut saw_candidate = false;
        let mut last_error = None;
        for candidate in CATALOG_PRIORITY
            .iter()
            .filter(|candidate| candidate.provider == provider)
        {
            saw_candidate = true;
            match try_register_candidate(catalog_api, catalog, candidate, allow_not_present) {
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
        let state = WINDOWS_ML_BOOTSTRAP
            .get_or_init(initialize)
            .as_ref()
            .map_err(|err| anyhow!(err.clone()))?;
        let module = load_runtime_library(
            &state
                .runtime_directory
                .join("Microsoft.Windows.AI.MachineLearning.dll"),
        )?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_package(family: &str, version: u64, path: &str) -> GraphPackage {
        GraphPackage {
            family: family.into(),
            version,
            path: path.into(),
        }
    }

    #[test]
    fn runtime_selection_uses_graph_order_and_required_family_and_version() {
        let packages = [
            graph_package("Other.Framework_8wekyb3d8bbwe", u64::MAX, r"C:\other"),
            graph_package(WINDOWS_APP_RUNTIME_FAMILY, 2u64 << 48, r"C:\old"),
            graph_package(
                WINDOWS_APP_RUNTIME_FAMILY,
                WINDOWS_APP_RUNTIME_MIN_VERSION,
                r"C:\selected",
            ),
            graph_package(WINDOWS_APP_RUNTIME_FAMILY, u64::MAX, r"C:\newer"),
        ];
        assert_eq!(
            runtime_directory(&packages).unwrap(),
            Path::new(r"C:\selected")
        );
        assert!(runtime_directory(&packages[..2]).is_err());
        assert!(runtime_directory(&[]).is_err());
        assert!(runtime_directory(&[graph_package(
            WINDOWS_APP_RUNTIME_FAMILY,
            WINDOWS_APP_RUNTIME_MIN_VERSION,
            "relative",
        )])
        .is_err());
    }

    #[test]
    fn graph_strings_are_copied_within_the_owning_buffer() {
        let mut buffer = [u64::MAX; 2];
        let units = unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr().cast::<u16>(), 8) };
        units[..3].copy_from_slice(&[0x65E5, 0x672C, 0]);
        let value = buffer.as_ptr().cast::<u16>();
        assert_eq!(
            graph_string(&buffer, value).unwrap(),
            OsString::from("日本")
        );
        assert!(graph_string(&buffer, ptr::null()).is_err());
        assert!(graph_string(&buffer, unsafe { value.add(8) }).is_err());
        assert!(graph_string(&buffer, unsafe { value.add(3) }).is_err());
    }

    #[test]
    #[ignore = "requires an installed Windows App Runtime and bootstrap DLL"]
    fn runtime_ignores_executable_adjacent_onnxruntime() {
        // Reproduce the name collision even on Windows 11: ort::init_from with
        // a basename prefers an EXE-adjacent DLL. A renamed system DLL is valid
        // to load but has no OrtGetApiBase, like the Windows 10 System32 stub.
        // Run in a fresh process so another test cannot initialize ORT first.
        let bootstrap = std::env::var_os(WINDOWS_ML_BOOTSTRAP_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| bundled_bootstrap_path().unwrap())
            .canonicalize()
            .unwrap();
        let directory = std::env::temp_dir().join(format!(
            "vc-rs-runtime-shadow-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let executable = directory.join("runtime-test.exe");
        let decoy = directory.join("onnxruntime.dll");
        std::fs::copy(std::env::current_exe().unwrap(), &executable).unwrap();
        let system = PathBuf::from(std::env::var_os("SystemRoot").unwrap());
        std::fs::copy(system.join("System32/version.dll"), &decoy).unwrap();
        let result = std::process::Command::new(&executable)
            .env(WINDOWS_ML_BOOTSTRAP_ENV, bootstrap)
            .args([
                "--exact",
                "windows_ml::tests::runtime_libraries_load_from_the_resolved_package_graph",
                "--ignored",
                "--nocapture",
            ])
            .output();
        std::fs::remove_file(&decoy).unwrap();
        std::fs::remove_file(&executable).unwrap();
        std::fs::remove_dir(&directory).unwrap();
        let result = result.unwrap();
        assert!(
            result.status.success(),
            "child failed: {}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }

    #[test]
    #[ignore = "requires an installed Windows App Runtime and bootstrap DLL"]
    fn runtime_libraries_load_from_the_resolved_package_graph() {
        ensure_initialized().unwrap();
        let expected = runtime_directory(&current_graph_packages().unwrap()).unwrap();
        let catalog = CatalogApi::load().unwrap();
        for (name, symbol) in [
            ("onnxruntime.dll", b"OrtGetApiBase\0".as_slice()),
            ("DirectML.dll", b"DMLCreateDevice\0".as_slice()),
            (
                "Microsoft.Windows.AI.MachineLearning.dll",
                b"WinMLEpCatalogCreate\0".as_slice(),
            ),
        ] {
            let module = load_runtime_library(&expected.join(name)).unwrap();
            assert!(!unsafe { GetProcAddress(module, symbol.as_ptr().cast()) }.is_null());
            let mut path = vec![0u16; 32768];
            let len = unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) };
            assert!(len > 0 && (len as usize) < path.len());
            let loaded = PathBuf::from(OsString::from_wide(&path[..len as usize]));
            assert_eq!(loaded, expected.join(name));
            eprintln!("Windows ML loaded {}", loaded.display());
        }
        assert!(!catalog._module.is_null());
        // A real ORT environment is also created; the tiny-RVC tests cover inference.
        Environment::current().unwrap();
    }

    #[test]
    fn bundled_bootstrap_is_next_to_the_containing_module() {
        // Unit tests are EXEs. A hosted DLL smoke test must additionally check
        // this resolves beside the plugin rather than beside its host EXE.
        let expected = std::env::current_exe()
            .unwrap()
            .with_file_name("Microsoft.WindowsAppRuntime.Bootstrap.dll");
        assert_eq!(bundled_bootstrap_path().unwrap(), expected);
    }

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
