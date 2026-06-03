//! Make bundled DLLs load from next to the plugin.
//!
//! The CUDA package loads provider/CUDA/cuDNN DLLs from beside the plugin. The
//! Windows ML package only bundles `Microsoft.WindowsAppRuntime.Bootstrap.dll`
//! there, while ONNX Runtime and DirectML come from Windows App SDK Runtime.
//! Windows resolves those DLLs against the *host process* (the DAW), not the
//! plugin's own folder. Add the plugin's directory to the user DLL directory
//! list before any ONNX Runtime session is created, while deliberately avoiding
//! process-wide default DLL policy changes that could affect the DAW or other
//! plugins.
//!
//! Windows ML dynamically loads the Windows App SDK Runtime ORT core. CUDA
//! builds use the traditional ORT CUDA EP packaging.

#[cfg(windows)]
fn plugin_dir() -> Option<std::path::PathBuf> {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use std::path::PathBuf;

    use windows_sys::Win32::System::LibraryLoader::{
        GetModuleFileNameW, GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    };

    unsafe {
        // Find this module (the plugin DLL) via the address of a local symbol.
        let mut module = std::ptr::null_mut();
        let addr = plugin_dir as *const () as *const u16;
        if GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            addr,
            &mut module,
        ) == 0
        {
            return None;
        }

        let mut buf = [0u16; 1024];
        let len = GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32);
        if len == 0 || len as usize >= buf.len() {
            return None;
        }

        let mut dir = PathBuf::from(OsString::from_wide(&buf[..len as usize]));
        dir.pop(); // drop the DLL file name, leaving its directory
        Some(dir)
    }
}

#[cfg(windows)]
pub fn add_plugin_dir_to_dll_search_path() {
    use std::os::windows::ffi::OsStrExt;
    use std::sync::Once;

    use windows_sys::Win32::System::LibraryLoader::AddDllDirectory;

    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        let Some(dir) = plugin_dir() else {
            return;
        };

        let wide: Vec<u16> = dir
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        if AddDllDirectory(wide.as_ptr()).is_null() {
            nice_plug::nice_warn!(
                "vc-vst3: failed to add plugin directory to DLL search path: {}",
                dir.display()
            );
        }
    });
}

#[cfg(not(windows))]
pub fn add_plugin_dir_to_dll_search_path() {}

#[cfg(all(windows, feature = "cuda"))]
pub fn preload_bundled_cuda_dlls() -> anyhow::Result<()> {
    use anyhow::Context;

    let dir = plugin_dir().context("failed to locate plugin DLL directory")?;
    if !dir.join("onnxruntime_providers_cuda.dll").exists() {
        nice_plug::nice_warn!(
            "vc-vst3: no bundled ONNX Runtime CUDA provider DLLs found in {}; falling back to the host/system DLL search path",
            dir.display()
        );
        return Ok(());
    }

    ort::ep::cuda::preload_dylibs(Some(&dir), Some(&dir))
        .map_err(|err| anyhow::anyhow!("failed to preload bundled CUDA/cuDNN DLLs: {err}"))?;
    Ok(())
}

// No-op when the CUDA EP is compiled out (TensorRT-only build) or off-Windows:
// there are no provider DLLs to preload, and the GPU path runs through native
// TensorRT instead. The caller only invokes this on the `Provider::Cuda` branch.
#[cfg(not(all(windows, feature = "cuda")))]
pub fn preload_bundled_cuda_dlls() -> anyhow::Result<()> {
    Ok(())
}

#[cfg(windows)]
pub fn with_bundled_dll_directory<T>(f: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
    use std::os::windows::ffi::OsStrExt;
    use std::sync::Mutex;

    use anyhow::{bail, Context};
    use windows_sys::Win32::System::LibraryLoader::SetDllDirectoryW;

    static DLL_DIRECTORY_LOCK: Mutex<()> = Mutex::new(());
    let _guard = DLL_DIRECTORY_LOCK.lock().unwrap();

    let dir = plugin_dir().context("failed to locate plugin DLL directory")?;
    let previous = current_dll_directory();
    let wide_dir: Vec<u16> = dir
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    // SetDllDirectoryW is process-wide, so hold the lock for the shortest
    // possible worker-thread window: only the ORT session creation that loads
    // provider DLLs by name. Do not move this into initialize() or process().
    if unsafe { SetDllDirectoryW(wide_dir.as_ptr()) } == 0 {
        bail!(
            "failed to add bundled DLL directory for CUDA provider load: {}",
            std::io::Error::last_os_error()
        );
    }

    let result = f();

    let restore_ok = match previous {
        Some(previous) => {
            let wide_previous: Vec<u16> = previous
                .as_os_str()
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            unsafe { SetDllDirectoryW(wide_previous.as_ptr()) != 0 }
        }
        None => unsafe { SetDllDirectoryW(std::ptr::null()) != 0 },
    };
    if !restore_ok {
        nice_plug::nice_warn!(
            "vc-vst3: failed to restore previous DLL directory: {}",
            std::io::Error::last_os_error()
        );
    }
    result
}

#[cfg(windows)]
fn current_dll_directory() -> Option<std::ffi::OsString> {
    use std::os::windows::ffi::OsStringExt;

    use windows_sys::Win32::System::LibraryLoader::GetDllDirectoryW;

    let required = unsafe { GetDllDirectoryW(0, std::ptr::null_mut()) };
    if required == 0 {
        return None;
    }
    let mut buf = vec![0u16; required as usize];
    let len = unsafe { GetDllDirectoryW(buf.len() as u32, buf.as_mut_ptr()) };
    if len == 0 || len as usize >= buf.len() {
        return None;
    }
    Some(std::ffi::OsString::from_wide(&buf[..len as usize]))
}

#[cfg(not(windows))]
pub fn with_bundled_dll_directory<T>(f: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
    f()
}
