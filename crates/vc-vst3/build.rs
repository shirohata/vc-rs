//! Delay-load the native TensorRT / CUDA runtime DLLs (`nvinfer_<N>.dll`,
//! `nvinfer_plugin_<N>.dll`, `cudart64_<M>.dll`) so Windows resolves them at
//! first use instead of at module load. The shim's `__pfnDliNotifyHook2`
//! (native_tensorrt_shim.cpp) then loads them from the plugin's own directory,
//! letting a self-contained bundle load in a DAW without PATH and avoiding the
//! silent load failure caused by unresolved load-time imports.
//!
//! `cargo:rustc-link-arg` does not propagate from a dependency's build script,
//! so the final crate must emit these. The DLL versions come from vc-core's
//! build script via its `links` metadata (`DEP_VC_RS_NATIVE_TENSORRT_*`), which
//! is only set when the native TensorRT shim is actually built.

fn main() {
    let Some(nvinfer_major) = std::env::var_os("DEP_VC_RS_NATIVE_TENSORRT_NVINFER_MAJOR") else {
        // No native TensorRT shim in this build (e.g. the CUDA package, or native
        // TRT disabled): nothing to delay-load.
        return;
    };
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        // `/DELAYLOAD` and delayimp.lib are MSVC-only.
        return;
    }
    let nvinfer_major = nvinfer_major.to_string_lossy();
    let cuda_major = std::env::var("DEP_VC_RS_NATIVE_TENSORRT_CUDA_MAJOR").unwrap_or_default();
    for dll in [
        format!("nvinfer_{nvinfer_major}.dll"),
        format!("nvinfer_plugin_{nvinfer_major}.dll"),
        format!("cudart64_{cuda_major}.dll"),
    ] {
        println!("cargo:rustc-link-arg=/DELAYLOAD:{dll}");
    }
    println!("cargo:rustc-link-arg=delayimp.lib");
}
