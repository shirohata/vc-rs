//! Delay-load the native TensorRT / CUDA runtime DLLs (`nvinfer_<N>.dll`,
//! `nvinfer_plugin_<N>.dll`, `cudart64_<M>.dll`) for the `vc-rs` binary and its
//! test executables. With delay-load, the exe starts without these DLLs present
//! and resolves them only on first TensorRT use (via the shim's
//! `__pfnDliNotifyHook2`, falling back to the default search / PATH). This keeps
//! `cargo test` from failing at process start with STATUS_DLL_NOT_FOUND when the
//! TensorRT bin is not on PATH and the test never touches the GPU.
//!
//! `cargo:rustc-link-arg` does not propagate from a dependency's build script,
//! so the final crate must emit these. The DLL versions come from vc-core's
//! build script via its `links` metadata (`DEP_VC_RS_NATIVE_TENSORRT_*`), which
//! is only set when the native TensorRT shim is actually built.

fn main() {
    let Some(nvinfer_major) = std::env::var_os("DEP_VC_RS_NATIVE_TENSORRT_NVINFER_MAJOR") else {
        return;
    };
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
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
