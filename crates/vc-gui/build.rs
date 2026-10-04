//! Like the CLI and VST3, the GUI must delay-load the native TensorRT DLLs.
//! Otherwise an explicitly enabled dual-backend build fails before main with
//! STATUS_DLL_NOT_FOUND, even when the user only wants Windows ML.
//! Linker arguments do not propagate from vc-core; emit them in this final
//! executable crate using the SDK versions from vc-core's links metadata.

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
    // Some CUDA SDKs link cudart statically, leaving its optional delay entry
    // unused. Suppress only MSVC's "no imports for this delay DLL" diagnostic;
    // the native TensorRT entries are verified in the built PE import tables.
    println!("cargo:rustc-link-arg=/IGNORE:4199");
    println!("cargo:rustc-link-arg=delayimp.lib");
}
