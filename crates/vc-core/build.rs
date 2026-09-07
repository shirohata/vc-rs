use std::{env, path::PathBuf};

#[path = "../../build_support/tensorrt_sdk.rs"]
mod tensorrt_sdk;
use tensorrt_sdk::{cuda_major_for_trt, detect_sdk, resolve_cuda_root};

fn main() {
    println!("cargo:rerun-if-changed=../../build_support/tensorrt_sdk.rs");
    println!("cargo:rerun-if-env-changed=TENSORRT_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=VC_RS_ENABLE_NATIVE_TENSORRT");
    println!("cargo:rerun-if-changed=src/model_rvc/native_tensorrt_shim.cpp");
    println!("cargo:rustc-check-cfg=cfg(native_tensorrt)");

    // The native TensorRT shim links `nvinfer_<major>.dll` at load time. Only
    // build it when the `tensorrt` cargo feature is enabled (the CLI), so a
    // plugin build without that feature never picks up the dependency.
    if env::var_os("CARGO_FEATURE_TENSORRT").is_none() {
        return;
    }

    if env::var("VC_RS_ENABLE_NATIVE_TENSORRT")
        .is_ok_and(|value| matches!(value.as_str(), "0" | "false" | "off" | "no"))
    {
        return;
    }

    let Some(paths) = NativeTensorRtPaths::detect() else {
        println!("cargo:warning=native TensorRT shim disabled; set TENSORRT_ROOT and CUDA_PATH to enable it");
        return;
    };

    println!(
        "cargo:warning=native TensorRT shim using TensorRT {} ({}), CUDA ({})",
        paths.trt_version,
        paths
            .tensorrt_lib
            .parent()
            .unwrap_or(&paths.tensorrt_lib)
            .display(),
        paths
            .cuda_lib
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or(&paths.cuda_lib)
            .display(),
    );

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(&paths.tensorrt_include)
        .include(&paths.cuda_include)
        .file("src/model_rvc/native_tensorrt_shim.cpp")
        .compile("vc_rs_native_tensorrt");

    println!("cargo:rustc-cfg=native_tensorrt");
    println!(
        "cargo:rustc-link-search=native={}",
        paths.tensorrt_lib.display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        paths.cuda_lib.display()
    );
    println!("cargo:rustc-link-lib=dylib=nvinfer_{}", paths.trt_major);
    println!(
        "cargo:rustc-link-lib=dylib=nvinfer_plugin_{}",
        paths.trt_major
    );
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Propagate the resolved DLL versions to dependent build scripts (vc-cli,
    // vc-vst3) as DEP_VC_RS_NATIVE_TENSORRT_{NVINFER_MAJOR,CUDA_MAJOR}. They use
    // these to emit `/DELAYLOAD:` linker args (which do not propagate from a lib
    // crate's build script), so nvinfer_<N>.dll / nvinfer_plugin_<N>.dll /
    // cudart64_<M>.dll are delay-loaded and resolved from the module directory.
    println!("cargo:nvinfer_major={}", paths.trt_major);
    println!("cargo:cuda_major={}", cuda_major_for_trt(paths.trt_major));
}

struct NativeTensorRtPaths {
    trt_major: u32,
    trt_version: tensorrt_sdk::Version,
    tensorrt_include: PathBuf,
    tensorrt_lib: PathBuf,
    cuda_include: PathBuf,
    cuda_lib: PathBuf,
}

impl NativeTensorRtPaths {
    fn detect() -> Option<Self> {
        let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR")?);
        // Local SDK archives live under `external/` so the repository root stays
        // reserved for source-owned paths. This crate's manifest is at
        // `crates/vc-core`, so walk two levels up before searching there.
        let workspace_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .map(PathBuf::from)
            .unwrap_or_else(|| manifest_dir.clone());

        let sdk = detect_sdk(&workspace_root).unwrap_or_else(|error| panic!("{error}"))?;
        let tensorrt_root = sdk.root;
        let trt_major = sdk.version.0[0];
        let cuda_root = resolve_cuda_root(cuda_major_for_trt(trt_major))?;

        let paths = Self {
            trt_major,
            trt_version: sdk.version,
            tensorrt_include: tensorrt_root.join("include"),
            tensorrt_lib: tensorrt_root.join("lib"),
            cuda_include: cuda_root.join("include"),
            cuda_lib: cuda_root.join("lib").join("x64"),
        };
        [
            &paths.tensorrt_include,
            &paths.tensorrt_lib,
            &paths.cuda_include,
            &paths.cuda_lib,
        ]
        .iter()
        .all(|path| path.exists())
        .then_some(paths)
    }
}
