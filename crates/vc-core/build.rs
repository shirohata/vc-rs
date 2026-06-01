use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=TENSORRT_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=VC_RS_ENABLE_NATIVE_TENSORRT");
    println!("cargo:rerun-if-changed=src/model_rvc/native_tensorrt_shim.cpp");
    println!("cargo:rustc-check-cfg=cfg(native_tensorrt)");

    if env::var("VC_RS_ENABLE_NATIVE_TENSORRT")
        .is_ok_and(|value| matches!(value.as_str(), "0" | "false" | "off" | "no"))
    {
        return;
    }

    let Some(paths) = NativeTensorRtPaths::detect() else {
        println!("cargo:warning=native TensorRT shim disabled; set TENSORRT_ROOT and CUDA_PATH to enable it");
        return;
    };

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
    println!("cargo:rustc-link-lib=dylib=nvinfer_10");
    println!("cargo:rustc-link-lib=dylib=nvinfer_plugin_10");
    println!("cargo:rustc-link-lib=dylib=cudart");
}

struct NativeTensorRtPaths {
    tensorrt_include: PathBuf,
    tensorrt_lib: PathBuf,
    cuda_include: PathBuf,
    cuda_lib: PathBuf,
}

impl NativeTensorRtPaths {
    fn detect() -> Option<Self> {
        let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR")?);
        // The bundled TensorRT folder lives at the workspace root. This crate's
        // manifest is at `crates/vc-core`, so walk two levels up to find it when
        // `TENSORRT_ROOT` is not set explicitly.
        let workspace_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .map(PathBuf::from)
            .unwrap_or_else(|| manifest_dir.clone());
        let tensorrt_root = env::var_os("TENSORRT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                workspace_root
                    .join("TensorRT-10.16.1.11.Windows.amd64.cuda-12.9")
                    .join("TensorRT-10.16.1.11")
            });
        let cuda_root = env::var_os("CUDA_PATH")
            .or_else(|| env::var_os("CUDA_HOME"))
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9")
            });

        let paths = Self {
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
