use std::{env, path::PathBuf};

#[path = "../../build_support/tensorrt_sdk.rs"]
mod tensorrt_sdk;
use tensorrt_sdk::{cuda_major_for_trt, detect_sdk, resolve_cuda_root};

fn main() {
    println!("cargo:rerun-if-changed=../../build_support/tensorrt_sdk.rs");
    println!("cargo:rerun-if-env-changed=TENSORRT_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-changed=src/trt_builder_shim.cpp");

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("tools/tensorrt_builder must live two levels below the repository root")
        .to_path_buf();

    let sdk = detect_sdk(&repo_root)
        .unwrap_or_else(|error| panic!("{error}"))
        .expect("no complete TensorRT 11 SDK found; set TENSORRT_ROOT");
    let tensorrt_root = sdk.root;
    let trt_major = sdk.version.0[0];
    let cuda_root = resolve_cuda_root(cuda_major_for_trt(trt_major))
        .expect("no matching CUDA toolkit found; set CUDA_PATH");

    let tensorrt_include = tensorrt_root.join("include");
    let tensorrt_lib = tensorrt_root.join("lib");
    let cuda_include = cuda_root.join("include");
    let cuda_lib = cuda_root.join("lib").join("x64");

    for path in [&tensorrt_include, &tensorrt_lib, &cuda_include, &cuda_lib] {
        if !path.exists() {
            panic!(
                "required TensorRT/CUDA path does not exist: {}",
                path.display()
            );
        }
    }

    println!(
        "cargo:warning=tensorrt_builder using TensorRT {} ({}), CUDA ({})",
        sdk.version,
        tensorrt_root.display(),
        cuda_root.display()
    );

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(&tensorrt_include)
        .include(&cuda_include)
        .file("src/trt_builder_shim.cpp")
        .compile("trt_builder_shim");

    println!("cargo:rustc-link-search=native={}", tensorrt_lib.display());
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=dylib=nvinfer_{trt_major}");
    println!("cargo:rustc-link-lib=dylib=nvinfer_plugin_{trt_major}");
    println!("cargo:rustc-link-lib=dylib=nvonnxparser_{trt_major}");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
