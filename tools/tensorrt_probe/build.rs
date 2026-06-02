use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=TENSORRT_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-changed=src/trt_probe_shim.cpp");

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("tools/tensorrt_probe must live two levels below the repository root");

    let tensorrt_root = env::var_os("TENSORRT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            repo_root
                .join("TensorRT-10.16.1.11.Windows.amd64.cuda-12.9")
                .join("TensorRT-10.16.1.11")
        });
    let cuda_root = env::var_os("CUDA_PATH")
        .or_else(|| env::var_os("CUDA_HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9")
        });

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

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(&tensorrt_include)
        .include(&cuda_include)
        .file("src/trt_probe_shim.cpp")
        .compile("trt_probe_shim");

    println!("cargo:rustc-link-search=native={}", tensorrt_lib.display());
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=dylib=nvinfer_10");
    println!("cargo:rustc-link-lib=dylib=nvinfer_plugin_10");
    println!("cargo:rustc-link-lib=dylib=nvonnxparser_10");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
