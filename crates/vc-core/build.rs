use std::{env, fs, path::PathBuf};

fn main() {
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
        paths.trt_major,
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
            .or_else(|| discover_newest_tensorrt(&workspace_root))?;
        let trt_major = detect_nvinfer_major(&tensorrt_root.join("lib"))?;
        let cuda_root = resolve_cuda_root(cuda_major_for_trt(trt_major))?;

        let paths = Self {
            trt_major,
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

// --- TensorRT / CUDA discovery shared with tools/tensorrt_probe/build.rs ---

/// Find the newest TensorRT install under `workspace_root`. Entries whose name
/// contains "TensorRT" are inspected; the real root is the entry itself when it
/// holds an `include/`, otherwise its single nested `TensorRT-*` subdir. The
/// candidate with the highest `nvinfer_<major>.lib` wins.
fn discover_newest_tensorrt(workspace_root: &std::path::Path) -> Option<PathBuf> {
    let mut best: Option<(u32, PathBuf)> = None;
    for entry in fs::read_dir(workspace_root).ok()?.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if !entry
            .file_name()
            .to_string_lossy()
            .to_lowercase()
            .contains("tensorrt")
        {
            continue;
        }
        for root in tensorrt_root_candidates(&path) {
            if let Some(major) = detect_nvinfer_major(&root.join("lib")) {
                if root.join("include").is_dir()
                    && best
                        .as_ref()
                        .is_none_or(|(best_major, _)| major > *best_major)
                {
                    best = Some((major, root));
                }
            }
        }
    }
    best.map(|(_, path)| path)
}

/// A TensorRT folder may be the install root itself or wrap a single
/// `TensorRT-*` subdirectory (the layout NVIDIA's Windows archives use).
fn tensorrt_root_candidates(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut candidates = vec![dir.to_path_buf()];
    if let Ok(children) = fs::read_dir(dir) {
        for child in children.flatten() {
            let child_path = child.path();
            if child_path.is_dir()
                && child
                    .file_name()
                    .to_string_lossy()
                    .to_lowercase()
                    .starts_with("tensorrt-")
            {
                candidates.push(child_path);
            }
        }
    }
    candidates
}

/// Scan a `lib` directory for `nvinfer_<digits>.lib` and return the highest
/// major. Excludes `nvinfer_plugin_*`, `nvinfer_lean_*`, `nvinfer_dispatch_*`,
/// etc. because the segment after the suffix does not parse as an integer.
fn detect_nvinfer_major(lib_dir: &std::path::Path) -> Option<u32> {
    let mut best: Option<u32> = None;
    for entry in fs::read_dir(lib_dir).ok()?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(rest) = name.strip_prefix("nvinfer_") {
            if let Some(digits) = rest.strip_suffix(".lib") {
                if let Ok(major) = digits.parse::<u32>() {
                    best = Some(best.map_or(major, |b| b.max(major)));
                }
            }
        }
    }
    best
}

/// Map a TensorRT major version to the CUDA major it links against, per
/// NVIDIA's support matrix: TensorRT 10 → CUDA 12, TensorRT 11 → CUDA 13.
fn cuda_major_for_trt(trt_major: u32) -> u32 {
    match trt_major {
        10 => 12,
        11 => 13,
        other => other + 2,
    }
}

/// Resolve the CUDA toolkit for `cuda_major`. `CUDA_PATH` / `CUDA_HOME` is used
/// only when its trailing `v<major>.<minor>` component already matches;
/// otherwise the newest matching toolkit under the standard install dir is
/// chosen so the CUDA runtime stays paired with the selected TensorRT.
fn resolve_cuda_root(cuda_major: u32) -> Option<PathBuf> {
    if let Some(root) = env::var_os("CUDA_PATH")
        .or_else(|| env::var_os("CUDA_HOME"))
        .map(PathBuf::from)
    {
        if cuda_dir_major(&root) == Some(cuda_major) {
            return Some(root);
        }
    }
    discover_newest_cuda(cuda_major)
}

/// Pick the newest `v<cuda_major>.<minor>` toolkit under the default Windows
/// CUDA install directory.
fn discover_newest_cuda(cuda_major: u32) -> Option<PathBuf> {
    let base = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
    let mut best: Option<(u32, PathBuf)> = None;
    for entry in fs::read_dir(&base).ok()?.flatten() {
        let path = entry.path();
        if !path.is_dir() || cuda_dir_major(&path) != Some(cuda_major) {
            continue;
        }
        let minor = cuda_dir_minor(&path).unwrap_or(0);
        if best
            .as_ref()
            .is_none_or(|(best_minor, _)| minor > *best_minor)
        {
            best = Some((minor, path));
        }
    }
    best.map(|(_, path)| path)
}

/// Parse the major from a CUDA toolkit dir named like `v13.2`.
fn cuda_dir_major(dir: &std::path::Path) -> Option<u32> {
    cuda_dir_version(dir).map(|(major, _)| major)
}

fn cuda_dir_minor(dir: &std::path::Path) -> Option<u32> {
    cuda_dir_version(dir).map(|(_, minor)| minor)
}

fn cuda_dir_version(dir: &std::path::Path) -> Option<(u32, u32)> {
    let name = dir.file_name()?.to_string_lossy();
    let version = name.strip_prefix('v').or_else(|| name.strip_prefix('V'))?;
    let (major, minor) = version.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}
