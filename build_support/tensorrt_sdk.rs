//! Build-time SDK discovery shared by vc-core and the standalone builder.
//! Keep the selection rules in sync with scripts/tensorrt-sdk.ps1. Directory
//! names and nvinfer_11.lib cannot distinguish minor releases; the header can.

use std::{
    collections::HashMap,
    env, fmt, fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Version(pub [u32; 4]);

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [major, minor, patch, build] = self.0;
        write!(f, "{major}.{minor}.{patch}.{build}")
    }
}

pub fn parse_version(header: &str) -> Option<Version> {
    let definitions: HashMap<_, _> = header
        .lines()
        .filter_map(|line| {
            let mut words = line.trim().strip_prefix('#')?.split_whitespace();
            (words.next()? == "define").then_some(())?;
            Some((words.next()?, words.next()?))
        })
        .collect();
    let mut parts = [0; 4];
    for (part, name) in parts.iter_mut().zip([
        "NV_TENSORRT_MAJOR",
        "NV_TENSORRT_MINOR",
        "NV_TENSORRT_PATCH",
        "NV_TENSORRT_BUILD",
    ]) {
        let mut value = name;
        // 11.x uses aliases such as TRT_MINOR_ENTERPRISE. Never evaluate header
        // text as code; accept integer tokens / aliases and bound cyclic input.
        let mut resolved = None;
        for _ in 0..16 {
            if let Ok(number) = value.trim_matches(['(', ')']).parse::<u32>() {
                resolved = Some(number);
                break;
            }
            value = definitions.get(value.trim_matches(['(', ')']))?;
        }
        *part = resolved?;
    }
    Some(Version(parts))
}

#[derive(Debug)]
pub struct Sdk {
    pub root: PathBuf,
    pub version: Version,
}

pub fn inspect_sdk(root: &Path) -> Option<Sdk> {
    let version = parse_version(&fs::read_to_string(root.join("include/NvInferVersion.h")).ok()?)?;
    let major = version.0[0];
    // A runtime-only directory is not a build SDK. Both binaries must select
    // the same complete install, even though vc-core itself does not link the parser.
    for header in ["NvInfer.h", "NvInferPlugin.h", "NvOnnxParser.h"] {
        if !root.join("include").join(header).is_file() {
            return None;
        }
    }
    for library in ["nvinfer", "nvinfer_plugin", "nvonnxparser"] {
        if !root
            .join("lib")
            .join(format!("{library}_{major}.lib"))
            .is_file()
            || !root
                .join("bin")
                .join(format!("{library}_{major}.dll"))
                .is_file()
        {
            return None;
        }
    }
    Some(Sdk {
        root: root.to_path_buf(),
        version,
    })
}

fn child_dirs(root: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<_> = fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    paths.sort();
    paths
}

pub fn select_sdk(workspace: &Path, explicit: Option<&Path>) -> Result<Option<Sdk>, String> {
    if let Some(root) = explicit {
        return inspect_sdk(root).map(Some)
            .ok_or_else(|| format!("incomplete TensorRT SDK at {}; expected headers, import libraries and runtime DLLs", root.display()));
    }
    let mut best: Option<Sdk> = None;
    for search_root in [
        workspace.join("external/nvidia"),
        workspace.join("external"),
        workspace.to_path_buf(),
    ] {
        for dir in child_dirs(&search_root).into_iter().filter(|dir| {
            dir.file_name()
                .is_some_and(|name| name.to_string_lossy().to_lowercase().contains("tensorrt"))
        }) {
            let candidates =
                std::iter::once(dir.clone()).chain(child_dirs(&dir).into_iter().filter(|child| {
                    child.file_name().is_some_and(|name| {
                        name.to_string_lossy()
                            .to_lowercase()
                            .starts_with("tensorrt-")
                    })
                }));
            for root in candidates {
                if let Some(sdk) = inspect_sdk(&root) {
                    if sdk.version.0[0] == 11
                        && best.as_ref().is_none_or(|best| sdk.version > best.version)
                    {
                        best = Some(sdk);
                    }
                }
            }
        }
    }
    Ok(best)
}

pub fn detect_sdk(workspace: &Path) -> Result<Option<Sdk>, String> {
    let explicit = env::var_os("TENSORRT_ROOT")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    // Track SDK additions as well as replacement of headers in an existing root.
    // Do not watch the whole workspace: that would include Cargo build outputs.
    for dir in [
        workspace.join("external/nvidia"),
        workspace.join("external"),
    ] {
        if dir.is_dir() {
            println!("cargo:rerun-if-changed={}", dir.display());
            break;
        }
    }
    let sdk = select_sdk(workspace, explicit.as_deref())?;
    if let Some(sdk) = &sdk {
        println!(
            "cargo:rerun-if-changed={}",
            sdk.root.join("include").display()
        );
        // Only a numeric version is embedded; never ship a developer SDK path.
        println!("cargo:rustc-env=VC_RS_TENSORRT_VERSION={}", sdk.version);
    }
    Ok(sdk)
}

pub fn cuda_major_for_trt(trt_major: u32) -> u32 {
    match trt_major {
        10 => 12,
        11 => 13,
        _ => panic!("unsupported TensorRT major {trt_major}; update the CUDA mapping explicitly"),
    }
}

pub fn resolve_cuda_root(cuda_major: u32) -> Option<PathBuf> {
    for variable in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(root) = env::var_os(variable)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
        {
            if cuda_dir_version(&root).is_some_and(|(major, _)| major == cuda_major) {
                return Some(root);
            }
        }
    }
    child_dirs(Path::new(
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    ))
    .into_iter()
    .filter_map(|root| {
        let (major, minor) = cuda_dir_version(&root)?;
        (major == cuda_major).then_some((minor, root))
    })
    .max_by_key(|(minor, _)| *minor)
    .map(|(_, root)| root)
}

fn cuda_dir_version(dir: &Path) -> Option<(u32, u32)> {
    let name = dir.file_name()?.to_str()?;
    let (major, minor) = name.strip_prefix(['v', 'V'])?.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}
