// Exercise the exact build-script implementation without installing any SDK.
#[allow(dead_code)]
#[path = "../../../build_support/tensorrt_sdk.rs"]
mod sdk;

use sdk::{parse_version, select_sdk, Version};
use std::{
    fs,
    path::{Path, PathBuf},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        Self(std::env::temp_dir().join(
            format!("vc-rs-sdk-{}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()),
        ))
    }
    fn install(&self, relative: &str, version: [u32; 4]) -> PathBuf {
        let root = self.0.join(relative);
        for dir in ["include", "lib", "bin"] {
            fs::create_dir_all(root.join(dir)).unwrap();
        }
        let mut header = String::new();
        for (name, value) in ["MAJOR", "MINOR", "PATCH", "BUILD"]
            .into_iter()
            .zip(version)
        {
            header.push_str(&format!("#define NV_TENSORRT_{name} TRT_{name}_ENTERPRISE // alias\n#define TRT_{name}_ENTERPRISE {value}\n"));
        }
        fs::write(root.join("include/NvInferVersion.h"), header).unwrap();
        for header in ["NvInfer.h", "NvInferPlugin.h", "NvOnnxParser.h"] {
            fs::write(root.join("include").join(header), "").unwrap();
        }
        for lib in ["nvinfer", "nvinfer_plugin", "nvonnxparser"] {
            fs::write(
                root.join("lib").join(format!("{lib}_{}.lib", version[0])),
                "",
            )
            .unwrap();
            fs::write(
                root.join("bin").join(format!("{lib}_{}.dll", version[0])),
                "",
            )
            .unwrap();
        }
        root
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn numeric_version_order_and_nested_sdk_override_folder_order() {
    let fixture = Fixture::new();
    fixture.install("external/nvidia/TensorRT-11.0", [11, 0, 0, 114]);
    let old = fixture.install("external/nvidia/TensorRT-11.1", [11, 1, 0, 106]);
    let latest = fixture.install(
        "external/nvidia/TensorRT-Enterprise/TensorRT-11.2.1",
        [11, 2, 1, 2],
    );
    assert_eq!(select_sdk(&fixture.0, None).unwrap().unwrap().root, latest);
    assert_eq!(
        select_sdk(&fixture.0, Some(&old)).unwrap().unwrap().root,
        old
    );
    let numeric = fixture.install("external/TensorRT-11.10", [11, 10, 0, 2]);
    fixture.install("external/TensorRT-11.9", [11, 9, 0, 999]);
    fixture.install("TensorRT-12", [12, 0, 0, 1]);
    assert_eq!(select_sdk(&fixture.0, None).unwrap().unwrap().root, numeric);
    let patch = fixture.install("external/nvidia/TensorRT-patch", [11, 10, 1, 1]);
    assert_eq!(select_sdk(&fixture.0, None).unwrap().unwrap().root, patch);
    let build = fixture.install("external/nvidia/TensorRT-build", [11, 10, 1, 10]);
    assert_eq!(select_sdk(&fixture.0, None).unwrap().unwrap().root, build);
}

#[test]
fn incomplete_sdk_is_skipped_but_explicit_invalid_path_fails() {
    let fixture = Fixture::new();
    let good = fixture.install("external/nvidia/TensorRT-good", [11, 1, 0, 106]);
    let bad = fixture.install("external/nvidia/TensorRT-newer", [11, 2, 1, 2]);
    fs::remove_file(bad.join("lib/nvonnxparser_11.lib")).unwrap();
    assert_eq!(select_sdk(&fixture.0, None).unwrap().unwrap().root, good);
    assert!(select_sdk(&fixture.0, Some(&bad)).is_err());
    assert!(select_sdk(&fixture.0, Some(Path::new("missing-sdk"))).is_err());
    fs::remove_file(good.join("bin/nvinfer_11.dll")).unwrap();
    assert!(select_sdk(&fixture.0, None).unwrap().is_none());
}

#[test]
fn explicit_legacy_sdk_remains_available_without_automatic_downgrade() {
    let fixture = Fixture::new();
    let old = fixture.install("external/nvidia/TensorRT-10", [10, 16, 1, 1]);
    assert!(select_sdk(&fixture.0, None).unwrap().is_none());
    assert_eq!(
        select_sdk(&fixture.0, Some(&old)).unwrap().unwrap().root,
        old
    );
}

#[test]
fn version_parser_rejects_missing_cyclic_and_expression_values() {
    let header = "# define NV_TENSORRT_MAJOR 11\n#define NV_TENSORRT_MINOR (2)\n#define NV_TENSORRT_PATCH 1\n#define NV_TENSORRT_BUILD BUILD\n#define BUILD 2";
    assert_eq!(parse_version(header), Some(Version([11, 2, 1, 2])));
    assert_eq!(Version([11, 2, 1, 2]).to_string(), "11.2.1.2");
    assert_eq!(
        parse_version(&header.replace("#define BUILD 2", "#define BUILD BUILD")),
        None
    );
    assert_eq!(
        parse_version(&header.replace("#define BUILD 2", "#define BUILD 1+1")),
        None
    );
    assert_eq!(parse_version(&header.replace("#define BUILD 2", "")), None);
}
