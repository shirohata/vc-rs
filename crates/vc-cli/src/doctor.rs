use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Result};
use cpal::traits::{DeviceTrait, HostTrait};

use crate::cli::default_provider;
use vc_core::model_rvc::{self, EngineCacheInfo};
use vc_core::Provider;

pub fn run() -> Result<()> {
    let mut report = DoctorReport::default();

    println!("vc-rs doctor");
    println!("Read-only runtime diagnostics. No install, repair, or cache cleanup is performed.");
    println!();

    check_build(&mut report);
    check_windows_ml(&mut report);
    check_tensorrt(&mut report);
    check_nvidia(&mut report);
    check_engine_cache(&mut report);
    check_audio(&mut report);

    report.print();

    if report.has_failures() {
        bail!("doctor found FAIL items; see diagnostics above");
    }
    Ok(())
}

#[derive(Default)]
struct DoctorReport {
    checks: Vec<DoctorCheck>,
}

impl DoctorReport {
    fn add(&mut self, status: CheckStatus, name: impl Into<String>, message: impl Into<String>) {
        self.checks.push(DoctorCheck {
            status,
            name: name.into(),
            message: message.into(),
            details: Vec::new(),
        });
    }

    fn add_with_details(
        &mut self,
        status: CheckStatus,
        name: impl Into<String>,
        message: impl Into<String>,
        details: Vec<String>,
    ) {
        self.checks.push(DoctorCheck {
            status,
            name: name.into(),
            message: message.into(),
            details,
        });
    }

    fn has_failures(&self) -> bool {
        self.checks
            .iter()
            .any(|check| check.status == CheckStatus::Fail)
    }

    fn print(&self) {
        for check in &self.checks {
            println!(
                "[{}] {}: {}",
                check.status.label(),
                check.name,
                check.message
            );
            for detail in &check.details {
                println!("      {detail}");
            }
        }
    }
}

struct DoctorCheck {
    status: CheckStatus,
    name: String,
    message: String,
    details: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CheckStatus {
    Ok,
    Warn,
    Fail,
}

impl CheckStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }
}

fn check_build(report: &mut DoctorReport) {
    let default = default_provider();
    let mut features = Vec::new();
    if cfg!(feature = "windowsml") {
        features.push("windowsml");
    }
    if cfg!(feature = "tensorrt") {
        features.push("tensorrt");
    }
    if cfg!(feature = "cuda") {
        features.push("cuda");
    }
    if features.is_empty() {
        features.push("none");
    }

    report.add(
        CheckStatus::Ok,
        "Build",
        format!(
            "provider features={} default-provider={}",
            features.join(","),
            default.label()
        ),
    );
}

#[cfg(all(windows, feature = "windowsml"))]
fn check_windows_ml(report: &mut DoctorReport) {
    use vc_core::windows_ml::{self, CatalogReadyState};

    match windows_ml::list_catalog_providers() {
        Ok(providers) => {
            report.add(
                CheckStatus::Ok,
                "Windows ML runtime",
                "Windows App SDK Runtime initialized; DirectML/CPU fallback is available",
            );

            let supported: Vec<_> = providers
                .iter()
                .filter(|provider| provider.vc_provider.is_some())
                .collect();
            if supported.is_empty() {
                report.add(
                    CheckStatus::Warn,
                    "Windows ML catalog EPs",
                    "no vc-rs-supported catalog EP is visible; --provider windowsml can still fall back to DirectML/CPU",
                );
                return;
            }

            let selected = windows_ml::select_best_catalog_provider(&providers);
            let mut details = Vec::new();
            for provider in &supported {
                let vc_name = provider
                    .vc_provider
                    .map(|provider| provider.vc_provider_name())
                    .unwrap_or("-");
                details.push(format!(
                    "{} status={} vc-provider={}",
                    provider.name,
                    provider.ready_state.label(),
                    vc_name
                ));
            }

            match selected.and_then(|selected| {
                supported
                    .iter()
                    .find(|provider| provider.vc_provider == Some(selected))
            }) {
                Some(provider) if provider.ready_state == CatalogReadyState::Ready => {
                    report.add_with_details(
                        CheckStatus::Ok,
                        "Windows ML catalog EPs",
                        format!(
                            "best supported EP is ready: {} ({})",
                            provider
                                .vc_provider
                                .map(|provider| provider.label())
                                .unwrap_or("unknown"),
                            provider
                                .vc_provider
                                .map(|provider| provider.vc_provider_name())
                                .unwrap_or("-")
                        ),
                        details,
                    );
                }
                Some(provider) => {
                    report.add_with_details(
                        CheckStatus::Warn,
                        "Windows ML catalog EPs",
                        format!(
                            "best supported EP is listed but not ready: {} status={}; run `vc-rs windowsml-eps install` if you want catalog acceleration",
                            provider.name,
                            provider.ready_state.label()
                        ),
                        details,
                    );
                }
                None => {
                    report.add_with_details(
                        CheckStatus::Warn,
                        "Windows ML catalog EPs",
                        "supported catalog EPs were listed but none matched vc-rs priority; --provider windowsml can still fall back to DirectML/CPU",
                        details,
                    );
                }
            }
        }
        Err(err) => {
            let status = if default_provider().is_windows_ml() {
                CheckStatus::Fail
            } else {
                CheckStatus::Warn
            };
            report.add_with_details(
                status,
                "Windows ML runtime",
                "failed to initialize Windows ML runtime",
                vec![
                    format!("{err:#}"),
                    "Install Windows App SDK Runtime 2.x, or use a non-WindowsML provider build."
                        .to_string(),
                ],
            );
        }
    }
}

#[cfg(not(all(windows, feature = "windowsml")))]
fn check_windows_ml(report: &mut DoctorReport) {
    report.add(
        CheckStatus::Ok,
        "Windows ML runtime",
        "windowsml provider is not built into this binary",
    );
}

fn check_tensorrt(report: &mut DoctorReport) {
    if !cfg!(feature = "tensorrt") {
        report.add(
            CheckStatus::Ok,
            "TensorRT runtime",
            "tensorrt provider is not built into this binary",
        );
        return;
    }

    let scan = scan_tensorrt_dlls_in_dirs(tensorrt_search_dirs());
    let mut details = Vec::new();
    details.push(format_search_result("nvinfer", scan.nvinfer.as_deref()));
    details.push(format_search_result(
        "nvinfer_plugin",
        scan.nvinfer_plugin.as_deref(),
    ));
    details.push(match scan.expected_cuda_major.as_deref() {
        Some(major) => format_search_result(&format!("cudart64_{major}"), scan.cudart.as_deref()),
        None => "cudart64: not checked because TensorRT major was not found".to_string(),
    });

    let runtime_ready =
        scan.nvinfer.is_some() && scan.nvinfer_plugin.is_some() && scan.cudart.is_some();
    let runtime_status = if runtime_ready {
        CheckStatus::Ok
    } else if default_provider() == Provider::TensorRt {
        CheckStatus::Fail
    } else {
        CheckStatus::Warn
    };

    let message = if runtime_ready {
        let major = scan.nvinfer_major.as_deref().unwrap_or("unknown");
        format!("required runtime DLLs are visible (TensorRT major {major})")
    } else {
        "required TensorRT runtime DLLs are missing; keep the TensorRT package DLLs beside vc-rs.exe or on PATH".to_string()
    };
    report.add_with_details(runtime_status, "TensorRT runtime", message, details);

    if runtime_ready {
        if scan.builder_resources.is_empty() {
            report.add(
                CheckStatus::Warn,
                "TensorRT builder resources",
                "no nvinfer_builder_resource DLL was found; first-run engine builds may fail unless the model is already cached",
            );
        } else {
            report.add(
                CheckStatus::Ok,
                "TensorRT builder resources",
                format!(
                    "{} builder resource DLL(s) visible for first-run engine builds",
                    scan.builder_resources.len()
                ),
            );
        }
    }
}

fn check_nvidia(report: &mut DoctorReport) {
    match Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader",
        ])
        .output()
    {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let gpus: Vec<String> = stdout
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToOwned::to_owned)
                .collect();
            if gpus.is_empty() {
                report.add(
                    CheckStatus::Warn,
                    "NVIDIA GPU",
                    "nvidia-smi ran but reported no GPU rows",
                );
            } else {
                report.add_with_details(
                    CheckStatus::Ok,
                    "NVIDIA GPU",
                    format!("nvidia-smi reported {} GPU(s)", gpus.len()),
                    gpus,
                );
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            report.add(
                CheckStatus::Warn,
                "NVIDIA GPU",
                format!(
                    "nvidia-smi exited with {}; TensorRT/CUDA providers may be unavailable{}",
                    output.status,
                    optional_command_stderr(&stderr)
                ),
            );
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            report.add(
                CheckStatus::Warn,
                "NVIDIA GPU",
                "nvidia-smi was not found on PATH; this is expected on non-NVIDIA systems",
            );
        }
        Err(err) => {
            report.add(
                CheckStatus::Warn,
                "NVIDIA GPU",
                format!("failed to run nvidia-smi: {err}"),
            );
        }
    }
}

fn check_engine_cache(report: &mut DoctorReport) {
    match model_rvc::engine_cache_info() {
        Ok(info) => report.add(CheckStatus::Ok, "Engine cache", format_engine_cache(&info)),
        Err(err) => report.add_with_details(
            CheckStatus::Warn,
            "Engine cache",
            "failed to inspect engine cache",
            vec![
                format!("{err:#}"),
                format!(
                    "Override the location with {}=<dir> if the default path is unusable.",
                    model_rvc::ENGINE_CACHE_DIR_ENV
                ),
            ],
        ),
    }
}

fn check_audio(report: &mut DoctorReport) {
    // Doctor only enumerates endpoints from the foreground command. It does not
    // open streams or touch the real-time callback path, so audio safety
    // invariants stay owned by the run command.
    let host = cpal::default_host();
    let input_count = match host.input_devices() {
        Ok(devices) => devices.count(),
        Err(err) => {
            report.add(
                CheckStatus::Warn,
                "Audio devices",
                format!("failed to enumerate CPAL input devices: {err}"),
            );
            return;
        }
    };
    let output_count = match host.output_devices() {
        Ok(devices) => devices.count(),
        Err(err) => {
            report.add(
                CheckStatus::Warn,
                "Audio devices",
                format!("failed to enumerate CPAL output devices: {err}"),
            );
            return;
        }
    };

    let input_default = host
        .default_input_device()
        .map(|device| device_name(&device))
        .unwrap_or_else(|| "<none>".to_string());
    let output_default = host
        .default_output_device()
        .map(|device| device_name(&device))
        .unwrap_or_else(|| "<none>".to_string());

    let status = if input_count > 0 && output_count > 0 {
        CheckStatus::Ok
    } else {
        CheckStatus::Warn
    };
    report.add(
        status,
        "Audio devices",
        format!(
            "CPAL inputs={} outputs={} default-input={} default-output={}",
            input_count, output_count, input_default, output_default
        ),
    );
}

fn device_name(device: &cpal::Device) -> String {
    device
        .description()
        .map(|description| description.name().to_string())
        .unwrap_or_else(|_| "<unknown>".to_string())
}

fn tensorrt_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.to_path_buf());
        }
    }
    if let Ok(current_dir) = env::current_dir() {
        dirs.push(current_dir);
    }
    if let Some(path) = env::var_os("PATH") {
        dirs.extend(env::split_paths(&path));
    }
    dedup_existing_dirs(dirs)
}

fn dedup_existing_dirs(dirs: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut seen = Vec::<String>::new();
    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        let key = dir.to_string_lossy().to_lowercase();
        if seen.iter().any(|seen| seen == &key) {
            continue;
        }
        seen.push(key);
        out.push(dir);
    }
    out
}

#[derive(Debug, Default)]
struct TensorRtDllScan {
    nvinfer: Option<PathBuf>,
    nvinfer_major: Option<String>,
    expected_cuda_major: Option<String>,
    nvinfer_plugin: Option<PathBuf>,
    cudart: Option<PathBuf>,
    builder_resources: Vec<PathBuf>,
}

fn scan_tensorrt_dlls_in_dirs(dirs: Vec<PathBuf>) -> TensorRtDllScan {
    // This is intentionally file-system-only. Loading TensorRT here would make a
    // diagnostic command mutate process-global CUDA/TensorRT state before the
    // user has selected a model or backend.
    let nvinfer = find_first_file(&dirs, parse_nvinfer_major);
    let nvinfer_major = nvinfer
        .as_deref()
        .and_then(|path| path.file_name())
        .and_then(parse_nvinfer_major);
    let nvinfer_plugin = match nvinfer_major.as_deref() {
        Some(major) => find_named_file(&dirs, &format!("nvinfer_plugin_{major}.dll")),
        None => find_first_file(&dirs, parse_nvinfer_plugin_major),
    };
    let expected_cuda_major = nvinfer_major
        .as_deref()
        .and_then(cuda_major_for_tensorrt_major);
    let cudart = match expected_cuda_major.as_deref() {
        Some(major) => find_named_file(&dirs, &format!("cudart64_{major}.dll")),
        None => None,
    };
    let builder_resources = find_builder_resource_files(&dirs, nvinfer_major.as_deref());

    TensorRtDllScan {
        nvinfer,
        nvinfer_major,
        expected_cuda_major,
        nvinfer_plugin,
        cudart,
        builder_resources,
    }
}

fn find_first_file(dirs: &[PathBuf], parse: fn(&OsStr) -> Option<String>) -> Option<PathBuf> {
    let mut best: Option<(u32, PathBuf)> = None;
    for dir in dirs {
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(version) = path.file_name().and_then(parse) else {
                continue;
            };
            let version = version.parse::<u32>().unwrap_or(0);
            if best
                .as_ref()
                .is_none_or(|(best_version, _)| version > *best_version)
            {
                best = Some((version, path));
            }
        }
    }
    best.map(|(_, path)| path)
}

fn find_named_file(dirs: &[PathBuf], file_name: &str) -> Option<PathBuf> {
    for dir in dirs {
        let candidate = dir.join(file_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn find_builder_resource_files(dirs: &[PathBuf], nvinfer_major: Option<&str>) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for dir in dirs {
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(OsStr::to_str) else {
                continue;
            };
            let name = name.to_ascii_lowercase();
            if !name.starts_with("nvinfer_builder_resource_") || !name.ends_with(".dll") {
                continue;
            }
            if let Some(major) = nvinfer_major {
                if !name.ends_with(&format!("_{major}.dll")) {
                    continue;
                }
            }
            files.push(path);
        }
    }
    files.sort();
    files
}

fn parse_nvinfer_major(name: &OsStr) -> Option<String> {
    let name = name.to_str()?.to_ascii_lowercase();
    let major = name.strip_prefix("nvinfer_")?.strip_suffix(".dll")?;
    major
        .chars()
        .all(|ch| ch.is_ascii_digit())
        .then(|| major.to_string())
}

fn parse_nvinfer_plugin_major(name: &OsStr) -> Option<String> {
    let name = name.to_str()?.to_ascii_lowercase();
    let major = name.strip_prefix("nvinfer_plugin_")?.strip_suffix(".dll")?;
    major
        .chars()
        .all(|ch| ch.is_ascii_digit())
        .then(|| major.to_string())
}

fn cuda_major_for_tensorrt_major(major: &str) -> Option<String> {
    let major = major.parse::<u32>().ok()?;
    let cuda_major = match major {
        10 => 12,
        11 => 13,
        other => other + 2,
    };
    Some(cuda_major.to_string())
}

fn format_search_result(label: &str, path: Option<&Path>) -> String {
    match path {
        Some(path) => format!("{label}: {}", path.display()),
        None => format!("{label}: not found"),
    }
}

fn format_engine_cache(info: &EngineCacheInfo) -> String {
    if !info.exists {
        return format!(
            "{} not created yet; override with {}=<dir>",
            info.root.display(),
            model_rvc::ENGINE_CACHE_DIR_ENV
        );
    }

    format!(
        "{} contains {} across {} file(s); clear manually with `vc-rs engine-cache clear` if needed",
        info.root.display(),
        format_bytes(info.size_bytes),
        info.file_count
    )
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

fn optional_command_stderr(stderr: &str) -> String {
    let stderr = stderr.trim();
    if stderr.is_empty() {
        String::new()
    } else {
        format!(" ({stderr})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_fails_only_when_fail_item_exists() {
        let mut report = DoctorReport::default();
        report.add(CheckStatus::Ok, "ok", "ok");
        report.add(CheckStatus::Warn, "warn", "warn");
        assert!(!report.has_failures());

        report.add(CheckStatus::Fail, "fail", "fail");
        assert!(report.has_failures());
    }

    #[test]
    fn scans_tensorrt_dlls_from_directory() {
        let temp = unique_temp_dir("vc-rs-doctor-trt");
        fs::create_dir_all(&temp).unwrap();
        fs::write(temp.join("nvinfer_11.dll"), []).unwrap();
        fs::write(temp.join("nvinfer_plugin_11.dll"), []).unwrap();
        fs::write(temp.join("cudart64_13.dll"), []).unwrap();
        fs::write(temp.join("nvinfer_builder_resource_sm86_11.dll"), []).unwrap();

        let scan = scan_tensorrt_dlls_in_dirs(vec![temp.clone()]);
        assert_eq!(scan.nvinfer_major.as_deref(), Some("11"));
        assert!(scan.nvinfer.unwrap().ends_with("nvinfer_11.dll"));
        assert!(scan
            .nvinfer_plugin
            .unwrap()
            .ends_with("nvinfer_plugin_11.dll"));
        assert!(scan.cudart.unwrap().ends_with("cudart64_13.dll"));
        assert_eq!(scan.builder_resources.len(), 1);

        fs::remove_dir_all(temp).unwrap();
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        env::temp_dir().join(format!(
            "{}-{}-{}",
            prefix,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
}
