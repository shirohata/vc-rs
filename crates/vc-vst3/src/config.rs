//! Plugin settings: model paths and conversion defaults.
//!
//! These persist in the plugin state (via `#[persist]` on the params), so the
//! host saves/restores them per project/preset. A TOML config file is still
//! supported as a headless seed for fresh instances (see [`PluginConfig::discover`]).
//! Field names and defaults mirror the CLI `Run` arguments so a working CLI
//! setup transfers directly.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use vc_core::Provider;

/// Search order for the config file:
/// 1. `VC_RS_VST3_CONFIG` environment variable (explicit path)
/// 2. `<os-config-dir>/vc-rs/vst3.toml` (see [`os_config_dir`])
/// 3. `vc-rs-vst3.toml` in the host's current working directory
pub const CONFIG_ENV: &str = "VC_RS_VST3_CONFIG";

#[derive(Debug, Clone, Serialize, Deserialize)]
// Lenient: unknown/legacy keys (e.g. `pitch_shift`, now a DAW parameter) are
// ignored rather than rejected, so older config files keep parsing.
#[serde(default)]
pub struct PluginConfig {
    pub model: PathBuf,
    pub embedder: PathBuf,
    pub f0_model: PathBuf,
    pub embedder_output: Option<String>,
    /// Legacy config key kept for lenient parsing only. TensorRT engine paths
    /// are no longer user-provided; native TensorRT builds cache entries from
    /// the ONNX model and fixed profile.
    pub rvc_engine: Option<PathBuf>,
    /// "cpu" | "windowsml*" | "cuda" | "tensorrt". GPU spellings resolve
    /// to whichever GPU-capable backend this package was built with (see
    /// [`PluginConfig::provider`]).
    pub provider: String,
    pub f0_threshold: f32,
    pub silence_threshold: f32,
    pub chunk_ms: u32,
    pub crossfade_ms: u32,
    pub sola_search_ms: u32,
    pub rvc_output_tail_discard_ms: u32,
    pub extra_convert_ms: u32,
    /// "sola" | "psola".
    pub smoother: String,
    pub volume_envelope: bool,
    pub rms_mix_rate: f32,
    pub auto_output_gain: bool,
    pub target_output_rms: f32,
    pub max_output_gain: f32,
}

impl Default for PluginConfig {
    fn default() -> Self {
        // Defaults track `crates/vc-cli/src/cli.rs` RunArgs.
        Self {
            model: PathBuf::new(),
            embedder: PathBuf::new(),
            f0_model: PathBuf::new(),
            embedder_output: None,
            rvc_engine: None,
            provider: default_provider().to_string(),
            f0_threshold: 0.3,
            silence_threshold: 0.0001,
            chunk_ms: 500,
            crossfade_ms: 85,
            sola_search_ms: 12,
            rvc_output_tail_discard_ms: 10,
            extra_convert_ms: 100,
            smoother: "sola".to_string(),
            volume_envelope: false,
            rms_mix_rate: 0.0,
            auto_output_gain: false,
            target_output_rms: 0.03,
            max_output_gain: 512.0,
        }
    }
}

impl PluginConfig {
    /// True when all three required model paths are set. When false the plugin
    /// runs in silent mode (the worker never loads a pipeline).
    pub fn has_models(&self) -> bool {
        !self.model.as_os_str().is_empty()
            && !self.embedder.as_os_str().is_empty()
            && !self.f0_model.as_os_str().is_empty()
    }

    pub fn provider(&self) -> Provider {
        match self.provider.trim().to_ascii_lowercase().as_str() {
            "windowsml" | "windows-ml" | "winml" => Provider::WindowsMl,
            "windowsml-cpu" | "windows-ml-cpu" | "winml-cpu" => Provider::WindowsMlCpu,
            "windowsml-directml"
            | "windows-ml-directml"
            | "winml-directml"
            | "windowsml-dml"
            | "winml-dml" => Provider::WindowsMlDirectMl,
            "windowsml-nvtrtx" | "windows-ml-nvtrtx" | "winml-nvtrtx" | "windowsml-tensorrt"
            | "winml-tensorrt" => Provider::WindowsMlNvTensorRtRtx,
            "windowsml-openvino" | "windows-ml-openvino" | "winml-openvino" => {
                Provider::WindowsMlOpenVino
            }
            "windowsml-qnn" | "windows-ml-qnn" | "winml-qnn" => Provider::WindowsMlQnn,
            "windowsml-migraphx" | "windows-ml-migraphx" | "winml-migraphx" => {
                Provider::WindowsMlMiGraphX
            }
            "windowsml-vitisai" | "windows-ml-vitisai" | "winml-vitisai" => {
                Provider::WindowsMlVitisAi
            }
            "cuda" => gpu_provider("cuda"),
            "tensorrt" | "trt" | "tensor-rt" => gpu_provider("tensorrt"),
            _ => Provider::Cpu,
        }
    }

    pub fn smoothing_kind(&self) -> vc_core::sola::SmoothingKind {
        match self.smoother.trim().to_ascii_lowercase().as_str() {
            "psola" => vc_core::sola::SmoothingKind::Psola,
            _ => vc_core::sola::SmoothingKind::Sola,
        }
    }

    /// Locate and parse the config file. Returns the default config when no file
    /// is found so the plugin still loads (in silent mode).
    pub fn discover() -> Self {
        let Some(path) = Self::config_path() else {
            return Self::default();
        };
        match std::fs::read_to_string(&path) {
            Ok(text) => match toml::from_str::<PluginConfig>(&text) {
                Ok(config) => {
                    nice_plug::nice_log!("vc-vst3: loaded config from {}", path.display());
                    config
                }
                Err(err) => {
                    nice_plug::nice_error!("vc-vst3: failed to parse {}: {err}", path.display());
                    Self::default()
                }
            },
            Err(err) => {
                nice_plug::nice_error!("vc-vst3: failed to read {}: {err}", path.display());
                Self::default()
            }
        }
    }

    fn config_path() -> Option<PathBuf> {
        if let Some(explicit) = std::env::var_os(CONFIG_ENV) {
            let path = PathBuf::from(explicit);
            if path.is_file() {
                return Some(path);
            }
        }
        if let Some(dir) = os_config_dir() {
            let path = dir.join("vc-rs").join("vst3.toml");
            if path.is_file() {
                return Some(path);
            }
        }
        let cwd = std::env::current_dir().ok()?.join("vc-rs-vst3.toml");
        cwd.is_file().then_some(cwd)
    }
}

// The default provider tracks the backend this package was built with, so a
// fresh instance is usable without first opening the GUI to pick one. The
// variants are mutually exclusive by cargo feature (one per distributed
// package); a CPU-only build with none of them falls back to "cpu".
#[cfg(feature = "windowsml")]
fn default_provider() -> &'static str {
    "windowsml"
}

#[cfg(all(feature = "tensorrt", not(feature = "windowsml")))]
fn default_provider() -> &'static str {
    "tensorrt"
}

#[cfg(all(
    feature = "cuda",
    not(feature = "windowsml"),
    not(feature = "tensorrt")
))]
fn default_provider() -> &'static str {
    "cuda"
}

#[cfg(not(any(feature = "windowsml", feature = "tensorrt", feature = "cuda")))]
fn default_provider() -> &'static str {
    "cpu"
}

/// Resolve a requested GPU provider ("cuda" or "tensorrt") to the GPU backend
/// this package was compiled with. The variants are mutually exclusive by cargo
/// feature, so each build sees exactly one of these.
#[cfg(feature = "tensorrt")]
fn gpu_provider(requested: &str) -> Provider {
    if requested != "tensorrt" {
        nice_plug::nice_warn!(
            "vc-vst3: '{requested}' provider is not enabled in this package; using TensorRT"
        );
    }
    Provider::TensorRt
}

#[cfg(all(feature = "cuda", not(feature = "tensorrt")))]
fn gpu_provider(requested: &str) -> Provider {
    if requested != "cuda" {
        nice_plug::nice_warn!(
            "vc-vst3: '{requested}' provider is not enabled in this package; using CUDA"
        );
    }
    Provider::Cuda
}

#[cfg(all(
    feature = "windowsml",
    not(any(feature = "cuda", feature = "tensorrt"))
))]
fn gpu_provider(requested: &str) -> Provider {
    nice_plug::nice_warn!(
        "vc-vst3: '{requested}' provider is not enabled in this package; using Windows ML"
    );
    Provider::WindowsMl
}

#[cfg(not(any(feature = "cuda", feature = "tensorrt", feature = "windowsml")))]
fn gpu_provider(requested: &str) -> Provider {
    nice_plug::nice_warn!(
        "vc-vst3: '{requested}' provider is not enabled in this CPU-only package; using CPU"
    );
    Provider::Cpu
}

/// The per-user config directory for the current OS:
/// `%APPDATA%` on Windows, `$XDG_CONFIG_HOME` (or `$HOME/.config`) elsewhere.
#[cfg(windows)]
fn os_config_dir() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(PathBuf::from)
}

#[cfg(not(windows))]
fn os_config_dir() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(xdg));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config"))
}
