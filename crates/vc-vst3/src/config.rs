//! Plugin settings: model paths and conversion defaults.
//!
//! These persist in the plugin state (via `#[persist]` on the params), so the
//! host saves/restores them per project/preset. A TOML config file is still
//! supported as a headless seed for fresh instances (see [`PluginConfig::discover`]).
//! Field names and defaults mirror the CLI `Run` arguments so a working CLI
//! setup transfers directly.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use vc_core::model_rvc::GpuPriority;
use vc_core::validation::{
    validate_conversion_timing, validate_non_negative_f32, validate_unit_interval,
    ConversionTiming, ConversionTimingLimits, CONVERSION_TIMING_LIMITS,
};
use vc_core::Provider;

/// Search order for the config file:
/// 1. `VC_RS_VST3_CONFIG` environment variable (explicit path)
/// 2. `<os-config-dir>/vc-rs/vst3.toml` (see [`os_config_dir`])
/// 3. `vc-rs-vst3.toml` in the host's current working directory
pub const CONFIG_ENV: &str = "VC_RS_VST3_CONFIG";
pub const PLUGIN_MIN_EXTRA_CONVERT_MS: u32 = 100;

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
    /// "high" | "normal". Only native TensorRT consumes this setting.
    pub gpu_priority: String,
    /// CUDA device ID used by CUDA and native TensorRT backends.
    pub gpu_device_id: u32,
    pub f0_threshold: f32,
    pub silence_threshold: f32,
    /// Noise gate attack/release/floor. Static (applied at Load/Reload); the
    /// gate's on/off and threshold are DAW parameters (see `VcRvcParams`).
    pub noise_gate_attack_ms: f32,
    pub noise_gate_release_ms: f32,
    pub noise_gate_floor: f32,
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
            gpu_priority: "high".to_string(),
            gpu_device_id: 0,
            f0_threshold: 0.3,
            silence_threshold: 0.0001,
            noise_gate_attack_ms: 5.0,
            noise_gate_release_ms: 50.0,
            noise_gate_floor: 0.0,
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
        // Shared parser (canonical names + aliases); unknown spellings fall back
        // to CPU as before. Native GPU spellings ("cuda"/"tensorrt") still route
        // through `gpu_provider` so they resolve to whichever GPU backend this
        // package was compiled with; the windowsml catalog EPs pass through.
        match Provider::from_name(&self.provider) {
            Some(provider) if provider.is_cuda() || provider.is_tensorrt() => {
                gpu_provider(provider.label())
            }
            Some(provider) => provider,
            None => Provider::Cpu,
        }
    }

    pub fn smoothing_kind(&self) -> vc_core::sola::SmoothingKind {
        match self.smoother.trim().to_ascii_lowercase().as_str() {
            "psola" => vc_core::sola::SmoothingKind::Psola,
            _ => vc_core::sola::SmoothingKind::Sola,
        }
    }

    pub fn gpu_priority(&self) -> GpuPriority {
        match self.gpu_priority.trim().to_ascii_lowercase().as_str() {
            "normal" => GpuPriority::Normal,
            _ => GpuPriority::High,
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let limits = ConversionTimingLimits {
            min_extra_convert_ms: PLUGIN_MIN_EXTRA_CONVERT_MS,
            ..CONVERSION_TIMING_LIMITS
        };
        validate_conversion_timing(
            ConversionTiming {
                chunk_ms: self.chunk_ms,
                crossfade_ms: self.crossfade_ms,
                sola_search_ms: self.sola_search_ms,
                tail_discard_ms: self.rvc_output_tail_discard_ms,
                extra_convert_ms: self.extra_convert_ms,
            },
            limits,
        )?;
        validate_non_negative_f32("F0 threshold", self.f0_threshold)?;
        validate_non_negative_f32("silence threshold", self.silence_threshold)?;
        validate_non_negative_f32("noise gate attack (ms)", self.noise_gate_attack_ms)?;
        validate_non_negative_f32("noise gate release (ms)", self.noise_gate_release_ms)?;
        validate_unit_interval("noise gate floor", self.noise_gate_floor)?;
        validate_unit_interval("RMS mix rate", self.rms_mix_rate)?;
        validate_non_negative_f32("target output RMS", self.target_output_rms)?;
        validate_non_negative_f32("max output gain", self.max_output_gain)?;
        Ok(())
    }

    /// Locate and parse the config file. Returns the default config when no file
    /// is found so the plugin still loads (in silent mode).
    pub fn discover() -> Self {
        let Some(path) = Self::config_path() else {
            return Self::default();
        };
        match std::fs::read_to_string(&path) {
            Ok(text) => match toml::from_str::<PluginConfig>(&text) {
                Ok(config) => match config.validate() {
                    Ok(()) => {
                        nice_plug::nice_log!("vc-vst3: loaded config from {}", path.display());
                        config
                    }
                    Err(err) => {
                        nice_plug::nice_error!("vc-vst3: invalid config {}: {err}", path.display());
                        Self::default()
                    }
                },
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
// fresh instance is usable without first opening the GUI to pick one. Shared
// with the CLI/GUI via `vc_core::default_provider`; the variants are mutually
// exclusive by cargo feature (one per distributed package), and a CPU-only
// build with none of them falls back to "cpu".
fn default_provider() -> &'static str {
    vc_core::default_provider().label()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_priority_defaults_high_and_parses_normal() {
        assert_eq!(PluginConfig::default().gpu_priority(), GpuPriority::High);
        assert_eq!(PluginConfig::default().gpu_device_id, 0);
        let config: PluginConfig = toml::from_str("gpu_priority = \"normal\"").unwrap();
        assert_eq!(config.gpu_priority(), GpuPriority::Normal);
        let config: PluginConfig = toml::from_str("gpu_device_id = 2").unwrap();
        assert_eq!(config.gpu_device_id, 2);
    }

    #[test]
    fn validates_timing_ranges() {
        assert!(PluginConfig::default().validate().is_ok());
        let config: PluginConfig = toml::from_str("chunk_ms = 19").unwrap();
        assert!(config.validate().is_err());
        let config: PluginConfig = toml::from_str(&format!(
            "extra_convert_ms = {}",
            PLUGIN_MIN_EXTRA_CONVERT_MS - 1
        ))
        .unwrap();
        assert!(config.validate().is_err());
        let config: PluginConfig = toml::from_str("extra_convert_ms = 3001").unwrap();
        assert!(config.validate().is_err());
    }
}
