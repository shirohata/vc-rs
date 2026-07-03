use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use vc_core::validation::{
    validate_conversion_timing, validate_finite_f32, validate_non_negative_f32,
    validate_unit_interval, ConversionTiming, CONVERSION_TIMING_LIMITS,
};
pub use vc_core::Provider;

/// Default inference backend for this build. Delegates to the shared
/// `vc_core::default_provider` so the CLI, GUI, and VST3 agree: a
/// single-provider distribution package defaults to its one backend, while the
/// combined dev binary and the CPU-only build fall back to `cpu`.
pub fn default_provider() -> Provider {
    vc_core::default_provider()
}

pub const DEFAULT_CROSSFADE_MS: u32 = 85;
pub const DEFAULT_EXTRA_CONVERT_MS: u32 = 100;
pub const DEFAULT_INPUT_GAIN: f32 = 1.0;
pub const DEFAULT_NOISE_GATE_THRESHOLD: f32 = 0.01;
pub const DEFAULT_NOISE_GATE_ATTACK_MS: f32 = 5.0;
pub const DEFAULT_NOISE_GATE_RELEASE_MS: f32 = 50.0;
pub const DEFAULT_NOISE_GATE_FLOOR: f32 = 0.0;
pub const DEFAULT_MAX_OUTPUT_GAIN: f32 = 512.0;
pub const DEFAULT_OUTPUT_GAIN: f32 = 1.0;
pub const DEFAULT_RMS_MIX_RATE: f32 = 0.0;
pub const DEFAULT_RT_CHUNK_MS: u32 = 500;
pub const DEFAULT_WAV_CHUNK_MS: u32 = 2000;
pub const DEFAULT_RVC_OUTPUT_TAIL_DISCARD_MS: u32 = 10;
pub const DEFAULT_SOLA_SEARCH_MS: u32 = 12;
pub const DEFAULT_TARGET_OUTPUT_RMS: f32 = 0.03;
pub const DEFAULT_VOLUME_ENVELOPE: bool = false;
pub const DEFAULT_WASAPI_BUFFER_MS: u32 = 0;

#[derive(Debug, Parser)]
#[command(
    name = "vc-rs",
    version,
    about = "Minimal Rust voice changer prototype"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Diagnose runtime dependencies and local device visibility.
    Doctor,
    /// List audio input and output devices.
    Devices(DevicesArgs),
    /// Inspect ONNX model inputs, outputs, and metadata.
    Inspect(InspectArgs),
    /// List or install Windows ML catalog execution providers.
    #[cfg(all(windows, feature = "windowsml"))]
    #[command(name = "windowsml-eps", alias = "windows-ml-eps", alias = "winml-eps")]
    WindowsMlEps(WindowsMlEpsArgs),
    /// Inspect or clear the on-disk GPU engine cache (TensorRT / Windows ML
    /// TensorRT-RTX).
    #[command(name = "engine-cache", alias = "cache", alias = "trt-cache")]
    EngineCache(EngineCacheArgs),
    /// Run realtime microphone-to-speaker conversion.
    Run(RunArgs),
    /// Run conversion against a wav file for model/DSP verification.
    Wav(WavArgs),
}

#[derive(Debug, Parser)]
pub struct EngineCacheArgs {
    #[command(subcommand)]
    pub command: EngineCacheCommand,
}

#[derive(Debug, Subcommand)]
pub enum EngineCacheCommand {
    /// Show the cache location and size (per-model breakdown).
    #[command(alias = "show", alias = "list")]
    Info,
    /// Delete all cached engines (they rebuild on the next model load).
    #[command(alias = "clean")]
    Clear(EngineCacheClearArgs),
}

#[derive(Debug, Parser)]
pub struct EngineCacheClearArgs {
    /// Skip the confirmation prompt before deleting.
    #[arg(long)]
    pub yes: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Smoother {
    Sola,
    Psola,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Denoiser {
    Off,
    NoiseGate,
    Rnnoise,
    Gtcrn,
}

impl From<Denoiser> for vc_app::DenoiserMode {
    fn from(value: Denoiser) -> Self {
        match value {
            Denoiser::Off => Self::Off,
            Denoiser::NoiseGate => Self::NoiseGate,
            Denoiser::Rnnoise => Self::Rnnoise,
            Denoiser::Gtcrn => Self::Gtcrn,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum GpuPriority {
    Normal,
    #[default]
    High,
}

impl From<GpuPriority> for vc_core::model_rvc::GpuPriority {
    fn from(value: GpuPriority) -> Self {
        match value {
            GpuPriority::Normal => Self::Normal,
            GpuPriority::High => Self::High,
        }
    }
}

// OS audio host, tokens aligned with cpal's HostId. Selecting one unavailable on
// the current platform/build (e.g. `asio` without `--features asio`, or
// `coreaudio` on Windows) errors at open/list time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum AudioHost {
    Wasapi,
    Asio,
    #[value(name = "coreaudio")]
    CoreAudio,
    Alsa,
    Jack,
}

impl From<AudioHost> for vc_app::AudioHost {
    fn from(value: AudioHost) -> Self {
        match value {
            AudioHost::Wasapi => Self::Wasapi,
            AudioHost::Asio => Self::Asio,
            AudioHost::CoreAudio => Self::CoreAudio,
            AudioHost::Alsa => Self::Alsa,
            AudioHost::Jack => Self::Jack,
        }
    }
}

impl From<Smoother> for vc_app::Smoother {
    fn from(value: Smoother) -> Self {
        match value {
            Smoother::Sola => Self::Sola,
            Smoother::Psola => Self::Psola,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DeviceHost {
    All,
    Wasapi,
    Asio,
    #[value(name = "coreaudio")]
    CoreAudio,
    Alsa,
    Jack,
}

#[derive(Debug, Parser)]
pub struct DevicesArgs {
    #[arg(long, value_enum, default_value_t = DeviceHost::All)]
    pub audio_backend: DeviceHost,
}

#[derive(Debug, Parser)]
pub struct RunArgs {
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[arg(long)]
    pub embedder: Option<PathBuf>,
    #[arg(long)]
    pub embedder_output: Option<String>,
    #[arg(long)]
    pub f0_model: Option<PathBuf>,
    #[arg(long)]
    pub input: Option<String>,
    #[arg(long)]
    pub output: Option<String>,
    #[arg(
        long,
        value_enum,
        help = "Audio backend for both directions (default: platform default); override per direction with --input-backend/--output-backend"
    )]
    pub audio_backend: Option<AudioHost>,
    #[arg(
        long,
        value_enum,
        help = "Input audio backend; defaults to --audio-backend"
    )]
    pub input_backend: Option<AudioHost>,
    #[arg(
        long,
        value_enum,
        help = "Output audio backend; defaults to --audio-backend"
    )]
    pub output_backend: Option<AudioHost>,
    #[arg(long)]
    pub wasapi_exclusive: bool,
    #[arg(long)]
    pub wasapi_exclusive_input: bool,
    #[arg(long)]
    pub wasapi_exclusive_output: bool,
    #[arg(
        long,
        default_value_t = DEFAULT_WASAPI_BUFFER_MS,
        help = "WASAPI event buffer in milliseconds; 0 uses the device minimum period"
    )]
    pub wasapi_buffer_ms: u32,
    #[arg(long, default_value_t = DEFAULT_RT_CHUNK_MS)]
    pub chunk_ms: u32,
    #[arg(long, default_value_t = DEFAULT_CROSSFADE_MS)]
    pub crossfade_ms: u32,
    #[arg(long, default_value_t = DEFAULT_SOLA_SEARCH_MS)]
    pub sola_search_ms: u32,
    #[arg(long, value_enum, default_value_t = Smoother::Sola)]
    pub smoother: Smoother,
    #[arg(long, default_value_t = DEFAULT_RVC_OUTPUT_TAIL_DISCARD_MS)]
    pub rvc_output_tail_discard_ms: u32,
    #[arg(
        long,
        default_value_t = DEFAULT_EXTRA_CONVERT_MS,
        help = "Additional RVC conversion context in milliseconds"
    )]
    pub extra_convert_ms: u32,
    #[arg(long, default_value_t = 0.0001)]
    pub silence_threshold: f32,
    #[arg(long, value_enum, default_value_t = default_provider())]
    pub provider: Provider,
    #[arg(long, value_enum, default_value_t = GpuPriority::High)]
    pub gpu_priority: GpuPriority,
    #[arg(long, default_value_t = 0, help = "CUDA device ID for CUDA/TensorRT")]
    pub gpu_device_id: u32,
    #[arg(long, default_value_t = 0)]
    pub speaker_id: i64,
    #[arg(long, default_value_t = 0.0)]
    pub pitch_shift: f32,
    #[arg(long, default_value_t = 0.3)]
    pub f0_threshold: f32,
    #[arg(long, default_value_t = DEFAULT_INPUT_GAIN)]
    pub input_gain: f32,
    #[arg(long, default_value_t = DEFAULT_OUTPUT_GAIN)]
    pub output_gain: f32,
    #[arg(
        long,
        value_enum,
        help = "Input denoiser: off, noise-gate, rnnoise, or gtcrn"
    )]
    pub denoiser: Option<Denoiser>,
    #[arg(
        long = "gtcrn-model",
        value_name = "DIR",
        help = "GTCRN model directory holding gtcrn_stream.onnx (required for --denoiser gtcrn)"
    )]
    pub gtcrn_model: Option<PathBuf>,
    #[arg(long = "noise-gate", conflicts_with = "denoiser", action = clap::ArgAction::SetTrue, help = "Deprecated alias for --denoiser noise-gate")]
    pub noise_gate: bool,
    #[arg(
        long,
        default_value_t = DEFAULT_NOISE_GATE_THRESHOLD,
        help = "Noise gate threshold (linear amplitude; signal below this is attenuated)"
    )]
    pub noise_gate_threshold: f32,
    #[arg(long, default_value_t = DEFAULT_NOISE_GATE_ATTACK_MS, help = "Noise gate attack time (ms)")]
    pub noise_gate_attack_ms: f32,
    #[arg(long, default_value_t = DEFAULT_NOISE_GATE_RELEASE_MS, help = "Noise gate release time (ms)")]
    pub noise_gate_release_ms: f32,
    #[arg(long, default_value_t = DEFAULT_NOISE_GATE_FLOOR, help = "Noise gate floor gain when closed (0.0..=1.0)")]
    pub noise_gate_floor: f32,
    #[arg(long = "volume-envelope", action = clap::ArgAction::SetTrue, default_value_t = DEFAULT_VOLUME_ENVELOPE)]
    pub volume_envelope: bool,
    #[arg(long, default_value_t = DEFAULT_RMS_MIX_RATE, value_parser = parse_unit_f32)]
    pub rms_mix_rate: f32,
    #[arg(long)]
    pub auto_output_gain: bool,
    #[arg(long, default_value_t = DEFAULT_TARGET_OUTPUT_RMS)]
    pub target_output_rms: f32,
    #[arg(long, default_value_t = DEFAULT_MAX_OUTPUT_GAIN)]
    pub max_output_gain: f32,
    #[arg(long)]
    pub debug_output_wav: Option<PathBuf>,
    #[arg(long)]
    pub debug_input_wav: Option<PathBuf>,
    #[arg(long)]
    pub duration_seconds: Option<u64>,
    #[arg(long)]
    pub passthrough: bool,
}

#[derive(Debug, Parser)]
pub struct WavArgs {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub embedder: PathBuf,
    #[arg(long)]
    pub embedder_output: Option<String>,
    #[arg(long)]
    pub f0_model: PathBuf,
    #[arg(long)]
    pub input: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = DEFAULT_WAV_CHUNK_MS)]
    pub chunk_ms: u32,
    #[arg(long, value_enum, default_value_t = Smoother::Sola)]
    pub smoother: Smoother,
    // Exposed so `--join-report` can sweep the SOLA/crossfade window offline;
    // mirrors the realtime command's flags. The 3/4-of-chunk cap in
    // `sola::model_domain_crossfade_samples` still applies on top of this.
    #[arg(long, default_value_t = DEFAULT_CROSSFADE_MS)]
    pub crossfade_ms: u32,
    #[arg(long, default_value_t = DEFAULT_SOLA_SEARCH_MS)]
    pub sola_search_ms: u32,
    #[arg(long, default_value_t = DEFAULT_RVC_OUTPUT_TAIL_DISCARD_MS)]
    pub rvc_output_tail_discard_ms: u32,
    #[arg(long, value_enum, default_value_t = default_provider())]
    pub provider: Provider,
    #[arg(long, value_enum, default_value_t = GpuPriority::High)]
    pub gpu_priority: GpuPriority,
    #[arg(long, default_value_t = 0, help = "CUDA device ID for CUDA/TensorRT")]
    pub gpu_device_id: u32,
    #[arg(long, default_value_t = 0)]
    pub speaker_id: i64,
    #[arg(long, default_value_t = 0.0)]
    pub pitch_shift: f32,
    #[arg(long, default_value_t = 0.3)]
    pub f0_threshold: f32,
    #[arg(long, default_value_t = DEFAULT_INPUT_GAIN)]
    pub input_gain: f32,
    #[arg(long, default_value_t = DEFAULT_OUTPUT_GAIN)]
    pub output_gain: f32,
    #[arg(
        long,
        value_enum,
        help = "Input denoiser: off, noise-gate, rnnoise, or gtcrn"
    )]
    pub denoiser: Option<Denoiser>,
    #[arg(
        long = "gtcrn-model",
        value_name = "DIR",
        help = "GTCRN model directory holding gtcrn_stream.onnx (required for --denoiser gtcrn)"
    )]
    pub gtcrn_model: Option<PathBuf>,
    #[arg(long = "noise-gate", conflicts_with = "denoiser", action = clap::ArgAction::SetTrue, help = "Deprecated alias for --denoiser noise-gate")]
    pub noise_gate: bool,
    #[arg(
        long,
        default_value_t = DEFAULT_NOISE_GATE_THRESHOLD,
        help = "Noise gate threshold (linear amplitude; signal below this is attenuated)"
    )]
    pub noise_gate_threshold: f32,
    #[arg(long, default_value_t = DEFAULT_NOISE_GATE_ATTACK_MS, help = "Noise gate attack time (ms)")]
    pub noise_gate_attack_ms: f32,
    #[arg(long, default_value_t = DEFAULT_NOISE_GATE_RELEASE_MS, help = "Noise gate release time (ms)")]
    pub noise_gate_release_ms: f32,
    #[arg(long, default_value_t = DEFAULT_NOISE_GATE_FLOOR, help = "Noise gate floor gain when closed (0.0..=1.0)")]
    pub noise_gate_floor: f32,
    #[arg(
        long,
        default_value_t = DEFAULT_EXTRA_CONVERT_MS,
        help = "Additional RVC conversion context in milliseconds"
    )]
    pub extra_convert_ms: u32,
    #[arg(long = "volume-envelope", action = clap::ArgAction::SetTrue, default_value_t = DEFAULT_VOLUME_ENVELOPE)]
    pub volume_envelope: bool,
    #[arg(long, default_value_t = DEFAULT_RMS_MIX_RATE, value_parser = parse_unit_f32)]
    pub rms_mix_rate: f32,
    #[arg(long)]
    pub auto_output_gain: bool,
    #[arg(long, default_value_t = DEFAULT_TARGET_OUTPUT_RMS)]
    pub target_output_rms: f32,
    #[arg(long, default_value_t = DEFAULT_MAX_OUTPUT_GAIN)]
    pub max_output_gain: f32,
    #[arg(
        long,
        value_name = "PATH",
        help = "Write a per-chunk SOLA/PSOLA join report (CSV) for offline seam analysis"
    )]
    pub join_report: Option<PathBuf>,
}

#[derive(Debug, Parser)]
pub struct InspectArgs {
    #[arg(long)]
    pub model: PathBuf,
}

#[cfg(all(windows, feature = "windowsml"))]
#[derive(Debug, Parser)]
pub struct WindowsMlEpsArgs {
    #[command(subcommand)]
    pub command: WindowsMlEpsCommand,
}

#[cfg(all(windows, feature = "windowsml"))]
#[derive(Debug, Subcommand)]
pub enum WindowsMlEpsCommand {
    /// List Windows ML catalog EPs visible on this device.
    List,
    /// Download/install or prepare a Windows ML catalog EP.
    Install(WindowsMlEpsInstallArgs),
}

#[cfg(all(windows, feature = "windowsml"))]
#[derive(Debug, Parser)]
pub struct WindowsMlEpsInstallArgs {
    /// EP to install. Omit to select the best compatible EP by vc-rs priority.
    #[arg(long, value_enum)]
    pub provider: Option<WindowsMlEpProvider>,
    /// Skip the confirmation prompt before downloading/installing.
    #[arg(long)]
    pub yes: bool,
}

#[cfg(all(windows, feature = "windowsml"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum WindowsMlEpProvider {
    #[value(alias = "tensorrt", alias = "windowsml-nvtrtx", alias = "winml-nvtrtx")]
    Nvtrtx,
    #[value(alias = "windowsml-qnn", alias = "winml-qnn")]
    Qnn,
    #[value(alias = "windowsml-openvino", alias = "winml-openvino")]
    Openvino,
    #[value(alias = "windowsml-migraphx", alias = "winml-migraphx")]
    Migraphx,
    #[value(alias = "windowsml-vitisai", alias = "winml-vitisai")]
    Vitisai,
}

impl Cli {
    pub fn parse_args() -> Self {
        Self::parse()
    }
}

fn parse_unit_f32(value: &str) -> Result<f32, String> {
    let value = value
        .parse::<f32>()
        .map_err(|err| format!("invalid float value: {err}"))?;
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("value must be a finite number in 0.0..=1.0".to_string())
    }
}

impl RunArgs {
    pub fn denoiser_mode(&self) -> Denoiser {
        self.denoiser.unwrap_or(if self.noise_gate {
            Denoiser::NoiseGate
        } else {
            Denoiser::Off
        })
    }

    // Effective per-direction hosts: --input-backend/--output-backend override the
    // --audio-backend default, which applies to both directions otherwise; an
    // entirely unset host falls back to the platform default.
    pub fn effective_input_host(&self) -> vc_app::AudioHost {
        self.input_backend
            .or(self.audio_backend)
            .map(Into::into)
            .unwrap_or_default()
    }

    pub fn effective_output_host(&self) -> vc_app::AudioHost {
        self.output_backend
            .or(self.audio_backend)
            .map(Into::into)
            .unwrap_or_default()
    }

    pub fn validate_audio_options(&self) -> Result<(), String> {
        if (self.wasapi_exclusive || self.wasapi_exclusive_input)
            && self.effective_input_host() != vc_app::AudioHost::Wasapi
        {
            return Err(
                "--wasapi-exclusive/--wasapi-exclusive-input require the WASAPI input host"
                    .to_string(),
            );
        }
        if (self.wasapi_exclusive || self.wasapi_exclusive_output)
            && self.effective_output_host() != vc_app::AudioHost::Wasapi
        {
            return Err(
                "--wasapi-exclusive/--wasapi-exclusive-output require the WASAPI output host"
                    .to_string(),
            );
        }
        Ok(())
    }

    pub fn validate_conversion_options(&self) -> Result<(), String> {
        validate_common_conversion_options(
            ConversionTiming {
                chunk_ms: self.chunk_ms,
                crossfade_ms: self.crossfade_ms,
                sola_search_ms: self.sola_search_ms,
                tail_discard_ms: self.rvc_output_tail_discard_ms,
                extra_convert_ms: self.extra_convert_ms,
            },
            CONVERSION_TIMING_LIMITS,
            self.noise_gate_attack_ms,
            self.noise_gate_release_ms,
            self.noise_gate_floor,
            self.rms_mix_rate,
            self.f0_threshold,
            self.silence_threshold,
            self.target_output_rms,
            self.max_output_gain,
        )
        .and_then(|()| validate_live_conversion_options(self))
    }

    pub fn wasapi_input_exclusive(&self) -> bool {
        self.wasapi_exclusive || self.wasapi_exclusive_input
    }

    pub fn wasapi_output_exclusive(&self) -> bool {
        self.wasapi_exclusive || self.wasapi_exclusive_output
    }
}

impl WavArgs {
    pub fn denoiser_mode(&self) -> Denoiser {
        self.denoiser.unwrap_or(if self.noise_gate {
            Denoiser::NoiseGate
        } else {
            Denoiser::Off
        })
    }

    pub fn validate_conversion_options(&self) -> Result<(), String> {
        validate_common_conversion_options(
            ConversionTiming {
                chunk_ms: self.chunk_ms,
                crossfade_ms: self.crossfade_ms,
                sola_search_ms: self.sola_search_ms,
                tail_discard_ms: self.rvc_output_tail_discard_ms,
                extra_convert_ms: self.extra_convert_ms,
            },
            CONVERSION_TIMING_LIMITS,
            self.noise_gate_attack_ms,
            self.noise_gate_release_ms,
            self.noise_gate_floor,
            self.rms_mix_rate,
            self.f0_threshold,
            0.0,
            self.target_output_rms,
            self.max_output_gain,
        )
        .and_then(|()| validate_wav_live_conversion_options(self))
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_common_conversion_options(
    timing: ConversionTiming,
    limits: vc_core::validation::ConversionTimingLimits,
    noise_gate_attack_ms: f32,
    noise_gate_release_ms: f32,
    noise_gate_floor: f32,
    rms_mix_rate: f32,
    f0_threshold: f32,
    silence_threshold: f32,
    target_output_rms: f32,
    max_output_gain: f32,
) -> Result<(), String> {
    validate_conversion_timing(timing, limits).map_err(|err| err.to_string())?;
    validate_non_negative_f32("F0 threshold", f0_threshold).map_err(|err| err.to_string())?;
    validate_non_negative_f32("silence threshold", silence_threshold)
        .map_err(|err| err.to_string())?;
    validate_non_negative_f32("noise gate attack (ms)", noise_gate_attack_ms)
        .map_err(|err| err.to_string())?;
    validate_non_negative_f32("noise gate release (ms)", noise_gate_release_ms)
        .map_err(|err| err.to_string())?;
    validate_unit_interval("noise gate floor", noise_gate_floor).map_err(|err| err.to_string())?;
    validate_unit_interval("RMS mix rate", rms_mix_rate).map_err(|err| err.to_string())?;
    validate_non_negative_f32("target output RMS", target_output_rms)
        .map_err(|err| err.to_string())?;
    validate_non_negative_f32("max output gain", max_output_gain).map_err(|err| err.to_string())?;
    Ok(())
}

fn validate_live_values(
    pitch_shift: f32,
    input_gain: f32,
    output_gain: f32,
    noise_gate_threshold: f32,
) -> Result<(), String> {
    validate_finite_f32("pitch shift", pitch_shift).map_err(|err| err.to_string())?;
    validate_non_negative_f32("input gain", input_gain).map_err(|err| err.to_string())?;
    validate_non_negative_f32("output gain", output_gain).map_err(|err| err.to_string())?;
    validate_non_negative_f32("noise gate threshold", noise_gate_threshold)
        .map_err(|err| err.to_string())?;
    Ok(())
}

fn validate_live_conversion_options(args: &RunArgs) -> Result<(), String> {
    validate_live_values(
        args.pitch_shift,
        args.input_gain,
        args.output_gain,
        args.noise_gate_threshold,
    )
}

fn validate_wav_live_conversion_options(args: &WavArgs) -> Result<(), String> {
    validate_live_values(
        args.pitch_shift,
        args.input_gain,
        args.output_gain,
        args.noise_gate_threshold,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_defaults_to_platform_host() {
        let cli = Cli::try_parse_from(["vc-rs", "run", "--passthrough"]).unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };

        // No backend flags → unset, resolving to the platform default host.
        assert_eq!(args.audio_backend, None);
        assert_eq!(args.effective_input_host(), vc_app::AudioHost::default());
        assert_eq!(args.effective_output_host(), vc_app::AudioHost::default());
        assert!(!args.wasapi_exclusive);
        assert!(!args.wasapi_input_exclusive());
        assert!(!args.wasapi_output_exclusive());
        assert_eq!(args.wasapi_buffer_ms, DEFAULT_WASAPI_BUFFER_MS);
        assert_eq!(args.rms_mix_rate, 0.0);
        assert_eq!(args.smoother, Smoother::Sola);
        assert_eq!(
            args.rvc_output_tail_discard_ms,
            DEFAULT_RVC_OUTPUT_TAIL_DISCARD_MS
        );
        assert_eq!(args.extra_convert_ms, DEFAULT_EXTRA_CONVERT_MS);
        assert!(args.validate_audio_options().is_ok());
    }

    #[test]
    fn rejects_removed_chunking_options() {
        assert!(
            Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--infer-chunks", "2"]).is_err()
        );
        assert!(
            Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--prefill-ms", "20"]).is_err()
        );
        assert!(Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
            "--infer-chunks",
            "2",
        ])
        .is_err());
    }

    #[test]
    fn rejects_removed_wav_realtime_command() {
        let err = Cli::try_parse_from([
            "vc-rs",
            "wav-realtime",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
        ])
        .unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidSubcommand);
    }

    #[test]
    fn parses_tensorrt_provider_and_alias() {
        let cli = Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--provider", "tensorrt"])
            .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(args.provider, Provider::TensorRt);

        let cli = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
            "--provider",
            "trt",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };
        assert_eq!(args.provider, Provider::TensorRt);
        assert_eq!(args.gpu_priority, GpuPriority::High);
    }

    #[test]
    fn parses_normal_gpu_priority() {
        let cli =
            Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--gpu-priority", "normal"])
                .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(args.gpu_priority, GpuPriority::Normal);
    }

    #[test]
    fn parses_gpu_device_id() {
        let cli =
            Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--gpu-device-id", "2"]).unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(args.gpu_device_id, 2);
    }

    #[test]
    fn rejects_removed_native_rvc_engine_for_rvc_commands() {
        let err = Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--rvc-engine",
            "present.engine",
        ])
        .unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);

        let err = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
            "--rvc-engine",
            "present.engine",
        ])
        .unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
    }

    #[test]
    fn parses_windowsml_providers() {
        let cli = Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--provider", "windowsml"])
            .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(args.provider, Provider::WindowsMl);

        let cli = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
            "--provider",
            "winml-dml",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };
        assert_eq!(args.provider, Provider::WindowsMlDirectMl);

        let cli = Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--provider",
            "windowsml-nvtrtx",
        ])
        .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(args.provider, Provider::WindowsMlNvTensorRtRtx);

        let cli = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
            "--provider",
            "windowsml-openvino",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };
        assert_eq!(args.provider, Provider::WindowsMlOpenVino);
    }

    #[cfg(all(windows, feature = "windowsml"))]
    #[test]
    fn parses_windowsml_eps_commands() {
        let cli = Cli::try_parse_from(["vc-rs", "windowsml-eps", "list"]).unwrap();
        let Command::WindowsMlEps(args) = cli.command else {
            panic!("expected windowsml-eps command");
        };
        assert!(matches!(args.command, WindowsMlEpsCommand::List));

        let cli = Cli::try_parse_from([
            "vc-rs",
            "windowsml-eps",
            "install",
            "--provider",
            "nvtrtx",
            "--yes",
        ])
        .unwrap();
        let Command::WindowsMlEps(args) = cli.command else {
            panic!("expected windowsml-eps command");
        };
        let WindowsMlEpsCommand::Install(args) = args.command else {
            panic!("expected install command");
        };
        assert_eq!(args.provider, Some(WindowsMlEpProvider::Nvtrtx));
        assert!(args.yes);
    }

    #[test]
    fn wav_defaults_to_cpu_provider() {
        let cli = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };

        assert_eq!(args.provider, Provider::Cpu);
    }

    #[test]
    fn parses_inspect_model_without_provider() {
        let cli = Cli::try_parse_from(["vc-rs", "inspect", "--model", "model.onnx"]).unwrap();
        let Command::Inspect(args) = cli.command else {
            panic!("expected inspect command");
        };
        assert_eq!(args.model, PathBuf::from("model.onnx"));
    }

    #[test]
    fn rejects_provider_for_inspect() {
        assert!(Cli::try_parse_from([
            "vc-rs",
            "inspect",
            "--model",
            "model.onnx",
            "--provider",
            "cpu"
        ])
        .is_err());
    }

    #[test]
    fn parses_smoother_for_rvc_commands() {
        let cli =
            Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--smoother", "psola"]).unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(args.smoother, Smoother::Psola);

        let cli = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
            "--smoother",
            "psola",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };
        assert_eq!(args.smoother, Smoother::Psola);
    }

    #[test]
    fn rejects_unknown_smoother() {
        assert!(Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--smoother",
            "overlap-add",
        ])
        .is_err());
    }

    #[test]
    fn rvc_output_tail_discard_defaults_to_10_ms_for_rvc_commands() {
        let cli = Cli::try_parse_from(["vc-rs", "run", "--passthrough"]).unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(
            args.rvc_output_tail_discard_ms,
            DEFAULT_RVC_OUTPUT_TAIL_DISCARD_MS
        );

        let cli = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };
        assert_eq!(args.smoother, Smoother::Sola);
        assert_eq!(
            args.rvc_output_tail_discard_ms,
            DEFAULT_RVC_OUTPUT_TAIL_DISCARD_MS
        );
    }

    #[test]
    fn parses_rvc_output_tail_discard_ms_for_rvc_commands() {
        let cli = Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--rvc-output-tail-discard-ms",
            "25",
        ])
        .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(args.rvc_output_tail_discard_ms, 25);

        let cli = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
            "--rvc-output-tail-discard-ms",
            "0",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };
        assert_eq!(args.rvc_output_tail_discard_ms, 0);
    }

    #[test]
    fn parses_extra_convert_ms_for_rvc_commands() {
        let cli =
            Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--extra-convert-ms", "125"])
                .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(args.extra_convert_ms, 125);

        let cli = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
            "--extra-convert-ms",
            "20",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };
        assert_eq!(args.extra_convert_ms, 20);
        assert!(Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--extra-convert-samples",
            "4800"
        ])
        .is_err());
    }

    #[test]
    fn wav_crossfade_and_sola_search_default_and_parse() {
        // Defaults match the shared constants when the flags are omitted.
        let cli = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };
        assert_eq!(args.crossfade_ms, DEFAULT_CROSSFADE_MS);
        assert_eq!(args.sola_search_ms, DEFAULT_SOLA_SEARCH_MS);

        // Both flags are sweepable on the wav command (used by --join-report).
        let cli = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
            "--crossfade-ms",
            "40",
            "--sola-search-ms",
            "16",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };
        assert_eq!(args.crossfade_ms, 40);
        assert_eq!(args.sola_search_ms, 16);
        assert!(args.validate_conversion_options().is_ok());
    }

    #[test]
    fn validates_realtime_conversion_ranges() {
        let cli =
            Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--chunk-ms", "19"]).unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(args.validate_conversion_options().is_err());

        let cli =
            Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--chunk-ms", "2001"]).unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(args.validate_conversion_options().is_err());

        let cli =
            Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--extra-convert-ms", "19"])
                .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(args.validate_conversion_options().is_err());

        let cli = Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--extra-convert-ms",
            "3001",
        ])
        .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(args.validate_conversion_options().is_err());
    }

    #[test]
    fn validates_wav_conversion_ranges_without_rejecting_default_chunk() {
        let cli = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };
        assert_eq!(args.chunk_ms, DEFAULT_WAV_CHUNK_MS);
        assert!(args.validate_conversion_options().is_ok());

        let cli = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
            "--extra-convert-ms",
            "19",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };
        assert!(args.validate_conversion_options().is_err());

        let cli = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
            "--chunk-ms",
            "2001",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };
        assert!(args.validate_conversion_options().is_err());
    }

    #[test]
    fn parses_rms_mix_rate() {
        let cli = Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--rms-mix-rate", "0.25"])
            .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };

        assert_eq!(args.rms_mix_rate, 0.25);
    }

    #[test]
    fn wav_defaults_rms_mix_rate_to_input_reference_matching() {
        let cli = Cli::try_parse_from([
            "vc-rs",
            "wav",
            "--model",
            "model.onnx",
            "--embedder",
            "embedder.onnx",
            "--f0-model",
            "f0.onnx",
            "--input",
            "input.wav",
            "--output",
            "output.wav",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };

        assert_eq!(args.rms_mix_rate, 0.0);
    }

    #[test]
    fn rejects_rms_mix_rate_out_of_range() {
        assert!(
            Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--rms-mix-rate", "1.01",])
                .is_err()
        );
        assert!(
            Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--rms-mix-rate", "NaN",])
                .is_err()
        );
    }

    #[test]
    fn parses_wasapi_shared_audio() {
        let cli =
            Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--audio-backend", "wasapi"])
                .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };

        assert_eq!(args.audio_backend, Some(AudioHost::Wasapi));
        assert!(!args.wasapi_exclusive);
        assert!(!args.wasapi_input_exclusive());
        assert!(!args.wasapi_output_exclusive());
        assert!(args.validate_audio_options().is_ok());
    }

    #[test]
    fn parses_wasapi_exclusive_audio() {
        let cli = Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--audio-backend",
            "wasapi",
            "--wasapi-exclusive",
        ])
        .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };

        assert_eq!(args.audio_backend, Some(AudioHost::Wasapi));
        assert!(args.wasapi_exclusive);
        assert!(args.wasapi_input_exclusive());
        assert!(args.wasapi_output_exclusive());
        assert!(args.validate_audio_options().is_ok());
    }

    #[test]
    fn parses_wasapi_buffer_ms() {
        let cli = Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--audio-backend",
            "wasapi",
            "--wasapi-buffer-ms",
            "3",
        ])
        .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };

        assert_eq!(args.wasapi_buffer_ms, 3);
        assert!(args.validate_audio_options().is_ok());
    }

    #[test]
    fn allows_zero_wasapi_buffer_for_minimum() {
        let cli = Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--wasapi-buffer-ms", "0"])
            .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };

        assert_eq!(args.wasapi_buffer_ms, 0);
        assert!(args.validate_audio_options().is_ok());
    }

    #[test]
    fn rejects_removed_wasapi_timing_options() {
        assert!(
            Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--wasapi-period-ms", "0"])
                .is_err()
        );
        assert!(Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--wasapi-buffer-periods",
            "1"
        ])
        .is_err());
    }

    #[test]
    fn parses_wasapi_output_only_exclusive_audio() {
        let cli = Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--audio-backend",
            "wasapi",
            "--wasapi-exclusive-output",
        ])
        .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };

        assert_eq!(args.audio_backend, Some(AudioHost::Wasapi));
        assert!(!args.wasapi_input_exclusive());
        assert!(args.wasapi_output_exclusive());
        assert!(args.validate_audio_options().is_ok());
    }

    #[test]
    fn rejects_exclusive_options_without_wasapi_host() {
        // A non-WASAPI input host with --wasapi-exclusive must be rejected.
        let cli = Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--input-backend",
            "asio",
            "--wasapi-exclusive",
        ])
        .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };

        assert_eq!(
            args.validate_audio_options().unwrap_err(),
            "--wasapi-exclusive/--wasapi-exclusive-input require the WASAPI input host"
        );
    }

    #[test]
    fn parses_asio_backend() {
        let cli = Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--audio-backend", "asio"])
            .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };

        assert_eq!(args.audio_backend, Some(AudioHost::Asio));
        assert_eq!(args.effective_input_host(), vc_app::AudioHost::Asio);
        assert_eq!(args.effective_output_host(), vc_app::AudioHost::Asio);
        assert!(args.validate_audio_options().is_ok());
    }

    #[test]
    fn per_direction_hosts_override_audio_backend() {
        // --audio-backend is the default for both directions; the per-direction
        // flags override only their side.
        let cli = Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--input-backend",
            "wasapi",
            "--output-backend",
            "asio",
        ])
        .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };

        assert_eq!(args.effective_input_host(), vc_app::AudioHost::Wasapi);
        assert_eq!(args.effective_output_host(), vc_app::AudioHost::Asio);
        assert!(args.validate_audio_options().is_ok());
    }

    #[test]
    fn audio_backend_seeds_unset_direction() {
        // Only the input direction is overridden; output falls back to
        // --audio-backend (asio here).
        let cli = Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--audio-backend",
            "asio",
            "--input-backend",
            "wasapi",
        ])
        .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };

        assert_eq!(args.effective_input_host(), vc_app::AudioHost::Wasapi);
        assert_eq!(args.effective_output_host(), vc_app::AudioHost::Asio);
    }

    #[test]
    fn exclusive_output_requires_wasapi_output_host() {
        // Input is WASAPI but output is ASIO, so --wasapi-exclusive (both) fails on
        // the output direction.
        let cli = Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--input-backend",
            "wasapi",
            "--output-backend",
            "asio",
            "--wasapi-exclusive",
        ])
        .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };

        assert_eq!(
            args.validate_audio_options().unwrap_err(),
            "--wasapi-exclusive/--wasapi-exclusive-output require the WASAPI output host"
        );
    }

    #[test]
    fn parses_exclusive_denoiser_modes_and_legacy_gate_alias() {
        let cli = Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--denoiser", "rnnoise"])
            .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(args.denoiser_mode(), Denoiser::Rnnoise);

        let cli = Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--noise-gate"]).unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(args.denoiser_mode(), Denoiser::NoiseGate);

        assert!(Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--noise-gate",
            "--denoiser",
            "rnnoise",
        ])
        .is_err());
    }

    #[test]
    fn parses_engine_cache_commands() {
        let cli = Cli::try_parse_from(["vc-rs", "engine-cache", "info"]).unwrap();
        let Command::EngineCache(args) = cli.command else {
            panic!("expected engine-cache command");
        };
        assert!(matches!(args.command, EngineCacheCommand::Info));

        // `cache` alias and the `clear --yes` form.
        let cli = Cli::try_parse_from(["vc-rs", "cache", "clear", "--yes"]).unwrap();
        let Command::EngineCache(args) = cli.command else {
            panic!("expected engine-cache command");
        };
        let EngineCacheCommand::Clear(args) = args.command else {
            panic!("expected clear command");
        };
        assert!(args.yes);

        // `show` alias for info defaults `--yes` off on clear.
        let cli = Cli::try_parse_from(["vc-rs", "engine-cache", "clear"]).unwrap();
        let Command::EngineCache(args) = cli.command else {
            panic!("expected engine-cache command");
        };
        let EngineCacheCommand::Clear(args) = args.command else {
            panic!("expected clear command");
        };
        assert!(!args.yes);
    }

    #[test]
    fn doctor() {
        let cli = Cli::try_parse_from(["vc-rs", "doctor"]).unwrap();
        assert!(matches!(cli.command, Command::Doctor));
    }

    #[test]
    fn devices() {
        let cli = Cli::try_parse_from(["vc-rs", "devices"]).unwrap();
        let Command::Devices(_args) = cli.command else {
            panic!("expected devices command");
        };
    }
}
