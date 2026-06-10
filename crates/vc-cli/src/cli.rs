use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

pub use vc_core::Provider;

/// Default inference backend for this build. The distribution packages are
/// single-provider (`package.ps1` builds `--no-default-features --features
/// windowsml|tensorrt`), so the default tracks the one backend that package
/// ships — Windows ML for the windowsml package, native TensorRT for the
/// tensorrt package. The combined dev binary (both features, via `cargo build`)
/// and any CPU-only build fall back to `cpu`, which always works there.
#[cfg(all(feature = "windowsml", not(feature = "tensorrt")))]
pub fn default_provider() -> Provider {
    Provider::WindowsMl
}

#[cfg(all(feature = "tensorrt", not(feature = "windowsml")))]
pub fn default_provider() -> Provider {
    Provider::TensorRt
}

#[cfg(not(any(
    all(feature = "windowsml", not(feature = "tensorrt")),
    all(feature = "tensorrt", not(feature = "windowsml")),
)))]
pub fn default_provider() -> Provider {
    Provider::Cpu
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
}

impl From<Denoiser> for vc_app::DenoiserMode {
    fn from(value: Denoiser) -> Self {
        match value {
            Denoiser::Off => Self::Off,
            Denoiser::NoiseGate => Self::NoiseGate,
            Denoiser::Rnnoise => Self::Rnnoise,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum AudioBackend {
    Cpal,
    Wasapi,
}

impl From<AudioBackend> for vc_app::AudioBackend {
    fn from(value: AudioBackend) -> Self {
        match value {
            AudioBackend::Cpal => Self::Cpal,
            AudioBackend::Wasapi => Self::Wasapi,
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
pub enum DeviceAudioBackend {
    All,
    Cpal,
    Wasapi,
}

#[derive(Debug, Parser)]
pub struct DevicesArgs {
    #[arg(long, value_enum, default_value_t = DeviceAudioBackend::Cpal)]
    pub audio_backend: DeviceAudioBackend,
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
    #[arg(long, value_enum, default_value_t = AudioBackend::Cpal)]
    pub audio_backend: AudioBackend,
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
    #[arg(long, value_enum, help = "Input denoiser: off, noise-gate, or rnnoise")]
    pub denoiser: Option<Denoiser>,
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
    #[arg(long, default_value_t = DEFAULT_RVC_OUTPUT_TAIL_DISCARD_MS)]
    pub rvc_output_tail_discard_ms: u32,
    #[arg(long, value_enum, default_value_t = default_provider())]
    pub provider: Provider,
    #[arg(long, value_enum, default_value_t = GpuPriority::High)]
    pub gpu_priority: GpuPriority,
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
    #[arg(long, value_enum, help = "Input denoiser: off, noise-gate, or rnnoise")]
    pub denoiser: Option<Denoiser>,
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

    pub fn validate_audio_options(&self) -> Result<(), String> {
        if (self.wasapi_exclusive || self.wasapi_exclusive_input || self.wasapi_exclusive_output)
            && self.audio_backend != AudioBackend::Wasapi
        {
            return Err("--wasapi-exclusive* options require --audio-backend wasapi".to_string());
        }
        Ok(())
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_defaults_to_cpal_shared_audio() {
        let cli = Cli::try_parse_from(["vc-rs", "run", "--passthrough"]).unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };

        assert_eq!(args.audio_backend, AudioBackend::Cpal);
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
            "0",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };
        assert_eq!(args.extra_convert_ms, 0);
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

        assert_eq!(args.audio_backend, AudioBackend::Wasapi);
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

        assert_eq!(args.audio_backend, AudioBackend::Wasapi);
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

        assert_eq!(args.audio_backend, AudioBackend::Wasapi);
        assert!(!args.wasapi_input_exclusive());
        assert!(args.wasapi_output_exclusive());
        assert!(args.validate_audio_options().is_ok());
    }

    #[test]
    fn rejects_exclusive_options_without_wasapi_backend() {
        let cli =
            Cli::try_parse_from(["vc-rs", "run", "--passthrough", "--wasapi-exclusive"]).unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };

        assert_eq!(
            args.validate_audio_options().unwrap_err(),
            "--wasapi-exclusive* options require --audio-backend wasapi"
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
