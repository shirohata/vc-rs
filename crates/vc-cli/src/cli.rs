use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

pub use vc_core::Provider;

pub const DEFAULT_CROSSFADE_MS: u32 = 85;
pub const DEFAULT_EXTRA_CONVERT_MS: u32 = 100;
pub const DEFAULT_INPUT_GAIN: f32 = 1.0;
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
    /// List audio input and output devices.
    Devices(DevicesArgs),
    /// Inspect ONNX model inputs, outputs, and metadata.
    Inspect(InspectArgs),
    /// Run realtime microphone-to-speaker conversion.
    Run(RunArgs),
    /// Run conversion against a wav file for model/DSP verification.
    Wav(WavArgs),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Smoother {
    Sola,
    Psola,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum AudioBackend {
    Cpal,
    Wasapi,
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
    #[arg(
        long,
        help = "Serialized TensorRT engine for the RVC model; with --provider tensorrt this bypasses ONNX Runtime TensorRT EP for RVC"
    )]
    pub rvc_engine: Option<PathBuf>,
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
    #[arg(long, value_enum, default_value_t = Provider::Cpu)]
    pub provider: Provider,
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
    #[arg(
        long,
        help = "Serialized TensorRT engine for the RVC model; with --provider tensorrt this bypasses ONNX Runtime TensorRT EP for RVC"
    )]
    pub rvc_engine: Option<PathBuf>,
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
    #[arg(long, value_enum, default_value_t = Provider::Cpu)]
    pub provider: Provider,
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
    }

    #[test]
    fn parses_native_rvc_engine_for_rvc_commands() {
        let cli = Cli::try_parse_from([
            "vc-rs",
            "run",
            "--passthrough",
            "--rvc-engine",
            "present.engine",
        ])
        .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(args.rvc_engine, Some(PathBuf::from("present.engine")));

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
            "--rvc-engine",
            "present.engine",
        ])
        .unwrap();
        let Command::Wav(args) = cli.command else {
            panic!("expected wav command");
        };
        assert_eq!(args.rvc_engine, Some(PathBuf::from("present.engine")));
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
    fn devices() {
        let cli = Cli::try_parse_from(["vc-rs", "devices"]).unwrap();
        let Command::Devices(_args) = cli.command else {
            panic!("expected devices command");
        };
    }
}
