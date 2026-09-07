use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use tracing::{debug, info};

use crate::dsp;
use crate::validation::RvcChunkTiming;
use crate::Provider;

use super::api::{ContentDelay, ModelOutput, VoiceModel};
use super::f0_postprocess::{F0PostprocessConfig, F0Postprocessor};
use super::feature::FeatureTensor;
use super::inspect::{inspect_contentvec_input_name, inspect_rvc_model};
use super::native_tensorrt::native_engine_is_cached;
#[cfg(feature = "gtcrn")]
use super::native_tensorrt::native_gtcrn_engine_is_cached;
use super::pitch::{
    align_pitchf_to_features_into, center_crop_pitchf_to_features_into, coarse_pitch_into,
    pitchf_tail_for_output_into, voiced_ratio,
};
use super::sessions::{HubertEmbedderSession, RmvpePitchSession, RvcModelSession};
use super::shape::{
    extra_convert_samples_from_ms, keep_tail_in_place, ms_to_samples,
    onnx_silence_front_feature_frames, rmvpe_model_input_samples_for_context_16k,
    tensor_rt_model_input_samples_16k, RVC_SAMPLE_RATE,
};
use super::stream::RvcStreamState;
use super::tensorrt::{
    derive_rvc_feature_len, provider_uses_fixed_shape, tensor_rt_model_cache_key, ModelRole,
    TensorRtRunMode, TensorRtSessionProfile, TensorRtSessionPurpose, CUDA_GRAPH_ENV,
};
#[cfg(feature = "ort")]
use super::tensorrt::{tensor_rt_warmup_feature_len, TensorRtSharedWaveform};
use super::time_state::StreamParams;

const SKIP_SILENT_CHUNKS: bool = false;

/// Input-side denoising stage, applied to the raw input (after `input_gain`)
/// before RMS/silence detection and feature/F0 extraction. Keep future denoisers
/// behind variants and match arms here so the timing-sensitive call site in
/// `process()` does not change.
enum InputDenoiser {
    Off,
    Gate(dsp::NoiseGate),
    #[cfg(feature = "rnnoise")]
    Rnnoise(Box<crate::denoise::RnnoiseDenoiser>),
}

impl InputDenoiser {
    fn content_delay_samples(&self) -> usize {
        match self {
            Self::Off | Self::Gate(_) => 0,
            #[cfg(feature = "rnnoise")]
            Self::Rnnoise(denoiser) => denoiser.latency_samples(),
        }
    }

    fn process_in_place(&mut self, buf: &mut [f32]) -> Result<()> {
        match self {
            InputDenoiser::Off => {}
            InputDenoiser::Gate(gate) => gate.process_in_place(buf),
            #[cfg(feature = "rnnoise")]
            InputDenoiser::Rnnoise(denoiser) => denoiser.process_in_place(buf)?,
        }
        Ok(())
    }

    fn reset(&mut self, sample_rate: f32, threshold: f32, shaping: NoiseGateShaping) -> Result<()> {
        *self = match self {
            InputDenoiser::Off => InputDenoiser::Off,
            InputDenoiser::Gate(_) => InputDenoiser::Gate(dsp::NoiseGate::new(
                sample_rate,
                threshold,
                shaping.attack_ms,
                shaping.release_ms,
                shaping.floor,
            )),
            #[cfg(feature = "rnnoise")]
            InputDenoiser::Rnnoise(_) => InputDenoiser::Rnnoise(Box::new(
                crate::denoise::RnnoiseDenoiser::new(sample_rate as u32)?,
            )),
        };
        Ok(())
    }
}

fn build_input_denoiser(config: &RvcPipelineConfig<'_>) -> InputDenoiser {
    if config.noise_gate_enabled {
        InputDenoiser::Gate(dsp::NoiseGate::new(
            config.sample_rate as f32,
            config.noise_gate_threshold,
            config.noise_gate_shaping.attack_ms,
            config.noise_gate_shaping.release_ms,
            config.noise_gate_shaping.floor,
        ))
    } else {
        InputDenoiser::Off
    }
}

pub struct RvcPipeline {
    embedder: HubertEmbedderSession,
    pitch: RmvpePitchSession,
    rvc: RvcModelSession,
    #[cfg(feature = "ort")]
    shared_waveform: Option<TensorRtSharedWaveform>,
    speaker_id: i64,
    pitch_shift: f32,
    f0_threshold: f32,
    silence_threshold: f32,
    input_gain: f32,
    // Input noise reduction. `input_denoiser` is the active stage (Off when
    // disabled); the remaining fields let `set_noise_gate` rebuild the gate
    // when it is toggled back on without a full pipeline reload.
    input_denoiser: InputDenoiser,
    noise_gate_threshold: f32,
    noise_gate_attack_ms: f32,
    noise_gate_release_ms: f32,
    noise_gate_floor: f32,
    noise_gate_sample_rate: f32,
    output_extra_ms: u32,
    volume_excluded_ms: u32,
    // The RVC model's native output sample rate (metadata `samplingRate`, default
    // RVC_SAMPLE_RATE). All convert/output-window sizing and the reported output
    // rate use this so non-48 kHz models (e.g. 32 kHz) are not mis-sized.
    rvc_sample_rate: u32,
    extra_convert_samples: usize,
    rmvpe_input_samples_16k: usize,
    output_gain: f32,
    volume_envelope: bool,
    rms_mix_rate: f32,
    auto_output_gain: bool,
    target_output_rms: f32,
    max_output_gain: f32,
    stream_state: RvcStreamState,
    // The input hop and rate are fixed with the loaded profiles and denoisers.
    // A host changing either must reload; allowing a new duration here would
    // advance audio, F0, latent noise and NSF phase on different timelines.
    chunk_timing: RvcChunkTiming,
    input_sample_rate: u32,
    // The model's `rnd` (latent-noise) channel count, retained so the rolling
    // noise state can be rebuilt with the same shape on `reset_streaming_state`.
    // `None` when the model samples its own noise.
    rnd_channels: Option<usize>,
    // Streaming NSF time base (frame hop / sample rate), retained to rebuild the
    // streaming time state on reset. `None` for conventional exports.
    stream_params: Option<StreamParams>,
    // Reused channel-major `[1, channels, feature_len]` latent-noise tensor,
    // refilled per chunk from the rolling noise state and bound by every backend.
    rnd_scratch: Vec<f32>,
    // Reused `[1, audio_len, 1]` NSF source-noise tensor for streaming exports,
    // refilled per chunk from the rolling NSF noise state.
    nsf_scratch: Vec<f32>,
    // Reused buffer for the model's per-sample `streaming_nsf_phase` output, read
    // back each streaming chunk to pick the next window's `phase_in`.
    phase_out_scratch: Vec<f32>,
    // Reused per-chunk buffer for the gain-scaled / denoised input, so `process`
    // does not allocate a fresh Vec every chunk when input_gain != 1.0 or a
    // denoiser is active. Empty when the zero-copy (gain==1.0, denoiser-off) path
    // is taken.
    input_scratch: Vec<f32>,
    // Reused embedder output tensor, refilled in place each chunk by
    // `extract_into` so the per-chunk ContentVec output is not reallocated.
    feature_tensor: FeatureTensor,
    input_reference_scratch: Vec<f32>,
    rms_mix_scratch: dsp::RmsMixScratch,
    pitchf_untrimmed_scratch: Vec<f32>,
    pitchf_scratch: Vec<f32>,
    pitch_scratch: Vec<i64>,
    f0_postprocess: F0Postprocessor,
    // Non-destructive output of `process_pitchf_into`: `pitchf_scratch` holds the
    // aligned raw F0 input, so the post-processed result needs its own buffer.
    pitchf_postprocessed_scratch: Vec<f32>,
}

/// Post-conversion output level shaping, applied after inference.
///
/// Grouped because every front-end carries the same five knobs verbatim and
/// passes them straight through `RealtimeConfig` into `RvcPipelineConfig`;
/// keeping them as one unit means adding an output-level knob touches the
/// struct, not each front-end's field-by-field config mapping. `output_gain`
/// is deliberately *not* here: it is a live (per-block) parameter, not static
/// load-time config.
#[derive(Clone, Copy, Debug)]
pub struct OutputDynamicsConfig {
    /// Match the converted output's short-term envelope to the input's.
    pub volume_envelope: bool,
    /// Blend ratio (0..=1) for mixing input RMS back into the output level.
    pub rms_mix_rate: f32,
    /// Automatically scale output toward `target_output_rms`.
    pub auto_output_gain: bool,
    pub target_output_rms: f32,
    pub max_output_gain: f32,
}

impl Default for OutputDynamicsConfig {
    fn default() -> Self {
        Self {
            volume_envelope: false,
            rms_mix_rate: 0.0,
            auto_output_gain: false,
            target_output_rms: 0.03,
            max_output_gain: 512.0,
        }
    }
}

/// Static (load-time) shaping for the input noise gate.
///
/// Same rationale as [`OutputDynamicsConfig`]: every front-end carries these
/// three knobs verbatim and passes them straight through, so grouping them
/// means adding a gate-shaping knob touches the struct, not each front-end's
/// field-by-field mapping. The gate's `enabled`/`threshold` are deliberately
/// *not* here: they are live (per-block) parameters, kept as separate
/// initial-value fields on `RvcPipelineConfig`. Attack/release/floor shape the
/// smoothing coefficients fixed when the gate is constructed.
#[derive(Clone, Copy, Debug)]
pub struct NoiseGateShaping {
    pub attack_ms: f32,
    pub release_ms: f32,
    pub floor: f32,
}

impl Default for NoiseGateShaping {
    fn default() -> Self {
        Self {
            attack_ms: 5.0,
            release_ms: 50.0,
            floor: 0.0,
        }
    }
}

/// Static (load-time) F0 configuration.
///
/// Same grouping rationale as [`OutputDynamicsConfig`]: these knobs travel
/// together from every front-end through `RealtimeConfig` into
/// `RvcPipelineConfig`. `f0_postprocess` is plumbed but inert by default
/// (`F0PostprocessConfig::default()` has `enabled: false`); exposing it to the
/// front-ends is a separate, behaviour-changing task. Keeping it in this struct
/// means that wiring becomes "fill in a field" rather than threading a new knob
/// through every boundary.
#[derive(Clone, Debug)]
pub struct F0Config {
    /// RMVPE voiced/unvoiced confidence threshold.
    pub f0_threshold: f32,
    /// Input RMS below which a chunk is treated as silence.
    pub silence_threshold: f32,
    pub postprocess: F0PostprocessConfig,
}

impl Default for F0Config {
    fn default() -> Self {
        Self {
            f0_threshold: 0.3,
            silence_threshold: 0.0001,
            postprocess: F0PostprocessConfig::default(),
        }
    }
}

/// Live (per-block) conversion parameters: the knobs a host can change between
/// chunks without reloading the pipeline. Applied through
/// [`RvcPipeline::apply_live`], which is the single live-update entry point
/// shared by every front-end (the standalone worker and the VST3 host callback).
///
/// `noise_gate_enabled`/`noise_gate_threshold` live here, but the gate's
/// attack/release/floor and the denoiser *variant* selection (off / gate /
/// rnnoise) are static load-time config: switching to/from rnnoise rebuilds a
/// stateful denoiser and so requires a reload, and `apply_live` cannot replace
/// an active rnnoise stage (see `set_noise_gate`).
#[derive(Clone, Copy, Debug)]
pub struct LiveParams {
    pub pitch_shift: f32,
    pub speaker_id: i64,
    pub input_gain: f32,
    pub output_gain: f32,
    pub noise_gate_enabled: bool,
    pub noise_gate_threshold: f32,
}

impl Default for LiveParams {
    fn default() -> Self {
        Self {
            pitch_shift: 0.0,
            speaker_id: 0,
            input_gain: 1.0,
            output_gain: 1.0,
            noise_gate_enabled: false,
            noise_gate_threshold: 0.01,
        }
    }
}

pub struct RvcPipelineConfig<'a> {
    pub model: &'a Path,
    pub embedder: &'a Path,
    pub embedder_output: Option<&'a str>,
    pub f0_model: &'a Path,
    pub provider: Provider,
    pub gpu_priority: super::GpuPriority,
    pub gpu_device_id: u32,
    pub sample_rate: u32,
    pub chunk_samples: usize,
    pub speaker_id: i64,
    pub pitch_shift: f32,
    pub f0: F0Config,
    pub input_gain: f32,
    pub noise_gate_enabled: bool,
    pub noise_gate_threshold: f32,
    pub noise_gate_shaping: NoiseGateShaping,
    pub output_extra_ms: u32,
    pub volume_excluded_ms: u32,
    pub extra_convert_ms: u32,
    pub output_gain: f32,
    pub output_dynamics: OutputDynamicsConfig,
    /// Optional load-time progress callback. It is invoked only while building
    /// the pipeline, never from inference or an audio callback.
    pub progress: Option<&'a dyn Fn(LoadProgress)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoadModelRole {
    ContentVec,
    Rmvpe,
    Rvc,
    Gtcrn,
}

impl LoadModelRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::ContentVec => "ContentVec",
            Self::Rmvpe => "RMVPE",
            Self::Rvc => "RVC",
            Self::Gtcrn => "GTCRN",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoadProgress {
    Idle,
    ValidatingConfig,
    PreparingProvider,
    DownloadingProvider,
    BuildingEngine { role: LoadModelRole },
    LoadingModel { role: LoadModelRole },
    OpeningAudioDevices,
    Running,
    Failed,
}

fn report_progress(config: &RvcPipelineConfig<'_>, progress: LoadProgress) {
    if let Some(report) = config.progress {
        report(progress);
    }
}

fn report_native_load_progress(
    config: &RvcPipelineConfig<'_>,
    profile: &TensorRtSessionProfile,
    role: LoadModelRole,
) {
    if let Some(progress) = native_engine_build_progress(native_engine_is_cached(profile), role) {
        report_progress(config, progress);
    }
    report_progress(config, LoadProgress::LoadingModel { role });
}

fn native_engine_build_progress(cached: bool, role: LoadModelRole) -> Option<LoadProgress> {
    (!cached).then_some(LoadProgress::BuildingEngine { role })
}

impl RvcPipeline {
    #[cfg(feature = "rnnoise")]
    pub fn load_with_rnnoise(config: RvcPipelineConfig<'_>) -> Result<Self> {
        if config.noise_gate_enabled {
            bail!("RNNoise and the input noise gate are mutually exclusive");
        }
        let sample_rate = config.sample_rate;
        let mut pipeline = Self::load(config)?;
        pipeline.input_denoiser =
            InputDenoiser::Rnnoise(Box::new(crate::denoise::RnnoiseDenoiser::new(sample_rate)?));
        Ok(pipeline)
    }

    /// Load with GTCRN input denoising at the 16 kHz RVC seam. Unlike RNNoise
    /// (a device-rate `InputDenoiser`), GTCRN lives in the stream state and
    /// denoises the new 16 kHz increment inside `generate_input`.
    #[cfg(feature = "gtcrn")]
    pub fn load_with_gtcrn(
        config: RvcPipelineConfig<'_>,
        gtcrn: crate::denoise::GtcrnConfig<'_>,
    ) -> Result<Self> {
        // This entry point loads the denoiser before Self::load; reject an
        // unsupported hop before it can build a GPU engine or allocate caches.
        RvcChunkTiming::from_samples(config.chunk_samples, config.sample_rate)?;
        if config.noise_gate_enabled {
            bail!("GTCRN and the input noise gate are mutually exclusive");
        }
        if let crate::denoise::GtcrnBackend::NativeTensorRt { gpu_device_id, .. } = gtcrn.backend {
            let model = crate::denoise::model_file_for_cache_probe(gtcrn.model_dir)?;
            if !native_gtcrn_engine_is_cached(&model, gpu_device_id) {
                report_progress(
                    &config,
                    LoadProgress::BuildingEngine {
                        role: LoadModelRole::Gtcrn,
                    },
                );
            }
            report_progress(
                &config,
                LoadProgress::LoadingModel {
                    role: LoadModelRole::Gtcrn,
                },
            );
        }
        // Build the adapter at 16 kHz so its resamplers are bypass — only the
        // hop FIFO and fixed delay run on the increment fed by `resampler_16k`.
        let denoiser =
            crate::denoise::GtcrnDenoiser::new(gtcrn, super::shape::EMBEDDER_SAMPLE_RATE)?;
        let mut pipeline = Self::load(config)?;
        pipeline.stream_state.gtcrn = Some(denoiser);
        Ok(pipeline)
    }

    pub fn load(config: RvcPipelineConfig<'_>) -> Result<Self> {
        report_progress(&config, LoadProgress::ValidatingConfig);
        let chunk_timing = RvcChunkTiming::from_samples(config.chunk_samples, config.sample_rate)?;
        if provider_needs_fixed_shape_profile(config.provider) {
            return Self::load_fixed_shape(config);
        }

        report_progress(&config, LoadProgress::PreparingProvider);
        // Resolve the generator's I/O names (vcclient vs RVC WebUI / converter
        // export aliases) and the model's native sample rate before sizing the
        // convert/output windows, so the ORT bind sites use the right names and
        // all sizing math runs in the model's actual rate domain.
        let rvc_info = inspect_rvc_model(config.model)?;
        let rvc_sample_rate = rvc_info.rvc_sample_rate.unwrap_or(RVC_SAMPLE_RATE);
        chunk_timing.samples_at_rate(rvc_sample_rate)?;
        // Capture the `rnd` channel count before `io_names` is moved into the
        // session, so the rolling-noise state can be sized (and rebuilt on reset).
        let rnd_channels = rvc_info
            .io_names
            .rnd
            .as_ref()
            .and_then(|rnd| usize::try_from(rnd.channels).ok());
        // Streaming exports add NSF noise/phase state; the dynamic-shape `load`
        // path (CPU/CUDA/DirectML) handles their extra I/O directly.
        let stream_params = stream_params_from_info(&rvc_info)?;
        // CLI-facing configuration is milliseconds for consistency with other latency knobs.
        // The RVC shape and trimming code below use the model's sample-rate domain, so keep the
        // conversion at load time and leave the per-chunk processing path in samples.
        let extra_convert_samples =
            extra_convert_samples_from_ms(config.extra_convert_ms, rvc_sample_rate);
        let input_samples_16k = tensor_rt_model_input_samples_16k(
            config.chunk_samples,
            config.sample_rate,
            config.output_extra_ms,
            extra_convert_samples,
            rvc_sample_rate,
        );
        let rmvpe_input_samples_16k = rmvpe_model_input_samples_for_context_16k(
            config.chunk_samples,
            config.sample_rate,
            input_samples_16k,
        );
        report_progress(
            &config,
            LoadProgress::LoadingModel {
                role: LoadModelRole::Rvc,
            },
        );
        let rvc = RvcModelSession::load(
            config.model,
            config.provider,
            None,
            Some(rvc_info.expected_feat_channels),
            TensorRtRunMode::PinnedCpu,
            TensorRtSessionPurpose::Main,
            rvc_info.io_names,
        )?;
        let expected_feat_channels = rvc.expected_feat_channels;
        report_progress(
            &config,
            LoadProgress::LoadingModel {
                role: LoadModelRole::ContentVec,
            },
        );
        let embedder = HubertEmbedderSession::load(
            config.embedder,
            config.provider,
            expected_feat_channels,
            config.embedder_output,
            None,
            TensorRtRunMode::PinnedCpu,
            TensorRtSessionPurpose::Main,
        )?;
        report_progress(
            &config,
            LoadProgress::LoadingModel {
                role: LoadModelRole::Rmvpe,
            },
        );
        let pitch = RmvpePitchSession::load(
            config.f0_model,
            config.provider,
            None,
            TensorRtRunMode::PinnedCpu,
            TensorRtSessionPurpose::Main,
        )?;
        Ok(Self {
            embedder,
            pitch,
            rvc,
            #[cfg(feature = "ort")]
            shared_waveform: None,
            speaker_id: config.speaker_id,
            pitch_shift: config.pitch_shift,
            f0_threshold: config.f0.f0_threshold,
            silence_threshold: config.f0.silence_threshold,
            input_gain: config.input_gain,
            input_denoiser: build_input_denoiser(&config),
            noise_gate_threshold: config.noise_gate_threshold,
            noise_gate_attack_ms: config.noise_gate_shaping.attack_ms,
            noise_gate_release_ms: config.noise_gate_shaping.release_ms,
            noise_gate_floor: config.noise_gate_shaping.floor,
            noise_gate_sample_rate: config.sample_rate as f32,
            output_extra_ms: config.output_extra_ms,
            volume_excluded_ms: config.volume_excluded_ms,
            rvc_sample_rate,
            extra_convert_samples,
            rmvpe_input_samples_16k,
            output_gain: config.output_gain,
            volume_envelope: config.output_dynamics.volume_envelope,
            rms_mix_rate: config.output_dynamics.rms_mix_rate,
            auto_output_gain: config.output_dynamics.auto_output_gain,
            target_output_rms: config.output_dynamics.target_output_rms,
            max_output_gain: config.output_dynamics.max_output_gain,
            stream_state: RvcStreamState::new_configured(
                rvc_sample_rate,
                rnd_channels,
                stream_params,
                config.sample_rate,
            )?,
            chunk_timing,
            input_sample_rate: config.sample_rate,
            rnd_channels,
            stream_params,
            rnd_scratch: Vec::new(),
            nsf_scratch: Vec::new(),
            phase_out_scratch: Vec::new(),
            input_scratch: Vec::new(),
            feature_tensor: FeatureTensor::default(),
            input_reference_scratch: Vec::new(),
            rms_mix_scratch: dsp::RmsMixScratch::default(),
            pitchf_untrimmed_scratch: Vec::new(),
            pitchf_scratch: Vec::new(),
            pitch_scratch: Vec::new(),
            f0_postprocess: F0Postprocessor::new(config.f0.postprocess.clone()),
            pitchf_postprocessed_scratch: Vec::new(),
        })
    }

    fn load_fixed_shape(config: RvcPipelineConfig<'_>) -> Result<Self> {
        let chunk_timing = RvcChunkTiming::from_samples(config.chunk_samples, config.sample_rate)?;
        report_progress(&config, LoadProgress::PreparingProvider);
        // Windows ML catalog providers may also use fixed-shape profiles, but
        // their adapter selection is owned by Windows ML. Only explicit CUDA
        // backends consume the user-selected CUDA device ID.
        let gpu_device_id = if config.provider.is_cuda() || config.provider.is_tensorrt() {
            config.gpu_device_id
        } else {
            0
        };
        // NvTensorRtRtx (explicit, or auto-selected by `Provider::WindowsMl`) uses
        // the pinned-CPU run mode: it has no CUDA IoBinding/CUDA-graph path, so the
        // CUDA device-I/O modes from the env must not be applied to it.
        let tensor_rt_run_mode =
            if config.provider.is_tensorrt() || provider_drives_nvtrtx(config.provider) {
                TensorRtRunMode::PinnedCpu
            } else {
                TensorRtRunMode::cuda_from_env()
            };
        info!(
            "{} run mode selected mode={} cuda_graph={} device_io={} env_var={}",
            config.provider.label(),
            tensor_rt_run_mode.label(),
            tensor_rt_run_mode.cuda_graph(),
            tensor_rt_run_mode.device_io(),
            if config.provider.is_tensorrt() {
                "native-tensorrt"
            } else {
                CUDA_GRAPH_ENV
            }
        );
        let rvc_info = inspect_rvc_model(config.model)?;
        let stream_params = stream_params_from_info(&rvc_info)?;
        // Streaming NSF noise/phase I/O is bound on two fixed-shape backends:
        // native TensorRT (its engine profile + bindings) and the pinned-CPU ORT
        // IoBinding (Windows ML TensorRT-RTX). The CUDA device-I/O and CUDA-graph
        // IoBinding paths do not bind it yet, so for those a streaming model must
        // run on the dynamic-shape `load` path (CPU/CUDA/DirectML); fail clearly
        // rather than build a binding missing those inputs.
        if stream_params.is_some()
            && !config.provider.is_tensorrt()
            && tensor_rt_run_mode != TensorRtRunMode::PinnedCpu
        {
            bail!(
                "provider {} does not support rvc-onnx-web streaming exports in this run mode; use native tensorrt, windowsml-nvtrtx, or a CPU/CUDA/DirectML provider (windowsml) for streaming models",
                config.provider.label()
            );
        }
        let expected_feat_channels = rvc_info.expected_feat_channels;
        let expected_feat_channels_usize = usize::try_from(expected_feat_channels)
            .context("RVC expected feature channel count does not fit in usize")?;
        let rvc_sample_rate = rvc_info.rvc_sample_rate.unwrap_or(RVC_SAMPLE_RATE);
        chunk_timing.samples_at_rate(rvc_sample_rate)?;
        // Capture the `rnd` channel count before `io_names` is cloned/moved into
        // the sessions, so the rolling-noise state can be sized (and reset).
        let rnd_channels = rvc_info
            .io_names
            .rnd
            .as_ref()
            .and_then(|rnd| usize::try_from(rnd.channels).ok());
        let extra_convert_samples =
            extra_convert_samples_from_ms(config.extra_convert_ms, rvc_sample_rate);
        let input_samples_16k = tensor_rt_model_input_samples_16k(
            config.chunk_samples,
            config.sample_rate,
            config.output_extra_ms,
            extra_convert_samples,
            rvc_sample_rate,
        );
        let rmvpe_input_samples_16k = rmvpe_model_input_samples_for_context_16k(
            config.chunk_samples,
            config.sample_rate,
            input_samples_16k,
        );
        let (contentvec_model_cache_key, rmvpe_model_cache_key, rvc_model_cache_key) =
            if provider_needs_fixed_shape_profile(config.provider) {
                (
                    Some(tensor_rt_model_cache_key(config.embedder)?),
                    Some(tensor_rt_model_cache_key(config.f0_model)?),
                    Some(tensor_rt_model_cache_key(config.model)?),
                )
            } else {
                (None, None, None)
            };
        // Fixed-shape GPU profiles must use the model's exported input name.
        // Keep this CPU-only probe at load time; the realtime path relies on
        // the resulting profile for CUDA/TensorRT validation and IoBinding.
        let contentvec_input_name = inspect_contentvec_input_name(
            config.embedder,
            expected_feat_channels,
            config.embedder_output,
        )?;
        let contentvec_profile = TensorRtSessionProfile::single_input(
            ModelRole::ContentVec,
            contentvec_input_name,
            input_samples_16k,
        )
        .with_gpu_priority(config.gpu_priority)
        .with_gpu_device_id(gpu_device_id)
        .with_optional_model_cache_key(contentvec_model_cache_key);
        let rmvpe_profile = TensorRtSessionProfile::single_input(
            ModelRole::Rmvpe,
            "waveform",
            rmvpe_input_samples_16k,
        )
        .with_gpu_priority(config.gpu_priority)
        .with_gpu_device_id(gpu_device_id)
        .with_optional_model_cache_key(rmvpe_model_cache_key);
        #[cfg(feature = "ort")]
        let shared_waveform_shape = [1usize, input_samples_16k];
        #[cfg(feature = "ort")]
        let mut shared_waveform: Option<TensorRtSharedWaveform> = None;

        let (embedder, pitch, rvc) = if tensor_rt_run_mode.cuda_graph() {
            #[cfg(not(feature = "ort"))]
            {
                unreachable!("cuda_graph run mode requires the `ort` feature")
            }
            #[cfg(feature = "ort")]
            {
                report_progress(
                    &config,
                    LoadProgress::LoadingModel {
                        role: LoadModelRole::ContentVec,
                    },
                );
                let mut embedder_probe = HubertEmbedderSession::load(
                    config.embedder,
                    config.provider,
                    expected_feat_channels,
                    config.embedder_output,
                    Some(contentvec_profile.clone()),
                    TensorRtRunMode::PinnedCpu,
                    TensorRtSessionPurpose::Probe,
                )?;
                let warmup = tensor_rt_warmup_feature_len(
                    &mut embedder_probe,
                    input_samples_16k,
                    extra_convert_samples,
                    rvc_sample_rate,
                )?;
                drop(embedder_probe);
                let feature_len = warmup.rvc_feature_len;
                // ORT fixed-shape (CUDA graph) does not bind streaming I/O; such
                // models are rejected at load above, so pass no stream hop here.
                let rvc_profile = TensorRtSessionProfile::rvc(
                    feature_len,
                    expected_feat_channels_usize,
                    &rvc_info.io_names,
                    None,
                )
                .with_gpu_priority(config.gpu_priority)
                .with_gpu_device_id(gpu_device_id)
                .with_optional_model_cache_key(rvc_model_cache_key.clone());
                info!(
                "fixed runtime profiles backend={} sample_rate={} chunk_samples={} contentvec={} rmvpe={} rvc={}",
                config.provider.label(),
                config.sample_rate,
                config.chunk_samples,
                contentvec_profile.profile_shapes,
                rmvpe_profile.profile_shapes,
                rvc_profile.profile_shapes
            );

                report_progress(
                    &config,
                    LoadProgress::LoadingModel {
                        role: LoadModelRole::Rmvpe,
                    },
                );
                let mut pitch_probe = RmvpePitchSession::load(
                    config.f0_model,
                    config.provider,
                    Some(rmvpe_profile.clone()),
                    TensorRtRunMode::PinnedCpu,
                    TensorRtSessionPurpose::Probe,
                )?;
                let rmvpe_output_shape = pitch_probe
                    .warmup_output_shape(rmvpe_input_samples_16k, config.f0.f0_threshold)?;
                drop(pitch_probe);

                report_progress(
                    &config,
                    LoadProgress::LoadingModel {
                        role: LoadModelRole::Rvc,
                    },
                );
                let mut rvc_probe = RvcModelSession::load(
                    config.model,
                    config.provider,
                    Some(rvc_profile.clone()),
                    Some(rvc_info.expected_feat_channels),
                    TensorRtRunMode::PinnedCpu,
                    TensorRtSessionPurpose::Probe,
                    rvc_info.io_names.clone(),
                )?;
                let rvc_output_shape = rvc_probe.warmup_output_shape(
                    feature_len,
                    rvc_info.expected_feat_channels,
                    config.speaker_id,
                )?;
                drop(rvc_probe);

                let mut embedder = HubertEmbedderSession::load(
                    config.embedder,
                    config.provider,
                    expected_feat_channels,
                    config.embedder_output,
                    Some(contentvec_profile),
                    tensor_rt_run_mode,
                    TensorRtSessionPurpose::Final,
                )?;
                shared_waveform = if tensor_rt_run_mode.device_io() {
                    Some(TensorRtSharedWaveform::new(
                        &embedder.session,
                        &shared_waveform_shape,
                        gpu_device_id,
                    )?)
                } else {
                    None
                };
                embedder.enable_tensorrt_binding(
                    &warmup.contentvec_output_shape,
                    shared_waveform.as_ref(),
                )?;

                let mut pitch = RmvpePitchSession::load(
                    config.f0_model,
                    config.provider,
                    Some(rmvpe_profile),
                    tensor_rt_run_mode,
                    TensorRtSessionPurpose::Final,
                )?;
                pitch.enable_tensorrt_binding(&rmvpe_output_shape, config.f0.f0_threshold, None)?;

                let mut rvc = RvcModelSession::load(
                    config.model,
                    config.provider,
                    Some(rvc_profile),
                    Some(rvc_info.expected_feat_channels),
                    tensor_rt_run_mode,
                    TensorRtSessionPurpose::Final,
                    rvc_info.io_names.clone(),
                )?;
                rvc.enable_tensorrt_binding(&rvc_output_shape, config.speaker_id)?;
                (embedder, pitch, rvc)
            }
        } else if config.provider.is_tensorrt() {
            // Native TensorRT engines self-report their fixed output shapes after
            // deserialize, so there is no warmup inference here: the RVC
            // `feature_len` is derived arithmetically from the ContentVec engine's
            // output frame count. Engine builds run in an isolated helper process
            // (native_tensorrt.rs has no in-process Builder), so the historical
            // "build RVC before other TensorRT runtimes in the same process"
            // ordering no longer applies and ContentVec can load first.
            report_native_load_progress(&config, &contentvec_profile, LoadModelRole::ContentVec);
            let embedder = HubertEmbedderSession::load(
                config.embedder,
                config.provider,
                expected_feat_channels,
                config.embedder_output,
                Some(contentvec_profile),
                tensor_rt_run_mode,
                TensorRtSessionPurpose::Final,
            )?;
            let contentvec_frames = match embedder.native_contentvec_output_frames() {
                Some(frames) => frames?,
                None => bail!("native TensorRT embedder is missing its engine"),
            };
            let feature_len =
                derive_rvc_feature_len(contentvec_frames, extra_convert_samples, rvc_sample_rate)?;
            // Native TensorRT models streaming: pass the frame hop so the profile
            // includes the `nsf_noise` `[1, feature_len*frame_hop, 1]` input.
            let rvc_profile = TensorRtSessionProfile::rvc(
                feature_len,
                expected_feat_channels_usize,
                &rvc_info.io_names,
                stream_params.map(|stream| stream.frame_hop),
            )
            .with_gpu_priority(config.gpu_priority)
            .with_gpu_device_id(gpu_device_id)
            .with_optional_model_cache_key(rvc_model_cache_key.clone());
            info!(
                "fixed runtime profiles backend={} sample_rate={} chunk_samples={} contentvec={} rmvpe={} rvc={}",
                config.provider.label(),
                config.sample_rate,
                config.chunk_samples,
                embedder
                    .tensor_rt_profile
                    .as_ref()
                    .map(|profile| profile.profile_shapes.as_str())
                    .unwrap_or("none"),
                rmvpe_profile.profile_shapes,
                rvc_profile.profile_shapes
            );
            report_native_load_progress(&config, &rvc_profile, LoadModelRole::Rvc);
            let mut rvc = RvcModelSession::load(
                config.model,
                config.provider,
                Some(rvc_profile),
                Some(rvc_info.expected_feat_channels),
                tensor_rt_run_mode,
                TensorRtSessionPurpose::Final,
                rvc_info.io_names.clone(),
            )?;
            // Validates the engine frame/channel counts against the runtime
            // profile; native engines self-report their output shape and use no
            // ORT IoBinding, so the returned shape is intentionally discarded.
            rvc.warmup_output_shape(
                feature_len,
                rvc_info.expected_feat_channels,
                config.speaker_id,
            )?;

            report_native_load_progress(&config, &rmvpe_profile, LoadModelRole::Rmvpe);
            let mut pitch = RmvpePitchSession::load(
                config.f0_model,
                config.provider,
                Some(rmvpe_profile),
                tensor_rt_run_mode,
                TensorRtSessionPurpose::Final,
            )?;
            pitch.warmup_output_shape(rmvpe_input_samples_16k, config.f0.f0_threshold)?;

            (embedder, pitch, rvc)
        } else {
            #[cfg(not(feature = "ort"))]
            {
                bail!(
                    "provider {} requires the `ort` feature; this build supports native TensorRT only",
                    config.provider.label()
                )
            }
            #[cfg(feature = "ort")]
            {
                report_progress(
                    &config,
                    LoadProgress::LoadingModel {
                        role: LoadModelRole::ContentVec,
                    },
                );
                let mut embedder = HubertEmbedderSession::load(
                    config.embedder,
                    config.provider,
                    expected_feat_channels,
                    config.embedder_output,
                    Some(contentvec_profile),
                    tensor_rt_run_mode,
                    TensorRtSessionPurpose::Final,
                )?;
                let warmup = tensor_rt_warmup_feature_len(
                    &mut embedder,
                    input_samples_16k,
                    extra_convert_samples,
                    rvc_sample_rate,
                )?;
                let feature_len = warmup.rvc_feature_len;
                shared_waveform = if tensor_rt_run_mode.device_io() {
                    Some(TensorRtSharedWaveform::new(
                        &embedder.session,
                        &shared_waveform_shape,
                        gpu_device_id,
                    )?)
                } else {
                    None
                };
                embedder.enable_tensorrt_binding(
                    &warmup.contentvec_output_shape,
                    shared_waveform.as_ref(),
                )?;
                // Pinned-CPU IoBinding (Windows ML TensorRT-RTX) binds streaming
                // I/O, so pass the frame hop to add the `nsf_noise`
                // `[1, feature_len*frame_hop, 1]` input to its profile. The
                // device-I/O / CUDA-graph modes do not, and streaming models are
                // rejected for them at load above, so they get no stream hop.
                let rvc_stream_frame_hop = if tensor_rt_run_mode == TensorRtRunMode::PinnedCpu {
                    stream_params.map(|stream| stream.frame_hop)
                } else {
                    None
                };
                let rvc_profile = TensorRtSessionProfile::rvc(
                    feature_len,
                    expected_feat_channels_usize,
                    &rvc_info.io_names,
                    rvc_stream_frame_hop,
                )
                .with_gpu_priority(config.gpu_priority)
                .with_gpu_device_id(gpu_device_id)
                .with_optional_model_cache_key(rvc_model_cache_key.clone());
                info!(
                "fixed runtime profiles backend={} sample_rate={} chunk_samples={} contentvec={} rmvpe={} rvc={}",
                config.provider.label(),
                config.sample_rate,
                config.chunk_samples,
                embedder
                    .tensor_rt_profile
                    .as_ref()
                    .map(|profile| profile.profile_shapes.as_str())
                    .unwrap_or("none"),
                rmvpe_profile.profile_shapes,
                rvc_profile.profile_shapes
            );

                report_progress(
                    &config,
                    LoadProgress::LoadingModel {
                        role: LoadModelRole::Rmvpe,
                    },
                );
                let mut pitch = RmvpePitchSession::load(
                    config.f0_model,
                    config.provider,
                    Some(rmvpe_profile),
                    tensor_rt_run_mode,
                    TensorRtSessionPurpose::Final,
                )?;
                let rmvpe_output_shape =
                    pitch.warmup_output_shape(rmvpe_input_samples_16k, config.f0.f0_threshold)?;
                pitch.enable_tensorrt_binding(&rmvpe_output_shape, config.f0.f0_threshold, None)?;

                report_progress(
                    &config,
                    LoadProgress::LoadingModel {
                        role: LoadModelRole::Rvc,
                    },
                );
                let mut rvc = RvcModelSession::load(
                    config.model,
                    config.provider,
                    Some(rvc_profile),
                    Some(rvc_info.expected_feat_channels),
                    tensor_rt_run_mode,
                    TensorRtSessionPurpose::Final,
                    rvc_info.io_names.clone(),
                )?;
                let rvc_output_shape = rvc.warmup_output_shape(
                    feature_len,
                    rvc_info.expected_feat_channels,
                    config.speaker_id,
                )?;
                rvc.enable_tensorrt_binding(&rvc_output_shape, config.speaker_id)?;
                (embedder, pitch, rvc)
            }
        };

        Ok(Self {
            embedder,
            pitch,
            rvc,
            #[cfg(feature = "ort")]
            shared_waveform,
            speaker_id: config.speaker_id,
            pitch_shift: config.pitch_shift,
            f0_threshold: config.f0.f0_threshold,
            silence_threshold: config.f0.silence_threshold,
            input_gain: config.input_gain,
            input_denoiser: build_input_denoiser(&config),
            noise_gate_threshold: config.noise_gate_threshold,
            noise_gate_attack_ms: config.noise_gate_shaping.attack_ms,
            noise_gate_release_ms: config.noise_gate_shaping.release_ms,
            noise_gate_floor: config.noise_gate_shaping.floor,
            noise_gate_sample_rate: config.sample_rate as f32,
            output_extra_ms: config.output_extra_ms,
            volume_excluded_ms: config.volume_excluded_ms,
            rvc_sample_rate,
            extra_convert_samples,
            rmvpe_input_samples_16k,
            output_gain: config.output_gain,
            volume_envelope: config.output_dynamics.volume_envelope,
            rms_mix_rate: config.output_dynamics.rms_mix_rate,
            auto_output_gain: config.output_dynamics.auto_output_gain,
            target_output_rms: config.output_dynamics.target_output_rms,
            max_output_gain: config.output_dynamics.max_output_gain,
            stream_state: RvcStreamState::new_configured(
                rvc_sample_rate,
                rnd_channels,
                stream_params,
                config.sample_rate,
            )?,
            chunk_timing,
            input_sample_rate: config.sample_rate,
            rnd_channels,
            stream_params,
            rnd_scratch: Vec::new(),
            nsf_scratch: Vec::new(),
            phase_out_scratch: Vec::new(),
            input_scratch: Vec::new(),
            feature_tensor: FeatureTensor::default(),
            input_reference_scratch: Vec::new(),
            rms_mix_scratch: dsp::RmsMixScratch::default(),
            pitchf_untrimmed_scratch: Vec::new(),
            pitchf_scratch: Vec::new(),
            pitch_scratch: Vec::new(),
            f0_postprocess: F0Postprocessor::new(config.f0.postprocess.clone()),
            pitchf_postprocessed_scratch: Vec::new(),
        })
    }

    /// Runtime-mutable conversion parameters. These mirror the matching
    /// `RvcPipelineConfig` fields and let a host (e.g. the VST3 plugin) drive
    /// them from automation between chunks without reloading the pipeline.
    pub fn set_pitch_shift(&mut self, pitch_shift: f32) {
        self.pitch_shift = pitch_shift;
    }

    pub fn set_speaker_id(&mut self, speaker_id: i64) {
        self.speaker_id = speaker_id;
    }

    pub fn set_input_gain(&mut self, input_gain: f32) {
        self.input_gain = input_gain;
    }

    /// Live-update the input noise gate. Toggling on (re)builds the gate from
    /// the stored attack/release/floor; while it stays on, only the threshold
    /// changes so the envelope/gain state carries across chunks. Attack and
    /// release are not live-adjustable (they shape the smoothing coefficients
    /// fixed at construction).
    pub fn set_noise_gate(&mut self, enabled: bool, threshold: f32) {
        self.noise_gate_threshold = threshold;
        // Standalone live-parameter updates must not replace a configured
        // stateful denoiser. Future denoiser variants need the same guard.
        #[cfg(feature = "rnnoise")]
        if matches!(self.input_denoiser, InputDenoiser::Rnnoise(_)) {
            return;
        }
        // GTCRN owns the 16 kHz input-denoise seam in `stream_state`; a live gate
        // toggle must never build a device-rate gate that would fight it.
        #[cfg(feature = "gtcrn")]
        if self.stream_state.gtcrn.is_some() {
            return;
        }
        if !enabled {
            self.input_denoiser = InputDenoiser::Off;
            return;
        }
        match &mut self.input_denoiser {
            InputDenoiser::Gate(gate) => gate.set_threshold(threshold),
            _ => {
                self.input_denoiser = InputDenoiser::Gate(dsp::NoiseGate::new(
                    self.noise_gate_sample_rate,
                    threshold,
                    self.noise_gate_attack_ms,
                    self.noise_gate_release_ms,
                    self.noise_gate_floor,
                ));
            }
        }
    }

    pub fn set_output_gain(&mut self, output_gain: f32) {
        self.output_gain = output_gain;
    }

    /// Apply a full [`LiveParams`] snapshot. The single per-chunk live-update
    /// path: the standalone worker and the VST3 host callback both build a
    /// `LiveParams` and call this, so the live knobs stay wired identically
    /// across front-ends. `set_noise_gate` keeps the rnnoise guard, so passing
    /// `noise_gate_enabled: false` never tears down an active rnnoise stage.
    pub fn apply_live(&mut self, live: &LiveParams) {
        self.set_pitch_shift(live.pitch_shift);
        self.set_speaker_id(live.speaker_id);
        self.set_input_gain(live.input_gain);
        self.set_output_gain(live.output_gain);
        self.set_noise_gate(live.noise_gate_enabled, live.noise_gate_threshold);
    }

    /// Discards rolling audio/F0 context while retaining loaded inference
    /// sessions.
    ///
    /// Standalone passthrough can pause RVC processing for an arbitrary time.
    /// On resume, old audio context must not be concatenated with the new live
    /// input. Denoiser state is reset too because RNNoise owns fixed-delay
    /// buffers that may otherwise emit audio captured before the pause.
    pub fn reset_streaming_state(&mut self) -> Result<()> {
        self.input_denoiser.reset(
            self.noise_gate_sample_rate,
            self.noise_gate_threshold,
            NoiseGateShaping {
                attack_ms: self.noise_gate_attack_ms,
                release_ms: self.noise_gate_release_ms,
                floor: self.noise_gate_floor,
            },
        )?;
        // Preserve the loaded GTCRN denoiser across a context reset, but reset its
        // fixed-delay/cache state (mirroring the RNNoise reset above) so it does
        // not emit audio captured before the pause.
        #[cfg(feature = "gtcrn")]
        let gtcrn = self.stream_state.gtcrn.take();
        // Rebuilding the stream state re-seeds the rolling noise/phase state and
        // zeroes the NSF phase / absolute position, so a resumed stream is
        // reproducible from its start.
        self.stream_state = RvcStreamState::new_configured(
            self.rvc_sample_rate,
            self.rnd_channels,
            self.stream_params,
            self.input_sample_rate,
        )?;
        #[cfg(feature = "gtcrn")]
        if let Some(mut gtcrn) = gtcrn {
            gtcrn.reset()?;
            self.stream_state.gtcrn = Some(gtcrn);
        }
        self.input_scratch.clear();
        self.rnd_scratch.clear();
        self.nsf_scratch.clear();
        self.phase_out_scratch.clear();
        self.feature_tensor = FeatureTensor::default();
        self.input_reference_scratch.clear();
        self.rms_mix_scratch = dsp::RmsMixScratch::default();
        self.pitchf_untrimmed_scratch.clear();
        self.pitchf_scratch.clear();
        self.pitch_scratch.clear();
        self.pitchf_postprocessed_scratch.clear();
        Ok(())
    }

    /// Explicit input content delay, available before priming. Keep both rate
    /// domains rational until the converter adds the joiner's delay, then round
    /// once on its output grid; rounding stages separately can trim real audio.
    pub fn input_content_delay(&self) -> ContentDelay {
        ContentDelay::from_samples(
            self.input_denoiser.content_delay_samples(),
            self.input_sample_rate,
        ) + ContentDelay::from_samples(
            self.stream_state.input_content_delay_16k(),
            super::shape::EMBEDDER_SAMPLE_RATE,
        )
    }

    pub fn input_content_delay_samples(&self, output_sample_rate: u32) -> usize {
        self.input_content_delay()
            .output_samples(output_sample_rate)
    }
}

impl VoiceModel for RvcPipeline {
    fn input_content_delay(&self) -> ContentDelay {
        RvcPipeline::input_content_delay(self)
    }

    fn input_content_delay_samples(&self, output_sample_rate: u32) -> usize {
        RvcPipeline::input_content_delay_samples(self, output_sample_rate)
    }

    fn process(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        out_audio: &mut Vec<f32>,
        out_pitchf: &mut Vec<f32>,
    ) -> Result<ModelOutput> {
        validate_process_timing(
            self.chunk_timing,
            self.input_sample_rate,
            audio.len(),
            sample_rate,
        )?;
        let total_start = Instant::now();
        let input_gain = self.input_gain.max(0.0);
        let apply_gain = (input_gain - 1.0).abs() > f32::EPSILON;
        let denoiser_active = !matches!(self.input_denoiser, InputDenoiser::Off);
        // Only gain≠1.0 or an active denoiser need an owned buffer; the no-op
        // path keeps `audio` as a zero-copy borrow. When an owned buffer is
        // needed, reuse `input_scratch` instead of allocating a fresh Vec every
        // chunk. Take it into a local so the borrow does not collide with the
        // later `&mut self.stream_state` call; it is written back at the end of
        // the function to retain the allocation.
        let mut input_scratch = std::mem::take(&mut self.input_scratch);
        if apply_gain || denoiser_active {
            input_scratch.clear();
            if apply_gain {
                input_scratch.extend(
                    audio
                        .iter()
                        .map(|sample| (*sample * input_gain).clamp(-1.0, 1.0)),
                );
            } else {
                input_scratch.extend_from_slice(audio);
            }
            // Noise reduction runs before RMS/silence detection and feature/F0
            // extraction so the model sees the cleaned signal.
            if denoiser_active {
                self.input_denoiser.process_in_place(&mut input_scratch)?;
            }
        }
        let input_audio: &[f32] = if apply_gain || denoiser_active {
            &input_scratch
        } else {
            audio
        };
        let output_extra_len = ms_to_samples(self.rvc_sample_rate, self.output_extra_ms);
        let volume_excluded_len = ms_to_samples(self.rvc_sample_rate, self.volume_excluded_ms);
        let stream_input = self.stream_state.generate_input(
            input_audio,
            sample_rate,
            output_extra_len,
            volume_excluded_len,
            self.extra_convert_samples,
        )?;
        // `input_audio` is no longer borrowed past this point; return the buffer
        // so its capacity is reused on the next chunk.
        self.input_scratch = input_scratch;

        // input_rms/silence are derived from the new 16 kHz increment (the same
        // post-input-denoiser signal ContentVec/F0 consume), not the raw
        // device-rate chunk — for every denoiser mode. This also changes the
        // `input_rms` reported in ModelOutput to the 16 kHz value (intended).
        let input_rms = stream_input.input_rms;
        let is_silent = self.silence_threshold > 0.0 && input_rms < self.silence_threshold;
        let output_silent = is_silent && self.stream_state.prev_silence;
        self.stream_state.prev_silence = is_silent;

        // Features
        let embedder_start = Instant::now();
        #[cfg(feature = "ort")]
        if let Some(shared_waveform) = self.shared_waveform.as_mut() {
            // Shared CUDA input is charged to embedder_time because the public
            // metrics do not have a separate transfer bucket. RMVPE uses a
            // separate upstream-RVC bucket window, so this stable device address
            // is now owned by ContentVec only.
            let h2d_us = shared_waveform.copy_from_slice(&self.stream_state.audio_16k_buffer)?;
            debug!(
                "shared waveform h2d backend={} samples={} consumers=contentvec h2d_us={}",
                self.embedder.provider.label(),
                self.stream_state.audio_16k_buffer.len(),
                h2d_us
            );
        }
        self.embedder.extract_into(
            &self.stream_state.audio_16k_buffer,
            &mut self.feature_tensor,
        )?;
        let raw_feature_len = self
            .feature_tensor
            .shape
            .get(1)
            .copied()
            .context("embedder output must be rank-3 [1, frames, channels]")?;
        if raw_feature_len <= 0 {
            bail!("embedder produced zero frames");
        }
        let raw_feature_len = usize::try_from(raw_feature_len)
            .context("embedder frame length does not fit in usize")?;
        let feature_len_before_trim = raw_feature_len
            .checked_mul(2)
            .context("repeated embedder frame length overflowed")?;

        let silence_front_frames =
            onnx_silence_front_feature_frames(self.extra_convert_samples, self.rvc_sample_rate);
        if silence_front_frames > 0 && silence_front_frames < feature_len_before_trim {
            if silence_front_frames.is_multiple_of(2) {
                // `silence_front_frames` is on RVC's repeated 10 ms grid. Drop
                // the equivalent ContentVec frames before repeat so discarded
                // context is not duplicated and shifted every chunk.
                self.feature_tensor
                    .trim_front_frames(silence_front_frames / 2)?;
                self.feature_tensor.repeat_frames(2)?;
            } else {
                self.feature_tensor.repeat_frames(2)?;
                self.feature_tensor
                    .trim_front_frames(silence_front_frames)?;
            }
        } else {
            self.feature_tensor.repeat_frames(2)?;
        }
        let embedder_time = embedder_start.elapsed();
        let feature_len = self
            .feature_tensor
            .shape
            .get(1)
            .copied()
            .and_then(|len| usize::try_from(len).ok())
            .context("trimmed embedder frame length does not fit in usize")?;
        // Pitch
        let pitch_start = Instant::now();
        // Extract raw (natural) F0: pass 0.0 so RMVPE does not pre-apply pitch
        // shift. pitch_shift is applied once in f0_postprocess after smoothing,
        // so clamp/octave/median act on natural F0. pitchf_buffer therefore
        // accumulates raw F0 (see the guardrail above the post-process call).
        let audio_16k_len = self.stream_state.audio_16k_buffer.len();
        let rmvpe_input_samples_16k = self.rmvpe_input_samples_16k.min(audio_16k_len);
        let rmvpe_window_start_samples = audio_16k_len - rmvpe_input_samples_16k;
        let rmvpe_audio_16k = &self.stream_state.audio_16k_buffer[rmvpe_window_start_samples..];
        let pitchf_raw = self
            .pitch
            .extract(rmvpe_audio_16k, 0.0, self.f0_threshold)?;
        let rmvpe_audio_16k_len = rmvpe_audio_16k.len();
        let pitchf_raw_len = pitchf_raw.len();
        self.stream_state
            .update_pitchf_from_rmvpe_window(pitchf_raw, rmvpe_window_start_samples);
        let pitch_frames = self.stream_state.pitchf_buffer.len();
        let pitch_time = pitch_start.elapsed();
        // RMVPE's center-padded STFT and ContentVec's convolutional frontend do
        // not expose the same frame count for the same waveform. First center
        // crop to the untrimmed ContentVec grid so a 183->180 case uses
        // pitchf[1..181], then apply the existing tail crop for silence_front.
        center_crop_pitchf_to_features_into(
            &self.stream_state.pitchf_buffer,
            feature_len_before_trim,
            &mut self.pitchf_untrimmed_scratch,
        );
        align_pitchf_to_features_into(
            &self.pitchf_untrimmed_scratch,
            feature_len,
            &mut self.pitchf_scratch,
        );
        // Guardrail: pitchf_buffer / pitchf_scratch hold raw (un-transposed) F0
        // because extract() above is called with 0.0 shift. Post-process the
        // RVC-aligned natural F0 and apply pitch_shift exactly once at the end.
        // Always run this, even when post-processing is disabled: the shift is
        // applied here, so skipping it would drop pitch shift entirely.
        // voiced_ratio and coarse_pitch_into below must use this post-processed
        // pitchf so the RVC inputs stay consistent (island/gap edits change the
        // voiced frame count).
        self.f0_postprocess.process_pitchf_into(
            &self.pitchf_scratch,
            self.pitch_shift,
            &mut self.pitchf_postprocessed_scratch,
        );
        let pitchf = self.pitchf_postprocessed_scratch.as_slice();
        debug!(
            "pitch update: audio_16k_samples={}, rmvpe_input_samples={}, rmvpe_window_start_samples={}, pitchf_raw_len={}, pitchf_buffer_len={}, feature_len={}",
            self.stream_state.audio_16k_buffer.len(),
            rmvpe_audio_16k_len,
            rmvpe_window_start_samples,
            pitchf_raw_len,
            self.stream_state.pitchf_buffer.len(),
            feature_len,
        );
        let voiced_ratio = voiced_ratio(pitchf);
        coarse_pitch_into(pitchf, &mut self.pitch_scratch);
        let pitch = self.pitch_scratch.as_slice();

        if SKIP_SILENT_CHUNKS && output_silent {
            // If previous chunk was also silent, keep returning silence without running the model to reduce CPU usage and avoid latency spikes from the embedder when silence ends.
            out_audio.clear();
            out_audio.resize(stream_input.out_size, 0.0);
            out_pitchf.clear();
            return Ok(ModelOutput {
                sample_rate: self.rvc_sample_rate,
                inference_time: total_start.elapsed(),
                embedder_time: Duration::ZERO,
                pitch_time: Duration::ZERO,
                rvc_time: Duration::ZERO,
                input_rms,
                voiced_ratio: 0.0,
                raw_output_samples: stream_input.out_size,
                output_rms: 0.0,
                applied_output_gain: 1.0,
                feature_frames: 0,
                pitch_frames: 0,
                silent: true,
                convert_size: stream_input.convert_size,
                out_size: stream_input.out_size,
                model_input_samples: self.stream_state.audio_buffer.len(),
                volume: stream_input.volume,
            });
        }

        // Latent noise: produce this chunk's `[1, channels, feature_len]` rnd
        // window into the reused `rnd_scratch`, aligned to `pitchf`/`feats` by the
        // same center-crop + tail selection used above. `has_rnd` is false (and
        // the buffer untouched) for models that sample their own noise.
        let has_rnd = self.stream_state.time_state.rnd_window_into(
            feature_len_before_trim,
            feature_len,
            &mut self.rnd_scratch,
        );
        let rnd = has_rnd.then_some(self.rnd_scratch.as_slice());

        // Streaming exports: the NSF source noise `[1, audio_len, 1]` (selected on
        // the output-sample grid by the same alignment as `rnd`) and the
        // window-start NSF phase. Both are `None`/false for conventional exports.
        let has_nsf = self.stream_state.time_state.nsf_noise_window_into(
            feature_len_before_trim,
            feature_len,
            &mut self.nsf_scratch,
        );
        let nsf_noise = has_nsf.then_some(self.nsf_scratch.as_slice());
        let phase_in = self.stream_state.time_state.phase_in();
        if let Some(phase) = phase_in {
            let (abs_frame, abs_sample) = self.stream_state.time_state.absolute_position();
            debug!(
                "rvc streaming state phase_in={phase} abs_frame={abs_frame} abs_sample={abs_sample} nsf_noise_len={}",
                self.nsf_scratch.len()
            );
        }

        // RVC. The converted samples are written straight into the caller-owned
        // `out_audio` buffer (reused across chunks) and all post-processing runs
        // in place on it; the output pitchf goes into `out_pitchf`.
        let rvc_start = Instant::now();
        // Streaming exports return the per-sample `streaming_nsf_phase`; collect it
        // so the next chunk's `phase_in` can be read at the next window start.
        let phase_out = phase_in.is_some().then_some(&mut self.phase_out_scratch);
        self.rvc.infer(
            &self.feature_tensor.data,
            &self.feature_tensor.shape,
            feature_len,
            pitch,
            pitchf,
            self.speaker_id,
            rnd,
            nsf_noise,
            phase_in,
            phase_out,
            out_audio,
        )?;
        let rvc_time = rvc_start.elapsed();
        // Carry the NSF phase to the next chunk. Prefer the model's per-sample
        // phase output (exact, per the streaming contract); fall back to CPU
        // accumulation when the export does not provide a usable per-sample output.
        // No-op for conventional exports.
        if !self
            .stream_state
            .time_state
            .set_phase_from_output(&self.phase_out_scratch)
        {
            self.stream_state.time_state.advance_phase(pitchf);
        }
        let raw_output_samples = out_audio.len();
        keep_tail_in_place(out_audio, stream_input.out_size);
        pitchf_tail_for_output_into(pitchf, out_audio.len(), self.rvc_sample_rate, out_pitchf);
        let output_envelope = if self.volume_envelope {
            stream_input.volume.sqrt().clamp(0.0, 1.0)
        } else {
            1.0
        };
        dsp::clamp_scale_in_place(out_audio, output_envelope);
        if self.rms_mix_rate < 1.0 {
            // Captured before apply_rms_mix mutates `out_audio`, but only used
            // in the debug! below; skip the extra RMS pass when debug is off.
            let output_rms_before_mix = if tracing::enabled!(tracing::Level::DEBUG) {
                dsp::rms(out_audio)
            } else {
                0.0
            };
            // `out_audio` has already been trimmed to the same tail that SOLA
            // will search over. Use the input buffer tail with the same
            // duration; taking the head would compare against past context
            // added only to stabilize the model.
            // Reference the 16 kHz rolling signal (resampled 16 kHz -> RVC rate),
            // the same signal ContentVec/F0 see, so the RMS-mix level matches the
            // model's input for every denoiser mode.
            let input_reference = self.stream_state.output_reference_audio(
                super::shape::EMBEDDER_SAMPLE_RATE,
                self.rvc_sample_rate,
                out_audio.len(),
                &mut self.input_reference_scratch,
            )?;
            dsp::apply_rms_mix_with_scratch(
                input_reference,
                out_audio,
                self.rvc_sample_rate as usize,
                self.rms_mix_rate,
                &mut self.rms_mix_scratch,
            );
            debug!(
                "rms_mix_rate={:.3} input_ref_rms={:.8} output_rms_before_mix={:.8} output_rms_after_mix={:.8}",
                self.rms_mix_rate,
                dsp::rms(input_reference),
                output_rms_before_mix,
                dsp::rms(out_audio)
            );
        }
        let output_rms_before_gain = dsp::rms(out_audio);
        let applied_output_gain = self.applied_output_gain(output_rms_before_gain);
        let output_rms = if (applied_output_gain - 1.0).abs() > f32::EPSILON {
            dsp::apply_gain_and_rms(out_audio, applied_output_gain)
        } else {
            output_rms_before_gain
        };

        Ok(ModelOutput {
            sample_rate: self.rvc_sample_rate,
            inference_time: total_start.elapsed(),
            embedder_time,
            pitch_time,
            rvc_time,
            input_rms,
            voiced_ratio,
            raw_output_samples,
            output_rms,
            applied_output_gain,
            feature_frames: feature_len,
            pitch_frames,
            silent: output_silent,
            convert_size: stream_input.convert_size,
            out_size: stream_input.out_size,
            model_input_samples: self.stream_state.audio_buffer.len(),
            volume: stream_input.volume,
        })
    }
}

fn validate_process_timing(
    timing: RvcChunkTiming,
    configured_rate: u32,
    samples: usize,
    sample_rate: u32,
) -> Result<()> {
    if sample_rate != configured_rate || samples != timing.input_chunk_samples {
        bail!(
            "RVC pipeline expects {} samples at {} Hz, received {} samples at {} Hz; pad the final chunk or reload the pipeline when timing changes",
            timing.input_chunk_samples,
            configured_rate,
            samples,
            sample_rate,
        );
    }
    Ok(())
}

/// Derive the streaming NSF time base from inspected metadata, validating the
/// format version we implement. `None` for conventional (non-streaming) exports.
fn stream_params_from_info(info: &super::inspect::RvcModelInfo) -> Result<Option<StreamParams>> {
    let Some(stream) = info.stream else {
        return Ok(None);
    };
    if stream.version != 1 {
        bail!(
            "unsupported RVC stream_format_version {}; this build implements version 1",
            stream.version
        );
    }
    Ok(Some(StreamParams {
        frame_hop: stream.frame_hop,
        sample_rate: stream.sample_rate,
    }))
}

fn provider_needs_fixed_shape_profile(provider: Provider) -> bool {
    provider_uses_fixed_shape(provider) || provider_drives_nvtrtx(provider)
}

/// True when sessions for this provider run through the NvTensorRtRtx
/// (TensorRT-RTX) Windows ML catalog EP — either because it was requested
/// explicitly, or because the "Auto" Windows ML provider (`Provider::WindowsMl`)
/// resolves to it as the best available catalog EP on this machine.
///
/// That EP rejects dynamic shapes, so it needs the same fixed-shape profile and
/// pinned-CPU run mode as the explicit provider. Without this, `Provider::WindowsMl`
/// on a machine with TensorRT-RTX installed takes the dynamic-shape `load` path,
/// passes no profile, and the session build fails with
/// "Windows ML NvTensorRtRtx requires a fixed-shape profile". The catalog lookup
/// is cached (OnceLock) and matches what `load_session` selects later, so the
/// load-time routing decision and the session build stay in agreement.
fn provider_drives_nvtrtx(provider: Provider) -> bool {
    if provider == Provider::WindowsMlNvTensorRtRtx {
        return true;
    }
    #[cfg(all(windows, feature = "windowsml"))]
    if provider == Provider::WindowsMl {
        return matches!(
            crate::windows_ml::try_register_best_catalog_ep(),
            Ok(Some(
                crate::windows_ml::CatalogExecutionProvider::NvTensorRtRtx
            ))
        );
    }
    false
}

impl RvcPipeline {
    fn applied_output_gain(&self, output_rms: f32) -> f32 {
        let manual_gain = self.output_gain.max(0.0);
        if !self.auto_output_gain || output_rms <= 1e-8 {
            return manual_gain;
        }
        let auto_gain = (self.target_output_rms.max(0.0) / output_rms)
            .clamp(1.0, self.max_output_gain.max(1.0));
        manual_gain * auto_gain
    }
}

impl std::fmt::Debug for RvcPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RvcPipeline")
            .field("speaker_id", &self.speaker_id)
            .field("pitch_shift", &self.pitch_shift)
            .field("f0_threshold", &self.f0_threshold)
            .field("silence_threshold", &self.silence_threshold)
            .field("input_gain", &self.input_gain)
            .field("output_extra_ms", &self.output_extra_ms)
            .field("volume_excluded_ms", &self.volume_excluded_ms)
            .field("extra_convert_samples", &self.extra_convert_samples)
            .field("rmvpe_input_samples_16k", &self.rmvpe_input_samples_16k)
            .field("output_gain", &self.output_gain)
            .field("volume_envelope", &self.volume_envelope)
            .field("rms_mix_rate", &self.rms_mix_rate)
            .field("auto_output_gain", &self.auto_output_gain)
            .field("target_output_rms", &self.target_output_rms)
            .field("max_output_gain", &self.max_output_gain)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod progress_tests {
    use super::*;

    #[test]
    fn process_timing_rejects_rate_and_chunk_changes_before_mutating_state() {
        let timing = RvcChunkTiming::from_ms(20, 44_100).unwrap();
        assert!(validate_process_timing(timing, 44_100, 882, 44_100).is_ok());
        assert!(validate_process_timing(timing, 44_100, 881, 44_100).is_err());
        assert!(validate_process_timing(timing, 44_100, 882, 48_000).is_err());
        assert!(validate_process_timing(timing, 44_100, 0, 44_100).is_err());
    }

    #[test]
    fn native_engine_build_progress_is_only_reported_for_cache_miss() {
        assert_eq!(native_engine_build_progress(true, LoadModelRole::Rvc), None);
        assert_eq!(
            native_engine_build_progress(false, LoadModelRole::Rvc),
            Some(LoadProgress::BuildingEngine {
                role: LoadModelRole::Rvc
            })
        );
    }
}
