use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use tracing::{debug, info};
use vc_app::{
    write_wav_mono, DenoiserMode, EngineController, EngineState, LiveParams, RealtimeConfig,
};
use vc_core::model_rvc::{
    convert_finite, set_process_gpu_priority, set_process_power_throttling, ChunkConverter,
    ChunkOutputConfig, F0Config, GpuPriority, NoiseGateShaping, OutputDynamicsConfig, RvcPipeline,
    RvcPipelineConfig,
};
use vc_core::sola::SmoothingKind;
use vc_core::validation::RvcChunkTiming;

use crate::cli::{Denoiser, RunArgs, Smoother, WavArgs};
use crate::join_report::JoinReport;

pub fn run_realtime(args: RunArgs) -> Result<()> {
    args.validate_audio_options().map_err(anyhow::Error::msg)?;
    args.validate_conversion_options()
        .map_err(anyhow::Error::msg)?;
    let denoiser_mode: DenoiserMode = args.denoiser_mode().into();
    let live = LiveParams {
        pitch_shift: args.pitch_shift,
        speaker_id: args.speaker_id,
        input_gain: args.input_gain,
        output_gain: args.output_gain,
        // Gate on/off is static for the CLI session (no live denoiser control),
        // so derive it from the selected mode; the unified live path applies it.
        noise_gate_enabled: denoiser_mode == DenoiserMode::NoiseGate,
        noise_gate_threshold: args.noise_gate_threshold,
    };
    let wasapi_input_exclusive = args.wasapi_input_exclusive();
    let wasapi_output_exclusive = args.wasapi_output_exclusive();
    let input_host = args.effective_input_host();
    let output_host = args.effective_output_host();
    let controller = EngineController::new(live);
    controller.apply_config(RealtimeConfig {
        model: args.model,
        embedder: args.embedder,
        embedder_output: args.embedder_output,
        f0_model: args.f0_model,
        provider: args.provider,
        gpu_priority: args.gpu_priority.into(),
        gpu_device_id: args.gpu_device_id,
        input_host,
        output_host,
        input_device: args.input,
        output_device: args.output,
        wasapi_input_exclusive,
        wasapi_output_exclusive,
        wasapi_buffer_ms: args.wasapi_buffer_ms,
        chunk_ms: args.chunk_ms,
        crossfade_ms: args.crossfade_ms,
        sola_search_ms: args.sola_search_ms,
        smoother: args.smoother.into(),
        rvc_output_tail_discard_ms: args.rvc_output_tail_discard_ms,
        extra_convert_ms: args.extra_convert_ms,
        f0: F0Config {
            f0_threshold: args.f0_threshold,
            silence_threshold: args.silence_threshold,
            ..F0Config::default()
        },
        denoiser_mode,
        gtcrn_model_dir: args.gtcrn_model,
        noise_gate_shaping: NoiseGateShaping {
            attack_ms: args.noise_gate_attack_ms,
            release_ms: args.noise_gate_release_ms,
            floor: args.noise_gate_floor,
        },
        output_dynamics: OutputDynamicsConfig {
            volume_envelope: args.volume_envelope,
            rms_mix_rate: args.rms_mix_rate,
            auto_output_gain: args.auto_output_gain,
            target_output_rms: args.target_output_rms,
            max_output_gain: args.max_output_gain,
        },
        passthrough: args.passthrough,
        debug_input_wav: args.debug_input_wav,
        debug_output_wav: args.debug_output_wav,
    })?;

    let running = Arc::new(AtomicBool::new(true));
    let ctrl_running = Arc::clone(&running);
    ctrlc::set_handler(move || ctrl_running.store(false, Ordering::SeqCst))?;
    let started = Instant::now();
    let mut last_log = Instant::now();
    info!("starting; press Ctrl+C to stop");
    while running.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(100));
        let (status, metrics, _) = controller.snapshot();
        if status.state == EngineState::Error {
            return Err(anyhow!(status.message));
        }
        if let Some(seconds) = args.duration_seconds {
            if started.elapsed() >= Duration::from_secs(seconds) {
                break;
            }
        }
        if last_log.elapsed() >= Duration::from_secs(1) {
            last_log = Instant::now();
            info!(
                "state={:?} chunks={} infer={}us input_rms={:.8} output_rms={:.8} input_overruns={} output_underruns={} output_dropped_samples={} output_buffer_samples={}",
                status.state,
                metrics.chunks,
                metrics.inference_us,
                metrics.input_rms,
                metrics.output_rms,
                metrics.input_overruns,
                metrics.output_underruns,
                metrics.output_dropped_samples,
                metrics.output_buffer_samples,
            );
        }
    }
    controller.stop()?;
    Ok(())
}

fn smoothing_kind(smoother: Smoother) -> SmoothingKind {
    match smoother {
        Smoother::Sola => SmoothingKind::Sola,
        Smoother::Psola => SmoothingKind::Psola,
    }
}

pub fn run_wav(args: WavArgs) -> Result<()> {
    args.validate_conversion_options()
        .map_err(anyhow::Error::msg)?;
    let (mut samples, spec) = read_wav_mono(&args.input)?;
    let chunk_samples =
        RvcChunkTiming::from_ms(args.chunk_ms, spec.sample_rate)?.input_chunk_samples;
    let denoiser_mode = args.denoiser_mode();
    let pipeline_input_gain = if denoiser_mode == Denoiser::Rnnoise {
        for sample in &mut samples {
            *sample = (*sample * args.input_gain.max(0.0)).clamp(-1.0, 1.0);
        }
        samples = process_rnnoise_finite(&samples, spec.sample_rate)?;
        1.0
    } else {
        args.input_gain
    };
    // Must cover the SOLA window the joiner actually uses, so the model emits
    // enough extra audio to feed `crossfade_ms` + `sola_search_ms` + tail.
    let output_extra_ms = args
        .crossfade_ms
        .saturating_add(args.sola_search_ms)
        .saturating_add(args.rvc_output_tail_discard_ms);
    // WAV mode builds the pipeline directly (realtime goes via vc-app, which
    // applies these on session start); set the process GPU priority and power
    // throttling here too. High also opts out of EcoQoS so a background run
    // keeps full clock.
    let gpu_priority: GpuPriority = args.gpu_priority.into();
    set_process_gpu_priority(gpu_priority);
    set_process_power_throttling(gpu_priority == GpuPriority::High);
    let pipeline_config = RvcPipelineConfig {
        model: &args.model,
        embedder: &args.embedder,
        embedder_output: args.embedder_output.as_deref(),
        f0_model: &args.f0_model,
        provider: args.provider,
        gpu_priority,
        gpu_device_id: args.gpu_device_id,
        sample_rate: spec.sample_rate,
        chunk_samples,
        speaker_id: args.speaker_id,
        pitch_shift: args.pitch_shift,
        f0: F0Config {
            f0_threshold: args.f0_threshold,
            // WAV mode treats nothing as silence so the whole clip converts.
            silence_threshold: 0.0,
            ..F0Config::default()
        },
        input_gain: pipeline_input_gain,
        noise_gate_enabled: denoiser_mode == Denoiser::NoiseGate,
        noise_gate_threshold: args.noise_gate_threshold,
        noise_gate_shaping: NoiseGateShaping {
            attack_ms: args.noise_gate_attack_ms,
            release_ms: args.noise_gate_release_ms,
            floor: args.noise_gate_floor,
        },
        output_extra_ms,
        volume_excluded_ms: args.crossfade_ms,
        extra_convert_ms: args.extra_convert_ms,
        output_gain: args.output_gain,
        output_dynamics: OutputDynamicsConfig {
            volume_envelope: args.volume_envelope,
            rms_mix_rate: args.rms_mix_rate,
            auto_output_gain: args.auto_output_gain,
            target_output_rms: args.target_output_rms,
            max_output_gain: args.max_output_gain,
        },
        progress: None,
    };
    // GTCRN stays on the shared 16 kHz seam. The finite core adapter drains and
    // removes its declared content delay together with the other streaming
    // buffers, rather than truncating that delayed speech at the clip end.
    let model = load_wav_pipeline(denoiser_mode, args.gtcrn_model.as_deref(), pipeline_config)?;
    let converter = ChunkConverter::new(
        model,
        ChunkOutputConfig {
            kind: smoothing_kind(args.smoother),
            output_sample_rate: spec.sample_rate,
            output_chunk_samples: chunk_samples,
            crossfade_ms: args.crossfade_ms,
            sola_search_ms: args.sola_search_ms,
            tail_discard_ms: args.rvc_output_tail_discard_ms,
        },
    );
    let converted = convert_finite(converter, &samples, spec.sample_rate, chunk_samples)?;
    let output = converted.audio;
    let chunks = converted
        .chunks
        .iter()
        .filter(|chunk| !chunk.flushing)
        .count();
    let mut join_report = args
        .join_report
        .as_ref()
        .map(|_| JoinReport::new(spec.sample_rate));

    for chunk in converted.chunks {
        debug!(
            "wav chunk={} flushing={} model_output_samples={}",
            chunk.index, chunk.flushing, chunk.stats.model_output_samples
        );
        if let (Some(report), Some(seam)) = (join_report.as_mut(), chunk.seam_sample) {
            // Measure the written, delay-compensated WAV, not the discarded
            // startup or FFT FIFO boundary. Drain calls can contain real speech.
            report.record_at(
                chunk.index,
                &output,
                seam,
                chunk.join,
                chunk.requested_crossfade_samples,
            );
        }
    }
    write_wav_mono(&args.output, &output, spec.sample_rate)?;
    info!(
        "wrote {} samples at {} Hz to {} (chunks={})",
        output.len(),
        spec.sample_rate,
        args.output.display(),
        chunks
    );
    if let (Some(report), Some(path)) = (join_report, args.join_report.as_ref()) {
        report.write_csv(path)?;
        info!("wrote join report to {}", path.display());
        // Summary goes to stderr via info! line-by-line so it is visible without
        // opening the CSV.
        for line in report.summary().lines() {
            info!("{line}");
        }
    }
    Ok(())
}

/// Load the WAV-mode pipeline, selecting the GTCRN loader when requested. The
/// disabled-feature path is a runtime error, not a build error.
fn load_wav_pipeline(
    denoiser_mode: Denoiser,
    gtcrn_model: Option<&std::path::Path>,
    config: RvcPipelineConfig<'_>,
) -> Result<RvcPipeline> {
    if denoiser_mode == Denoiser::Gtcrn {
        #[cfg(feature = "gtcrn")]
        {
            let model_dir = gtcrn_model
                .ok_or_else(|| anyhow!("GTCRN denoiser requires --gtcrn-model <dir>"))?;
            let backend = vc_core::denoise::GtcrnBackend::for_provider(
                config.provider,
                config.gpu_priority,
                config.gpu_device_id,
            );
            return RvcPipeline::load_with_gtcrn(
                config,
                vc_core::denoise::GtcrnConfig { model_dir, backend },
            );
        }
        #[cfg(not(feature = "gtcrn"))]
        {
            let _ = gtcrn_model;
            anyhow::bail!("GTCRN support is not enabled in this build");
        }
    }
    RvcPipeline::load(config)
}

#[cfg(feature = "rnnoise")]
fn process_rnnoise_finite(samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
    vc_core::denoise::RnnoiseDenoiser::process_finite(samples, sample_rate)
}

#[cfg(not(feature = "rnnoise"))]
fn process_rnnoise_finite(_samples: &[f32], _sample_rate: u32) -> Result<Vec<f32>> {
    anyhow::bail!("RNNoise support is not enabled in this build")
}

fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, hound::WavSpec)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let samples = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .collect::<Result<Vec<_>, _>>()?
            .chunks(channels)
            .map(|f| f.iter().map(|&x| x as f32 / 32768.0).sum::<f32>() / f.len() as f32)
            .collect(),
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()?
            .chunks(channels)
            .map(|f| f.iter().copied().sum::<f32>() / f.len() as f32)
            .collect(),
    };
    Ok((samples, spec))
}
