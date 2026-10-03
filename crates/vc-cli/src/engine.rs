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
            let content_delay_ms = metrics
                .content_delay_samples
                .filter(|_| status.output_sample_rate > 0)
                .map(|samples| {
                    format!(
                        "{:.3}",
                        samples as f64 * 1000.0 / status.output_sample_rate as f64
                    )
                })
                .unwrap_or_else(|| "unknown".to_string());
            info!(
                "state={:?} chunks={} infer={}us processing={}us content_delay_ms={} (nominal; excludes devices, queues and chunk accumulation) input_rms={:.8} output_rms={:.8} input_overruns={} output_underruns={} output_dropped_samples={} output_buffer_samples={}",
                status.state,
                metrics.chunks,
                metrics.inference_us,
                metrics.processing_us,
                content_delay_ms,
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
    let report_file = args
        .performance_report
        .as_deref()
        .map(crate::performance_report::create)
        .transpose()?;
    if let Some(path) = &args.performance_report {
        let report_path = path.canonicalize()?;
        for output_path in std::iter::once(&args.output).chain(args.join_report.iter()) {
            if output_path.canonicalize().is_ok_and(|p| p == report_path) {
                anyhow::bail!("performance report must differ from WAV and join report outputs");
            }
        }
    }
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
    let load_started = Instant::now();
    let load_events = std::cell::RefCell::new(Vec::new());
    let progress = |event| {
        load_events.borrow_mut().push(serde_json::json!({
            "event": format!("{event:?}"), "elapsed_ms": load_started.elapsed().as_secs_f64() * 1000.0
        }));
    };
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
        progress: args
            .performance_report
            .as_ref()
            .map(|_| &progress as &dyn Fn(_)),
    };
    // GTCRN stays on the shared 16 kHz seam. The finite core adapter drains and
    // removes its declared content delay together with the other streaming
    // buffers, rather than truncating that delayed speech at the clip end.
    let model = load_wav_pipeline(denoiser_mode, args.gtcrn_model.as_deref(), pipeline_config)?;
    let load_duration = load_started.elapsed();
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
    let performance = report_file.as_ref().map(|_| {
        let mut report = crate::performance_report::build(&converted, args.performance_warmup_chunks,
            chunk_samples as f64 / f64::from(spec.sample_rate) * 1000.0, load_duration);
        report["load_progress_events"] = serde_json::json!(load_events.into_inner());
        report["load_progress_scope"] = serde_json::json!("Progress boundaries include preparation; intervals are not isolated model load measurements.");
        report["requested_provider"] = serde_json::json!(args.provider.label());
        report["configuration"] = serde_json::json!(format!("{args:?}"));
        report["sample_rate"] = serde_json::json!(spec.sample_rate);
        report["catalog_after_conversion"] = crate::performance_report::catalog_snapshot();
        report
    });
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
    if let (Some(file), Some(report)) = (report_file, performance) {
        serde_json::to_writer_pretty(file, &report)
            .context("failed to write performance report")?;
    }
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
    let channels = usize::from(spec.channels);
    let decoded = match spec.sample_format {
        hound::SampleFormat::Int => {
            anyhow::ensure!(
                matches!(spec.bits_per_sample, 8 | 16 | 24 | 32),
                "unsupported integer WAV bit depth: {}",
                spec.bits_per_sample
            );
            // Hound centers unsigned PCM8 and sign-extends wider PCM into i32
            // without scaling. Full scale depends on the source precision,
            // not the Rust sample type: a fixed i16 scale attenuates PCM8 and
            // an i16 decoder rejects PCM24/32.
            let full_scale = (1_u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / full_scale))
                .collect::<Result<Vec<_>, _>>()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>(),
    };
    let decoded = decoded.with_context(|| format!("failed to decode {}", path.display()))?;
    let frames = decoded.chunks_exact(channels);
    anyhow::ensure!(
        frames.remainder().is_empty(),
        "WAV contains an incomplete channel frame"
    );
    let samples = frames
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect();
    Ok((samples, spec))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::read_wav_mono;

    struct TestWav(PathBuf);

    impl TestWav {
        fn write<T: hound::Sample>(
            spec: hound::WavSpec,
            samples: impl IntoIterator<Item = T>,
        ) -> Self {
            static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "vc-rs-wav-test-{}-{}-{}.wav",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT_ID.fetch_add(1, Ordering::Relaxed)
            ));
            let file = Self(path);
            let mut writer = hound::WavWriter::create(&file.0, spec).unwrap();
            for sample in samples {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
            file
        }
    }

    impl Drop for TestWav {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn spec(
        bits_per_sample: u16,
        sample_format: hound::SampleFormat,
        channels: u16,
    ) -> hound::WavSpec {
        hound::WavSpec {
            channels,
            sample_rate: 48_000,
            bits_per_sample,
            sample_format,
        }
    }

    #[test]
    fn integer_wav_bit_depths_preserve_equal_full_scale_amplitudes() {
        let expected: [f32; 8] = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.0];
        for bits in [8, 16, 24, 32] {
            let full_scale = 1_i64 << (bits - 1);
            let encoded = expected
                .iter()
                .map(|&sample| (f64::from(sample) * full_scale as f64) as i32);
            let file = TestWav::write(spec(bits, hound::SampleFormat::Int, 1), encoded);
            let (samples, decoded_spec) = read_wav_mono(&file.0).unwrap();
            assert_eq!(samples.as_slice(), &expected, "PCM{bits} changed level");
            assert_eq!(decoded_spec.bits_per_sample, bits);
            assert_eq!(decoded_spec.sample_rate, 48_000);
        }
    }

    #[test]
    fn integer_and_float_stereo_wav_average_complete_channel_frames() {
        let integer = TestWav::write(
            spec(24, hound::SampleFormat::Int, 2),
            [1_i32 << 22, 0, -(1_i32 << 22), 1_i32 << 21],
        );
        let (samples, _) = read_wav_mono(&integer.0).unwrap();
        assert_eq!(samples, [0.25, -0.125]);

        let float = TestWav::write(
            spec(32, hound::SampleFormat::Float, 2),
            [1.5_f32, 0.5, -0.5, 0.25],
        );
        let (samples, _) = read_wav_mono(&float.0).unwrap();
        assert_eq!(samples, [1.0, -0.125]);
    }

    #[test]
    fn float_mono_wav_preserves_sample_values_without_integer_scaling() {
        let expected = [1.25_f32, -0.5, 0.0, 0.25];
        let file = TestWav::write(spec(32, hound::SampleFormat::Float, 1), expected);
        let (samples, _) = read_wav_mono(&file.0).unwrap();
        assert_eq!(samples, expected);
    }

    #[test]
    fn wav_with_incomplete_channel_frame_is_rejected() {
        let file = TestWav::write(spec(16, hound::SampleFormat::Int, 2), [1_i32, 2, 3, 4]);
        let mut bytes = fs::read(&file.0).unwrap();
        let data_start = hound::WavReader::new(Cursor::new(&bytes))
            .unwrap()
            .into_inner()
            .position() as usize;
        // Declare three interleaved samples for a stereo stream. Averaging a
        // short last frame would silently change the final sample's level.
        bytes[data_start - 4..data_start].copy_from_slice(&6_u32.to_le_bytes());
        fs::write(&file.0, bytes).unwrap();
        assert!(read_wav_mono(&file.0).is_err());
    }

    #[test]
    fn truncated_wav_sample_is_rejected() {
        let file = TestWav::write(spec(16, hound::SampleFormat::Int, 1), [1_i32, 2]);
        let mut bytes = fs::read(&file.0).unwrap();
        bytes.pop();
        fs::write(&file.0, bytes).unwrap();
        assert!(read_wav_mono(&file.0).is_err());
    }

    #[test]
    fn invalid_wav_header_is_rejected() {
        let file = TestWav::write(spec(16, hound::SampleFormat::Int, 1), [0_i32]);
        fs::write(&file.0, b"invalid WAV header").unwrap();
        assert!(read_wav_mono(&file.0).is_err());
    }
}
