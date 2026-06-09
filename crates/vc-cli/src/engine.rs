use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use tracing::{debug, info};
use vc_app::{EngineController, EngineState, LiveParams, RealtimeConfig};
use vc_core::dsp;
use vc_core::model_rvc::{F0PostprocessConfig, RvcPipeline, RvcPipelineConfig, VoiceModel};
use vc_core::sola::{self, ChunkSmootherConfig, SmoothingKind};

use crate::cli::{RunArgs, Smoother, WavArgs, DEFAULT_CROSSFADE_MS, DEFAULT_SOLA_SEARCH_MS};

pub fn run_realtime(args: RunArgs) -> Result<()> {
    args.validate_audio_options().map_err(anyhow::Error::msg)?;
    let live = LiveParams {
        pitch_shift: args.pitch_shift,
        speaker_id: args.speaker_id,
        input_gain: args.input_gain,
        output_gain: args.output_gain,
        noise_gate_enabled: args.noise_gate,
        noise_gate_threshold: args.noise_gate_threshold,
    };
    let wasapi_input_exclusive = args.wasapi_input_exclusive();
    let wasapi_output_exclusive = args.wasapi_output_exclusive();
    let controller = EngineController::new(live);
    controller.apply_config(RealtimeConfig {
        model: args.model,
        embedder: args.embedder,
        embedder_output: args.embedder_output,
        f0_model: args.f0_model,
        provider: args.provider,
        gpu_priority: args.gpu_priority.into(),
        audio_backend: args.audio_backend.into(),
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
        f0_threshold: args.f0_threshold,
        silence_threshold: args.silence_threshold,
        noise_gate_attack_ms: args.noise_gate_attack_ms,
        noise_gate_release_ms: args.noise_gate_release_ms,
        noise_gate_floor: args.noise_gate_floor,
        volume_envelope: args.volume_envelope,
        rms_mix_rate: args.rms_mix_rate,
        auto_output_gain: args.auto_output_gain,
        target_output_rms: args.target_output_rms,
        max_output_gain: args.max_output_gain,
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

fn chunk_samples_for_rate(sample_rate: u32, chunk_ms: u32) -> usize {
    ((sample_rate as u64 * chunk_ms as u64) / 1000).max(128) as usize
}

pub fn run_wav(args: WavArgs) -> Result<()> {
    let (samples, spec) = read_wav_mono(&args.input)?;
    let chunk_samples = chunk_samples_for_rate(spec.sample_rate, args.chunk_ms);
    let output_extra_ms = DEFAULT_CROSSFADE_MS
        .saturating_add(DEFAULT_SOLA_SEARCH_MS)
        .saturating_add(args.rvc_output_tail_discard_ms);
    let mut model = RvcPipeline::load(RvcPipelineConfig {
        model: &args.model,
        embedder: &args.embedder,
        embedder_output: args.embedder_output.as_deref(),
        f0_model: &args.f0_model,
        provider: args.provider,
        gpu_priority: args.gpu_priority.into(),
        sample_rate: spec.sample_rate,
        chunk_samples,
        speaker_id: args.speaker_id,
        pitch_shift: args.pitch_shift,
        f0_threshold: args.f0_threshold,
        silence_threshold: 0.0,
        input_gain: args.input_gain,
        noise_gate_enabled: args.noise_gate,
        noise_gate_threshold: args.noise_gate_threshold,
        noise_gate_attack_ms: args.noise_gate_attack_ms,
        noise_gate_release_ms: args.noise_gate_release_ms,
        noise_gate_floor: args.noise_gate_floor,
        output_extra_ms,
        volume_excluded_ms: DEFAULT_CROSSFADE_MS,
        extra_convert_ms: args.extra_convert_ms,
        output_gain: args.output_gain,
        volume_envelope: args.volume_envelope,
        rms_mix_rate: args.rms_mix_rate,
        auto_output_gain: args.auto_output_gain,
        target_output_rms: args.target_output_rms,
        max_output_gain: args.max_output_gain,
        // F0 post-processing is disabled by default; CLI/preset wiring is a
        // separate task.
        f0_postprocess: F0PostprocessConfig::default(),
    })?;
    let mut output = Vec::with_capacity(samples.len());
    let mut chunks = 0usize;
    let mut final_tail = Vec::new();
    let preroll = vec![0.0; chunk_samples];
    let preroll_out = model.process(&preroll, spec.sample_rate)?;
    let mut joiner = sola::model_domain_chunk_smoother(ChunkSmootherConfig {
        kind: smoothing_kind(args.smoother),
        output_chunk_samples: chunk_samples,
        output_sample_rate: spec.sample_rate,
        model_sample_rate: preroll_out.sample_rate,
        crossfade_ms: DEFAULT_CROSSFADE_MS,
        sola_search_ms: DEFAULT_SOLA_SEARCH_MS,
        tail_discard_ms: args.rvc_output_tail_discard_ms,
    });
    joiner.prime_model_output(&preroll_out.audio, &preroll_out.pitchf);

    let mut fixed_chunk_pad = Vec::new();
    for chunk in samples.chunks(chunk_samples) {
        let model_input = wav_model_input_chunk(chunk, chunk_samples, &mut fixed_chunk_pad);
        let out = model.process(model_input, spec.sample_rate)?;
        debug!(
            "wav chunk={} input_samples={} output_samples={}",
            chunks,
            chunk.len(),
            out.audio.len()
        );
        chunks += 1;
        let prepared = sola::prepare_model_output(
            out,
            spec.sample_rate,
            chunk_samples,
            &mut joiner,
            Some(&mut final_tail),
        )?;
        output.extend_from_slice(&prepared.audio);
    }
    if output.len() < samples.len() {
        let missing = samples.len() - output.len();
        output.extend_from_slice(&final_tail[..missing.min(final_tail.len())]);
    }
    output.resize(samples.len(), 0.0);
    output.truncate(samples.len());
    write_wav_mono(&args.output, &output, spec.sample_rate)?;
    info!(
        "wrote {} samples at {} Hz to {} (chunks={})",
        output.len(),
        spec.sample_rate,
        args.output.display(),
        chunks
    );
    Ok(())
}

fn wav_model_input_chunk<'a>(
    chunk: &'a [f32],
    chunk_samples: usize,
    scratch: &'a mut Vec<f32>,
) -> &'a [f32] {
    if chunk.len() < chunk_samples {
        scratch.clear();
        scratch.extend_from_slice(chunk);
        scratch.resize(chunk_samples, 0.0);
        scratch.as_slice()
    } else {
        chunk
    }
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

fn write_wav_mono(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for sample in dsp::f32_to_i16(samples) {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{chunk_samples_for_rate, wav_model_input_chunk};

    #[test]
    fn computes_chunk_samples_per_sample_rate() {
        assert_eq!(chunk_samples_for_rate(48_000, 10), 480);
        assert_eq!(chunk_samples_for_rate(48_000, 1), 128);
    }

    #[test]
    fn pads_short_wav_chunk() {
        let mut scratch = Vec::new();
        assert_eq!(
            wav_model_input_chunk(&[0.25, -0.5], 4, &mut scratch),
            &[0.25, -0.5, 0.0, 0.0]
        );
    }
}
