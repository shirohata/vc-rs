use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use rtrb::RingBuffer;
use thread_priority::{set_current_thread_priority, ThreadPriority};
use tracing::{debug, error, info};

use crate::audio;
use crate::cli::{RunArgs, Smoother, WavArgs, DEFAULT_CROSSFADE_MS, DEFAULT_SOLA_SEARCH_MS};
use crate::dsp;
use crate::model_rvc::{PassthroughModel, RvcPipeline, RvcPipelineConfig, VoiceModel};
use crate::sola::{self, ChunkSmootherConfig, SmoothingKind};

const INPUT_QUEUE_CHUNKS: usize = 4;
const OUTPUT_QUEUE_CHUNKS: usize = 4;

#[derive(Default)]
struct Metrics {
    chunks: AtomicU64,
    input_overruns: AtomicU64,
    input_overrun_samples: AtomicU64,
    output_underruns: AtomicU64,
    output_underrun_samples: AtomicU64,
    output_dropped_samples: AtomicU64,
    output_buffer_samples: AtomicU64,
    last_inference_us: AtomicU64,
    last_embedder_us: AtomicU64,
    last_pitch_us: AtomicU64,
    last_rvc_us: AtomicU64,
    last_output_samples: AtomicU64,
    last_raw_output_samples: AtomicU64,
    last_feature_frames: AtomicU64,
    last_pitch_frames: AtomicU64,
    last_convert_size: AtomicU64,
    last_out_size: AtomicU64,
    last_model_input_samples: AtomicU64,
    last_volume_ppb: AtomicU64,
    last_sola_offset: AtomicU64,
    last_input_rms_ppb: AtomicU64,
    last_output_rms_ppb: AtomicU64,
    last_voiced_ratio_ppm: AtomicU64,
    last_applied_output_gain_ppm: AtomicU64,
    silent_chunks: AtomicU64,
    crossfade_samples: AtomicU64,
    sola_search_samples: AtomicU64,
}

pub fn run_realtime(args: RunArgs) -> Result<()> {
    set_current_thread_priority(ThreadPriority::Max).expect("failed to set thread priority");
    args.validate_audio_options().map_err(anyhow::Error::msg)?;

    if !args.passthrough && args.model.is_none() {
        return Err(anyhow!("--model is required unless --passthrough is set"));
    }
    if !args.passthrough && args.embedder.is_none() {
        return Err(anyhow!(
            "--embedder is required unless --passthrough is set"
        ));
    }
    if !args.passthrough && args.f0_model.is_none() {
        return Err(anyhow!(
            "--f0-model is required unless --passthrough is set"
        ));
    }

    let realtime_audio = audio::RealtimeAudio::open(
        args.audio_backend,
        args.wasapi_input_exclusive(),
        args.wasapi_output_exclusive(),
        args.input.as_deref(),
        args.output.as_deref(),
        args.wasapi_period_ms,
        args.wasapi_buffer_periods,
    )?;
    let sample_rate = realtime_audio.sample_rate();
    let chunk_samples = ((sample_rate as u64 * args.chunk_ms as u64) / 1000).max(128) as usize;
    let crossfade_samples = sola::ms_to_samples(sample_rate, args.crossfade_ms);
    let sola_search_samples = sola::ms_to_samples(sample_rate, args.sola_search_ms);
    let extra_convert_ms = args.extra_convert_ms;
    let smoother_kind = args.smoother;
    let smoothing_enabled = !args.passthrough;
    let rvc_output_tail_discard_ms = if args.passthrough {
        0
    } else {
        args.rvc_output_tail_discard_ms
    };
    let output_extra_ms = if args.passthrough {
        0
    } else {
        args.crossfade_ms
            .saturating_add(args.sola_search_ms)
            .saturating_add(rvc_output_tail_discard_ms)
    };
    let duration_seconds = args.duration_seconds;
    let debug_output_wav = args.debug_output_wav.clone();
    let debug_input_wav = args.debug_input_wav.clone();

    info!("audio backend: {}", realtime_audio.backend_label());
    info!("input device: {}", realtime_audio.input_name());
    info!("output device: {}", realtime_audio.output_name());
    info!(
        "sample_rate={} chunk_ms={} chunk_samples={} wasapi_period_ms={} wasapi_buffer_periods={} crossfade_samples={} sola_search_samples={} smoother={:?} rvc_output_tail_discard_ms={} extra_convert_ms={}",
        sample_rate, args.chunk_ms, chunk_samples, args.wasapi_period_ms, args.wasapi_buffer_periods, crossfade_samples, sola_search_samples, smoother_kind, rvc_output_tail_discard_ms, extra_convert_ms
    );
    if !args.passthrough {
        info!("silence_threshold={}", args.silence_threshold);
        if args.chunk_ms >= 500 {
            info!(
                "large chunk_ms={} adds at least one chunk of startup latency before first converted audio",
                args.chunk_ms
            );
        }
    }

    let mut model: Box<dyn VoiceModel> = if args.passthrough {
        info!("running in passthrough mode");
        Box::new(PassthroughModel)
    } else {
        let model_path = args.model.as_ref().expect("checked above");
        let embedder_path = args.embedder.as_ref().expect("checked above");
        let f0_model_path = args.f0_model.as_ref().expect("checked above");
        Box::new(RvcPipeline::load(RvcPipelineConfig {
            model: model_path,
            embedder: embedder_path,
            embedder_output: args.embedder_output.as_deref(),
            f0_model: f0_model_path,
            provider: args.provider,
            sample_rate,
            chunk_samples,
            speaker_id: args.speaker_id,
            pitch_shift: args.pitch_shift,
            f0_threshold: args.f0_threshold,
            silence_threshold: args.silence_threshold,
            input_gain: args.input_gain,
            output_extra_ms,
            volume_excluded_ms: args.crossfade_ms,
            extra_convert_ms,
            output_gain: args.output_gain,
            volume_envelope: args.volume_envelope,
            rms_mix_rate: args.rms_mix_rate,
            auto_output_gain: args.auto_output_gain,
            target_output_rms: args.target_output_rms,
            max_output_gain: args.max_output_gain,
        })?)
    };

    let input_buffer_capacity = chunk_samples * INPUT_QUEUE_CHUNKS;
    let (mut input_producer, mut input_consumer) = RingBuffer::<f32>::new(input_buffer_capacity);
    let output_buffer_capacity = chunk_samples * OUTPUT_QUEUE_CHUNKS;
    let (mut output_producer, mut output_consumer) = RingBuffer::<f32>::new(output_buffer_capacity);
    let metrics = Arc::new(Metrics::default());
    metrics
        .crossfade_samples
        .store(crossfade_samples as u64, Ordering::Relaxed);
    metrics
        .sola_search_samples
        .store(sola_search_samples as u64, Ordering::Relaxed);
    let running = Arc::new(AtomicBool::new(true));
    let debug_output_samples = Arc::new(Mutex::new(Vec::<f32>::new()));
    let debug_input_samples = Arc::new(Mutex::new(Vec::<f32>::new()));

    let worker_running = Arc::clone(&running);
    let worker_metrics = Arc::clone(&metrics);
    let worker_debug_output_samples = Arc::clone(&debug_output_samples);
    let worker_debug_input_samples = Arc::clone(&debug_input_samples);
    let capture_debug_output = debug_output_wav.is_some();
    let capture_debug_input = debug_input_wav.is_some();
    let crossfade_ms = args.crossfade_ms;
    let sola_search_ms = args.sola_search_ms;
    let worker = thread::spawn(move || {
        let mut smoother = None::<(u32, sola::ChunkSmoother)>;
        let mut input_acc = Vec::<f32>::with_capacity(chunk_samples * 2);
        while worker_running.load(Ordering::SeqCst) {
            while input_acc.len() < chunk_samples {
                match input_consumer.pop() {
                    Ok(sample) => input_acc.push(sample),
                    Err(_) => break,
                }
            }
            if input_acc.len() < chunk_samples {
                thread::sleep(Duration::from_millis(2));
                continue;
            }

            let chunk: Vec<f32> = input_acc.drain(..chunk_samples).collect();
            if capture_debug_input {
                if let Ok(mut debug) = worker_debug_input_samples.lock() {
                    debug.extend_from_slice(&chunk);
                }
            }

            match model.process(&chunk, sample_rate) {
                Ok(out) => {
                    if out.inference_time.as_millis() > 150 {
                        debug!("model inference took {} ms (embedder: {} ms, pitch: {} ms, rvc: {} ms)", out.inference_time.as_millis(), out.embedder_time.as_millis(), out.pitch_time.as_millis(), out.rvc_time.as_millis());
                        let out_cap = output_producer.buffer().capacity();
                        debug!(
                            "in buffer {}/{}, out buffer {}/{};\n",
                            input_consumer.slots(),
                            input_consumer.buffer().capacity(),
                            out_cap - output_producer.slots(),
                            out_cap
                        );
                    }
                    worker_metrics.chunks.fetch_add(1, Ordering::Relaxed);
                    worker_metrics
                        .last_inference_us
                        .store(out.inference_time.as_micros() as u64, Ordering::Relaxed);
                    worker_metrics
                        .last_embedder_us
                        .store(out.embedder_time.as_micros() as u64, Ordering::Relaxed);
                    worker_metrics
                        .last_pitch_us
                        .store(out.pitch_time.as_micros() as u64, Ordering::Relaxed);
                    worker_metrics
                        .last_rvc_us
                        .store(out.rvc_time.as_micros() as u64, Ordering::Relaxed);
                    worker_metrics
                        .last_raw_output_samples
                        .store(out.raw_output_samples as u64, Ordering::Relaxed);
                    worker_metrics
                        .last_feature_frames
                        .store(out.feature_frames as u64, Ordering::Relaxed);
                    worker_metrics
                        .last_pitch_frames
                        .store(out.pitch_frames as u64, Ordering::Relaxed);
                    worker_metrics
                        .last_convert_size
                        .store(out.convert_size as u64, Ordering::Relaxed);
                    worker_metrics
                        .last_out_size
                        .store(out.out_size as u64, Ordering::Relaxed);
                    worker_metrics
                        .last_model_input_samples
                        .store(out.model_input_samples as u64, Ordering::Relaxed);
                    worker_metrics.last_volume_ppb.store(
                        (out.volume.max(0.0) * 1_000_000_000.0) as u64,
                        Ordering::Relaxed,
                    );
                    worker_metrics.last_input_rms_ppb.store(
                        (out.input_rms.max(0.0) * 1_000_000_000.0) as u64,
                        Ordering::Relaxed,
                    );
                    worker_metrics.last_output_rms_ppb.store(
                        (out.output_rms.max(0.0) * 1_000_000_000.0) as u64,
                        Ordering::Relaxed,
                    );
                    worker_metrics.last_applied_output_gain_ppm.store(
                        (out.applied_output_gain.max(0.0) * 1_000_000.0) as u64,
                        Ordering::Relaxed,
                    );
                    worker_metrics.last_voiced_ratio_ppm.store(
                        (out.voiced_ratio.clamp(0.0, 1.0) * 1_000_000.0) as u64,
                        Ordering::Relaxed,
                    );
                    if out.silent {
                        worker_metrics.silent_chunks.fetch_add(1, Ordering::Relaxed);
                    }
                    let output_sample_rate = out.sample_rate;
                    let output_silent = out.silent;
                    let audio = if smoothing_enabled {
                        if smoother.as_ref().map(|(sample_rate, _)| *sample_rate)
                            != Some(output_sample_rate)
                        {
                            let joiner = sola::model_domain_chunk_smoother(ChunkSmootherConfig {
                                kind: smoothing_kind(smoother_kind),
                                output_chunk_samples: chunk_samples,
                                output_sample_rate: sample_rate,
                                model_sample_rate: output_sample_rate,
                                crossfade_ms,
                                sola_search_ms,
                                tail_discard_ms: rvc_output_tail_discard_ms,
                            });
                            worker_metrics
                                .crossfade_samples
                                .store(joiner.crossfade_samples() as u64, Ordering::Relaxed);
                            worker_metrics
                                .sola_search_samples
                                .store(joiner.sola_search_samples() as u64, Ordering::Relaxed);
                            smoother = Some((output_sample_rate, joiner));
                        }
                        let smoother =
                            &mut smoother.as_mut().expect("smoother initialized above").1;
                        match sola::prepare_model_output(
                            out,
                            sample_rate,
                            chunk_samples,
                            smoother,
                            None,
                        ) {
                            Ok(prepared) => {
                                worker_metrics
                                    .last_sola_offset
                                    .store(prepared.sola_offset as u64, Ordering::Relaxed);
                                prepared.audio
                            }
                            Err(err) => {
                                error!("output smoothing/resampling failed: {err:#}");
                                worker_running.store(false, Ordering::SeqCst);
                                break;
                            }
                        }
                    } else {
                        match dsp::resample_mono(
                            &out.audio,
                            output_sample_rate as usize,
                            sample_rate as usize,
                        ) {
                            Ok(audio) => audio,
                            Err(err) => {
                                error!("output resampling failed: {err:#}");
                                worker_running.store(false, Ordering::SeqCst);
                                break;
                            }
                        }
                    };
                    worker_metrics
                        .last_output_samples
                        .store(audio.len() as u64, Ordering::Relaxed);
                    if capture_debug_output {
                        if let Ok(mut debug) = worker_debug_output_samples.lock() {
                            debug.extend_from_slice(&audio);
                        }
                    }
                    let mut dropped = 0u64;
                    if output_silent
                        && output_buffer_capacity - output_producer.slots() > chunk_samples
                    {
                        debug!("model output is silent, pushing silence to output buffer to reduce latency");
                    } else {
                        for sample in audio {
                            if output_producer.push(sample).is_err() {
                                // rtrb is SPSC and only the consumer can pop, so on full buffer
                                // we discard the new sample (latest) instead of forcibly evicting
                                // the oldest sample as the previous VecDeque implementation did.
                                dropped += 1;
                            }
                        }
                    }
                    if dropped > 0 {
                        worker_metrics
                            .output_dropped_samples
                            .fetch_add(dropped, Ordering::Relaxed);
                    }
                    worker_metrics.output_buffer_samples.store(
                        output_buffer_capacity.saturating_sub(output_producer.slots()) as u64,
                        Ordering::Relaxed,
                    );
                }
                Err(err) => {
                    error!("model processing failed: {err:#}");
                    worker_running.store(false, Ordering::SeqCst);
                }
            }
        }
    });

    let input_metrics = Arc::clone(&metrics);
    let input_running = Arc::clone(&running);
    let input_stream = realtime_audio.build_input_stream(move |samples| {
        if !input_running.load(Ordering::Relaxed) {
            return;
        }
        let mut dropped = 0u64;
        for sample in samples {
            if input_producer.push(*sample).is_err() {
                dropped += 1;
            }
        }
        if dropped > 0 {
            input_metrics.input_overruns.fetch_add(1, Ordering::Relaxed);
            input_metrics
                .input_overrun_samples
                .fetch_add(dropped, Ordering::Relaxed);
        }
    })?;

    let output_metrics = Arc::clone(&metrics);
    let output_running = Arc::clone(&running);
    let output_stream = realtime_audio.build_output_stream(move |out| {
        if !output_running.load(Ordering::Relaxed) {
            out.fill(0.0);
            return;
        }
        let mut underrun = false;
        let mut underrun_samples = 0u64;
        for sample in out {
            if let Ok(next) = output_consumer.pop() {
                *sample = next;
            } else {
                *sample = 0.0;
                underrun = true;
                underrun_samples += 1;
            }
        }
        output_metrics
            .output_buffer_samples
            .store(output_consumer.cached_slots() as u64, Ordering::Relaxed);
        if underrun {
            output_metrics
                .output_underruns
                .fetch_add(1, Ordering::Relaxed);
            output_metrics
                .output_underrun_samples
                .fetch_add(underrun_samples, Ordering::Relaxed);
        }
    })?;

    let ctrl_running = Arc::clone(&running);
    ctrlc::set_handler(move || {
        ctrl_running.store(false, Ordering::SeqCst);
    })?;

    output_stream.play()?;
    input_stream.play()?;
    info!("running; press Ctrl+C to stop");

    let started = Instant::now();
    let mut last = Instant::now();
    while running.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(100));
        if let Some(duration_seconds) = duration_seconds {
            if started.elapsed() >= Duration::from_secs(duration_seconds) {
                running.store(false, Ordering::SeqCst);
            }
        }
        if last.elapsed() >= Duration::from_secs(1) {
            last = Instant::now();
            info!(
                "chunks={} infer={}us embedder={}us pitch={}us rvc={}us feature_frames={} pitch_frames={} convert_size={} out_size={} model_input_samples={} raw_out_samples={} out_samples={} input_rms={:.8} vol={:.8} output_rms={:.8} output_gain={:.2} voiced_ratio={:.3} silent_chunks={} sola_offset={} output_buffer_samples={} output_dropped_samples={} output_underrun_samples={} input_overrun_samples={} crossfade_samples={} sola_search_samples={} input_overruns={} output_underruns={}",
                metrics.chunks.load(Ordering::Relaxed),
                metrics.last_inference_us.load(Ordering::Relaxed),
                metrics.last_embedder_us.load(Ordering::Relaxed),
                metrics.last_pitch_us.load(Ordering::Relaxed),
                metrics.last_rvc_us.load(Ordering::Relaxed),
                metrics.last_feature_frames.load(Ordering::Relaxed),
                metrics.last_pitch_frames.load(Ordering::Relaxed),
                metrics.last_convert_size.load(Ordering::Relaxed),
                metrics.last_out_size.load(Ordering::Relaxed),
                metrics.last_model_input_samples.load(Ordering::Relaxed),
                metrics.last_raw_output_samples.load(Ordering::Relaxed),
                metrics.last_output_samples.load(Ordering::Relaxed),
                metrics.last_input_rms_ppb.load(Ordering::Relaxed) as f64 / 1_000_000_000.0,
                metrics.last_volume_ppb.load(Ordering::Relaxed) as f64 / 1_000_000_000.0,
                metrics.last_output_rms_ppb.load(Ordering::Relaxed) as f64 / 1_000_000_000.0,
                metrics.last_applied_output_gain_ppm.load(Ordering::Relaxed) as f64 / 1_000_000.0,
                metrics.last_voiced_ratio_ppm.load(Ordering::Relaxed) as f64 / 1_000_000.0,
                metrics.silent_chunks.load(Ordering::Relaxed),
                metrics.last_sola_offset.load(Ordering::Relaxed),
                metrics.output_buffer_samples.load(Ordering::Relaxed),
                metrics.output_dropped_samples.load(Ordering::Relaxed),
                metrics.output_underrun_samples.load(Ordering::Relaxed),
                metrics.input_overrun_samples.load(Ordering::Relaxed),
                metrics.crossfade_samples.load(Ordering::Relaxed),
                metrics.sola_search_samples.load(Ordering::Relaxed),
                metrics.input_overruns.load(Ordering::Relaxed),
                metrics.output_underruns.load(Ordering::Relaxed),
            );
        }
    }

    drop(input_stream);
    drop(output_stream);
    worker
        .join()
        .map_err(|_| anyhow!("worker thread panicked"))?;
    if let Some(path) = debug_output_wav {
        if let Ok(samples) = debug_output_samples.lock() {
            write_wav_mono(&path, &samples, sample_rate)?;
            info!(
                "wrote realtime debug output {} samples at {} Hz to {}",
                samples.len(),
                sample_rate,
                path.display()
            );
        }
    }
    if let Some(path) = debug_input_wav {
        if let Ok(samples) = debug_input_samples.lock() {
            write_wav_mono(&path, &samples, sample_rate)?;
            info!(
                "wrote realtime debug input {} samples at {} Hz to {}",
                samples.len(),
                sample_rate,
                path.display()
            );
        }
    }
    Ok(())
}

fn smoothing_kind(smoother: Smoother) -> SmoothingKind {
    match smoother {
        Smoother::Sola => SmoothingKind::Sola,
        Smoother::Psola => SmoothingKind::Psola,
    }
}

pub fn run_wav(args: WavArgs) -> Result<()> {
    let (samples, spec) = read_wav_mono(&args.input)?;
    let chunk_samples = ((spec.sample_rate as u64 * args.chunk_ms as u64) / 1000).max(128) as usize;
    let output_extra_ms = DEFAULT_CROSSFADE_MS
        .saturating_add(DEFAULT_SOLA_SEARCH_MS)
        .saturating_add(args.rvc_output_tail_discard_ms);
    let mut model = RvcPipeline::load(RvcPipelineConfig {
        model: &args.model,
        embedder: &args.embedder,
        embedder_output: args.embedder_output.as_deref(),
        f0_model: &args.f0_model,
        provider: args.provider,
        sample_rate: spec.sample_rate,
        chunk_samples,
        speaker_id: args.speaker_id,
        pitch_shift: args.pitch_shift,
        f0_threshold: args.f0_threshold,
        silence_threshold: 0.0,
        input_gain: args.input_gain,
        output_extra_ms,
        volume_excluded_ms: DEFAULT_CROSSFADE_MS,
        extra_convert_ms: args.extra_convert_ms,
        output_gain: args.output_gain,
        volume_envelope: args.volume_envelope,
        rms_mix_rate: args.rms_mix_rate,
        auto_output_gain: args.auto_output_gain,
        target_output_rms: args.target_output_rms,
        max_output_gain: args.max_output_gain,
    })?;
    let mut output = Vec::with_capacity(samples.len());
    let mut total_inference_us = 0u128;
    let mut total_embedder_us = 0u128;
    let mut total_pitch_us = 0u128;
    let mut total_rvc_us = 0u128;
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
            "wav chunk={} input_samples={} model_input_samples={} output_samples={} raw_output_samples={} convert_size={} out_size={} feature_frames={} pitch_frames={} volume={}",
            chunks,
            chunk.len(),
            model_input.len(),
            out.audio.len(),
            out.raw_output_samples,
            out.convert_size,
            out.out_size,
            out.feature_frames,
            out.pitch_frames,
            out.volume
        );
        total_inference_us += out.inference_time.as_micros();
        total_embedder_us += out.embedder_time.as_micros();
        total_pitch_us += out.pitch_time.as_micros();
        total_rvc_us += out.rvc_time.as_micros();
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
    if output.len() < samples.len() {
        output.resize(samples.len(), 0.0);
    }
    output.truncate(samples.len());
    write_wav_mono(&args.output, &output, spec.sample_rate)?;
    info!(
        "wrote {} samples at {} Hz to {} (chunks={} chunk_ms={} smoother={:?} inference {}us embedder {}us pitch {}us rvc {}us)",
        output.len(),
        spec.sample_rate,
        args.output.display(),
        chunks,
        args.chunk_ms,
        args.smoother,
        total_inference_us,
        total_embedder_us,
        total_pitch_us,
        total_rvc_us
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
        hound::SampleFormat::Int => {
            let raw: Vec<i16> = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;
            raw.chunks(channels)
                .map(|frame| {
                    frame.iter().map(|&x| x as f32 / 32768.0).sum::<f32>() / frame.len() as f32
                })
                .collect()
        }
        hound::SampleFormat::Float => {
            let raw: Vec<f32> = reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?;
            raw.chunks(channels)
                .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
                .collect()
        }
    };
    Ok((samples, spec))
}

fn write_wav_mono(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in dsp::f32_to_i16(samples) {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::wav_model_input_chunk;

    #[test]
    fn pads_short_wav_chunk_for_fixed_shape_model_input() {
        let mut scratch = Vec::new();
        let input = [0.25, -0.5];

        let out = wav_model_input_chunk(&input, 4, &mut scratch);

        assert_eq!(out, &[0.25, -0.5, 0.0, 0.0]);
    }

    #[test]
    fn leaves_full_wav_chunk_unpadded() {
        let mut scratch = Vec::new();
        let input = [0.25, -0.5, 0.5, -0.25];

        let out = wav_model_input_chunk(&input, 4, &mut scratch);

        assert_eq!(out, input);
        assert!(scratch.is_empty());
    }
}
