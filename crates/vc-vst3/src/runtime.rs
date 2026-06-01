//! Realtime bridge between the host's `process()` callback and the RVC pipeline.
//!
//! Mirrors the CLI's `engine.rs` worker model: the audio thread only pushes
//! input and pops output through lock-free SPSC ring buffers, while a dedicated
//! worker thread owns the `RvcPipeline`, runs inference, and smooths/resamples
//! the result back to the host sample rate. Inference and allocation never run
//! on the audio thread.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use nih_plug::prelude::util;
use rtrb::{Consumer, Producer, RingBuffer};
use vc_core::model_rvc::{RvcPipeline, RvcPipelineConfig, VoiceModel};
use vc_core::sola::{self, ChunkSmoother, ChunkSmootherConfig};

use crate::config::PluginConfig;
use crate::params::VcRvcParams;

const INPUT_QUEUE_CHUNKS: usize = 4;
const OUTPUT_QUEUE_CHUNKS: usize = 4;

fn chunk_samples_for_rate(sample_rate: u32, chunk_ms: u32) -> usize {
    ((sample_rate as u64 * chunk_ms as u64) / 1000).max(128) as usize
}

/// How long [`PluginRuntime::drop`] waits for the worker to stop before
/// detaching it. Keeps the host's unload/deactivate call from blocking on a
/// slow, non-cancelable model load.
const SHUTDOWN_JOIN_TIMEOUT: Duration = Duration::from_millis(200);

/// Owns the worker thread and the audio-thread ends of the ring buffers.
pub struct PluginRuntime {
    input_producer: Producer<f32>,
    output_consumer: Consumer<f32>,
    running: Arc<AtomicBool>,
    /// Set by the worker (via [`FinishGuard`]) when `run` returns, so `drop` can
    /// tell "stopped quickly" from "still loading" without a blocking join.
    finished: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    mono_in: Vec<f32>,
    mono_out: Vec<f32>,
    /// Reported plugin latency in host samples (for the host's PDC).
    pub latency_samples: u32,
}

/// Marks `finished` on worker exit, even on early return or panic.
struct FinishGuard(Arc<AtomicBool>);

impl Drop for FinishGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl PluginRuntime {
    /// Start the worker and allocate the ring buffers for the given host rate.
    /// `max_block` is the host's maximum block size used to pre-size scratch.
    pub fn start(
        config: PluginConfig,
        params: Arc<VcRvcParams>,
        sample_rate: u32,
        max_block: usize,
    ) -> Self {
        let chunk_samples = chunk_samples_for_rate(sample_rate, config.chunk_ms);
        let crossfade_ms = config.crossfade_ms;
        let sola_search_ms = config.sola_search_ms;
        let tail_discard_ms = config.rvc_output_tail_discard_ms;
        let output_extra_ms = crossfade_ms
            .saturating_add(sola_search_ms)
            .saturating_add(tail_discard_ms);

        let (input_producer, input_consumer) =
            RingBuffer::<f32>::new(chunk_samples * INPUT_QUEUE_CHUNKS);
        let (output_producer, output_consumer) =
            RingBuffer::<f32>::new(chunk_samples * OUTPUT_QUEUE_CHUNKS);

        let running = Arc::new(AtomicBool::new(true));
        let finished = Arc::new(AtomicBool::new(false));

        // Latency estimate: one chunk of input buffering plus the smoothing /
        // tail context, expressed in host samples. RVC has additional inherent
        // algorithmic latency; this is a starting estimate the host can use for
        // delay compensation and may be refined empirically.
        let extra_samples = chunk_samples_for_rate(sample_rate, output_extra_ms);
        let latency_samples = (chunk_samples + extra_samples) as u32;

        let worker = WorkerCtx {
            config,
            params,
            sample_rate,
            chunk_samples,
            crossfade_ms,
            sola_search_ms,
            tail_discard_ms,
            running: Arc::clone(&running),
            finished: Arc::clone(&finished),
            input_consumer,
            output_producer,
        }
        .spawn();

        Self {
            input_producer,
            output_consumer,
            running,
            finished,
            worker: Some(worker),
            mono_in: Vec::with_capacity(max_block),
            mono_out: vec![0.0; max_block],
            latency_samples,
        }
    }

    /// Audio-thread entry point. Downmixes input to mono, queues it, and fills
    /// the output channels from the worker's converted audio (silence on
    /// underrun). Allocation-free and lock-free.
    pub fn process_block(&mut self, channels: &mut [&mut [f32]]) {
        if channels.is_empty() {
            return;
        }
        let n = channels[0].len();

        // Downmix to mono.
        self.mono_in.clear();
        self.mono_in.resize(n, 0.0);
        if channels.len() >= 2 {
            let (left, right) = (&channels[0], &channels[1]);
            for i in 0..n {
                self.mono_in[i] = 0.5 * (left[i] + right[i]);
            }
        } else {
            self.mono_in.copy_from_slice(&channels[0][..n]);
        }

        // Queue input; drop on overflow (worker is behind, audio keeps flowing).
        let _ = self.input_producer.push_partial_slice(&self.mono_in);

        // Pull up to n converted samples; pad the remainder with silence.
        if self.mono_out.len() < n {
            self.mono_out.resize(n, 0.0);
        }
        let want = n.min(self.output_consumer.slots());
        let mut filled = 0;
        if want > 0 {
            if let Ok(chunk) = self.output_consumer.read_chunk(want) {
                let (a, b) = chunk.as_slices();
                self.mono_out[..a.len()].copy_from_slice(a);
                self.mono_out[a.len()..a.len() + b.len()].copy_from_slice(b);
                filled = a.len() + b.len();
                chunk.commit_all();
            }
        }
        for sample in &mut self.mono_out[filled..n] {
            *sample = 0.0;
        }

        // Fan out mono to every output channel.
        for channel in channels.iter_mut() {
            channel[..n].copy_from_slice(&self.mono_out[..n]);
        }
    }
}

impl Drop for PluginRuntime {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        let Some(handle) = self.worker.take() else {
            return;
        };
        // If the worker is idle or finishing a chunk it exits almost
        // immediately, so wait briefly and join to reclaim it cleanly. If it is
        // mid model-load (seconds, and not cancelable) we detach instead of
        // blocking the host's unload/deactivate thread; the orphaned worker sees
        // `running == false` and exits on its own once the load completes,
        // freeing its pipeline and GPU resources.
        let deadline = Instant::now() + SHUTDOWN_JOIN_TIMEOUT;
        while !self.finished.load(Ordering::SeqCst) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        if self.finished.load(Ordering::SeqCst) {
            let _ = handle.join();
        }
        // else: detached — dropping `handle` does not join.
    }
}

/// Everything the worker thread needs, moved into it on spawn.
struct WorkerCtx {
    config: PluginConfig,
    params: Arc<VcRvcParams>,
    sample_rate: u32,
    chunk_samples: usize,
    crossfade_ms: u32,
    sola_search_ms: u32,
    tail_discard_ms: u32,
    running: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
    input_consumer: Consumer<f32>,
    output_producer: Producer<f32>,
}

impl WorkerCtx {
    fn spawn(self) -> JoinHandle<()> {
        thread::Builder::new()
            .name("vc-vst3-rvc".to_string())
            .spawn(move || self.run())
            .expect("failed to spawn vc-vst3 worker thread")
    }

    fn run(mut self) {
        // Signal `finished` on exit (normal, error, or panic) so a concurrent
        // `PluginRuntime::drop` can join or detach without blocking.
        let _finish_guard = FinishGuard(Arc::clone(&self.finished));

        // Load the pipeline up front (seconds for GPU backends). Until it
        // succeeds — or if no models are configured — the output ring stays
        // empty and the plugin emits silence.
        let mut pipeline = if self.config.has_models() {
            match self.load_pipeline() {
                Ok(pipeline) => Some(pipeline),
                Err(err) => {
                    nih_plug::nih_error!("vc-vst3: failed to load RVC pipeline: {err:#}");
                    None
                }
            }
        } else {
            nih_plug::nih_warn!("vc-vst3: no models configured; running silent");
            None
        };

        // Loading can take seconds, during which the audio thread kept queuing
        // input. Drop that backlog so conversion starts from the current audio
        // instead of replaying samples from before the model was ready.
        if pipeline.is_some() {
            self.drain_input();
        }

        let output_sample_rate = self.sample_rate;
        let output_chunk_samples = self.chunk_samples;
        let mut smoother: Option<(u32, ChunkSmoother)> = None;
        let mut input_acc = Vec::<f32>::with_capacity(self.chunk_samples * 2);
        let mut prepared = Vec::<f32>::with_capacity(output_chunk_samples * 2);

        while self.running.load(Ordering::SeqCst) {
            // Accumulate one input chunk.
            while input_acc.len() < self.chunk_samples {
                let needed = self.chunk_samples - input_acc.len();
                let available = self.input_consumer.slots().min(needed);
                if available == 0 {
                    break;
                }
                let old_len = input_acc.len();
                input_acc.resize(old_len + available, 0.0);
                if self
                    .input_consumer
                    .pop_entire_slice(&mut input_acc[old_len..])
                    .is_err()
                {
                    input_acc.truncate(old_len);
                    break;
                }
            }
            if input_acc.len() < self.chunk_samples {
                thread::sleep(Duration::from_millis(2));
                continue;
            }

            let chunk = &input_acc[..self.chunk_samples];
            let Some(pipeline) = pipeline.as_mut() else {
                // No pipeline: discard input and stay silent.
                input_acc.clear();
                continue;
            };

            // Apply automatable parameters before converting this chunk.
            pipeline.set_pitch_shift(self.params.pitch_shift.value());
            pipeline.set_speaker_id(self.params.speaker_id.value() as i64);
            pipeline.set_input_gain(util::db_to_gain(self.params.input_gain_db.value()));
            pipeline.set_output_gain(util::db_to_gain(self.params.output_gain_db.value()));

            let out = match pipeline.process(chunk, self.sample_rate) {
                Ok(out) => out,
                Err(err) => {
                    nih_plug::nih_error!("vc-vst3: model processing failed: {err:#}");
                    self.running.store(false, Ordering::SeqCst);
                    break;
                }
            };
            input_acc.clear();

            // (Re)build the chunk smoother when the model output rate changes.
            let model_sample_rate = out.sample_rate;
            if smoother.as_ref().map(|(rate, _)| *rate) != Some(model_sample_rate) {
                let joiner = sola::model_domain_chunk_smoother(ChunkSmootherConfig {
                    kind: self.config.smoothing_kind(),
                    output_chunk_samples,
                    output_sample_rate,
                    model_sample_rate,
                    crossfade_ms: self.crossfade_ms,
                    sola_search_ms: self.sola_search_ms,
                    tail_discard_ms: self.tail_discard_ms,
                });
                smoother = Some((model_sample_rate, joiner));
            }
            let joiner = &mut smoother.as_mut().expect("smoother set above").1;

            prepared.clear();
            match sola::prepare_model_output(
                out,
                output_sample_rate,
                output_chunk_samples,
                joiner,
                None,
            ) {
                Ok(result) => prepared = result.audio,
                Err(err) => {
                    nih_plug::nih_error!("vc-vst3: output smoothing failed: {err:#}");
                    self.running.store(false, Ordering::SeqCst);
                    break;
                }
            }

            // Push to the output ring; drop the tail if the consumer is behind.
            let _ = self.output_producer.push_partial_slice(&prepared);
        }
    }

    /// Discard everything currently queued in the input ring.
    fn drain_input(&mut self) {
        let backlog = self.input_consumer.slots();
        if backlog > 0 {
            if let Ok(chunk) = self.input_consumer.read_chunk(backlog) {
                chunk.commit_all();
            }
        }
    }

    fn load_pipeline(&self) -> anyhow::Result<RvcPipeline> {
        RvcPipeline::load(RvcPipelineConfig {
            model: &self.config.model,
            embedder: &self.config.embedder,
            embedder_output: self.config.embedder_output.as_deref(),
            f0_model: &self.config.f0_model,
            rvc_engine: self.config.rvc_engine.as_deref(),
            provider: self.config.provider(),
            sample_rate: self.sample_rate,
            chunk_samples: self.chunk_samples,
            speaker_id: self.config.speaker_id,
            pitch_shift: self.config.pitch_shift,
            f0_threshold: self.config.f0_threshold,
            silence_threshold: self.config.silence_threshold,
            input_gain: self.config.input_gain,
            output_extra_ms: self
                .crossfade_ms
                .saturating_add(self.sola_search_ms)
                .saturating_add(self.tail_discard_ms),
            volume_excluded_ms: self.crossfade_ms,
            extra_convert_ms: self.config.extra_convert_ms,
            output_gain: self.config.output_gain,
            volume_envelope: self.config.volume_envelope,
            rms_mix_rate: self.config.rms_mix_rate,
            auto_output_gain: self.config.auto_output_gain,
            target_output_rms: self.config.target_output_rms,
            max_output_gain: self.config.max_output_gain,
        })
    }
}
