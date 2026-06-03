//! Realtime bridge between the host's `process()` callback and the RVC pipeline.
//!
//! Mirrors the CLI's `engine.rs` worker model: the audio thread only pushes
//! input and pops output through lock-free SPSC ring buffers, while a dedicated
//! worker thread owns the `RvcPipeline`, runs inference, and smooths/resamples
//! the result back to the host sample rate. Inference and allocation never run
//! on the audio thread.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use nice_plug::prelude::util;
use rtrb::{Consumer, Producer, RingBuffer};
use vc_core::model_rvc::{RvcPipeline, RvcPipelineConfig, VoiceModel};
use vc_core::sola::{self, ChunkSmoother, ChunkSmootherConfig, SmoothingKind};

use crate::config::PluginConfig;
use crate::params::VcRvcParams;

const INPUT_QUEUE_CHUNKS: usize = 4;
const OUTPUT_QUEUE_CHUNKS: usize = 4;

/// Allowed range for the user-tunable chunk size (ms). The ring buffers are
/// sized for `MAX_CHUNK_MS` up front so chunk changes apply live (on reload)
/// without reallocating them.
pub const MIN_CHUNK_MS: u32 = 50;
pub const MAX_CHUNK_MS: u32 = 1000;

fn chunk_samples_for_rate(sample_rate: u32, chunk_ms: u32) -> usize {
    ((sample_rate as u64 * chunk_ms as u64) / 1000).max(128) as usize
}

/// Settle time before acting on a reload request. A burst of changes (e.g.
/// toggling CPU/CUDA quickly) collapses into a single reload, because rebuilding
/// the ONNX Runtime CUDA provider in rapid succession is fragile.
const RELOAD_DEBOUNCE: Duration = Duration::from_millis(300);

/// Owns the worker thread and the audio-thread ends of the ring buffers.
pub struct PluginRuntime {
    input_producer: Producer<f32>,
    output_consumer: Consumer<f32>,
    running: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    mono_in: Vec<f32>,
    mono_out: Vec<f32>,
    /// Initial plugin latency in host samples, reported at `initialize`.
    pub latency_samples: u32,
    /// Current latency, updated by the worker when `chunk_ms` changes. The audio
    /// thread re-reports it to the host (see [`PluginRuntime::poll_latency_update`]).
    latency: Arc<AtomicU32>,
    last_reported_latency: u32,
}

impl PluginRuntime {
    /// Start the worker and allocate the ring buffers for the given host rate.
    /// `max_block` is the host's maximum block size used to pre-size scratch.
    ///
    /// `crossfade`/`sola`/`tail` (and the ring capacity) are fixed here from the
    /// settings present at init. `chunk_ms` can change live: the rings are sized
    /// for `MAX_CHUNK_MS` so the worker can adopt a new chunk size on reload, and
    /// the reported latency is updated from the audio thread afterwards.
    pub fn start(
        params: Arc<VcRvcParams>,
        reload: Arc<AtomicBool>,
        dirty: Arc<AtomicBool>,
        status: Arc<Mutex<String>>,
        sample_rate: u32,
        max_block: usize,
    ) -> Self {
        let settings0 = params.settings.read().unwrap().clone();
        let crossfade_ms = settings0.crossfade_ms;
        let sola_search_ms = settings0.sola_search_ms;
        let tail_discard_ms = settings0.rvc_output_tail_discard_ms;
        let output_extra_ms = crossfade_ms
            .saturating_add(sola_search_ms)
            .saturating_add(tail_discard_ms);

        // Size the rings for the largest allowed chunk so `chunk_ms` can change
        // without reallocating. Extra capacity does not add latency (the worker
        // pops as soon as a chunk is available).
        let max_chunk_samples = chunk_samples_for_rate(sample_rate, MAX_CHUNK_MS);
        let (input_producer, input_consumer) =
            RingBuffer::<f32>::new(max_chunk_samples * INPUT_QUEUE_CHUNKS);
        let (output_producer, output_consumer) =
            RingBuffer::<f32>::new(max_chunk_samples * OUTPUT_QUEUE_CHUNKS);

        let running = Arc::new(AtomicBool::new(true));

        // Initial latency: one (clamped) chunk of input buffering plus the
        // smoothing/tail context, in host samples. Updated live when chunk_ms
        // changes; RVC has additional inherent latency this estimate omits.
        let chunk_ms = settings0.chunk_ms.clamp(MIN_CHUNK_MS, MAX_CHUNK_MS);
        let chunk_samples = chunk_samples_for_rate(sample_rate, chunk_ms);
        let extra_samples = chunk_samples_for_rate(sample_rate, output_extra_ms);
        let latency_samples = (chunk_samples + extra_samples) as u32;
        let latency = Arc::new(AtomicU32::new(latency_samples));

        let worker = WorkerCtx {
            params,
            reload,
            dirty,
            status,
            sample_rate,
            crossfade_ms,
            sola_search_ms,
            tail_discard_ms,
            latency: Arc::clone(&latency),
            running: Arc::clone(&running),
            input_consumer,
            output_producer,
        }
        .spawn();

        Self {
            input_producer,
            output_consumer,
            running,
            worker: Some(worker),
            mono_in: Vec::with_capacity(max_block),
            mono_out: vec![0.0; max_block],
            latency_samples,
            latency,
            last_reported_latency: latency_samples,
        }
    }

    /// Returns a new latency value if the worker changed it (chunk_ms edit) since
    /// the last call, so the audio thread can re-report it to the host.
    pub fn poll_latency_update(&mut self) -> Option<u32> {
        let current = self.latency.load(Ordering::Relaxed);
        if current != self.last_reported_latency {
            self.last_reported_latency = current;
            Some(current)
        } else {
            None
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
        // The host may call deactivate/drop from a thread that blocks while a
        // slow model load finishes. We still join here: detaching would allow a
        // worker to continue executing plugin code after the DAW unloads this
        // DLL, which is a harder crash mode than a bounded unload wait.
        let _ = handle.join();
    }
}

/// Everything the worker thread needs, moved into it on spawn.
struct WorkerCtx {
    params: Arc<VcRvcParams>,
    reload: Arc<AtomicBool>,
    dirty: Arc<AtomicBool>,
    status: Arc<Mutex<String>>,
    sample_rate: u32,
    crossfade_ms: u32,
    sola_search_ms: u32,
    tail_discard_ms: u32,
    latency: Arc<AtomicU32>,
    running: Arc<AtomicBool>,
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
        let output_sample_rate = self.sample_rate;
        let mut chunk_samples = self.current_chunk_samples();
        let mut output_chunk_samples = chunk_samples;
        let mut smoother: Option<(u32, ChunkSmoother)> = None;
        let mut input_acc = Vec::<f32>::with_capacity(chunk_samples * 2);
        let mut prepared = Vec::<f32>::with_capacity(output_chunk_samples * 2);

        // Do not load models during host startup, plugin scan, or project
        // restore. Some DAWs instantiate and tear down plugins on UI/control
        // threads, and CUDA/ORT loading can crash or stall the entire host if it
        // happens implicitly. The editor's Load / Reload button is the explicit
        // boundary for model (re)initialization.
        let mut pipeline = None;
        let mut smoothing_kind = self.current_smoothing_kind();
        self.set_idle_status();
        let mut reload_at: Option<Instant> = None;

        while self.running.load(Ordering::SeqCst) {
            // Coalesce reload requests (model/provider/chunk changes) and only act
            // once they settle, then rebuild — dropping the old pipeline first so
            // we never hold two CUDA contexts at once.
            if self.reload.swap(false, Ordering::SeqCst) {
                reload_at = Some(Instant::now());
            }
            if reload_at.is_some_and(|t| t.elapsed() >= RELOAD_DEBOUNCE) {
                reload_at = None;
                // We're applying the current settings now; clear the
                // "unapplied" indicator (a later edit re-sets it).
                self.dirty.store(false, Ordering::SeqCst);
                // chunk_ms may have changed; recompute and re-report latency.
                chunk_samples = self.current_chunk_samples();
                output_chunk_samples = chunk_samples;
                self.latency
                    .store(self.latency_samples(chunk_samples), Ordering::Relaxed);
                // Drop the old pipeline (releasing its CUDA context) before
                // building the new one, so the two never coexist.
                drop(pipeline.take());
                smoother = None;
                let (new_pipeline, new_kind) = self.load_current(chunk_samples);
                pipeline = new_pipeline;
                smoothing_kind = new_kind;
                input_acc.clear();
                self.drain_input();
            }

            // Accumulate one input chunk.
            while input_acc.len() < chunk_samples {
                let needed = chunk_samples - input_acc.len();
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
            if input_acc.len() < chunk_samples {
                thread::sleep(Duration::from_millis(2));
                continue;
            }

            let chunk = &input_acc[..chunk_samples];
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
                    nice_plug::nice_error!("vc-vst3: model processing failed: {err:#}");
                    self.running.store(false, Ordering::SeqCst);
                    break;
                }
            };
            input_acc.clear();

            // (Re)build the chunk smoother when the model output rate changes.
            let model_sample_rate = out.sample_rate;
            if smoother.as_ref().map(|(rate, _)| *rate) != Some(model_sample_rate) {
                let joiner = sola::model_domain_chunk_smoother(ChunkSmootherConfig {
                    kind: smoothing_kind,
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
                    nice_plug::nice_error!("vc-vst3: output smoothing failed: {err:#}");
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

    fn set_status(&self, text: impl Into<String>) {
        if let Ok(mut status) = self.status.lock() {
            *status = text.into();
        }
    }

    fn set_idle_status(&self) {
        let settings = self.params.settings.read().unwrap();
        if settings.has_models() {
            self.set_status("models configured; click Load / Reload");
        } else {
            self.set_status("no models configured");
        }
    }

    fn current_smoothing_kind(&self) -> SmoothingKind {
        self.params.settings.read().unwrap().smoothing_kind()
    }

    /// Current chunk size in samples, from the persisted (clamped) `chunk_ms`.
    fn current_chunk_samples(&self) -> usize {
        let chunk_ms = self
            .params
            .settings
            .read()
            .unwrap()
            .chunk_ms
            .clamp(MIN_CHUNK_MS, MAX_CHUNK_MS);
        chunk_samples_for_rate(self.sample_rate, chunk_ms)
    }

    fn output_extra_ms(&self) -> u32 {
        self.crossfade_ms
            .saturating_add(self.sola_search_ms)
            .saturating_add(self.tail_discard_ms)
    }

    fn latency_samples(&self, chunk_samples: usize) -> u32 {
        let extra = chunk_samples_for_rate(self.sample_rate, self.output_extra_ms());
        (chunk_samples + extra) as u32
    }

    /// Build a pipeline from the current persisted settings, reporting status.
    /// Returns the pipeline (None if no models / load failed) and the smoothing
    /// kind to use until the next reload.
    fn load_current(&self, chunk_samples: usize) -> (Option<RvcPipeline>, SmoothingKind) {
        let settings = self.params.settings.read().unwrap().clone();
        let kind = settings.smoothing_kind();
        if !settings.has_models() {
            nice_plug::nice_warn!("vc-vst3: no models configured; running silent");
            self.set_status("no models configured");
            return (None, kind);
        }
        self.set_status("loading…");
        let provider = settings.provider();
        match self.load_pipeline(&settings, provider, chunk_samples) {
            Ok(pipeline) => {
                self.set_status(format!("running ({})", provider.label()));
                (Some(pipeline), kind)
            }
            Err(err) => {
                nice_plug::nice_error!("vc-vst3: failed to load RVC pipeline: {err:#}");
                self.set_status(format!("load failed: {err}"));
                (None, kind)
            }
        }
    }

    fn load_pipeline(
        &self,
        settings: &PluginConfig,
        provider: vc_core::Provider,
        chunk_samples: usize,
    ) -> anyhow::Result<RvcPipeline> {
        if provider.is_cuda() {
            // This is deliberately on the worker's explicit Load / Reload path,
            // not plugin initialization or the realtime callback. It prevents a
            // DAW's PATH or already-installed CUDA stack from silently winning
            // DLL resolution before ONNX Runtime creates the CUDA EP session.
            crate::dll_path::preload_bundled_cuda_dlls()?;
            return crate::dll_path::with_bundled_dll_directory(|| {
                self.load_pipeline_inner(settings, provider, chunk_samples)
            });
        }
        if provider.is_windows_ml() {
            // Windows ML's small bootstrapper DLL is bundled beside the plugin,
            // while ONNX Runtime/DirectML come from Windows App SDK Runtime.
            // Keep this on the worker load path so the realtime callback never
            // performs package bootstrap or DLL resolution work.
            return crate::dll_path::with_bundled_dll_directory(|| {
                self.load_pipeline_inner(settings, provider, chunk_samples)
            });
        }
        self.load_pipeline_inner(settings, provider, chunk_samples)
    }

    fn load_pipeline_inner(
        &self,
        settings: &PluginConfig,
        provider: vc_core::Provider,
        chunk_samples: usize,
    ) -> anyhow::Result<RvcPipeline> {
        RvcPipeline::load(RvcPipelineConfig {
            model: &settings.model,
            embedder: &settings.embedder,
            embedder_output: settings.embedder_output.as_deref(),
            f0_model: &settings.f0_model,
            provider,
            sample_rate: self.sample_rate,
            chunk_samples,
            // pitch / speaker / gains are DAW parameters; the worker applies the
            // current parameter values before every chunk, so these load-time
            // values are placeholders that get overwritten on the first chunk.
            speaker_id: 0,
            pitch_shift: 0.0,
            f0_threshold: settings.f0_threshold,
            silence_threshold: settings.silence_threshold,
            input_gain: 1.0,
            output_extra_ms: self.output_extra_ms(),
            volume_excluded_ms: self.crossfade_ms,
            extra_convert_ms: settings.extra_convert_ms,
            output_gain: 1.0,
            volume_envelope: settings.volume_envelope,
            rms_mix_rate: settings.rms_mix_rate,
            auto_output_gain: settings.auto_output_gain,
            target_output_rms: settings.target_output_rms,
            max_output_gain: settings.max_output_gain,
        })
    }
}
