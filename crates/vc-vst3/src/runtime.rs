//! Realtime bridge between the host's `process()` callback and the RVC pipeline.
//!
//! Mirrors the CLI's `engine.rs` worker model: the audio thread only pushes
//! input and pops output through lock-free SPSC ring buffers, while a dedicated
//! worker thread owns the `RvcPipeline`, runs inference, and smooths/resamples
//! the result back to the host sample rate. Inference and allocation never run
//! on the audio thread.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle, Thread};
use std::time::Duration;

use nice_plug::prelude::util;
use rtrb::{Consumer, Producer, RingBuffer};
use vc_core::dsp::chunk_samples_for_rate;
use vc_core::model_rvc::{
    ChunkConverter, ChunkOutputConfig, F0Config, LiveParams, LoadProgress, NoiseGateShaping,
    OutputDynamicsConfig, RvcPipeline, RvcPipelineConfig,
};
use vc_core::sola::SmoothingKind;
use vc_core::validation::CONVERSION_TIMING_LIMITS;

use crate::config::PluginConfig;
use crate::params::VcRvcParams;

#[derive(Clone, Debug)]
pub(crate) struct PluginStatus {
    pub summary: String,
    pub detail: Option<String>,
}

impl PluginStatus {
    pub fn new(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            detail: None,
        }
    }
}

const INPUT_QUEUE_CHUNKS: usize = 4;
const OUTPUT_QUEUE_CHUNKS: usize = 4;

/// Allowed range for the user-tunable chunk size (ms). The ring buffers are
/// sized for `MAX_CHUNK_MS` up front so chunk changes apply live (on reload)
/// without reallocating them.
pub const MIN_CHUNK_MS: u32 = CONVERSION_TIMING_LIMITS.min_chunk_ms;
pub const MAX_CHUNK_MS: u32 = CONVERSION_TIMING_LIMITS.max_chunk_ms;

/// Lets the editor's Load / Reload submit (a non-realtime UI thread) wake the
/// current worker even while the host is idle and not calling `process()`, so a
/// reload starts immediately instead of waiting for the worker's park timeout.
///
/// The worker handle is republished on every `PluginRuntime::start`, so this
/// stays correct across `initialize()`-driven worker restarts (sample-rate or
/// block-size changes). The realtime `process()` path deliberately does NOT use
/// this — it unparks through the [`Thread`] stored directly in [`PluginRuntime`]
/// so the audio callback never takes a lock. The editor and worker registration
/// are both off the audio thread, so the `Mutex` here is fine.
#[derive(Default)]
pub(crate) struct ReloadWaker {
    thread: Mutex<Option<Thread>>,
}

impl ReloadWaker {
    fn register(&self, thread: Thread) {
        if let Ok(mut slot) = self.thread.lock() {
            *slot = Some(thread);
        }
    }

    /// Unpark the current worker, if any. Called after the editor sets `reload`.
    pub(crate) fn wake(&self) {
        if let Ok(slot) = self.thread.lock() {
            if let Some(thread) = slot.as_ref() {
                thread.unpark();
            }
        }
    }
}

/// Owns the worker thread and the audio-thread ends of the ring buffers.
pub struct PluginRuntime {
    /// Worker `Thread` handle for the realtime wake path (input queued in
    /// `process_block`, and Drop). Stored directly so the audio callback unparks
    /// without a lock; refreshed whenever the worker is (re)spawned.
    worker_thread: Thread,
    input_producer: Producer<f32>,
    output_consumer: Consumer<f32>,
    running: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    mono_in: Vec<f32>,
    mono_out: Vec<f32>,
    /// Initial plugin latency in host samples, reported at `initialize`.
    pub latency_samples: u32,
    /// Current latency, updated after model/output initialization or reload. The audio
    /// thread re-reports it to the host (see [`PluginRuntime::poll_latency_update`]).
    latency: Arc<AtomicU32>,
    last_reported_latency: u32,
    loading: Arc<AtomicBool>,
}

impl PluginRuntime {
    /// Start the worker and allocate the ring buffers for the given host rate.
    /// `max_block` is the host's maximum block size used to pre-size scratch.
    ///
    /// `crossfade`/`sola`/`tail` (and the ring capacity) are fixed here from the
    /// settings present at init. `chunk_ms` can change live: the rings are sized
    /// for `MAX_CHUNK_MS` so the worker can adopt a new chunk size on reload, and
    /// the reported latency is updated from the audio thread afterwards.
    // The worker is wired up from a handful of independent shared handles
    // (editor/worker handshake flags, status, reload waker) plus the audio
    // format; bundling them into a struct would only move the same fields behind
    // an extra type without making the lifecycle clearer.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        params: Arc<VcRvcParams>,
        reload: Arc<AtomicBool>,
        loading: Arc<AtomicBool>,
        dirty: Arc<AtomicBool>,
        status: Arc<Mutex<PluginStatus>>,
        reload_waker: &Arc<ReloadWaker>,
        sample_rate: u32,
        max_block: usize,
    ) -> Self {
        // Reload requests belong to one runtime lifecycle. Dropping a stale
        // request here also guarantees the replacement starts with an enabled
        // button rather than accepting a duplicate while it handles old work.
        reload.store(false, Ordering::SeqCst);
        loading.store(false, Ordering::SeqCst);
        let settings0 = params.settings.read().unwrap().clone();
        let initial_timing_settings =
            if let Err(err) = settings0.validated_chunk_samples(sample_rate) {
                nice_plug::nice_error!("vc-vst3: invalid persisted settings: {err}");
                if let Ok(mut current) = status.lock() {
                    *current = PluginStatus {
                        summary: format!("invalid settings: {err}"),
                        detail: Some(format!("{err:#}")),
                    };
                }
                PluginConfig::default()
            } else {
                settings0.clone()
            };
        let crossfade_ms = initial_timing_settings.crossfade_ms;
        let sola_search_ms = initial_timing_settings.sola_search_ms;
        let tail_discard_ms = initial_timing_settings.rvc_output_tail_discard_ms;
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

        // Before explicit model loading, only host chunk/context timing is
        // known. The worker replaces this estimate with the shared converter's
        // actual input/join/output buffering delay after its first model call.
        let chunk_ms = initial_timing_settings.chunk_ms;
        let chunk_samples = chunk_samples_for_rate(sample_rate, chunk_ms);
        let extra_samples = chunk_samples_for_rate(sample_rate, output_extra_ms);
        let latency_samples = (chunk_samples + extra_samples) as u32;
        let latency = Arc::new(AtomicU32::new(latency_samples));

        let worker = WorkerCtx {
            params,
            reload,
            loading: Arc::clone(&loading),
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

        // Publish the worker handle for both wake paths: the realtime path keeps
        // its own clone, while the editor's reload submit goes through the shared
        // ReloadWaker (re-registered here so it tracks worker restarts).
        let worker_thread = worker.thread().clone();
        reload_waker.register(worker_thread.clone());

        Self {
            worker_thread,
            input_producer,
            output_consumer,
            running,
            worker: Some(worker),
            mono_in: Vec::with_capacity(max_block),
            mono_out: vec![0.0; max_block],
            latency_samples,
            latency,
            last_reported_latency: latency_samples,
            loading,
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
        let (pushed, _) = self.input_producer.push_partial_slice(&self.mono_in);
        if !pushed.is_empty() {
            // Wake the worker now that input is queued instead of letting it find
            // the data on a fixed poll. unpark is wait-free (token store, or one
            // OS wakeup when actually parked), so it is safe on the audio thread.
            self.worker_thread.unpark();
        }

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
        // Wake a parked worker so it observes the cleared running flag and exits.
        // The host may unload us while idle (no process() wakes arriving), so the
        // join below would otherwise block until the worker's park timeout.
        self.worker_thread.unpark();
        let Some(handle) = self.worker.take() else {
            self.loading.store(false, Ordering::SeqCst);
            return;
        };
        // The host may call deactivate/drop from a thread that blocks while a
        // slow model load finishes. We still join here: detaching would allow a
        // worker to continue executing plugin code after the DAW unloads this
        // DLL, which is a harder crash mode than a bounded unload wait.
        let _ = handle.join();
        self.loading.store(false, Ordering::SeqCst);
    }
}

/// Everything the worker thread needs, moved into it on spawn.
struct WorkerCtx {
    params: Arc<VcRvcParams>,
    reload: Arc<AtomicBool>,
    loading: Arc<AtomicBool>,
    dirty: Arc<AtomicBool>,
    status: Arc<Mutex<PluginStatus>>,
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
        let initial_settings = self.params.settings.read().unwrap().clone();
        let initial_chunk = initial_settings.validated_chunk_samples(self.sample_rate);
        // Invalid persisted settings never load a model. The fallback only
        // sizes the idle/silent worker until the user submits valid settings.
        let mut chunk_samples = initial_chunk.as_ref().copied().unwrap_or_else(|_| {
            chunk_samples_for_rate(self.sample_rate, PluginConfig::default().chunk_ms).max(1)
        });
        let mut input_acc = Vec::<f32>::with_capacity(chunk_samples * 2);
        // Reused output buffer for the converted chunk, filled by `process_chunk`.
        let mut chunk_out = Vec::<f32>::with_capacity(chunk_samples * 2);

        // Do not load models during host startup, plugin scan, or project
        // restore. Some DAWs instantiate and tear down plugins on UI/control
        // threads, and CUDA/ORT loading can crash or stall the entire host if it
        // happens implicitly. The editor's Load / Reload button is the explicit
        // boundary for model (re)initialization.
        let mut converter = None;
        if let Err(err) = initial_chunk {
            self.set_error(format!("invalid settings: {err}"), format!("{err:#}"));
        } else {
            self.set_idle_status();
        }

        while self.running.load(Ordering::SeqCst) {
            // The editor prevents another request while this one is loading, so
            // the worker can act immediately without a debounce timer.
            if self.reload.swap(false, Ordering::SeqCst) {
                // Clear before taking the snapshot. An edit after this point
                // re-sets dirty and remains visibly staged for the next load.
                self.dirty.store(false, Ordering::SeqCst);
                let settings = self.params.settings.read().unwrap().clone();
                let new_chunk_samples = match settings.validated_chunk_samples(self.sample_rate) {
                    Ok(samples) => samples,
                    Err(err) => {
                        nice_plug::nice_error!("vc-vst3: invalid settings: {err}");
                        self.set_error(format!("invalid settings: {err}"), format!("{err:#}"));
                        self.dirty.store(true, Ordering::SeqCst);
                        self.loading.store(false, Ordering::SeqCst);
                        continue;
                    }
                };
                // All join geometry belongs to the validated reload snapshot.
                // Keeping startup values here would disagree with settings
                // recovered from an invalid persisted configuration.
                self.crossfade_ms = settings.crossfade_ms;
                self.sola_search_ms = settings.sola_search_ms;
                self.tail_discard_ms = settings.rvc_output_tail_discard_ms;
                // chunk_ms may have changed; recompute and re-report latency.
                chunk_samples = new_chunk_samples;
                self.latency
                    .store(self.latency_samples(chunk_samples), Ordering::Relaxed);
                // Drop the old pipeline (releasing its CUDA context) before
                // building the new one, so the two never coexist.
                drop(converter.take());
                converter = match self.load_current(&settings, chunk_samples) {
                    Ok((new_pipeline, new_kind)) => new_pipeline
                        .map(|pipeline| self.chunk_converter(pipeline, new_kind, chunk_samples)),
                    Err(()) => {
                        self.dirty.store(true, Ordering::SeqCst);
                        None
                    }
                };
                input_acc.clear();
                self.drain_input();
                self.loading.store(false, Ordering::SeqCst);
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
                // Re-check the stop flag before parking so a stop requested
                // between the loop head and here exits without waiting.
                if !self.running.load(Ordering::SeqCst) {
                    break;
                }
                // Wait for an unpark — input queued in process_block, a reload
                // submitted from the editor, or Drop — or the safety timeout.
                // Replaces the fixed 2 ms poll: a reload submitted while the host
                // is idle starts immediately, and there is no idle spin when the
                // host stops calling process().
                thread::park_timeout(Duration::from_millis(100));
                continue;
            }

            let chunk = &input_acc[..chunk_samples];
            let Some(converter) = converter.as_mut() else {
                // No pipeline: discard input and stay silent.
                input_acc.clear();
                continue;
            };

            // Apply automatable parameters before converting this chunk. Builds
            // the same `LiveParams` the standalone worker does, so both drive the
            // single `apply_live` entry point rather than diverging set_* calls.
            converter.model_mut().apply_live(&LiveParams {
                pitch_shift: self.params.pitch_shift.value(),
                speaker_id: self.params.speaker_id.value() as i64,
                input_gain: util::db_to_gain(self.params.input_gain_db.value()),
                output_gain: util::db_to_gain(self.params.output_gain_db.value()),
                noise_gate_enabled: self.params.noise_gate.value(),
                noise_gate_threshold: util::db_to_gain(self.params.noise_gate_threshold_db.value()),
            });

            if let Err(err) = converter.process_chunk(chunk, self.sample_rate, &mut chunk_out) {
                nice_plug::nice_error!("vc-vst3: chunk conversion failed: {err:#}");
                self.running.store(false, Ordering::SeqCst);
                break;
            }
            // The shared converter now knows the model's native rate, actual
            // capped overlap, and both rate-adapter delays. Report these once
            // available; the callback only relays this precomputed atomic value.
            // SOLA's bounded search can advance audio within that nominal hold.
            let content_delay = converter.output_content_delay_samples();
            self.latency.store(
                u32::try_from(chunk_samples.saturating_add(content_delay)).unwrap_or(u32::MAX),
                Ordering::Relaxed,
            );
            input_acc.clear();

            // Push to the output ring; drop the tail if the consumer is behind.
            let _ = self.output_producer.push_partial_slice(&chunk_out);
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
            *status = PluginStatus::new(text);
        }
    }

    fn set_error(&self, summary: impl Into<String>, detail: impl Into<String>) {
        if let Ok(mut status) = self.status.lock() {
            *status = PluginStatus {
                summary: summary.into(),
                detail: Some(detail.into()),
            };
        }
    }

    fn report_load_progress(&self, progress: LoadProgress) {
        self.set_status(match progress {
            LoadProgress::Idle => "idle".to_string(),
            LoadProgress::ValidatingConfig => "validating configuration".to_string(),
            LoadProgress::PreparingProvider => "preparing execution provider".to_string(),
            LoadProgress::DownloadingProvider => "downloading execution provider".to_string(),
            LoadProgress::BuildingEngine { role } => {
                format!("building {} TensorRT engine", role.label())
            }
            LoadProgress::LoadingModel { role } => format!("loading {} model", role.label()),
            LoadProgress::OpeningAudioDevices => "opening audio devices".to_string(),
            LoadProgress::Running => "running".to_string(),
            LoadProgress::Failed => "failed".to_string(),
        });
    }

    fn set_idle_status(&self) {
        let settings = self.params.settings.read().unwrap();
        if settings.has_models() {
            self.set_status("models configured; click Load / Reload");
        } else {
            self.set_status("no models configured");
        }
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

    fn chunk_converter(
        &self,
        pipeline: RvcPipeline,
        kind: SmoothingKind,
        chunk_samples: usize,
    ) -> ChunkConverter<RvcPipeline> {
        ChunkConverter::new(
            pipeline,
            ChunkOutputConfig {
                kind,
                output_sample_rate: self.sample_rate,
                output_chunk_samples: chunk_samples,
                crossfade_ms: self.crossfade_ms,
                sola_search_ms: self.sola_search_ms,
                tail_discard_ms: self.tail_discard_ms,
            },
        )
    }

    /// Build a pipeline from one settings snapshot, reporting status. Missing
    /// models are a valid silent configuration; load failures return `Err`.
    fn load_current(
        &self,
        settings: &PluginConfig,
        chunk_samples: usize,
    ) -> Result<(Option<RvcPipeline>, SmoothingKind), ()> {
        let kind = settings.smoothing_kind();
        if !settings.has_models() {
            nice_plug::nice_warn!("vc-vst3: no models configured; running silent");
            self.set_status("no models configured");
            return Ok((None, kind));
        }
        self.set_status("loading…");
        let provider = settings.provider();
        match self.load_pipeline(settings, provider, chunk_samples) {
            Ok(pipeline) => {
                self.set_status(format!("running ({})", provider.label()));
                Ok((Some(pipeline), kind))
            }
            Err(err) => {
                nice_plug::nice_error!("vc-vst3: failed to load RVC pipeline: {err:#}");
                self.set_error(format!("load failed: {err}"), format!("{err:#}"));
                Err(())
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
        let report_progress = |progress| self.report_load_progress(progress);
        RvcPipeline::load(RvcPipelineConfig {
            model: &settings.model,
            embedder: &settings.embedder,
            embedder_output: settings.embedder_output.as_deref(),
            f0_model: &settings.f0_model,
            provider,
            gpu_priority: settings.gpu_priority(),
            gpu_device_id: settings.gpu_device_id,
            sample_rate: self.sample_rate,
            chunk_samples,
            // pitch / speaker / gains are DAW parameters; the worker applies the
            // current parameter values before every chunk, so these load-time
            // values are placeholders that get overwritten on the first chunk.
            speaker_id: 0,
            pitch_shift: 0.0,
            f0: F0Config {
                f0_threshold: settings.f0_threshold,
                silence_threshold: settings.silence_threshold,
                ..F0Config::default()
            },
            input_gain: 1.0,
            // Gate on/off + threshold are DAW parameters applied per chunk
            // (overwriting these load-time placeholders); attack/release/floor
            // are static and shape the gate built here.
            noise_gate_enabled: false,
            noise_gate_threshold: 0.01,
            noise_gate_shaping: NoiseGateShaping {
                attack_ms: settings.noise_gate_attack_ms,
                release_ms: settings.noise_gate_release_ms,
                floor: settings.noise_gate_floor,
            },
            output_extra_ms: self.output_extra_ms(),
            volume_excluded_ms: self.crossfade_ms,
            extra_convert_ms: settings.extra_convert_ms,
            output_gain: 1.0,
            output_dynamics: OutputDynamicsConfig {
                volume_envelope: settings.volume_envelope,
                rms_mix_rate: settings.rms_mix_rate,
                auto_output_gain: settings.auto_output_gain,
                target_output_rms: settings.target_output_rms,
                max_output_gain: settings.max_output_gain,
            },
            progress: Some(&report_progress),
        })
    }
}
