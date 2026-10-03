//! Realtime bridge between the host's `process()` callback and the RVC pipeline.
//!
//! Mirrors the CLI's `engine.rs` worker model: the audio thread only pushes
//! input and pops output through lock-free SPSC ring buffers, while a dedicated
//! worker thread owns the `RvcPipeline`, runs inference, and smooths/resamples
//! the result back to the host sample rate. Inference and allocation never run
//! on the audio thread.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle, Thread};
use std::time::Duration;

use nice_plug::prelude::{util, ProcessMode};
use rtrb::{Consumer, Producer, RingBuffer};
use vc_core::dsp::chunk_samples_for_rate;
use vc_core::model_rvc::{
    ChunkConverter, ChunkOutputConfig, ChunkStats, F0Config, LiveParams, LoadProgress,
    NoiseGateShaping, OutputDynamicsConfig, RvcPipeline, RvcPipelineConfig,
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

#[derive(Clone, Copy, Default)]
struct QueuedSample {
    value: f32,
    generation: u32,
}

struct RuntimeState {
    generation: AtomicU32,
    ready_generation: AtomicU32,
    input_hop: AtomicUsize,
    offline: AtomicBool,
    has_converter: AtomicBool,
    applied_settings: Mutex<Option<PluginConfig>>,
    // Only offline process() waits/takes this lock. Realtime callbacks never do.
    progress: Mutex<()>,
    progress_changed: Condvar,
}

impl RuntimeState {
    fn notify_offline(&self) {
        if self.offline.load(Ordering::Acquire) {
            let _guard = self.progress.lock().unwrap_or_else(|e| e.into_inner());
            self.progress_changed.notify_all();
        }
    }
}

// Also runs when inference panics: an offline host must not wait forever for a
// worker which has exited without publishing another output chunk.
struct WorkerExit {
    running: Arc<AtomicBool>,
    state: Arc<RuntimeState>,
}

impl Drop for WorkerExit {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
        self.state.has_converter.store(false, Ordering::Release);
        self.state.notify_offline();
    }
}

trait WorkerConversion: Send {
    fn apply_live(&mut self, live: &LiveParams);
    fn reset(&mut self) -> anyhow::Result<()>;
    fn prime(&mut self, input: &[f32], rate: u32) -> anyhow::Result<()>;
    fn process(
        &mut self,
        input: &[f32],
        rate: u32,
        output: &mut Vec<f32>,
    ) -> anyhow::Result<ChunkStats>;
    fn content_delay(&self) -> usize;
}

impl WorkerConversion for ChunkConverter<RvcPipeline> {
    fn apply_live(&mut self, live: &LiveParams) {
        self.model_mut().apply_live(live);
    }

    fn reset(&mut self) -> anyhow::Result<()> {
        self.model_mut().reset_streaming_state()?;
        self.reset_streaming_state();
        Ok(())
    }

    fn process(
        &mut self,
        input: &[f32],
        rate: u32,
        output: &mut Vec<f32>,
    ) -> anyhow::Result<ChunkStats> {
        self.process_chunk(input, rate, output)
    }

    fn prime(&mut self, input: &[f32], rate: u32) -> anyhow::Result<()> {
        ChunkConverter::prime(self, input, rate).map(|_| ())
    }

    fn content_delay(&self) -> usize {
        self.output_content_delay_samples()
    }
}

/// Discard obsolete generations without scanning a potentially multi-second
/// backlog in the callback. A single producer publishes generations in order;
/// the current generation, if present, is the suffix of each ring slice.
fn read_generation(
    consumer: &mut Consumer<QueuedSample>,
    generation: u32,
    output: &mut [f32],
) -> usize {
    let available = consumer.slots();
    if available == 0 {
        return 0;
    }
    let Ok(chunk) = consumer.read_chunk(available) else {
        return 0;
    };
    let (a, b) = chunk.as_slices();
    let mut consumed = 0;
    let mut filled = 0;
    for slice in [a, b] {
        let stale =
            slice.partition_point(|sample| (generation.wrapping_sub(sample.generation) as i32) > 0);
        consumed += stale;
        // A reset can race the worker's input read. Preserve newer input until
        // its corresponding model reset, rather than discarding or converting it.
        let current = slice[stale..].partition_point(|sample| sample.generation == generation);
        let take = current.min(output.len() - filled);
        for (sample, target) in slice[stale..stale + take]
            .iter()
            .zip(&mut output[filled..filled + take])
        {
            *target = sample.value;
        }
        filled += take;
        consumed += take;
        if take < current
            || current < slice.len() - stale
            || (filled == output.len() && current > 0)
        {
            break;
        }
    }
    chunk.commit(consumed);
    filled
}

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
    input_producer: Producer<QueuedSample>,
    output_consumer: Consumer<QueuedSample>,
    running: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    mono_in: Vec<QueuedSample>,
    mono_out: Vec<f32>,
    /// Initial plugin latency in host samples, reported at `initialize`.
    pub latency_samples: u32,
    /// Current latency, updated after model/output initialization or reload. The audio
    /// thread re-reports it to the host (see [`PluginRuntime::poll_latency_update`]).
    latency: Arc<AtomicU32>,
    last_reported_latency: u32,
    loading: Arc<AtomicBool>,
    state: Arc<RuntimeState>,
    sample_rate: u32,
    max_block_capacity: usize,
    offline_generation: u32,
    offline_startup_remaining: usize,
}

impl PluginRuntime {
    /// Start the worker and allocate the ring buffers for the given host rate.
    /// `max_block` is the host's maximum block size used to pre-size scratch.
    ///
    /// Ring capacity is fixed here for `MAX_CHUNK_MS`. Chunk/join geometry comes
    /// from the applied Load / Reload snapshot; the callback relays the worker's
    /// updated latency. A previously loaded snapshot may rebuild fixed profiles
    /// after a host rate change, without applying staged editor changes.
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
        mode: ProcessMode,
        applied_settings: Option<PluginConfig>,
    ) -> Self {
        Self::start_inner(
            params,
            reload,
            loading,
            dirty,
            status,
            reload_waker,
            sample_rate,
            max_block,
            mode,
            applied_settings,
            #[cfg(test)]
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn start_inner(
        params: Arc<VcRvcParams>,
        reload: Arc<AtomicBool>,
        loading: Arc<AtomicBool>,
        dirty: Arc<AtomicBool>,
        status: Arc<Mutex<PluginStatus>>,
        reload_waker: &Arc<ReloadWaker>,
        sample_rate: u32,
        max_block: usize,
        mode: ProcessMode,
        applied_settings: Option<PluginConfig>,
        #[cfg(test)] initial_conversion: Option<Box<dyn WorkerConversion>>,
    ) -> Self {
        // Reload requests belong to one runtime lifecycle. Dropping a stale
        // request here also guarantees the replacement starts with an enabled
        // button rather than accepting a duplicate while it handles old work.
        reload.store(false, Ordering::SeqCst);
        let autoload = applied_settings.is_some();
        loading.store(autoload, Ordering::SeqCst);
        let settings0 = applied_settings.unwrap_or_else(|| params.settings.read().unwrap().clone());
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
        let input_capacity =
            (max_chunk_samples * INPUT_QUEUE_CHUNKS).max(max_block + max_chunk_samples);
        let output_capacity =
            (max_chunk_samples * OUTPUT_QUEUE_CHUNKS).max(max_block + max_chunk_samples);
        let (input_producer, input_consumer) = RingBuffer::<QueuedSample>::new(input_capacity);
        let (output_producer, output_consumer) = RingBuffer::<QueuedSample>::new(output_capacity);

        let running = Arc::new(AtomicBool::new(true));

        // Before explicit model loading, only host chunk/context timing is
        // known. The worker replaces this estimate with the shared converter's
        // actual input/join/output buffering delay after its first model call.
        let chunk_ms = initial_timing_settings.chunk_ms;
        let chunk_samples = chunk_samples_for_rate(sample_rate, chunk_ms);
        let extra_samples = chunk_samples_for_rate(sample_rate, output_extra_ms);
        let latency_samples = (chunk_samples + extra_samples) as u32;
        let latency = Arc::new(AtomicU32::new(latency_samples));
        let state = Arc::new(RuntimeState {
            generation: AtomicU32::new(1),
            ready_generation: AtomicU32::new(0),
            input_hop: AtomicUsize::new(chunk_samples),
            offline: AtomicBool::new(mode == ProcessMode::Offline),
            has_converter: AtomicBool::new(false),
            applied_settings: Mutex::new(autoload.then(|| settings0.clone())),
            progress: Mutex::new(()),
            progress_changed: Condvar::new(),
        });

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
            state: Arc::clone(&state),
            initial_settings: settings0,
            autoload,
            #[cfg(test)]
            initial_conversion,
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
            state,
            sample_rate,
            max_block_capacity: input_capacity.min(output_capacity) - max_chunk_samples,
            offline_generation: 0,
            offline_startup_remaining: 0,
        }
    }

    /// Mode/block changes retain successfully loaded models and staged settings.
    /// Only initialize() may grow scratch; process() accepts the advertised bound.
    pub fn reconfigure(&mut self, rate: u32, max_block: usize, mode: ProcessMode) -> bool {
        if !self.running.load(Ordering::Acquire)
            || rate != self.sample_rate
            || max_block > self.max_block_capacity
        {
            return false;
        }
        self.mono_in.resize(max_block, QueuedSample::default());
        self.mono_in.clear();
        self.mono_out.resize(max_block, 0.0);
        self.state
            .offline
            .store(mode == ProcessMode::Offline, Ordering::Release);
        self.reset();
        self.worker_thread.unpark();
        true
    }

    pub fn applied_settings(&self) -> Option<PluginConfig> {
        self.state.applied_settings.lock().unwrap().clone()
    }

    pub fn current_latency(&self) -> u32 {
        self.latency.load(Ordering::Relaxed)
    }

    pub fn tail_samples(&self) -> Option<u32> {
        self.state
            .has_converter
            .load(Ordering::Acquire)
            .then(|| self.current_latency())
    }

    /// Called on the audio thread. In-flight inference may finish later, so
    /// flushing queues alone is insufficient: tag every sample and wait for the
    /// worker's model + smoother reset acknowledgement before emitting new audio.
    pub fn reset(&mut self) {
        self.state.generation.fetch_add(1, Ordering::AcqRel);
        let queued = self.output_consumer.slots();
        if queued > 0 {
            if let Ok(chunk) = self.output_consumer.read_chunk(queued) {
                chunk.commit_all();
            }
        }
        self.offline_generation = 0;
        self.worker_thread.unpark();
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
    /// underrun). Realtime processing is allocation/lock/wait-free. Offline
    /// rendering alone waits for the same worker, paced by sample counts rather
    /// than wall time, so freewheeling cannot discard input or insert underruns.
    pub fn process_block(&mut self, channels: &mut [&mut [f32]]) {
        if channels.is_empty() {
            return;
        }
        let n = channels[0].len();
        if n == 0 {
            return;
        }
        if n > self.mono_in.capacity() || n > self.mono_out.len() {
            // A host violating max_buffer_size must not trigger callback allocation.
            for channel in channels.iter_mut() {
                channel.fill(0.0);
            }
            return;
        }
        let generation = self.state.generation.load(Ordering::Acquire);
        let offline = self.state.offline.load(Ordering::Acquire);
        if offline && !self.wait_until_ready(generation) {
            for channel in channels.iter_mut() {
                channel.fill(0.0);
            }
            return;
        }

        // Generation travels with audio, including samples queued while reset
        // is still pending; the worker rejects the old input timeline.
        self.mono_in.clear();
        self.mono_in.resize(n, QueuedSample::default());
        if channels.len() >= 2 {
            let (left, right) = (&channels[0], &channels[1]);
            for i in 0..n {
                self.mono_in[i] = QueuedSample {
                    value: 0.5 * (left[i] + right[i]),
                    generation,
                };
            }
        } else {
            for (target, value) in self.mono_in.iter_mut().zip(channels[0].iter().copied()) {
                *target = QueuedSample { value, generation };
            }
        }

        // Queue input; drop on overflow (worker is behind, audio keeps flowing).
        let (pushed, _) = self.input_producer.push_partial_slice(&self.mono_in);
        if !pushed.is_empty() {
            // Wake the worker now that input is queued instead of letting it find
            // the data on a fixed poll. unpark is wait-free (token store, or one
            // OS wakeup when actually parked), so it is safe on the audio thread.
            self.worker_thread.unpark();
        }

        self.mono_out[..n].fill(0.0);
        if offline {
            if self.offline_generation != generation {
                self.offline_generation = generation;
                self.offline_startup_remaining = self.state.input_hop.load(Ordering::Acquire);
            }
            // One input hop is the host-scheduling delay already included in
            // latency_samples. Shared model/join/resampler content delay stays
            // in the converter; do not add or remove it again here.
            let startup = n.min(self.offline_startup_remaining);
            self.offline_startup_remaining -= startup;
            let state = Arc::clone(&self.state);
            let mut guard = state.progress.lock().unwrap_or_else(|e| e.into_inner());
            let mut filled = startup;
            while filled < n {
                filled += read_generation(
                    &mut self.output_consumer,
                    generation,
                    &mut self.mono_out[filled..n],
                );
                self.worker_thread.unpark();
                if filled == n
                    || !self.running.load(Ordering::Acquire)
                    || !state.has_converter.load(Ordering::Acquire)
                    || state.generation.load(Ordering::Acquire) != generation
                {
                    break;
                }
                guard = state
                    .progress_changed
                    .wait_timeout(guard, Duration::from_millis(100))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        } else if self.state.ready_generation.load(Ordering::Acquire) == generation
            && !self.loading.load(Ordering::Acquire)
        {
            read_generation(
                &mut self.output_consumer,
                generation,
                &mut self.mono_out[..n],
            );
        }
        if self.state.generation.load(Ordering::Acquire) != generation {
            self.mono_out[..n].fill(0.0);
        }

        // Fan out mono to every output channel.
        for channel in channels.iter_mut() {
            channel[..n].copy_from_slice(&self.mono_out[..n]);
        }
    }

    fn wait_until_ready(&self, generation: u32) -> bool {
        let mut guard = self
            .state
            .progress
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        loop {
            if !self.running.load(Ordering::Acquire)
                || self.state.generation.load(Ordering::Acquire) != generation
            {
                return false;
            }
            if !self.loading.load(Ordering::Acquire) {
                if !self.state.has_converter.load(Ordering::Acquire) {
                    return false;
                }
                if self.state.ready_generation.load(Ordering::Acquire) == generation {
                    return true;
                }
            }
            guard = self
                .state
                .progress_changed
                .wait_timeout(guard, Duration::from_millis(100))
                .unwrap_or_else(|e| e.into_inner())
                .0;
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
        self.state.notify_offline();
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
    input_consumer: Consumer<QueuedSample>,
    output_producer: Producer<QueuedSample>,
    state: Arc<RuntimeState>,
    initial_settings: PluginConfig,
    autoload: bool,
    #[cfg(test)]
    initial_conversion: Option<Box<dyn WorkerConversion>>,
}

impl WorkerCtx {
    fn spawn(self) -> JoinHandle<()> {
        thread::Builder::new()
            .name("vc-vst3-rvc".to_string())
            .spawn(move || self.run())
            .expect("failed to spawn vc-vst3 worker thread")
    }

    fn run(mut self) {
        let _exit = WorkerExit {
            running: Arc::clone(&self.running),
            state: Arc::clone(&self.state),
        };
        let initial_settings = self.initial_settings.clone();
        let initial_chunk = initial_settings.validated_chunk_samples(self.sample_rate);
        // Invalid persisted settings never load a model. The fallback only
        // sizes the idle/silent worker until the user submits valid settings.
        let mut chunk_samples = initial_chunk.as_ref().copied().unwrap_or_else(|_| {
            chunk_samples_for_rate(self.sample_rate, PluginConfig::default().chunk_ms).max(1)
        });
        let mut input_acc = Vec::<f32>::with_capacity(chunk_samples * 2);
        // Reused output buffer for the converted chunk, filled by `process_chunk`.
        let mut chunk_out = Vec::<f32>::with_capacity(chunk_samples * 2);
        let mut queued_output = Vec::<QueuedSample>::with_capacity(chunk_samples * 2);
        let mut zero_input = vec![0.0; chunk_samples];

        // Do not load models during host startup, plugin scan, or project
        // restore. Some DAWs instantiate and tear down plugins on UI/control
        // threads, and CUDA/ORT loading can crash or stall the entire host if it
        // happens implicitly. The editor's Load / Reload button is the explicit
        // boundary for model (re)initialization.
        #[cfg(not(test))]
        let mut converter: Option<Box<dyn WorkerConversion>> = None;
        #[cfg(test)]
        let mut converter = self.initial_conversion.take();
        if let Err(err) = initial_chunk {
            self.set_error(format!("invalid settings: {err}"), format!("{err:#}"));
        } else if self.autoload && converter.is_none() {
            converter = self.load_conversion(&initial_settings, chunk_samples);
        } else {
            if converter.is_none() {
                self.set_idle_status();
            }
        }
        if let Some(current) = converter.as_mut() {
            if let Err(err) = self.prepare_offline(current.as_mut(), &zero_input) {
                self.set_error("offline preroll failed", format!("{err:#}"));
                converter = None;
            }
        }
        self.publish_conversion(&initial_settings, converter.is_some());
        let mut generation = self.state.generation.load(Ordering::Acquire);
        self.state
            .ready_generation
            .store(generation, Ordering::Release);
        self.loading.store(false, Ordering::Release);
        self.state.notify_offline();

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
                        self.state.notify_offline();
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
                zero_input.resize(chunk_samples, 0.0);
                self.state.input_hop.store(chunk_samples, Ordering::Release);
                self.latency
                    .store(self.latency_samples(chunk_samples), Ordering::Relaxed);
                // Drop the old pipeline (releasing its CUDA context) before
                // building the new one, so the two never coexist.
                drop(converter.take());
                self.state.generation.fetch_add(1, Ordering::AcqRel);
                self.publish_conversion(&settings, false);
                converter = self.load_conversion(&settings, chunk_samples);
                if let Some(current) = converter.as_mut() {
                    if let Err(err) = self.prepare_offline(current.as_mut(), &zero_input) {
                        self.set_error("offline preroll failed", format!("{err:#}"));
                        converter = None;
                    }
                }
                input_acc.clear();
                self.drain_input();
                // Loading made a fresh model timeline. A host reset issued
                // during the load is covered by that same fresh initialization.
                generation = self.state.generation.load(Ordering::Acquire);
                self.publish_conversion(&settings, converter.is_some());
                self.state
                    .ready_generation
                    .store(generation, Ordering::Release);
                self.loading.store(false, Ordering::SeqCst);
                self.state.notify_offline();
            }

            let requested_generation = self.state.generation.load(Ordering::Acquire);
            if requested_generation != generation {
                input_acc.clear();
                if let Some(converter) = converter.as_mut() {
                    if let Err(err) = converter.reset() {
                        self.set_error("stream reset failed", format!("{err:#}"));
                        break;
                    }
                    if let Err(err) = self.prepare_offline(converter.as_mut(), &zero_input) {
                        self.set_error("offline preroll failed", format!("{err:#}"));
                        break;
                    }
                }
                generation = requested_generation;
                read_generation(&mut self.input_consumer, generation, &mut []);
                self.state
                    .ready_generation
                    .store(generation, Ordering::Release);
                self.state.notify_offline();
            }

            // Accumulate one input chunk.
            while input_acc.len() < chunk_samples {
                let needed = chunk_samples - input_acc.len();
                if self.input_consumer.slots() == 0 {
                    break;
                }
                let old_len = input_acc.len();
                input_acc.resize(old_len + needed, 0.0);
                let read = read_generation(
                    &mut self.input_consumer,
                    generation,
                    &mut input_acc[old_len..],
                );
                input_acc.truncate(old_len + read);
                if read == 0 {
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
            converter.apply_live(&self.live_params());

            if let Err(err) = converter.process(chunk, self.sample_rate, &mut chunk_out) {
                nice_plug::nice_error!("vc-vst3: chunk conversion failed: {err:#}");
                self.set_error("chunk conversion failed", format!("{err:#}"));
                break;
            }
            // The shared converter now knows the model's native rate, actual
            // capped overlap, and both rate-adapter delays. Report these once
            // available; the callback only relays this precomputed atomic value.
            // SOLA's bounded search can advance audio within that nominal hold.
            let content_delay = converter.content_delay();
            self.latency.store(
                u32::try_from(chunk_samples.saturating_add(content_delay)).unwrap_or(u32::MAX),
                Ordering::Relaxed,
            );
            input_acc.clear();
            if self.state.generation.load(Ordering::Acquire) != generation {
                continue;
            }
            queued_output.clear();
            queued_output.extend(
                chunk_out
                    .iter()
                    .map(|&value| QueuedSample { value, generation }),
            );
            let mut pushed = 0;
            while pushed < queued_output.len() {
                let (written, _) = self
                    .output_producer
                    .push_partial_slice(&queued_output[pushed..]);
                pushed += written.len();
                self.state.notify_offline();
                if pushed == queued_output.len()
                    || !self.state.offline.load(Ordering::Acquire)
                    || !self.running.load(Ordering::Acquire)
                    || self.state.generation.load(Ordering::Acquire) != generation
                {
                    break;
                }
                // Offline output is never dropped. Its callback consumes then
                // unparks us; realtime retains the bounded drop-on-overflow rule.
                thread::park_timeout(Duration::from_millis(100));
            }
        }
    }

    fn load_conversion(
        &self,
        settings: &PluginConfig,
        chunk_samples: usize,
    ) -> Option<Box<dyn WorkerConversion>> {
        match self.load_current(settings, chunk_samples) {
            Ok((pipeline, kind)) => pipeline.map(|pipeline| {
                Box::new(self.chunk_converter(pipeline, kind, chunk_samples))
                    as Box<dyn WorkerConversion>
            }),
            Err(()) => {
                self.dirty.store(true, Ordering::SeqCst);
                None
            }
        }
    }

    fn prepare_offline(
        &self,
        converter: &mut dyn WorkerConversion,
        zeros: &[f32],
    ) -> anyhow::Result<()> {
        if self.state.offline.load(Ordering::Acquire) {
            // A finite host render must preserve its first input hop. Reuse the
            // shared finite path's zero-hop prime; realtime keeps its existing
            // first-real-chunk startup policy. Reset/mode change owns both states.
            converter.apply_live(&self.live_params());
            converter.prime(zeros, self.sample_rate)?;
            self.latency.store(
                u32::try_from(zeros.len().saturating_add(converter.content_delay()))
                    .unwrap_or(u32::MAX),
                Ordering::Relaxed,
            );
        }
        Ok(())
    }

    fn live_params(&self) -> LiveParams {
        LiveParams {
            pitch_shift: self.params.pitch_shift.value(),
            speaker_id: self.params.speaker_id.value() as i64,
            input_gain: util::db_to_gain(self.params.input_gain_db.value()),
            output_gain: util::db_to_gain(self.params.output_gain_db.value()),
            noise_gate_enabled: self.params.noise_gate.value(),
            noise_gate_threshold: util::db_to_gain(self.params.noise_gate_threshold_db.value()),
        }
    }

    fn publish_conversion(&self, settings: &PluginConfig, loaded: bool) {
        if loaded || !settings.has_models() {
            // A failed rebuild/reload does not erase the last successful Load
            // authorization. Explicitly clearing the model set does revoke it.
            *self.state.applied_settings.lock().unwrap() = loaded.then(|| settings.clone());
        }
        self.state.has_converter.store(loaded, Ordering::Release);
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

#[cfg(test)]
pub(crate) fn test_loaded_runtime(
    params: Arc<VcRvcParams>,
    mode: ProcessMode,
    max_block: usize,
) -> PluginRuntime {
    tests::loaded_runtime(params, mode, max_block, tests::TestControl::default())
}

#[cfg(test)]
impl PluginRuntime {
    pub(crate) fn test_worker_id(&self) -> thread::ThreadId {
        self.worker_thread.id()
    }
    pub(crate) fn test_output_samples(&mut self) -> usize {
        self.output_consumer.slots()
    }
    pub(crate) fn test_input_hop(&self) -> usize {
        self.state.input_hop.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::time::Instant;
    use vc_core::model_rvc::{ContentDelay, ModelOutput, PassthroughModel, VoiceModel};

    const RATE: u32 = 16_000;
    const HOP: usize = 320;

    #[derive(Default)]
    pub(super) struct TestControl {
        gate: Option<(Sender<()>, Receiver<()>)>,
        panic_on_audio: bool,
        resets: Arc<AtomicUsize>,
        input_delay: usize,
    }

    // A rolling identity generator with the same candidate/context and declared
    // delay contracts as RVC. Actual ChunkConverter/SOLA/PSOLA/output resampling
    // run in these tests; only neural inference is replaced.
    struct TestVoice {
        history: Vec<f32>,
        candidate_samples: usize,
        control: TestControl,
    }

    impl VoiceModel for TestVoice {
        fn input_content_delay(&self) -> ContentDelay {
            ContentDelay::from_samples(self.control.input_delay, RATE)
        }

        fn process(
            &mut self,
            input: &[f32],
            rate: u32,
            out: &mut Vec<f32>,
            pitch: &mut Vec<f32>,
        ) -> anyhow::Result<ModelOutput> {
            if input.iter().any(|&sample| sample != 0.0) {
                if let Some((started, release)) = self.control.gate.take() {
                    started.send(()).unwrap();
                    release.recv_timeout(Duration::from_secs(2)).unwrap();
                }
                assert!(!self.control.panic_on_audio, "intentional inference panic");
                thread::sleep(Duration::from_millis(1));
            }
            self.history.extend_from_slice(input);
            let needed = self.candidate_samples + self.control.input_delay;
            if self.history.len() < needed {
                let mut padded = vec![0.0; needed - self.history.len()];
                padded.append(&mut self.history);
                self.history = padded;
            }
            let end = self.history.len() - self.control.input_delay;
            out.clear();
            out.extend_from_slice(&self.history[end - self.candidate_samples..end]);
            if self.history.len() > needed {
                self.history.drain(..self.history.len() - needed);
            }
            let mut metadata_audio = Vec::new();
            let mut metadata = PassthroughModel.process(input, rate, &mut metadata_audio, pitch)?;
            pitch.resize(64, 220.0);
            metadata.raw_output_samples = out.len();
            Ok(metadata)
        }
    }

    struct TestConversion(ChunkConverter<TestVoice>);
    impl WorkerConversion for TestConversion {
        fn apply_live(&mut self, _: &LiveParams) {}
        fn reset(&mut self) -> anyhow::Result<()> {
            self.0.model_mut().history.clear();
            self.0
                .model_mut()
                .control
                .resets
                .fetch_add(1, Ordering::Relaxed);
            self.0.reset_streaming_state();
            Ok(())
        }
        fn prime(&mut self, input: &[f32], rate: u32) -> anyhow::Result<()> {
            self.0.prime(input, rate).map(|_| ())
        }
        fn process(
            &mut self,
            input: &[f32],
            rate: u32,
            out: &mut Vec<f32>,
        ) -> anyhow::Result<ChunkStats> {
            self.0.process_chunk(input, rate, out)
        }
        fn content_delay(&self) -> usize {
            self.0.output_content_delay_samples()
        }
    }

    pub(super) fn loaded_runtime(
        params: Arc<VcRvcParams>,
        mode: ProcessMode,
        max_block: usize,
        control: TestControl,
    ) -> PluginRuntime {
        let settings = params.settings.read().unwrap().clone();
        let extra = RATE as usize
            * (settings.crossfade_ms
                + settings.sola_search_ms
                + settings.rvc_output_tail_discard_ms) as usize
            / 1000;
        let converter = TestConversion(ChunkConverter::new(
            TestVoice {
                history: Vec::new(),
                candidate_samples: HOP + extra,
                control,
            },
            ChunkOutputConfig {
                kind: settings.smoothing_kind(),
                output_sample_rate: RATE,
                output_chunk_samples: HOP,
                crossfade_ms: settings.crossfade_ms,
                sola_search_ms: settings.sola_search_ms,
                tail_discard_ms: settings.rvc_output_tail_discard_ms,
            },
        ));
        let runtime = PluginRuntime::start_inner(
            params,
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(Mutex::new(PluginStatus::new("test running"))),
            &Arc::new(ReloadWaker::default()),
            RATE,
            max_block,
            mode,
            None,
            Some(Box::new(converter)),
        );
        wait_for(|| {
            runtime.state.has_converter.load(Ordering::Acquire)
                && runtime.state.ready_generation.load(Ordering::Acquire) == 1
        });
        runtime
    }

    pub(crate) fn configured_params(fade: u32, tail: u32, kind: &str) -> Arc<VcRvcParams> {
        let params = Arc::new(VcRvcParams::default());
        *params.settings.write().unwrap() = PluginConfig {
            model: "test-rvc.onnx".into(),
            embedder: "test-embedder.onnx".into(),
            f0_model: "test-f0.onnx".into(),
            chunk_ms: 20,
            crossfade_ms: fade,
            sola_search_ms: 0,
            rvc_output_tail_discard_ms: tail,
            smoother: kind.into(),
            ..PluginConfig::default()
        };
        params
    }

    fn wait_for(mut predicate: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !predicate() {
            assert!(Instant::now() < deadline, "worker did not make progress");
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn render(runtime: &mut PluginRuntime, input: &[f32], blocks: &[usize]) -> Vec<f32> {
        let mut output = Vec::new();
        let mut offset = 0;
        let mut index = 0;
        while offset < input.len() {
            let take = blocks[index % blocks.len()].min(input.len() - offset);
            let mut block = input[offset..offset + take].to_vec();
            runtime.process_block(&mut [&mut block]);
            output.extend_from_slice(&block);
            offset += take;
            index += 1;
        }
        output
    }

    #[test]
    fn offline_freewheeling_preserves_first_short_and_final_partial_audio() {
        for kind in ["sola", "psola"] {
            for len in [1, 37, HOP - 1, HOP + 1, HOP * 4 + 79] {
                let params = configured_params(10, 10, kind);
                let mut runtime = loaded_runtime(
                    params,
                    ProcessMode::Offline,
                    2048,
                    TestControl {
                        input_delay: 17,
                        ..TestControl::default()
                    },
                );
                let latency = runtime.current_latency() as usize;
                assert_eq!(latency, HOP + 160 + 160 + 17);
                assert_eq!(runtime.tail_samples(), Some(latency as u32));
                let mut input: Vec<f32> = (0..len)
                    .map(|i| 0.2 + 0.1 * (i as f32 * 0.017).cos())
                    .collect();
                input[0] = 0.8;
                input[len - 1] = 0.7;
                let mut host_input = input.clone();
                host_input.resize(len + latency, 0.0);
                let output = render(&mut runtime, &host_input, &[37, 1024, 63, 17]);
                let error = output[latency..]
                    .iter()
                    .zip(&input)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f32, f32::max);
                assert!(error < 1e-5, "{kind} len={len} first/tail error={error}");
            }
        }
    }

    #[test]
    fn reset_rejects_queued_and_in_flight_audio_and_clears_model_context() {
        let params = configured_params(10, 0, "sola");
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let resets = Arc::new(AtomicUsize::new(0));
        let mut runtime = loaded_runtime(
            params,
            ProcessMode::Realtime,
            HOP,
            TestControl {
                gate: Some((started_tx, release_rx)),
                resets: Arc::clone(&resets),
                input_delay: HOP + 17,
                ..TestControl::default()
            },
        );
        runtime.process_block(&mut [&mut [0.75; HOP]]);
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        runtime.reset();
        let mut during_reset = [0.25; HOP];
        runtime.process_block(&mut [&mut during_reset]);
        assert!(during_reset.iter().all(|&value| value == 0.0));
        release_tx.send(()).unwrap();
        wait_for(|| {
            runtime.state.ready_generation.load(Ordering::Acquire) == 2
                && runtime.output_consumer.slots() >= HOP
        });
        assert_eq!(resets.load(Ordering::Relaxed), 1);
        let mut resumed = [0.0; HOP];
        runtime.process_block(&mut [&mut resumed]);
        assert!(resumed.iter().all(|&value| value == 0.0)); // fresh unprimed realtime policy
        wait_for(|| runtime.output_consumer.slots() >= HOP);
        let mut after_context_reset = [0.0; HOP];
        runtime.process_block(&mut [&mut after_context_reset]);
        assert!(after_context_reset
            .iter()
            .all(|&value| value.abs() <= 0.25 + 1e-6));
        wait_for(|| runtime.output_consumer.slots() >= HOP);
        runtime.reset();
        let mut queued_stale = [0.0; HOP];
        runtime.process_block(&mut [&mut queued_stale]);
        assert!(queued_stale.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn realtime_to_offline_discards_full_old_input_backlog_before_accepting_audio() {
        let params = configured_params(0, 0, "sola");
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let mut runtime = loaded_runtime(
            params,
            ProcessMode::Realtime,
            2048,
            TestControl {
                gate: Some((started_tx, release_rx)),
                ..TestControl::default()
            },
        );
        runtime.process_block(&mut [&mut [0.75; HOP]]);
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let old_backlog = vec![
            QueuedSample {
                value: 0.75,
                generation: 1
            };
            runtime.input_producer.slots()
        ];
        runtime
            .input_producer
            .push_entire_slice(&old_backlog)
            .unwrap();
        assert!(runtime.reconfigure(RATE, 2048, ProcessMode::Offline));
        release_tx.send(()).unwrap();
        let input = vec![0.25; HOP + 41];
        let mut host_input = input.clone();
        host_input.resize(input.len() + HOP, 0.0);
        let output = render(&mut runtime, &host_input, &[37, 1024, 63]);
        assert_eq!(&output[HOP..], input.as_slice());
    }

    #[test]
    fn offline_wait_returns_when_inference_panics() {
        let (done_tx, done_rx) = mpsc::channel();
        thread::spawn(move || {
            let params = configured_params(0, 0, "sola");
            let mut runtime = loaded_runtime(
                params,
                ProcessMode::Offline,
                HOP,
                TestControl {
                    panic_on_audio: true,
                    ..TestControl::default()
                },
            );
            runtime.process_block(&mut [&mut [0.25; HOP]]);
            let mut block = [0.25; HOP];
            runtime.process_block(&mut [&mut block]);
            assert!(
                !runtime.reconfigure(RATE, HOP, ProcessMode::Offline),
                "a dead worker must be replaced on host reinitialization"
            );
            done_tx.send(block).unwrap();
        });
        let output = done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("offline host hung after worker panic");
        assert_eq!(output, [0.0; HOP]);
    }

    #[test]
    fn generation_read_preserves_future_input_and_handles_ring_wrap() {
        let (mut producer, mut consumer) = RingBuffer::<QueuedSample>::new(8);
        for _ in 0..6 {
            producer
                .push(QueuedSample {
                    value: 0.0,
                    generation: u32::MAX - 2,
                })
                .unwrap();
        }
        read_generation(&mut consumer, u32::MAX - 1, &mut []);
        for generation in [u32::MAX - 1, u32::MAX, 0, 1] {
            producer
                .push(QueuedSample {
                    value: generation as f32,
                    generation,
                })
                .unwrap();
        }
        let mut output = [0.0; 4];
        assert_eq!(read_generation(&mut consumer, 0, &mut output), 1);
        assert_eq!(consumer.slots(), 1); // generation 1 awaits its own reset
        producer
            .push(QueuedSample {
                value: 0.5,
                generation: 1,
            })
            .unwrap();
        assert_eq!(read_generation(&mut consumer, 1, &mut output), 2);
        assert_eq!(output[1], 0.5);
    }
}
