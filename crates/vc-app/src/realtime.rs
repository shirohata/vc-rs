use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use rtrb::RingBuffer;
use thread_priority::{set_current_thread_priority, ThreadPriority};
use vc_core::dsp;
use vc_core::model_rvc::{
    GpuPriority, PassthroughModel, RvcPipeline, RvcPipelineConfig, VoiceModel,
};
use vc_core::sola::{self, ChunkSmootherConfig, SmoothingKind};
use vc_core::Provider;

use crate::audio::{self, AudioStream, RealtimeAudio};

const INPUT_QUEUE_CHUNKS: usize = 4;
const OUTPUT_QUEUE_CHUNKS: usize = 4;
const COMMAND_CAPACITY: usize = 8;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AudioBackend {
    #[default]
    Cpal,
    Wasapi,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Smoother {
    #[default]
    Sola,
    Psola,
}

impl Smoother {
    fn kind(self) -> SmoothingKind {
        match self {
            Self::Sola => SmoothingKind::Sola,
            Self::Psola => SmoothingKind::Psola,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RealtimeConfig {
    pub model: Option<PathBuf>,
    pub embedder: Option<PathBuf>,
    pub embedder_output: Option<String>,
    pub f0_model: Option<PathBuf>,
    pub provider: Provider,
    pub gpu_priority: GpuPriority,
    pub audio_backend: AudioBackend,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub wasapi_input_exclusive: bool,
    pub wasapi_output_exclusive: bool,
    pub wasapi_buffer_ms: u32,
    pub chunk_ms: u32,
    pub crossfade_ms: u32,
    pub sola_search_ms: u32,
    pub smoother: Smoother,
    pub rvc_output_tail_discard_ms: u32,
    pub extra_convert_ms: u32,
    pub f0_threshold: f32,
    pub silence_threshold: f32,
    pub volume_envelope: bool,
    pub rms_mix_rate: f32,
    pub auto_output_gain: bool,
    pub target_output_rms: f32,
    pub max_output_gain: f32,
    pub passthrough: bool,
    pub debug_input_wav: Option<PathBuf>,
    pub debug_output_wav: Option<PathBuf>,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            model: None,
            embedder: None,
            embedder_output: None,
            f0_model: None,
            provider: Provider::Cpu,
            gpu_priority: GpuPriority::default(),
            audio_backend: AudioBackend::Cpal,
            input_device: None,
            output_device: None,
            wasapi_input_exclusive: false,
            wasapi_output_exclusive: false,
            wasapi_buffer_ms: 0,
            chunk_ms: 500,
            crossfade_ms: 85,
            sola_search_ms: 12,
            smoother: Smoother::Sola,
            rvc_output_tail_discard_ms: 10,
            extra_convert_ms: 100,
            f0_threshold: 0.3,
            silence_threshold: 0.0001,
            volume_envelope: false,
            rms_mix_rate: 0.0,
            auto_output_gain: false,
            target_output_rms: 0.03,
            max_output_gain: 512.0,
            passthrough: false,
            debug_input_wav: None,
            debug_output_wav: None,
        }
    }
}

impl RealtimeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.wasapi_input_exclusive || self.wasapi_output_exclusive {
            if self.audio_backend != AudioBackend::Wasapi {
                bail!("WASAPI exclusive options require the WASAPI backend");
            }
        }
        if self.chunk_ms == 0 {
            bail!("chunk size must be greater than zero");
        }
        if !(0.0..=1.0).contains(&self.rms_mix_rate) || !self.rms_mix_rate.is_finite() {
            bail!("RMS mix rate must be a finite value in 0.0..=1.0");
        }
        if !self.passthrough {
            if self.model.is_none() || self.embedder.is_none() || self.f0_model.is_none() {
                bail!("model, embedder, and F0 model are required");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LiveParams {
    pub pitch_shift: f32,
    pub speaker_id: i64,
    pub input_gain: f32,
    pub output_gain: f32,
}

impl Default for LiveParams {
    fn default() -> Self {
        Self {
            pitch_shift: 0.0,
            speaker_id: 0,
            input_gain: 1.0,
            output_gain: 1.0,
        }
    }
}

#[derive(Default)]
struct AtomicLiveParams {
    pitch_shift: AtomicU32,
    speaker_id: AtomicI64,
    input_gain: AtomicU32,
    output_gain: AtomicU32,
}

impl AtomicLiveParams {
    fn new(value: LiveParams) -> Self {
        let this = Self::default();
        this.store(value);
        this
    }

    fn store(&self, value: LiveParams) {
        self.pitch_shift
            .store(value.pitch_shift.to_bits(), Ordering::Relaxed);
        self.speaker_id.store(value.speaker_id, Ordering::Relaxed);
        self.input_gain
            .store(value.input_gain.to_bits(), Ordering::Relaxed);
        self.output_gain
            .store(value.output_gain.to_bits(), Ordering::Relaxed);
    }

    fn load(&self) -> LiveParams {
        LiveParams {
            pitch_shift: f32::from_bits(self.pitch_shift.load(Ordering::Relaxed)),
            speaker_id: self.speaker_id.load(Ordering::Relaxed),
            input_gain: f32::from_bits(self.input_gain.load(Ordering::Relaxed)),
            output_gain: f32::from_bits(self.output_gain.load(Ordering::Relaxed)),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EngineState {
    #[default]
    Stopped,
    Starting,
    Running,
    Stopping,
    Error,
}

#[derive(Clone, Debug, Default)]
pub struct EngineStatusSnapshot {
    pub state: EngineState,
    pub message: String,
    pub input_device: String,
    pub output_device: String,
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
}

#[derive(Clone, Debug, Default)]
pub struct DeviceList {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TelemetrySnapshot {
    pub chunks: u64,
    pub inference_us: u64,
    pub input_rms: f32,
    pub output_rms: f32,
    pub input_overruns: u64,
    pub output_underruns: u64,
    pub output_dropped_samples: u64,
    pub output_buffer_samples: u64,
}

#[derive(Default)]
struct Telemetry {
    chunks: AtomicU64,
    inference_us: AtomicU64,
    input_rms_bits: AtomicU32,
    output_rms_bits: AtomicU32,
    input_overruns: AtomicU64,
    output_underruns: AtomicU64,
    output_dropped_samples: AtomicU64,
    output_buffer_samples: AtomicU64,
}

impl Telemetry {
    fn reset(&self) {
        self.chunks.store(0, Ordering::Relaxed);
        self.inference_us.store(0, Ordering::Relaxed);
        self.input_rms_bits.store(0, Ordering::Relaxed);
        self.output_rms_bits.store(0, Ordering::Relaxed);
        self.input_overruns.store(0, Ordering::Relaxed);
        self.output_underruns.store(0, Ordering::Relaxed);
        self.output_dropped_samples.store(0, Ordering::Relaxed);
        self.output_buffer_samples.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            chunks: self.chunks.load(Ordering::Relaxed),
            inference_us: self.inference_us.load(Ordering::Relaxed),
            input_rms: f32::from_bits(self.input_rms_bits.load(Ordering::Relaxed)),
            output_rms: f32::from_bits(self.output_rms_bits.load(Ordering::Relaxed)),
            input_overruns: self.input_overruns.load(Ordering::Relaxed),
            output_underruns: self.output_underruns.load(Ordering::Relaxed),
            output_dropped_samples: self.output_dropped_samples.load(Ordering::Relaxed),
            output_buffer_samples: self.output_buffer_samples.load(Ordering::Relaxed),
        }
    }
}

enum Command {
    Apply(RealtimeConfig),
    Stop,
    RefreshDevices(AudioBackend),
    Shutdown,
}

pub struct EngineController {
    tx: SyncSender<Command>,
    status: Arc<Mutex<EngineStatusSnapshot>>,
    devices: Arc<Mutex<DeviceList>>,
    telemetry: Arc<Telemetry>,
    live: Arc<AtomicLiveParams>,
    control: Option<JoinHandle<()>>,
}

impl EngineController {
    pub fn new(initial_live: LiveParams) -> Self {
        let (tx, rx) = mpsc::sync_channel(COMMAND_CAPACITY);
        let status = Arc::new(Mutex::new(EngineStatusSnapshot::default()));
        let devices = Arc::new(Mutex::new(DeviceList::default()));
        let telemetry = Arc::new(Telemetry::default());
        let live = Arc::new(AtomicLiveParams::new(initial_live));
        let control = {
            let status = Arc::clone(&status);
            let devices = Arc::clone(&devices);
            let telemetry = Arc::clone(&telemetry);
            let live = Arc::clone(&live);
            thread::Builder::new()
                .name("vc-app-control".to_string())
                .stack_size(64 * 1024 * 1024)
                .spawn(move || control_loop(rx, status, devices, telemetry, live))
                .expect("failed to spawn vc-app control thread")
        };
        Self {
            tx,
            status,
            devices,
            telemetry,
            live,
            control: Some(control),
        }
    }

    pub fn apply_config(&self, config: RealtimeConfig) -> Result<()> {
        config.validate()?;
        self.try_command(Command::Apply(config))
    }

    pub fn stop(&self) -> Result<()> {
        self.try_command(Command::Stop)
    }

    pub fn refresh_devices(&self, backend: AudioBackend) -> Result<()> {
        self.try_command(Command::RefreshDevices(backend))
    }

    pub fn set_live_params(&self, params: LiveParams) {
        self.live.store(params);
    }

    pub fn snapshot(&self) -> (EngineStatusSnapshot, TelemetrySnapshot, DeviceList) {
        let status = self.status.lock().map(|s| s.clone()).unwrap_or_default();
        let devices = self.devices.lock().map(|d| d.clone()).unwrap_or_default();
        (status, self.telemetry.snapshot(), devices)
    }

    fn try_command(&self, command: Command) -> Result<()> {
        self.tx.try_send(command).map_err(|err| match err {
            TrySendError::Full(_) => anyhow!("engine command queue is full"),
            TrySendError::Disconnected(_) => anyhow!("engine control thread has stopped"),
        })
    }
}

impl Drop for EngineController {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(control) = self.control.take() {
            let _ = control.join();
        }
    }
}

fn control_loop(
    rx: Receiver<Command>,
    status: Arc<Mutex<EngineStatusSnapshot>>,
    devices: Arc<Mutex<DeviceList>>,
    telemetry: Arc<Telemetry>,
    live: Arc<AtomicLiveParams>,
) {
    let mut session: Option<RealtimeSession> = None;
    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Command::Apply(config)) => {
                set_status(&status, EngineState::Stopping, "Stopping previous session");
                drop(session.take());
                set_status(
                    &status,
                    EngineState::Starting,
                    "Loading model and audio devices",
                );
                telemetry.reset();
                match RealtimeSession::start(config, Arc::clone(&telemetry), Arc::clone(&live)) {
                    Ok(new_session) => {
                        if let Ok(mut current) = status.lock() {
                            *current = new_session.status();
                        }
                        session = Some(new_session);
                    }
                    Err(err) => set_status(&status, EngineState::Error, format!("{err:#}")),
                }
            }
            Ok(Command::Stop) => {
                set_status(&status, EngineState::Stopping, "Stopping");
                drop(session.take());
                set_status(&status, EngineState::Stopped, "Stopped");
            }
            Ok(Command::RefreshDevices(backend)) => {
                let result = device_list(backend);
                if let Ok(mut current) = devices.lock() {
                    *current = result;
                }
            }
            Ok(Command::Shutdown) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
        if session
            .as_ref()
            .is_some_and(|s| !s.running.load(Ordering::Relaxed))
        {
            drop(session.take());
            set_status(&status, EngineState::Error, "Realtime worker stopped");
        }
    }
    drop(session);
}

fn set_status(
    status: &Mutex<EngineStatusSnapshot>,
    state: EngineState,
    message: impl Into<String>,
) {
    if let Ok(mut status) = status.lock() {
        status.state = state;
        status.message = message.into();
        if state != EngineState::Running {
            status.input_device.clear();
            status.output_device.clear();
            status.input_sample_rate = 0;
            status.output_sample_rate = 0;
        }
    }
}

fn device_list(backend: AudioBackend) -> DeviceList {
    let result = match backend {
        AudioBackend::Cpal => audio::cpal_device_names(),
        AudioBackend::Wasapi => wasapi_device_names(),
    };
    match result {
        Ok((inputs, outputs)) => DeviceList {
            inputs,
            outputs,
            error: None,
        },
        Err(err) => DeviceList {
            error: Some(format!("{err:#}")),
            ..Default::default()
        },
    }
}

#[cfg(windows)]
fn wasapi_device_names() -> Result<(Vec<String>, Vec<String>)> {
    crate::audio::wasapi_audio::device_names()
}

#[cfg(not(windows))]
fn wasapi_device_names() -> Result<(Vec<String>, Vec<String>)> {
    bail!("WASAPI is only available on Windows")
}

enum RuntimeModel {
    Passthrough(PassthroughModel),
    Rvc(RvcPipeline),
}

impl RuntimeModel {
    fn apply_live(&mut self, live: LiveParams) {
        if let Self::Rvc(model) = self {
            model.set_pitch_shift(live.pitch_shift);
            model.set_speaker_id(live.speaker_id);
            model.set_input_gain(live.input_gain);
            model.set_output_gain(live.output_gain);
        }
    }
}

impl VoiceModel for RuntimeModel {
    fn process(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<vc_core::model_rvc::ModelOutput> {
        match self {
            Self::Passthrough(model) => model.process(audio, sample_rate),
            Self::Rvc(model) => model.process(audio, sample_rate),
        }
    }
}

struct RealtimeSession {
    running: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    input_stream: Option<AudioStream>,
    output_stream: Option<AudioStream>,
    status: EngineStatusSnapshot,
    debug_input_wav: Option<PathBuf>,
    debug_output_wav: Option<PathBuf>,
    debug_input: Arc<Mutex<Vec<f32>>>,
    debug_output: Arc<Mutex<Vec<f32>>>,
    input_rate: u32,
    output_rate: u32,
}

impl RealtimeSession {
    fn start(
        config: RealtimeConfig,
        telemetry: Arc<Telemetry>,
        live: Arc<AtomicLiveParams>,
    ) -> Result<Self> {
        let audio = RealtimeAudio::open(
            config.audio_backend,
            config.wasapi_input_exclusive,
            config.wasapi_output_exclusive,
            config.input_device.as_deref(),
            config.output_device.as_deref(),
            config.wasapi_buffer_ms,
        )?;
        let input_rate = audio.input_sample_rate();
        let output_rate = audio.output_sample_rate();
        let input_chunk = chunk_samples_for_rate(input_rate, config.chunk_ms);
        let output_chunk = chunk_samples_for_rate(output_rate, config.chunk_ms);
        let output_extra_ms = if config.passthrough {
            0
        } else {
            config
                .crossfade_ms
                .saturating_add(config.sola_search_ms)
                .saturating_add(config.rvc_output_tail_discard_ms)
        };
        let current_live = live.load();
        let debug_input = Arc::new(Mutex::new(Vec::new()));
        let debug_output = Arc::new(Mutex::new(Vec::new()));
        let model = if config.passthrough {
            RuntimeModel::Passthrough(PassthroughModel)
        } else {
            RuntimeModel::Rvc(RvcPipeline::load(RvcPipelineConfig {
                model: config.model.as_ref().expect("validated"),
                embedder: config.embedder.as_ref().expect("validated"),
                embedder_output: config.embedder_output.as_deref(),
                f0_model: config.f0_model.as_ref().expect("validated"),
                provider: config.provider,
                gpu_priority: config.gpu_priority,
                sample_rate: input_rate,
                chunk_samples: input_chunk,
                speaker_id: current_live.speaker_id,
                pitch_shift: current_live.pitch_shift,
                f0_threshold: config.f0_threshold,
                silence_threshold: config.silence_threshold,
                input_gain: current_live.input_gain,
                output_extra_ms,
                volume_excluded_ms: config.crossfade_ms,
                extra_convert_ms: config.extra_convert_ms,
                output_gain: current_live.output_gain,
                volume_envelope: config.volume_envelope,
                rms_mix_rate: config.rms_mix_rate,
                auto_output_gain: config.auto_output_gain,
                target_output_rms: config.target_output_rms,
                max_output_gain: config.max_output_gain,
            })?)
        };

        let (mut input_producer, mut input_consumer) =
            RingBuffer::<f32>::new(input_chunk * INPUT_QUEUE_CHUNKS);
        let output_capacity = output_chunk * OUTPUT_QUEUE_CHUNKS;
        let (mut output_producer, mut output_consumer) = RingBuffer::<f32>::new(output_capacity);
        let running = Arc::new(AtomicBool::new(true));
        let worker_running = Arc::clone(&running);
        let worker_telemetry = Arc::clone(&telemetry);
        let worker_debug_input = Arc::clone(&debug_input);
        let worker_debug_output = Arc::clone(&debug_output);
        let capture_input = config.debug_input_wav.is_some();
        let capture_output = config.debug_output_wav.is_some();
        let smoothing_enabled = !config.passthrough;
        let mut worker = Some(
            thread::Builder::new()
                .name("vc-app-inference".to_string())
                .spawn(move || {
                    if let Err(err) = set_current_thread_priority(ThreadPriority::Max) {
                        tracing::warn!("failed to set inference worker thread priority: {err}");
                    }
                    let mut model = model;
                    let mut input_acc = Vec::<f32>::with_capacity(input_chunk * 2);
                    let mut prepared = Vec::<f32>::with_capacity(output_chunk * 2);
                    let mut resampler = match dsp::StreamingResampleMono::new(
                        input_rate as usize,
                        output_rate as usize,
                    ) {
                        Ok(value) => value,
                        Err(_) => {
                            worker_running.store(false, Ordering::SeqCst);
                            return;
                        }
                    };
                    let mut smoother = None::<(u32, sola::ChunkSmoother)>;
                    while worker_running.load(Ordering::SeqCst) {
                        let available = input_consumer
                            .slots()
                            .min(input_chunk.saturating_sub(input_acc.len()));
                        if available > 0 {
                            let old = input_acc.len();
                            input_acc.resize(old + available, 0.0);
                            if input_consumer
                                .pop_entire_slice(&mut input_acc[old..])
                                .is_err()
                            {
                                input_acc.truncate(old);
                            }
                        }
                        if input_acc.len() < input_chunk {
                            thread::sleep(Duration::from_millis(2));
                            continue;
                        }
                        if capture_input {
                            if let Ok(mut samples) = worker_debug_input.lock() {
                                samples.extend_from_slice(&input_acc[..input_chunk]);
                            }
                        }
                        model.apply_live(live.load());
                        let out = model.process(&input_acc[..input_chunk], input_rate);
                        input_acc.clear();
                        let Ok(out) = out else {
                            worker_running.store(false, Ordering::SeqCst);
                            break;
                        };
                        worker_telemetry.chunks.fetch_add(1, Ordering::Relaxed);
                        worker_telemetry
                            .inference_us
                            .store(out.inference_time.as_micros() as u64, Ordering::Relaxed);
                        worker_telemetry
                            .input_rms_bits
                            .store(out.input_rms.to_bits(), Ordering::Relaxed);
                        worker_telemetry
                            .output_rms_bits
                            .store(out.output_rms.to_bits(), Ordering::Relaxed);
                        let output_silent = out.silent;
                        prepared.clear();
                        if smoothing_enabled {
                            if smoother.as_ref().map(|(rate, _)| *rate) != Some(out.sample_rate) {
                                smoother = Some((
                                    out.sample_rate,
                                    sola::model_domain_chunk_smoother(ChunkSmootherConfig {
                                        kind: config.smoother.kind(),
                                        output_chunk_samples: output_chunk,
                                        output_sample_rate: output_rate,
                                        model_sample_rate: out.sample_rate,
                                        crossfade_ms: config.crossfade_ms,
                                        sola_search_ms: config.sola_search_ms,
                                        tail_discard_ms: config.rvc_output_tail_discard_ms,
                                    }),
                                ));
                            }
                            match sola::prepare_model_output(
                                out,
                                output_rate,
                                output_chunk,
                                &mut smoother.as_mut().unwrap().1,
                                None,
                            ) {
                                Ok(value) => prepared = value.audio,
                                Err(_) => {
                                    worker_running.store(false, Ordering::SeqCst);
                                    break;
                                }
                            }
                        } else if resampler.process_into(&out.audio, &mut prepared).is_err() {
                            worker_running.store(false, Ordering::SeqCst);
                            break;
                        }
                        if capture_output {
                            if let Ok(mut samples) = worker_debug_output.lock() {
                                samples.extend_from_slice(&prepared);
                            }
                        }
                        let should_queue = !output_silent
                            || should_queue_silent_output(
                                output_capacity - output_producer.slots(),
                                output_chunk,
                            );
                        if should_queue {
                            let (_, remainder) = output_producer.push_partial_slice(&prepared);
                            worker_telemetry
                                .output_dropped_samples
                                .fetch_add(remainder.len() as u64, Ordering::Relaxed);
                        }
                        worker_telemetry.output_buffer_samples.store(
                            (output_capacity - output_producer.slots()) as u64,
                            Ordering::Relaxed,
                        );
                    }
                })?,
        );

        let input_running = Arc::clone(&running);
        let input_telemetry = Arc::clone(&telemetry);
        let input_stream = match audio.build_input_stream(move |samples| {
            if !input_running.load(Ordering::Relaxed) {
                return;
            }
            let (_, remainder) = input_producer.push_partial_slice(samples);
            if !remainder.is_empty() {
                input_telemetry
                    .input_overruns
                    .fetch_add(1, Ordering::Relaxed);
            }
        }) {
            Ok(stream) => stream,
            Err(err) => {
                stop_startup_worker(&running, &mut worker);
                return Err(err);
            }
        };
        let output_running = Arc::clone(&running);
        let output_telemetry = Arc::clone(&telemetry);
        let output_stream = match audio.build_output_stream(move |out| {
            if !output_running.load(Ordering::Relaxed) {
                out.fill(0.0);
                return;
            }
            let (_, remainder) = output_consumer.pop_partial_slice(out);
            if !remainder.is_empty() {
                remainder.fill(0.0);
                output_telemetry
                    .output_underruns
                    .fetch_add(1, Ordering::Relaxed);
            }
            output_telemetry
                .output_buffer_samples
                .store(output_consumer.cached_slots() as u64, Ordering::Relaxed);
        }) {
            Ok(stream) => stream,
            Err(err) => {
                drop(input_stream);
                stop_startup_worker(&running, &mut worker);
                return Err(err);
            }
        };
        if let Err(err) = output_stream.play().and_then(|_| input_stream.play()) {
            drop(input_stream);
            drop(output_stream);
            stop_startup_worker(&running, &mut worker);
            return Err(err);
        }

        Ok(Self {
            running,
            worker: worker.take(),
            input_stream: Some(input_stream),
            output_stream: Some(output_stream),
            status: EngineStatusSnapshot {
                state: EngineState::Running,
                message: format!("Running ({})", audio.backend_label()),
                input_device: audio.input_name().to_string(),
                output_device: audio.output_name().to_string(),
                input_sample_rate: input_rate,
                output_sample_rate: output_rate,
            },
            debug_input_wav: config.debug_input_wav,
            debug_output_wav: config.debug_output_wav,
            debug_input,
            debug_output,
            input_rate,
            output_rate,
        })
    }

    fn status(&self) -> EngineStatusSnapshot {
        self.status.clone()
    }
}

impl Drop for RealtimeSession {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        drop(self.input_stream.take());
        drop(self.output_stream.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        if let Some(path) = &self.debug_input_wav {
            if let Ok(samples) = self.debug_input.lock() {
                let _ = write_wav(path, &samples, self.input_rate);
            }
        }
        if let Some(path) = &self.debug_output_wav {
            if let Ok(samples) = self.debug_output.lock() {
                let _ = write_wav(path, &samples, self.output_rate);
            }
        }
    }
}

fn chunk_samples_for_rate(sample_rate: u32, chunk_ms: u32) -> usize {
    ((sample_rate as u64 * chunk_ms as u64) / 1000).max(128) as usize
}

fn should_queue_silent_output(buffered: usize, output_chunk: usize) -> bool {
    // Keep at most one generated-silence chunk queued. Filling the output ring
    // during quiet periods delays or drops the first converted speech when
    // input resumes.
    buffered <= output_chunk
}

fn stop_startup_worker(running: &AtomicBool, worker: &mut Option<JoinHandle<()>>) {
    // Stream construction can fail after the inference worker starts. Always
    // stop and join it before returning so failed Apply attempts cannot leave a
    // model/CUDA context alive behind the next session.
    running.store(false, Ordering::SeqCst);
    if let Some(worker) = worker.take() {
        let _ = worker.join();
    }
}

fn write_wav(path: &PathBuf, samples: &[f32], sample_rate: u32) -> Result<()> {
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
    use super::*;

    #[test]
    fn live_params_round_trip_through_atomics() {
        let params = LiveParams {
            pitch_shift: -3.5,
            speaker_id: 7,
            input_gain: 0.5,
            output_gain: 2.0,
        };
        let atomic = AtomicLiveParams::new(params);
        let out = atomic.load();
        assert_eq!(out.pitch_shift, params.pitch_shift);
        assert_eq!(out.speaker_id, params.speaker_id);
        assert_eq!(out.input_gain, params.input_gain);
        assert_eq!(out.output_gain, params.output_gain);
    }

    #[test]
    fn validation_requires_models_unless_passthrough() {
        assert!(RealtimeConfig::default().validate().is_err());
        assert!(RealtimeConfig {
            passthrough: true,
            ..Default::default()
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn silent_output_does_not_fill_the_output_ring() {
        assert!(should_queue_silent_output(0, 1_000));
        assert!(should_queue_silent_output(1_000, 1_000));
        assert!(!should_queue_silent_output(1_001, 1_000));
    }
}
