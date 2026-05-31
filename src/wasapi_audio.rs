use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use thread_priority::{set_current_thread_priority, ThreadPriority};
use tracing::{debug, warn};
use wasapi::{
    calculate_period_100ns, initialize_mta, initialize_sta, AudioClient, Device, DeviceEnumerator,
    Direction, SampleType, StreamMode, WaveFormat,
};

const EVENT_WAIT_TIMEOUT_MS: u32 = 100;
const AUDIO_CLIENT_INIT_RETRY_DELAYS_MS: [u64; 5] = [25, 50, 100, 200, 400];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasapiSampleFormat {
    F32,
    I16,
}

#[derive(Clone, Debug)]
pub struct WasapiStreamConfig {
    pub device_id: String,
    pub device_name: String,
    pub wave_format: WaveFormat,
    pub sample_rate: u32,
    pub channels: usize,
    pub sample_format: WasapiSampleFormat,
    pub exclusive: bool,
    period_hns: i64,
    buffer_duration_hns: i64,
}

pub struct WasapiEndpoints {
    pub input: WasapiStreamConfig,
    pub output: WasapiStreamConfig,
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
}

pub struct WasapiStream {
    name: &'static str,
    running: Arc<AtomicBool>,
    started: Arc<AtomicBool>,
    start_result: Mutex<Option<mpsc::Receiver<Result<()>>>>,
    handle: Option<JoinHandle<Result<()>>>,
}

impl WasapiStream {
    pub fn play(&self) -> Result<()> {
        if !self.running.load(Ordering::SeqCst) {
            bail!("WASAPI {} stream is already stopped", self.name);
        }

        let start_result = self
            .start_result
            .lock()
            .map_err(|_| anyhow!("WASAPI {} start result lock poisoned", self.name))?
            .take();
        self.started.store(true, Ordering::SeqCst);
        if let Some(start_result) = start_result {
            start_result
                .recv()
                .with_context(|| format!("WASAPI {} stream exited before start", self.name))?
        } else {
            Ok(())
        }
    }
}

impl Drop for WasapiStream {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        self.started.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => warn!("WASAPI {} stream stopped with error: {err:#}", self.name),
                Err(_) => warn!("WASAPI {} stream thread panicked", self.name),
            }
        }
    }
}

pub fn print_devices() -> Result<()> {
    let _com = ComGuard::initialize()?;
    let enumerator =
        DeviceEnumerator::new().context("failed to create WASAPI device enumerator")?;

    println!("WASAPI input devices:");
    print_devices_for_direction(&enumerator, Direction::Capture)?;

    println!();
    println!("WASAPI output devices:");
    print_devices_for_direction(&enumerator, Direction::Render)?;

    Ok(())
}

pub fn open_realtime(
    input_name: Option<&str>,
    output_name: Option<&str>,
    input_exclusive: bool,
    output_exclusive: bool,
    wasapi_buffer_ms: u32,
) -> Result<WasapiEndpoints> {
    let _com = ComGuard::initialize()?;
    let enumerator =
        DeviceEnumerator::new().context("failed to create WASAPI device enumerator")?;

    let input_device = find_device(&enumerator, Direction::Capture, input_name)?;
    let output_device = find_device(&enumerator, Direction::Render, output_name)?;
    let input_summary = summarize_device(&input_device)?;
    let output_summary = summarize_device(&output_device)?;

    let input_base_format = device_format(&input_device, Direction::Capture)?;
    let input_sample_rate = input_base_format.get_samplespersec();
    if input_sample_rate == 0 {
        bail!("WASAPI input device reported an invalid sample rate");
    }

    let input_channels = input_base_format.get_nchannels().max(1) as usize;
    let output_base_format = device_format(&output_device, Direction::Render)?;
    let output_sample_rate = output_base_format.get_samplespersec();
    if output_sample_rate == 0 {
        bail!("WASAPI output device reported an invalid sample rate");
    }
    let output_channels = output_base_format.get_nchannels().max(1) as usize;

    let input = select_stream_config(
        &input_device,
        input_summary,
        StreamConfigSelection {
            sample_rate: input_sample_rate,
            channels: input_channels,
            exclusive: input_exclusive,
            wasapi_buffer_ms,
            direction: Direction::Capture,
        },
    )?;
    let output = select_stream_config(
        &output_device,
        output_summary,
        StreamConfigSelection {
            sample_rate: output_sample_rate,
            channels: output_channels,
            exclusive: output_exclusive,
            wasapi_buffer_ms,
            direction: Direction::Render,
        },
    )?;

    let input_sample_rate = input.sample_rate;
    let output_sample_rate = output.sample_rate;
    Ok(WasapiEndpoints {
        input,
        output,
        input_sample_rate,
        output_sample_rate,
    })
}

pub fn build_input_stream<F>(config: WasapiStreamConfig, on_samples: F) -> Result<WasapiStream>
where
    F: FnMut(&[f32]) + Send + 'static,
{
    spawn_stream("input", move |running, started, init_tx, start_tx| {
        run_input_stream(config, running, started, init_tx, start_tx, on_samples)
    })
}

pub fn build_output_stream<F>(config: WasapiStreamConfig, fill: F) -> Result<WasapiStream>
where
    F: FnMut(&mut [f32]) + Send + 'static,
{
    spawn_stream("output", move |running, started, init_tx, start_tx| {
        run_output_stream(config, running, started, init_tx, start_tx, fill)
    })
}

fn print_devices_for_direction(enumerator: &DeviceEnumerator, direction: Direction) -> Result<()> {
    let devices = enumerator
        .get_device_collection(&direction)
        .with_context(|| format!("failed to enumerate WASAPI {direction} devices"))?;
    for index in 0..devices.get_nbr_devices()? {
        let device = devices.get_device_at_index(index)?;
        let summary = summarize_device(&device)?;
        println!(
            "  {} | {} | {}",
            summary.friendly_name, summary.description, summary.id
        );
    }
    Ok(())
}

fn spawn_stream<F>(name: &'static str, run: F) -> Result<WasapiStream>
where
    F: FnOnce(
            Arc<AtomicBool>,
            Arc<AtomicBool>,
            SyncSender<Result<()>>,
            SyncSender<Result<()>>,
        ) -> Result<()>
        + Send
        + 'static,
{
    let running = Arc::new(AtomicBool::new(true));
    let started = Arc::new(AtomicBool::new(false));
    let (init_tx, init_rx) = mpsc::sync_channel(1);
    let (start_tx, start_rx) = mpsc::sync_channel(1);
    let thread_running = Arc::clone(&running);
    let thread_started = Arc::clone(&started);
    let handle = thread::Builder::new()
        .name(format!("wasapi_{name}"))
        .spawn(move || run(thread_running, thread_started, init_tx, start_tx))
        .with_context(|| format!("failed to spawn WASAPI {name} thread"))?;

    let init_result = init_rx
        .recv()
        .with_context(|| format!("WASAPI {name} thread exited before initialization"))?;
    if let Err(err) = init_result {
        running.store(false, Ordering::SeqCst);
        started.store(true, Ordering::SeqCst);
        let _ = handle.join();
        return Err(err);
    }

    Ok(WasapiStream {
        name,
        running,
        started,
        start_result: Mutex::new(Some(start_rx)),
        handle: Some(handle),
    })
}

fn run_input_stream<F>(
    config: WasapiStreamConfig,
    running: Arc<AtomicBool>,
    started: Arc<AtomicBool>,
    init_tx: SyncSender<Result<()>>,
    start_tx: SyncSender<Result<()>>,
    mut on_samples: F,
) -> Result<()>
where
    F: FnMut(&[f32]) + Send + 'static,
{
    raise_current_thread_priority("input");
    let initialized = init_input_stream(&config);
    let stream = match initialized {
        Ok(stream) => {
            let _ = init_tx.send(Ok(()));
            stream
        }
        Err(err) => {
            let _ = init_tx.send(Err(anyhow!("{err:#}")));
            return Err(err);
        }
    };

    wait_until_started(&running, &started);
    if !running.load(Ordering::SeqCst) {
        return Ok(());
    }

    if let Err(err) = stream.audio_client.start_stream().with_context(|| {
        format!(
            "failed to start WASAPI input stream for {}",
            config.device_name
        )
    }) {
        let message = format!("{err:#}");
        let _ = start_tx.send(Err(anyhow!(message.clone())));
        return Err(anyhow!(message));
    }
    let _ = start_tx.send(Ok(()));
    let result = (|| -> Result<()> {
        let bytes_per_frame = config.wave_format.get_blockalign() as usize;
        let buffer_frames = stream.audio_client.get_buffer_size()? as usize;
        let bytes_needed = buffer_frames * bytes_per_frame;
        let mut raw = Vec::<u8>::with_capacity(bytes_needed);
        let mut mono = Vec::<f32>::with_capacity(buffer_frames);

        while running.load(Ordering::SeqCst) {
            while running.load(Ordering::SeqCst) && stream.audio_client.get_current_padding()? > 0 {
                raw.resize(bytes_needed, 0);
                let (frames_read, buffer_info) =
                    stream.capture_client.read_from_device(&mut raw)?;
                if frames_read == 0 {
                    break;
                }

                decode_interleaved_to_mono(
                    &raw[..frames_read as usize * bytes_per_frame],
                    frames_read as usize,
                    config.channels,
                    config.sample_format,
                    buffer_info.flags.silent,
                    &mut mono,
                );
                on_samples(&mono);
            }
            wait_for_stream_event(&stream.event_handle, &running);
        }
        Ok(())
    })();
    stop_stream(&stream.audio_client, "input");
    result
}

fn run_output_stream<F>(
    config: WasapiStreamConfig,
    running: Arc<AtomicBool>,
    started: Arc<AtomicBool>,
    init_tx: SyncSender<Result<()>>,
    start_tx: SyncSender<Result<()>>,
    mut fill: F,
) -> Result<()>
where
    F: FnMut(&mut [f32]) + Send + 'static,
{
    raise_current_thread_priority("output");
    let initialized = init_output_stream(&config);
    let stream = match initialized {
        Ok(stream) => {
            let _ = init_tx.send(Ok(()));
            stream
        }
        Err(err) => {
            let _ = init_tx.send(Err(anyhow!("{err:#}")));
            return Err(err);
        }
    };

    wait_until_started(&running, &started);
    if !running.load(Ordering::SeqCst) {
        return Ok(());
    }

    if let Err(err) = stream.audio_client.start_stream().with_context(|| {
        format!(
            "failed to start WASAPI output stream for {}",
            config.device_name
        )
    }) {
        let message = format!("{err:#}");
        let _ = start_tx.send(Err(anyhow!(message.clone())));
        return Err(anyhow!(message));
    }
    let _ = start_tx.send(Ok(()));
    let result = (|| -> Result<()> {
        let buffer_frames = stream.audio_client.get_buffer_size()? as usize;
        let bytes_per_frame = config.wave_format.get_blockalign() as usize;
        let mut mono = Vec::<f32>::with_capacity(buffer_frames);
        let mut raw = Vec::<u8>::with_capacity(buffer_frames * bytes_per_frame);

        while running.load(Ordering::SeqCst) {
            let frames_available = stream.audio_client.get_available_space_in_frames()?;
            if frames_available > 0 {
                mono.resize(frames_available as usize, 0.0);
                fill(&mut mono);
                encode_mono_to_interleaved(&mono, config.channels, config.sample_format, &mut raw);
                stream
                    .render_client
                    .write_to_device(frames_available as usize, &raw, None)?;
            }
            wait_for_stream_event(&stream.event_handle, &running);
        }
        Ok(())
    })();
    stop_stream(&stream.audio_client, "output");
    result
}

struct ActiveInputStream {
    audio_client: AudioClient,
    capture_client: wasapi::AudioCaptureClient,
    event_handle: wasapi::Handle,
    _com: ComGuard,
}

struct ActiveOutputStream {
    audio_client: AudioClient,
    render_client: wasapi::AudioRenderClient,
    event_handle: wasapi::Handle,
    _com: ComGuard,
}

fn init_input_stream(config: &WasapiStreamConfig) -> Result<ActiveInputStream> {
    let _com = ComGuard::initialize()?;
    let audio_client = initialized_client(config, Direction::Capture)?;
    let event_handle = audio_client.set_get_eventhandle().with_context(|| {
        format!(
            "failed to create WASAPI event handle for {}",
            config.device_name
        )
    })?;
    let capture_client = audio_client.get_audiocaptureclient().with_context(|| {
        format!(
            "failed to create WASAPI capture client for {}",
            config.device_name
        )
    })?;
    Ok(ActiveInputStream {
        audio_client,
        capture_client,
        event_handle,
        _com,
    })
}

fn init_output_stream(config: &WasapiStreamConfig) -> Result<ActiveOutputStream> {
    let _com = ComGuard::initialize()?;
    let audio_client = initialized_client(config, Direction::Render)?;
    let event_handle = audio_client.set_get_eventhandle().with_context(|| {
        format!(
            "failed to create WASAPI event handle for {}",
            config.device_name
        )
    })?;
    let render_client = audio_client.get_audiorenderclient().with_context(|| {
        format!(
            "failed to create WASAPI render client for {}",
            config.device_name
        )
    })?;
    Ok(ActiveOutputStream {
        audio_client,
        render_client,
        event_handle,
        _com,
    })
}

fn initialized_client(config: &WasapiStreamConfig, direction: Direction) -> Result<AudioClient> {
    // Some USB/Bluetooth endpoints, especially headset "Chat" devices, can
    // briefly reject IMMDevice activation right after the previous stream was
    // stopped. Keep this retry bounded and initialization-only; the audio loop
    // must remain event-driven and must not fall back to sleep polling.
    for (attempt, retry_delay_ms) in AUDIO_CLIENT_INIT_RETRY_DELAYS_MS
        .iter()
        .copied()
        .map(Some)
        .chain(std::iter::once(None))
        .enumerate()
    {
        match initialized_client_once(config, direction) {
            Ok(audio_client) => return Ok(audio_client),
            Err(err) => {
                if let Some(retry_delay_ms) = retry_delay_ms {
                    debug!(
                        "retrying WASAPI {direction} stream initialization for {} in {retry_delay_ms}ms after attempt {} failed: {err:#}",
                        config.device_name,
                        attempt + 1
                    );
                    thread::sleep(Duration::from_millis(retry_delay_ms));
                } else {
                    return Err(err);
                }
            }
        }
    }
    unreachable!("WASAPI initialization retry loop always returns");
}

fn initialized_client_once(
    config: &WasapiStreamConfig,
    direction: Direction,
) -> Result<AudioClient> {
    let enumerator =
        DeviceEnumerator::new().context("failed to create WASAPI device enumerator")?;
    let device = enumerator
        .get_device(&config.device_id)
        .with_context(|| format!("failed to reopen WASAPI device {}", config.device_name))?;
    let mut audio_client = device.get_iaudioclient().with_context(|| {
        format!(
            "failed to create WASAPI audio client for {}",
            config.device_name
        )
    })?;
    let stream_mode = stream_mode(config);
    audio_client
        .initialize_client(&config.wave_format, &direction, &stream_mode)
        .with_context(|| {
            format!(
                "failed to initialize WASAPI {} stream for {} in {} mode",
                direction,
                config.device_name,
                if config.exclusive {
                    "exclusive"
                } else {
                    "shared"
                }
            )
        })?;
    Ok(audio_client)
}

fn stop_stream(audio_client: &AudioClient, name: &str) {
    if let Err(err) = audio_client.stop_stream() {
        warn!("failed to stop WASAPI {name} stream: {err}");
        return;
    }
    // Reset is shutdown-only. It clears endpoint buffers before COM objects are
    // dropped, which makes rapid stop/start cycles less dependent on driver
    // teardown timing without adding work to the real-time event loop.
    if let Err(err) = audio_client.reset_stream() {
        warn!("failed to reset WASAPI {name} stream: {err}");
    }
}

fn raise_current_thread_priority(name: &str) {
    if let Err(err) = set_current_thread_priority(ThreadPriority::Max) {
        warn!("failed to set WASAPI {name} thread priority: {err}");
    }
}

fn wait_for_stream_event(event_handle: &wasapi::Handle, running: &AtomicBool) {
    // Keep the timeout finite so Drop can join promptly without having to signal
    // the WASAPI event handle. Timeouts are expected during shutdown and should
    // not be logged from the audio thread.
    if running.load(Ordering::SeqCst) {
        let _ = event_handle.wait_for_event(EVENT_WAIT_TIMEOUT_MS);
    }
}

fn wait_until_started(running: &AtomicBool, started: &AtomicBool) {
    while running.load(Ordering::SeqCst) && !started.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(1));
    }
}

fn stream_mode(config: &WasapiStreamConfig) -> StreamMode {
    // This backend is intentionally event-only. Exclusive event mode requires
    // buffer duration to equal the period, so only the aligned period is passed.
    if config.exclusive {
        StreamMode::EventsExclusive {
            period_hns: config.period_hns,
        }
    } else {
        StreamMode::EventsShared {
            autoconvert: true,
            buffer_duration_hns: config.buffer_duration_hns,
        }
    }
}

fn find_device(
    enumerator: &DeviceEnumerator,
    direction: Direction,
    name: Option<&str>,
) -> Result<Device> {
    if let Some(name) = name {
        let needle = name.to_lowercase();
        let devices = enumerator
            .get_device_collection(&direction)
            .with_context(|| format!("failed to enumerate WASAPI {direction} devices"))?;
        for index in 0..devices.get_nbr_devices()? {
            let device = devices.get_device_at_index(index)?;
            let summary = summarize_device(&device)?;
            if summary.matches(&needle) {
                return Ok(device);
            }
        }
        bail!("WASAPI {direction} device not found: {name}");
    }

    enumerator
        .get_default_device(&direction)
        .with_context(|| format!("failed to get default WASAPI {direction} device"))
}

#[derive(Debug)]
struct DeviceSummary {
    id: String,
    friendly_name: String,
    description: String,
    interface_name: String,
}

impl DeviceSummary {
    fn matches(&self, needle: &str) -> bool {
        self.id.to_lowercase().contains(needle)
            || self.friendly_name.to_lowercase().contains(needle)
            || self.description.to_lowercase().contains(needle)
            || self.interface_name.to_lowercase().contains(needle)
    }
}

fn summarize_device(device: &Device) -> Result<DeviceSummary> {
    let id = device.get_id().context("failed to read WASAPI device id")?;
    let friendly_name = device
        .get_friendlyname()
        .unwrap_or_else(|_| "<unknown>".to_string());
    let description = device
        .get_description()
        .unwrap_or_else(|_| "<unknown>".to_string());
    let interface_name = device
        .get_interface_friendlyname()
        .unwrap_or_else(|_| "<unknown>".to_string());

    Ok(DeviceSummary {
        id,
        friendly_name,
        description,
        interface_name,
    })
}

fn device_format(device: &Device, direction: Direction) -> Result<WaveFormat> {
    match device.get_device_format() {
        Ok(format) => Ok(format),
        Err(device_format_err) => {
            debug!(
                "failed to read WASAPI device format for {direction}: {device_format_err}; falling back to mix format"
            );
            let client = device
                .get_iaudioclient()
                .with_context(|| format!("failed to create WASAPI {direction} audio client"))?;
            client
                .get_mixformat()
                .with_context(|| format!("failed to read WASAPI {direction} mix format"))
        }
    }
}

struct StreamConfigSelection {
    sample_rate: u32,
    channels: usize,
    exclusive: bool,
    wasapi_buffer_ms: u32,
    direction: Direction,
}

fn select_stream_config(
    device: &Device,
    summary: DeviceSummary,
    selection: StreamConfigSelection,
) -> Result<WasapiStreamConfig> {
    let StreamConfigSelection {
        sample_rate,
        channels,
        exclusive,
        wasapi_buffer_ms,
        direction,
    } = selection;
    let audio_client = device
        .get_iaudioclient()
        .with_context(|| format!("failed to create WASAPI {direction} audio client"))?;
    let (wave_format, sample_format) =
        choose_format(&audio_client, sample_rate, channels, exclusive, direction)?;
    let sample_rate = wave_format.get_samplespersec();
    let channels = wave_format.get_nchannels().max(1) as usize;
    let (period_hns, buffer_duration_hns) = stream_periods(
        &audio_client,
        &wave_format,
        wasapi_buffer_ms,
        sample_rate,
        exclusive,
    )?;

    Ok(WasapiStreamConfig {
        device_id: summary.id,
        device_name: summary.friendly_name,
        wave_format,
        sample_rate,
        channels,
        sample_format,
        exclusive,
        period_hns,
        buffer_duration_hns,
    })
}

fn choose_format(
    audio_client: &AudioClient,
    sample_rate: u32,
    channels: usize,
    exclusive: bool,
    direction: Direction,
) -> Result<(WaveFormat, WasapiSampleFormat)> {
    let float_format = WaveFormat::new(
        32,
        32,
        &SampleType::Float,
        sample_rate as usize,
        channels,
        None,
    );
    if !exclusive {
        return Ok((float_format, WasapiSampleFormat::F32));
    }

    match audio_client.is_supported_exclusive_with_quirks(&float_format) {
        Ok(format) => Ok((format, WasapiSampleFormat::F32)),
        Err(float_err) => {
            let int_format = WaveFormat::new(
                16,
                16,
                &SampleType::Int,
                sample_rate as usize,
                channels,
                None,
            );
            match audio_client.is_supported_exclusive_with_quirks(&int_format) {
                Ok(format) => Ok((format, WasapiSampleFormat::I16)),
                Err(int_err) => Err(anyhow!(
                    "WASAPI exclusive {direction} format unsupported for {sample_rate} Hz, {channels} channels (float32: {float_err}; int16: {int_err})"
                )),
            }
        }
    }
}

fn stream_periods(
    audio_client: &AudioClient,
    wave_format: &WaveFormat,
    wasapi_buffer_ms: u32,
    sample_rate: u32,
    exclusive: bool,
) -> Result<(i64, i64)> {
    let (_default_period, min_period) = audio_client.get_device_period()?;
    debug!(
        "WASAPI device default period: {} ms, minimum period: {} ms",
        _default_period / 10000,
        min_period / 10000
    );
    let desired_period = if wasapi_buffer_ms == 0 || wasapi_buffer_ms < (min_period / 10000) as u32
    {
        min_period
    } else {
        let desired_frames = ((sample_rate as u64 * wasapi_buffer_ms as u64) / 1000).max(1) as i64;
        calculate_period_100ns(desired_frames, sample_rate as i64)
    };
    let period = desired_period.max(min_period);
    let period = if exclusive {
        audio_client.calculate_aligned_period_near(period, None, wave_format)?
    } else {
        period
    };
    // In event-exclusive mode WASAPI requires hnsBufferDuration == hnsPeriodicity;
    // keep the config fields equal so future polling-era multipliers do not leak back in.
    debug!("WASAPI event period/buffer duration: {} ms", period / 10000);
    Ok((period, period))
}

struct ComGuard;

impl ComGuard {
    fn initialize() -> Result<Self> {
        match initialize_mta().ok() {
            Ok(()) => {}
            Err(mta_err) => {
                initialize_sta().ok().with_context(|| {
                    format!("failed to initialize COM for WASAPI as MTA ({mta_err}) or STA")
                })?;
            }
        }
        Ok(Self)
    }
}

impl Drop for ComGuard {
    fn drop(&mut self) {
        wasapi::deinitialize();
    }
}

pub(crate) fn decode_interleaved_to_mono(
    input: &[u8],
    frames: usize,
    channels: usize,
    sample_format: WasapiSampleFormat,
    silent: bool,
    output: &mut Vec<f32>,
) {
    output.clear();
    output.resize(frames, 0.0);
    if silent || channels == 0 {
        return;
    }

    match sample_format {
        WasapiSampleFormat::F32 => {
            for (frame, out_sample) in output.iter_mut().enumerate().take(frames) {
                let mut sum = 0.0;
                for channel in 0..channels {
                    let offset = (frame * channels + channel) * 4;
                    if offset + 4 <= input.len() {
                        sum += f32::from_le_bytes([
                            input[offset],
                            input[offset + 1],
                            input[offset + 2],
                            input[offset + 3],
                        ]);
                    }
                }
                *out_sample = sum / channels as f32;
            }
        }
        WasapiSampleFormat::I16 => {
            for (frame, out_sample) in output.iter_mut().enumerate().take(frames) {
                let mut sum = 0.0;
                for channel in 0..channels {
                    let offset = (frame * channels + channel) * 2;
                    if offset + 2 <= input.len() {
                        let sample = i16::from_le_bytes([input[offset], input[offset + 1]]);
                        sum += sample as f32 / 32768.0;
                    }
                }
                *out_sample = sum / channels as f32;
            }
        }
    }
}

pub(crate) fn encode_mono_to_interleaved(
    input: &[f32],
    channels: usize,
    sample_format: WasapiSampleFormat,
    output: &mut Vec<u8>,
) {
    output.clear();
    if channels == 0 {
        return;
    }

    match sample_format {
        WasapiSampleFormat::F32 => {
            output.resize(input.len() * channels * 4, 0);
            for (frame, sample) in input.iter().copied().enumerate() {
                let bytes = sample.clamp(-1.0, 1.0).to_le_bytes();
                for channel in 0..channels {
                    let offset = (frame * channels + channel) * 4;
                    output[offset..offset + 4].copy_from_slice(&bytes);
                }
            }
        }
        WasapiSampleFormat::I16 => {
            output.resize(input.len() * channels * 2, 0);
            for (frame, sample) in input.iter().copied().enumerate() {
                let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
                let bytes = value.to_le_bytes();
                for channel in 0..channels {
                    let offset = (frame * channels + channel) * 2;
                    output[offset..offset + 2].copy_from_slice(&bytes);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn test_config(exclusive: bool) -> WasapiStreamConfig {
        WasapiStreamConfig {
            device_id: "test-device".to_string(),
            device_name: "test device".to_string(),
            wave_format: WaveFormat::new(32, 32, &SampleType::Float, 48_000, 2, None),
            sample_rate: 48_000,
            channels: 2,
            sample_format: WasapiSampleFormat::F32,
            exclusive,
            period_hns: 30_000,
            buffer_duration_hns: 40_000,
        }
    }

    #[test]
    fn shared_stream_config_uses_event_timing() {
        match stream_mode(&test_config(false)) {
            StreamMode::EventsShared {
                autoconvert,
                buffer_duration_hns,
            } => {
                assert!(autoconvert);
                assert_eq!(buffer_duration_hns, 40_000);
            }
            mode => panic!("expected shared event stream mode, got {mode:?}"),
        }
    }

    #[test]
    fn exclusive_stream_config_uses_event_period() {
        match stream_mode(&test_config(true)) {
            StreamMode::EventsExclusive { period_hns } => assert_eq!(period_hns, 30_000),
            mode => panic!("expected exclusive event stream mode, got {mode:?}"),
        }
    }

    #[test]
    fn decodes_stereo_f32_to_mono() {
        let mut input = Vec::new();
        for sample in [0.5f32, -0.25, 0.25, 0.75] {
            input.extend_from_slice(&sample.to_le_bytes());
        }

        let mut output = Vec::new();
        decode_interleaved_to_mono(&input, 2, 2, WasapiSampleFormat::F32, false, &mut output);

        assert_abs_diff_eq!(output[0], 0.125, epsilon = 0.000001);
        assert_abs_diff_eq!(output[1], 0.5, epsilon = 0.000001);
    }

    #[test]
    fn decodes_silent_buffer_to_zeroes() {
        let input = [255u8; 16];
        let mut output = Vec::new();
        decode_interleaved_to_mono(&input, 4, 2, WasapiSampleFormat::I16, true, &mut output);

        assert_eq!(output, vec![0.0; 4]);
    }

    #[test]
    fn encodes_mono_f32_to_all_channels() {
        let mut output = Vec::new();
        encode_mono_to_interleaved(&[0.25, -0.5], 2, WasapiSampleFormat::F32, &mut output);

        let samples: Vec<f32> = output
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        assert_eq!(samples, vec![0.25, 0.25, -0.5, -0.5]);
    }

    #[test]
    fn encodes_i16_with_clipping() {
        let mut output = Vec::new();
        encode_mono_to_interleaved(&[-2.0, 0.0, 2.0], 1, WasapiSampleFormat::I16, &mut output);

        let samples: Vec<i16> = output
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        assert_eq!(samples, vec![-32767, 0, 32767]);
    }
}
