use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

use crate::AudioHost;
use vc_core::dsp;

const CPAL_SCRATCH_FALLBACK_SAMPLES: usize = 65_536;
const CPAL_MAX_SCRATCH_SAMPLES: usize = 65_536;

#[cfg(windows)]
#[path = "wasapi_audio.rs"]
pub(crate) mod wasapi_audio;

// Maps an `AudioHost` to its cpal `Host`. Each arm is gated by the platform (and
// feature) that provides that cpal `HostId`; an unavailable selection (e.g. ASIO
// without the `asio` feature, or CoreAudio on Windows) returns an actionable error
// instead of failing to compile or panicking. WASAPI *exclusive* mode does not go
// through here — it uses the bespoke `wasapi_audio` path — but WASAPI *shared* is a
// normal cpal host.
fn cpal_host(host: AudioHost) -> Result<cpal::Host> {
    let id = match host {
        #[cfg(windows)]
        AudioHost::Wasapi => cpal::HostId::Wasapi,
        #[cfg(all(windows, feature = "asio"))]
        AudioHost::Asio => cpal::HostId::Asio,
        #[cfg(target_os = "macos")]
        AudioHost::CoreAudio => cpal::HostId::CoreAudio,
        #[cfg(target_os = "linux")]
        AudioHost::Alsa => cpal::HostId::Alsa,
        #[cfg(feature = "jack")]
        AudioHost::Jack => cpal::HostId::Jack,
        other => return Err(host_unavailable_error(other)),
    };
    cpal::host_from_id(id).with_context(|| format!("failed to initialize {host:?} host"))
}

fn host_unavailable_error(host: AudioHost) -> anyhow::Error {
    if host == AudioHost::Asio {
        anyhow!("ASIO is unavailable; on Windows rebuild with --features asio")
    } else {
        anyhow!("audio host {host:?} is not available on this platform/build")
    }
}

pub fn print_cpal_devices(host: AudioHost) -> Result<()> {
    let cpal_host = cpal_host(host)?;
    let label = host_label(host, false);

    println!("{label} input devices:");
    for device in cpal_host.input_devices()? {
        println!("  {}", device_name(&device));
    }

    println!();
    println!("{label} output devices:");
    for device in cpal_host.output_devices()? {
        println!("  {}", device_name(&device));
    }

    Ok(())
}

#[cfg(windows)]
pub fn print_wasapi_devices() -> Result<()> {
    wasapi_audio::print_devices()
}

#[cfg(not(windows))]
pub fn print_wasapi_devices() -> Result<()> {
    bail!("WASAPI audio backend is only available on Windows")
}

pub struct RealtimeAudio {
    input_host: AudioHost,
    output_host: AudioHost,
    wasapi_input_exclusive: bool,
    wasapi_output_exclusive: bool,
    input: InputEndpoint,
    output: OutputEndpoint,
    input_sample_rate: u32,
    output_sample_rate: u32,
    input_name: String,
    output_name: String,
}

enum InputEndpoint {
    Cpal {
        device: cpal::Device,
        config: cpal::SupportedStreamConfig,
    },
    #[cfg(windows)]
    Wasapi(wasapi_audio::WasapiStreamConfig),
}

enum OutputEndpoint {
    Cpal {
        device: cpal::Device,
        config: cpal::SupportedStreamConfig,
    },
    #[cfg(windows)]
    Wasapi(wasapi_audio::WasapiStreamConfig),
}

impl RealtimeAudio {
    // Input and output are opened independently so each direction can use a
    // different host (e.g. input WASAPI + output ASIO). The exception is
    // ASIO-on-both, which must share one driver (cpal loads a single ASIO driver
    // globally) and is handled by `open_asio_duplex`.
    pub fn open(
        input_host: AudioHost,
        output_host: AudioHost,
        wasapi_input_exclusive: bool,
        wasapi_output_exclusive: bool,
        input_name: Option<&str>,
        output_name: Option<&str>,
        wasapi_buffer_ms: u32,
    ) -> Result<Self> {
        if wasapi_input_exclusive && input_host != AudioHost::Wasapi {
            bail!("WASAPI exclusive input requires the WASAPI input host");
        }
        if wasapi_output_exclusive && output_host != AudioHost::Wasapi {
            bail!("WASAPI exclusive output requires the WASAPI output host");
        }

        if input_host == AudioHost::Asio && output_host == AudioHost::Asio {
            return Self::open_asio_duplex(input_name, output_name);
        }

        let (input, input_sample_rate, input_name) = open_input_endpoint(
            input_host,
            input_name,
            wasapi_input_exclusive,
            wasapi_buffer_ms,
        )?;
        let (output, output_sample_rate, output_name) = open_output_endpoint(
            output_host,
            output_name,
            wasapi_output_exclusive,
            wasapi_buffer_ms,
        )?;

        Ok(Self {
            input_host,
            output_host,
            wasapi_input_exclusive,
            wasapi_output_exclusive,
            input,
            output,
            input_sample_rate,
            output_sample_rate,
            input_name,
            output_name,
        })
    }

    // Both directions on ASIO: resolve one driver and share it across the input
    // and output endpoints. Names (if given) must select the same driver.
    fn open_asio_duplex(input_name: Option<&str>, output_name: Option<&str>) -> Result<Self> {
        let host = cpal_host(AudioHost::Asio)?;
        let device = input_device(&host, input_name.or(output_name))?;
        let resolved = device_name(&device).to_lowercase();
        let mismatches =
            |name: Option<&str>| name.is_some_and(|name| !resolved.contains(&name.to_lowercase()));
        if mismatches(input_name) || mismatches(output_name) {
            bail!(
                "ASIO uses a single driver for both directions; input '{}' and output '{}' must name the same driver",
                input_name.unwrap_or("<default>"),
                output_name.unwrap_or("<default>"),
            );
        }
        let input_config = default_input_config(&device)?;
        let output_config = default_output_config(&device)?;
        let input_sample_rate = input_config.sample_rate();
        let output_sample_rate = output_config.sample_rate();
        let name = device_name(&device);

        Ok(Self {
            input_host: AudioHost::Asio,
            output_host: AudioHost::Asio,
            wasapi_input_exclusive: false,
            wasapi_output_exclusive: false,
            input: InputEndpoint::Cpal {
                device: device.clone(),
                config: input_config,
            },
            output: OutputEndpoint::Cpal {
                device,
                config: output_config,
            },
            input_sample_rate,
            output_sample_rate,
            input_name: name.clone(),
            output_name: name,
        })
    }

    pub fn input_host_label(&self) -> &'static str {
        host_label(self.input_host, self.wasapi_input_exclusive)
    }

    pub fn output_host_label(&self) -> &'static str {
        host_label(self.output_host, self.wasapi_output_exclusive)
    }

    pub fn input_sample_rate(&self) -> u32 {
        self.input_sample_rate
    }

    pub fn output_sample_rate(&self) -> u32 {
        self.output_sample_rate
    }

    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    pub fn output_name(&self) -> &str {
        &self.output_name
    }

    pub fn build_input_stream<F>(&self, on_samples: F) -> Result<AudioStream>
    where
        F: FnMut(&[f32]) + Send + 'static,
    {
        match &self.input {
            InputEndpoint::Cpal { device, config } => Ok(AudioStream::Cpal(
                build_cpal_input_stream(device, config, on_samples)?,
            )),
            #[cfg(windows)]
            InputEndpoint::Wasapi(config) => Ok(AudioStream::Wasapi(
                wasapi_audio::build_input_stream(config.clone(), on_samples)?,
            )),
        }
    }

    pub fn build_output_stream<F>(&self, fill: F) -> Result<AudioStream>
    where
        F: FnMut(&mut [f32]) + Send + 'static,
    {
        match &self.output {
            OutputEndpoint::Cpal { device, config } => Ok(AudioStream::Cpal(
                build_cpal_output_stream(device, config, fill)?,
            )),
            #[cfg(windows)]
            OutputEndpoint::Wasapi(config) => Ok(AudioStream::Wasapi(
                wasapi_audio::build_output_stream(config.clone(), fill)?,
            )),
        }
    }
}

pub enum AudioStream {
    Cpal(CpalStream),
    #[cfg(windows)]
    Wasapi(wasapi_audio::WasapiStream),
}

impl AudioStream {
    pub fn play(&self) -> Result<()> {
        match self {
            AudioStream::Cpal(stream) => stream
                .stream
                .as_ref()
                .expect("live CPAL stream")
                .play()
                .context("failed to start CPAL stream"),
            #[cfg(windows)]
            AudioStream::Wasapi(stream) => stream.play(),
        }
    }
}

impl AudioStream {
    /// Called only by the session control thread, never the audio callback.
    pub(crate) fn report_errors(&mut self) {
        if let Self::Cpal(stream) = self {
            if stream.last_report.elapsed() >= Duration::from_secs(1) {
                stream.errors.report(stream.direction);
                stream.last_report = Instant::now();
            }
        }
    }
}

const ERROR_KINDS: [cpal::ErrorKind; 14] = [
    cpal::ErrorKind::DeviceBusy,
    cpal::ErrorKind::DeviceChanged,
    cpal::ErrorKind::DeviceNotAvailable,
    cpal::ErrorKind::HostUnavailable,
    cpal::ErrorKind::InvalidInput,
    cpal::ErrorKind::PermissionDenied,
    cpal::ErrorKind::RealtimeDenied,
    cpal::ErrorKind::ResourceExhausted,
    cpal::ErrorKind::StreamInvalidated,
    cpal::ErrorKind::UnsupportedConfig,
    cpal::ErrorKind::UnsupportedOperation,
    cpal::ErrorKind::Xrun,
    cpal::ErrorKind::BackendError,
    cpal::ErrorKind::Other,
];

#[derive(Default)]
struct StreamErrors {
    counts: [AtomicUsize; ERROR_KINDS.len()],
}

impl StreamErrors {
    // CPAL can deliver xruns on its audio thread. Keep recording bounded and
    // free of formatting, allocation, logging and locks. Retain categories and
    // counts instead of owned backend text, so repeated errors cannot grow a
    // queue. Relaxed suffices: these counters publish no other shared data.
    fn record(&self, kind: cpal::ErrorKind) {
        let index = ERROR_KINDS
            .iter()
            .position(|candidate| *candidate == kind)
            .unwrap_or(ERROR_KINDS.len() - 1);
        self.counts[index].fetch_add(1, Ordering::Relaxed);
    }

    fn report(&self, direction: &str) {
        for (kind, count) in ERROR_KINDS.iter().zip(&self.counts) {
            let count = count.swap(0, Ordering::Relaxed);
            if count != 0 {
                tracing::warn!(direction, ?kind, count, "audio stream errors: {kind}");
            }
        }
    }
}

pub struct CpalStream {
    stream: Option<cpal::Stream>,
    errors: Arc<StreamErrors>,
    direction: &'static str,
    last_report: Instant,
}

impl Drop for CpalStream {
    fn drop(&mut self) {
        // Teardown runs on the control thread. Stop callbacks before draining
        // final counts, including errors from short-lived streams.
        drop(self.stream.take());
        self.errors.report(self.direction);
    }
}

// Resolves one direction's endpoint for its host. Every host except WASAPI
// *exclusive* mode goes through the shared cpal stream path (they differ only by
// which cpal host the device comes from); WASAPI exclusive uses the bespoke path
// (until cpal gains exclusive mode). Returns the endpoint, its sample rate, and
// the resolved device name.
fn open_input_endpoint(
    host: AudioHost,
    name: Option<&str>,
    exclusive: bool,
    wasapi_buffer_ms: u32,
) -> Result<(InputEndpoint, u32, String)> {
    if host == AudioHost::Wasapi && exclusive {
        return open_wasapi_input(name, wasapi_buffer_ms);
    }
    let cpal_host = cpal_host(host)?;
    let device = input_device(&cpal_host, name)?;
    let config = default_input_config(&device)?;
    let sample_rate = config.sample_rate();
    let label = device_name(&device);
    Ok((InputEndpoint::Cpal { device, config }, sample_rate, label))
}

fn open_output_endpoint(
    host: AudioHost,
    name: Option<&str>,
    exclusive: bool,
    wasapi_buffer_ms: u32,
) -> Result<(OutputEndpoint, u32, String)> {
    if host == AudioHost::Wasapi && exclusive {
        return open_wasapi_output(name, wasapi_buffer_ms);
    }
    let cpal_host = cpal_host(host)?;
    let device = output_device(&cpal_host, name)?;
    let config = default_output_config(&device)?;
    let sample_rate = config.sample_rate();
    let label = device_name(&device);
    Ok((OutputEndpoint::Cpal { device, config }, sample_rate, label))
}

// The bespoke WASAPI path serves exclusive mode only (shared WASAPI goes through
// cpal), so these always request exclusive.
#[cfg(windows)]
fn open_wasapi_input(
    name: Option<&str>,
    wasapi_buffer_ms: u32,
) -> Result<(InputEndpoint, u32, String)> {
    let config = wasapi_audio::open_input(name, true, wasapi_buffer_ms)?;
    let sample_rate = config.sample_rate;
    let label = config.device_name.clone();
    Ok((InputEndpoint::Wasapi(config), sample_rate, label))
}

#[cfg(not(windows))]
fn open_wasapi_input(
    _name: Option<&str>,
    _wasapi_buffer_ms: u32,
) -> Result<(InputEndpoint, u32, String)> {
    bail!("the WASAPI host is only available on Windows")
}

#[cfg(windows)]
fn open_wasapi_output(
    name: Option<&str>,
    wasapi_buffer_ms: u32,
) -> Result<(OutputEndpoint, u32, String)> {
    let config = wasapi_audio::open_output(name, true, wasapi_buffer_ms)?;
    let sample_rate = config.sample_rate;
    let label = config.device_name.clone();
    Ok((OutputEndpoint::Wasapi(config), sample_rate, label))
}

#[cfg(not(windows))]
fn open_wasapi_output(
    _name: Option<&str>,
    _wasapi_buffer_ms: u32,
) -> Result<(OutputEndpoint, u32, String)> {
    bail!("the WASAPI host is only available on Windows")
}

// Canonical, cpal-aligned token for each host (WASAPI exclusive is annotated).
// The GUI maps these to friendlier labels; the CLI shows them as-is.
fn host_label(host: AudioHost, exclusive: bool) -> &'static str {
    match host {
        AudioHost::Wasapi => {
            if exclusive {
                "wasapi-exclusive"
            } else {
                "wasapi"
            }
        }
        AudioHost::Asio => "asio",
        AudioHost::CoreAudio => "coreaudio",
        AudioHost::Alsa => "alsa",
        AudioHost::Jack => "jack",
    }
}

pub fn input_device(host: &cpal::Host, name: Option<&str>) -> Result<cpal::Device> {
    find_device(host.input_devices()?, name)
        .or_else(|| host.default_input_device())
        .ok_or_else(|| anyhow!("input device not found"))
}

pub fn output_device(host: &cpal::Host, name: Option<&str>) -> Result<cpal::Device> {
    find_device(host.output_devices()?, name)
        .or_else(|| host.default_output_device())
        .ok_or_else(|| anyhow!("output device not found"))
}

pub fn cpal_input_names(host: AudioHost) -> Result<Vec<String>> {
    let cpal_host = cpal_host(host)?;
    Ok(cpal_host
        .input_devices()?
        .map(|d| device_name(&d))
        .collect())
}

pub fn cpal_output_names(host: AudioHost) -> Result<Vec<String>> {
    let cpal_host = cpal_host(host)?;
    Ok(cpal_host
        .output_devices()?
        .map(|d| device_name(&d))
        .collect())
}

fn find_device<I>(devices: I, name: Option<&str>) -> Option<cpal::Device>
where
    I: Iterator<Item = cpal::Device>,
{
    let needle = name?.to_lowercase();
    devices
        .filter_map(|device| {
            let device_name = device_name(&device);
            device_name
                .to_lowercase()
                .contains(&needle)
                .then_some(device)
        })
        .next()
}

pub fn device_name(device: &cpal::Device) -> String {
    device
        .description()
        .map(|description| description.name().to_string())
        .unwrap_or_else(|_| "<unknown>".to_string())
}

// The engine works in mono, but the stream must be opened with the device's
// native channel count: WASAPI shared mode only accepts the mix-format channel
// count, and since cpal 0.18 `build_*_stream` enforces that via
// `IsFormatSupported` instead of relying on AUTOCONVERTPCM. Channel
// up/downmixing therefore happens in our callbacks, not in the OS.
pub fn default_input_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
    device
        .default_input_config()
        .context("failed to get default input config")
}

pub fn default_output_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
    device
        .default_output_config()
        .context("failed to get default output config")
}

fn cpal_scratch_frames(config: &cpal::SupportedStreamConfig) -> usize {
    let channels = config.channels().max(1) as usize;
    let frames = match *config.buffer_size() {
        cpal::SupportedBufferSize::Range { max, .. } => max as usize,
        cpal::SupportedBufferSize::Unknown => CPAL_SCRATCH_FALLBACK_SAMPLES,
    };
    // These buffers are moved into CPAL callbacks so sample-format conversion
    // and channel up/downmixing stay allocation-free on the real-time path.
    // Interleaved scratch is frames * channels, so cap the total accordingly.
    frames.clamp(1, (CPAL_MAX_SCRATCH_SAMPLES / channels).max(1))
}

// FormatMessageW cannot resolve AUDCLNT_E_* HRESULTs, so cpal stream errors
// reach the GUI status line as a bare "OS Error -2004287478". Translate the
// codes users actually hit into something actionable. The decimal codes are
// matched against the error text because cpal does not expose the raw HRESULT;
// if the formatting ever changes the hint silently disappears, nothing breaks.
fn with_audclnt_hint<E>(err: E) -> anyhow::Error
where
    E: std::error::Error + Send + Sync + 'static,
{
    let text = err.to_string();
    let hint = [
        // 0x8889000A AUDCLNT_E_DEVICE_IN_USE
        (
            "-2004287478",
            "audio device is in use in exclusive mode by another application",
        ),
        // 0x88890004 AUDCLNT_E_DEVICE_INVALIDATED
        (
            "-2004287484",
            "audio device was removed or its configuration changed",
        ),
        // 0x88890008 AUDCLNT_E_UNSUPPORTED_FORMAT
        ("-2004287480", "audio device rejected the stream format"),
    ]
    .iter()
    .find_map(|(code, hint)| text.contains(code).then_some(*hint));
    match hint {
        Some(hint) => anyhow::Error::new(err).context(hint),
        None => anyhow::Error::new(err),
    }
}

fn build_cpal_input_stream<F>(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    mut on_samples: F,
) -> Result<CpalStream>
where
    F: FnMut(&[f32]) + Send + 'static,
{
    let stream_config: cpal::StreamConfig = (*config).into();
    // CPAL guarantees `data.len()` is a multiple of the channel count, and the
    // chunk size below is too, so every chunk holds whole frames.
    let channels = config.channels().max(1) as usize;
    let frames = cpal_scratch_frames(config);
    let errors = Arc::new(StreamErrors::default());
    let callback_errors = Arc::clone(&errors);
    let err_fn = move |err: cpal::Error| callback_errors.record(err.kind());
    match config.sample_format() {
        cpal::SampleFormat::F32 if channels == 1 => device.build_input_stream(
            stream_config,
            move |data: &[f32], _| on_samples(data),
            err_fn,
            None,
        ),
        cpal::SampleFormat::F32 => {
            let mut mono = vec![0.0; frames];
            device.build_input_stream(
                stream_config,
                move |data: &[f32], _| {
                    for input in data.chunks(frames * channels) {
                        let mono = &mut mono[..input.len() / channels];
                        dsp::downmix_to_mono_into(input, channels, mono);
                        on_samples(mono);
                    }
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::I16 => {
            let mut interleaved = vec![0.0; frames * channels];
            let mut mono = vec![0.0; frames];
            device.build_input_stream(
                stream_config,
                move |data: &[i16], _| {
                    for input in data.chunks(frames * channels) {
                        let converted = &mut interleaved[..input.len()];
                        dsp::i16_to_f32_into(input, converted);
                        let mono = &mut mono[..input.len() / channels];
                        dsp::downmix_to_mono_into(converted, channels, mono);
                        on_samples(mono);
                    }
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            let mut interleaved = vec![0.0; frames * channels];
            let mut mono = vec![0.0; frames];
            device.build_input_stream(
                stream_config,
                move |data: &[u16], _| {
                    for input in data.chunks(frames * channels) {
                        let converted = &mut interleaved[..input.len()];
                        dsp::u16_to_f32_into(input, converted);
                        let mono = &mut mono[..input.len() / channels];
                        dsp::downmix_to_mono_into(converted, channels, mono);
                        on_samples(mono);
                    }
                },
                err_fn,
                None,
            )
        }
        // 32-bit PCM, common on ASIO drivers.
        cpal::SampleFormat::I32 => {
            let mut interleaved = vec![0.0; frames * channels];
            let mut mono = vec![0.0; frames];
            device.build_input_stream(
                stream_config,
                move |data: &[i32], _| {
                    for input in data.chunks(frames * channels) {
                        let converted = &mut interleaved[..input.len()];
                        dsp::i32_to_f32_into(input, converted);
                        let mono = &mut mono[..input.len() / channels];
                        dsp::downmix_to_mono_into(converted, channels, mono);
                        on_samples(mono);
                    }
                },
                err_fn,
                None,
            )
        }
        sample_format => {
            return Err(anyhow!(
                "unsupported input sample format: {sample_format:?}"
            ))
        }
    }
    .map_err(with_audclnt_hint)
    .context("failed to build input stream")
    .map(|stream| CpalStream {
        stream: Some(stream),
        errors,
        direction: "input",
        last_report: Instant::now(),
    })
}

fn build_cpal_output_stream<F>(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    mut fill: F,
) -> Result<CpalStream>
where
    F: FnMut(&mut [f32]) + Send + 'static,
{
    let stream_config: cpal::StreamConfig = (*config).into();
    // Same framing guarantee as the input path: chunks hold whole frames.
    let channels = config.channels().max(1) as usize;
    let frames = cpal_scratch_frames(config);
    let errors = Arc::new(StreamErrors::default());
    let callback_errors = Arc::clone(&errors);
    let err_fn = move |err: cpal::Error| callback_errors.record(err.kind());
    match config.sample_format() {
        cpal::SampleFormat::F32 if channels == 1 => device.build_output_stream(
            stream_config,
            move |data: &mut [f32], _| fill(data),
            err_fn,
            None,
        ),
        cpal::SampleFormat::F32 => {
            let mut mono = vec![0.0; frames];
            device.build_output_stream(
                stream_config,
                move |data: &mut [f32], _| {
                    for output in data.chunks_mut(frames * channels) {
                        let mono = &mut mono[..output.len() / channels];
                        fill(mono);
                        dsp::upmix_mono_into(mono, channels, output);
                    }
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::I16 => {
            let mut mono = vec![0.0; frames];
            let mut converted = vec![0_i16; frames];
            device.build_output_stream(
                stream_config,
                move |data: &mut [i16], _| {
                    for output in data.chunks_mut(frames * channels) {
                        let frames_in_chunk = output.len() / channels;
                        let mono = &mut mono[..frames_in_chunk];
                        fill(mono);
                        let converted = &mut converted[..frames_in_chunk];
                        dsp::f32_to_i16_into(mono, converted);
                        dsp::upmix_mono_into(converted, channels, output);
                    }
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            let mut mono = vec![0.0; frames];
            let mut converted = vec![0_u16; frames];
            device.build_output_stream(
                stream_config,
                move |data: &mut [u16], _| {
                    for output in data.chunks_mut(frames * channels) {
                        let frames_in_chunk = output.len() / channels;
                        let mono = &mut mono[..frames_in_chunk];
                        fill(mono);
                        let converted = &mut converted[..frames_in_chunk];
                        dsp::f32_to_u16_into(mono, converted);
                        dsp::upmix_mono_into(converted, channels, output);
                    }
                },
                err_fn,
                None,
            )
        }
        // 32-bit PCM, common on ASIO drivers.
        cpal::SampleFormat::I32 => {
            let mut mono = vec![0.0; frames];
            let mut converted = vec![0_i32; frames];
            device.build_output_stream(
                stream_config,
                move |data: &mut [i32], _| {
                    for output in data.chunks_mut(frames * channels) {
                        let frames_in_chunk = output.len() / channels;
                        let mono = &mut mono[..frames_in_chunk];
                        fill(mono);
                        let converted = &mut converted[..frames_in_chunk];
                        dsp::f32_to_i32_into(mono, converted);
                        dsp::upmix_mono_into(converted, channels, output);
                    }
                },
                err_fn,
                None,
            )
        }
        sample_format => {
            return Err(anyhow!(
                "unsupported output sample format: {sample_format:?}"
            ))
        }
    }
    .map_err(with_audclnt_hint)
    .context("failed to build output stream")
    .map(|stream| CpalStream {
        stream: Some(stream),
        errors,
        direction: "output",
        last_report: Instant::now(),
    })
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn concurrent_error_reporting_preserves_counts() {
        let errors = Arc::new(StreamErrors::default());
        let producer = Arc::clone(&errors);
        let worker = std::thread::spawn(move || {
            for _ in 0..10_000 {
                producer.record(cpal::ErrorKind::Xrun);
                producer.record(cpal::ErrorKind::DeviceNotAvailable);
            }
        });
        let mut totals = [0; ERROR_KINDS.len()];
        while !worker.is_finished() {
            for (total, count) in totals.iter_mut().zip(&errors.counts) {
                *total += count.swap(0, Ordering::Relaxed);
            }
            std::thread::yield_now();
        }
        worker.join().unwrap();
        for (total, count) in totals.iter_mut().zip(&errors.counts) {
            *total += count.swap(0, Ordering::Relaxed);
        }
        for (kind, total) in ERROR_KINDS.iter().zip(totals) {
            let expected = if matches!(
                kind,
                cpal::ErrorKind::Xrun | cpal::ErrorKind::DeviceNotAvailable
            ) {
                10_000
            } else {
                0
            };
            assert_eq!(total, expected, "{kind:?}");
        }
        assert!(errors
            .counts
            .iter()
            .all(|count| count.load(Ordering::Relaxed) == 0));
    }
}
