use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::cli::{AudioBackend, DeviceAudioBackend};
use crate::dsp;

const CPAL_SCRATCH_FALLBACK_SAMPLES: usize = 65_536;
const CPAL_MAX_SCRATCH_SAMPLES: usize = 65_536;

#[cfg(windows)]
#[path = "wasapi_audio.rs"]
mod wasapi_audio;

pub fn print_devices(backend: DeviceAudioBackend) -> Result<()> {
    match backend {
        DeviceAudioBackend::All => {
            print_cpal_devices()?;
            println!();
            print_wasapi_devices()
        }
        DeviceAudioBackend::Cpal => print_cpal_devices(),
        DeviceAudioBackend::Wasapi => print_wasapi_devices(),
    }
}

fn print_cpal_devices() -> Result<()> {
    let host = cpal::default_host();

    println!("CPAL input devices:");
    for device in host.input_devices()? {
        println!("  {}", device_name(&device));
    }

    println!();
    println!("CPAL output devices:");
    for device in host.output_devices()? {
        println!("  {}", device_name(&device));
    }

    Ok(())
}

#[cfg(windows)]
fn print_wasapi_devices() -> Result<()> {
    wasapi_audio::print_devices()
}

#[cfg(not(windows))]
fn print_wasapi_devices() -> Result<()> {
    bail!("WASAPI audio backend is only available on Windows")
}

pub struct RealtimeAudio {
    backend: AudioBackend,
    wasapi_input_exclusive: bool,
    wasapi_output_exclusive: bool,
    input: InputEndpoint,
    output: OutputEndpoint,
    sample_rate: u32,
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
    pub fn open(
        backend: AudioBackend,
        wasapi_input_exclusive: bool,
        wasapi_output_exclusive: bool,
        input_name: Option<&str>,
        output_name: Option<&str>,
        wasapi_buffer_ms: u32,
    ) -> Result<Self> {
        if (wasapi_input_exclusive || wasapi_output_exclusive) && backend != AudioBackend::Wasapi {
            bail!("--wasapi-exclusive* options require --audio-backend wasapi");
        }

        match backend {
            AudioBackend::Cpal => Self::open_cpal(input_name, output_name),
            AudioBackend::Wasapi => Self::open_wasapi(
                input_name,
                output_name,
                wasapi_input_exclusive,
                wasapi_output_exclusive,
                wasapi_buffer_ms,
            ),
        }
    }

    fn open_cpal(input_name: Option<&str>, output_name: Option<&str>) -> Result<Self> {
        let input_device = input_device(input_name)?;
        let input_config = mono_input_config(&input_device)?;
        let output_device = output_device(output_name)?;
        let output_config = mono_output_config(&output_device, input_config.sample_rate())?;
        let sample_rate = input_config.sample_rate();
        let input_name = device_name(&input_device);
        let output_name = device_name(&output_device);

        Ok(Self {
            backend: AudioBackend::Cpal,
            wasapi_input_exclusive: false,
            wasapi_output_exclusive: false,
            input: InputEndpoint::Cpal {
                device: input_device,
                config: input_config,
            },
            output: OutputEndpoint::Cpal {
                device: output_device,
                config: output_config,
            },
            sample_rate,
            input_name,
            output_name,
        })
    }

    #[cfg(windows)]
    fn open_wasapi(
        input_name: Option<&str>,
        output_name: Option<&str>,
        wasapi_input_exclusive: bool,
        wasapi_output_exclusive: bool,
        wasapi_buffer_ms: u32,
    ) -> Result<Self> {
        let endpoints = wasapi_audio::open_realtime(
            input_name,
            output_name,
            wasapi_input_exclusive,
            wasapi_output_exclusive,
            wasapi_buffer_ms,
        )?;
        let input_name = endpoints.input.device_name.clone();
        let output_name = endpoints.output.device_name.clone();
        let sample_rate = endpoints.sample_rate;

        Ok(Self {
            backend: AudioBackend::Wasapi,
            wasapi_input_exclusive,
            wasapi_output_exclusive,
            input: InputEndpoint::Wasapi(endpoints.input),
            output: OutputEndpoint::Wasapi(endpoints.output),
            sample_rate,
            input_name,
            output_name,
        })
    }

    #[cfg(not(windows))]
    fn open_wasapi(
        _input_name: Option<&str>,
        _output_name: Option<&str>,
        _wasapi_input_exclusive: bool,
        _wasapi_output_exclusive: bool,
        _wasapi_buffer_ms: u32,
    ) -> Result<Self> {
        bail!("WASAPI audio backend is only available on Windows")
    }

    pub fn backend_label(&self) -> &'static str {
        match self.backend {
            AudioBackend::Cpal => "cpal",
            AudioBackend::Wasapi => {
                match (self.wasapi_input_exclusive, self.wasapi_output_exclusive) {
                    (true, true) => "wasapi-exclusive",
                    (true, false) => "wasapi-input-exclusive",
                    (false, true) => "wasapi-output-exclusive",
                    (false, false) => "wasapi-shared",
                }
            }
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
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
    Cpal(cpal::Stream),
    #[cfg(windows)]
    Wasapi(wasapi_audio::WasapiStream),
}

impl AudioStream {
    pub fn play(&self) -> Result<()> {
        match self {
            AudioStream::Cpal(stream) => stream.play().context("failed to start CPAL stream"),
            #[cfg(windows)]
            AudioStream::Wasapi(stream) => stream.play(),
        }
    }
}

pub fn input_device(name: Option<&str>) -> Result<cpal::Device> {
    let host = cpal::default_host();
    find_device(host.input_devices()?, name)
        .or_else(|| host.default_input_device())
        .ok_or_else(|| anyhow!("input device not found"))
}

pub fn output_device(name: Option<&str>) -> Result<cpal::Device> {
    let host = cpal::default_host();
    find_device(host.output_devices()?, name)
        .or_else(|| host.default_output_device())
        .ok_or_else(|| anyhow!("output device not found"))
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

pub fn mono_input_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
    let mut config = device
        .default_input_config()
        .context("failed to get default input config")?;
    if config.channels() != 1 {
        let sample_format = config.sample_format();
        let sample_rate = config.sample_rate();
        let buffer_size = *config.buffer_size();
        config = cpal::SupportedStreamConfig::new(1, sample_rate, buffer_size, sample_format);
    }
    Ok(config)
}

pub fn mono_output_config(
    device: &cpal::Device,
    sample_rate: cpal::SampleRate,
) -> Result<cpal::SupportedStreamConfig> {
    let default = device
        .default_output_config()
        .context("failed to get default output config")?;
    Ok(cpal::SupportedStreamConfig::new(
        1,
        sample_rate,
        *default.buffer_size(),
        default.sample_format(),
    ))
}

fn cpal_scratch_samples(config: &cpal::SupportedStreamConfig) -> usize {
    let samples = match *config.buffer_size() {
        cpal::SupportedBufferSize::Range { max, .. } => {
            (max as usize).saturating_mul(config.channels().max(1) as usize)
        }
        cpal::SupportedBufferSize::Unknown => CPAL_SCRATCH_FALLBACK_SAMPLES,
    };
    // This buffer is moved into CPAL callbacks so sample-format conversion can
    // stay allocation-free on the real-time path.
    samples.clamp(1, CPAL_MAX_SCRATCH_SAMPLES)
}

fn build_cpal_input_stream<F>(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    mut on_samples: F,
) -> Result<cpal::Stream>
where
    F: FnMut(&[f32]) + Send + 'static,
{
    let stream_config = config.clone().into();
    let err_fn = |err| tracing::warn!("input stream error: {err}");
    match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| on_samples(data),
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => {
            let mut scratch = vec![0.0; cpal_scratch_samples(config)];
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| {
                    for input in data.chunks(scratch.len()) {
                        let converted = &mut scratch[..input.len()];
                        dsp::i16_to_f32_into(input, converted);
                        on_samples(converted);
                    }
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            let mut scratch = vec![0.0; cpal_scratch_samples(config)];
            device.build_input_stream(
                &stream_config,
                move |data: &[u16], _| {
                    for input in data.chunks(scratch.len()) {
                        let converted = &mut scratch[..input.len()];
                        dsp::u16_to_f32_into(input, converted);
                        on_samples(converted);
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
    .context("failed to build input stream")
}

fn build_cpal_output_stream<F>(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    mut fill: F,
) -> Result<cpal::Stream>
where
    F: FnMut(&mut [f32]) + Send + 'static,
{
    let stream_config = config.clone().into();
    let err_fn = |err| tracing::warn!("output stream error: {err}");
    match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_output_stream(
            &stream_config,
            move |data: &mut [f32], _| fill(data),
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => {
            let mut scratch = vec![0.0; cpal_scratch_samples(config)];
            device.build_output_stream(
                &stream_config,
                move |data: &mut [i16], _| {
                    for output in data.chunks_mut(scratch.len()) {
                        let tmp = &mut scratch[..output.len()];
                        fill(tmp);
                        dsp::f32_to_i16_into(tmp, output);
                    }
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            let mut scratch = vec![0.0; cpal_scratch_samples(config)];
            device.build_output_stream(
                &stream_config,
                move |data: &mut [u16], _| {
                    for output in data.chunks_mut(scratch.len()) {
                        let tmp = &mut scratch[..output.len()];
                        fill(tmp);
                        dsp::f32_to_u16_into(tmp, output);
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
    .context("failed to build output stream")
}
