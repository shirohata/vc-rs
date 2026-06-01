//! VST3 / CLAP plugin front-end for the vc-rs RVC pipeline.
//!
//! The plugin reuses `vc_core` (the same RVC pipeline the CLI drives) and feeds
//! it from the host's `process()` callback instead of driving an audio device
//! directly. Heavy work runs on a worker thread; see [`runtime`] for the
//! realtime bridge.

use std::num::NonZeroU32;
use std::sync::Arc;

use nih_plug::prelude::*;

mod config;
mod params;
mod runtime;

use config::PluginConfig;
use params::VcRvcParams;
use runtime::PluginRuntime;

pub struct VcRvcPlugin {
    params: Arc<VcRvcParams>,
    config: PluginConfig,
    runtime: Option<PluginRuntime>,
}

impl Default for VcRvcPlugin {
    fn default() -> Self {
        let config = PluginConfig::discover();
        let params = Arc::new(VcRvcParams::from_config(&config));
        Self {
            params,
            config,
            runtime: None,
        }
    }
}

impl Plugin for VcRvcPlugin {
    const NAME: &'static str = "VC-RS RVC";
    const VENDOR: &'static str = "vc-rs";
    const URL: &'static str = "https://github.com/wok-rvc/vc-rs";
    const EMAIL: &'static str = "noreply@vc-rs.invalid";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Prefer stereo; mono is offered as a fallback. The pipeline is mono
    // internally, so stereo input is downmixed and the converted mono is fanned
    // out to all output channels.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        // Tear down any previous worker before starting a new one (sample rate
        // or block size may have changed).
        self.runtime = None;

        let sample_rate = buffer_config.sample_rate.round() as u32;
        let max_block = buffer_config.max_buffer_size as usize;
        let runtime = PluginRuntime::start(
            self.config.clone(),
            self.params.clone(),
            sample_rate,
            max_block,
        );
        context.set_latency_samples(runtime.latency_samples);
        self.runtime = Some(runtime);
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        match self.runtime.as_mut() {
            Some(runtime) => runtime.process_block(buffer.as_slice()),
            None => {
                for channel in buffer.as_slice() {
                    channel.fill(0.0);
                }
            }
        }
        ProcessStatus::Normal
    }

    fn deactivate(&mut self) {
        self.runtime = None;
    }
}

impl ClapPlugin for VcRvcPlugin {
    const CLAP_ID: &'static str = "rs.vc.rvc";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Realtime RVC voice conversion");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for VcRvcPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VcRsRvcVoiceConv";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Fx];
}

nih_export_clap!(VcRvcPlugin);
nih_export_vst3!(VcRvcPlugin);
