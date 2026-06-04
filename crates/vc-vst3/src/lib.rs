//! VST3 plugin front-end for the vc-rs RVC pipeline.
//!
//! The plugin reuses `vc_core` (the same RVC pipeline the CLI drives) and feeds
//! it from the host's `process()` callback instead of driving an audio device
//! directly. Heavy work runs on a worker thread; see [`runtime`] for the
//! realtime bridge.

use std::num::NonZeroU32;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use nice_plug::prelude::*;

mod config;
mod dll_path;
mod editor;
mod params;
mod runtime;

use config::PluginConfig;
use params::VcRvcParams;
use runtime::PluginRuntime;

pub struct VcRvcPlugin {
    params: Arc<VcRvcParams>,
    runtime: Option<PluginRuntime>,
    /// GUI → worker: request a pipeline rebuild from the current settings.
    reload: Arc<AtomicBool>,
    /// GUI sets on edit, worker clears on apply: drives the "unapplied" hint.
    dirty: Arc<AtomicBool>,
    /// worker → GUI: human-readable status.
    status: Arc<Mutex<String>>,
}

impl Default for VcRvcPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(VcRvcParams::default()),
            runtime: None,
            reload: Arc::new(AtomicBool::new(false)),
            dirty: Arc::new(AtomicBool::new(false)),
            status: Arc::new(Mutex::new("idle".to_string())),
        }
    }
}

impl Plugin for VcRvcPlugin {
    const NAME: &'static str = "VC-RS RVC";
    const VENDOR: &'static str = "vc-rs";
    const URL: &'static str = "https://github.com/wok-rvc/vc-rs";
    const EMAIL: &'static str = "noreply@vc-rs.invalid";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Stereo is the default layout; mono is offered as a fallback. The host uses
    // the first layout as the default. The pipeline is mono internally, so stereo
    // input is downmixed and the converted mono is fanned out to all channels.
    //
    // NOTE: a mono-first default makes Element terminate when the plugin is added
    // to a (stereo) track — the Rust code loads and processes fine (verified by
    // trace: default -> editor -> initialize -> process all succeed, no panic),
    // but Element bails after the first process block. Keep stereo first.
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

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            self.params.clone(),
            self.reload.clone(),
            self.dirty.clone(),
            self.status.clone(),
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        // Let bundled provider/CUDA/cuDNN DLLs load from beside the plugin
        // before any ONNX Runtime session is created on the worker thread.
        dll_path::add_plugin_dir_to_dll_search_path();

        // Bootstrap: if the persisted settings have no models yet (fresh
        // instance), seed them from the headless TOML config when present. A
        // restored project already has its own settings and is left untouched.
        if !self.params.settings.read().unwrap().has_models() {
            let seed = PluginConfig::discover();
            if seed.has_models() {
                *self.params.settings.write().unwrap() = seed;
            }
        }

        // Tear down any previous worker before starting a new one (sample rate
        // or block size may have changed).
        self.runtime = None;

        let sample_rate = buffer_config.sample_rate.round() as u32;
        let max_block = buffer_config.max_buffer_size as usize;
        let runtime = PluginRuntime::start(
            self.params.clone(),
            self.reload.clone(),
            self.dirty.clone(),
            self.status.clone(),
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
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        match self.runtime.as_mut() {
            Some(runtime) => {
                runtime.process_block(buffer.as_slice());
                // The worker updates latency when chunk_ms changes; relay it to
                // the host (this just sets a pending flag in the wrapper).
                if let Some(latency) = runtime.poll_latency_update() {
                    context.set_latency_samples(latency);
                }
            }
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

impl Vst3Plugin for VcRvcPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VcRsRvcVoiceConv";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Fx];
}

nice_export_vst3!(VcRvcPlugin);
