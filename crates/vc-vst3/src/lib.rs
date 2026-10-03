//! VST3 plugin front-end for the vc-rs RVC pipeline.
//!
//! The plugin reuses `vc_core` (the same RVC pipeline the CLI drives) and feeds
//! it from the host's `process()` callback instead of driving an audio device
//! directly. Heavy work runs on a worker thread; see [`runtime`] for the
//! realtime bridge.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use nice_plug::prelude::*;

mod config;
mod dll_path;
mod editor;
mod params;
mod runtime;

use config::PluginConfig;
use params::VcRvcParams;
use runtime::{PluginRuntime, PluginStatus, ReloadWaker};

pub(crate) mod plugin_identity {
    #[cfg(feature = "tensorrt")]
    pub const NAME: &str = "VC-RS RVC (TensorRT)";
    #[cfg(feature = "tensorrt")]
    pub const VST3_CLASS_ID: [u8; 16] = *b"VcRsRvcTensorTRT";

    #[cfg(all(not(feature = "tensorrt"), feature = "windowsml"))]
    pub const NAME: &str = "VC-RS RVC (Windows ML)";
    #[cfg(all(not(feature = "tensorrt"), feature = "windowsml"))]
    pub const VST3_CLASS_ID: [u8; 16] = *b"VcRsRvcWinMLPlug";

    #[cfg(all(
        not(feature = "tensorrt"),
        not(feature = "windowsml"),
        feature = "cuda"
    ))]
    pub const NAME: &str = "VC-RS RVC (CUDA)";
    #[cfg(all(
        not(feature = "tensorrt"),
        not(feature = "windowsml"),
        feature = "cuda"
    ))]
    pub const VST3_CLASS_ID: [u8; 16] = *b"VcRsRvcCudaDev01";

    #[cfg(not(any(feature = "tensorrt", feature = "windowsml", feature = "cuda")))]
    pub const NAME: &str = "VC-RS RVC";
    #[cfg(not(any(feature = "tensorrt", feature = "windowsml", feature = "cuda")))]
    pub const VST3_CLASS_ID: [u8; 16] = *b"VcRsRvcVoiceConv";
}

pub struct VcRvcPlugin {
    params: Arc<VcRvcParams>,
    runtime: Option<PluginRuntime>,
    /// GUI → worker: request a pipeline rebuild from the current settings.
    reload: Arc<AtomicBool>,
    /// GUI/worker handshake: disables duplicate reload requests while loading.
    loading: Arc<AtomicBool>,
    /// GUI sets on edit, worker clears on apply: drives the "unapplied" hint.
    dirty: Arc<AtomicBool>,
    /// worker → GUI: short status plus optional expandable error detail.
    status: Arc<Mutex<PluginStatus>>,
    /// Editor → worker: wakes the parked worker when a reload is submitted, so a
    /// Load / Reload applies immediately even while the host is idle.
    reload_waker: Arc<ReloadWaker>,
    /// Successful explicit Load survives host deactivate/reinitialize, but is
    /// never persisted or inferred merely from a restored project's model paths.
    applied_settings: Option<PluginConfig>,
}

impl Default for VcRvcPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(VcRvcParams::default()),
            runtime: None,
            reload: Arc::new(AtomicBool::new(false)),
            loading: Arc::new(AtomicBool::new(false)),
            dirty: Arc::new(AtomicBool::new(false)),
            status: Arc::new(Mutex::new(PluginStatus::new("idle"))),
            reload_waker: Arc::new(ReloadWaker::default()),
            applied_settings: None,
        }
    }
}

impl Plugin for VcRvcPlugin {
    const NAME: &'static str = plugin_identity::NAME;
    const VENDOR: &'static str = "vc-rs";
    const URL: &'static str = "https://github.com/shirohata/vc-rs";
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
            self.loading.clone(),
            self.dirty.clone(),
            self.status.clone(),
            self.reload_waker.clone(),
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
        if self.runtime.is_none()
            && self.applied_settings.is_none()
            && !self.params.settings.read().unwrap().has_models()
        {
            let seed = PluginConfig::discover();
            if seed.has_models() {
                *self.params.settings.write().unwrap() = seed;
            }
        }

        let sample_rate = buffer_config.sample_rate.round() as u32;
        let max_block = buffer_config.max_buffer_size as usize;
        if let Some(runtime) = self.runtime.as_mut() {
            if runtime.reconfigure(sample_rate, max_block, buffer_config.process_mode) {
                context.set_latency_samples(runtime.current_latency());
                return true;
            }
        }
        // Fixed inference profiles require a new pipeline after rate changes.
        // Reuse only the last successfully applied snapshot, preserving staged
        // editor changes and the explicit Load boundary for fresh/restored hosts.
        let applied_settings = match self.runtime.as_ref() {
            Some(runtime) => runtime.applied_settings(),
            None => self.applied_settings.take(),
        };
        self.runtime = None;
        let runtime = PluginRuntime::start(
            self.params.clone(),
            self.reload.clone(),
            self.loading.clone(),
            self.dirty.clone(),
            self.status.clone(),
            &self.reload_waker,
            sample_rate,
            max_block,
            buffer_config.process_mode,
            applied_settings,
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
        self.runtime
            .as_ref()
            .and_then(PluginRuntime::tail_samples)
            .map_or(ProcessStatus::Normal, ProcessStatus::Tail)
    }

    fn reset(&mut self) {
        if let Some(runtime) = self.runtime.as_mut() {
            runtime.reset();
        }
    }

    fn deactivate(&mut self) {
        self.applied_settings = self
            .runtime
            .as_ref()
            .and_then(PluginRuntime::applied_settings);
        self.runtime = None;
        self.reload.store(false, Ordering::SeqCst);
        self.loading.store(false, Ordering::SeqCst);
    }
}

impl Vst3Plugin for VcRvcPlugin {
    const VST3_CLASS_ID: [u8; 16] = plugin_identity::VST3_CLASS_ID;
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Fx];
}

nice_export_vst3!(VcRvcPlugin);

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    struct TestInitContext;
    impl InitContext<VcRvcPlugin> for TestInitContext {
        fn plugin_api(&self) -> PluginApi {
            PluginApi::Vst3
        }
        fn execute(&self, _: ()) {}
        fn set_latency_samples(&self, _: u32) {}
        fn set_current_voice_capacity(&self, _: u32) {}
    }

    fn configured_plugin(fade: u32) -> VcRvcPlugin {
        let plugin = VcRvcPlugin::default();
        *plugin.params.settings.write().unwrap() = PluginConfig {
            model: "test-rvc.onnx".into(),
            embedder: "test-embedder.onnx".into(),
            f0_model: "test-f0.onnx".into(),
            chunk_ms: 20,
            crossfade_ms: fade,
            sola_search_ms: 0,
            rvc_output_tail_discard_ms: 0,
            ..PluginConfig::default()
        };
        plugin
    }

    fn initialize(plugin: &mut VcRvcPlugin, rate: f32, max_block: u32, mode: ProcessMode) {
        assert!(plugin.initialize(
            &VcRvcPlugin::AUDIO_IO_LAYOUTS[1],
            &BufferConfig {
                sample_rate: rate,
                min_buffer_size: Some(1),
                max_buffer_size: max_block,
                process_mode: mode,
            },
            &mut TestInitContext
        ));
    }

    #[test]
    fn mode_and_block_reinitialization_preserves_loaded_models_and_staged_edits() {
        let mut plugin = configured_plugin(0);
        plugin.runtime = Some(runtime::test_loaded_runtime(
            plugin.params.clone(),
            ProcessMode::Realtime,
            64,
        ));
        let worker = plugin.runtime.as_ref().unwrap().test_worker_id();
        plugin.params.settings.write().unwrap().chunk_ms = 30;
        initialize(&mut plugin, 16_000.0, 2048, ProcessMode::Offline);
        let runtime = plugin.runtime.as_mut().unwrap();
        assert_eq!(runtime.test_worker_id(), worker);
        assert_eq!(runtime.applied_settings().unwrap().chunk_ms, 20);
        assert_eq!(plugin.params.settings.read().unwrap().chunk_ms, 30);
        let mut input = [0.25; 640];
        runtime.process_block(&mut [&mut input]);
        assert_eq!(&input[..320], &[0.0; 320]);
        assert_eq!(&input[320..], &[0.25; 320]);
    }

    #[test]
    fn public_host_reset_discards_previously_queued_output() {
        let mut plugin = configured_plugin(0);
        plugin.runtime = Some(runtime::test_loaded_runtime(
            plugin.params.clone(),
            ProcessMode::Realtime,
            320,
        ));
        plugin
            .runtime
            .as_mut()
            .unwrap()
            .process_block(&mut [&mut [0.75; 320]]);
        let deadline = Instant::now() + Duration::from_secs(2);
        while plugin.runtime.as_mut().unwrap().test_output_samples() < 320 {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        Plugin::reset(&mut plugin);
        let mut new_input = [0.0; 320];
        plugin
            .runtime
            .as_mut()
            .unwrap()
            .process_block(&mut [&mut new_input]);
        assert_eq!(new_input, [0.0; 320]);
    }

    #[test]
    fn rate_reinitialization_uses_successful_snapshot_instead_of_staged_settings() {
        let mut plugin = configured_plugin(0);
        plugin.runtime = Some(runtime::test_loaded_runtime(
            plugin.params.clone(),
            ProcessMode::Realtime,
            64,
        ));
        plugin.params.settings.write().unwrap().chunk_ms = 30;
        initialize(&mut plugin, 44_100.0, 64, ProcessMode::Offline);
        assert_eq!(plugin.runtime.as_ref().unwrap().test_input_hop(), 882);
        assert_eq!(plugin.params.settings.read().unwrap().chunk_ms, 30);
        // Fixture model files do not exist: reloading may fail, but it must use
        // the last successful 20 ms shape and never apply the staged 30 ms edit.
    }

    #[test]
    fn deactivate_retains_explicit_load_but_configured_fresh_instance_stays_idle() {
        let mut plugin = configured_plugin(0);
        plugin.runtime = Some(runtime::test_loaded_runtime(
            plugin.params.clone(),
            ProcessMode::Realtime,
            64,
        ));
        plugin.deactivate();
        assert_eq!(plugin.applied_settings.as_ref().unwrap().chunk_ms, 20);

        let mut fresh = configured_plugin(0);
        initialize(&mut fresh, 16_000.0, 64, ProcessMode::Offline);
        let deadline = Instant::now() + Duration::from_secs(2);
        while fresh.status.lock().unwrap().summary == "idle" {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(
            fresh.status.lock().unwrap().summary,
            "models configured; click Load / Reload"
        );
        assert!(fresh.runtime.as_ref().unwrap().applied_settings().is_none());
        let mut input = [0.25; 64];
        fresh
            .runtime
            .as_mut()
            .unwrap()
            .process_block(&mut [&mut input]);
        assert_eq!(input, [0.0; 64]);
    }
}
