#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;
use vc_app::{
    AudioHost, DenoiserMode, EngineController, EngineState, F0Config, LiveParams, NoiseGateShaping,
    OutputDynamicsConfig, RealtimeConfig, Smoother, TelemetrySnapshot,
};
#[cfg(not(test))]
use vc_core::gpu::list_cuda_devices;
use vc_core::gpu::GpuDevice;
use vc_core::validation::CONVERSION_TIMING_LIMITS;
use vc_core::Provider;

mod model_setup;
mod onboarding;
mod ui_text;

const SAVE_DEBOUNCE: Duration = Duration::from_millis(500);
const TELEMETRY_REFRESH: Duration = Duration::from_millis(250);
const GUI_CROSSFADE_MS: u32 = 85;
const GUI_SOLA_SEARCH_MS: u32 = 12;
const GUI_MIN_EXTRA_CONVERT_MS: u32 = 100;
const GPU_DEVICE_SELECTOR_AVAILABLE: bool = cfg!(any(feature = "cuda", feature = "tensorrt"));

fn main() -> eframe::Result {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    eframe::run_native(
        "vc-rs",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([880.0, 680.0])
                .with_min_inner_size([520.0, 440.0]),
            ..Default::default()
        },
        Box::new(|cc| {
            install_system_japanese_font(&cc.egui_ctx);
            let mut style = (*cc.egui_ctx.global_style()).clone();
            style.spacing.item_spacing = egui::vec2(10.0, 10.0);
            style.spacing.button_padding = egui::vec2(12.0, 7.0);
            style
                .text_styles
                .insert(egui::TextStyle::Body, egui::FontId::proportional(16.0));
            style
                .text_styles
                .insert(egui::TextStyle::Button, egui::FontId::proportional(16.0));
            cc.egui_ctx.set_global_style(style);
            Ok(Box::new(VcGui::new()))
        }),
    )
}

fn install_system_japanese_font(ctx: &egui::Context) {
    let Some((bytes, face_index)) = system_japanese_font_candidates()
        .into_iter()
        .find_map(|(path, face_index)| fs::read(path).ok().map(|bytes| (bytes, face_index)))
    else {
        return;
    };

    // Keep egui's compact Latin fonts first and use the OS font only for
    // missing glyphs. Bundling a CJK font would add roughly 5-15 MB to every
    // package, while loading it here has no impact on the real-time audio path.
    let font_name = "system_japanese".to_owned();
    let mut font_data = egui::FontData::from_owned(bytes);
    font_data.index = face_index;

    let mut fonts = egui::FontDefinitions::default();
    fonts
        .font_data
        .insert(font_name.clone(), Arc::new(font_data));
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push(font_name.clone());
    }
    ctx.set_fonts(fonts);
}

fn system_japanese_font_candidates() -> Vec<(PathBuf, u32)> {
    #[cfg(target_os = "windows")]
    {
        let fonts = std::env::var_os("WINDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\Windows"))
            .join("Fonts");
        return [
            "NotoSansJP-VF.ttf",
            "BIZ-UDGothicR.ttc",
            "YuGothM.ttc",
            "meiryo.ttc",
            "msgothic.ttc",
        ]
        .into_iter()
        .map(|name| (fonts.join(name), 0))
        .collect();
    }

    #[cfg(target_os = "macos")]
    {
        return [
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
            "/Library/Fonts/NotoSansJP-Regular.ttf",
        ]
        .into_iter()
        .map(|path| (PathBuf::from(path), 0))
        .collect();
    }

    #[cfg(target_os = "linux")]
    {
        return [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf",
        ]
        .into_iter()
        .map(|path| (PathBuf::from(path), 0))
        .collect();
    }

    #[allow(unreachable_code)]
    Vec::new()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
struct GuiSettings {
    language: ui_text::Language,
    setup_completed: bool,
    // Skipping dismisses the tutorial without claiming models/audio were tested.
    setup_skipped: bool,
    tutorial: Option<onboarding::Progress>,
    accepted_terms: Vec<String>,
    model: String,
    embedder: String,
    f0_model: String,
    support_custom_mode: [String; 2],
    support_custom_paths: [String; 2],
    provider: String,
    gpu_priority: String,
    gpu_device_id: u32,
    // Host is per direction (input/output independent); the GUI offers the
    // platform's host (WASAPI/CoreAudio/ALSA) plus, when built with the `asio`
    // feature on Windows, ASIO. WASAPI exclusive mode stays CLI-only. Tokens are
    // cpal HostId names ("wasapi"/"asio"/...). Unknown/legacy keys fall back to the
    // platform default below.
    input_host: String,
    output_host: String,
    input_device: String,
    output_device: String,
    recent_input_devices: Vec<String>,
    recent_output_devices: Vec<String>,
    wasapi_input_exclusive: bool,
    wasapi_output_exclusive: bool,
    wasapi_buffer_ms: u32,
    chunk_ms: u32,
    crossfade_ms: u32,
    sola_search_ms: u32,
    rvc_output_tail_discard_ms: u32,
    extra_convert_ms: u32,
    smoother: String,
    f0_threshold: f32,
    silence_threshold: f32,
    pitch_shift: f32,
    speaker_id: i64,
    input_gain: f32,
    output_gain: f32,
    denoiser: String,
    #[serde(default)]
    gtcrn_model_dir: String,
    #[serde(skip_serializing)]
    noise_gate_enabled: bool,
    noise_gate_threshold: f32,
    noise_gate_attack_ms: f32,
    noise_gate_release_ms: f32,
    noise_gate_floor: f32,
    volume_envelope: bool,
    rms_mix_rate: f32,
    auto_output_gain: bool,
    target_output_rms: f32,
    max_output_gain: f32,
    passthrough: bool,
}

impl Default for GuiSettings {
    fn default() -> Self {
        Self {
            language: ui_text::Language::English,
            setup_completed: false,
            setup_skipped: false,
            tutorial: None,
            accepted_terms: Vec::new(),
            model: String::new(),
            embedder: String::new(),
            f0_model: String::new(),
            support_custom_mode: Default::default(),
            support_custom_paths: Default::default(),
            provider: default_provider_name().to_string(),
            gpu_priority: "high".to_string(),
            gpu_device_id: 0,
            input_host: default_host_token().to_string(),
            output_host: default_host_token().to_string(),
            input_device: String::new(),
            output_device: String::new(),
            recent_input_devices: Vec::new(),
            recent_output_devices: Vec::new(),
            wasapi_input_exclusive: false,
            wasapi_output_exclusive: false,
            wasapi_buffer_ms: 0,
            chunk_ms: 500,
            crossfade_ms: GUI_CROSSFADE_MS,
            sola_search_ms: GUI_SOLA_SEARCH_MS,
            rvc_output_tail_discard_ms: 10,
            extra_convert_ms: 100,
            smoother: "phase-vocoder".to_string(),
            f0_threshold: 0.3,
            silence_threshold: 0.0001,
            pitch_shift: 0.0,
            speaker_id: 0,
            input_gain: 1.0,
            output_gain: 1.0,
            denoiser: "off".to_string(),
            gtcrn_model_dir: String::new(),
            noise_gate_enabled: false,
            noise_gate_threshold: 0.01,
            noise_gate_attack_ms: 5.0,
            noise_gate_release_ms: 50.0,
            noise_gate_floor: 0.0,
            volume_envelope: false,
            rms_mix_rate: 0.0,
            auto_output_gain: false,
            target_output_rms: 0.03,
            max_output_gain: 512.0,
            passthrough: false,
        }
    }
}

impl GuiSettings {
    fn normalize_gui_managed_settings(&mut self) {
        // WASAPI exclusive mode and these smoothing timings remain available to the
        // CLI, but the GUI intentionally pins them until their safe tuning and
        // failure behavior are clear enough to expose to general users. Per-direction
        // hosts are clamped to what the GUI offers on this platform (e.g. WASAPI,
        // plus ASIO when built in); WASAPI exclusive mode is never selectable here.
        if !gui_host_names().contains(&self.input_host.as_str()) {
            self.input_host = default_host_token().to_string();
        }
        if !gui_host_names().contains(&self.output_host.as_str()) {
            self.output_host = default_host_token().to_string();
        }
        self.wasapi_input_exclusive = false;
        self.wasapi_output_exclusive = false;
        self.wasapi_buffer_ms = 0;
        self.crossfade_ms = GUI_CROSSFADE_MS;
        self.sola_search_ms = GUI_SOLA_SEARCH_MS;
        self.extra_convert_ms = self.extra_convert_ms.max(GUI_MIN_EXTRA_CONVERT_MS);
        // Validate the persisted provider against what this build can run
        // (compile-time), not the live picker list: a saved catalog EP stays
        // valid even if the device's catalog does not list it right now.
        if !Provider::from_name(&self.provider).is_some_and(Provider::available_in_build) {
            self.provider = default_provider_name().to_string();
        }
        if cfg!(feature = "windowsml") && self.provider == "cpu" {
            // Both names use the same Windows ML runtime and CPU session.
            // Canonicalize old settings so the single CPU option stays selected.
            self.provider = "windowsml-cpu".into();
        }
        if !gpu_priority_names().contains(&self.gpu_priority.as_str()) {
            self.gpu_priority = "high".to_string();
        }
        // Migrate settings written before the exclusive denoiser selector.
        if self.noise_gate_enabled && self.denoiser == "off" {
            self.denoiser = "noise-gate".to_string();
        }
        self.noise_gate_enabled = false;
        if !denoiser_names().contains(&self.denoiser.as_str()) {
            self.denoiser = "off".to_string();
        }
    }

    fn live(&self) -> LiveParams {
        LiveParams {
            pitch_shift: self.pitch_shift,
            speaker_id: self.speaker_id,
            input_gain: self.input_gain,
            output_gain: self.output_gain,
            // Gate on/off rides the unified live path now, so toggling the
            // denoiser between off and noise-gate takes effect without a reload;
            // rnnoise still needs a reload (it rebuilds a stateful denoiser).
            noise_gate_enabled: self.denoiser == "noise-gate",
            noise_gate_threshold: self.noise_gate_threshold,
        }
    }

    fn realtime(&self) -> Result<RealtimeConfig, String> {
        if self.extra_convert_ms < GUI_MIN_EXTRA_CONVERT_MS {
            return Err(format!(
                "Extra convert ms must be at least {GUI_MIN_EXTRA_CONVERT_MS} ms in the GUI"
            ));
        }
        Ok(RealtimeConfig {
            model: path_option(&self.model),
            embedder: path_option(&self.embedder),
            embedder_output: None,
            f0_model: path_option(&self.f0_model),
            provider: parse_provider(&self.provider)?,
            gpu_priority: parse_gpu_priority(&self.gpu_priority)?,
            gpu_device_id: self.gpu_device_id,
            input_host: self.input_host(),
            output_host: self.output_host(),
            input_device: string_option(&self.input_device),
            output_device: string_option(&self.output_device),
            wasapi_input_exclusive: false,
            wasapi_output_exclusive: false,
            wasapi_buffer_ms: 0,
            chunk_ms: self.chunk_ms,
            crossfade_ms: GUI_CROSSFADE_MS,
            sola_search_ms: GUI_SOLA_SEARCH_MS,
            // Missing keys use the new default; explicit saved choices retain
            // their behavior, including the legacy SOLA fallback for unknown values.
            smoother: match self.smoother.as_str() {
                "psola" => Smoother::Psola,
                "phase-vocoder" => Smoother::PhaseVocoder,
                _ => Smoother::Sola,
            },
            rvc_output_tail_discard_ms: self.rvc_output_tail_discard_ms,
            extra_convert_ms: self.extra_convert_ms,
            f0: F0Config {
                f0_threshold: self.f0_threshold,
                silence_threshold: self.silence_threshold,
                ..F0Config::default()
            },
            denoiser_mode: parse_denoiser(&self.denoiser)?,
            gtcrn_model_dir: if self.gtcrn_model_dir.is_empty() {
                None
            } else {
                Some(PathBuf::from(&self.gtcrn_model_dir))
            },
            noise_gate_shaping: NoiseGateShaping {
                attack_ms: self.noise_gate_attack_ms,
                release_ms: self.noise_gate_release_ms,
                floor: self.noise_gate_floor,
            },
            output_dynamics: OutputDynamicsConfig {
                volume_envelope: self.volume_envelope,
                rms_mix_rate: self.rms_mix_rate,
                auto_output_gain: self.auto_output_gain,
                target_output_rms: self.target_output_rms,
                max_output_gain: self.max_output_gain,
            },
            passthrough: self.passthrough,
            debug_input_wav: None,
            debug_output_wav: None,
        })
    }

    fn input_host(&self) -> AudioHost {
        parse_gui_host(&self.input_host)
    }

    fn output_host(&self) -> AudioHost {
        parse_gui_host(&self.output_host)
    }
}

struct VcGui {
    controller: EngineController,
    settings: GuiSettings,
    dirty_since: Option<Instant>,
    ui_error: Option<String>,
    telemetry: TelemetrySnapshot,
    telemetry_updated_at: Instant,
    applied_chunk_ms: Option<u32>,
    gpu_devices: Arc<Mutex<GpuDeviceDiscovery>>,
    openvino_devices: vc_core::openvino::DeviceDiscovery,
    pth_convert: Option<PthConvert>,
    model_download: Option<model_setup::Download>,
    onboarding: onboarding::Onboarding,
}

/// State of the `.pth` → `.onnx` conversion dialog. The conversion itself
/// runs on a named worker thread (same pattern as GPU discovery); the shared
/// state is how the thread reports progress back to the UI.
struct PthConvert {
    source: PathBuf,
    mode: vc_convert::ExportMode,
    state: Arc<Mutex<PthConvertState>>,
}

#[derive(Clone)]
enum PthConvertState {
    Configuring,
    Running { stage: &'static str },
    Done { output: PathBuf },
    Failed { error: String },
}

impl PthConvert {
    fn new(source: PathBuf) -> Self {
        Self {
            source,
            mode: vc_convert::ExportMode::Streaming,
            state: Arc::new(Mutex::new(PthConvertState::Configuring)),
        }
    }

    fn output_path(&self) -> PathBuf {
        self.source.with_extension("onnx")
    }

    fn start(&self) {
        let state = Arc::clone(&self.state);
        let source = self.source.clone();
        let options = vc_convert::ConvertOptions {
            export_mode: self.mode,
            ..Default::default()
        };
        if let Ok(mut current) = self.state.lock() {
            *current = PthConvertState::Running {
                stage: vc_convert::ProgressStage::ReadArchive.label(),
            };
        }
        let spawned = std::thread::Builder::new()
            .name("vc-gui-pth-convert".to_string())
            .spawn(move || {
                let progress_state = Arc::clone(&state);
                let mut progress = move |stage: vc_convert::ProgressStage| {
                    if let Ok(mut current) = progress_state.lock() {
                        *current = PthConvertState::Running {
                            stage: stage.label(),
                        };
                    }
                };
                let result = vc_convert::convert_pth_file(&source, &options, &mut progress);
                if let Ok(mut current) = state.lock() {
                    *current = match result {
                        Ok(output) => PthConvertState::Done { output },
                        Err(err) => PthConvertState::Failed {
                            error: format!("{err:#}"),
                        },
                    };
                }
            });
        if let Err(err) = spawned {
            if let Ok(mut current) = self.state.lock() {
                *current = PthConvertState::Failed {
                    error: format!("failed to spawn conversion thread: {err}"),
                };
            }
        }
    }
}

fn is_pth_path(value: &str) -> bool {
    Path::new(value)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pth"))
}

#[derive(Clone, Debug, Default)]
struct GpuDeviceDiscovery {
    #[cfg(not(test))]
    started: bool,
    devices: Option<Vec<GpuDevice>>,
    error: Option<String>,
}

impl VcGui {
    fn language_picker(&mut self, ui: &mut egui::Ui) {
        let previous = self.settings.language;
        egui::ComboBox::new("language-picker", "Language / 言語")
            .selected_text(match self.settings.language {
                ui_text::Language::English => "English",
                ui_text::Language::Japanese => "日本語",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.settings.language,
                    ui_text::Language::English,
                    "English",
                );
                ui.selectable_value(
                    &mut self.settings.language,
                    ui_text::Language::Japanese,
                    "日本語",
                );
            });
        ui_text::set_language(ui.ctx(), self.settings.language);
        if previous != self.settings.language {
            self.changed();
        }
    }

    fn new() -> Self {
        let (mut settings, ui_error) = load_settings();
        settings.normalize_gui_managed_settings();
        let discovered = discover_support_models(&mut settings);
        let onboarding = onboarding::Onboarding::new(&settings);
        let controller = EngineController::new(settings.live());
        let _ = controller.refresh_devices(settings.input_host(), settings.output_host());
        Self {
            controller,
            settings,
            dirty_since: discovered.then(Instant::now),
            ui_error,
            telemetry: TelemetrySnapshot::default(),
            telemetry_updated_at: Instant::now() - TELEMETRY_REFRESH,
            applied_chunk_ms: None,
            gpu_devices: Arc::new(Mutex::new(GpuDeviceDiscovery::default())),
            openvino_devices: vc_core::openvino::DeviceDiscovery::default(),
            pth_convert: None,
            model_download: None,
            onboarding,
        }
    }

    fn changed(&mut self) {
        self.dirty_since = Some(Instant::now());
        self.controller.set_live_params(self.settings.live());
    }

    fn maybe_save(&mut self) {
        if self
            .dirty_since
            .is_some_and(|at| at.elapsed() >= SAVE_DEBOUNCE)
        {
            if let Err(err) = save_settings(&self.settings) {
                self.ui_error = Some(err);
                self.dirty_since = Some(Instant::now());
            } else {
                self.dirty_since = None;
            }
        }
    }

    fn browse_into(&mut self, kind: ModelKind) {
        let lang = self.settings.language;
        // Only the RVC slot accepts .pth: picking one opens the conversion
        // dialog instead of storing the path (the engine only loads .onnx).
        let dialog = match kind {
            ModelKind::Rvc => rfd::FileDialog::new()
                .add_filter(lang.text("RVC model"), &["onnx", "pth"])
                .add_filter(lang.text("ONNX model"), &["onnx"])
                .add_filter(lang.text("PyTorch checkpoint"), &["pth"]),
            ModelKind::Embedder | ModelKind::F0 => {
                rfd::FileDialog::new().add_filter(lang.text("ONNX model"), &["onnx"])
            }
        };
        if let Some(path) = dialog.pick_file() {
            let value = path.to_string_lossy().into_owned();
            if matches!(kind, ModelKind::Rvc) && is_pth_path(&value) {
                self.pth_convert = Some(PthConvert::new(path));
                return;
            }
            match kind {
                ModelKind::Rvc => self.settings.model = value,
                ModelKind::Embedder => self.settings.embedder = value,
                ModelKind::F0 => self.settings.f0_model = value,
            }
            self.changed();
        }
    }

    /// Render the `.pth` conversion dialog and apply its state transitions.
    /// Actions mutate `self` after the window closure to keep borrows simple.
    fn pth_convert_window(&mut self, ctx: &egui::Context) {
        let lang = self.settings.language;
        enum Action {
            None,
            Start,
            Close,
            Retry,
            Accept(PathBuf),
        }

        let Some(convert) = &mut self.pth_convert else {
            return;
        };
        let state = convert
            .state
            .lock()
            .map(|state| state.clone())
            .unwrap_or(PthConvertState::Configuring);
        let mut action = Action::None;

        egui::Window::new(lang.text("Convert .pth to ONNX"))
            .id(egui::Id::new("pth-conversion"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| match &state {
                PthConvertState::Configuring => {
                    ui.label(format!(
                        "{}: {}",
                        lang.text("Source"),
                        convert.source.display()
                    ));
                    let output = convert.output_path();
                    ui.label(format!("{}: {}", lang.text("Output"), output.display()));
                    if output.exists() {
                        ui.small(lang.text("The existing file will be overwritten."));
                    }
                    egui::ComboBox::new("Export mode", lang.text("Export mode"))
                        .selected_text(match convert.mode {
                            vc_convert::ExportMode::Streaming => {
                                lang.text("Streaming (recommended)")
                            }
                            vc_convert::ExportMode::Webui => lang.text("WebUI-compatible"),
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut convert.mode,
                                vc_convert::ExportMode::Streaming,
                                lang.text("Streaming (recommended)"),
                            );
                            ui.selectable_value(
                                &mut convert.mode,
                                vc_convert::ExportMode::Webui,
                                lang.text("WebUI-compatible"),
                            );
                        });
                    ui.small(lang.text(
                        "Streaming exports carry NSF phase across chunks for the realtime engine.",
                    ));
                    ui.horizontal(|ui| {
                        if ui.button(lang.text("Convert")).clicked() {
                            action = Action::Start;
                        }
                        if ui.button(lang.text("Cancel")).clicked() {
                            action = Action::Close;
                        }
                    });
                }
                PthConvertState::Running { stage } => {
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new());
                        ui.label(lang.text(stage));
                    });
                    // Keep repainting so the worker thread's progress shows
                    // without user input.
                    ctx.request_repaint_after(Duration::from_millis(100));
                }
                PthConvertState::Done { output } => {
                    action = Action::Accept(output.clone());
                }
                PthConvertState::Failed { error } => {
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                    ui.horizontal(|ui| {
                        if ui.button(lang.text("Retry")).clicked() {
                            action = Action::Retry;
                        }
                        if ui.button(lang.text("Close")).clicked() {
                            action = Action::Close;
                        }
                    });
                }
            });

        match action {
            Action::None => {}
            Action::Start => {
                if let Some(convert) = &self.pth_convert {
                    convert.start();
                }
            }
            Action::Retry => {
                if let Some(convert) = &self.pth_convert {
                    if let Ok(mut state) = convert.state.lock() {
                        *state = PthConvertState::Configuring;
                    }
                }
            }
            Action::Close => {
                self.pth_convert = None;
            }
            Action::Accept(output) => {
                self.settings.model = output.to_string_lossy().into_owned();
                self.onboarding.model_check = None;
                self.pth_convert = None;
                self.changed();
            }
        }
    }

    fn apply_or_start(&mut self) {
        // Setup checks are transient UI state, not a prerequisite for Start.
        // The shared engine validates actual files/runtime and reports failures here.
        if !self.settings.passthrough
            && (!model_setup::available(&self.settings.model)
                || is_pth_path(&self.settings.model)
                || !model_setup::available(&self.settings.embedder)
                || !model_setup::available(&self.settings.f0_model))
        {
            self.ui_error =
                Some("Choose a voice model and prepare the embedder / F0 model in Setup.".into());
            return;
        }
        self.controller.set_live_params(self.settings.live());
        #[cfg(not(test))]
        let revision = self.controller.snapshot().0.session_revision + 1;
        #[cfg(test)]
        let revision = self.onboarding.effects.session_revision + 1;
        match self.settings.realtime().and_then(|config| {
            #[cfg(not(test))]
            {
                self.controller
                    .apply_config(config)
                    .map_err(|e| format!("{e:#}"))
            }
            #[cfg(test)]
            {
                let _ = config;
                self.onboarding.effects.start_requests += 1;
                self.onboarding
                    .effects
                    .start_error
                    .clone()
                    .map_or(Ok(()), Err)
            }
        }) {
            Ok(()) => {
                self.ui_error = None;
                self.onboarding.normal.requested = Some(self.settings.clone());
                self.onboarding.normal.expected_revision = revision;
            }
            Err(err) => self.ui_error = Some(err),
        }
    }

    fn stop(&mut self) {
        #[cfg(not(test))]
        let result = self.controller.stop().map_err(|err| format!("{err:#}"));
        #[cfg(test)]
        let result = {
            self.onboarding.effects.stop_requests += 1;
            self.onboarding
                .effects
                .stop_error
                .clone()
                .map_or(Ok(()), Err)
        };
        if let Err(err) = result {
            self.ui_error = Some(format!("{err:#}"));
        } else {
            self.ui_error = None;
            self.applied_chunk_ms = None;
            self.onboarding.normal.requested = None;
            self.onboarding.normal.applied = None;
        }
    }

    // Shared by eframe and the headless fixture so scrolling and error placement
    // cannot drift. Lifecycle I/O (autosave, telemetry polling) stays in App::ui.
    fn screen_ui(
        &mut self,
        ui: &mut egui::Ui,
        status: &vc_app::EngineStatusSnapshot,
        devices: &vc_app::DeviceList,
    ) {
        ui_text::set_language(ui.ctx(), self.settings.language);
        self.pth_convert_window(ui.ctx());
        if self.onboarding.active() {
            if let Some(error) = &self.ui_error {
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    ui_text::diagnostic_message(self.settings.language, error),
                );
            }
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.download_status(ui);
                self.onboarding_ui(ui, status, devices);
            });
        } else {
            self.basic_ui(ui, status, devices);
        }
    }
}

impl eframe::App for VcGui {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop();
        if self.dirty_since.is_some() {
            if let Err(error) = save_settings(&self.settings) {
                eprintln!("Failed to save settings on exit: {error}");
            }
        }
    }
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if ui.ctx().input(|i| i.viewport().close_requested()) && self.dirty_since.is_some() {
            match save_settings(&self.settings) {
                Ok(()) => self.dirty_since = None,
                Err(error) => {
                    self.ui_error = Some(error);
                    ui.ctx()
                        .send_viewport_cmd(egui::ViewportCommand::CancelClose);
                }
            }
        }
        egui::Frame::new()
            .fill(ui.visuals().panel_fill)
            .inner_margin(20)
            .show(ui, |ui| {
                self.maybe_save();
                let (status, latest, devices) = self.controller.snapshot();
                if self.telemetry_updated_at.elapsed() >= TELEMETRY_REFRESH {
                    self.telemetry = latest;
                    self.telemetry_updated_at = Instant::now();
                }
                self.screen_ui(ui, &status, &devices);
                ui.ctx().request_repaint_after(Duration::from_millis(33));
            });
    }
}

fn format_content_delay(samples: Option<u64>, output_rate: u32) -> String {
    match (samples, output_rate) {
        (Some(samples), rate) if rate > 0 => {
            format!("{:.2} ms", samples as f64 * 1000.0 / rate as f64)
        }
        _ => "Unknown".to_string(),
    }
}

fn gpu_device_selector_visible(provider: &str) -> bool {
    // Capability lives on `Provider`; parse the stored string and ask it, so the
    // GUI and VST3 can't drift from the engine's notion of a GPU backend.
    GPU_DEVICE_SELECTOR_AVAILABLE
        && Provider::from_name(provider).is_some_and(Provider::shows_gpu_device_selector)
}

#[cfg(not(test))]
fn ensure_gpu_device_discovery(discovery: &Arc<Mutex<GpuDeviceDiscovery>>) {
    if let Ok(mut current) = discovery.lock() {
        if current.started {
            return;
        }
        current.started = true;
    } else {
        return;
    }

    let result = Arc::clone(discovery);
    if let Err(error) = std::thread::Builder::new()
        .name("vc-gui-gpu-discovery".to_string())
        .spawn(move || {
            let update = match list_cuda_devices() {
                Ok(devices) => GpuDeviceDiscovery {
                    started: true,
                    devices: Some(devices),
                    error: None,
                },
                Err(error) => GpuDeviceDiscovery {
                    started: true,
                    devices: None,
                    error: Some(format!("{error:#}")),
                },
            };
            if let Ok(mut current) = result.lock() {
                *current = update;
            }
        })
    {
        if let Ok(mut current) = discovery.lock() {
            current.error = Some(format!("failed to spawn GPU discovery thread: {error}"));
        }
    }
}

fn gpu_device_control(
    ui: &mut egui::Ui,
    selected_id: &mut u32,
    discovery: &Mutex<GpuDeviceDiscovery>,
) -> bool {
    let lang = ui_text::language(ui);
    let discovery = discovery
        .lock()
        .map(|value| value.clone())
        .unwrap_or_default();
    if let Some(devices) = discovery.devices {
        let selected_text = if devices.iter().any(|device| device.id == *selected_id) {
            gpu_device_label(*selected_id, &devices)
        } else {
            format!("{} {}", lang.text("Unavailable: device"), selected_id)
        };
        let mut changed = false;
        egui::ComboBox::new("GPU Device", lang.text("GPU Device"))
            .selected_text(selected_text)
            .show_ui(ui, |ui| {
                for device in devices {
                    changed |= ui
                        .selectable_value(
                            selected_id,
                            device.id,
                            format!("{}: {}", device.id, device.display_name),
                        )
                        .changed();
                }
            });
        changed
    } else if let Some(error) = discovery.error {
        let changed = ui
            .add(
                egui::DragValue::new(selected_id)
                    .prefix(lang.text("GPU Device ID: "))
                    .range(0..=i32::MAX as u32),
            )
            .changed();
        ui.small(format!("{}: {error}", lang.text("GPU enumeration failed")));
        changed
    } else {
        ui.label(lang.text("Detecting CUDA devices..."));
        false
    }
}

fn gpu_device_label(selected_id: u32, devices: &[GpuDevice]) -> String {
    devices
        .iter()
        .find(|device| device.id == selected_id)
        .map(|device| format!("{}: {}", device.id, device.display_name))
        .unwrap_or_else(|| format!("Unavailable: device {selected_id}"))
}

enum ModelKind {
    Rvc,
    Embedder,
    F0,
}

fn model_path_control(ui: &mut egui::Ui, label: &str, value: &mut String) -> (bool, bool) {
    let lang = ui_text::language(ui);
    let browse_clicked = ui
        .horizontal(|ui| {
            ui.label(label);
            ui.button(lang.text("Browse")).clicked()
        })
        .inner;
    let available_width = ui.available_width();
    let changed = ui
        .add(egui::TextEdit::singleline(value).desired_width(available_width))
        .changed();
    (changed, browse_clicked)
}

fn backend_combo(ui: &mut egui::Ui, label: &str, value: &mut String, changed: &mut bool) {
    let lang = ui_text::language(ui);
    // The stored value stays a canonical cpal HostId token (`wasapi`/`asio`/...)
    // for config + mapping stability; only the shown text is user-facing.
    egui::ComboBox::new(label, lang.text(label))
        .width((ui.available_width() - 120.0).clamp(90.0, 280.0))
        .truncate()
        .selected_text(gui_host_label(value))
        .show_ui(ui, |ui| {
            for name in gui_host_names() {
                *changed |= ui
                    .selectable_value(value, (*name).to_string(), gui_host_label(name))
                    .changed();
            }
        });
}

fn device_combo(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut String,
    names: &[String],
    recent: &mut Vec<String>,
    changed: &mut bool,
    show_label: bool,
) {
    let lang = ui_text::language(ui);
    let combo = if show_label {
        egui::ComboBox::new(label, lang.text(label))
    } else {
        egui::ComboBox::from_id_salt(label)
    };
    // An empty value follows the system default; only explicit device names
    // can be missing from the latest enumeration. Preserve the saved selection.
    let missing = !value.is_empty() && !names.contains(value);
    let mut selected = egui::RichText::new(if value.is_empty() {
        lang.text("System default")
    } else {
        value.as_str()
    });
    if missing {
        selected = selected.color(egui::Color32::LIGHT_RED);
    }
    let mut picked = None;
    combo.selected_text(selected).show_ui(ui, |ui| {
        *changed |= ui
            .selectable_value(value, String::new(), lang.text("System default"))
            .changed();
        let available_recent: Vec<_> = recent
            .iter()
            .filter(|name| !name.is_empty() && names.contains(name))
            .take(3)
            .collect();
        for name in &available_recent {
            let response = ui.selectable_value(value, (*name).clone(), name.as_str());
            *changed |= response.changed();
            if response.clicked() {
                picked = Some((*name).clone());
            }
        }
        if !available_recent.is_empty() {
            ui.separator();
        }
        for name in names.iter().filter(|name| !available_recent.contains(name)) {
            let response = ui.selectable_value(value, name.clone(), name);
            *changed |= response.changed();
            if response.clicked() {
                picked = Some(name.clone());
            }
        }
    });
    // Update after rendering: selecting closes the popup, so its rows never move
    // beneath the pointer. Disconnected names stay in history for reconnection.
    if let Some(name) = picked {
        remember_device(recent, name);
        *changed = true;
    }
}

fn remember_device(recent: &mut Vec<String>, name: String) {
    if name.is_empty() {
        return;
    }
    recent.retain(|previous| previous != &name && !previous.is_empty());
    recent.insert(0, name);
    recent.truncate(3);
}

fn metric(ui: &mut egui::Ui, label: &str, value: impl ToString) {
    let lang = ui_text::language(ui);
    ui.label(lang.text(label));
    ui.monospace(lang.text(&value.to_string()));
    ui.end_row();
}

fn colored_metric(ui: &mut egui::Ui, label: &str, value: impl ToString, color: egui::Color32) {
    ui.colored_label(color, ui_text::language(ui).text(label));
    ui.colored_label(color, egui::RichText::new(value.to_string()).monospace());
    ui.end_row();
}

fn inference_color(inference_us: u64, chunk_ms: u32) -> Option<egui::Color32> {
    let budget_us = u64::from(chunk_ms).saturating_mul(1_000);
    if inference_us > budget_us {
        Some(egui::Color32::LIGHT_RED)
    } else if inference_us.saturating_mul(5) >= budget_us.saturating_mul(4) {
        Some(egui::Color32::YELLOW)
    } else {
        None
    }
}

fn string_option(value: &str) -> Option<String> {
    (!value.trim().is_empty()).then(|| value.trim().to_string())
}

fn path_option(value: &str) -> Option<PathBuf> {
    string_option(value).map(PathBuf::from)
}

fn settings_path() -> Result<PathBuf, String> {
    std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .map(|dir| dir.join("vc-rs").join("gui.toml"))
        .ok_or_else(|| "APPDATA is not set; GUI settings cannot be persisted".to_string())
}

fn discover_support_models(settings: &mut GuiSettings) -> bool {
    let mut roots = Vec::new();
    if let Ok(dir) = model_setup::cache_dir() {
        roots.push(dir);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            roots.push(parent.join("assets"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd.join("assets"));
    }
    let embedder =
        model_setup::discover(&mut settings.embedder, model_setup::MODELS[0].file, &roots);
    let f0 = model_setup::discover(&mut settings.f0_model, model_setup::MODELS[1].file, &roots);
    let gtcrn = model_setup::discover_gtcrn(&mut settings.gtcrn_model_dir, &roots);
    embedder || f0 || gtcrn
}

fn load_settings() -> (GuiSettings, Option<String>) {
    let Ok(path) = settings_path() else {
        return (
            GuiSettings::default(),
            Some("APPDATA is not set".to_string()),
        );
    };
    if !path.exists() {
        return (GuiSettings::new_user(), None);
    }
    match fs::read_to_string(&path)
        .map_err(|e| e.to_string())
        .and_then(|s| toml::from_str(&s).map_err(|e| e.to_string()))
    {
        Ok(settings) => (settings, None),
        Err(err) => (
            GuiSettings::default(),
            Some(format!("Failed to load {}: {err}", path.display())),
        ),
    }
}

fn save_settings(settings: &GuiSettings) -> Result<(), String> {
    let path = settings_path()?;
    save_settings_at(settings, &path)
}

fn save_settings_at(settings: &GuiSettings, path: &Path) -> Result<(), String> {
    use std::io::Write;
    fs::create_dir_all(path.parent().unwrap())
        .map_err(|e| format!("Failed to create settings directory: {e}"))?;
    let text = toml::to_string_pretty(settings)
        .map_err(|e| format!("Failed to serialize settings: {e}"))?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let temporary = path.with_extension(format!("{}.{nonce}.tmp", std::process::id()));
    // A failed write must not destroy the previous tutorial/consent record.
    // Rename is on the same filesystem and happens only after flush succeeds.
    let result = (|| -> std::io::Result<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(|e| format!("Failed to save {}: {e}", path.display()))
}

fn parse_provider(value: &str) -> Result<Provider, String> {
    // Shared parser (canonical names + aliases) so the GUI accepts exactly what
    // the CLI and VST3 do, including the windowsml catalog EPs this used to omit.
    Provider::from_name(value).ok_or_else(|| format!("Unsupported provider: {value}"))
}

fn parse_gpu_priority(value: &str) -> Result<vc_core::model_rvc::GpuPriority, String> {
    match value {
        "normal" => Ok(vc_core::model_rvc::GpuPriority::Normal),
        "high" => Ok(vc_core::model_rvc::GpuPriority::High),
        _ => Err(format!("Unsupported GPU priority: {value}")),
    }
}

fn parse_denoiser(value: &str) -> Result<DenoiserMode, String> {
    match value {
        "off" => Ok(DenoiserMode::Off),
        "noise-gate" => Ok(DenoiserMode::NoiseGate),
        "rnnoise" => Ok(DenoiserMode::Rnnoise),
        "gtcrn" => Ok(DenoiserMode::Gtcrn),
        _ => Err(format!("Unsupported denoiser: {value}")),
    }
}

fn denoiser_names() -> &'static [&'static str] {
    &["off", "noise-gate", "rnnoise", "gtcrn"]
}

// Hosts the GUI exposes per direction, as cpal HostId tokens. Platform-gated to
// what cpal provides on this target; ASIO only appears with the `asio` feature.
// The bespoke WASAPI *exclusive* mode stays CLI-only (the GUI's "wasapi" is shared).
fn gui_host_names() -> &'static [&'static str] {
    #[cfg(windows)]
    {
        &[
            "wasapi",
            #[cfg(feature = "asio")]
            "asio",
        ]
    }
    #[cfg(target_os = "macos")]
    {
        &["coreaudio"]
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        &["alsa"]
    }
}

// Platform default host token (mirrors AudioHost::default() / cpal::default_host()).
fn default_host_token() -> &'static str {
    #[cfg(windows)]
    {
        "wasapi"
    }
    #[cfg(target_os = "macos")]
    {
        "coreaudio"
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        "alsa"
    }
}

fn parse_gui_host(name: &str) -> AudioHost {
    match name {
        "wasapi" => AudioHost::Wasapi,
        #[cfg(feature = "asio")]
        "asio" => AudioHost::Asio,
        "coreaudio" => AudioHost::CoreAudio,
        "alsa" => AudioHost::Alsa,
        _ => AudioHost::default(),
    }
}

// User-facing label for a host token; plain users do not know cpal HostId names.
fn gui_host_label(name: &str) -> &'static str {
    match name {
        "asio" => "ASIO",
        "coreaudio" => "Core Audio",
        "alsa" => "ALSA",
        "jack" => "JACK",
        _ => "WASAPI",
    }
}

// The engine's running status reads "Running (in: <host> / out: <host>)" with the
// canonical tokens. Present them with the same names as the selectors. Scoped to
// the "in:/out:" patterns so other messages are untouched.
fn friendly_status_message(message: &str) -> String {
    let mut text = message.to_string();
    for (raw, friendly) in [
        ("wasapi", "WASAPI"),
        ("asio", "ASIO"),
        ("coreaudio", "Core Audio"),
        ("alsa", "ALSA"),
        ("jack", "JACK"),
    ] {
        text = text
            .replace(&format!("in: {raw}"), &format!("in: {friendly}"))
            .replace(&format!("out: {raw}"), &format!("out: {friendly}"));
    }
    text
}

fn gpu_priority_names() -> &'static [&'static str] {
    &["high", "normal"]
}

// Shared with the CLI/VST3 via `vc_core::default_provider`, rendered as the
// config string the dropdown stores.
fn default_provider_name() -> &'static str {
    vc_core::default_provider().label()
}

fn gui_provider_visible(provider: Provider) -> bool {
    !(cfg!(feature = "windowsml") && provider == Provider::Cpu)
}

fn gui_provider_label(provider: &str) -> &str {
    match provider {
        "cpu" | "windowsml-cpu" => "CPU",
        _ => provider,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_history_survives_restart_and_keeps_three_unique_selections() {
        let mut settings: GuiSettings = toml::from_str("input_device = 'Old mic'").unwrap();
        assert!(settings.recent_input_devices.is_empty());
        for name in ["A", "B", "C", "D", "B", ""] {
            remember_device(&mut settings.recent_input_devices, name.into());
        }
        remember_device(&mut settings.recent_output_devices, "Speaker".into());
        let restored: GuiSettings = toml::from_str(&toml::to_string(&settings).unwrap()).unwrap();
        assert_eq!(restored.recent_input_devices, ["B", "D", "C"]);
        assert_eq!(restored.recent_output_devices, ["Speaker"]);
    }

    #[test]
    fn content_delay_display_keeps_unknown_distinct_from_zero() {
        assert_eq!(format_content_delay(None, 48_000), "Unknown");
        assert_eq!(format_content_delay(Some(240), 0), "Unknown");
        assert_eq!(format_content_delay(Some(0), 48_000), "0.00 ms");
        assert_eq!(format_content_delay(Some(240), 48_000), "5.00 ms");
        assert_eq!(format_content_delay(Some(1323), 44_100), "30.00 ms");
    }

    #[test]
    fn settings_toml_ignores_unknown_fields() {
        let settings: GuiSettings = toml::from_str("unknown = 1\npitch_shift = 2.5").unwrap();
        assert_eq!(settings.pitch_shift, 2.5);
        assert_eq!(settings.gpu_priority, "high");
    }

    #[test]
    fn legacy_noise_gate_setting_migrates_to_denoiser_mode() {
        let mut settings: GuiSettings =
            toml::from_str("noise_gate_enabled = true\npassthrough = true").unwrap();
        settings.normalize_gui_managed_settings();

        assert_eq!(settings.denoiser, "noise-gate");
        assert!(!settings.noise_gate_enabled);
        assert_eq!(
            settings.realtime().unwrap().denoiser_mode,
            DenoiserMode::NoiseGate
        );
    }

    #[test]
    fn status_message_shows_friendly_host_names() {
        assert_eq!(
            friendly_status_message("Running (in: wasapi / out: asio)"),
            "Running (in: WASAPI / out: ASIO)"
        );
        // Non-host messages pass through untouched.
        assert_eq!(
            friendly_status_message("Opening audio devices"),
            "Opening audio devices"
        );
    }

    #[test]
    fn gui_host_label_presents_friendly_names() {
        assert_eq!(gui_host_label("wasapi"), "WASAPI");
        assert_eq!(gui_host_label("asio"), "ASIO");
        assert_eq!(gui_host_label("coreaudio"), "Core Audio");
    }

    #[test]
    fn gui_gpu_priority_parses_and_normalizes() {
        assert_eq!(
            parse_gpu_priority("normal").unwrap(),
            vc_core::model_rvc::GpuPriority::Normal
        );
        let mut settings = GuiSettings {
            gpu_priority: "unsupported".to_string(),
            ..GuiSettings::default()
        };
        settings.normalize_gui_managed_settings();
        assert_eq!(settings.gpu_priority, "high");
    }

    #[test]
    fn gpu_device_label_preserves_unknown_saved_id() {
        let devices = vec![GpuDevice {
            id: 0,
            display_name: "NVIDIA Test GPU".to_string(),
        }];
        assert_eq!(gpu_device_label(0, &devices), "0: NVIDIA Test GPU");
        assert_eq!(gpu_device_label(7, &devices), "Unavailable: device 7");
    }

    #[test]
    fn gpu_device_selector_is_hidden_for_windows_ml_providers() {
        assert!(!gpu_device_selector_visible("windowsml"));
        assert!(!gpu_device_selector_visible("windowsml-directml"));
    }

    #[test]
    fn default_realtime_config_requires_models() {
        assert!(GuiSettings::default()
            .realtime()
            .unwrap()
            .validate()
            .is_err());
    }

    #[test]
    fn gui_realtime_config_forces_safe_audio_and_smoothing_settings() {
        let settings: GuiSettings = toml::from_str(
            r#"
input_host = "wasapi"
output_host = "wasapi"
wasapi_input_exclusive = true
wasapi_output_exclusive = true
wasapi_buffer_ms = 1
crossfade_ms = 1
sola_search_ms = 99
passthrough = true
"#,
        )
        .unwrap();

        let config = settings.realtime().unwrap();
        // WASAPI (shared) is a valid GUI host and is kept; the GUI only pins the
        // unsafe knobs — exclusive mode, buffer ms, and the smoothing timings.
        assert_eq!(config.input_host, AudioHost::Wasapi);
        assert_eq!(config.output_host, AudioHost::Wasapi);
        assert!(!config.wasapi_input_exclusive);
        assert!(!config.wasapi_output_exclusive);
        assert_eq!(config.wasapi_buffer_ms, 0);
        assert_eq!(config.crossfade_ms, GUI_CROSSFADE_MS);
        assert_eq!(config.sola_search_ms, GUI_SOLA_SEARCH_MS);
    }

    #[test]
    fn smoother_defaults_only_when_missing_and_preserves_saved_choices() {
        for (source, expected) in [
            ("", Smoother::PhaseVocoder),
            ("smoother = 'sola'", Smoother::Sola),
            ("smoother = 'psola'", Smoother::Psola),
            ("smoother = 'phase-vocoder'", Smoother::PhaseVocoder),
            ("smoother = 'unknown'", Smoother::Sola),
        ] {
            let mut settings: GuiSettings = toml::from_str(source).unwrap();
            settings.normalize_gui_managed_settings();
            assert_eq!(settings.realtime().unwrap().smoother, expected);
            let saved = toml::to_string(&settings).unwrap();
            let restored: GuiSettings = toml::from_str(&saved).unwrap();
            assert_eq!(restored.smoother, settings.smoother);
            assert_eq!(restored.realtime().unwrap().smoother, expected);
        }
    }

    #[test]
    fn normalization_removes_hidden_unsafe_gui_settings() {
        let mut settings = GuiSettings {
            // An unsupported/edited host token clamps to the platform default.
            input_host: "totally-invalid".to_string(),
            output_host: "totally-invalid".to_string(),
            wasapi_input_exclusive: true,
            wasapi_output_exclusive: true,
            wasapi_buffer_ms: 1,
            crossfade_ms: 1,
            sola_search_ms: 99,
            extra_convert_ms: 20,
            ..GuiSettings::default()
        };

        settings.normalize_gui_managed_settings();
        assert_eq!(settings.input_host, default_host_token());
        assert_eq!(settings.output_host, default_host_token());
        assert!(!settings.wasapi_input_exclusive);
        assert!(!settings.wasapi_output_exclusive);
        assert_eq!(settings.wasapi_buffer_ms, 0);
        assert_eq!(settings.crossfade_ms, GUI_CROSSFADE_MS);
        assert_eq!(settings.sola_search_ms, GUI_SOLA_SEARCH_MS);
        assert_eq!(settings.extra_convert_ms, GUI_MIN_EXTRA_CONVERT_MS);
    }

    #[test]
    fn persisted_chunk_is_preserved_and_rejected_when_off_rvc_frame_grid() {
        let mut settings = GuiSettings {
            model: "rvc.onnx".to_string(),
            embedder: "embedder.onnx".to_string(),
            f0_model: "f0.onnx".to_string(),
            chunk_ms: 25,
            ..Default::default()
        };
        settings.normalize_gui_managed_settings();
        assert_eq!(settings.chunk_ms, 25);
        let config = settings.realtime().unwrap();
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("multiple of 10"));
    }

    #[test]
    fn gui_realtime_rejects_extra_convert_below_gui_minimum() {
        let settings = GuiSettings {
            extra_convert_ms: GUI_MIN_EXTRA_CONVERT_MS - 1,
            passthrough: true,
            ..GuiSettings::default()
        };

        assert!(settings.realtime().is_err());
    }

    #[cfg(all(feature = "tensorrt", not(feature = "windowsml")))]
    #[test]
    fn tensorrt_only_gui_removes_cpu_provider() {
        // CPU has no ORT in the tensorrt-only build, so it is neither selectable
        // nor a valid persisted provider.
        assert!(!vc_core::selectable_providers().contains(&Provider::Cpu));
        assert!(!Provider::Cpu.available_in_build());
        let mut settings = GuiSettings {
            provider: "cpu".to_string(),
            ..GuiSettings::default()
        };
        settings.normalize_gui_managed_settings();
        assert_eq!(settings.provider, "tensorrt");
    }

    #[test]
    fn inference_color_warns_at_eighty_percent_and_errors_over_budget() {
        assert_eq!(inference_color(399_999, 500), None);
        assert_eq!(inference_color(400_000, 500), Some(egui::Color32::YELLOW));
        assert_eq!(inference_color(500_000, 500), Some(egui::Color32::YELLOW));
        assert_eq!(
            inference_color(500_001, 500),
            Some(egui::Color32::LIGHT_RED)
        );
    }
}
