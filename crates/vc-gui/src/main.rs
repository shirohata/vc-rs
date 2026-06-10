#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;
use vc_app::{
    AudioBackend, DenoiserMode, EngineController, EngineState, LiveParams, RealtimeConfig,
    Smoother, TelemetrySnapshot,
};
use vc_core::Provider;

const SAVE_DEBOUNCE: Duration = Duration::from_millis(500);
const TELEMETRY_REFRESH: Duration = Duration::from_millis(250);
const GUI_CROSSFADE_MS: u32 = 85;
const GUI_SOLA_SEARCH_MS: u32 = 12;
const RMS_HEALTHY_MIN: f32 = 0.01;
const RMS_HEALTHY_MAX: f32 = 0.10;
const RMS_HIGH_MAX: f32 = 0.25;

fn main() -> eframe::Result {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    eframe::run_native(
        "vc-rs",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            install_system_japanese_font(&cc.egui_ctx);
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
    model: String,
    embedder: String,
    f0_model: String,
    provider: String,
    gpu_priority: String,
    audio_backend: String,
    input_device: String,
    output_device: String,
    wasapi_input_exclusive: bool,
    wasapi_output_exclusive: bool,
    wasapi_buffer_ms: u32,
    chunk_ms: u32,
    crossfade_ms: u32,
    sola_search_ms: u32,
    smoother: String,
    rvc_output_tail_discard_ms: u32,
    extra_convert_ms: u32,
    f0_threshold: f32,
    silence_threshold: f32,
    pitch_shift: f32,
    speaker_id: i64,
    input_gain: f32,
    output_gain: f32,
    denoiser: String,
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
            model: String::new(),
            embedder: String::new(),
            f0_model: String::new(),
            provider: default_provider_name().to_string(),
            gpu_priority: "high".to_string(),
            audio_backend: "cpal".to_string(),
            input_device: String::new(),
            output_device: String::new(),
            wasapi_input_exclusive: false,
            wasapi_output_exclusive: false,
            wasapi_buffer_ms: 0,
            chunk_ms: 500,
            crossfade_ms: GUI_CROSSFADE_MS,
            sola_search_ms: GUI_SOLA_SEARCH_MS,
            smoother: "sola".to_string(),
            rvc_output_tail_discard_ms: 10,
            extra_convert_ms: 100,
            f0_threshold: 0.3,
            silence_threshold: 0.0001,
            pitch_shift: 0.0,
            speaker_id: 0,
            input_gain: 1.0,
            output_gain: 1.0,
            denoiser: "off".to_string(),
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
        // WASAPI and these smoothing timings remain available to the CLI, but
        // the GUI intentionally pins them until their safe tuning and failure
        // behavior are clear enough to expose to general users.
        self.audio_backend = "cpal".to_string();
        self.wasapi_input_exclusive = false;
        self.wasapi_output_exclusive = false;
        self.wasapi_buffer_ms = 0;
        self.crossfade_ms = GUI_CROSSFADE_MS;
        self.sola_search_ms = GUI_SOLA_SEARCH_MS;
        if !provider_names().contains(&self.provider.as_str()) {
            self.provider = default_provider_name().to_string();
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
            noise_gate_threshold: self.noise_gate_threshold,
        }
    }

    fn realtime(&self) -> Result<RealtimeConfig, String> {
        Ok(RealtimeConfig {
            model: path_option(&self.model),
            embedder: path_option(&self.embedder),
            embedder_output: None,
            f0_model: path_option(&self.f0_model),
            provider: parse_provider(&self.provider)?,
            gpu_priority: parse_gpu_priority(&self.gpu_priority)?,
            audio_backend: AudioBackend::Cpal,
            input_device: string_option(&self.input_device),
            output_device: string_option(&self.output_device),
            wasapi_input_exclusive: false,
            wasapi_output_exclusive: false,
            wasapi_buffer_ms: 0,
            chunk_ms: self.chunk_ms,
            crossfade_ms: GUI_CROSSFADE_MS,
            sola_search_ms: GUI_SOLA_SEARCH_MS,
            smoother: if self.smoother == "psola" {
                Smoother::Psola
            } else {
                Smoother::Sola
            },
            rvc_output_tail_discard_ms: self.rvc_output_tail_discard_ms,
            extra_convert_ms: self.extra_convert_ms,
            f0_threshold: self.f0_threshold,
            silence_threshold: self.silence_threshold,
            denoiser_mode: parse_denoiser(&self.denoiser)?,
            noise_gate_attack_ms: self.noise_gate_attack_ms,
            noise_gate_release_ms: self.noise_gate_release_ms,
            noise_gate_floor: self.noise_gate_floor,
            volume_envelope: self.volume_envelope,
            rms_mix_rate: self.rms_mix_rate,
            auto_output_gain: self.auto_output_gain,
            target_output_rms: self.target_output_rms,
            max_output_gain: self.max_output_gain,
            passthrough: self.passthrough,
            debug_input_wav: None,
            debug_output_wav: None,
        })
    }

    fn backend(&self) -> AudioBackend {
        AudioBackend::Cpal
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
}

impl VcGui {
    fn new() -> Self {
        let (mut settings, ui_error) = load_settings();
        settings.normalize_gui_managed_settings();
        let controller = EngineController::new(settings.live());
        let _ = controller.refresh_devices(settings.backend());
        Self {
            controller,
            settings,
            dirty_since: None,
            ui_error,
            telemetry: TelemetrySnapshot::default(),
            telemetry_updated_at: Instant::now() - TELEMETRY_REFRESH,
            applied_chunk_ms: None,
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
            self.dirty_since = None;
            if let Err(err) = save_settings(&self.settings) {
                self.ui_error = Some(err);
            }
        }
    }

    fn browse_into(&mut self, kind: ModelKind) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("ONNX model", &["onnx"])
            .pick_file()
        {
            let value = path.to_string_lossy().into_owned();
            match kind {
                ModelKind::Rvc => self.settings.model = value,
                ModelKind::Embedder => self.settings.embedder = value,
                ModelKind::F0 => self.settings.f0_model = value,
            }
            self.changed();
        }
    }

    fn apply_or_start(&mut self) {
        self.controller.set_live_params(self.settings.live());
        let chunk_ms = self.settings.chunk_ms;
        match self.settings.realtime().and_then(|config| {
            self.controller
                .apply_config(config)
                .map_err(|e| format!("{e:#}"))
        }) {
            Ok(()) => {
                self.ui_error = None;
                self.applied_chunk_ms = Some(chunk_ms);
            }
            Err(err) => self.ui_error = Some(err),
        }
    }

    fn stop(&mut self) {
        if let Err(err) = self.controller.stop() {
            self.ui_error = Some(format!("{err:#}"));
        } else {
            self.applied_chunk_ms = None;
        }
    }
}

impl eframe::App for VcGui {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.maybe_save();
        let (status, latest_telemetry, devices) = self.controller.snapshot();
        if self.telemetry_updated_at.elapsed() >= TELEMETRY_REFRESH {
            self.telemetry = latest_telemetry;
            self.telemetry_updated_at = Instant::now();
        }
        let telemetry = self.telemetry;
        ui.heading("vc-rs Standalone");
        ui.horizontal(|ui| {
            ui.label(format!("Status: {:?} - {}", status.state, status.message));
            if status.state == EngineState::Running {
                ui.label(format!(
                    "{} Hz -> {} Hz",
                    status.input_sample_rate, status.output_sample_rate
                ));
            }
        });
        if let Some(error) = &self.ui_error {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }
        if let Some(error) = &devices.error {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }
        ui.horizontal(|ui| {
            if ui.button("Apply / Start").clicked() {
                self.apply_or_start();
            }
            if ui.button("Stop").clicked() {
                self.stop();
            }
            if ui
                .checkbox(&mut self.settings.passthrough, "Passthrough")
                .changed()
            {
                self.changed();
            }
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            let mut changed = false;
            ui.heading("Models");
            let (path_changed, browse_clicked) =
                model_path_control(ui, "RVC model", &mut self.settings.model);
            changed |= path_changed;
            if browse_clicked {
                self.browse_into(ModelKind::Rvc);
            }
            let (path_changed, browse_clicked) =
                model_path_control(ui, "Embedder", &mut self.settings.embedder);
            changed |= path_changed;
            if browse_clicked {
                self.browse_into(ModelKind::Embedder);
            }
            let (path_changed, browse_clicked) =
                model_path_control(ui, "F0 model", &mut self.settings.f0_model);
            changed |= path_changed;
            if browse_clicked {
                self.browse_into(ModelKind::F0);
            }

            egui::ComboBox::from_label("Provider")
                .selected_text(&self.settings.provider)
                .show_ui(ui, |ui| {
                    for provider in provider_names() {
                        changed |= ui
                            .selectable_value(
                                &mut self.settings.provider,
                                provider.to_string(),
                                *provider,
                            )
                            .changed();
                    }
                });
            egui::ComboBox::from_label("GPU Priority")
                .selected_text(&self.settings.gpu_priority)
                .show_ui(ui, |ui| {
                    for priority in gpu_priority_names() {
                        changed |= ui
                            .selectable_value(
                                &mut self.settings.gpu_priority,
                                priority.to_string(),
                                *priority,
                            )
                            .changed();
                    }
                });

            ui.separator();
            ui.heading("Audio");
            if ui.button("Refresh devices").clicked() {
                let _ = self.controller.refresh_devices(self.settings.backend());
            }
            device_combo(
                ui,
                "Input device",
                &mut self.settings.input_device,
                &devices.inputs,
                &mut changed,
            );
            device_combo(
                ui,
                "Output device",
                &mut self.settings.output_device,
                &devices.outputs,
                &mut changed,
            );

            ui.separator();
            ui.heading("Engine configuration (Apply to restart)");
            changed |= ui
                .add(egui::Slider::new(&mut self.settings.chunk_ms, 50..=1000).text("Chunk ms"))
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.extra_convert_ms, 0..=2000)
                        .text("Extra convert ms"),
                )
                .changed();

            ui.separator();
            ui.heading("Live parameters");
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.pitch_shift, -24.0..=24.0)
                        .text("Pitch shift"),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.settings.speaker_id, 0..=255).text("Speaker ID"))
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.input_gain, 0.0..=12.0).text("Input gain"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.output_gain, 0.0..=12.0)
                        .text("Output gain"),
                )
                .changed();
            egui::ComboBox::from_label("Input denoiser")
                .selected_text(&self.settings.denoiser)
                .show_ui(ui, |ui| {
                    for denoiser in denoiser_names() {
                        changed |= ui
                            .selectable_value(
                                &mut self.settings.denoiser,
                                denoiser.to_string(),
                                *denoiser,
                            )
                            .changed();
                    }
                });
            if self.settings.denoiser == "noise-gate" {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.settings.noise_gate_threshold, 0.0001..=0.5)
                            .logarithmic(true)
                            .text("Gate threshold"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.settings.noise_gate_attack_ms, 0.0..=200.0)
                            .text("Gate attack (ms)"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.settings.noise_gate_release_ms, 0.0..=1000.0)
                            .text("Gate release (ms)"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.settings.noise_gate_floor, 0.0..=1.0)
                            .text("Gate floor"),
                    )
                    .changed();
            }
            if changed {
                self.changed();
            }
            ui.separator();
            ui.heading("Telemetry");
            egui::Grid::new("telemetry").show(ui, |ui| {
                let inference_ms = telemetry.inference_us.saturating_add(500) / 1_000;
                let inference_color = (status.state == EngineState::Running)
                    .then(|| {
                        self.applied_chunk_ms
                            .and_then(|chunk_ms| inference_color(telemetry.inference_us, chunk_ms))
                    })
                    .flatten();
                if let Some(color) = inference_color {
                    colored_metric(ui, "Inference", format!("{inference_ms} ms"), color);
                } else {
                    metric(ui, "Inference", format!("{inference_ms} ms"));
                }
                rms_metric(ui, "Input RMS", telemetry.input_rms);
                rms_metric(ui, "Output RMS", telemetry.output_rms);
                metric(ui, "Input overruns", telemetry.input_overruns);
                metric(ui, "Output underruns", telemetry.output_underruns);
                metric(
                    ui,
                    "Dropped output samples",
                    telemetry.output_dropped_samples,
                );
                metric(
                    ui,
                    "Output buffered samples",
                    telemetry.output_buffer_samples,
                );
            });
        });
        ui.ctx().request_repaint_after(Duration::from_millis(33));
    }
}

enum ModelKind {
    Rvc,
    Embedder,
    F0,
}

fn model_path_control(ui: &mut egui::Ui, label: &str, value: &mut String) -> (bool, bool) {
    let browse_clicked = ui
        .horizontal(|ui| {
            ui.label(label);
            ui.button("Browse").clicked()
        })
        .inner;
    let available_width = ui.available_width();
    let changed = ui
        .add(egui::TextEdit::singleline(value).desired_width(available_width))
        .changed();
    (changed, browse_clicked)
}

fn device_combo(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut String,
    names: &[String],
    changed: &mut bool,
) {
    egui::ComboBox::from_label(label)
        .selected_text(if value.is_empty() {
            "System default"
        } else {
            value.as_str()
        })
        .show_ui(ui, |ui| {
            *changed |= ui
                .selectable_value(value, String::new(), "System default")
                .changed();
            for name in names {
                *changed |= ui.selectable_value(value, name.clone(), name).changed();
            }
        });
}

fn metric(ui: &mut egui::Ui, label: &str, value: impl ToString) {
    ui.label(label);
    ui.monospace(value.to_string());
    ui.end_row();
}

fn colored_metric(ui: &mut egui::Ui, label: &str, value: impl ToString, color: egui::Color32) {
    ui.colored_label(color, label);
    ui.colored_label(color, egui::RichText::new(value.to_string()).monospace());
    ui.end_row();
}

fn rms_metric(ui: &mut egui::Ui, label: &str, rms: f32) {
    colored_metric(ui, label, format!("{rms:.6}"), rms_color(rms));
}

fn rms_color(rms: f32) -> egui::Color32 {
    if !rms.is_finite() || rms > RMS_HIGH_MAX {
        egui::Color32::LIGHT_RED
    } else if rms < RMS_HEALTHY_MIN {
        egui::Color32::GRAY
    } else if rms > RMS_HEALTHY_MAX {
        egui::Color32::YELLOW
    } else {
        egui::Color32::LIGHT_GREEN
    }
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

fn load_settings() -> (GuiSettings, Option<String>) {
    let Ok(path) = settings_path() else {
        return (
            GuiSettings::default(),
            Some("APPDATA is not set".to_string()),
        );
    };
    if !path.exists() {
        return (GuiSettings::default(), None);
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
    fs::create_dir_all(path.parent().unwrap())
        .map_err(|e| format!("Failed to create settings directory: {e}"))?;
    let text = toml::to_string_pretty(settings)
        .map_err(|e| format!("Failed to serialize settings: {e}"))?;
    fs::write(&path, text).map_err(|e| format!("Failed to save {}: {e}", path.display()))
}

fn parse_provider(value: &str) -> Result<Provider, String> {
    match value {
        "cpu" => Ok(Provider::Cpu),
        "cuda" => Ok(Provider::Cuda),
        "tensorrt" => Ok(Provider::TensorRt),
        "windowsml" => Ok(Provider::WindowsMl),
        "windowsml-cpu" => Ok(Provider::WindowsMlCpu),
        "windowsml-directml" => Ok(Provider::WindowsMlDirectMl),
        "windowsml-nvtrtx" => Ok(Provider::WindowsMlNvTensorRtRtx),
        _ => Err(format!("Unsupported provider: {value}")),
    }
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
        _ => Err(format!("Unsupported denoiser: {value}")),
    }
}

fn denoiser_names() -> &'static [&'static str] {
    &["off", "noise-gate", "rnnoise"]
}

fn gpu_priority_names() -> &'static [&'static str] {
    &["high", "normal"]
}

fn provider_names() -> &'static [&'static str] {
    &[
        #[cfg(not(all(feature = "tensorrt", not(feature = "windowsml"))))]
        "cpu",
        #[cfg(feature = "cuda")]
        "cuda",
        #[cfg(feature = "tensorrt")]
        "tensorrt",
        #[cfg(feature = "windowsml")]
        "windowsml",
        #[cfg(feature = "windowsml")]
        "windowsml-cpu",
        #[cfg(feature = "windowsml")]
        "windowsml-directml",
        #[cfg(feature = "windowsml")]
        "windowsml-nvtrtx",
    ]
}

#[cfg(all(feature = "windowsml", not(feature = "tensorrt")))]
fn default_provider_name() -> &'static str {
    "windowsml"
}

#[cfg(all(feature = "tensorrt", not(feature = "windowsml")))]
fn default_provider_name() -> &'static str {
    "tensorrt"
}

#[cfg(not(any(
    all(feature = "windowsml", not(feature = "tensorrt")),
    all(feature = "tensorrt", not(feature = "windowsml")),
)))]
fn default_provider_name() -> &'static str {
    "cpu"
}

#[cfg(test)]
mod tests {
    use super::*;

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
audio_backend = "wasapi"
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
        assert_eq!(config.audio_backend, AudioBackend::Cpal);
        assert!(!config.wasapi_input_exclusive);
        assert!(!config.wasapi_output_exclusive);
        assert_eq!(config.wasapi_buffer_ms, 0);
        assert_eq!(config.crossfade_ms, GUI_CROSSFADE_MS);
        assert_eq!(config.sola_search_ms, GUI_SOLA_SEARCH_MS);
    }

    #[test]
    fn normalization_removes_hidden_unsafe_gui_settings() {
        let mut settings = GuiSettings {
            audio_backend: "wasapi".to_string(),
            wasapi_input_exclusive: true,
            wasapi_output_exclusive: true,
            wasapi_buffer_ms: 1,
            crossfade_ms: 1,
            sola_search_ms: 99,
            ..GuiSettings::default()
        };

        settings.normalize_gui_managed_settings();
        assert_eq!(settings.audio_backend, "cpal");
        assert!(!settings.wasapi_input_exclusive);
        assert!(!settings.wasapi_output_exclusive);
        assert_eq!(settings.wasapi_buffer_ms, 0);
        assert_eq!(settings.crossfade_ms, GUI_CROSSFADE_MS);
        assert_eq!(settings.sola_search_ms, GUI_SOLA_SEARCH_MS);
    }

    #[cfg(all(feature = "tensorrt", not(feature = "windowsml")))]
    #[test]
    fn tensorrt_only_gui_removes_cpu_provider() {
        assert!(!provider_names().contains(&"cpu"));
        let mut settings = GuiSettings {
            provider: "cpu".to_string(),
            ..GuiSettings::default()
        };
        settings.normalize_gui_managed_settings();
        assert_eq!(settings.provider, "tensorrt");
    }

    #[test]
    fn rms_colors_distinguish_silence_healthy_and_excessive_levels() {
        assert_eq!(rms_color(0.0), egui::Color32::GRAY);
        assert_eq!(rms_color(0.005), egui::Color32::GRAY);
        assert_eq!(rms_color(0.03), egui::Color32::LIGHT_GREEN);
        assert_eq!(rms_color(0.15), egui::Color32::YELLOW);
        assert_eq!(rms_color(0.30), egui::Color32::LIGHT_RED);
        assert_eq!(rms_color(f32::NAN), egui::Color32::LIGHT_RED);
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
