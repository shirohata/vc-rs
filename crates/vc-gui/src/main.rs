use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::egui;
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;
use vc_app::{AudioBackend, EngineController, EngineState, LiveParams, RealtimeConfig, Smoother};
use vc_core::Provider;

const SAVE_DEBOUNCE: Duration = Duration::from_millis(500);

fn main() -> eframe::Result {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    eframe::run_native(
        "vc-rs",
        eframe::NativeOptions::default(),
        Box::new(|_| Ok(Box::new(VcGui::new()))),
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
struct GuiSettings {
    model: String,
    embedder: String,
    f0_model: String,
    provider: String,
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
            audio_backend: "cpal".to_string(),
            input_device: String::new(),
            output_device: String::new(),
            wasapi_input_exclusive: false,
            wasapi_output_exclusive: false,
            wasapi_buffer_ms: 0,
            chunk_ms: 500,
            crossfade_ms: 85,
            sola_search_ms: 12,
            smoother: "sola".to_string(),
            rvc_output_tail_discard_ms: 10,
            extra_convert_ms: 100,
            f0_threshold: 0.3,
            silence_threshold: 0.0001,
            pitch_shift: 0.0,
            speaker_id: 0,
            input_gain: 1.0,
            output_gain: 1.0,
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
    fn live(&self) -> LiveParams {
        LiveParams {
            pitch_shift: self.pitch_shift,
            speaker_id: self.speaker_id,
            input_gain: self.input_gain,
            output_gain: self.output_gain,
        }
    }

    fn realtime(&self) -> Result<RealtimeConfig, String> {
        Ok(RealtimeConfig {
            model: path_option(&self.model),
            embedder: path_option(&self.embedder),
            embedder_output: None,
            f0_model: path_option(&self.f0_model),
            provider: parse_provider(&self.provider)?,
            audio_backend: self.backend(),
            input_device: string_option(&self.input_device),
            output_device: string_option(&self.output_device),
            wasapi_input_exclusive: self.wasapi_input_exclusive,
            wasapi_output_exclusive: self.wasapi_output_exclusive,
            wasapi_buffer_ms: self.wasapi_buffer_ms,
            chunk_ms: self.chunk_ms,
            crossfade_ms: self.crossfade_ms,
            sola_search_ms: self.sola_search_ms,
            smoother: if self.smoother == "psola" {
                Smoother::Psola
            } else {
                Smoother::Sola
            },
            rvc_output_tail_discard_ms: self.rvc_output_tail_discard_ms,
            extra_convert_ms: self.extra_convert_ms,
            f0_threshold: self.f0_threshold,
            silence_threshold: self.silence_threshold,
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
        if self.audio_backend == "wasapi" {
            AudioBackend::Wasapi
        } else {
            AudioBackend::Cpal
        }
    }
}

struct VcGui {
    controller: EngineController,
    settings: GuiSettings,
    dirty_since: Option<Instant>,
    ui_error: Option<String>,
}

impl VcGui {
    fn new() -> Self {
        let (settings, ui_error) = load_settings();
        let controller = EngineController::new(settings.live());
        let _ = controller.refresh_devices(settings.backend());
        Self {
            controller,
            settings,
            dirty_since: None,
            ui_error,
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
}

impl eframe::App for VcGui {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.maybe_save();
        let (status, telemetry, devices) = self.controller.snapshot();
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
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            let mut changed = false;
            ui.heading("Models");
            changed |= path_row(ui, "RVC model", &mut self.settings.model);
            if ui.button("Browse RVC model").clicked() {
                self.browse_into(ModelKind::Rvc);
            }
            changed |= path_row(ui, "Embedder", &mut self.settings.embedder);
            if ui.button("Browse embedder").clicked() {
                self.browse_into(ModelKind::Embedder);
            }
            changed |= path_row(ui, "F0 model", &mut self.settings.f0_model);
            if ui.button("Browse F0 model").clicked() {
                self.browse_into(ModelKind::F0);
            }

            changed |= ui
                .checkbox(&mut self.settings.passthrough, "Passthrough (no models)")
                .changed();
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

            ui.separator();
            ui.heading("Audio");
            let old_backend = self.settings.audio_backend.clone();
            egui::ComboBox::from_label("Backend")
                .selected_text(&self.settings.audio_backend)
                .show_ui(ui, |ui| {
                    changed |= ui
                        .selectable_value(
                            &mut self.settings.audio_backend,
                            "cpal".to_string(),
                            "CPAL",
                        )
                        .changed();
                    changed |= ui
                        .selectable_value(
                            &mut self.settings.audio_backend,
                            "wasapi".to_string(),
                            "WASAPI",
                        )
                        .changed();
                });
            if old_backend != self.settings.audio_backend {
                let _ = self.controller.refresh_devices(self.settings.backend());
            }
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
            if self.settings.backend() == AudioBackend::Wasapi {
                changed |= ui
                    .checkbox(&mut self.settings.wasapi_input_exclusive, "Exclusive input")
                    .changed();
                changed |= ui
                    .checkbox(
                        &mut self.settings.wasapi_output_exclusive,
                        "Exclusive output",
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.settings.wasapi_buffer_ms, 0..=50)
                            .text("WASAPI buffer ms"),
                    )
                    .changed();
            }

            ui.separator();
            ui.heading("Engine configuration (Apply to restart)");
            changed |= ui
                .add(egui::Slider::new(&mut self.settings.chunk_ms, 50..=2000).text("Chunk ms"))
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.crossfade_ms, 0..=500)
                        .text("Crossfade ms"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.sola_search_ms, 0..=100)
                        .text("SOLA search ms"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.extra_convert_ms, 0..=1000)
                        .text("Extra convert ms"),
                )
                .changed();
            egui::ComboBox::from_label("Smoother")
                .selected_text(&self.settings.smoother)
                .show_ui(ui, |ui| {
                    changed |= ui
                        .selectable_value(&mut self.settings.smoother, "sola".to_string(), "SOLA")
                        .changed();
                    changed |= ui
                        .selectable_value(&mut self.settings.smoother, "psola".to_string(), "PSOLA")
                        .changed();
                });

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
                .add(egui::Slider::new(&mut self.settings.input_gain, 0.0..=4.0).text("Input gain"))
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.output_gain, 0.0..=4.0)
                        .text("Output gain"),
                )
                .changed();
            if changed {
                self.changed();
            }

            ui.horizontal(|ui| {
                if ui.button("Apply / Start").clicked() {
                    self.controller.set_live_params(self.settings.live());
                    match self.settings.realtime().and_then(|c| {
                        self.controller
                            .apply_config(c)
                            .map_err(|e| format!("{e:#}"))
                    }) {
                        Ok(()) => self.ui_error = None,
                        Err(err) => self.ui_error = Some(err),
                    }
                }
                if ui.button("Stop").clicked() {
                    if let Err(err) = self.controller.stop() {
                        self.ui_error = Some(format!("{err:#}"));
                    }
                }
            });

            ui.separator();
            ui.heading("Telemetry");
            egui::Grid::new("telemetry").show(ui, |ui| {
                metric(ui, "Chunks", telemetry.chunks);
                metric(ui, "Inference", format!("{} us", telemetry.inference_us));
                metric(ui, "Input RMS", format!("{:.6}", telemetry.input_rms));
                metric(ui, "Output RMS", format!("{:.6}", telemetry.output_rms));
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

fn path_row(ui: &mut egui::Ui, label: &str, value: &mut String) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.text_edit_singleline(value).changed()
    })
    .inner
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

fn provider_names() -> &'static [&'static str] {
    &[
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
    }

    #[test]
    fn default_realtime_config_requires_models() {
        assert!(GuiSettings::default()
            .realtime()
            .unwrap()
            .validate()
            .is_err());
    }
}
