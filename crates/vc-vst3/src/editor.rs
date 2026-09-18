//! Minimal egui editor: pick model files, choose the backend, watch status,
//! and tweak the live parameters.
//!
//! Apply model is **manual**: editing the model paths or backend only stages
//! the change into the persisted `settings` and marks it `dirty`; nothing is
//! rebuilt until the user presses **Load / Reload** (which sets `reload`). This
//! keeps the expensive/fragile ONNX Runtime (re)initialisation under explicit
//! user control instead of firing on every edit.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use egui::{self, Vec2};
use nice_plug::prelude::{Editor, ParamSetter};
use nice_plug_egui::{create_egui_editor, resizable_window::ResizableWindow, widgets};
use vc_core::gpu::{list_cuda_devices, GpuDevice};
use vc_core::validation::CONVERSION_TIMING_LIMITS;
use vc_core::Provider;

use crate::config::PLUGIN_MIN_EXTRA_CONVERT_MS;
use crate::params::VcRvcParams;
use crate::plugin_identity;
use crate::runtime::{PluginStatus, ReloadWaker, MAX_CHUNK_MS, MIN_CHUNK_MS};

/// Granularity for the millisecond sliders.
const MS_STEP: u32 = 10;
const MIN_EXTRA_CONVERT_MS: u32 = PLUGIN_MIN_EXTRA_CONVERT_MS;
const MAX_EXTRA_CONVERT_MS: u32 = CONVERSION_TIMING_LIMITS.max_extra_convert_ms;
const GPU_DEVICE_SELECTOR_AVAILABLE: bool = cfg!(any(feature = "cuda", feature = "tensorrt"));

/// Shared state handed to the egui update closure.
pub struct EditorState {
    pub params: Arc<VcRvcParams>,
    /// Set by the **Load / Reload** button to request a pipeline rebuild.
    pub reload: Arc<AtomicBool>,
    /// True while the worker is rebuilding the pipeline. Prevents duplicate
    /// reload requests without delaying the first request.
    pub loading: Arc<AtomicBool>,
    /// Set when settings are edited, cleared by the worker once it applies them.
    /// Drives the "unapplied changes" indicator.
    pub dirty: Arc<AtomicBool>,
    /// Worker status shown in the UI.
    pub status: Arc<Mutex<PluginStatus>>,
    /// Wakes the worker the instant a reload is submitted, so Load / Reload
    /// applies even when the host is idle and not calling `process()`.
    pub reload_waker: Arc<ReloadWaker>,
    /// Populated only after the editor opens. CUDA discovery must never run
    /// during plugin scan, project restore, or from the audio callback.
    gpu_devices: Arc<Mutex<GpuDeviceDiscovery>>,
    gpu_discovery_thread: Option<std::thread::JoinHandle<()>>,
    openvino_devices: vc_core::openvino::DeviceDiscovery,
}

#[derive(Clone, Debug, Default)]
struct GpuDeviceDiscovery {
    devices: Option<Vec<GpuDevice>>,
    error: Option<String>,
}

pub fn create(
    params: Arc<VcRvcParams>,
    reload: Arc<AtomicBool>,
    loading: Arc<AtomicBool>,
    dirty: Arc<AtomicBool>,
    status: Arc<Mutex<PluginStatus>>,
    reload_waker: Arc<ReloadWaker>,
) -> Option<Box<dyn Editor>> {
    let egui_state = params.editor_state.clone();
    let (gpu_devices, gpu_discovery_thread) = spawn_gpu_device_discovery();
    create_egui_editor(
        egui_state,
        EditorState {
            params,
            reload,
            loading,
            dirty,
            status,
            reload_waker,
            gpu_devices,
            gpu_discovery_thread,
            openvino_devices: vc_core::openvino::DeviceDiscovery::default(),
        },
        Default::default(),
        |_, _, _| {},
        |ui, setter, _queue, state| draw(ui, setter, state),
    )
}

impl Drop for EditorState {
    fn drop(&mut self) {
        if let Some(thread) = self.gpu_discovery_thread.take() {
            let _ = thread.join();
        }
    }
}

fn draw(ui: &mut egui::Ui, setter: &ParamSetter, state: &mut EditorState) {
    let egui_state = state.params.editor_state.clone();
    ResizableWindow::new("vc-rs-rvc-editor")
        .min_size(Vec2::new(440.0, 380.0))
        .show(ui, egui_state.as_ref(), |ui| {
            // Hosts restore persisted editor sizes, and some DPI/host combinations
            // leave less usable space than requested. Keep the content reachable
            // even when the host cannot or will not grow the outer plugin view.
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| draw_contents(ui, setter, state));
        });
}

fn draw_contents(ui: &mut egui::Ui, setter: &ParamSetter, state: &mut EditorState) {
    ui.heading(plugin_identity::NAME);

    // Status line (updated by the worker thread).
    let status = state
        .status
        .lock()
        .map(|s| s.clone())
        .unwrap_or_else(|_| PluginStatus::new("status unavailable"));
    ui.label(format!("Status: {}", status.summary));
    if let Some(detail) = status.detail {
        egui::CollapsingHeader::new("Error details")
            .default_open(false)
            .show(ui, |ui| {
                ui.monospace(detail);
            });
    }
    ui.separator();

    ui.heading("Live controls");
    ui.small("Changes apply immediately and can be automated in your DAW.");
    egui::Grid::new("params").num_columns(2).show(ui, |ui| {
        ui.label("Pitch");
        ui.add(widgets::ParamSlider::for_param(
            &state.params.pitch_shift,
            setter,
        ));
        ui.end_row();
        ui.label("Speaker");
        ui.add(widgets::ParamSlider::for_param(
            &state.params.speaker_id,
            setter,
        ));
        ui.end_row();
        ui.label("Input gain");
        ui.add(widgets::ParamSlider::for_param(
            &state.params.input_gain_db,
            setter,
        ));
        ui.end_row();
        ui.label("Output gain");
        ui.add(widgets::ParamSlider::for_param(
            &state.params.output_gain_db,
            setter,
        ));
        ui.end_row();
        ui.label("Noise gate");
        ui.add(widgets::ParamSlider::for_param(
            &state.params.noise_gate,
            setter,
        ));
        ui.end_row();
        ui.label("Gate threshold");
        ui.add(widgets::ParamSlider::for_param(
            &state.params.noise_gate_threshold_db,
            setter,
        ));
        ui.end_row();
    });

    ui.separator();
    ui.heading("Setup");
    ui.small("Models, backend, chunk, Extra convert, and gate timing apply on Load / Reload.");
    ui.horizontal_wrapped(|ui| {
        let loading = state.loading.load(Ordering::SeqCst);
        if ui
            .add_enabled(!loading, egui::Button::new("Load / Reload"))
            .clicked()
            && state
                .loading
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            state.reload.store(true, Ordering::SeqCst);
            // Wake the worker so it picks up the request immediately, even if the
            // host is idle and not driving process() (no input wakes arriving).
            state.reload_waker.wake();
        }
        if state.dirty.load(Ordering::Relaxed) {
            ui.colored_label(egui::Color32::from_rgb(220, 180, 60), "Unapplied changes");
        }
    });

    // New instances expose setup; configured instances prioritize live controls.
    // The apply action stays outside the fold so pending edits remain actionable.
    let needs_models = {
        let settings = state.params.settings.read().unwrap();
        settings.model.as_os_str().is_empty()
            || settings.embedder.as_os_str().is_empty()
            || settings.f0_model.as_os_str().is_empty()
    };
    egui::CollapsingHeader::new("Models and conversion settings")
        .default_open(needs_models)
        .show(ui, |ui| {
            draw_staged_settings(ui, state);
        });
}

fn draw_staged_settings(ui: &mut egui::Ui, state: &mut EditorState) {
    // Snapshot current settings for display, releasing the lock immediately.
    let (model, embedder, f0_model, provider, gpu_device_id) = {
        let s = state.params.settings.read().unwrap();
        (
            s.model.clone(),
            s.embedder.clone(),
            s.f0_model.clone(),
            s.provider.clone(),
            s.gpu_device_id,
        )
    };

    ui.label("Models (.onnx)");
    if file_row(ui, "RVC model", &model) {
        spawn_picker(state, ModelKind::Rvc);
    }
    if file_row(ui, "Embedder", &embedder) {
        spawn_picker(state, ModelKind::Embedder);
    }
    if file_row(ui, "F0 (RMVPE)", &f0_model) {
        spawn_picker(state, ModelKind::F0);
    }

    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.label("Backend");
        let mut selected_provider =
            Provider::from_name(&provider).unwrap_or(vc_core::default_provider());
        let mut backend = selected_provider.backend();
        let mut provider_changed = false;
        egui::ComboBox::from_id_salt("provider")
            .selected_text(backend.label().to_uppercase())
            .show_ui(ui, |ui| {
                // Build's base backends plus the host device's live Windows ML
                // catalog EPs (cached in vc-core), so the package offers what is
                // actually usable rather than a fixed per-build list.
                for candidate in vc_core::selectable_providers()
                    .into_iter()
                    .filter(|p| p.backend() == *p)
                {
                    let option = candidate.label();
                    if ui
                        .selectable_value(&mut backend, candidate, option.to_uppercase())
                        .changed()
                    {
                        selected_provider = if backend == Provider::WindowsMlOpenVino {
                            Provider::WindowsMlOpenVinoCpu
                        } else {
                            backend
                        };
                        provider_changed = true;
                    }
                }
            });
        if let Some(label) = selected_provider.openvino_device_label() {
            state.openvino_devices.poll();
            let availability = &state.openvino_devices.status;
            if availability.show_picker() {
                ui.label("Device");
                egui::ComboBox::from_id_salt("openvino-device")
                    .selected_text(if selected_provider == Provider::WindowsMlOpenVino {
                        label.to_owned()
                    } else {
                        format!("{label} ({})", availability.label(selected_provider))
                    })
                    .show_ui(ui, |ui| {
                        for &candidate in Provider::OPENVINO_DEVICES {
                            provider_changed |= ui
                                .add_enabled_ui(
                                    availability.availability(candidate) != Some(false),
                                    |ui| {
                                        ui.selectable_value(
                                            &mut selected_provider,
                                            candidate,
                                            format!(
                                                "{} ({})",
                                                candidate.openvino_device_label().unwrap(),
                                                availability.label(candidate)
                                            ),
                                        )
                                    },
                                )
                                .inner
                                .changed();
                        }
                    });
            }
            let mut download = false;
            let mut retry = false;
            if matches!(
                availability,
                vc_core::openvino::DeviceStatus::DownloadRequired
            ) {
                ui.small("Download OpenVINO to check available devices.");
                download = ui.button("Download and check").clicked();
            }
            if matches!(availability, vc_core::openvino::DeviceStatus::Downloading) {
                ui.spinner();
                ui.small("Downloading and preparing OpenVINO...");
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(100));
            }
            if matches!(
                availability,
                vc_core::openvino::DeviceStatus::Checking
                    | vc_core::openvino::DeviceStatus::NotStarted
            ) {
                ui.small("Checking OpenVINO devices...");
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(100));
            }
            if let vc_core::openvino::DeviceStatus::Unknown(error) = availability {
                ui.small("Could not verify OpenVINO devices. Hover for details.")
                    .on_hover_text(error);
                retry = ui.button("Retry device check").clicked();
            }
            if availability.availability(selected_provider) == Some(false) {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "Selected device is unavailable. Choose an available device.",
                );
            }
            if download {
                state.openvino_devices.download_and_check();
            } else if retry {
                state.openvino_devices.retry();
            }
        }
        if provider_changed {
            state.params.settings.write().unwrap().provider = selected_provider.label().to_owned();
            mark_dirty(state);
        }
        if gpu_device_selector_visible(selected_provider.label()) {
            let mut selected_gpu_device_id = gpu_device_id;
            if gpu_device_control(ui, &mut selected_gpu_device_id, &state.gpu_devices) {
                state.params.settings.write().unwrap().gpu_device_id = selected_gpu_device_id;
                mark_dirty(state);
            }
        }
    });

    // Latency / context sliders (10 ms steps). Applied on Load / Reload.
    let (chunk_ms, extra_convert_ms) = {
        let s = state.params.settings.read().unwrap();
        (s.chunk_ms, s.extra_convert_ms)
    };
    if let Some(v) = ms_slider(ui, "Chunk", chunk_ms, MIN_CHUNK_MS, MAX_CHUNK_MS) {
        state.params.settings.write().unwrap().chunk_ms = v;
        mark_dirty(state);
    }
    if let Some(v) = ms_slider(
        ui,
        "Extra convert",
        extra_convert_ms,
        MIN_EXTRA_CONVERT_MS,
        MAX_EXTRA_CONVERT_MS,
    ) {
        state.params.settings.write().unwrap().extra_convert_ms = v;
        mark_dirty(state);
    }
    ui.small("Chunk = latency vs. context. Extra convert = extra model context.");

    // Keep these controls in the staged section: unlike the gate toggle and
    // threshold, attack/release/floor require a worker reload.
    ui.separator();
    ui.label("Gate timing");
    let (gate_attack_ms, gate_release_ms, gate_floor) = {
        let s = state.params.settings.read().unwrap();
        (
            s.noise_gate_attack_ms,
            s.noise_gate_release_ms,
            s.noise_gate_floor,
        )
    };
    if let Some(v) = f32_slider(ui, "Gate attack", gate_attack_ms, 0.0, 200.0, " ms") {
        state.params.settings.write().unwrap().noise_gate_attack_ms = v;
        mark_dirty(state);
    }
    if let Some(v) = f32_slider(ui, "Gate release", gate_release_ms, 0.0, 1000.0, " ms") {
        state.params.settings.write().unwrap().noise_gate_release_ms = v;
        mark_dirty(state);
    }
    if let Some(v) = f32_slider(ui, "Gate floor", gate_floor, 0.0, 1.0, "") {
        state.params.settings.write().unwrap().noise_gate_floor = v;
        mark_dirty(state);
    }

    ui.small("Crossfade and SOLA settings are configured in the config file and apply on reinstantiation.");
}

/// A labelled control: the name + Browse button on one line, and the current
/// path wrapped on the line below (so long paths don't get cut off on the
/// right). Returns whether Browse was clicked this frame.
fn file_row(ui: &mut egui::Ui, label: &str, current: &Path) -> bool {
    let clicked = ui
        .horizontal(|ui| {
            ui.label(label);
            ui.button("Browse…").clicked()
        })
        .inner;
    let shown = if current.as_os_str().is_empty() {
        "(not set)".to_string()
    } else {
        current.display().to_string()
    };
    ui.label(egui::RichText::new(shown).small().weak());
    ui.add_space(4.0);
    clicked
}

/// A labelled millisecond slider snapping to [`MS_STEP`]. Returns the new value
/// when the user changes it.
fn ms_slider(ui: &mut egui::Ui, label: &str, current: u32, min: u32, max: u32) -> Option<u32> {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut v = current;
        let changed = ui
            .add(
                egui::Slider::new(&mut v, min..=max)
                    // Merely showing an invalid saved value must not silently
                    // snap it before the worker can report the validation error.
                    .clamping(egui::SliderClamping::Edits)
                    .step_by(MS_STEP as f64)
                    .suffix(" ms"),
            )
            .changed();
        changed.then_some(v)
    })
    .inner
}

/// A labelled `f32` slider. Returns the new value when the user changes it.
fn f32_slider(
    ui: &mut egui::Ui,
    label: &str,
    current: f32,
    min: f32,
    max: f32,
    suffix: &str,
) -> Option<f32> {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut v = current.clamp(min, max);
        let changed = ui
            .add(egui::Slider::new(&mut v, min..=max).suffix(suffix))
            .changed();
        changed.then_some(v)
    })
    .inner
}

#[derive(Clone, Copy)]
enum ModelKind {
    Rvc,
    Embedder,
    F0,
}

/// Open the native file dialog on a separate thread and stage the chosen path
/// into the persisted settings (marking it dirty; not applied until the user
/// clicks Load / Reload). Running the modal dialog off the GUI thread is
/// required: rfd pumps a nested message loop, and doing that inside the
/// egui/baseview draw callback re-enters baseview's window proc and panics with
/// "RefCell already borrowed".
fn spawn_picker(state: &EditorState, kind: ModelKind) {
    let params = state.params.clone();
    let dirty = state.dirty.clone();
    std::thread::spawn(move || {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("ONNX model", &["onnx"])
            .pick_file()
        {
            if let Ok(mut settings) = params.settings.write() {
                match kind {
                    ModelKind::Rvc => settings.model = path,
                    ModelKind::Embedder => settings.embedder = path,
                    ModelKind::F0 => settings.f0_model = path,
                }
            }
            dirty.store(true, Ordering::SeqCst);
        }
    });
}

fn spawn_gpu_device_discovery() -> (
    Arc<Mutex<GpuDeviceDiscovery>>,
    Option<std::thread::JoinHandle<()>>,
) {
    let discovery = Arc::new(Mutex::new(GpuDeviceDiscovery::default()));
    if !GPU_DEVICE_SELECTOR_AVAILABLE {
        return (discovery, None);
    }
    let result = Arc::clone(&discovery);
    let thread = match std::thread::Builder::new()
        .name("vc-vst3-gpu-discovery".to_string())
        .spawn(move || {
            let update = match list_cuda_devices() {
                Ok(devices) => GpuDeviceDiscovery {
                    devices: Some(devices),
                    error: None,
                },
                Err(error) => GpuDeviceDiscovery {
                    devices: None,
                    error: Some(format!("{error:#}")),
                },
            };
            if let Ok(mut current) = result.lock() {
                *current = update;
            }
        }) {
        Ok(thread) => Some(thread),
        Err(error) => {
            if let Ok(mut current) = discovery.lock() {
                current.error = Some(format!("failed to spawn GPU discovery thread: {error}"));
            }
            None
        }
    };
    (discovery, thread)
}

fn gpu_device_selector_visible(provider: &str) -> bool {
    // Capability lives on `Provider`; parse the stored string and ask it, so the
    // VST3 and GUI can't drift from the engine's notion of a GPU backend.
    GPU_DEVICE_SELECTOR_AVAILABLE
        && vc_core::Provider::from_name(provider)
            .is_some_and(vc_core::Provider::shows_gpu_device_selector)
}

fn gpu_device_control(
    ui: &mut egui::Ui,
    selected_id: &mut u32,
    discovery: &Mutex<GpuDeviceDiscovery>,
) -> bool {
    let discovery = discovery
        .lock()
        .map(|value| value.clone())
        .unwrap_or_default();
    if let Some(devices) = discovery.devices {
        let selected_text = gpu_device_label(*selected_id, &devices);
        let mut changed = false;
        egui::ComboBox::from_id_salt("gpu-device")
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
                    .prefix("GPU Device ID: ")
                    .range(0..=i32::MAX as u32),
            )
            .changed();
        ui.small(format!("GPU enumeration failed: {error}"));
        changed
    } else {
        ui.label("Detecting CUDA devices...");
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

fn mark_dirty(state: &EditorState) {
    state.dirty.store(true, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displaying_chunk_slider_does_not_round_invalid_persisted_values() {
        let context = egui::Context::default();
        for chunk_ms in [19, 25, 2001] {
            let mut edited = None;
            let _ = context.run_ui(egui::RawInput::default(), |ui| {
                edited = ms_slider(ui, "Chunk", chunk_ms, MIN_CHUNK_MS, MAX_CHUNK_MS);
            });
            assert_eq!(edited, None, "display silently changed {chunk_ms} ms");
        }
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
}
