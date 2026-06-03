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

use crate::params::VcRvcParams;
use crate::runtime::{MAX_CHUNK_MS, MIN_CHUNK_MS};

/// Granularity for the millisecond sliders.
const MS_STEP: u32 = 10;
const MIN_EXTRA_CONVERT_MS: u32 = 0;
const MAX_EXTRA_CONVERT_MS: u32 = 3000;

/// Shared state handed to the egui update closure.
pub struct EditorState {
    pub params: Arc<VcRvcParams>,
    /// Set by the **Load / Reload** button to request a pipeline rebuild.
    pub reload: Arc<AtomicBool>,
    /// Set when settings are edited, cleared by the worker once it applies them.
    /// Drives the "unapplied changes" indicator.
    pub dirty: Arc<AtomicBool>,
    /// Human-readable worker status shown in the UI.
    pub status: Arc<Mutex<String>>,
}

pub fn create(
    params: Arc<VcRvcParams>,
    reload: Arc<AtomicBool>,
    dirty: Arc<AtomicBool>,
    status: Arc<Mutex<String>>,
) -> Option<Box<dyn Editor>> {
    let egui_state = params.editor_state.clone();
    create_egui_editor(
        egui_state,
        EditorState {
            params,
            reload,
            dirty,
            status,
        },
        Default::default(),
        |_, _, _| {},
        |ui, setter, _queue, state| draw(ui, setter, state),
    )
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
    ui.heading("VC-RS RVC");

    // Status line (updated by the worker thread).
    let status = state.status.lock().map(|s| s.clone()).unwrap_or_default();
    ui.label(format!("Status: {status}"));
    ui.separator();

    // Snapshot current settings for display, releasing the lock immediately.
    let (model, embedder, f0_model, provider) = {
        let s = state.params.settings.read().unwrap();
        (
            s.model.clone(),
            s.embedder.clone(),
            s.f0_model.clone(),
            s.provider.clone(),
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
    ui.horizontal(|ui| {
        ui.label("Backend");
        egui::ComboBox::from_id_salt("provider")
            .selected_text(provider.to_uppercase())
            .show_ui(ui, |ui| {
                for option in ["cpu", "cuda"] {
                    if ui
                        .selectable_label(provider == option, option.to_uppercase())
                        .clicked()
                        && provider != option
                    {
                        state.params.settings.write().unwrap().provider = option.to_string();
                        mark_dirty(state);
                    }
                }
            });
        if ui.button("Load / Reload").clicked() {
            state.reload.store(true, Ordering::SeqCst);
        }
        if state.dirty.load(Ordering::Relaxed) {
            ui.colored_label(
                egui::Color32::from_rgb(220, 180, 60),
                "● unapplied — click Load / Reload",
            );
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

    ui.separator();
    ui.label("Live parameters");
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
    });

    ui.separator();
    ui.small("Model / backend / chunk edits apply when you click Load / Reload. Other latency settings (crossfade, SOLA, extra-convert) come from the config file and apply on reinstantiation.");
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
        let mut v = current.clamp(min, max);
        let changed = ui
            .add(
                egui::Slider::new(&mut v, min..=max)
                    .step_by(MS_STEP as f64)
                    .suffix(" ms"),
            )
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

fn mark_dirty(state: &EditorState) {
    state.dirty.store(true, Ordering::SeqCst);
}
