//! Editor-only acquisition. No hashing, networking, or settings locks are added
//! to process(), and completion only stages paths for explicit Load / Reload.
use std::path::PathBuf;

use super::*;
use crate::config::{CustomEmbedder, PluginConfig};

#[derive(Clone, PartialEq, Eq)]
struct SelectionSnapshot {
    restore_revision: u64,
    voice: PathBuf,
    embedder: CustomEmbedder,
    package: Option<vc_models::Selection>,
    custom: Option<CustomEmbedder>,
}

impl SelectionSnapshot {
    fn new(settings: &PluginConfig, restore_revision: u64) -> Self {
        Self {
            restore_revision,
            voice: settings.model.clone(),
            embedder: CustomEmbedder {
                path: settings.embedder.clone(),
                output: settings.embedder_output.clone(),
            },
            package: settings.embedder_package.clone(),
            custom: settings.custom_embedder.clone(),
        }
    }
}

#[derive(Clone)]
enum JobState {
    Working(vc_models::Progress),
    Finished(Result<vc_models::Installed, String>),
}

/// Executed by nice-plug's lifetime-managed background executor, never as a
/// detached plugin thread that could outlive the host's plugin library.
pub struct EmbedderJob {
    model: &'static vc_models::Model,
    root: PathBuf,
    cache_only: bool,
    cancel: Arc<AtomicBool>,
    state: Arc<Mutex<JobState>>,
}

impl EmbedderJob {
    pub fn run(self) {
        if self.cancel.load(Ordering::Relaxed) {
            return;
        }
        let result = if self.cache_only {
            vc_models::verify_cached(self.model, &self.root, &self.cancel).map(|()| {
                vc_models::Installed {
                    path: self.model.model_path(&self.root),
                    reused: true,
                }
            })
        } else {
            vc_models::download(self.model, &self.root, &self.cancel, |progress| {
                *self.state.lock().unwrap() = JobState::Working(progress);
            })
        };
        *self.state.lock().unwrap() = JobState::Finished(result.map_err(|e| format!("{e:#}")));
    }
}

pub(super) struct EmbedderPicker {
    root: Result<PathBuf, String>,
    choice: Option<(&'static vc_models::Model, SelectionSnapshot)>,
    task: Option<(Arc<AtomicBool>, Arc<Mutex<JobState>>)>,
    message: Option<String>,
}

impl EmbedderPicker {
    pub(super) fn new(root: Result<PathBuf, String>) -> Self {
        Self {
            root,
            choice: None,
            task: None,
            message: None,
        }
    }

    pub(super) fn pending(&self) -> bool {
        self.choice.is_some()
    }

    pub(super) fn working(&self) -> bool {
        self.task
            .as_ref()
            .is_some_and(|(_, state)| matches!(*state.lock().unwrap(), JobState::Working(_)))
    }

    fn cached(&self, model: &vc_models::Model) -> bool {
        self.root
            .as_ref()
            .is_ok_and(|root| model.model_path(root).is_file())
    }

    fn cancel(&mut self) {
        if let Some((cancel, _)) = self.task.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.choice = None;
        self.message = None;
    }

    fn observe(&mut self, settings: &PluginConfig, revision: u64) {
        if self
            .choice
            .as_ref()
            .is_some_and(|(_, origin)| *origin != SelectionSnapshot::new(settings, revision))
        {
            // Sticky cancellation also rejects changed-then-restored selections
            // observed between frames and late results from closed editors.
            self.cancel();
        }
    }
}

impl Drop for EmbedderPicker {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl EditorState {
    fn active_package(&self, settings: &PluginConfig) -> Option<&'static vc_models::Model> {
        settings
            .embedder_package
            .as_ref()
            .and_then(vc_models::selected)
            .filter(|model| {
                self.embedders
                    .root
                    .as_ref()
                    .is_ok_and(|root| settings.embedder == model.model_path(root))
            })
    }

    fn choose_embedder(&mut self, model: &'static vc_models::Model) {
        self.embedders.cancel();
        let origin = SelectionSnapshot::new(
            &self.params.settings.read().unwrap(),
            self.params.settings.restore_revision(),
        );
        self.embedders.choice = Some((model, origin));
        if self.embedders.cached(model) {
            self.start_embedder(true);
        }
    }

    fn start_embedder(&mut self, cache_only: bool) {
        let root = match &self.embedders.root {
            Ok(root) => root.clone(),
            Err(error) => {
                self.embedders.message = Some(error.clone());
                return;
            }
        };
        let Some((model, _)) = &self.embedders.choice else {
            return;
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new(JobState::Working(vc_models::Progress {
            stage: vc_models::Stage::Checking,
            bytes: 0,
            total: 0,
        })));
        self.embedders.task = Some((cancel.clone(), state.clone()));
        self.embedders.message = None;
        self.async_executor.execute_background(EmbedderJob {
            model,
            root,
            cache_only,
            cancel,
            state,
        });
    }

    pub(super) fn poll_embedders(&mut self) {
        let mut settings = self.params.settings.write().unwrap();
        let revision = self.params.settings.restore_revision();
        self.embedders.observe(&settings, revision);
        let result =
            self.embedders
                .task
                .as_ref()
                .and_then(|(_, state)| match &*state.lock().unwrap() {
                    JobState::Finished(Ok(installed)) => Some(installed.clone()),
                    _ => None,
                });
        if let Some(installed) = result {
            let (model, origin) = self.embedders.choice.take().unwrap();
            self.embedders.task = None;
            if self.active_package(&settings).is_none() {
                settings.custom_embedder = Some(CustomEmbedder {
                    path: settings.embedder.clone(),
                    output: settings.embedder_output.clone(),
                });
            }
            settings.embedder = installed.path;
            settings.embedder_package = Some(model.selection());
            // Published profiles select their output by the generator's channel
            // count; an old Custom override must not leak into another profile.
            settings.embedder_output = None;
            if SelectionSnapshot::new(&settings, revision) != origin {
                self.dirty.store(true, Ordering::SeqCst);
                self.embedders.message =
                    Some("Package verified and selected. Load / Reload to apply.".into());
            }
        }
    }

    fn choose_custom_embedder(&mut self) {
        self.embedders.cancel();
        let mut settings = self.params.settings.write().unwrap();
        if settings.embedder_package.is_some() {
            if self.active_package(&settings).is_some() {
                if let Some(custom) = settings.custom_embedder.clone() {
                    settings.embedder = custom.path;
                    settings.embedder_output = custom.output;
                }
            }
            settings.embedder_package = None;
            self.dirty.store(true, Ordering::SeqCst);
        }
    }

    pub(super) fn draw_embedder(&mut self, ui: &mut egui::Ui) {
        let settings = self.params.settings.read().unwrap().clone();
        let selected = self
            .embedders
            .choice
            .as_ref()
            .map(|(model, _)| *model)
            .or_else(|| self.active_package(&settings));
        let mut requested = None;
        let mut custom = false;
        egui::ComboBox::new("embedder-profile", "Embedder")
            .selected_text(selected.map_or("Custom", |model| model.label.as_str()))
            .show_ui(ui, |ui| {
                for model in vc_models::catalog() {
                    if ui
                        .selectable_label(selected.is_some_and(|s| s.id == model.id), &model.label)
                        .clicked()
                    {
                        requested = Some(model);
                        ui.close();
                    }
                }
                if ui.selectable_label(selected.is_none(), "Custom").clicked() {
                    custom = true;
                    ui.close();
                }
            });
        if let Some(model) = requested {
            self.choose_embedder(model);
        }
        if custom {
            self.choose_custom_embedder();
        }
        let settings = self.params.settings.read().unwrap().clone();
        let selected = self
            .embedders
            .choice
            .as_ref()
            .map(|(model, _)| *model)
            .or_else(|| self.active_package(&settings));
        let Some(model) = selected else {
            if file_row(ui, "Embedder file (.onnx)", &settings.embedder) {
                spawn_picker(self, ModelKind::Embedder);
            }
            return;
        };
        ui.small(format!(
            "{} · v{} · {} · {:.1} MB ZIP",
            model.id,
            model.version,
            model.license,
            model.archive.size as f64 / 1_000_000.0
        ));
        let cached = self.embedders.cached(model);
        let status = self
            .embedders
            .task
            .as_ref()
            .map(|(_, state)| state.lock().unwrap().clone());
        let failed = matches!(status, Some(JobState::Finished(Err(_))));
        let busy = matches!(status, Some(JobState::Working(_)));
        if cached && !failed && !busy {
            ui.small("Downloaded");
        }
        if (!cached || failed)
            && model.url.is_some()
            && ui
                .add_enabled(!busy, egui::Button::new("Download and use"))
                .clicked()
        {
            if self.embedders.choice.is_none() {
                self.choose_embedder(model);
            }
            self.start_embedder(false);
        }
        if let Some(JobState::Working(progress)) = status {
            ui.small(match progress.stage {
                vc_models::Stage::Waiting => "Waiting for another package operation…",
                vc_models::Stage::Checking => "Checking package…",
                vc_models::Stage::Acquiring => "Reading package…",
                vc_models::Stage::Extracting => "Extracting package…",
            });
            if progress.total > 0 {
                ui.add(
                    egui::ProgressBar::new(progress.bytes as f32 / progress.total as f32).text(
                        format!(
                            "{:.1} / {:.1} MB",
                            progress.bytes as f64 / 1e6,
                            progress.total as f64 / 1e6
                        ),
                    ),
                );
            }
            if ui.button("Cancel").clicked() {
                self.embedders.cancel();
            }
        } else if let Some(JobState::Finished(Err(error))) = status {
            ui.colored_label(
                egui::Color32::LIGHT_RED,
                "Package acquisition failed. Your current selection was kept.",
            );
            egui::CollapsingHeader::new("Package error details").show(ui, |ui| {
                ui.label(error);
            });
        }
        if let Some(message) = &self.embedders.message {
            ui.small(message);
        }
        if cached && !failed {
            if let Ok(root) = &self.embedders.root {
                ui.small("Package documents (README / README.ja / LICENSE):");
                ui.add(egui::Label::new(model.directory(root).to_string_lossy()).wrap());
            }
        }
    }
}

#[cfg(test)]
mod tests;
