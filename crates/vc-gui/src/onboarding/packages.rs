//! UI adaptation only. Acquisition belongs to vc-models, not the audio engine.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
pub(super) struct Packages {
    selected: String,
    picker_origin: Option<SelectionSnapshot>,
    task: Option<Task>,
    message: Option<&'static str>,
}

#[derive(Clone, PartialEq, Eq)]
struct SelectionSnapshot {
    voice: String,
    embedder: String,
    mode: String,
    package: Option<vc_models::Selection>,
}

impl SelectionSnapshot {
    fn new(settings: &GuiSettings) -> Self {
        Self {
            voice: settings.model.clone(),
            embedder: settings.embedder.clone(),
            mode: settings.support_custom_mode[0].clone(),
            package: settings.embedder_package.clone(),
        }
    }
}

struct Task {
    model: &'static vc_models::Model,
    origin: SelectionSnapshot,
    may_apply: bool,
    cancel: Arc<AtomicBool>,
    state: Arc<Mutex<TaskState>>,
}

#[derive(Clone)]
enum TaskState {
    Working(vc_models::Progress),
    Finished(Result<vc_models::Installed, String>),
}

impl Drop for Task {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Packages {
    pub(super) fn cancel(&mut self) {
        self.task = None;
        self.message = None;
        self.picker_origin = None;
    }
    pub(super) fn select_from_picker(
        &mut self,
        model: &'static vc_models::Model,
        settings: &GuiSettings,
    ) {
        self.cancel();
        self.selected = model.id.clone();
        // The picker can preview an uninstalled package while the actual path
        // and saved source remain usable until acquisition verifies all files.
        self.picker_origin = Some(SelectionSnapshot::new(settings));
    }
    pub(super) fn pending_choice(
        &self,
        settings: &GuiSettings,
    ) -> Option<&'static vc_models::Model> {
        self.picker_origin
            .as_ref()
            .filter(|origin| **origin == SelectionSnapshot::new(settings))
            .and_then(|_| vc_models::catalog().iter().find(|m| m.id == self.selected))
    }
    pub(super) fn invalidate_application(&mut self) {
        self.picker_origin = None;
        if let Some(task) = &mut self.task {
            task.may_apply = false;
        }
    }
    pub(super) fn observe_selection(&mut self, settings: &GuiSettings) {
        if self
            .picker_origin
            .as_ref()
            .is_some_and(|origin| *origin != SelectionSnapshot::new(settings))
        {
            self.picker_origin = None;
        }
        if let Some(task) = &mut self.task {
            // Sticky invalidation also covers switching away and back before the
            // worker finishes. A late result may populate cache, never settings.
            if task.origin != SelectionSnapshot::new(settings) {
                task.may_apply = false;
            }
        }
    }
}

impl VcGui {
    fn package_root(&self) -> Result<PathBuf, String> {
        #[cfg(test)]
        return Ok(self.onboarding.effects.support_cache_dir.clone());
        #[cfg(not(test))]
        model_setup::cache_dir()
    }

    pub(super) fn embedder_package_cached(&self, model: &vc_models::Model) -> bool {
        self.package_root()
            .is_ok_and(|root| model.model_path(&root).is_file())
    }

    pub(super) fn select_embedder_package(&mut self, model: &'static vc_models::Model) {
        self.onboarding
            .packages
            .select_from_picker(model, &self.settings);
        if self.embedder_package_cached(model) {
            // Selecting an existing package may verify local files, but never
            // authorizes a network transfer if those files are incomplete.
            self.start_embedder_package(model, true);
        }
    }

    pub(crate) fn poll_embedder_package(&mut self) {
        self.onboarding.packages.observe_selection(&self.settings);
        let result = self.onboarding.packages.task.as_ref().and_then(|task| {
            match &*task.state.lock().unwrap() {
                TaskState::Finished(Ok(installed)) => Some(installed.clone()),
                _ => None,
            }
        });
        let Some(installed) = result else {
            return;
        };
        let task = self.onboarding.packages.task.take().unwrap();
        let applies = task.may_apply
            && task.origin == SelectionSnapshot::new(&self.settings)
            && task.model.id == self.onboarding.packages.selected;
        if applies {
            // Re-selecting the active package can leave the saved settings
            // identical. End the preview explicitly rather than relying on
            // changed() to notice a different selection and unblock Start.
            self.onboarding.packages.picker_origin = None;
            // Keep the Custom slot as a separate choice, including while a
            // package is active. No failed/cancelled job changes any selection.
            let custom = match self.settings.support_custom_mode[0].as_str() {
                "custom" => true,
                "package" | "downloaded" => false,
                _ => {
                    !self.settings.embedder.is_empty()
                        && self.package_root().is_ok_and(|root| {
                            Path::new(&self.settings.embedder)
                                != root.join(model_setup::MODELS[0].file)
                        })
                }
            };
            if custom {
                self.settings.support_custom_paths[0] = self.settings.embedder.clone();
            }
            self.settings.embedder = installed.path.to_string_lossy().into_owned();
            self.settings.embedder_package = Some(task.model.selection());
            self.settings.support_custom_mode[0] = "package".into();
            self.onboarding.support_check = None;
            self.onboarding.packages.message =
                Some("Package verified and selected. Start or reload conversion to apply.");
            self.changed();
        } else {
            self.onboarding.packages.message =
                Some("Package cached. Your current model selection was kept.");
        }
    }

    fn start_embedder_package(&mut self, model: &'static vc_models::Model, cache_only: bool) {
        let root = match self.package_root() {
            Ok(root) => root,
            Err(error) => {
                self.ui_error = Some(error);
                return;
            }
        };
        let task = Task {
            model,
            origin: SelectionSnapshot::new(&self.settings),
            may_apply: true,
            cancel: Arc::new(AtomicBool::new(false)),
            state: Arc::new(Mutex::new(TaskState::Working(vc_models::Progress {
                stage: vc_models::Stage::Checking,
                bytes: 0,
                total: 0,
            }))),
        };
        #[cfg(not(test))]
        {
            let state = task.state.clone();
            let cancel = task.cancel.clone();
            let spawned = std::thread::Builder::new()
                .name("vc-embedder-package".into())
                .spawn(move || {
                    let progress = |value| *state.lock().unwrap() = TaskState::Working(value);
                    // Hashing stays on this worker, away from audio/UI callbacks.
                    // Only an explicit Download action may fetch missing files.
                    let result = if cache_only {
                        vc_models::verify_cached(model, &root, &cancel).map(|()| {
                            vc_models::Installed {
                                path: model.model_path(&root),
                                reused: true,
                            }
                        })
                    } else {
                        vc_models::download(model, &root, &cancel, progress)
                    }
                    .map_err(|e| format!("{e:#}"));
                    *state.lock().unwrap() = TaskState::Finished(result);
                });
            if let Err(error) = spawned {
                *task.state.lock().unwrap() = TaskState::Finished(Err(error.to_string()));
            }
        }
        #[cfg(test)]
        {
            let _ = root;
            if cache_only {
                self.onboarding
                    .effects
                    .package_cache_checks
                    .push(model.id.clone());
            } else {
                self.onboarding
                    .effects
                    .package_requests
                    .push(model.id.clone());
            }
        }
        self.onboarding.packages.task = Some(task);
        self.onboarding.packages.message = None;
    }

    pub(super) fn embedder_packages_ui(&mut self, ui: &mut egui::Ui) {
        let lang = self.settings.language;
        egui::CollapsingHeader::new(lang.text("Embedder packages")).id_salt("embedder-packages").show(ui, |ui| {
            // Keep the body width stable across locales. Otherwise short text
            // can shrink the header's clip rect after a long path was laid out.
            ui.set_min_width(ui.available_width());
            ui.small(lang.text("Choose the embedder used to train your voice model. Matching dimensions alone do not establish compatibility."));
            if self.onboarding.packages.selected.is_empty() {
                self.onboarding.packages.selected = self.settings.embedder_package.as_ref()
                    .and_then(vc_models::selected).unwrap_or(&vc_models::catalog()[0]).id.clone();
            }
            let selected = vc_models::catalog().iter().find(|m| m.id == self.onboarding.packages.selected).unwrap_or(&vc_models::catalog()[0]);
            let mut requested = None;
            egui::ComboBox::new("embedder-package-choice", lang.text("Package")).selected_text(lang.text(&selected.label)).show_ui(ui, |ui| {
                for model in vc_models::catalog() {
                    if ui.selectable_label(model.id == selected.id, lang.text(&model.label)).clicked() {
                        requested = Some(model);
                        ui.close();
                    }
                }
            });
            if let Some(model) = requested {
                self.select_embedder_package(model);
            }
            let model = vc_models::catalog().iter().find(|m| m.id == self.onboarding.packages.selected).unwrap();
            self.embedder_package_controls_ui(ui, model);
        });
    }

    pub(super) fn embedder_package_controls_ui(
        &mut self,
        ui: &mut egui::Ui,
        model: &'static vc_models::Model,
    ) {
        let lang = self.settings.language;
        self.onboarding.packages.selected = model.id.clone();
        ui.small(format!(
            "{} · v{} · {} · {:.1} MB ZIP",
            model.id,
            model.version,
            model.license,
            model.archive.size as f64 / 1_000_000.0
        ));
        ui.small(format!(
            "{}: {}",
            lang.text("Outputs"),
            model
                .outputs
                .iter()
                .map(|o| format!("{} / {}", o.name, o.channels))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        let cached = self.embedder_package_cached(model);
        if model.url.is_none() && !cached {
            ui.small(lang.text("No public download source is configured."));
        }
        let busy = self
            .onboarding
            .packages
            .task
            .as_ref()
            .is_some_and(|t| matches!(*t.state.lock().unwrap(), TaskState::Working(_)));
        let failed = self
            .onboarding
            .packages
            .task
            .as_ref()
            .is_some_and(|t| matches!(*t.state.lock().unwrap(), TaskState::Finished(Err(_))));
        if cached && !failed && !busy {
            ui.small(lang.text("Downloaded"));
        }
        if model.url.is_some()
            && (!cached || failed)
            && ui
                .add_enabled(!busy, egui::Button::new(lang.text("Download and use")))
                .clicked()
        {
            self.start_embedder_package(model, false);
        }
        let status = self
            .onboarding
            .packages
            .task
            .as_ref()
            .map(|t| t.state.lock().unwrap().clone());
        match status {
            Some(TaskState::Working(progress)) => {
                let stage = match progress.stage {
                    vc_models::Stage::Waiting => "Waiting for another package operation…",
                    vc_models::Stage::Checking => "Checking package…",
                    vc_models::Stage::Acquiring => "Reading package…",
                    vc_models::Stage::Extracting => "Extracting package…",
                };
                ui.label(lang.text(stage));
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
                if ui.button(lang.text(text::CANCEL)).clicked() {
                    // Both acquisition and local verification return to the
                    // previous choice; a cancelled preview must not block Start.
                    self.onboarding.packages.cancel();
                }
            }
            Some(TaskState::Finished(Err(error))) => {
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    lang.text("Package acquisition failed. Your current selection was kept."),
                );
                egui::CollapsingHeader::new(lang.text("Error details")).show(ui, |ui| {
                    ui.label(error);
                });
            }
            _ => {}
        }
        if let Some(message) = self.onboarding.packages.message {
            ui.small(lang.text(message));
        }
        if cached && !failed {
            if let Ok(root) = self.package_root() {
                ui.small(lang.text("Package documents (README / README.ja / LICENSE):"));
                // Describe the profile displayed here, not the saved selection
                // that remains active until this new choice has been verified.
                ui.add(egui::Label::new(model.directory(&root).to_string_lossy()).wrap());
            }
        }
    }
}

#[cfg(test)]
pub(super) mod tests;
