//! GUI-only workflow. Preview goes through EngineController, never a second
//! inference/audio path. No setup worker may run on an audio callback.
use super::*;
use ui_text as text;
use vc_app::{DeviceList, EngineStatusSnapshot};

#[cfg(all(test, feature = "ui-snapshots"))]
mod texture_to_image;
#[cfg(all(test, feature = "ui-snapshots"))]
mod ui_renderer;
#[cfg(test)]
mod ui_tests;

type CheckResult = Arc<Mutex<Option<Result<(), String>>>>;
type SupportResult = Arc<Mutex<Option<[Result<(), String>; 2]>>>;
type RuntimeInspection = Arc<Mutex<Option<Result<bool, String>>>>;
mod normal;
mod packages;
mod terms;
mod workflow;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Progress {
    pub step: Option<Step>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum Step {
    Language,
    Terms,
    Audio,
    Voice,
    // Older tutorials resume the removed conversion test at Preparation.
    #[serde(alias = "Listen")]
    Prepare,
}

pub(super) struct Onboarding {
    packages: packages::Packages,
    step: Option<Step>,
    pub(super) normal: normal::NormalState,
    pub(super) model_check: Option<(String, CheckResult)>,
    support_check: Option<(String, SupportResult)>,
    runtime_check: Option<(String, CheckResult)>,
    runtime_inspection: Option<(String, RuntimeInspection)>,
    return_after_terms: Option<Step>,
    test_signature: Option<String>,
    prepare_requested: bool,
    #[cfg(test)]
    pub(super) effects: workflow::TestEffects,
}

fn voice_exists(settings: &GuiSettings) -> bool {
    model_setup::available(&settings.model) && !is_pth_path(&settings.model)
}

fn support_exists(settings: &GuiSettings) -> bool {
    model_setup::available(&settings.embedder) && model_setup::available(&settings.f0_model)
}

fn support_signature(settings: &GuiSettings) -> String {
    [&settings.embedder, &settings.f0_model]
        .iter()
        .map(|path| {
            let metadata = fs::metadata(path).ok();
            format!(
                "{}:{:?}:{:?}",
                path,
                metadata.as_ref().map(|m| m.len()),
                metadata.and_then(|m| m.modified().ok())
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

impl Onboarding {
    pub(super) fn new(settings: &GuiSettings) -> Self {
        let mut step = settings
            .tutorial
            .as_ref()
            .map(|p| p.step)
            .unwrap_or_else(|| {
                if settings.setup_completed || settings.setup_skipped {
                    None
                } else {
                    Some(Step::Language)
                }
            });
        // Only in-progress tutorials rewind for missing prerequisites. Completed
        // users repair models from the normal screen instead of being trapped.
        if matches!(step, Some(Step::Prepare)) && !voice_exists(settings) {
            step = Some(Step::Voice);
        }
        let return_after_terms = step;
        if step != Some(Step::Language) && !terms::accepted(settings) {
            step = Some(Step::Terms);
        }
        Self {
            packages: Default::default(),
            step,
            normal: Default::default(),
            model_check: None,
            support_check: None,
            runtime_check: None,
            runtime_inspection: None,
            return_after_terms,
            test_signature: None,
            prepare_requested: false,
            #[cfg(test)]
            effects: workflow::TestEffects::default(),
        }
    }

    pub(super) fn active(&self) -> bool {
        self.step.is_some()
    }

    pub(super) fn observe_package_selection(&mut self, settings: &GuiSettings) {
        self.packages.observe_selection(settings);
    }

    pub(super) fn invalidate_package_application(&mut self) {
        self.packages.invalidate_application();
    }

    fn check_support(&mut self, settings: &GuiSettings) -> Option<Result<(), String>> {
        let signature = support_signature(settings);
        if self
            .support_check
            .as_ref()
            .is_none_or(|(key, _)| key != &signature)
        {
            let state: SupportResult = Arc::new(Mutex::new(None));
            let worker = state.clone();
            let paths = [settings.embedder.clone(), settings.f0_model.clone()];
            let spawned = std::thread::Builder::new()
                .name("vc-support-check".into())
                .spawn(move || {
                    *worker.lock().unwrap() = Some(std::array::from_fn(|i| {
                        model_setup::validate_support(Path::new(&paths[i]), i)
                    }));
                });
            if let Err(error) = spawned {
                *state.lock().unwrap() = Some([Err(error.to_string()), Err(error.to_string())]);
            }
            self.support_check = Some((signature, state));
        }
        self.support_check
            .as_ref()
            .unwrap()
            .1
            .lock()
            .unwrap()
            .clone()
            .map(|results| results.into_iter().collect())
    }

    fn check_model(&mut self, path: &str) -> Option<Result<(), String>> {
        if self
            .model_check
            .as_ref()
            .is_none_or(|(checked, _)| checked != path)
        {
            let state = Arc::new(Mutex::new(None));
            let worker = state.clone();
            let source = PathBuf::from(path);
            let spawned = std::thread::Builder::new()
                .name("vc-gui-model-check".into())
                .spawn(move || {
                    *worker.lock().unwrap() = Some(
                        vc_core::model_rvc::validate_rvc_model(&source)
                            .map_err(|e| format!("{e:#}")),
                    );
                });
            if let Err(error) = spawned {
                *state.lock().unwrap() = Some(Err(error.to_string()));
            }
            // Each result belongs to its selected path; obsolete workers cannot
            // mark a newer selection as ready.
            self.model_check = Some((path.to_string(), state));
        }
        self.model_check.as_ref().unwrap().1.lock().unwrap().clone()
    }
}

impl VcGui {
    /// Shared by catalog inspection and preparation failures. Runtime bootstrap
    /// failures are cached in vc-core for this process, so installation recovery
    /// must explicitly instruct users to restart rather than promise a live fix.
    fn runtime_failure_ui(&self, ui: &mut egui::Ui, error: &str) -> bool {
        let lang = self.settings.language;
        ui.colored_label(
            egui::Color32::LIGHT_RED,
            lang.text("Runtime check could not complete."),
        );
        if self.settings.provider.starts_with("windowsml") {
            ui.label(lang.text(text::RUNTIME_HELP));
            ui.label(lang.text(text::RUNTIME_INSTALL));
            ui.hyperlink_to(
                lang.text(text::RUNTIME_LINK),
                "https://learn.microsoft.com/windows/apps/windows-app-sdk/downloads",
            );
            ui.label(lang.text(text::RUNTIME_RESTART));
            ui.small(lang.text(text::RUNTIME_INSTALLED));
        }
        egui::CollapsingHeader::new(lang.text("Runtime details"))
            .id_salt("Runtime details")
            .show(ui, |ui| {
                ui.label(error);
            });
        ui.button(lang.text("Retry runtime check")).clicked()
    }

    #[cfg(not(test))]
    fn runtime_preflight(&mut self, ui: &mut egui::Ui) {
        let lang = self.settings.language;
        let selected = self.settings.provider.clone();
        if self
            .onboarding
            .runtime_check
            .as_ref()
            .is_none_or(|(name, _)| name != &selected)
        {
            let state: CheckResult = Arc::new(Mutex::new(None));
            let worker = state.clone();
            let provider_name = selected.clone();
            let spawned = std::thread::Builder::new().name("vc-gui-runtime-check".into()).spawn(move || {
                let result = (|| {
                    let provider = parse_provider(&provider_name)?;
                    if !provider.available_in_build() { return Err("This backend is not included in this build.".into()); }
                    #[cfg(all(windows, feature = "windowsml"))]
                    if provider.is_windows_ml() {
                        vc_core::windows_ml::prepare_provider(provider).map_err(|e| format!("{e:#}"))?;
                    }
                    if provider.is_tensorrt() || provider.is_cuda() {
                        let devices = list_cuda_devices().map_err(|e| format!("{e:#}"))?;
                        if devices.is_empty() { return Err("No NVIDIA GPU was detected. Check your driver or use the Windows ML build.".into()); }
                    }
                    Ok(())
                })();
                *worker.lock().unwrap() = Some(result);
            });
            if let Err(error) = spawned {
                *state.lock().unwrap() = Some(Err(error.to_string()));
            }
            self.onboarding.runtime_check = Some((selected, state));
        }
        let result = self
            .onboarding
            .runtime_check
            .as_ref()
            .unwrap()
            .1
            .lock()
            .unwrap()
            .clone();
        match result {
            None => {
                ui.spinner();
                ui.label(lang.text("Checking the runtime…"));
            }
            Some(Ok(())) => {
                ui.label(lang.text(
                    "Runtime preflight passed. Starting conversion will verify model loading and audio.",
                ));
            }
            Some(Err(error)) => {
                if self.runtime_failure_ui(ui, &error) {
                    self.onboarding.runtime_check = None;
                }
            }
        }
    }

    pub(super) fn download_status(&mut self, ui: &mut egui::Ui) {
        let lang = self.settings.language;
        let state = self
            .model_download
            .as_ref()
            .map(|d| d.state.lock().unwrap().clone());
        match state {
            Some(model_setup::State::Done) => {
                self.model_download = None;
                if discover_support_models(&mut self.settings) {
                    self.changed();
                }
            }
            Some(model_setup::State::Running { name, bytes, total }) => {
                ui.add(
                    egui::ProgressBar::new(if total == 0 {
                        0.0
                    } else {
                        bytes as f32 / total as f32
                    })
                    .text(format!(
                        "{}: {:.1} / {:.1} MB",
                        lang.text(name),
                        bytes as f64 / 1e6,
                        total as f64 / 1e6
                    )),
                );
                if ui.button(lang.text(text::CANCEL)).clicked() {
                    self.model_download = None;
                    if discover_support_models(&mut self.settings) {
                        self.changed();
                    }
                }
            }
            Some(model_setup::State::Failed(error)) => {
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    text::diagnostic_message(lang, &error),
                );
                ui.small(lang.text(text::RETRY));
                if discover_support_models(&mut self.settings) {
                    self.changed();
                }
                if ui.button(lang.text(text::DISMISS)).clicked() {
                    self.model_download = None;
                }
            }
            None => {}
        }
    }

    fn support_download(&mut self, ui: &mut egui::Ui) {
        let lang = self.settings.language;
        ui.label(lang.text(text::DOWNLOAD_INFO));
        ui.label(
            lang.text("Downloading uses the GPL-3.0 model license. Completed files are reused."),
        );
        ui.hyperlink_to("GPL-3.0", "https://www.gnu.org/licenses/gpl-3.0.html");
        if cfg!(feature = "gtcrn") {
            ui.label(lang.text("GTCRN noise reduction model is included: 352 KB, MIT license. Noise reduction will not be enabled automatically."));
            egui::CollapsingHeader::new("GTCRN · MIT").show(ui, |ui| {
                ui.label(include_str!("gtcrn-license.txt"));
            });
        }
        let running = self.model_download.as_ref().is_some_and(|d| {
            matches!(*d.state.lock().unwrap(), model_setup::State::Running { .. })
        });
        let checking = self
            .onboarding
            .support_check
            .as_ref()
            .is_none_or(|(_, state)| state.lock().unwrap().is_none());
        if ui
            .add_enabled(
                !running && !checking,
                egui::Button::new(lang.text(text::DOWNLOAD)),
            )
            .clicked()
        {
            if !terms::accepted(&self.settings) {
                self.onboarding.return_after_terms = Some(Step::Prepare);
                self.onboarding.step = Some(Step::Terms);
                return;
            }
            let mut candidate = self.settings.clone();
            if !candidate
                .accepted_terms
                .iter()
                .any(|id| id == "support-models-gpl3-v1")
            {
                candidate
                    .accepted_terms
                    .push("support-models-gpl3-v1".into());
            }
            if cfg!(feature = "gtcrn")
                && !candidate
                    .accepted_terms
                    .iter()
                    .any(|id| id == "gtcrn-mit-502ebfab")
            {
                candidate.accepted_terms.push("gtcrn-mit-502ebfab".into());
            }
            if !self.persist_tutorial(candidate) {
                return;
            }
            let checked = self
                .onboarding
                .support_check
                .as_ref()
                .and_then(|(_, state)| state.lock().unwrap().clone());
            let mut indices: Vec<usize> = (0..2)
                .filter(|&i| checked.as_ref().is_none_or(|results| results[i].is_err()))
                .collect();
            if cfg!(feature = "gtcrn") {
                // Use the pinned cache download even if a custom denoiser is
                // selected. Discovery preserves that selection; the worker
                // verifies and reuses an already downloaded GTCRN file.
                indices.push(model_setup::GTCRN_INDEX);
            }
            #[cfg(test)]
            {
                self.onboarding.effects.download_requests += 1;
                self.onboarding.effects.download_indices = indices;
            }
            #[cfg(not(test))]
            match model_setup::cache_dir() {
                Ok(dir) => {
                    // The download action explicitly chooses the reference
                    // files. Verified files are reused by the worker; corrupt
                    // cache entries are repaired instead of accepted by size.
                    for &i in indices.iter().filter(|&&i| i < 2) {
                        let path = dir
                            .join(model_setup::MODELS[i].file)
                            .to_string_lossy()
                            .into_owned();
                        if i == 0 {
                            self.settings.embedder = path;
                        } else {
                            self.settings.f0_model = path;
                        }
                    }
                    self.changed();
                    self.onboarding.support_check = None;
                    self.model_download = Some(model_setup::Download::start(dir, indices));
                }
                Err(error) => self.ui_error = Some(error),
            }
        }
    }

    fn select_voice(&mut self, ui: &mut egui::Ui) {
        let lang = self.settings.language;
        let name = Path::new(&self.settings.model)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(lang.text("No voice model selected"));
        ui.label(name).on_hover_text(&self.settings.model);
        let choose_label = if self.onboarding.step == Some(Step::Voice) {
            "Choose .pth file…"
        } else {
            text::CHOOSE
        };
        if ui.button(lang.text(choose_label)).clicked() {
            self.browse_into(ModelKind::Rvc);
            self.onboarding.model_check = None;
        }
        if is_pth_path(&self.settings.model) && ui.button(lang.text("Convert to ONNX…")).clicked()
        {
            self.pth_convert = Some(PthConvert::new(PathBuf::from(&self.settings.model)));
        }
        let dropped = ui
            .ctx()
            .input(|i| i.raw.dropped_files.first().and_then(|f| f.path.clone()));
        if let Some(path) = dropped {
            match path
                .extension()
                .and_then(|s| s.to_str())
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                Some("pth") => self.pth_convert = Some(PthConvert::new(path)),
                Some("onnx") => {
                    self.settings.model = path.to_string_lossy().into_owned();
                    self.onboarding.model_check = None;
                    self.changed();
                }
                _ => self.ui_error = Some("Choose an RVC .onnx or .pth file.".into()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn new_settings_start_with_language() {
        let settings: GuiSettings = toml::from_str("").unwrap();
        assert_eq!(settings.language, ui_text::Language::English);
        assert!(!settings.setup_completed);
        assert_eq!(Onboarding::new(&settings).step, Some(Step::Language));
    }
    #[test]
    fn completed_setup_only_requires_missing_consent() {
        let mut settings = GuiSettings {
            setup_completed: true,
            ..Default::default()
        };
        assert_eq!(Onboarding::new(&settings).step, Some(Step::Terms));
        settings.accepted_terms = terms::required(&settings);
        assert_eq!(Onboarding::new(&settings).step, None);
    }

    #[test]
    fn completion_roundtrips_and_keeps_normal_screen_when_models_disappear() {
        let dir = std::env::temp_dir().join(format!("vc-onboarding-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.onnx");
        fs::write(&file, b"selected model placeholder").unwrap();
        let path = file.to_string_lossy().into_owned();
        let mut settings = GuiSettings {
            setup_completed: true,
            model: path.clone(),
            embedder: path.clone(),
            f0_model: path,
            ..Default::default()
        };
        settings.accepted_terms = terms::required(&settings);
        let restored: GuiSettings = toml::from_str(&toml::to_string(&settings).unwrap()).unwrap();
        assert_eq!(Onboarding::new(&restored).step, None);
        fs::remove_file(&file).unwrap();
        assert_eq!(Onboarding::new(&restored).step, None);
        fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn resumed_audio_step_and_missing_terms_preserve_destination() {
        let mut settings = GuiSettings {
            tutorial: Some(Progress {
                step: Some(Step::Audio),
            }),
            ..Default::default()
        };
        let pending = Onboarding::new(&settings);
        assert_eq!(pending.step, Some(Step::Terms));
        assert_eq!(pending.return_after_terms, Some(Step::Audio));
        settings.accepted_terms = terms::required(&settings);
        let saved: GuiSettings = toml::from_str(&toml::to_string(&settings).unwrap()).unwrap();
        let resumed = Onboarding::new(&saved);
        assert_eq!(resumed.step, Some(Step::Audio));
        assert!(resumed.test_signature.is_none());
        assert!(!resumed.prepare_requested);
        settings.accepted_terms = vec!["obsolete-terms".into()];
        assert_eq!(Onboarding::new(&settings).step, Some(Step::Terms));
    }

    #[test]
    fn interrupted_preparation_with_missing_voice_rewinds() {
        let mut settings = GuiSettings {
            tutorial: Some(Progress {
                step: Some(Step::Prepare),
            }),
            ..Default::default()
        };
        settings.accepted_terms = terms::required(&settings);
        assert_eq!(Onboarding::new(&settings).step, Some(Step::Voice));
    }

    #[test]
    fn legacy_listen_step_resumes_preparation_without_starting_work() {
        let mut settings: GuiSettings = toml::from_str("[tutorial]\nstep = 'Listen'\n").unwrap();
        settings.model = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        settings.accepted_terms = terms::required(&settings);
        let resumed = Onboarding::new(&settings);
        assert_eq!(resumed.step, Some(Step::Prepare));
        assert!(!resumed.prepare_requested);
        assert!(resumed.runtime_check.is_none());
        assert!(resumed.support_check.is_none());
        assert!(!settings.setup_completed);
        let saved = toml::to_string(&settings).unwrap();
        assert!(!saved.contains("Listen"));
        assert!(saved.contains("Prepare"));
        settings.model.clear();
        assert_eq!(Onboarding::new(&settings).step, Some(Step::Voice));
    }

    #[test]
    fn new_user_volume_matches_legacy_defaults() {
        assert_eq!(GuiSettings::new_user().output_gain, 1.0);
        assert_eq!(toml::from_str::<GuiSettings>("").unwrap().output_gain, 1.0);
    }

    #[test]
    fn settings_replace_is_atomic_and_failed_replace_keeps_existing_data() {
        let directory =
            std::env::temp_dir().join(format!("vc-settings-atomic-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("settings.toml");
        let mut settings = GuiSettings::default();
        save_settings_at(&settings, &path).unwrap();
        settings.input_gain = 2.0;
        save_settings_at(&settings, &path).unwrap();
        assert_eq!(
            toml::from_str::<GuiSettings>(&fs::read_to_string(&path).unwrap())
                .unwrap()
                .input_gain,
            2.0
        );
        let blocked = directory.join("blocked");
        fs::create_dir_all(&blocked).unwrap();
        fs::write(blocked.join("preserved"), b"keep").unwrap();
        assert!(save_settings_at(&settings, &blocked).is_err());
        assert_eq!(fs::read(blocked.join("preserved")).unwrap(), b"keep");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 2);
        fs::remove_file(path).unwrap();
        fs::remove_file(blocked.join("preserved")).unwrap();
        fs::remove_dir(blocked).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
