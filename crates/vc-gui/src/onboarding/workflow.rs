use super::*;
#[cfg(test)]
use vc_app::DeviceTestSnapshot;
use vc_app::{DeviceTestConfig, TestOutput};

#[cfg(test)]
#[derive(Default)]
pub(crate) struct TestEffects {
    pub catalog_providers: Vec<Provider>,
    pub session_revision: u64,
    pub stop_requests: usize,
    pub stop_error: Option<String>,
    pub support_cache_dir: PathBuf,
    pub start_requests: usize,
    pub start_error: Option<String>,
    pub download_requests: usize,
    pub download_indices: Vec<usize>,
    pub save_error: Option<String>,
    pub saved: Vec<GuiSettings>,
    pub device_requests: Vec<TestOutput>,
    pub device_configs: Vec<DeviceTestConfig>,
    pub state: DeviceTestSnapshot,
}

impl GuiSettings {
    pub(crate) fn new_user() -> Self {
        Self {
            language: text::Language::English,
            provider: if cfg!(feature = "windowsml") {
                "windowsml".into()
            } else {
                default_provider_name().into()
            },
            ..Default::default()
        }
    }
}

impl VcGui {
    fn inspect_runtime(&mut self) -> Option<Result<bool, String>> {
        let provider = self.settings.provider.clone();
        if self
            .onboarding
            .runtime_inspection
            .as_ref()
            .is_none_or(|(key, _)| key != &provider)
        {
            let state: RuntimeInspection = Arc::new(Mutex::new(None));
            let worker = state.clone();
            let selected = provider.clone();
            #[cfg(not(test))]
            let spawned = std::thread::Builder::new()
                .name("vc-runtime-inspect".into())
                .spawn(move || {
                    let result = (|| {
                        let provider = parse_provider(&selected)?;
                        #[cfg(all(windows, feature = "windowsml"))]
                        if provider.is_windows_ml() {
                            return vc_core::windows_ml::preparation_required(provider)
                                .map_err(|e| format!("{e:#}"));
                        }
                        let _ = provider;
                        Ok(false)
                    })();
                    *worker.lock().unwrap() = Some(result);
                });
            #[cfg(not(test))]
            if let Err(error) = spawned {
                *state.lock().unwrap() = Some(Err(error.to_string()));
            }
            #[cfg(test)]
            {
                let _ = (worker, selected);
                *state.lock().unwrap() = Some(Ok(true));
            }
            self.onboarding.runtime_inspection = Some((provider, state));
        }
        self.onboarding
            .runtime_inspection
            .as_ref()
            .unwrap()
            .1
            .lock()
            .unwrap()
            .clone()
    }
    pub(super) fn persist_tutorial(&mut self, candidate: GuiSettings) -> bool {
        #[cfg(not(test))]
        let result = save_settings(&candidate);
        #[cfg(test)]
        let result = self
            .onboarding
            .effects
            .save_error
            .clone()
            .map_or(Ok(()), Err);
        match result {
            Ok(()) => {
                #[cfg(test)]
                self.onboarding.effects.saved.push(candidate.clone());
                self.settings = candidate;
                self.dirty_since = None;
                self.ui_error = None;
                true
            }
            Err(error) => {
                self.ui_error = Some(format!(
                    "{} {error}",
                    self.settings.language.text(text::FINISH_ERROR)
                ));
                false
            }
        }
    }

    /// Stop test output first, then persist before changing the visible page. Store
    /// the intended destination even when a missing consent temporarily inserts
    /// Terms, so closing that page resumes the correct route after acceptance.
    fn go(&mut self, next: Option<Step>, skipped: bool, completed: bool) -> bool {
        self.ui_error = None;
        self.stop();
        if self.ui_error.is_some() {
            return false;
        }
        let mut candidate = self.settings.clone();
        candidate.tutorial = Some(Progress { step: next });
        candidate.setup_skipped |= skipped;
        candidate.setup_completed |= completed;
        if !self.persist_tutorial(candidate) {
            return false;
        }
        self.onboarding.test_signature = None;
        self.onboarding.prepare_requested = false;
        self.model_download = None;
        // An in-flight exporter may finish its explicitly requested file, but
        // leaving the page must prevent its result from replacing a later choice.
        self.pth_convert = None;
        if !terms::accepted(&self.settings) && next != Some(Step::Language) {
            self.onboarding.return_after_terms = next;
            self.onboarding.step = Some(Step::Terms);
        } else {
            self.onboarding.step = next;
        }
        true
    }

    pub(super) fn reopen_tutorial(&mut self) {
        self.go(Some(Step::Audio), false, false);
    }

    pub(crate) fn ensure_download_terms(&mut self) -> bool {
        if !terms::accepted(&self.settings) {
            self.stop();
            self.onboarding.return_after_terms = self.onboarding.step;
            self.onboarding.step = Some(Step::Terms);
            return false;
        }
        true
    }

    fn support_ready(&self) -> bool {
        self.onboarding
            .support_check
            .as_ref()
            .is_some_and(|(key, result)| {
                key == &support_signature(&self.settings)
                    && result
                        .lock()
                        .unwrap()
                        .as_ref()
                        .is_some_and(|results| results.iter().all(Result::is_ok))
            })
    }

    fn runtime_ready(&self) -> bool {
        !self.settings.provider.starts_with("windowsml")
            || self
                .onboarding
                .runtime_check
                .as_ref()
                .is_some_and(|(provider, result)| {
                    provider == &self.settings.provider
                        && matches!(*result.lock().unwrap(), Some(Ok(())))
                })
    }

    fn test_device_signature(&self) -> String {
        format!(
            "{:?}|{:?}|{}|{}|{}",
            self.settings.input_host(),
            self.settings.output_host(),
            self.settings.input_device,
            self.settings.output_device,
            self.settings.denoiser
        )
    }

    fn start_audio_test(&mut self, output: TestOutput) {
        let config = DeviceTestConfig {
            input_host: self.settings.input_host(),
            output_host: self.settings.output_host(),
            input_device: (!self.settings.input_device.is_empty())
                .then(|| self.settings.input_device.clone()),
            output_device: (!self.settings.output_device.is_empty())
                .then(|| self.settings.output_device.clone()),
            input_exclusive: self.settings.wasapi_input_exclusive,
            output_exclusive: self.settings.wasapi_output_exclusive,
            buffer_ms: self.settings.wasapi_buffer_ms,
            rnnoise: cfg!(feature = "rnnoise") && self.settings.denoiser == "rnnoise",
            output,
        };
        self.controller.set_live_params(self.settings.live());
        #[cfg(not(test))]
        let result = self.controller.start_device_test(config);
        #[cfg(test)]
        let result: Result<(), String> = {
            self.onboarding.effects.device_configs.push(config);
            self.onboarding.effects.device_requests.push(output);
            Ok(())
        };
        match result {
            Ok(()) => self.onboarding.test_signature = Some(self.test_device_signature()),
            Err(error) => self.ui_error = Some(format!("{error:#}")),
        }
    }

    fn audio_test_ui(&mut self, ui: &mut egui::Ui, devices: &DeviceList) {
        let lang = self.settings.language;
        ui.heading(lang.text("Check your microphone and headphones"));
        ui.label(lang.text("Speak normally. Adjust the microphone volume so the meter does not frequently turn red."));
        if self.onboarding.test_signature.as_ref() != Some(&self.test_device_signature()) {
            self.start_audio_test(TestOutput::Silent);
        }
        #[cfg(not(test))]
        let state = self.controller.device_test_snapshot();
        #[cfg(test)]
        let state = self.onboarding.effects.state.clone();
        let signature_before = self.test_device_signature();
        let mut changed = false;
        device_combo(
            ui,
            lang.text(text::INPUT),
            &mut self.settings.input_device,
            &devices.inputs,
            &mut self.settings.recent_input_devices,
            &mut changed,
            true,
        );
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.input_gain, 0.0..=12.0)
                    .text(lang.text("Microphone volume")),
            )
            .changed();
        let db = |value: f32| 20.0 * value.max(0.000001).log10();
        let color = if state.clipping {
            egui::Color32::LIGHT_RED
        } else if db(state.peak) >= -6.0 {
            egui::Color32::YELLOW
        } else {
            egui::Color32::LIGHT_GREEN
        };
        ui.label(lang.text("Checking microphone…"));
        ui.label(format!(
            "{:.1} dBFS · {} {:.1} dBFS",
            db(state.rms),
            lang.text("Peak"),
            db(state.peak)
        ));
        ui.add(egui::ProgressBar::new(((db(state.rms) + 60.0) / 60.0).clamp(0.0, 1.0)).fill(color));
        if state.clipping {
            ui.colored_label(
                color,
                lang.text("Input is too loud. Lower the microphone volume."),
            );
        }
        if let Some(error) = &state.input_error {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }
        ui.separator();
        device_combo(
            ui,
            lang.text(text::OUTPUT),
            &mut self.settings.output_device,
            &devices.outputs,
            &mut self.settings.recent_output_devices,
            &mut changed,
            true,
        );
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.output_gain, 0.0..=2.0)
                    .text(lang.text(text::VOLUME)),
            )
            .changed();
        ui.horizontal_wrapped(|ui| {
            if ui
                .button(lang.text(if state.output == TestOutput::Tone {
                    text::STOP
                } else {
                    "Play test sound"
                }))
                .clicked()
            {
                self.start_audio_test(if state.output == TestOutput::Tone {
                    TestOutput::Silent
                } else {
                    TestOutput::Tone
                });
            }
            if ui
                .button(lang.text(if state.output == TestOutput::Monitor {
                    text::STOP
                } else {
                    "Hear my voice"
                }))
                .clicked()
            {
                self.start_audio_test(if state.output == TestOutput::Monitor {
                    TestOutput::Silent
                } else {
                    TestOutput::Monitor
                });
            }
        });
        ui.small(lang.text("Use headphones to hear your voice without feedback. Sound starts only when you press a button."));
        if let Some(error) = &state.output_error {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button(lang.text("Reconnect microphone")).clicked() {
                self.start_audio_test(TestOutput::Silent);
            }
            if ui.button(lang.text(text::REFRESH)).clicked() {
                #[cfg(not(test))]
                let _ = self
                    .controller
                    .refresh_devices(self.settings.input_host(), self.settings.output_host());
                self.start_audio_test(TestOutput::Silent);
            }
        });
        #[cfg(feature = "rnnoise")]
        egui::CollapsingHeader::new(lang.text("If background noise bothers you")).show(ui, |ui| {
            let mut enabled = self.settings.denoiser == "rnnoise";
            if ui.checkbox(&mut enabled, lang.text("Reduce background noise")).changed() {
                self.settings.denoiser = if enabled { "rnnoise" } else { "off" }.into(); changed = true;
                self.start_audio_test(state.output);
            }
            ui.small(lang.text("Reduces environmental noise. It can also change how your voice sounds."));
            if !matches!(self.settings.denoiser.as_str(), "off" | "rnnoise") {
                ui.small(lang.text("Your saved denoiser is preserved for conversion. This device test offers only noise reduction without extra downloads."));
            }
        });
        if changed {
            self.changed();
        }
        // Device changes always silence output. A denoiser toggle alone may
        // continue an explicitly requested monitor after worker reconstruction.
        if signature_before != self.test_device_signature()
            && self.onboarding.test_signature.as_ref() != Some(&self.test_device_signature())
        {
            self.start_audio_test(TestOutput::Silent);
        }
        ui.ctx().request_repaint_after(Duration::from_millis(50));
    }

    pub(crate) fn onboarding_ui(
        &mut self,
        ui: &mut egui::Ui,
        _status: &EngineStatusSnapshot,
        devices: &DeviceList,
    ) -> bool {
        let Some(step) = self.onboarding.step else {
            return false;
        };
        // Only the tutorial gets the larger, less dense controls. The normal
        // settings panel keeps its existing layout and sizing.
        ui.set_max_width(760.0);
        ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
        ui.spacing_mut().interact_size.y = 28.0;
        ui.spacing_mut().slider_width = 200.0;
        for (style, size) in [
            (egui::TextStyle::Body, 16.0),
            (egui::TextStyle::Button, 16.0),
            (egui::TextStyle::Small, 13.0),
            (egui::TextStyle::Heading, 24.0),
        ] {
            ui.style_mut()
                .text_styles
                .insert(style, egui::FontId::proportional(size));
        }
        let lang = self.settings.language;
        ui.label(lang.text(match step {
            Step::Language => "1. Language",
            Step::Terms => "2. Terms",
            Step::Audio => "3. Audio devices",
            Step::Voice => "4. Voice model",
            Step::Prepare => "5. Preparation",
        }));
        let mut primary_action = None;
        match step {
            Step::Language => {
                ui.heading("Choose your language / 言語を選択");
                ui.radio_value(
                    &mut self.settings.language,
                    text::Language::English,
                    "English",
                );
                ui.radio_value(
                    &mut self.settings.language,
                    text::Language::Japanese,
                    "日本語",
                );
                let selected = self.settings.language;
                ui.small(selected.text("You can change this later in settings."));
                if ui.button(selected.text("Next")).clicked() {
                    self.go(Some(Step::Audio), false, false);
                }
            }
            Step::Terms => {
                ui.heading(lang.text("Terms of use"));
                terms::view(ui, &self.settings);
                ui.horizontal_wrapped(|ui| {
                    if ui.button(lang.text(text::BACK)).clicked() {
                        self.go(Some(Step::Language), false, false);
                    }
                    if ui.button(lang.text("Agree and continue")).clicked() {
                        let mut candidate = self.settings.clone();
                        for id in terms::required(&candidate) {
                            if !candidate.accepted_terms.contains(&id) {
                                candidate.accepted_terms.push(id);
                            }
                        }
                        if self.persist_tutorial(candidate) {
                            self.onboarding.step = self.onboarding.return_after_terms;
                        }
                    }
                });
            }
            Step::Audio => {
                self.audio_test_ui(ui, devices);
                primary_action = Some(("Continue with these devices", true));
            }
            Step::Voice => {
                ui.heading(lang.text(text::VOICE_TITLE));
                ui.label(lang.text(text::VOICE_HELP));
                self.select_voice(ui);
                ui.small(lang.text(text::VOICE_ONNX));
                ui.label(lang.text(text::VOICE_COMPATIBILITY));
                egui::CollapsingHeader::new(lang.text(text::MODEL_HELP)).show(ui, |ui| {
                    ui.label(lang.text(text::MODEL_GUIDE));
                });
                let ready = if voice_exists(&self.settings) {
                    match self.onboarding.check_model(&self.settings.model) {
                        Some(Ok(())) => {
                            ui.label(lang.text(text::CHECKED));
                            true
                        }
                        Some(Err(error)) => {
                            ui.colored_label(egui::Color32::LIGHT_RED, error);
                            false
                        }
                        None => {
                            ui.spinner();
                            ui.label(lang.text(text::CHECKING));
                            ui.ctx().request_repaint_after(Duration::from_millis(100));
                            false
                        }
                    }
                } else {
                    false
                };
                primary_action = Some((text::NEXT, ready));
            }
            Step::Prepare => {
                ui.heading(lang.text("Prepare voice conversion"));
                ui.label(lang.text(text::PREPARE_HELP));
                let support_result = self.onboarding.check_support(&self.settings);
                let checked = self
                    .onboarding
                    .support_check
                    .as_ref()
                    .and_then(|(_, result)| result.lock().unwrap().clone());
                for (i, name) in ["Embedder", "F0 model"].iter().enumerate() {
                    let state = match checked.as_ref().map(|results| &results[i]) {
                        Some(Ok(())) => "Ready",
                        Some(Err(_)) => "Download required",
                        None => text::CHECKING,
                    };
                    ui.label(format!("{}: {}", lang.text(name), lang.text(state)));
                }
                if matches!(support_result, Some(Ok(()))) {
                    ui.label(lang.text(text::SUPPORT_READY));
                    if self.model_download.is_some() {
                        self.support_download(ui);
                    }
                } else if let Some(Err(error)) = support_result {
                    if support_exists(&self.settings) {
                        ui.colored_label(egui::Color32::LIGHT_RED, error);
                    }
                    self.support_download(ui);
                }
                egui::CollapsingHeader::new(lang.text("Details and existing files")).show(
                    ui,
                    |ui| {
                        ui.hyperlink_to(
                            lang.text(text::SOURCE),
                            "https://huggingface.co/wok000/weights_gpl",
                        );
                        if ui.button(lang.text("Choose embedder…")).clicked() {
                            self.browse_into(ModelKind::Embedder);
                        }
                        if ui.button(lang.text("Choose RMVPE…")).clicked() {
                            self.browse_into(ModelKind::F0);
                        }
                    },
                );
                if !self.runtime_ready() {
                    let inspection = self.inspect_runtime();
                    if terms::accepted(&self.settings) && matches!(inspection, Some(Ok(false))) {
                        // Registration of already-ready components uses the same
                        // process cache as conversion, without an acquisition step.
                        self.onboarding.prepare_requested = true;
                    }
                    // Unknown availability is not a download requirement. Only
                    // offer acquisition after inspection confirms it is needed.
                    if inspection.is_none() && !self.onboarding.prepare_requested {
                        ui.label(lang.text("Checking the runtime…"));
                    }
                    if matches!(inspection, Some(Ok(true))) && !self.onboarding.prepare_requested {
                        ui.label(lang.text("Prepare the processing components for this PC. Windows ML may download vendor components; their size depends on your device."));
                        if ui
                            .button(lang.text("Prepare processing components"))
                            .clicked()
                        {
                            if terms::accepted(&self.settings) {
                                self.onboarding.prepare_requested = true;
                                self.onboarding.runtime_check = None;
                            } else {
                                self.onboarding.return_after_terms = Some(Step::Prepare);
                                self.onboarding.step = Some(Step::Terms);
                            }
                        }
                    }
                    if let Some(Err(error)) = self
                        .onboarding
                        .runtime_inspection
                        .as_ref()
                        .and_then(|(_, result)| result.lock().unwrap().clone())
                    {
                        if self.runtime_failure_ui(ui, &error) {
                            self.onboarding.runtime_inspection = None;
                        }
                    }
                    if self.onboarding.prepare_requested {
                        #[cfg(not(test))]
                        self.runtime_preflight(ui);
                    }
                }
                // Finishing the conversion models must not dismiss an in-flight
                // GTCRN download. Keep failures here so the batch can be retried.
                let ready = voice_exists(&self.settings)
                    && self.support_ready()
                    && self.runtime_ready()
                    && self.model_download.is_none();
                if ready {
                    ui.label(
                        lang.text(
                            "Setup is ready. Open the main screen to start voice conversion.",
                        ),
                    );
                }
                // Completion updates readiness, never navigation. Let users
                // review the result even when all files were already cached.
                primary_action = Some((text::OPEN_MAIN, ready));
                ui.ctx().request_repaint_after(Duration::from_millis(100));
            }
        }
        if matches!(step, Step::Audio | Step::Voice | Step::Prepare) {
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                if ui.button(lang.text(text::BACK)).clicked() {
                    self.go(
                        Some(match step {
                            Step::Audio => Step::Language,
                            Step::Voice => Step::Audio,
                            Step::Prepare => Step::Voice,
                            _ => Step::Prepare,
                        }),
                        false,
                        false,
                    );
                }
                let skip = match step {
                    Step::Voice => "Choose a voice later",
                    _ => text::SKIP,
                };
                if ui.button(lang.text(skip)).clicked() {
                    self.go(None, true, false);
                }
                if let Some((label, enabled)) = primary_action {
                    if ui
                        .add_enabled(enabled, egui::Button::new(lang.text(label)))
                        .clicked()
                    {
                        let next = match step {
                            Step::Audio => Some(Step::Voice),
                            Step::Voice => Some(Step::Prepare),
                            _ => None,
                        };
                        self.go(next, false, step == Step::Prepare);
                    }
                }
            });
        }
        // Keep the frame owned until the next repaint, avoiding normal controls
        // underneath a just-dismissed page with stale status/device snapshots.
        true
    }
}
