use super::*;

mod details;

#[derive(Default)]
pub(crate) struct NormalState {
    pub requested: Option<GuiSettings>,
    pub applied: Option<GuiSettings>,
    pub expected_revision: u64,
    pub requested_runtime: Option<u64>,
    pub deferred_runtime: DeferredRuntimeChanges,
    details: details::ModelDetailsState,
}

#[derive(Default)]
pub(crate) struct DeferredRuntimeChanges {
    input: bool,
    output: bool,
    denoiser: bool,
}

// Compare reload-scoped engine configuration only. Live controls and UI state
// must never trigger the Restart reminder. Keep in sync with GuiSettings::live.
fn reload_signature(settings: &GuiSettings) -> String {
    let mut config = settings.clone();
    config.language = text::Language::English;
    config.input_gain = 1.0;
    config.output_gain = 1.0;
    config.pitch_shift = 0.0;
    config.speaker_id = 0;
    config.noise_gate_threshold = 0.0;
    // Device selections are acknowledged separately; history is UI-only state.
    config.input_device.clear();
    config.input_host.clear();
    config.output_device.clear();
    config.output_host.clear();
    config.denoiser.clear();
    config.gtcrn_model_dir.clear();
    config.recent_input_devices.clear();
    config.recent_output_devices.clear();
    config.passthrough = false;
    config.tutorial = None;
    config.accepted_terms.clear();
    config.setup_completed = false;
    config.setup_skipped = false;
    config.support_custom_mode = Default::default();
    config.support_custom_paths = Default::default();
    config.embedder_package = None;
    toml::to_string(&config).unwrap_or_default()
}

impl VcGui {
    pub(super) fn request_input_change(&mut self, status: &EngineStatusSnapshot) {
        self.request_device_change(status, true);
    }

    pub(super) fn request_output_change(&mut self, status: &EngineStatusSnapshot) {
        self.request_device_change(status, false);
    }

    fn request_device_change(&mut self, status: &EngineStatusSnapshot, input: bool) {
        if self.onboarding.normal.requested.is_some() {
            if input {
                self.onboarding.normal.deferred_runtime.input = true;
            } else {
                self.onboarding.normal.deferred_runtime.output = true;
            }
            return;
        }
        if !status.input_recoverable && status.state != EngineState::Running {
            return;
        }
        let (host, device) = if input {
            (
                self.settings.input_host(),
                string_option(&self.settings.input_device),
            )
        } else {
            (
                self.settings.output_host(),
                string_option(&self.settings.output_device),
            )
        };
        #[cfg(not(test))]
        let result = if input {
            self.controller.change_input(host, device)
        } else {
            self.controller.change_output(host, device)
        }
        .map_err(|e| format!("{e:#}"));
        #[cfg(test)]
        let result: Result<u64, String> = {
            if input {
                self.onboarding.effects.input_requests.push((host, device));
            } else {
                self.onboarding.effects.output_requests.push((host, device));
            }
            Ok((self.onboarding.effects.input_requests.len()
                + self.onboarding.effects.output_requests.len()
                + self.onboarding.effects.denoiser_requests.len()) as u64)
        };
        match result {
            Ok(revision) => {
                self.ui_error = None;
                self.onboarding.normal.requested_runtime = Some(revision);
            }
            Err(error) => self.ui_error = Some(error),
        }
    }

    pub(super) fn request_denoiser_change(&mut self, status: &EngineStatusSnapshot) {
        if self.onboarding.normal.requested.is_some() {
            self.onboarding.normal.deferred_runtime.denoiser = true;
            return;
        }
        if !status.input_recoverable && status.state != EngineState::Running {
            return;
        }
        let mode = match parse_denoiser(&self.settings.denoiser) {
            Ok(mode) => mode,
            Err(error) => {
                self.ui_error = Some(error);
                return;
            }
        };
        let dir = path_option(&self.settings.gtcrn_model_dir);
        #[cfg(not(test))]
        let result = self
            .controller
            .change_denoiser(mode, dir)
            .map_err(|error| format!("{error:#}"));
        #[cfg(test)]
        let result: Result<u64, String> = {
            self.onboarding.effects.denoiser_requests.push((mode, dir));
            Ok((self.onboarding.effects.input_requests.len()
                + self.onboarding.effects.output_requests.len()
                + self.onboarding.effects.denoiser_requests.len()) as u64)
        };
        match result {
            Ok(revision) => {
                self.ui_error = None;
                self.onboarding.normal.requested_runtime = Some(revision);
            }
            Err(error) => self.ui_error = Some(error),
        }
    }

    fn normal_toolbar(&mut self, ui: &mut egui::Ui, status: &EngineStatusSnapshot) {
        let lang = self.settings.language;
        if self
            .onboarding
            .normal
            .requested_runtime
            .is_some_and(|revision| status.reconfiguration_revision >= revision)
        {
            self.onboarding.normal.requested_runtime = None;
            if let Some(error) = &status.reconfiguration_error {
                self.ui_error = Some(error.clone());
            }
        }
        if status.state == EngineState::Error && !status.input_recoverable {
            self.onboarding.normal.requested = None;
            self.onboarding.normal.applied = None;
            self.onboarding.normal.requested_runtime = None;
            self.onboarding.normal.deferred_runtime = Default::default();
        }
        if status.state == EngineState::Running
            && status.session_revision >= self.onboarding.normal.expected_revision
        {
            if let Some(applied) = self.onboarding.normal.requested.take() {
                self.applied_chunk_ms = Some(applied.chunk_ms);
                self.onboarding.normal.applied = Some(applied);
            }
        }
        // A GUI poll may skip successful intermediate requests. Synchronize the
        // actual applied runtime even when the latest request failed; never infer
        // it from the selected controls or overwrite unrelated staged settings.
        if status.state == EngineState::Running || status.input_recoverable {
            if let (Some(actual), Some(applied)) =
                (&status.applied_runtime, &mut self.onboarding.normal.applied)
            {
                applied.input_host = match actual.input_host {
                    AudioHost::Wasapi => "wasapi",
                    AudioHost::Asio => "asio",
                    AudioHost::CoreAudio => "coreaudio",
                    AudioHost::Alsa => "alsa",
                    AudioHost::Jack => "jack",
                }
                .into();
                applied.input_device = actual.input_device.clone().unwrap_or_default();
                applied.output_host = match actual.output_host {
                    AudioHost::Wasapi => "wasapi",
                    AudioHost::Asio => "asio",
                    AudioHost::CoreAudio => "coreaudio",
                    AudioHost::Alsa => "alsa",
                    AudioHost::Jack => "jack",
                }
                .into();
                applied.output_device = actual.output_device.clone().unwrap_or_default();
                applied.denoiser = match actual.denoiser_mode {
                    DenoiserMode::Off => "off",
                    DenoiserMode::NoiseGate => "noise-gate",
                    DenoiserMode::Rnnoise => "rnnoise",
                    DenoiserMode::Gtcrn => "gtcrn",
                }
                .into();
                applied.gtcrn_model_dir = actual
                    .gtcrn_model_dir
                    .as_ref()
                    .map(|path| path.to_string_lossy().into_owned())
                    .unwrap_or_default();
            }
        }
        if status.state == EngineState::Running
            && self.onboarding.normal.requested.is_none()
            && status.session_revision >= self.onboarding.normal.expected_revision
        {
            // Full Apply owns an immutable settings snapshot. Edits made while
            // it loads must wait for its revision, then patch only their own
            // runtime fields through the normal coalescing mailbox. Do not
            // promote unrelated staged model/provider settings into that load.
            let deferred = std::mem::take(&mut self.onboarding.normal.deferred_runtime);
            if deferred.input {
                self.request_input_change(status);
            }
            if deferred.output {
                self.request_output_change(status);
            }
            if deferred.denoiser {
                self.request_denoiser_change(status);
            }
        }
        let busy = matches!(status.state, EngineState::Starting | EngineState::Stopping)
            || self.onboarding.normal.requested.is_some()
            || self.onboarding.normal.requested_runtime.is_some();
        let package_pending = self
            .onboarding
            .packages
            .pending_choice(&self.settings)
            .is_some();
        let running = status.state == EngineState::Running;
        ui.horizontal_wrapped(|ui| {
            if busy {
                ui.spinner();
            }
            let status_color = if status.state == EngineState::Error {
                egui::Color32::LIGHT_RED
            } else {
                ui.visuals().text_color()
            };
            let state_label = format!("{:?}", status.state);
            // The normal running message only repeats the state and audio hosts.
            // Omit it here, but keep preparation progress and diagnostics visible.
            let routine_message = status.message == state_label
                || (running
                    && status.message.starts_with("Running (in: ")
                    && status.message.ends_with(')'));
            ui.colored_label(status_color, lang.text(&state_label));
            if !status.message.is_empty() && !routine_message {
                ui.colored_label(
                    status_color,
                    ui_text::engine_message(lang, &friendly_status_message(&status.message)),
                );
            }
        });
        ui.horizontal_wrapped(|ui| {
            let label = if running { "Restart" } else { text::START };
            if ui
                .add_enabled(
                    !busy && !package_pending,
                    egui::Button::new(lang.text(label)),
                )
                .clicked()
            {
                self.apply_or_start();
            }
            if (running || busy || status.input_recoverable)
                && ui
                    .add_enabled(
                        status.state != EngineState::Stopping,
                        egui::Button::new(lang.text(text::STOP)),
                    )
                    .clicked()
            {
                self.stop();
                self.onboarding.normal.requested = None;
            }
            let live_switchable = !running || status.passthrough_live_switchable;
            if ui
                .add_enabled(
                    !busy && live_switchable,
                    egui::Checkbox::new(&mut self.settings.passthrough, lang.text("Passthrough")),
                )
                .changed()
            {
                self.controller.set_passthrough(self.settings.passthrough);
                self.changed();
            }
        });
        if running
            && self
                .onboarding
                .normal
                .applied
                .as_ref()
                .is_some_and(|s| reload_signature(s) != reload_signature(&self.settings))
        {
            ui.colored_label(
                egui::Color32::YELLOW,
                lang.text("Unapplied changes — Restart to apply."),
            );
        }
        if self.settings.passthrough {
            ui.small(lang.text("Send input audio to the output without voice conversion."));
        }
        if busy {
            ui.small(lang.text("Preparing or stopping. Stop requests are handled after the current preparation step."));
        }
        if package_pending {
            ui.small(lang.text("Acquire the selected embedder before starting."));
        }
        if let Some(error) = &self.ui_error {
            ui.colored_label(
                egui::Color32::LIGHT_RED,
                ui_text::diagnostic_message(lang, error),
            );
        }
    }

    pub(crate) fn basic_ui(
        &mut self,
        ui: &mut egui::Ui,
        status: &EngineStatusSnapshot,
        devices: &DeviceList,
    ) {
        if self.onboarding.active() {
            return;
        }
        let lang = self.settings.language;
        self.normal_toolbar(ui, status);
        ui.separator();
        // Keep transport reachable while settings, including language/setup,
        // scroll together at small window sizes or with detail sections expanded.
        let height = ui.available_height().max(80.0);
        egui::ScrollArea::vertical()
            .id_salt("normal-body")
            .auto_shrink([false, false])
            .max_height(height)
            .min_scrolled_height(height)
            .show(ui, |ui| {
                self.download_status(ui);
                ui.heading(lang.text("Voice"));
                self.select_voice(ui);
                // Structural validation belongs to Setup, not normal rendering:
                // scanning large models here competes with engine loading at Start.
                egui::CollapsingHeader::new(lang.text("Model"))
                    .id_salt("model-settings")
                    .show(ui, |ui| {
                        self.support_selector(ui, 0);
                        self.support_selector(ui, 1);
                        if ui
                            .add(
                                egui::Slider::new(&mut self.settings.speaker_id, 0..=255)
                                    .text(lang.text("Speaker ID")),
                            )
                            .changed()
                        {
                            self.changed();
                        }
                        ui.separator();
                        self.model_details_ui(ui, status);
                    });
                ui.horizontal_wrapped(|ui| {
                    ui.add_enabled_ui(!self.settings.passthrough, |ui| {
                        if ui
                            .add(
                                egui::Slider::new(&mut self.settings.pitch_shift, -24.0..=24.0)
                                    .text(lang.text(text::PITCH)),
                            )
                            .changed()
                        {
                            self.changed();
                        }
                        if ui.small_button(lang.text("Reset to 0")).clicked() {
                            self.settings.pitch_shift = 0.0;
                            self.changed();
                        }
                    });
                });
                ui.small(lang.text("Volume and pitch update live."));
                ui.separator();
                if ui.available_width() >= 620.0 {
                    ui.columns(2, |columns| {
                        self.channel_ui(&mut columns[0], devices, true, status);
                        self.channel_ui(&mut columns[1], devices, false, status);
                    });
                } else {
                    self.channel_ui(ui, devices, true, status);
                    self.channel_ui(ui, devices, false, status);
                }
                egui::CollapsingHeader::new(lang.text("Audio"))
                    .id_salt("audio-settings")
                    .show(ui, |ui| {
                        self.connection_settings_ui(ui, status, devices);
                        ui.separator();
                        details::audio_details_ui(ui, lang, status);
                    });
                egui::CollapsingHeader::new(lang.text("Backend Details"))
                    .id_salt("performance")
                    .show(ui, |ui| {
                        self.performance_settings_ui(ui, status, devices);
                        self.performance_metrics_ui(ui, status);
                        egui::CollapsingHeader::new(lang.text("Error details")).show(ui, |ui| {
                            if let Some(detail) = &status.detail {
                                ui.colored_label(egui::Color32::LIGHT_RED, detail);
                            } else {
                                ui.label(lang.text("No engine error details."));
                            }
                            if let Some(error) = &devices.error {
                                ui.colored_label(egui::Color32::LIGHT_RED, error);
                            }
                        });
                    });
                ui.separator();
                self.language_picker(ui);
                if ui.button(lang.text(text::SETUP)).clicked() {
                    self.reopen_tutorial();
                }
            });
    }

    fn channel_ui(
        &mut self,
        ui: &mut egui::Ui,
        devices: &DeviceList,
        input: bool,
        status: &EngineStatusSnapshot,
    ) {
        let lang = self.settings.language;
        let old_device = if input {
            self.settings.input_device.clone()
        } else {
            self.settings.output_device.clone()
        };
        let mut changed = false;
        ui.push_id(
            if input {
                "input-channel"
            } else {
                "output-channel"
            },
            |ui| {
                ui.heading(lang.text(if input { "Audio input" } else { "Audio output" }));
                device_combo(
                    ui,
                    lang.text(if input { text::INPUT } else { text::OUTPUT }),
                    if input {
                        &mut self.settings.input_device
                    } else {
                        &mut self.settings.output_device
                    },
                    if input {
                        &devices.inputs
                    } else {
                        &devices.outputs
                    },
                    if input {
                        &mut self.settings.recent_input_devices
                    } else {
                        &mut self.settings.recent_output_devices
                    },
                    &mut changed,
                    false,
                );
                let (active, selected) = if input {
                    (&status.input_device, &self.settings.input_device)
                } else {
                    (&status.output_device, &self.settings.output_device)
                };
                if !active.is_empty() && !selected.is_empty() && active != selected {
                    ui.small(format!(
                        "{}: {}",
                        lang.text(if input {
                            "Active input"
                        } else {
                            "Active output"
                        }),
                        active
                    ));
                }
                changed |= ui
                    .add(
                        egui::Slider::new(
                            if input {
                                &mut self.settings.input_gain
                            } else {
                                &mut self.settings.output_gain
                            },
                            0.0..=12.0,
                        )
                        .text(lang.text(if input {
                            "Input gain"
                        } else {
                            "Output gain"
                        })),
                    )
                    .changed();
                let rms = if status.state == EngineState::Running {
                    if input {
                        self.telemetry.input_rms
                    } else {
                        self.telemetry.output_rms
                    }
                } else {
                    0.0
                };
                let peak = if status.state != EngineState::Running {
                    0.0
                } else if input {
                    self.telemetry.input_peak
                } else {
                    self.telemetry.output_peak
                };
                // RMS drives the bar length, but brief full-scale peaks must
                // warn even when the average level is low. Match setup's -6 dB
                // caution threshold without exposing technical meter numbers.
                let color = if peak >= 1.0 {
                    egui::Color32::LIGHT_RED
                } else if peak >= 10.0_f32.powf(-6.0 / 20.0) {
                    egui::Color32::YELLOW
                } else {
                    egui::Color32::LIGHT_GREEN
                };
                ui.add(
                    egui::ProgressBar::new(
                        ((20.0 * rms.max(0.000001).log10() + 60.0) / 60.0).clamp(0.0, 1.0),
                    )
                    .fill(color),
                );
            },
        );
        if changed {
            self.changed();
            if input && old_device != self.settings.input_device {
                self.request_input_change(status);
            } else if !input && old_device != self.settings.output_device {
                self.request_output_change(status);
            }
        }
    }

    pub(super) fn support_selector(&mut self, ui: &mut egui::Ui, index: usize) {
        let lang = self.settings.language;
        let path = if index == 0 {
            self.settings.embedder.clone()
        } else {
            self.settings.f0_model.clone()
        };
        #[cfg(not(test))]
        let cache_dir = model_setup::cache_dir().ok();
        // Fixtures must not depend on whether this user's reference cache exists.
        #[cfg(test)]
        let cache_dir = Some(self.onboarding.effects.support_cache_dir.clone());
        let default = cache_dir
            .map(|p| {
                p.join(model_setup::MODELS[index].file)
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_default();
        let name = model_setup::MODELS[index].name;
        let package = (index == 0)
            .then(|| {
                self.settings
                    .embedder_package
                    .as_ref()
                    .and_then(vc_models::selected)
            })
            .flatten();
        let mut source = match self.settings.support_custom_mode[index].as_str() {
            "custom" => "custom",
            "downloaded" => "downloaded",
            "package" if package.is_some() => "package",
            _ => {
                if !path.is_empty() && path != default {
                    "custom"
                } else {
                    "downloaded"
                }
            }
        };
        let before = source;
        let pending = (index == 0)
            .then(|| self.onboarding.packages.pending_choice(&self.settings))
            .flatten();
        if pending.is_some() {
            source = "pending";
        }
        let mut requested_package = None;
        egui::ComboBox::new(
            ("support-source", index),
            lang.text(if index == 0 { "Embedder" } else { "F0 model" }),
        )
        .selected_text(match source {
            "custom" => lang.text("Custom"),
            "pending" => lang.text(&pending.unwrap().label),
            "package" => lang.text(&package.unwrap().label),
            _ => name,
        })
        .show_ui(ui, |ui| {
            // Preserve restored legacy ContentVec paths and their source key,
            // while offering only the published profiles for new selections.
            if index != 0 {
                ui.selectable_value(&mut source, "downloaded", name);
            }
            if index == 0 {
                for model in vc_models::catalog() {
                    let selected = pending
                        .map(|m| m.id.as_str())
                        .or_else(|| (source == "package").then(|| package.unwrap().id.as_str()));
                    if ui
                        .selectable_label(
                            selected == Some(model.id.as_str()),
                            lang.text(&model.label),
                        )
                        .clicked()
                    {
                        requested_package = Some(model);
                        ui.close();
                    }
                }
            }
            ui.selectable_value(&mut source, "custom", lang.text("Custom"));
        });
        if let Some(model) = requested_package {
            if !self.embedder_package_cached(model)
                && package.is_some_and(|installed| installed.id == model.id)
            {
                source = "package";
            } else {
                self.select_embedder_package(model);
            }
        }
        if source != "pending" && (source != before || pending.is_some()) {
            if index == 0 {
                self.onboarding.packages.cancel();
            }
            if before == "custom" {
                self.settings.support_custom_paths[index] = path.clone();
            }
            let value = if source == "custom" {
                self.settings.support_custom_paths[index].clone()
            } else if source == "package" {
                #[cfg(not(test))]
                let root = model_setup::cache_dir().ok();
                #[cfg(test)]
                let root = Some(self.onboarding.effects.support_cache_dir.clone());
                root.map(|root| {
                    package
                        .unwrap()
                        .model_path(&root)
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_default()
            } else {
                default.clone()
            };
            if index == 0 {
                self.settings.embedder = value;
            } else {
                self.settings.f0_model = value;
            }
            self.settings.support_custom_mode[index] = source.into();
            self.changed();
        }
        if index == 0 {
            let selected = self
                .onboarding
                .packages
                .pending_choice(&self.settings)
                .or_else(|| (source == "package").then_some(package).flatten());
            if let Some(model) = selected {
                self.embedder_package_controls_ui(ui, model);
                return;
            }
        }
        ui.push_id(("support", index), |ui| {
            if source == "custom" {
                let (changed, browse) = model_path_control(
                    ui,
                    lang.text("File path"),
                    if index == 0 {
                        &mut self.settings.embedder
                    } else {
                        &mut self.settings.f0_model
                    },
                );
                if browse {
                    self.browse_into(if index == 0 {
                        ModelKind::Embedder
                    } else {
                        ModelKind::F0
                    });
                }
                if changed || browse {
                    self.changed();
                }
            }
            let selected = if index == 0 {
                &self.settings.embedder
            } else {
                &self.settings.f0_model
            };
            if model_setup::available(selected) {
                ui.small(lang.text("File found"));
            } else {
                if source != "downloaded" {
                    ui.colored_label(egui::Color32::LIGHT_RED, lang.text("File not found"));
                } else {
                    ui.small(format!(
                        "{} · {:.0} MB · GPL-3.0",
                        lang.text("Download required"),
                        model_setup::MODELS[index].size as f64 / 1_000_000.0
                    ));
                    ui.hyperlink_to("GPL-3.0", "https://www.gnu.org/licenses/gpl-3.0.html");
                    if ui
                        .add_enabled(
                            self.model_download.is_none(),
                            egui::Button::new(lang.text("Agree and download")),
                        )
                        .clicked()
                    {
                        self.download_support_role(index, &default);
                    }
                }
            }
        });
    }

    fn download_support_role(&mut self, index: usize, path: &str) {
        if !self.ensure_download_terms() {
            return;
        }
        let mut candidate = self.settings.clone();
        let id = if index == 2 {
            "gtcrn-mit-502ebfab"
        } else {
            "support-models-gpl3-v1"
        }
        .to_string();
        if !candidate.accepted_terms.contains(&id) {
            candidate.accepted_terms.push(id);
        }
        if index == 0 {
            candidate.embedder = path.into();
        } else if index == 1 {
            candidate.f0_model = path.into();
        }
        if index < 2 {
            candidate.support_custom_mode[index] = "downloaded".into();
        }
        if !self.persist_tutorial(candidate) {
            return;
        }
        #[cfg(test)]
        {
            self.onboarding.effects.download_requests += 1;
            self.onboarding.effects.download_indices = vec![index];
        }
        #[cfg(not(test))]
        match model_setup::cache_dir() {
            Ok(dir) => self.model_download = Some(model_setup::Download::start(dir, vec![index])),
            Err(e) => self.ui_error = Some(e),
        }
    }
}

impl VcGui {
    fn performance_settings_ui(
        &mut self,
        ui: &mut egui::Ui,
        _status: &EngineStatusSnapshot,
        _devices: &DeviceList,
    ) {
        let lang = self.settings.language;
        let mut changed = false;
        let selected =
            Provider::from_name(&self.settings.provider).unwrap_or(vc_core::default_provider());
        let mut backend = selected.backend();
        egui::ComboBox::new("Provider", lang.text("Provider"))
            .selected_text(lang.text(backend.display_name()))
            .show_ui(ui, |ui| {
                // Build's base backends plus the device's live Windows ML
                // catalog EPs (cached in vc-core), so the picker offers what
                // is actually usable here rather than a fixed per-build list.
                for provider in self
                    .normal_providers()
                    .into_iter()
                    .filter(|p| gui_provider_visible(*p))
                    .filter(|p| p.backend() == *p)
                {
                    if ui
                        .selectable_value(
                            &mut backend,
                            provider,
                            lang.text(provider.display_name()),
                        )
                        .changed()
                    {
                        // New selections are explicit. Only restored settings
                        // retain the legacy unrestricted OpenVINO provider.
                        self.settings.provider = if backend == Provider::WindowsMlOpenVino {
                            Provider::WindowsMlOpenVinoCpu
                        } else {
                            backend
                        }
                        .label()
                        .to_owned();
                        changed = true;
                    }
                }
            });
        if let Some(mut device) = Provider::from_name(&self.settings.provider)
            .filter(|p| p.openvino_device_label().is_some())
        {
            #[cfg(not(test))]
            self.openvino_devices.poll();
            let availability = &self.openvino_devices.status;
            if availability.show_picker() {
                egui::ComboBox::new("OpenVINO Device", lang.text("OpenVINO Device"))
                    .selected_text(if device == Provider::WindowsMlOpenVino {
                        lang.text(device.openvino_device_label().unwrap())
                            .to_owned()
                    } else {
                        format!(
                            "{} ({})",
                            device.openvino_device_label().unwrap(),
                            lang.text(availability.label(device))
                        )
                    })
                    .show_ui(ui, |ui| {
                        for &candidate in Provider::OPENVINO_DEVICES {
                            if ui
                                .add_enabled_ui(
                                    availability.availability(candidate) != Some(false),
                                    |ui| {
                                        ui.selectable_value(
                                            &mut device,
                                            candidate,
                                            format!(
                                                "{} ({})",
                                                candidate.openvino_device_label().unwrap(),
                                                lang.text(availability.label(candidate))
                                            ),
                                        )
                                    },
                                )
                                .inner
                                .changed()
                            {
                                self.settings.provider = device.label().to_owned();
                                changed = true;
                            }
                        }
                    });
            }
            let mut download = false;
            let mut retry = false;
            if matches!(
                availability,
                vc_core::openvino::DeviceStatus::DownloadRequired
            ) {
                ui.small(lang.text("Download OpenVINO to check available devices."));
                download = ui.button(lang.text("Download and check")).clicked();
            }
            if matches!(availability, vc_core::openvino::DeviceStatus::Downloading) {
                ui.spinner();
                ui.small(lang.text("Downloading and preparing OpenVINO..."));
                ui.ctx().request_repaint_after(Duration::from_millis(100));
            }
            if matches!(
                availability,
                vc_core::openvino::DeviceStatus::Checking
                    | vc_core::openvino::DeviceStatus::NotStarted
            ) {
                ui.small(lang.text("Checking OpenVINO devices..."));
                ui.ctx().request_repaint_after(Duration::from_millis(100));
            }
            if let vc_core::openvino::DeviceStatus::Unknown(error) = availability {
                ui.small(lang.text("Could not verify OpenVINO devices. Hover for details."))
                    .on_hover_text(error);
                retry = ui.button(lang.text("Retry device check")).clicked();
            }
            if availability.availability(device) == Some(false) {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    lang.text("Selected device is unavailable. Choose an available device."),
                );
            }
            if download {
                self.openvino_devices.download_and_check();
            } else if retry {
                self.openvino_devices.retry();
            }
        }
        if let Some(hint) =
            Provider::from_name(&self.settings.provider).and_then(Provider::display_hint)
        {
            ui.small(lang.text(hint));
        }
        // GPU priority now applies to every backend: a process-wide Windows
        // GPU scheduling priority class (set on engine start) plus, on the
        // TensorRT path, a CUDA stream priority. So it's shown for all builds.
        egui::ComboBox::new("GPU Priority", lang.text("GPU Priority"))
            .selected_text(lang.text(&self.settings.gpu_priority))
            .show_ui(ui, |ui| {
                for priority in gpu_priority_names() {
                    changed |= ui
                        .selectable_value(
                            &mut self.settings.gpu_priority,
                            priority.to_string(),
                            lang.text(priority),
                        )
                        .changed();
                }
            });
        if gpu_device_selector_visible(&self.settings.provider) {
            #[cfg(not(test))]
            ensure_gpu_device_discovery(&self.gpu_devices);
            changed |= gpu_device_control(ui, &mut self.settings.gpu_device_id, &self.gpu_devices);
        }

        changed |= ui
            .add(
                egui::Slider::new(
                    &mut self.settings.chunk_ms,
                    CONVERSION_TIMING_LIMITS.min_chunk_ms..=CONVERSION_TIMING_LIMITS.max_chunk_ms,
                )
                // Preserve invalid saved values for validation; default
                // slider clamping would silently snap 25 ms on display.
                .clamping(egui::SliderClamping::Edits)
                .step_by(10.0)
                .text(lang.text("Chunk ms")),
            )
            .changed();
        ui.small(lang.text("RVC: 10 ms steps, with integer samples at both device rates."));
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut self.settings.extra_convert_ms,
                    GUI_MIN_EXTRA_CONVERT_MS..=CONVERSION_TIMING_LIMITS.max_extra_convert_ms,
                )
                .text(lang.text("Extra convert ms")),
            )
            .changed();

        if changed {
            self.changed();
        }
    }

    fn connection_settings_ui(
        &mut self,
        ui: &mut egui::Ui,
        status: &EngineStatusSnapshot,
        _devices: &DeviceList,
    ) {
        let lang = self.settings.language;
        let old_input_host = self.settings.input_host.clone();
        let old_output_host = self.settings.output_host.clone();
        let old_denoiser = self.settings.denoiser.clone();
        let mut changed = false;
        let mut host_changed = false;
        if gui_host_names().len() > 1 {
            backend_combo(
                ui,
                lang.text("Input backend"),
                &mut self.settings.input_host,
                &mut host_changed,
            );
            backend_combo(
                ui,
                lang.text("Output backend"),
                &mut self.settings.output_host,
                &mut host_changed,
            );
            ui.label(
                    lang.text("ASIO uses one driver for both directions; pick the same device for input and output."),
                );
        }
        if ui.button(lang.text("Refresh devices")).clicked() || host_changed {
            #[cfg(not(test))]
            let _ = self
                .controller
                .refresh_devices(self.settings.input_host(), self.settings.output_host());
        }
        changed |= host_changed;
        egui::ComboBox::new("Input denoiser", lang.text("Input denoiser"))
            .selected_text(lang.text(&self.settings.denoiser))
            .show_ui(ui, |ui| {
                for denoiser in denoiser_names() {
                    changed |= ui
                        .selectable_value(
                            &mut self.settings.denoiser,
                            denoiser.to_string(),
                            lang.text(denoiser),
                        )
                        .changed();
                }
            });
        if let Some(applied) = &self.onboarding.normal.applied {
            if applied.denoiser != self.settings.denoiser {
                ui.small(format!(
                    "{}: {}",
                    lang.text("Active noise reduction"),
                    lang.text(&applied.denoiser)
                ));
            }
        }
        if self.settings.denoiser == "noise-gate" {
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.noise_gate_threshold, 0.0001..=0.5)
                        .logarithmic(true)
                        .text(lang.text("Gate threshold")),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.noise_gate_attack_ms, 0.0..=200.0)
                        .text(lang.text("Gate attack (ms)")),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.noise_gate_release_ms, 0.0..=1000.0)
                        .text(lang.text("Gate release (ms)")),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.noise_gate_floor, 0.0..=1.0)
                        .text(lang.text("Gate floor")),
                )
                .changed();
        }
        // Model acquisition stays explicit. A prepared method change applies
        // this directory without reloading the voice-model sessions.
        if self.settings.denoiser == "gtcrn" {
            if model_setup::gtcrn_available(&self.settings.gtcrn_model_dir) {
                ui.small(lang.text("GTCRN: ready"));
                if status.input_recoverable
                    && self
                        .onboarding
                        .normal
                        .applied
                        .as_ref()
                        .is_some_and(|applied| {
                            applied.denoiser != "gtcrn"
                                || applied.gtcrn_model_dir != self.settings.gtcrn_model_dir
                        })
                    && ui.button(lang.text("Apply noise reduction")).clicked()
                {
                    self.request_denoiser_change(status);
                }
            } else {
                ui.horizontal_wrapped(|ui| {
                        ui.small("GTCRN: 352 KB · MIT · © 2024 Rong Xiaobin");
                        ui.hyperlink_to(lang.text("Source"), "https://github.com/Xiaobin-Rong/gtcrn");
                        ui.hyperlink_to(lang.text("License"), "https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/LICENSE");
                    });
                let running = self.model_download.as_ref().is_some_and(|download| {
                    matches!(
                        *download.state.lock().unwrap(),
                        model_setup::State::Running { .. }
                    )
                });
                if ui
                    .add_enabled(!running, egui::Button::new(lang.text("Download GTCRN")))
                    .clicked()
                {
                    self.download_support_role(model_setup::GTCRN_INDEX, "");
                }
                if running {
                    ui.small(lang.text("Download progress and cancellation are shown above."));
                }
            }
        }

        if changed {
            self.changed();
            if old_input_host != self.settings.input_host {
                self.request_input_change(status);
            }
            if old_output_host != self.settings.output_host {
                self.request_output_change(status);
            }
            if old_denoiser != self.settings.denoiser {
                self.request_denoiser_change(status);
            }
        }
    }
    fn performance_metrics_ui(&self, ui: &mut egui::Ui, status: &EngineStatusSnapshot) {
        let lang = self.settings.language;
        let telemetry = self.telemetry;
        egui::Grid::new("telemetry").show(ui, |ui| {
            let processing = format!("{:.1} ms", telemetry.processing_us as f64 / 1000.0);
            let color = (status.state == EngineState::Running)
                .then(|| {
                    self.applied_chunk_ms
                        .and_then(|ms| inference_color(telemetry.processing_us, ms))
                })
                .flatten();
            if let Some(color) = color {
                colored_metric(ui, lang.text("Processing (total)"), processing, color);
            } else {
                metric(ui, lang.text("Processing (total)"), processing);
            }
            metric(
                ui,
                lang.text("Content delay (nominal)"),
                format_content_delay(telemetry.content_delay_samples, status.output_sample_rate),
            );
            if status.state == EngineState::Running {
                if let Some(ms) = self.applied_chunk_ms {
                    metric(
                        ui,
                        lang.text("Processing time / available time"),
                        format!(
                            "{:.1}%",
                            telemetry.processing_us as f64 / (f64::from(ms.max(1)) * 10.0)
                        ),
                    );
                }
            }
            let inference_ms = telemetry.inference_us.saturating_add(500) / 1_000;
            let inference_color = (status.state == EngineState::Running)
                .then(|| {
                    self.applied_chunk_ms
                        .and_then(|chunk_ms| inference_color(telemetry.inference_us, chunk_ms))
                })
                .flatten();
            if let Some(color) = inference_color {
                colored_metric(
                    ui,
                    lang.text("Inference"),
                    format!("{inference_ms} ms"),
                    color,
                );
            } else {
                metric(ui, lang.text("Inference"), format!("{inference_ms} ms"));
            }
            metric(ui, lang.text("Input overruns"), telemetry.input_overruns);
            metric(
                ui,
                lang.text("Output underruns"),
                telemetry.output_underruns,
            );
            metric(
                ui,
                lang.text("Dropped output samples"),
                telemetry.output_dropped_samples,
            );
        });
        ui.small(lang.text(
            "Content delay excludes devices, queues, chunk accumulation and processing time.",
        ));
    }
}

impl VcGui {
    fn normal_providers(&self) -> Vec<Provider> {
        #[cfg(not(test))]
        {
            vc_core::selectable_providers()
        }
        #[cfg(test)]
        {
            Provider::ALL
                .iter()
                .copied()
                .filter(|p| p.available_in_build() && !p.is_catalog_ep())
                // Inject only the runtime catalog boundary; render and selection use
                // the same widgets as production, including optional vendor EPs.
                .chain(self.onboarding.effects.catalog_providers.iter().copied())
                .collect()
        }
    }
}
