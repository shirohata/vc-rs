use super::*;
use egui_kittest::{
    kittest::{NodeT, Queryable},
    Harness,
};

pub(in crate::onboarding) struct CacheFixture(pub PathBuf);

impl CacheFixture {
    pub(in crate::onboarding) fn new(model: &vc_models::Model) -> Self {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "vc-gui-package-cache-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::create_dir_all(model.directory(&root)).unwrap();
        // UI tests mock the worker completion, not real package verification.
        fs::write(model.model_path(&root), b"cache-existence fixture").unwrap();
        Self(root)
    }
}

impl Drop for CacheFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn app(language: text::Language) -> VcGui {
    let settings = GuiSettings {
        language,
        setup_completed: true,
        model: "voice.onnx".into(),
        embedder: "my-custom.onnx".into(),
        support_custom_mode: ["custom".into(), "custom".into()],
        support_custom_paths: ["my-custom.onnx".into(), "my-f0.onnx".into()],
        ..Default::default()
    };
    let mut onboarding = Onboarding::new(&settings);
    onboarding.effects.support_cache_dir =
        std::env::temp_dir().join("vc-gui-package-uncreated-fixture");
    VcGui {
        controller: EngineController::new(settings.live()),
        settings,
        dirty_since: None,
        ui_error: None,
        telemetry: TelemetrySnapshot::default(),
        telemetry_updated_at: Instant::now(),
        applied_chunk_ms: None,
        gpu_devices: Arc::new(Mutex::new(GpuDeviceDiscovery::default())),
        openvino_devices: vc_core::openvino::DeviceDiscovery::default(),
        pth_convert: None,
        model_download: None,
        onboarding,
    }
}

fn harness(app: VcGui) -> Harness<'static, VcGui> {
    harness_sized(app, egui::vec2(900.0, 900.0))
}

fn harness_sized(app: VcGui, size: egui::Vec2) -> Harness<'static, VcGui> {
    let builder = Harness::builder();
    #[cfg(feature = "ui-snapshots")]
    let builder = builder.renderer(egui_kittest::LazyRenderer::new(
        crate::onboarding::ui_renderer::WgpuTestRenderer::new,
    ));
    builder.with_size(size).build_ui_state(
        |ui, app: &mut VcGui| {
            app.poll_embedder_package();
            app.embedder_packages_ui(ui);
        },
        app,
    )
}

#[cfg(feature = "ui-snapshots")]
#[test]
#[ignore = "Render expanded package controls without acquisition or settings writes"]
fn render_embedder_packages() {
    let destination =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/embedder-package-ui");
    fs::create_dir_all(&destination).unwrap();
    for language in [text::Language::English, text::Language::Japanese] {
        for width in [880.0, 520.0] {
            let mut fixture = app(language);
            let model = vc_models::find("applio-chinese-hubert-base", "0.1.0").unwrap();
            // Use an isolated dummy cache so the downloaded state and its long
            // document path render without touching any real model packages.
            let cache = CacheFixture::new(model);
            fixture.onboarding.effects.support_cache_dir = cache.0.clone();
            fixture.settings.embedder_package = Some(model.selection());
            fixture.settings.embedder = model
                .model_path(&fixture.onboarding.effects.support_cache_dir)
                .to_string_lossy()
                .into_owned();
            fixture.settings.support_custom_mode[0] = "package".into();
            let mut h = harness_sized(fixture, egui::vec2(width, 600.0));
            install_system_japanese_font(&h.ctx);
            h.run_steps(4);
            h.get_by_label(language.text("Embedder packages")).click();
            // Settle font/layout updates before inspecting the expanded body.
            h.run_steps(20);
            let document_path = model
                .directory(&h.state().onboarding.effects.support_cache_dir)
                .to_string_lossy()
                .into_owned();
            let rect = h.get_by_label(&document_path).rect();
            assert!(rect.right() <= width && rect.bottom() <= 600.0);
            for shape in &h.output().shapes {
                if let egui::epaint::Shape::Text(text) = &shape.shape {
                    if text.galley.job.text == document_path {
                        assert!(
                            shape.clip_rect.contains_rect(rect),
                            "Clipped document path in {language:?} at {width}"
                        );
                    }
                }
            }
            h.render()
                .unwrap()
                .save(destination.join(format!("{language:?}-{width}.png")))
                .unwrap();
        }
    }
}

pub(in crate::onboarding) fn finish(app: &mut VcGui, success: bool) {
    let task = app.onboarding.packages.task.as_ref().unwrap();
    let result = if success {
        Ok(vc_models::Installed {
            path: task.model.model_path(&app.package_root().unwrap()),
            reused: false,
        })
    } else {
        Err("fixture checksum mismatch".into())
    };
    *task.state.lock().unwrap() = TaskState::Finished(result);
}

#[test]
fn ui_package_acquisition_is_explicit_localized_and_keeps_custom_until_success() {
    for language in [text::Language::English, text::Language::Japanese] {
        let mut h = harness(app(language));
        assert!(h.state().onboarding.effects.package_requests.is_empty());
        h.get_by_label(language.text("Embedder packages")).click();
        h.run_steps(4);
        assert!(h
            .query_by_label(language.text("No public download source is configured."),)
            .is_none());
        h.get_by_label(language.text("Download and use"));
        h.get_by_label(language.text("Package")).click();
        h.run_steps(4);
        h.get_by_label(language.text("SPIN v2 (Applio)")).click();
        h.run_steps(4);
        assert!(h.state().onboarding.effects.package_requests.is_empty());
        assert!(h.query_by_label("Import ZIP and use…").is_none());
        assert!(h.query_by_label("ZIPを取り込んで使用…").is_none());
        h.get_by_label(language.text("Download and use")).click();
        h.run_steps(4);
        assert_eq!(
            h.state().onboarding.effects.package_requests,
            ["applio-spin-v2"]
        );
        assert_eq!(h.state().settings.embedder, "my-custom.onnx");
        assert!(h.state().settings.embedder_package.is_none());
        finish(h.state_mut(), true);
        h.run_steps(4);
        let model = vc_models::find("applio-spin-v2", "0.1.0").unwrap();
        let settings = &h.state().settings;
        assert_eq!(settings.embedder_package, Some(model.selection()));
        assert_eq!(settings.support_custom_mode[0], "package");
        assert_eq!(settings.support_custom_paths[0], "my-custom.onnx");
        h.get_by_label(
            language.text("Package verified and selected. Start or reload conversion to apply."),
        );
        let restored: GuiSettings = toml::from_str(&toml::to_string(settings).unwrap()).unwrap();
        assert_eq!(restored.embedder, settings.embedder);
        assert_eq!(restored.embedder_package, settings.embedder_package);
        assert_eq!(restored.smoother, settings.smoother);
    }
}

#[test]
fn package_completion_cannot_overwrite_changed_or_aba_restored_selections() {
    for change_voice in [false, true] {
        let mut app = app(text::Language::English);
        let model = vc_models::find("applio-spin", "0.1.0").unwrap();
        app.onboarding.packages.selected = model.id.clone();
        app.start_embedder_package(model, false);
        if change_voice {
            app.settings.model = "new-voice.onnx".into();
        } else {
            app.settings.embedder = "new-custom.onnx".into();
        }
        app.changed();
        if change_voice {
            app.settings.model = "voice.onnx".into();
        } else {
            app.settings.embedder = "my-custom.onnx".into();
        }
        app.changed();
        finish(&mut app, true);
        app.poll_embedder_package();
        assert_eq!(app.settings.embedder, "my-custom.onnx");
        assert!(app.settings.embedder_package.is_none());
        assert_eq!(
            app.onboarding.packages.message,
            Some("Package cached. Your current model selection was kept.")
        );
    }
}

#[test]
fn ui_package_failure_retry_and_cancel_preserve_selection_and_ignore_late_result() {
    let language = text::Language::Japanese;
    let mut h = harness(app(language));
    h.get_by_label(language.text("Embedder packages")).click();
    h.run_steps(4);
    h.get_by_label(language.text("Download and use")).click();
    h.run_steps(4);
    finish(h.state_mut(), false);
    h.run_steps(4);
    h.get_by_label(language.text("Package acquisition failed. Your current selection was kept."));
    assert_eq!(h.state().settings.embedder, "my-custom.onnx");
    h.get_by_label(language.text("Download and use")).click();
    h.run_steps(4);
    assert_eq!(h.state().onboarding.effects.package_requests.len(), 2);
    let task = h.state().onboarding.packages.task.as_ref().unwrap();
    let cancel = task.cancel.clone();
    let state = task.state.clone();
    h.get_by_label(language.text(text::CANCEL)).click();
    h.run_steps(4);
    assert!(cancel.load(Ordering::Relaxed));
    *state.lock().unwrap() = TaskState::Finished(Ok(vc_models::Installed {
        path: "late.onnx".into(),
        reused: false,
    }));
    h.run_steps(4);
    assert_eq!(h.state().settings.embedder, "my-custom.onnx");
    assert!(h.state().settings.embedder_package.is_none());
}

#[test]
fn ui_package_choice_change_cancels_previous_work_without_changing_active_path() {
    let mut h = harness(app(text::Language::English));
    h.get_by_label("Embedder packages").click();
    h.run_steps(4);
    h.get_by_label("Download and use").click();
    h.run_steps(4);
    assert!(h
        .get_by_label("Download and use")
        .accesskit_node()
        .is_disabled());
    let cancel = h
        .state()
        .onboarding
        .packages
        .task
        .as_ref()
        .unwrap()
        .cancel
        .clone();
    h.get_by_label("Package").click();
    h.run_steps(4);
    h.get_by_label("SPIN (Applio)").click();
    h.run_steps(4);
    assert!(cancel.load(Ordering::Relaxed));
    assert_eq!(h.state().settings.embedder, "my-custom.onnx");
    assert!(h.state().onboarding.packages.task.is_none());
}

#[test]
fn explicit_missing_custom_and_package_paths_survive_discovery_and_old_settings() {
    let legacy: GuiSettings =
        toml::from_str("embedder = 'custom.onnx'\nsmoother = 'psola'\n").unwrap();
    assert!(legacy.embedder_package.is_none());
    assert_eq!(legacy.smoother, "psola");
    for mode in ["custom", "package"] {
        let mut settings = legacy.clone();
        settings.support_custom_mode[0] = mode.into();
        settings.embedder_package = Some(vc_models::Selection {
            id: "future-id".into(),
            version: "99.0.0".into(),
            onnx_sha256: "f".repeat(64),
        });
        discover_support_models(&mut settings);
        assert_eq!(settings.embedder, legacy.embedder);
        assert_eq!(settings.smoother, legacy.smoother);
        let restored: GuiSettings = toml::from_str(&toml::to_string(&settings).unwrap()).unwrap();
        assert_eq!(restored.embedder_package, settings.embedder_package);
        assert_eq!(restored.embedder, legacy.embedder);
    }
}

#[test]
fn staging_a_voice_export_invalidates_a_package_before_model_path_changes() {
    let mut app = app(text::Language::English);
    let model = vc_models::find("applio-spin", "0.1.0").unwrap();
    app.onboarding.packages.selected = model.id.clone();
    app.start_embedder_package(model, false);
    app.onboarding.invalidate_package_application();
    finish(&mut app, true);
    app.poll_embedder_package();
    assert_eq!(app.settings.embedder, "my-custom.onnx");
    assert!(app.settings.embedder_package.is_none());
}
