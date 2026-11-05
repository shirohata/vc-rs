use super::*;
use egui_kittest::{
    kittest::{NodeT, Queryable},
    Harness,
};
use nice_plug::prelude::Params;
use std::fs;

struct Cache(PathBuf);
impl Cache {
    fn new() -> Self {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("vc-vst3-embedder-{}-{suffix}", std::process::id()));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn seed(&self, model: &vc_models::Model) {
        fs::create_dir_all(model.directory(&self.0)).unwrap();
        fs::write(
            model.model_path(&self.0),
            b"UI fixture only; never a verified ONNX",
        )
        .unwrap();
    }
}
impl Drop for Cache {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Fixture {
    editor: EditorState,
    jobs: Arc<Mutex<Vec<EmbedderJob>>>,
}

fn fixture(root: &Path) -> Fixture {
    let jobs = Arc::new(Mutex::new(Vec::new()));
    let captured = jobs.clone();
    let executor = AsyncExecutor::new(
        Arc::new(move |job| captured.lock().unwrap().push(job)),
        Arc::new(|_| panic!("Acquisition must not use the GUI executor")),
    );
    let params = Arc::new(VcRvcParams::default());
    *params.settings.write().unwrap() = PluginConfig {
        model: "voice.onnx".into(),
        embedder: "original.onnx".into(),
        f0_model: "pitch.onnx".into(),
        embedder_output: Some("custom-output".into()),
        smoother: "psola".into(),
        ..PluginConfig::default()
    };
    Fixture {
        editor: EditorState {
            params,
            reload: Arc::new(AtomicBool::new(false)),
            loading: Arc::new(AtomicBool::new(false)),
            dirty: Arc::new(AtomicBool::new(false)),
            status: Arc::new(Mutex::new(PluginStatus::new("running"))),
            reload_waker: Arc::new(ReloadWaker::default()),
            gpu_devices: Arc::new(Mutex::new(GpuDeviceDiscovery::default())),
            gpu_discovery_thread: None,
            openvino_devices: vc_core::openvino::DeviceDiscovery::default(),
            embedders: EmbedderPicker::new(Ok(root.to_owned())),
            async_executor: executor,
            model_pickers: Arc::new(pickers::ModelPickers::default()),
        },
        jobs,
    }
}

fn harness(fixture: Fixture, width: f32) -> Harness<'static, Fixture> {
    let builder = Harness::builder();
    #[cfg(feature = "ui-snapshots")]
    let builder = builder.renderer(egui_kittest::LazyRenderer::new(
        crate::ui_renderer::WgpuTestRenderer::new,
    ));
    let mut h = builder.with_size(egui::vec2(width, 1000.0)).build_ui_state(
        |ui, f: &mut Fixture| {
            f.editor.poll_embedders();
            egui::ScrollArea::vertical()
                .show(ui, |ui| crate::editor::draw_setup(ui, &mut f.editor));
        },
        fixture,
    );
    h.get_by_label("Models and conversion settings").click();
    h.run_steps(4);
    h
}

fn choose(h: &mut Harness<'_, Fixture>, label: &str) {
    h.get_by_label("Embedder").click();
    h.run_steps(4);
    h.get_by_label(label).click();
    h.run_steps(4);
}

fn finish(h: &mut Harness<'_, Fixture>) {
    let jobs = h.state().jobs.lock().unwrap();
    let job = jobs.last().unwrap();
    *job.state.lock().unwrap() = JobState::Finished(Ok(vc_models::Installed {
        path: job.model.model_path(&job.root),
        reused: job.cache_only,
    }));
    drop(jobs);
    h.run_steps(4);
}

#[test]
fn ui_download_is_explicit_staged_and_roundtrips_custom_output_and_project_state() {
    let cache = Cache::new();
    let mut h = harness(fixture(&cache.0), 880.0);
    assert!(h.state().jobs.lock().unwrap().is_empty());
    h.get_by_label("Embedder").click();
    h.run_steps(4);
    let mut previous = 0.0;
    for model in vc_models::catalog() {
        let y = h.get_by_label(&model.label).rect().min.y;
        assert!(y > previous);
        previous = y;
    }
    assert!(h.get_by_label("Custom").rect().min.y > previous);
    let model = vc_models::find("applio-spin-v2", "0.1.0").unwrap();
    h.get_by_label(&model.label).click();
    h.run_steps(4);
    assert!(h.state().jobs.lock().unwrap().is_empty());
    assert!(h
        .get_by_label("Load / Reload")
        .accesskit_node()
        .is_disabled());
    h.get_by_label("Download and use").click();
    h.run_steps(4);
    assert!(!h.state().jobs.lock().unwrap()[0].cache_only);
    assert_eq!(
        h.state().editor.params.settings.read().unwrap().embedder,
        PathBuf::from("original.onnx")
    );
    cache.seed(model);
    finish(&mut h);
    h.get_by_label("Downloaded");
    assert!(h.query_by_label("Download and use").is_none());
    assert!(h.query_by_label("Import ZIP and use…").is_none());
    assert!(h.query_by_label("Verify cache and use").is_none());
    assert!(!h
        .get_by_label("Load / Reload")
        .accesskit_node()
        .is_disabled());
    assert!(!h.state().editor.reload.load(Ordering::Relaxed));
    // Exercise the actual persisted Params fields the DAW stores, not merely
    // the config-seed serializer. Saved projects must restore both choices.
    let saved = h.state().editor.params.serialize_fields();
    let reopened = fixture(&cache.0);
    reopened.editor.params.deserialize_fields(&saved);
    let restored = reopened.editor.params.settings.read().unwrap().clone();
    assert_eq!(restored.embedder_package, Some(model.selection()));
    assert!(restored.embedder_output.is_none());
    assert_eq!(restored.smoother, "psola");
    let mut reopened = harness(reopened, 880.0);
    reopened.get_by_label("Downloaded");
    assert!(reopened.state().jobs.lock().unwrap().is_empty());
    choose(&mut reopened, "Custom");
    let restored = reopened.state().editor.params.settings.read().unwrap();
    assert_eq!(restored.embedder, PathBuf::from("original.onnx"));
    assert_eq!(restored.embedder_output.as_deref(), Some("custom-output"));
    assert!(restored.embedder_package.is_none());
    drop(restored);
    h.get_by_label("Load / Reload").click();
    h.run_steps(4);
    assert!(h.state().editor.reload.load(Ordering::Relaxed));
}

#[test]
fn ui_cached_selection_checks_locally_reselects_and_repairs_only_after_download_action() {
    let cache = Cache::new();
    let model = vc_models::find("applio-spin-v2", "0.1.0").unwrap();
    cache.seed(model);
    let mut h = harness(fixture(&cache.0), 880.0);
    choose(&mut h, &model.label);
    assert!(h.state().jobs.lock().unwrap()[0].cache_only);
    assert!(h.query_by_label("Download and use").is_none());
    // Execute the real worker against an invalid local package. No download
    // may happen, even though a public URL is present in the catalog.
    h.state().jobs.lock().unwrap().pop().unwrap().run();
    h.run_steps(4);
    h.get_by_label("Package acquisition failed. Your current selection was kept.");
    assert_eq!(
        h.state().editor.params.settings.read().unwrap().embedder,
        PathBuf::from("original.onnx")
    );
    h.get_by_label("Download and use").click();
    h.run_steps(4);
    assert!(!h.state().jobs.lock().unwrap()[0].cache_only);
    finish(&mut h);
    h.state().editor.dirty.store(false, Ordering::SeqCst);
    choose(&mut h, &model.label);
    finish(&mut h);
    assert!(!h
        .get_by_label("Load / Reload")
        .accesskit_node()
        .is_disabled());
    assert!(!h.state().editor.dirty.load(Ordering::Relaxed));
    assert!(!h.state().editor.reload.load(Ordering::Relaxed));
}

#[test]
fn ui_cancel_switch_restore_and_close_reject_late_completions() {
    for action in [
        "cancel",
        "custom",
        "voice",
        "restore",
        "same-state",
        "close",
    ] {
        let cache = Cache::new();
        let mut h = harness(fixture(&cache.0), 880.0);
        let model = vc_models::find("applio-spin-v2", "0.1.0").unwrap();
        choose(&mut h, &model.label);
        h.get_by_label("Download and use").click();
        h.run_steps(4);
        let settings = h.state().editor.params.clone();
        let cancel = h.state().jobs.lock().unwrap()[0].cancel.clone();
        let result = h.state().jobs.lock().unwrap()[0].state.clone();
        match action {
            "same-state" => {
                // A real host restore of the same settings still supersedes
                // editor-only work that was not part of that saved project.
                let saved = settings.serialize_fields();
                settings.deserialize_fields(&saved);
                h.run_steps(4);
            }
            "cancel" => {
                h.get_by_label("Cancel").click();
                h.run_steps(4);
            }
            "custom" => choose(&mut h, "Custom"),
            "voice" => {
                settings.settings.write().unwrap().model = "another.onnx".into();
                h.run_steps(4);
            }
            "restore" => {
                settings.settings.write().unwrap().embedder = "project.onnx".into();
                h.run_steps(4);
                settings.settings.write().unwrap().embedder = "original.onnx".into();
                h.run_steps(4);
            }
            "close" => {
                drop(h);
                assert!(cancel.load(Ordering::Relaxed));
                *result.lock().unwrap() = JobState::Finished(Ok(vc_models::Installed {
                    path: model.model_path(&cache.0),
                    reused: false,
                }));
                assert_eq!(
                    settings.settings.read().unwrap().embedder,
                    PathBuf::from("original.onnx")
                );
                continue;
            }
            _ => unreachable!(),
        }
        assert!(cancel.load(Ordering::Relaxed), "{action}");
        *result.lock().unwrap() = JobState::Finished(Ok(vc_models::Installed {
            path: model.model_path(&cache.0),
            reused: false,
        }));
        h.run_steps(4);
        assert!(!h
            .get_by_label("Load / Reload")
            .accesskit_node()
            .is_disabled());
        assert_eq!(
            settings.settings.read().unwrap().embedder,
            PathBuf::from("original.onnx")
        );
        assert!(!h.state().editor.reload.load(Ordering::Relaxed));
    }
}

#[test]
fn legacy_paths_are_custom_and_not_migrated_by_display() {
    let cache = Cache::new();
    let f = fixture(&cache.0);
    let old: PluginConfig = toml::from_str("model='voice.onnx'\nembedder='old.onnx'\nf0_model='pitch.onnx'\nembedder_output='output'\nsmoother='sola'").unwrap();
    *f.editor.params.settings.write().unwrap() = old;
    let h = harness(f, 520.0);
    let settings = h.state().editor.params.settings.read().unwrap();
    assert_eq!(settings.embedder, PathBuf::from("old.onnx"));
    assert_eq!(settings.embedder_output.as_deref(), Some("output"));
    assert_eq!(settings.smoother, "sola");
    assert!(settings.embedder_package.is_none());
    assert!(h.state().jobs.lock().unwrap().is_empty());
}

#[cfg(feature = "ui-snapshots")]
#[test]
#[ignore = "Render VST3 Embedder choices, missing, downloaded and failure states"]
fn render_embedder_picker() {
    let output = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/vst3-embedder-ui");
    fs::create_dir_all(&output).unwrap();
    for width in [880.0, 440.0] {
        let cache = Cache::new();
        let mut h = harness(fixture(&cache.0), width);
        h.get_by_label("Embedder").click();
        h.run_steps(4);
        h.render()
            .unwrap()
            .save(output.join(format!("{width}-choices.png")))
            .unwrap();
        let model = vc_models::find("applio-spin-v2", "0.1.0").unwrap();
        h.get_by_label(&model.label).click();
        h.run_steps(4);
        h.render()
            .unwrap()
            .save(output.join(format!("{width}-missing.png")))
            .unwrap();
        h.get_by_label("Download and use").click();
        h.run_steps(4);
        cache.seed(model);
        finish(&mut h);
        h.render()
            .unwrap()
            .save(output.join(format!("{width}-downloaded.png")))
            .unwrap();
        choose(&mut h, &model.label);
        h.state().jobs.lock().unwrap().pop().unwrap().run();
        h.run_steps(4);
        h.render()
            .unwrap()
            .save(output.join(format!("{width}-failure.png")))
            .unwrap();
    }
}

#[test]
#[ignore = "requires an explicit scratch cache; may download a pinned published package"]
fn published_embedder_worker_diagnostic() {
    let root = PathBuf::from(
        std::env::var_os("VC_RS_VST3_EMBEDDER_CACHE").expect("explicit scratch cache"),
    );
    let id = std::env::var("VC_RS_VST3_EMBEDDER_ID").unwrap_or_else(|_| "applio-spin-v2".into());
    let model = vc_models::find(&id, "0.1.0").unwrap();
    let was_cached = model.model_path(&root).is_file();
    let mut h = harness(fixture(&root), 880.0);
    choose(&mut h, &model.label);
    if !was_cached {
        h.get_by_label("Download and use").click();
        h.run_steps(4);
    }
    let job = h.state().jobs.lock().unwrap().pop().unwrap();
    assert_eq!(job.cache_only, was_cached);
    job.run();
    h.run_steps(4);
    let settings = h.state().editor.params.settings.read().unwrap();
    assert_eq!(settings.embedder_package, Some(model.selection()));
    assert_eq!(settings.embedder, model.model_path(&root));
    assert!(settings.embedder_output.is_none());
    assert!(!h.state().editor.reload.load(Ordering::Relaxed));
    h.get_by_label("Downloaded");
    assert!(h.query_by_label("Download and use").is_none());
    eprintln!(
        "published package {id}: {} verified; staged without reloading",
        if was_cached { "cache" } else { "download" }
    );
}
