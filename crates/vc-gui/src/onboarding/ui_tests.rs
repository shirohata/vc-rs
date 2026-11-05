//! Exercise production widgets with deterministic state, without opening devices,
//! discovering models, or touching the user's persisted settings. Keep these
//! tests below the eframe lifecycle: `VcGui::new` and `maybe_save` perform I/O.
use super::*;
use vc_app::AppliedRuntimeConfig;
#[cfg(any(feature = "windowsml", feature = "tensorrt"))]
#[path = "backend_tests.rs"]
mod backend_tests;
use egui_kittest::{
    kittest::{NodeT, Queryable},
    Harness,
};

struct Fixture {
    app: VcGui,
    status: EngineStatusSnapshot,
    devices: DeviceList,
    setup_owns_screen: bool,
}

fn ready_main_fixture(language: text::Language) -> Fixture {
    let mut fixture = Fixture::new(None, EngineState::Stopped);
    fixture.app.settings.language = language;
    let file = std::env::current_exe()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    fixture.app.settings.model = file.clone();
    fixture.app.settings.embedder = file.clone();
    fixture.app.settings.f0_model = file.clone();
    fixture.app.onboarding.model_check = Some((file, Arc::new(Mutex::new(Some(Ok(()))))));
    fixture.app.onboarding.support_check = Some((
        support_signature(&fixture.app.settings),
        Arc::new(Mutex::new(Some([Ok(()), Ok(())]))),
    ));
    fixture
}

#[test]
fn ui_runtime_edits_during_start_wait_for_the_loaded_revision_and_apply_once() {
    for restarting in [false, true] {
        let mut fixture = ready_main_fixture(text::Language::English);
        fixture.app.settings.input_device = "Old microphone".into();
        fixture.app.settings.output_device = "Old speaker".into();
        fixture.devices.inputs = vec!["Old microphone".into(), "New microphone".into()];
        fixture.devices.outputs = vec!["Old speaker".into(), "New speaker".into()];
        if restarting {
            fixture.status.state = EngineState::Running;
            fixture.status.session_revision = 7;
            fixture.status.input_recoverable = true;
            fixture.app.onboarding.normal.applied = Some(fixture.app.settings.clone());
        }
        let mut h = harness(fixture);
        h.get_by_label(if restarting { "Restart" } else { text::START })
            .click();
        h.run_steps(4);
        let revision = h.state().app.onboarding.normal.expected_revision;
        // Editing before the controller publishes Starting is possible, too.
        h.get_by_value("Old microphone").click();
        h.run_steps(4);
        h.get_by_label("New microphone").click();
        h.run_steps(4);
        h.state_mut().status.state = EngineState::Starting;
        h.run_steps(4);
        h.get_by_value("Old speaker").click();
        h.run_steps(4);
        h.get_by_label("New speaker").click();
        h.run_steps(4);
        let original = h.state().app.onboarding.normal.requested.clone().unwrap();
        {
            let fixture = h.state_mut();
            fixture.app.settings.denoiser = "noise-gate".into();
            fixture.app.request_denoiser_change(&fixture.status);
            // Repeated edits should coalesce to the latest field values.
            fixture.app.settings.input_device = "Final microphone".into();
            fixture.app.request_input_change(&fixture.status);
            fixture.app.settings.provider = "staged backend".into();
            fixture.app.settings.model = "staged voice.onnx".into();
            fixture.status.state = EngineState::Running;
            fixture.status.session_revision = revision - 1;
        }
        h.run_steps(4);
        assert!(h.state().app.onboarding.effects.input_requests.is_empty());
        assert!(h.state().app.onboarding.effects.output_requests.is_empty());
        assert!(h
            .state()
            .app
            .onboarding
            .effects
            .denoiser_requests
            .is_empty());
        // Even a matching revision must not acknowledge an unfinished load.
        h.state_mut().status.state = EngineState::Starting;
        h.state_mut().status.session_revision = revision;
        h.run_steps(4);
        assert!(h.state().app.onboarding.normal.requested.is_some());
        h.state_mut().status.state = EngineState::Running;
        h.run_steps(8);
        let fixture = h.state();
        assert_eq!(fixture.app.onboarding.effects.start_requests, 1);
        assert_eq!(
            fixture.app.onboarding.effects.input_requests,
            [(AudioHost::default(), Some("Final microphone".into()))]
        );
        assert_eq!(
            fixture.app.onboarding.effects.output_requests,
            [(AudioHost::default(), Some("New speaker".into()))]
        );
        assert_eq!(
            fixture.app.onboarding.effects.denoiser_requests,
            [(DenoiserMode::NoiseGate, None)]
        );
        assert_eq!(fixture.app.onboarding.normal.requested_runtime, Some(3));
        let applied = fixture.app.onboarding.normal.applied.as_ref().unwrap();
        assert_eq!(applied.model, original.model);
        assert_eq!(applied.provider, original.provider);
        assert_eq!(applied.input_device, "Old microphone");
        assert_eq!(applied.output_device, "Old speaker");
        assert_eq!(applied.denoiser, "off");
        {
            let fixture = h.state_mut();
            fixture.status.reconfiguration_revision = 3;
            fixture.status.applied_runtime = Some(AppliedRuntimeConfig {
                input_device: Some("Final microphone".into()),
                output_device: Some("New speaker".into()),
                denoiser_mode: DenoiserMode::NoiseGate,
                ..Default::default()
            });
        }
        h.run_steps(4);
        let fixture = h.state();
        assert!(fixture.app.onboarding.normal.requested_runtime.is_none());
        let applied = fixture.app.onboarding.normal.applied.as_ref().unwrap();
        assert_eq!(applied.input_device, "Final microphone");
        assert_eq!(applied.output_device, "New speaker");
        assert_eq!(applied.denoiser, "noise-gate");
        assert_eq!(applied.model, original.model);
        assert_eq!(applied.provider, original.provider);
        assert_eq!(fixture.app.onboarding.effects.input_requests.len(), 1);
    }
}

#[test]
fn ui_stop_and_load_failure_clear_deferred_runtime_edits_before_a_new_apply() {
    for action in ["stop", "load failure", "new apply"] {
        let mut fixture = ready_main_fixture(text::Language::English);
        fixture.app.apply_or_start();
        fixture.app.settings.input_device = "New microphone".into();
        fixture.app.request_input_change(&fixture.status);
        fixture.app.settings.output_device = "New speaker".into();
        fixture.app.request_output_change(&fixture.status);
        fixture.app.settings.denoiser = "noise-gate".into();
        fixture.app.request_denoiser_change(&fixture.status);
        let mut h = harness(fixture);
        match action {
            "stop" => {
                h.get_by_label(text::STOP).click();
                h.run_steps(4);
            }
            "load failure" => {
                h.state_mut().status.state = EngineState::Error;
                h.run_steps(4);
            }
            _ => {}
        }
        if action != "new apply" {
            // A late Running snapshot of the abandoned attempt must not revive
            // its deferred edits, even before a replacement Apply is submitted.
            h.state_mut().status.state = EngineState::Running;
            h.state_mut().status.session_revision = 1;
            h.run_steps(4);
            assert!(
                h.state().app.onboarding.effects.input_requests.is_empty(),
                "{action}"
            );
            assert!(
                h.state().app.onboarding.effects.output_requests.is_empty(),
                "{action}"
            );
            assert!(
                h.state()
                    .app
                    .onboarding
                    .effects
                    .denoiser_requests
                    .is_empty(),
                "{action}"
            );
        }
        // A fresh Apply already includes all selected runtime fields. Deferred
        // requests from the previous attempt must not restart it a second time.
        h.state_mut().app.apply_or_start();
        h.state_mut().status.state = EngineState::Running;
        h.state_mut().status.session_revision = h.state().app.onboarding.normal.expected_revision;
        h.run_steps(8);
        let fixture = h.state();
        assert!(
            fixture.app.onboarding.effects.input_requests.is_empty(),
            "{action}"
        );
        assert!(
            fixture.app.onboarding.effects.output_requests.is_empty(),
            "{action}"
        );
        assert!(
            fixture.app.onboarding.effects.denoiser_requests.is_empty(),
            "{action}"
        );
        let applied = fixture.app.onboarding.normal.applied.as_ref().unwrap();
        assert_eq!(applied.input_device, "New microphone");
        assert_eq!(applied.output_device, "New speaker");
        assert_eq!(applied.denoiser, "noise-gate");
    }
}

#[test]
fn ui_output_widget_switches_without_restart_and_keeps_staged_models() {
    for lang in [text::Language::English, text::Language::Japanese] {
        let mut fixture = ready_main_fixture(lang);
        fixture.status.state = EngineState::Running;
        fixture.status.input_recoverable = true;
        fixture.status.output_device = "Old speaker".into();
        fixture.devices.outputs = vec!["Old speaker".into(), "New speaker".into()];
        fixture.app.settings.output_device = "Old speaker".into();
        fixture.app.settings.input_device = "Applied microphone".into();
        fixture.app.onboarding.normal.applied = Some(fixture.app.settings.clone());
        let old_model = fixture.app.settings.model.clone();
        let mut h = harness(fixture);
        h.get_by_value("Old speaker").click();
        h.run_steps(4);
        h.get_by_label("New speaker").click();
        h.run_steps(4);
        assert_eq!(
            h.state().app.onboarding.effects.output_requests,
            vec![(AudioHost::default(), Some("New speaker".into()))]
        );
        assert!(h.state().app.onboarding.effects.input_requests.is_empty());
        assert_eq!(h.state().app.onboarding.effects.start_requests, 0);
        assert!(h
            .query_by_label(lang.text("Unapplied changes — Restart to apply."))
            .is_none());
        {
            let fixture = h.state_mut();
            fixture.app.settings.model = "Staged voice.onnx".into();
            fixture.app.settings.provider = "staged provider".into();
            fixture.status.output_device = "New speaker".into();
            fixture.status.applied_runtime = Some(AppliedRuntimeConfig {
                input_device: Some("Applied microphone".into()),
                output_device: Some("New speaker".into()),
                ..Default::default()
            });
            fixture.status.reconfiguration_revision = 1;
        }
        h.run_steps(4);
        let applied = h.state().app.onboarding.normal.applied.as_ref().unwrap();
        assert_eq!(applied.output_device, "New speaker");
        assert_eq!(applied.input_device, "Applied microphone");
        assert_eq!(applied.model, old_model);
        assert_ne!(applied.provider, "staged provider");
        assert!(h.state().app.onboarding.normal.requested_runtime.is_none());
    }
}

#[test]
fn ui_unobserved_default_output_success_survives_later_failure() {
    let mut fixture = ready_main_fixture(text::Language::English);
    fixture.status.state = EngineState::Running;
    fixture.status.input_recoverable = true;
    fixture.app.settings.output_device = "Old speaker".into();
    fixture.app.onboarding.normal.applied = Some(fixture.app.settings.clone());
    fixture.app.settings.output_device.clear();
    fixture.app.request_output_change(&fixture.status);
    fixture.app.settings.output_device = "Unavailable speaker".into();
    fixture.app.request_output_change(&fixture.status);
    fixture.status.output_device = "Resolved default speaker".into();
    fixture.status.applied_runtime = Some(AppliedRuntimeConfig {
        output_host: AudioHost::Asio,
        output_device: None,
        ..Default::default()
    });
    fixture.status.reconfiguration_revision = 2;
    fixture.status.reconfiguration_error = Some("Selected output is unavailable".into());
    let mut h = harness(fixture);
    h.run_steps(4);
    let applied = h.state().app.onboarding.normal.applied.as_ref().unwrap();
    assert!(
        applied.output_device.is_empty(),
        "Default intent must survive missed ACKs"
    );
    assert_eq!(applied.output_host, "asio");
    assert_eq!(h.state().app.settings.output_device, "Unavailable speaker");
    assert!(h.state().app.ui_error.is_some());
    assert_eq!(h.state().app.onboarding.effects.start_requests, 0);
}

#[test]
fn ui_unobserved_success_before_latest_failure_keeps_actual_applied_runtime() {
    let mut fixture = ready_main_fixture(text::Language::English);
    fixture.status.state = EngineState::Running;
    fixture.status.input_recoverable = true;
    fixture.app.settings.input_device = "mic A".into();
    fixture.app.onboarding.normal.applied = Some(fixture.app.settings.clone());
    let old_model = fixture.app.settings.model.clone();
    fixture.app.settings.model = "staged voice.onnx".into();
    fixture.app.settings.provider = "staged provider".into();

    // A succeeds on the engine before the GUI polls. B is submitted with the
    // stale snapshot, and the GUI observes only B's failed completion.
    fixture.app.settings.input_device = "mic B".into();
    fixture.app.request_input_change(&fixture.status);
    fixture.app.settings.denoiser = "noise-gate".into();
    fixture.app.request_denoiser_change(&fixture.status);
    fixture.app.settings.denoiser = "gtcrn".into();
    fixture.app.settings.gtcrn_model_dir = "staged GTCRN path".into();
    fixture.app.request_denoiser_change(&fixture.status);
    fixture.status.input_device = "mic B".into();
    fixture.status.applied_runtime = Some(AppliedRuntimeConfig {
        input_host: AudioHost::Asio,
        input_device: Some("mic B".into()),
        denoiser_mode: DenoiserMode::NoiseGate,
        gtcrn_model_dir: Some(PathBuf::from("accepted GTCRN path")),
        ..Default::default()
    });
    fixture.status.reconfiguration_revision = 3;
    fixture.status.reconfiguration_error = Some("GTCRN preparation failed".into());
    let mut h = harness(fixture);
    h.run_steps(4);
    let applied = h.state().app.onboarding.normal.applied.as_ref().unwrap();
    assert_eq!(applied.input_device, "mic B");
    assert_eq!(applied.input_host, "asio");
    assert_eq!(applied.denoiser, "noise-gate");
    assert_eq!(applied.gtcrn_model_dir, "accepted GTCRN path");
    assert_eq!(applied.model, old_model);
    assert_ne!(applied.provider, "staged provider");
    assert_eq!(h.state().app.settings.gtcrn_model_dir, "staged GTCRN path");
    assert!(h.state().app.ui_error.is_some());
}

#[test]
fn ui_unobserved_default_input_success_preserves_intent_and_saved_paths() {
    for fail_input in [false, true] {
        let mut fixture = ready_main_fixture(text::Language::English);
        fixture.status.state = EngineState::Running;
        fixture.status.input_recoverable = true;
        fixture.app.settings.input_device = "old explicit microphone".into();
        fixture.app.settings.gtcrn_model_dir = "saved model directory".into();
        fixture.app.onboarding.normal.applied = Some(fixture.app.settings.clone());
        fixture.app.settings.input_device.clear();
        fixture.app.request_input_change(&fixture.status);
        if fail_input {
            fixture.app.settings.input_device = "unavailable microphone".into();
            fixture.app.request_input_change(&fixture.status);
        } else {
            fixture.app.settings.denoiser = "gtcrn".into();
            fixture.app.settings.gtcrn_model_dir = "new saved selection".into();
            fixture.app.request_denoiser_change(&fixture.status);
        }
        fixture.status.input_device = "resolved Default endpoint".into();
        fixture.status.applied_runtime = Some(AppliedRuntimeConfig {
            input_device: None,
            denoiser_mode: DenoiserMode::Off,
            gtcrn_model_dir: Some(PathBuf::from("saved model directory")),
            ..Default::default()
        });
        fixture.status.reconfiguration_revision = 2;
        fixture.status.reconfiguration_error = Some("Latest selection failed".into());
        let mut h = harness(fixture);
        h.run_steps(4);
        let applied = h.state().app.onboarding.normal.applied.as_ref().unwrap();
        assert!(
            applied.input_device.is_empty(),
            "Default must not become a resolved endpoint name"
        );
        assert_eq!(applied.gtcrn_model_dir, "saved model directory");
        if fail_input {
            assert_eq!(
                h.state().app.settings.input_device,
                "unavailable microphone"
            );
        } else {
            assert!(h.state().app.settings.input_device.is_empty());
            assert_eq!(
                h.state().app.settings.gtcrn_model_dir,
                "new saved selection"
            );
        }
    }
}

#[test]
fn ui_unobserved_gtcrn_success_keeps_accepted_path_after_replacement_failure() {
    let mut fixture = ready_main_fixture(text::Language::English);
    fixture.status.state = EngineState::Running;
    fixture.status.input_recoverable = true;
    fixture.app.onboarding.normal.applied = Some(fixture.app.settings.clone());
    fixture.app.settings.denoiser = "gtcrn".into();
    fixture.app.settings.gtcrn_model_dir = "accepted directory".into();
    fixture.app.request_denoiser_change(&fixture.status);
    fixture.app.settings.gtcrn_model_dir = "rejected directory".into();
    fixture.app.request_denoiser_change(&fixture.status);
    fixture.status.applied_runtime = Some(AppliedRuntimeConfig {
        denoiser_mode: DenoiserMode::Gtcrn,
        gtcrn_model_dir: Some(PathBuf::from("accepted directory")),
        ..Default::default()
    });
    fixture.status.reconfiguration_revision = 2;
    fixture.status.reconfiguration_error = Some("Replacement model failed".into());
    let mut h = harness(fixture);
    h.run_steps(4);
    let applied = h.state().app.onboarding.normal.applied.as_ref().unwrap();
    assert_eq!(applied.denoiser, "gtcrn");
    assert_eq!(applied.gtcrn_model_dir, "accepted directory");
    assert_eq!(h.state().app.settings.gtcrn_model_dir, "rejected directory");
}

#[test]
fn ui_input_change_acknowledges_only_input_and_failure_keeps_applied_selection() {
    let mut fixture = ready_main_fixture(text::Language::English);
    fixture.status.state = EngineState::Running;
    fixture.status.input_recoverable = true;
    fixture.app.onboarding.normal.applied = Some(fixture.app.settings.clone());
    let original_model = fixture.app.settings.model.clone();
    fixture.app.settings.model = "unapplied-model.onnx".into();
    fixture.app.settings.output_device = "unapplied-output".into();
    fixture.app.settings.input_device = "new microphone".into();
    fixture.app.request_input_change(&fixture.status);
    assert_eq!(fixture.app.onboarding.effects.input_requests.len(), 1);
    fixture.status.reconfiguration_revision = 1;
    fixture.status.applied_runtime = Some(AppliedRuntimeConfig {
        input_device: Some("new microphone".into()),
        ..Default::default()
    });
    let mut h = harness(fixture);
    h.run_steps(4);
    let applied = h.state().app.onboarding.normal.applied.as_ref().unwrap();
    assert_eq!(applied.input_device, "new microphone");
    assert_eq!(applied.model, original_model);
    assert_ne!(applied.output_device, "unapplied-output");
    {
        let fixture = h.state_mut();
        fixture.app.settings.input_device = "missing microphone".into();
        fixture.app.request_input_change(&fixture.status);
        fixture.status.reconfiguration_revision = 2;
        fixture.status.reconfiguration_error = Some("Selected microphone is unavailable".into());
    }
    h.run_steps(4);
    assert_eq!(
        h.state()
            .app
            .onboarding
            .normal
            .applied
            .as_ref()
            .unwrap()
            .input_device,
        "new microphone"
    );
    assert!(h.state().app.ui_error.is_some());
    assert_eq!(h.state().app.onboarding.effects.start_requests, 0);
}

#[test]
fn ui_denoiser_changes_and_coalesced_input_wait_for_latest_ack_without_restart() {
    let mut fixture = ready_main_fixture(text::Language::English);
    fixture.status.state = EngineState::Running;
    fixture.status.input_recoverable = true;
    fixture.app.onboarding.normal.applied = Some(fixture.app.settings.clone());
    fixture.app.settings.input_device = "new microphone".into();
    fixture.app.request_input_change(&fixture.status);
    fixture.app.settings.denoiser = "noise-gate".into();
    fixture.app.request_denoiser_change(&fixture.status);
    fixture.status.reconfiguration_revision = 1;
    fixture.status.applied_runtime = Some(AppliedRuntimeConfig {
        input_device: Some("new microphone".into()),
        ..Default::default()
    });
    let mut h = harness(fixture);
    h.run_steps(4);
    assert_eq!(
        h.state()
            .app
            .onboarding
            .normal
            .applied
            .as_ref()
            .unwrap()
            .input_device,
        "new microphone"
    );
    assert_eq!(
        h.state()
            .app
            .onboarding
            .normal
            .applied
            .as_ref()
            .unwrap()
            .denoiser,
        "off"
    );
    assert!(h.state().app.onboarding.normal.requested_runtime.is_some());
    h.state_mut().status.reconfiguration_revision = 2;
    h.state_mut()
        .status
        .applied_runtime
        .as_mut()
        .unwrap()
        .denoiser_mode = DenoiserMode::NoiseGate;
    h.run_steps(4);
    let applied = h.state().app.onboarding.normal.applied.as_ref().unwrap();
    assert_eq!(applied.input_device, "new microphone");
    assert_eq!(applied.denoiser, "noise-gate");
    assert!(h
        .query_by_label("Unapplied changes — Restart to apply.")
        .is_none());
    {
        let fixture = h.state_mut();
        fixture.app.settings.denoiser = "rnnoise".into();
        fixture.app.request_denoiser_change(&fixture.status);
        fixture.status.reconfiguration_revision = 3;
        fixture
            .status
            .applied_runtime
            .as_mut()
            .unwrap()
            .denoiser_mode = DenoiserMode::Rnnoise;
    }
    h.run_steps(4);
    assert_eq!(
        h.state()
            .app
            .onboarding
            .normal
            .applied
            .as_ref()
            .unwrap()
            .denoiser,
        "rnnoise"
    );
    {
        let fixture = h.state_mut();
        fixture.app.settings.denoiser = "gtcrn".into();
        fixture.app.request_denoiser_change(&fixture.status);
        fixture.status.reconfiguration_revision = 4;
        fixture.status.reconfiguration_error = Some("GTCRN preparation failed".into());
    }
    h.run_steps(4);
    assert_eq!(
        h.state()
            .app
            .onboarding
            .normal
            .applied
            .as_ref()
            .unwrap()
            .denoiser,
        "rnnoise"
    );
    assert!(h.state().app.ui_error.is_some());
    assert!(h
        .query_by_label("Unapplied changes — Restart to apply.")
        .is_none());
    assert_eq!(h.state().app.onboarding.effects.start_requests, 0);
}

#[test]
fn ui_normal_screen_and_start_do_not_spawn_model_validation() {
    for lang in [text::Language::English, text::Language::Japanese] {
        let mut fixture = ready_main_fixture(lang);
        fixture.app.onboarding.model_check = None;
        fixture.app.onboarding.support_check = None;
        let mut h = harness(fixture);
        h.get_by_label(lang.text("Model")).click();
        h.run_steps(4);
        assert!(h.state().app.onboarding.model_check.is_none());
        assert!(h.state().app.onboarding.support_check.is_none());
        assert!(h.query_by_label(lang.text(text::CHECKING)).is_none());
        h.get_by_label(lang.text(text::START)).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.effects.start_requests, 1);
        assert!(h.state().app.onboarding.model_check.is_none());
        assert!(h.state().app.onboarding.support_check.is_none());
    }
}

#[test]
fn ui_hides_smoother_and_preserves_saved_choice_through_start() {
    for lang in [text::Language::English, text::Language::Japanese] {
        for (value, expected) in [
            ("sola", Smoother::Sola),
            ("psola", Smoother::Psola),
            ("phase-vocoder", Smoother::PhaseVocoder),
        ] {
            let mut fixture = ready_main_fixture(lang);
            fixture.app.settings.smoother = value.into();
            let mut h = harness(fixture);
            h.get_by_label(lang.text("Backend Details")).scroll_to_me();
            h.run_steps(8);
            h.get_by_label(lang.text("Backend Details")).click();
            h.run_steps(4);
            for label in [
                "Smoother",
                "チャンク接続方式",
                "SOLA",
                "PSOLA",
                "Phase vocoder",
            ] {
                assert!(
                    h.query_by_label(label).is_none(),
                    "unexpected control: {label}"
                );
            }
            assert_eq!(h.state().app.settings.smoother, value);
            let saved = toml::to_string(&h.state().app.settings).unwrap();
            let restored: GuiSettings = toml::from_str(&saved).unwrap();
            assert_eq!(restored.realtime().unwrap().smoother, expected);
            assert_eq!(h.state().app.onboarding.effects.start_requests, 0);
            h.get_by_label(lang.text(text::START)).scroll_to_me();
            h.run_steps(8);
            h.get_by_label(lang.text(text::START)).click();
            h.run_steps(4);
            assert_eq!(h.state().app.onboarding.effects.start_requests, 1);
            assert_eq!(
                h.state()
                    .app
                    .onboarding
                    .normal
                    .requested
                    .as_ref()
                    .unwrap()
                    .realtime()
                    .unwrap()
                    .smoother,
                expected
            );
        }
    }
}

#[test]
fn ui_start_does_not_redirect_for_setup_checks() {
    for lang in [text::Language::English, text::Language::Japanese] {
        for check_state in 0..3 {
            let mut fixture = ready_main_fixture(lang);
            #[cfg(feature = "windowsml")]
            {
                fixture.app.settings.provider = "windowsml-cpu".into();
            }
            fixture.app.onboarding.runtime_check = None;
            fixture.app.onboarding.support_check = match check_state {
                0 => None,
                1 => Some((
                    support_signature(&fixture.app.settings),
                    Arc::new(Mutex::new(None)),
                )),
                _ => Some((
                    support_signature(&fixture.app.settings),
                    Arc::new(Mutex::new(Some([
                        Err("Test validation failure".into()),
                        Ok(()),
                    ]))),
                )),
            };
            if check_state == 0 {
                fixture.app.settings.accepted_terms.clear();
            }
            let mut h = harness(fixture);
            h.get_by_label(lang.text(text::START)).click();
            h.run_steps(4);
            assert_eq!(h.state().app.onboarding.effects.start_requests, 1);
            assert!(h.state().app.onboarding.step.is_none());
            assert!(!h.state().setup_owns_screen);
        }
    }
}

#[test]
fn ui_normal_transport_pending_and_passthrough() {
    for lang in [text::Language::English, text::Language::Japanese] {
        let mut h = harness(ready_main_fixture(lang));
        assert!(h.query_by_label(lang.text(text::STOP)).is_none());
        h.get_by_label(lang.text(text::START)).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.effects.start_requests, 1);
        assert!(h
            .get_by_label(lang.text(text::START))
            .accesskit_node()
            .is_disabled());
        h.state_mut().status.state = EngineState::Running;
        h.state_mut().status.session_revision = 1;
        h.state_mut().status.passthrough_live_switchable = true;
        h.run_steps(4);
        h.get_by_label(lang.text("Restart"));
        h.state_mut().app.settings.input_gain = 2.0;
        h.state_mut().app.settings.pitch_shift = 3.0;
        h.run_steps(4);
        assert!(h
            .query_by_label(lang.text("Unapplied changes — Restart to apply."))
            .is_none());
        h.state_mut().app.settings.extra_convert_ms += 10;
        h.run_steps(4);
        h.get_by_label(lang.text("Unapplied changes — Restart to apply."));
        h.get_by_label(lang.text("Passthrough")).click();
        h.run_steps(4);
        assert!(h.state().app.settings.passthrough);
        assert!(h.query_all_by_label(lang.text(text::PITCH)).count() > 0);
        assert!(h
            .query_all_by_label(lang.text(text::PITCH))
            .all(|node| node.accesskit_node().is_disabled()));
        h.get_by_label(lang.text("Restart")).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.effects.start_requests, 2);
        assert_eq!(h.state().app.onboarding.normal.expected_revision, 2);
        assert!(h.state().app.onboarding.normal.requested.is_some());
        assert!(h
            .get_by_label(lang.text("Restart"))
            .accesskit_node()
            .is_disabled());
        h.get_by_label(lang.text("Restart")).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.effects.start_requests, 2);
        // An old Running snapshot must not acknowledge the queued restart.
        h.state_mut().status.state = EngineState::Starting;
        h.state_mut().status.session_revision = 2;
        h.run_steps(4);
        assert!(h.state().app.onboarding.normal.requested.is_some());
        h.state_mut().status.state = EngineState::Running;
        h.run_steps(4);
        assert!(h.state().app.onboarding.normal.requested.is_none());
        assert_eq!(
            h.state()
                .app
                .onboarding
                .normal
                .applied
                .as_ref()
                .unwrap()
                .extra_convert_ms,
            h.state().app.settings.extra_convert_ms
        );
        assert!(h
            .query_by_label(lang.text("Unapplied changes — Restart to apply."))
            .is_none());
        h.get_by_label(lang.text(text::STOP)).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.effects.stop_requests, 1);
        assert!(h.state().app.onboarding.normal.applied.is_none());
        h.state_mut().status.state = EngineState::Stopped;
        h.run_steps(4);
        assert!(!h
            .get_by_label(lang.text(text::START))
            .accesskit_node()
            .is_disabled());
    }
}

#[test]
fn ui_f0_sources_preserve_custom_path_and_download_only_selected_role() {
    for lang in [text::Language::English, text::Language::Japanese] {
        let index = 1;
        let mut h = harness(ready_main_fixture(lang));
        h.get_by_label(lang.text("Model")).click();
        h.run_steps(4);
        let original = h.state().app.settings.embedder.clone();
        // Combo controls have distinct labels even though both offer Custom.
        h.get_by_label(lang.text("F0 model")).click();
        h.run_steps(2);
        h.get_by_label("RMVPE").click();
        h.run_steps(4);
        assert_eq!(h.state().app.settings.support_custom_paths[index], original);
        assert_eq!(
            h.state().app.settings.support_custom_mode[index],
            "downloaded"
        );
        h.get_by_label(lang.text("Agree and download")).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.effects.download_requests, 1);
        assert_eq!(h.state().app.onboarding.effects.download_indices, [index]);
        let saved = h.state().app.onboarding.effects.saved.last().unwrap();
        assert_eq!(&saved.embedder, &original);
        let expected = h
            .state()
            .app
            .onboarding
            .effects
            .support_cache_dir
            .join(model_setup::MODELS[index].file);
        assert_eq!(&saved.f0_model, &expected.to_string_lossy());
        h.get_by_label(lang.text("F0 model")).click();
        h.run_steps(2);
        h.get_by_label(lang.text("Custom")).click();
        h.run_steps(4);
        let settings = &h.state().app.settings;
        assert_eq!(&settings.f0_model, &original);
    }
}

#[test]
fn ui_small_screen_scroll_keeps_transport_and_reaches_language_and_setup() {
    for lang in [text::Language::English, text::Language::Japanese] {
        let mut h = harness_sized(ready_main_fixture(lang), egui::vec2(520.0, 480.0));
        let start = h.get_by_label(lang.text(text::START)).rect();
        h.get_by_label(lang.text("Backend Details")).scroll_to_me();
        h.run_steps(8);
        h.get_by_label(lang.text("Backend Details")).click();
        h.run_steps(4);
        h.get_by_label("Language / 言語").scroll_to_me();
        h.run_steps(8);
        assert_eq!(h.get_by_label(lang.text(text::START)).rect(), start);
        h.get_by_label("Language / 言語").click();
        h.run_steps(4);
        let (choice, changed) = if lang == text::Language::English {
            ("日本語", text::Language::Japanese)
        } else {
            ("English", text::Language::English)
        };
        h.get_by_label(choice).click();
        h.run_steps(4);
        assert_eq!(h.state().app.settings.language, changed);
        h.get_by_label(changed.text(text::SETUP)).scroll_to_me();
        h.run_steps(8);
        let setup = h.get_by_label(changed.text(text::SETUP)).rect();
        assert!(setup.min.y >= 20.0 && setup.max.y <= 460.0);
        assert!(h.get_by_label(changed.text(text::START)).rect().min.y < setup.min.y);
        h.get_by_label(changed.text(text::SETUP)).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.step, Some(Step::Audio));
    }
}

#[test]
fn ui_embedder_picker_lists_every_package_before_download_and_applies_after_verification() {
    for lang in [text::Language::English, text::Language::Japanese] {
        let fixture = ready_main_fixture(lang);
        let original = fixture.app.settings.embedder.clone();
        let mut h = harness(fixture);
        h.get_by_label(lang.text("Model")).click();
        h.run_steps(4);
        h.get_by_label(lang.text("Embedder")).click();
        h.run_steps(4);
        assert!(h.query_by_label("ContentVec").is_none());
        let mut previous_y = 0.0;
        for model in vc_models::catalog() {
            let node = h.get_by_label(lang.text(&model.label));
            assert!(node.rect().min.y > previous_y);
            previous_y = node.rect().min.y;
        }
        assert!(h.get_by_label(lang.text("Custom")).rect().min.y > previous_y);
        h.get_by_label(lang.text("SPIN v2 (Applio)")).click();
        h.run_steps(4);
        assert_eq!(h.state().app.settings.embedder, original);
        assert!(h.state().app.settings.embedder_package.is_none());
        assert!(h.state().app.onboarding.effects.package_requests.is_empty());
        assert!(h
            .get_by_label(lang.text(text::START))
            .accesskit_node()
            .is_disabled());
        assert!(h.query_by_label(lang.text("Embedder packages")).is_none());
        h.get_by_label(lang.text("Download and use")).click();
        h.run_steps(4);
        assert_eq!(
            h.state().app.onboarding.effects.package_requests,
            ["applio-spin-v2"]
        );
        assert_eq!(h.state().app.settings.embedder, original);
        packages::tests::finish(&mut h.state_mut().app, true);
        h.run_steps(4);
        let model = vc_models::find("applio-spin-v2", "0.1.0").unwrap();
        let settings = &h.state().app.settings;
        assert_eq!(settings.embedder_package, Some(model.selection()));
        assert_eq!(settings.support_custom_paths[0], original);
        assert_eq!(settings.support_custom_mode[0], "package");
        assert!(!h
            .get_by_label(lang.text(text::START))
            .accesskit_node()
            .is_disabled());
    }
}

#[test]
fn ui_cached_embedder_selection_verifies_without_download_and_preserves_settings_on_failure() {
    for lang in [text::Language::English, text::Language::Japanese] {
        for outcome in ["success", "failure", "cancel"] {
            let model = vc_models::find("applio-spin-v2", "0.1.0").unwrap();
            let cache = packages::tests::CacheFixture::new(model);
            let mut fixture = ready_main_fixture(lang);
            let original = fixture.app.settings.embedder.clone();
            fixture.app.onboarding.effects.support_cache_dir = cache.0.clone();
            let mut h = harness(fixture);
            h.get_by_label(lang.text("Model")).click();
            h.run_steps(4);
            h.get_by_label(lang.text("Embedder")).click();
            h.run_steps(4);
            h.get_by_label(lang.text(&model.label)).click();
            h.run_steps(4);
            assert_eq!(h.state().app.settings.embedder, original);
            assert!(h.state().app.onboarding.effects.package_requests.is_empty());
            assert_eq!(
                h.state()
                    .app
                    .onboarding
                    .effects
                    .package_cache_checks
                    .as_slice(),
                std::slice::from_ref(&model.id)
            );
            assert!(h.query_by_label(lang.text("Download and use")).is_none());
            assert!(h
                .get_by_label(lang.text(text::START))
                .accesskit_node()
                .is_disabled());
            if outcome == "cancel" {
                h.get_by_label(lang.text(text::CANCEL)).click();
                h.run_steps(4);
                assert_eq!(h.state().app.settings.embedder, original);
                assert!(h.state().app.settings.embedder_package.is_none());
                assert!(!h
                    .get_by_label(lang.text(text::START))
                    .accesskit_node()
                    .is_disabled());
            } else {
                packages::tests::finish(&mut h.state_mut().app, outcome == "success");
                h.run_steps(4);
                if outcome == "success" {
                    assert_eq!(
                        h.state().app.settings.embedder_package,
                        Some(model.selection())
                    );
                    assert_eq!(h.state().app.settings.support_custom_paths[0], original);
                    h.get_by_label(lang.text("Downloaded"));
                    assert!(h.query_by_label(lang.text("Download and use")).is_none());
                    assert!(!h
                        .get_by_label(lang.text(text::START))
                        .accesskit_node()
                        .is_disabled());
                    // Restore the settings in a fresh GUI, without doing work just
                    // to render the already selected, downloaded profile.
                    let mut restored = ready_main_fixture(lang);
                    restored.app.settings = h.state().app.settings.clone();
                    restored.app.onboarding.effects.support_cache_dir = cache.0.clone();
                    let mut restored = harness(restored);
                    restored.get_by_label(lang.text("Model")).click();
                    restored.run_steps(4);
                    restored.get_by_label(lang.text("Downloaded"));
                    assert!(restored
                        .query_by_label(lang.text("Download and use"))
                        .is_none());
                    assert!(restored
                        .state()
                        .app
                        .onboarding
                        .effects
                        .package_cache_checks
                        .is_empty());
                    // Re-selecting the active model must also finish the preview
                    // when verification leaves every saved setting unchanged.
                    restored.get_by_label(lang.text("Embedder")).click();
                    restored.run_steps(4);
                    restored.get_by_label(lang.text(&model.label)).click();
                    restored.run_steps(4);
                    packages::tests::finish(&mut restored.state_mut().app, true);
                    restored.run_steps(4);
                    assert!(!restored
                        .get_by_label(lang.text(text::START))
                        .accesskit_node()
                        .is_disabled());
                } else {
                    assert_eq!(h.state().app.settings.embedder, original);
                    assert!(h.state().app.settings.embedder_package.is_none());
                    h.get_by_label(lang.text("Download and use")).click();
                    h.run_steps(4);
                    assert_eq!(
                        h.state().app.onboarding.effects.package_requests.as_slice(),
                        std::slice::from_ref(&model.id)
                    );
                    assert_eq!(h.state().app.settings.embedder, original);
                }
            }
        }
    }
}

#[test]
fn ui_embedder_download_cancel_restores_running_choice_without_showing_its_docs_for_another() {
    for lang in [text::Language::English, text::Language::Japanese] {
        let installed = vc_models::find("applio-spin-v2", "0.1.0").unwrap();
        let cache = packages::tests::CacheFixture::new(installed);
        let mut fixture = ready_main_fixture(lang);
        let installed_path = installed
            .model_path(&cache.0)
            .to_string_lossy()
            .into_owned();
        fixture.app.settings.embedder = installed_path.clone();
        fixture.app.settings.embedder_package = Some(installed.selection());
        fixture.app.settings.support_custom_mode[0] = "package".into();
        fixture.app.onboarding.effects.support_cache_dir = cache.0.clone();
        fixture.status.state = EngineState::Running;
        fixture.app.onboarding.normal.applied = Some(fixture.app.settings.clone());
        let mut h = harness(fixture);
        h.get_by_label(lang.text("Model")).click();
        h.run_steps(4);
        h.get_by_label(lang.text("Embedder")).click();
        h.run_steps(4);
        h.get_by_label(lang.text("SPIN (Applio)")).click();
        h.run_steps(4);
        assert_eq!(h.state().app.settings.embedder, installed_path);
        assert!(h
            .get_by_label(lang.text("Restart"))
            .accesskit_node()
            .is_disabled());
        assert!(!h
            .get_by_label(lang.text(text::STOP))
            .accesskit_node()
            .is_disabled());
        assert!(h
            .query_by_label(lang.text("Package documents (README / README.ja / LICENSE):"))
            .is_none());
        h.get_by_label(lang.text("Download and use")).click();
        h.run_steps(4);
        h.get_by_label(lang.text(text::CANCEL)).click();
        h.run_steps(4);
        h.get_by_value(lang.text(&installed.label));
        assert!(!h
            .get_by_label(lang.text("Restart"))
            .accesskit_node()
            .is_disabled());
        assert_eq!(h.state().app.settings.embedder, installed_path);
        assert_eq!(h.state().app.onboarding.effects.start_requests, 0);
        assert_eq!(h.state().app.onboarding.effects.stop_requests, 0);
        assert_eq!(h.state().status.state, EngineState::Running);
    }
}

#[test]
fn ui_embedder_pending_choice_can_return_to_custom_after_failed_download() {
    for lang in [text::Language::English, text::Language::Japanese] {
        let fixture = ready_main_fixture(lang);
        let original = fixture.app.settings.embedder.clone();
        let mut h = harness(fixture);
        h.get_by_label(lang.text("Model")).click();
        h.run_steps(4);
        h.get_by_label(lang.text("Embedder")).click();
        h.run_steps(4);
        h.get_by_label(lang.text("SPIN (Applio)")).click();
        h.run_steps(4);
        h.get_by_label(lang.text("Download and use")).click();
        h.run_steps(4);
        packages::tests::finish(&mut h.state_mut().app, false);
        h.run_steps(4);
        h.get_by_label(lang.text("Package acquisition failed. Your current selection was kept."));
        h.get_by_label(lang.text("Embedder")).click();
        h.run_steps(4);
        h.get_by_label(lang.text("Custom")).click();
        h.run_steps(4);
        assert_eq!(h.state().app.settings.embedder, original);
        assert!(h.state().app.settings.embedder_package.is_none());
        assert!(h
            .state()
            .app
            .onboarding
            .packages
            .pending_choice(&h.state().app.settings)
            .is_none());
        assert!(!h
            .get_by_label(lang.text(text::START))
            .accesskit_node()
            .is_disabled());
    }
}

#[test]
fn ui_legacy_contentvec_configuration_survives_picker_simplification() {
    for lang in [text::Language::English, text::Language::Japanese] {
        let mut fixture = ready_main_fixture(lang);
        let legacy_path = fixture
            .app
            .onboarding
            .effects
            .support_cache_dir
            .join(model_setup::MODELS[0].file)
            .to_string_lossy()
            .into_owned();
        fixture.app.settings.embedder = legacy_path.clone();
        fixture.app.settings.support_custom_mode[0] = "downloaded".into();
        fixture.app.settings.support_custom_paths[0] = "remembered-custom.onnx".into();
        fixture.app.settings.smoother = "psola".into();
        let mut h = harness(fixture);
        h.get_by_label(lang.text("Model")).click();
        h.run_steps(4);
        let settings = &h.state().app.settings;
        assert_eq!(settings.embedder, legacy_path);
        assert_eq!(settings.support_custom_mode[0], "downloaded");
        assert_eq!(settings.support_custom_paths[0], "remembered-custom.onnx");
        assert_eq!(settings.smoother, "psola");
        assert!(settings.embedder_package.is_none());
        assert!(h.state().app.onboarding.effects.package_requests.is_empty());
        assert_eq!(h.state().app.onboarding.effects.download_requests, 0);
    }
}

#[cfg(feature = "ui-snapshots")]
#[test]
#[ignore = "Render the ordinary Embedder picker; no acquisition or settings writes"]
fn render_normal_embedder_picker() {
    let destination =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/embedder-picker-ui");
    fs::create_dir_all(&destination).unwrap();
    for lang in [text::Language::English, text::Language::Japanese] {
        for width in [880.0, 520.0] {
            let mut h = harness_sized(ready_main_fixture(lang), egui::vec2(width, 720.0));
            install_system_japanese_font(&h.ctx);
            h.run_steps(4);
            h.get_by_label(lang.text("Model")).click();
            h.run_steps(4);
            h.get_by_label(lang.text("Embedder")).scroll_to_me();
            h.run_steps(8);
            h.get_by_label(lang.text("Embedder")).click();
            h.run_steps(4);
            h.render()
                .unwrap()
                .save(destination.join(format!("{lang:?}-{width}-choices.png")))
                .unwrap();
            h.get_by_label(lang.text("SPIN v2 (Applio)")).click();
            h.run_steps(4);
            h.get_by_label(lang.text("Download and use")).scroll_to_me();
            h.run_steps(8);
            h.render()
                .unwrap()
                .save(destination.join(format!("{lang:?}-{width}-selected.png")))
                .unwrap();
            let model = vc_models::find("applio-spin-v2", "0.1.0").unwrap();
            let cache = packages::tests::CacheFixture::new(model);
            h.state_mut().app.onboarding.effects.support_cache_dir = cache.0.clone();
            h.state_mut().app.select_embedder_package(model);
            packages::tests::finish(&mut h.state_mut().app, true);
            h.run_steps(4);
            h.get_by_label(lang.text("Downloaded")).scroll_to_me();
            h.run_steps(8);
            h.render()
                .unwrap()
                .save(destination.join(format!("{lang:?}-{width}-downloaded.png")))
                .unwrap();
        }
    }
}

#[test]
fn ui_installed_package_and_custom_remain_distinct_saved_choices() {
    for lang in [text::Language::English, text::Language::Japanese] {
        let mut fixture = ready_main_fixture(lang);
        let model = vc_models::find("applio-spin-v2", "0.1.0").unwrap();
        let package_path = model
            .model_path(&fixture.app.onboarding.effects.support_cache_dir)
            .to_string_lossy()
            .into_owned();
        fixture.app.settings.embedder = package_path.clone();
        fixture.app.settings.embedder_package = Some(model.selection());
        fixture.app.settings.support_custom_mode[0] = "package".into();
        fixture.app.settings.support_custom_paths[0] = "my-preserved-custom.onnx".into();
        let mut h = harness(fixture);
        h.get_by_label(lang.text("Model")).click();
        h.run_steps(4);
        // A missing managed package offers its published download.
        assert!(h.query_by_label(lang.text("Agree and download")).is_none());
        h.get_by_label(lang.text("Embedder")).click();
        h.run_steps(2);
        h.get_by_label(lang.text("Custom")).click();
        h.run_steps(4);
        assert_eq!(h.state().app.settings.embedder, "my-preserved-custom.onnx");
        assert_eq!(
            h.state().app.settings.embedder_package,
            Some(model.selection())
        );
        h.get_by_label(lang.text("Embedder")).click();
        h.run_steps(2);
        h.get_by_label(lang.text(&model.label)).click();
        h.run_steps(4);
        let settings = &h.state().app.settings;
        assert_eq!(settings.embedder, package_path);
        assert_eq!(settings.support_custom_paths[0], "my-preserved-custom.onnx");
        assert_eq!(settings.support_custom_mode[0], "package");
        assert_eq!(h.state().app.onboarding.effects.download_requests, 0);
        assert!(h.state().app.onboarding.effects.package_requests.is_empty());
    }
}

#[test]
fn ui_stop_failure_keeps_applied_settings_and_shows_error() {
    let mut h = harness(ready_main_fixture(text::Language::English));
    h.get_by_label(text::START).click();
    h.run_steps(4);
    h.state_mut().status.state = EngineState::Running;
    h.state_mut().status.session_revision = 1;
    h.run_steps(4);
    h.state_mut().app.onboarding.effects.stop_error = Some("Test stop failure".into());
    h.get_by_label(text::STOP).click();
    h.run_steps(4);
    assert_eq!(h.state().app.onboarding.effects.stop_requests, 1);
    assert!(h.state().app.onboarding.normal.applied.is_some());
    h.get_by_label("Test stop failure");

    h.state_mut().app.onboarding.effects.stop_error = None;
    h.get_by_label(text::STOP).click();
    h.run_steps(4);
    assert_eq!(h.state().app.onboarding.effects.stop_requests, 2);
    assert!(h.state().app.onboarding.normal.applied.is_none());
    assert!(h.state().app.ui_error.is_none());
    assert!(h.query_by_label("Test stop failure").is_none());
}

#[test]
fn ui_device_selection_orders_history_and_preserves_disconnected_devices() {
    for lang in [text::Language::English, text::Language::Japanese] {
        for input in [true, false] {
            let mut fixture = ready_main_fixture(lang);
            fixture.devices.inputs = vec!["A".into(), "B".into(), "C".into(), "D".into()];
            fixture.devices.outputs = fixture.devices.inputs.clone();
            let settings = &mut fixture.app.settings;
            if input {
                settings.input_device = "Disconnected".into();
                settings.recent_input_devices = vec!["C".into(), "Disconnected".into(), "B".into()];
            } else {
                settings.output_device = "Disconnected".into();
                settings.recent_output_devices =
                    vec!["C".into(), "Disconnected".into(), "B".into()];
            }
            let mut h = harness(fixture);
            h.get_by_value("Disconnected").click();
            h.run_steps(4);
            // The disconnected selection remains displayed, but is not an option.
            assert_eq!(h.query_all_by_label("Disconnected").count(), 0);
            let mut previous_y = 0.0;
            for name in ["C", "B", "A", "D"] {
                let node = h.get_by_label(name); // also rejects duplicate rows
                assert!(node.rect().min.y > previous_y);
                previous_y = node.rect().min.y;
            }
            h.get_by_label("A").click();
            h.run_steps(4);
            let settings = &h.state().app.settings;
            let (selected, history, other, other_history) = if input {
                (
                    &settings.input_device,
                    &settings.recent_input_devices,
                    &settings.output_device,
                    &settings.recent_output_devices,
                )
            } else {
                (
                    &settings.output_device,
                    &settings.recent_output_devices,
                    &settings.input_device,
                    &settings.recent_input_devices,
                )
            };
            assert_eq!(selected, "A");
            assert_eq!(history, &["A", "C", "Disconnected"]);
            assert!(other.is_empty() && other_history.is_empty());
            assert!(h.state().app.dirty_since.is_some());
            // Reconnecting makes the remembered device selectable in MRU order.
            if input {
                h.state_mut().devices.inputs.push("Disconnected".into());
            } else {
                h.state_mut().devices.outputs.push("Disconnected".into());
            }
            h.get_by_value("A").click();
            h.run_steps(4);
            assert!(h.get_by_label("C").rect().min.y < h.get_by_label("Disconnected").rect().min.y);
            assert!(h.get_by_label("Disconnected").rect().min.y < h.get_by_label("B").rect().min.y);
            h.get_by_label("Disconnected").click();
            h.run_steps(4);
            h.get_by_value("Disconnected").click();
            h.run_steps(4);
            h.get_by_label(lang.text("System default")).click();
            h.run_steps(4);
            let settings = &h.state().app.settings;
            let (selected, history) = if input {
                (&settings.input_device, &settings.recent_input_devices)
            } else {
                (&settings.output_device, &settings.recent_output_devices)
            };
            assert!(selected.is_empty());
            assert_eq!(history, &["Disconnected", "A", "C"]);
        }
    }
}

#[test]
fn ui_failed_start_can_retry_and_language_is_not_a_restart_change() {
    let mut fixture = ready_main_fixture(text::Language::English);
    fixture.app.onboarding.effects.start_error = Some("Test start failure".into());
    let mut h = harness(fixture);
    h.get_by_label(text::START).click();
    h.run_steps(4);
    assert!(h.state().app.onboarding.normal.requested.is_none());
    assert!(!h.get_by_label(text::START).accesskit_node().is_disabled());
    h.state_mut().app.onboarding.effects.start_error = None;
    h.get_by_label(text::START).click();
    h.run_steps(4);
    h.state_mut().status.state = EngineState::Running;
    h.state_mut().status.session_revision = 1;
    h.run_steps(4);
    h.state_mut().app.settings.language = text::Language::Japanese;
    h.run_steps(4);
    assert!(h
        .query_by_label(text::Language::Japanese.text("Unapplied changes — Restart to apply."))
        .is_none());
    assert_eq!(h.state().app.onboarding.effects.start_requests, 2);
}

#[cfg(feature = "ui-snapshots")]
#[test]
#[ignore = "Render the normal screen without devices, inference, or settings writes"]
fn render_normal_pages() {
    let destination = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/normal-ui");
    fs::create_dir_all(&destination).unwrap();
    for lang in [text::Language::English, text::Language::Japanese] {
        for width in [880.0, 520.0] {
            let mut h = harness_sized(ready_main_fixture(lang), egui::vec2(width, 680.0));
            install_system_japanese_font(&h.ctx);
            h.run_steps(4);
            h.render()
                .unwrap()
                .save(destination.join(format!("{lang:?}-{width}.png")))
                .unwrap();
            h.get_by_label(lang.text("Backend Details")).click();
            h.run_steps(4);
            h.render()
                .unwrap()
                .save(destination.join(format!("{lang:?}-{width}-expanded.png")))
                .unwrap();
        }
    }
}

impl Fixture {
    fn new(step: Option<Step>, state: EngineState) -> Self {
        let mut settings = GuiSettings::default();
        settings.accepted_terms = terms::required(&settings);
        let mut onboarding = Onboarding::new(&settings);
        onboarding.step = step;
        onboarding.effects.support_cache_dir =
            std::env::temp_dir().join(format!("vc-gui-missing-cache-{}", std::process::id()));
        assert!(!onboarding.effects.support_cache_dir.exists());
        onboarding.runtime_check = Some((
            settings.provider.clone(),
            Arc::new(Mutex::new(Some(Ok(())))),
        ));
        onboarding.support_check = Some((
            support_signature(&settings),
            Arc::new(Mutex::new(Some([
                Err("Missing test support files".into()),
                Err("Missing test support files".into()),
            ]))),
        ));
        Self {
            app: VcGui {
                // Start/Stop and downloads are injected. Device discovery still
                // belongs to integration tests; do not click Refresh here.
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
            },
            status: EngineStatusSnapshot {
                state,
                ..Default::default()
            },
            devices: DeviceList {
                inputs: vec!["Test microphone".into()],
                outputs: vec!["Test headphones".into()],
                error: None,
            },
            setup_owns_screen: false,
        }
    }
}

fn harness(fixture: Fixture) -> Harness<'static, Fixture> {
    harness_sized(fixture, egui::vec2(880.0, 680.0))
}

fn harness_sized(fixture: Fixture, size: egui::Vec2) -> Harness<'static, Fixture> {
    let builder = Harness::builder();
    #[cfg(feature = "ui-snapshots")]
    let builder = builder.renderer(egui_kittest::LazyRenderer::new(
        super::ui_renderer::WgpuTestRenderer::new,
    ));
    builder.with_size(size).build_ui_state(
        |ui, fixture: &mut Fixture| {
            egui::Frame::new()
                .fill(ui.visuals().panel_fill)
                .inner_margin(20)
                .show(ui, |ui| {
                    fixture.app.onboarding.effects.session_revision =
                        fixture.status.session_revision;
                    fixture.app.screen_ui(ui, &fixture.status, &fixture.devices);
                    fixture.setup_owns_screen = fixture.app.onboarding.active();
                });
        },
        fixture,
    )
}

#[test]
fn ui_setup_cannot_advance_without_a_voice() {
    for language in [ui_text::Language::English, ui_text::Language::Japanese] {
        let mut fixture = Fixture::new(Some(Step::Voice), EngineState::Stopped);
        fixture.app.settings.language = language;
        let mut h = harness(fixture);
        h.get_by_label(language.text(text::VOICE_TITLE));
        assert!(h
            .get_by_label(language.text(text::NEXT))
            .accesskit_node()
            .is_disabled());
        h.get_by_label(language.text(text::NEXT)).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.step, Some(Step::Voice));
        assert!(!h.state().app.settings.setup_completed);
        assert!(h.state().setup_owns_screen);
    }
}

#[test]
fn ui_support_batch_includes_gtcrn_only_when_built_without_enabling_it() {
    let mut h = harness(Fixture::new(Some(Step::Prepare), EngineState::Stopped));
    let denoiser = h.state().app.settings.denoiser.clone();
    h.get_by_label(text::DOWNLOAD).click();
    h.run_steps(4);
    let mut expected = vec![0, 1];
    if cfg!(feature = "gtcrn") {
        expected.push(model_setup::GTCRN_INDEX);
        assert!(h
            .state()
            .app
            .settings
            .accepted_terms
            .iter()
            .any(|id| id == "gtcrn-mit-502ebfab"));
    }
    assert_eq!(h.state().app.onboarding.effects.download_indices, expected);
    assert_eq!(h.state().app.settings.denoiser, denoiser);
}

#[cfg(feature = "windowsml")]
#[test]
fn ui_legacy_cpu_uses_the_single_windows_ml_cpu_option() {
    let mut fixture = Fixture::new(None, EngineState::Stopped);
    fixture.app.settings.provider = "cpu".into();
    fixture.app.settings.chunk_ms = 230;
    fixture.app.settings.normalize_gui_managed_settings();
    assert_eq!(fixture.app.settings.provider, "windowsml-cpu");
    assert_eq!(fixture.app.settings.chunk_ms, 230);
    assert!(!gui_provider_visible(vc_core::Provider::Cpu));
    assert!(gui_provider_visible(vc_core::Provider::WindowsMlCpu));
    assert_eq!(Provider::WindowsMlCpu.display_name(), "CPU (Windows ML)");
}

#[test]
fn ui_language_terms_audio_flow_does_not_start_output() {
    for language in [text::Language::English, text::Language::Japanese] {
        let mut fixture = Fixture::new(Some(Step::Language), EngineState::Stopped);
        fixture.app.settings.accepted_terms.clear();
        fixture.app.settings.language = language;
        let mut h = harness(fixture);
        h.get_by_label(language.text("Next")).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.step, Some(Step::Terms));
        assert!(h.state().app.onboarding.effects.device_requests.is_empty());
        h.get_by_label(language.text(text::BACK)).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.step, Some(Step::Language));
        assert!(h.state().app.settings.accepted_terms.is_empty());
        assert_eq!(
            h.state().app.settings.tutorial.as_ref().unwrap().step,
            Some(Step::Language)
        );
        h.get_by_label(language.text("Next")).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.step, Some(Step::Terms));
        h.get_by_label(language.text("Agree and continue")).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.step, Some(Step::Audio));
        assert!(terms::accepted(&h.state().app.settings));
        assert_eq!(
            h.state().app.onboarding.effects.device_requests,
            [vc_app::TestOutput::Silent]
        );
        h.get_by_label(language.text("Continue with these devices"))
            .click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.step, Some(Step::Voice));
    }
}

#[test]
fn ui_save_failure_preserves_page_and_does_not_accept_terms() {
    let mut fixture = Fixture::new(Some(Step::Terms), EngineState::Stopped);
    fixture.app.settings.accepted_terms.clear();
    fixture.app.onboarding.effects.save_error = Some("Read-only test settings".into());
    let mut h = harness(fixture);
    h.get_by_label("Agree and continue").click();
    h.run_steps(4);
    assert_eq!(h.state().app.onboarding.step, Some(Step::Terms));
    assert!(h.state().app.settings.accepted_terms.is_empty());
    assert!(h.state().app.onboarding.effects.saved.is_empty());
    assert!(h.state().app.ui_error.is_some());
}

#[test]
fn ui_skip_save_failure_stays_on_audio_page() {
    let mut fixture = Fixture::new(Some(Step::Audio), EngineState::Stopped);
    fixture.app.onboarding.effects.save_error = Some("Test save failure".into());
    let mut h = harness(fixture);
    h.get_by_label(text::SKIP).click();
    h.run_steps(4);
    assert_eq!(h.state().app.onboarding.step, Some(Step::Audio));
    assert!(!h.state().app.settings.setup_skipped);
}

#[test]
fn ui_device_change_silences_monitor_and_uses_new_device() {
    let mut h = harness(Fixture::new(Some(Step::Audio), EngineState::Stopped));
    h.get_by_label("Hear my voice").click();
    h.run_steps(4);
    assert_eq!(
        h.state().app.onboarding.effects.device_requests.last(),
        Some(&vc_app::TestOutput::Monitor)
    );
    h.get_by_label(text::OUTPUT).click();
    h.run_steps(4);
    h.get_by_label("Test headphones").click();
    h.run_steps(4);
    assert_eq!(
        h.state().app.onboarding.effects.device_requests.last(),
        Some(&vc_app::TestOutput::Silent)
    );
    let config = h
        .state()
        .app
        .onboarding
        .effects
        .device_configs
        .last()
        .unwrap();
    assert_eq!(config.output_device.as_deref(), Some("Test headphones"));
    assert_eq!(config.output, vc_app::TestOutput::Silent);
}

#[cfg(feature = "ui-snapshots")]
#[test]
#[ignore = "Renders setup pages to target/onboarding-ui for visual review"]
fn render_tutorial_pages() {
    let destination = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/onboarding-ui");
    fs::create_dir_all(&destination).unwrap();
    for language in [text::Language::English, text::Language::Japanese] {
        for step in [
            Step::Language,
            Step::Terms,
            Step::Audio,
            Step::Voice,
            Step::Prepare,
        ] {
            let mut fixture = Fixture::new(Some(step), EngineState::Stopped);
            fixture.app.settings.language = language;
            let mut h = harness(fixture);
            install_system_japanese_font(&h.ctx);
            h.run_steps(4);
            h.render()
                .unwrap()
                .save(destination.join(format!("{language:?}-{step:?}.png")))
                .unwrap();
        }
    }
}

#[test]
fn ui_prepared_voice_waits_for_continue_on_preparation() {
    for language in [ui_text::Language::English, ui_text::Language::Japanese] {
        let mut fixture = Fixture::new(Some(Step::Voice), EngineState::Stopped);
        fixture.app.settings.language = language;
        // Supply a cached successful validation and an existing file solely for
        // availability checks. This fixture must never load or parse this file.
        let path = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        fixture.app.settings.model = path.clone();
        fixture.app.settings.embedder = path.clone();
        fixture.app.settings.f0_model = path.clone();
        fixture.app.onboarding.support_check = Some((
            support_signature(&fixture.app.settings),
            Arc::new(Mutex::new(Some([Ok(()), Ok(())]))),
        ));
        fixture.app.onboarding.model_check = Some((path, Arc::new(Mutex::new(Some(Ok(()))))));
        let mut h = harness(fixture);
        h.get_by_label(language.text(text::NEXT)).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.step, Some(Step::Prepare));
        h.get_by_label(&format!(
            "{}: {}",
            language.text("Embedder"),
            language.text("Ready")
        ));
        h.get_by_label(&format!(
            "{}: {}",
            language.text("F0 model"),
            language.text("Ready")
        ));
        assert!(h
            .query_by_label(&format!("ContentVec: {}", language.text("Ready")))
            .is_none());
        assert_eq!(h.state().app.onboarding.effects.download_requests, 0);
        h.get_by_label(language.text(text::OPEN_MAIN)).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.step, None);
        assert!(h.state().app.settings.setup_completed);
        assert_eq!(h.state().status.state, EngineState::Stopped);
        assert!(h.state().app.onboarding.effects.device_requests.is_empty());
        assert_eq!(h.state().app.settings.output_gain, 1.0);
        h.get_by_label(language.text(text::SETUP)).click();
        h.run_steps(4);
        assert_eq!(h.state().app.onboarding.step, Some(Step::Audio));
    }
}

#[test]
fn ui_valid_voice_still_requires_shared_files() {
    let mut fixture = Fixture::new(Some(Step::Voice), EngineState::Stopped);
    let path = std::env::current_exe()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    fixture.app.settings.model = path.clone();
    fixture.app.onboarding.model_check = Some((path, Arc::new(Mutex::new(Some(Ok(()))))));
    let mut h = harness(fixture);
    h.get_by_label(text::NEXT).click();
    h.run_steps(4);
    assert_eq!(h.state().app.onboarding.step, Some(Step::Prepare));
    h.get_by_label(text::DOWNLOAD);
}

#[test]
fn ui_preparation_downloads_missing_files_and_waits_after_completion() {
    for language in [text::Language::English, text::Language::Japanese] {
        for missing in 0..2 {
            let mut fixture = Fixture::new(Some(Step::Prepare), EngineState::Stopped);
            fixture.app.settings.language = language;
            fixture.app.settings.model = std::env::current_exe()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            let check = fixture
                .app
                .onboarding
                .support_check
                .as_ref()
                .unwrap()
                .1
                .clone();
            *check.lock().unwrap() = None;
            let mut h = harness(fixture);
            assert!(h.query_by_label(language.text(text::DOWNLOAD)).is_none());
            assert!(h
                .query_by_label(language.text(text::DOWNLOAD_INFO))
                .is_none());
            assert!(h
                .get_by_label(language.text(text::OPEN_MAIN))
                .accesskit_node()
                .is_disabled());
            let mut results = [Ok(()), Ok(())];
            results[missing] = Err("Missing or corrupt file".into());
            *check.lock().unwrap() = Some(results);
            h.run_steps(4);
            assert!(h
                .get_by_label(language.text(text::OPEN_MAIN))
                .accesskit_node()
                .is_disabled());
            h.get_by_label(language.text(text::DOWNLOAD)).click();
            h.run_steps(4);
            let mut expected = vec![missing];
            if cfg!(feature = "gtcrn") {
                expected.push(model_setup::GTCRN_INDEX);
            }
            assert_eq!(h.state().app.onboarding.effects.download_indices, expected);
            *check.lock().unwrap() = Some([Ok(()), Ok(())]);
            h.run_steps(4);
            assert!(h.query_by_label(language.text(text::DOWNLOAD)).is_none());
            assert_eq!(h.state().app.onboarding.step, Some(Step::Prepare));
            assert!(!h
                .get_by_label(language.text(text::OPEN_MAIN))
                .accesskit_node()
                .is_disabled());
            h.state_mut().app.onboarding.effects.save_error = Some("Test save failure".into());
            h.get_by_label(language.text(text::OPEN_MAIN)).click();
            h.run_steps(4);
            assert_eq!(h.state().app.onboarding.step, Some(Step::Prepare));
            h.state_mut().app.onboarding.effects.save_error = None;
            h.get_by_label(language.text(text::OPEN_MAIN)).click();
            h.run_steps(4);
            assert_eq!(h.state().app.onboarding.step, None);
        }
    }
}

#[cfg(feature = "windowsml")]
#[test]
fn ui_runtime_download_prompt_waits_for_inspection() {
    for language in [text::Language::English, text::Language::Japanese] {
        for result in [Ok(true), Ok(false), Err("Test inspection failure".into())] {
            let mut fixture = Fixture::new(Some(Step::Prepare), EngineState::Stopped);
            fixture.app.settings.language = language;
            fixture.app.settings.provider = "windowsml".into();
            fixture.app.onboarding.runtime_check = None;
            let inspection = Arc::new(Mutex::new(None));
            fixture.app.onboarding.runtime_inspection =
                Some(("windowsml".into(), inspection.clone()));
            let mut h = harness(fixture);
            let label = language.text("Prepare processing components");
            assert!(h.query_by_label(label).is_none());
            h.get_by_label(language.text("Checking the runtime…"));
            *inspection.lock().unwrap() = Some(result.clone());
            h.run_steps(4);
            assert_eq!(h.query_by_label(label).is_some(), result == Ok(true));
            assert_eq!(h.state().app.onboarding.step, Some(Step::Prepare));
            if result.is_err() {
                h.get_by_label(language.text(text::RUNTIME_HELP));
                h.get_by_label(language.text(text::RUNTIME_LINK));
                h.get_by_label(language.text(text::RUNTIME_RESTART));
                h.get_by_label(language.text("Runtime details"))
                    .scroll_to_me();
                h.run_steps(4);
                h.get_by_label(language.text("Runtime details")).click();
                h.run_steps(4);
                h.get_by_label("Test inspection failure");
                h.get_by_label(language.text("Retry runtime check"))
                    .scroll_to_me();
                h.run_steps(4);
                h.get_by_label(language.text("Retry runtime check")).click();
                h.run_steps(4);
                assert!(h
                    .query_by_label(language.text(text::RUNTIME_LINK))
                    .is_none());
                assert_eq!(h.state().app.onboarding.step, Some(Step::Prepare));
            } else {
                assert!(h
                    .query_by_label(language.text(text::RUNTIME_LINK))
                    .is_none());
                if result == Ok(true) {
                    assert!(!h.state().app.onboarding.prepare_requested);
                    h.get_by_label(label).scroll_to_me();
                    h.run_steps(8);
                    h.get_by_label(label).click();
                    h.run_steps(4);
                }
                assert!(h.state().app.onboarding.prepare_requested);
                assert!(h.query_by_label(label).is_none());
            }
        }
    }
}

#[test]
fn ui_skip_opens_normal_screen_and_persists_without_claiming_preview() {
    for language in [ui_text::Language::English, ui_text::Language::Japanese] {
        for step in [Step::Audio, Step::Voice, Step::Prepare] {
            let mut fixture = Fixture::new(Some(step), EngineState::Stopped);
            fixture.app.settings.language = language;
            let mut h = harness(fixture);
            let skip = match step {
                Step::Voice => "Choose a voice later",
                _ => text::SKIP,
            };
            h.get_by_label(language.text(skip)).click();
            h.run_steps(4);
            assert!(!h.state().setup_owns_screen);
            assert_eq!(h.state().app.onboarding.step, None);
            assert!(h.state().app.settings.setup_skipped);
            assert!(!h.state().app.settings.setup_completed);
            assert!(h.state().app.dirty_since.is_none());
            assert!(!h.state().app.onboarding.effects.saved.is_empty());
            h.get_by_label(language.text(text::START));
            let saved = toml::to_string(&h.state().app.settings).unwrap();
            let restored: GuiSettings = toml::from_str(&saved).unwrap();
            assert_eq!(Onboarding::new(&restored).step, None);
            h.get_by_label(language.text(text::SETUP)).click();
            h.run_steps(4);
            assert_eq!(h.state().app.onboarding.step, Some(Step::Audio));
        }
    }
}

#[test]
fn ui_busy_engine_disables_start() {
    for state in [EngineState::Starting, EngineState::Stopping] {
        let mut h = harness(Fixture::new(None, state));
        assert!(h.get_by_label(text::START).accesskit_node().is_disabled());
        h.get_by_label(text::START).click();
        h.run_steps(4);
        // An enabled Start would reach model validation and set an error.
        assert!(h.state().app.ui_error.is_none());
        assert!(h.state().app.applied_chunk_ms.is_none());
    }
}

#[test]
fn ui_start_with_missing_models_reports_validation_error() {
    let mut h = harness(Fixture::new(None, EngineState::Stopped));
    h.get_by_label(text::START).click();
    h.run_steps(4);
    assert_eq!(
        h.state().app.ui_error.as_deref(),
        Some("Choose a voice model and prepare the embedder / F0 model in Setup.")
    );
    assert!(h.state().app.applied_chunk_ms.is_none());
}

#[test]
fn ui_setup_button_returns_to_device_test() {
    let fixture = Fixture::new(None, EngineState::Stopped);
    let mut h = harness(fixture);
    h.get_by_label(text::SETUP).click();
    h.run_steps(4);
    h.get_by_label("Check your microphone and headphones");
    assert!(h.state().setup_owns_screen);
}

#[test]
fn ui_failed_conversion_can_retry_then_cancel_without_changing_model() {
    let mut fixture = Fixture::new(Some(Step::Voice), EngineState::Stopped);
    let convert = PthConvert::new(PathBuf::from("ui-test-voice.pth"));
    *convert.state.lock().unwrap() = PthConvertState::Failed {
        error: "Test conversion failure".into(),
    };
    fixture.app.pth_convert = Some(convert);
    let mut h = harness(fixture);
    h.run_steps(4);
    h.get_by_label("Test conversion failure");
    h.get_by_label("Retry").click();
    h.run_steps(4);
    assert!(matches!(
        *h.state()
            .app
            .pth_convert
            .as_ref()
            .unwrap()
            .state
            .lock()
            .unwrap(),
        PthConvertState::Configuring
    ));
    h.get_by_label("Cancel").click();
    h.run_steps(4);
    assert!(h.state().app.pth_convert.is_none());
    assert!(h.state().app.settings.model.is_empty());
    assert!(h.state().app.dirty_since.is_none());
}
