//! Run with each distribution feature set, not just their unified dev build.
use super::*;

#[cfg(all(feature = "windowsml", feature = "ui-snapshots"))]
#[test]
#[ignore = "Render backend labels with injected devices; no inference, downloads or settings writes"]
fn render_backend_pages() {
    let destination = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/backend-ui");
    fs::create_dir_all(&destination).unwrap();
    for lang in [text::Language::English, text::Language::Japanese] {
        for width in [880.0, 520.0] {
            for provider in [
                Provider::WindowsMl,
                Provider::WindowsMlDirectMl,
                Provider::WindowsMlNvTensorRtRtx,
                Provider::WindowsMlOpenVinoCpu,
                Provider::WindowsMlOpenVinoGpu,
                Provider::WindowsMlOpenVinoNpu,
                Provider::WindowsMlOpenVino,
            ] {
                let mut fixture = ready_main_fixture(lang);
                fixture.app.settings.provider = provider.label().into();
                fixture.app.onboarding.effects.catalog_providers = vec![
                    Provider::WindowsMlNvTensorRtRtx,
                    Provider::WindowsMlOpenVino,
                ];
                fixture.app.openvino_devices.status =
                    vc_core::openvino::DeviceStatus::Detected(Provider::OPENVINO_DEVICES.to_vec());
                let mut h = harness_sized(fixture, egui::vec2(width, 680.0));
                install_system_japanese_font(&h.ctx);
                // Match production font sizes and spacing for layout inspection.
                let mut style = (*h.ctx.global_style()).clone();
                style.spacing.item_spacing = egui::vec2(10.0, 10.0);
                style.spacing.button_padding = egui::vec2(12.0, 7.0);
                for text_style in [egui::TextStyle::Body, egui::TextStyle::Button] {
                    style
                        .text_styles
                        .insert(text_style, egui::FontId::proportional(16.0));
                }
                h.ctx.set_global_style(style);
                h.run_steps(4);
                h.get_by_label(lang.text("Backend Details")).scroll_to_me();
                h.run_steps(8);
                h.get_by_label(lang.text("Backend Details")).click();
                h.run_steps(4);
                h.get_by_label(lang.text("Provider")).scroll_to_me();
                h.run_steps(8);
                let stem = format!("{lang:?}-{width}-{}", provider.label());
                h.render()
                    .unwrap()
                    .save(destination.join(format!("{stem}.png")))
                    .unwrap();
                if provider == Provider::WindowsMlNvTensorRtRtx {
                    h.get_by_label(lang.text("Provider")).click();
                    h.run_steps(4);
                    h.render()
                        .unwrap()
                        .save(destination.join(format!("{stem}-choices.png")))
                        .unwrap();
                } else if provider == Provider::WindowsMlOpenVinoGpu {
                    h.get_by_label(lang.text("OpenVINO Device")).click();
                    h.run_steps(4);
                    h.render()
                        .unwrap()
                        .save(destination.join(format!("{stem}-devices.png")))
                        .unwrap();
                }
                assert_eq!(h.state().app.settings.provider, provider.label());
            }
        }
    }
}

fn backend_harness(fixture: Fixture) -> Harness<'static, Fixture> {
    let lang = fixture.app.settings.language;
    let mut h = harness(fixture);
    h.get_by_label(lang.text("Backend Details")).scroll_to_me();
    h.run_steps(8);
    h.get_by_label(lang.text("Backend Details")).click();
    h.run_steps(4);
    h.get_by_label(lang.text("Provider")).scroll_to_me();
    h.run_steps(8);
    h
}

#[test]
fn ui_provider_options_match_build_and_priority_is_always_available() {
    for lang in [text::Language::English, text::Language::Japanese] {
        let mut fixture = ready_main_fixture(lang);
        fixture.app.settings.normalize_gui_managed_settings();
        let mut h = backend_harness(fixture);
        h.get_by_label(lang.text("GPU Priority"));
        h.get_by_label(lang.text("Provider")).click();
        h.run_steps(4);
        for (label, expected) in [
            ("Windows ML (automatic)", cfg!(feature = "windowsml")),
            ("CPU (Windows ML)", cfg!(feature = "windowsml")),
            ("DirectML (GPU)", cfg!(feature = "windowsml")),
            ("TensorRT (NVIDIA GPU)", cfg!(feature = "tensorrt")),
        ] {
            assert_eq!(
                h.query_by_label(lang.text(label)).is_some(),
                expected,
                "{label}"
            );
        }
        for raw in [
            "cpu",
            "cuda",
            "tensorrt",
            "windowsml",
            "windowsml-cpu",
            "windowsml-directml",
            "windowsml-openvino",
        ] {
            assert!(h.query_by_label(raw).is_none(), "raw provider: {raw}");
        }
        let selected = if cfg!(feature = "windowsml") {
            "CPU (Windows ML)"
        } else {
            "TensorRT (NVIDIA GPU)"
        };
        h.get_by_label(lang.text(selected)).click();
        h.run_steps(4);
        assert_eq!(
            h.state().app.settings.provider,
            if cfg!(feature = "windowsml") {
                "windowsml-cpu"
            } else {
                "tensorrt"
            }
        );
        assert_eq!(
            h.query_by_label(lang.text("Detecting CUDA devices..."))
                .is_some(),
            !cfg!(feature = "windowsml")
        );
    }
}

#[cfg(feature = "windowsml")]
#[test]
fn ui_catalog_providers_can_be_selected_without_exposing_cuda_device_controls() {
    for lang in [text::Language::English, text::Language::Japanese] {
        let mut fixture = ready_main_fixture(lang);
        fixture.app.onboarding.effects.catalog_providers = vec![
            Provider::WindowsMlNvTensorRtRtx,
            Provider::WindowsMlOpenVino,
            Provider::WindowsMlOpenVinoCpu,
            Provider::WindowsMlOpenVinoGpu,
            Provider::WindowsMlOpenVinoNpu,
        ];
        fixture.app.openvino_devices.status =
            vc_core::openvino::DeviceStatus::Detected(Provider::OPENVINO_DEVICES.to_vec());
        let mut h = backend_harness(fixture);
        for label in [
            "windowsml-nvtrtx",
            "windowsml-openvino-cpu",
            "windowsml-openvino-gpu",
            "windowsml-openvino-npu",
            "windowsml",
            "windowsml-directml",
        ] {
            h.get_by_label(lang.text("Provider")).click();
            h.run_steps(4);
            assert!(h.query_by_label("windowsml-qnn").is_none());
            for device_name in [
                "windowsml-openvino-cpu",
                "windowsml-openvino-gpu",
                "windowsml-openvino-npu",
            ] {
                assert!(h.query_by_label(device_name).is_none());
            }
            let provider = Provider::from_name(label).unwrap();
            h.get_by_label(lang.text(provider.backend().display_name()))
                .click();
            h.run_steps(4);
            if let Some(device) = provider.openvino_device_label() {
                h.get_by_label(lang.text("OpenVINO Device")).click();
                h.run_steps(4);
                assert!(h.query_by_label(lang.text("Default")).is_none());
                h.get_by_label(&format!("{} ({})", device, lang.text("Available")))
                    .click();
                h.run_steps(4);
            } else {
                assert!(h.query_by_label(lang.text("OpenVINO Device")).is_none());
            }
            if let Some(hint) = provider.display_hint() {
                h.get_by_label(lang.text(hint));
            }
            assert_eq!(h.state().app.settings.provider, label);
            let saved = toml::to_string(&h.state().app.settings).unwrap();
            let mut restored: GuiSettings = toml::from_str(&saved).unwrap();
            restored.normalize_gui_managed_settings();
            assert_eq!(restored.provider, label);
            assert!(h.query_by_label(lang.text("GPU Device")).is_none());
            assert!(h
                .query_by_label(lang.text("Detecting CUDA devices..."))
                .is_none());
            h.get_by_label(lang.text("GPU Priority"));
        }
    }
}

#[cfg(feature = "windowsml")]
#[test]
fn ui_restores_openvino_hardware_without_changing_saved_provider() {
    for lang in [text::Language::English, text::Language::Japanese] {
        for provider in std::iter::once(Provider::WindowsMlOpenVino)
            .chain(Provider::OPENVINO_DEVICES.iter().copied())
        {
            let mut fixture = ready_main_fixture(lang);
            fixture.app.settings.provider = provider.label().into();
            fixture.app.onboarding.effects.catalog_providers = vec![Provider::WindowsMlOpenVino];
            fixture.app.openvino_devices.status =
                vc_core::openvino::DeviceStatus::Detected(Provider::OPENVINO_DEVICES.to_vec());
            let mut h = backend_harness(fixture);
            h.get_by_label(lang.text("OpenVINO Device"));
            assert_eq!(h.state().app.settings.provider, provider.label());
            if provider == Provider::WindowsMlOpenVino {
                h.get_by_value(lang.text("Unspecified (legacy setting)"));
            }
            // Re-selecting the same backend must not reset its device choice.
            h.get_by_label(lang.text("Provider")).click();
            h.run_steps(4);
            h.get_by_label("OpenVINO").click();
            h.run_steps(4);
            assert_eq!(h.state().app.settings.provider, provider.label());
        }
    }
}

#[cfg(feature = "windowsml")]
#[test]
fn ui_openvino_disables_only_confirmed_missing_devices() {
    for lang in [text::Language::English, text::Language::Japanese] {
        for detected in [false, true] {
            let mut fixture = ready_main_fixture(lang);
            fixture.app.settings.provider = "windowsml-openvino-npu".into();
            fixture.app.onboarding.effects.catalog_providers = vec![Provider::WindowsMlOpenVino];
            fixture.app.openvino_devices.status = if detected {
                vc_core::openvino::DeviceStatus::Detected(vec![Provider::WindowsMlOpenVinoCpu])
            } else {
                vc_core::openvino::DeviceStatus::Unknown("Test EP not prepared".into())
            };
            let mut h = backend_harness(fixture);
            // Discovery must not silently replace a saved, now-missing device.
            assert_eq!(h.state().app.settings.provider, "windowsml-openvino-npu");
            h.get_by_label(lang.text("OpenVINO Device")).click();
            h.run_steps(4);
            assert!(h.query_by_label(lang.text("Default")).is_none());
            let npu = format!(
                "NPU ({})",
                lang.text(if detected {
                    "Unavailable"
                } else {
                    "Unverified"
                })
            );
            assert_eq!(
                h.get_by_label(&npu).accesskit_node().is_disabled(),
                detected
            );
            let cpu = format!(
                "CPU ({})",
                lang.text(if detected { "Available" } else { "Unverified" })
            );
            assert!(!h.get_by_label(&cpu).accesskit_node().is_disabled());
            h.get_by_label(&cpu).click();
            h.run_steps(4);
            assert_eq!(h.state().app.settings.provider, "windowsml-openvino-cpu");
        }
    }
}

#[cfg(feature = "windowsml")]
#[test]
fn ui_openvino_first_download_then_devices_without_refresh_button() {
    use vc_core::openvino::DeviceStatus;
    for lang in [text::Language::English, text::Language::Japanese] {
        let mut fixture = ready_main_fixture(lang);
        fixture.app.settings.provider = "windowsml-openvino-gpu".into();
        fixture.app.onboarding.effects.catalog_providers = vec![Provider::WindowsMlOpenVino];
        fixture.app.openvino_devices.status = DeviceStatus::DownloadRequired;
        let mut h = backend_harness(fixture);
        h.get_by_label(lang.text("Download OpenVINO to check available devices."));
        assert!(h.query_by_label(lang.text("OpenVINO Device")).is_none());
        assert!(matches!(
            h.state().app.openvino_devices.status,
            DeviceStatus::DownloadRequired
        ));
        h.get_by_label(lang.text("Download and check")).click();
        h.run_steps(4);
        h.get_by_label(lang.text("Downloading and preparing OpenVINO..."));
        assert!(h.query_by_label(lang.text("Download and check")).is_none());
        assert!(h.query_by_label(lang.text("OpenVINO Device")).is_none());

        // Inject only the asynchronous result. The click and queued retry use
        // production state transitions, without acquiring an EP in a UI test.
        h.state_mut().app.openvino_devices.status =
            DeviceStatus::Unknown("Test download failed".into());
        h.run_steps(4);
        h.get_by_label(lang.text("Retry device check")).click();
        h.run_steps(4);
        assert!(matches!(
            h.state().app.openvino_devices.status,
            DeviceStatus::Downloading
        ));
        h.state_mut().app.openvino_devices.status = DeviceStatus::Detected(vec![
            Provider::WindowsMlOpenVinoCpu,
            Provider::WindowsMlOpenVinoGpu,
        ]);
        h.run_steps(4);
        h.get_by_label(lang.text("OpenVINO Device"));
        assert!(h.query_by_label(lang.text("Retry device check")).is_none());
        assert!(h.query_by_label(lang.text("Download and check")).is_none());
        for removed in ["Refresh OpenVINO devices", "OpenVINOデバイスを再確認"] {
            assert!(h.query_by_label(removed).is_none());
        }
        assert_eq!(h.state().app.settings.provider, "windowsml-openvino-gpu");
    }
}

#[cfg(feature = "windowsml")]
#[test]
fn ui_new_openvino_selection_is_explicit_cpu() {
    let mut fixture = ready_main_fixture(text::Language::English);
    fixture.app.settings.provider = "windowsml".into();
    fixture.app.onboarding.effects.catalog_providers = vec![Provider::WindowsMlOpenVino];
    fixture.app.openvino_devices.status =
        vc_core::openvino::DeviceStatus::Detected(Provider::OPENVINO_DEVICES.to_vec());
    let mut h = backend_harness(fixture);
    h.get_by_label("Provider").click();
    h.run_steps(4);
    h.get_by_label("OpenVINO").click();
    h.run_steps(4);
    assert_eq!(h.state().app.settings.provider, "windowsml-openvino-cpu");
    h.get_by_label("OpenVINO Device").click();
    h.run_steps(4);
    assert!(h.query_by_label("Default").is_none());
    assert!(h.query_by_label("Unspecified (legacy setting)").is_none());
}

#[cfg(feature = "tensorrt")]
#[test]
fn ui_tensorrt_gpu_discovery_pending_success_missing_and_failure() {
    for lang in [text::Language::English, text::Language::Japanese] {
        let mut fixture = ready_main_fixture(lang);
        fixture.app.settings.provider = "tensorrt".into();
        fixture.app.settings.gpu_device_id = 7;
        let mut h = backend_harness(fixture);
        h.get_by_label(lang.text("Detecting CUDA devices..."));
        h.state().app.gpu_devices.lock().unwrap().devices = Some(vec![
            GpuDevice {
                id: 0,
                display_name: "Test GPU A".into(),
            },
            GpuDevice {
                id: 1,
                display_name: "Test GPU B".into(),
            },
        ]);
        h.run_steps(4);
        h.get_by_value(&format!("{} 7", lang.text("Unavailable: device")));
        assert_eq!(h.state().app.settings.gpu_device_id, 7);
        h.get_by_label(lang.text("GPU Device")).click();
        h.run_steps(4);
        h.get_by_label("1: Test GPU B").click();
        h.run_steps(4);
        assert_eq!(h.state().app.settings.gpu_device_id, 1);
        assert!(h.state().app.dirty_since.is_some());
        h.state().app.gpu_devices.lock().unwrap().devices = Some(vec![]);
        h.run_steps(4);
        h.get_by_value(&format!("{} 1", lang.text("Unavailable: device")));
        *h.state().app.gpu_devices.lock().unwrap() = GpuDeviceDiscovery {
            devices: None,
            error: Some("Test enumeration error".into()),
        };
        h.run_steps(4);
        assert!(h.query_by_label(lang.text("GPU Device")).is_none());
        h.get_by_label(&format!(
            "{}: Test enumeration error",
            lang.text("GPU enumeration failed")
        ));
        assert_eq!(h.state().app.settings.gpu_device_id, 1);
    }
}

#[test]
fn ui_single_backend_defaults_migrate_other_package_and_show_matching_terms() {
    if cfg!(all(feature = "windowsml", feature = "tensorrt")) {
        return;
    }
    let expected = if cfg!(feature = "windowsml") {
        "windowsml"
    } else {
        "tensorrt"
    };
    assert_eq!(GuiSettings::new_user().provider, expected);
    for lang in [text::Language::English, text::Language::Japanese] {
        for old_provider in if cfg!(feature = "windowsml") {
            ["tensorrt", "cuda"]
        } else {
            ["windowsml", "windowsml-openvino"]
        } {
            let mut fixture = ready_main_fixture(lang);
            fixture.app.settings.provider = old_provider.into();
            fixture.app.settings.normalize_gui_managed_settings();
            assert_eq!(fixture.app.settings.provider, expected);
            fixture.app.settings.accepted_terms.clear();
            fixture.app.onboarding.step = Some(Step::Terms);
            let mut h = harness(fixture);
            assert_eq!(
                h.query_by_label("Windows App SDK").is_some(),
                cfg!(feature = "windowsml")
            );
            assert_eq!(
                h.query_by_label("NVIDIA TensorRT SDK license").is_some(),
                cfg!(feature = "tensorrt")
            );
            assert_eq!(
                h.query_by_label("NVIDIA CUDA license").is_some(),
                cfg!(feature = "tensorrt")
            );
            h.get_by_label(lang.text("Agree and continue"))
                .scroll_to_me();
            h.run_steps(8);
            h.get_by_label(lang.text("Agree and continue")).click();
            h.run_steps(4);
            let ids = &h.state().app.settings.accepted_terms;
            assert_eq!(
                ids.iter().any(|id| id.starts_with("windowsml:")),
                cfg!(feature = "windowsml")
            );
            assert_eq!(
                ids.iter().any(|id| id.starts_with("nvidia:")),
                cfg!(feature = "tensorrt")
            );
        }
    }
}

#[cfg(all(feature = "tensorrt", not(feature = "windowsml")))]
#[test]
fn ui_tensorrt_setup_never_requires_windows_ml_preparation() {
    for lang in [text::Language::English, text::Language::Japanese] {
        let mut fixture = ready_main_fixture(lang);
        fixture.app.onboarding.step = Some(Step::Prepare);
        fixture.app.onboarding.runtime_check = None;
        let h = harness(fixture);
        for label in [
            "Checking the runtime…",
            "Prepare processing components",
            text::RUNTIME_LINK,
            text::RUNTIME_RESTART,
        ] {
            assert!(h.query_by_label(lang.text(label)).is_none());
        }
        assert!(!h
            .get_by_label(lang.text(text::OPEN_MAIN))
            .accesskit_node()
            .is_disabled());
        assert!(h.state().app.onboarding.runtime_inspection.is_none());
    }
}
