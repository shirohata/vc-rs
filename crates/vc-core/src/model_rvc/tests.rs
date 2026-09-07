use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::Provider;

use super::onnx_meta::RvcIoNames;
use super::pitch::{
    align_pitchf_to_features, center_crop_pitchf_to_features, pitchf_tail_for_output,
};
use super::shape::{
    aligned_rvc_input_len, extra_convert_samples_from_ms, keep_tail_in_place,
    onnx_silence_front_feature_frames, output_len_from_convert_size, rmvpe_model_input_samples_16k,
    rmvpe_model_input_samples_for_context_16k, tensor_rt_model_input_samples_16k,
    EMBEDDER_SAMPLE_RATE, RVC_SAMPLE_RATE,
};
use super::stream::{RvcStreamState, VOLUME_DECAY};
use super::tensorrt::{
    format_usize_shape, i64_shape_to_usize, tensor_rt_benchmark_profile, tensor_rt_cache_key,
    tensor_rt_cache_root_from_override, tensor_rt_model_cache_key, tensor_rt_model_file_hash,
    tensor_rt_sanitize_cache_component, validate_tensorrt_input_shape, ModelRole, TensorRtRunMode,
    TensorRtSessionProfile,
};

fn tensor_rt_temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("vc-rs-{name}-{}-{nanos}", std::process::id()))
}

#[test]
fn parses_cuda_graph_env_as_opt_in() {
    assert_eq!(
        TensorRtRunMode::parse_cuda_env(None),
        TensorRtRunMode::DeviceIo
    );
    assert_eq!(
        TensorRtRunMode::parse_cuda_env(Some("")),
        TensorRtRunMode::DeviceIo
    );
    assert_eq!(
        TensorRtRunMode::parse_cuda_env(Some("0")),
        TensorRtRunMode::DeviceIo
    );
    assert_eq!(
        TensorRtRunMode::parse_cuda_env(Some("false")),
        TensorRtRunMode::DeviceIo
    );
    assert_eq!(
        TensorRtRunMode::parse_cuda_env(Some("1")),
        TensorRtRunMode::CudaGraph
    );
    assert_eq!(
        TensorRtRunMode::parse_cuda_env(Some("true")),
        TensorRtRunMode::CudaGraph
    );
    assert_eq!(
        TensorRtRunMode::parse_cuda_env(Some("on")),
        TensorRtRunMode::CudaGraph
    );
}

#[test]
fn tensorrt_run_mode_controls_graph_device_io() {
    assert!(TensorRtRunMode::CudaGraph.cuda_graph());
    assert!(TensorRtRunMode::CudaGraph.device_io());
    assert!(!TensorRtRunMode::DeviceIo.cuda_graph());
    assert!(TensorRtRunMode::DeviceIo.device_io());
    assert!(!TensorRtRunMode::PinnedCpu.cuda_graph());
    assert!(!TensorRtRunMode::PinnedCpu.device_io());
    assert!(Provider::TensorRt.is_tensorrt());
    assert!(Provider::Cuda.is_cuda());
    assert!(Provider::WindowsMl.is_windows_ml());
    assert!(Provider::WindowsMlDirectMl.is_windows_ml_directml());
    assert!(Provider::WindowsMlNvTensorRtRtx.is_windows_ml());
    assert!(Provider::WindowsMlOpenVino.is_windows_ml());
    assert!(Provider::WindowsMlQnn.is_windows_ml());
    assert!(Provider::WindowsMlMiGraphX.is_windows_ml());
    assert!(Provider::WindowsMlVitisAi.is_windows_ml());
    assert!(!Provider::WindowsMl.is_cuda());
    assert!(!Provider::WindowsMl.is_tensorrt());
    assert!(!Provider::WindowsMlNvTensorRtRtx.is_tensorrt());
    assert!(!Provider::Cpu.is_tensorrt());
}

#[test]
fn tensorrt_profiles_match_validated_shapes() {
    let contentvec = tensor_rt_benchmark_profile(ModelRole::ContentVec)
        .unwrap()
        .with_model_cache_key("content_vec_500_0123456789abcdef");
    let rmvpe = tensor_rt_benchmark_profile(ModelRole::Rmvpe).unwrap();
    let rvc = tensor_rt_benchmark_profile(ModelRole::Rvc).unwrap();

    assert_eq!(contentvec.profile_shapes, "audio:1x24000");
    assert_eq!(rmvpe.profile_shapes, "waveform:1x24000");
    assert_eq!(rvc.profile_shapes, "feats:1x75x768,pitch:1x75,pitchf:1x75");
    assert_eq!(
        contentvec
            .cache_dir_from_root(Path::new("cache-root"))
            .unwrap(),
        Path::new("cache-root")
            .join("device-0")
            .join("contentvec")
            .join("content_vec_500_0123456789abcdef")
            .join("audio_1x24000")
    );
}

#[test]
fn derives_tensorrt_contentvec_profile_from_default_realtime_chunking() {
    assert_eq!(
        tensor_rt_model_input_samples_16k(960, 48_000, 107, 48_000, 48_000),
        18_240
    );
    assert_eq!(rmvpe_model_input_samples_16k(960, 48_000), 4_960);
    let contentvec = TensorRtSessionProfile::single_input(ModelRole::ContentVec, "audio", 18_240);
    let rmvpe = TensorRtSessionProfile::single_input(ModelRole::Rmvpe, "waveform", 4_960);
    let rvc = TensorRtSessionProfile::rvc(114, 768, &RvcIoNames::canonical(), None);

    assert_eq!(contentvec.profile_shapes, "audio:1x18240");
    assert_eq!(rmvpe.profile_shapes, "waveform:1x4960");
    assert_eq!(
        rvc.profile_shapes,
        "feats:1x114x768,pitch:1x114,pitchf:1x114"
    );
    assert_eq!(
        tensor_rt_cache_key("feats:1x114x768,pitch:1x114,pitchf:1x114"),
        "feats_1x114x768_pitch_1x114_pitchf_1x114"
    );
}

#[test]
fn rmvpe_input_uses_upstream_rvc_bucket_boundaries() {
    assert_eq!(rmvpe_model_input_samples_16k(12_960, 48_000), 4_960);
    assert_eq!(rmvpe_model_input_samples_16k(13_440, 48_000), 10_080);
    assert_eq!(rmvpe_model_input_samples_16k(28_320, 48_000), 10_080);
    assert_eq!(rmvpe_model_input_samples_16k(28_800, 48_000), 15_200);
    assert_eq!(rmvpe_model_input_samples_16k(43_680, 48_000), 15_200);
    assert_eq!(rmvpe_model_input_samples_16k(44_160, 48_000), 20_320);
}

#[test]
fn rmvpe_input_is_capped_to_available_context_without_padding() {
    assert_eq!(
        rmvpe_model_input_samples_for_context_16k(13_440, 48_000, 4_480),
        4_480
    );
    assert_eq!(
        rmvpe_model_input_samples_for_context_16k(13_440, 48_000, 12_000),
        10_080
    );
}

#[test]
fn contentvec_fixed_profile_allows_non_default_input_name() {
    let contentvec =
        TensorRtSessionProfile::single_input(ModelRole::ContentVec, "input_values", 18_240)
            .with_model_cache_key("content_vec_500_0123456789abcdef");

    assert_eq!(contentvec.profile_shapes, "input_values:1x18240");
    assert_eq!(
        contentvec.fixed_input_dims("input_values").unwrap(),
        &[1, 18_240]
    );
    assert!(contentvec.fixed_input_dims("audio").is_err());
    assert_eq!(
        contentvec
            .cache_dir_from_root(Path::new("cache-root"))
            .unwrap(),
        Path::new("cache-root")
            .join("device-0")
            .join("contentvec")
            .join("content_vec_500_0123456789abcdef")
            .join("input_values_1x18240")
    );
}

#[test]
fn tensor_rt_cache_root_override_wins() {
    assert_eq!(
        tensor_rt_cache_root_from_override(Some(OsStr::new("override-cache"))).unwrap(),
        PathBuf::from("override-cache")
    );
}

#[test]
fn tensor_rt_model_cache_key_hashes_file_contents() {
    let dir = tensor_rt_temp_dir("model-cache-key");
    fs::create_dir_all(&dir).unwrap();
    let model_a = dir.join("voice opt.onnx");
    let model_a_copy = dir.join("voice copy.onnx");
    let model_b = dir.join("voice changed.onnx");
    fs::write(&model_a, b"same model bytes").unwrap();
    fs::write(&model_a_copy, b"same model bytes").unwrap();
    fs::write(&model_b, b"different model bytes").unwrap();

    let hash_a = tensor_rt_model_file_hash(&model_a).unwrap();
    let hash_a_copy = tensor_rt_model_file_hash(&model_a_copy).unwrap();
    let hash_b = tensor_rt_model_file_hash(&model_b).unwrap();
    assert_eq!(hash_a, hash_a_copy);
    assert_ne!(hash_a, hash_b);

    let key = tensor_rt_model_cache_key(&model_a).unwrap();
    assert!(key.starts_with("voice_opt_"));
    assert!(key.ends_with(&format!("{hash_a:016x}")));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tensor_rt_sanitizes_model_cache_components() {
    assert_eq!(
        tensor_rt_sanitize_cache_component("voice opt+v1"),
        "voice_opt_v1"
    );
    assert_eq!(
        tensor_rt_sanitize_cache_component("abc_DEF-01"),
        "abc_DEF_01"
    );
    assert_eq!(tensor_rt_sanitize_cache_component(""), "model");
}

#[test]
fn validates_tensorrt_profile_input_shapes() {
    let contentvec = TensorRtSessionProfile::single_input(ModelRole::ContentVec, "audio", 24_000);
    let rmvpe = TensorRtSessionProfile::single_input(ModelRole::Rmvpe, "waveform", 24_000);
    let rvc = TensorRtSessionProfile::rvc(75, 768, &RvcIoNames::canonical(), None);

    validate_tensorrt_input_shape(Provider::TensorRt, Some(&contentvec), "audio", &[1, 24_000])
        .unwrap();
    validate_tensorrt_input_shape(Provider::TensorRt, Some(&rmvpe), "waveform", &[1, 24_000])
        .unwrap();
    validate_tensorrt_input_shape(Provider::TensorRt, Some(&rvc), "feats", &[1, 75, 768]).unwrap();
    validate_tensorrt_input_shape(Provider::TensorRt, Some(&rvc), "pitch", &[1, 75]).unwrap();
    validate_tensorrt_input_shape(Provider::Cpu, Some(&rvc), "pitch", &[1, 74]).unwrap();

    let err = validate_tensorrt_input_shape(Provider::TensorRt, Some(&rvc), "pitch", &[1, 74])
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("requires input 'pitch' shape 1x75"));
    let err =
        validate_tensorrt_input_shape(Provider::Cuda, Some(&rvc), "pitch", &[1, 74]).unwrap_err();
    assert!(err
        .to_string()
        .contains("requires input 'pitch' shape 1x75"));
    assert_eq!(format_usize_shape(&[1, 75, 768]), "1x75x768");
}

#[test]
fn looks_up_tensorrt_fixed_input_dims_explicitly() {
    let rvc = TensorRtSessionProfile::rvc(113, 768, &RvcIoNames::canonical(), None);

    assert_eq!(rvc.fixed_input_dims("feats").unwrap(), &[1, 113, 768]);
    assert_eq!(rvc.fixed_input_dims("pitchf").unwrap(), &[1, 113]);
    let err = rvc.fixed_input_dims("threshold").unwrap_err();

    assert!(err
        .to_string()
        .contains("does not include input 'threshold'"));
}

#[test]
fn rejects_negative_runtime_output_shape_dims() {
    let err = i64_shape_to_usize(&[1, -1, 768], "contentvec output").unwrap_err();

    assert!(err.to_string().contains("negative or too-large dim -1"));
}

#[test]
fn aligns_pitchf_by_taking_tail_frames() {
    assert_eq!(
        align_pitchf_to_features(&[1.0, 2.0, 3.0, 4.0], 2),
        vec![3.0, 4.0]
    );
}

#[test]
fn keeps_only_requested_output_tail() {
    let mut audio = vec![1, 2, 3, 4, 5];
    keep_tail_in_place(&mut audio, 3);
    assert_eq!(audio, vec![3, 4, 5]);
}

// The RMS-mix reference now reads the 16 kHz rolling buffer (the signal
// ContentVec/F0 see), not the device-rate `audio_buffer`. These unit tests keep
// the matched-rate (16 kHz in == 16 kHz out) tail/pad math by populating
// `audio_16k_buffer` and passing `EMBEDDER_SAMPLE_RATE` — the assertion values
// are unchanged; only the buffer the reference is drawn from moved.
#[test]
fn output_reference_audio_uses_tail_matching_trimmed_output() {
    let mut state = RvcStreamState::new(48_000, None, None);
    state.audio_16k_buffer = (0..8).map(|value| value as f32).collect();
    let mut scratch = Vec::new();

    let reference = state
        .output_reference_audio(EMBEDDER_SAMPLE_RATE, EMBEDDER_SAMPLE_RATE, 5, &mut scratch)
        .unwrap();

    assert_eq!(reference, &[3.0, 4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn output_reference_audio_left_pads_when_history_is_short() {
    let mut state = RvcStreamState::new(48_000, None, None);
    state.audio_16k_buffer = vec![1.0, 2.0];
    let mut scratch = Vec::new();

    let reference = state
        .output_reference_audio(EMBEDDER_SAMPLE_RATE, EMBEDDER_SAMPLE_RATE, 4, &mut scratch)
        .unwrap();

    assert_eq!(reference, &[0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn aligns_pitchf_by_left_padding_short_inputs() {
    assert_eq!(
        align_pitchf_to_features(&[3.0, 4.0], 4),
        vec![0.0, 0.0, 3.0, 4.0]
    );
}

#[test]
fn aligns_realtime_rvc_input_to_16k_hop_samples() {
    assert_eq!(aligned_rvc_input_len(4800, 48_000, 5632), 10560);
}

#[test]
fn derives_output_len_like_reference_pipeline() {
    assert_eq!(
        output_len_from_convert_size(3520, 48_000, 4096, 48_000),
        6465
    );
}

#[test]
fn stream_state_aligns_convert_size_to_16k_hop_samples() {
    let mut state = RvcStreamState::new(48_000, None, None);
    let input = vec![0.0; 24_000];
    let out = state
        .generate_input(&input, 48_000, 1_536, 1_536, 4_096)
        .unwrap();
    assert_eq!(out.convert_size, 29_760);
}

#[test]
fn stream_state_derives_out_size_from_extra_convert_size() {
    let mut state = RvcStreamState::new(48_000, None, None);
    let input = vec![0.0; 24_000];
    let out = state
        .generate_input(&input, 48_000, 1_536, 1_536, 4_096)
        .unwrap();
    assert_eq!(out.out_size, 25_665);
}

#[test]
fn stream_state_zero_pads_initial_buffer() {
    let mut state = RvcStreamState::new(48_000, None, None);
    let out = state
        .generate_input(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 48_000, 0, 0, 4_096)
        .unwrap();
    assert_eq!(state.audio_buffer.len(), out.convert_size);
    assert!(state.audio_buffer[..state.audio_buffer.len() - 6]
        .iter()
        .all(|x| *x == 0.0));
    assert_eq!(
        &state.audio_buffer[state.audio_buffer.len() - 6..],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn stream_state_keeps_16k_history_for_embedder() {
    let mut state = RvcStreamState::new(48_000, None, None);
    let input = vec![0.25; 4_800];

    state.generate_input(&input, 48_000, 0, 0, 0).unwrap();

    assert_eq!(state.audio_buffer.len(), 4_800);
    assert_eq!(state.audio_16k_buffer.len(), 1_600);
    assert!(state
        .audio_16k_buffer
        .iter()
        .any(|sample| sample.abs() > 1e-4));
}

#[test]
fn stream_state_volume_excludes_crossfade_not_sola_search() {
    let mut state = RvcStreamState::new(48_000, None, None);
    let mut input = vec![1.0; 80];
    input.extend(std::iter::repeat_n(0.0, 80));

    let out = state.generate_input(&input, 16_000, 480, 240, 0).unwrap();

    assert!((out.volume - 0.5f32.sqrt()).abs() < 1e-6);
}

#[test]
fn stream_state_volume_keeps_decay_from_previous_chunk() {
    let mut state = RvcStreamState::new(48_000, None, None);
    let loud = vec![1.0; 160];
    let quiet = vec![0.0; 160];

    let first = state.generate_input(&loud, 16_000, 0, 0, 0).unwrap();
    let second = state.generate_input(&quiet, 16_000, 0, 0, 0).unwrap();

    assert!((first.volume - 1.0).abs() < 1e-6);
    assert!((second.volume - VOLUME_DECAY).abs() < 1e-6);
}

#[test]
fn align_pitchf_to_features_uses_tail_for_feature_length() {
    let pitchf = vec![0.0, 10.0, 20.0, 30.0, 40.0];
    assert_eq!(align_pitchf_to_features(&pitchf, 3), vec![20.0, 30.0, 40.0]);
}

#[test]
fn center_crops_pitchf_to_feature_grid() {
    let pitchf: Vec<f32> = (0..183).map(|frame| frame as f32).collect();

    let aligned = center_crop_pitchf_to_features(&pitchf, 180);

    assert_eq!(aligned.len(), 180);
    assert_eq!(aligned[0], 1.0);
    assert_eq!(aligned[179], 180.0);
}

#[test]
fn pitchf_tail_for_output_matches_10ms_output_frames() {
    let pitchf = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    assert_eq!(
        pitchf_tail_for_output(&pitchf, 1_440, RVC_SAMPLE_RATE),
        vec![30.0, 40.0, 50.0]
    );
}

#[test]
fn stream_state_pitch_update_places_rmvpe_tail_window_at_absolute_frame() {
    let mut state = RvcStreamState::new(48_000, None, None);
    state.pitchf_buffer = (0..34).map(|frame| frame as f32).collect();

    state.update_pitchf_from_rmvpe_window(&[100.0, 101.0, 102.0, 103.0], 480);

    assert_eq!(state.pitchf_buffer[2], 2.0);
    assert_eq!(&state.pitchf_buffer[3..7], &[100.0, 101.0, 102.0, 103.0]);
}

#[test]
fn stream_state_pitch_update_drops_center_padded_tail_frame() {
    let mut state = RvcStreamState::new(48_000, None, None);
    state.pitchf_buffer = vec![0.0, 1.0, 2.0];

    state.update_pitchf_from_rmvpe_window(&[10.0, 20.0, 30.0, 40.0], 0);

    assert_eq!(state.pitchf_buffer, vec![10.0, 20.0, 30.0]);
}

#[test]
fn derives_vcclient_onnx_silence_front_feature_offset() {
    assert_eq!(onnx_silence_front_feature_frames(4096, 48_000), 6);
}

#[test]
fn extra_convert_samples_scale_with_model_sample_rate() {
    // The same convert-context duration is fewer samples at a lower model rate.
    assert_eq!(extra_convert_samples_from_ms(100, 48_000), 4_800);
    assert_eq!(extra_convert_samples_from_ms(100, 40_000), 4_000);
    assert_eq!(extra_convert_samples_from_ms(100, 32_000), 3_200);
}

#[test]
fn out_size_tracks_model_sample_rate() {
    // Run the same device-rate input through models of different native rates.
    // The fix is rate-generic (reads `samplingRate`), not special-cased per rate,
    // so 32 kHz and 40 kHz are both handled like 48 kHz.
    let chunk = vec![0.1f32; 4_800]; // 100 ms at the 48 kHz device rate
    let device_rate = 48_000;

    let run = |rvc_rate: u32| {
        RvcStreamState::new(rvc_rate, None, None)
            .generate_input(
                &chunk,
                device_rate,
                0,
                0,
                extra_convert_samples_from_ms(100, rvc_rate),
            )
            .unwrap()
    };

    let base = run(48_000);
    for rvc_rate in [32_000u32, 40_000] {
        let out = run(rvc_rate);
        // The ContentVec/F0 window lives in the 16 kHz domain (independent of the
        // model output rate), so `convert_size` matches the 48 kHz baseline.
        assert_eq!(
            base.convert_size, out.convert_size,
            "device-rate convert window must not depend on model rate ({rvc_rate} Hz)"
        );
        // `out_size` is in the model's output-rate domain, so it scales by
        // rvc_rate/48000 relative to the 48 kHz baseline.
        assert!(
            out.out_size < base.out_size,
            "{rvc_rate} Hz output window must be shorter than 48 kHz: {} vs {}",
            out.out_size,
            base.out_size
        );
        let expected = base.out_size * rvc_rate as usize / 48_000;
        assert!(
            (out.out_size as i64 - expected as i64).abs() <= 1,
            "{rvc_rate} Hz out_size {} not ~{rvc_rate}/48000 of 48 kHz {} (expected {})",
            out.out_size,
            base.out_size,
            expected
        );
    }
}
#[test]
fn gpu_priority_defaults_to_high() {
    assert_eq!(super::GpuPriority::default(), super::GpuPriority::High);
}

#[test]
fn tensor_rt_cache_is_separated_by_gpu_device_id() {
    let profile = TensorRtSessionProfile::single_input(ModelRole::ContentVec, "audio", 24_000)
        .with_model_cache_key("model")
        .with_gpu_device_id(2);
    assert_eq!(
        profile
            .cache_dir_from_root(Path::new("cache-root"))
            .unwrap(),
        Path::new("cache-root")
            .join("device-2")
            .join("contentvec")
            .join("model")
            .join("audio_1x24000")
    );
}

// =============================================================================
// vc-convert output acceptance
//
// vc-convert (the .pth→.onnx converter) must produce models this crate's
// loaders accept. These tests run its tiny checked-in checkpoint through the
// exact private gatekeepers a user's model passes at load time, and — under
// the `ort` feature — through a real ORT CPU session, porting the streaming
// phase-contract assertions of rvc-onnx-web's streaming-export.spec.ts.
// =============================================================================

fn vc_convert_tiny(mode: vc_convert::ExportMode) -> Vec<u8> {
    let options = vc_convert::ConvertOptions {
        export_mode: mode,
        ..Default::default()
    };
    vc_convert::pth_to_onnx(
        vc_convert::test_fixtures::tiny_v2_f0_pth(),
        &options,
        &mut |_| {},
    )
    .expect("tiny fixture converts")
    .onnx_bytes
}

/// onnx_meta reads from a path, so round-trip through a temp file.
fn with_temp_model<T>(name: &str, bytes: &[u8], f: impl FnOnce(&Path) -> T) -> T {
    let dir = tensor_rt_temp_dir(name);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.onnx");
    fs::write(&path, bytes).unwrap();
    let result = f(&path);
    let _ = fs::remove_dir_all(&dir);
    result
}

#[test]
fn vc_convert_streaming_export_passes_onnx_meta_gatekeepers() {
    let bytes = vc_convert_tiny(vc_convert::ExportMode::Streaming);
    with_temp_model("convert-streaming", &bytes, |path| {
        let io = super::onnx_meta::read_model_io(path).unwrap();

        let names = io.resolve_rvc_io_names().unwrap();
        assert_eq!(names.feats, "phone");
        assert_eq!(names.p_len, "phone_lengths");
        assert_eq!(names.pitch, "pitch");
        assert_eq!(names.pitchf, "pitchf");
        assert_eq!(names.sid, "ds");
        assert_eq!(names.audio, "audio");
        let rnd = names.rnd.expect("rnd input resolved");
        assert_eq!(rnd.name, "rnd");
        assert_eq!(rnd.channels, 8); // tiny fixture inter_channels
        assert_eq!(names.nsf_noise.as_deref(), Some("nsf_noise"));
        assert_eq!(names.phase_in.as_deref(), Some("phase_in"));
        assert_eq!(names.phase_out.as_deref(), Some("streaming_nsf_phase"));

        io.validate_rvc_metadata().unwrap();
        assert_eq!(io.rvc_sample_rate(), Some(40_000));
        assert_eq!(io.feat_channels(&names.feats).unwrap(), 768);

        let stream = io.stream_format().unwrap().expect("streaming export");
        assert_eq!(stream.version, 1);
        assert_eq!(stream.frame_hop, 400);
        assert_eq!(stream.sample_rate, 40_000);
    });
}

#[test]
fn vc_convert_webui_export_passes_onnx_meta_gatekeepers() {
    let bytes = vc_convert_tiny(vc_convert::ExportMode::Webui);
    with_temp_model("convert-webui", &bytes, |path| {
        let io = super::onnx_meta::read_model_io(path).unwrap();

        let names = io.resolve_rvc_io_names().unwrap();
        assert_eq!(names.feats, "phone");
        assert_eq!(names.sid, "ds");
        assert!(names.rnd.is_some());
        assert_eq!(names.nsf_noise, None);
        assert_eq!(names.phase_in, None);
        assert_eq!(names.phase_out, None);

        io.validate_rvc_metadata().unwrap();
        assert_eq!(io.rvc_sample_rate(), Some(40_000));
        assert!(io.stream_format().unwrap().is_none());
    });
}

/// ORT CPU execution of the converted streaming model (port of the runtime
/// half of streaming-export.spec.ts). Runs in the CI `cpu` feature job; the
/// tiny fixture keeps it sub-second.
#[cfg(feature = "ort")]
mod vc_convert_ort {
    use ort::session::Session;
    use ort::value::Tensor;

    use super::vc_convert_tiny;

    const HOP: usize = 400; // tiny fixture upsample rates 10*10*2*2
    const INTER: usize = 8; // tiny fixture inter_channels

    fn session() -> Session {
        // Under the windowsml feature ORT is load-dynamic; bind it to the
        // Windows App SDK runtime exactly like the real session path does.
        #[cfg(all(windows, feature = "windowsml"))]
        crate::windows_ml::ensure_initialized().unwrap();
        Session::builder()
            .unwrap()
            .with_intra_threads(1)
            .unwrap()
            .commit_from_memory(&vc_convert_tiny(vc_convert::ExportMode::Streaming))
            .expect("converted model loads under ORT CPU")
    }

    /// Port of the spec's makeFeeds: constant voiced pitch, zeroed NSF
    /// noise, fixed posterior noise, one-hot phone frame.
    fn run(session: &mut Session, phone_len: usize, phase_in: f32) -> (Vec<f32>, Vec<f32>) {
        let audio_len = phone_len * HOP;
        let mut phone = vec![0.0f32; phone_len * 768];
        phone[0] = 1.0;
        let outputs = session
            .run(ort::inputs![
                "phone" => Tensor::from_array(([1usize, phone_len, 768], phone)).unwrap(),
                "phone_lengths" => Tensor::from_array(([1usize], vec![phone_len as i64])).unwrap(),
                "pitch" => Tensor::from_array(([1usize, phone_len], vec![100i64; phone_len])).unwrap(),
                "pitchf" => Tensor::from_array(([1usize, phone_len], vec![100.0f32; phone_len])).unwrap(),
                "ds" => Tensor::from_array(([1usize], vec![0i64])).unwrap(),
                "rnd" => Tensor::from_array(([1usize, INTER, phone_len], vec![0.25f32; INTER * phone_len])).unwrap(),
                "nsf_noise" => Tensor::from_array(([1usize, audio_len, 1], vec![0.0f32; audio_len])).unwrap(),
                "phase_in" => Tensor::from_array(([1usize, 1, 1], vec![phase_in])).unwrap(),
            ])
            .expect("streaming model runs");
        let (audio_shape, audio) = outputs["audio"].try_extract_tensor::<f32>().unwrap();
        let (phase_shape, phase) = outputs["streaming_nsf_phase"]
            .try_extract_tensor::<f32>()
            .unwrap();
        assert_eq!(audio_shape.iter().product::<i64>(), audio_len as i64);
        assert_eq!(
            phase_shape.to_vec(),
            vec![1, audio_len as i64, 1],
            "phase output must be per-sample"
        );
        (audio.to_vec(), phase.to_vec())
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn streaming_session_is_reproducible_with_fixed_noise() {
        let mut session = session();
        let (audio1, phase1) = run(&mut session, 4, 0.125);
        let (audio2, phase2) = run(&mut session, 4, 0.125);
        assert_eq!(max_abs_diff(&audio1, &audio2), 0.0);
        assert_eq!(max_abs_diff(&phase1, &phase2), 0.0);
    }

    #[test]
    fn streaming_phase_matches_across_overlapping_windows() {
        use super::super::time_state::{RvcTimeState, StreamParams};

        let mut session = session();
        let frames = 4;
        let advance = 2;
        let mut state = RvcTimeState::new(
            None,
            Some(StreamParams {
                frame_hop: HOP,
                sample_rate: 40_000,
            }),
        );
        let (_, mut previous) = run(&mut session, frames, state.phase_in().unwrap());
        // Exercise the production carry, not a hand-picked output index. Repeated
        // overlapping windows expose a per-chunk phase increment that adjacent
        // windows using first.last() would never catch. Phase is periodic, so
        // compare circular distance to tolerate equivalent 0/1 representations.
        for _ in 0..4 {
            state.roll(advance, frames);
            assert!(state.set_phase_from_output(&previous));
            let (_, current) = run(&mut session, frames, state.phase_in().unwrap());
            let error = previous[advance * HOP..]
                .iter()
                .zip(&current)
                .map(|(a, b)| {
                    let delta = (a - b).abs().rem_euclid(1.0);
                    delta.min(1.0 - delta)
                })
                .fold(0.0, f32::max);
            assert!(error < 1e-5, "overlapping phase diverged by {error} cycles");
            previous = current;
        }
    }

    #[test]
    fn streaming_phase_is_continuous_across_split_windows() {
        let mut session = session();

        let (_, continuous) = run(&mut session, 4, 0.0);
        let (_, first) = run(&mut session, 2, 0.0);
        // Adjacent windows carry the phase after the previous window's last sample.
        let next_phase = *first.last().unwrap();
        let (_, second) = run(&mut session, 2, next_phase);

        let mut split = first.clone();
        split.extend_from_slice(&second);
        assert!(
            max_abs_diff(&continuous, &split) < 1e-5,
            "split-window phase diverged from the continuous run by {}",
            max_abs_diff(&continuous, &split)
        );
        assert!((continuous.last().unwrap() - second.last().unwrap()).abs() < 1e-5);
    }
}

/// Local-only validation against a real checkpoint (never runs in CI).
///
/// Set `VC_CONVERT_TEST_PTH` to a real RVC v2/F0 `.pth` and run
/// `cargo test -p vc-core --features ort vc_convert_real -- --ignored --nocapture`.
/// Optionally set `VC_CONVERT_TEST_REF_ONNX` to a streaming export of the same
/// checkpoint produced by rvc-onnx-web to compare audio under identical noise.
#[cfg(feature = "ort")]
#[test]
#[ignore = "needs VC_CONVERT_TEST_PTH pointing at a real checkpoint"]
fn vc_convert_real_model_matches_reference() {
    use ort::session::Session;
    use ort::value::Tensor;

    let pth_path = std::env::var("VC_CONVERT_TEST_PTH")
        .expect("set VC_CONVERT_TEST_PTH to a real RVC v2/F0 .pth");
    let pth = fs::read(&pth_path).unwrap();
    let conversion =
        vc_convert::pth_to_onnx(&pth, &vc_convert::ConvertOptions::default(), &mut |_| {})
            .expect("real checkpoint converts");
    println!(
        "converted {} -> {} bytes, sr {}",
        pth_path,
        conversion.onnx_bytes.len(),
        conversion.sample_rate
    );

    // The converted model must pass the same loader gatekeepers as the tiny
    // fixture, and run under ORT CPU.
    with_temp_model("convert-real", &conversion.onnx_bytes, |path| {
        let io = super::onnx_meta::read_model_io(path).unwrap();
        io.resolve_rvc_io_names().unwrap();
        io.validate_rvc_metadata().unwrap();
        assert!(io.stream_format().unwrap().is_some());
    });

    #[cfg(all(windows, feature = "windowsml"))]
    crate::windows_ml::ensure_initialized().unwrap();
    let build_session = |bytes: &[u8]| -> Session {
        Session::builder()
            .unwrap()
            .with_intra_threads(1)
            .unwrap()
            .commit_from_memory(bytes)
            .unwrap()
    };
    let mut session = build_session(&conversion.onnx_bytes);

    // Deterministic feeds sized for a real v2 model (768 features,
    // inter_channels 192). Fixed noise so a reference model given the same
    // feeds must produce the same audio.
    let phone_len = 32usize;
    let inter = 192usize;
    let rnd = io_names_rnd_channels(&conversion.onnx_bytes).unwrap_or(inter);
    let hop = real_model_frame_hop(&conversion.onnx_bytes);
    let audio_len = phone_len * hop;
    // Exporters use different aliases (e.g. pitchf/nsff0 and ds/sid).
    // Resolve each graph independently, as the production loader does.
    let resolve_names = |bytes: &[u8]| {
        with_temp_model("convert-real-names", bytes, |path| {
            super::onnx_meta::read_model_io(path)
                .unwrap()
                .resolve_rvc_io_names()
                .unwrap()
        })
    };
    let names = resolve_names(&conversion.onnx_bytes);
    let feeds = |phase_in: f32, names: &RvcIoNames| {
        let mut phone = vec![0.0f32; phone_len * 768];
        for (i, v) in phone.iter_mut().enumerate() {
            *v = ((i % 97) as f32 / 97.0 - 0.5) * 0.1;
        }
        ort::inputs![
            names.feats.clone() => Tensor::from_array(([1usize, phone_len, 768], phone)).unwrap(),
            names.p_len.clone() => Tensor::from_array(([1usize], vec![phone_len as i64])).unwrap(),
            names.pitch.clone() => Tensor::from_array(([1usize, phone_len], vec![120i64; phone_len])).unwrap(),
            names.pitchf.clone() => Tensor::from_array(([1usize, phone_len], vec![160.0f32; phone_len])).unwrap(),
            names.sid.clone() => Tensor::from_array(([1usize], vec![0i64])).unwrap(),
            names.rnd.as_ref().unwrap().name.clone() => Tensor::from_array(([1usize, rnd, phone_len], vec![0.1f32; rnd * phone_len])).unwrap(),
            names.nsf_noise.clone().unwrap() => Tensor::from_array(([1usize, audio_len, 1], vec![0.0f32; audio_len])).unwrap(),
            names.phase_in.clone().unwrap() => Tensor::from_array(([1usize, 1, 1], vec![phase_in])).unwrap(),
        ]
    };
    let outputs = session.run(feeds(0.0, &names)).expect("real model runs");
    let (_, audio) = outputs[names.audio.as_str()]
        .try_extract_tensor::<f32>()
        .unwrap();
    let peak = audio.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!("audio: {} samples, peak {peak}", audio.len());
    assert_eq!(audio.len(), audio_len);
    assert!(
        peak > 1e-4 && peak <= 1.0,
        "audio is silent or clipped: peak {peak}"
    );
    let audio = audio.to_vec();
    drop(outputs);

    if let Ok(ref_path) = std::env::var("VC_CONVERT_TEST_REF_ONNX") {
        let reference = fs::read(&ref_path).unwrap();
        let mut ref_session = build_session(&reference);
        let ref_names = resolve_names(&reference);
        let ref_outputs = ref_session
            .run(feeds(0.0, &ref_names))
            .expect("reference model runs");
        let (_, ref_audio) = ref_outputs[ref_names.audio.as_str()]
            .try_extract_tensor::<f32>()
            .unwrap();
        assert_eq!(audio.len(), ref_audio.len());
        assert!(audio.iter().chain(ref_audio).all(|v| v.is_finite()));
        let max_diff = audio
            .iter()
            .zip(ref_audio)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("max abs diff vs rvc-onnx-web reference: {max_diff}");
        assert!(
            max_diff < 1e-4,
            "converted audio diverges from rvc-onnx-web reference by {max_diff}"
        );
    }
}

/// Best-effort rnd channel count from the converted model's own metadata
/// (avoids hardcoding inter_channels for arbitrary real checkpoints).
#[cfg(feature = "ort")]
fn io_names_rnd_channels(onnx_bytes: &[u8]) -> Option<usize> {
    with_temp_model("convert-real-io", onnx_bytes, |path| {
        let io = super::onnx_meta::read_model_io(path).ok()?;
        let names = io.resolve_rvc_io_names().ok()?;
        usize::try_from(names.rnd?.channels).ok()
    })
}

#[cfg(feature = "ort")]
fn real_model_frame_hop(onnx_bytes: &[u8]) -> usize {
    with_temp_model("convert-real-hop", onnx_bytes, |path| {
        super::onnx_meta::read_model_io(path)
            .ok()
            .and_then(|io| io.stream_format().ok().flatten())
            .map(|s| s.frame_hop)
            .unwrap_or(400)
    })
}
