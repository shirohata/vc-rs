use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::Provider;

use super::pitch::{
    align_pitchf_to_features, center_crop_pitchf_to_features, pitchf_tail_for_output,
};
use super::shape::{
    aligned_rvc_input_len, keep_tail_in_place, onnx_silence_front_feature_frames,
    output_len_from_convert_size, tensor_rt_model_input_samples_16k, RVC_SAMPLE_RATE,
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
            .join("contentvec")
            .join("content_vec_500_0123456789abcdef")
            .join("audio_1x24000")
    );
}

#[test]
fn derives_tensorrt_contentvec_profile_from_default_realtime_chunking() {
    assert_eq!(
        tensor_rt_model_input_samples_16k(960, 48_000, 107, 48_000),
        18_240
    );
    let contentvec = TensorRtSessionProfile::single_input(ModelRole::ContentVec, "audio", 18_240);
    let rmvpe = TensorRtSessionProfile::single_input(ModelRole::Rmvpe, "waveform", 18_240);
    let rvc = TensorRtSessionProfile::rvc(114, 768);

    assert_eq!(contentvec.profile_shapes, "audio:1x18240");
    assert_eq!(rmvpe.profile_shapes, "waveform:1x18240");
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
    let rvc = TensorRtSessionProfile::rvc(75, 768);

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
    let rvc = TensorRtSessionProfile::rvc(113, 768);

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

#[test]
fn output_reference_audio_uses_tail_matching_trimmed_output() {
    let mut state = RvcStreamState::new();
    state.audio_buffer = (0..8).map(|value| value as f32).collect();
    let mut scratch = Vec::new();

    let reference = state
        .output_reference_audio(RVC_SAMPLE_RATE, RVC_SAMPLE_RATE, 5, &mut scratch)
        .unwrap();

    assert_eq!(reference, &[3.0, 4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn output_reference_audio_left_pads_when_history_is_short() {
    let mut state = RvcStreamState::new();
    state.audio_buffer = vec![1.0, 2.0];
    let mut scratch = Vec::new();

    let reference = state
        .output_reference_audio(RVC_SAMPLE_RATE, RVC_SAMPLE_RATE, 4, &mut scratch)
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
    let mut state = RvcStreamState::new();
    let input = vec![0.0; 24_000];
    let out = state
        .generate_input(&input, 48_000, 1_536, 1_536, 4_096)
        .unwrap();
    assert_eq!(out.convert_size, 29_760);
}

#[test]
fn stream_state_derives_out_size_from_extra_convert_size() {
    let mut state = RvcStreamState::new();
    let input = vec![0.0; 24_000];
    let out = state
        .generate_input(&input, 48_000, 1_536, 1_536, 4_096)
        .unwrap();
    assert_eq!(out.out_size, 25_665);
}

#[test]
fn stream_state_zero_pads_initial_buffer() {
    let mut state = RvcStreamState::new();
    let out = state
        .generate_input(&[1.0, 2.0, 3.0, 4.0], 48_000, 0, 0, 4_096)
        .unwrap();
    assert_eq!(state.audio_buffer.len(), out.convert_size);
    assert!(state.audio_buffer[..state.audio_buffer.len() - 4]
        .iter()
        .all(|x| *x == 0.0));
    assert_eq!(
        &state.audio_buffer[state.audio_buffer.len() - 4..],
        &[1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn stream_state_keeps_16k_history_for_embedder() {
    let mut state = RvcStreamState::new();
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
    let mut state = RvcStreamState::new();
    let mut input = vec![1.0; 80];
    input.extend(std::iter::repeat_n(0.0, 80));

    let out = state.generate_input(&input, 16_000, 480, 240, 0).unwrap();

    assert!((out.volume - 0.5f32.sqrt()).abs() < 1e-6);
}

#[test]
fn stream_state_volume_keeps_decay_from_previous_chunk() {
    let mut state = RvcStreamState::new();
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
fn stream_state_updates_pitchf_buffer_like_vcclient() {
    let mut state = RvcStreamState::new();
    state.pitchf_buffer = vec![0.0, 1.0, 2.0];

    state.update_pitchf_from_rmvpe_frames(&[10.0, 20.0, 30.0, 40.0, 50.0]);

    assert_eq!(state.pitchf_buffer, vec![10.0, 20.0, 30.0]);
}

#[test]
fn stream_state_pitch_update_writes_short_rmvpe_to_tail() {
    let mut state = RvcStreamState::new();
    state.pitchf_buffer = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    state.update_pitchf_from_rmvpe_frames(&[10.0, 20.0, 30.0]);

    assert_eq!(state.pitchf_buffer, vec![1.0, 2.0, 10.0, 20.0, 30.0]);
}

#[test]
fn derives_vcclient_onnx_silence_front_feature_offset() {
    assert_eq!(onnx_silence_front_feature_frames(4096), 6);
}
