//! NSF source: sine generator with cumulative phase (and the streaming
//! phase-carry contract) plus SourceModuleHnNSF.
//!
//! Ported from rvc-onnx-web (https://github.com/visgotti/rvc-onnx-web),
//! MIT License — `buildSineGenerator` / `buildSourceModuleHnNSF` in
//! `src/synthesizer-builder.ts`.
//!
//! Streaming contract (see `rvc.streaming.phase_contract` metadata):
//! `phase_in [1,1,1]` is the normalized fundamental phase at the window
//! start; every generated sample's normalized phase is emitted as
//! `streaming_nsf_phase [1, audio_len, 1]`, and the caller feeds the value
//! at the next window's start position back in as `phase_in`.

use anyhow::{anyhow, Result};

use crate::onnx::DataType;

use super::{weight_norm, GraphBuilder};

pub(crate) struct DecoderOptions<'a> {
    pub streaming: bool,
    pub nsf_noise_input: Option<&'a str>,
    pub phase_input: Option<&'a str>,
}

pub(crate) struct SourceOut {
    pub source: String,
    pub phase_trace: Option<String>,
}

struct SineOut {
    sine_waveforms: String,
    phase_trace: Option<String>,
}

const SINE_AMPLITUDE: f32 = 0.1;
const NOISE_STDDEV: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 0.0;

/// SineGenerator: voiced mask, per-sample phase accumulation (cumsum with
/// fmod wrapping), sine synthesis, and voiced/unvoiced noise mixing.
fn build_sine_generator(
    b: &mut GraphBuilder<'_>,
    f0: &str,
    upsampling_factor: usize,
    sampling_rate: u32,
    options: &DecoderOptions<'_>,
) -> Result<SineOut> {
    // Step 1: voiced mask = (f0 > threshold) as float
    let threshold = {
        let n = b.unique("voiced_threshold");
        b.add_scalar(&n, VOICED_THRESHOLD)
    };
    let voiced_bool = b.binary_new("Greater", f0, &threshold, "voiced_bool");
    let voiced_mask_f32 = b.unique("voiced_mask_f32");
    b.cast(&voiced_bool, &voiced_mask_f32, DataType::Float);

    // Step 2: f0 [B, L] -> [B, L, 1]
    let unsqueeze_neg1 = {
        let n = b.unique("unsqueeze_neg1");
        b.add_i64(&n, vec![-1], vec![1])
    };
    let f0_expanded = b.unique("f0_expanded");
    b.unsqueeze(f0, &unsqueeze_neg1, &f0_expanded);

    // Step 3: upsampling grid [1..=factor] as float
    let one_int = {
        let n = b.unique("one_int");
        b.add_i64(&n, vec![1], vec![])
    };
    let up_plus_one = {
        let n = b.unique("up_factor_plus_one");
        b.add_i64(&n, vec![upsampling_factor as i64 + 1], vec![])
    };
    let grid = b.unique("upsampling_grid");
    b.range(&one_int, &up_plus_one, &one_int, &grid);
    let grid_f32 = b.unique("upsampling_grid_f32");
    b.cast(&grid, &grid_f32, DataType::Float);

    // Step 4: phase_increments_raw = (f0 / sr) * grid → [B, L, F]
    let sr_const = {
        let n = b.unique("sampling_rate");
        b.add_scalar(&n, sampling_rate as f32)
    };
    let f0_normalized = b.binary_new("Div", &f0_expanded, &sr_const, "f0_normalized");
    let phase_increments_raw =
        b.binary_new("Mul", &f0_normalized, &grid_f32, "phase_increments_raw");

    // Step 5: carry phase across frames.
    // phase_remainder = fmod(phase[:, :-1, -1:] + 0.5, 1.0) - 0.5
    // cumulative = cumsum(phase_remainder, dim=1).fmod(1.0), padded by one
    // leading zero frame, broadcast-added back.
    let start_5a = {
        let n = b.unique("slice_start_5a");
        b.add_i64(&n, vec![0, 0, upsampling_factor as i64 - 1], vec![3])
    };
    let end_5a = {
        let n = b.unique("slice_end_5a");
        b.add_i64(&n, vec![2_147_483_647, -1, 2_147_483_647], vec![3])
    };
    let axes_5a = {
        let n = b.unique("slice_axes_5a");
        b.add_i64(&n, vec![0, 1, 2], vec![3])
    };
    let steps_5a = {
        let n = b.unique("slice_steps_5a");
        b.add_i64(&n, vec![1, 1, 1], vec![3])
    };
    let phase_last_col = b.unique("phase_last_col");
    b.slice(
        &phase_increments_raw,
        &start_5a,
        &end_5a,
        &axes_5a,
        &steps_5a,
        &phase_last_col,
    );

    let half = {
        let n = b.unique("half");
        b.add_scalar(&n, 0.5)
    };
    let one_f32 = {
        let n = b.unique("one_f32");
        b.add_scalar(&n, 1.0)
    };
    let phase_plus_half = b.binary_new("Add", &phase_last_col, &half, "phase_plus_half");
    let phase_mod_one = b.unique("phase_mod_one");
    b.fmod(&phase_plus_half, &one_f32, &phase_mod_one);
    let phase_remainder = b.binary_new("Sub", &phase_mod_one, &half, "phase_remainder");

    let cumsum_axis = {
        let n = b.unique("cumsum_axis");
        b.add_i64(&n, vec![1], vec![])
    };
    let cumulative_raw = b.unique("cumulative_phase_raw");
    b.cumsum(&phase_remainder, &cumsum_axis, &cumulative_raw);
    let cumulative = b.unique("cumulative_phase");
    b.fmod(&cumulative_raw, &one_f32, &cumulative);

    // Pad one frame of zeros at the start of dim 1: [B, L-1, 1] → [B, L, 1]
    let pad_const = {
        let n = b.unique("pad_const");
        b.add_i64(&n, vec![0, 1, 0, 0, 0, 0], vec![6])
    };
    let zero_f32 = {
        let n = b.unique("zero_f32");
        b.add_scalar(&n, 0.0)
    };
    let cumulative_padded = b.unique("cumulative_phase_padded");
    b.push(
        "Pad",
        &[&cumulative, &pad_const, &zero_f32],
        &[&cumulative_padded],
        vec![],
    );

    let phase_increments = b.binary_new(
        "Add",
        &phase_increments_raw,
        &cumulative_padded,
        "phase_increments",
    );

    // Step 6: reshape [B, L, F] → [B, L*F, 1]
    let batch_idx = {
        let n = b.unique("batch_idx_sine");
        b.add_i64(&n, vec![0], vec![1])
    };
    let phase_shape = b.unique("phase_shape");
    b.unary("Shape", &phase_increments, &phase_shape);
    let batch_dim = b.unique("batch_dim_phase");
    b.gather(&phase_shape, &batch_idx, &batch_dim, 0);

    let neg_one = {
        let n = b.unique("neg_one");
        b.add_i64(&n, vec![-1], vec![1])
    };
    let one_dim = {
        let n = b.unique("one_dim");
        b.add_i64(&n, vec![1], vec![1])
    };
    let reshape_shape = b.unique("reshape_phase_shape");
    b.concat(&[&batch_dim, &neg_one, &one_dim], &reshape_shape, 0);
    let phase_flat = b.unique("phase_increments_flat");
    b.reshape(&phase_increments, &reshape_shape, &phase_flat);

    // Streaming phase carry: add the caller's window-start phase and wrap.
    // The wrapped phase is the graph output "streaming_nsf_phase".
    let mut phase_for_sine = phase_flat.clone();
    let mut phase_trace = None;
    if options.streaming {
        let phase_input = options
            .phase_input
            .ok_or_else(|| anyhow!("streaming NSF export requires phase_in"))?;
        let phase_with_input = b.binary_new("Add", &phase_flat, phase_input, "phase_with_input");
        phase_for_sine = "streaming_nsf_phase".to_owned();
        b.fmod(&phase_with_input, &one_f32, &phase_for_sine);
        phase_trace = Some(phase_for_sine.clone());
    }

    // Steps 7–8 (harmonic scaling / random harmonic phase) are no-ops for
    // RVC's harmonic_num=0, mirroring the TS.

    // Step 9: sine = sin(2π · phase) · amplitude
    let two_pi = {
        let n = b.unique("two_pi");
        b.add_scalar(&n, 2.0 * std::f32::consts::PI)
    };
    let phase_radians = b.binary_new("Mul", &phase_for_sine, &two_pi, "phase_radians");
    let sine_raw = b.unary_new("Sin", &phase_radians, "sine_raw");
    let sine_amp = {
        let n = b.unique("sine_amplitude");
        b.add_scalar(&n, SINE_AMPLITUDE)
    };
    let sine_scaled = b.binary_new("Mul", &sine_raw, &sine_amp, "sine_scaled");

    // Step 10: nearest-upsample the voiced mask to sample rate
    let voiced_expanded = b.unique("voiced_mask_expanded");
    b.unsqueeze(&voiced_mask_f32, &unsqueeze_neg1, &voiced_expanded);
    let voiced_t = b.unique("voiced_mask_transposed");
    b.transpose(&voiced_expanded, &voiced_t, vec![0, 2, 1]);

    let scales = {
        let n = b.unique("resize_scales");
        b.add_f32(&n, vec![1.0, 1.0, upsampling_factor as f32], vec![3])
    };
    let voiced_resized = b.unique("voiced_mask_resized");
    b.resize_nearest(&voiced_t, &scales, &voiced_resized);
    let voiced_upsampled = b.unique("voiced_mask_upsampled");
    b.transpose(&voiced_resized, &voiced_upsampled, vec![0, 2, 1]);

    // Step 11: noise_amplitude = voiced·stddev + (1-voiced)·(amp/3)
    let noise_std = {
        let n = b.unique("noise_stddev");
        b.add_scalar(&n, NOISE_STDDEV)
    };
    let unvoiced_amp = {
        let n = b.unique("unvoiced_amp");
        b.add_scalar(&n, SINE_AMPLITUDE / 3.0)
    };
    let voiced_noise_amp = b.binary_new("Mul", &voiced_upsampled, &noise_std, "voiced_noise_amp");
    let one_minus_voiced = b.binary_new("Sub", &one_f32, &voiced_upsampled, "one_minus_voiced");
    let unvoiced_noise_amp = b.binary_new(
        "Mul",
        &one_minus_voiced,
        &unvoiced_amp,
        "unvoiced_noise_amp",
    );
    let noise_amplitude = b.binary_new(
        "Add",
        &voiced_noise_amp,
        &unvoiced_noise_amp,
        "noise_amplitude",
    );

    // Step 12: noise source — external input when streaming, else
    // RandomNormalLike (the one random node a webui export keeps).
    let noise_random = if options.streaming {
        options
            .nsf_noise_input
            .ok_or_else(|| anyhow!("streaming NSF export requires nsf_noise"))?
            .to_owned()
    } else {
        let name = b.unique("noise_random");
        b.random_normal_like(&sine_scaled, &name);
        name
    };
    let noise = b.binary_new("Mul", &noise_amplitude, &noise_random, "noise");

    // Step 13: sine_waveforms = sine · voiced + noise
    let sine_voiced = b.binary_new("Mul", &sine_scaled, &voiced_upsampled, "sine_voiced");
    let sine_waveforms = b.binary_new("Add", &sine_voiced, &noise, "sine_waveforms");

    Ok(SineOut {
        sine_waveforms,
        phase_trace,
    })
}

/// SourceModuleHnNSF: sine source shaped by `l_linear` + tanh, transposed to
/// `[B, C, T]` for the decoder's noise convs.
pub(crate) fn build_source_module(
    b: &mut GraphBuilder<'_>,
    f0: &str,
    upsampling_factor: usize,
    sampling_rate: u32,
    prefix: &str,
    options: &DecoderOptions<'_>,
) -> Result<SourceOut> {
    let sine = build_sine_generator(b, f0, upsampling_factor, sampling_rate, options)?;

    let linear_weight = format!("{prefix}l_linear.weight");
    if b.has_weight(&linear_weight) {
        let linear_bias = if b.has_weight(&format!("{prefix}l_linear.bias")) {
            Some(b.add_weight(&format!("{prefix}l_linear.bias"))?)
        } else {
            None
        };
        let linear_out = b.unique("nsf_linear_out");
        weight_norm::linear(
            b,
            &sine.sine_waveforms,
            &linear_weight,
            linear_bias.as_deref(),
            &linear_out,
        )?;
        let tanh_out = b.unary_new("Tanh", &linear_out, "nsf_tanh_out");
        let source = b.unique("nsf_source");
        b.transpose(&tanh_out, &source, vec![0, 2, 1]);
        return Ok(SourceOut {
            source,
            phase_trace: sine.phase_trace,
        });
    }

    let source = b.unique("nsf_source_transposed");
    b.transpose(&sine.sine_waveforms, &source, vec![0, 2, 1]);
    Ok(SourceOut {
        source,
        phase_trace: sine.phase_trace,
    })
}
