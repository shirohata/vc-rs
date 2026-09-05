//! TextEncoder (enc_p): phone/pitch embedding, transformer encoder with
//! relative positional multi-head attention, FFN, and projection to
//! (m_p, logs_p).
//!
//! Ported from rvc-onnx-web (https://github.com/visgotti/rvc-onnx-web),
//! MIT License — `buildTextEncoder` / `buildTransformerEncoder` /
//! `buildMultiHeadAttention` / `buildFFN` / the relative-position helpers in
//! `src/synthesizer-builder.ts`. Comments citing Python refer to RVC's
//! `attentions.py`, same as the TS source.

use anyhow::{anyhow, Result};

use crate::checkpoint::RvcConfig;
use crate::onnx::DataType;

use super::{weight_norm, GraphBuilder};

pub(crate) struct TextEncoderOut {
    pub m_p: String,
    pub logs_p: String,
    pub x_mask: String,
}

/// Build sequence mask: `mask[i, j] = 1.0 if j < lengths[i] else 0.0`,
/// shaped `[batch, 1, max_len]` (port of `buildSequenceMask`).
fn build_sequence_mask(b: &mut GraphBuilder<'_>, lengths: &str, max_len: &str, out: &str) {
    let zero = {
        let n = b.unique("zero");
        b.add_i64(&n, vec![0], vec![])
    };
    let one = {
        let n = b.unique("one");
        b.add_i64(&n, vec![1], vec![])
    };

    let range_vals = b.unique("range_vals");
    b.range(&zero, max_len, &one, &range_vals);

    let axes0 = {
        let n = b.unique("unsqueeze_axes_0");
        b.add_i64(&n, vec![0], vec![1])
    };
    let range_unsqueezed = b.unique("range_unsqueezed");
    b.unsqueeze(&range_vals, &axes0, &range_unsqueezed);

    let axes1 = {
        let n = b.unique("unsqueeze_axes_1");
        b.add_i64(&n, vec![1], vec![1])
    };
    let lengths_unsqueezed = b.unique("lengths_unsqueezed");
    b.unsqueeze(lengths, &axes1, &lengths_unsqueezed);

    let mask_bool = b.unique("mask_bool");
    b.binary("Less", &range_unsqueezed, &lengths_unsqueezed, &mask_bool);

    let mask_float = b.unique("mask_float");
    b.cast(&mask_bool, &mask_float, DataType::Float);

    let axes_mid = {
        let n = b.unique("unsqueeze_axes_mid");
        b.add_i64(&n, vec![1], vec![1])
    };
    b.unsqueeze(&mask_float, &axes_mid, out);
}

/// enc_p: returns (m_p, logs_p, x_mask).
pub(crate) fn build_text_encoder(
    b: &mut GraphBuilder<'_>,
    config: &RvcConfig,
    use_f0: bool,
) -> Result<TextEncoderOut> {
    let prefix = "enc_p.";

    // Linear embedding: emb_phone(phone), [B, T, feat] -> [B, T, hidden]
    let emb_phone_bias = b.add_weight(&format!("{prefix}emb_phone.bias"))?;
    let phone_embedded = b.unique("phone_embedded");
    weight_norm::linear(
        b,
        "phone",
        &format!("{prefix}emb_phone.weight"),
        Some(&emb_phone_bias),
        &phone_embedded,
    )?;

    let mut x = phone_embedded;

    // Pitch embedding lookup, added to the phone embedding.
    if use_f0 && b.has_weight(&format!("{prefix}emb_pitch.weight")) {
        let emb_pitch = b.add_weight(&format!("{prefix}emb_pitch.weight"))?;
        let pitch_embedded = b.unique("pitch_embedded");
        b.gather(&emb_pitch, "pitch", &pitch_embedded, 0);
        let x_with_pitch = b.binary_new("Add", &x, &pitch_embedded, "x_with_pitch");
        x = x_with_pitch;
    }

    // Scale by sqrt(hidden_channels)
    let scale = b.add_scalar("enc_scale", (config.hidden_channels as f32).sqrt());
    let x_scaled = b.binary_new("Mul", &x, &scale, "x_scaled");

    // LeakyReLU(0.1) — RVC's TextEncoder applies lrelu right after embedding.
    let x_activated = b.unique("x_activated");
    b.leaky_relu(&x_scaled, &x_activated, 0.1);

    // [B, T, H] -> [B, H, T]
    let x_transposed = b.unique("x_transposed");
    b.transpose(&x_activated, &x_transposed, vec![0, 2, 1]);

    // Dynamic T from the phone input shape (index 1). The index stays a
    // scalar so Gather yields a scalar Range limit; TensorRT rejects a
    // rank-1 limit here (comment carried from the TS source).
    let phone_shape = b.unique("phone_shape");
    b.unary("Shape", "phone", &phone_shape);
    let dim_one_idx = b.add_i64("dim_one_idx", vec![1], vec![]);
    let phone_len_dynamic = b.unique("phone_len_dynamic");
    b.gather(&phone_shape, &dim_one_idx, &phone_len_dynamic, 0);

    let x_mask = b.unique("x_mask");
    build_sequence_mask(b, "phone_lengths", &phone_len_dynamic, &x_mask);

    let x_masked = b.binary_new("Mul", &x_transposed, &x_mask, "x_masked");

    // Transformer encoder layers
    let x_encoded =
        build_transformer_encoder(b, config, &x_masked, &x_mask, &format!("{prefix}encoder."))?;

    // proj(x) -> [B, inter*2, T], masked, split into m_p / logs_p
    let proj_weight = b.add_weight(&format!("{prefix}proj.weight"))?;
    let proj_bias = b.add_weight(&format!("{prefix}proj.bias"))?;
    let stats = b.unique("stats");
    b.conv1d(
        &x_encoded,
        &proj_weight,
        Some(&proj_bias),
        &stats,
        1,
        1,
        0,
        1,
    );

    let stats_masked = b.binary_new("Mul", &stats, &x_mask, "stats_masked");

    let split_sizes = {
        let n = b.unique("stats_split_sizes");
        b.add_i64(
            &n,
            vec![config.inter_channels as i64, config.inter_channels as i64],
            vec![2],
        )
    };
    let m_p = b.unique("m_p");
    let logs_p = b.unique("logs_p");
    b.split_sizes(&stats_masked, &split_sizes, &[&m_p, &logs_p], 1);

    Ok(TextEncoderOut {
        m_p,
        logs_p,
        x_mask,
    })
}

fn build_transformer_encoder(
    b: &mut GraphBuilder<'_>,
    config: &RvcConfig,
    input: &str,
    mask: &str,
    prefix: &str,
) -> Result<String> {
    let mut x = input.to_owned();

    // attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1): [B, 1, T, T]
    let axes3 = {
        let n = b.unique("mask_unsqueeze2_axes");
        b.add_i64(&n, vec![3], vec![1])
    };
    let mask_unsq2 = b.unique("mask_unsqueeze2");
    b.unsqueeze(mask, &axes3, &mask_unsq2); // [B, 1, T, 1]

    let axes2 = {
        let n = b.unique("mask_unsqueeze_m1_axes");
        b.add_i64(&n, vec![2], vec![1])
    };
    let mask_unsq_m1 = b.unique("mask_unsqueeze_minus1");
    b.unsqueeze(mask, &axes2, &mask_unsq_m1); // [B, 1, 1, T]

    let attn_mask = b.binary_new("Mul", &mask_unsq2, &mask_unsq_m1, "attn_mask");

    for i in 0..config.n_layers {
        // Self-attention
        let attn_out = build_multi_head_attention(
            b,
            config,
            &x,
            &attn_mask,
            &format!("{prefix}attn_layers.{i}."),
        )?;

        // Residual + LayerNorm 1 (LN acts on channels: transpose around it)
        let x_residual1 = b.binary_new("Add", &x, &attn_out, "x_residual1");
        let norm_w1 = b.add_weight(&format!("{prefix}norm_layers_1.{i}.gamma"))?;
        let norm_b1 = b.add_weight(&format!("{prefix}norm_layers_1.{i}.beta"))?;
        let x_norm1 = b.unique("x_norm1");
        let x_t1 = b.unique("x_transpose1");
        b.transpose(&x_residual1, &x_t1, vec![0, 2, 1]);
        let x_ln1 = b.unique("x_ln1");
        b.layer_norm(&x_t1, &norm_w1, &norm_b1, &x_ln1);
        b.transpose(&x_ln1, &x_norm1, vec![0, 2, 1]);

        // FFN
        let ffn_out = build_ffn(
            b,
            config,
            &x_norm1,
            mask,
            &format!("{prefix}ffn_layers.{i}."),
        )?;

        // Residual + LayerNorm 2
        let x_residual2 = b.binary_new("Add", &x_norm1, &ffn_out, "x_residual2");
        let norm_w2 = b.add_weight(&format!("{prefix}norm_layers_2.{i}.gamma"))?;
        let norm_b2 = b.add_weight(&format!("{prefix}norm_layers_2.{i}.beta"))?;
        let x_t2 = b.unique("x_transpose2");
        b.transpose(&x_residual2, &x_t2, vec![0, 2, 1]);
        let x_ln2 = b.unique("x_ln2");
        b.layer_norm(&x_t2, &norm_w2, &norm_b2, &x_ln2);
        let x_norm2 = b.unique("x_norm2");
        b.transpose(&x_ln2, &x_norm2, vec![0, 2, 1]);

        x = x_norm2;
    }

    Ok(b.binary_new("Mul", &x, mask, "x_enc_final"))
}

// =============================================================================
// Relative positional encoding helpers
// =============================================================================

/// `_get_relative_embeddings(embeddings, length)`: pad the `[1, 2W+1, k]`
/// table to `2*length-1` positions and slice the used window, all with
/// dynamic shapes.
fn build_get_relative_embeddings(
    b: &mut GraphBuilder<'_>,
    embeddings: &str,
    window_size: usize,
    time_dim: &str,
    out: &str,
) {
    let one = {
        let n = b.unique("rel_one");
        b.add_i64(&n, vec![1], vec![1])
    };
    let two = {
        let n = b.unique("rel_two");
        b.add_i64(&n, vec![2], vec![1])
    };
    let zero = {
        let n = b.unique("rel_zero");
        b.add_i64(&n, vec![0], vec![1])
    };
    let window_plus1 = {
        let n = b.unique("rel_window_plus_1");
        b.add_i64(&n, vec![window_size as i64 + 1], vec![1])
    };

    // padLength = max(length - (windowSize + 1), 0)
    let len_minus = b.binary_new("Sub", time_dim, &window_plus1, "rel_len_minus_wp1");
    let pad_length = b.binary_new("Max", &len_minus, &zero, "rel_pad_length");

    // start = max((windowSize + 1) - length, 0)
    let wp1_minus_len = b.binary_new("Sub", &window_plus1, time_dim, "rel_wp1_minus_len");
    let start = b.binary_new("Max", &wp1_minus_len, &zero, "rel_start");

    // end = start + 2 * length - 1
    let two_times_len = b.binary_new("Mul", &two, time_dim, "rel_two_times_len");
    let start_plus = b.binary_new("Add", &start, &two_times_len, "rel_start_plus_two_len");
    let end = b.binary_new("Sub", &start_plus, &one, "rel_end");

    // Pad positions dim: 3-D pads layout [d0_b, d1_b, d2_b, d0_a, d1_a, d2_a]
    let pads_shape = b.unique("rel_pads_shape");
    b.concat(
        &[&zero, &pad_length, &zero, &zero, &pad_length, &zero],
        &pads_shape,
        0,
    );
    let pad_value = {
        let n = b.unique("rel_pad_value");
        b.add_scalar(&n, 0.0)
    };
    let padded = b.unique("rel_emb_padded");
    b.pad_constant(embeddings, &pads_shape, &pad_value, &padded);

    // embeddings[:, start:end, :]
    let axes = {
        let n = b.unique("rel_axes");
        b.add_i64(&n, vec![1], vec![1])
    };
    let steps = {
        let n = b.unique("rel_steps");
        b.add_i64(&n, vec![1], vec![1])
    };
    b.slice(&padded, &start, &end, &axes, &steps, out);
}

/// `matmul(x, y.unsqueeze(0).transpose(-2, -1))`.
fn build_matmul_with_relative_keys(
    b: &mut GraphBuilder<'_>,
    query: &str,
    rel_emb: &str,
    out: &str,
) {
    let axes = {
        let n = b.unique("rel_unsqueeze_axes");
        b.add_i64(&n, vec![0], vec![1])
    };
    let unsqueezed = b.unique("rel_emb_unsqueezed");
    b.unsqueeze(rel_emb, &axes, &unsqueezed);
    let transposed = b.unique("rel_emb_transposed");
    b.transpose(&unsqueezed, &transposed, vec![0, 1, 3, 2]);
    b.binary("MatMul", query, &transposed, out);
}

/// `matmul(x, y.unsqueeze(0))`.
fn build_matmul_with_relative_values(
    b: &mut GraphBuilder<'_>,
    rel_weights: &str,
    rel_emb: &str,
    out: &str,
) {
    let axes = {
        let n = b.unique("relv_unsqueeze_axes");
        b.add_i64(&n, vec![0], vec![1])
    };
    let unsqueezed = b.unique("relv_emb_unsqueezed");
    b.unsqueeze(rel_emb, &axes, &unsqueezed);
    b.binary("MatMul", rel_weights, &unsqueezed, out);
}

/// `_relative_position_to_absolute_position`:
/// `[B, H, L, 2L-1] → [B, H, L, L]` via pad/flatten/pad/reshape/slice.
fn build_relative_to_absolute_position(
    b: &mut GraphBuilder<'_>,
    rel_logits: &str,
    batch_dim: &str,
    heads_dim: &str,
    out: &str,
) {
    let input_shape = b.unique("r2a_input_shape");
    b.unary("Shape", rel_logits, &input_shape);
    let dim2_idx = {
        let n = b.unique("r2a_dim2_idx");
        b.add_i64(&n, vec![2], vec![1])
    };
    let length_dim = b.unique("r2a_length");
    b.gather(&input_shape, &dim2_idx, &length_dim, 0);

    let one = {
        let n = b.unique("r2a_one");
        b.add_i64(&n, vec![1], vec![1])
    };
    let two = {
        let n = b.unique("r2a_two");
        b.add_i64(&n, vec![2], vec![1])
    };

    let length_plus1 = b.binary_new("Add", &length_dim, &one, "r2a_len_plus_1");
    let length_minus1 = b.binary_new("Sub", &length_dim, &one, "r2a_len_minus_1");
    let two_len = b.binary_new("Mul", &two, &length_dim, "r2a_two_len");
    let two_len_minus1 = b.binary_new("Sub", &two_len, &one, "r2a_two_len_minus_1");
    let flat_size = b.binary_new("Mul", &length_dim, &two_len, "r2a_flat_size");

    // Pad last dim by one zero → [B, H, L, 2L]
    let pad1 = {
        let n = b.unique("r2a_pad1");
        b.add_i64(&n, vec![0, 0, 0, 0, 0, 0, 0, 1], vec![8])
    };
    let pad_val1 = {
        let n = b.unique("r2a_pad_val1");
        b.add_scalar(&n, 0.0)
    };
    let padded1 = b.unique("r2a_padded1");
    b.pad_constant(rel_logits, &pad1, &pad_val1, &padded1);

    // Flatten → [B, H, L*2L]
    let flat_shape = b.unique("r2a_flat_shape");
    b.concat(&[batch_dim, heads_dim, &flat_size], &flat_shape, 0);
    let flattened = b.unique("r2a_flattened");
    b.reshape(&padded1, &flat_shape, &flattened);

    // Pad end of last dim by L-1
    let zero = {
        let n = b.unique("r2a_zero");
        b.add_i64(&n, vec![0], vec![1])
    };
    let pad2_shape = b.unique("r2a_pad2_shape");
    b.concat(
        &[&zero, &zero, &zero, &zero, &zero, &length_minus1],
        &pad2_shape,
        0,
    );
    let pad_val2 = {
        let n = b.unique("r2a_pad_val2");
        b.add_scalar(&n, 0.0)
    };
    let padded2 = b.unique("r2a_padded2");
    b.pad_constant(&flattened, &pad2_shape, &pad_val2, &padded2);

    // Reshape → [B, H, L+1, 2L-1]
    let view_shape = b.unique("r2a_view_shape");
    b.concat(
        &[batch_dim, heads_dim, &length_plus1, &two_len_minus1],
        &view_shape,
        0,
    );
    let reshaped = b.unique("r2a_reshaped");
    b.reshape(&padded2, &view_shape, &reshaped);

    // [:, :, :L, L-1:]
    let start_dim2 = {
        let n = b.unique("r2a_start_dim2");
        b.add_i64(&n, vec![0], vec![1])
    };
    let axes_dim2 = {
        let n = b.unique("r2a_axes_dim2");
        b.add_i64(&n, vec![2], vec![1])
    };
    let steps_dim2 = {
        let n = b.unique("r2a_steps_dim2");
        b.add_i64(&n, vec![1], vec![1])
    };
    let sliced1 = b.unique("r2a_sliced1");
    b.slice(
        &reshaped,
        &start_dim2,
        &length_dim,
        &axes_dim2,
        &steps_dim2,
        &sliced1,
    );

    let axes_dim3 = {
        let n = b.unique("r2a_axes_dim3");
        b.add_i64(&n, vec![3], vec![1])
    };
    let steps_dim3 = {
        let n = b.unique("r2a_steps_dim3");
        b.add_i64(&n, vec![1], vec![1])
    };
    b.slice(
        &sliced1,
        &length_minus1,
        &two_len_minus1,
        &axes_dim3,
        &steps_dim3,
        out,
    );
}

/// `_absolute_position_to_relative_position`:
/// `[B, H, L, L] → [B, H, L, 2L-1]`.
fn build_absolute_to_relative_position(
    b: &mut GraphBuilder<'_>,
    abs_weights: &str,
    batch_dim: &str,
    heads_dim: &str,
    out: &str,
) {
    let input_shape = b.unique("a2r_input_shape");
    b.unary("Shape", abs_weights, &input_shape);
    let dim2_idx = {
        let n = b.unique("a2r_dim2_idx");
        b.add_i64(&n, vec![2], vec![1])
    };
    let length_dim = b.unique("a2r_length");
    b.gather(&input_shape, &dim2_idx, &length_dim, 0);

    let one = {
        let n = b.unique("a2r_one");
        b.add_i64(&n, vec![1], vec![1])
    };
    let two = {
        let n = b.unique("a2r_two");
        b.add_i64(&n, vec![2], vec![1])
    };
    let zero = {
        let n = b.unique("a2r_zero");
        b.add_i64(&n, vec![0], vec![1])
    };

    let length_minus1 = b.binary_new("Sub", &length_dim, &one, "a2r_len_minus_1");
    let two_len = b.binary_new("Mul", &two, &length_dim, "a2r_two_len");
    let length_squared = b.binary_new("Mul", &length_dim, &length_dim, "a2r_len_squared");
    let len_times_lm1 = b.binary_new(
        "Mul",
        &length_dim,
        &length_minus1,
        "a2r_len_times_len_minus_1",
    );
    let flat_size = b.binary_new("Add", &length_squared, &len_times_lm1, "a2r_flat_size");

    // Pad last dim by L-1 → [B, H, L, 2L-1]
    let pad1_shape = b.unique("a2r_pad1_shape");
    b.concat(
        &[
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &length_minus1,
        ],
        &pad1_shape,
        0,
    );
    let pad_val1 = {
        let n = b.unique("a2r_pad_val1");
        b.add_scalar(&n, 0.0)
    };
    let padded1 = b.unique("a2r_padded1");
    b.pad_constant(abs_weights, &pad1_shape, &pad_val1, &padded1);

    // Flatten → [B, H, L² + L(L-1)]
    let flat_shape = b.unique("a2r_flat_shape");
    b.concat(&[batch_dim, heads_dim, &flat_size], &flat_shape, 0);
    let flattened = b.unique("a2r_flattened");
    b.reshape(&padded1, &flat_shape, &flattened);

    // Pad start of last dim by L
    let pad2_shape = b.unique("a2r_pad2_shape");
    b.concat(
        &[&zero, &zero, &length_dim, &zero, &zero, &zero],
        &pad2_shape,
        0,
    );
    let pad_val2 = {
        let n = b.unique("a2r_pad_val2");
        b.add_scalar(&n, 0.0)
    };
    let padded2 = b.unique("a2r_padded2");
    b.pad_constant(&flattened, &pad2_shape, &pad_val2, &padded2);

    // Reshape → [B, H, L, 2L]
    let view_shape = b.unique("a2r_view_shape");
    b.concat(
        &[batch_dim, heads_dim, &length_dim, &two_len],
        &view_shape,
        0,
    );
    let reshaped = b.unique("a2r_reshaped");
    b.reshape(&padded2, &view_shape, &reshaped);

    // [:, :, :, 1:]
    let start_dim3 = {
        let n = b.unique("a2r_start");
        b.add_i64(&n, vec![1], vec![1])
    };
    let axes_dim3 = {
        let n = b.unique("a2r_axes");
        b.add_i64(&n, vec![3], vec![1])
    };
    let steps_dim3 = {
        let n = b.unique("a2r_steps");
        b.add_i64(&n, vec![1], vec![1])
    };
    b.slice(
        &reshaped,
        &start_dim3,
        &two_len,
        &axes_dim3,
        &steps_dim3,
        out,
    );
}

// =============================================================================
// Multi-head attention
// =============================================================================

fn build_multi_head_attention(
    b: &mut GraphBuilder<'_>,
    config: &RvcConfig,
    x: &str,
    mask: &str,
    prefix: &str,
) -> Result<String> {
    let n_heads = config.n_heads;
    let channels = config.hidden_channels;
    if n_heads == 0 || !channels.is_multiple_of(n_heads) {
        return Err(anyhow!(
            "hidden_channels {channels} not divisible by n_heads {n_heads}"
        ));
    }
    let k_channels = channels / n_heads;

    // Relative positional encoding tables (RVC always has them).
    let has_relative_pos = b.has_weight(&format!("{prefix}emb_rel_k"));
    let mut window_size = 0usize;
    let mut emb_rel_k = String::new();
    let mut emb_rel_v = String::new();
    if has_relative_pos {
        emb_rel_k = b.add_weight(&format!("{prefix}emb_rel_k"))?;
        emb_rel_v = b.add_weight(&format!("{prefix}emb_rel_v"))?;
        // [n_heads_rel, 2*window+1, k_channels]
        let emb_shape = b.weight_shape(&format!("{prefix}emb_rel_k"))?;
        window_size = (emb_shape
            .get(1)
            .copied()
            .ok_or_else(|| anyhow!("emb_rel_k has no position dimension"))?
            - 1)
            / 2;
    }

    // Q, K, V, O projection weights (1×1 convs)
    let conv_q_w = b.add_weight(&format!("{prefix}conv_q.weight"))?;
    let conv_q_b = b.add_weight(&format!("{prefix}conv_q.bias"))?;
    let conv_k_w = b.add_weight(&format!("{prefix}conv_k.weight"))?;
    let conv_k_b = b.add_weight(&format!("{prefix}conv_k.bias"))?;
    let conv_v_w = b.add_weight(&format!("{prefix}conv_v.weight"))?;
    let conv_v_b = b.add_weight(&format!("{prefix}conv_v.bias"))?;
    let conv_o_w = b.add_weight(&format!("{prefix}conv_o.weight"))?;
    let conv_o_b = b.add_weight(&format!("{prefix}conv_o.bias"))?;

    let q_proj = b.unique("q_proj");
    let k_proj = b.unique("k_proj");
    let v_proj = b.unique("v_proj");
    b.conv1d(x, &conv_q_w, Some(&conv_q_b), &q_proj, 1, 1, 0, 1);
    b.conv1d(x, &conv_k_w, Some(&conv_k_b), &k_proj, 1, 1, 0, 1);
    b.conv1d(x, &conv_v_w, Some(&conv_v_b), &v_proj, 1, 1, 0, 1);

    // Dynamic batch/time dims from the projected query shape
    let q_shape = b.unique("q_shape");
    b.unary("Shape", &q_proj, &q_shape);
    let batch_idx = {
        let n = b.unique("batch_idx");
        b.add_i64(&n, vec![0], vec![1])
    };
    let time_idx = {
        let n = b.unique("time_idx");
        b.add_i64(&n, vec![2], vec![1])
    };
    let batch_dim = b.unique("batch_dim");
    let time_dim = b.unique("time_dim");
    b.gather(&q_shape, &batch_idx, &batch_dim, 0);
    b.gather(&q_shape, &time_idx, &time_dim, 0);

    // Reshape [B, C, T] -> [B, heads, k, T], transpose to [B, heads, T, k]
    let n_heads_const = {
        let n = b.unique("n_heads");
        b.add_i64(&n, vec![n_heads as i64], vec![1])
    };
    let k_channels_const = {
        let n = b.unique("k_channels");
        b.add_i64(&n, vec![k_channels as i64], vec![1])
    };
    let reshape_shape = b.unique("reshape_shape");
    b.concat(
        &[&batch_dim, &n_heads_const, &k_channels_const, &time_dim],
        &reshape_shape,
        0,
    );

    let q_reshaped = b.unique("q_reshaped");
    b.reshape(&q_proj, &reshape_shape, &q_reshaped);
    let q_heads = b.unique("q_heads");
    b.transpose(&q_reshaped, &q_heads, vec![0, 1, 3, 2]);

    let k_reshaped = b.unique("k_reshaped");
    b.reshape(&k_proj, &reshape_shape, &k_reshaped);
    let k_heads = b.unique("k_heads");
    b.transpose(&k_reshaped, &k_heads, vec![0, 1, 3, 2]);

    let v_reshaped = b.unique("v_reshaped");
    b.reshape(&v_proj, &reshape_shape, &v_reshaped);
    let v_heads = b.unique("v_heads");
    b.transpose(&v_reshaped, &v_heads, vec![0, 1, 3, 2]);

    // scores = (q / sqrt(k)) @ k^T
    let scale_factor = {
        let n = b.unique("attn_scale");
        b.add_scalar(&n, 1.0 / (k_channels as f32).sqrt())
    };
    let q_scaled = b.binary_new("Mul", &q_heads, &scale_factor, "q_scaled");

    let k_t = b.unique("k_transposed");
    b.transpose(&k_heads, &k_t, vec![0, 1, 3, 2]);

    let mut scores = b.binary_new("MatMul", &q_scaled, &k_t, "attn_scores");

    // scores += relative_scores(query, T)
    if has_relative_pos {
        // (The TS computes rel_batch/heads gathers here but passes the
        // constants below; kept as-is for parity.)
        let rel_batch_dim = b.unique("rel_batch_dim");
        let rel_heads_dim = b.unique("rel_heads_dim");
        let heads_idx = {
            let n = b.unique("heads_idx");
            b.add_i64(&n, vec![1], vec![1])
        };
        b.gather(&q_shape, &batch_idx, &rel_batch_dim, 0);
        b.gather(&q_shape, &heads_idx, &rel_heads_dim, 0);

        let rel_emb_k = b.unique("rel_emb_k");
        build_get_relative_embeddings(b, &emb_rel_k, window_size, &time_dim, &rel_emb_k);
        let rel_logits = b.unique("rel_logits");
        build_matmul_with_relative_keys(b, &q_scaled, &rel_emb_k, &rel_logits);
        let rel_scores = b.unique("rel_scores");
        build_relative_to_absolute_position(
            b,
            &rel_logits,
            &rel_batch_dim,
            &n_heads_const,
            &rel_scores,
        );

        scores = b.binary_new("Add", &scores, &rel_scores, "scores_with_rel");
    }

    // scores.masked_fill(mask == 0, -1e4)
    let zero = {
        let n = b.unique("zero");
        b.add_scalar(&n, 0.0)
    };
    let mask_is_zero = b.binary_new("Equal", mask, &zero, "mask_is_zero");
    let neg_inf = {
        let n = b.unique("neg_inf");
        b.add_scalar(&n, -10000.0)
    };
    let scores_masked = b.unique("scores_masked");
    b.where_(&mask_is_zero, &neg_inf, &scores, &scores_masked);

    let attn_weights = b.unique("attn_weights");
    b.softmax(&scores_masked, &attn_weights, -1);

    let mut attn_output = b.binary_new("MatMul", &attn_weights, &v_heads, "attn_out");

    // output += relative_values(p_attn, T)
    if has_relative_pos {
        let rel_weights = b.unique("rel_weights");
        build_absolute_to_relative_position(
            b,
            &attn_weights,
            &batch_dim,
            &n_heads_const,
            &rel_weights,
        );
        let rel_emb_v = b.unique("rel_emb_v");
        build_get_relative_embeddings(b, &emb_rel_v, window_size, &time_dim, &rel_emb_v);
        let rel_values = b.unique("rel_values");
        build_matmul_with_relative_values(b, &rel_weights, &rel_emb_v, &rel_values);

        attn_output = b.binary_new("Add", &attn_output, &rel_values, "attn_with_rel");
    }

    // [B, heads, T, k] -> [B, heads, k, T] -> [B, C, T]
    let attn_out_t = b.unique("attn_out_transposed");
    b.transpose(&attn_output, &attn_out_t, vec![0, 1, 3, 2]);
    let channels_const = {
        let n = b.unique("channels");
        b.add_i64(&n, vec![channels as i64], vec![1])
    };
    let out_shape = b.unique("out_shape");
    b.concat(&[&batch_dim, &channels_const, &time_dim], &out_shape, 0);
    let attn_out_reshaped = b.unique("attn_out_reshaped");
    b.reshape(&attn_out_t, &out_shape, &attn_out_reshaped);

    let output = b.unique("mha_output");
    b.conv1d(
        &attn_out_reshaped,
        &conv_o_w,
        Some(&conv_o_b),
        &output,
        1,
        1,
        0,
        1,
    );
    Ok(output)
}

// =============================================================================
// FFN
// =============================================================================

fn build_ffn(
    b: &mut GraphBuilder<'_>,
    config: &RvcConfig,
    input: &str,
    mask: &str,
    prefix: &str,
) -> Result<String> {
    let conv1_w = b.add_weight(&format!("{prefix}conv_1.weight"))?;
    let conv1_b = b.add_weight(&format!("{prefix}conv_1.bias"))?;
    let conv2_w = b.add_weight(&format!("{prefix}conv_2.weight"))?;
    let conv2_b = b.add_weight(&format!("{prefix}conv_2.bias"))?;

    let kernel_size = config.kernel_size;
    let padding = (kernel_size - 1) / 2;

    let x_masked1 = b.binary_new("Mul", input, mask, "ffn_x_masked1");

    let h = b.unique("ffn_h");
    b.conv1d(
        &x_masked1,
        &conv1_w,
        Some(&conv1_b),
        &h,
        kernel_size,
        1,
        padding,
        1,
    );

    let h_act = b.unary_new("Relu", &h, "ffn_h_act");

    let h_masked = b.binary_new("Mul", &h_act, mask, "ffn_h_masked");

    let output = b.unique("ffn_output");
    b.conv1d(
        &h_masked,
        &conv2_w,
        Some(&conv2_b),
        &output,
        kernel_size,
        1,
        padding,
        1,
    );

    Ok(b.binary_new("Mul", &output, mask, "ffn_output_masked"))
}
