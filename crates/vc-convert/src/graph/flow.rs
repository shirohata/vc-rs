//! Normalizing flow (ResidualCouplingBlock) with WaveNet coupling layers.
//!
//! Ported from rvc-onnx-web (https://github.com/visgotti/rvc-onnx-web),
//! MIT License — `buildResidualCouplingBlock` / `buildResidualCouplingLayer`
//! / `buildWaveNet` / `buildFusedAddTanhSigmoidMultiply` in
//! `src/synthesizer-builder.ts`. Python comments refer to RVC's
//! `residuals.py` / `commons.py`, same as the TS source.

use anyhow::Result;

use crate::checkpoint::RvcConfig;

use super::weight_norm::{self, has_weight_norm};
use super::GraphBuilder;

/// Fused `tanh(a[:, :n, :]) * sigmoid(a[:, n:, :])` where `a = input_a + g_l`.
fn build_fused_add_tanh_sigmoid_multiply(
    b: &mut GraphBuilder<'_>,
    input_a: &str,
    input_b: Option<&str>,
    n_channels: usize,
    output: &str,
) {
    let in_act = match input_b {
        Some(input_b) => b.binary_new("Add", input_a, input_b, "in_act"),
        None => input_a.to_owned(),
    };

    let split_sizes = {
        let n = b.unique("split_sizes");
        b.add_i64(&n, vec![n_channels as i64, n_channels as i64], vec![2])
    };
    let tanh_input = b.unique("tanh_input");
    let sigmoid_input = b.unique("sigmoid_input");
    b.split_sizes(&in_act, &split_sizes, &[&tanh_input, &sigmoid_input], 1);

    let t_act = b.unary_new("Tanh", &tanh_input, "t_act");
    let s_act = b.unary_new("Sigmoid", &sigmoid_input, "s_act");
    b.binary("Mul", &t_act, &s_act, output);
}

/// WaveNet residual stack (`modules.WN`): dilated in_layers feeding
/// gated activations, res/skip 1×1 convs, global conditioning via a single
/// cond_layer sliced per layer.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the TS buildWaveNet signature"
)]
fn build_wavenet(
    b: &mut GraphBuilder<'_>,
    input: &str,
    mask: &str,
    g: Option<&str>,
    prefix: &str,
    hidden_channels: usize,
    kernel_size: usize,
    dilation_rate: usize,
    n_layers: usize,
) -> Result<String> {
    let mut x = input.to_owned();

    // output = zeros_like(x), built as x * 0 so the shape follows x.
    let zero_const = {
        let n = b.unique("wavenet_zero");
        b.add_scalar(&n, 0.0)
    };
    let mut output = b.binary_new("Mul", input, &zero_const, "wavenet_output_init");

    // g = cond_layer(g): [B, 2*hidden*n_layers, 1]
    let cond_prefix = format!("{prefix}cond_layer");
    let mut g_conditioned: Option<String> = None;
    if let Some(g) = g {
        if b.has_weight(&format!("{cond_prefix}.weight")) || has_weight_norm(b, &cond_prefix) {
            let cond_weight = weight_norm::conv_weight(b, &cond_prefix, "cond_layer_weight")?;
            let cond_bias = if b.has_weight(&format!("{cond_prefix}.bias")) {
                Some(b.add_weight(&format!("{cond_prefix}.bias"))?)
            } else {
                None
            };
            let out = b.unique("g_conditioned");
            b.conv1d(g, &cond_weight, cond_bias.as_deref(), &out, 1, 1, 0, 1);
            g_conditioned = Some(out);
        }
    }

    for i in 0..n_layers {
        let dilation = dilation_rate.pow(i as u32);
        let padding = (kernel_size * dilation - dilation) / 2;

        // x_in = in_layers[i](x): [B, 2*hidden, T]
        let in_prefix = format!("{prefix}in_layers.{i}");
        let in_weight = weight_norm::conv_weight(b, &in_prefix, &format!("in_layer_{i}_weight"))?;
        let in_bias = if b.has_weight(&format!("{in_prefix}.bias")) {
            Some(b.add_weight(&format!("{in_prefix}.bias"))?)
        } else {
            None
        };
        let x_in = b.unique(&format!("x_in_{i}"));
        b.conv1d(
            &x,
            &in_weight,
            in_bias.as_deref(),
            &x_in,
            kernel_size,
            1,
            padding,
            dilation,
        );

        // g_l = g[:, i*2*hidden : (i+1)*2*hidden, :]
        let g_l = if let Some(g_conditioned) = &g_conditioned {
            let start_idx = (i * 2 * hidden_channels) as i64;
            let end_idx = ((i + 1) * 2 * hidden_channels) as i64;
            let start = {
                let n = b.unique(&format!("slice_start_{i}"));
                b.add_i64(&n, vec![0, start_idx, 0], vec![3])
            };
            // INT32_MAX for the unbounded axes, matching the TS
            let end = {
                let n = b.unique(&format!("slice_end_{i}"));
                b.add_i64(&n, vec![2_147_483_647, end_idx, 2_147_483_647], vec![3])
            };
            let axes = {
                let n = b.unique(&format!("slice_axes_{i}"));
                b.add_i64(&n, vec![0, 1, 2], vec![3])
            };
            let steps = {
                let n = b.unique(&format!("slice_steps_{i}"));
                b.add_i64(&n, vec![1, 1, 1], vec![3])
            };
            let g_l = b.unique(&format!("g_l_{i}"));
            b.slice(g_conditioned, &start, &end, &axes, &steps, &g_l);
            Some(g_l)
        } else {
            None
        };

        let acts = b.unique(&format!("acts_{i}"));
        build_fused_add_tanh_sigmoid_multiply(b, &x_in, g_l.as_deref(), hidden_channels, &acts);

        // res_skip_layers[i]: 2*hidden out except the last layer (hidden).
        let is_last_layer = i == n_layers - 1;
        let res_skip_prefix = format!("{prefix}res_skip_layers.{i}");
        let res_skip_weight =
            weight_norm::conv_weight(b, &res_skip_prefix, &format!("res_skip_{i}_weight"))?;
        let res_skip_bias = if b.has_weight(&format!("{res_skip_prefix}.bias")) {
            Some(b.add_weight(&format!("{res_skip_prefix}.bias"))?)
        } else {
            None
        };
        let res_skip_acts = b.unique(&format!("res_skip_acts_{i}"));
        b.conv1d(
            &acts,
            &res_skip_weight,
            res_skip_bias.as_deref(),
            &res_skip_acts,
            1,
            1,
            0,
            1,
        );

        if !is_last_layer {
            let split_sizes = {
                let n = b.unique(&format!("split_sizes_{i}"));
                b.add_i64(
                    &n,
                    vec![hidden_channels as i64, hidden_channels as i64],
                    vec![2],
                )
            };
            let res_acts = b.unique(&format!("res_acts_{i}"));
            let skip_acts = b.unique(&format!("skip_acts_{i}"));
            b.split_sizes(&res_skip_acts, &split_sizes, &[&res_acts, &skip_acts], 1);

            // x = (x + res_acts) * x_mask
            let x_plus_res = b.binary_new("Add", &x, &res_acts, &format!("x_plus_res_{i}"));
            x = b.binary_new("Mul", &x_plus_res, mask, &format!("x_masked_{i}"));

            output = b.binary_new("Add", &output, &skip_acts, &format!("output_{i}"));
        } else {
            output = b.binary_new("Add", &output, &res_skip_acts, "output_final");
        }
    }

    Ok(b.binary_new("Mul", &output, mask, "wavenet_final"))
}

/// One `ResidualCouplingLayer` (mean-only): split, WaveNet on x0, shift x1.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the TS builder signature"
)]
fn build_residual_coupling_layer(
    b: &mut GraphBuilder<'_>,
    config: &RvcConfig,
    input: &str,
    mask: &str,
    g: &str,
    reverse: bool,
    prefix: &str,
) -> Result<String> {
    let channels = config.inter_channels;
    let half_channels = channels / 2;
    let hidden_channels = config.hidden_channels;

    // x0, x1 = split(x, [half, half], dim=1)
    let split_sizes = {
        let n = b.unique("split_sizes");
        b.add_i64(
            &n,
            vec![half_channels as i64, half_channels as i64],
            vec![2],
        )
    };
    let x0 = b.unique("x0");
    let x1 = b.unique("x1");
    b.split_sizes(input, &split_sizes, &[&x0, &x1], 1);

    // h = pre(x0) * x_mask
    let pre_weight = b.add_weight(&format!("{prefix}pre.weight"))?;
    let pre_bias = b.add_weight(&format!("{prefix}pre.bias"))?;
    let h_pre = b.unique("h_pre");
    b.conv1d(&x0, &pre_weight, Some(&pre_bias), &h_pre, 1, 1, 0, 1);
    let h_pre_masked = b.binary_new("Mul", &h_pre, mask, "h_pre_masked");

    // WaveNet layer count from the weights (like the TS detection loop).
    let mut wavenet_layers = 0;
    for i in 0..20 {
        let has_layer = b.has_weight(&format!("{prefix}enc.in_layers.{i}.bias"))
            || b.has_weight(&format!("{prefix}enc.in_layers.{i}.weight"))
            || has_weight_norm(b, &format!("{prefix}enc.in_layers.{i}"));
        if has_layer {
            wavenet_layers = i + 1;
        } else {
            break;
        }
    }
    if wavenet_layers == 0 {
        wavenet_layers = 4;
    }

    // Kernel size from the first in_layer weight shape [out, in, k].
    let mut wavenet_kernel_size = 5;
    let first_layer = format!("{prefix}enc.in_layers.0");
    if let Some(params) = crate::checkpoint::weight_norm_for(b.weights, &first_layer) {
        if let Ok(shape) = b.weight_shape(&params.weight_v) {
            if shape.len() >= 3 {
                wavenet_kernel_size = shape[2];
            }
        }
    } else if let Ok(shape) = b.weight_shape(&format!("{first_layer}.weight")) {
        if shape.len() >= 3 {
            wavenet_kernel_size = shape[2];
        }
    }

    let h = build_wavenet(
        b,
        &h_pre_masked,
        mask,
        Some(g),
        &format!("{prefix}enc."),
        hidden_channels,
        wavenet_kernel_size,
        1, // dilation_rate is 1 in RVC coupling layers
        wavenet_layers,
    )?;

    // stats = post(h) * x_mask; mean_only=True so m = stats, logs = 0
    let post_weight = b.add_weight(&format!("{prefix}post.weight"))?;
    let post_bias = b.add_weight(&format!("{prefix}post.bias"))?;
    let stats_raw = b.unique("stats_raw");
    b.conv1d(&h, &post_weight, Some(&post_bias), &stats_raw, 1, 1, 0, 1);
    let m = b.binary_new("Mul", &stats_raw, mask, "stats");

    let x_out = if !reverse {
        // x1 = m + x1 * x_mask  (exp(logs) = 1)
        let x1_masked = b.binary_new("Mul", &x1, mask, "x1_masked");
        let x1_new = b.binary_new("Add", &m, &x1_masked, "x1_new");
        let out = b.unique("x_coupled");
        b.concat(&[&x0, &x1_new], &out, 1);
        out
    } else {
        // x1 = (x1 - m) * x_mask  (exp(-logs) = 1)
        let x1_sub_m = b.binary_new("Sub", &x1, &m, "x1_sub_m");
        let x1_new = b.binary_new("Mul", &x1_sub_m, mask, "x1_new");
        let out = b.unique("x_coupled");
        b.concat(&[&x0, &x1_new], &out, 1);
        out
    };
    Ok(x_out)
}

/// The full flow: coupling layers interleaved with channel Flips
/// (implemented as Gather with reversed indices), reversed order when
/// `reverse` (inference direction).
pub(crate) fn build_residual_coupling_block(
    b: &mut GraphBuilder<'_>,
    config: &RvcConfig,
    input: &str,
    mask: &str,
    g: &str,
    reverse: bool,
) -> Result<String> {
    let mut x = input.to_owned();
    // RVC always uses 4 coupling flows (not stored in the config array).
    let n_flows = 4usize;

    // Reversed channel indices for Flip = Gather(axis=1).
    let reversed: Vec<i64> = (0..config.inter_channels).rev().map(|i| i as i64).collect();
    let flip_indices = {
        let n = b.unique("flip_indices");
        b.add_i64(&n, reversed, vec![config.inter_channels])
    };

    if !reverse {
        for i in 0..n_flows {
            let coupling_prefix = format!("flow.flows.{}.", i * 2);
            x = build_residual_coupling_layer(b, config, &x, mask, g, false, &coupling_prefix)?;
            let x_flipped = b.unique("x_flipped");
            b.gather(&x, &flip_indices, &x_flipped, 1);
            x = x_flipped;
        }
    } else {
        for i in (0..n_flows).rev() {
            let x_flipped = b.unique("x_flipped_rev");
            b.gather(&x, &flip_indices, &x_flipped, 1);
            x = x_flipped;
            let coupling_prefix = format!("flow.flows.{}.", i * 2);
            x = build_residual_coupling_layer(b, config, &x, mask, g, true, &coupling_prefix)?;
        }
    }

    Ok(x)
}
