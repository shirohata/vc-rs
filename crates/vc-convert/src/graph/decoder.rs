//! HiFiGAN / GeneratorNSF decoder: upsampling ConvTranspose stack with
//! multi-receptive-field ResBlocks and NSF source injection.
//!
//! Ported from rvc-onnx-web (https://github.com/visgotti/rvc-onnx-web),
//! MIT License — `buildHiFiGANDecoder` / `buildResBlock` /
//! `buildGeneratorNSF` in `src/synthesizer-builder.ts`.

use anyhow::{anyhow, Result};

use crate::checkpoint::RvcConfig;

use super::nsf::{self, DecoderOptions};
use super::{weight_norm, GraphBuilder};

pub(crate) struct DecoderOut {
    pub audio: String,
    pub phase_trace: Option<String>,
}

/// Entry point: dispatches to the NSF path when the checkpoint has the
/// source module and F0 is in play (always true for the supported models,
/// which `validate_checkpoint` enforces up front).
pub(crate) fn build_decoder(
    b: &mut GraphBuilder<'_>,
    config: &RvcConfig,
    input: &str,
    f0_input: Option<&str>,
    g: &str,
    use_f0: bool,
    options: &DecoderOptions<'_>,
) -> Result<DecoderOut> {
    let prefix = "dec.";
    let is_nsf = use_f0 && b.has_weight(&format!("{prefix}m_source.l_linear.weight"));

    if is_nsf {
        if let Some(f0) = f0_input {
            return build_generator_nsf(b, config, input, f0, g, prefix, options);
        }
    }

    // Plain HiFiGAN path (non-NSF). Kept for parity with the TS source even
    // though the supported v2+F0 checkpoints always take the NSF path.
    let conv_pre_w = b.add_weight(&format!("{prefix}conv_pre.weight"))?;
    let conv_pre_b = b.add_weight(&format!("{prefix}conv_pre.bias"))?;
    let mut x = b.unique("dec_x");
    b.conv1d(input, &conv_pre_w, Some(&conv_pre_b), &x, 7, 1, 3, 1);

    if b.has_weight(&format!("{prefix}cond.weight")) {
        let cond_w = b.add_weight(&format!("{prefix}cond.weight"))?;
        let cond_b = if b.has_weight(&format!("{prefix}cond.bias")) {
            Some(b.add_weight(&format!("{prefix}cond.bias"))?)
        } else {
            None
        };
        let g_cond = b.unique("g_cond");
        b.conv1d(g, &cond_w, cond_b.as_deref(), &g_cond, 1, 1, 0, 1);
        x = b.binary_new("Add", &x, &g_cond, "x_cond");
    }

    let num_kernels = config.resblock_kernel_sizes.len();
    for i in 0..config.upsample_rates.len() {
        let x_act = b.unique("dec_x_act");
        b.leaky_relu(&x, &x_act, 0.1);

        let up_weight =
            weight_norm::conv_weight(b, &format!("{prefix}ups.{i}"), &format!("ups_{i}_weight"))?;
        let up_bias = if b.has_weight(&format!("{prefix}ups.{i}.bias")) {
            Some(b.add_weight(&format!("{prefix}ups.{i}.bias"))?)
        } else {
            None
        };
        let (rate, kernel_size) = upsample_params(config, i)?;
        let (padding, output_padding) = upsample_padding(rate, kernel_size);
        let x_up = b.unique("dec_x_up");
        b.conv_transpose1d(
            &x_act,
            &up_weight,
            up_bias.as_deref(),
            &x_up,
            kernel_size,
            rate,
            padding,
            output_padding,
        );

        x = resblock_average(
            b,
            config,
            &x_up,
            prefix,
            i,
            num_kernels,
            "xs_acc",
            "dec_x_avg",
        )?;
    }

    let x_act_final = b.unique("dec_x_act_final");
    b.leaky_relu(&x, &x_act_final, 0.1);
    let conv_post_w = b.add_weight(&format!("{prefix}conv_post.weight"))?;
    let audio = b.unique("audio_raw");
    // conv_post has bias=False in RVC
    b.conv1d(&x_act_final, &conv_post_w, None, &audio, 7, 1, 3, 1);
    let audio_final = b.unary_new("Tanh", &audio, "audio_final");

    Ok(DecoderOut {
        audio: audio_final,
        phase_trace: None,
    })
}

/// GeneratorNSF: HiFiGAN plus the NSF source signal injected at every
/// upsample stage through the strided noise_convs.
fn build_generator_nsf(
    b: &mut GraphBuilder<'_>,
    config: &RvcConfig,
    input: &str,
    f0: &str,
    g: &str,
    prefix: &str,
    options: &DecoderOptions<'_>,
) -> Result<DecoderOut> {
    let total_upsample = config.frame_hop();

    let mut source = None;
    let mut phase_trace = None;
    if b.has_weight(&format!("{prefix}m_source.l_linear.weight")) {
        let out = nsf::build_source_module(
            b,
            f0,
            total_upsample,
            config.sr,
            &format!("{prefix}m_source."),
            options,
        )?;
        source = Some(out.source);
        phase_trace = out.phase_trace;
    }

    let conv_pre_w = b.add_weight(&format!("{prefix}conv_pre.weight"))?;
    let conv_pre_b = b.add_weight(&format!("{prefix}conv_pre.bias"))?;
    let mut x = b.unique("nsf_x");
    b.conv1d(input, &conv_pre_w, Some(&conv_pre_b), &x, 7, 1, 3, 1);

    if b.has_weight(&format!("{prefix}cond.weight")) {
        let cond_w = b.add_weight(&format!("{prefix}cond.weight"))?;
        let cond_b = if b.has_weight(&format!("{prefix}cond.bias")) {
            Some(b.add_weight(&format!("{prefix}cond.bias"))?)
        } else {
            None
        };
        let g_cond = b.unique("nsf_g_cond");
        b.conv1d(g, &cond_w, cond_b.as_deref(), &g_cond, 1, 1, 0, 1);
        x = b.binary_new("Add", &x, &g_cond, "nsf_x_cond");
    }

    let num_upsamples = config.upsample_rates.len();
    let num_kernels = config.resblock_kernel_sizes.len();

    for i in 0..num_upsamples {
        let x_act = b.unique("nsf_x_act");
        b.leaky_relu(&x, &x_act, 0.1);

        let up_weight = weight_norm::conv_weight(
            b,
            &format!("{prefix}ups.{i}"),
            &format!("nsf_ups_{i}_weight"),
        )?;
        let up_bias = if b.has_weight(&format!("{prefix}ups.{i}.bias")) {
            Some(b.add_weight(&format!("{prefix}ups.{i}.bias"))?)
        } else {
            None
        };
        let (rate, kernel_size) = upsample_params(config, i)?;
        let (padding, output_padding) = upsample_padding(rate, kernel_size);
        let x_up = b.unique("nsf_x_up");
        b.conv_transpose1d(
            &x_act,
            &up_weight,
            up_bias.as_deref(),
            &x_up,
            kernel_size,
            rate,
            padding,
            output_padding,
        );

        // Inject the NSF source through this stage's noise conv. Its stride
        // is the product of the remaining upsample rates so the source (at
        // audio rate) lands on this stage's time resolution.
        let noise_conv_weight = format!("{prefix}noise_convs.{i}.weight");
        if source.is_some() && b.has_weight(&noise_conv_weight) {
            let noise_w = b.add_weight(&noise_conv_weight)?;
            let noise_b = if b.has_weight(&format!("{prefix}noise_convs.{i}.bias")) {
                Some(b.add_weight(&format!("{prefix}noise_convs.{i}.bias"))?)
            } else {
                None
            };
            let noise_shape = b.weight_shape(&noise_conv_weight)?;
            let noise_kernel = *noise_shape
                .get(2)
                .ok_or_else(|| anyhow!("noise conv weight is not 3-D"))?;
            let noise_stride: usize = config.upsample_rates[i + 1..].iter().product();
            let noise_padding = if noise_stride == 1 {
                0
            } else {
                (noise_kernel - noise_stride) / 2
            };
            let source_name = source.as_deref().expect("source checked above");
            let source_conv = b.unique("nsf_source_conv");
            b.conv1d(
                source_name,
                &noise_w,
                noise_b.as_deref(),
                &source_conv,
                noise_kernel,
                noise_stride,
                noise_padding,
                1,
            );
            x = b.binary_new("Add", &x_up, &source_conv, "nsf_x_with_source");
        } else {
            x = x_up;
        }

        x = resblock_average(
            b,
            config,
            &x,
            prefix,
            i,
            num_kernels,
            "nsf_xs_acc",
            "nsf_x_avg",
        )?;
    }

    let x_act_final = b.unique("nsf_x_act_final");
    b.leaky_relu(&x, &x_act_final, 0.1);
    let conv_post_w = b.add_weight(&format!("{prefix}conv_post.weight"))?;
    let audio = b.unique("nsf_audio_raw");
    b.conv1d(&x_act_final, &conv_post_w, None, &audio, 7, 1, 3, 1);
    let audio_final = b.unary_new("Tanh", &audio, "nsf_audio_final");

    Ok(DecoderOut {
        audio: audio_final,
        phase_trace,
    })
}

fn upsample_params(config: &RvcConfig, i: usize) -> Result<(usize, usize)> {
    let rate = *config
        .upsample_rates
        .get(i)
        .ok_or_else(|| anyhow!("missing upsample rate {i}"))?;
    let kernel = *config
        .upsample_kernel_sizes
        .get(i)
        .ok_or_else(|| anyhow!("missing upsample kernel size {i}"))?;
    Ok((rate, kernel))
}

/// PyTorch: even rate → pad (k-u)/2; odd rate → pad u/2 + u%2 with
/// output_padding u%2.
fn upsample_padding(rate: usize, kernel_size: usize) -> (usize, usize) {
    if rate.is_multiple_of(2) {
        ((kernel_size - rate) / 2, 0)
    } else {
        (rate / 2 + rate % 2, rate % 2)
    }
}

/// Run this stage's `num_kernels` ResBlocks on `x` and average their sums.
#[allow(clippy::too_many_arguments, reason = "shared by both decoder variants")]
fn resblock_average(
    b: &mut GraphBuilder<'_>,
    config: &RvcConfig,
    x: &str,
    prefix: &str,
    stage: usize,
    num_kernels: usize,
    acc_prefix: &str,
    avg_prefix: &str,
) -> Result<String> {
    let mut xs: Option<String> = None;
    for j in 0..num_kernels {
        let res_idx = stage * num_kernels + j;
        let res_out = build_res_block(b, config, x, &format!("{prefix}resblocks.{res_idx}."), j)?;
        xs = Some(match xs {
            None => res_out,
            Some(acc) => b.binary_new("Add", &acc, &res_out, acc_prefix),
        });
    }
    let xs = xs.ok_or_else(|| anyhow!("no resblock kernels configured"))?;
    let inv = {
        let n = b.unique("num_kernels_inv");
        b.add_scalar(&n, 1.0 / num_kernels as f32)
    };
    Ok(b.binary_new("Mul", &xs, &inv, avg_prefix))
}

/// ResBlock1: pairs of dilated + undilated convs with residual connections.
fn build_res_block(
    b: &mut GraphBuilder<'_>,
    config: &RvcConfig,
    input: &str,
    prefix: &str,
    kernel_idx: usize,
) -> Result<String> {
    let mut x = input.to_owned();
    let kernel_size = config
        .resblock_kernel_sizes
        .get(kernel_idx)
        .copied()
        .unwrap_or(3);
    let default_dilations = vec![1, 3, 5];
    let dilations = config
        .resblock_dilation_sizes
        .get(kernel_idx)
        .unwrap_or(&default_dilations);

    for (i, &dilation) in dilations.iter().enumerate() {
        let residual = x.clone();

        let x_act1 = b.unique("resblock_act1");
        b.leaky_relu(&x, &x_act1, 0.1);

        let conv1_weight = weight_norm::conv_weight(
            b,
            &format!("{prefix}convs1.{i}"),
            &format!("resblock_conv1_{i}_weight"),
        )?;
        let conv1_bias = if b.has_weight(&format!("{prefix}convs1.{i}.bias")) {
            Some(b.add_weight(&format!("{prefix}convs1.{i}.bias"))?)
        } else {
            None
        };
        let padding = (kernel_size * dilation - dilation) / 2;
        let h = b.unique("resblock_h");
        b.conv1d(
            &x_act1,
            &conv1_weight,
            conv1_bias.as_deref(),
            &h,
            kernel_size,
            1,
            padding,
            dilation,
        );

        let h_act = b.unique("resblock_h_act");
        b.leaky_relu(&h, &h_act, 0.1);

        let conv2_weight = weight_norm::conv_weight(
            b,
            &format!("{prefix}convs2.{i}"),
            &format!("resblock_conv2_{i}_weight"),
        )?;
        let conv2_bias = if b.has_weight(&format!("{prefix}convs2.{i}.bias")) {
            Some(b.add_weight(&format!("{prefix}convs2.{i}.bias"))?)
        } else {
            None
        };
        let padding2 = (kernel_size - 1) / 2;
        let h_conv = b.unique("resblock_h_conv");
        b.conv1d(
            &h_act,
            &conv2_weight,
            conv2_bias.as_deref(),
            &h_conv,
            kernel_size,
            1,
            padding2,
            1,
        );

        x = b.binary_new("Add", &h_conv, &residual, "resblock_x_new");
    }

    Ok(x)
}
