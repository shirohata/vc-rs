//! Applio decoders, pinned to IAHispano/Applio 5171e1dc (MIT).
//! Kaiser sinc kernel: torchaudio 2.7.1 (BSD-2-Clause). Full notices are in
//! this crate's THIRD_PARTY.md and accompanying LICENSE-* files.
//!
//! V2 exports pack independent Gaussian domains *per feature frame*. This keeps
//! all scales attached to the same absolute frame when the host crops or rolls
//! overlapping windows. Do not replace AdaIN noise with zeros or graph RNGs.

use anyhow::{bail, ensure, Context, Result};

use super::{decoder::DecoderOut, weight_norm, GraphBuilder};
use crate::checkpoint::{ParsedCheckpoint, RvcConfig};
use crate::onnx::{attr, Attr};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Vocoder {
    Hifi,
    Mrf,
    Refine,
}

impl Vocoder {
    pub(crate) fn of(c: &ParsedCheckpoint) -> Result<Self> {
        match c.vocoder.as_str() {
            "HiFi-GAN" => Ok(Self::Hifi),
            "MRF HiFi-GAN" => Ok(Self::Mrf),
            "RefineGAN" => Ok(Self::Refine),
            name => bail!("unsupported RVC vocoder: {name}"),
        }
    }
    pub(crate) fn phases(self) -> usize {
        if self == Self::Mrf {
            9
        } else {
            1
        }
    }
}

pub(crate) fn noise_hop(c: &ParsedCheckpoint, v: Vocoder) -> Result<usize> {
    let hop = c.config.frame_hop();
    if v != Vocoder::Refine {
        return Ok(hop * v.phases());
    }
    let mut count = hop;
    let mut rate = 1;
    for (i, u) in c.config.upsample_rates.iter().enumerate() {
        rate *= u;
        let channels = c
            .weight(&format!("dec.upsample_conv_blocks.{i}.input_conv.weight"))?
            .shape
            .first()
            .copied()
            .context("RefineGAN input_conv has no output channels")?;
        ensure!(
            channels > 0 && channels <= 4096,
            "invalid RefineGAN channel count"
        );
        count = count
            .checked_add(6 * channels * rate)
            .ok_or_else(|| anyhow::anyhow!("RefineGAN noise layout overflow"))?;
    }
    ensure!(
        count <= 1_000_000,
        "RefineGAN noise frame exceeds supported size"
    );
    Ok(count)
}

fn shape(b: &mut GraphBuilder<'_>, values: Vec<i64>) -> String {
    let n = b.unique("shape");
    let len = values.len();
    b.add_i64(&n, values, vec![len])
}
fn reshape(b: &mut GraphBuilder<'_>, x: &str, dims: Vec<i64>) -> String {
    let s = shape(b, dims);
    let y = b.unique("reshape");
    b.reshape(x, &s, &y);
    y
}
fn slice(b: &mut GraphBuilder<'_>, x: &str, axis: i64, start: i64, end: i64) -> String {
    let starts = shape(b, vec![start]);
    let ends = shape(b, vec![end]);
    let axes = shape(b, vec![axis]);
    let steps = shape(b, vec![1]);
    let y = b.unique("slice");
    b.slice(x, &starts, &ends, &axes, &steps, &y);
    y
}
fn resize(b: &mut GraphBuilder<'_>, x: &str, factor: usize, linear: bool) -> String {
    let n = b.unique("scales");
    let scales = b.add_f32(&n, vec![1., 1., factor as f32], vec![3]);
    let y = b.unique("resize");
    b.push(
        "Resize",
        &[x, "", &scales],
        &[&y],
        vec![
            attr(
                "mode",
                Attr::String(if linear { "linear" } else { "nearest" }.into()),
            ),
            attr(
                "coordinate_transformation_mode",
                Attr::String(if linear { "half_pixel" } else { "asymmetric" }.into()),
            ),
            attr("nearest_mode", Attr::String("floor".into())),
        ],
    );
    y
}
fn activate(b: &mut GraphBuilder<'_>, x: &str, slope: f32) -> String {
    let y = b.unique("leaky");
    b.leaky_relu(x, &y, slope);
    y
}
fn conv(
    b: &mut GraphBuilder<'_>,
    x: &str,
    base: &str,
    stride: usize,
    dilation: usize,
) -> Result<String> {
    let w = weight_norm::conv_weight(b, base, "conv_weight")?;
    let kernel = *b
        .initializers
        .iter()
        .find(|i| i.name == w)
        .unwrap()
        .tensor
        .shape
        .get(2)
        .context("decoder convolution weight must have rank 3")?;
    let bias = if b.has_weight(&format!("{base}.bias")) {
        Some(b.add_weight(&format!("{base}.bias"))?)
    } else {
        None
    };
    let y = b.unique("conv");
    let padding = if stride > 1 {
        (kernel - stride) / 2
    } else {
        (kernel - 1) * dilation / 2
    };
    b.conv1d(
        x,
        &w,
        bias.as_deref(),
        &y,
        kernel,
        stride,
        padding,
        dilation,
    );
    Ok(y)
}

struct Noise {
    packed: Option<String>,
    offset: usize,
    total: usize,
}
impl Noise {
    fn new(b: &mut GraphBuilder<'_>, total: usize, streaming: bool) -> Self {
        Self {
            packed: streaming.then(|| reshape(b, "nsf_noise", vec![1, -1, total as i64])),
            offset: 0,
            total,
        }
    }
    fn take(
        &mut self,
        b: &mut GraphBuilder<'_>,
        reference: &str,
        channels: usize,
        rate: usize,
        channel_last: bool,
    ) -> String {
        let width = channels * rate;
        let out = if let Some(packed) = &self.packed {
            let selected = slice(
                b,
                packed,
                2,
                self.offset as i64,
                (self.offset + width) as i64,
            );
            if channel_last {
                reshape(b, &selected, vec![1, -1, channels as i64])
            } else {
                // frame-major [T, samples_per_frame, C] -> [1,C,T*rate].
                let time = reshape(b, &selected, vec![1, -1, channels as i64]);
                let y = b.unique("noise_domain");
                b.transpose(&time, &y, vec![0, 2, 1]);
                y
            }
        } else {
            let y = b.unique("noise");
            b.random_normal_like(reference, &y);
            y
        };
        self.offset += width;
        out
    }
}

/// Full-sample accumulation is required for MRF and RefineGAN's interpolated F0.
/// The host carries every harmonic's phase, with independent initial offsets.
fn source(
    b: &mut GraphBuilder<'_>,
    config: &RvcConfig,
    v: Vocoder,
    streaming: bool,
    noise: &mut Noise,
) -> Result<(String, Option<String>)> {
    let hop = config.frame_hop();
    let channels = v.phases();
    let f0 = reshape(b, "pitchf", vec![1, 1, -1]);
    let f0 = resize(b, &f0, hop, v == Vocoder::Refine);
    let f0t = b.unique("f0_samples");
    b.transpose(&f0, &f0t, vec![0, 2, 1]);
    let n = b.unique("harmonics");
    let harmonics = b.add_f32(
        &n,
        (1..=channels).map(|i| i as f32).collect(),
        vec![1, 1, channels],
    );
    let harmonic_f0 = b.binary_new("Mul", &f0t, &harmonics, "harmonic_f0");
    let sr = b.add_scalar("source_sr", config.sr as f32);
    let delta = b.binary_new("Div", &harmonic_f0, &sr, "phase_delta_raw");
    let whole = b.unary_new("Floor", &delta, "delta_whole");
    let delta = b.binary_new("Sub", &delta, &whole, "phase_delta");
    let axis_name = b.unique("phase_axis");
    let axis_scalar = b.add_i64(&axis_name, vec![1], vec![]);
    let cumulative = b.unique("phase_cumulative");
    b.cumsum(&delta, &axis_scalar, &cumulative);
    // Match upstream's compensated accumulation: remove whole-cycle crossings
    // before the final sum. A raw sample-rate cumsum loses fractional phase on
    // long context windows, especially at the higher MRF harmonics.
    let whole = b.unary_new("Floor", &cumulative, "cumulative_whole");
    let turns = b.binary_new("Sub", &cumulative, &whole, "turns");
    let previous = slice(b, &turns, 1, 0, -1);
    let next = slice(b, &turns, 1, 1, i32::MAX as i64);
    let crossed = b.binary_new("Less", &next, &previous, "cycle_crossing");
    let crossings = b.unique("crossings");
    b.cast(&crossed, &crossings, crate::onnx::DataType::Float);
    let pads = shape(b, vec![0, 1, 0, 0, 0, 0]);
    let shift = b.unique("cycle_shift");
    b.push("Pad", &[&crossings, &pads], &[&shift], vec![]);
    let compensated = b.binary_new("Sub", &delta, &shift, "phase_compensated");
    let cumulative = if v == Vocoder::Refine {
        // ORT's float32 scan accumulates rounding across the entire window,
        // unlike Torch CPU's accumulator. A trained RefineGAN amplifies that
        // drift even for constant F0. Sum compensated increments within short
        // blocks, then scan block totals; all backends retain float32 operations.
        // Choose a divisor so arbitrary supported frame hops still reshape.
        let block = (1..=16).rev().find(|n| hop.is_multiple_of(*n)).unwrap() as i64;
        let frames = reshape(b, &compensated, vec![1, -1, block, channels as i64]);
        let local_axis_name = b.unique("local_phase_axis");
        let local_axis = b.add_i64(&local_axis_name, vec![2], vec![]);
        let local = b.unique("local_phase");
        b.cumsum(&frames, &local_axis, &local);
        let totals = slice(b, &local, 2, block - 1, block);
        let prefixes = b.unique("frame_phase");
        b.cumsum(&totals, &axis_scalar, &prefixes);
        let previous = slice(b, &prefixes, 1, 0, -1);
        let pads = shape(b, vec![0, 1, 0, 0, 0, 0, 0, 0]);
        let shifted = b.unique("previous_frame_phase");
        b.push("Pad", &[&previous, &pads], &[&shifted], vec![]);
        let summed = b.binary_new("Add", &local, &shifted, "summed_phase");
        reshape(b, &summed, vec![1, -1, channels as i64])
    } else {
        let cumulative = b.unique("phase_precise");
        b.cumsum(&compensated, &axis_scalar, &cumulative);
        cumulative
    };
    let phase = if streaming {
        b.binary_new("Add", &cumulative, "phase_in", "phase")
    } else if channels > 1 {
        let initial = b.unique("initial_phase");
        b.push(
            "RandomUniform",
            &[],
            &[&initial],
            vec![
                attr("shape", Attr::Ints(vec![1, 1, channels as i64])),
                attr("dtype", Attr::Int(1)),
            ],
        );
        let mask = b.add_f32(
            "initial_phase_mask",
            (0..channels)
                .map(|i| if i == 0 { 0. } else { 1. })
                .collect(),
            vec![1, 1, channels],
        );
        let initial = b.binary_new("Mul", &initial, &mask, "harmonic_initial_phase");
        b.binary_new("Add", &cumulative, &initial, "phase")
    } else {
        cumulative
    };
    let one = b.add_scalar("source_one", 1.);
    let whole = b.unary_new("Floor", &phase, "phase_whole");
    let wrapped = b.binary_new("Sub", &phase, &whole, "phase_wrapped");
    let tau = b.add_scalar("source_tau", std::f32::consts::TAU);
    let rad = b.binary_new("Mul", &wrapped, &tau, "phase_rad");
    let sine = b.unary_new("Sin", &rad, "sine");
    let amp = b.add_scalar("source_amplitude", 0.1);
    let sine = b.binary_new("Mul", &sine, &amp, "sine_scaled");
    let zero = b.add_scalar("source_zero", 0.);
    let uv = b.binary_new("Greater", &f0t, &zero, "voiced");
    let uvf = b.unique("voiced_f32");
    b.cast(&uv, &uvf, crate::onnx::DataType::Float);
    let std = b.add_scalar("source_std", 0.003);
    let unvoiced_std = b.add_scalar("source_unvoiced_std", 0.1 / 3.);
    let voiced_noise = b.binary_new("Mul", &uvf, &std, "voiced_noise_amp");
    let unvoiced = b.binary_new("Sub", &one, &uvf, "unvoiced");
    let unvoiced_noise = b.binary_new("Mul", &unvoiced, &unvoiced_std, "unvoiced_noise_amp");
    let noise_amp = b.binary_new("Add", &voiced_noise, &unvoiced_noise, "source_noise_amp");
    let random = noise.take(b, &sine, channels, hop, true);
    let random = b.binary_new("Mul", &random, &noise_amp, "source_noise");
    let sine = b.binary_new("Mul", &sine, &uvf, "voiced_sine");
    let sine = b.binary_new("Add", &sine, &random, "excitation");
    let base = if v == Vocoder::Mrf {
        "dec.m_source.l_linear"
    } else {
        "dec.m_source.merge.0"
    };
    let weight = format!("{base}.weight");
    ensure!(
        b.weight_shape(&weight)? == [1, channels],
        "invalid {base} harmonic count"
    );
    let bias = if b.has_weight(&format!("{base}.bias")) {
        Some(b.add_weight(&format!("{base}.bias"))?)
    } else {
        None
    };
    let merged = b.unique("source_merged");
    weight_norm::linear(b, &sine, &weight, bias.as_deref(), &merged)?;
    let merged = b.unary_new("Tanh", &merged, "source_tanh");
    let out = b.unique("source");
    b.transpose(&merged, &out, vec![0, 2, 1]);
    Ok((out, streaming.then_some(wrapped)))
}

pub(crate) fn build(
    b: &mut GraphBuilder<'_>,
    c: &ParsedCheckpoint,
    v: Vocoder,
    latent: &str,
    g: &str,
    streaming: bool,
) -> Result<DecoderOut> {
    let mut noise = Noise::new(b, noise_hop(c, v)?, streaming);
    let (source, phase_trace) = source(b, &c.config, v, streaming, &mut noise)?;
    let audio = if v == Vocoder::Mrf {
        mrf(b, &c.config, latent, g, &source)?
    } else {
        refine(b, &c.config, latent, g, &source, &mut noise)?
    };
    ensure!(
        noise.offset == noise.total,
        "incomplete decoder noise layout"
    );
    Ok(DecoderOut { audio, phase_trace })
}

fn mrf(
    b: &mut GraphBuilder<'_>,
    c: &RvcConfig,
    latent: &str,
    g: &str,
    source: &str,
) -> Result<String> {
    let mut x = conv(b, latent, "dec.conv_pre", 1, 1)?;
    let cond = conv(b, g, "dec.cond", 1, 1)?;
    x = b.binary_new("Add", &x, &cond, "conditioned");
    for (i, (&u, &k)) in c
        .upsample_rates
        .iter()
        .zip(&c.upsample_kernel_sizes)
        .enumerate()
    {
        let act = activate(b, &x, 0.1);
        let base = format!("dec.upsamples.{i}");
        let w = weight_norm::conv_weight(b, &base, "ups_weight")?;
        let bias = b.add_weight(&format!("{base}.bias"))?;
        let up = b.unique("up");
        let p = if u % 2 == 0 {
            (k - u) / 2
        } else {
            u / 2 + u % 2
        };
        b.conv_transpose1d(&act, &w, Some(&bias), &up, k, u, p, u % 2);
        let injected = conv(
            b,
            source,
            &format!("dec.noise_convs.{i}"),
            c.upsample_rates[i + 1..].iter().product(),
            1,
        )?;
        x = b.binary_new("Add", &up, &injected, "injected");
        let mut branches = Vec::new();
        for (j, dilations) in c.resblock_dilation_sizes.iter().enumerate() {
            let mut branch = x.clone();
            for (l, &d) in dilations.iter().enumerate() {
                let act = activate(b, &branch, 0.1);
                let y = conv(b, &act, &format!("dec.mrfs.{i}.{j}.layers.{l}.conv1"), 1, d)?;
                let act = activate(b, &y, 0.1);
                let y = conv(b, &act, &format!("dec.mrfs.{i}.{j}.layers.{l}.conv2"), 1, 1)?;
                branch = b.binary_new("Add", &branch, &y, "residual");
            }
            branches.push(branch);
        }
        x = average(b, &branches)?;
    }
    let act = activate(b, &x, 0.01);
    let y = conv(b, &act, "dec.conv_post", 1, 1)?;
    Ok(b.unary_new("Tanh", &y, "audio"))
}
fn average(b: &mut GraphBuilder<'_>, xs: &[String]) -> Result<String> {
    ensure!(!xs.is_empty(), "decoder has no residual branches");
    let mut y = xs[0].clone();
    for x in &xs[1..] {
        y = b.binary_new("Add", &y, x, "sum");
    }
    let n = b.unique("inverse_branches");
    let scale = b.add_scalar(&n, 1. / xs.len() as f32);
    Ok(b.binary_new("Mul", &y, &scale, "average"))
}

/// Integer downsampling equivalent to torchaudio's Kaiser sinc resampler.
/// The f0_size factor in upstream frequencies cancels through gcd. Compute the
/// kernel once at export, preserving torchaudio's asymmetric padding and phase.
fn sinc_downsample(b: &mut GraphBuilder<'_>, x: &str, factor: usize) -> String {
    // torchaudio bypasses its filter when the input/output rates are equal.
    if factor == 1 {
        return x.to_owned();
    }
    let rolloff = 0.947_593_716_739_959_6_f64;
    let beta = 14.769_656_459_379_492;
    let width = (64. * factor as f64 / rolloff).ceil() as usize;
    let kernel_len = 2 * width + factor;
    let denom = bessel_i0(beta);
    let kernel = (0..kernel_len)
        .map(|i| {
            let t = ((i as f64 - width as f64) / factor as f64 * rolloff).clamp(-64., 64.);
            let window = bessel_i0(beta * (1. - (t / 64.).powi(2)).max(0.).sqrt()) / denom;
            let a = t * std::f64::consts::PI;
            ((if a == 0. { 1. } else { a.sin() / a }) * window * rolloff / factor as f64) as f32
        })
        .collect();
    let n = b.unique("sinc_kernel");
    let kernel = b.add_f32(&n, kernel, vec![1, 1, kernel_len]);
    let input_shape = b.unique("sinc_input_shape");
    b.unary("Shape", x, &input_shape);
    let flat = reshape(b, x, vec![-1, 1, 0]);
    let pads = shape(b, vec![0, 0, width as i64, 0, 0, (width + factor) as i64]);
    let padded = b.unique("sinc_padded");
    b.push("Pad", &[&flat, &pads], &[&padded], vec![]);
    let filtered = b.unique("sinc_filtered");
    b.conv1d(&padded, &kernel, None, &filtered, kernel_len, factor, 0, 1);
    // Exact divisibility follows from generator frame lengths and stage ratios.
    // Slice off Conv's extra terminal sample before restoring [1,C,T/factor].
    let trimmed = slice(b, &filtered, 2, 0, -1);
    let channels_index = shape(b, vec![1]);
    let channels = b.unique("sinc_channels");
    b.gather(&input_shape, &channels_index, &channels, 0);
    let one = shape(b, vec![1]);
    let infer = shape(b, vec![-1]);
    let out_shape = b.unique("sinc_output_shape");
    b.concat(&[&one, &channels, &infer], &out_shape, 0);
    let out = b.unique("sinc_downsample");
    b.reshape(&trimmed, &out_shape, &out);
    out
}
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.;
    let mut term = 1.;
    for k in 1..100 {
        term *= x * x / (4. * f64::from(k * k));
        sum += term;
        if term < sum * 1e-16 {
            break;
        }
    }
    sum
}

fn refine(
    b: &mut GraphBuilder<'_>,
    c: &RvcConfig,
    latent: &str,
    g: &str,
    source: &str,
    noise: &mut Noise,
) -> Result<String> {
    let mut x = conv(b, source, "dec.pre_conv", 1, 1)?;
    let mut downs = Vec::new();
    for (i, &factor) in c.upsample_rates.iter().rev().enumerate() {
        x = activate(b, &x, 0.2);
        downs.push(x.clone());
        x = sinc_downsample(b, &x, factor);
        x = conv(b, &x, &format!("dec.downsample_blocks.{i}"), 1, 1)?;
    }
    let mel = conv(b, latent, "dec.mel_conv", 1, 1)?;
    let cond = conv(b, g, "dec.cond", 1, 1)?;
    let mel = b.binary_new("Add", &mel, &cond, "conditioned");
    let mut merged = b.unique("bottleneck");
    b.concat(&[&mel, &x], &merged, 1);
    let mut rate = 1;
    for (i, (&factor, down)) in c.upsample_rates.iter().zip(downs.iter().rev()).enumerate() {
        rate *= factor;
        let act = activate(b, &merged, 0.2);
        let up = resize(b, &act, factor, true);
        let joined = b.unique("skip");
        b.concat(&[&up, down], &joined, 1);
        let base = format!("dec.upsample_conv_blocks.{i}");
        let input = conv(b, &joined, &format!("{base}.input_conv"), 1, 1)?;
        let channels = b.weight_shape(&format!("{base}.input_conv.weight"))?[0];
        let mut branches = Vec::new();
        for j in 0..3 {
            let mut branch = adain(
                b,
                &input,
                &format!("{base}.blocks.{j}.0"),
                channels,
                rate,
                noise,
            )?;
            for (l, d) in [1, 3, 5].into_iter().enumerate() {
                let act = activate(b, &branch, 0.2);
                let y = conv(b, &act, &format!("{base}.blocks.{j}.1.convs1.{l}"), 1, d)?;
                let act = activate(b, &y, 0.2);
                let y = conv(b, &act, &format!("{base}.blocks.{j}.1.convs2.{l}"), 1, 1)?;
                branch = b.binary_new("Add", &branch, &y, "residual");
            }
            branches.push(adain(
                b,
                &branch,
                &format!("{base}.blocks.{j}.2"),
                channels,
                rate,
                noise,
            )?);
        }
        merged = average(b, &branches)?;
    }
    let act = activate(b, &merged, 0.2);
    let y = conv(b, &act, "dec.conv_post", 1, 1)?;
    Ok(b.unary_new("Tanh", &y, "audio"))
}
fn adain(
    b: &mut GraphBuilder<'_>,
    x: &str,
    base: &str,
    channels: usize,
    rate: usize,
    noise: &mut Noise,
) -> Result<String> {
    let gaussian = noise.take(b, x, channels, rate, false);
    let weight = b.add_weight(&format!("{base}.weight"))?;
    let weight = reshape(b, &weight, vec![1, channels as i64, 1]);
    let gaussian = b.binary_new("Mul", &gaussian, &weight, "adain_noise");
    let y = b.binary_new("Add", x, &gaussian, "adain");
    Ok(activate(b, &y, 0.2))
}
