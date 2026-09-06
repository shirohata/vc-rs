#![allow(dead_code)]

use anyhow::{anyhow, Result};
// Use rubato's re-export so the buffer adapter always implements the exact
// trait version accepted by the resampler across dependency upgrades.
use rubato::audioadapter_buffers::direct::SequentialSlice;
use rubato::{Fft, FixedSync, Resampler, WindowFunction};

const STREAM_RESAMPLE_CHUNK: usize = 480;
const STREAM_RESAMPLE_COMPACT_THRESHOLD: usize = STREAM_RESAMPLE_CHUNK * 8;

/// Samples in a `chunk_ms` window at `sample_rate`, with a hard 128-sample
/// floor.
///
/// The floor keeps every front-end's chunk above the minimum the feature/F0
/// extractors need at low sample rates or tiny chunk sizes; it is a shared
/// invariant, so the three realtime drivers (CLI/GUI worker, VST3 worker) call
/// this rather than each re-deriving the formula.
pub fn chunk_samples_for_rate(sample_rate: u32, chunk_ms: u32) -> usize {
    ((sample_rate as u64 * chunk_ms as u64) / 1000).max(128) as usize
}

pub fn i16_to_f32(input: &[i16]) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    i16_to_f32_into(input, &mut output);
    output
}

pub fn i16_to_f32_into(input: &[i16], output: &mut [f32]) {
    for (dst, &src) in output.iter_mut().zip(input) {
        *dst = src as f32 / 32768.0;
    }
}

pub fn u16_to_f32_into(input: &[u16], output: &mut [f32]) {
    for (dst, &src) in output.iter_mut().zip(input) {
        *dst = (src as f32 - 32768.0) / 32768.0;
    }
}

pub fn f32_to_i16(input: &[f32]) -> Vec<i16> {
    let mut output = vec![0; input.len()];
    f32_to_i16_into(input, &mut output);
    output
}

pub fn f32_to_i16_into(input: &[f32], output: &mut [i16]) {
    for (dst, &src) in output.iter_mut().zip(input) {
        *dst = (src.clamp(-1.0, 1.0) * 32767.0).round() as i16;
    }
}

pub fn f32_to_u16_into(input: &[f32], output: &mut [u16]) {
    for (dst, &src) in output.iter_mut().zip(input) {
        *dst = ((src.clamp(-1.0, 1.0) * 32767.0) + 32768.0).round() as u16;
    }
}

// 32-bit PCM (common on ASIO devices). Scale by 2^31 on decode and by i32::MAX on
// encode; the f64 multiply avoids f32 mantissa loss at the int32 scale, and the
// float->int cast saturates (so +1.0 lands on i32::MAX rather than wrapping).
pub fn i32_to_f32_into(input: &[i32], output: &mut [f32]) {
    for (dst, &src) in output.iter_mut().zip(input) {
        *dst = (src as f64 / 2_147_483_648.0) as f32;
    }
}

pub fn f32_to_i32_into(input: &[f32], output: &mut [i32]) {
    for (dst, &src) in output.iter_mut().zip(input) {
        *dst = (f64::from(src.clamp(-1.0, 1.0)) * 2_147_483_647.0).round() as i32;
    }
}

/// Averages interleaved `channels`-channel frames down to mono.
///
/// `output` receives one sample per frame; `input.len()` must be
/// `output.len() * channels`. `channels == 1` degenerates to a copy.
pub fn downmix_to_mono_into(input: &[f32], channels: usize, output: &mut [f32]) {
    debug_assert!(channels >= 1);
    debug_assert_eq!(input.len(), output.len() * channels);
    match channels {
        1 => output.copy_from_slice(input),
        2 => {
            for (dst, frame) in output.iter_mut().zip(input.chunks_exact(2)) {
                *dst = (frame[0] + frame[1]) * 0.5;
            }
        }
        4 => {
            for (dst, frame) in output.iter_mut().zip(input.chunks_exact(4)) {
                *dst = (frame[0] + frame[1] + frame[2] + frame[3]) * 0.25;
            }
        }
        _ => {
            let scale = 1.0 / channels as f32;
            for (dst, frame) in output.iter_mut().zip(input.chunks_exact(channels)) {
                *dst = frame.iter().sum::<f32>() * scale;
            }
        }
    }
}

/// Duplicates each mono sample across all `channels` slots of an interleaved
/// frame (the same upmix WASAPI's AUTOCONVERTPCM applied for mono sources).
///
/// `output.len()` must be `mono.len() * channels`.
pub fn upmix_mono_into<T: Copy>(mono: &[T], channels: usize, output: &mut [T]) {
    debug_assert!(channels >= 1);
    debug_assert_eq!(output.len(), mono.len() * channels);
    match channels {
        1 => output.copy_from_slice(mono),
        2 => {
            for (frame, &sample) in output.chunks_exact_mut(2).zip(mono) {
                frame[0] = sample;
                frame[1] = sample;
            }
        }
        4 => {
            for (frame, &sample) in output.chunks_exact_mut(4).zip(mono) {
                frame[0] = sample;
                frame[1] = sample;
                frame[2] = sample;
                frame[3] = sample;
            }
        }
        _ => {
            for (frame, &sample) in output.chunks_exact_mut(channels).zip(mono) {
                frame.fill(sample);
            }
        }
    }
}

pub fn rms(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }
    let sum = input.iter().map(|x| x * x).sum::<f32>();
    (sum / input.len() as f32).sqrt()
}

#[inline]
pub fn clamp_scale_in_place(input: &mut [f32], scale: f32) {
    if (scale - 1.0).abs() <= f32::EPSILON {
        for sample in input {
            *sample = sample.clamp(-1.0, 1.0);
        }
    } else {
        for sample in input {
            *sample = sample.clamp(-1.0, 1.0) * scale;
        }
    }
}

#[inline]
pub fn apply_gain_and_rms(input: &mut [f32], gain: f32) -> f32 {
    if input.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0;
    for sample in input.iter_mut() {
        let scaled = (*sample * gain).clamp(-1.0, 1.0);
        *sample = scaled;
        sum += scaled * scaled;
    }
    (sum / input.len() as f32).sqrt()
}

#[derive(Default)]
pub struct RmsMixScratch {
    input_rms: Vec<f32>,
    output_rms: Vec<f32>,
}

#[derive(Clone, Copy)]
enum RmsMixGainCurve {
    Linear,
    SquareRoot,
    Power(f32),
}

impl RmsMixGainCurve {
    #[inline]
    fn for_exponent(exponent: f32) -> Self {
        if exponent.to_bits() == 1.0f32.to_bits() {
            Self::Linear
        } else if exponent.to_bits() == 0.5f32.to_bits() {
            Self::SquareRoot
        } else {
            Self::Power(exponent)
        }
    }

    #[inline]
    fn gain(self, ratio: f32) -> f32 {
        match self {
            Self::Linear => ratio,
            Self::SquareRoot => ratio.sqrt(),
            Self::Power(exponent) => ratio.powf(exponent),
        }
    }
}

pub fn compute_rms_envelope(input: &[f32], sample_rate: usize) -> Vec<f32> {
    let mut envelope = Vec::new();
    compute_rms_envelope_into(input, sample_rate, &mut envelope);
    envelope
}

pub fn compute_rms_envelope_into(input: &[f32], sample_rate: usize, output: &mut Vec<f32>) {
    output.clear();
    if input.is_empty() {
        return;
    }

    let hop_len = (sample_rate / 100).max(1);
    let frame_len = hop_len.saturating_mul(4).max(1);
    let frame_count = input.len().div_ceil(hop_len);
    output.reserve(frame_count);

    for frame in 0..frame_count {
        let start = frame * hop_len;
        let end = start.saturating_add(frame_len).min(input.len());
        let mut sum = 0.0;
        for &sample in &input[start..end] {
            let square = sample * sample;
            if square.is_finite() {
                sum += square;
            }
        }
        // Match the RVC WebUI-style frame grid by keeping 0, hop, 2*hop...
        // starts and treating missing tail samples as zero padding. Do not
        // change this to a short-frame denominator without retuning tests and
        // comparing the SOLA-before envelope behavior.
        output.push((sum / frame_len as f32).sqrt());
    }
}

pub fn linear_resample_envelope(points: &[f32], output_len: usize) -> Vec<f32> {
    if output_len == 0 {
        return Vec::new();
    }
    if points.is_empty() {
        return vec![0.0; output_len];
    }
    if points.len() == 1 || output_len == 1 {
        return vec![finite_nonnegative(points[0]); output_len];
    }

    let last_point = points.len() - 1;
    let last_output = output_len - 1;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let position = i as f32 * last_point as f32 / last_output as f32;
        let left = position.floor() as usize;
        let right = (left + 1).min(last_point);
        let frac = position - left as f32;
        let left_value = finite_nonnegative(points[left]);
        let right_value = finite_nonnegative(points[right]);
        output.push(left_value + (right_value - left_value) * frac);
    }

    output
}

pub fn apply_rms_mix(
    input_reference: &[f32],
    output: &mut [f32],
    sample_rate: usize,
    rms_mix_rate: f32,
) {
    let mut scratch = RmsMixScratch::default();
    apply_rms_mix_with_scratch(
        input_reference,
        output,
        sample_rate,
        rms_mix_rate,
        &mut scratch,
    );
}

pub fn apply_rms_mix_with_scratch(
    input_reference: &[f32],
    output: &mut [f32],
    sample_rate: usize,
    rms_mix_rate: f32,
    scratch: &mut RmsMixScratch,
) {
    if input_reference.is_empty() || output.is_empty() {
        return;
    }
    let rms_mix_rate = if rms_mix_rate.is_finite() {
        rms_mix_rate.clamp(0.0, 1.0)
    } else {
        1.0
    };
    if (rms_mix_rate - 1.0).abs() <= f32::EPSILON {
        return;
    }

    // Keep only the frame-rate RMS envelopes in reusable scratch. Expanding
    // them to per-sample Vecs here used to allocate two output-length buffers
    // on every model chunk.
    compute_rms_envelope_into(input_reference, sample_rate, &mut scratch.input_rms);
    compute_rms_envelope_into(output, sample_rate, &mut scratch.output_rms);
    let exponent = 1.0 - rms_mix_rate;
    // Common user settings map to exact exponent values. Keep those off the
    // slower powf path without approximating nearby rates, which would change
    // the level curve.
    let gain_curve = RmsMixGainCurve::for_exponent(exponent);
    let output_len = output.len();

    for (index, sample) in output.iter_mut().enumerate() {
        if !sample.is_finite() {
            *sample = 0.0;
            continue;
        }
        let in_rms = linear_resample_envelope_at(&scratch.input_rms, index, output_len);
        let out_rms = linear_resample_envelope_at(&scratch.output_rms, index, output_len);
        let out_rms = finite_nonnegative(out_rms).max(1e-3);
        let ratio = finite_nonnegative(in_rms) / out_rms;
        let gain = gain_curve.gain(ratio);
        let mixed = *sample * gain;
        *sample = if mixed.is_finite() { mixed } else { 0.0 };
    }
}

pub fn resample_mono(input: &[f32], from_hz: usize, to_hz: usize) -> Result<Vec<f32>> {
    if from_hz == to_hz {
        return Ok(input.to_vec());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let requested_chunk = 1024;
    // Preserve rubato 3's FFT partitioning and anti-aliasing filter. The newer
    // `new` auto-selects sub-chunks, which can change delay and the passband.
    let mut resampler = Fft::<f32>::new_custom(
        from_hz,
        to_hz,
        requested_chunk,
        1,
        1,
        WindowFunction::BlackmanHarris2,
        FixedSync::Both,
    )?;
    let out_frames = resampler.process_all_needed_output_len(input.len()).max(1);
    let input_adapter = SequentialSlice::new(input, 1, input.len())?;
    let mut output = vec![0.0; out_frames];
    let mut output_adapter = SequentialSlice::new_mut(&mut output, 1, out_frames)?;
    let (_used_in, produced_out) = resampler.process_all_into_buffer(
        &input_adapter,
        &mut output_adapter,
        input.len(),
        None,
    )?;

    output.truncate(produced_out);
    Ok(output)
}

pub struct StreamingResampleMono {
    from_hz: usize,
    to_hz: usize,
    resampler: Option<Fft<f32>>,
    pending_input: Vec<f32>,
    pending_input_start: usize,
    output_scratch: Vec<f32>,
    discard_output: usize,
}

impl StreamingResampleMono {
    pub fn new(from_hz: usize, to_hz: usize) -> Result<Self> {
        let resampler = if from_hz == to_hz {
            None
        } else {
            // Keep this filter and single sub-chunk aligned with resample_mono;
            // changing them requires checking waveforms and startup trimming.
            Some(Fft::<f32>::new_custom(
                from_hz,
                to_hz,
                STREAM_RESAMPLE_CHUNK,
                1,
                1,
                WindowFunction::BlackmanHarris2,
                FixedSync::Input,
            )?)
        };
        let discard_output = resampler
            .as_ref()
            .map(|resampler| resampler.output_delay())
            .unwrap_or(0);
        Ok(Self {
            from_hz,
            to_hz,
            resampler,
            pending_input: Vec::new(),
            pending_input_start: 0,
            output_scratch: Vec::new(),
            discard_output,
        })
    }

    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let mut output = Vec::new();
        self.process_into(input, &mut output)?;
        Ok(output)
    }

    pub fn process_into(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<()> {
        if self.from_hz == self.to_hz {
            output.extend_from_slice(input);
            return Ok(());
        }
        if input.is_empty() {
            return Ok(());
        }

        let resampler = self
            .resampler
            .as_mut()
            .ok_or_else(|| anyhow!("streaming resampler is not initialized"))?;
        self.pending_input.extend_from_slice(input);
        output.reserve(
            (self.pending_input[self.pending_input_start..].len() as f64
                * resampler.resample_ratio())
            .ceil() as usize,
        );

        while self.pending_input[self.pending_input_start..].len() >= resampler.input_frames_next()
        {
            let input_frames = resampler.input_frames_next();
            let output_frames = resampler.output_frames_next();
            let input_start = self.pending_input_start;
            let input_end = input_start + input_frames;
            let input_adapter =
                SequentialSlice::new(&self.pending_input[input_start..input_end], 1, input_frames)?;
            self.output_scratch.resize(output_frames, 0.0);
            let mut output_adapter = SequentialSlice::new_mut(
                &mut self.output_scratch[..output_frames],
                1,
                output_frames,
            )?;
            let (used_in, produced_out) =
                resampler.process_into_buffer(&input_adapter, &mut output_adapter, None)?;
            if used_in == 0 {
                break;
            }
            self.pending_input_start += used_in;

            let skip = self.discard_output.min(produced_out);
            self.discard_output -= skip;
            output.extend_from_slice(&self.output_scratch[skip..produced_out]);
        }

        self.compact_pending_input();
        Ok(())
    }

    fn compact_pending_input(&mut self) {
        if self.pending_input_start == 0 {
            return;
        }
        if self.pending_input_start >= self.pending_input.len() {
            self.pending_input.clear();
            self.pending_input_start = 0;
            return;
        }
        // The resampler consumes fixed-size chunks. Keep a logical head so the
        // common path does not memmove the pending buffer after every chunk;
        // compact only when the skipped prefix has grown large enough to matter.
        if self.pending_input_start >= STREAM_RESAMPLE_COMPACT_THRESHOLD
            && self.pending_input_start * 2 >= self.pending_input.len()
        {
            self.pending_input.drain(..self.pending_input_start);
            self.pending_input_start = 0;
        }
    }
}

pub fn crossfade(prev_tail: &[f32], current: &mut [f32]) {
    let n = prev_tail.len().min(current.len());
    if n == 0 {
        return;
    }

    for i in 0..n {
        let t = if n == 1 {
            1.0
        } else {
            i as f32 / (n - 1) as f32
        };
        let prev_gain = (t * std::f32::consts::FRAC_PI_2).cos().powi(2);
        let cur_gain = (t * std::f32::consts::FRAC_PI_2).sin().powi(2);
        current[i] = prev_tail[i] * prev_gain + current[i] * cur_gain;
    }
}

pub fn sola_offset(candidate: &[f32], reference: &[f32], search: usize) -> usize {
    let frame = reference.len().min(candidate.len());
    if frame == 0 {
        return 0;
    }

    let max_offset = search.min(candidate.len().saturating_sub(frame));
    let reference = &reference[..frame];
    // Reference energy is independent of `offset`; hoist it out of the search
    // loop so the normalized cross-correlation denominator only recomputes the
    // window-dependent term per iteration.
    let reference_energy = dot(reference, reference);
    // The window energy `dot(window, window)` is the only other offset-dependent
    // term, and sliding the frame by one sample just drops the leaving sample's
    // square and adds the entering sample's. Maintaining it incrementally keeps
    // the denominator O(1) per offset, so each step costs one cross-correlation
    // `dot` instead of two. Numerically equivalent to the full recompute (f32
    // accumulation drifts negligibly over the bounded search range).
    let mut window_energy = dot(&candidate[..frame], &candidate[..frame]);
    let mut best_offset = 0;
    let mut best_score = f32::MIN;
    for offset in 0..=max_offset {
        if offset > 0 {
            let leaving = candidate[offset - 1];
            let entering = candidate[offset + frame - 1];
            window_energy += entering * entering - leaving * leaving;
        }
        let window = &candidate[offset..offset + frame];
        let nom = dot(window, reference);
        let den = (window_energy * reference_energy + 1e-9).sqrt();
        let score = nom / den;
        if score > best_score {
            best_score = score;
            best_offset = offset;
        }
    }
    best_offset
}

/// Normalized cross-correlation of two equal-length windows (the shorter length
/// is used if they differ). This is the exact score [`sola_offset`] maximizes at
/// a single offset: `dot(a, b) / sqrt(dot(a, a) * dot(b, b))`. Returns a value in
/// roughly `[-1, 1]`, and `0.0` for empty / zero-energy inputs. Diagnostics-only;
/// the SOLA search keeps its own incremental energy bookkeeping.
pub fn normalized_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let a = &a[..n];
    let b = &b[..n];
    let nom = dot(a, b);
    let den = (dot(a, a) * dot(b, b) + 1e-9).sqrt();
    nom / den
}

pub fn sola_offset_with_threshold(
    candidate: &[f32],
    reference: &[f32],
    search: usize,
    min_rms: f32,
) -> usize {
    if rms(candidate) < min_rms || rms(reference) < min_rms {
        return 0;
    }
    sola_offset(candidate, reference, search)
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Reduce into independent accumulator lanes rather than a single `.sum()`.
    // f32 addition is not associative, so a plain `.sum()` forces a strict
    // left-to-right reduction that the compiler cannot auto-vectorize; eight
    // lanes (a 256-bit register's worth) let it emit SIMD. The lane assignment
    // is fixed, so the result is deterministic run-to-run (not bit-identical to
    // the scalar sum, which the SOLA search tolerates).
    const LANES: usize = 8;
    let n = a.len().min(b.len());
    let a = &a[..n];
    let b = &b[..n];
    let mut acc = [0.0f32; LANES];
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);
    for (ca, cb) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for lane in 0..LANES {
            acc[lane] += ca[lane] * cb[lane];
        }
    }
    let tail: f32 = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder())
        .map(|(x, y)| x * y)
        .sum();
    acc.iter().sum::<f32>() + tail
}

/// One-pole smoothing coefficient for an exponential follower with the given
/// time constant. `time_ms <= 0` yields `1.0` (instantaneous), so callers can
/// request a zero-length attack/release without special-casing.
// Negated `>` (not `<= 0.0`) is deliberate: it also routes NaN into the safe
// 1.0 branch, since every comparison with NaN is false. clippy's partial_cmp
// suggestion would drop that guarantee.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn one_pole_coef(time_ms: f32, sample_rate: f32) -> f32 {
    if !(time_ms > 0.0) || !(sample_rate > 0.0) {
        return 1.0;
    }
    let samples = time_ms * 0.001 * sample_rate;
    1.0 - (-1.0 / samples).exp()
}

/// Streaming amplitude noise gate. Tracks a smoothed amplitude envelope of the
/// input and ramps a per-sample gain between `1.0` (open, when the envelope is
/// at/above `threshold`) and `floor` (closed) using attack/release smoothing,
/// so level transitions do not click.
///
/// Operates at the input sample rate and only scales amplitude — it never
/// touches timing or the frame grid, so it is safe to run ahead of the RVC
/// feature/F0 extraction (audio-quality guardrail). State (`env`, `gain`)
/// persists across calls so chunk boundaries are seamless; processing one
/// buffer is identical to processing its concatenated halves.
pub struct NoiseGate {
    threshold: f32,
    floor: f32,
    attack_coef: f32,
    release_coef: f32,
    env: f32,
    gain: f32,
}

impl NoiseGate {
    pub fn new(
        sample_rate: f32,
        threshold: f32,
        attack_ms: f32,
        release_ms: f32,
        floor: f32,
    ) -> Self {
        Self {
            threshold: threshold.max(0.0),
            floor: floor.clamp(0.0, 1.0),
            attack_coef: one_pole_coef(attack_ms, sample_rate),
            release_coef: one_pole_coef(release_ms, sample_rate),
            // Start fully open so the first chunk after (re)load is not gated
            // shut before the envelope has seen any signal.
            env: 0.0,
            gain: 1.0,
        }
    }

    /// Update the live-adjustable threshold (linear amplitude). Attack/release
    /// are baked into the smoothing coefficients at construction.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.max(0.0);
    }

    pub fn process_in_place(&mut self, buf: &mut [f32]) {
        for sample in buf.iter_mut() {
            let level = sample.abs();
            // Detector: fast on the way up, slower on the way down, so brief
            // dips inside speech do not slam the gate shut.
            let det_coef = if level > self.env {
                self.attack_coef
            } else {
                self.release_coef
            };
            self.env += det_coef * (level - self.env);

            let target = if self.env >= self.threshold {
                1.0
            } else {
                self.floor
            };
            let gain_coef = if target > self.gain {
                self.attack_coef
            } else {
                self.release_coef
            };
            self.gain += gain_coef * (target - self.gain);

            *sample *= self.gain;
        }
    }
}

fn finite_nonnegative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn linear_resample_envelope_at(points: &[f32], index: usize, output_len: usize) -> f32 {
    if output_len == 0 || points.is_empty() {
        return 0.0;
    }
    if points.len() == 1 || output_len == 1 {
        return finite_nonnegative(points[0]);
    }

    let last_point = points.len() - 1;
    let last_output = output_len - 1;
    let position = index as f32 * last_point as f32 / last_output as f32;
    let left = position.floor() as usize;
    let right = (left + 1).min(last_point);
    let frac = position - left as f32;
    let left_value = finite_nonnegative(points[left]);
    let right_value = finite_nonnegative(points[right]);
    left_value + (right_value - left_value) * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn chunk_samples_apply_the_floor() {
        assert_eq!(chunk_samples_for_rate(48_000, 10), 480);
        assert_eq!(chunk_samples_for_rate(48_000, 1), 128);
    }

    #[test]
    fn converts_i16_roundtrip() {
        let src = [-32768, -1000, 0, 1000, 32767];
        let f = i16_to_f32(&src);
        let out = f32_to_i16(&f);
        assert_eq!(out[2], 0);
        assert!((out[1] + 1000).abs() <= 1);
    }

    #[test]
    fn converts_i16_to_f32_in_place() {
        let input = [-32768, 0, 32767];
        let mut output = [9.0; 3];

        i16_to_f32_into(&input, &mut output);

        assert_abs_diff_eq!(output[0], -1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(output[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(output[2], 32767.0 / 32768.0, epsilon = 1e-6);
    }

    #[test]
    fn converts_u16_to_f32_in_place() {
        let input = [0, 32768, 65535];
        let mut output = [9.0; 3];

        u16_to_f32_into(&input, &mut output);

        assert_abs_diff_eq!(output[0], -1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(output[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(output[2], 32767.0 / 32768.0, epsilon = 1e-6);
    }

    #[test]
    fn downmix_averages_interleaved_frames() {
        let stereo = [1.0, 0.0, 0.5, -0.5, -1.0, 1.0];
        let mut mono = [9.0; 3];

        downmix_to_mono_into(&stereo, 2, &mut mono);

        assert_abs_diff_eq!(mono[0], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(mono[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(mono[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn downmix_averages_four_channel_frames() {
        let quad = [1.0, 0.0, 0.5, -0.5, -1.0, 1.0, 0.25, -0.25];
        let mut mono = [9.0; 2];

        downmix_to_mono_into(&quad, 4, &mut mono);

        assert_abs_diff_eq!(mono[0], 0.25, epsilon = 1e-6);
        assert_abs_diff_eq!(mono[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn downmix_mono_is_copy() {
        let input = [0.25, -0.75];
        let mut output = [0.0; 2];

        downmix_to_mono_into(&input, 1, &mut output);

        assert_eq!(output, input);
    }

    #[test]
    fn upmix_duplicates_mono_across_channels() {
        let mono = [0.25, -0.5];
        let mut stereo = [9.0; 4];

        upmix_mono_into(&mono, 2, &mut stereo);

        assert_eq!(stereo, [0.25, 0.25, -0.5, -0.5]);
    }

    #[test]
    fn upmix_mono_channel_count_copies() {
        let mono = [0.25, -0.5];
        let mut output = [9.0; 2];

        upmix_mono_into(&mono, 1, &mut output);

        assert_eq!(output, mono);
    }

    #[test]
    fn upmix_duplicates_mono_across_four_channels() {
        let mono = [0.25, -0.5];
        let mut quad = [9.0; 8];

        upmix_mono_into(&mono, 4, &mut quad);

        assert_eq!(quad, [0.25, 0.25, 0.25, 0.25, -0.5, -0.5, -0.5, -0.5]);
    }

    #[test]
    fn converts_f32_to_i16_in_place_with_existing_rounding() {
        let input = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let mut output = [123; 7];

        f32_to_i16_into(&input, &mut output);

        assert_eq!(output, [-32767, -32767, -16384, 0, 16384, 32767, 32767]);
        assert_eq!(output, f32_to_i16(&input)[..]);
    }

    #[test]
    fn converts_f32_to_u16_in_place_with_existing_rounding() {
        let input = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let mut output = [123; 7];

        f32_to_u16_into(&input, &mut output);

        assert_eq!(output, [1, 1, 16385, 32768, 49152, 65535, 65535]);
    }

    #[test]
    fn converts_f32_to_i32_in_place_with_saturating_extremes() {
        let input = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let mut output = [123_i32; 7];

        f32_to_i32_into(&input, &mut output);

        assert_eq!(
            output,
            [
                -2_147_483_647,
                -2_147_483_647,
                -1_073_741_824,
                0,
                1_073_741_824,
                2_147_483_647,
                2_147_483_647,
            ]
        );
    }

    #[test]
    fn converts_i32_to_f32_in_place() {
        let input = [i32::MIN, -1_073_741_824, 0, 1_073_741_824, i32::MAX];
        let mut output = [0.0_f32; 5];

        i32_to_f32_into(&input, &mut output);

        assert_abs_diff_eq!(output[0], -1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(output[1], -0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(output[2], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(output[3], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(output[4], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn computes_rms() {
        assert_abs_diff_eq!(rms(&[1.0, -1.0]), 1.0);
        assert_eq!(rms(&[]), 0.0);
    }

    #[test]
    fn clamp_scale_in_place_matches_separate_passes() {
        let mut combined = [-2.0, -0.5, 0.25, 2.0];
        let mut separate = combined;

        clamp_scale_in_place(&mut combined, 0.5);
        separate
            .iter_mut()
            .for_each(|sample| *sample = sample.clamp(-1.0, 1.0));
        for sample in &mut separate {
            *sample *= 0.5;
        }

        assert_eq!(combined, separate);
    }

    #[test]
    fn apply_gain_and_rms_matches_separate_passes() {
        let mut combined = [-0.75, -0.25, 0.25, 0.75];
        let mut separate = combined;

        let combined_rms = apply_gain_and_rms(&mut combined, 2.0);
        for sample in &mut separate {
            *sample = (*sample * 2.0).clamp(-1.0, 1.0);
        }

        assert_eq!(combined, separate);
        assert_abs_diff_eq!(combined_rms, rms(&separate), epsilon = 1e-6);
    }

    #[test]
    fn computes_rms_envelope_with_zero_padded_tail_frames() {
        let envelope = compute_rms_envelope(&[1.0, 1.0], 100);

        assert_eq!(envelope.len(), 2);
        assert_abs_diff_eq!(envelope[0], 0.5f32.sqrt(), epsilon = 1e-6);
        assert_abs_diff_eq!(envelope[1], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn linear_envelope_resampling_handles_single_and_empty_inputs() {
        assert_eq!(linear_resample_envelope(&[0.25], 4), vec![0.25; 4]);
        assert_eq!(
            linear_resample_envelope(&[0.0, 1.0], 3),
            vec![0.0, 0.5, 1.0]
        );
        assert_eq!(linear_resample_envelope(&[], 3), vec![0.0; 3]);
        assert!(linear_resample_envelope(&[1.0], 0).is_empty());
    }

    #[test]
    fn rms_mix_rate_one_keeps_output_unchanged() {
        let input = [0.1, 0.2, 0.3, 0.4];
        let mut output = [0.4, -0.2, 0.1, -0.3];
        let before = output;

        apply_rms_mix(&input, &mut output, 100, 1.0);

        assert_eq!(output, before);
    }

    #[test]
    fn rms_mix_keeps_output_when_envelopes_match() {
        let input = [0.25, -0.5, 0.75, -1.0, 0.5, -0.25];
        let mut output = input;

        apply_rms_mix(&input, &mut output, 100, 0.35);

        for (actual, expected) in output.iter().zip(input) {
            assert_abs_diff_eq!(*actual, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn rms_mix_zero_moves_output_rms_toward_input_rms() {
        let input = vec![0.4; 16];
        let mut output = vec![0.8; 16];

        apply_rms_mix(&input, &mut output, 100, 0.0);

        assert_abs_diff_eq!(rms(&output), rms(&input), epsilon = 1e-6);
    }

    #[test]
    fn rms_mix_uses_expected_gain_for_intermediate_mix() {
        let input = vec![0.5; 4];
        let mut output = vec![1.0; 4];

        apply_rms_mix(&input, &mut output, 100, 0.5);

        for sample in output {
            assert_abs_diff_eq!(sample, 0.5f32.sqrt(), epsilon = 1e-6);
        }
    }

    #[test]
    fn rms_mix_uses_powf_for_non_special_rates() {
        let input = vec![0.25; 4];
        let mut output = vec![1.0; 4];

        apply_rms_mix(&input, &mut output, 100, 0.9);

        let expected = 0.25f32.powf(0.1);
        for sample in output {
            assert_abs_diff_eq!(sample, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn rms_mix_handles_empty_and_short_inputs() {
        let mut empty = Vec::new();
        apply_rms_mix(&[], &mut empty, 100, 0.0);
        assert!(empty.is_empty());

        let mut short = vec![1.0];
        apply_rms_mix(&[0.0], &mut short, 100, 0.0);
        assert_abs_diff_eq!(short[0], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn resamples_when_rubato_adjusts_input_frame_count() {
        let input = vec![0.0; 10_496];
        let out = resample_mono(&input, 48_000, 16_000).unwrap();
        assert_eq!(out.len(), 3_499);
    }

    #[test]
    fn streaming_resampler_handles_output_device_rate() {
        let input = vec![0.0; 48_000];
        let mut resampler = StreamingResampleMono::new(48_000, 44_100).unwrap();
        let mut output = Vec::new();

        for chunk in input.chunks(480) {
            resampler.process_into(chunk, &mut output).unwrap();
        }

        assert!((43_000..=44_100).contains(&output.len()));
    }

    #[test]
    fn finds_sola_offset() {
        let reference = [0.0, 1.0, 0.5, 0.0];
        let candidate = [0.2, 0.1, 0.0, 1.0, 0.5, 0.0, -0.1];
        assert_eq!(sola_offset(&candidate, &reference, 4), 2);
    }

    #[test]
    fn normalized_correlation_matches_sola_score() {
        // Perfectly aligned identical windows correlate to ~1.0.
        let a = [0.0, 1.0, 0.5, -0.2];
        assert_abs_diff_eq!(normalized_correlation(&a, &a), 1.0, epsilon = 1e-5);
        // A scaled copy is still perfectly correlated (normalization removes gain).
        let scaled: Vec<f32> = a.iter().map(|x| x * 3.0).collect();
        assert_abs_diff_eq!(normalized_correlation(&a, &scaled), 1.0, epsilon = 1e-5);
        // The value reported at the offset SOLA picks equals the score SOLA used.
        let reference = [0.0, 1.0, 0.5, 0.0];
        let candidate = [0.2, 0.1, 0.0, 1.0, 0.5, 0.0, -0.1];
        let offset = sola_offset(&candidate, &reference, 4);
        let window = &candidate[offset..offset + reference.len()];
        assert_abs_diff_eq!(
            normalized_correlation(window, &reference),
            1.0,
            epsilon = 1e-5
        );
    }

    #[test]
    fn normalized_correlation_handles_empty_and_silent() {
        assert_eq!(normalized_correlation(&[], &[1.0, 0.0]), 0.0);
        assert_eq!(normalized_correlation(&[0.0; 4], &[0.0; 4]), 0.0);
    }

    #[test]
    fn crossfade_moves_from_previous_to_current() {
        let prev = [1.0; 4];
        let mut current = [0.0; 4];
        crossfade(&prev, &mut current);

        assert_abs_diff_eq!(current[0], 1.0);
        assert!(current[1] > current[2]);
        assert_abs_diff_eq!(current[3], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn sola_offset_skips_low_rms() {
        let reference = [0.0; 4];
        let candidate = [0.0; 8];
        assert_eq!(
            sola_offset_with_threshold(&candidate, &reference, 4, 1e-4),
            0
        );
    }

    #[test]
    fn sola_offset_handles_short_inputs() {
        assert_eq!(sola_offset(&[], &[1.0, 0.0], 4), 0);
        assert_eq!(sola_offset(&[1.0], &[1.0, 0.0], 4), 0);
    }

    #[test]
    fn noise_gate_passes_loud_signal_through() {
        // Loud, sustained tone stays open: after the attack ramp, amplitude is
        // preserved.
        let sr = 48_000.0;
        let mut gate = NoiseGate::new(sr, 0.05, 1.0, 50.0, 0.0);
        let mut buf: Vec<f32> = (0..4_800).map(|i| 0.5 * (i as f32 * 0.05).sin()).collect();
        let before = buf.clone();
        gate.process_in_place(&mut buf);

        // Tail (well past the 1 ms attack) should match the input closely.
        for (out, inp) in buf[2_400..].iter().zip(&before[2_400..]) {
            assert_abs_diff_eq!(*out, *inp, epsilon = 1e-3);
        }
    }

    #[test]
    fn noise_gate_attenuates_quiet_noise_to_floor() {
        let sr = 48_000.0;
        let mut gate = NoiseGate::new(sr, 0.1, 1.0, 5.0, 0.0);
        // Steady low-level noise below threshold.
        let mut buf = vec![0.02f32; 4_800];
        gate.process_in_place(&mut buf);

        // After release, output should be driven toward the floor (~0).
        assert!(rms(&buf[2_400..]) < 1e-3);
    }

    #[test]
    fn noise_gate_opens_smoothly_without_click() {
        let sr = 48_000.0;
        let mut gate = NoiseGate::new(sr, 0.05, 10.0, 50.0, 0.0);
        // Drive the gate fully closed first (well past several release time
        // constants) so the step below exercises the opening ramp, not a gate
        // that was still partly open.
        let mut silence = vec![0.0f32; 24_000];
        gate.process_in_place(&mut silence);

        // Loud step into the closed gate. The onset is attenuated and the gain
        // ramps rather than jumping (no click).
        let mut buf = vec![0.5f32; 4_800];
        gate.process_in_place(&mut buf);

        assert!(
            buf[0].abs() < 0.1,
            "onset should be attenuated, got {}",
            buf[0]
        );
        for w in buf.windows(2) {
            assert!(
                (w[1] - w[0]).abs() < 0.05,
                "per-sample step too large (click)"
            );
        }
        // The gate fully opens within the buffer: the tail recovers the input.
        assert_abs_diff_eq!(buf[4_799], 0.5, epsilon = 1e-2);
    }

    #[test]
    fn noise_gate_is_continuous_across_chunks() {
        let sr = 48_000.0;
        let signal: Vec<f32> = (0..2_000).map(|i| 0.3 * (i as f32 * 0.02).sin()).collect();

        let mut whole = signal.clone();
        NoiseGate::new(sr, 0.05, 5.0, 20.0, 0.0).process_in_place(&mut whole);

        let mut split = signal.clone();
        let mut gate = NoiseGate::new(sr, 0.05, 5.0, 20.0, 0.0);
        let (head, tail) = split.split_at_mut(777);
        gate.process_in_place(head);
        gate.process_in_place(tail);

        for (a, b) in whole.iter().zip(&split) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-6);
        }
    }
}
