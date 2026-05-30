#![allow(dead_code)]

use anyhow::{anyhow, Result};
use audioadapter_buffers::direct::SequentialSlice;
use rubato::{Fft, FixedSync, Resampler};

const STREAM_RESAMPLE_CHUNK: usize = 480;

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

pub fn rms(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }
    let sum = input.iter().map(|x| x * x).sum::<f32>();
    (sum / input.len() as f32).sqrt()
}

pub fn compute_rms_envelope(input: &[f32], sample_rate: usize) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    let hop_len = (sample_rate / 100).max(1);
    let frame_len = hop_len.saturating_mul(4).max(1);
    let frame_count = input.len().div_ceil(hop_len);
    let mut envelope = Vec::with_capacity(frame_count);

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
        envelope.push((sum / frame_len as f32).sqrt());
    }

    envelope
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

    let input_rms = linear_resample_envelope(
        &compute_rms_envelope(input_reference, sample_rate),
        output.len(),
    );
    let output_rms =
        linear_resample_envelope(&compute_rms_envelope(output, sample_rate), output.len());
    let exponent = 1.0 - rms_mix_rate;

    for ((sample, &in_rms), &out_rms) in output.iter_mut().zip(&input_rms).zip(&output_rms) {
        if !sample.is_finite() {
            *sample = 0.0;
            continue;
        }
        let out_rms = finite_nonnegative(out_rms).max(1e-3);
        let ratio = finite_nonnegative(in_rms) / out_rms;
        let gain = ratio.powf(exponent);
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
    let mut resampler = Fft::<f32>::new(from_hz, to_hz, requested_chunk, 1, 1, FixedSync::Both)?;
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
    output_scratch: Vec<f32>,
    discard_output: usize,
}

impl StreamingResampleMono {
    pub fn new(from_hz: usize, to_hz: usize) -> Result<Self> {
        let resampler = if from_hz == to_hz {
            None
        } else {
            Some(Fft::<f32>::new(
                from_hz,
                to_hz,
                STREAM_RESAMPLE_CHUNK,
                1,
                1,
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
            (self.pending_input.len() as f64 * resampler.resample_ratio()).ceil() as usize,
        );

        while self.pending_input.len() >= resampler.input_frames_next() {
            let input_frames = resampler.input_frames_next();
            let output_frames = resampler.output_frames_next();
            let input_adapter =
                SequentialSlice::new(&self.pending_input[..input_frames], 1, input_frames)?;
            self.output_scratch.resize(output_frames, 0.0);
            let mut output_adapter = SequentialSlice::new_mut(
                &mut self.output_scratch[..output_frames],
                1,
                output_frames,
            )?;
            let (used_in, produced_out) =
                resampler.process_into_buffer(&input_adapter, &mut output_adapter, None)?;
            self.pending_input.drain(..used_in);

            let skip = self.discard_output.min(produced_out);
            self.discard_output -= skip;
            output.extend_from_slice(&self.output_scratch[skip..produced_out]);
        }

        Ok(())
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
    let mut best_offset = 0;
    let mut best_score = f32::MIN;
    for offset in 0..=max_offset {
        let window = &candidate[offset..offset + frame];
        let nom = dot(window, &reference[..frame]);
        let den =
            (dot(window, window) * dot(&reference[..frame], &reference[..frame]) + 1e-9).sqrt();
        let score = nom / den;
        if score > best_score {
            best_score = score;
            best_offset = offset;
        }
    }
    best_offset
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

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn finite_nonnegative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

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
    fn computes_rms() {
        assert_abs_diff_eq!(rms(&[1.0, -1.0]), 1.0);
        assert_eq!(rms(&[]), 0.0);
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
    fn finds_sola_offset() {
        let reference = [0.0, 1.0, 0.5, 0.0];
        let candidate = [0.2, 0.1, 0.0, 1.0, 0.5, 0.0, -0.1];
        assert_eq!(sola_offset(&candidate, &reference, 4), 2);
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
}
