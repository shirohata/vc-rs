use super::shape::feature_len_for_samples;

#[cfg(test)]
pub(super) fn align_pitchf_to_features(input: &[f32], target_len: usize) -> Vec<f32> {
    let mut out = Vec::new();
    align_pitchf_to_features_into(input, target_len, &mut out);
    out
}

pub(super) fn align_pitchf_to_features_into(
    input: &[f32],
    target_len: usize,
    output: &mut Vec<f32>,
) {
    output.clear();
    if input.len() >= target_len {
        output.extend_from_slice(&input[input.len() - target_len..]);
    } else {
        output.resize(target_len - input.len(), 0.0);
        output.extend_from_slice(input);
    }
}

#[cfg(test)]
pub(super) fn center_crop_pitchf_to_features(input: &[f32], target_len: usize) -> Vec<f32> {
    let mut out = Vec::new();
    center_crop_pitchf_to_features_into(input, target_len, &mut out);
    out
}

pub(super) fn center_crop_pitchf_to_features_into(
    input: &[f32],
    target_len: usize,
    output: &mut Vec<f32>,
) {
    if input.len() > target_len {
        let excess = input.len() - target_len;
        let front_drop = excess / 2;
        output.clear();
        output.extend_from_slice(&input[front_drop..front_drop + target_len]);
    } else {
        align_pitchf_to_features_into(input, target_len, output);
    }
}

pub(super) fn pitchf_tail_for_output(
    pitchf: &[f32],
    output_samples: usize,
    sample_rate: u32,
) -> Vec<f32> {
    if pitchf.is_empty() || output_samples == 0 || sample_rate == 0 {
        return Vec::new();
    }
    let frames = feature_len_for_samples(output_samples, sample_rate)
        .max(1)
        .min(pitchf.len());
    pitchf[pitchf.len() - frames..].to_vec()
}

pub(super) fn voiced_ratio(pitchf: &[f32]) -> f32 {
    if pitchf.is_empty() {
        return 0.0;
    }
    let voiced = pitchf.iter().filter(|&&f0| f0 > 0.0).count();
    voiced as f32 / pitchf.len() as f32
}

pub(super) fn coarse_pitch_into(pitchf: &[f32], output: &mut Vec<i64>) {
    let f0_min = 50.0f32;
    let f0_max = 1100.0f32;
    let mel_min = 1127.0 * (1.0 + f0_min / 700.0).ln();
    let mel_max = 1127.0 * (1.0 + f0_max / 700.0).ln();
    output.clear();
    output.reserve(pitchf.len());
    for &f0 in pitchf {
        let mel = if f0 > 0.0 {
            1127.0 * (1.0 + f0 / 700.0).ln()
        } else {
            0.0
        };
        let coarse = if mel > 0.0 {
            (mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0
        } else {
            1.0
        };
        output.push(coarse.clamp(1.0, 255.0).round() as i64);
    }
}
