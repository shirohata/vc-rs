use super::shape::feature_len_for_samples;

pub(super) fn align_pitchf_to_features(input: &[f32], target_len: usize) -> Vec<f32> {
    if input.len() >= target_len {
        input[input.len() - target_len..].to_vec()
    } else {
        let mut out = vec![0.0; target_len - input.len()];
        out.extend_from_slice(input);
        out
    }
}

pub(super) fn center_crop_pitchf_to_features(input: &[f32], target_len: usize) -> Vec<f32> {
    if input.len() > target_len {
        let excess = input.len() - target_len;
        let front_drop = excess / 2;
        input[front_drop..front_drop + target_len].to_vec()
    } else {
        align_pitchf_to_features(input, target_len)
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

pub(super) fn coarse_pitch(pitchf: &[f32]) -> Vec<i64> {
    let f0_min = 50.0f32;
    let f0_max = 1100.0f32;
    let mel_min = 1127.0 * (1.0 + f0_min / 700.0).ln();
    let mel_max = 1127.0 * (1.0 + f0_max / 700.0).ln();
    pitchf
        .iter()
        .map(|&f0| {
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
            coarse.clamp(1.0, 255.0).round() as i64
        })
        .collect()
}
