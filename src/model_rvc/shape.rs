// ContentVec emits frames on a 20 ms stride at 16 kHz. RVC's 10 ms grid is
// created later by repeating feature frames, so keep this alignment scoped to
// the shared ContentVec/RMVPE waveform context and not pitch/output sizing.
pub(super) const EMBEDDER_SAMPLE_RATE: u32 = 16_000;
pub(super) const RVC_SAMPLE_RATE: u32 = 48_000;
pub(super) const CONTENTVEC_CONTEXT_ALIGN_SAMPLES: usize = 320;

pub(super) fn ms_to_samples(sample_rate: u32, ms: u32) -> usize {
    ((sample_rate as u64 * ms as u64) / 1000) as usize
}

pub(super) fn extra_convert_samples_from_ms(ms: u32) -> usize {
    ms_to_samples(RVC_SAMPLE_RATE, ms)
}

pub(super) fn feature_len_for_samples(samples: usize, sample_rate: u32) -> usize {
    (samples as u64 * 100 / sample_rate as u64) as usize
}

pub(super) enum Rounding {
    Floor,
    Ceil,
}

pub(super) fn samples_between_rates(
    samples: usize,
    from_sample_rate: u32,
    to_sample_rate: u32,
    rounding: Rounding,
) -> usize {
    let numerator = samples as u64 * to_sample_rate as u64;
    let denominator = from_sample_rate as u64;
    match rounding {
        Rounding::Floor => (numerator / denominator) as usize,
        Rounding::Ceil => numerator.div_ceil(denominator) as usize,
    }
}

pub(super) fn onnx_silence_front_feature_frames(extra_convert_samples: usize) -> usize {
    let extra_16k_samples = (extra_convert_samples as u64 * EMBEDDER_SAMPLE_RATE as u64
        / RVC_SAMPLE_RATE as u64) as usize;
    (extra_16k_samples / 360) * 2
}

pub(super) fn keep_tail_in_place<T>(values: &mut Vec<T>, len: usize) {
    if values.len() > len {
        values.drain(..values.len() - len);
    }
}

pub(super) fn tail_or_left_pad(mut values: Vec<f32>, len: usize) -> Vec<f32> {
    keep_tail_in_place(&mut values, len);
    if values.len() < len {
        let mut padded = vec![0.0; len - values.len()];
        padded.extend(values);
        padded
    } else {
        values
    }
}

#[cfg(test)]
pub(super) fn aligned_rvc_input_len(
    chunk_len: usize,
    sample_rate: u32,
    extra_48k_samples: usize,
) -> usize {
    let chunk_16k = samples_between_rates(
        chunk_len,
        sample_rate,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    let extra_16k = samples_between_rates(
        extra_48k_samples,
        RVC_SAMPLE_RATE,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    let convert_16k = align_up(chunk_16k + extra_16k, CONTENTVEC_CONTEXT_ALIGN_SAMPLES);
    samples_between_rates(
        convert_16k,
        EMBEDDER_SAMPLE_RATE,
        sample_rate,
        Rounding::Ceil,
    )
}

#[cfg(test)]
pub(super) fn output_len_from_convert_size(
    convert_len_16k: usize,
    _input_sample_rate: u32,
    extra_48k_samples: usize,
    output_sample_rate: u32,
) -> usize {
    let extra_16k = samples_between_rates(
        extra_48k_samples,
        RVC_SAMPLE_RATE,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    samples_between_rates(
        convert_len_16k.saturating_sub(extra_16k),
        EMBEDDER_SAMPLE_RATE,
        output_sample_rate,
        Rounding::Floor,
    )
    .max(1)
}

pub(super) fn align_up(value: usize, align: usize) -> usize {
    if align == 0 || value.is_multiple_of(align) {
        value
    } else {
        value + (align - value % align)
    }
}

pub(super) fn tensor_rt_model_input_samples_16k(
    chunk_samples: usize,
    sample_rate: u32,
    output_extra_ms: u32,
    extra_convert_samples: usize,
) -> usize {
    tensor_rt_convert_size_16k(
        chunk_samples,
        sample_rate,
        ms_to_samples(RVC_SAMPLE_RATE, output_extra_ms),
        extra_convert_samples,
    )
}

pub(super) fn tensor_rt_convert_size_16k(
    new_audio_samples: usize,
    sample_rate: u32,
    output_extra_samples: usize,
    extra_convert_samples: usize,
) -> usize {
    let new_audio_16k_samples = samples_between_rates(
        new_audio_samples,
        sample_rate,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    let output_extra_16k_samples = samples_between_rates(
        output_extra_samples,
        RVC_SAMPLE_RATE,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    let extra_16k_samples = samples_between_rates(
        extra_convert_samples,
        RVC_SAMPLE_RATE,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    align_up(
        new_audio_16k_samples + output_extra_16k_samples + extra_16k_samples,
        CONTENTVEC_CONTEXT_ALIGN_SAMPLES,
    )
}
