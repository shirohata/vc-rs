// ContentVec emits frames on a 20 ms stride at 16 kHz. RVC's 10 ms grid is
// created later by repeating feature frames, so keep this alignment scoped to
// the shared ContentVec/RMVPE waveform context and not pitch/output sizing.
pub(super) const EMBEDDER_SAMPLE_RATE: u32 = 16_000;
pub(super) const RVC_SAMPLE_RATE: u32 = 48_000;
pub(super) const CONTENTVEC_CONTEXT_ALIGN_SAMPLES: usize = 320;
pub(super) const RMVPE_FRAME_SAMPLES_16K: usize = 160;
pub(super) const RMVPE_BUCKET_FRAMES: usize = 32;
pub(super) const RMVPE_GUARD_FRAMES: usize = 5;

pub(super) fn ms_to_samples(sample_rate: u32, ms: u32) -> usize {
    ((sample_rate as u64 * ms as u64) / 1000) as usize
}

pub(super) fn extra_convert_samples_from_ms(ms: u32, rvc_sample_rate: u32) -> usize {
    ms_to_samples(rvc_sample_rate, ms)
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

pub(super) fn onnx_silence_front_feature_frames(
    extra_convert_samples: usize,
    rvc_sample_rate: u32,
) -> usize {
    let extra_16k_samples = (extra_convert_samples as u64 * EMBEDDER_SAMPLE_RATE as u64
        / rvc_sample_rate as u64) as usize;
    (extra_16k_samples / 360) * 2
}

pub(super) fn keep_tail_in_place<T>(values: &mut Vec<T>, len: usize) {
    if values.len() > len {
        values.drain(..values.len() - len);
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
    rvc_sample_rate: u32,
) -> usize {
    tensor_rt_convert_size_16k(
        chunk_samples,
        sample_rate,
        ms_to_samples(rvc_sample_rate, output_extra_ms),
        extra_convert_samples,
        rvc_sample_rate,
    )
}

pub(super) fn rmvpe_model_input_samples_16k(chunk_samples: usize, sample_rate: u32) -> usize {
    // Match upstream RVC's RMVPE framing: 10 ms hop at 16 kHz, then mel2hidden
    // pads hidden frames to 32-frame buckets. The guard frames preserve a small
    // amount of F0 context without coupling RMVPE to ContentVec's larger window.
    let chunk_frames = (chunk_samples as u64 * 100).div_ceil(sample_rate as u64) as usize;
    let required_frames = chunk_frames.saturating_add(RMVPE_GUARD_FRAMES).max(1);
    let bucket_frames = align_up(required_frames, RMVPE_BUCKET_FRAMES);
    bucket_frames.saturating_sub(1) * RMVPE_FRAME_SAMPLES_16K
}

pub(super) fn rmvpe_model_input_samples_for_context_16k(
    chunk_samples: usize,
    sample_rate: u32,
    max_context_samples_16k: usize,
) -> usize {
    rmvpe_model_input_samples_16k(chunk_samples, sample_rate).min(max_context_samples_16k)
}

pub(super) fn tensor_rt_convert_size_16k(
    new_audio_samples: usize,
    sample_rate: u32,
    output_extra_samples: usize,
    extra_convert_samples: usize,
    rvc_sample_rate: u32,
) -> usize {
    // Pipeline construction validates the new-audio hop with RvcChunkTiming,
    // so this rate conversion is exact for production chunks. Only the full
    // context window is rounded to ContentVec's 20 ms stride below; never use
    // that aligned window size to advance audio, F0, latent noise, or NSF phase.
    let new_audio_16k_samples = samples_between_rates(
        new_audio_samples,
        sample_rate,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    let output_extra_16k_samples = samples_between_rates(
        output_extra_samples,
        rvc_sample_rate,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    let extra_16k_samples = samples_between_rates(
        extra_convert_samples,
        rvc_sample_rate,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    align_up(
        new_audio_16k_samples + output_extra_16k_samples + extra_16k_samples,
        CONTENTVEC_CONTEXT_ALIGN_SAMPLES,
    )
}
