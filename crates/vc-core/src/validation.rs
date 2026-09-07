use anyhow::{bail, Result};

pub const CONVERSION_TIMING_LIMITS: ConversionTimingLimits = ConversionTimingLimits {
    min_chunk_ms: 20,
    max_chunk_ms: 2000,
    max_crossfade_ms: 1000,
    max_sola_search_ms: 1000,
    max_tail_discard_ms: 1000,
    max_output_extra_ms: 3000,
    min_extra_convert_ms: 20,
    max_extra_convert_ms: 3000,
};

#[derive(Clone, Copy, Debug)]
pub struct ConversionTiming {
    pub chunk_ms: u32,
    pub crossfade_ms: u32,
    pub sola_search_ms: u32,
    pub tail_discard_ms: u32,
    pub extra_convert_ms: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ConversionTimingLimits {
    pub min_chunk_ms: u32,
    pub max_chunk_ms: u32,
    pub max_crossfade_ms: u32,
    pub max_sola_search_ms: u32,
    pub max_tail_discard_ms: u32,
    pub max_output_extra_ms: u32,
    pub min_extra_convert_ms: u32,
    pub max_extra_convert_ms: u32,
}

/// Fixed RVC hop on the shared 10 ms timeline. Build this once rates are known,
/// before allocating FIFOs or model profiles. ContentVec's 20 ms *context*
/// alignment is independent: it must never round this hop up to a larger step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RvcChunkTiming {
    pub input_chunk_samples: usize,
    pub chunk_samples_16k: usize,
    pub advance_frames: usize,
}

impl RvcChunkTiming {
    pub fn from_ms(chunk_ms: u32, input_rate: u32) -> Result<Self> {
        validate_u32_range(
            "chunk_ms",
            chunk_ms,
            CONVERSION_TIMING_LIMITS.min_chunk_ms,
            CONVERSION_TIMING_LIMITS.max_chunk_ms,
        )?;
        validate_rvc_chunk_ms(chunk_ms)?;
        let advance_frames = (chunk_ms / 10) as usize;
        Ok(Self {
            input_chunk_samples: rvc_samples_for_frames(advance_frames, input_rate)?,
            chunk_samples_16k: advance_frames * 160,
            advance_frames,
        })
    }

    /// Public model APIs receive samples instead of milliseconds. Validate the
    /// same rational duration without rounding, including on nonstandard rates.
    pub fn from_samples(input_chunk_samples: usize, input_rate: u32) -> Result<Self> {
        if input_rate == 0 || input_chunk_samples == 0 {
            bail!("RVC input sample rate and chunk length must be positive");
        }
        let frame_numerator = input_chunk_samples as u128 * 100;
        if !frame_numerator.is_multiple_of(u128::from(input_rate)) {
            bail!(
                "RVC chunk of {input_chunk_samples} samples at {input_rate} Hz must span whole 10 ms frames; use a supported chunk_ms value such as 20 or 30, with an integer sample count at this rate"
            );
        }
        let advance_frames = usize::try_from(frame_numerator / u128::from(input_rate))?;
        let chunk_samples_16k = advance_frames
            .checked_mul(160)
            .ok_or_else(|| anyhow::anyhow!("RVC 16 kHz chunk size overflows usize"))?;
        Ok(Self {
            input_chunk_samples,
            chunk_samples_16k,
            advance_frames,
        })
    }

    /// Exact hop at a model/output rate. A 30 ms chunk at 22.05 kHz is rejected
    /// (661.5 samples), while a 20 ms chunk is valid; truncation accumulates drift.
    pub fn samples_at_rate(&self, sample_rate: u32) -> Result<usize> {
        rvc_samples_for_frames(self.advance_frames, sample_rate)
    }
}

pub fn validate_rvc_chunk_ms(chunk_ms: u32) -> Result<()> {
    if chunk_ms == 0 || !chunk_ms.is_multiple_of(10) {
        let lower = (chunk_ms / 10 * 10).max(CONVERSION_TIMING_LIMITS.min_chunk_ms);
        let upper = lower.saturating_add(10);
        bail!(
            "RVC chunk_ms must be a multiple of 10 ms; {chunk_ms} ms is unsupported (try {lower} or {upper} ms)"
        );
    }
    Ok(())
}

fn rvc_samples_for_frames(advance_frames: usize, sample_rate: u32) -> Result<usize> {
    if sample_rate == 0 {
        bail!("RVC sample rate must be positive");
    }
    let numerator = advance_frames as u128 * u128::from(sample_rate);
    if !numerator.is_multiple_of(100) {
        let mut divisor = sample_rate;
        let mut remainder = 100;
        while remainder != 0 {
            (divisor, remainder) = (remainder, divisor % remainder);
        }
        let step_ms = 1000 / divisor;
        bail!(
            "RVC chunk at {sample_rate} Hz must contain an integer number of samples; use chunk_ms in multiples of {step_ms} ms"
        );
    }
    Ok(usize::try_from(numerator / 100)?)
}

pub fn validate_conversion_timing(
    timing: ConversionTiming,
    limits: ConversionTimingLimits,
) -> Result<()> {
    validate_u32_range(
        "chunk_ms",
        timing.chunk_ms,
        limits.min_chunk_ms,
        limits.max_chunk_ms,
    )?;
    validate_u32_range(
        "crossfade_ms",
        timing.crossfade_ms,
        0,
        limits.max_crossfade_ms,
    )?;
    validate_u32_range(
        "sola_search_ms",
        timing.sola_search_ms,
        0,
        limits.max_sola_search_ms,
    )?;
    validate_u32_range(
        "rvc_output_tail_discard_ms",
        timing.tail_discard_ms,
        0,
        limits.max_tail_discard_ms,
    )?;
    validate_u32_range(
        "extra_convert_ms",
        timing.extra_convert_ms,
        limits.min_extra_convert_ms,
        limits.max_extra_convert_ms,
    )?;

    // The three output-context knobs are added before pipeline construction and
    // then converted to sample counts. Keep this bound shared so front-ends
    // cannot accidentally request a huge TensorRT profile or ring buffer.
    let output_extra_ms = timing
        .crossfade_ms
        .checked_add(timing.sola_search_ms)
        .and_then(|v| v.checked_add(timing.tail_discard_ms))
        .ok_or_else(|| anyhow::anyhow!("output context milliseconds overflow u32"))?;
    validate_u32_range(
        "crossfade_ms + sola_search_ms + rvc_output_tail_discard_ms",
        output_extra_ms,
        0,
        limits.max_output_extra_ms,
    )?;

    Ok(())
}

pub fn validate_unit_interval(name: &str, value: f32) -> Result<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        bail!("{name} must be a finite value in 0.0..=1.0")
    }
}

pub fn validate_non_negative_f32(name: &str, value: f32) -> Result<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        bail!("{name} must be a finite, non-negative value")
    }
}

pub fn validate_finite_f32(name: &str, value: f32) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        bail!("{name} must be finite")
    }
}

fn validate_u32_range(name: &str, value: u32, min: u32, max: u32) -> Result<()> {
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        bail!("{name} must be in {min}..={max} ms")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_timing() -> ConversionTiming {
        ConversionTiming {
            chunk_ms: 500,
            crossfade_ms: 85,
            sola_search_ms: 12,
            tail_discard_ms: 10,
            extra_convert_ms: 100,
        }
    }

    #[test]
    fn accepts_default_realtime_timing() {
        validate_conversion_timing(valid_timing(), CONVERSION_TIMING_LIMITS).unwrap();
    }

    #[test]
    fn rvc_rejects_fractional_frame_chunks_without_restricting_passthrough() {
        for chunk_ms in [21, 25, 29] {
            assert!(validate_rvc_chunk_ms(chunk_ms).is_err());
            assert!(RvcChunkTiming::from_ms(chunk_ms, 16_000).is_err());
            assert!(RvcChunkTiming::from_samples(chunk_ms as usize * 16, 16_000).is_err());
            // The bounds validator is also used by model-free passthrough.
            validate_conversion_timing(
                ConversionTiming {
                    chunk_ms,
                    ..valid_timing()
                },
                CONVERSION_TIMING_LIMITS,
            )
            .unwrap();
        }
    }

    #[test]
    fn rvc_timing_preserves_exact_duration_across_rates() {
        for chunk_ms in [20, 30, 100, 500, 2000] {
            for input_rate in [16_000, 32_000, 44_100, 48_000, 96_000] {
                let timing = RvcChunkTiming::from_ms(chunk_ms, input_rate).unwrap();
                assert_eq!(
                    timing,
                    RvcChunkTiming::from_samples(timing.input_chunk_samples, input_rate).unwrap(),
                );
                assert_eq!(timing.chunk_samples_16k, timing.advance_frames * 160);
                assert_eq!(
                    timing.input_chunk_samples as u128 * 16_000,
                    timing.chunk_samples_16k as u128 * u128::from(input_rate),
                );
                for output_rate in [16_000, 32_000, 40_000, 44_100, 48_000] {
                    let output_samples = timing.samples_at_rate(output_rate).unwrap();
                    assert_eq!(
                        output_samples as u128 * 16_000,
                        timing.chunk_samples_16k as u128 * u128::from(output_rate),
                    );
                }
            }
        }
    }

    #[test]
    fn rvc_rejects_fractional_samples_at_input_and_output_rates() {
        let valid = RvcChunkTiming::from_ms(20, 22_050).unwrap();
        assert_eq!(valid.input_chunk_samples, 441);
        assert_eq!(valid.samples_at_rate(22_050).unwrap(), 441);
        let err = RvcChunkTiming::from_ms(30, 22_050).unwrap_err();
        assert!(err.to_string().contains("multiples of 20 ms"));
        let odd_frames = RvcChunkTiming::from_ms(30, 48_000).unwrap();
        assert!(odd_frames.samples_at_rate(22_050).is_err());
        assert!(RvcChunkTiming::from_samples(661, 22_050).is_err());
        assert!(RvcChunkTiming::from_samples(662, 22_050).is_err());
    }

    #[test]
    fn rvc_timing_rejects_zero_rates_and_overflow() {
        assert!(RvcChunkTiming::from_ms(20, 0).is_err());
        assert!(RvcChunkTiming::from_samples(0, 48_000).is_err());
        assert!(RvcChunkTiming::from_samples(usize::MAX, 100).is_err());
        assert!(RvcChunkTiming::from_ms(20, 48_000)
            .unwrap()
            .samples_at_rate(0)
            .is_err());
    }

    #[test]
    fn rejects_chunk_values_outside_profile() {
        assert!(validate_conversion_timing(
            ConversionTiming {
                chunk_ms: CONVERSION_TIMING_LIMITS.min_chunk_ms - 1,
                ..valid_timing()
            },
            CONVERSION_TIMING_LIMITS,
        )
        .is_err());
        assert!(validate_conversion_timing(
            ConversionTiming {
                chunk_ms: CONVERSION_TIMING_LIMITS.max_chunk_ms + 1,
                ..valid_timing()
            },
            CONVERSION_TIMING_LIMITS,
        )
        .is_err());
    }

    #[test]
    fn rejects_extra_convert_values_outside_profile() {
        assert!(validate_conversion_timing(
            ConversionTiming {
                extra_convert_ms: CONVERSION_TIMING_LIMITS.min_extra_convert_ms - 1,
                ..valid_timing()
            },
            CONVERSION_TIMING_LIMITS,
        )
        .is_err());
        assert!(validate_conversion_timing(
            ConversionTiming {
                extra_convert_ms: CONVERSION_TIMING_LIMITS.max_extra_convert_ms + 1,
                ..valid_timing()
            },
            CONVERSION_TIMING_LIMITS,
        )
        .is_err());
    }

    #[test]
    fn rejects_excessive_output_context_sum() {
        assert!(validate_conversion_timing(
            ConversionTiming {
                crossfade_ms: 1000,
                sola_search_ms: 1000,
                tail_discard_ms: 1001,
                ..valid_timing()
            },
            CONVERSION_TIMING_LIMITS,
        )
        .is_err());
    }
}
