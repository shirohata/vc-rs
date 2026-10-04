use anyhow::{ensure, Result};
use rubato::audioadapter_buffers::direct::SequentialSlice;
use rubato::{Adjustable, Async, FixedAsync, Resampler, SincInterpolationParameters};

/// Maximum device-clock correction. Overload must remain an explicit queue
/// failure, not be disguised as a large speed/pitch change.
pub const MAX_CLOCK_CORRECTION_PPM: f64 = 2_000.0;

/// Worker-side continuous sinc resampling for independent device clocks.
/// Unlike the fixed WAV/host adapter, this consumes one unchanged input hop
/// and emits a variable number of samples. Never round that output back to a
/// nominal hop: rubato retains fractional phase across every call, even when
/// the negotiated input/output rates are identical.
pub struct AdaptiveOutputResampler {
    resampler: Async<f32>,
    input_hop: usize,
    scratch: Vec<f32>,
    nominal_delay: usize,
}

impl AdaptiveOutputResampler {
    pub fn new(from_hz: usize, to_hz: usize, input_hop: usize) -> Result<Self> {
        ensure!(
            from_hz > 0 && to_hz > 0 && input_hop > 0,
            "invalid adaptive resampler format"
        );
        let resampler = Async::new_sinc(
            to_hz as f64 / from_hz as f64,
            // Include the reciprocal lower bound plus floating-point slack:
            // ratio*(1-2000ppm) can round one ULP below ratio/max_relative.
            // Our public setter still enforces the exact +/-2000ppm policy.
            (1.0 / (1.0 - MAX_CLOCK_CORRECTION_PPM * 1e-6)) * (1.0 + 1e-12),
            &SincInterpolationParameters::default(),
            input_hop,
            1,
            FixedAsync::Input,
        )?;
        let nominal_delay = resampler.output_delay();
        let scratch = vec![0.0; resampler.output_frames_max()];
        Ok(Self {
            resampler,
            input_hop,
            scratch,
            nominal_delay,
        })
    }

    pub fn delay_samples(&self) -> usize {
        self.nominal_delay
    }

    pub fn set_correction_ppm(&mut self, ppm: f64) -> Result<()> {
        ensure!(
            ppm.is_finite() && ppm.abs() <= MAX_CLOCK_CORRECTION_PPM,
            "invalid device clock correction"
        );
        self.resampler
            .set_resample_ratio_relative(1.0 + ppm * 1e-6, true)?;
        Ok(())
    }

    pub fn process_into(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<()> {
        ensure!(
            input.len() == self.input_hop,
            "adaptive resampler input hop changed"
        );
        let output_len = self.scratch.len();
        let (consumed, produced) = self.resampler.process_into_buffer(
            &SequentialSlice::new(input, 1, input.len())?,
            &mut SequentialSlice::new_mut(&mut self.scratch, 1, output_len)?,
            None,
        )?;
        ensure!(
            consumed == input.len(),
            "adaptive resampler did not consume the hop"
        );
        out.clear();
        out.extend_from_slice(&self.scratch[..produced]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: usize, samples: usize, hz: f64) -> Vec<f32> {
        (0..samples)
            .map(|i| (0.2 * (std::f64::consts::TAU * hz * i as f64 / rate as f64).sin()) as f32)
            .collect()
    }

    #[test]
    fn partitioning_preserves_waveform_and_fractional_phase() {
        for (from, to) in [
            (48_000, 48_000),
            (44_100, 48_000),
            (48_000, 44_100),
            (32_000, 48_000),
            (40_000, 44_100),
        ] {
            let hop = from / 50;
            let mut signal = tone(from, hop * 53, 997.0);
            signal[hop + 7] += 0.7;
            let mut reference = AdaptiveOutputResampler::new(from, to, signal.len()).unwrap();
            let mut expected = Vec::new();
            reference.process_into(&signal, &mut expected).unwrap();
            let mut stream = AdaptiveOutputResampler::new(from, to, hop).unwrap();
            let mut actual = Vec::new();
            let mut out = Vec::new();
            let scratch_capacity = stream.scratch.capacity();
            for chunk in signal.chunks_exact(hop) {
                stream.process_into(chunk, &mut out).unwrap();
                actual.extend_from_slice(&out);
                assert_eq!(stream.scratch.capacity(), scratch_capacity);
            }
            assert!(actual.len().abs_diff(expected.len()) <= 1, "{from}->{to}");
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(error < 1e-5, "{from}->{to}: {error}");
        }
    }

    #[test]
    fn correction_changes_sample_count_without_per_hop_rounding() {
        for ppm in [-500.0, 0.0, 500.0] {
            let mut stream = AdaptiveOutputResampler::new(48_000, 48_000, 960).unwrap();
            stream.set_correction_ppm(ppm).unwrap();
            let mut count = 0;
            let mut out = Vec::new();
            for _ in 0..500 {
                stream.process_into(&[0.25; 960], &mut out).unwrap();
                count += out.len();
                if count > 4_800 {
                    assert!(out.iter().all(|x| (*x - 0.25).abs() < 1e-4));
                }
            }
            let expected = 480_000.0 * (1.0 + ppm * 1e-6);
            assert!(
                (count as f64 - expected).abs() < stream.delay_samples() as f64 + 4.0,
                "{ppm}: {count} vs {expected}"
            );
        }
    }

    #[test]
    fn sinc_preserves_voice_band_and_rejects_downsample_aliasing() {
        for (hz, expected_rms) in [
            (997.0, 0.2 / 2.0_f64.sqrt()),
            (14_000.0, 0.2 / 2.0_f64.sqrt()),
            (22_000.0, 0.0),
        ] {
            let input = tone(48_000, 48_000, hz);
            let mut stream = AdaptiveOutputResampler::new(48_000, 32_000, input.len()).unwrap();
            let mut out = Vec::new();
            stream.process_into(&input, &mut out).unwrap();
            let rms = crate::dsp::rms(&out[1_000..]) as f64;
            assert!((rms - expected_rms).abs() < 0.0015, "{hz}: {rms}");
        }
    }

    #[test]
    fn ramped_correction_has_no_waveform_seams() {
        let input = tone(48_000, 960 * 200, 997.0);
        let mut stream = AdaptiveOutputResampler::new(48_000, 48_000, 960).unwrap();
        let mut joined = Vec::new();
        let mut out = Vec::new();
        for (index, chunk) in input.as_chunks::<960>().0.iter().enumerate() {
            let ppm = if index < 100 {
                index as f64 * 5.0
            } else {
                500.0 - (index - 100) as f64 * 10.0
            };
            stream.set_correction_ppm(ppm).unwrap();
            stream.process_into(chunk, &mut out).unwrap();
            joined.extend_from_slice(&out);
        }
        assert!(joined.iter().all(|v| v.is_finite()));
        assert!(joined[500..]
            .windows(2)
            .all(|w| (w[1] - w[0]).abs() < 0.027));
    }

    #[test]
    fn impulse_delay_matches_reported_latency() {
        for (from, to) in [(48_000, 48_000), (32_000, 48_000), (48_000, 44_100)] {
            let mut input = vec![0.0; from / 50];
            input[0] = 1.0;
            let mut stream = AdaptiveOutputResampler::new(from, to, input.len()).unwrap();
            let mut out = Vec::new();
            stream.process_into(&input, &mut out).unwrap();
            let peak = out
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                .unwrap()
                .0;
            assert!(
                peak.abs_diff(stream.delay_samples()) <= 1,
                "{from}->{to}: peak={peak}, delay={}",
                stream.delay_samples()
            );
        }
    }

    #[test]
    fn full_correction_range_is_accepted_across_nominal_rates() {
        for (from, to) in [
            (32_000, 48_000),
            (40_000, 48_000),
            (44_100, 48_000),
            (48_000, 44_100),
            (48_000, 48_000),
            (44_100, 16_000),
        ] {
            let mut stream = AdaptiveOutputResampler::new(from, to, from / 50).unwrap();
            for ppm in [-MAX_CLOCK_CORRECTION_PPM, MAX_CLOCK_CORRECTION_PPM, 0.0] {
                stream.set_correction_ppm(ppm).unwrap();
            }
        }
    }

    #[test]
    fn invalid_parameters_do_not_consume_audio() {
        let mut stream = AdaptiveOutputResampler::new(48_000, 48_000, 960).unwrap();
        for ppm in [f64::NAN, f64::INFINITY, 2_001.0, -2_001.0] {
            assert!(stream.set_correction_ppm(ppm).is_err());
        }
        let mut actual = Vec::new();
        assert!(stream.process_into(&[1.0; 959], &mut actual).is_err());
        stream.process_into(&[1.0; 960], &mut actual).unwrap();
        let mut reference = AdaptiveOutputResampler::new(48_000, 48_000, 960).unwrap();
        let mut expected = Vec::new();
        reference.process_into(&[1.0; 960], &mut expected).unwrap();
        assert_eq!(actual, expected);
    }
}
