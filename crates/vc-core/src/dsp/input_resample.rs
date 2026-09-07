//! Fixed model-input timeline over rubato's burst-producing streaming output.
//!
//! Owned by the inference worker. Initialization fixes the content offset on
//! every subsequent hop, including after silence; skipping downstream silent
//! output does not reset this timeline. Arbitrary call boundaries must never
//! insert, discard, duplicate, or reprocess speech samples.

use anyhow::{bail, Result};

use super::{fixed_hop::FixedHop, StreamingResampleMono};
use rubato::Resampler;

pub(crate) struct FixedInputResampler {
    resampler: StreamingResampleMono,
    pending_output: Vec<f32>,
    pending_start: usize,
    delay_samples: usize,
    hop: FixedHop,
}

impl FixedInputResampler {
    pub(crate) fn new(
        input_rate: usize,
        output_rate: usize,
        input_hop: usize,
        output_hop: usize,
    ) -> Result<Self> {
        let hop = FixedHop::new(input_rate, output_rate, input_hop, output_hop)?;
        let resampler = StreamingResampleMono::for_fixed_hop(input_rate, output_rate)?;
        let delay_samples = match resampler.resampler.as_ref() {
            Some(fft) => hop.delay(
                fft.input_frames_max(),
                fft.fft_size_in(),
                fft.fft_size_out(),
                fft.output_delay(),
            )?,
            None => 0,
        };
        Ok(Self {
            resampler,
            pending_output: vec![0.0; delay_samples],
            pending_start: 0,
            delay_samples,
            hop,
        })
    }

    /// Fixed content delay in output samples, available before the first input.
    /// Zero for matched rates. Finite conversion feeds zeros on the same input
    /// timeline until this delayed content reaches the model, then trims once.
    pub(crate) fn delay_samples(&self) -> usize {
        self.delay_samples
    }

    /// Append exactly the nominal model increment. The caller's timing plan
    /// must ensure consecutive requests equal the rational input duration.
    pub(crate) fn process_into(
        &mut self,
        input: &[f32],
        output_samples: usize,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.hop.validate(input.len(), output_samples)?;
        self.resampler
            .process_into(input, &mut self.pending_output)?;
        let available = self.pending_output.len() - self.pending_start;
        if available < output_samples {
            // An underrun is an invalid timing contract or a changed resampler
            // invariant. Zero-filling here would hide lost/shifted phonemes.
            bail!(
                "input resampler timeline underrun: need {output_samples} samples, have {available}"
            );
        }
        let end = self.pending_start + output_samples;
        output.extend_from_slice(&self.pending_output[self.pending_start..end]);
        self.pending_start = end;
        if end == self.pending_output.len() {
            self.pending_output.clear();
            self.pending_start = 0;
        } else if end >= 4096 && end >= self.pending_output.len() / 2 {
            self.pending_output.drain(..end);
            self.pending_start = 0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_increments_preserve_continuous_signal_across_resampler_phases() {
        for input_rate in [16_000, 32_000, 44_100, 48_000, 96_000] {
            for chunk_ms in [20, 30, 100, 200, 600] {
                let input_chunk = input_rate * chunk_ms / 1000;
                let output_chunk = 16_000 * chunk_ms / 1000;
                let mut fixed =
                    FixedInputResampler::new(input_rate, 16_000, input_chunk, output_chunk)
                        .unwrap();
                let mut raw = StreamingResampleMono::for_fixed_hop(input_rate, 16_000).unwrap();
                let mut legacy = StreamingResampleMono::new(input_rate, 16_000).unwrap();
                let mut fixed_signal = Vec::new();
                let mut raw_signal = Vec::new();
                let mut legacy_signal = Vec::new();
                let mut input = vec![0.0; input_chunk];
                // Traverse both modes' complete phase cycles, including the
                // former Input mode's 160-call cycle at 44.1 kHz / 30 ms.
                let period = fixed.resampler.resampler.as_ref().map_or(1, |fft| {
                    fixed
                        .hop
                        .phase_period(fft.input_frames_max(), fft.fft_size_in())
                        .unwrap()
                });
                let period = period.max(legacy.resampler.as_ref().map_or(1, |fft| {
                    fixed
                        .hop
                        .phase_period(fft.input_frames_max(), fft.fft_size_in())
                        .unwrap()
                }));
                let calls = 2 * period + 3;
                let mut maximum_deficit = 0;
                for call in 0..calls {
                    for (i, sample) in input.iter_mut().enumerate() {
                        let position = (call * input_chunk + i) as f64 / input_rate as f64;
                        *sample = ((position * 1003.0).sin() * 0.4 + (position * 173.0).cos() * 0.2)
                            as f32;
                    }
                    raw.process_into(&input, &mut raw_signal).unwrap();
                    legacy.process_into(&input, &mut legacy_signal).unwrap();
                    maximum_deficit = maximum_deficit
                        .max(((call + 1) * output_chunk).saturating_sub(raw_signal.len()));
                    let start = fixed_signal.len();
                    fixed
                        .process_into(&input, output_chunk, &mut fixed_signal)
                        .unwrap();
                    assert_eq!(fixed_signal.len() - start, output_chunk);
                    let queued = fixed.pending_output.len() - fixed.pending_start;
                    assert!(queued <= fixed.delay_samples());
                }
                let delay = fixed.delay_samples();
                assert_eq!(
                    delay, maximum_deficit,
                    "preload must be minimal for {input_rate} Hz / {chunk_ms} ms"
                );
                assert!(fixed_signal[..delay].iter().all(|sample| *sample == 0.0));
                let content = &fixed_signal[delay..];
                assert_eq!(
                    content,
                    &raw_signal[..content.len()],
                    "{input_rate} Hz {chunk_ms} ms"
                );
                // Input mode can still be holding already-computable content.
                // Drain its batching delay, then compare the complete waveform
                // rather than requiring identical per-call production lengths.
                input.fill(0.0);
                while legacy_signal.len() < content.len() {
                    legacy.process_into(&input, &mut legacy_signal).unwrap();
                }
                assert_eq!(content, &legacy_signal[..content.len()]);
                assert_eq!(delay == 0, input_rate == 16_000);
            }
        }
    }

    #[test]
    fn both_preserves_fft_and_filter_and_removes_44100_batch_delay() {
        for (rate, expected_delay) in [(32_000, 280), (44_100, 160), (48_000, 80), (96_000, 40)] {
            let fixed = FixedInputResampler::new(rate, 16_000, rate / 5, 3200).unwrap();
            let generic = StreamingResampleMono::new(rate, 16_000).unwrap();
            let previous = generic.resampler.as_ref().unwrap();
            let current = fixed.resampler.resampler.as_ref().unwrap();
            assert_eq!(current.fft_size_in(), previous.fft_size_in());
            assert_eq!(current.fft_size_out(), previous.fft_size_out());
            assert_eq!(current.cutoff(), previous.cutoff());
            assert_eq!(current.output_delay(), previous.output_delay());
            assert_eq!(current.input_frames_next(), current.fft_size_in());
            assert_eq!(previous.input_frames_next(), 480);
            assert_eq!(fixed.delay_samples(), expected_delay);
            if rate == 44_100 {
                assert_eq!(current.input_frames_next(), 882);
                let old_delay = fixed
                    .hop
                    .delay(
                        previous.input_frames_max(),
                        previous.fft_size_in(),
                        previous.fft_size_out(),
                        previous.output_delay(),
                    )
                    .unwrap();
                assert_eq!(old_delay - fixed.delay_samples(), 320);
            }
        }
    }

    #[test]
    fn impulse_and_final_tail_land_at_reported_delay() {
        for input_rate in [16_000, 32_000, 44_100, 48_000, 96_000] {
            let input_chunk = input_rate / 50;
            let output_chunk = 320;
            let mut fixed =
                FixedInputResampler::new(input_rate, 16_000, input_chunk, output_chunk).unwrap();
            let mut signal = vec![0.0; input_chunk * 10];
            signal[0] = 1.0;
            // A second impulse near finite input end tests the zero-fed drain,
            // independent of the model/SOLA finite finalizer.
            signal[input_rate / 5 - input_rate / 1000] = 1.0;
            let mut output = Vec::new();
            for chunk in signal.chunks(input_chunk) {
                fixed
                    .process_into(chunk, output_chunk, &mut output)
                    .unwrap();
            }
            let content_len = output.len();
            let zeros = vec![0.0; input_chunk];
            while output.len() < content_len + fixed.delay_samples() {
                fixed
                    .process_into(&zeros, output_chunk, &mut output)
                    .unwrap();
            }
            let first_peak = output[..fixed.delay_samples() + 16]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                .unwrap()
                .0;
            assert_eq!(first_peak, fixed.delay_samples(), "rate {input_rate}");
            let expected_last = fixed.delay_samples() + content_len - 16;
            let last_peak = output[expected_last - 8..expected_last + 8]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                .unwrap()
                .0;
            assert_eq!(last_peak, 8, "rate {input_rate}");
        }
    }

    #[test]
    fn rebuilding_discards_previous_stream_content() {
        let mut used = FixedInputResampler::new(44_100, 16_000, 882, 320).unwrap();
        used.process_into(&vec![0.7; 882], 320, &mut Vec::new())
            .unwrap();
        used = FixedInputResampler::new(44_100, 16_000, 4410, 1600).unwrap();
        let mut fresh = FixedInputResampler::new(44_100, 16_000, 4410, 1600).unwrap();
        let input = vec![0.2; 4410];
        let mut resumed = Vec::new();
        let mut expected = Vec::new();
        used.process_into(&input, 1600, &mut resumed).unwrap();
        fresh.process_into(&input, 1600, &mut expected).unwrap();
        assert_eq!(resumed, expected);
    }

    #[test]
    fn fixed_hop_rejects_changes_before_consuming_and_saves_20_ms_at_48k() {
        for rate in [16_000, 48_000] {
            let mut used = FixedInputResampler::new(rate, 16_000, rate / 5, 3200).unwrap();
            let mut fresh = FixedInputResampler::new(rate, 16_000, rate / 5, 3200).unwrap();
            assert_eq!(used.delay_samples(), if rate == 48_000 { 80 } else { 0 });
            if rate == 48_000 {
                assert_eq!(
                    used.resampler.fixed_output_delay_samples() - used.delay_samples(),
                    320
                );
            }
            let mut out = vec![7.0];
            assert!(used
                .process_into(&vec![1.0; rate / 10], 1600, &mut out)
                .is_err());
            assert!(used
                .process_into(&vec![1.0; rate / 5], 3199, &mut out)
                .is_err());
            assert_eq!(out, [7.0]);
            out.clear();
            let mut expected = Vec::new();
            let input = vec![0.2; rate / 5];
            used.process_into(&input, 3200, &mut out).unwrap();
            fresh.process_into(&input, 3200, &mut expected).unwrap();
            assert_eq!(out, expected);
        }
        assert!(FixedInputResampler::new(48_000, 16_000, 960, 319).is_err());
    }

    #[test]
    fn smaller_preload_advances_content_inside_unchanged_200ms_hops() {
        let mut current = FixedInputResampler::new(48_000, 16_000, 9600, 3200).unwrap();
        let mut legacy = FixedInputResampler::new(48_000, 16_000, 9600, 3200).unwrap();
        // Reproduce only the former constructor's FIFO preload. FFT processing,
        // filter coefficients, input calls and output reads stay identical.
        legacy.delay_samples = legacy.resampler.fixed_output_delay_samples();
        legacy.pending_output.resize(legacy.delay_samples, 0.0);
        assert_eq!((legacy.delay_samples(), current.delay_samples()), (400, 80));

        let mut input = vec![0.0; 9600];
        let mut old_signal = Vec::new();
        let mut new_signal = Vec::new();
        // Ten seconds of silence must not be confused with resetting the input
        // FIFO. The realtime worker may skip enqueueing a silent output chunk,
        // but that downstream decision cannot remove this retained input lag.
        for hop in 0..54 {
            let has_impulse = hop == 0 || hop >= 51;
            input.fill(0.0);
            if has_impulse {
                input[4800] = 1.0; // Impulse at 100 ms within the 200 ms hop.
            }
            legacy.process_into(&input, 3200, &mut old_signal).unwrap();
            current.process_into(&input, 3200, &mut new_signal).unwrap();
            assert_eq!(old_signal.len(), (hop + 1) * 3200);
            assert_eq!(new_signal.len(), old_signal.len());
            let peak = |signal: &[f32]| {
                signal[hop * 3200..]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                    .unwrap()
                    .0
            };
            if has_impulse {
                assert_eq!((peak(&old_signal), peak(&new_signal)), (2000, 1680));
            } else {
                assert!(old_signal[hop * 3200..].iter().all(|&sample| sample == 0.0));
                assert!(new_signal[hop * 3200..].iter().all(|&sample| sample == 0.0));
            }
            assert_eq!(legacy.pending_output.len() - legacy.pending_start, 320);
            assert_eq!(current.pending_output.len() - current.pending_start, 0);
        }
        // Same waveform shifted by 320 samples (20 ms at 16 kHz), including
        // subsequent hops. This is retained content, not a different call rate
        // or merely a smaller reported delay with unchanged audio.
        assert_eq!(&old_signal[320..], &new_signal[..new_signal.len() - 320]);
    }
}
