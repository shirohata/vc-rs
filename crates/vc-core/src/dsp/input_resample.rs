//! Fixed model-input timeline over rubato's burst-producing streaming output.
//!
//! Owned by the inference worker. The FIFO contains one startup delay followed
//! by the unmodified continuous resampling result; arbitrary call boundaries
//! must never insert, discard, duplicate, or reprocess speech samples.

use anyhow::{bail, Result};

use super::StreamingResampleMono;

pub(crate) struct FixedInputResampler {
    resampler: StreamingResampleMono,
    pending_output: Vec<f32>,
    pending_start: usize,
    delay_samples: usize,
}

impl FixedInputResampler {
    pub(crate) fn new(input_rate: usize, output_rate: usize) -> Result<Self> {
        let resampler = StreamingResampleMono::new(input_rate, output_rate)?;
        let delay_samples = resampler.fixed_output_delay_samples();
        Ok(Self {
            resampler,
            pending_output: vec![0.0; delay_samples],
            pending_start: 0,
            delay_samples,
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
            for chunk_ms in [20, 30, 100, 500] {
                let input_chunk = input_rate * chunk_ms / 1000;
                let output_chunk = 16_000 * chunk_ms / 1000;
                let mut fixed = FixedInputResampler::new(input_rate, 16_000).unwrap();
                let mut raw = StreamingResampleMono::new(input_rate, 16_000).unwrap();
                let mut fixed_signal = Vec::new();
                let mut raw_signal = Vec::new();
                let mut input = vec![0.0; input_chunk];
                // Include odd 10 ms hops and traverse the full 44.1 kHz batch
                // phase cycle (160 calls for the 30 ms case).
                let calls = if chunk_ms == 30 { 200 } else { 100 };
                for call in 0..calls {
                    for (i, sample) in input.iter_mut().enumerate() {
                        let position = (call * input_chunk + i) as f64 / input_rate as f64;
                        *sample = ((position * 1003.0).sin() * 0.4 + (position * 173.0).cos() * 0.2)
                            as f32;
                    }
                    raw.process_into(&input, &mut raw_signal).unwrap();
                    let start = fixed_signal.len();
                    fixed
                        .process_into(&input, output_chunk, &mut fixed_signal)
                        .unwrap();
                    assert_eq!(fixed_signal.len() - start, output_chunk);
                    let queued = fixed.pending_output.len() - fixed.pending_start;
                    assert!(queued <= fixed.delay_samples());
                }
                let delay = fixed.delay_samples();
                assert!(fixed_signal[..delay].iter().all(|sample| *sample == 0.0));
                let content = &fixed_signal[delay..];
                assert_eq!(
                    content,
                    &raw_signal[..content.len()],
                    "{input_rate} Hz {chunk_ms} ms"
                );
                assert_eq!(delay == 0, input_rate == 16_000);
            }
        }
    }

    #[test]
    fn impulse_and_final_tail_land_at_reported_delay() {
        for input_rate in [16_000, 32_000, 44_100, 48_000, 96_000] {
            let input_chunk = input_rate / 50;
            let output_chunk = 320;
            let mut fixed = FixedInputResampler::new(input_rate, 16_000).unwrap();
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
        let mut used = FixedInputResampler::new(44_100, 16_000).unwrap();
        used.process_into(&vec![0.7; 882], 320, &mut Vec::new())
            .unwrap();
        used = FixedInputResampler::new(44_100, 16_000).unwrap();
        let mut fresh = FixedInputResampler::new(44_100, 16_000).unwrap();
        let input = vec![0.2; 4410];
        let mut resumed = Vec::new();
        let mut expected = Vec::new();
        used.process_into(&input, 1600, &mut resumed).unwrap();
        fresh.process_into(&input, 1600, &mut expected).unwrap();
        assert_eq!(resumed, expected);
    }
}
