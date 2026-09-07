use std::collections::VecDeque;

use anyhow::{ensure, Result};
use rubato::audioadapter_buffers::direct::SequentialSlice;
use rubato::{Fft, FixedSync, Resampler, WindowFunction};

/// Worker-side model-output resampling. Only committed, non-overlapping joined
/// samples belong here: feeding each candidate's search/crossfade margin again
/// would repeat audio and corrupt the filter timeline.
pub(crate) struct OutputResampler {
    from_hz: usize,
    to_hz: usize,
    resampler: Option<Fft<f32>>,
    input_block: Vec<f32>,
    input_block_samples: usize,
    output_scratch: Vec<f32>,
    fifo: VecDeque<f32>,
    delay_samples: usize,
    discard_output: usize,
    total_input: u64,
    total_taken: u64,
    finished: bool,
}

impl OutputResampler {
    pub(crate) fn new(from_hz: usize, to_hz: usize, max_input_chunk: usize) -> Result<Self> {
        ensure!(
            from_hz > 0 && to_hz > 0,
            "output sample rates must be positive"
        );
        // Match resample_mono's actual FFT partitioning and anti-aliasing window;
        // the input-side streaming resampler intentionally has other settings.
        let resampler = if from_hz == to_hz {
            None
        } else {
            Some(Fft::<f32>::new_custom(
                from_hz,
                to_hz,
                1024,
                1,
                1,
                WindowFunction::BlackmanHarris2,
                FixedSync::Both,
            )?)
        };
        let input_block_samples = resampler.as_ref().map_or(0, |r| r.input_frames_next());
        let output_block_samples = resampler.as_ref().map_or(0, |r| r.output_frames_next());
        let discard_output = resampler.as_ref().map_or(0, |r| r.output_delay());
        // Both fixes B input -> O output, with O/B == to/from. After N input
        // samples fewer than O output samples can be waiting for a full block.
        // Trimming the filter delay D once and preloading D+O zeros therefore
        // guarantees fixed-duration reads forever, not just for the first hop:
        // (D+O) + floor(N/B)*O - D >= N*to/from. Never conceal a later deficit
        // with fresh zeros; it indicates a broken hop-duration contract.
        let delay_samples = discard_output + output_block_samples;
        let max_output_chunk =
            (max_input_chunk as u128 * to_hz as u128).div_ceil(from_hz as u128) as usize;
        let mut fifo =
            VecDeque::with_capacity(delay_samples + max_output_chunk + output_block_samples);
        fifo.resize(delay_samples, 0.0);
        Ok(Self {
            from_hz,
            to_hz,
            resampler,
            input_block: Vec::with_capacity(input_block_samples),
            input_block_samples,
            output_scratch: vec![0.0; output_block_samples],
            fifo,
            delay_samples,
            discard_output,
            total_input: 0,
            total_taken: 0,
            finished: false,
        })
    }

    /// Content delay in output samples; includes incomplete-block buffering as
    /// well as the filter delay, and excludes the model and joiner's delays.
    pub(crate) fn delay_samples(&self) -> usize {
        self.delay_samples
    }

    pub(crate) fn process_fixed(
        &mut self,
        input: &[f32],
        output_samples: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        ensure!(
            input.len() as u128 * self.to_hz as u128
                == output_samples as u128 * self.from_hz as u128,
            "model and output chunks must have exactly the same duration"
        );
        self.append(input)?;
        self.take(output_samples, out)
    }

    pub(crate) fn append(&mut self, mut input: &[f32]) -> Result<()> {
        ensure!(!self.finished, "output resampler is already finished");
        self.total_input += input.len() as u64;
        if self.resampler.is_none() {
            self.fifo.extend(input.iter().copied());
            return Ok(());
        }
        while !input.is_empty() {
            let count = input
                .len()
                .min(self.input_block_samples - self.input_block.len());
            self.input_block.extend_from_slice(&input[..count]);
            input = &input[count..];
            if self.input_block.len() == self.input_block_samples {
                self.process_block()?;
            }
        }
        Ok(())
    }

    fn process_block(&mut self) -> Result<()> {
        let resampler = self.resampler.as_mut().expect("resampling enabled");
        let input = SequentialSlice::new(&self.input_block, 1, self.input_block_samples)?;
        let output_len = self.output_scratch.len();
        let mut output = SequentialSlice::new_mut(&mut self.output_scratch, 1, output_len)?;
        let (consumed, produced) = resampler.process_into_buffer(&input, &mut output, None)?;
        ensure!(
            consumed == self.input_block_samples && produced == output_len,
            "fixed output resampler changed its block size"
        );
        let skip = self.discard_output.min(produced);
        self.discard_output -= skip;
        self.fifo
            .extend(self.output_scratch[skip..produced].iter().copied());
        self.input_block.clear();
        Ok(())
    }

    fn take(&mut self, samples: usize, out: &mut Vec<f32>) -> Result<()> {
        ensure!(
            self.fifo.len() >= samples,
            "output resampler FIFO underflow"
        );
        out.clear();
        out.extend(self.fifo.drain(..samples));
        self.total_taken += samples as u64;
        Ok(())
    }

    /// Drains exactly the delayed finite input, including a partial last FFT
    /// block. EOF padding is only filter support and is not counted as input.
    /// The returned stream still includes `delay_samples()` initial zeros, so
    /// the finite owner must crop that delay exactly once. A second finish is
    /// empty; no tail can be emitted twice.
    pub(crate) fn finish(&mut self, out: &mut Vec<f32>) -> Result<()> {
        out.clear();
        if self.finished {
            return Ok(());
        }
        if self.total_input == 0 {
            self.fifo.clear();
            self.finished = true;
            return Ok(());
        }
        let total_output = (self.total_input as u128 * self.to_hz as u128)
            .div_ceil(self.from_hz as u128)
            + self.delay_samples as u128;
        let remaining = usize::try_from(total_output - self.total_taken as u128)?;
        while self.fifo.len() < remaining {
            ensure!(self.resampler.is_some(), "same-rate output FIFO underflow");
            self.input_block.resize(self.input_block_samples, 0.0);
            self.process_block()?;
        }
        self.take(remaining, out)?;
        self.fifo.clear();
        self.finished = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Explicit full blocks provide a reference independent of this adapter's
    // partial-input/FIFO/EOF bookkeeping and rubato's finite helper (which loses
    // one startup sample for odd output blocks in rubato 5.0).
    fn continuous_block_reference(signal: &[f32], from: usize, to: usize) -> Vec<f32> {
        let mut fft = Fft::<f32>::new_custom(
            from,
            to,
            1024,
            1,
            1,
            WindowFunction::BlackmanHarris2,
            FixedSync::Both,
        )
        .unwrap();
        let input_block = fft.input_frames_next();
        let output_block = fft.output_frames_next();
        let delay = fft.output_delay();
        let target = (signal.len() * to).div_ceil(from);
        let block_count = (target + delay).div_ceil(output_block);
        let mut padded = signal.to_vec();
        padded.resize(block_count * input_block, 0.0);
        let mut raw = Vec::new();
        let mut block = vec![0.0; output_block];
        for input in padded.chunks_exact(input_block) {
            fft.process_into_buffer(
                &SequentialSlice::new(input, 1, input_block).unwrap(),
                &mut SequentialSlice::new_mut(&mut block, 1, output_block).unwrap(),
                None,
            )
            .unwrap();
            raw.extend_from_slice(&block);
        }
        raw[delay..delay + target].to_vec()
    }

    #[test]
    fn output_stream_matches_one_shot_filter_and_recovers_partial_tail() {
        for (from, to) in [
            (32_000, 48_000),
            (40_000, 48_000),
            (48_000, 44_100),
            (44_100, 48_000),
        ] {
            // A noninteger number of FFT blocks and output samples tests the EOF
            // phase, with impulses away from zero and on the final input sample.
            let mut signal: Vec<f32> = (0..from / 3 + 17)
                .map(|i| {
                    let t = i as f64 / from as f64;
                    (0.2 * (t * 2.0 * std::f64::consts::PI * 997.0).sin()
                        + 0.07 * (t * 2.0 * std::f64::consts::PI * 7103.0).sin())
                        as f32
                })
                .collect();
            signal[101] += 0.5;
            *signal.last_mut().unwrap() += 0.7;
            let reference = continuous_block_reference(&signal, from, to);
            let mut resampler = OutputResampler::new(from, to, 137).unwrap();
            for chunk in signal.chunks(137) {
                resampler.append(chunk).unwrap();
            }
            let mut actual = Vec::new();
            resampler.finish(&mut actual).unwrap();
            let actual = &actual[resampler.delay_samples()..];
            assert_eq!(actual.len(), reference.len(), "{from}->{to}");
            let (max_index, max_error) = actual
                .iter()
                .zip(&reference)
                .enumerate()
                .map(|(i, (a, b))| (i, (a - b).abs()))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            assert!(
                max_error < 1e-6,
                "{from}->{to}: max error {max_error} at {max_index}"
            );
            let mut second_finish = vec![9.0];
            resampler.finish(&mut second_finish).unwrap();
            assert!(second_finish.is_empty());
            assert!(resampler.append(&[1.0]).is_err());
        }
    }

    #[test]
    fn output_stream_has_no_dc_seams_or_unbounded_fifo_at_supported_hops() {
        for (from, to) in [
            (32_000, 48_000),
            (40_000, 48_000),
            (48_000, 44_100),
            (44_100, 48_000),
        ] {
            for ms in [20, 100, 500, 2000] {
                let input_hop = from * ms / 1000;
                let output_hop = to * ms / 1000;
                let mut resampler = OutputResampler::new(from, to, input_hop).unwrap();
                let capacity = resampler.fifo.capacity();
                let input = vec![0.25; input_hop];
                let mut out = Vec::new();
                for index in 0..40 {
                    resampler
                        .process_fixed(&input, output_hop, &mut out)
                        .unwrap();
                    assert_eq!(out.len(), output_hop);
                    assert_eq!(resampler.fifo.capacity(), capacity, "{from}->{to}, {ms}ms");
                    assert!(resampler.fifo.len() <= resampler.delay_samples());
                    if index * output_hop >= resampler.delay_samples() + to / 10 {
                        let error = out.iter().map(|v| (v - 0.25).abs()).fold(0.0, f32::max);
                        assert!(error < 1e-4, "{from}->{to}, {ms}ms: {error}");
                    }
                }
            }
        }
    }

    #[test]
    fn same_rate_is_exact_and_bad_hop_does_not_consume_audio() {
        let mut resampler = OutputResampler::new(48_000, 48_000, 3).unwrap();
        assert_eq!(resampler.delay_samples(), 0);
        let mut out = Vec::new();
        assert!(resampler
            .process_fixed(&[1.0, 2.0, 3.0], 2, &mut out)
            .is_err());
        resampler
            .process_fixed(&[1.0, 2.0, 3.0], 3, &mut out)
            .unwrap();
        assert_eq!(out, [1.0, 2.0, 3.0]);
        resampler.append(&[4.0, 5.0]).unwrap();
        resampler.finish(&mut out).unwrap();
        assert_eq!(out, [4.0, 5.0]);
    }
}
