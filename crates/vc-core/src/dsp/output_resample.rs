use std::collections::VecDeque;

use anyhow::{ensure, Result};
use rubato::audioadapter_buffers::direct::SequentialSlice;
use rubato::{Fft, FixedSync, Resampler, WindowFunction};

use super::fixed_hop::FixedHop;

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
    fixed_hop: Option<FixedHop>,
}

impl OutputResampler {
    pub(crate) fn new(from_hz: usize, to_hz: usize, max_input_chunk: usize) -> Result<Self> {
        Self::build(from_hz, to_hz, max_input_chunk, None)
    }

    pub(crate) fn new_fixed(
        from_hz: usize,
        to_hz: usize,
        input_hop: usize,
        output_hop: usize,
    ) -> Result<Self> {
        let hop = FixedHop::new(from_hz, to_hz, input_hop, output_hop)?;
        Self::build(from_hz, to_hz, input_hop, Some(hop))
    }

    fn build(
        from_hz: usize,
        to_hz: usize,
        max_input_chunk: usize,
        fixed_hop: Option<FixedHop>,
    ) -> Result<Self> {
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
        // The generic append/finish adapter keeps its conservative bound. Only
        // lifetime-fixed reads may use the smaller phase-specific preload.
        let delay_samples = match (fixed_hop, resampler.as_ref()) {
            (Some(hop), Some(fft)) => hop.delay(
                input_block_samples,
                fft.fft_size_in(),
                fft.fft_size_out(),
                discard_output,
            )?,
            _ => discard_output + output_block_samples,
        };
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
            fixed_hop,
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
        if let Some(hop) = self.fixed_hop {
            hop.validate(input.len(), output_samples)?;
        }
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
        if let Some(hop) = self.fixed_hop {
            hop.validate(input.len(), hop.output)?;
        }
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

    /// Restart an isolated finite window without rebuilding FFT plans/scratch.
    /// RMS references overlap across calls: carrying filter state between them
    /// would process repeated content as new audio and alter the gain envelope.
    fn reset(&mut self) {
        if let Some(fft) = self.resampler.as_mut() {
            fft.reset();
        }
        self.input_block.clear();
        self.fifo.clear();
        self.fifo.resize(self.delay_samples, 0.0);
        self.discard_output = self.resampler.as_ref().map_or(0, |r| r.output_delay());
        self.total_input = 0;
        self.total_taken = 0;
        self.finished = false;
    }
}

/// Reusable worker-side scratch for *independent*, possibly overlapping finite
/// windows. Same filter/EOF semantics as `resample_mono`; equal-rate calls copy.
/// This must not replace the persistent resampler of committed output audio.
#[derive(Default)]
pub struct ResampleMonoScratch {
    resampler: Option<OutputResampler>,
}

impl ResampleMonoScratch {
    pub fn process_into(
        &mut self,
        input: &[f32],
        from_hz: usize,
        to_hz: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        ensure!(
            from_hz > 0 && to_hz > 0,
            "resampler sample rates must be positive"
        );
        out.clear();
        if input.is_empty() {
            return Ok(());
        }
        if from_hz == to_hz {
            out.extend_from_slice(input);
            return Ok(());
        }
        if !self
            .resampler
            .as_ref()
            .is_some_and(|r| r.from_hz == from_hz && r.to_hz == to_hz)
        {
            self.resampler = Some(OutputResampler::new(from_hz, to_hz, input.len())?);
        }
        let resampler = self.resampler.as_mut().expect("resampler initialized");
        resampler.reset();
        resampler.append(input)?;
        resampler.finish(out)?;
        let delay = resampler.delay_samples();
        out.copy_within(delay.., 0);
        out.truncate(out.len() - delay);
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
            (32_000, 44_100),
            (32_000, 48_000),
            (40_000, 44_100),
            (40_000, 48_000),
            (48_000, 44_100),
            (48_000, 48_000),
            (44_100, 48_000),
        ] {
            for ms in [20, 30, 100, 200, 600] {
                let input_hop = from * ms / 1000;
                let output_hop = to * ms / 1000;
                let mut resampler =
                    OutputResampler::new_fixed(from, to, input_hop, output_hop).unwrap();
                let capacity = resampler.fifo.capacity();
                let input = vec![0.25; input_hop];
                let mut out = Vec::new();
                let period = resampler.resampler.as_ref().map_or(1, |fft| {
                    resampler
                        .fixed_hop
                        .unwrap()
                        .phase_period(resampler.input_block_samples, fft.fft_size_in())
                        .unwrap()
                });
                for index in 0..(2 * period + 10) {
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

    #[test]
    fn fixed_output_preserves_waveform_and_tail_at_every_phase() {
        for from in [32_000, 40_000, 48_000] {
            for to in [44_100, 48_000] {
                for ms in [20, 30, 100, 200, 600] {
                    let ih = from * ms / 1000;
                    let oh = to * ms / 1000;
                    let mut fixed = OutputResampler::new_fixed(from, to, ih, oh).unwrap();
                    let period = fixed.resampler.as_ref().map_or(1, |fft| {
                        fixed
                            .fixed_hop
                            .unwrap()
                            .phase_period(fixed.input_block_samples, fft.fft_size_in())
                            .unwrap()
                    });
                    let mut signal: Vec<f32> = (0..ih * (2 * period + 3))
                        .map(|i| ((i as f64 * 0.071).sin() * 0.2) as f32)
                        .collect();
                    signal[0] = 1.0;
                    *signal.last_mut().unwrap() = 0.9;
                    let expected = if from == to {
                        signal.clone()
                    } else {
                        continuous_block_reference(&signal, from, to)
                    };
                    let mut actual = Vec::new();
                    let mut out = Vec::new();
                    let mut max_deficit = 0;
                    for chunk in signal.chunks_exact(ih) {
                        fixed.process_fixed(chunk, oh, &mut out).unwrap();
                        actual.extend_from_slice(&out);
                        let produced =
                            fixed.total_taken as usize + fixed.fifo.len() - fixed.delay_samples();
                        max_deficit =
                            max_deficit.max((fixed.total_taken as usize).saturating_sub(produced));
                    }
                    assert_eq!(
                        fixed.delay_samples(),
                        max_deficit,
                        "minimal preload {from}->{to} / {ms}"
                    );
                    fixed.finish(&mut out).unwrap();
                    actual.extend_from_slice(&out);
                    assert_eq!(
                        &actual[fixed.delay_samples()..],
                        expected,
                        "{from}->{to} / {ms}"
                    );
                }
            }
        }
    }

    #[test]
    fn fixed_output_rejects_equal_duration_hop_changes_before_mutation() {
        for (from, to) in [(40_000, 48_000), (48_000, 48_000)] {
            let ih = from / 5;
            let oh = to / 5;
            let mut used = OutputResampler::new_fixed(from, to, ih, oh).unwrap();
            let mut fresh = OutputResampler::new_fixed(from, to, ih, oh).unwrap();
            let mut actual = vec![7.0];
            assert!(used
                .process_fixed(&vec![1.0; ih / 2], oh / 2, &mut actual)
                .is_err());
            assert!(used.append(&vec![1.0; ih / 2]).is_err());
            assert_eq!(actual, [7.0]);
            assert_eq!(used.total_input, 0);
            let mut expected = Vec::new();
            used.process_fixed(&vec![0.2; ih], oh, &mut actual).unwrap();
            fresh
                .process_fixed(&vec![0.2; ih], oh, &mut expected)
                .unwrap();
            assert_eq!(actual, expected);
            used.reset();
            fresh.reset();
            used.process_fixed(&vec![0.3; ih], oh, &mut actual).unwrap();
            fresh
                .process_fixed(&vec![0.3; ih], oh, &mut expected)
                .unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn finite_scratch_matches_independent_windows_and_keeps_fft_storage() {
        let mut scratch = ResampleMonoScratch::default();
        let mut out = Vec::new();
        for (from, to) in [
            (16_000, 32_000),
            (16_000, 40_000),
            (16_000, 48_000),
            (48_000, 44_100),
        ] {
            for len in [1, 7, 1023, 1024, 1025, 4912, 0, 3] {
                let signal: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.17).sin()) * 0.3).collect();
                let expected = if len == 0 {
                    Vec::new()
                } else {
                    continuous_block_reference(&signal, from, to)
                };
                scratch.process_into(&signal, from, to, &mut out).unwrap();
                assert_eq!(out.len(), expected.len());
                assert!(out.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-6));
                if len > 0 {
                    let r = scratch.resampler.as_ref().unwrap();
                    let storage = (r.input_block.as_ptr(), r.output_scratch.as_ptr());
                    scratch.process_into(&signal, from, to, &mut out).unwrap();
                    let r = scratch.resampler.as_ref().unwrap();
                    assert_eq!(storage, (r.input_block.as_ptr(), r.output_scratch.as_ptr()));
                }
            }
        }
        scratch
            .process_into(&[0.3, 0.7], 48_000, 48_000, &mut out)
            .unwrap();
        assert_eq!(out, [0.3, 0.7]);
    }
}
