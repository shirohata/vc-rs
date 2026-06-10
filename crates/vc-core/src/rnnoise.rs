//! Fixed-delay streaming adapter for the pure-Rust `nnnoiseless` RNNoise port.
//!
//! RNNoise consumes 480-sample frames at 48 kHz in i16-scale `f32`. The RVC
//! pipeline accepts arbitrary device rates and chunk boundaries, so this adapter
//! owns both streaming resamplers, frame accumulation, and a delayed output
//! timeline. Keep those states continuous across calls: resetting them per RVC
//! chunk creates audible seams and changes the feature/F0 time grid.

use anyhow::{bail, Result};
use nnnoiseless::DenoiseState;

use crate::dsp::StreamingResampleMono;

const RNNOISE_SAMPLE_RATE: usize = 48_000;
const PCM_SCALE: f32 = 32_768.0;

pub struct RnnoiseDenoiser {
    sample_rate: usize,
    state: Box<DenoiseState<'static>>,
    to_rnnoise: StreamingResampleMono,
    from_rnnoise: StreamingResampleMono,
    rnnoise_input: Vec<f32>,
    rnnoise_input_start: usize,
    rnnoise_output: Vec<f32>,
    device_output: Vec<f32>,
    device_output_start: usize,
    frame_input: [f32; DenoiseState::FRAME_SIZE],
    frame_output: [f32; DenoiseState::FRAME_SIZE],
    latency_samples: usize,
}

impl RnnoiseDenoiser {
    pub fn new(sample_rate: u32) -> Result<Self> {
        let sample_rate = usize::try_from(sample_rate)?;
        if sample_rate == 0 {
            bail!("RNNoise sample rate must be greater than zero");
        }

        // Rubato's streaming adapter operates in 1024-sample batches. Prime a
        // conservative, fixed timeline that covers both resampler batches and
        // RNNoise frame scheduling. This work is
        // on the inference worker, never the realtime audio callback.
        let resampler_input_delay: usize = if sample_rate == RNNOISE_SAMPLE_RATE {
            0
        } else {
            1024
        };
        let rnnoise_domain_delay = 1024 + 2 * DenoiseState::FRAME_SIZE;
        let converted_delay = rnnoise_domain_delay
            .saturating_mul(sample_rate)
            .div_ceil(RNNOISE_SAMPLE_RATE);
        let latency_samples = resampler_input_delay.saturating_add(converted_delay);

        let mut state = DenoiseState::new();
        let silent_frame = [0.0; DenoiseState::FRAME_SIZE];
        let mut warmup_output = [0.0; DenoiseState::FRAME_SIZE];
        // Discard a silent warmup frame, not the caller's first speech frame.
        // This avoids nnnoiseless' documented fade-in artifact without losing
        // the first 10 ms of real input.
        state.process_frame(&mut warmup_output, &silent_frame);

        Ok(Self {
            sample_rate,
            state,
            to_rnnoise: StreamingResampleMono::new(sample_rate, RNNOISE_SAMPLE_RATE)?,
            from_rnnoise: StreamingResampleMono::new(RNNOISE_SAMPLE_RATE, sample_rate)?,
            rnnoise_input: Vec::new(),
            rnnoise_input_start: 0,
            rnnoise_output: Vec::new(),
            device_output: vec![0.0; latency_samples],
            device_output_start: 0,
            frame_input: [0.0; DenoiseState::FRAME_SIZE],
            frame_output: [0.0; DenoiseState::FRAME_SIZE],
            latency_samples,
        })
    }

    pub fn latency_samples(&self) -> usize {
        self.latency_samples
    }

    /// Process arbitrary device-rate input while preserving the per-call length.
    pub fn process_in_place(&mut self, samples: &mut [f32]) -> Result<()> {
        self.process_input(samples)?;
        self.emit_exact(samples)
    }

    /// Denoise finite input, remove the streaming delay, and preserve its length.
    pub fn process_finite(input: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let mut denoiser = Self::new(sample_rate)?;
        let mut delayed = Vec::with_capacity(input.len() + denoiser.latency_samples);
        let block = (usize::try_from(sample_rate)? / 20).max(128);

        for chunk in input.chunks(block) {
            let mut out = chunk.to_vec();
            denoiser.process_in_place(&mut out)?;
            delayed.extend_from_slice(&out);
        }

        // Feed bounded zero blocks until the delayed tail corresponding to all
        // finite input samples has reached the output timeline.
        let target = input.len().saturating_add(denoiser.latency_samples);
        while delayed.len() < target {
            let mut zeros = vec![0.0; block.min(target - delayed.len())];
            denoiser.process_in_place(&mut zeros)?;
            delayed.extend_from_slice(&zeros);
        }
        let start = denoiser.latency_samples.min(delayed.len());
        let end = start.saturating_add(input.len()).min(delayed.len());
        let mut aligned = delayed[start..end].to_vec();
        aligned.resize(input.len(), 0.0);
        Ok(aligned)
    }

    fn process_input(&mut self, samples: &[f32]) -> Result<()> {
        self.to_rnnoise
            .process_into(samples, &mut self.rnnoise_input)?;

        while self
            .rnnoise_input
            .len()
            .saturating_sub(self.rnnoise_input_start)
            >= DenoiseState::FRAME_SIZE
        {
            let end = self.rnnoise_input_start + DenoiseState::FRAME_SIZE;
            for (dst, src) in self
                .frame_input
                .iter_mut()
                .zip(&self.rnnoise_input[self.rnnoise_input_start..end])
            {
                let finite = if src.is_finite() { *src } else { 0.0 };
                *dst = (finite.clamp(-1.0, 1.0) * PCM_SCALE).clamp(-32_768.0, 32_767.0);
            }
            self.state
                .process_frame(&mut self.frame_output, &self.frame_input);
            self.rnnoise_input_start = end;

            for sample in &mut self.frame_output {
                *sample = if sample.is_finite() {
                    (*sample / PCM_SCALE).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
            }
            self.rnnoise_output.extend_from_slice(&self.frame_output);
        }
        self.compact_rnnoise_input();

        if !self.rnnoise_output.is_empty() {
            self.from_rnnoise
                .process_into(&self.rnnoise_output, &mut self.device_output)?;
            self.rnnoise_output.clear();
        }
        Ok(())
    }

    fn emit_exact(&mut self, samples: &mut [f32]) -> Result<()> {
        let available = self
            .device_output
            .len()
            .saturating_sub(self.device_output_start);
        if available < samples.len() {
            bail!(
                "RNNoise output underrun: need {} device samples, have {} (rate={} Hz)",
                samples.len(),
                available,
                self.sample_rate
            );
        }
        let end = self.device_output_start + samples.len();
        samples.copy_from_slice(&self.device_output[self.device_output_start..end]);
        self.device_output_start = end;
        self.compact_device_output();
        Ok(())
    }

    fn compact_rnnoise_input(&mut self) {
        if self.rnnoise_input_start == self.rnnoise_input.len() {
            self.rnnoise_input.clear();
            self.rnnoise_input_start = 0;
        } else if self.rnnoise_input_start >= 4096
            && self.rnnoise_input_start * 2 >= self.rnnoise_input.len()
        {
            self.rnnoise_input.drain(..self.rnnoise_input_start);
            self.rnnoise_input_start = 0;
        }
    }

    fn compact_device_output(&mut self) {
        if self.device_output_start == self.device_output.len() {
            self.device_output.clear();
            self.device_output_start = 0;
        } else if self.device_output_start >= 4096
            && self.device_output_start * 2 >= self.device_output.len()
        {
            self.device_output.drain(..self.device_output_start);
            self.device_output_start = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_calls_preserve_length_at_common_rates() {
        for rate in [44_100, 48_000, 96_000] {
            let mut denoiser = RnnoiseDenoiser::new(rate).unwrap();
            for len in [1, 127, 480, 777, 2205, 4800] {
                let mut audio = vec![0.0; len];
                denoiser.process_in_place(&mut audio).unwrap();
                assert_eq!(audio.len(), len);
                assert!(audio.iter().all(|x| x.is_finite()));
            }
        }
    }

    #[test]
    fn finite_processing_preserves_length() {
        for rate in [44_100, 48_000, 96_000] {
            for len in [0, 1, 127, 1000, rate as usize / 3] {
                let input = vec![0.0; len];
                let output = RnnoiseDenoiser::process_finite(&input, rate).unwrap();
                assert_eq!(output.len(), input.len());
                assert!(output.iter().all(|x| x.is_finite()));
            }
        }
    }

    #[test]
    fn chunk_partition_does_not_reset_rnnoise_state() {
        let input: Vec<f32> = (0..48_000).map(|i| 0.4 * (i as f32 * 0.04).sin()).collect();
        let mut whole = input.clone();
        RnnoiseDenoiser::new(48_000)
            .unwrap()
            .process_in_place(&mut whole)
            .unwrap();

        let mut split_denoiser = RnnoiseDenoiser::new(48_000).unwrap();
        let mut split = Vec::with_capacity(input.len());
        for chunk in input.chunks(777) {
            let mut output = chunk.to_vec();
            split_denoiser.process_in_place(&mut output).unwrap();
            split.extend_from_slice(&output);
        }
        assert_eq!(whole, split);
    }

    #[test]
    fn finite_processing_keeps_non_silent_signal() {
        let input: Vec<f32> = (0..48_000).map(|i| 0.4 * (i as f32 * 0.04).sin()).collect();
        let output = RnnoiseDenoiser::process_finite(&input, 48_000).unwrap();
        assert!(crate::dsp::rms(&output) > 0.01);
    }
}
