//! Model-agnostic fixed-delay streaming adapter shared by every input denoiser.
//!
//! A [`FrameDenoiser`] owns only the model state (its native sample rate, frame
//! size, and per-frame processing). Everything around it — the two streaming
//! resamplers (device rate ↔ model rate), frame accumulation, and the
//! fixed-delay output timeline — lives here so RNNoise, GTCRN, and any future
//! model share one streaming contract. Keep these states continuous across
//! calls: resetting them per RVC chunk creates audible seams and changes the
//! feature/F0 time grid.

use anyhow::{bail, Result};

use crate::dsp::StreamingResampleMono;

// Preserve the historical minimum delay where it already suffices. The actual
// bound below can be larger: the FFT block depends on the rate pair, and the
// wrapper also removes startup filter samples. A fixed 1024-sample estimate
// alone underruns at rates such as 11,025 and 22,050 Hz.
const RESAMPLER_BATCH_DELAY: usize = 1024;

/// A model-specific denoiser that consumes and produces fixed-size frames at a
/// single native sample rate.
///
/// Contract: `process_frame` is strictly 1:1 — exactly `frame_size()` input
/// samples in, `frame_size()` output samples out, every call. A model may
/// *delay* its content (e.g. STFT/overlap-add reconstruction emits samples that
/// correspond to an earlier window) but must still emit one output sample per
/// input sample; report that inherent delay via [`output_delay_frames`] so the
/// adapter surfaces it as latency without ever reshaping the sample grid.
///
/// [`output_delay_frames`]: FrameDenoiser::output_delay_frames
pub(crate) trait FrameDenoiser {
    /// Native sample rate the model frames run at (e.g. 48_000 for RNNoise).
    fn sample_rate(&self) -> usize;
    /// Frame/hop size in samples at [`sample_rate`](FrameDenoiser::sample_rate).
    fn frame_size(&self) -> usize;
    /// Inherent output delay in whole frames (0 if the model is sample-aligned).
    ///
    /// This is content the model *already* withholds in its output stream (it is
    /// not buffered by the adapter); the adapter only reflects it in
    /// [`latency_samples`](FixedDelayAdapter::latency_samples) and the
    /// finite-drain trim position.
    fn output_delay_frames(&self) -> usize;
    /// Process exactly one frame. `input.len() == output.len() == frame_size()`.
    /// Implementations are responsible for sanitizing non-finite samples.
    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<()>;
    /// Restore the model to its post-construction (warmed-up) state.
    fn reset(&mut self) -> Result<()>;
}

/// Wraps a [`FrameDenoiser`] with resampling, frame accumulation, and a
/// fixed-delay output timeline so callers can feed arbitrary device rates and
/// chunk boundaries while the per-call sample count is preserved exactly.
pub(crate) struct FixedDelayAdapter<D: FrameDenoiser> {
    processor: D,
    device_rate: usize,
    // Read only by `reset` (rebuilding the resamplers). `reset` itself has no
    // non-test caller until GTCRN wires it into `reset_streaming_state`
    // (Phase 3), so keep these from tripping dead-code lints meanwhile.
    #[allow(dead_code)]
    model_rate: usize,
    frame_size: usize,
    to_model: StreamingResampleMono,
    from_model: StreamingResampleMono,
    model_input: Vec<f32>,
    model_input_start: usize,
    model_output: Vec<f32>,
    device_output: Vec<f32>,
    device_output_start: usize,
    frame_input: Vec<f32>,
    frame_output: Vec<f32>,
    // Output zeros primed at startup to cover resampler batching + frame
    // scheduling. Does NOT include the model's reconstruction delay (that is
    // already present in the model's own output stream). Read only by `reset`
    // (see `model_rate`); allow dead_code until GTCRN drives `reset`.
    #[allow(dead_code)]
    priming_samples: usize,
    // Total end-to-end content delay: `priming_samples` + the model's inherent
    // reconstruction delay, converted to the device rate.
    latency_samples: usize,
}

impl<D: FrameDenoiser> FixedDelayAdapter<D> {
    pub(crate) fn new(processor: D, device_sample_rate: u32) -> Result<Self> {
        let device_rate = usize::try_from(device_sample_rate)?;
        if device_rate == 0 {
            bail!("denoiser device sample rate must be greater than zero");
        }
        let model_rate = processor.sample_rate();
        let frame_size = processor.frame_size();
        if model_rate == 0 || frame_size == 0 {
            bail!("denoiser model rate and frame size must be greater than zero");
        }

        // At matched rates the streaming resamplers are pure passthrough, so the
        // batch delay collapses to zero on both sides — only the frame FIFO and
        // the fixed-delay output FIFO do work (the GTCRN 16 kHz seam relies on
        // this).
        let resampling = device_rate != model_rate;
        let to_model_delay = if resampling { RESAMPLER_BATCH_DELAY } else { 0 };
        let from_model_delay = if resampling { RESAMPLER_BATCH_DELAY } else { 0 };
        // The frame FIFO can hold just under one frame of unprocessed input; keep
        // a second frame of slack. This is the adapter's own scheduling margin,
        // independent of any model reconstruction delay.
        let frame_margin = 2 * frame_size;
        let priming_model_domain = from_model_delay + frame_margin;
        let minimum_priming_samples = to_model_delay.saturating_add(
            priming_model_domain
                .saturating_mul(device_rate)
                .div_ceil(model_rate),
        );
        let to_model = StreamingResampleMono::new(device_rate, model_rate)?;
        let from_model = StreamingResampleMono::new(model_rate, device_rate)?;
        // Both bounds are in their respective OUTPUT domains. Convert the
        // first deficit plus frame scheduling to device samples, then add the
        // second deficit. Keep this coupled to StreamingResampleMono's batching
        // contract; changing chunk partitions must never insert/drop content.
        let bounded_priming_samples = to_model
            .fixed_output_delay_samples()
            .saturating_add(frame_margin)
            .saturating_mul(device_rate)
            .div_ceil(model_rate)
            .saturating_add(from_model.fixed_output_delay_samples());
        let priming_samples = minimum_priming_samples.max(bounded_priming_samples);
        // The model's reconstruction delay is already baked into its output
        // samples (it emits delayed content, still 1:1). Don't prime zeros for
        // it — only report it as latency so callers and the finite drain trim at
        // the right position.
        let recon_samples = processor
            .output_delay_frames()
            .saturating_mul(frame_size)
            .saturating_mul(device_rate)
            .div_ceil(model_rate);
        let latency_samples = priming_samples.saturating_add(recon_samples);

        Ok(Self {
            processor,
            device_rate,
            model_rate,
            frame_size,
            to_model,
            from_model,
            model_input: Vec::new(),
            model_input_start: 0,
            model_output: Vec::new(),
            device_output: vec![0.0; priming_samples],
            device_output_start: 0,
            frame_input: vec![0.0; frame_size],
            frame_output: vec![0.0; frame_size],
            priming_samples,
            latency_samples,
        })
    }

    pub(crate) fn latency_samples(&self) -> usize {
        self.latency_samples
    }

    /// Process arbitrary device-rate input while preserving the per-call length.
    pub(crate) fn process_in_place(&mut self, samples: &mut [f32]) -> Result<()> {
        self.process_input(samples)?;
        self.emit_exact(samples)
    }

    /// Denoise finite input, remove the streaming delay, and preserve its length.
    ///
    /// Assumes a freshly-built (or freshly-`reset`) adapter so the startup
    /// priming is deterministic.
    pub(crate) fn process_finite(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let block = (self.device_rate / 20).max(128);
        let mut delayed = Vec::with_capacity(input.len() + self.latency_samples);

        for chunk in input.chunks(block) {
            let mut out = chunk.to_vec();
            self.process_in_place(&mut out)?;
            delayed.extend_from_slice(&out);
        }

        // Feed bounded zero blocks until the delayed tail corresponding to all
        // finite input samples has reached the output timeline.
        let target = input.len().saturating_add(self.latency_samples);
        while delayed.len() < target {
            let mut zeros = vec![0.0; block.min(target - delayed.len())];
            self.process_in_place(&mut zeros)?;
            delayed.extend_from_slice(&zeros);
        }
        let start = self.latency_samples.min(delayed.len());
        let end = start.saturating_add(input.len()).min(delayed.len());
        let mut aligned = delayed[start..end].to_vec();
        aligned.resize(input.len(), 0.0);
        Ok(aligned)
    }

    /// Restore the post-construction (warmed-up) state, including the model.
    ///
    /// Exercised by the adapter tests today; the first non-test caller is GTCRN's
    /// `reset_streaming_state` wiring (Phase 3). Allow dead_code until then so the
    /// shared contract can land ahead of its consumer.
    #[allow(dead_code)]
    pub(crate) fn reset(&mut self) -> Result<()> {
        self.processor.reset()?;
        self.to_model = StreamingResampleMono::new(self.device_rate, self.model_rate)?;
        self.from_model = StreamingResampleMono::new(self.model_rate, self.device_rate)?;
        self.model_input.clear();
        self.model_input_start = 0;
        self.model_output.clear();
        self.device_output.clear();
        self.device_output.resize(self.priming_samples, 0.0);
        self.device_output_start = 0;
        Ok(())
    }

    fn process_input(&mut self, samples: &[f32]) -> Result<()> {
        self.to_model.process_into(samples, &mut self.model_input)?;

        let frame = self.frame_size;
        while self
            .model_input
            .len()
            .saturating_sub(self.model_input_start)
            >= frame
        {
            let end = self.model_input_start + frame;
            self.frame_input
                .copy_from_slice(&self.model_input[self.model_input_start..end]);
            self.processor
                .process_frame(&self.frame_input, &mut self.frame_output)?;
            self.model_input_start = end;
            self.model_output.extend_from_slice(&self.frame_output);
        }
        self.compact_model_input();

        if !self.model_output.is_empty() {
            self.from_model
                .process_into(&self.model_output, &mut self.device_output)?;
            self.model_output.clear();
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
                "denoiser output underrun: need {} device samples, have {} (rate={} Hz)",
                samples.len(),
                available,
                self.device_rate
            );
        }
        let end = self.device_output_start + samples.len();
        samples.copy_from_slice(&self.device_output[self.device_output_start..end]);
        self.device_output_start = end;
        self.compact_device_output();
        Ok(())
    }

    fn compact_model_input(&mut self) {
        if self.model_input_start == self.model_input.len() {
            self.model_input.clear();
            self.model_input_start = 0;
        } else if self.model_input_start >= 4096
            && self.model_input_start * 2 >= self.model_input.len()
        {
            self.model_input.drain(..self.model_input_start);
            self.model_input_start = 0;
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

    /// Deterministic fake `FrameDenoiser` for the adapter tests: copies input to
    /// output but delays the content by `delay_frames` whole frames, so the
    /// nonzero-`output_delay_frames` path (which GTCRN needs) is exercised
    /// without any model dependency.
    struct DelayCopy {
        sample_rate: usize,
        frame_size: usize,
        delay_frames: usize,
        ring: Vec<Vec<f32>>,
        next: usize,
    }

    impl DelayCopy {
        fn new(sample_rate: usize, frame_size: usize, delay_frames: usize) -> Self {
            Self {
                sample_rate,
                frame_size,
                delay_frames,
                ring: vec![vec![0.0; frame_size]; delay_frames],
                next: 0,
            }
        }
    }

    impl FrameDenoiser for DelayCopy {
        fn sample_rate(&self) -> usize {
            self.sample_rate
        }
        fn frame_size(&self) -> usize {
            self.frame_size
        }
        fn output_delay_frames(&self) -> usize {
            self.delay_frames
        }
        fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
            if self.delay_frames == 0 {
                for (dst, src) in output.iter_mut().zip(input) {
                    *dst = if src.is_finite() { *src } else { 0.0 };
                }
                return Ok(());
            }
            // Emit the frame from `delay_frames` calls ago, then store this one.
            output.copy_from_slice(&self.ring[self.next]);
            for (dst, src) in self.ring[self.next].iter_mut().zip(input) {
                *dst = if src.is_finite() { *src } else { 0.0 };
            }
            self.next = (self.next + 1) % self.delay_frames;
            Ok(())
        }
        fn reset(&mut self) -> Result<()> {
            for frame in &mut self.ring {
                frame.iter_mut().for_each(|s| *s = 0.0);
            }
            self.next = 0;
            Ok(())
        }
    }

    fn adapter(
        device_rate: u32,
        frame_size: usize,
        delay_frames: usize,
    ) -> FixedDelayAdapter<DelayCopy> {
        // Model rate fixed at 16 kHz so device_rate == 16_000 exercises the
        // resampler-bypass (matched) path and other rates exercise resampling.
        FixedDelayAdapter::new(
            DelayCopy::new(16_000, frame_size, delay_frames),
            device_rate,
        )
        .unwrap()
    }

    #[test]
    fn per_call_length_is_preserved_matched_and_resampled() {
        for &rate in &[16_000u32, 44_100, 48_000, 96_000] {
            for &delay in &[0usize, 1, 3] {
                let mut a = adapter(rate, 256, delay);
                for len in [1usize, 127, 256, 777, 2205, 4800] {
                    let mut audio = vec![0.0; len];
                    a.process_in_place(&mut audio).unwrap();
                    assert_eq!(audio.len(), len);
                    assert!(audio.iter().all(|x| x.is_finite()));
                }
            }
        }
    }

    #[test]
    fn chunk_partition_matches_whole_run() {
        for &rate in &[16_000u32, 48_000] {
            for &delay in &[0usize, 2] {
                let input: Vec<f32> = (0..40_000).map(|i| 0.4 * (i as f32 * 0.05).sin()).collect();

                let mut whole = input.clone();
                adapter(rate, 256, delay)
                    .process_in_place(&mut whole)
                    .unwrap();

                let mut split_adapter = adapter(rate, 256, delay);
                let mut split = Vec::with_capacity(input.len());
                for chunk in input.chunks(777) {
                    let mut out = chunk.to_vec();
                    split_adapter.process_in_place(&mut out).unwrap();
                    split.extend_from_slice(&out);
                }
                assert_eq!(whole, split);
            }
        }
    }

    #[test]
    fn matched_rate_impulse_lands_at_reported_latency() {
        // At the matched (bypass) rate the arithmetic is exact: a delaying model
        // must surface its content at exactly `latency_samples`.
        for &delay in &[0usize, 1, 4] {
            let frame = 256usize;
            let mut a = adapter(16_000, frame, delay);
            let latency = a.latency_samples();

            let total = latency + 4 * frame;
            let mut input = vec![0.0f32; total];
            input[0] = 1.0;
            let mut out = input.clone();
            a.process_in_place(&mut out).unwrap();

            let peak = out
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.abs().partial_cmp(&y.abs()).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            assert_eq!(peak, latency, "delay_frames={delay}");
        }
    }

    #[test]
    fn finite_processing_preserves_length_and_signal() {
        for &rate in &[16_000u32, 48_000] {
            for &delay in &[0usize, 2] {
                let input: Vec<f32> = (0..rate as usize / 2)
                    .map(|i| 0.4 * (i as f32 * 0.05).sin())
                    .collect();
                let mut a = adapter(rate, 256, delay);
                let out = a.process_finite(&input).unwrap();
                assert_eq!(out.len(), input.len());
                assert!(out.iter().all(|x| x.is_finite()));
                assert!(crate::dsp::rms(&out) > 0.01, "rate={rate} delay={delay}");
            }
        }
    }

    #[test]
    fn reset_reproduces_startup() {
        let input: Vec<f32> = (0..20_000).map(|i| 0.3 * (i as f32 * 0.07).sin()).collect();
        let mut a = adapter(48_000, 256, 2);

        let mut first = input.clone();
        a.process_in_place(&mut first).unwrap();

        a.reset().unwrap();
        let mut second = input.clone();
        a.process_in_place(&mut second).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn awkward_rates_preserve_continuous_audio_across_batch_phases() {
        for (model_rate, frame) in [(16_000, 256), (48_000, 480)] {
            for rate in [11_025u32, 22_050] {
                let input: Vec<f32> = (0..rate as usize * 3)
                    .map(|i| 0.3 * (std::f32::consts::TAU * 733.0 * i as f32 / rate as f32).sin())
                    .collect();
                let build =
                    || FixedDelayAdapter::new(DelayCopy::new(model_rate, frame, 0), rate).unwrap();
                let mut whole = input.clone();
                build().process_in_place(&mut whole).unwrap();
                // Include the 50 ms finite-WAV partition and the 20 ms GTCRN
                // partition that used to fail, plus one-sample calls covering
                // every possible phase of the two resampler accumulators.
                for hop in [1, (rate as usize / 20).max(128), rate as usize / 50] {
                    let mut adapter = build();
                    let mut split = input.clone();
                    for chunk in split.chunks_mut(hop) {
                        adapter.process_in_place(chunk).unwrap();
                    }
                    assert_eq!(split, whole, "model={model_rate} rate={rate} hop={hop}");
                    assert!(crate::dsp::rms(&split) > 0.1);
                    adapter.reset().unwrap();
                    let mut reset = input.clone();
                    adapter.process_in_place(&mut reset).unwrap();
                    assert_eq!(reset, whole);
                }
            }
        }
    }

    #[test]
    fn resampled_finite_impulse_keeps_its_position() {
        for rate in [11_025, 22_050, 44_100, 48_000] {
            let mut input = vec![0.0; rate as usize];
            let position = 1000;
            input[position] = 1.0;
            for delay_frames in [0, 2] {
                let mut adapter = adapter(rate, 256, delay_frames);
                let output = adapter.process_finite(&input).unwrap();
                let peak = output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                    .unwrap()
                    .0;
                assert_eq!(output.len(), input.len());
                assert!(
                    peak.abs_diff(position) <= 1,
                    "rate={rate} delay_frames={delay_frames} peak={peak}"
                );
            }
        }
    }
}
