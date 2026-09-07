use std::time::{Duration, Instant};

use anyhow::Result;

use crate::dsp::OutputResampler;
use crate::sola::{self, ChunkSmoother, ChunkSmootherConfig, JoinDiagnostics, SmoothingKind};

use super::{ContentDelay, ModelOutput, VoiceModel};

#[derive(Clone, Copy, Debug)]
pub struct ChunkOutputConfig {
    pub kind: SmoothingKind,
    pub output_sample_rate: u32,
    pub output_chunk_samples: usize,
    pub crossfade_ms: u32,
    pub sola_search_ms: u32,
    pub tail_discard_ms: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ChunkStats {
    pub silent: bool,
    pub inference_time: Duration,
    /// Worker time including model processing, joining and output resampling.
    pub processing_time: Duration,
    /// Nominal retained content in output samples, excluding chunk accumulation,
    /// device/queue latency and inference wall time. None until initialized.
    pub content_delay_samples: Option<usize>,
    pub input_rms: f32,
    pub output_rms: f32,
    pub model_output_samples: usize,
}

/// Owns the stateful model-to-fixed-output conversion shared by WAV and the
/// worker-side realtime paths.
///
/// Keep this off audio callbacks: model processing, smoothing, and resampling
/// may allocate. Output settings are intentionally fixed for the converter
/// lifetime; rebuild the converter together with the model when a stream's chunk
/// size or output format changes.
pub struct ChunkConverter<M> {
    model: M,
    output: ChunkOutputConfig,
    smoother: Option<(u32, ChunkSmoother)>,
    output_resampler: Option<OutputResampler>,
    // Reused per-chunk buffers for the model's converted audio and output
    // pitchf, so `process_chunk` does not allocate them every chunk.
    model_audio: Vec<f32>,
    model_pitchf: Vec<f32>,
}

impl<M: VoiceModel> ChunkConverter<M> {
    pub fn new(model: M, output: ChunkOutputConfig) -> Self {
        Self {
            model,
            output,
            smoother: None,
            output_resampler: None,
            model_audio: Vec::new(),
            model_pitchf: Vec::new(),
        }
    }

    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    pub fn output_chunk_samples(&self) -> usize {
        self.output.output_chunk_samples
    }

    /// Output-side filter and incomplete-block buffering delay. This is known
    /// after the first process/prime and is zero for identical sample rates.
    pub fn output_resample_delay_samples(&self) -> Option<usize> {
        self.output_resampler
            .as_ref()
            .map(OutputResampler::delay_samples)
    }

    /// Content delay after priming, in output samples. Finite conversion removes
    /// this once and recovers the source tail by processing additional zero hops
    /// through this same converter. SOLA's bounded search may advance content
    /// within its window; the maximum nominal hold keeps that tail recoverable.
    /// The smoother's first unprimed silent hop is separate startup behavior.
    pub fn output_content_delay_samples(&self) -> usize {
        let join_delay = self
            .smoother
            .as_ref()
            .map_or(ContentDelay::ZERO, |(rate, smoother)| {
                ContentDelay::from_samples(smoother.content_delay_samples(), *rate)
            });
        (self.model.input_content_delay() + join_delay)
            .output_samples(self.output.output_sample_rate)
            + self.output_resample_delay_samples().unwrap_or(0)
    }

    /// Diagnostics for the most recent [`Self::process_chunk`] / [`Self::prime`]
    /// join, or `None` before the first chunk builds the smoother. Diagnostics
    /// only (offline join analysis); realtime callers can ignore it.
    pub fn last_join_diagnostics(&self) -> Option<JoinDiagnostics> {
        self.smoother
            .as_ref()
            .map(|(_, smoother)| smoother.last_diagnostics())
    }

    /// The crossfade window the joiner actually applies, in model-domain samples,
    /// or `None` before the first chunk builds the smoother. This is *after* the
    /// 3/4-of-chunk cap in [`sola::model_domain_crossfade_samples`]; compare it
    /// with [`Self::join_requested_crossfade_samples`] to see the cap's effect.
    pub fn join_crossfade_samples(&self) -> Option<usize> {
        self.smoother
            .as_ref()
            .map(|(_, smoother)| smoother.crossfade_samples())
    }

    /// The configured (pre-cap) crossfade window in model-domain samples — what
    /// `crossfade_ms` asks for before the 3/4-of-chunk cap. `None` before the
    /// first chunk builds the smoother. Diagnostics: comparing a chunk's
    /// `crossfade_len` against this shows when the cap (short chunk) or the
    /// per-chunk runtime clamp shortened the overlap.
    pub fn join_requested_crossfade_samples(&self) -> Option<usize> {
        self.smoother
            .as_ref()
            .map(|(rate, _)| sola::ms_to_samples(*rate, self.output.crossfade_ms))
    }

    /// Discards output-joining history without rebuilding the owned model.
    ///
    /// Realtime callers use this after a period where model processing was
    /// paused. Reusing the old smoother history would join fresh model output
    /// against audio emitted before the pause.
    pub fn reset_streaming_state(&mut self) {
        self.smoother = None;
        // Join history and the FFT/FIFO timeline are a single stream. Retaining
        // either across pass-through would replay stale audio on resumption.
        self.output_resampler = None;
    }

    pub fn process_chunk(
        &mut self,
        input: &[f32],
        input_sample_rate: u32,
        out: &mut Vec<f32>,
    ) -> Result<ChunkStats> {
        let started = Instant::now();
        let meta = self.model.process(
            input,
            input_sample_rate,
            &mut self.model_audio,
            &mut self.model_pitchf,
        )?;
        let mut stats = chunk_stats(&meta, self.model_audio.len());
        let model_sample_rate = meta.sample_rate;
        let output_chunk_samples = self.output.output_chunk_samples;
        self.ensure_smoother(model_sample_rate)?;
        // Disjoint field borrows: the smoother and the model output buffers are
        // separate fields, so this does not conflict.
        let smoother = &mut self.smoother.as_mut().expect("smoother set above").1;
        smoother.process(&self.model_audio, &self.model_pitchf);
        self.output_resampler
            .as_mut()
            .expect("resampler set above")
            .process_fixed(smoother.output(), output_chunk_samples, out)?;
        stats.content_delay_samples = Some(self.output_content_delay_samples());
        stats.processing_time = started.elapsed();
        Ok(stats)
    }

    /// Runs the same model/smoother initialization path as a normal chunk but
    /// emits no audio. WAV conversion uses this for its historical silent
    /// preroll; realtime paths deliberately start from their first real chunk.
    pub fn prime(&mut self, input: &[f32], input_sample_rate: u32) -> Result<ChunkStats> {
        let started = Instant::now();
        let meta = self.model.process(
            input,
            input_sample_rate,
            &mut self.model_audio,
            &mut self.model_pitchf,
        )?;
        let mut stats = chunk_stats(&meta, self.model_audio.len());
        self.ensure_smoother(meta.sample_rate)?;
        let smoother = &mut self.smoother.as_mut().expect("smoother set above").1;
        smoother.prime_model_output(&self.model_audio, &self.model_pitchf);
        stats.content_delay_samples = Some(self.output_content_delay_samples());
        stats.processing_time = started.elapsed();
        Ok(stats)
    }

    /// Ensures `self.smoother` matches `model_sample_rate`, rebuilding it on a
    /// rate change. Split out of the per-chunk path so callers can then take a
    /// disjoint borrow of the smoother alongside the model output buffers.
    fn ensure_smoother(&mut self, model_sample_rate: u32) -> Result<()> {
        if self.smoother.as_ref().map(|(rate, _)| *rate) != Some(model_sample_rate) {
            let smoother = sola::model_domain_chunk_smoother(ChunkSmootherConfig {
                kind: self.output.kind,
                output_chunk_samples: self.output.output_chunk_samples,
                output_sample_rate: self.output.output_sample_rate,
                model_sample_rate,
                crossfade_ms: self.output.crossfade_ms,
                sola_search_ms: self.output.sola_search_ms,
                tail_discard_ms: self.output.tail_discard_ms,
            });
            let resampler = OutputResampler::new_fixed(
                model_sample_rate as usize,
                self.output.output_sample_rate as usize,
                smoother.chunk_samples(),
                self.output.output_chunk_samples,
            )?;
            self.smoother = Some((model_sample_rate, smoother));
            self.output_resampler = Some(resampler);
        }
        Ok(())
    }
}

fn chunk_stats(meta: &ModelOutput, model_output_samples: usize) -> ChunkStats {
    ChunkStats {
        silent: meta.silent,
        inference_time: meta.inference_time,
        processing_time: Duration::ZERO,
        content_delay_samples: None,
        input_rms: meta.input_rms,
        output_rms: meta.output_rms,
        // This is the length immediately before smoothing, not the pipeline's
        // separately reported `raw_output_samples` diagnostic.
        model_output_samples,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use anyhow::{anyhow, Result};

    use super::*;

    struct FakeModel {
        // Each entry pairs the model-domain audio the fake emits with its
        // metadata; `process` writes the audio into the caller's out buffer.
        outputs: VecDeque<Result<(Vec<f32>, ModelOutput)>>,
        calls: usize,
        pitch_hz: Option<f32>,
        input_delay: ContentDelay,
    }

    impl FakeModel {
        fn new(outputs: impl IntoIterator<Item = Result<(Vec<f32>, ModelOutput)>>) -> Self {
            Self {
                outputs: outputs.into_iter().collect(),
                calls: 0,
                pitch_hz: None,
                input_delay: ContentDelay::ZERO,
            }
        }
    }

    impl VoiceModel for FakeModel {
        fn input_content_delay(&self) -> ContentDelay {
            self.input_delay
        }

        fn process(
            &mut self,
            _audio: &[f32],
            _sample_rate: u32,
            out_audio: &mut Vec<f32>,
            out_pitchf: &mut Vec<f32>,
        ) -> Result<ModelOutput> {
            self.calls += 1;
            let (audio, meta) = self.outputs.pop_front().expect("fake output")?;
            out_audio.clear();
            out_audio.extend_from_slice(&audio);
            out_pitchf.clear();
            if let Some(pitch_hz) = self.pitch_hz {
                out_pitchf.resize(12, pitch_hz);
            }
            Ok(meta)
        }
    }

    fn config() -> ChunkOutputConfig {
        ChunkOutputConfig {
            kind: SmoothingKind::Sola,
            output_sample_rate: 1_000,
            output_chunk_samples: 4,
            crossfade_ms: 2,
            sola_search_ms: 2,
            tail_discard_ms: 0,
        }
    }

    fn output(audio: Vec<f32>, sample_rate: u32) -> (Vec<f32>, ModelOutput) {
        let meta = ModelOutput {
            raw_output_samples: 999,
            output_rms: 0.75,
            convert_size: audio.len(),
            out_size: audio.len(),
            model_input_samples: audio.len(),
            sample_rate,
            inference_time: Duration::from_micros(123),
            embedder_time: Duration::ZERO,
            pitch_time: Duration::ZERO,
            rvc_time: Duration::ZERO,
            input_rms: 0.25,
            voiced_ratio: 0.0,
            applied_output_gain: 1.0,
            feature_frames: 0,
            pitch_frames: 0,
            silent: true,
            volume: 0.0,
        };
        (audio, meta)
    }

    #[test]
    fn processes_once_and_returns_fixed_audio_and_stats() {
        let mut converter =
            ChunkConverter::new(FakeModel::new([Ok(output(vec![1.0; 8], 1_000))]), config());

        let mut out = Vec::new();
        let stats = converter.process_chunk(&[0.0; 4], 1_000, &mut out).unwrap();

        assert_eq!(converter.model_mut().calls, 1);
        assert_eq!(out.len(), 4);
        assert!(stats.silent);
        assert_eq!(stats.inference_time, Duration::from_micros(123));
        assert!(stats.processing_time > Duration::ZERO);
        assert_eq!(
            stats.content_delay_samples,
            Some(converter.output_content_delay_samples())
        );
        assert_eq!(stats.input_rms, 0.25);
        assert_eq!(stats.output_rms, 0.75);
        assert_eq!(stats.model_output_samples, 8);
    }

    #[test]
    fn smoother_persists_until_model_rate_changes() {
        let outputs = [
            Ok(output(vec![1.0; 8], 1_000)),
            Ok(output(vec![2.0; 8], 1_000)),
            Ok(output(vec![3.0; 16], 2_000)),
        ];
        let mut converter = ChunkConverter::new(FakeModel::new(outputs), config());

        let mut first = Vec::new();
        converter
            .process_chunk(&[0.0; 4], 1_000, &mut first)
            .unwrap();
        let mut second = Vec::new();
        converter
            .process_chunk(&[0.0; 4], 1_000, &mut second)
            .unwrap();
        let mut changed_rate = Vec::new();
        converter
            .process_chunk(&[0.0; 4], 1_000, &mut changed_rate)
            .unwrap();

        assert_eq!(first, vec![0.0; 4]);
        assert_ne!(second, vec![0.0; 4]);
        assert_eq!(changed_rate, vec![0.0; 4]);
    }

    #[test]
    fn reset_streaming_state_discards_smoother_history() {
        let outputs = [
            Ok(output(vec![1.0; 8], 1_000)),
            Ok(output(vec![2.0; 8], 1_000)),
            Ok(output(vec![3.0; 8], 1_000)),
        ];
        let mut converter = ChunkConverter::new(FakeModel::new(outputs), config());

        let mut scratch = Vec::new();
        converter
            .process_chunk(&[0.0; 4], 1_000, &mut scratch)
            .unwrap();
        let mut joined = Vec::new();
        converter
            .process_chunk(&[0.0; 4], 1_000, &mut joined)
            .unwrap();
        assert_ne!(joined, vec![0.0; 4]);

        converter.reset_streaming_state();
        let mut reset = Vec::new();
        converter
            .process_chunk(&[0.0; 4], 1_000, &mut reset)
            .unwrap();
        assert_eq!(reset, vec![0.0; 4]);
    }

    #[test]
    fn prime_initializes_smoother_without_emitting_audio() {
        let prime_audio = vec![0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0];
        let real_audio = vec![0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0];
        let mut converter = ChunkConverter::new(
            FakeModel::new([
                Ok(output(prime_audio, 1_000)),
                Ok(output(real_audio.clone(), 1_000)),
            ]),
            config(),
        );
        let mut without_prime =
            ChunkConverter::new(FakeModel::new([Ok(output(real_audio, 1_000))]), config());

        let stats = converter.prime(&[0.0; 4], 1_000).unwrap();
        let mut primed = Vec::new();
        converter
            .process_chunk(&[0.0; 4], 1_000, &mut primed)
            .unwrap();
        let mut unprimed = Vec::new();
        without_prime
            .process_chunk(&[0.0; 4], 1_000, &mut unprimed)
            .unwrap();

        assert_eq!(stats.model_output_samples, 8);
        assert_ne!(primed, unprimed);
    }

    #[test]
    fn content_delay_uses_capped_crossfade_and_disabled_join_geometry() {
        let mut settings = config();
        settings.crossfade_ms = 20;
        settings.tail_discard_ms = 3;
        let mut converter =
            ChunkConverter::new(FakeModel::new([Ok(output(vec![0.0; 32], 1_000))]), settings);
        converter.prime(&[0.0; 4], 1_000).unwrap();
        assert_eq!(converter.output_content_delay_samples(), 3 + 2 + 3);
        assert_eq!(converter.output_resample_delay_samples(), Some(0));

        settings.crossfade_ms = 0;
        let mut converter =
            ChunkConverter::new(FakeModel::new([Ok(output(vec![0.0; 32], 1_000))]), settings);
        converter.prime(&[0.0; 4], 1_000).unwrap();
        assert_eq!(converter.output_content_delay_samples(), 3);
    }

    #[test]
    fn fractional_input_and_join_delays_round_only_after_summing() {
        let settings = ChunkOutputConfig {
            kind: SmoothingKind::Sola,
            output_sample_rate: 44_100,
            output_chunk_samples: 882,
            crossfade_ms: 20,
            sola_search_ms: 0,
            tail_discard_ms: 0,
        };
        let mut model = FakeModel::new([Ok(output(vec![0.0; 1280], 32_000))]);
        model.input_delay = ContentDelay::from_samples(3, 16_000);
        let mut converter = ChunkConverter::new(model, settings);
        converter.prime(&[], 32_000).unwrap();
        // The capped 15 ms fade is 661.5 output samples, and 3/16k seconds
        // contributes 8.26875. ceil(669.76875) is 670, not ceil(661.5)+ceil(8.26875)=671.
        assert_eq!(
            converter.output_content_delay_samples(),
            670 + converter.output_resample_delay_samples().unwrap()
        );
    }

    #[test]
    fn persistent_output_resampling_removes_join_boundary_dc_distortion() {
        for kind in [SmoothingKind::Sola, SmoothingKind::Psola] {
            for (from, to) in [(32_000, 48_000), (48_000, 44_100)] {
                let settings = ChunkOutputConfig {
                    kind,
                    output_sample_rate: to,
                    output_chunk_samples: to as usize / 10,
                    crossfade_ms: 10,
                    sola_search_ms: 0,
                    tail_discard_ms: 0,
                };
                let outputs =
                    (0..20).map(|_| Ok(output(vec![0.25; from as usize * 11 / 100], from)));
                let mut converter = ChunkConverter::new(FakeModel::new(outputs), settings);
                let mut out = Vec::new();
                for index in 0..20 {
                    converter.process_chunk(&[], from, &mut out).unwrap();
                    assert_eq!(out.len(), to as usize / 10);
                    if index >= 3 {
                        let error = out.iter().map(|v| (v - 0.25).abs()).fold(0.0, f32::max);
                        assert!(error < 1e-4, "{kind:?} {from}->{to}: {error}");
                    }
                }
            }
        }
    }

    #[test]
    fn reset_discards_resampler_filter_fifo_and_delay_state_together() {
        let settings = ChunkOutputConfig {
            kind: SmoothingKind::Sola,
            output_sample_rate: 44_100,
            output_chunk_samples: 4410,
            crossfade_ms: 10,
            sola_search_ms: 0,
            tail_discard_ms: 0,
        };
        let outputs = (0..6).map(|_| Ok(output(vec![0.25; 5280], 48_000)));
        let mut converter = ChunkConverter::new(FakeModel::new(outputs), settings);
        let mut initial = Vec::new();
        converter.process_chunk(&[], 48_000, &mut initial).unwrap();
        let mut out = Vec::new();
        for _ in 0..4 {
            converter.process_chunk(&[], 48_000, &mut out).unwrap();
        }
        assert!(out.iter().any(|sample| sample.abs() > 0.1));
        converter.reset_streaming_state();
        assert_eq!(converter.output_resample_delay_samples(), None);
        converter.process_chunk(&[], 48_000, &mut out).unwrap();
        assert_eq!(out, initial);
    }

    #[test]
    fn joined_voiced_sines_match_continuous_output_filter_after_delay() {
        for kind in [SmoothingKind::Sola, SmoothingKind::Psola] {
            for (from, to) in [(32_000, 48_000), (48_000, 44_100)] {
                let hop = from as usize / 10;
                let fade = from as usize / 100;
                let settings = ChunkOutputConfig {
                    kind,
                    output_sample_rate: to,
                    output_chunk_samples: to as usize / 10,
                    crossfade_ms: 10,
                    sola_search_ms: 0,
                    tail_discard_ms: 0,
                };
                let sample = |i: usize| {
                    (0.25 * (2.0 * std::f64::consts::PI * 213.7 * i as f64 / from as f64).sin())
                        as f32
                };
                let outputs = (0..20).map(|index| {
                    Ok(output(
                        (index * hop..(index + 1) * hop + fade)
                            .map(sample)
                            .collect(),
                        from,
                    ))
                });
                let mut model = FakeModel::new(outputs);
                model.pitch_hz = Some(213.7);
                let mut converter = ChunkConverter::new(model, settings);
                let mut actual = Vec::new();
                let mut chunk = Vec::new();
                for _ in 0..20 {
                    converter.process_chunk(&[], from, &mut chunk).unwrap();
                    actual.extend_from_slice(&chunk);
                }
                if kind == SmoothingKind::Psola {
                    let diagnostics = converter.last_join_diagnostics().unwrap();
                    assert!(diagnostics.pitch_period.is_some());
                    assert!(!diagnostics.psola_fallback);
                }
                let mut joined = vec![0.0; hop];
                joined.extend((hop..20 * hop).map(sample));
                let reference =
                    crate::dsp::resample_mono(&joined, from as usize, to as usize).unwrap();
                let delay = converter.output_resample_delay_samples().unwrap();
                let max_error = actual[delay..]
                    .iter()
                    .zip(&reference)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(max_error < 1e-6, "{kind:?} {from}->{to}: {max_error}");
            }
        }
    }

    #[test]
    fn model_and_output_errors_are_returned() {
        let mut out = Vec::new();
        let mut model_error =
            ChunkConverter::new(FakeModel::new([Err(anyhow!("model failed"))]), config());
        assert!(model_error
            .process_chunk(&[0.0; 4], 1_000, &mut out)
            .is_err());

        let mut output_error =
            ChunkConverter::new(FakeModel::new([Ok(output(vec![1.0; 8], 0))]), config());
        assert!(output_error
            .process_chunk(&[0.0; 4], 1_000, &mut out)
            .is_err());
    }
}
