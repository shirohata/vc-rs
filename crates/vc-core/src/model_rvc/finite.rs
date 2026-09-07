//! Finite-input scheduling around the normal worker-side conversion pipeline.
//!
//! A fixed-length output chunk does not mean all its input content has emerged:
//! input denoisers, rate adapters, and the joiner's lookahead retain audio. Drain
//! them by processing zero input through the same model before removing the
//! known content delay. Never append an independently resampled overlap tail.

use anyhow::{ensure, Context, Result};

use crate::sola::JoinDiagnostics;

use super::{ChunkConverter, ChunkStats, VoiceModel};

/// One model call, including zero-input calls needed to recover the final tail.
pub struct FiniteChunk {
    pub index: usize,
    pub flushing: bool,
    pub stats: ChunkStats,
    pub join: JoinDiagnostics,
    pub requested_crossfade_samples: usize,
    /// Join location in the final, delay-compensated output. `None` means the
    /// boundary fell outside the retained clip. Resampler delay moves the join
    /// away from its original fixed-chunk boundary and must be included here.
    pub seam_sample: Option<usize>,
}

pub struct FiniteOutput {
    pub audio: Vec<f32>,
    pub chunks: Vec<FiniteChunk>,
}

/// Convert a finite mono signal using a newly constructed converter. Ownership
/// prevents a caller from continuing the stream after its zero-input drain.
/// `input_chunk_samples` must match the shape used to load the model. The output
/// duration follows the configured input/output chunk ratio (rounding up only
/// for the final partial sample), not the number of padded model calls.
pub fn convert_finite<M: VoiceModel>(
    mut converter: ChunkConverter<M>,
    input: &[f32],
    input_sample_rate: u32,
    input_chunk_samples: usize,
) -> Result<FiniteOutput> {
    ensure!(
        input_sample_rate > 0,
        "finite input sample rate must be positive"
    );
    ensure!(
        input_chunk_samples > 0,
        "finite input chunk must be nonempty"
    );
    let output_chunk_samples = converter.output_chunk_samples();
    ensure!(
        output_chunk_samples > 0,
        "finite output chunk must be nonempty"
    );
    if input.is_empty() {
        return Ok(FiniteOutput {
            audio: Vec::new(),
            chunks: Vec::new(),
        });
    }
    let target_len = usize::try_from(
        (input.len() as u128 * output_chunk_samples as u128).div_ceil(input_chunk_samples as u128),
    )
    .context("finite output length overflow")?;

    let zeros = vec![0.0; input_chunk_samples];
    converter.prime(&zeros, input_sample_rate)?;
    let content_delay = converter.output_content_delay_samples();
    let output_resample_delay = converter.output_resample_delay_samples().unwrap_or(0);
    let end = content_delay
        .checked_add(target_len)
        .context("finite delayed length overflow")?;
    let mut audio = Vec::with_capacity(end);
    let mut chunks = Vec::new();
    let mut padded = zeros.clone();
    let mut converted = Vec::with_capacity(output_chunk_samples);
    let input_chunks = input.len().div_ceil(input_chunk_samples);
    let required_chunks = end.div_ceil(output_chunk_samples).max(input_chunks);

    // The bound comes from the exact retained content interval, never from a
    // silence detector: a voice model can generate nonzero audio for zero input.
    // Padding already supplied with a partial final chunk counts toward drain.
    for index in 0..required_chunks {
        let start = index
            .checked_mul(input_chunk_samples)
            .context("finite input position overflow")?;
        let flushing = index >= input_chunks;
        let model_input = if flushing {
            zeros.as_slice()
        } else {
            let end = start.saturating_add(input_chunk_samples).min(input.len());
            let real = &input[start..end];
            if real.len() == input_chunk_samples {
                real
            } else {
                padded.fill(0.0);
                padded[..real.len()].copy_from_slice(real);
                &padded
            }
        };
        let stats = converter.process_chunk(model_input, input_sample_rate, &mut converted)?;
        ensure!(
            converted.len() == output_chunk_samples,
            "finite converter changed output chunk size"
        );
        ensure!(
            converter.output_content_delay_samples() == content_delay,
            "finite converter changed content delay during conversion"
        );

        let seam_sample = audio
            .len()
            .checked_add(output_resample_delay)
            .and_then(|position| position.checked_sub(content_delay))
            .filter(|&position| position < target_len);
        audio.extend_from_slice(&converted);
        chunks.push(FiniteChunk {
            index,
            flushing,
            stats,
            join: converter.last_join_diagnostics().unwrap_or_default(),
            requested_crossfade_samples: converter.join_requested_crossfade_samples().unwrap_or(0),
            seam_sample,
        });
    }
    ensure!(
        audio.len() >= end,
        "finite conversion did not recover the entire input interval"
    );
    audio.copy_within(content_delay..end, 0);
    audio.truncate(target_len);
    Ok(FiniteOutput { audio, chunks })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::model_rvc::{ChunkOutputConfig, ContentDelay, ModelOutput};
    use crate::sola::SmoothingKind;

    /// An exact identity generator with rolling model context and optional
    /// sample delay, isolating finite scheduling from learned-model variation.
    struct DelayedIdentity {
        history: Vec<f32>,
        input_delay: usize,
        candidate_samples: usize,
        sample_rate: u32,
    }

    impl VoiceModel for DelayedIdentity {
        fn input_content_delay(&self) -> ContentDelay {
            ContentDelay::from_samples(self.input_delay, self.sample_rate)
        }

        fn process(
            &mut self,
            input: &[f32],
            rate: u32,
            out: &mut Vec<f32>,
            pitch: &mut Vec<f32>,
        ) -> Result<ModelOutput> {
            assert_eq!(rate, self.sample_rate);
            self.history.extend_from_slice(input);
            let needed = self.candidate_samples + self.input_delay;
            if self.history.len() < needed {
                let mut padded = vec![0.0; needed - self.history.len()];
                padded.append(&mut self.history);
                self.history = padded;
            }
            let end = self.history.len() - self.input_delay;
            out.clear();
            out.extend_from_slice(&self.history[end - self.candidate_samples..end]);
            if self.history.len() > needed {
                self.history.drain(..self.history.len() - needed);
            }
            pitch.clear();
            pitch.resize(self.candidate_samples / 160, 200.0);
            Ok(ModelOutput {
                sample_rate: rate,
                inference_time: Duration::ZERO,
                embedder_time: Duration::ZERO,
                pitch_time: Duration::ZERO,
                rvc_time: Duration::ZERO,
                input_rms: crate::dsp::rms(input),
                voiced_ratio: 1.0,
                raw_output_samples: out.len(),
                output_rms: crate::dsp::rms(out),
                applied_output_gain: 1.0,
                feature_frames: pitch.len(),
                pitch_frames: pitch.len(),
                silent: false,
                convert_size: needed,
                out_size: out.len(),
                model_input_samples: needed,
                volume: 1.0,
            })
        }
    }

    fn converter(
        kind: SmoothingKind,
        input_delay: usize,
        crossfade_ms: u32,
        tail_ms: u32,
        output_rate: u32,
    ) -> ChunkConverter<DelayedIdentity> {
        converter_with_search(kind, input_delay, crossfade_ms, tail_ms, output_rate, 0)
    }

    fn converter_with_search(
        kind: SmoothingKind,
        input_delay: usize,
        crossfade_ms: u32,
        tail_ms: u32,
        output_rate: u32,
        search_ms: u32,
    ) -> ChunkConverter<DelayedIdentity> {
        let sample_rate = 16_000;
        let model = DelayedIdentity {
            history: Vec::new(),
            input_delay,
            candidate_samples: 1600 + 16 * (crossfade_ms + tail_ms + search_ms) as usize,
            sample_rate,
        };
        ChunkConverter::new(
            model,
            ChunkOutputConfig {
                kind,
                output_sample_rate: output_rate,
                output_chunk_samples: output_rate as usize / 10,
                crossfade_ms,
                sola_search_ms: search_ms,
                tail_discard_ms: tail_ms,
            },
        )
    }

    #[test]
    fn finite_identity_preserves_both_ends_and_partial_chunks() {
        for kind in [SmoothingKind::Sola, SmoothingKind::Psola] {
            for len in [0, 1, 79, 1599, 1600, 1601, 3199, 3200, 3201] {
                for (delay, fade, tail) in [(0, 0, 0), (0, 10, 0), (768, 10, 10), (2400, 85, 20)] {
                    let input: Vec<f32> = (0..len)
                        .map(|i| 0.2 + 0.15 * (i as f32 * 0.017).cos())
                        .collect();
                    let result = convert_finite(
                        converter(kind, delay, fade, tail, 16_000),
                        &input,
                        16_000,
                        1600,
                    )
                    .unwrap();
                    assert_eq!(result.audio.len(), input.len());
                    let max_error = result
                        .audio
                        .iter()
                        .zip(&input)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f32, f32::max);
                    assert!(
                        max_error < 1e-5,
                        "len={len} delay={delay} fade={fade} tail={tail} error={max_error}"
                    );
                }
            }
        }
    }

    #[test]
    fn finite_recovers_end_pulse_that_fixed_length_assembly_lost() {
        let mut input = vec![0.0; 3200];
        input[3120..].fill(0.5);
        let result = convert_finite(
            converter(SmoothingKind::Sola, 0, 10, 0, 16_000),
            &input,
            16_000,
            1600,
        )
        .unwrap();
        assert_eq!(result.audio.len(), input.len());
        assert!(result.audio[..3120].iter().all(|&sample| sample == 0.0));
        assert!(result.audio[3120..]
            .iter()
            .all(|&sample| (sample - 0.5).abs() < 1e-6));
        assert!(result.chunks.iter().any(|c| c.flushing));
    }

    #[test]
    fn finite_search_preserves_boundary_bursts_within_search_allowance() {
        let mut input = vec![0.0; 3201];
        input[..160].fill(0.5);
        input[3041..].fill(0.4);
        for kind in [SmoothingKind::Sola, SmoothingKind::Psola] {
            let converter = converter_with_search(kind, 768, 10, 10, 16_000, 12);
            let result = convert_finite(converter, &input, 16_000, 1600).unwrap();
            assert_eq!(result.audio.len(), input.len());
            // SOLA may advance either burst within its bounded search. This
            // checks recovery, not sample-exact phase preservation under search.
            assert!(result.audio[..640].iter().any(|&sample| sample > 0.3));
            assert!(result.audio[2561..].iter().any(|&sample| sample > 0.3));
        }
    }

    #[test]
    fn finite_resampling_uses_one_continuous_timeline_through_drain() {
        let input: Vec<f32> = (0..3201)
            .map(|i| 0.2 * (i as f32 * 0.073 + 0.4).cos())
            .collect();
        let expected = crate::dsp::resample_mono(&input, 16_000, 48_000).unwrap();
        let result = convert_finite(
            converter(SmoothingKind::Sola, 768, 10, 10, 48_000),
            &input,
            16_000,
            1600,
        )
        .unwrap();
        assert_eq!(result.audio.len(), input.len() * 3);
        let max_error = result
            .audio
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error < 1e-4, "max_error={max_error}");
        assert!(result
            .chunks
            .iter()
            .filter_map(|c| c.seam_sample)
            .all(|s| s < result.audio.len()));
    }
}
