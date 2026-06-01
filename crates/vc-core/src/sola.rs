use anyhow::Result;

use crate::dsp;
use crate::model_rvc::ModelOutput;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothingKind {
    Sola,
    Psola,
}

pub struct ChunkSmootherConfig {
    pub kind: SmoothingKind,
    pub output_chunk_samples: usize,
    pub output_sample_rate: u32,
    pub model_sample_rate: u32,
    pub crossfade_ms: u32,
    pub sola_search_ms: u32,
    pub tail_discard_ms: u32,
}

pub struct PreparedModelAudio {
    pub audio: Vec<f32>,
    pub sola_offset: usize,
}

struct SmoothedAudio {
    audio: Vec<f32>,
    sola_offset: usize,
}

const PSOLA_MIN_F0_HZ: f32 = 50.0;
const PSOLA_MAX_F0_HZ: f32 = 1_100.0;
const PSOLA_MAX_RELATIVE_F0_STDDEV: f32 = 0.20;
const PSOLA_MIN_RMS: f32 = 1e-4;
const PSOLA_MIN_SCORE: f32 = 0.05;

pub struct SolaChunkJoiner {
    chunk_samples: usize,
    crossfade_samples: usize,
    sola_search_samples: usize,
    tail_discard_samples: usize,
    sola_buffer: Vec<f32>,
    weighted_reference: Vec<f32>,
}

impl SolaChunkJoiner {
    fn new(
        chunk_samples: usize,
        crossfade_samples: usize,
        sola_search_samples: usize,
        tail_discard_samples: usize,
    ) -> Self {
        Self {
            chunk_samples,
            crossfade_samples,
            sola_search_samples,
            tail_discard_samples,
            sola_buffer: Vec::new(),
            weighted_reference: Vec::new(),
        }
    }

    fn prime(&mut self, audio: &[f32]) {
        let audio = self.candidate_audio(audio);
        if self.crossfade_samples == 0 || audio.is_empty() {
            self.sola_buffer.clear();
            return;
        }
        self.sola_buffer.clear();
        self.sola_buffer
            .extend_from_slice(tail_slice(audio, self.crossfade_samples));
    }

    fn process(&mut self, audio: &[f32]) -> SmoothedAudio {
        self.process_with_offset_selector(audio, |candidate, weighted_reference, max_offset| {
            dsp::sola_offset_with_threshold(candidate, weighted_reference, max_offset, 1e-4)
        })
    }

    fn process_with_offset_selector<F>(
        &mut self,
        audio: &[f32],
        mut select_offset: F,
    ) -> SmoothedAudio
    where
        F: FnMut(&[f32], &[f32], usize) -> usize,
    {
        let target_len = self.chunk_samples.max(1);
        let audio = self.candidate_audio(audio);

        if self.crossfade_samples == 0 || audio.is_empty() {
            let output = last_or_pad(audio, target_len);
            self.sola_buffer.clear();
            return SmoothedAudio {
                audio: output,
                sola_offset: 0,
            };
        }

        if self.sola_buffer.is_empty() {
            self.prime(audio);
            return SmoothedAudio {
                audio: vec![0.0; target_len],
                sola_offset: 0,
            };
        }

        let crossfade_len = self
            .sola_buffer
            .len()
            .min(self.crossfade_samples)
            .min(audio.len())
            .min(target_len);
        if crossfade_len == 0 {
            let output = last_or_pad(audio, target_len);
            self.update_sola_buffer(audio, 0);
            return SmoothedAudio {
                audio: output,
                sola_offset: 0,
            };
        }

        let max_offset = self
            .sola_search_samples
            .min(audio.len().saturating_sub(target_len));
        let candidate_len = (crossfade_len + max_offset).min(audio.len());
        let reference = &self.sola_buffer[self.sola_buffer.len() - crossfade_len..];
        vcclient_prev_strength_into(reference, &mut self.weighted_reference);
        let weighted_reference = self.weighted_reference.as_slice();
        let sola_offset = if max_offset > 0 {
            select_offset(&audio[..candidate_len], weighted_reference, max_offset).min(max_offset)
        } else {
            0
        };

        let output_end = sola_offset.saturating_add(target_len).min(audio.len());
        let mut output = audio[sola_offset..output_end].to_vec();
        pad_to_len_in_place(&mut output, target_len);
        output.truncate(target_len);
        vcclient_crossfade(reference, &mut output[..crossfade_len]);
        self.update_sola_buffer(audio, sola_offset);

        SmoothedAudio {
            audio: output,
            sola_offset,
        }
    }

    fn candidate_audio<'a>(&self, audio: &'a [f32]) -> &'a [f32] {
        // Drop unstable RVC tail samples before SOLA offset selection. The
        // worker asks the model for extra audio so emitted chunk length stays
        // fixed; do not move this trimming onto the real-time audio callback.
        let stable_len = audio.len().saturating_sub(self.tail_discard_samples);
        let audio = &audio[..stable_len];
        let window_len = self
            .chunk_samples
            .max(1)
            .saturating_add(self.crossfade_samples)
            .saturating_add(self.sola_search_samples);
        if audio.len() > window_len {
            &audio[audio.len() - window_len..]
        } else {
            audio
        }
    }

    fn update_sola_buffer(&mut self, audio: &[f32], sola_offset: usize) {
        if self.crossfade_samples == 0 {
            self.sola_buffer.clear();
            return;
        }

        let candidate = if sola_offset < self.sola_search_samples {
            let start = audio
                .len()
                .saturating_sub(self.sola_search_samples + self.crossfade_samples - sola_offset);
            let end = audio
                .len()
                .saturating_sub(self.sola_search_samples - sola_offset);
            if start < end && end <= audio.len() {
                &audio[start..end]
            } else {
                tail_slice(audio, self.crossfade_samples)
            }
        } else {
            tail_slice(audio, self.crossfade_samples)
        };
        self.sola_buffer.clear();
        self.sola_buffer.extend_from_slice(candidate);
    }
}

pub struct PsolaChunkJoiner {
    inner: SolaChunkJoiner,
    sample_rate: u32,
    pitch_mark_weights: Vec<f32>,
}

impl PsolaChunkJoiner {
    fn new(
        chunk_samples: usize,
        crossfade_samples: usize,
        sola_search_samples: usize,
        tail_discard_samples: usize,
        sample_rate: u32,
    ) -> Self {
        Self {
            inner: SolaChunkJoiner::new(
                chunk_samples,
                crossfade_samples,
                sola_search_samples,
                tail_discard_samples,
            ),
            sample_rate,
            pitch_mark_weights: Vec::new(),
        }
    }

    #[cfg(test)]
    fn prime(&mut self, audio: &[f32]) {
        self.inner.prime(audio);
    }

    fn process(&mut self, audio: &[f32], pitchf: &[f32]) -> SmoothedAudio {
        let Some(period_samples) = stable_pitch_period_samples(pitchf, self.sample_rate) else {
            return self.inner.process(audio);
        };

        // PSOLA is deliberately kept in the worker-side model domain. Moving
        // this into the audio callback would add allocation and O(search*fade)
        // work to the real-time path.
        let pitch_mark_weights = &mut self.pitch_mark_weights;
        self.inner.process_with_offset_selector(
            audio,
            |candidate, weighted_reference, max_offset| {
                psola_offset_with_period_with_scratch(
                    candidate,
                    weighted_reference,
                    max_offset,
                    period_samples,
                    pitch_mark_weights,
                )
                .unwrap_or_else(|| {
                    dsp::sola_offset_with_threshold(
                        candidate,
                        weighted_reference,
                        max_offset,
                        PSOLA_MIN_RMS,
                    )
                })
            },
        )
    }

    fn candidate_audio<'a>(&self, audio: &'a [f32]) -> &'a [f32] {
        self.inner.candidate_audio(audio)
    }

    fn chunk_samples(&self) -> usize {
        self.inner.chunk_samples
    }

    fn crossfade_samples(&self) -> usize {
        self.inner.crossfade_samples
    }

    fn sola_search_samples(&self) -> usize {
        self.inner.sola_search_samples
    }
}

pub enum ChunkSmoother {
    Sola(SolaChunkJoiner),
    Psola(PsolaChunkJoiner),
}

impl ChunkSmoother {
    #[cfg(test)]
    fn prime(&mut self, audio: &[f32]) {
        match self {
            Self::Sola(joiner) => joiner.prime(audio),
            Self::Psola(joiner) => joiner.prime(audio),
        }
    }

    fn process(&mut self, audio: &[f32], pitchf: &[f32]) -> SmoothedAudio {
        match self {
            Self::Sola(joiner) => joiner.process(audio),
            Self::Psola(joiner) => joiner.process(audio, pitchf),
        }
    }

    pub fn prime_model_output(&mut self, audio: &[f32], pitchf: &[f32]) {
        let _ = self.process(audio, pitchf);
    }

    fn candidate_audio<'a>(&self, audio: &'a [f32]) -> &'a [f32] {
        match self {
            Self::Sola(joiner) => joiner.candidate_audio(audio),
            Self::Psola(joiner) => joiner.candidate_audio(audio),
        }
    }

    fn chunk_samples(&self) -> usize {
        match self {
            Self::Sola(joiner) => joiner.chunk_samples,
            Self::Psola(joiner) => joiner.chunk_samples(),
        }
    }

    pub fn crossfade_samples(&self) -> usize {
        match self {
            Self::Sola(joiner) => joiner.crossfade_samples,
            Self::Psola(joiner) => joiner.crossfade_samples(),
        }
    }

    pub fn sola_search_samples(&self) -> usize {
        match self {
            Self::Sola(joiner) => joiner.sola_search_samples,
            Self::Psola(joiner) => joiner.sola_search_samples(),
        }
    }
}

fn stable_pitch_period_samples(pitchf: &[f32], sample_rate: u32) -> Option<usize> {
    if pitchf.is_empty() || sample_rate == 0 {
        return None;
    }

    let (voiced_count, voiced_sum) = pitchf
        .iter()
        .copied()
        .filter(|f0| f0.is_finite() && (PSOLA_MIN_F0_HZ..=PSOLA_MAX_F0_HZ).contains(f0))
        .fold((0usize, 0.0f32), |(count, sum), f0| (count + 1, sum + f0));
    if voiced_count * 2 < pitchf.len() || voiced_count == 0 {
        return None;
    }

    let mean = voiced_sum / voiced_count as f32;
    if mean <= 0.0 {
        return None;
    }
    let variance = pitchf
        .iter()
        .copied()
        .filter(|f0| f0.is_finite() && (PSOLA_MIN_F0_HZ..=PSOLA_MAX_F0_HZ).contains(f0))
        .map(|f0| {
            let delta = f0 - mean;
            delta * delta
        })
        .sum::<f32>()
        / voiced_count as f32;
    if variance.sqrt() / mean > PSOLA_MAX_RELATIVE_F0_STDDEV {
        return None;
    }

    let period = (sample_rate as f32 / mean).round() as usize;
    (period >= 2).then_some(period)
}

#[cfg(test)]
fn psola_offset_with_period(
    candidate: &[f32],
    reference: &[f32],
    search: usize,
    period: usize,
) -> Option<usize> {
    let mut weights = Vec::new();
    psola_offset_with_period_with_scratch(candidate, reference, search, period, &mut weights)
}

fn psola_offset_with_period_with_scratch(
    candidate: &[f32],
    reference: &[f32],
    search: usize,
    period: usize,
    weights: &mut Vec<f32>,
) -> Option<usize> {
    let frame = reference.len().min(candidate.len());
    if frame == 0 || period < 2 {
        return None;
    }

    let max_offset = search.min(candidate.len().saturating_sub(frame));
    let candidate_len = (frame + max_offset).min(candidate.len());
    if dsp::rms(&candidate[..candidate_len]) < PSOLA_MIN_RMS
        || dsp::rms(&reference[..frame]) < PSOLA_MIN_RMS
    {
        return None;
    }

    psola_pitch_mark_weights_into(&reference[..frame], period, weights)?;
    let mut best_offset = 0;
    let mut best_score = f32::MIN;
    for offset in 0..=max_offset {
        let window = &candidate[offset..offset + frame];
        let pitch_score = weighted_correlation(window, &reference[..frame], weights);
        let full_score = normalized_correlation(window, &reference[..frame]);
        let score = pitch_score * 0.8 + full_score * 0.2;
        if score.is_finite() && score > best_score {
            best_score = score;
            best_offset = offset;
        }
    }

    (best_score >= PSOLA_MIN_SCORE).then_some(best_offset)
}

fn psola_pitch_mark_weights_into(
    reference: &[f32],
    period: usize,
    weights: &mut Vec<f32>,
) -> Option<()> {
    weights.clear();
    let (center, peak) = reference
        .iter()
        .enumerate()
        .map(|(index, sample)| (index, sample.abs()))
        .max_by(|(_, a), (_, b)| a.total_cmp(b))?;
    if peak < PSOLA_MIN_RMS {
        return None;
    }

    let radius = (period / 6).max(1);
    weights.resize(reference.len(), 0.0);
    let mut mark = center;
    loop {
        add_pitch_mark_weight(weights, mark, radius);
        if mark < period {
            break;
        }
        mark -= period;
    }
    mark = center + period;
    while mark < reference.len() {
        add_pitch_mark_weight(weights, mark, radius);
        mark += period;
    }

    Some(())
}

fn add_pitch_mark_weight(weights: &mut [f32], mark: usize, radius: usize) {
    let start = mark.saturating_sub(radius);
    let end = (mark + radius + 1).min(weights.len());
    for (index, weight) in weights.iter_mut().enumerate().take(end).skip(start) {
        let distance = index.abs_diff(mark);
        let mark_weight = 1.0 - distance as f32 / (radius + 1) as f32;
        *weight = (*weight).max(mark_weight);
    }
}

fn normalized_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut nom = 0.0;
    let mut a_energy = 0.0;
    let mut b_energy = 0.0;
    for (&x, &y) in a.iter().zip(b).take(n) {
        nom += x * y;
        a_energy += x * x;
        b_energy += y * y;
    }
    nom / (a_energy * b_energy + 1e-9).sqrt()
}

fn weighted_correlation(a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
    let n = a.len().min(b.len()).min(weights.len());
    let mut nom = 0.0;
    let mut a_energy = 0.0;
    let mut b_energy = 0.0;
    for ((&x, &y), &weight) in a.iter().zip(b).zip(weights).take(n) {
        nom += x * y * weight;
        a_energy += x * x * weight;
        b_energy += y * y * weight;
    }
    nom / (a_energy * b_energy + 1e-9).sqrt()
}

fn vcclient_prev_strength_into(input: &[f32], output: &mut Vec<f32>) {
    let n = input.len();
    output.clear();
    output.reserve(n);
    output.extend(
        input
            .iter()
            .enumerate()
            .map(|(i, &sample)| sample * vcclient_crossfade_gains(i, n).0),
    );
}

fn vcclient_crossfade(prev_tail: &[f32], current: &mut [f32]) {
    let n = prev_tail.len().min(current.len());
    for i in 0..n {
        let (prev_gain, cur_gain) = vcclient_crossfade_gains(i, n);
        current[i] = prev_tail[i] * prev_gain + current[i] * cur_gain;
    }
}

fn vcclient_crossfade_gains(index: usize, len: usize) -> (f32, f32) {
    if len == 0 {
        return (0.0, 1.0);
    }
    let fade_start = len / 10;
    let fade_end = (len * 9) / 10;
    if index < fade_start {
        return (1.0, 0.0);
    }
    if index >= fade_end || fade_end <= fade_start {
        return (0.0, 1.0);
    }
    let t = (index - fade_start) as f32 / (fade_end - fade_start) as f32;
    let prev_gain = (t * std::f32::consts::FRAC_PI_2).cos().powi(2);
    let cur_gain = ((1.0 - t) * std::f32::consts::FRAC_PI_2).cos().powi(2);
    (prev_gain, cur_gain)
}

pub fn ms_to_samples(sample_rate: u32, ms: u32) -> usize {
    ((sample_rate as u64 * ms as u64) / 1000) as usize
}

fn rescale_samples(samples: usize, from_sample_rate: u32, to_sample_rate: u32) -> usize {
    if samples == 0 || from_sample_rate == 0 || to_sample_rate == 0 {
        return 0;
    }

    let numerator = samples as u64 * to_sample_rate as u64;
    ((numerator + from_sample_rate as u64 / 2) / from_sample_rate as u64) as usize
}

fn model_domain_sola_joiner(config: &ChunkSmootherConfig) -> SolaChunkJoiner {
    // SOLA/PSOLA must stay on the worker side in the model output domain.
    // Moving this work to the real-time callback would reintroduce allocation
    // and O(search*fade) processing on the audio thread.
    SolaChunkJoiner::new(
        rescale_samples(
            config.output_chunk_samples,
            config.output_sample_rate,
            config.model_sample_rate,
        )
        .max(1),
        ms_to_samples(config.model_sample_rate, config.crossfade_ms),
        ms_to_samples(config.model_sample_rate, config.sola_search_ms),
        ms_to_samples(config.model_sample_rate, config.tail_discard_ms),
    )
}

pub fn model_domain_chunk_smoother(config: ChunkSmootherConfig) -> ChunkSmoother {
    match config.kind {
        SmoothingKind::Sola => ChunkSmoother::Sola(model_domain_sola_joiner(&config)),
        SmoothingKind::Psola => ChunkSmoother::Psola(PsolaChunkJoiner::new(
            rescale_samples(
                config.output_chunk_samples,
                config.output_sample_rate,
                config.model_sample_rate,
            )
            .max(1),
            ms_to_samples(config.model_sample_rate, config.crossfade_ms),
            ms_to_samples(config.model_sample_rate, config.sola_search_ms),
            ms_to_samples(config.model_sample_rate, config.tail_discard_ms),
            config.model_sample_rate,
        )),
    }
}

fn resample_to_output_domain(
    audio: &[f32],
    from_sample_rate: u32,
    to_sample_rate: u32,
) -> Result<Vec<f32>> {
    dsp::resample_mono(audio, from_sample_rate as usize, to_sample_rate as usize)
}

fn resample_owned_to_output_domain(
    audio: Vec<f32>,
    from_sample_rate: u32,
    to_sample_rate: u32,
) -> Result<Vec<f32>> {
    if from_sample_rate == to_sample_rate {
        Ok(audio)
    } else {
        resample_to_output_domain(&audio, from_sample_rate, to_sample_rate)
    }
}

fn fit_to_len_in_place(input: &mut Vec<f32>, len: usize) {
    pad_to_len_in_place(input, len);
    input.truncate(len);
}

fn tail_slice(input: &[f32], len: usize) -> &[f32] {
    if input.len() <= len {
        input
    } else {
        &input[input.len() - len..]
    }
}

fn pad_to_len_in_place(input: &mut Vec<f32>, len: usize) {
    if input.len() < len {
        input.resize(len, 0.0);
    }
}

fn last_or_pad(input: &[f32], len: usize) -> Vec<f32> {
    if input.len() >= len {
        input[input.len() - len..].to_vec()
    } else {
        let mut output = vec![0.0; len - input.len()];
        output.extend_from_slice(input);
        output
    }
}

pub fn prepare_model_output(
    out: ModelOutput,
    output_sample_rate: u32,
    output_chunk_samples: usize,
    joiner: &mut ChunkSmoother,
    final_tail: Option<&mut Vec<f32>>,
) -> Result<PreparedModelAudio> {
    let model_sample_rate = out.sample_rate;
    let candidate = joiner.candidate_audio(&out.audio);
    let smoothed = joiner.process(&out.audio, &out.pitchf);
    if let Some(final_tail) = final_tail {
        let tail_start = smoothed
            .sola_offset
            .saturating_add(joiner.chunk_samples())
            .min(candidate.len());
        final_tail.clear();
        let tail = &candidate[tail_start..];
        if model_sample_rate == output_sample_rate {
            final_tail.extend_from_slice(tail);
        } else {
            final_tail.extend(resample_to_output_domain(
                tail,
                model_sample_rate,
                output_sample_rate,
            )?);
        }
    }

    let mut audio =
        resample_owned_to_output_domain(smoothed.audio, model_sample_rate, output_sample_rate)?;
    fit_to_len_in_place(&mut audio, output_chunk_samples);
    Ok(PreparedModelAudio {
        audio,
        sola_offset: smoothed.sola_offset,
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{
        prepare_model_output, psola_offset_with_period, stable_pitch_period_samples,
        vcclient_crossfade_gains, ChunkSmoother, PsolaChunkJoiner, SolaChunkJoiner,
    };
    use crate::model_rvc::ModelOutput;

    #[test]
    fn sola_chunk_joiner_uses_detected_offset() {
        let mut joiner = SolaChunkJoiner::new(4, 2, 2, 0);

        let first = joiner.process(&[0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(first.audio, vec![0.0, 0.0, 0.0, 0.0]);

        let second = joiner.process(&[0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0]);

        assert_eq!(second.sola_offset, 2);
        assert_eq!(second.audio.len(), 4);
    }

    #[test]
    fn sola_chunk_joiner_primes_from_startup_tail() {
        let mut joiner = SolaChunkJoiner::new(4, 2, 2, 0);

        joiner.prime(&[0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0]);
        let output = joiner.process(&[0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0]);

        assert_eq!(output.sola_offset, 2);
        assert_eq!(output.audio, vec![4.0, 0.5, 6.0, 7.0]);
    }

    #[test]
    fn vcclient_crossfade_gains_keep_flat_edges() {
        assert_eq!(vcclient_crossfade_gains(0, 10), (1.0, 0.0));
        assert_eq!(vcclient_crossfade_gains(9, 10), (0.0, 1.0));
    }

    #[test]
    fn sola_chunk_joiner_right_aligns_short_outputs_without_crossfade() {
        let mut joiner = SolaChunkJoiner::new(5, 0, 1, 0);

        let output = joiner.process(&[1.0, 2.0]);

        assert_eq!(output.audio, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn sola_chunk_joiner_discards_unstable_tail_before_output_selection() {
        let mut joiner = SolaChunkJoiner::new(4, 0, 0, 2);

        let output = joiner.process(&[1.0, 2.0, 3.0, 4.0, 100.0, 101.0]);

        assert_eq!(output.audio, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn stable_pitch_period_requires_voiced_stable_f0() {
        assert_eq!(
            stable_pitch_period_samples(&[100.0, 102.0, 98.0], 48_000),
            Some(480)
        );
        assert_eq!(
            stable_pitch_period_samples(&[0.0, 0.0, 100.0], 48_000),
            None
        );
        assert_eq!(
            stable_pitch_period_samples(&[100.0, 300.0, 500.0], 48_000),
            None
        );
    }

    #[test]
    fn psola_offset_uses_pitch_period_marks() {
        let reference = [0.0, 1.0, 0.0, -0.1, 0.0, 1.0, 0.0, -0.1];
        let mut candidate = vec![0.2, -0.2, 0.2, -0.2];
        candidate.extend_from_slice(&reference);

        assert_eq!(
            psola_offset_with_period(&candidate, &reference, 4, 4),
            Some(4)
        );
    }

    #[test]
    fn psola_chunk_joiner_falls_back_to_sola_without_voiced_f0() {
        let mut joiner = PsolaChunkJoiner::new(4, 2, 2, 0, 48_000);

        joiner.prime(&[0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0]);
        let output = joiner.process(&[0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0], &[]);

        assert_eq!(output.sola_offset, 2);
        assert_eq!(output.audio, vec![4.0, 0.5, 6.0, 7.0]);
    }

    #[test]
    fn smoothed_model_output_reports_sola_offset_before_resampling() {
        let mut joiner = ChunkSmoother::Sola(SolaChunkJoiner::new(4, 2, 2, 0));
        joiner.prime(&[0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0]);

        let prepared = prepare_model_output(
            synthetic_model_output(vec![0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0], 48_000),
            24_000,
            2,
            &mut joiner,
            None,
        )
        .unwrap();

        assert_eq!(prepared.sola_offset, 2);
        assert_eq!(prepared.audio.len(), 2);
    }

    #[test]
    fn smoothed_model_output_resamples_final_tail_after_sola() {
        let mut joiner = ChunkSmoother::Sola(SolaChunkJoiner::new(4, 2, 2, 0));
        joiner.prime(&[0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0]);
        let mut final_tail = Vec::new();

        let _ = prepare_model_output(
            synthetic_model_output(vec![0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0], 48_000),
            24_000,
            2,
            &mut joiner,
            Some(&mut final_tail),
        )
        .unwrap();
        let expected_tail = crate::dsp::resample_mono(&[8.0, 9.0], 48_000, 24_000).unwrap();

        assert_eq!(final_tail, expected_tail);
    }

    #[test]
    fn smoothed_model_output_excludes_discarded_tail_from_final_tail() {
        let mut joiner = ChunkSmoother::Sola(SolaChunkJoiner::new(4, 2, 2, 2));
        joiner.prime(&[0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0, 100.0, 101.0]);
        let mut final_tail = Vec::new();

        let prepared = prepare_model_output(
            synthetic_model_output(
                vec![0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0, 100.0, 101.0],
                48_000,
            ),
            48_000,
            4,
            &mut joiner,
            Some(&mut final_tail),
        )
        .unwrap();

        assert_eq!(prepared.audio, vec![4.0, 0.5, 6.0, 7.0]);
        assert_eq!(final_tail, vec![8.0, 9.0]);
    }

    #[test]
    fn smoothed_model_output_keeps_final_tail_with_psola() {
        let mut joiner = ChunkSmoother::Psola(PsolaChunkJoiner::new(4, 2, 2, 0, 48_000));
        joiner.prime(&[0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0]);
        let mut final_tail = Vec::new();

        let prepared = prepare_model_output(
            synthetic_model_output_with_pitchf(
                vec![0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0],
                48_000,
                vec![100.0; 8],
            ),
            48_000,
            4,
            &mut joiner,
            Some(&mut final_tail),
        )
        .unwrap();

        assert_eq!(prepared.audio.len(), 4);
        assert_eq!(final_tail.len(), 2);
    }

    fn synthetic_model_output(audio: Vec<f32>, sample_rate: u32) -> ModelOutput {
        synthetic_model_output_with_pitchf(audio, sample_rate, Vec::new())
    }

    fn synthetic_model_output_with_pitchf(
        audio: Vec<f32>,
        sample_rate: u32,
        pitchf: Vec<f32>,
    ) -> ModelOutput {
        ModelOutput {
            raw_output_samples: audio.len(),
            output_rms: crate::dsp::rms(&audio),
            convert_size: audio.len(),
            out_size: audio.len(),
            model_input_samples: audio.len(),
            audio,
            pitchf,
            sample_rate,
            inference_time: Duration::ZERO,
            embedder_time: Duration::ZERO,
            pitch_time: Duration::ZERO,
            rvc_time: Duration::ZERO,
            input_rms: 0.0,
            voiced_ratio: 0.0,
            applied_output_gain: 1.0,
            feature_frames: 0,
            pitch_frames: 0,
            silent: false,
            volume: 0.0,
        }
    }
}
