use anyhow::Result;

use crate::dsp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothingKind {
    Sola,
    Psola,
}

/// Per-chunk record of how the smoother joined the latest chunk. Diagnostics
/// only: filled on every `process` call so offline tooling (the CLI join report)
/// can correlate audible seam artifacts with the joiner's decisions. The values
/// are in the model output domain (same domain SOLA runs in), not the device
/// output domain. Read via [`ChunkSmoother::last_diagnostics`].
#[derive(Clone, Copy, Debug, Default)]
pub struct JoinDiagnostics {
    pub kind: Option<SmoothingKind>,
    /// Offset chosen within the candidate window (samples).
    pub sola_offset: usize,
    /// Maximum offset the search was allowed to consider this chunk.
    pub max_offset: usize,
    /// Normalized cross-correlation between the chosen window and the weighted
    /// reference tail (the metric SOLA maximizes). `0.0` when no join happened.
    pub correlation: f32,
    /// Crossfade length actually applied. Capped by the output chunk length, so
    /// a value below `crossfade_samples` means the chunk was shorter than the
    /// configured crossfade window.
    pub crossfade_len: usize,
    /// PSOLA pitch period (samples) when a stable voiced pitch was detected.
    pub pitch_period: Option<usize>,
    /// `true` when PSOLA could not use pitch-synchronous alignment and fell back
    /// to plain SOLA (unvoiced/unstable F0, or no acceptable pitch-mark score).
    pub psola_fallback: bool,
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
    // Reused storage for the joined chunk so `process` does not allocate a fresh
    // Vec every chunk. Holds the latest smoothed output; read via `output()`.
    output_buffer: Vec<f32>,
    // Diagnostics for the most recent `process` call. Stored (not returned) so the
    // hot `process` signature stays a plain `usize`; read via `last_diagnostics`.
    last_diagnostics: JoinDiagnostics,
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
            output_buffer: Vec::new(),
            last_diagnostics: JoinDiagnostics::default(),
        }
    }

    #[cfg(test)]
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

    /// Joins `audio` against the retained crossfade history, writing the result
    /// into `self.output_buffer` (reused across chunks) and returning the chosen
    /// SOLA offset. Read the joined samples via [`Self::output`].
    fn process(&mut self, audio: &[f32]) -> usize {
        self.process_with_offset_selector(audio, |candidate, weighted_reference, max_offset| {
            dsp::sola_offset_with_threshold(candidate, weighted_reference, max_offset, 1e-4)
        })
    }

    fn output(&self) -> &[f32] {
        &self.output_buffer
    }

    fn process_with_offset_selector<F>(&mut self, audio: &[f32], mut select_offset: F) -> usize
    where
        F: FnMut(&[f32], &[f32], usize) -> usize,
    {
        let target_len = self.chunk_samples.max(1);
        let audio = self.candidate_audio(audio);

        if self.crossfade_samples == 0 || audio.is_empty() {
            last_or_pad_into(audio, target_len, &mut self.output_buffer);
            self.sola_buffer.clear();
            self.last_diagnostics = JoinDiagnostics::default();
            return 0;
        }

        if self.sola_buffer.is_empty() {
            // `audio` is already the candidate: calling `prime` here would
            // apply tail_discard a second time and seed the wrong startup seam.
            self.sola_buffer
                .extend_from_slice(tail_slice(audio, self.crossfade_samples));
            self.output_buffer.clear();
            self.output_buffer.resize(target_len, 0.0);
            self.last_diagnostics = JoinDiagnostics::default();
            return 0;
        }

        let crossfade_len = self
            .sola_buffer
            .len()
            .min(self.crossfade_samples)
            .min(audio.len())
            .min(target_len);
        if crossfade_len == 0 {
            last_or_pad_into(audio, target_len, &mut self.output_buffer);
            self.update_sola_buffer(audio, 0);
            self.last_diagnostics = JoinDiagnostics::default();
            return 0;
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

        // Diagnostics only: the normalized correlation at the chosen offset is the
        // score the SOLA search maximizes (`dsp::normalized_correlation` mirrors
        // `dsp::sola_offset`). One extra dot over the crossfade window per chunk,
        // on the worker thread — negligible and never on the audio callback.
        let correlation = dsp::normalized_correlation(
            &audio[sola_offset..sola_offset + crossfade_len],
            weighted_reference,
        );

        let output_end = sola_offset.saturating_add(target_len).min(audio.len());
        // `reference` borrows `self.sola_buffer`; `output_buffer` is a disjoint
        // field, so writing the joined chunk here while reading `reference` is a
        // valid split borrow (same pattern as `weighted_reference` above).
        let output = &mut self.output_buffer;
        output.clear();
        output.extend_from_slice(&audio[sola_offset..output_end]);
        pad_to_len_in_place(output, target_len);
        output.truncate(target_len);
        vcclient_crossfade(reference, &mut output[..crossfade_len]);
        self.update_sola_buffer(audio, sola_offset);

        self.last_diagnostics = JoinDiagnostics {
            sola_offset,
            max_offset,
            correlation,
            crossfade_len,
            ..JoinDiagnostics::default()
        };

        sola_offset
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

    fn process(&mut self, audio: &[f32], pitchf: &[f32]) -> usize {
        let Some(period_samples) = stable_pitch_period_samples(pitchf, self.sample_rate) else {
            // No stable voiced pitch: plain SOLA. `inner.process` records the base
            // diagnostics; tag this as a fallback with no pitch period.
            let offset = self.inner.process(audio);
            self.inner.last_diagnostics.pitch_period = None;
            self.inner.last_diagnostics.psola_fallback = true;
            return offset;
        };

        // PSOLA is deliberately kept in the worker-side model domain. Moving
        // this into the audio callback would add allocation and O(search*fade)
        // work to the real-time path.
        let pitch_mark_weights = &mut self.pitch_mark_weights;
        let mut fell_back = false;
        let offset = self.inner.process_with_offset_selector(
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
                    fell_back = true;
                    dsp::sola_offset_with_threshold(
                        candidate,
                        weighted_reference,
                        max_offset,
                        PSOLA_MIN_RMS,
                    )
                })
            },
        );
        self.inner.last_diagnostics.pitch_period = Some(period_samples);
        self.inner.last_diagnostics.psola_fallback = fell_back;
        offset
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

    pub(crate) fn process(&mut self, audio: &[f32], pitchf: &[f32]) -> usize {
        match self {
            Self::Sola(joiner) => joiner.process(audio),
            Self::Psola(joiner) => joiner.process(audio, pitchf),
        }
    }

    /// The most recent joined chunk (model domain), valid after [`Self::process`].
    pub(crate) fn output(&self) -> &[f32] {
        match self {
            Self::Sola(joiner) => joiner.output(),
            Self::Psola(joiner) => joiner.inner.output(),
        }
    }

    pub fn prime_model_output(&mut self, audio: &[f32], pitchf: &[f32]) {
        let _ = self.process(audio, pitchf);
    }

    /// Diagnostics for the most recent join (the latest [`prepare_model_output`]
    /// or [`Self::prime_model_output`] call), with `kind` set to the active
    /// smoother. Diagnostics-only; see [`JoinDiagnostics`].
    pub fn last_diagnostics(&self) -> JoinDiagnostics {
        match self {
            Self::Sola(joiner) => JoinDiagnostics {
                kind: Some(SmoothingKind::Sola),
                ..joiner.last_diagnostics
            },
            Self::Psola(joiner) => JoinDiagnostics {
                kind: Some(SmoothingKind::Psola),
                ..joiner.inner.last_diagnostics
            },
        }
    }

    fn candidate_audio<'a>(&self, audio: &'a [f32]) -> &'a [f32] {
        match self {
            Self::Sola(joiner) => joiner.candidate_audio(audio),
            Self::Psola(joiner) => joiner.candidate_audio(audio),
        }
    }

    pub(crate) fn chunk_samples(&self) -> usize {
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

    /// Nominal content held behind the model candidate's end. SOLA can advance
    /// the selected candidate by up to the search window, but finite cropping
    /// must use this maximum hold so a later offset never drops the source tail.
    /// With overlap disabled `last_or_pad_into` selects the final hop directly.
    pub(crate) fn content_delay_samples(&self) -> usize {
        let joiner = match self {
            Self::Sola(joiner) => joiner,
            Self::Psola(joiner) => &joiner.inner,
        };
        joiner.tail_discard_samples
            + if joiner.crossfade_samples == 0 {
                0
            } else {
                joiner.crossfade_samples + joiner.sola_search_samples
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
    let reference = &reference[..frame];
    let weights = &weights[..frame];

    // The reference-side energies (`sum(y*y)` and `sum(y*y*w)`) are
    // offset-independent; hoist them out of the search instead of recomputing
    // them per offset as the previous `weighted_correlation` /
    // `normalized_correlation` calls did.
    let mut ref_energy_weighted = 0.0f32;
    let mut ref_energy_plain = 0.0f32;
    for (&y, &w) in reference.iter().zip(weights) {
        ref_energy_weighted += y * y * w;
        ref_energy_plain += y * y;
    }

    // The unweighted window energy slides by one sample per offset (see
    // `dsp::sola_offset`); only the cross-correlation numerators and the
    // weighted window energy must be recomputed per offset.
    let mut window_energy_plain = dsp::dot(&candidate[..frame], &candidate[..frame]);
    let mut best_offset = 0;
    let mut best_score = f32::MIN;
    for offset in 0..=max_offset {
        if offset > 0 {
            let leaving = candidate[offset - 1];
            let entering = candidate[offset + frame - 1];
            window_energy_plain += entering * entering - leaving * leaving;
        }
        let window = &candidate[offset..offset + frame];
        let (nom_weighted, window_energy_weighted) =
            weighted_window_terms(window, reference, weights);
        let pitch_score =
            nom_weighted / (window_energy_weighted * ref_energy_weighted + 1e-9).sqrt();
        let nom_plain = dsp::dot(window, reference);
        let full_score = nom_plain / (window_energy_plain * ref_energy_plain + 1e-9).sqrt();
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

/// Offset-dependent terms of the pitch-weighted correlation: the numerator
/// `sum(x*y*w)` and the weighted window energy `sum(x*x*w)`. The reference-side
/// energy `sum(y*y*w)` is offset-independent and hoisted by the caller, so it is
/// not recomputed here.
fn weighted_window_terms(window: &[f32], reference: &[f32], weights: &[f32]) -> (f32, f32) {
    // Split both reductions into independent accumulator lanes so the compiler
    // can auto-vectorize them (same rationale as `dsp::dot`: a single `+=`
    // reduction forces a strict, non-associative f32 chain that blocks SIMD).
    // Eight lanes match a 256-bit register; lane assignment is fixed, so the
    // result is deterministic run-to-run.
    const LANES: usize = 8;
    let n = window.len().min(reference.len()).min(weights.len());
    let window = &window[..n];
    let reference = &reference[..n];
    let weights = &weights[..n];
    let mut nom = [0.0f32; LANES];
    let mut window_energy = [0.0f32; LANES];
    let mut w_chunks = window.chunks_exact(LANES);
    let mut r_chunks = reference.chunks_exact(LANES);
    let mut wt_chunks = weights.chunks_exact(LANES);
    for ((cw, cr), cwt) in w_chunks
        .by_ref()
        .zip(r_chunks.by_ref())
        .zip(wt_chunks.by_ref())
    {
        for lane in 0..LANES {
            let xw = cw[lane] * cwt[lane];
            nom[lane] += xw * cr[lane];
            window_energy[lane] += xw * cw[lane];
        }
    }
    let mut nom_tail = 0.0;
    let mut energy_tail = 0.0;
    for ((&x, &y), &weight) in w_chunks
        .remainder()
        .iter()
        .zip(r_chunks.remainder())
        .zip(wt_chunks.remainder())
    {
        let xw = x * weight;
        nom_tail += xw * y;
        energy_tail += xw * x;
    }
    (
        nom.iter().sum::<f32>() + nom_tail,
        window_energy.iter().sum::<f32>() + energy_tail,
    )
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

/// Crossfade window (model-domain samples) for a chunk, capped so the overlap
/// stays below the chunk hop with a pure-current margin.
///
/// The configured `crossfade_ms` is chunk-independent. When the output chunk is
/// shorter than the crossfade window the pairwise crossfade can no longer span a
/// seam (the overlap reaches the full hop and successive overlaps collide),
/// which leaves an audible step at every boundary — the small-`chunk_ms` artifact
/// this guards against. Capping at 3/4 of the chunk keeps at least a 25%
/// pure-current region (overlap/hop ≤ 0.75, comfortably inside the measured
/// clean regime). Large chunks (realtime 500 ms, WAV 2000 ms) sit far under the
/// cap and are unaffected; only sub-~110 ms chunks clamp.
fn model_domain_crossfade_samples(config: &ChunkSmootherConfig, chunk_samples: usize) -> usize {
    let crossfade = ms_to_samples(config.model_sample_rate, config.crossfade_ms);
    crossfade.min(chunk_samples * 3 / 4)
}

fn model_domain_chunk_samples(config: &ChunkSmootherConfig) -> usize {
    rescale_samples(
        config.output_chunk_samples,
        config.output_sample_rate,
        config.model_sample_rate,
    )
    .max(1)
}

fn model_domain_sola_joiner(config: &ChunkSmootherConfig) -> SolaChunkJoiner {
    // SOLA/PSOLA must stay on the worker side in the model output domain.
    // Moving this work to the real-time callback would reintroduce allocation
    // and O(search*fade) processing on the audio thread.
    let chunk_samples = model_domain_chunk_samples(config);
    SolaChunkJoiner::new(
        chunk_samples,
        model_domain_crossfade_samples(config, chunk_samples),
        ms_to_samples(config.model_sample_rate, config.sola_search_ms),
        ms_to_samples(config.model_sample_rate, config.tail_discard_ms),
    )
}

pub fn model_domain_chunk_smoother(config: ChunkSmootherConfig) -> ChunkSmoother {
    match config.kind {
        SmoothingKind::Sola => ChunkSmoother::Sola(model_domain_sola_joiner(&config)),
        SmoothingKind::Psola => {
            let chunk_samples = model_domain_chunk_samples(&config);
            ChunkSmoother::Psola(PsolaChunkJoiner::new(
                chunk_samples,
                model_domain_crossfade_samples(&config, chunk_samples),
                ms_to_samples(config.model_sample_rate, config.sola_search_ms),
                ms_to_samples(config.model_sample_rate, config.tail_discard_ms),
                config.model_sample_rate,
            ))
        }
    }
}

fn resample_to_output_domain(
    audio: &[f32],
    from_sample_rate: u32,
    to_sample_rate: u32,
) -> Result<Vec<f32>> {
    dsp::resample_mono(audio, from_sample_rate as usize, to_sample_rate as usize)
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

fn last_or_pad_into(input: &[f32], len: usize, output: &mut Vec<f32>) {
    output.clear();
    if input.len() >= len {
        output.extend_from_slice(&input[input.len() - len..]);
    } else {
        output.resize(len - input.len(), 0.0);
        output.extend_from_slice(input);
    }
}

/// Legacy finite-buffer helper. Stateful streams must use `ChunkConverter`,
/// which retains one output resampler for the entire stream. Calling this
/// helper repeatedly at different sample rates restarts the filter each time;
/// its independent `final_tail` conversion must not be appended to such streams.
///
/// Runs the chunk smoother on a model-domain chunk and writes the fixed-length
/// output-domain audio into `out` (cleared first), returning the chosen SOLA
/// offset. `model_audio` / `model_pitchf` are the model's per-chunk output and
/// output pitchf; both they and `out` are caller-owned and reused across chunks
/// so the steady-state path allocates only when the model and output sample
/// rates differ (the resample below).
#[allow(clippy::too_many_arguments)]
pub fn prepare_model_output(
    model_audio: &[f32],
    model_pitchf: &[f32],
    model_sample_rate: u32,
    output_sample_rate: u32,
    output_chunk_samples: usize,
    joiner: &mut ChunkSmoother,
    final_tail: Option<&mut Vec<f32>>,
    out: &mut Vec<f32>,
) -> Result<usize> {
    let candidate = joiner.candidate_audio(model_audio);
    let chunk_samples = joiner.chunk_samples();
    let sola_offset = joiner.process(model_audio, model_pitchf);
    if let Some(final_tail) = final_tail {
        let tail_start = sola_offset
            .saturating_add(chunk_samples)
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

    out.clear();
    if model_sample_rate == output_sample_rate {
        out.extend_from_slice(joiner.output());
    } else {
        out.extend(resample_to_output_domain(
            joiner.output(),
            model_sample_rate,
            output_sample_rate,
        )?);
    }
    fit_to_len_in_place(out, output_chunk_samples);
    Ok(sola_offset)
}

#[cfg(test)]
mod tests {
    use super::{
        model_domain_chunk_smoother, prepare_model_output, psola_offset_with_period,
        stable_pitch_period_samples, vcclient_crossfade_gains, ChunkSmoother, ChunkSmootherConfig,
        PsolaChunkJoiner, SmoothingKind, SolaChunkJoiner,
    };

    #[test]
    fn sola_chunk_joiner_uses_detected_offset() {
        let mut joiner = SolaChunkJoiner::new(4, 2, 2, 0);

        let _ = joiner.process(&[0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(joiner.output(), vec![0.0, 0.0, 0.0, 0.0].as_slice());

        let second = joiner.process(&[0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0]);

        assert_eq!(second, 2);
        assert_eq!(joiner.output().len(), 4);
    }

    #[test]
    fn sola_chunk_joiner_primes_from_startup_tail() {
        let mut joiner = SolaChunkJoiner::new(4, 2, 2, 0);

        joiner.prime(&[0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0]);
        let sola_offset = joiner.process(&[0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0]);

        assert_eq!(sola_offset, 2);
        assert_eq!(joiner.output(), vec![4.0, 0.5, 6.0, 7.0].as_slice());
    }

    #[test]
    fn vcclient_crossfade_gains_keep_flat_edges() {
        assert_eq!(vcclient_crossfade_gains(0, 10), (1.0, 0.0));
        assert_eq!(vcclient_crossfade_gains(9, 10), (0.0, 1.0));
    }

    #[test]
    fn sola_chunk_joiner_right_aligns_short_outputs_without_crossfade() {
        let mut joiner = SolaChunkJoiner::new(5, 0, 1, 0);

        let _ = joiner.process(&[1.0, 2.0]);

        assert_eq!(joiner.output(), vec![0.0, 0.0, 0.0, 1.0, 2.0].as_slice());
    }

    #[test]
    fn sola_chunk_joiner_discards_unstable_tail_before_output_selection() {
        let mut joiner = SolaChunkJoiner::new(4, 0, 0, 2);

        let _ = joiner.process(&[1.0, 2.0, 3.0, 4.0, 100.0, 101.0]);

        assert_eq!(joiner.output(), vec![1.0, 2.0, 3.0, 4.0].as_slice());
    }

    #[test]
    fn startup_applies_tail_discard_once_when_priming_overlap() {
        let mut joiner = SolaChunkJoiner::new(4, 2, 0, 2);
        joiner.process(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 90.0, 91.0]);
        assert_eq!(joiner.sola_buffer, [4.0, 5.0]);
        joiner.process(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 90.0, 91.0]);
        assert_eq!(joiner.output()[0], 4.0);
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
        let sola_offset = joiner.process(&[0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0], &[]);

        assert_eq!(sola_offset, 2);
        assert_eq!(joiner.inner.output(), vec![4.0, 0.5, 6.0, 7.0].as_slice());
    }

    #[test]
    fn smoothed_model_output_reports_sola_offset_before_resampling() {
        let mut joiner = ChunkSmoother::Sola(SolaChunkJoiner::new(4, 2, 2, 0));
        joiner.prime(&[0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0]);
        let mut out = Vec::new();

        let sola_offset = prepare_model_output(
            &[0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0],
            &[],
            48_000,
            24_000,
            2,
            &mut joiner,
            None,
            &mut out,
        )
        .unwrap();

        assert_eq!(sola_offset, 2);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn smoothed_model_output_resamples_final_tail_after_sola() {
        let mut joiner = ChunkSmoother::Sola(SolaChunkJoiner::new(4, 2, 2, 0));
        joiner.prime(&[0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0]);
        let mut final_tail = Vec::new();
        let mut out = Vec::new();

        let _ = prepare_model_output(
            &[0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0],
            &[],
            48_000,
            24_000,
            2,
            &mut joiner,
            Some(&mut final_tail),
            &mut out,
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
        let mut out = Vec::new();

        prepare_model_output(
            &[0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0, 100.0, 101.0],
            &[],
            48_000,
            48_000,
            4,
            &mut joiner,
            Some(&mut final_tail),
            &mut out,
        )
        .unwrap();

        assert_eq!(out, vec![4.0, 0.5, 6.0, 7.0]);
        assert_eq!(final_tail, vec![8.0, 9.0]);
    }

    #[test]
    fn smoothed_model_output_keeps_final_tail_with_psola() {
        let mut joiner = ChunkSmoother::Psola(PsolaChunkJoiner::new(4, 2, 2, 0, 48_000));
        joiner.prime(&[0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0]);
        let mut final_tail = Vec::new();
        let mut out = Vec::new();

        prepare_model_output(
            &[0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0],
            &[100.0; 8],
            48_000,
            48_000,
            4,
            &mut joiner,
            Some(&mut final_tail),
            &mut out,
        )
        .unwrap();

        assert_eq!(out.len(), 4);
        assert_eq!(final_tail.len(), 2);
    }

    fn smoother_config(kind: SmoothingKind, chunk_ms: u32) -> ChunkSmootherConfig {
        ChunkSmootherConfig {
            kind,
            output_chunk_samples: (48_000 * chunk_ms / 1000) as usize,
            output_sample_rate: 48_000,
            model_sample_rate: 48_000,
            crossfade_ms: 85,
            sola_search_ms: 12,
            tail_discard_ms: 10,
        }
    }

    #[test]
    fn crossfade_window_is_clamped_below_short_chunks() {
        // 85 ms crossfade would exceed an 80 ms chunk; it must clamp to 3/4 of the
        // chunk so the overlap stays below the hop (the small-chunk artifact fix).
        for kind in [SmoothingKind::Sola, SmoothingKind::Psola] {
            let smoother = model_domain_chunk_smoother(smoother_config(kind, 80));
            let chunk = 48_000 * 80 / 1000;
            assert_eq!(smoother.crossfade_samples(), chunk * 3 / 4);
            assert!(smoother.crossfade_samples() < chunk);
        }
    }

    #[test]
    fn crossfade_window_unchanged_for_large_chunks() {
        // 500 ms chunk leaves the 85 ms crossfade far under the 3/4 cap: untouched.
        for kind in [SmoothingKind::Sola, SmoothingKind::Psola] {
            let smoother = model_domain_chunk_smoother(smoother_config(kind, 500));
            assert_eq!(smoother.crossfade_samples(), 48_000 * 85 / 1000);
        }
    }
}
