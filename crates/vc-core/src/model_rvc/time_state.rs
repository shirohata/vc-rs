//! CPU-resident, backend-neutral time state for the RVC generator.
//!
//! Some RVC exports take time-varying noise/phase as graph inputs instead of
//! sampling them internally. The realtime path feeds the model an *overlapping*
//! rolling window (past context + SOLA search region), so a given absolute frame
//! recurs across consecutive chunks. If those recurring frames were fed fresh
//! noise every chunk, the overlapping region would differ between chunks and the
//! SOLA join / crossfade would degrade.
//!
//! This module keeps the per-frame noise (and, in the streaming export, the NSF
//! phase / per-sample NSF noise — added in Step 2) as a single CPU-side rolling
//! state, shared by every backend (ORT CPU/CUDA/DirectML and native TensorRT).
//! The backends only *bind* the buffers produced here; they never generate noise
//! themselves, so there is exactly one noise timeline regardless of provider.
//!
//! Owned by the model worker, never touched from the audio callback — the
//! per-chunk arithmetic here is intentionally off the realtime path.
//!
//! The state is deliberately held in plain `Vec<f32>` so a future change can swap
//! it for a GPU-resident buffer (VRAM) behind the same `roll`/`window_into` API
//! without touching callers.

use super::noise::GaussianNoise;

// Fixed seed for the RVC latent-noise generator: reproducible across runs while
// advancing per generated frame, so a stream is byte-reproducible from a fixed
// seed + a fixed chunk sequence. Spells "RVCRND" in ASCII for grep-ability.
// Centralized here so the ORT and native TensorRT paths share one seed instead
// of each defining their own.
pub(super) const RVC_RND_SEED: u64 = 0x0000_5256_4352_4e44;

// Distinct seed for the streaming NSF source noise (`nsf_noise`), so it draws an
// independent stream from `rnd` (the reference samples each from a separate
// `torch.randn`). Spells "NSFRND" in ASCII.
const NSF_NOISE_SEED: u64 = 0x0000_4e53_4652_4e44;

/// Rolling buffer of latent noise (`rnd`, the VITS reparameterization noise `z`)
/// keyed to absolute feature-frame position.
///
/// Stored frame-major: frame `f` occupies `buffer[f*channels .. (f+1)*channels]`.
/// Rolling is therefore a contiguous append-at-tail / drop-at-front, so a frame's
/// noise value is fixed once generated and stays attached to its absolute frame
/// as the window slides — overlapping frames between chunks read identical noise.
///
/// The window handed to the model is selected with the *same* center-crop + tail
/// alignment the pipeline applies to `pitchf` (see [`window_into`]), so `rnd`
/// frame `i` lines up with `pitchf`/`feats` frame `i` by construction.
pub(super) struct RndRollingBuffer {
    channels: usize,
    seed: u64,
    generator: GaussianNoise,
    /// Frame-major rolling noise, length `frames * channels`.
    buffer: Vec<f32>,
    /// Reused scratch for the center-cropped frames (frame-major).
    crop_scratch: Vec<f32>,
    /// Reused scratch for the tail-aligned window (frame-major), transposed into
    /// the caller's channel-major output by `window_into`.
    frame_scratch: Vec<f32>,
}

impl RndRollingBuffer {
    fn with_seed(channels: usize, seed: u64) -> Self {
        Self {
            channels,
            seed,
            generator: GaussianNoise::new(seed),
            buffer: Vec::new(),
            crop_scratch: Vec::new(),
            frame_scratch: Vec::new(),
        }
    }

    /// Advance the rolling window: append fresh `N(0, 1)` noise for `new_frames`
    /// newly-arrived frames, then size the buffer to exactly `total_frames`.
    ///
    /// Startup / post-reset history is zero-padded at the front — identical to how
    /// `audio_16k_buffer`/`pitchf_buffer` left-pad before enough context exists.
    /// The audio there is silence and its converted output is discarded (past
    /// context / SOLA search region), so the zero pad is inaudible; in steady
    /// state every absolute frame carries stable per-frame noise.
    fn roll(&mut self, new_frames: usize, total_frames: usize) {
        let new_len = new_frames.saturating_mul(self.channels);
        let start = self.buffer.len();
        self.buffer.resize(start + new_len, 0.0);
        self.generator.fill(&mut self.buffer[start..]);

        let total_len = total_frames.saturating_mul(self.channels);
        // Left-pad whole frames with zeros when history is short (startup), then
        // keep the tail — mirroring `left_pad_to_len_in_place` + `keep_tail`.
        if self.buffer.len() < total_len {
            let pad = total_len - self.buffer.len();
            let old_len = self.buffer.len();
            self.buffer.resize(total_len, 0.0);
            self.buffer.copy_within(0..old_len, pad);
            self.buffer[..pad].fill(0.0);
        } else if self.buffer.len() > total_len {
            self.buffer.drain(..self.buffer.len() - total_len);
        }
    }

    /// Produce the `[1, channels, feature_len]` (channel-major) rnd tensor for
    /// this chunk into `out`, applying the same two-step frame selection the
    /// pipeline uses for `pitchf`: center-crop the rolling buffer to
    /// `feature_len_before_trim`, then tail-align to `feature_len`. Returns the
    /// channel-major length written (`channels * feature_len`).
    fn window_into(
        &mut self,
        feature_len_before_trim: usize,
        feature_len: usize,
        out: &mut Vec<f32>,
    ) -> usize {
        let buffer_frames = self.buffer.len() / self.channels.max(1);
        center_crop_frames(
            &self.buffer,
            self.channels,
            buffer_frames,
            feature_len_before_trim,
            &mut self.crop_scratch,
        );
        align_tail_frames(
            &self.crop_scratch,
            self.channels,
            feature_len_before_trim,
            feature_len,
            &mut self.frame_scratch,
        );
        // Transpose frame-major [feature_len, channels] -> channel-major
        // [channels, feature_len] expected by the `rnd` input.
        out.clear();
        out.resize(feature_len * self.channels, 0.0);
        for f in 0..feature_len {
            for c in 0..self.channels {
                out[c * feature_len + f] = self.frame_scratch[f * self.channels + c];
            }
        }
        out.len()
    }

    fn reset(&mut self) {
        self.buffer.clear();
        // Re-seed so the stream is reproducible from its start after a reset.
        self.generator = GaussianNoise::new(self.seed);
    }
}

/// Time base for a streaming export's NSF phase / `nsf_noise`, from the model's
/// `rvc.frame_hop` / `rvc.sample_rate` metadata.
#[derive(Debug, Clone, Copy)]
pub(super) struct StreamParams {
    /// Output samples per feature frame (`audio_len == feature_frames * frame_hop`).
    pub(super) frame_hop: usize,
    /// Output sample rate — the phase advance time base (`f0 / sample_rate`).
    pub(super) sample_rate: u32,
}

/// NSF fundamental-phase carry for a streaming export.
///
/// The exporter's contract (`rvc.streaming.phase_contract`) is
/// `absolute_window_start`: `phase_in` is the normalized phase at the window's
/// first generated sample, `phase_out` the phase after its last. The model adds
/// `phase_in` to the within-window cumulative phase, which starts at 0.
///
/// vc-rs feeds *overlapping* windows, so the exporter's "carry phase_out only to
/// an immediately adjacent next window" rule does not apply — successive windows
/// are not adjacent. Instead we carry the phase at the window start and advance it
/// by exactly the frames the window scrolls past each chunk. Because the chunk
/// size is fixed, the window advances by a constant `advance_frames` and those are
/// the first `advance_frames` frames of the current window's `pitchf` (which is
/// tail-aligned), so the advance reproduces the model's per-frame phase step
/// `f0 / sample_rate * frame_hop` exactly, on the CPU.
struct PhaseState {
    /// Normalized phase (fraction of a cycle, `[0, 1)`) at the current window's
    /// first generated sample — fed straight to `phase_in`.
    window_start_phase: f64,
}

impl PhaseState {
    fn new() -> Self {
        Self {
            window_start_phase: 0.0,
        }
    }

    /// Advance the carried window-start phase past `advance_frames` frames using
    /// their `pitchf` (the same `nsff0` the model consumes), reproducing the
    /// model's per-frame step `f0 / sample_rate * frame_hop`.
    fn advance(&mut self, pitchf: &[f32], advance_frames: usize, params: StreamParams) {
        let n = advance_frames.min(pitchf.len());
        let mut acc = self.window_start_phase;
        let hop = params.frame_hop as f64;
        let sr = params.sample_rate.max(1) as f64;
        for &f0 in &pitchf[..n] {
            acc += (f0 as f64 / sr) * hop;
        }
        // Wrap into [0, 1); the sine source is periodic so only the fraction
        // matters, and keeping it bounded avoids f64 precision loss over a long
        // stream.
        self.window_start_phase = acc.rem_euclid(1.0);
    }
}

/// Aggregated CPU-side time state for the RVC generator: the optional `rnd`
/// rolling buffer (every `rnd`-input model) plus, for streaming exports, the NSF
/// source-noise rolling buffer and the NSF phase carry. All time-varying state
/// lives behind one `roll`/`reset` surface so every backend binds identical
/// buffers and there is a single place to reset when the timeline breaks.
pub(super) struct RvcTimeState {
    /// `Some` only when the model exposes an `rnd` input; `None` leaves models
    /// that sample their own noise completely unchanged.
    rnd: Option<RndRollingBuffer>,
    /// Streaming NSF source noise (`nsf_noise`, 1 channel on the output-sample
    /// grid). `Some` only for streaming exports.
    nsf: Option<RndRollingBuffer>,
    /// NSF phase carry; `Some` only for streaming exports.
    phase: Option<PhaseState>,
    /// Streaming time base (frame hop / sample rate); `Some` only for streaming.
    stream: Option<StreamParams>,
    /// Frames the window advanced this chunk (constant in steady state), captured
    /// in `roll` and consumed by `advance_phase` after inference.
    last_advance_frames: usize,
    /// Absolute feature-frame / output-sample position of the window tail. Kept so
    /// downstream state stays coherent and resets are observable; reset to 0.
    abs_frame: u64,
    abs_sample: u64,
}

impl RvcTimeState {
    /// `rnd_channels` is the static `inter_channels` of the model's `rnd` input
    /// (or `None`). `stream` is `Some` only for streaming exports and enables the
    /// NSF noise buffer + phase carry.
    pub(super) fn new(rnd_channels: Option<usize>, stream: Option<StreamParams>) -> Self {
        Self {
            rnd: rnd_channels.map(|channels| RndRollingBuffer::with_seed(channels, RVC_RND_SEED)),
            nsf: stream.map(|_| RndRollingBuffer::with_seed(1, NSF_NOISE_SEED)),
            phase: stream.map(|_| PhaseState::new()),
            stream,
            last_advance_frames: 0,
            abs_frame: 0,
            abs_sample: 0,
        }
    }

    /// Roll every per-chunk timeline forward by `new_frames` (on the feature grid),
    /// keeping a `total_frames` window — in lockstep with `pitchf_buffer`. The NSF
    /// noise rolls on the output-sample grid (`* frame_hop`). No-op for the parts
    /// the model does not use.
    pub(super) fn roll(&mut self, new_frames: usize, total_frames: usize) {
        if let Some(rnd) = self.rnd.as_mut() {
            rnd.roll(new_frames, total_frames);
        }
        if let (Some(nsf), Some(params)) = (self.nsf.as_mut(), self.stream) {
            nsf.roll(
                new_frames * params.frame_hop,
                total_frames * params.frame_hop,
            );
            self.abs_sample = self
                .abs_sample
                .saturating_add((new_frames * params.frame_hop) as u64);
        }
        self.last_advance_frames = new_frames;
        self.abs_frame = self.abs_frame.saturating_add(new_frames as u64);
    }

    /// Fill `out` with this chunk's `[1, channels, feature_len]` rnd tensor
    /// (channel-major) and return `true`. Returns `false` (leaving `out`
    /// untouched) when the model has no `rnd` input.
    pub(super) fn rnd_window_into(
        &mut self,
        feature_len_before_trim: usize,
        feature_len: usize,
        out: &mut Vec<f32>,
    ) -> bool {
        match self.rnd.as_mut() {
            Some(rnd) => {
                rnd.window_into(feature_len_before_trim, feature_len, out);
                true
            }
            None => false,
        }
    }

    /// Fill `out` with this chunk's `[1, audio_len, 1]` NSF noise (audio_len ==
    /// `feature_len * frame_hop`), selected on the output-sample grid with the
    /// same center-crop + tail alignment as `rnd`/`pitchf` so it tracks the same
    /// absolute frames. Returns `false` for non-streaming exports.
    pub(super) fn nsf_noise_window_into(
        &mut self,
        feature_len_before_trim: usize,
        feature_len: usize,
        out: &mut Vec<f32>,
    ) -> bool {
        match (self.nsf.as_mut(), self.stream) {
            (Some(nsf), Some(params)) => {
                nsf.window_into(
                    feature_len_before_trim * params.frame_hop,
                    feature_len * params.frame_hop,
                    out,
                );
                true
            }
            _ => false,
        }
    }

    /// The `phase_in` value for this chunk (the carried window-start phase), or
    /// `None` for non-streaming exports.
    pub(super) fn phase_in(&self) -> Option<f32> {
        self.phase.as_ref().map(|p| p.window_start_phase as f32)
    }

    /// Absolute (feature-frame, output-sample) position of the rolling window
    /// tail. Advances every chunk and resets to `(0, 0)`; exposed for diagnostics
    /// and drift detection on the streaming timeline.
    pub(super) fn absolute_position(&self) -> (u64, u64) {
        (self.abs_frame, self.abs_sample)
    }

    /// Set the next chunk's window-start phase directly from the model's
    /// per-sample phase output (`streaming_nsf_phase`), per the exporter contract
    /// "select the next phase_in from streaming_nsf_phase at the next input window
    /// start". The next window starts `advance_frames * frame_hop` samples into the
    /// current output, so that element is the next `phase_in`.
    ///
    /// Returns `false` (leaving the phase unchanged) for non-streaming exports or
    /// when the output is too short to contain that sample — the caller then falls
    /// back to [`advance_phase`]. Preferring the model output keeps the carry exact
    /// even if the exporter's internal phase math changes.
    pub(super) fn set_phase_from_output(&mut self, phase_samples: &[f32]) -> bool {
        let Some(params) = self.stream else {
            return false;
        };
        let advance_samples = self.last_advance_frames.saturating_mul(params.frame_hop);
        let Some(phase) = self.phase.as_mut() else {
            return false;
        };
        let Some(&value) = phase_samples.get(advance_samples) else {
            return false;
        };
        phase.window_start_phase = (value as f64).rem_euclid(1.0);
        true
    }

    /// Advance the carried NSF phase past this chunk's window advance, using the
    /// final `pitchf` (the model's `nsff0`). The CPU fallback for exports that do
    /// not emit a usable per-sample phase output. Call once after each inference.
    /// No-op for non-streaming exports.
    pub(super) fn advance_phase(&mut self, pitchf: &[f32]) {
        if let (Some(phase), Some(params)) = (self.phase.as_mut(), self.stream) {
            phase.advance(pitchf, self.last_advance_frames, params);
        }
    }

    /// Reset every timeline (clears buffers, re-seeds generators, zeroes the phase
    /// and absolute position). Called when the audio timeline breaks: stream
    /// restart, sample-rate / chunk change, model reload, or RVC enable/disable.
    pub(super) fn reset(&mut self) {
        if let Some(rnd) = self.rnd.as_mut() {
            rnd.reset();
        }
        if let Some(nsf) = self.nsf.as_mut() {
            nsf.reset();
        }
        if let Some(phase) = self.phase.as_mut() {
            *phase = PhaseState::new();
        }
        self.last_advance_frames = 0;
        self.abs_frame = 0;
        self.abs_sample = 0;
    }
}

/// Center-crop frame-major `input` (with `channels` values per frame) from
/// `in_frames` to `target_frames`, mirroring `center_crop_pitchf_to_features_into`
/// on the frame axis: drop `excess/2` frames from the front when longer, else
/// front-pad. Result is frame-major in `out`.
fn center_crop_frames(
    input: &[f32],
    channels: usize,
    in_frames: usize,
    target_frames: usize,
    out: &mut Vec<f32>,
) {
    if in_frames > target_frames {
        let front_drop = (in_frames - target_frames) / 2;
        out.clear();
        out.extend_from_slice(
            &input[front_drop * channels..(front_drop + target_frames) * channels],
        );
    } else {
        align_tail_frames(input, channels, in_frames, target_frames, out);
    }
}

/// Tail-align frame-major `input` from `in_frames` to `target_frames`, mirroring
/// `align_pitchf_to_features_into` on the frame axis: keep the last
/// `target_frames` frames when longer, else front-pad with zero frames.
fn align_tail_frames(
    input: &[f32],
    channels: usize,
    in_frames: usize,
    target_frames: usize,
    out: &mut Vec<f32>,
) {
    out.clear();
    if in_frames >= target_frames {
        out.extend_from_slice(&input[(in_frames - target_frames) * channels..]);
    } else {
        out.resize((target_frames - in_frames) * channels, 0.0);
        out.extend_from_slice(input);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A frame's channel-0 value identifies the frame across chunks. In the
    // channel-major `[channels, feature_len]` layout, channel 0 occupies
    // `window[0..feature_len]`, so frame `f`'s channel-0 value is `window[f]`.
    fn frame_c0(window: &[f32], frame: usize) -> f32 {
        window[frame]
    }

    #[test]
    fn overlapping_frames_keep_their_noise_across_chunks() {
        // Steady-state window of 8 frames, advancing 2 frames per chunk. The last
        // 6 frames of chunk 1 must equal the first 6 frames of chunk 2.
        let channels = 4;
        let total = 8;
        let advance = 2;
        let mut state = RvcTimeState::new(Some(channels), None);

        // Warm up so the buffer is full (no zero pad) before comparing.
        for _ in 0..6 {
            state.roll(advance, total);
        }
        let mut w1 = Vec::new();
        state.roll(advance, total);
        assert!(state.rnd_window_into(total, total, &mut w1));
        let mut w2 = Vec::new();
        state.roll(advance, total);
        assert!(state.rnd_window_into(total, total, &mut w2));

        // w1 frames [advance..total] are the same absolute frames as w2
        // [0..total-advance].
        for f in 0..(total - advance) {
            assert_eq!(
                frame_c0(&w1, f + advance),
                frame_c0(&w2, f),
                "overlapping absolute frame {f} changed between chunks"
            );
        }
    }

    #[test]
    fn only_new_frames_get_fresh_noise() {
        let channels = 2;
        let total = 6;
        let advance = 2;
        let mut state = RvcTimeState::new(Some(channels), None);
        for _ in 0..6 {
            state.roll(advance, total);
        }
        let mut w1 = Vec::new();
        state.roll(advance, total);
        state.rnd_window_into(total, total, &mut w1);
        let mut w2 = Vec::new();
        state.roll(advance, total);
        state.rnd_window_into(total, total, &mut w2);
        // The newest `advance` frames of w2 are fresh: they should differ from the
        // frames that occupied those positions in w1.
        let mut any_new_differs = false;
        for f in (total - advance)..total {
            if frame_c0(&w2, f) != frame_c0(&w1, f) {
                any_new_differs = true;
            }
        }
        assert!(
            any_new_differs,
            "new frames were not refreshed with new noise"
        );
    }

    #[test]
    fn same_seed_and_chunk_sequence_reproduce_identical_noise() {
        let channels = 3;
        let total = 8;
        let advance = 2;
        let run = || {
            let mut state = RvcTimeState::new(Some(channels), None);
            let mut windows = Vec::new();
            for _ in 0..5 {
                state.roll(advance, total);
                let mut w = Vec::new();
                state.rnd_window_into(total, total, &mut w);
                windows.push(w);
            }
            windows
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn reset_returns_to_initial_state() {
        let channels = 3;
        let total = 8;
        let advance = 2;
        let mut state = RvcTimeState::new(Some(channels), None);
        let first = {
            state.roll(advance, total);
            let mut w = Vec::new();
            state.rnd_window_into(total, total, &mut w);
            w
        };
        // Advance further, then reset and replay the first chunk.
        for _ in 0..4 {
            state.roll(advance, total);
        }
        state.reset();
        state.roll(advance, total);
        let mut after = Vec::new();
        state.rnd_window_into(total, total, &mut after);
        assert_eq!(
            first, after,
            "reset did not restore the initial noise stream"
        );
    }

    #[test]
    fn no_rnd_input_is_a_noop() {
        let mut state = RvcTimeState::new(None, None);
        state.roll(2, 8);
        let mut out = vec![1.0, 2.0, 3.0];
        assert!(!state.rnd_window_into(8, 8, &mut out));
        // `out` is left untouched when there is no rnd input.
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn window_selection_matches_pitch_center_crop_then_align() {
        // Verify the frame selection equals running pitch's helpers on a per-frame
        // witness (channel 0). Build a buffer where frame f's channel-0 value = f.
        let channels = 2;
        let buffer_frames = 9;
        let before_trim = 7;
        let feature_len = 5;
        let mut rnd = RndRollingBuffer::with_seed(channels, RVC_RND_SEED);
        rnd.buffer = vec![0.0; buffer_frames * channels];
        for f in 0..buffer_frames {
            rnd.buffer[f * channels] = f as f32; // channel 0 witness
            rnd.buffer[f * channels + 1] = 100.0 + f as f32; // channel 1 witness
        }
        let mut out = Vec::new();
        rnd.window_into(before_trim, feature_len, &mut out);

        // Expected frame indices: center-crop 9->7 drops front (9-7)/2=1 =>
        // frames [1..8); tail-align 7->5 keeps last 5 => frames [3..8).
        let expected_frames = [3usize, 4, 5, 6, 7];
        for (i, &fr) in expected_frames.iter().enumerate() {
            assert_eq!(out[i], fr as f32, "channel 0 frame mismatch at {i}");
            assert_eq!(
                out[feature_len + i],
                100.0 + fr as f32,
                "channel 1 frame mismatch at {i}"
            );
        }
    }

    fn streaming_state(frame_hop: usize, sample_rate: u32) -> RvcTimeState {
        RvcTimeState::new(
            Some(4),
            Some(StreamParams {
                frame_hop,
                sample_rate,
            }),
        )
    }

    #[test]
    fn nsf_noise_window_is_audio_len_and_overlaps_across_chunks() {
        // feature window of 8 frames, hop 4 -> audio_len 32; advance 2 frames ->
        // 8 samples per chunk. Overlap region must match across chunks.
        let frame_hop = 4;
        let total = 8;
        let advance = 2;
        let mut state = streaming_state(frame_hop, 16_000);
        for _ in 0..6 {
            state.roll(advance, total);
        }
        let mut n1 = Vec::new();
        state.roll(advance, total);
        assert!(state.nsf_noise_window_into(total, total, &mut n1));
        assert_eq!(n1.len(), total * frame_hop, "nsf_noise must be audio_len");
        let mut n2 = Vec::new();
        state.roll(advance, total);
        state.nsf_noise_window_into(total, total, &mut n2);
        // n1 samples [advance*hop..] equal n2 [..len-advance*hop] (overlap).
        let shift = advance * frame_hop;
        for i in 0..(total * frame_hop - shift) {
            assert_eq!(n1[i + shift], n2[i], "nsf overlap sample {i} changed");
        }
    }

    #[test]
    fn rnd_and_nsf_noise_are_independent_streams() {
        // Same shape on channel 0 vs the 1-channel nsf buffer must differ (distinct
        // seeds), otherwise the two graph inputs would receive correlated noise.
        let mut state = streaming_state(1, 16_000);
        for _ in 0..4 {
            state.roll(8, 8);
        }
        state.roll(8, 8);
        let mut rnd = Vec::new();
        state.rnd_window_into(8, 8, &mut rnd);
        let mut nsf = Vec::new();
        state.nsf_noise_window_into(8, 8, &mut nsf);
        // Channel 0 of rnd (first 8 values) vs nsf (8 values) should not be equal.
        assert_ne!(rnd[..8], nsf[..8]);
    }

    #[test]
    fn phase_carry_advances_by_frame_step_and_wraps() {
        // Constant f0 so the step is predictable: per frame, phase advances by
        // f0/sr*hop. With f0=100, sr=16000, hop=160 -> 1.0 per frame -> wraps to 0.
        let frame_hop = 160;
        let sample_rate = 16_000;
        let total = 8;
        let advance = 2;
        let mut state = streaming_state(frame_hop, sample_rate);
        state.roll(advance, total);
        assert_eq!(state.phase_in(), Some(0.0), "stream starts at phase 0");
        let pitchf = vec![100.0f32; total];
        state.advance_phase(&pitchf); // advance 2 frames * 1.0 = 2.0 -> wraps to 0
        let p = state.phase_in().unwrap();
        assert!(p.abs() < 1e-5, "phase {p} should wrap to ~0");

        // Half-step f0 so it does not wrap: f0=50 -> step 0.5/frame, 2 frames -> 1.0
        // -> wraps to 0; use f0=25 -> 0.25/frame, 2 frames -> 0.5.
        let mut state = streaming_state(frame_hop, sample_rate);
        state.roll(advance, total);
        let pitchf = vec![25.0f32; total];
        state.advance_phase(&pitchf);
        let p = state.phase_in().unwrap();
        assert!((p - 0.5).abs() < 1e-5, "phase {p} should be 0.5");
    }

    #[test]
    fn phase_from_output_picks_next_window_start_sample() {
        let frame_hop = 4;
        let total = 8;
        let advance = 2;
        let mut state = streaming_state(frame_hop, 48_000);
        state.roll(advance, total); // last_advance_frames = 2 -> advance_samples = 8
                                    // Per-sample phase output, audio_len = total * frame_hop = 32.
        let phase: Vec<f32> = (0..total * frame_hop).map(|i| i as f32 / 100.0).collect();
        assert!(state.set_phase_from_output(&phase));
        // The next window starts at sample advance*frame_hop = 8 -> phase[8] = 0.08.
        assert!((state.phase_in().unwrap() - 0.08).abs() < 1e-5);
        // A too-short output leaves the phase unchanged and reports no update.
        assert!(!state.set_phase_from_output(&phase[..4]));
    }

    #[test]
    fn streaming_reset_clears_noise_phase_and_position() {
        let frame_hop = 4;
        let total = 8;
        let advance = 2;
        let mut state = streaming_state(frame_hop, 16_000);
        let first_nsf = {
            state.roll(advance, total);
            let mut n = Vec::new();
            state.nsf_noise_window_into(total, total, &mut n);
            n
        };
        for _ in 0..4 {
            state.roll(advance, total);
            state.advance_phase(&vec![120.0f32; total]);
        }
        state.reset();
        assert_eq!(state.phase_in(), Some(0.0), "phase not reset");
        state.roll(advance, total);
        let mut after = Vec::new();
        state.nsf_noise_window_into(total, total, &mut after);
        assert_eq!(first_nsf, after, "nsf noise stream not reset");
    }

    #[test]
    fn non_streaming_state_has_no_nsf_or_phase() {
        let mut state = RvcTimeState::new(Some(4), None);
        state.roll(2, 8);
        let mut out = vec![1.0f32];
        assert!(!state.nsf_noise_window_into(8, 8, &mut out));
        assert_eq!(state.phase_in(), None);
    }
}
