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
    fn new(channels: usize) -> Self {
        Self {
            channels,
            generator: GaussianNoise::new(RVC_RND_SEED),
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
        self.generator = GaussianNoise::new(RVC_RND_SEED);
    }
}

/// Aggregated CPU-side time state for the RVC generator. Step 1 carries only the
/// optional `rnd` rolling buffer; Step 2 adds NSF phase / `nsf_noise` here so all
/// time-varying state lives behind one reset/roll surface.
pub(super) struct RvcTimeState {
    /// `Some` only when the model exposes an `rnd` input; `None` leaves models
    /// that sample their own noise completely unchanged.
    rnd: Option<RndRollingBuffer>,
}

impl RvcTimeState {
    /// `rnd_channels` is the static `inter_channels` of the model's `rnd` input,
    /// or `None` when the model has no such input.
    pub(super) fn new(rnd_channels: Option<usize>) -> Self {
        Self {
            rnd: rnd_channels.map(RndRollingBuffer::new),
        }
    }

    /// Roll the per-frame noise to track the same feature window as `pitchf`.
    /// No-op when the model has no `rnd` input.
    pub(super) fn roll_rnd(&mut self, new_frames: usize, total_frames: usize) {
        if let Some(rnd) = self.rnd.as_mut() {
            rnd.roll(new_frames, total_frames);
        }
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

    /// Reset every timeline (clears buffers, re-seeds generators). Called when the
    /// audio timeline breaks: stream restart, sample-rate / chunk change, model
    /// reload, or RVC enable/disable.
    pub(super) fn reset(&mut self) {
        if let Some(rnd) = self.rnd.as_mut() {
            rnd.reset();
        }
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
        let mut state = RvcTimeState::new(Some(channels));

        // Warm up so the buffer is full (no zero pad) before comparing.
        for _ in 0..6 {
            state.roll_rnd(advance, total);
        }
        let mut w1 = Vec::new();
        state.roll_rnd(advance, total);
        assert!(state.rnd_window_into(total, total, &mut w1));
        let mut w2 = Vec::new();
        state.roll_rnd(advance, total);
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
        let mut state = RvcTimeState::new(Some(channels));
        for _ in 0..6 {
            state.roll_rnd(advance, total);
        }
        let mut w1 = Vec::new();
        state.roll_rnd(advance, total);
        state.rnd_window_into(total, total, &mut w1);
        let mut w2 = Vec::new();
        state.roll_rnd(advance, total);
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
            let mut state = RvcTimeState::new(Some(channels));
            let mut windows = Vec::new();
            for _ in 0..5 {
                state.roll_rnd(advance, total);
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
        let mut state = RvcTimeState::new(Some(channels));
        let first = {
            state.roll_rnd(advance, total);
            let mut w = Vec::new();
            state.rnd_window_into(total, total, &mut w);
            w
        };
        // Advance further, then reset and replay the first chunk.
        for _ in 0..4 {
            state.roll_rnd(advance, total);
        }
        state.reset();
        state.roll_rnd(advance, total);
        let mut after = Vec::new();
        state.rnd_window_into(total, total, &mut after);
        assert_eq!(
            first, after,
            "reset did not restore the initial noise stream"
        );
    }

    #[test]
    fn no_rnd_input_is_a_noop() {
        let mut state = RvcTimeState::new(None);
        state.roll_rnd(2, 8);
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
        let mut rnd = RndRollingBuffer::new(channels);
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
}
