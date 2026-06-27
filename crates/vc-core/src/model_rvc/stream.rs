use anyhow::{anyhow, Result};

use crate::dsp;

use super::shape::{
    feature_len_for_samples, keep_tail_in_place, samples_between_rates, tensor_rt_convert_size_16k,
    Rounding, EMBEDDER_SAMPLE_RATE, RMVPE_FRAME_SAMPLES_16K,
};
use super::time_state::RvcTimeState;

pub(super) const VOLUME_DECAY: f32 = 0.97;

// This state is owned by the model worker, not the realtime audio callback.
// Keep resizing and resampling work here so callback code remains queue-only.

pub(super) struct RvcStreamInput {
    pub(super) convert_size: usize,
    pub(super) out_size: usize,
    pub(super) volume: f32,
    // RMS of the new 16 kHz increment (post-input-denoiser), i.e. the same signal
    // ContentVec/F0 consume. Drives input_rms/silence on the 16 kHz timeline for
    // every denoiser mode — see the guardrail in `process`.
    pub(super) input_rms: f32,
}

impl RvcStreamState {
    /// Tail of the (post-input-denoiser) 16 kHz rolling signal, resampled to the
    /// RVC output rate, for the RMS-mix reference. Uses `audio_16k_buffer` — the
    /// same signal ContentVec/F0 consume — so the level reference matches the
    /// model's input for every denoiser mode. Callers pass `input_sample_rate =
    /// EMBEDDER_SAMPLE_RATE`.
    pub(super) fn output_reference_audio<'a>(
        &'a self,
        input_sample_rate: u32,
        output_sample_rate: u32,
        output_samples: usize,
        scratch: &'a mut Vec<f32>,
    ) -> Result<&'a [f32]> {
        scratch.clear();
        if self.audio_16k_buffer.is_empty()
            || output_samples == 0
            || input_sample_rate == 0
            || output_sample_rate == 0
        {
            return Ok(scratch.as_slice());
        }

        let input_samples = samples_between_rates(
            output_samples,
            output_sample_rate,
            input_sample_rate,
            Rounding::Ceil,
        )
        .max(1);
        let start = self.audio_16k_buffer.len().saturating_sub(input_samples);
        let input_tail = &self.audio_16k_buffer[start..];
        if input_sample_rate == output_sample_rate && input_tail.len() >= output_samples {
            return Ok(&input_tail[input_tail.len() - output_samples..]);
        }

        if input_sample_rate == output_sample_rate {
            scratch.extend_from_slice(input_tail);
        } else {
            let reference = dsp::resample_mono(
                input_tail,
                input_sample_rate as usize,
                output_sample_rate as usize,
            )?;
            scratch.extend_from_slice(&reference);
        }
        keep_tail_in_place(scratch, output_samples);
        left_pad_to_len_in_place(scratch, output_samples);

        Ok(scratch.as_slice())
    }
}

fn left_pad_to_len_in_place(values: &mut Vec<f32>, len: usize) {
    if values.len() >= len {
        return;
    }
    let old_len = values.len();
    let pad = len - old_len;
    values.resize(len, 0.0);
    values.copy_within(0..old_len, pad);
    values[..pad].fill(0.0);
}

pub(super) struct RvcStreamState {
    pub(super) audio_buffer: Vec<f32>,
    pub(super) audio_16k_buffer: Vec<f32>,
    pub(super) pitchf_buffer: Vec<f32>,
    pub(super) prev_vol: f32,
    pub(super) prev_silence: bool,
    pub(super) sample_rate: u32,
    /// The RVC model's native output rate (from metadata `samplingRate`, default
    /// `RVC_SAMPLE_RATE`). Fixed per model — distinct from `sample_rate`, which is
    /// the device/input rate. Sizes `out_size` and the RVC-domain conversions.
    pub(super) rvc_sample_rate: u32,
    pub(super) resampler_16k: Option<dsp::StreamingResampleMono>,
    // Backend-neutral CPU time state (latent `rnd` noise; Step 2 adds NSF
    // phase / nsf_noise). Rolled in lockstep with `pitchf_buffer` so per-frame
    // noise tracks the same absolute feature window. Inert when the model has no
    // `rnd` input.
    pub(super) time_state: RvcTimeState,
    // GTCRN input denoiser applied to each new 16 kHz increment before it is
    // appended to the windowed `audio_16k_buffer` (the RVC-path seam). `Some`
    // only when the pipeline was built with `load_with_gtcrn`. At 16 kHz the
    // adapter's resamplers are bypass, so only its frame FIFO + fixed delay run.
    #[cfg(feature = "gtcrn")]
    pub(super) gtcrn: Option<crate::denoise::GtcrnDenoiser>,
}

impl RvcStreamState {
    /// `rnd_channels` is the model's `rnd` input channel count (`inter_channels`),
    /// or `None` when the model samples its own noise — see [`RvcTimeState::new`].
    pub(super) fn new(rvc_sample_rate: u32, rnd_channels: Option<usize>) -> Self {
        Self {
            audio_buffer: Vec::new(),
            audio_16k_buffer: Vec::new(),
            pitchf_buffer: Vec::new(),
            prev_vol: 0.0,
            prev_silence: false,
            sample_rate: 0,
            rvc_sample_rate,
            resampler_16k: None,
            time_state: RvcTimeState::new(rnd_channels),
            #[cfg(feature = "gtcrn")]
            gtcrn: None,
        }
    }

    pub(super) fn generate_input(
        &mut self,
        new_audio: &[f32],
        sample_rate: u32,
        crossfade_and_search_samples: usize,
        volume_excluded_samples: usize,
        extra_convert_samples: usize,
    ) -> Result<RvcStreamInput> {
        if self.sample_rate != sample_rate {
            self.audio_buffer.clear();
            self.audio_16k_buffer.clear();
            self.pitchf_buffer.clear();
            self.prev_vol = 0.0;
            self.prev_silence = false;
            self.sample_rate = sample_rate;
            self.resampler_16k = Some(dsp::StreamingResampleMono::new(
                sample_rate as usize,
                EMBEDDER_SAMPLE_RATE as usize,
            )?);
            // A device sample-rate change restarts the stream; the audio timeline
            // breaks, so drop the per-frame noise/phase history too.
            self.time_state.reset();
            // A device sample-rate change restarts the stream; reset GTCRN's
            // fixed-delay/cache state so it does not emit pre-restart audio.
            #[cfg(feature = "gtcrn")]
            if let Some(gtcrn) = self.gtcrn.as_mut() {
                gtcrn.reset()?;
            }
        }

        let new_audio_16k_samples = samples_between_rates(
            new_audio.len(),
            sample_rate,
            EMBEDDER_SAMPLE_RATE,
            Rounding::Floor,
        );
        let new_feature_len = feature_len_for_samples(new_audio_16k_samples, EMBEDDER_SAMPLE_RATE);
        self.audio_buffer.extend_from_slice(new_audio);
        let new_16k_start = self.audio_16k_buffer.len();
        self.resampler_16k
            .as_mut()
            .ok_or_else(|| anyhow!("16kHz stream resampler is not initialized"))?
            .process_into(new_audio, &mut self.audio_16k_buffer)?;
        // GTCRN denoises exactly the new 16 kHz increment, in place, BEFORE the
        // windowing below. Guardrail: process only the increment, never the
        // re-windowed `audio_16k_buffer`, so its length — and thus
        // `new_audio_16k_samples` and the ContentVec/F0 window length — stays
        // unchanged. The fixed delay is internal (adds latency, never shifts the
        // sample grid). Each sample is denoised exactly once, at append time.
        #[cfg(feature = "gtcrn")]
        if let Some(gtcrn) = self.gtcrn.as_mut() {
            gtcrn.process_in_place(&mut self.audio_16k_buffer[new_16k_start..])?;
        }
        // input_rms/silence run on the 16 kHz post-input-denoiser signal (the same
        // one ContentVec/F0 see) for every mode. Measure the raw new increment
        // here, before the windowing below left-pads the front — so this never
        // touches the zero pad.
        let input_rms = dsp::rms(&self.audio_16k_buffer[new_16k_start..]);
        self.pitchf_buffer
            .extend(std::iter::repeat_n(0.0, new_feature_len));

        let extra_16k_samples = samples_between_rates(
            extra_convert_samples,
            self.rvc_sample_rate,
            EMBEDDER_SAMPLE_RATE,
            Rounding::Floor,
        );
        let volume_excluded_16k_samples = samples_between_rates(
            volume_excluded_samples,
            self.rvc_sample_rate,
            EMBEDDER_SAMPLE_RATE,
            Rounding::Floor,
        );
        let convert_size_16k = tensor_rt_convert_size_16k(
            new_audio.len(),
            sample_rate,
            crossfade_and_search_samples,
            extra_convert_samples,
            self.rvc_sample_rate,
        );
        let convert_size = samples_between_rates(
            convert_size_16k,
            EMBEDDER_SAMPLE_RATE,
            sample_rate,
            Rounding::Ceil,
        );
        let out_size = samples_between_rates(
            convert_size_16k.saturating_sub(extra_16k_samples),
            EMBEDDER_SAMPLE_RATE,
            self.rvc_sample_rate,
            Rounding::Floor,
        );
        let out_size = out_size.max(1);
        let feature_size = feature_len_for_samples(convert_size_16k, EMBEDDER_SAMPLE_RATE);

        // Left-pad with zeros in place (reusing the buffers) when a chunk arrives
        // before enough history has accumulated — startup and just after a
        // passthrough->RVC switch resets the state.
        left_pad_to_len_in_place(&mut self.audio_buffer, convert_size);
        left_pad_to_len_in_place(&mut self.audio_16k_buffer, convert_size_16k);
        left_pad_to_len_in_place(&mut self.pitchf_buffer, feature_size);

        keep_tail_in_place(&mut self.audio_buffer, convert_size);
        keep_tail_in_place(&mut self.audio_16k_buffer, convert_size_16k);
        keep_tail_in_place(&mut self.pitchf_buffer, feature_size);

        // Roll the latent-noise window in lockstep with `pitchf_buffer`: same new
        // frame count, same total window length, same 10 ms grid. This keeps a
        // given absolute frame's `rnd` value stable across overlapping chunks.
        // Inert when the model has no `rnd` input.
        self.time_state.roll_rnd(new_feature_len, feature_size);

        // Volume envelope memory on the 16 kHz timeline (same signal as
        // ContentVec/F0), the new-increment region minus the excluded tail. The
        // crop is at the back of the buffer, so in steady state it avoids the
        // front zero pad; at stream start it may dip into the pad exactly as the
        // former device-rate crop did (parity, not a regression).
        let crop_len_16k = new_audio_16k_samples + volume_excluded_16k_samples;
        let crop_end_16k = volume_excluded_16k_samples;
        let volume = if crop_len_16k > crop_end_16k && self.audio_16k_buffer.len() >= crop_len_16k {
            let end = self.audio_16k_buffer.len().saturating_sub(crop_end_16k);
            let start = self.audio_16k_buffer.len().saturating_sub(crop_len_16k);
            dsp::rms(&self.audio_16k_buffer[start..end])
        } else {
            0.0
        };
        // Keep a short memory of previous chunk loudness so envelope-based
        // output shaping does not collapse instantly between adjacent chunks.
        let volume = volume.max(self.prev_vol * VOLUME_DECAY);
        self.prev_vol = volume;

        Ok(RvcStreamInput {
            convert_size,
            out_size,
            volume,
            input_rms,
        })
    }

    pub(super) fn update_pitchf_from_rmvpe_window(
        &mut self,
        f0: &[f32],
        window_start_samples_16k: usize,
    ) {
        let dst_start = window_start_samples_16k / RMVPE_FRAME_SAMPLES_16K;
        if dst_start >= self.pitchf_buffer.len() {
            return;
        }
        let n = (self.pitchf_buffer.len() - dst_start).min(f0.len());
        if n == 0 {
            return;
        }
        // RMVPE emits one center-padded frame past `(samples / hop)` for the
        // upstream bucket sizes. Copy from the front and let any trailing frame
        // fall off, matching the full-window WebUI assignment above while
        // preserving the absolute frame offset of a tail-only RMVPE window.
        self.pitchf_buffer[dst_start..dst_start + n].copy_from_slice(&f0[..n]);
    }
}
