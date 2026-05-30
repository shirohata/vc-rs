use anyhow::{anyhow, Result};

use crate::dsp;

use super::feature::FeatureTensor;
use super::shape::{
    feature_len_for_samples, keep_tail_in_place, samples_between_rates, tail_or_left_pad,
    tensor_rt_convert_size_16k, Rounding, EMBEDDER_SAMPLE_RATE, RVC_SAMPLE_RATE,
};

pub(super) const VOLUME_DECAY: f32 = 0.97;

// This state is owned by the model worker, not the realtime audio callback.
// Keep resizing and resampling work here so callback code remains queue-only.

pub(super) struct RvcStreamInput {
    pub(super) convert_size: usize,
    pub(super) out_size: usize,
    pub(super) volume: f32,
}

impl RvcStreamState {
    pub(super) fn output_reference_audio(
        &self,
        input_sample_rate: u32,
        output_sample_rate: u32,
        output_samples: usize,
    ) -> Result<Vec<f32>> {
        if self.audio_buffer.is_empty()
            || output_samples == 0
            || input_sample_rate == 0
            || output_sample_rate == 0
        {
            return Ok(Vec::new());
        }

        let input_samples = samples_between_rates(
            output_samples,
            output_sample_rate,
            input_sample_rate,
            Rounding::Ceil,
        )
        .max(1);
        let start = self.audio_buffer.len().saturating_sub(input_samples);
        let input_tail = &self.audio_buffer[start..];
        let reference = if input_sample_rate == output_sample_rate {
            input_tail.to_vec()
        } else {
            dsp::resample_mono(
                input_tail,
                input_sample_rate as usize,
                output_sample_rate as usize,
            )?
        };

        Ok(tail_or_left_pad(reference, output_samples))
    }
}

pub(super) struct RvcStreamState {
    pub(super) audio_buffer: Vec<f32>,
    pub(super) audio_16k_buffer: Vec<f32>,
    pub(super) pitchf_buffer: Vec<f32>,
    pub(super) feature_buffer: Vec<f32>,
    pub(super) feature_channels: usize,
    pub(super) prev_vol: f32,
    pub(super) prev_silence: bool,
    pub(super) sample_rate: u32,
    pub(super) resampler_16k: Option<dsp::StreamingResampleMono>,
}

impl RvcStreamState {
    pub(super) fn new(feature_channels: i64) -> Self {
        Self {
            audio_buffer: Vec::new(),
            audio_16k_buffer: Vec::new(),
            pitchf_buffer: Vec::new(),
            feature_buffer: Vec::new(),
            feature_channels: feature_channels.max(0) as usize,
            prev_vol: 0.0,
            prev_silence: false,
            sample_rate: 0,
            resampler_16k: None,
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
            self.feature_buffer.clear();
            self.prev_vol = 0.0;
            self.prev_silence = false;
            self.sample_rate = sample_rate;
            self.resampler_16k = Some(dsp::StreamingResampleMono::new(
                sample_rate as usize,
                EMBEDDER_SAMPLE_RATE as usize,
            )?);
        }

        let new_audio_16k_samples = samples_between_rates(
            new_audio.len(),
            sample_rate,
            EMBEDDER_SAMPLE_RATE,
            Rounding::Floor,
        );
        let new_feature_len = feature_len_for_samples(new_audio_16k_samples, EMBEDDER_SAMPLE_RATE);
        self.audio_buffer.extend_from_slice(new_audio);
        let new_audio_16k = self
            .resampler_16k
            .as_mut()
            .ok_or_else(|| anyhow!("16kHz stream resampler is not initialized"))?
            .process(new_audio)?;
        self.audio_16k_buffer.extend_from_slice(&new_audio_16k);
        self.pitchf_buffer
            .extend(std::iter::repeat_n(0.0, new_feature_len));
        self.feature_buffer.extend(std::iter::repeat_n(
            0.0,
            new_feature_len * self.feature_channels,
        ));

        let extra_16k_samples = samples_between_rates(
            extra_convert_samples,
            RVC_SAMPLE_RATE,
            EMBEDDER_SAMPLE_RATE,
            Rounding::Floor,
        );
        let volume_excluded_16k_samples = samples_between_rates(
            volume_excluded_samples,
            RVC_SAMPLE_RATE,
            EMBEDDER_SAMPLE_RATE,
            Rounding::Floor,
        );
        let volume_excluded_input_samples = samples_between_rates(
            volume_excluded_16k_samples,
            EMBEDDER_SAMPLE_RATE,
            sample_rate,
            Rounding::Ceil,
        );
        let convert_size_16k = tensor_rt_convert_size_16k(
            new_audio.len(),
            sample_rate,
            crossfade_and_search_samples,
            extra_convert_samples,
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
            RVC_SAMPLE_RATE,
            Rounding::Floor,
        );
        let out_size = out_size.max(1);
        let feature_size = feature_len_for_samples(convert_size_16k, EMBEDDER_SAMPLE_RATE);

        if self.audio_buffer.len() < convert_size {
            let mut padded = vec![0.0; convert_size - self.audio_buffer.len()];
            padded.extend_from_slice(&self.audio_buffer);
            self.audio_buffer = padded;
        }
        if self.audio_16k_buffer.len() < convert_size_16k {
            let mut padded = vec![0.0; convert_size_16k - self.audio_16k_buffer.len()];
            padded.extend_from_slice(&self.audio_16k_buffer);
            self.audio_16k_buffer = padded;
        }
        if self.pitchf_buffer.len() < feature_size {
            let mut padded = vec![0.0; feature_size - self.pitchf_buffer.len()];
            padded.extend_from_slice(&self.pitchf_buffer);
            self.pitchf_buffer = padded;
        }
        let feature_sample_size = feature_size * self.feature_channels;
        if self.feature_buffer.len() < feature_sample_size {
            let mut padded = vec![0.0; feature_sample_size - self.feature_buffer.len()];
            padded.extend_from_slice(&self.feature_buffer);
            self.feature_buffer = padded;
        }

        keep_tail_in_place(&mut self.audio_buffer, convert_size);
        keep_tail_in_place(&mut self.audio_16k_buffer, convert_size_16k);
        keep_tail_in_place(&mut self.pitchf_buffer, feature_size);
        keep_tail_in_place(&mut self.feature_buffer, feature_sample_size);

        let crop_len = new_audio.len() + volume_excluded_input_samples;
        let crop_end = volume_excluded_input_samples;
        let volume = if crop_len > crop_end && self.audio_buffer.len() >= crop_len {
            let end = self.audio_buffer.len().saturating_sub(crop_end);
            let start = self.audio_buffer.len().saturating_sub(crop_len);
            dsp::rms(&self.audio_buffer[start..end])
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
        })
    }

    pub(super) fn update_pitchf_from_rmvpe_frames(&mut self, f0: &[f32]) {
        let n = self.pitchf_buffer.len().min(f0.len());
        if n == 0 {
            return;
        }
        // Match the WebUI pitch cache assignment:
        // `pitchf[-f0_len:] = f0[:pitchf_len]`. RMVPE usually emits one
        // center-padded frame beyond the model pitch buffer, and taking the
        // tail of f0 instead shifts the pitch contour later in time.
        let dst_start = self.pitchf_buffer.len() - n;
        self.pitchf_buffer[dst_start..].copy_from_slice(&f0[..n]);
    }

    pub(super) fn update_feature_buffer(&mut self, features: &FeatureTensor, feature_len: usize) {
        if self.feature_channels == 0 || features.data.is_empty() {
            return;
        }
        let frames = feature_len.min(features.data.len() / self.feature_channels);
        let samples = frames * self.feature_channels;
        if samples == 0 {
            return;
        }
        self.feature_buffer = features.data[features.data.len() - samples..].to_vec();
    }
}
