use std::time::Duration;

use anyhow::Result;

use crate::dsp;

/// Exact content delay in seconds. Keep delays in their native sample domains
/// while combining stages; rounding each stage at the output rate can crop an
/// extra source sample, especially at 44.1 kHz. Only the final owner rounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContentDelay {
    numerator: u128,
    denominator: u128,
}

impl ContentDelay {
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };

    /// `sample_rate` is a validated, nonzero stream or model sample rate.
    pub fn from_samples(samples: usize, sample_rate: u32) -> Self {
        assert!(
            sample_rate > 0,
            "content delay sample rate must be positive"
        );
        Self::reduced(samples as u128, sample_rate as u128)
    }

    pub fn output_samples(self, output_sample_rate: u32) -> usize {
        let samples = (self.numerator * output_sample_rate as u128).div_ceil(self.denominator);
        usize::try_from(samples).expect("content delay exceeds addressable audio length")
    }

    fn reduced(numerator: u128, denominator: u128) -> Self {
        let divisor = greatest_common_divisor(numerator, denominator);
        Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        }
    }
}

impl std::ops::Add for ContentDelay {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // Cancel the shared denominator first: delays commonly share 16/32/48k
        // sample domains, and repeated aggregation must not multiply those
        // denominators unnecessarily.
        let divisor = greatest_common_divisor(self.denominator, other.denominator);
        let left_scale = other.denominator / divisor;
        let right_scale = self.denominator / divisor;
        Self::reduced(
            self.numerator * left_scale + other.numerator * right_scale,
            self.denominator * left_scale,
        )
    }
}

fn greatest_common_divisor(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

/// Per-chunk metadata returned by [`VoiceModel::process`]. The converted audio
/// and output `pitchf` are written into caller-owned buffers passed as
/// out-parameters (so the worker reuses them across chunks instead of
/// allocating); this struct carries only the scalar stats describing the chunk.
pub struct ModelOutput {
    pub sample_rate: u32,
    pub inference_time: Duration,
    pub embedder_time: Duration,
    pub pitch_time: Duration,
    pub rvc_time: Duration,
    pub input_rms: f32,
    pub voiced_ratio: f32,
    pub raw_output_samples: usize,
    pub output_rms: f32,
    pub applied_output_gain: f32,
    pub feature_frames: usize,
    pub pitch_frames: usize,
    pub silent: bool,
    pub convert_size: usize,
    pub out_size: usize,
    pub model_input_samples: usize,
    pub volume: f32,
}

pub trait VoiceModel: Send {
    /// Exact delay before generation. Override this native-domain contract;
    /// converter joining adds its delay before the one final output rounding.
    fn input_content_delay(&self) -> ContentDelay {
        ContentDelay::ZERO
    }

    /// Fixed content delay introduced before generation, expressed at the
    /// requested output rate. This excludes inference scheduling, device
    /// buffering, and chunk joining. Finite conversion removes it only after
    /// draining the same streaming pipeline; do not include context-only input
    /// padding or denoiser delay already removed by an offline preprocessing pass.
    fn input_content_delay_samples(&self, output_sample_rate: u32) -> usize {
        self.input_content_delay()
            .output_samples(output_sample_rate)
    }

    /// Convert one chunk. The converted samples are written into `out_audio` and
    /// the output `pitchf` into `out_pitchf` (both cleared first), so callers
    /// can reuse the buffers across chunks. Returns scalar chunk metadata.
    fn process(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        out_audio: &mut Vec<f32>,
        out_pitchf: &mut Vec<f32>,
    ) -> Result<ModelOutput>;
}

pub struct PassthroughModel;

impl VoiceModel for PassthroughModel {
    fn process(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        out_audio: &mut Vec<f32>,
        out_pitchf: &mut Vec<f32>,
    ) -> Result<ModelOutput> {
        out_audio.clear();
        out_audio.extend_from_slice(audio);
        out_pitchf.clear();
        Ok(ModelOutput {
            sample_rate,
            inference_time: Duration::ZERO,
            embedder_time: Duration::ZERO,
            pitch_time: Duration::ZERO,
            rvc_time: Duration::ZERO,
            input_rms: dsp::rms(audio),
            voiced_ratio: 0.0,
            raw_output_samples: audio.len(),
            output_rms: dsp::rms(audio),
            applied_output_gain: 1.0,
            feature_frames: 0,
            pitch_frames: 0,
            silent: false,
            convert_size: audio.len(),
            out_size: audio.len(),
            model_input_samples: audio.len(),
            volume: dsp::rms(audio),
        })
    }
}
