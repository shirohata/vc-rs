use std::time::Duration;

use anyhow::Result;

use crate::dsp;

pub struct ModelOutput {
    pub audio: Vec<f32>,
    pub pitchf: Vec<f32>,
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
    fn process(&mut self, audio: &[f32], sample_rate: u32) -> Result<ModelOutput>;
}

pub struct PassthroughModel;

impl VoiceModel for PassthroughModel {
    fn process(&mut self, audio: &[f32], sample_rate: u32) -> Result<ModelOutput> {
        Ok(ModelOutput {
            audio: audio.to_vec(),
            pitchf: Vec::new(),
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
