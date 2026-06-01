//! Host-automatable parameters. These override the config-file values at
//! runtime; the worker thread reads them once per chunk and pushes them into
//! the `RvcPipeline` via its setters.

use nih_plug::prelude::*;

use crate::config::PluginConfig;

#[derive(Params)]
pub struct VcRvcParams {
    #[id = "pitch"]
    pub pitch_shift: FloatParam,
    #[id = "speaker"]
    pub speaker_id: IntParam,
    #[id = "ingain"]
    pub input_gain_db: FloatParam,
    #[id = "outgain"]
    pub output_gain_db: FloatParam,
}

impl VcRvcParams {
    /// Build parameters with initial values taken from the config file so the
    /// host's "default" matches the headless configuration.
    pub fn from_config(config: &PluginConfig) -> Self {
        Self {
            pitch_shift: FloatParam::new(
                "Pitch",
                config.pitch_shift,
                FloatRange::Linear {
                    min: -24.0,
                    max: 24.0,
                },
            )
            .with_unit(" st")
            .with_step_size(0.5),
            speaker_id: IntParam::new(
                "Speaker",
                config.speaker_id as i32,
                IntRange::Linear { min: 0, max: 255 },
            ),
            input_gain_db: FloatParam::new(
                "Input Gain",
                util::gain_to_db(config.input_gain.max(f32::EPSILON)),
                FloatRange::Linear {
                    min: -36.0,
                    max: 36.0,
                },
            )
            .with_unit(" dB"),
            output_gain_db: FloatParam::new(
                "Output Gain",
                util::gain_to_db(config.output_gain.max(f32::EPSILON)),
                FloatRange::Linear {
                    min: -36.0,
                    max: 36.0,
                },
            )
            .with_unit(" dB"),
        }
    }
}
