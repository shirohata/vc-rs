//! Plugin parameters and persisted state.
//!
//! - The live, automatable knobs (`pitch`/`speaker`/gains) are `#[id]` params
//!   owned by the DAW (the host persists/automates them per project).
//! - `settings` (model paths, provider, chunking, thresholds) and the editor
//!   window size are non-parameter state persisted via `#[persist]`, so they
//!   are saved/restored with the project too.

use std::sync::{Arc, RwLock};

use nice_plug::prelude::*;
use nice_plug_egui::EguiState;

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

    /// Model paths and conversion settings. Set via the GUI / config seed and
    /// persisted with the project.
    #[persist = "settings"]
    pub settings: RwLock<PluginConfig>,

    /// Editor window size, persisted so the host remembers it.
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
}

impl Default for VcRvcParams {
    fn default() -> Self {
        Self {
            pitch_shift: FloatParam::new(
                "Pitch",
                0.0,
                FloatRange::Linear {
                    min: -24.0,
                    max: 24.0,
                },
            )
            .with_unit(" st")
            .with_step_size(0.5),
            speaker_id: IntParam::new("Speaker", 0, IntRange::Linear { min: 0, max: 255 }),
            input_gain_db: FloatParam::new(
                "Input Gain",
                0.0,
                FloatRange::Linear {
                    min: -36.0,
                    max: 36.0,
                },
            )
            .with_unit(" dB"),
            output_gain_db: FloatParam::new(
                "Output Gain",
                0.0,
                FloatRange::Linear {
                    min: -36.0,
                    max: 36.0,
                },
            )
            .with_unit(" dB"),
            settings: RwLock::new(PluginConfig::default()),
            editor_state: EguiState::from_size(480, 520),
        }
    }
}
