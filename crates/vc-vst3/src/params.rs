//! Plugin parameters and persisted state.
//!
//! - The live, automatable knobs (`pitch`/`speaker`/gains) are `#[id]` params
//!   owned by the DAW (the host persists/automates them per project).
//! - `settings` (model paths, provider, chunking, thresholds) and the editor
//!   window size are non-parameter state persisted via `#[persist]`, so they
//!   are saved/restored with the project too.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use nice_plug::prelude::*;
use nice_plug_egui::EguiState;

use crate::config::PluginConfig;

/// Preserve the existing serialized `settings` object while observing host
/// restores, including restoring identical bytes during a pending UI download.
/// The revision is session-only and checked under the existing settings lock.
#[derive(Default)]
pub struct SettingsState {
    value: RwLock<PluginConfig>,
    restore_revision: AtomicU64,
}

impl std::ops::Deref for SettingsState {
    type Target = RwLock<PluginConfig>;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl SettingsState {
    pub(crate) fn restore_revision(&self) -> u64 {
        self.restore_revision.load(Ordering::SeqCst)
    }

    pub(crate) fn seed_if_pristine(&self, discover: impl FnOnce() -> PluginConfig) {
        let pristine = |value: &PluginConfig| {
            self.restore_revision() == 0 && *value == PluginConfig::default()
        };
        if !pristine(&self.value.read().unwrap()) {
            return;
        }
        let seed = discover();
        if !seed.has_models() {
            return;
        }
        let mut value = self.value.write().unwrap();
        // Config discovery does file I/O without holding the settings lock.
        // A project restore (even empty/default state) or an editor change
        // which happened meanwhile always wins over the fresh-instance seed.
        if pristine(&value) {
            *value = seed;
        }
    }
}

impl nice_plug::params::persist::PersistentField<'_, PluginConfig> for SettingsState {
    fn set(&self, new_value: PluginConfig) {
        let mut value = self.value.write().unwrap();
        *value = new_value;
        self.restore_revision.fetch_add(1, Ordering::SeqCst);
    }
    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&PluginConfig) -> R,
    {
        f(&self.value.read().unwrap())
    }
}

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
    #[id = "ngate"]
    pub noise_gate: BoolParam,
    /// Gate threshold in dB; converted to a linear amplitude before it reaches
    /// the core gate (`util::db_to_gain`).
    #[id = "ngthr"]
    pub noise_gate_threshold_db: FloatParam,

    /// Model paths and conversion settings. Set via the GUI / config seed and
    /// persisted with the project.
    #[persist = "settings"]
    pub settings: SettingsState,

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
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(2)),
            output_gain_db: FloatParam::new(
                "Output Gain",
                0.0,
                FloatRange::Linear {
                    min: -36.0,
                    max: 36.0,
                },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(2)),
            noise_gate: BoolParam::new("Noise Gate", false),
            noise_gate_threshold_db: FloatParam::new(
                "Gate Threshold",
                -40.0,
                FloatRange::Linear {
                    min: -80.0,
                    max: 0.0,
                },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(2)),
            settings: SettingsState::default(),
            editor_state: EguiState::from_size(480, 520),
        }
    }
}
