//! Shared application runtime used by the CLI and standalone GUI.
//!
//! Frontends communicate with [`EngineController`]. Audio callbacks only touch
//! preallocated lock-free sample rings and atomics; they must never be coupled
//! to GUI rendering, model loading, or other blocking work.

pub mod audio;
mod clock_drift;
mod realtime;

pub use realtime::{
    write_wav_mono, AudioHost, DenoiserMode, DeviceList, DeviceTestConfig, DeviceTestSnapshot,
    EngineController, EngineState, EngineStatusSnapshot, RealtimeConfig, Smoother,
    TelemetrySnapshot, TestOutput,
};
pub use vc_core::model_rvc::{F0Config, LiveParams, NoiseGateShaping, OutputDynamicsConfig};
