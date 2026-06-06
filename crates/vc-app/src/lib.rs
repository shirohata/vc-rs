//! Shared application runtime used by the CLI and standalone GUI.
//!
//! Frontends communicate with [`EngineController`]. Audio callbacks only touch
//! preallocated lock-free sample rings and atomics; they must never be coupled
//! to GUI rendering, model loading, or other blocking work.

pub mod audio;
mod realtime;

pub use realtime::{
    AudioBackend, DeviceList, EngineController, EngineState, EngineStatusSnapshot, LiveParams,
    RealtimeConfig, Smoother, TelemetrySnapshot,
};
