//! Reusable RVC voice conversion core.
//!
//! This crate holds the audio-I/O-agnostic pieces of the voice changer: the
//! RVC inference pipeline (`model_rvc`), DSP helpers (`dsp`), and chunk
//! smoothing (`sola`). Both the CLI (`vc-cli`) and the VST3 plugin (`vc-vst3`)
//! depend on it and drive `model_rvc::RvcPipeline::process` from their own I/O
//! layer.

mod provider;

pub mod dsp;
pub mod model_rvc;
pub mod sola;

pub use provider::Provider;
