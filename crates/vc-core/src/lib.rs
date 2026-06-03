//! Reusable RVC voice conversion core.
//!
//! This crate holds the audio-I/O-agnostic pieces of the voice changer: the
//! RVC inference pipeline (`model_rvc`), DSP helpers (`dsp`), and chunk
//! smoothing (`sola`). Both the CLI (`vc-cli`) and the VST3 plugin (`vc-vst3`)
//! depend on it and drive `model_rvc::RvcPipeline::process` from their own I/O
//! layer.

// The TensorRT-only build (no `ort` feature) intentionally leaves a number of
// ORT-supporting fields, constants, helpers, and run-mode parameters inert; they
// exist only for the ONNX Runtime backend. The full (ORT) build still enforces
// these lints, so real dead code there is still caught.
#![cfg_attr(not(feature = "ort"), allow(dead_code, unused_variables))]

mod provider;
#[cfg(all(windows, feature = "windowsml"))]
pub mod windows_ml;

pub mod dsp;
pub mod model_rvc;
pub mod sola;

pub use provider::Provider;
