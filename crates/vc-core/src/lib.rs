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

// Input-denoiser family (shared fixed-delay adapter + per-model frame
// processors). Gated on the union of denoiser features; DeepFilterNet3 adds
// itself to this `cfg` when it lands.
#[cfg(any(feature = "rnnoise", feature = "gtcrn"))]
pub mod denoise;
pub mod dsp;
pub mod gpu;
pub mod model_rvc;
pub mod sola;
pub mod validation;

pub use provider::{default_provider, selectable_providers, Provider};
