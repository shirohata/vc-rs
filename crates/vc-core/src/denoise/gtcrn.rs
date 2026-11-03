//! GTCRN streaming frame processor: STFT ↔ ONNX streaming graph ↔ iSTFT.
//!
//! One `process_frame` call maps 256 new 16 kHz samples to 256 denoised samples:
//! analyze one STFT frame, run the streaming graph (feeding its `conv`/`tra`/
//! `inter` caches forward), synthesize the enhanced spectrum back. The inference
//! backend sits behind the [`InferSession`] seam so a future native-TensorRT
//! GTCRN can add a new seam impl (mirroring `model_rvc/native_tensorrt.rs`)
//! without disturbing the STFT/cache orchestration here.
//!
//! Empirical streaming-graph contract (Phase 0, from the upstream export — all
//! shapes fixed, B=1, C=16, dtype f32):
//!   mix  [1,257,1,2]            enh             [1,257,1,2]
//!   conv_cache  [2,1,16,16,33]  conv_cache_out  [2,1,16,16,33]
//!   tra_cache   [2,3,1,1,16]    tra_cache_out   [2,3,1,1,16]
//!   inter_cache [2,1,33,16]     inter_cache_out [2,1,33,16]
//! The cache frequency axis is 33 (sub-bands), not the 257 spectrum bins.

use std::path::{Path, PathBuf};

#[cfg(feature = "ort")]
use anyhow::{anyhow, Context};
use anyhow::{bail, Result};
#[cfg(feature = "ort")]
use ort::ep;
#[cfg(feature = "ort")]
use ort::session::Session;
#[cfg(feature = "ort")]
use ort::value::TensorRef;
use realfft::num_complex::Complex;

use super::adapter::{FixedDelayAdapter, FrameDenoiser};
use super::stft::{StftStreamer, GTCRN_BINS, GTCRN_HOP, GTCRN_RECON_DELAY_FRAMES};
use crate::model_rvc::GpuPriority;

// Streaming-graph tensor names and fixed shapes (Phase 0).
const MIX_NAME: &str = "mix";
const CONV_IN: &str = "conv_cache";
const TRA_IN: &str = "tra_cache";
const INTER_IN: &str = "inter_cache";
const ENH_OUT: &str = "enh";
const CONV_OUT: &str = "conv_cache_out";
const TRA_OUT: &str = "tra_cache_out";
const INTER_OUT: &str = "inter_cache_out";

const MIX_SHAPE: [usize; 4] = [1, GTCRN_BINS, 1, 2];
const CONV_SHAPE: [usize; 5] = [2, 1, 16, 16, 33];
const TRA_SHAPE: [usize; 5] = [2, 3, 1, 1, 16];
const INTER_SHAPE: [usize; 4] = [2, 1, 33, 16];

const MIX_LEN: usize = GTCRN_BINS * 2; // real/imag interleaved
const CONV_LEN: usize = 2 * 16 * 16 * 33;
const TRA_LEN: usize = 2 * 3 * 16;
const INTER_LEN: usize = 2 * 33 * 16;

/// Canonical model file name for our self-generated export; the upstream
/// prebuilt export (`gtcrn.onnx`) is accepted as a fallback.
const MODEL_FILE: &str = "gtcrn_stream.onnx";
const MODEL_FILE_FALLBACK: &str = "gtcrn.onnx";

/// The streaming caches fed forward between frames.
struct GtcrnCaches {
    conv: Vec<f32>,
    tra: Vec<f32>,
    inter: Vec<f32>,
}

impl GtcrnCaches {
    fn zeros() -> Self {
        Self {
            conv: vec![0.0; CONV_LEN],
            tra: vec![0.0; TRA_LEN],
            inter: vec![0.0; INTER_LEN],
        }
    }

    fn reset(&mut self) {
        self.conv.fill(0.0);
        self.tra.fill(0.0);
        self.inter.fill(0.0);
    }
}

/// Backend-agnostic "run one streaming step" seam: enhance `mix` (257×2
/// interleaved) into `enh`, advancing the caches in place. Implementations do
/// only the tensor I/O; the STFT and cache bookkeeping stay in
/// [`GtcrnFrameProcessor`].
///
/// `Send` because the whole pipeline (and thus the boxed session) is moved onto
/// the inference worker thread.
trait InferSession: Send {
    fn run(&mut self, mix: &[f32], caches: &mut GtcrnCaches, enh: &mut [f32]) -> Result<()>;
}

/// ONNX Runtime implementation of [`InferSession`] (CPU EP — GTCRN is tiny, and
/// CPU keeps it off the GPU that RVC inference uses).
#[cfg(feature = "ort")]
struct OrtInferSession {
    session: Session,
}

#[cfg(feature = "ort")]
impl OrtInferSession {
    fn new(path: &Path) -> Result<Self> {
        // Windows ML loads ORT dynamically from the Windows App SDK Runtime; the
        // bootstrap must run before any session is built. It is idempotent.
        #[cfg(all(windows, feature = "windowsml"))]
        crate::windows_ml::ensure_initialized()?;

        // The builder's intermediate errors carry the (non-`Send`)
        // `SessionBuilder` for recovery, so they cannot flow through `anyhow`'s
        // `?`; stringify them like the RVC session loader does.
        let session = Session::builder()
            .map_err(|err| anyhow!("failed to create GTCRN session builder: {err}"))?
            .with_execution_providers([ep::CPU::default().build()])
            .map_err(|err| anyhow!("failed to register CPU EP for GTCRN: {err}"))?
            .commit_from_file(path)
            .with_context(|| format!("failed to load GTCRN model {}", path.display()))?;
        Ok(Self { session })
    }
}

#[cfg(feature = "ort")]
impl InferSession for OrtInferSession {
    fn run(&mut self, mix: &[f32], caches: &mut GtcrnCaches, enh: &mut [f32]) -> Result<()> {
        // Zero-copy views into the caller's buffers; released when `outputs` is
        // produced, after which the caches are overwritten with the updates.
        let outputs = self.session.run(ort::inputs![
            MIX_NAME => TensorRef::from_array_view((MIX_SHAPE, mix))?,
            CONV_IN => TensorRef::from_array_view((CONV_SHAPE, caches.conv.as_slice()))?,
            TRA_IN => TensorRef::from_array_view((TRA_SHAPE, caches.tra.as_slice()))?,
            INTER_IN => TensorRef::from_array_view((INTER_SHAPE, caches.inter.as_slice()))?,
        ])?;

        enh.copy_from_slice(outputs[ENH_OUT].try_extract_tensor::<f32>()?.1);
        caches
            .conv
            .copy_from_slice(outputs[CONV_OUT].try_extract_tensor::<f32>()?.1);
        caches
            .tra
            .copy_from_slice(outputs[TRA_OUT].try_extract_tensor::<f32>()?.1);
        caches
            .inter
            .copy_from_slice(outputs[INTER_OUT].try_extract_tensor::<f32>()?.1);
        Ok(())
    }
}

struct NativeTensorRtInferSession {
    engine: crate::model_rvc::native_tensorrt::NativeGtcrnEngine,
}

impl NativeTensorRtInferSession {
    fn new(path: &Path, gpu_priority: GpuPriority, gpu_device_id: u32) -> Result<Self> {
        Ok(Self {
            engine: crate::model_rvc::native_tensorrt::NativeGtcrnEngine::load(
                path,
                gpu_priority,
                gpu_device_id,
            )?,
        })
    }
}

impl InferSession for NativeTensorRtInferSession {
    fn run(&mut self, mix: &[f32], caches: &mut GtcrnCaches, enh: &mut [f32]) -> Result<()> {
        self.engine.infer(
            mix,
            &mut caches.conv,
            &mut caches.tra,
            &mut caches.inter,
            enh,
        )
    }
}

impl GtcrnBackend {
    pub fn for_provider(
        provider: crate::Provider,
        gpu_priority: GpuPriority,
        gpu_device_id: u32,
    ) -> Self {
        if provider.is_tensorrt() {
            Self::NativeTensorRt {
                gpu_priority,
                gpu_device_id,
            }
        } else {
            Self::OrtCpu
        }
    }
}

/// GTCRN [`FrameDenoiser`]: STFT + streaming graph + iSTFT, all at 16 kHz with a
/// 256-sample hop.
pub(crate) struct GtcrnFrameProcessor {
    stft: StftStreamer,
    session: Box<dyn InferSession>,
    caches: GtcrnCaches,
    mix: Vec<f32>,
    enh: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
}

impl GtcrnFrameProcessor {
    fn with_session(session: Box<dyn InferSession>) -> Self {
        Self {
            stft: StftStreamer::new(),
            session,
            caches: GtcrnCaches::zeros(),
            mix: vec![0.0; MIX_LEN],
            enh: vec![0.0; MIX_LEN],
            spectrum: vec![Complex::new(0.0, 0.0); GTCRN_BINS],
        }
    }

    fn from_onnx(path: &Path, backend: GtcrnBackend) -> Result<Self> {
        match backend {
            GtcrnBackend::OrtCpu => {
                #[cfg(feature = "ort")]
                {
                    Ok(Self::with_session(Box::new(OrtInferSession::new(path)?)))
                }
                #[cfg(not(feature = "ort"))]
                {
                    let _ = path;
                    bail!("GTCRN ORT backend is unavailable in this build")
                }
            }
            GtcrnBackend::NativeTensorRt {
                gpu_priority,
                gpu_device_id,
            } => Ok(Self::with_session(Box::new(
                NativeTensorRtInferSession::new(path, gpu_priority, gpu_device_id)?,
            ))),
        }
    }
}

impl FrameDenoiser for GtcrnFrameProcessor {
    fn sample_rate(&self) -> usize {
        16_000
    }

    fn frame_size(&self) -> usize {
        GTCRN_HOP
    }

    fn output_delay_frames(&self) -> usize {
        // The model is causal (T=1 per step, cache feedback, no lookahead); the
        // only delay is the STFT/overlap-add reconstruction.
        GTCRN_RECON_DELAY_FRAMES
    }

    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let spectrum = self.stft.analyze(input)?;
        for (i, bin) in spectrum.iter().enumerate() {
            self.mix[2 * i] = bin.re;
            self.mix[2 * i + 1] = bin.im;
        }
        self.session
            .run(&self.mix, &mut self.caches, &mut self.enh)?;
        for (i, dst) in self.spectrum.iter_mut().enumerate() {
            *dst = Complex::new(self.enh[2 * i], self.enh[2 * i + 1]);
        }
        self.stft.synthesize(&self.spectrum, output)
    }

    fn reset(&mut self) -> Result<()> {
        self.stft.reset();
        self.caches.reset();
        Ok(())
    }
}

/// Inference backend for the GTCRN streaming graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GtcrnBackend {
    /// Run the tiny graph through ONNX Runtime's CPU execution provider.
    OrtCpu,
    /// Run the graph through the repository's native TensorRT engine path.
    NativeTensorRt {
        gpu_priority: GpuPriority,
        gpu_device_id: u32,
    },
}

/// Minimal GTCRN model config (no DFN-style attenuation knobs — keep it minimal).
pub struct GtcrnConfig<'a> {
    /// Directory holding `gtcrn_stream.onnx` (or the upstream `gtcrn.onnx`).
    pub model_dir: &'a Path,
    pub backend: GtcrnBackend,
}

/// Fixed-delay GTCRN input denoiser. Thin wrapper over the shared
/// [`FixedDelayAdapter`], mirroring `RnnoiseDenoiser`.
pub struct GtcrnDenoiser {
    inner: FixedDelayAdapter<GtcrnFrameProcessor>,
}

impl GtcrnDenoiser {
    pub fn new(config: GtcrnConfig<'_>, sample_rate: u32) -> Result<Self> {
        let model = resolve_model_file(config.model_dir)?;
        let processor = GtcrnFrameProcessor::from_onnx(&model, config.backend)?;
        Ok(Self {
            inner: FixedDelayAdapter::new(processor, sample_rate)?,
        })
    }

    pub fn latency_samples(&self) -> usize {
        self.inner.latency_samples()
    }

    /// Process arbitrary device-rate input while preserving the per-call length.
    pub fn process_in_place(&mut self, samples: &mut [f32]) -> Result<()> {
        self.inner.process_in_place(samples)
    }

    /// Restore the post-construction (warmed-up) state, including model caches.
    pub fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }
}

/// Locate the GTCRN model file inside `dir`, preferring the canonical name.
fn resolve_model_file(dir: &Path) -> Result<PathBuf> {
    for name in [MODEL_FILE, MODEL_FILE_FALLBACK] {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!(
        "no GTCRN model found in {}: expected {MODEL_FILE} (or {MODEL_FILE_FALLBACK})",
        dir.display()
    );
}

pub(crate) fn model_file_for_cache_probe(dir: &Path) -> Result<PathBuf> {
    resolve_model_file(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    // These exercise a real ORT session; gated on a local model dir so CI (which
    // has no model) skips them. The ort-free `stft`/adapter layers are the CI
    // coverage. Run with: VC_RS_GTCRN_MODEL=<dir with gtcrn[_stream].onnx>.
    //
    // Run these under a CPU-ORT feature set (e.g. `--features cpu,gtcrn`). They
    // hang under a bare-`cargo test` **windowsml** build because
    // `ensure_initialized` can't bring up the dynamic Windows App SDK runtime in
    // an unpackaged test process with no bootstrap DLL on the loader path — the
    // api-24 path is verified through the packaged GUI/CLI, not a unit test.
    fn model_dir() -> Option<PathBuf> {
        std::env::var_os("VC_RS_GTCRN_MODEL").map(PathBuf::from)
    }

    fn tone(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| 0.2 * (i as f32 * 0.05).sin() + 0.05 * (i as f32 * 0.31).sin())
            .collect()
    }

    #[test]
    fn missing_model_dir_errors() {
        let err = resolve_model_file(Path::new("definitely/not/here")).unwrap_err();
        assert!(err.to_string().contains("no GTCRN model"));
    }

    #[test]
    #[cfg(feature = "ort")]
    fn loads_and_preserves_length() {
        let Some(dir) = model_dir() else {
            eprintln!("skip loads_and_preserves_length: VC_RS_GTCRN_MODEL unset");
            return;
        };
        let mut denoiser = GtcrnDenoiser::new(ort_config(&dir), 16_000).unwrap();
        for len in [1usize, 200, 256, 777, 4096] {
            let mut buf = tone(len);
            denoiser.process_in_place(&mut buf).unwrap();
            assert_eq!(buf.len(), len);
            assert!(buf.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    #[cfg(feature = "ort")]
    fn cache_continuity_across_chunk_partitions() {
        let Some(dir) = model_dir() else {
            eprintln!("skip cache_continuity_across_chunk_partitions: VC_RS_GTCRN_MODEL unset");
            return;
        };
        let input = tone(16_000);

        let mut whole = input.clone();
        GtcrnDenoiser::new(ort_config(&dir), 16_000)
            .unwrap()
            .process_in_place(&mut whole)
            .unwrap();

        let mut split_denoiser = GtcrnDenoiser::new(ort_config(&dir), 16_000).unwrap();
        let mut split = Vec::with_capacity(input.len());
        for chunk in input.chunks(777) {
            let mut out = chunk.to_vec();
            split_denoiser.process_in_place(&mut out).unwrap();
            split.extend_from_slice(&out);
        }

        // The adapter feeds an identical hop sequence in both cases, so the
        // streaming caches must evolve identically and the outputs match.
        let max_err = whole
            .iter()
            .zip(&split)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "whole vs split max error {max_err}");
    }

    #[test]
    #[cfg(feature = "ort")]
    fn reset_repeats_startup() {
        let Some(dir) = model_dir() else {
            eprintln!("skip reset_repeats_startup: VC_RS_GTCRN_MODEL unset");
            return;
        };
        let input = tone(8_000);
        let mut denoiser = GtcrnDenoiser::new(ort_config(&dir), 16_000).unwrap();

        let mut first = input.clone();
        denoiser.process_in_place(&mut first).unwrap();
        denoiser.reset().unwrap();
        let mut second = input.clone();
        denoiser.process_in_place(&mut second).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    #[cfg(native_tensorrt)]
    fn native_tensorrt_loads_and_preserves_length() {
        let Some(dir) = model_dir() else {
            eprintln!("skip native_tensorrt_loads_and_preserves_length: VC_RS_GTCRN_MODEL unset");
            return;
        };
        let mut denoiser = GtcrnDenoiser::new(native_config(&dir), 16_000).unwrap();
        for len in [1usize, 200, 256, 777, 4096] {
            let mut buf = tone(len);
            denoiser.process_in_place(&mut buf).unwrap();
            assert_eq!(buf.len(), len);
            assert!(buf.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    #[cfg(all(windows, native_tensorrt))]
    #[ignore = "requires VC_RS_GTCRN_MODEL, native builder and GPU; builds a fresh long Unicode cache"]
    fn native_tensorrt_long_unicode_cache_round_trip() {
        use std::os::windows::ffi::OsStrExt;
        let _model = model_dir().expect("set VC_RS_GTCRN_MODEL to the GTCRN model directory");
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "vc-gtcrn-native-paths-{}-{unique}",
            std::process::id()
        ));
        let cache = base
            .join("日本語のキャッシュ")
            .join("long-path-".repeat(18))
            .join("nested".repeat(10));
        assert!(cache.as_os_str().encode_wide().count() > 260);
        std::fs::create_dir_all(&cache).unwrap();
        // Use a subprocess rather than changing the shared process environment.
        // The first start must write the plan; the second must read that exact
        // cached plan without a usable builder. This catches both native seams.
        for cached in [false, true] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "denoise::gtcrn::tests::native_tensorrt_loads_and_preserves_length",
                    "--nocapture",
                ])
                .env("VC_RS_TENSORRT_CACHE_DIR", &cache);
            if cached {
                command.env(
                    "VC_RS_TENSORRT_BUILDER_HELPER",
                    base.join("missing-builder.exe"),
                );
            }
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "long Unicode cache (cached={cached}):\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        // Timing-cache paths use the same wide I/O contract as engine files.
        let versions: Vec<_> = std::fs::read_dir(&cache)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("native-trt-")
            })
            .collect();
        assert_eq!(versions.len(), 1);
        assert!(
            versions[0]
                .path()
                .join("build-v2/device-0/timing.cache")
                .metadata()
                .unwrap()
                .len()
                > 0
        );
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    #[cfg(native_tensorrt)]
    fn native_tensorrt_cache_continuity_across_chunk_partitions() {
        let Some(dir) = model_dir() else {
            eprintln!(
                "skip native_tensorrt_cache_continuity_across_chunk_partitions: VC_RS_GTCRN_MODEL unset"
            );
            return;
        };
        let input = tone(16_000);

        let mut whole = input.clone();
        GtcrnDenoiser::new(native_config(&dir), 16_000)
            .unwrap()
            .process_in_place(&mut whole)
            .unwrap();

        let mut split_denoiser = GtcrnDenoiser::new(native_config(&dir), 16_000).unwrap();
        let mut split = Vec::with_capacity(input.len());
        for chunk in input.chunks(777) {
            let mut out = chunk.to_vec();
            split_denoiser.process_in_place(&mut out).unwrap();
            split.extend_from_slice(&out);
        }

        let max_err = whole
            .iter()
            .zip(&split)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "whole vs split max error {max_err}");
    }

    #[test]
    #[cfg(native_tensorrt)]
    fn native_tensorrt_reset_repeats_startup() {
        let Some(dir) = model_dir() else {
            eprintln!("skip native_tensorrt_reset_repeats_startup: VC_RS_GTCRN_MODEL unset");
            return;
        };
        let input = tone(8_000);
        let mut denoiser = GtcrnDenoiser::new(native_config(&dir), 16_000).unwrap();

        let mut first = input.clone();
        denoiser.process_in_place(&mut first).unwrap();
        denoiser.reset().unwrap();
        let mut second = input.clone();
        denoiser.process_in_place(&mut second).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    #[cfg(all(feature = "ort", native_tensorrt))]
    fn native_tensorrt_matches_ort_within_tolerance() {
        let Some(dir) = model_dir() else {
            eprintln!("skip native_tensorrt_matches_ort_within_tolerance: VC_RS_GTCRN_MODEL unset");
            return;
        };
        let input = tone(4096);
        let mut ort = input.clone();
        let mut native = input.clone();

        GtcrnDenoiser::new(ort_config(&dir), 16_000)
            .unwrap()
            .process_in_place(&mut ort)
            .unwrap();
        GtcrnDenoiser::new(native_config(&dir), 16_000)
            .unwrap()
            .process_in_place(&mut native)
            .unwrap();

        let max_err = ort
            .iter()
            .zip(&native)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // TensorRT tactic selection can move the tiny recurrent graph by ~1e-2
        // at sample peaks; keep this as a backend-sanity bound, not bit parity.
        assert!(max_err < 2e-2, "ORT vs native TensorRT max error {max_err}");
    }

    #[cfg(feature = "ort")]
    fn ort_config(dir: &Path) -> GtcrnConfig<'_> {
        GtcrnConfig {
            model_dir: dir,
            backend: GtcrnBackend::OrtCpu,
        }
    }

    #[cfg(native_tensorrt)]
    fn native_config(dir: &Path) -> GtcrnConfig<'_> {
        GtcrnConfig {
            model_dir: dir,
            backend: GtcrnBackend::NativeTensorRt {
                gpu_priority: GpuPriority::default(),
                gpu_device_id: 0,
            },
        }
    }
}
