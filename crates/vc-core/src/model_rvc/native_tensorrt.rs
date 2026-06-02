use std::env;
#[cfg(native_tensorrt)]
use std::ffi::CString;
use std::num::NonZeroUsize;
#[cfg(native_tensorrt)]
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};

#[cfg(native_tensorrt)]
use anyhow::Context;
use anyhow::{anyhow, bail, Result};
#[cfg(native_tensorrt)]
use tracing::info;

pub(super) const NATIVE_TENSORRT_RVC_ENGINE_ENV: &str = "VC_RS_TENSORRT_RVC_ENGINE";

#[cfg(native_tensorrt)]
mod ffi {
    use super::*;

    unsafe extern "C" {
        pub(super) fn vc_rs_trt_rvc_create(
            engine_path: *const c_char,
            frames: c_int,
            channels: c_int,
            message: *mut c_char,
            message_len: usize,
        ) -> *mut c_void;
        pub(super) fn vc_rs_trt_rvc_destroy(native: *mut c_void);
        pub(super) fn vc_rs_trt_rvc_output_len(native: *const c_void) -> usize;
        pub(super) fn vc_rs_trt_rvc_infer(
            native: *mut c_void,
            feats: *const f32,
            feats_len: usize,
            pitch: *const i64,
            pitch_len: usize,
            pitchf: *const f32,
            pitchf_len: usize,
            speaker_id: i64,
            output: *mut f32,
            output_len: usize,
            message: *mut c_char,
            message_len: usize,
        ) -> c_int;
    }
}

pub(super) fn configured_rvc_engine_path(override_path: Option<&Path>) -> Option<PathBuf> {
    override_path.map(Path::to_path_buf).or_else(|| {
        env::var_os(NATIVE_TENSORRT_RVC_ENGINE_ENV)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    })
}

pub(super) struct NativeRvcEngine {
    #[cfg(native_tensorrt)]
    handle: std::ptr::NonNull<c_void>,
    frames: NonZeroUsize,
    channels: NonZeroUsize,
    #[cfg(native_tensorrt)]
    output_len: NonZeroUsize,
    path: PathBuf,
}

// The native TensorRT handle owns one execution context, one CUDA stream, and
// fixed device buffers. The model worker moves this object across threads but
// does not share it; inference requires `&mut self`, so concurrent enqueue on
// the same TensorRT context is not exposed.
unsafe impl Send for NativeRvcEngine {}

impl NativeRvcEngine {
    pub(super) fn load(path: &Path, frames: usize, channels: usize) -> Result<Self> {
        let frames = NonZeroUsize::new(frames).ok_or_else(|| anyhow!("RVC frame count is zero"))?;
        let channels =
            NonZeroUsize::new(channels).ok_or_else(|| anyhow!("RVC channel count is zero"))?;
        load_impl(path, frames, channels)
    }

    pub(super) fn infer(
        &mut self,
        feats: &[f32],
        pitch: &[i64],
        pitchf: &[f32],
        speaker_id: i64,
    ) -> Result<Vec<f32>> {
        if feats.len() != self.frames.get() * self.channels.get() {
            bail!(
                "native TensorRT RVC feats length mismatch: got {}, expected {}",
                feats.len(),
                self.frames.get() * self.channels.get()
            );
        }
        if pitch.len() != self.frames.get() || pitchf.len() != self.frames.get() {
            bail!(
                "native TensorRT RVC pitch length mismatch: pitch={} pitchf={} expected={}",
                pitch.len(),
                pitchf.len(),
                self.frames.get()
            );
        }
        infer_impl(self, feats, pitch, pitchf, speaker_id)
    }

    pub(super) fn frames(&self) -> usize {
        self.frames.get()
    }

    pub(super) fn channels(&self) -> usize {
        self.channels.get()
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(native_tensorrt)]
impl Drop for NativeRvcEngine {
    fn drop(&mut self) {
        unsafe {
            ffi::vc_rs_trt_rvc_destroy(self.handle.as_ptr());
        }
    }
}

#[cfg(native_tensorrt)]
fn load_impl(path: &Path, frames: NonZeroUsize, channels: NonZeroUsize) -> Result<NativeRvcEngine> {
    let c_path = CString::new(path.to_string_lossy().as_bytes())
        .context("native TensorRT RVC engine path contains an interior NUL byte")?;
    let mut message = MessageBuffer::new();
    let handle = unsafe {
        ffi::vc_rs_trt_rvc_create(
            c_path.as_ptr(),
            usize_to_c_int(frames.get(), "RVC frame count")?,
            usize_to_c_int(channels.get(), "RVC channel count")?,
            message.as_mut_ptr(),
            message.len(),
        )
    };
    let handle = std::ptr::NonNull::new(handle).ok_or_else(|| {
        anyhow!(
            "failed to load native TensorRT RVC engine: {}",
            message.text()
        )
    })?;
    let output_len = unsafe { ffi::vc_rs_trt_rvc_output_len(handle.as_ptr()) };
    let output_len = NonZeroUsize::new(output_len)
        .ok_or_else(|| anyhow!("native TensorRT RVC engine reported zero output length"))?;
    info!("{}", message.text().trim_end());
    Ok(NativeRvcEngine {
        handle,
        frames,
        channels,
        output_len,
        path: path.to_path_buf(),
    })
}

#[cfg(not(native_tensorrt))]
fn load_impl(
    path: &Path,
    _frames: NonZeroUsize,
    _channels: NonZeroUsize,
) -> Result<NativeRvcEngine> {
    bail!(
        "native TensorRT RVC engine {} was requested via {}, but this binary was built without native_tensorrt support",
        path.display(),
        NATIVE_TENSORRT_RVC_ENGINE_ENV
    )
}

#[cfg(native_tensorrt)]
fn infer_impl(
    engine: &mut NativeRvcEngine,
    feats: &[f32],
    pitch: &[i64],
    pitchf: &[f32],
    speaker_id: i64,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; engine.output_len.get()];
    let mut message = MessageBuffer::new();
    let status = unsafe {
        ffi::vc_rs_trt_rvc_infer(
            engine.handle.as_ptr(),
            feats.as_ptr(),
            feats.len(),
            pitch.as_ptr(),
            pitch.len(),
            pitchf.as_ptr(),
            pitchf.len(),
            speaker_id,
            output.as_mut_ptr(),
            output.len(),
            message.as_mut_ptr(),
            message.len(),
        )
    };
    if status != 0 {
        bail!("native TensorRT RVC inference failed: {}", message.text());
    }
    Ok(output)
}

#[cfg(not(native_tensorrt))]
fn infer_impl(
    _engine: &mut NativeRvcEngine,
    _feats: &[f32],
    _pitch: &[i64],
    _pitchf: &[f32],
    _speaker_id: i64,
) -> Result<Vec<f32>> {
    bail!("native TensorRT RVC inference is unavailable in this binary")
}

#[cfg(native_tensorrt)]
fn usize_to_c_int(value: usize, label: &str) -> Result<c_int> {
    c_int::try_from(value).with_context(|| format!("{label} does not fit in c_int"))
}

#[cfg(native_tensorrt)]
struct MessageBuffer {
    data: Vec<c_char>,
}

#[cfg(native_tensorrt)]
impl MessageBuffer {
    fn new() -> Self {
        Self {
            data: vec![0; 16 * 1024],
        }
    }

    fn as_mut_ptr(&mut self) -> *mut c_char {
        self.data.as_mut_ptr()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn text(&self) -> String {
        let nul = self
            .data
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.data.len());
        let bytes = self.data[..nul]
            .iter()
            .map(|&b| b as u8)
            .collect::<Vec<_>>();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
