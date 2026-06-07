#[cfg(native_tensorrt)]
use std::ffi::CString;
use std::num::NonZeroUsize;
#[cfg(native_tensorrt)]
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
#[cfg(native_tensorrt)]
use std::process::Command;

use anyhow::Context;
use anyhow::{anyhow, bail, Result};
use tracing::info;

use super::feature::FeatureTensor;
use super::tensorrt::{
    format_usize_shape, tensor_rt_cache_root, TensorRtInputShape, TensorRtSessionProfile,
};

#[cfg(native_tensorrt)]
mod ffi {
    use super::*;

    unsafe extern "C" {
        pub(super) fn vc_rs_trt_engine_create(
            engine_path: *const c_char,
            profile_shapes: *const c_char,
            output_name: *const c_char,
            high_priority: i32,
            message: *mut c_char,
            message_len: usize,
        ) -> *mut c_void;
        pub(super) fn vc_rs_trt_engine_destroy(native: *mut c_void);
        pub(super) fn vc_rs_trt_engine_output_len(native: *const c_void) -> usize;
        pub(super) fn vc_rs_trt_contentvec_infer(
            native: *mut c_void,
            input_name: *const c_char,
            audio: *const f32,
            audio_len: usize,
            output: *mut f32,
            output_len: usize,
            message: *mut c_char,
            message_len: usize,
        ) -> c_int;
        pub(super) fn vc_rs_trt_rmvpe_infer(
            native: *mut c_void,
            waveform: *const f32,
            waveform_len: usize,
            threshold: f32,
            output: *mut f32,
            output_len: usize,
            message: *mut c_char,
            message_len: usize,
        ) -> c_int;
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

#[cfg_attr(not(native_tensorrt), allow(dead_code))]
pub(super) struct NativeContentVecEngine {
    #[cfg(native_tensorrt)]
    handle: std::ptr::NonNull<c_void>,
    input_name: String,
    input_len: NonZeroUsize,
    expected_channels: NonZeroUsize,
    output_len: NonZeroUsize,
}

pub(super) struct NativeRmvpeEngine {
    #[cfg(native_tensorrt)]
    handle: std::ptr::NonNull<c_void>,
    waveform_len: NonZeroUsize,
    output_len: NonZeroUsize,
}

#[cfg_attr(not(native_tensorrt), allow(dead_code))]
pub(super) struct NativeRvcEngine {
    #[cfg(native_tensorrt)]
    handle: std::ptr::NonNull<c_void>,
    frames: NonZeroUsize,
    channels: NonZeroUsize,
    output_len: NonZeroUsize,
}

// Native TensorRT handles own CUDA streams, execution contexts, and fixed device
// buffers. They move with the model worker but are never shared; inference takes
// &mut self so concurrent enqueue on one context is not exposed.
unsafe impl Send for NativeContentVecEngine {}
unsafe impl Send for NativeRmvpeEngine {}
unsafe impl Send for NativeRvcEngine {}

impl NativeContentVecEngine {
    pub(super) fn load(
        model_path: &Path,
        profile: &TensorRtSessionProfile,
        input_name: &str,
        output_name: &str,
        expected_channels: i64,
    ) -> Result<Self> {
        let input_shape = profile.fixed_input_dims(input_name)?;
        let input_len = shape_volume(input_shape, "ContentVec input")?;
        let expected_channels = usize::try_from(expected_channels)
            .ok()
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| anyhow!("ContentVec expected channel count must be positive"))?;
        let path = ensure_native_engine(model_path, profile, profile.profile_shapes.as_str())?;
        let handle = load_engine(
            &path,
            profile.profile_shapes.as_str(),
            output_name,
            profile.gpu_priority,
        )?;
        let output_len = engine_output_len(handle)?;
        info!(
            "loaded native TensorRT ContentVec engine model={} engine={} input={} input_shape={} output={} output_len={}",
            model_path.display(),
            path.display(),
            input_name,
            format_usize_shape(input_shape),
            output_name,
            output_len.get()
        );
        Ok(Self {
            #[cfg(native_tensorrt)]
            handle,
            input_name: input_name.to_string(),
            input_len,
            expected_channels,
            output_len,
        })
    }

    /// ContentVec frame count derived from the engine's self-reported output
    /// length. The engine knows its fixed output shape after deserialize, so the
    /// frame count needs no warmup inference.
    pub(super) fn output_frames(&self) -> Result<usize> {
        if !self
            .output_len
            .get()
            .is_multiple_of(self.expected_channels.get())
        {
            bail!(
                "native TensorRT ContentVec output length {} is not divisible by expected channels {}",
                self.output_len.get(),
                self.expected_channels.get()
            );
        }
        Ok(self.output_len.get() / self.expected_channels.get())
    }

    pub(super) fn extract(&mut self, audio_16k: &[f32]) -> Result<FeatureTensor> {
        if audio_16k.len() != self.input_len.get() {
            bail!(
                "native TensorRT ContentVec input length mismatch: got {}, expected {}",
                audio_16k.len(),
                self.input_len.get()
            );
        }
        let output = infer_contentvec(self, audio_16k)?;
        if output.len() % self.expected_channels.get() != 0 {
            bail!(
                "native TensorRT ContentVec output length {} is not divisible by expected channels {}",
                output.len(),
                self.expected_channels.get()
            );
        }
        let frames = output.len() / self.expected_channels.get();
        Ok(FeatureTensor {
            data: output,
            shape: vec![1, frames as i64, self.expected_channels.get() as i64],
        })
    }
}

impl NativeRmvpeEngine {
    pub(super) fn load(model_path: &Path, profile: &TensorRtSessionProfile) -> Result<Self> {
        let waveform_shape = profile.fixed_input_dims("waveform")?;
        let waveform_len = shape_volume(waveform_shape, "RMVPE waveform")?;
        let load_profile = profile_with_scalars(profile, &[("threshold", &[1usize])]);
        let path = ensure_native_engine(model_path, profile, profile.profile_shapes.as_str())?;
        let handle = load_engine(&path, load_profile.as_str(), "pitchf", profile.gpu_priority)?;
        let output_len = engine_output_len(handle)?;
        info!(
            "loaded native TensorRT RMVPE engine model={} engine={} waveform_shape={} output_len={}",
            model_path.display(),
            path.display(),
            format_usize_shape(waveform_shape),
            output_len.get()
        );
        Ok(Self {
            #[cfg(native_tensorrt)]
            handle,
            waveform_len,
            output_len,
        })
    }

    pub(super) fn warmup_output_shape(&self) -> Vec<i64> {
        vec![1, self.output_len.get() as i64]
    }

    pub(super) fn extract(
        &mut self,
        audio_16k: &[f32],
        pitch_shift: f32,
        threshold: f32,
    ) -> Result<Vec<f32>> {
        if audio_16k.len() != self.waveform_len.get() {
            bail!(
                "native TensorRT RMVPE waveform length mismatch: got {}, expected {}",
                audio_16k.len(),
                self.waveform_len.get()
            );
        }
        let mut output = infer_rmvpe(self, audio_16k, threshold)?;
        let factor = 2.0f32.powf(pitch_shift / 12.0);
        for value in &mut output {
            *value *= factor;
        }
        Ok(output)
    }
}

impl NativeRvcEngine {
    pub(super) fn load(
        model_path: &Path,
        profile: &TensorRtSessionProfile,
        channels: usize,
    ) -> Result<Self> {
        let pitch_shape = profile.fixed_input_dims("pitch")?;
        let frames = pitch_shape
            .get(1)
            .copied()
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| anyhow!("RVC pitch profile must be [1, frames] with frames > 0"))?;
        let channels =
            NonZeroUsize::new(channels).ok_or_else(|| anyhow!("RVC channels is zero"))?;
        let load_profile =
            profile_with_scalars(profile, &[("p_len", &[1usize]), ("sid", &[1usize])]);
        let path = ensure_native_engine(model_path, profile, profile.profile_shapes.as_str())?;
        let handle = load_engine(&path, load_profile.as_str(), "audio", profile.gpu_priority)?;
        let output_len = engine_output_len(handle)?;
        info!(
            "loaded native TensorRT RVC engine model={} engine={} frames={} channels={} output_len={}",
            model_path.display(),
            path.display(),
            frames.get(),
            channels.get(),
            output_len.get()
        );
        Ok(Self {
            #[cfg(native_tensorrt)]
            handle,
            frames,
            channels,
            output_len,
        })
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
        infer_rvc(self, feats, pitch, pitchf, speaker_id)
    }

    pub(super) fn frames(&self) -> usize {
        self.frames.get()
    }

    pub(super) fn channels(&self) -> usize {
        self.channels.get()
    }

    /// RVC `audio` output length self-reported by the engine after deserialize.
    /// Replaces the throwaway warmup inference previously used to learn it.
    pub(super) fn output_len(&self) -> usize {
        self.output_len.get()
    }
}

#[cfg(native_tensorrt)]
impl Drop for NativeContentVecEngine {
    fn drop(&mut self) {
        unsafe { ffi::vc_rs_trt_engine_destroy(self.handle.as_ptr()) };
    }
}

#[cfg(native_tensorrt)]
impl Drop for NativeRmvpeEngine {
    fn drop(&mut self) {
        unsafe { ffi::vc_rs_trt_engine_destroy(self.handle.as_ptr()) };
    }
}

#[cfg(native_tensorrt)]
impl Drop for NativeRvcEngine {
    fn drop(&mut self) {
        unsafe { ffi::vc_rs_trt_engine_destroy(self.handle.as_ptr()) };
    }
}

fn native_engine_path(profile: &TensorRtSessionProfile) -> Result<PathBuf> {
    Ok(profile
        .cache_dir_from_root(&tensor_rt_cache_root()?)?
        .join("native.engine"))
}

fn ensure_native_engine(
    model_path: &Path,
    profile: &TensorRtSessionProfile,
    build_profile_shapes: &str,
) -> Result<PathBuf> {
    let engine_path = native_engine_path(profile)?;
    if engine_path
        .metadata()
        .is_ok_and(|metadata| metadata.len() > 0)
    {
        return Ok(engine_path);
    }
    let parent = engine_path
        .parent()
        .ok_or_else(|| anyhow!("native TensorRT engine path has no parent"))?;
    std::fs::create_dir_all(parent)?;
    build_engine(model_path, &engine_path, build_profile_shapes)?;
    Ok(engine_path)
}

fn shape_volume(shape: &[usize], label: &str) -> Result<NonZeroUsize> {
    let len = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .with_context(|| format!("{label} shape length overflow"))?;
    NonZeroUsize::new(len).ok_or_else(|| anyhow!("{label} shape is zero"))
}

fn profile_with_scalars(profile: &TensorRtSessionProfile, scalars: &[(&str, &[usize])]) -> String {
    let mut inputs = profile.fixed_inputs.clone();
    for (name, dims) in scalars {
        if !inputs.iter().any(|input| input.name == *name) {
            inputs.push(TensorRtInputShape {
                name: (*name).to_string(),
                dims: dims.to_vec(),
            });
        }
    }
    super::tensorrt::tensor_rt_profile_shapes(&inputs)
}

#[cfg(native_tensorrt)]
fn build_engine(model_path: &Path, engine_path: &Path, profile_shapes: &str) -> Result<()> {
    let tmp_engine = engine_path.with_extension(format!("engine.tmp-{}", std::process::id()));
    let _ = std::fs::remove_file(&tmp_engine);

    // Persistent timing cache shared by every engine build (all model roles and
    // shapes), so high builder optimization levels reuse measured tactic timings
    // instead of re-timing from scratch on each cache miss. Lives at the cache
    // root; TensorRT validates its header and ignores an incompatible blob.
    let timing_cache = tensor_rt_cache_root()?.join("timing.cache");

    // Keep engine construction out of this process. ORT-free trtexec and the
    // ORT-free helper both build the RVC graph successfully, while the same
    // Builder API fails after ORT has initialized in the main process.
    let output =
        match tensor_rt_builder_command(model_path, &tmp_engine, profile_shapes, &timing_cache)? {
            BuilderCommand::Exe(mut command) | BuilderCommand::Cargo(mut command) => command
                .output()
                .with_context(|| "failed to launch native TensorRT builder helper")?,
        };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        let _ = std::fs::remove_file(&tmp_engine);
        bail!(
            "native TensorRT builder helper failed with status {}:\nstdout:\n{}\nstderr:\n{}",
            output.status,
            stdout.trim_end(),
            stderr.trim_end()
        );
    }
    if !tmp_engine
        .metadata()
        .is_ok_and(|metadata| metadata.len() > 0)
    {
        bail!(
            "native TensorRT builder helper completed but did not create a non-empty engine: {}",
            tmp_engine.display()
        );
    }
    let _ = std::fs::remove_file(engine_path);
    std::fs::rename(&tmp_engine, engine_path).with_context(|| {
        format!(
            "failed to install native TensorRT engine {}",
            engine_path.display()
        )
    })?;
    let stdout = stdout.trim_end();
    if !stdout.is_empty() {
        info!("{}", stdout);
    }
    let stderr = stderr.trim_end();
    if !stderr.is_empty() {
        info!("{}", stderr);
    }
    Ok(())
}

#[cfg(native_tensorrt)]
enum BuilderCommand {
    Exe(Command),
    Cargo(Command),
}

#[cfg(native_tensorrt)]
fn tensor_rt_builder_command(
    model_path: &Path,
    engine_path: &Path,
    profile_shapes: &str,
    timing_cache: &Path,
) -> Result<BuilderCommand> {
    if let Some(path) = std::env::var_os("VC_RS_TENSORRT_BUILDER_HELPER")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    {
        let mut command = Command::new(&path);
        add_builder_args(
            &mut command,
            model_path,
            engine_path,
            profile_shapes,
            timing_cache,
        );
        return Ok(BuilderCommand::Exe(command));
    }

    for candidate in tensor_rt_builder_candidates()? {
        if candidate.is_file() {
            let mut command = Command::new(&candidate);
            add_builder_args(
                &mut command,
                model_path,
                engine_path,
                profile_shapes,
                timing_cache,
            );
            return Ok(BuilderCommand::Exe(command));
        }
    }

    if let Some(manifest) = tensor_rt_builder_manifest() {
        let mut command = Command::new("cargo");
        command
            .arg("run")
            .arg("--manifest-path")
            .arg(manifest)
            .arg("--");
        add_builder_args(
            &mut command,
            model_path,
            engine_path,
            profile_shapes,
            timing_cache,
        );
        return Ok(BuilderCommand::Cargo(command));
    }

    bail!(
        "native TensorRT engine cache miss requires an ORT-free builder helper. \
         Bundle vc-tensorrt-builder.exe next to the plugin/executable (package-tensorrt.ps1 \
         does this automatically), or set VC_RS_TENSORRT_BUILDER_HELPER to override its path"
    )
}

#[cfg(native_tensorrt)]
fn add_builder_args(
    command: &mut Command,
    model_path: &Path,
    engine_path: &Path,
    profile_shapes: &str,
    timing_cache: &Path,
) {
    command
        .arg("--onnx")
        .arg(model_path)
        .arg("--save-engine")
        .arg(engine_path)
        .arg("--profile")
        .arg(profile_shapes)
        .arg("--timing-cache")
        .arg(timing_cache);

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW: the helper is a console-subsystem exe; without this a
        // console window flashes each time the plugin (a GUI DLL with no console
        // in the DAW process) spawns it for a first-run engine build. stdout/stderr
        // are still captured via Command::output(), so logs/errors are unaffected.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
}

#[cfg(native_tensorrt)]
fn tensor_rt_builder_candidates() -> Result<Vec<PathBuf>> {
    let mut candidates = Vec::new();
    // Highest priority: the directory holding *this* module's binary. For a VST3
    // plugin the host process is the DAW, so `current_exe()` is the DAW exe, not
    // the plugin — only the module directory points at the bundled helper that
    // `package-tensorrt.ps1` copies next to the plugin DLL (alongside the
    // nvonnxparser / builder-resource DLLs it loads at startup). This is what
    // makes a packaged TensorRT plugin self-contained without an env override.
    if let Some(module_dir) = current_module_dir() {
        candidates.push(module_dir.join("vc-tensorrt-builder.exe"));
    }
    // For the CLI the module dir equals the exe dir, but keep this fallback for
    // any front-end where `current_exe()` is the binary that links vc-core.
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
    {
        candidates.push(exe_dir.join("vc-tensorrt-builder.exe"));
    }
    if let Some(root) = tensor_rt_workspace_root() {
        candidates.push(
            root.join("target")
                .join("debug")
                .join("vc-tensorrt-builder.exe"),
        );
        candidates.push(
            root.join("target")
                .join("release")
                .join("vc-tensorrt-builder.exe"),
        );
    }
    Ok(candidates)
}

// Directory of the binary that contains this code (the plugin DLL inside a DAW,
// or the CLI exe), found from the address of a function in this module rather
// than from `current_exe()` — the two differ for a DLL loaded into a host.
#[cfg(all(native_tensorrt, windows))]
fn current_module_dir() -> Option<PathBuf> {
    use std::os::windows::ffi::OsStringExt;

    // GetModuleHandleExW with FROM_ADDRESS treats `lpModuleName` as an address
    // and returns the module containing it; UNCHANGED_REFCOUNT avoids leaking a
    // reference (we don't own/free the handle). 0x4 | 0x2 per Win32 headers.
    const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x0000_0004;
    const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 = 0x0000_0002;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleExW(
            dwFlags: u32,
            lpModuleName: *const u16,
            phModule: *mut *mut c_void,
        ) -> i32;
        fn GetModuleFileNameW(hModule: *mut c_void, lpFilename: *mut u16, nSize: u32) -> u32;
    }

    let mut module: *mut c_void = std::ptr::null_mut();
    // Address of a local function: guaranteed to live in this module's image.
    let anchor = current_module_dir as *const c_void as *const u16;
    let ok = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            anchor,
            &mut module,
        )
    };
    if ok == 0 {
        return None;
    }

    // Grow the buffer until the path fits (return value == capacity means the
    // name was truncated, per GetModuleFileNameW's contract).
    let mut buf = vec![0u16; 260];
    loop {
        let len = unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) };
        if len == 0 {
            return None;
        }
        if (len as usize) < buf.len() {
            buf.truncate(len as usize);
            break;
        }
        buf.resize(buf.len() * 2, 0);
    }

    let path = PathBuf::from(std::ffi::OsString::from_wide(&buf));
    path.parent().map(Path::to_path_buf)
}

#[cfg(all(native_tensorrt, not(windows)))]
fn current_module_dir() -> Option<PathBuf> {
    None
}

#[cfg(native_tensorrt)]
fn tensor_rt_builder_manifest() -> Option<PathBuf> {
    let manifest = tensor_rt_workspace_root()?
        .join("tools")
        .join("tensorrt_builder")
        .join("Cargo.toml");
    manifest.is_file().then_some(manifest)
}

// Dev-only fallback to the build-machine workspace path. Gated to debug builds:
// release (= distributed) binaries must NOT embed env!("CARGO_MANIFEST_DIR"),
// which is an absolute path containing the developer's user name. trim-paths does
// not cover env! values, so this must be cfg-gated (a runtime `if` would still
// bake the string literal in). A packaged build finds the bundled helper via
// current_module_dir() first, so this fallback is never needed in distribution.
#[cfg(all(native_tensorrt, debug_assertions))]
fn tensor_rt_workspace_root() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.parent()?.parent().map(Path::to_path_buf)
}

#[cfg(all(native_tensorrt, not(debug_assertions)))]
fn tensor_rt_workspace_root() -> Option<PathBuf> {
    None
}

#[cfg(not(native_tensorrt))]
fn build_engine(model_path: &Path, _engine_path: &Path, _profile_shapes: &str) -> Result<()> {
    bail!(
        "native TensorRT engine build was requested for {}, but this binary was built without native_tensorrt support",
        model_path.display()
    )
}

#[cfg(native_tensorrt)]
fn load_engine(
    engine_path: &Path,
    profile_shapes: &str,
    output_name: &str,
    gpu_priority: super::GpuPriority,
) -> Result<std::ptr::NonNull<c_void>> {
    let c_engine = path_cstring(engine_path, "TensorRT engine path")?;
    let c_profile = CString::new(profile_shapes)
        .context("TensorRT profile shape string contains an interior NUL byte")?;
    let c_output = CString::new(output_name).context("TensorRT output name contains NUL byte")?;
    let mut message = MessageBuffer::new();
    let handle = unsafe {
        ffi::vc_rs_trt_engine_create(
            c_engine.as_ptr(),
            c_profile.as_ptr(),
            c_output.as_ptr(),
            i32::from(gpu_priority == super::GpuPriority::High),
            message.as_mut_ptr(),
            message.len(),
        )
    };
    let handle = std::ptr::NonNull::new(handle)
        .ok_or_else(|| anyhow!("failed to load native TensorRT engine: {}", message.text()))?;
    info!("{}", message.text().trim_end());
    Ok(handle)
}

#[cfg(native_tensorrt)]
fn engine_output_len(handle: std::ptr::NonNull<c_void>) -> Result<NonZeroUsize> {
    let output_len = unsafe { ffi::vc_rs_trt_engine_output_len(handle.as_ptr()) };
    NonZeroUsize::new(output_len)
        .ok_or_else(|| anyhow!("native TensorRT engine reported zero output length"))
}

#[cfg(not(native_tensorrt))]
fn load_engine(
    _engine_path: &Path,
    _profile_shapes: &str,
    _output_name: &str,
    _gpu_priority: super::GpuPriority,
) -> Result<()> {
    bail!("native TensorRT engine loading is unavailable in this binary")
}

#[cfg(not(native_tensorrt))]
fn engine_output_len(_handle: ()) -> Result<NonZeroUsize> {
    bail!("native TensorRT engine loading is unavailable in this binary")
}

#[cfg(native_tensorrt)]
fn infer_contentvec(engine: &mut NativeContentVecEngine, audio: &[f32]) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; engine.output_len.get()];
    let input_name = CString::new(engine.input_name.as_str())
        .context("ContentVec input name contains an interior NUL byte")?;
    let mut message = MessageBuffer::new();
    let status = unsafe {
        ffi::vc_rs_trt_contentvec_infer(
            engine.handle.as_ptr(),
            input_name.as_ptr(),
            audio.as_ptr(),
            audio.len(),
            output.as_mut_ptr(),
            output.len(),
            message.as_mut_ptr(),
            message.len(),
        )
    };
    if status != 0 {
        bail!(
            "native TensorRT ContentVec inference failed: {}",
            message.text()
        );
    }
    Ok(output)
}

#[cfg(not(native_tensorrt))]
fn infer_contentvec(_engine: &mut NativeContentVecEngine, _audio: &[f32]) -> Result<Vec<f32>> {
    bail!("native TensorRT ContentVec inference is unavailable in this binary")
}

#[cfg(native_tensorrt)]
fn infer_rmvpe(
    engine: &mut NativeRmvpeEngine,
    waveform: &[f32],
    threshold: f32,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; engine.output_len.get()];
    let mut message = MessageBuffer::new();
    let status = unsafe {
        ffi::vc_rs_trt_rmvpe_infer(
            engine.handle.as_ptr(),
            waveform.as_ptr(),
            waveform.len(),
            threshold,
            output.as_mut_ptr(),
            output.len(),
            message.as_mut_ptr(),
            message.len(),
        )
    };
    if status != 0 {
        bail!("native TensorRT RMVPE inference failed: {}", message.text());
    }
    Ok(output)
}

#[cfg(not(native_tensorrt))]
fn infer_rmvpe(
    _engine: &mut NativeRmvpeEngine,
    _waveform: &[f32],
    _threshold: f32,
) -> Result<Vec<f32>> {
    bail!("native TensorRT RMVPE inference is unavailable in this binary")
}

#[cfg(native_tensorrt)]
fn infer_rvc(
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
fn infer_rvc(
    _engine: &mut NativeRvcEngine,
    _feats: &[f32],
    _pitch: &[i64],
    _pitchf: &[f32],
    _speaker_id: i64,
) -> Result<Vec<f32>> {
    bail!("native TensorRT RVC inference is unavailable in this binary")
}

#[cfg(native_tensorrt)]
fn path_cstring(path: &Path, label: &str) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes())
        .with_context(|| format!("{label} contains an interior NUL byte"))
}

#[cfg(native_tensorrt)]
struct MessageBuffer {
    data: Vec<c_char>,
}

#[cfg(native_tensorrt)]
impl MessageBuffer {
    fn new() -> Self {
        Self {
            data: vec![0; 64 * 1024],
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

#[cfg(all(test, native_tensorrt, windows))]
mod tests {
    use super::*;

    // The test binary statically links vc-core, so the module containing this
    // code IS the test exe — its module dir must equal the exe dir. This proves
    // the GetModuleHandleExW/GetModuleFileNameW FFI returns a real, correct path
    // (the property the packaged-plugin discovery relies on).
    #[test]
    fn module_dir_matches_exe_dir_for_static_binary() {
        let module_dir = current_module_dir().expect("module dir should resolve");
        let exe_dir = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        assert_eq!(module_dir, exe_dir);
    }

    // The bundled helper must be the first candidate so a packaged plugin finds
    // its co-located builder before any workspace/dev fallback.
    #[test]
    fn builder_candidates_lead_with_module_dir() {
        let candidates = tensor_rt_builder_candidates().unwrap();
        let module_dir = current_module_dir().expect("module dir should resolve");
        assert_eq!(
            candidates.first(),
            Some(&module_dir.join("vc-tensorrt-builder.exe"))
        );
    }
}
