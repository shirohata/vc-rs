use std::env;
use std::ffi::CString;
use std::fs;
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{IoBinding, Session};
use ort::value::Tensor;
use ort::{ortsys, AsPointer};
use tracing::{debug, info, warn};

use crate::cli::Provider;

use super::sessions::HubertEmbedderSession;
use super::shape::onnx_silence_front_feature_frames;

// Fixed-shape GPU bindings are intentionally model-worker state, not audio callback state.
// Keep CUDA Graph tensor addresses stable for each session; do not allocate or re-bind them on chunk runs.

pub(super) const TENSORRT_DEVICE_ID: i32 = 0;
pub(super) const TENSORRT_CACHE_DIR_ENV: &str = "VC_RS_TENSORRT_CACHE_DIR";
pub(super) const TENSORRT_MODEL_HASH_BUFFER_BYTES: usize = 1024 * 1024;
pub(super) const TENSORRT_CUDA_GRAPH_ENV: &str = "VC_RS_TENSORRT_CUDA_GRAPH";
pub(super) const CUDA_GRAPH_ENV: &str = "VC_RS_CUDA_GRAPH";
pub(super) const CPU_ONNX_INTRA_THREADS: usize = 4;
pub(super) const CPU_ONNX_INTER_THREADS: usize = 4;

// These benchmark profiles intentionally match the trtexec-validated static
// shapes from the original TensorRT investigation. Normal pipeline execution
// derives exact profiles from chunk settings instead; do not mix the two
// casually because CUDA graph capture and engine cache reuse depend on stable
// input dimensions.
#[cfg(test)]
pub(super) const CONTENTVEC_AUDIO_DIMS: &[usize] = &[1, 24_000];
#[cfg(test)]
pub(super) const RMVPE_WAVEFORM_DIMS: &[usize] = &[1, 24_000];
#[cfg(test)]
pub(super) const RVC_FEATS_DIMS: &[usize] = &[1, 75, 768];
#[cfg(test)]
pub(super) const RVC_PITCH_DIMS: &[usize] = &[1, 75];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TensorRtInputShape {
    pub(super) name: String,
    pub(super) dims: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TensorRtSessionProfile {
    pub(super) role: ModelRole,
    pub(super) model_cache_key: Option<String>,
    pub(super) profile_shapes: String,
    pub(super) fixed_inputs: Vec<TensorRtInputShape>,
}

impl TensorRtSessionProfile {
    pub(super) fn new(role: ModelRole, fixed_inputs: Vec<TensorRtInputShape>) -> Self {
        let profile_shapes = tensor_rt_profile_shapes(&fixed_inputs);
        Self {
            role,
            model_cache_key: None,
            profile_shapes,
            fixed_inputs,
        }
    }

    #[cfg(test)]
    pub(super) fn with_model_cache_key(mut self, model_cache_key: impl Into<String>) -> Self {
        self.model_cache_key = Some(model_cache_key.into());
        self
    }

    pub(super) fn with_optional_model_cache_key(mut self, model_cache_key: Option<String>) -> Self {
        self.model_cache_key = model_cache_key;
        self
    }

    pub(super) fn single_input(
        role: ModelRole,
        input_name: impl Into<String>,
        samples: usize,
    ) -> Self {
        Self::new(
            role,
            vec![TensorRtInputShape {
                name: input_name.into(),
                dims: vec![1, samples],
            }],
        )
    }

    pub(super) fn rvc(frames: usize, channels: usize) -> Self {
        Self::new(
            ModelRole::Rvc,
            vec![
                TensorRtInputShape {
                    name: "feats".to_string(),
                    dims: vec![1, frames, channels],
                },
                TensorRtInputShape {
                    name: "pitch".to_string(),
                    dims: vec![1, frames],
                },
                TensorRtInputShape {
                    name: "pitchf".to_string(),
                    dims: vec![1, frames],
                },
            ],
        )
    }

    pub(super) fn cache_dir_from_root(&self, cache_root: &Path) -> Result<PathBuf> {
        let model_cache_key = self.model_cache_key()?;
        Ok(cache_root
            .join(self.role.label())
            .join(model_cache_key)
            .join(tensor_rt_cache_key(&self.profile_shapes)))
    }

    pub(super) fn model_cache_key(&self) -> Result<&str> {
        self.model_cache_key.as_deref().ok_or_else(|| {
            anyhow!(
                "TensorRT cache profile for {} is missing a model cache key",
                self.role.label()
            )
        })
    }

    pub(super) fn fixed_input_dims(&self, input_name: &str) -> Result<&[usize]> {
        self.fixed_inputs
            .iter()
            .find(|shape| shape.name == input_name)
            .map(|shape| shape.dims.as_slice())
            .ok_or_else(|| {
                anyhow!(
                    "TensorRT profile for {} does not include input '{input_name}'",
                    self.role.label()
                )
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ModelRole {
    ContentVec,
    Rmvpe,
    Rvc,
    Inspect,
}

impl ModelRole {
    pub(super) fn label(self) -> &'static str {
        match self {
            ModelRole::ContentVec => "contentvec",
            ModelRole::Rmvpe => "rmvpe",
            ModelRole::Rvc => "rvc",
            ModelRole::Inspect => "inspect",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TensorRtRunMode {
    PinnedCpu,
    DeviceIo,
    CudaGraph,
}

impl TensorRtRunMode {
    pub(super) fn from_env() -> Self {
        let value = env::var(TENSORRT_CUDA_GRAPH_ENV).ok();
        let mode = Self::parse_env(value.as_deref());
        if let Some(value) = value.as_deref() {
            if !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "off" | "no" | "1" | "true" | "on" | "yes"
            ) {
                warn!(
                    "{}={value:?} is not recognized; defaulting TensorRT CUDA Graph to enabled",
                    TENSORRT_CUDA_GRAPH_ENV
                );
            }
        }
        mode
    }

    pub(super) fn parse_env(value: Option<&str>) -> Self {
        match value.map(|value| value.trim().to_ascii_lowercase()) {
            Some(value) if matches!(value.as_str(), "0" | "false" | "off" | "no") => {
                Self::PinnedCpu
            }
            _ => Self::CudaGraph,
        }
    }

    pub(super) fn cuda_from_env() -> Self {
        let value = env::var(CUDA_GRAPH_ENV).ok();
        let mode = Self::parse_cuda_env(value.as_deref());
        if let Some(value) = value.as_deref() {
            if !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "off" | "no" | "1" | "true" | "on" | "yes"
            ) {
                warn!(
                    "{}={value:?} is not recognized; defaulting CUDA Graph to disabled",
                    CUDA_GRAPH_ENV
                );
            }
        }
        mode
    }

    pub(super) fn parse_cuda_env(value: Option<&str>) -> Self {
        match value.map(|value| value.trim().to_ascii_lowercase()) {
            Some(value) if matches!(value.as_str(), "1" | "true" | "on" | "yes") => Self::CudaGraph,
            _ => Self::DeviceIo,
        }
    }

    pub(super) fn cuda_graph(self) -> bool {
        matches!(self, Self::CudaGraph)
    }

    pub(super) fn device_io(self) -> bool {
        matches!(self, Self::DeviceIo | Self::CudaGraph)
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::PinnedCpu => "pinned-cpu-iobinding",
            Self::DeviceIo => "cuda-device-iobinding",
            Self::CudaGraph => "cuda-graph-device-iobinding",
        }
    }

    pub(super) fn bound_input_memory(self) -> &'static str {
        match self {
            Self::PinnedCpu => "CUDA_PINNED/CPUInput",
            Self::DeviceIo => "CUDA/Default",
            Self::CudaGraph => "CUDA/Default",
        }
    }

    pub(super) fn bound_output_memory(self) -> &'static str {
        match self {
            Self::PinnedCpu => "CUDA_PINNED/CPUOutput",
            Self::DeviceIo => "CUDA/Default",
            Self::CudaGraph => "CUDA/Default",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TensorRtSessionPurpose {
    Main,
    Probe,
    Final,
}

impl TensorRtSessionPurpose {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Probe => "probe",
            Self::Final => "final",
        }
    }
}

pub(super) struct TensorRtWarmupInfo {
    pub(super) rvc_feature_len: usize,
    pub(super) contentvec_output_shape: Vec<i64>,
}

pub(super) struct HubertTensorRtPinnedBinding {
    pub(super) binding: IoBinding,
    pub(super) audio: Tensor<f32>,
    pub(super) output: Tensor<f32>,
    pub(super) _input_allocator: Allocator,
    pub(super) _output_allocator: Allocator,
    pub(super) input_shape: Vec<usize>,
    pub(super) output_shape: Vec<usize>,
}

impl HubertTensorRtPinnedBinding {
    pub(super) fn new(
        session: &Session,
        input_name: &str,
        input_shape: &[usize],
        output_name: &str,
        output_shape: &[usize],
    ) -> Result<Self> {
        let input_allocator = tensor_rt_pinned_allocator(session, MemoryType::CPUInput)?;
        let output_allocator = tensor_rt_pinned_allocator(session, MemoryType::CPUOutput)?;
        let audio =
            Tensor::<f32>::new(&input_allocator, input_shape.to_vec()).with_context(|| {
                format!("failed to allocate TensorRT ContentVec input '{input_name}'")
            })?;
        let mut output = Tensor::<f32>::new(&output_allocator, output_shape.to_vec())
            .with_context(|| {
                format!("failed to allocate TensorRT ContentVec output '{output_name}'")
            })?;
        let mut binding = session
            .create_binding()
            .context("failed to create TensorRT ContentVec IoBinding")?;
        bind_output_tensor(&mut binding, output_name, &mut output).with_context(|| {
            format!("failed to bind TensorRT ContentVec output '{output_name}'")
        })?;
        Ok(Self {
            binding,
            audio,
            output,
            _input_allocator: input_allocator,
            _output_allocator: output_allocator,
            input_shape: input_shape.to_vec(),
            output_shape: output_shape.to_vec(),
        })
    }
}

pub(super) struct RmvpeTensorRtPinnedBinding {
    pub(super) binding: IoBinding,
    pub(super) waveform: Tensor<f32>,
    pub(super) threshold: Tensor<f32>,
    pub(super) output: Tensor<f32>,
    pub(super) _input_allocator: Allocator,
    pub(super) _output_allocator: Allocator,
    pub(super) waveform_shape: Vec<usize>,
    pub(super) output_shape: Vec<usize>,
    pub(super) bound_threshold: f32,
}

impl RmvpeTensorRtPinnedBinding {
    pub(super) fn new(
        session: &Session,
        waveform_shape: &[usize],
        output_shape: &[usize],
        threshold_value: f32,
    ) -> Result<Self> {
        let input_allocator = tensor_rt_pinned_allocator(session, MemoryType::CPUInput)?;
        let output_allocator = tensor_rt_pinned_allocator(session, MemoryType::CPUOutput)?;
        let waveform = Tensor::<f32>::new(&input_allocator, waveform_shape.to_vec())
            .context("failed to allocate TensorRT RMVPE input 'waveform'")?;
        let mut threshold = Tensor::<f32>::new(&input_allocator, vec![1usize])
            .context("failed to allocate TensorRT RMVPE input 'threshold'")?;
        write_scalar_f32_tensor(&mut threshold, threshold_value, "threshold")?;
        let mut output = Tensor::<f32>::new(&output_allocator, output_shape.to_vec())
            .context("failed to allocate TensorRT RMVPE output 'pitchf'")?;
        let mut binding = session
            .create_binding()
            .context("failed to create TensorRT RMVPE IoBinding")?;
        binding
            .bind_input("threshold", &threshold)
            .context("failed to bind TensorRT RMVPE input 'threshold'")?;
        bind_output_tensor(&mut binding, "pitchf", &mut output)
            .context("failed to bind TensorRT RMVPE output 'pitchf'")?;
        Ok(Self {
            binding,
            waveform,
            threshold,
            output,
            _input_allocator: input_allocator,
            _output_allocator: output_allocator,
            waveform_shape: waveform_shape.to_vec(),
            output_shape: output_shape.to_vec(),
            bound_threshold: threshold_value,
        })
    }

    pub(super) fn bind_threshold_if_changed(&mut self, threshold_value: f32) -> Result<()> {
        if self.bound_threshold == threshold_value {
            return Ok(());
        }
        write_scalar_f32_tensor(&mut self.threshold, threshold_value, "threshold")?;
        self.binding
            .bind_input("threshold", &self.threshold)
            .context("failed to re-bind TensorRT RMVPE input 'threshold'")?;
        self.bound_threshold = threshold_value;
        Ok(())
    }
}

pub(super) struct RvcTensorRtPinnedBinding {
    pub(super) binding: IoBinding,
    pub(super) feats: Tensor<f32>,
    pub(super) pitch: Tensor<i64>,
    pub(super) pitchf: Tensor<f32>,
    pub(super) p_len: Tensor<i64>,
    pub(super) sid: Tensor<i64>,
    pub(super) output: Tensor<f32>,
    pub(super) _input_allocator: Allocator,
    pub(super) _output_allocator: Allocator,
    pub(super) feats_shape: Vec<usize>,
    pub(super) pitch_shape: Vec<usize>,
    pub(super) output_shape: Vec<usize>,
    pub(super) bound_p_len: i64,
    pub(super) bound_sid: i64,
}

impl RvcTensorRtPinnedBinding {
    pub(super) fn new(
        session: &Session,
        feats_shape: &[usize],
        pitch_shape: &[usize],
        output_shape: &[usize],
        frame_len: i64,
        speaker_id: i64,
    ) -> Result<Self> {
        let input_allocator = tensor_rt_pinned_allocator(session, MemoryType::CPUInput)?;
        let output_allocator = tensor_rt_pinned_allocator(session, MemoryType::CPUOutput)?;
        let feats = Tensor::<f32>::new(&input_allocator, feats_shape.to_vec())
            .context("failed to allocate TensorRT RVC input 'feats'")?;
        let pitch = Tensor::<i64>::new(&input_allocator, pitch_shape.to_vec())
            .context("failed to allocate TensorRT RVC input 'pitch'")?;
        let pitchf = Tensor::<f32>::new(&input_allocator, pitch_shape.to_vec())
            .context("failed to allocate TensorRT RVC input 'pitchf'")?;
        let mut p_len = Tensor::<i64>::new(&input_allocator, vec![1usize])
            .context("failed to allocate TensorRT RVC input 'p_len'")?;
        let mut sid = Tensor::<i64>::new(&input_allocator, vec![1usize])
            .context("failed to allocate TensorRT RVC input 'sid'")?;
        write_scalar_i64_tensor(&mut p_len, frame_len, "p_len")?;
        write_scalar_i64_tensor(&mut sid, speaker_id, "sid")?;
        let mut output = Tensor::<f32>::new(&output_allocator, output_shape.to_vec())
            .context("failed to allocate TensorRT RVC output 'audio'")?;
        let mut binding = session
            .create_binding()
            .context("failed to create TensorRT RVC IoBinding")?;
        binding
            .bind_input("p_len", &p_len)
            .context("failed to bind TensorRT RVC input 'p_len'")?;
        binding
            .bind_input("sid", &sid)
            .context("failed to bind TensorRT RVC input 'sid'")?;
        bind_output_tensor(&mut binding, "audio", &mut output)
            .context("failed to bind TensorRT RVC output 'audio'")?;
        Ok(Self {
            binding,
            feats,
            pitch,
            pitchf,
            p_len,
            sid,
            output,
            _input_allocator: input_allocator,
            _output_allocator: output_allocator,
            feats_shape: feats_shape.to_vec(),
            pitch_shape: pitch_shape.to_vec(),
            output_shape: output_shape.to_vec(),
            bound_p_len: frame_len,
            bound_sid: speaker_id,
        })
    }

    pub(super) fn bind_fixed_scalars_if_changed(
        &mut self,
        frame_len: i64,
        speaker_id: i64,
    ) -> Result<()> {
        if self.bound_p_len != frame_len {
            write_scalar_i64_tensor(&mut self.p_len, frame_len, "p_len")?;
            self.binding
                .bind_input("p_len", &self.p_len)
                .context("failed to re-bind TensorRT RVC input 'p_len'")?;
            self.bound_p_len = frame_len;
        }
        if self.bound_sid != speaker_id {
            write_scalar_i64_tensor(&mut self.sid, speaker_id, "sid")?;
            self.binding
                .bind_input("sid", &self.sid)
                .context("failed to re-bind TensorRT RVC input 'sid'")?;
            self.bound_sid = speaker_id;
        }
        Ok(())
    }
}

pub(super) struct HubertTensorRtGraphBinding {
    pub(super) binding: IoBinding,
    pub(super) host_audio: Tensor<f32>,
    pub(super) device_audio: Tensor<f32>,
    pub(super) device_output: Tensor<f32>,
    pub(super) host_output: Tensor<f32>,
    pub(super) _host_input_allocator: Allocator,
    pub(super) _host_output_allocator: Allocator,
    pub(super) _device_allocator: Allocator,
    pub(super) input_shape: Vec<usize>,
    pub(super) output_shape: Vec<usize>,
}

impl HubertTensorRtGraphBinding {
    pub(super) fn new(
        session: &Session,
        input_name: &str,
        input_shape: &[usize],
        output_name: &str,
        output_shape: &[usize],
    ) -> Result<Self> {
        let host_input_allocator = tensor_rt_pinned_allocator(session, MemoryType::CPUInput)?;
        let host_output_allocator = tensor_rt_pinned_allocator(session, MemoryType::CPUOutput)?;
        let device_allocator = tensor_rt_device_allocator(session)?;
        let mut host_audio = Tensor::<f32>::new(&host_input_allocator, input_shape.to_vec())
            .with_context(|| {
                format!("failed to allocate TensorRT ContentVec host input '{input_name}'")
            })?;
        zero_f32_tensor(&mut host_audio, input_name)?;
        let mut device_audio = Tensor::<f32>::new(&device_allocator, input_shape.to_vec())
            .with_context(|| {
                format!("failed to allocate TensorRT ContentVec CUDA input '{input_name}'")
            })?;
        copy_f32_tensor_to_device(&host_audio, &mut device_audio, input_name)?;
        let mut device_output = Tensor::<f32>::new(&device_allocator, output_shape.to_vec())
            .with_context(|| {
                format!("failed to allocate TensorRT ContentVec CUDA output '{output_name}'")
            })?;
        let host_output = Tensor::<f32>::new(&host_output_allocator, output_shape.to_vec())
            .with_context(|| {
                format!("failed to allocate TensorRT ContentVec host output '{output_name}'")
            })?;
        let mut binding = session
            .create_binding()
            .context("failed to create ContentVec CUDA device IoBinding")?;
        binding
            .bind_input(input_name, &device_audio)
            .with_context(|| {
                format!("failed to bind TensorRT ContentVec CUDA input '{input_name}'")
            })?;
        bind_output_tensor(&mut binding, output_name, &mut device_output).with_context(|| {
            format!("failed to bind TensorRT ContentVec CUDA output '{output_name}'")
        })?;
        Ok(Self {
            binding,
            host_audio,
            device_audio,
            device_output,
            host_output,
            _host_input_allocator: host_input_allocator,
            _host_output_allocator: host_output_allocator,
            _device_allocator: device_allocator,
            input_shape: input_shape.to_vec(),
            output_shape: output_shape.to_vec(),
        })
    }

    pub(super) fn warmup_capture(
        &mut self,
        session: &mut Session,
        output_name: &str,
        role: ModelRole,
        provider: Provider,
        cuda_graph: bool,
    ) -> Result<()> {
        let run_start = Instant::now();
        let _outputs = session.run_binding(&self.binding)?;
        let run_us = run_start.elapsed().as_micros();
        let d2h_start = Instant::now();
        copy_f32_tensor_to_host(&self.device_output, &mut self.host_output, output_name)?;
        let d2h_us = d2h_start.elapsed().as_micros();
        validate_host_output_shape(&self.host_output, &self.output_shape, output_name)?;
        info!(
            "{} device IoBinding warmup completed model_role={} cuda_graph={} graph_capture={} input_shape={} output_shape={} run_us={} d2h_us={}",
            provider.label(),
            role.label(),
            cuda_graph,
            cuda_graph,
            format_usize_shape(&self.input_shape),
            format_usize_shape(&self.output_shape),
            run_us,
            d2h_us
        );
        Ok(())
    }
}

pub(super) struct RmvpeTensorRtGraphBinding {
    pub(super) binding: IoBinding,
    pub(super) host_waveform: Tensor<f32>,
    pub(super) device_waveform: Tensor<f32>,
    pub(super) host_threshold: Tensor<f32>,
    pub(super) device_threshold: Tensor<f32>,
    pub(super) device_output: Tensor<f32>,
    pub(super) host_output: Tensor<f32>,
    pub(super) _host_input_allocator: Allocator,
    pub(super) _host_output_allocator: Allocator,
    pub(super) _device_allocator: Allocator,
    pub(super) waveform_shape: Vec<usize>,
    pub(super) output_shape: Vec<usize>,
    pub(super) bound_threshold: f32,
}

impl RmvpeTensorRtGraphBinding {
    pub(super) fn new(
        session: &Session,
        waveform_shape: &[usize],
        output_shape: &[usize],
        threshold_value: f32,
    ) -> Result<Self> {
        let host_input_allocator = tensor_rt_pinned_allocator(session, MemoryType::CPUInput)?;
        let host_output_allocator = tensor_rt_pinned_allocator(session, MemoryType::CPUOutput)?;
        let device_allocator = tensor_rt_device_allocator(session)?;
        let mut host_waveform = Tensor::<f32>::new(&host_input_allocator, waveform_shape.to_vec())
            .context("failed to allocate TensorRT RMVPE host input 'waveform'")?;
        zero_f32_tensor(&mut host_waveform, "waveform")?;
        let mut device_waveform = Tensor::<f32>::new(&device_allocator, waveform_shape.to_vec())
            .context("failed to allocate TensorRT RMVPE CUDA input 'waveform'")?;
        copy_f32_tensor_to_device(&host_waveform, &mut device_waveform, "waveform")?;
        let mut host_threshold = Tensor::<f32>::new(&host_input_allocator, vec![1usize])
            .context("failed to allocate TensorRT RMVPE host input 'threshold'")?;
        write_scalar_f32_tensor(&mut host_threshold, threshold_value, "threshold")?;
        let mut device_threshold = Tensor::<f32>::new(&device_allocator, vec![1usize])
            .context("failed to allocate TensorRT RMVPE CUDA input 'threshold'")?;
        copy_f32_tensor_to_device(&host_threshold, &mut device_threshold, "threshold")?;
        let mut device_output = Tensor::<f32>::new(&device_allocator, output_shape.to_vec())
            .context("failed to allocate TensorRT RMVPE CUDA output 'pitchf'")?;
        let host_output = Tensor::<f32>::new(&host_output_allocator, output_shape.to_vec())
            .context("failed to allocate TensorRT RMVPE host output 'pitchf'")?;
        let mut binding = session
            .create_binding()
            .context("failed to create RMVPE CUDA device IoBinding")?;
        binding
            .bind_input("waveform", &device_waveform)
            .context("failed to bind TensorRT RMVPE CUDA input 'waveform'")?;
        binding
            .bind_input("threshold", &device_threshold)
            .context("failed to bind TensorRT RMVPE CUDA input 'threshold'")?;
        bind_output_tensor(&mut binding, "pitchf", &mut device_output)
            .context("failed to bind TensorRT RMVPE CUDA output 'pitchf'")?;
        Ok(Self {
            binding,
            host_waveform,
            device_waveform,
            host_threshold,
            device_threshold,
            device_output,
            host_output,
            _host_input_allocator: host_input_allocator,
            _host_output_allocator: host_output_allocator,
            _device_allocator: device_allocator,
            waveform_shape: waveform_shape.to_vec(),
            output_shape: output_shape.to_vec(),
            bound_threshold: threshold_value,
        })
    }

    pub(super) fn copy_threshold_if_changed(&mut self, threshold_value: f32) -> Result<()> {
        if self.bound_threshold == threshold_value {
            return Ok(());
        }
        write_scalar_f32_tensor(&mut self.host_threshold, threshold_value, "threshold")?;
        copy_f32_tensor_to_device(
            &self.host_threshold,
            &mut self.device_threshold,
            "threshold",
        )?;
        self.bound_threshold = threshold_value;
        Ok(())
    }

    pub(super) fn warmup_capture(
        &mut self,
        session: &mut Session,
        provider: Provider,
        cuda_graph: bool,
    ) -> Result<()> {
        let run_start = Instant::now();
        let _outputs = session.run_binding(&self.binding)?;
        let run_us = run_start.elapsed().as_micros();
        let d2h_start = Instant::now();
        copy_f32_tensor_to_host(&self.device_output, &mut self.host_output, "pitchf")?;
        let d2h_us = d2h_start.elapsed().as_micros();
        validate_host_output_shape(&self.host_output, &self.output_shape, "pitchf")?;
        info!(
            "{} device IoBinding warmup completed model_role={} cuda_graph={} graph_capture={} input_shape={} output_shape={} run_us={} d2h_us={}",
            provider.label(),
            ModelRole::Rmvpe.label(),
            cuda_graph,
            cuda_graph,
            format_usize_shape(&self.waveform_shape),
            format_usize_shape(&self.output_shape),
            run_us,
            d2h_us
        );
        Ok(())
    }
}

pub(super) struct RvcTensorRtGraphBinding {
    pub(super) binding: IoBinding,
    pub(super) host_feats: Tensor<f32>,
    pub(super) device_feats: Tensor<f32>,
    pub(super) host_pitch: Tensor<i64>,
    pub(super) device_pitch: Tensor<i64>,
    pub(super) host_pitchf: Tensor<f32>,
    pub(super) device_pitchf: Tensor<f32>,
    pub(super) host_p_len: Tensor<i64>,
    pub(super) device_p_len: Tensor<i64>,
    pub(super) host_sid: Tensor<i64>,
    pub(super) device_sid: Tensor<i64>,
    pub(super) device_output: Tensor<f32>,
    pub(super) host_output: Tensor<f32>,
    pub(super) _host_input_allocator: Allocator,
    pub(super) _host_output_allocator: Allocator,
    pub(super) _device_allocator: Allocator,
    pub(super) feats_shape: Vec<usize>,
    pub(super) pitch_shape: Vec<usize>,
    pub(super) output_shape: Vec<usize>,
    pub(super) bound_p_len: i64,
    pub(super) bound_sid: i64,
}

impl RvcTensorRtGraphBinding {
    pub(super) fn new(
        session: &Session,
        feats_shape: &[usize],
        pitch_shape: &[usize],
        output_shape: &[usize],
        frame_len: i64,
        speaker_id: i64,
    ) -> Result<Self> {
        let host_input_allocator = tensor_rt_pinned_allocator(session, MemoryType::CPUInput)?;
        let host_output_allocator = tensor_rt_pinned_allocator(session, MemoryType::CPUOutput)?;
        let device_allocator = tensor_rt_device_allocator(session)?;
        let mut host_feats = Tensor::<f32>::new(&host_input_allocator, feats_shape.to_vec())
            .context("failed to allocate TensorRT RVC host input 'feats'")?;
        zero_f32_tensor(&mut host_feats, "feats")?;
        let mut device_feats = Tensor::<f32>::new(&device_allocator, feats_shape.to_vec())
            .context("failed to allocate TensorRT RVC CUDA input 'feats'")?;
        copy_f32_tensor_to_device(&host_feats, &mut device_feats, "feats")?;
        let mut host_pitch = Tensor::<i64>::new(&host_input_allocator, pitch_shape.to_vec())
            .context("failed to allocate TensorRT RVC host input 'pitch'")?;
        fill_i64_tensor(&mut host_pitch, 1, "pitch")?;
        let mut device_pitch = Tensor::<i64>::new(&device_allocator, pitch_shape.to_vec())
            .context("failed to allocate TensorRT RVC CUDA input 'pitch'")?;
        copy_i64_tensor_to_device(&host_pitch, &mut device_pitch, "pitch")?;
        let mut host_pitchf = Tensor::<f32>::new(&host_input_allocator, pitch_shape.to_vec())
            .context("failed to allocate TensorRT RVC host input 'pitchf'")?;
        zero_f32_tensor(&mut host_pitchf, "pitchf")?;
        let mut device_pitchf = Tensor::<f32>::new(&device_allocator, pitch_shape.to_vec())
            .context("failed to allocate TensorRT RVC CUDA input 'pitchf'")?;
        copy_f32_tensor_to_device(&host_pitchf, &mut device_pitchf, "pitchf")?;
        let mut host_p_len = Tensor::<i64>::new(&host_input_allocator, vec![1usize])
            .context("failed to allocate TensorRT RVC host input 'p_len'")?;
        write_scalar_i64_tensor(&mut host_p_len, frame_len, "p_len")?;
        let mut device_p_len = Tensor::<i64>::new(&device_allocator, vec![1usize])
            .context("failed to allocate TensorRT RVC CUDA input 'p_len'")?;
        copy_i64_tensor_to_device(&host_p_len, &mut device_p_len, "p_len")?;
        let mut host_sid = Tensor::<i64>::new(&host_input_allocator, vec![1usize])
            .context("failed to allocate TensorRT RVC host input 'sid'")?;
        write_scalar_i64_tensor(&mut host_sid, speaker_id, "sid")?;
        let mut device_sid = Tensor::<i64>::new(&device_allocator, vec![1usize])
            .context("failed to allocate TensorRT RVC CUDA input 'sid'")?;
        copy_i64_tensor_to_device(&host_sid, &mut device_sid, "sid")?;
        let mut device_output = Tensor::<f32>::new(&device_allocator, output_shape.to_vec())
            .context("failed to allocate TensorRT RVC CUDA output 'audio'")?;
        let host_output = Tensor::<f32>::new(&host_output_allocator, output_shape.to_vec())
            .context("failed to allocate TensorRT RVC host output 'audio'")?;
        let mut binding = session
            .create_binding()
            .context("failed to create RVC CUDA device IoBinding")?;
        binding
            .bind_input("feats", &device_feats)
            .context("failed to bind TensorRT RVC CUDA input 'feats'")?;
        binding
            .bind_input("pitch", &device_pitch)
            .context("failed to bind TensorRT RVC CUDA input 'pitch'")?;
        binding
            .bind_input("pitchf", &device_pitchf)
            .context("failed to bind TensorRT RVC CUDA input 'pitchf'")?;
        binding
            .bind_input("p_len", &device_p_len)
            .context("failed to bind TensorRT RVC CUDA input 'p_len'")?;
        binding
            .bind_input("sid", &device_sid)
            .context("failed to bind TensorRT RVC CUDA input 'sid'")?;
        bind_output_tensor(&mut binding, "audio", &mut device_output)
            .context("failed to bind TensorRT RVC CUDA output 'audio'")?;
        Ok(Self {
            binding,
            host_feats,
            device_feats,
            host_pitch,
            device_pitch,
            host_pitchf,
            device_pitchf,
            host_p_len,
            device_p_len,
            host_sid,
            device_sid,
            device_output,
            host_output,
            _host_input_allocator: host_input_allocator,
            _host_output_allocator: host_output_allocator,
            _device_allocator: device_allocator,
            feats_shape: feats_shape.to_vec(),
            pitch_shape: pitch_shape.to_vec(),
            output_shape: output_shape.to_vec(),
            bound_p_len: frame_len,
            bound_sid: speaker_id,
        })
    }

    pub(super) fn copy_fixed_scalars_if_changed(
        &mut self,
        frame_len: i64,
        speaker_id: i64,
    ) -> Result<()> {
        if self.bound_p_len != frame_len {
            write_scalar_i64_tensor(&mut self.host_p_len, frame_len, "p_len")?;
            copy_i64_tensor_to_device(&self.host_p_len, &mut self.device_p_len, "p_len")?;
            self.bound_p_len = frame_len;
        }
        if self.bound_sid != speaker_id {
            write_scalar_i64_tensor(&mut self.host_sid, speaker_id, "sid")?;
            copy_i64_tensor_to_device(&self.host_sid, &mut self.device_sid, "sid")?;
            self.bound_sid = speaker_id;
        }
        Ok(())
    }

    pub(super) fn warmup_capture(
        &mut self,
        session: &mut Session,
        provider: Provider,
        cuda_graph: bool,
    ) -> Result<()> {
        let run_start = Instant::now();
        let _outputs = session.run_binding(&self.binding)?;
        let run_us = run_start.elapsed().as_micros();
        let d2h_start = Instant::now();
        copy_f32_tensor_to_host(&self.device_output, &mut self.host_output, "audio")?;
        let d2h_us = d2h_start.elapsed().as_micros();
        validate_host_output_shape(&self.host_output, &self.output_shape, "audio")?;
        info!(
            "{} device IoBinding warmup completed model_role={} cuda_graph={} graph_capture={} feats_shape={} pitch_shape={} output_shape={} run_us={} d2h_us={}",
            provider.label(),
            ModelRole::Rvc.label(),
            cuda_graph,
            cuda_graph,
            format_usize_shape(&self.feats_shape),
            format_usize_shape(&self.pitch_shape),
            format_usize_shape(&self.output_shape),
            run_us,
            d2h_us
        );
        Ok(())
    }
}

pub(super) enum HubertTensorRtBinding {
    Pinned(HubertTensorRtPinnedBinding),
    CudaGraph(HubertTensorRtGraphBinding),
}

pub(super) enum RmvpeTensorRtBinding {
    Pinned(RmvpeTensorRtPinnedBinding),
    CudaGraph(RmvpeTensorRtGraphBinding),
}

pub(super) enum RvcTensorRtBinding {
    Pinned(RvcTensorRtPinnedBinding),
    CudaGraph(RvcTensorRtGraphBinding),
}

#[cfg(test)]
pub(super) fn tensor_rt_benchmark_profile(role: ModelRole) -> Result<TensorRtSessionProfile> {
    match role {
        ModelRole::ContentVec => Ok(TensorRtSessionProfile::new(
            role,
            vec![TensorRtInputShape {
                name: "audio".to_string(),
                dims: CONTENTVEC_AUDIO_DIMS.to_vec(),
            }],
        )),
        ModelRole::Rmvpe => Ok(TensorRtSessionProfile::new(
            role,
            vec![TensorRtInputShape {
                name: "waveform".to_string(),
                dims: RMVPE_WAVEFORM_DIMS.to_vec(),
            }],
        )),
        ModelRole::Rvc => Ok(TensorRtSessionProfile::new(
            role,
            vec![
                TensorRtInputShape {
                    name: "feats".to_string(),
                    dims: RVC_FEATS_DIMS.to_vec(),
                },
                TensorRtInputShape {
                    name: "pitch".to_string(),
                    dims: RVC_PITCH_DIMS.to_vec(),
                },
                TensorRtInputShape {
                    name: "pitchf".to_string(),
                    dims: RVC_PITCH_DIMS.to_vec(),
                },
            ],
        )),
        ModelRole::Inspect => bail!("TensorRT inspect profile requires a concrete model role"),
    }
}

pub(super) fn tensor_rt_warmup_feature_len(
    embedder: &mut HubertEmbedderSession,
    input_samples_16k: usize,
    extra_convert_samples: usize,
) -> Result<TensorRtWarmupInfo> {
    let silence = vec![0.0; input_samples_16k];
    let mut features = embedder.extract(&silence)?;
    let contentvec_output_shape = features.shape.clone();
    features.repeat_frames(2)?;
    let feature_len = feature_len_from_shape(&features.shape, "embedder warmup output")?;
    let silence_front_frames = onnx_silence_front_feature_frames(extra_convert_samples);
    if silence_front_frames > 0 && silence_front_frames < feature_len {
        features.trim_front_frames(silence_front_frames)?;
    }
    let feature_len = feature_len_from_shape(&features.shape, "trimmed embedder warmup output")?;
    if feature_len == 0 {
        bail!("TensorRT warmup produced zero RVC frames");
    }
    info!(
        "TensorRT warmup derived RVC frame count: contentvec_input_samples={} rvc_frames={}",
        input_samples_16k, feature_len
    );
    Ok(TensorRtWarmupInfo {
        rvc_feature_len: feature_len,
        contentvec_output_shape,
    })
}

pub(super) fn feature_len_from_shape(shape: &[i64], context: &str) -> Result<usize> {
    let len = shape
        .get(1)
        .copied()
        .with_context(|| format!("{context} must be rank-3 [1, frames, channels]"))?;
    if len <= 0 {
        bail!("{context} has non-positive frame length {len}");
    }
    usize::try_from(len).with_context(|| format!("{context} frame length does not fit in usize"))
}

pub(super) fn tensor_rt_profile_shapes(inputs: &[TensorRtInputShape]) -> String {
    inputs
        .iter()
        .map(|input| format!("{}:{}", input.name, format_usize_shape(&input.dims)))
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn tensor_rt_cache_root() -> Result<PathBuf> {
    let override_dir = env::var_os(TENSORRT_CACHE_DIR_ENV);
    tensor_rt_cache_root_from_override(override_dir.as_deref())
}

pub(super) fn tensor_rt_cache_root_from_override(
    override_dir: Option<&std::ffi::OsStr>,
) -> Result<PathBuf> {
    if let Some(root) = override_dir {
        if !root.is_empty() {
            return Ok(PathBuf::from(root));
        }
    }
    tensor_rt_default_cache_root()
}

// TensorRT timing cache stores tactic timings, not serialized model engines.
// Keep it shared under the TensorRT cache root so builds for different models
// can reuse compatible layer timings while model engines remain isolated below.
pub(super) fn tensor_rt_timing_cache_dir_from_root(cache_root: &Path) -> PathBuf {
    cache_root.join("timing")
}

#[cfg(windows)]
pub(super) fn tensor_rt_default_cache_root() -> Result<PathBuf> {
    let local_app_data = env::var_os("LOCALAPPDATA").ok_or_else(|| {
        anyhow!(
            "LOCALAPPDATA is not set; set {} to choose a TensorRT cache directory",
            TENSORRT_CACHE_DIR_ENV
        )
    })?;
    Ok(PathBuf::from(local_app_data)
        .join("vc-rs")
        .join("tensorrt-cache"))
}

#[cfg(not(windows))]
pub(super) fn tensor_rt_default_cache_root() -> Result<PathBuf> {
    if let Some(xdg_cache_home) = env::var_os("XDG_CACHE_HOME") {
        if !xdg_cache_home.is_empty() {
            return Ok(PathBuf::from(xdg_cache_home)
                .join("vc-rs")
                .join("tensorrt-cache"));
        }
    }
    let home = env::var_os("HOME").ok_or_else(|| {
        anyhow!(
            "HOME is not set; set {} to choose a TensorRT cache directory",
            TENSORRT_CACHE_DIR_ENV
        )
    })?;
    Ok(PathBuf::from(home)
        .join(".cache")
        .join("vc-rs")
        .join("tensorrt-cache"))
}

// Keep the model fingerprint in the cache path. ORT/TensorRT engine file names
// do not include the original ONNX path, so shape-only cache dirs let different
// models overwrite or invalidate each other's engines.
pub(super) fn tensor_rt_model_cache_key(path: &Path) -> Result<String> {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(tensor_rt_sanitize_cache_component)
        .unwrap_or_else(|| "model".to_string());
    let hash = tensor_rt_model_file_hash(path)?;
    Ok(format!("{stem}_{hash:016x}"))
}

pub(super) fn tensor_rt_model_file_hash(path: &Path) -> Result<u64> {
    let mut file = fs::File::open(path).with_context(|| {
        format!(
            "failed to open TensorRT model for cache key {}",
            path.display()
        )
    })?;
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    let mut buffer = vec![0u8; TENSORRT_MODEL_HASH_BUFFER_BYTES];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read TensorRT model {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.digest())
}

pub(super) fn tensor_rt_sanitize_cache_component(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect();
    if sanitized.is_empty() {
        "model".to_string()
    } else {
        sanitized
    }
}

pub(super) fn tensor_rt_cache_key(profile_shapes: &str) -> String {
    profile_shapes
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

pub(super) fn tensor_rt_cache_has_entries(path: &Path) -> bool {
    match fs::read_dir(path) {
        Ok(mut entries) => entries.next().is_some(),
        Err(err) if err.kind() == ErrorKind::NotFound => false,
        Err(err) => {
            debug!(
                "failed to inspect TensorRT cache directory {} before session creation: {err}",
                path.display()
            );
            false
        }
    }
}

pub(super) fn provider_uses_fixed_shape(provider: Provider) -> bool {
    provider.is_tensorrt() || provider.is_cuda()
}

pub(super) fn tensor_rt_pinned_allocator(
    session: &Session,
    memory_type: MemoryType,
) -> Result<Allocator> {
    let memory_info = MemoryInfo::new(
        AllocationDevice::CUDA_PINNED,
        TENSORRT_DEVICE_ID,
        AllocatorType::Device,
        memory_type,
    )
    .map_err(|err| anyhow!("failed to create CUDA pinned MemoryInfo ({memory_type:?}): {err}"))?;
    Allocator::new(session, memory_info)
        .map_err(|err| anyhow!("failed to create CUDA pinned allocator ({memory_type:?}): {err}"))
}

pub(super) fn tensor_rt_device_allocator(session: &Session) -> Result<Allocator> {
    let memory_info = MemoryInfo::new(
        AllocationDevice::CUDA,
        TENSORRT_DEVICE_ID,
        AllocatorType::Device,
        MemoryType::Default,
    )
    .map_err(|err| anyhow!("failed to create CUDA device MemoryInfo: {err}"))?;
    Allocator::new(session, memory_info)
        .map_err(|err| anyhow!("failed to create CUDA device allocator: {err}"))
}

pub(super) fn bind_output_tensor(
    binding: &mut IoBinding,
    output_name: &str,
    output: &mut Tensor<f32>,
) -> Result<()> {
    let output_name = CString::new(output_name)
        .with_context(|| format!("TensorRT output name contains NUL: {output_name:?}"))?;
    ortsys![unsafe BindOutput(binding.ptr_mut(), output_name.as_ptr(), output.ptr())?];
    Ok(())
}

// Guardrail: the pinned CPU path below is intentionally kept as the explicit
// TensorRT CUDA Graph opt-out. ort::IoBinding::bind_input copies CPU/CUDA-pinned
// data at bind time, so changing inputs must be re-bound before run_binding.
// The CUDA Graph path must not copy this pattern: graph capture requires stable
// CUDA device tensor addresses that remain bound for the session lifetime.
pub(super) fn copy_f32_tensor(tensor: &mut Tensor<f32>, src: &[f32], name: &str) -> Result<()> {
    let (_, dst) = tensor.extract_tensor_mut();
    if dst.len() != src.len() {
        bail!(
            "fixed-shape IoBinding input '{name}' length mismatch: expected {}, got {}",
            dst.len(),
            src.len()
        );
    }
    dst.copy_from_slice(src);
    Ok(())
}

pub(super) fn zero_f32_tensor(tensor: &mut Tensor<f32>, name: &str) -> Result<()> {
    let (_, dst) = tensor.extract_tensor_mut();
    if dst.is_empty() {
        bail!("fixed-shape IoBinding input '{name}' must not be empty");
    }
    dst.fill(0.0);
    Ok(())
}

pub(super) fn copy_i64_tensor(tensor: &mut Tensor<i64>, src: &[i64], name: &str) -> Result<()> {
    let (_, dst) = tensor.extract_tensor_mut();
    if dst.len() != src.len() {
        bail!(
            "fixed-shape IoBinding input '{name}' length mismatch: expected {}, got {}",
            dst.len(),
            src.len()
        );
    }
    dst.copy_from_slice(src);
    Ok(())
}

pub(super) fn fill_i64_tensor(tensor: &mut Tensor<i64>, value: i64, name: &str) -> Result<()> {
    let (_, dst) = tensor.extract_tensor_mut();
    if dst.is_empty() {
        bail!("fixed-shape IoBinding input '{name}' must not be empty");
    }
    dst.fill(value);
    Ok(())
}

pub(super) fn write_scalar_f32_tensor(
    tensor: &mut Tensor<f32>,
    value: f32,
    name: &str,
) -> Result<()> {
    let (_, dst) = tensor.extract_tensor_mut();
    if dst.len() != 1 {
        bail!(
            "fixed-shape IoBinding scalar input '{name}' length mismatch: expected 1, got {}",
            dst.len()
        );
    }
    dst[0] = value;
    Ok(())
}

pub(super) fn write_scalar_i64_tensor(
    tensor: &mut Tensor<i64>,
    value: i64,
    name: &str,
) -> Result<()> {
    let (_, dst) = tensor.extract_tensor_mut();
    if dst.len() != 1 {
        bail!(
            "fixed-shape IoBinding scalar input '{name}' length mismatch: expected 1, got {}",
            dst.len()
        );
    }
    dst[0] = value;
    Ok(())
}

// Guardrail: CUDA device IoBinding fixes tensor shapes and GPU addresses, which
// are separate requirements for CUDA Graph replay. Do not allocate or re-bind
// these device tensors on the runtime path. Tensor::copy_into currently routes
// through ort's cached identity sessions guarded by an internal mutex, so this
// model path must remain outside the real-time audio callback. The owning model
// runs through &mut self; do not introduce concurrent runs against a
// graph-enabled session.
pub(super) fn copy_f32_tensor_to_device(
    host_tensor: &Tensor<f32>,
    device_tensor: &mut Tensor<f32>,
    name: &str,
) -> Result<()> {
    host_tensor
        .copy_into(device_tensor)
        .with_context(|| format!("failed to copy TensorRT input '{name}' to CUDA device tensor"))
}

pub(super) fn copy_i64_tensor_to_device(
    host_tensor: &Tensor<i64>,
    device_tensor: &mut Tensor<i64>,
    name: &str,
) -> Result<()> {
    host_tensor
        .copy_into(device_tensor)
        .with_context(|| format!("failed to copy TensorRT input '{name}' to CUDA device tensor"))
}

pub(super) fn copy_f32_tensor_to_host(
    device_tensor: &Tensor<f32>,
    host_tensor: &mut Tensor<f32>,
    name: &str,
) -> Result<()> {
    device_tensor
        .copy_into(host_tensor)
        .with_context(|| format!("failed to copy TensorRT output '{name}' to host tensor"))
}

pub(super) fn validate_host_output_shape(
    tensor: &Tensor<f32>,
    expected_shape: &[usize],
    output_name: &str,
) -> Result<()> {
    let (shape, _) = tensor.try_extract_tensor::<f32>()?;
    let actual_shape = i64_shape_to_usize(shape, output_name)?;
    if actual_shape != expected_shape {
        bail!(
            "TensorRT bound output '{output_name}' shape changed from {} to {}",
            format_usize_shape(expected_shape),
            format_usize_shape(&actual_shape)
        );
    }
    Ok(())
}

pub(super) fn validate_tensorrt_input_shape(
    provider: Provider,
    tensor_rt_profile: Option<&TensorRtSessionProfile>,
    input_name: &str,
    actual: &[usize],
) -> Result<()> {
    if !provider_uses_fixed_shape(provider) {
        return Ok(());
    }
    let tensor_rt_profile = tensor_rt_profile
        .ok_or_else(|| anyhow!("fixed-shape input validation requires a session profile"))?;
    let expected = tensor_rt_profile.fixed_input_dims(input_name)?;
    if actual != expected {
        bail!(
            "{} fixed profile for {} requires input '{}' shape {}, got {}; changing chunk-related settings requires a matching fixed-shape model load",
            provider.label(),
            tensor_rt_profile.role.label(),
            input_name,
            format_usize_shape(expected),
            format_usize_shape(actual)
        );
    }
    Ok(())
}

pub(super) fn i64_shape_to_usize(shape: &[i64], input_name: &str) -> Result<Vec<usize>> {
    shape
        .iter()
        .map(|dim| {
            usize::try_from(*dim).with_context(|| {
                format!("input '{input_name}' shape contains negative or too-large dim {dim}")
            })
        })
        .collect()
}

pub(super) fn format_usize_shape(shape: &[usize]) -> String {
    shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join("x")
}
