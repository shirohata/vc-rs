use std::path::Path;
#[cfg(feature = "ort")]
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
#[cfg(feature = "ort")]
use ort::ep;
#[cfg(feature = "ort")]
use ort::memory::Allocator;
#[cfg(feature = "ort")]
use ort::session::builder::GraphOptimizationLevel;
#[cfg(feature = "ort")]
use ort::session::{IoBinding, Session};
#[cfg(feature = "ort")]
use ort::value::{Tensor, TensorRef, ValueType};
#[cfg(feature = "ort")]
use tracing::debug;
use tracing::info;

use crate::Provider;

use super::feature::FeatureTensor;
use super::native_tensorrt::{NativeContentVecEngine, NativeRmvpeEngine, NativeRvcEngine};
use super::onnx_meta::read_model_io;
use super::tensorrt::*;

pub(super) struct HubertEmbedderSession {
    // Present only in the ORT build, where it backs CPU/CUDA inference. The
    // native TensorRT-only build drops ORT entirely and runs through `native`.
    #[cfg(feature = "ort")]
    pub(super) session: Session,
    pub(super) provider: Provider,
    pub(super) tensor_rt_profile: Option<TensorRtSessionProfile>,
    pub(super) tensor_rt_run_mode: TensorRtRunMode,
    #[cfg(feature = "ort")]
    pub(super) tensor_rt_binding: Option<HubertTensorRtBinding>,
    native: Option<NativeContentVecEngine>,
    pub(super) input_name: String,
    pub(super) output_name: String,
}

impl HubertEmbedderSession {
    pub(super) fn load(
        path: &Path,
        provider: Provider,
        expected_channels: i64,
        requested_output: Option<&str>,
        tensor_rt_profile: Option<TensorRtSessionProfile>,
        tensor_rt_run_mode: TensorRtRunMode,
        tensor_rt_session_purpose: TensorRtSessionPurpose,
    ) -> Result<Self> {
        // Names/validation come from the provider-neutral ONNX reader so the
        // native TensorRT path needs no ORT session.
        let io = read_model_io(path)?;
        let input_name = io.single_input_name()?.to_string();
        let output_name = io.select_embedder_output(expected_channels, requested_output)?;
        let native = if provider.is_tensorrt() {
            let profile = tensor_rt_profile.as_ref().ok_or_else(|| {
                anyhow!("native TensorRT ContentVec requires a fixed-shape profile")
            })?;
            Some(NativeContentVecEngine::load(
                path,
                profile,
                input_name.as_str(),
                output_name.as_str(),
                expected_channels,
            )?)
        } else {
            None
        };
        // In the ORT build the session backs CPU/CUDA inference; the native
        // TensorRT path keeps a CPU session only as an unused placeholder there
        // (it is compiled out of the TensorRT-only build).
        #[cfg(feature = "ort")]
        let session = {
            let session_provider = if provider.is_tensorrt() {
                Provider::Cpu
            } else {
                provider
            };
            load_session(
                path,
                session_provider,
                ModelRole::ContentVec,
                tensor_rt_profile.as_ref(),
                tensor_rt_run_mode,
                tensor_rt_session_purpose,
            )?
        };
        info!(
            "loaded embedder: {} input={} output={}",
            path.display(),
            input_name,
            output_name
        );
        Ok(Self {
            #[cfg(feature = "ort")]
            session,
            provider,
            tensor_rt_profile,
            tensor_rt_run_mode,
            #[cfg(feature = "ort")]
            tensor_rt_binding: None,
            native,
            input_name,
            output_name,
        })
    }

    /// ContentVec output frame count from the native TensorRT engine, when this
    /// embedder is backed by one. `None` for ORT-backed sessions. The engine
    /// self-reports its fixed output length, so this needs no warmup inference.
    pub(super) fn native_contentvec_output_frames(&self) -> Option<Result<usize>> {
        self.native.as_ref().map(|native| native.output_frames())
    }

    #[cfg(feature = "ort")]
    pub(super) fn enable_tensorrt_binding(
        &mut self,
        output_shape: &[i64],
        shared_waveform: Option<&TensorRtSharedWaveform>,
    ) -> Result<()> {
        if !provider_uses_fixed_shape(self.provider) {
            return Ok(());
        }
        if self.native.is_some() {
            return Ok(());
        }
        let profile = self
            .tensor_rt_profile
            .as_ref()
            .ok_or_else(|| anyhow!("ContentVec IoBinding requires a fixed-shape profile"))?;
        let input_shape = profile.fixed_input_dims(self.input_name.as_str())?;
        let output_shape = i64_shape_to_usize(output_shape, "contentvec output")?;
        let binding = match self.tensor_rt_run_mode {
            TensorRtRunMode::PinnedCpu => {
                HubertTensorRtBinding::Pinned(HubertTensorRtPinnedBinding::new(
                    &self.session,
                    self.input_name.as_str(),
                    input_shape,
                    self.output_name.as_str(),
                    &output_shape,
                )?)
            }
            TensorRtRunMode::DeviceIo | TensorRtRunMode::CudaGraph => {
                let mut binding = HubertTensorRtGraphBinding::new(
                    &self.session,
                    self.input_name.as_str(),
                    input_shape,
                    self.output_name.as_str(),
                    &output_shape,
                    shared_waveform,
                )?;
                binding.warmup_capture(
                    &mut self.session,
                    self.output_name.as_str(),
                    ModelRole::ContentVec,
                    self.provider,
                    self.tensor_rt_run_mode.cuda_graph(),
                )?;
                HubertTensorRtBinding::CudaGraph(binding)
            }
        };
        let shared_waveform_input = match &binding {
            HubertTensorRtBinding::Pinned(_) => false,
            HubertTensorRtBinding::CudaGraph(binding) => binding.shared_waveform_input,
        };
        info!(
            "GPU IoBinding enabled backend={} model_role={} mode={} cuda_graph={} device_io={} input={} input_shape={} output={} output_shape={} shared_waveform_input={} host_input_memory=CUDA_PINNED/CPUInput host_output_memory=CUDA_PINNED/CPUOutput bound_input_memory={} bound_output_memory={}",
            self.provider.label(),
            ModelRole::ContentVec.label(),
            self.tensor_rt_run_mode.label(),
            self.tensor_rt_run_mode.cuda_graph(),
            self.tensor_rt_run_mode.device_io(),
            self.input_name,
            format_usize_shape(input_shape),
            self.output_name,
            format_usize_shape(&output_shape),
            shared_waveform_input,
            self.tensor_rt_run_mode.bound_input_memory(),
            self.tensor_rt_run_mode.bound_output_memory()
        );
        self.tensor_rt_binding = Some(binding);
        Ok(())
    }

    pub(super) fn extract(&mut self, audio_16k: &[f32]) -> Result<FeatureTensor> {
        let input_shape = [1usize, audio_16k.len()];
        validate_tensorrt_input_shape(
            self.provider,
            self.tensor_rt_profile.as_ref(),
            self.input_name.as_str(),
            &input_shape,
        )?;
        #[cfg(feature = "ort")]
        if self.tensor_rt_binding.is_some() {
            return self.extract_with_binding(audio_16k);
        }
        if let Some(native) = self.native.as_mut() {
            return native.extract(audio_16k);
        }
        #[cfg(feature = "ort")]
        {
            self.extract_with_session_run(audio_16k, &input_shape)
        }
        #[cfg(not(feature = "ort"))]
        bail!("ContentVec session inference requires the `ort` feature; this build supports native TensorRT only")
    }

    #[cfg(feature = "ort")]
    pub(super) fn extract_with_session_run(
        &mut self,
        audio_16k: &[f32],
        input_shape: &[usize; 2],
    ) -> Result<FeatureTensor> {
        // Borrow the worker-owned input slice for synchronous ORT runs. Using
        // Tensor::from_array here would allocate and copy the full waveform on
        // every realtime chunk.
        let input = TensorRef::from_array_view((*input_shape, audio_16k))?;
        let run_start = Instant::now();
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])?;
        debug!(
            "embedder session.run backend={} input={} shape={} elapsed_us={}",
            self.provider.label(),
            self.input_name,
            format_usize_shape(input_shape),
            run_start.elapsed().as_micros()
        );
        let value = outputs
            .get(self.output_name.as_str())
            .ok_or_else(|| anyhow!("embedder output '{}' not found", self.output_name))?;
        let (shape, data) = value.try_extract_tensor::<f32>()?;
        if shape.len() != 3 {
            bail!("embedder output must be rank-3 [1, frames, channels], got {shape}");
        }
        Ok(FeatureTensor {
            data: data.to_vec(),
            shape: shape.to_vec(),
        })
    }

    #[cfg(feature = "ort")]
    pub(super) fn extract_with_binding(&mut self, audio_16k: &[f32]) -> Result<FeatureTensor> {
        let binding = self
            .tensor_rt_binding
            .as_mut()
            .ok_or_else(|| anyhow!("TensorRT ContentVec IoBinding is not initialized"))?;
        match binding {
            HubertTensorRtBinding::Pinned(binding) => {
                copy_f32_tensor(&mut binding.audio, audio_16k, "audio")?;
                binding
                    .binding
                    .bind_input(self.input_name.as_str(), &binding.audio)
                    .with_context(|| {
                        format!(
                            "failed to bind TensorRT ContentVec input '{}'",
                            self.input_name
                        )
                    })?;
                let run_start = Instant::now();
                let _outputs = self.session.run_binding(&binding.binding)?;
                binding
                    .binding
                    .synchronize_outputs()
                    .context("failed to synchronize TensorRT ContentVec bound output")?;
                debug!(
                    "embedder session.run_binding backend={} cuda_graph=false device_io=false input={} shape={} output={} output_shape={} elapsed_us={}",
                    self.provider.label(),
                    self.input_name,
                    format_usize_shape(&binding.input_shape),
                    self.output_name,
                    format_usize_shape(&binding.output_shape),
                    run_start.elapsed().as_micros()
                );
                let (shape, data) = binding.output.try_extract_tensor::<f32>()?;
                if shape.len() != 3 {
                    bail!("embedder output must be rank-3 [1, frames, channels], got {shape}");
                }
                let actual_shape = i64_shape_to_usize(shape, "contentvec output")?;
                if actual_shape != binding.output_shape {
                    bail!(
                        "TensorRT ContentVec bound output shape changed from {} to {}",
                        format_usize_shape(&binding.output_shape),
                        format_usize_shape(&actual_shape)
                    );
                }
                Ok(FeatureTensor {
                    data: data.to_vec(),
                    shape: shape.to_vec(),
                })
            }
            HubertTensorRtBinding::CudaGraph(binding) => {
                let h2d_us = binding
                    .copy_audio_to_device_if_owned(audio_16k, self.input_name.as_str())?
                    .unwrap_or(0);
                let run_start = Instant::now();
                let _outputs = self.session.run_binding(&binding.binding)?;
                let run_us = run_start.elapsed().as_micros();
                let d2h_start = Instant::now();
                copy_f32_tensor_to_host(
                    &binding.device_output,
                    &mut binding.host_output,
                    self.output_name.as_str(),
                )?;
                let d2h_us = d2h_start.elapsed().as_micros();
                debug!(
                    "embedder session.run_binding(device_io=true) backend={} cuda_graph={} shared_waveform_input={} input={} shape={} output={} output_shape={} h2d_us={} run_us={} d2h_us={} elapsed_us={}",
                    self.provider.label(),
                    self.tensor_rt_run_mode.cuda_graph(),
                    binding.shared_waveform_input,
                    self.input_name,
                    format_usize_shape(&binding.input_shape),
                    self.output_name,
                    format_usize_shape(&binding.output_shape),
                    h2d_us,
                    run_us,
                    d2h_us,
                    h2d_us + run_us + d2h_us
                );
                let (shape, data) = binding.host_output.try_extract_tensor::<f32>()?;
                if shape.len() != 3 {
                    bail!("embedder output must be rank-3 [1, frames, channels], got {shape}");
                }
                let actual_shape = i64_shape_to_usize(shape, "contentvec output")?;
                if actual_shape != binding.output_shape {
                    bail!(
                        "TensorRT ContentVec bound output shape changed from {} to {}",
                        format_usize_shape(&binding.output_shape),
                        format_usize_shape(&actual_shape)
                    );
                }
                Ok(FeatureTensor {
                    data: data.to_vec(),
                    shape: shape.to_vec(),
                })
            }
        }
    }
}

pub(super) struct RmvpePitchSession {
    #[cfg(feature = "ort")]
    pub(super) session: Session,
    pub(super) provider: Provider,
    pub(super) tensor_rt_profile: Option<TensorRtSessionProfile>,
    pub(super) tensor_rt_run_mode: TensorRtRunMode,
    #[cfg(feature = "ort")]
    pub(super) tensor_rt_binding: Option<RmvpeTensorRtBinding>,
    native: Option<NativeRmvpeEngine>,
}

impl RmvpePitchSession {
    pub(super) fn load(
        path: &Path,
        provider: Provider,
        tensor_rt_profile: Option<TensorRtSessionProfile>,
        tensor_rt_run_mode: TensorRtRunMode,
        tensor_rt_session_purpose: TensorRtSessionPurpose,
    ) -> Result<Self> {
        let io = read_model_io(path)?;
        io.require_inputs(&["waveform", "threshold"])?;
        io.require_output("pitchf")?;
        let native = if provider.is_tensorrt() {
            let profile = tensor_rt_profile
                .as_ref()
                .ok_or_else(|| anyhow!("native TensorRT RMVPE requires a fixed-shape profile"))?;
            Some(NativeRmvpeEngine::load(path, profile)?)
        } else {
            None
        };
        #[cfg(feature = "ort")]
        let session = {
            let session_provider = if provider.is_tensorrt() {
                Provider::Cpu
            } else {
                provider
            };
            load_session(
                path,
                session_provider,
                ModelRole::Rmvpe,
                tensor_rt_profile.as_ref(),
                tensor_rt_run_mode,
                tensor_rt_session_purpose,
            )?
        };
        info!("loaded RMVPE f0 model: {}", path.display());
        Ok(Self {
            #[cfg(feature = "ort")]
            session,
            provider,
            tensor_rt_profile,
            tensor_rt_run_mode,
            #[cfg(feature = "ort")]
            tensor_rt_binding: None,
            native,
        })
    }

    pub(super) fn warmup_output_shape(
        &mut self,
        audio_16k_samples: usize,
        threshold: f32,
    ) -> Result<Vec<i64>> {
        let waveform_shape = [1usize, audio_16k_samples];
        validate_tensorrt_input_shape(
            self.provider,
            self.tensor_rt_profile.as_ref(),
            "waveform",
            &waveform_shape,
        )?;
        if let Some(native) = self.native.as_ref() {
            return Ok(native.warmup_output_shape());
        }
        #[cfg(feature = "ort")]
        {
            let waveform = Tensor::from_array((waveform_shape, vec![0.0f32; audio_16k_samples]))?;
            let threshold = Tensor::from_array(([1usize], vec![threshold]))?;
            let run_start = Instant::now();
            let outputs = self.session.run(ort::inputs![
                "waveform" => waveform,
                "threshold" => threshold,
            ])?;
            debug!(
                "rmvpe warmup session.run backend={} input=waveform shape={} elapsed_us={}",
                self.provider.label(),
                format_usize_shape(&waveform_shape),
                run_start.elapsed().as_micros()
            );
            let value = outputs
                .get("pitchf")
                .ok_or_else(|| anyhow!("RMVPE output 'pitchf' not found"))?;
            let (shape, _) = value.try_extract_tensor::<f32>()?;
            Ok(shape.to_vec())
        }
        #[cfg(not(feature = "ort"))]
        bail!("RMVPE warmup requires the `ort` feature; native TensorRT reports its own shape")
    }

    #[cfg(feature = "ort")]
    pub(super) fn enable_tensorrt_binding(
        &mut self,
        output_shape: &[i64],
        threshold: f32,
        shared_waveform: Option<&TensorRtSharedWaveform>,
    ) -> Result<()> {
        if !provider_uses_fixed_shape(self.provider) {
            return Ok(());
        }
        if self.native.is_some() {
            return Ok(());
        }
        let profile = self
            .tensor_rt_profile
            .as_ref()
            .ok_or_else(|| anyhow!("RMVPE IoBinding requires a fixed-shape profile"))?;
        let waveform_shape = profile.fixed_input_dims("waveform")?;
        let output_shape = i64_shape_to_usize(output_shape, "rmvpe output")?;
        let binding = match self.tensor_rt_run_mode {
            TensorRtRunMode::PinnedCpu => {
                RmvpeTensorRtBinding::Pinned(RmvpeTensorRtPinnedBinding::new(
                    &self.session,
                    waveform_shape,
                    &output_shape,
                    threshold,
                )?)
            }
            TensorRtRunMode::DeviceIo | TensorRtRunMode::CudaGraph => {
                let mut binding = RmvpeTensorRtGraphBinding::new(
                    &self.session,
                    waveform_shape,
                    &output_shape,
                    threshold,
                    shared_waveform,
                )?;
                binding.warmup_capture(
                    &mut self.session,
                    self.provider,
                    self.tensor_rt_run_mode.cuda_graph(),
                )?;
                RmvpeTensorRtBinding::CudaGraph(binding)
            }
        };
        let shared_waveform_input = match &binding {
            RmvpeTensorRtBinding::Pinned(_) => false,
            RmvpeTensorRtBinding::CudaGraph(binding) => binding.shared_waveform_input,
        };
        info!(
            "GPU IoBinding enabled backend={} model_role={} mode={} cuda_graph={} device_io={} input=waveform input_shape={} output=pitchf output_shape={} shared_waveform_input={} host_input_memory=CUDA_PINNED/CPUInput host_output_memory=CUDA_PINNED/CPUOutput bound_input_memory={} bound_output_memory={}",
            self.provider.label(),
            ModelRole::Rmvpe.label(),
            self.tensor_rt_run_mode.label(),
            self.tensor_rt_run_mode.cuda_graph(),
            self.tensor_rt_run_mode.device_io(),
            format_usize_shape(waveform_shape),
            format_usize_shape(&output_shape),
            shared_waveform_input,
            self.tensor_rt_run_mode.bound_input_memory(),
            self.tensor_rt_run_mode.bound_output_memory()
        );
        self.tensor_rt_binding = Some(binding);
        Ok(())
    }

    pub(super) fn extract(
        &mut self,
        audio_16k: &[f32],
        pitch_shift: f32,
        threshold: f32,
    ) -> Result<Vec<f32>> {
        let waveform_shape = [1usize, audio_16k.len()];
        validate_tensorrt_input_shape(
            self.provider,
            self.tensor_rt_profile.as_ref(),
            "waveform",
            &waveform_shape,
        )?;
        #[cfg(feature = "ort")]
        if self.tensor_rt_binding.is_some() {
            return self.extract_with_binding(audio_16k, pitch_shift, threshold);
        }
        if let Some(native) = self.native.as_mut() {
            return native.extract(audio_16k, pitch_shift, threshold);
        }
        #[cfg(feature = "ort")]
        {
            self.extract_with_session_run(audio_16k, pitch_shift, threshold, &waveform_shape)
        }
        #[cfg(not(feature = "ort"))]
        bail!("RMVPE session inference requires the `ort` feature; this build supports native TensorRT only")
    }

    #[cfg(feature = "ort")]
    pub(super) fn extract_with_session_run(
        &mut self,
        audio_16k: &[f32],
        pitch_shift: f32,
        threshold: f32,
        waveform_shape: &[usize; 2],
    ) -> Result<Vec<f32>> {
        let threshold_value = [threshold];
        let waveform = TensorRef::from_array_view((*waveform_shape, audio_16k))?;
        let threshold = TensorRef::from_array_view(([1usize], threshold_value.as_slice()))?;
        let run_start = Instant::now();
        let outputs = self.session.run(ort::inputs![
            "waveform" => waveform,
            "threshold" => threshold,
        ])?;
        debug!(
            "rmvpe session.run backend={} input=waveform shape={} elapsed_us={}",
            self.provider.label(),
            format_usize_shape(waveform_shape),
            run_start.elapsed().as_micros()
        );
        let value = outputs
            .get("pitchf")
            .ok_or_else(|| anyhow!("RMVPE output 'pitchf' not found"))?;
        let (_, data) = value.try_extract_tensor::<f32>()?;
        let factor = 2.0f32.powf(pitch_shift / 12.0);
        Ok(data.iter().map(|f0| f0 * factor).collect())
    }

    #[cfg(feature = "ort")]
    pub(super) fn extract_with_binding(
        &mut self,
        audio_16k: &[f32],
        pitch_shift: f32,
        threshold: f32,
    ) -> Result<Vec<f32>> {
        let binding = self
            .tensor_rt_binding
            .as_mut()
            .ok_or_else(|| anyhow!("TensorRT RMVPE IoBinding is not initialized"))?;
        match binding {
            RmvpeTensorRtBinding::Pinned(binding) => {
                copy_f32_tensor(&mut binding.waveform, audio_16k, "waveform")?;
                binding
                    .binding
                    .bind_input("waveform", &binding.waveform)
                    .context("failed to bind TensorRT RMVPE input 'waveform'")?;
                binding.bind_threshold_if_changed(threshold)?;
                let run_start = Instant::now();
                let _outputs = self.session.run_binding(&binding.binding)?;
                binding
                    .binding
                    .synchronize_outputs()
                    .context("failed to synchronize TensorRT RMVPE bound output")?;
                debug!(
                    "rmvpe session.run_binding backend={} cuda_graph=false device_io=false input=waveform shape={} output=pitchf output_shape={} elapsed_us={}",
                    self.provider.label(),
                    format_usize_shape(&binding.waveform_shape),
                    format_usize_shape(&binding.output_shape),
                    run_start.elapsed().as_micros()
                );
                let (shape, data) = binding.output.try_extract_tensor::<f32>()?;
                let actual_shape = i64_shape_to_usize(shape, "rmvpe output")?;
                if actual_shape != binding.output_shape {
                    bail!(
                        "TensorRT RMVPE bound output shape changed from {} to {}",
                        format_usize_shape(&binding.output_shape),
                        format_usize_shape(&actual_shape)
                    );
                }
                let factor = 2.0f32.powf(pitch_shift / 12.0);
                Ok(data.iter().map(|f0| f0 * factor).collect())
            }
            RmvpeTensorRtBinding::CudaGraph(binding) => {
                let h2d_start = Instant::now();
                let waveform_h2d_us = binding
                    .copy_waveform_to_device_if_owned(audio_16k)?
                    .unwrap_or(0);
                binding.copy_threshold_if_changed(threshold)?;
                let h2d_us = h2d_start.elapsed().as_micros();
                let run_start = Instant::now();
                let _outputs = self.session.run_binding(&binding.binding)?;
                let run_us = run_start.elapsed().as_micros();
                let d2h_start = Instant::now();
                copy_f32_tensor_to_host(
                    &binding.device_output,
                    &mut binding.host_output,
                    "pitchf",
                )?;
                let d2h_us = d2h_start.elapsed().as_micros();
                debug!(
                    "rmvpe session.run_binding(device_io=true) backend={} cuda_graph={} shared_waveform_input={} input=waveform shape={} output=pitchf output_shape={} waveform_h2d_us={} h2d_us={} run_us={} d2h_us={} elapsed_us={}",
                    self.provider.label(),
                    self.tensor_rt_run_mode.cuda_graph(),
                    binding.shared_waveform_input,
                    format_usize_shape(&binding.waveform_shape),
                    format_usize_shape(&binding.output_shape),
                    waveform_h2d_us,
                    h2d_us,
                    run_us,
                    d2h_us,
                    h2d_us + run_us + d2h_us
                );
                let (shape, data) = binding.host_output.try_extract_tensor::<f32>()?;
                let actual_shape = i64_shape_to_usize(shape, "rmvpe output")?;
                if actual_shape != binding.output_shape {
                    bail!(
                        "TensorRT RMVPE bound output shape changed from {} to {}",
                        format_usize_shape(&binding.output_shape),
                        format_usize_shape(&actual_shape)
                    );
                }
                let factor = 2.0f32.powf(pitch_shift / 12.0);
                Ok(data.iter().map(|f0| f0 * factor).collect())
            }
        }
    }
}

// CPU output binding is deliberately output-only: inputs still borrow the
// worker-owned buffers for each synchronous run, while the RVC "audio" tensor
// keeps stable preallocated storage across chunks with the same shapes.
#[cfg(feature = "ort")]
struct RvcCpuOutputBinding {
    binding: IoBinding,
    output: Tensor<f32>,
    output_shape: Vec<usize>,
    feats_shape: Vec<usize>,
    pitch_shape: Vec<usize>,
}

#[cfg(feature = "ort")]
impl RvcCpuOutputBinding {
    fn new(
        session: &Session,
        feats_shape: &[usize],
        pitch_shape: &[usize],
        output_shape: &[usize],
    ) -> Result<Self> {
        let allocator = Allocator::default();
        let mut output = Tensor::<f32>::new(&allocator, output_shape.to_vec())
            .context("failed to allocate CPU RVC output 'audio'")?;
        let mut binding = session
            .create_binding()
            .context("failed to create CPU RVC output IoBinding")?;
        bind_output_tensor(&mut binding, "audio", &mut output)
            .context("failed to bind CPU RVC output 'audio'")?;
        Ok(Self {
            binding,
            output,
            output_shape: output_shape.to_vec(),
            feats_shape: feats_shape.to_vec(),
            pitch_shape: pitch_shape.to_vec(),
        })
    }

    fn matches_input(&self, feats_shape: &[usize], pitch_shape: &[usize]) -> bool {
        self.feats_shape == feats_shape && self.pitch_shape == pitch_shape
    }
}

pub(super) struct RvcModelSession {
    #[cfg(feature = "ort")]
    pub(super) session: Option<Session>,
    pub(super) provider: Provider,
    pub(super) tensor_rt_profile: Option<TensorRtSessionProfile>,
    pub(super) tensor_rt_run_mode: TensorRtRunMode,
    #[cfg(feature = "ort")]
    pub(super) tensor_rt_binding: Option<RvcTensorRtBinding>,
    native_rvc: Option<NativeRvcEngine>,
    #[cfg(feature = "ort")]
    cpu_output_binding: Option<RvcCpuOutputBinding>,
    pub(super) expected_feat_channels: i64,
}

impl RvcModelSession {
    pub(super) fn load(
        path: &Path,
        provider: Provider,
        tensor_rt_profile: Option<TensorRtSessionProfile>,
        expected_feat_channels_override: Option<i64>,
        tensor_rt_run_mode: TensorRtRunMode,
        tensor_rt_session_purpose: TensorRtSessionPurpose,
    ) -> Result<Self> {
        if provider.is_tensorrt() {
            let profile = tensor_rt_profile.as_ref().ok_or_else(|| {
                anyhow!("native TensorRT RVC requires a fixed-shape TensorRT profile")
            })?;
            let feats_shape = profile.fixed_input_dims("feats")?;
            let channels = feats_shape
                .get(2)
                .copied()
                .ok_or_else(|| anyhow!("native TensorRT RVC feats profile must be rank-3"))?;
            let expected_feat_channels = expected_feat_channels_override
                .unwrap_or_else(|| i64::try_from(channels).unwrap_or(i64::MAX));
            let native_rvc = NativeRvcEngine::load(path, profile, channels)?;
            info!(
                "loaded native TensorRT RVC model={} frames={} channels={} session_purpose={}",
                path.display(),
                native_rvc.frames(),
                native_rvc.channels(),
                tensor_rt_session_purpose.label()
            );
            return Ok(Self {
                #[cfg(feature = "ort")]
                session: None,
                provider,
                tensor_rt_profile,
                tensor_rt_run_mode,
                #[cfg(feature = "ort")]
                tensor_rt_binding: None,
                native_rvc: Some(native_rvc),
                #[cfg(feature = "ort")]
                cpu_output_binding: None,
                expected_feat_channels,
            });
        }
        // CPU/CUDA only: validate via the provider-neutral reader, then load the
        // ORT session for inference. Unreachable in the TensorRT-only build.
        #[cfg(feature = "ort")]
        {
            let io = read_model_io(path)?;
            io.require_inputs(&["feats", "p_len", "pitch", "pitchf", "sid"])?;
            io.require_output("audio")?;
            let expected_feat_channels = match expected_feat_channels_override {
                Some(channels) => channels,
                None => io.expected_feat_channels()?,
            };
            io.validate_rvc_metadata()?;
            let session = load_session(
                path,
                provider,
                ModelRole::Rvc,
                tensor_rt_profile.as_ref(),
                tensor_rt_run_mode,
                tensor_rt_session_purpose,
            )?;
            info!("loaded RVC model: {}", path.display());
            Ok(Self {
                session: Some(session),
                provider,
                tensor_rt_profile,
                tensor_rt_run_mode,
                tensor_rt_binding: None,
                native_rvc: None,
                cpu_output_binding: None,
                expected_feat_channels,
            })
        }
        #[cfg(not(feature = "ort"))]
        bail!(
            "provider {} requires the `ort` feature; this build supports native TensorRT only",
            provider.label()
        )
    }

    pub(super) fn warmup_output_shape(
        &mut self,
        feature_len: usize,
        feature_channels: i64,
        speaker_id: i64,
    ) -> Result<Vec<i64>> {
        if let Some(native) = self.native_rvc.as_ref() {
            if native.frames() != feature_len {
                bail!(
                    "native TensorRT RVC engine frame count {} does not match runtime feature_len {}",
                    native.frames(),
                    feature_len
                );
            }
            if native.channels()
                != usize::try_from(feature_channels).context("invalid RVC channel count")?
            {
                bail!(
                    "native TensorRT RVC engine channel count {} does not match model channel count {}",
                    native.channels(),
                    feature_channels
                );
            }
            // The engine self-reports its fixed `audio` output length after
            // deserialize, so no warmup inference is needed to learn the shape.
            // `speaker_id` is consumed only by the ORT branch below.
            return Ok(vec![
                i64::try_from(native.output_len()).context("native RVC output length overflow")?
            ]);
        }
        #[cfg(not(feature = "ort"))]
        {
            let _ = (feature_len, feature_channels, speaker_id);
            bail!("RVC warmup requires the `ort` feature; native TensorRT reports its own shape")
        }
        #[cfg(feature = "ort")]
        {
            let feats_shape = vec![1i64, feature_len as i64, feature_channels];
            let feats_shape_usize = i64_shape_to_usize(&feats_shape, "feats")?;
            validate_tensorrt_input_shape(
                self.provider,
                self.tensor_rt_profile.as_ref(),
                "feats",
                &feats_shape_usize,
            )?;
            let pitch_shape = [1usize, feature_len];
            validate_tensorrt_input_shape(
                self.provider,
                self.tensor_rt_profile.as_ref(),
                "pitch",
                &pitch_shape,
            )?;
            validate_tensorrt_input_shape(
                self.provider,
                self.tensor_rt_profile.as_ref(),
                "pitchf",
                &pitch_shape,
            )?;
            let feats_len = feature_len
                .checked_mul(
                    usize::try_from(feature_channels).context("invalid RVC channel count")?,
                )
                .context("RVC warmup feats input length overflow")?;
            let feats = Tensor::from_array((feats_shape.clone(), vec![0.0f32; feats_len]))?;
            let p_len = Tensor::from_array(([1usize], vec![feature_len as i64]))?;
            let pitch = Tensor::from_array((pitch_shape, vec![1i64; feature_len]))?;
            let pitchf = Tensor::from_array((pitch_shape, vec![0.0f32; feature_len]))?;
            let sid = Tensor::from_array(([1usize], vec![speaker_id]))?;
            let run_start = Instant::now();
            let session = self
                .session
                .as_mut()
                .ok_or_else(|| anyhow!("RVC ORT session is not initialized"))?;
            let outputs = session.run(ort::inputs![
                "feats" => feats,
                "p_len" => p_len,
                "pitch" => pitch,
                "pitchf" => pitchf,
                "sid" => sid,
            ])?;
            debug!(
                "rvc warmup session.run backend={} feats_shape={} pitch_shape={} elapsed_us={}",
                self.provider.label(),
                format_usize_shape(&feats_shape_usize),
                format_usize_shape(&pitch_shape),
                run_start.elapsed().as_micros()
            );
            let value = outputs
                .get("audio")
                .ok_or_else(|| anyhow!("RVC output 'audio' not found"))?;
            let (shape, _) = value.try_extract_tensor::<f32>()?;
            Ok(shape.to_vec())
        }
    }

    #[cfg(feature = "ort")]
    pub(super) fn enable_tensorrt_binding(
        &mut self,
        output_shape: &[i64],
        speaker_id: i64,
    ) -> Result<()> {
        if self.native_rvc.is_some() {
            return Ok(());
        }
        if !provider_uses_fixed_shape(self.provider) {
            return Ok(());
        }
        let profile = self
            .tensor_rt_profile
            .as_ref()
            .ok_or_else(|| anyhow!("RVC IoBinding requires a fixed-shape profile"))?;
        let feats_shape = profile.fixed_input_dims("feats")?;
        let pitch_shape = profile.fixed_input_dims("pitch")?;
        let frame_len = pitch_shape
            .get(1)
            .copied()
            .ok_or_else(|| anyhow!("TensorRT RVC pitch profile must be rank-2"))?;
        let output_shape = i64_shape_to_usize(output_shape, "rvc output")?;
        let binding = match self.tensor_rt_run_mode {
            TensorRtRunMode::PinnedCpu => {
                RvcTensorRtBinding::Pinned(RvcTensorRtPinnedBinding::new(
                    self.session
                        .as_ref()
                        .ok_or_else(|| anyhow!("RVC ORT session is not initialized"))?,
                    feats_shape,
                    pitch_shape,
                    &output_shape,
                    frame_len as i64,
                    speaker_id,
                )?)
            }
            TensorRtRunMode::DeviceIo | TensorRtRunMode::CudaGraph => {
                let session = self
                    .session
                    .as_mut()
                    .ok_or_else(|| anyhow!("RVC ORT session is not initialized"))?;
                let mut binding = RvcTensorRtGraphBinding::new(
                    session,
                    feats_shape,
                    pitch_shape,
                    &output_shape,
                    frame_len as i64,
                    speaker_id,
                )?;
                binding.warmup_capture(
                    session,
                    self.provider,
                    self.tensor_rt_run_mode.cuda_graph(),
                )?;
                RvcTensorRtBinding::CudaGraph(binding)
            }
        };
        info!(
            "GPU IoBinding enabled backend={} model_role={} mode={} cuda_graph={} device_io={} inputs=feats:{},pitch:{},pitchf:{},p_len:1,sid:1 output=audio output_shape={} host_input_memory=CUDA_PINNED/CPUInput host_output_memory=CUDA_PINNED/CPUOutput bound_input_memory={} bound_output_memory={}",
            self.provider.label(),
            ModelRole::Rvc.label(),
            self.tensor_rt_run_mode.label(),
            self.tensor_rt_run_mode.cuda_graph(),
            self.tensor_rt_run_mode.device_io(),
            format_usize_shape(feats_shape),
            format_usize_shape(pitch_shape),
            format_usize_shape(pitch_shape),
            format_usize_shape(&output_shape),
            self.tensor_rt_run_mode.bound_input_memory(),
            self.tensor_rt_run_mode.bound_output_memory()
        );
        self.tensor_rt_binding = Some(binding);
        Ok(())
    }

    #[cfg(feature = "ort")]
    fn enable_cpu_output_binding(
        &mut self,
        feats_shape: &[usize],
        pitch_shape: &[usize],
        output_shape: &[usize],
    ) -> Result<()> {
        if self.provider != Provider::Cpu {
            return Ok(());
        }
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| anyhow!("RVC ORT session is not initialized"))?;
        let binding = RvcCpuOutputBinding::new(session, feats_shape, pitch_shape, output_shape)?;
        info!(
            "CPU output IoBinding enabled model_role={} inputs=feats:{},pitch:{},pitchf:{} output=audio output_shape={}",
            ModelRole::Rvc.label(),
            format_usize_shape(feats_shape),
            format_usize_shape(pitch_shape),
            format_usize_shape(pitch_shape),
            format_usize_shape(output_shape)
        );
        self.cpu_output_binding = Some(binding);
        Ok(())
    }

    pub(super) fn infer(
        &mut self,
        feats: &[f32],
        feats_shape: &[i64],
        frame_len: usize,
        pitch: &[i64],
        pitchf: &[f32],
        speaker_id: i64,
    ) -> Result<Vec<f32>> {
        let feats_shape_usize = i64_shape_to_usize(feats_shape, "feats")?;
        validate_tensorrt_input_shape(
            self.provider,
            self.tensor_rt_profile.as_ref(),
            "feats",
            &feats_shape_usize,
        )?;
        let pitch_shape = [1usize, frame_len];
        validate_tensorrt_input_shape(
            self.provider,
            self.tensor_rt_profile.as_ref(),
            "pitch",
            &pitch_shape,
        )?;
        validate_tensorrt_input_shape(
            self.provider,
            self.tensor_rt_profile.as_ref(),
            "pitchf",
            &pitch_shape,
        )?;
        if let Some(native) = self.native_rvc.as_mut() {
            return native.infer(feats, pitch, pitchf, speaker_id);
        }
        #[cfg(feature = "ort")]
        {
            if self.tensor_rt_binding.is_some() {
                return self.infer_with_binding(feats, frame_len, pitch, pitchf, speaker_id);
            }
            if self.provider == Provider::Cpu
                && self
                    .cpu_output_binding
                    .as_ref()
                    .is_some_and(|binding| binding.matches_input(&feats_shape_usize, &pitch_shape))
            {
                return self.infer_with_cpu_output_binding(
                    feats,
                    feats_shape,
                    &feats_shape_usize,
                    frame_len,
                    pitch,
                    pitchf,
                    speaker_id,
                    &pitch_shape,
                );
            }
            self.infer_with_session_run(
                feats,
                feats_shape,
                &feats_shape_usize,
                frame_len,
                pitch,
                pitchf,
                speaker_id,
                &pitch_shape,
            )
        }
        #[cfg(not(feature = "ort"))]
        bail!("RVC session inference requires the `ort` feature; this build supports native TensorRT only")
    }

    // Keep the RVC tensor inputs explicit here: collapsing them into an ad-hoc
    // struct would obscure the ONNX input contract this function validates.
    #[cfg(feature = "ort")]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn infer_with_session_run(
        &mut self,
        feats: &[f32],
        feats_shape: &[i64],
        feats_shape_usize: &[usize],
        frame_len: usize,
        pitch: &[i64],
        pitchf: &[f32],
        speaker_id: i64,
        pitch_shape: &[usize; 2],
    ) -> Result<Vec<f32>> {
        let p_len_value = [frame_len as i64];
        let sid_value = [speaker_id];
        let feats = TensorRef::from_array_view((feats_shape, feats))?;
        let p_len = TensorRef::from_array_view(([1usize], p_len_value.as_slice()))?;
        let pitch = TensorRef::from_array_view((*pitch_shape, pitch))?;
        let pitchf = TensorRef::from_array_view((*pitch_shape, pitchf))?;
        let sid = TensorRef::from_array_view(([1usize], sid_value.as_slice()))?;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("RVC ORT session is not initialized"))?;
        let (output_shape, output_data) = {
            let run_start = Instant::now();
            let outputs = session.run(ort::inputs![
                "feats" => feats,
                "p_len" => p_len,
                "pitch" => pitch,
                "pitchf" => pitchf,
                "sid" => sid,
            ])?;
            debug!(
                "rvc session.run backend={} feats_shape={} pitch_shape={} elapsed_us={}",
                self.provider.label(),
                format_usize_shape(feats_shape_usize),
                format_usize_shape(pitch_shape),
                run_start.elapsed().as_micros()
            );
            let value = outputs
                .get("audio")
                .ok_or_else(|| anyhow!("RVC output 'audio' not found"))?;
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            let output_shape = i64_shape_to_usize(shape, "rvc output")?;
            (output_shape, data.to_vec())
        };
        self.enable_cpu_output_binding(feats_shape_usize, pitch_shape, &output_shape)?;
        Ok(output_data)
    }

    // Keep the RVC tensor inputs explicit here: collapsing them into an ad-hoc
    // struct would obscure the ONNX input contract this function validates.
    #[cfg(feature = "ort")]
    #[allow(clippy::too_many_arguments)]
    fn infer_with_cpu_output_binding(
        &mut self,
        feats: &[f32],
        feats_shape: &[i64],
        feats_shape_usize: &[usize],
        frame_len: usize,
        pitch: &[i64],
        pitchf: &[f32],
        speaker_id: i64,
        pitch_shape: &[usize; 2],
    ) -> Result<Vec<f32>> {
        let p_len_value = [frame_len as i64];
        let sid_value = [speaker_id];
        let feats = TensorRef::from_array_view((feats_shape, feats))?;
        let p_len = TensorRef::from_array_view(([1usize], p_len_value.as_slice()))?;
        let pitch = TensorRef::from_array_view((*pitch_shape, pitch))?;
        let pitchf = TensorRef::from_array_view((*pitch_shape, pitchf))?;
        let sid = TensorRef::from_array_view(([1usize], sid_value.as_slice()))?;
        let provider = self.provider;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("RVC ORT session is not initialized"))?;
        let binding = self
            .cpu_output_binding
            .as_mut()
            .ok_or_else(|| anyhow!("CPU RVC output IoBinding is not initialized"))?;
        let run_start = Instant::now();
        // IoBinding retains bound input OrtValues after the run. These TensorRefs
        // borrow worker buffers, so clear inputs before returning on both success
        // and error paths; only the preallocated CPU output stays bound.
        let run_result: Result<()> = (|| {
            binding
                .binding
                .bind_input("feats", &feats)
                .context("failed to bind CPU RVC input 'feats'")?;
            binding
                .binding
                .bind_input("p_len", &p_len)
                .context("failed to bind CPU RVC input 'p_len'")?;
            binding
                .binding
                .bind_input("pitch", &pitch)
                .context("failed to bind CPU RVC input 'pitch'")?;
            binding
                .binding
                .bind_input("pitchf", &pitchf)
                .context("failed to bind CPU RVC input 'pitchf'")?;
            binding
                .binding
                .bind_input("sid", &sid)
                .context("failed to bind CPU RVC input 'sid'")?;
            let _outputs = session.run_binding(&binding.binding)?;
            binding
                .binding
                .synchronize_outputs()
                .context("failed to synchronize CPU RVC bound output")?;
            Ok(())
        })();
        binding.binding.clear_inputs();
        run_result?;
        debug!(
            "rvc session.run_binding backend={} cpu_output_binding=true feats_shape={} pitch_shape={} output_shape={} elapsed_us={}",
            provider.label(),
            format_usize_shape(feats_shape_usize),
            format_usize_shape(pitch_shape),
            format_usize_shape(&binding.output_shape),
            run_start.elapsed().as_micros()
        );
        let (shape, data) = binding.output.try_extract_tensor::<f32>()?;
        let actual_shape = i64_shape_to_usize(shape, "rvc output")?;
        if actual_shape != binding.output_shape {
            bail!(
                "CPU RVC bound output shape changed from {} to {}",
                format_usize_shape(&binding.output_shape),
                format_usize_shape(&actual_shape)
            );
        }
        Ok(data.to_vec())
    }

    #[cfg(feature = "ort")]
    pub(super) fn infer_with_binding(
        &mut self,
        feats: &[f32],
        frame_len: usize,
        pitch: &[i64],
        pitchf: &[f32],
        speaker_id: i64,
    ) -> Result<Vec<f32>> {
        let binding = self
            .tensor_rt_binding
            .as_mut()
            .ok_or_else(|| anyhow!("TensorRT RVC IoBinding is not initialized"))?;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("RVC ORT session is not initialized"))?;
        match binding {
            RvcTensorRtBinding::Pinned(binding) => {
                copy_f32_tensor(&mut binding.feats, feats, "feats")?;
                copy_i64_tensor(&mut binding.pitch, pitch, "pitch")?;
                copy_f32_tensor(&mut binding.pitchf, pitchf, "pitchf")?;
                binding.bind_fixed_scalars_if_changed(frame_len as i64, speaker_id)?;
                binding
                    .binding
                    .bind_input("feats", &binding.feats)
                    .context("failed to bind TensorRT RVC input 'feats'")?;
                binding
                    .binding
                    .bind_input("pitch", &binding.pitch)
                    .context("failed to bind TensorRT RVC input 'pitch'")?;
                binding
                    .binding
                    .bind_input("pitchf", &binding.pitchf)
                    .context("failed to bind TensorRT RVC input 'pitchf'")?;
                let run_start = Instant::now();
                let _outputs = session.run_binding(&binding.binding)?;
                binding
                    .binding
                    .synchronize_outputs()
                    .context("failed to synchronize TensorRT RVC bound output")?;
                debug!(
                    "rvc session.run_binding backend={} cuda_graph=false device_io=false feats_shape={} pitch_shape={} output_shape={} elapsed_us={}",
                    self.provider.label(),
                    format_usize_shape(&binding.feats_shape),
                    format_usize_shape(&binding.pitch_shape),
                    format_usize_shape(&binding.output_shape),
                    run_start.elapsed().as_micros()
                );
                let (shape, data) = binding.output.try_extract_tensor::<f32>()?;
                let actual_shape = i64_shape_to_usize(shape, "rvc output")?;
                if actual_shape != binding.output_shape {
                    bail!(
                        "TensorRT RVC bound output shape changed from {} to {}",
                        format_usize_shape(&binding.output_shape),
                        format_usize_shape(&actual_shape)
                    );
                }
                Ok(data.to_vec())
            }
            RvcTensorRtBinding::CudaGraph(binding) => {
                let h2d_start = Instant::now();
                copy_f32_tensor(&mut binding.host_feats, feats, "feats")?;
                copy_i64_tensor(&mut binding.host_pitch, pitch, "pitch")?;
                copy_f32_tensor(&mut binding.host_pitchf, pitchf, "pitchf")?;
                copy_f32_tensor_to_device(&binding.host_feats, &mut binding.device_feats, "feats")?;
                copy_i64_tensor_to_device(&binding.host_pitch, &mut binding.device_pitch, "pitch")?;
                copy_f32_tensor_to_device(
                    &binding.host_pitchf,
                    &mut binding.device_pitchf,
                    "pitchf",
                )?;
                binding.copy_fixed_scalars_if_changed(frame_len as i64, speaker_id)?;
                let h2d_us = h2d_start.elapsed().as_micros();
                let run_start = Instant::now();
                let _outputs = session.run_binding(&binding.binding)?;
                let run_us = run_start.elapsed().as_micros();
                let d2h_start = Instant::now();
                copy_f32_tensor_to_host(&binding.device_output, &mut binding.host_output, "audio")?;
                let d2h_us = d2h_start.elapsed().as_micros();
                debug!(
                    "rvc session.run_binding(device_io=true) backend={} cuda_graph={} feats_shape={} pitch_shape={} output_shape={} h2d_us={} run_us={} d2h_us={} elapsed_us={}",
                    self.provider.label(),
                    self.tensor_rt_run_mode.cuda_graph(),
                    format_usize_shape(&binding.feats_shape),
                    format_usize_shape(&binding.pitch_shape),
                    format_usize_shape(&binding.output_shape),
                    h2d_us,
                    run_us,
                    d2h_us,
                    h2d_us + run_us + d2h_us
                );
                let (shape, data) = binding.host_output.try_extract_tensor::<f32>()?;
                let actual_shape = i64_shape_to_usize(shape, "rvc output")?;
                if actual_shape != binding.output_shape {
                    bail!(
                        "TensorRT RVC bound output shape changed from {} to {}",
                        format_usize_shape(&binding.output_shape),
                        format_usize_shape(&actual_shape)
                    );
                }
                Ok(data.to_vec())
            }
        }
    }
}

#[cfg(feature = "ort")]
#[cfg(all(windows, feature = "windowsml"))]
fn windows_ml_catalog_ep_for_provider(
    provider: Provider,
) -> Option<crate::windows_ml::CatalogExecutionProvider> {
    match provider {
        Provider::WindowsMlNvTensorRtRtx => {
            Some(crate::windows_ml::CatalogExecutionProvider::NvTensorRtRtx)
        }
        Provider::WindowsMlQnn => Some(crate::windows_ml::CatalogExecutionProvider::Qnn),
        Provider::WindowsMlOpenVino => Some(crate::windows_ml::CatalogExecutionProvider::OpenVino),
        Provider::WindowsMlMiGraphX => Some(crate::windows_ml::CatalogExecutionProvider::MiGraphX),
        Provider::WindowsMlVitisAi => Some(crate::windows_ml::CatalogExecutionProvider::VitisAi),
        _ => None,
    }
}

#[cfg(feature = "ort")]
#[cfg(all(windows, feature = "windowsml"))]
fn windows_ml_catalog_ep_dispatch(
    catalog_ep: crate::windows_ml::CatalogExecutionProvider,
) -> ep::ExecutionProviderDispatch {
    match catalog_ep {
        crate::windows_ml::CatalogExecutionProvider::NvTensorRtRtx => ep::NVRTX::default().build(),
        crate::windows_ml::CatalogExecutionProvider::Qnn => ep::QNN::default().build(),
        crate::windows_ml::CatalogExecutionProvider::OpenVino => ep::OpenVINO::default().build(),
        crate::windows_ml::CatalogExecutionProvider::MiGraphX => ep::MIGraphX::default().build(),
        crate::windows_ml::CatalogExecutionProvider::VitisAi => ep::Vitis::default().build(),
    }
}

#[cfg(feature = "ort")]
#[cfg(all(windows, feature = "windowsml"))]
fn with_windows_ml_catalog_ep(
    builder: ort::session::builder::SessionBuilder,
    catalog_ep: crate::windows_ml::CatalogExecutionProvider,
    path: &Path,
) -> Result<ort::session::builder::SessionBuilder> {
    match catalog_ep {
        crate::windows_ml::CatalogExecutionProvider::NvTensorRtRtx => {
            info!(
                "using Windows ML catalog EP NvTensorRtRtx for {}",
                path.display()
            );
            builder
                .with_execution_providers([
                    windows_ml_catalog_ep_dispatch(catalog_ep).error_on_failure()
                ])
                .map_err(|err| anyhow!("failed to register Windows ML NvTensorRtRtx EP: {err}"))
        }
        crate::windows_ml::CatalogExecutionProvider::Qnn => {
            info!("using Windows ML catalog EP QNN for {}", path.display());
            builder
                .with_execution_providers([
                    windows_ml_catalog_ep_dispatch(catalog_ep).error_on_failure()
                ])
                .map_err(|err| anyhow!("failed to register Windows ML QNN EP: {err}"))
        }
        crate::windows_ml::CatalogExecutionProvider::OpenVino => {
            info!(
                "using Windows ML catalog EP OpenVINO for {}",
                path.display()
            );
            builder
                .with_execution_providers([
                    windows_ml_catalog_ep_dispatch(catalog_ep).error_on_failure()
                ])
                .map_err(|err| anyhow!("failed to register Windows ML OpenVINO EP: {err}"))
        }
        crate::windows_ml::CatalogExecutionProvider::MiGraphX => {
            info!(
                "using Windows ML catalog EP MIGraphX for {}",
                path.display()
            );
            builder
                .with_execution_providers([
                    windows_ml_catalog_ep_dispatch(catalog_ep).error_on_failure()
                ])
                .map_err(|err| anyhow!("failed to register Windows ML MIGraphX EP: {err}"))
        }
        crate::windows_ml::CatalogExecutionProvider::VitisAi => {
            info!("using Windows ML catalog EP VitisAI for {}", path.display());
            builder
                .with_execution_providers([
                    windows_ml_catalog_ep_dispatch(catalog_ep).error_on_failure()
                ])
                .map_err(|err| anyhow!("failed to register Windows ML VitisAI EP: {err}"))
        }
    }
}

#[cfg(feature = "ort")]
pub(super) fn load_session(
    path: &Path,
    provider: Provider,
    role: ModelRole,
    _tensor_rt_profile: Option<&TensorRtSessionProfile>,
    tensor_rt_run_mode: TensorRtRunMode,
    tensor_rt_session_purpose: TensorRtSessionPurpose,
) -> Result<Session> {
    if provider.is_windows_ml() {
        #[cfg(not(all(windows, feature = "windowsml")))]
        {
            bail!(
                "provider {} is unavailable in this build; rebuild on Windows with the `windowsml` feature for {}",
                provider.label(),
                path.display()
            );
        }
        #[cfg(all(windows, feature = "windowsml"))]
        {
            crate::windows_ml::ensure_initialized()?;
        }
    }

    let mut builder = Session::builder()?
        .with_intra_threads(1)
        .map_err(|err| anyhow!(err.to_string()))?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|err| anyhow!(err.to_string()))?;
    match provider {
        Provider::Cuda => {
            #[cfg(not(feature = "cuda"))]
            {
                // The ONNX Runtime CUDA EP is compiled out of this build. Keep
                // `tensor_rt_run_mode` referenced so the no-cuda build matches
                // the cuda build's signature without an unused-variable warning.
                let _ = tensor_rt_run_mode;
                bail!(
                    "Provider::Cuda is unavailable in this build (compiled without the `cuda` feature); rebuild with `--features cuda` or select a CPU/TensorRT provider for {}",
                    path.display()
                );
            }
            #[cfg(feature = "cuda")]
            {
                info!(
                    "requesting ONNX Runtime CUDA execution provider device_id={} cuda_graph={} device_io={} run_mode={}",
                    TENSORRT_DEVICE_ID,
                    tensor_rt_run_mode.cuda_graph(),
                    tensor_rt_run_mode.device_io(),
                    tensor_rt_run_mode.label()
                );
                if tensor_rt_run_mode.cuda_graph() {
                    builder = builder.with_disable_cpu_fallback().map_err(|err| {
                        anyhow!("failed to disable CPU fallback for CUDA backend: {err}")
                    })?;
                }
                builder = builder
                    .with_execution_providers([ep::CUDA::default()
                        .with_device_id(TENSORRT_DEVICE_ID)
                        .with_cuda_graph(tensor_rt_run_mode.cuda_graph())
                        .build()
                        .error_on_failure()])
                    .map_err(|err| anyhow!("failed to register CUDA execution provider: {err}"))?;
            }
        }
        Provider::WindowsMl => {
            #[cfg(not(all(windows, feature = "windowsml")))]
            {
                bail!(
                    "provider {} is unavailable in this build; rebuild on Windows with the `windowsml` feature for {}",
                    provider.label(),
                    path.display()
                );
            }
            #[cfg(all(windows, feature = "windowsml"))]
            {
                // Auto Windows ML optimizes for "works with the platform runtime":
                // catalog EP if ready, then DirectML, then ORT's CPU fallback.
                // Explicit windowsml-* providers below intentionally fail
                // instead of silently changing the requested accelerator.
                match crate::windows_ml::try_register_best_catalog_ep()? {
                    Some(catalog_ep) => {
                        info!(
                            "using Windows ML catalog EP {} with DirectML/CPU fallback for {}",
                            catalog_ep.label(),
                            path.display()
                        );
                        builder = builder
                            .with_execution_providers([
                                windows_ml_catalog_ep_dispatch(catalog_ep),
                                ep::DirectML::default().build(),
                            ])
                            .map_err(|err| {
                                anyhow!(
                                    "failed to configure Windows ML catalog/DirectML fallback EPs: {err}"
                                )
                            })?;
                    }
                    None => {
                        info!(
                            "no ready Windows ML catalog EP found; using DirectML/CPU fallback for {}",
                            path.display()
                        );
                        builder = builder
                            .with_execution_providers([ep::DirectML::default().build()])
                            .map_err(|err| {
                                anyhow!(
                                    "failed to configure Windows ML DirectML/CPU fallback EP: {err}"
                                )
                            })?;
                    }
                }
            }
        }
        Provider::WindowsMlNvTensorRtRtx
        | Provider::WindowsMlOpenVino
        | Provider::WindowsMlQnn
        | Provider::WindowsMlMiGraphX
        | Provider::WindowsMlVitisAi => {
            #[cfg(not(all(windows, feature = "windowsml")))]
            {
                bail!(
                    "provider {} is unavailable in this build; rebuild on Windows with the `windowsml` feature for {}",
                    provider.label(),
                    path.display()
                );
            }
            #[cfg(all(windows, feature = "windowsml"))]
            {
                let catalog_ep = windows_ml_catalog_ep_for_provider(provider).ok_or_else(|| {
                    anyhow!(
                        "provider {} has no Windows ML catalog EP mapping for {}",
                        provider.label(),
                        path.display()
                    )
                })?;
                if !crate::windows_ml::try_register_catalog_ep(catalog_ep)? {
                    bail!(
                        "Windows ML catalog EP {} requested by provider {} is not present or not ready for {}; install/enable that EP with Windows ML tooling, or use provider windowsml for DirectML/CPU fallback",
                        catalog_ep.label(),
                        provider.label(),
                        path.display()
                    );
                }
                builder = with_windows_ml_catalog_ep(builder, catalog_ep, path)?;
            }
        }
        Provider::WindowsMlDirectMl => {
            #[cfg(not(all(windows, feature = "windowsml")))]
            {
                bail!(
                    "provider {} is unavailable in this build; rebuild on Windows with the `windowsml` feature for {}",
                    provider.label(),
                    path.display()
                );
            }
            #[cfg(all(windows, feature = "windowsml"))]
            {
                info!(
                    "using Windows ML DirectML execution provider via Windows App SDK Runtime for {}",
                    path.display()
                );
                builder = builder
                    .with_execution_providers([ep::DirectML::default().build().error_on_failure()])
                    .map_err(|err| {
                        anyhow!("failed to register Windows ML DirectML execution provider: {err}")
                    })?;
            }
        }
        Provider::TensorRt => {
            bail!(
                "Provider::TensorRt is native-only; load a CPU inspection session or a native TensorRT engine for {}",
                path.display()
            );
        }
        Provider::Cpu | Provider::WindowsMlCpu => {
            info!(
                "using {} execution provider intra_threads={} inter_threads={} arena=true mem_pattern=true flush_to_zero=true",
                if provider == Provider::WindowsMlCpu {
                    "Windows ML CPU"
                } else {
                    "ONNX Runtime CPU"
                },
                CPU_ONNX_INTRA_THREADS,
                CPU_ONNX_INTER_THREADS
            );
            // CPU inference still feeds a latency-sensitive pipeline. Keep
            // these as load-time session options; per-chunk tuning here would
            // add allocation/logging pressure near the realtime path.
            builder = builder
                .with_optimization_level(GraphOptimizationLevel::All)
                .map_err(|err| anyhow!("failed to enable CPU graph optimizations: {err}"))?
                .with_intra_threads(CPU_ONNX_INTRA_THREADS)
                .map_err(|err| anyhow!("failed to set CPU intra-op threads: {err}"))?
                .with_parallel_execution(true)
                .map_err(|err| anyhow!("failed to enable CPU parallel execution: {err}"))?
                .with_inter_threads(CPU_ONNX_INTER_THREADS)
                .map_err(|err| anyhow!("failed to set CPU inter-op threads: {err}"))?
                // Repeated realtime chunks are shape-stable after stream
                // padding, so memory pattern plus the CPU arena avoids churn
                // in ORT's internal allocators. Revisit if CPU runs become
                // truly variable-shape.
                .with_memory_pattern(true)
                .map_err(|err| anyhow!("failed to enable CPU memory pattern: {err}"))?
                .with_prepacking(true)
                .map_err(|err| anyhow!("failed to enable CPU prepacking: {err}"))?
                .with_flush_to_zero()
                .map_err(|err| anyhow!("failed to enable CPU flush-to-zero: {err}"))?
                .with_intra_op_spinning(true)
                .map_err(|err| anyhow!("failed to enable CPU intra-op spinning: {err}"))?
                .with_inter_op_spinning(true)
                .map_err(|err| anyhow!("failed to enable CPU inter-op spinning: {err}"))?
                .with_execution_providers([ep::CPU::default()
                    .with_arena_allocator(true)
                    .build()
                    .error_on_failure()])
                .map_err(|err| anyhow!("failed to register CPU execution provider: {err}"))?;
        }
    }
    if provider.is_tensorrt() || provider.is_cuda() {
        info!(
            "starting {} session commit for {} session_purpose={} cuda_graph={}",
            provider.label(),
            role.label(),
            tensor_rt_session_purpose.label(),
            tensor_rt_run_mode.cuda_graph()
        );
    }
    let session = builder
        .commit_from_file(path)
        .with_context(|| format!("failed to load ONNX model {}", path.display()))?;
    info!(
        "created ONNX Runtime session backend={} model_role={} session_purpose={} cuda_graph={} model={}",
        provider.label(),
        role.label(),
        tensor_rt_session_purpose.label(),
        provider_uses_fixed_shape(provider) && tensor_rt_run_mode.cuda_graph(),
        path.display()
    );
    Ok(session)
}

/// Format an ORT output value type for the CLI `inspect` command. The pipeline's
/// own structural checks live on `onnx_meta::ModelIo` and need no ORT.
#[cfg(feature = "ort")]
pub(super) fn describe_value_type(value_type: &ValueType) -> String {
    match value_type {
        ValueType::Tensor { ty, shape, .. } => format!("{ty:?} {shape}"),
        other => format!("{other:?}"),
    }
}
