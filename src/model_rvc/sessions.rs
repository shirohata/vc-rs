use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use ort::ep;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{Tensor, TensorRef, ValueType};
use tracing::{debug, info};

use crate::cli::Provider;

use super::feature::FeatureTensor;
use super::tensorrt::*;

pub(super) struct HubertEmbedderSession {
    pub(super) session: Session,
    pub(super) provider: Provider,
    pub(super) tensor_rt_profile: Option<TensorRtSessionProfile>,
    pub(super) tensor_rt_run_mode: TensorRtRunMode,
    pub(super) tensor_rt_binding: Option<HubertTensorRtBinding>,
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
        let session = load_session(
            path,
            provider,
            ModelRole::ContentVec,
            tensor_rt_profile.as_ref(),
            tensor_rt_run_mode,
            tensor_rt_session_purpose,
        )?;
        let input_name = single_input_name(&session)?;
        let output_name = select_embedder_output(&session, expected_channels, requested_output)?;
        info!(
            "loaded embedder: {} input={} output={}",
            path.display(),
            input_name,
            output_name
        );
        Ok(Self {
            session,
            provider,
            tensor_rt_profile,
            tensor_rt_run_mode,
            tensor_rt_binding: None,
            input_name,
            output_name,
        })
    }

    pub(super) fn enable_tensorrt_binding(&mut self, output_shape: &[i64]) -> Result<()> {
        if !provider_uses_fixed_shape(self.provider) {
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
        info!(
            "GPU IoBinding enabled backend={} model_role={} mode={} cuda_graph={} device_io={} input={} input_shape={} output={} output_shape={} host_input_memory=CUDA_PINNED/CPUInput host_output_memory=CUDA_PINNED/CPUOutput bound_input_memory={} bound_output_memory={}",
            self.provider.label(),
            ModelRole::ContentVec.label(),
            self.tensor_rt_run_mode.label(),
            self.tensor_rt_run_mode.cuda_graph(),
            self.tensor_rt_run_mode.device_io(),
            self.input_name,
            format_usize_shape(input_shape),
            self.output_name,
            format_usize_shape(&output_shape),
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
        if self.tensor_rt_binding.is_some() {
            return self.extract_with_binding(audio_16k);
        }
        self.extract_with_session_run(audio_16k, &input_shape)
    }

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
                let h2d_start = Instant::now();
                copy_f32_tensor(&mut binding.host_audio, audio_16k, "audio")?;
                copy_f32_tensor_to_device(&binding.host_audio, &mut binding.device_audio, "audio")?;
                let h2d_us = h2d_start.elapsed().as_micros();
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
                    "embedder session.run_binding(device_io=true) backend={} cuda_graph={} input={} shape={} output={} output_shape={} h2d_us={} run_us={} d2h_us={} elapsed_us={}",
                    self.provider.label(),
                    self.tensor_rt_run_mode.cuda_graph(),
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
    pub(super) session: Session,
    pub(super) provider: Provider,
    pub(super) tensor_rt_profile: Option<TensorRtSessionProfile>,
    pub(super) tensor_rt_run_mode: TensorRtRunMode,
    pub(super) tensor_rt_binding: Option<RmvpeTensorRtBinding>,
}

impl RmvpePitchSession {
    pub(super) fn load(
        path: &Path,
        provider: Provider,
        tensor_rt_profile: Option<TensorRtSessionProfile>,
        tensor_rt_run_mode: TensorRtRunMode,
        tensor_rt_session_purpose: TensorRtSessionPurpose,
    ) -> Result<Self> {
        let session = load_session(
            path,
            provider,
            ModelRole::Rmvpe,
            tensor_rt_profile.as_ref(),
            tensor_rt_run_mode,
            tensor_rt_session_purpose,
        )?;
        require_inputs(&session, &["waveform", "threshold"])?;
        require_output(&session, "pitchf")?;
        info!("loaded RMVPE f0 model: {}", path.display());
        Ok(Self {
            session,
            provider,
            tensor_rt_profile,
            tensor_rt_run_mode,
            tensor_rt_binding: None,
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

    pub(super) fn enable_tensorrt_binding(
        &mut self,
        output_shape: &[i64],
        threshold: f32,
    ) -> Result<()> {
        if !provider_uses_fixed_shape(self.provider) {
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
                )?;
                binding.warmup_capture(
                    &mut self.session,
                    self.provider,
                    self.tensor_rt_run_mode.cuda_graph(),
                )?;
                RmvpeTensorRtBinding::CudaGraph(binding)
            }
        };
        info!(
            "GPU IoBinding enabled backend={} model_role={} mode={} cuda_graph={} device_io={} input=waveform input_shape={} output=pitchf output_shape={} host_input_memory=CUDA_PINNED/CPUInput host_output_memory=CUDA_PINNED/CPUOutput bound_input_memory={} bound_output_memory={}",
            self.provider.label(),
            ModelRole::Rmvpe.label(),
            self.tensor_rt_run_mode.label(),
            self.tensor_rt_run_mode.cuda_graph(),
            self.tensor_rt_run_mode.device_io(),
            format_usize_shape(waveform_shape),
            format_usize_shape(&output_shape),
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
        if self.tensor_rt_binding.is_some() {
            return self.extract_with_binding(audio_16k, pitch_shift, threshold);
        }
        self.extract_with_session_run(audio_16k, pitch_shift, threshold, &waveform_shape)
    }

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
                copy_f32_tensor(&mut binding.host_waveform, audio_16k, "waveform")?;
                copy_f32_tensor_to_device(
                    &binding.host_waveform,
                    &mut binding.device_waveform,
                    "waveform",
                )?;
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
                    "rmvpe session.run_binding(device_io=true) backend={} cuda_graph={} input=waveform shape={} output=pitchf output_shape={} h2d_us={} run_us={} d2h_us={} elapsed_us={}",
                    self.provider.label(),
                    self.tensor_rt_run_mode.cuda_graph(),
                    format_usize_shape(&binding.waveform_shape),
                    format_usize_shape(&binding.output_shape),
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

pub(super) struct RvcModelSession {
    pub(super) session: Session,
    pub(super) provider: Provider,
    pub(super) tensor_rt_profile: Option<TensorRtSessionProfile>,
    pub(super) tensor_rt_run_mode: TensorRtRunMode,
    pub(super) tensor_rt_binding: Option<RvcTensorRtBinding>,
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
        let session = load_session(
            path,
            provider,
            ModelRole::Rvc,
            tensor_rt_profile.as_ref(),
            tensor_rt_run_mode,
            tensor_rt_session_purpose,
        )?;
        require_inputs(&session, &["feats", "p_len", "pitch", "pitchf", "sid"])?;
        require_output(&session, "audio")?;
        let expected_feat_channels = match expected_feat_channels_override {
            Some(channels) => channels,
            None => expected_feat_channels(&session)?,
        };
        validate_rvc_metadata(&session)?;
        info!("loaded RVC model: {}", path.display());
        Ok(Self {
            session,
            provider,
            tensor_rt_profile,
            tensor_rt_run_mode,
            tensor_rt_binding: None,
            expected_feat_channels,
        })
    }

    pub(super) fn warmup_output_shape(
        &mut self,
        feature_len: usize,
        feature_channels: i64,
        speaker_id: i64,
    ) -> Result<Vec<i64>> {
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
            .checked_mul(usize::try_from(feature_channels).context("invalid RVC channel count")?)
            .context("RVC warmup feats input length overflow")?;
        let feats = Tensor::from_array((feats_shape.clone(), vec![0.0f32; feats_len]))?;
        let p_len = Tensor::from_array(([1usize], vec![feature_len as i64]))?;
        let pitch = Tensor::from_array((pitch_shape, vec![1i64; feature_len]))?;
        let pitchf = Tensor::from_array((pitch_shape, vec![0.0f32; feature_len]))?;
        let sid = Tensor::from_array(([1usize], vec![speaker_id]))?;
        let run_start = Instant::now();
        let outputs = self.session.run(ort::inputs![
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

    pub(super) fn enable_tensorrt_binding(
        &mut self,
        output_shape: &[i64],
        speaker_id: i64,
    ) -> Result<()> {
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
                    &self.session,
                    feats_shape,
                    pitch_shape,
                    &output_shape,
                    frame_len as i64,
                    speaker_id,
                )?)
            }
            TensorRtRunMode::DeviceIo | TensorRtRunMode::CudaGraph => {
                let mut binding = RvcTensorRtGraphBinding::new(
                    &self.session,
                    feats_shape,
                    pitch_shape,
                    &output_shape,
                    frame_len as i64,
                    speaker_id,
                )?;
                binding.warmup_capture(
                    &mut self.session,
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
        if self.tensor_rt_binding.is_some() {
            return self.infer_with_binding(feats, frame_len, pitch, pitchf, speaker_id);
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

    // Keep the RVC tensor inputs explicit here: collapsing them into an ad-hoc
    // struct would obscure the ONNX input contract this function validates.
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
        let run_start = Instant::now();
        let outputs = self.session.run(ort::inputs![
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
        let (_, data) = value.try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }

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
                let _outputs = self.session.run_binding(&binding.binding)?;
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
                let _outputs = self.session.run_binding(&binding.binding)?;
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

pub(super) fn load_session(
    path: &Path,
    provider: Provider,
    role: ModelRole,
    tensor_rt_profile: Option<&TensorRtSessionProfile>,
    tensor_rt_run_mode: TensorRtRunMode,
    tensor_rt_session_purpose: TensorRtSessionPurpose,
) -> Result<Session> {
    let mut builder = Session::builder()?
        .with_intra_threads(1)
        .map_err(|err| anyhow!(err.to_string()))?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|err| anyhow!(err.to_string()))?;
    match provider {
        Provider::Cuda => {
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
        Provider::TensorRt => {
            let tensor_rt_profile = tensor_rt_profile.ok_or_else(|| {
                anyhow!(
                    "TensorRT provider requires a profile for {}; use cpu/cuda for generic inspection",
                    path.display()
                )
            })?;
            if tensor_rt_profile.role != role {
                bail!(
                    "TensorRT profile role {} does not match requested model role {} for {}",
                    tensor_rt_profile.role.label(),
                    role.label(),
                    path.display()
                );
            }
            let profile_shapes = tensor_rt_profile.profile_shapes.as_str();
            let cache_root = tensor_rt_cache_root()?;
            let cache_dir = tensor_rt_profile.cache_dir_from_root(&cache_root)?;
            let timing_cache_dir = tensor_rt_timing_cache_dir_from_root(&cache_root);
            let model_cache_key = tensor_rt_profile.model_cache_key()?;
            let cache_has_entries = tensor_rt_cache_has_entries(&cache_dir);
            fs::create_dir_all(&cache_dir).with_context(|| {
                format!(
                    "failed to create TensorRT engine cache directory {}",
                    cache_dir.display()
                )
            })?;
            fs::create_dir_all(&timing_cache_dir).with_context(|| {
                format!(
                    "failed to create TensorRT timing cache directory {}",
                    timing_cache_dir.display()
                )
            })?;
            let cache_path = cache_dir.display().to_string();
            let timing_cache_path = timing_cache_dir.display().to_string();
            info!(
                "requesting ONNX Runtime TensorRT backend session_purpose={} model_role={} model={} provider_order=[TensorRT,CUDA] device_id={} fp16=true engine_cache=true timing_cache=true cuda_graph={} device_io={} run_mode={} cache_root={} model_cache_key={} cache_path={} timing_cache_path={} profile_shapes={} cache_status={}",
                tensor_rt_session_purpose.label(),
                role.label(),
                path.display(),
                TENSORRT_DEVICE_ID,
                tensor_rt_run_mode.cuda_graph(),
                tensor_rt_run_mode.device_io(),
                tensor_rt_run_mode.label(),
                cache_root.display(),
                model_cache_key,
                cache_path,
                timing_cache_path,
                profile_shapes,
                if cache_has_entries { "existing-files" } else { "empty-or-missing" }
            );
            if cache_has_entries {
                info!(
                    "TensorRT engine cache has existing files for {}; session commit may reuse cached engines",
                    role.label()
                );
            } else {
                info!(
                    "TensorRT engine cache is empty for {}; first session commit may build engines now",
                    role.label()
                );
            }
            let tensorrt = ep::TensorRT::default()
                .with_device_id(TENSORRT_DEVICE_ID)
                .with_fp16(true)
                .with_engine_cache(true)
                .with_engine_cache_path(cache_path)
                .with_timing_cache(true)
                .with_timing_cache_path(timing_cache_path)
                .with_force_timing_cache(false)
                .with_detailed_build_log(true)
                .with_cuda_graph(tensor_rt_run_mode.cuda_graph())
                .with_profile_min_shapes(profile_shapes)
                .with_profile_opt_shapes(profile_shapes)
                .with_profile_max_shapes(profile_shapes)
                .build()
                .error_on_failure();
            let cuda = ep::CUDA::default()
                .with_device_id(TENSORRT_DEVICE_ID)
                .build()
                .error_on_failure();
            builder = builder.with_disable_cpu_fallback().map_err(|err| {
                anyhow!("failed to disable CPU fallback for TensorRT backend: {err}")
            })?;
            builder = builder
                .with_execution_providers([tensorrt, cuda])
                .map_err(|err| {
                    anyhow!("failed to register TensorRT/CUDA execution providers: {err}")
                })?;
        }
        Provider::Cpu => {
            info!(
                "using ONNX Runtime CPU execution provider intra_threads={} inter_threads={} arena=true mem_pattern=true flush_to_zero=true",
                CPU_ONNX_INTRA_THREADS, CPU_ONNX_INTER_THREADS
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

pub(super) fn single_input_name(session: &Session) -> Result<String> {
    let inputs = session.inputs();
    if inputs.len() != 1 {
        bail!("expected a single input, got {}", inputs.len());
    }
    Ok(inputs[0].name().to_string())
}

pub(super) fn select_embedder_output(
    session: &Session,
    expected_channels: i64,
    requested_output: Option<&str>,
) -> Result<String> {
    let outputs = session.outputs();
    if let Some(name) = requested_output {
        let output = outputs
            .iter()
            .find(|output| output.name() == name)
            .ok_or_else(|| {
                let actual: Vec<&str> = outputs.iter().map(|output| output.name()).collect();
                anyhow!("requested embedder output '{name}' not found; outputs are {actual:?}")
            })?;
        validate_embedder_output_selection(
            "requested embedder output",
            name,
            output.dtype(),
            expected_channels,
        )?;
        return Ok(name.to_string());
    }

    let preferred_output = match expected_channels {
        768 => Some("unit12"),
        256 => Some("unit9"),
        _ => None,
    };
    if let Some(name) = preferred_output {
        for output in outputs {
            if output.name() == name && output_channels(output.dtype()) == Some(expected_channels) {
                return Ok(output.name().to_string());
            }
        }
    }
    for output in outputs {
        if output_channels(output.dtype()) == Some(expected_channels) {
            return Ok(output.name().to_string());
        }
    }
    if outputs.len() == 1 {
        let output = &outputs[0];
        validate_embedder_output_selection(
            "single embedder output",
            output.name(),
            output.dtype(),
            expected_channels,
        )?;
        return Ok(output.name().to_string());
    }
    let actual: Vec<String> = outputs
        .iter()
        .map(|output| format!("{}: {}", output.name(), describe_value_type(output.dtype())))
        .collect();
    bail!("no embedder output matches {expected_channels} channels; outputs are {actual:?}");
}

pub(super) fn validate_embedder_output_selection(
    label: &str,
    name: &str,
    value_type: &ValueType,
    expected_channels: i64,
) -> Result<()> {
    if !matches!(value_type, ValueType::Tensor { .. }) {
        bail!(
            "{label} '{name}' must be a tensor, got {}",
            describe_value_type(value_type)
        );
    }
    if let Some(channels) = output_channels(value_type) {
        if channels != expected_channels {
            bail!(
                "{label} '{name}' does not match expected {expected_channels} channels: {}",
                describe_value_type(value_type)
            );
        }
    }
    Ok(())
}

pub(super) fn output_channels(value_type: &ValueType) -> Option<i64> {
    match value_type {
        ValueType::Tensor { shape, .. } => shape.last().copied().filter(|channels| *channels > 0),
        _ => None,
    }
}

pub(super) fn expected_feat_channels(session: &Session) -> Result<i64> {
    let feats = session
        .inputs()
        .iter()
        .find(|input| input.name() == "feats")
        .ok_or_else(|| anyhow!("RVC model has no 'feats' input"))?;
    match feats.dtype() {
        ValueType::Tensor { shape, .. } => shape
            .last()
            .copied()
            .filter(|channels| *channels > 0)
            .ok_or_else(|| anyhow!("RVC 'feats' input does not expose a static channel count")),
        other => bail!("RVC 'feats' input must be tensor, got {other:?}"),
    }
}

pub(super) fn validate_rvc_metadata(session: &Session) -> Result<()> {
    let metadata = session.metadata().ok().and_then(|m| m.custom("metadata"));
    if let Some(metadata) = metadata {
        if !metadata.contains(r#""f0": 1"#) {
            bail!("RVC model metadata does not indicate f0=1: {metadata}");
        }
        info!("RVC metadata: {metadata}");
    }
    Ok(())
}

pub(super) fn require_inputs(session: &Session, names: &[&str]) -> Result<()> {
    let actual: HashSet<&str> = session.inputs().iter().map(|input| input.name()).collect();
    for name in names {
        if !actual.contains(name) {
            bail!("required input '{name}' not found; model inputs are {actual:?}");
        }
    }
    Ok(())
}

pub(super) fn require_output(session: &Session, name: &str) -> Result<()> {
    let exists = session.outputs().iter().any(|output| output.name() == name);
    if !exists {
        let actual: Vec<&str> = session
            .outputs()
            .iter()
            .map(|output| output.name())
            .collect();
        bail!("required output '{name}' not found; model outputs are {actual:?}");
    }
    Ok(())
}

pub(super) fn describe_value_type(value_type: &ValueType) -> String {
    match value_type {
        ValueType::Tensor { ty, shape, .. } => format!("{ty:?} {shape}"),
        other => format!("{other:?}"),
    }
}
