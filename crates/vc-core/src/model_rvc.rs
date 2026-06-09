mod api;
mod cache;
mod f0_postprocess;
mod feature;
mod inspect;
mod native_tensorrt;
mod onnx_meta;
mod pipeline;
mod pitch;
mod sessions;
mod shape;
mod stream;
mod tensorrt;

/// GPU scheduling priority requested for native TensorRT inference streams.
///
/// CUDA stream priority is a scheduling hint for compute kernels. It does not
/// guarantee execution order and does not prioritize host/device transfers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuPriority {
    Normal,
    #[default]
    High,
}

pub use api::{ModelOutput, PassthroughModel, VoiceModel};
pub use cache::{
    clear_engine_cache, engine_cache_info, engine_cache_root, ClearedEngineCache, EngineCacheEntry,
    EngineCacheInfo, ENGINE_CACHE_DIR_ENV,
};
// Re-exported so the standalone front-ends can name the config when building
// `RvcPipelineConfig`; the processor itself stays private to the engine.
pub use f0_postprocess::F0PostprocessConfig;
pub use inspect::inspect_model;
pub use pipeline::{RvcPipeline, RvcPipelineConfig};

#[cfg(test)]
mod tests;
