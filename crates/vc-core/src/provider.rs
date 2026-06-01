//! Inference backend selection shared by the CLI and plugin front-ends.
//!
//! The `clap` feature derives `clap::ValueEnum` so the CLI can accept
//! `--provider` directly; plugins keep the feature off and construct the enum
//! programmatically.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum Provider {
    Cpu,
    Cuda,
    #[cfg_attr(feature = "clap", value(name = "tensorrt", alias = "trt", alias = "tensor-rt"))]
    TensorRt,
}

impl Provider {
    pub fn label(self) -> &'static str {
        match self {
            Provider::Cpu => "cpu",
            Provider::Cuda => "cuda",
            Provider::TensorRt => "tensorrt",
        }
    }

    pub fn is_tensorrt(self) -> bool {
        matches!(self, Provider::TensorRt)
    }

    pub fn is_cuda(self) -> bool {
        matches!(self, Provider::Cuda)
    }
}
