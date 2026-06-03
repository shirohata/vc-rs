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
    #[cfg_attr(
        feature = "clap",
        value(name = "tensorrt", alias = "trt", alias = "tensor-rt")
    )]
    TensorRt,
    #[cfg_attr(
        feature = "clap",
        value(name = "windowsml", alias = "windows-ml", alias = "winml")
    )]
    WindowsMl,
    #[cfg_attr(
        feature = "clap",
        value(name = "windowsml-cpu", alias = "windows-ml-cpu", alias = "winml-cpu")
    )]
    WindowsMlCpu,
    #[cfg_attr(
        feature = "clap",
        value(
            name = "windowsml-directml",
            alias = "windows-ml-directml",
            alias = "winml-directml",
            alias = "windowsml-dml",
            alias = "winml-dml"
        )
    )]
    WindowsMlDirectMl,
}

impl Provider {
    pub fn label(self) -> &'static str {
        match self {
            Provider::Cpu => "cpu",
            Provider::Cuda => "cuda",
            Provider::TensorRt => "tensorrt",
            Provider::WindowsMl => "windowsml",
            Provider::WindowsMlCpu => "windowsml-cpu",
            Provider::WindowsMlDirectMl => "windowsml-directml",
        }
    }

    pub fn is_tensorrt(self) -> bool {
        matches!(self, Provider::TensorRt)
    }

    pub fn is_cuda(self) -> bool {
        matches!(self, Provider::Cuda)
    }

    pub fn is_windows_ml(self) -> bool {
        matches!(
            self,
            Provider::WindowsMl | Provider::WindowsMlCpu | Provider::WindowsMlDirectMl
        )
    }

    pub fn is_windows_ml_directml(self) -> bool {
        matches!(self, Provider::WindowsMl | Provider::WindowsMlDirectMl)
    }
}
