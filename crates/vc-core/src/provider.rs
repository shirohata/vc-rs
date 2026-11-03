//! Inference backend selection shared by the CLI and plugin front-ends.
//!
//! One descriptor table ([`Provider::info`]) is the single source of truth for
//! every provider's canonical name and accepted aliases. `label()`, string
//! parsing ([`Provider::from_name`] / `FromStr`), and the `clap::ValueEnum`
//! impl are all derived from it, so the CLI, GUI, and VST3 accept exactly the
//! same spellings and adding a variant is a single table row plus the
//! compiler-enforced `info` arm.

use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    Cpu,
    Cuda,
    TensorRt,
    WindowsMl,
    WindowsMlCpu,
    WindowsMlDirectMl,
    WindowsMlNvTensorRtRtx,
    WindowsMlOpenVino,
    WindowsMlOpenVinoCpu,
    WindowsMlOpenVinoGpu,
    WindowsMlOpenVinoNpu,
    WindowsMlQnn,
    WindowsMlMiGraphX,
    WindowsMlVitisAi,
}

/// Canonical name plus alternate spellings for one [`Provider`]. `name` doubles
/// as the value returned by `label()` and persisted by the CLI/VST3/GUI.
struct ProviderInfo {
    name: &'static str,
    aliases: &'static [&'static str],
}

impl Provider {
    /// Every provider variant, in the order the front-ends list them. Also the
    /// set `clap::ValueEnum` and `from_name` iterate.
    pub const ALL: &'static [Provider] = &[
        Provider::Cpu,
        Provider::Cuda,
        Provider::TensorRt,
        Provider::WindowsMl,
        Provider::WindowsMlCpu,
        Provider::WindowsMlDirectMl,
        Provider::WindowsMlNvTensorRtRtx,
        Provider::WindowsMlOpenVino,
        Provider::WindowsMlOpenVinoCpu,
        Provider::WindowsMlOpenVinoGpu,
        Provider::WindowsMlOpenVinoNpu,
        Provider::WindowsMlQnn,
        Provider::WindowsMlMiGraphX,
        Provider::WindowsMlVitisAi,
    ];

    // The exhaustive `match` makes the compiler reject any new variant that
    // lacks a name/alias row, so the table can never silently fall out of sync
    // with the enum.
    fn info(self) -> ProviderInfo {
        match self {
            Provider::Cpu => ProviderInfo {
                name: "cpu",
                aliases: &[],
            },
            Provider::Cuda => ProviderInfo {
                name: "cuda",
                aliases: &[],
            },
            Provider::TensorRt => ProviderInfo {
                name: "tensorrt",
                aliases: &["trt", "tensor-rt"],
            },
            Provider::WindowsMl => ProviderInfo {
                name: "windowsml",
                aliases: &["windows-ml", "winml"],
            },
            Provider::WindowsMlCpu => ProviderInfo {
                name: "windowsml-cpu",
                aliases: &["windows-ml-cpu", "winml-cpu"],
            },
            Provider::WindowsMlDirectMl => ProviderInfo {
                name: "windowsml-directml",
                aliases: &[
                    "windows-ml-directml",
                    "winml-directml",
                    "windowsml-dml",
                    "winml-dml",
                ],
            },
            Provider::WindowsMlNvTensorRtRtx => ProviderInfo {
                name: "windowsml-nvtrtx",
                aliases: &[
                    "windows-ml-nvtrtx",
                    "winml-nvtrtx",
                    "windowsml-tensorrt",
                    "winml-tensorrt",
                ],
            },
            Provider::WindowsMlOpenVino => ProviderInfo {
                name: "windowsml-openvino",
                aliases: &["windows-ml-openvino", "winml-openvino"],
            },
            Provider::WindowsMlOpenVinoCpu => ProviderInfo {
                name: "windowsml-openvino-cpu",
                aliases: &["windows-ml-openvino-cpu", "winml-openvino-cpu"],
            },
            Provider::WindowsMlOpenVinoGpu => ProviderInfo {
                name: "windowsml-openvino-gpu",
                aliases: &["windows-ml-openvino-gpu", "winml-openvino-gpu"],
            },
            Provider::WindowsMlOpenVinoNpu => ProviderInfo {
                name: "windowsml-openvino-npu",
                aliases: &["windows-ml-openvino-npu", "winml-openvino-npu"],
            },
            Provider::WindowsMlQnn => ProviderInfo {
                name: "windowsml-qnn",
                aliases: &["windows-ml-qnn", "winml-qnn"],
            },
            Provider::WindowsMlMiGraphX => ProviderInfo {
                name: "windowsml-migraphx",
                aliases: &["windows-ml-migraphx", "winml-migraphx"],
            },
            Provider::WindowsMlVitisAi => ProviderInfo {
                name: "windowsml-vitisai",
                aliases: &["windows-ml-vitisai", "winml-vitisai"],
            },
        }
    }

    pub fn label(self) -> &'static str {
        self.info().name
    }

    /// English UI text (also used as a GUI translation key). Never persist or
    /// parse this text: `label()` remains the compatible CLI/settings token.
    /// These names describe the requested configuration, not operation placement.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA (NVIDIA GPU)",
            Self::TensorRt => "TensorRT (NVIDIA GPU)",
            Self::WindowsMl => "Windows ML (automatic)",
            Self::WindowsMlCpu => "CPU (Windows ML)",
            Self::WindowsMlDirectMl => "DirectML (GPU)",
            Self::WindowsMlNvTensorRtRtx => "TensorRT RTX (NVIDIA GPU)",
            Self::WindowsMlOpenVino => "OpenVINO",
            Self::WindowsMlOpenVinoCpu => "OpenVINO (CPU)",
            Self::WindowsMlOpenVinoGpu => "OpenVINO (GPU)",
            Self::WindowsMlOpenVinoNpu => "OpenVINO (NPU)",
            Self::WindowsMlQnn => "QNN",
            Self::WindowsMlMiGraphX => "MIGraphX",
            Self::WindowsMlVitisAi => "Vitis AI",
        }
    }

    /// Short UI explanation shared by GUI and VST3; no runtime policy is
    /// inferred from this text. In particular, an OpenVINO device restriction
    /// does not promise that every model or operation runs on that device.
    pub fn display_hint(self) -> Option<&'static str> {
        match self {
            Self::WindowsMl => Some(
                "Windows ML automatically chooses an available backend. Some processing may use the CPU.",
            ),
            Self::WindowsMlOpenVino => Some(
                "The device is unspecified in this saved setting; this is not automatic device selection. Some processing may use the CPU.",
            ),
            Self::WindowsMlCpu | Self::Cpu => None,
            _ if self.is_windows_ml() => Some(
                "Some processing may use the CPU even when a GPU or NPU is selected.",
            ),
            _ => None,
        }
    }

    /// Backend shown in a picker. Hardware-specific persisted names remain
    /// unchanged so old GUI settings, plugin states and CLI commands still work.
    pub fn backend(self) -> Self {
        match self {
            Self::WindowsMlOpenVinoCpu
            | Self::WindowsMlOpenVinoGpu
            | Self::WindowsMlOpenVinoNpu => Self::WindowsMlOpenVino,
            other => other,
        }
    }

    /// New UI choices. Keep the unrestricted provider parseable for old state
    /// and CLI compatibility, but do not offer it in the device picker.
    pub const OPENVINO_DEVICES: &'static [Self] = &[
        Self::WindowsMlOpenVinoCpu,
        Self::WindowsMlOpenVinoGpu,
        Self::WindowsMlOpenVinoNpu,
    ];

    /// The unrestricted choice is not OpenVINO's AUTO device mode.
    pub fn openvino_device_label(self) -> Option<&'static str> {
        match self {
            Self::WindowsMlOpenVino => Some("Unspecified (legacy setting)"),
            Self::WindowsMlOpenVinoCpu => Some("CPU"),
            Self::WindowsMlOpenVinoGpu => Some("GPU"),
            Self::WindowsMlOpenVinoNpu => Some("NPU"),
            _ => None,
        }
    }

    /// Parse a provider from its canonical name or any alias, case-insensitively
    /// and ignoring surrounding whitespace. Returns `None` for unknown spellings
    /// so callers choose whether that is an error or a fallback. This is the one
    /// parser the CLI (`clap`), GUI, and VST3 share.
    pub fn from_name(name: &str) -> Option<Provider> {
        let name = name.trim();
        Provider::ALL.iter().copied().find(|provider| {
            let info = provider.info();
            info.name.eq_ignore_ascii_case(name)
                || info
                    .aliases
                    .iter()
                    .any(|alias| alias.eq_ignore_ascii_case(name))
        })
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
            Provider::WindowsMl
                | Provider::WindowsMlCpu
                | Provider::WindowsMlDirectMl
                | Provider::WindowsMlNvTensorRtRtx
                | Provider::WindowsMlOpenVino
                | Provider::WindowsMlOpenVinoCpu
                | Provider::WindowsMlOpenVinoGpu
                | Provider::WindowsMlOpenVinoNpu
                | Provider::WindowsMlQnn
                | Provider::WindowsMlMiGraphX
                | Provider::WindowsMlVitisAi
        )
    }

    pub fn is_windows_ml_directml(self) -> bool {
        matches!(self, Provider::WindowsMl | Provider::WindowsMlDirectMl)
    }

    /// True for the windowsml catalog execution providers (TensorRT-RTX,
    /// OpenVINO, QNN, MIGraphX, VitisAI). These are gated on the device's
    /// Windows ML catalog at runtime rather than purely by build features, so
    /// they are excluded from the build-time base list and appended from the
    /// live catalog. See [`selectable_providers`].
    pub fn is_catalog_ep(self) -> bool {
        matches!(
            self,
            Provider::WindowsMlNvTensorRtRtx
                | Provider::WindowsMlOpenVino
                | Provider::WindowsMlOpenVinoCpu
                | Provider::WindowsMlOpenVinoGpu
                | Provider::WindowsMlOpenVinoNpu
                | Provider::WindowsMlQnn
                | Provider::WindowsMlMiGraphX
                | Provider::WindowsMlVitisAi
        )
    }

    /// True for backends the engine drives with a fixed-shape TensorRT/CUDA
    /// profile (native TensorRT and the ORT CUDA EP). The one place this
    /// capability is defined; `model_rvc` and the front-ends both read it here.
    pub fn uses_fixed_shape(self) -> bool {
        self.is_tensorrt() || self.is_cuda()
    }

    /// True when a front-end should offer the GPU device-id selector for this
    /// provider. Kept separate from [`uses_fixed_shape`] so the two can diverge
    /// (e.g. if the Windows ML TensorRT-RTX EP later gains a device selector)
    /// without silently changing the fixed-shape path.
    pub fn shows_gpu_device_selector(self) -> bool {
        self.is_cuda() || self.is_tensorrt()
    }

    /// Explicit OpenVINO hardware restriction. The legacy provider intentionally
    /// leaves the runtime's device list unchanged; it is not an alias for CPU.
    #[cfg(all(feature = "ort", any(test, all(windows, feature = "windowsml"))))]
    pub(crate) fn openvino_device_type(self) -> Option<ort::memory::DeviceType> {
        use ort::memory::DeviceType;
        match self {
            Self::WindowsMlOpenVinoCpu => Some(DeviceType::CPU),
            Self::WindowsMlOpenVinoGpu => Some(DeviceType::GPU),
            Self::WindowsMlOpenVinoNpu => Some(DeviceType::NPU),
            _ => None,
        }
    }

    /// True when this provider's backend is compiled into the current build.
    /// The one place the per-backend feature gates live, so `load_session` and
    /// the front-ends agree on what a build can actually run.
    pub fn available_in_build(self) -> bool {
        if self.is_windows_ml() {
            // Every windowsml* variant (including WindowsMlCpu) rides the Windows
            // App SDK Runtime, so they share one gate.
            return cfg!(all(windows, feature = "windowsml"));
        }
        match self {
            Provider::Cpu => cfg!(feature = "ort"),
            Provider::Cuda => cfg!(feature = "cuda"),
            Provider::TensorRt => cfg!(feature = "tensorrt"),
            // `is_windows_ml` handled every remaining variant above.
            _ => false,
        }
    }
}

/// The provider a fresh config / `--provider` default selects for this build —
/// the single source for the CLI, GUI, and VST3 defaults.
///
/// An unambiguous single-backend build defaults to that backend (a GPU package
/// defaults to its GPU provider); the dev build that enables several backends at
/// once, and the CPU-only build, both default to CPU.
pub fn default_provider() -> Provider {
    match (
        cfg!(feature = "windowsml"),
        cfg!(feature = "tensorrt"),
        cfg!(feature = "cuda"),
    ) {
        (true, false, false) => Provider::WindowsMl,
        (false, true, false) => Provider::TensorRt,
        (false, false, true) => Provider::Cuda,
        _ => Provider::Cpu,
    }
}

/// Providers a front-end should offer in its picker: every non-catalog backend
/// compiled into this build, followed by the windowsml catalog EPs the device's
/// Windows ML catalog actually lists (including not-yet-installed ones, which
/// download on first load). The catalog is enumerated once and cached, so this
/// is cheap to call per frame. The single source both the GUI and VST3 render.
pub fn selectable_providers() -> Vec<Provider> {
    let list: Vec<Provider> = Provider::ALL
        .iter()
        .copied()
        .filter(|provider| provider.available_in_build() && !provider.is_catalog_ep())
        .collect();
    // Only the windowsml build appends live catalog EPs; shadow with a mutable
    // binding there so the base-only builds keep `list` immutable (no unused-mut).
    #[cfg(all(windows, feature = "windowsml"))]
    let list = {
        let mut list = list;
        for &provider in crate::windows_ml::available_catalog_providers() {
            if !list.contains(&provider) {
                list.push(provider);
            }
        }
        list
    };
    list
}

/// Returned by `Provider::from_str` for an unrecognized spelling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownProvider(pub String);

impl fmt::Display for UnknownProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown provider: {}", self.0)
    }
}

impl std::error::Error for UnknownProvider {}

impl FromStr for Provider {
    type Err = UnknownProvider;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Provider::from_name(s).ok_or_else(|| UnknownProvider(s.to_string()))
    }
}

// Drive clap off the same table so `--provider` accepts exactly the names and
// aliases `from_name` does; no second alias list to keep in sync.
#[cfg(feature = "clap")]
impl clap::ValueEnum for Provider {
    fn value_variants<'a>() -> &'a [Self] {
        Provider::ALL
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        let info = self.info();
        let mut value = clap::builder::PossibleValue::new(info.name);
        for alias in info.aliases {
            value = value.alias(*alias);
        }
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_names_leave_cli_and_persisted_tokens_unchanged() {
        let expected = [
            ("cpu", "CPU"),
            ("cuda", "CUDA (NVIDIA GPU)"),
            ("tensorrt", "TensorRT (NVIDIA GPU)"),
            ("windowsml", "Windows ML (automatic)"),
            ("windowsml-cpu", "CPU (Windows ML)"),
            ("windowsml-directml", "DirectML (GPU)"),
            ("windowsml-nvtrtx", "TensorRT RTX (NVIDIA GPU)"),
            ("windowsml-openvino", "OpenVINO"),
            ("windowsml-openvino-cpu", "OpenVINO (CPU)"),
            ("windowsml-openvino-gpu", "OpenVINO (GPU)"),
            ("windowsml-openvino-npu", "OpenVINO (NPU)"),
            ("windowsml-qnn", "QNN"),
            ("windowsml-migraphx", "MIGraphX"),
            ("windowsml-vitisai", "Vitis AI"),
        ];
        assert_eq!(expected.len(), Provider::ALL.len());
        for (&provider, (token, display)) in Provider::ALL.iter().zip(expected) {
            assert_eq!(provider.display_name(), display);
            assert_eq!(provider.label(), token);
            assert_eq!(Provider::from_name(token), Some(provider));
        }
    }

    #[cfg(feature = "ort")]
    #[test]
    fn openvino_hardware_selection_is_explicit_and_round_trips() {
        use ort::memory::DeviceType;
        for (suffix, expected) in [
            ("cpu", DeviceType::CPU),
            ("gpu", DeviceType::GPU),
            ("npu", DeviceType::NPU),
        ] {
            let name = format!("windowsml-openvino-{suffix}");
            let provider: Provider = name.parse().unwrap();
            assert_eq!(provider.label(), name);
            assert_eq!(provider.openvino_device_type(), Some(expected));
            assert!(provider.is_catalog_ep());
            assert!(provider.is_windows_ml());
            assert!(!provider.shows_gpu_device_selector());
            #[cfg(all(windows, feature = "windowsml"))]
            assert_eq!(
                provider.catalog_ep(),
                Some(crate::windows_ml::CatalogExecutionProvider::OpenVino)
            );
        }
        assert_eq!(Provider::WindowsMlOpenVino.openvino_device_type(), None);
        assert_eq!(Provider::WindowsMlCpu.openvino_device_type(), None);
        assert!("windowsml-openvino-auto".parse::<Provider>().is_err());
    }

    #[test]
    fn all_lists_every_variant_once() {
        // Each variant's canonical name round-trips, and `ALL` has no dupes.
        let mut seen: Vec<Provider> = Vec::new();
        for &provider in Provider::ALL {
            assert!(
                !seen.contains(&provider),
                "duplicate in Provider::ALL: {provider:?}"
            );
            seen.push(provider);
            assert_eq!(Provider::from_name(provider.label()), Some(provider));
        }
    }

    #[test]
    fn parses_canonical_names_aliases_and_case() {
        assert_eq!(Provider::from_name("cpu"), Some(Provider::Cpu));
        assert_eq!(Provider::from_name("trt"), Some(Provider::TensorRt));
        assert_eq!(
            Provider::from_name("  Windows-ML  "),
            Some(Provider::WindowsMl)
        );
        assert_eq!(
            Provider::from_name("windowsml-tensorrt"),
            Some(Provider::WindowsMlNvTensorRtRtx)
        );
        // The catalog EPs the GUI parser used to omit now parse everywhere.
        assert_eq!(
            Provider::from_name("windowsml-openvino"),
            Some(Provider::WindowsMlOpenVino)
        );
        assert_eq!(Provider::from_name("nope"), None);
        assert!("nope".parse::<Provider>().is_err());
    }

    #[test]
    fn default_provider_is_available_in_this_build() {
        // Whatever the feature set, the default must be a backend this build can
        // actually run — otherwise a fresh config/`--provider` default fails.
        let default = default_provider();
        assert!(
            default.available_in_build(),
            "default provider {default:?} is not available in this build"
        );
    }

    // clap and `from_name` are driven from the same table, so `--provider` must
    // accept exactly the canonical names and aliases `from_name` does.
    #[cfg(feature = "clap")]
    #[test]
    fn clap_and_from_name_accept_the_same_spellings() {
        use clap::ValueEnum;
        for &provider in Provider::ALL {
            let info = provider.info();
            for name in std::iter::once(info.name).chain(info.aliases.iter().copied()) {
                assert_eq!(
                    <Provider as ValueEnum>::from_str(name, false).ok(),
                    Some(provider),
                    "clap rejected {name}"
                );
                assert_eq!(Provider::from_name(name), Some(provider));
            }
        }
    }
}
