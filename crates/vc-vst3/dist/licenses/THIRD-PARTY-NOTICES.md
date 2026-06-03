# Third-party notices — bundled CUDA runtime DLLs

The GPU build of this plugin ships third-party DLLs so it can run the ONNX
Runtime CUDA execution provider without a separate CUDA/cuDNN install. Each
component keeps its own license; the texts are included in this folder.

| Component | DLLs | License |
|---|---|---|
| ONNX Runtime (CUDA EP) | `onnxruntime_providers_shared.dll`, `onnxruntime_providers_cuda.dll` | MIT — see [`onnxruntime.LICENSE.txt`](onnxruntime.LICENSE.txt) |
| NVIDIA CUDA Runtime | `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`, `cufft64_11.dll` | NVIDIA CUDA Toolkit EULA (redistributable runtime) — see `CUDA-EULA.txt` |
| NVIDIA cuDNN | `cudnn64_9.dll`, `cudnn_*64_9.dll` | NVIDIA cuDNN Software License Agreement — see `cuDNN-LICENSE.txt` |

Notes:

- The ONNX Runtime *core* is statically linked into the plugin binary (also
  MIT); no `onnxruntime.dll` is bundled separately.
- `CUDA-EULA.txt` and `cuDNN-LICENSE.txt` are copied from your local NVIDIA
  installations by `package-cuda.ps1`. If they are missing, obtain them from:
  - CUDA Toolkit EULA: https://docs.nvidia.com/cuda/eula/index.html
  - cuDNN SLA: https://docs.nvidia.com/deeplearning/cudnn/sla/index.html
- The NVIDIA CUDA runtime libraries and cuDNN are redistributable under the
  terms of those agreements. Review them before redistributing this build.
- An up-to-date NVIDIA GPU **driver** is still required on the end-user machine
  (it provides the CUDA driver, which is not redistributable and not bundled).
- Building the VST3 target links the Steinberg VST3 SDK bindings (GPLv3) via
  nice-plug, making the plugin binary GPLv3; the bundled NVIDIA/ONNX DLLs are
  separate aggregated works under their own licenses (mere aggregation).
