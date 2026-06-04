# Third-party notices

This folder ships with the vc-rs distributables. Which components actually apply
depends on the package variant you have (named in `INSTALL.txt`). The matching
license texts are included here; the GPU / Windows App SDK license files are
copied from your local install (or the downloaded NuGet package) at packaging
time.

## Windows ML package

- **Microsoft Windows App SDK bootstrapper** — `Microsoft.WindowsAppRuntime.Bootstrap.dll`
  is bundled beside the binary. It is redistributed under the Microsoft Windows
  App SDK license terms — see [`WindowsAppSDK-LICENSE.txt`](WindowsAppSDK-LICENSE.txt).
- **ONNX Runtime + DirectML** — `onnxruntime.dll` and `DirectML.dll` are **not**
  bundled. They are provided at runtime by the installed **Windows App SDK
  Runtime 2.x**; vc-rs loads them dynamically (ORT `load-dynamic`). vc-rs uses
  the ONNX Runtime API, which is MIT-licensed — see
  [`onnxruntime.LICENSE.txt`](onnxruntime.LICENSE.txt).

## TensorRT package

The TensorRT build runs the GPU path through native TensorRT and contains **no
ONNX Runtime** (`onnxruntime.LICENSE.txt` does not apply to this package). It
ships the NVIDIA TensorRT runtime DLLs so it can run engines without a separate
TensorRT install.

| Component | DLLs | License |
|---|---|---|
| NVIDIA TensorRT | `nvinfer_<N>.dll`, `nvinfer_plugin_<N>.dll`, `nvonnxparser_<N>.dll`, `nvinfer_builder_resource_sm*_<N>.dll` | NVIDIA TensorRT license — see `TensorRT-LICENSE.txt` |
| NVIDIA CUDA Runtime | `cudart64_<M>.dll` | NVIDIA CUDA Toolkit EULA (redistributable runtime) — https://docs.nvidia.com/cuda/eula/index.html |

Notes:

- `TensorRT-LICENSE.txt` is copied from your local TensorRT install by
  `package-tensorrt.ps1`. If missing, obtain it from your TensorRT package.
- The NVIDIA runtime libraries are redistributable under the terms of those
  agreements. Review them before redistributing this build.
- An up-to-date NVIDIA GPU **driver** is still required on the end-user machine
  (it provides the CUDA driver, which is not redistributable and not bundled).

## Plugin licensing (both variants)

The `.vst3` plugin binary is under this project's MIT license (the VST3 SDK
bindings used via nice-plug are MIT-licensed). The bundled NVIDIA / Microsoft
DLLs are separate aggregated works under their own licenses listed above (mere
aggregation).
