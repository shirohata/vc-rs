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
Standalone TensorRT packages also include GTCRN input denoising through native
TensorRT; its Rust DSP/FFT dependencies (`realfft`/`rustfft` and transitive
crates) are covered by the generated third-party license file in the archive.

| Component | DLLs | License |
|---|---|---|
| NVIDIA TensorRT | `nvinfer_<N>.dll`, `nvinfer_plugin_<N>.dll`, `nvonnxparser_<N>.dll`, `nvinfer_builder_resource_sm*_<N>.dll` | [NVIDIA TensorRT SDK License Agreement](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html) |
| NVIDIA CUDA Runtime | `cudart64_<M>.dll` | NVIDIA CUDA Toolkit EULA (redistributable runtime) — see `CUDA-EULA.txt` |

Notes:

- `CUDA-EULA.txt` is copied from the matching local CUDA install by
  `package-tensorrt.ps1`; packaging fails if it is unavailable.
- The TensorRT SDK agreement permits redistribution of runtime `.dll` files
  under its distribution requirements; review the linked terms before shipping.
- The NVIDIA runtime libraries are redistributable under the terms of those
  agreements. Review them before redistributing this build.
- An up-to-date NVIDIA GPU **driver** is still required on the end-user machine
  (it provides the CUDA driver, which is not redistributable and not bundled).

## Application and plugin licensing

The vc-rs executables and `.vst3` plugin are under this project's MIT license.
Bundled NVIDIA / Microsoft DLLs are separate aggregated works under their own
licenses listed above (mere aggregation).

The Rust crates statically linked into each shipped binary, and their full
license texts, are listed in the `THIRD-PARTY-LICENSES*.md` file(s) generated
directly into each distribution package.

## Ported source: rvc-onnx-web

The GUI's built-in `.pth` → `.onnx` model converter (the `vc-convert` crate)
is a Rust port of [rvc-onnx-web](https://github.com/visgotti/rvc-onnx-web) by
Joseph Viscardi (visgotti), MIT License. The full license text and porting
notes are in `crates/vc-convert/THIRD_PARTY.md` in the source tree. The port
is statically linked into `vc-gui.exe` only, so this section applies to the
packages that ship the GUI. (cargo-about only enumerates crate dependencies,
which is why this ported-source attribution is recorded here instead of the
generated `THIRD-PARTY-LICENSES*.md`.)


## Ported source: Applio decoders

The GUI converter includes MRF HiFi-GAN and RefineGAN decoder algorithms from
IAHispano/Applio, commit 5171e1dcf31a33d1afeb14ef7c6aac6dadae5dc1.
The following MIT notice applies to packages containing vc-gui.exe.

```
MIT License

Copyright (c) 2026 AI Hispano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Ported algorithm: torchaudio resampling

The converter's integer Kaiser sinc resampler follows torchaudio 2.7.1.
The following BSD-2-Clause notice applies to packages containing vc-gui.exe.

```
BSD 2-Clause License

Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
