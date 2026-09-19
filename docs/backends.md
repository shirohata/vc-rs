# Choosing an inference backend

> English | [日本語](backends_ja.md)

vc-rs offers several inference backends through the conversion pipeline shared by
the GUI, CLI, and VST3 plugin. This guide describes the current source. Check the
[release notes](https://github.com/shirohata/vc-rs/releases) for features in your download.

## Where to start

| Goal | Choice |
| --- | --- |
| Try the app or use a non-NVIDIA GPU | Windows ML package. The default `windowsml` selects automatically; use `windowsml-directml` to request DirectML |
| Use native TensorRT on NVIDIA | TensorRT package, `tensorrt`. Allow time and disk space for engine builds and caches |
| Evaluate MIGraphX on AMD | Windows ML package, `windowsml-migraphx` (experimental) |
| Evaluate OpenVINO on Intel | Windows ML package, `windowsml-openvino-cpu` / `windowsml-openvino-gpu` / `windowsml-openvino-npu` (experimental) |

Choose GUI + CLI for standalone use or VST3 for a DAW. Windows ML packages use
Windows App SDK Runtime 2.1 or later; TensorRT packages bundle their runtime.
Models are separate. See the [setup instructions](../README.md).

## Switching backends

In the GUI, choose a backend under **Backend Details**, then restart conversion
to reload the models. In VST3, choose the backend and press **Load / Reload**.
For CLI `run` / `wav`, pass the name from the table to `--provider`.
GUI/VST3 have separate OpenVINO backend and CPU/GPU/NPU selectors. If the EP is
missing, the UI offers a download action. Device detection does not establish model compatibility.
Windows ML catalog EPs appear in GUI/VST3 pickers according to this PC's catalog.

Use the CLI bundled with the Windows ML package to inspect the environment and EPs:

```powershell
.\vc-rs.exe doctor
.\vc-rs.exe windowsml-eps list
.\vc-rs.exe windowsml-eps install --help
```

An EP is an execution provider: a component ONNX Runtime uses to execute inference.
See the [CLI guide](cli.md) for installation and commands. There are no separate
MIGraphX or OpenVINO ZIPs.

## Native TensorRT

ContentVec, RMVPE, and the RVC generator run through native TensorRT rather than
the ONNX Runtime TensorRT EP. Fixed-shape engines are built and cached for reuse.
Beyond the NVIDIA driver, no separate CUDA or TensorRT installation is needed.

Expect a build on first use, after model/input-shape changes, or when the bundled
TensorRT version changes. Chunk size and extra context affect input shapes.
Choose your settings, wait for the build to finish, and evaluate dropouts and
audio quality in steady operation.

Run `.\vc-rs.exe engine-cache info` to inspect cache location and size. The ZIP,
extracted runtime, models, and engine caches consume space separately. See
[cache management](cli.md#engine-cache-management) for deletion and relocation.

The Windows ML package's `windowsml-nvtrtx` uses the separate **TensorRT-RTX EP**.
Its runtime, build process, model compatibility, performance, and caching differ.
For RVC sessions with streaming NSF inputs, the current implementation disables
the TensorRT-RTX runtime cache to avoid a teardown problem, so they rebuild on each load.

## Experimental MIGraphX / OpenVINO support

Both have Windows ML catalog registration and session/inference integration in
the shared pipeline. MIGraphX remains unverified on target hardware. OpenVINO has
model measurements and GUI listening checks on Intel Iris Xe / Core i7-1195G7,
not validation across all devices and models. See the configuration changes in the
[validation record](openvino-model-routing_ja.md). Installing an EP or creating a
session successfully does not prove successful acceleration.

- **MIGraphX:** requires a supported AMD GPU. Individual GPU selection and dedicated
  tuning/cache controls are not implemented; EP defaults apply. Dynamic shapes may
  trigger compilation during the first inference as well as loading.
- **OpenVINO:** CPU / GPU / NPU type selection is implemented. An unavailable type
  errors instead of silently selecting another type. Selecting an individual device
  among multiple devices of the same type is not supported. The legacy
  `windowsml-openvino` does not restrict the type. NPU operation especially needs
  model, operator, and shape compatibility validation.

The OpenVINO GPU path runs RMVPE on OpenVINO CPU to avoid audio-quality problems,
with fixed-input ContentVec and dynamic-shape RVC. It does not run every stage on
GPU. GPU precision is currently left to the EP rather than explicitly forced;
requesting accuracy alone does not establish effective FP32 execution.

`windowsml` Auto can retry catalog EP, DirectML, and CPU during model loading.
An explicit EP does not automatically retry a different EP. However, **ORT CPU
fallback** for unsupported operators is a separate mechanism and remains enabled
in sessions such as OpenVINO. Auto also does not recover from every failure during
first or subsequent inference. Device-selection logs do not establish GPU/NPU
placement of individual operators.

## Why changing Chunk ms requires a reload

Fixed shapes let inference optimize for known tensor dimensions. Native TensorRT
derives each model's input dimensions from the chunk and surrounding context, then
reuses engines built for those dimensions. The [fixed/dynamic shape comparison](tensorrt_performance_ja.md)
also found a performance benefit from fixed shapes. The dimensions are fixed, not
the contents of the audio.

For example, changing Chunk ms from 100 to 200 changes how much audio advances
per processing step. Models whose input dimensions change need matching TensorRT
engines: compatible cached engines are reused, and missing ones are built. A reload
does not necessarily mean a new engine build. The OpenVINO GPU path also fixes
ContentVec input dimensions, while RVC remains dynamic; the scope of fixed-shape
optimization depends on the backend and model.

Chunk size also affects audio history, F0 frame alignment, resampler FIFOs, chunk
joining, I/O buffering, and latency reporting. The current pipeline initializes
these together for one chunk size. Even a dynamic-shape backend cannot switch by
simply feeding a different chunk length into that running pipeline.

After changing the setting, restart conversion in the GUI or press **Load / Reload**
in VST3. Restart CLI real-time conversion with the new settings. This is a current
implementation choice to preserve fixed-shape optimization and consistent streaming
state, not a claim that TensorRT itself cannot handle dynamic shapes. See the
[chunk lifecycle](architecture.md#chunk-lifecycle) for the internal contract.

## Comparing speed and audio quality

Keep models, input audio, chunk size, extra context, and pre/post-processing the
same. Separate initial compilation from steady-state processing time. Smaller
chunks reduce input accumulation time, but also check processing headroom and
audio quality. Processing time, retained content delay, and actual microphone-to-output
latency are different measurements.

The [TensorRT investigation](tensorrt_performance_ja.md) measures individual models
with `trtexec` on RTX 3060 Ti using TensorRT 11.0 / CUDA 13.2. It does not establish
current package end-to-end performance or superiority over other products. The sum
of per-stage p95 values is neither a measured pipeline p95 nor input-to-output
latency. See the [architecture notes](architecture.md#latency-trade-offs) for details.
