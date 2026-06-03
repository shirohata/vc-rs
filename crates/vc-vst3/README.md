# vc-vst3 — RVC voice conversion VST3 / CLAP plugin

A DAW plugin front-end for the same RVC pipeline the CLI (`vc-cli`) uses. It
reuses `vc-core` and feeds the pipeline from the host's `process()` callback
instead of driving an audio device directly.

## Architecture

```
host process() ─┬─ downmix L/R → mono ─→ input ring ─┐
                │                                     ▼
                │                          worker thread (vc-core)
                │                          RvcPipeline::process @ host rate
                │                          → SOLA/PSOLA smooth + resample
                │                                     │
                └─ mono → L/R  ◀── output ring ◀──────┘
```

- The audio thread never allocates, locks, or runs inference — it only pushes /
  pops lock-free `rtrb` ring buffers ([`runtime.rs`](src/runtime.rs)).
- A worker thread owns the `RvcPipeline`, mirroring the CLI's `engine.rs` worker.
- RVC is inherently high-latency; the plugin reports its latency via
  `set_latency_samples` so the host can apply delay compensation.
- The model loads on the worker thread; until it is ready the plugin emits
  silence (the GUI shows the current status).

## GUI / settings

Open the plugin's editor in your DAW ([`editor.rs`](src/editor.rs), egui). From
there you can:

- **Browse** for the RVC model, embedder, and F0 (RMVPE) `.onnx` files
- choose the **backend** (CPU / Windows ML / CUDA / TensorRT, depending on the package)
- set the **chunk size** (ms) — larger means more latency but more context
- hit **Load / Reload** to apply model / backend / chunk edits
- watch the **status** line (`no models configured` / `models configured; click
  Load / Reload` / `loading…` / `running (cuda)` / `load failed: …`)
- adjust the live parameters (Pitch / Speaker / Input · Output gain)

Model/backend/chunk edits are **staged**: they only take effect when you press
**Load / Reload** (shown by an "unapplied" indicator). Live parameters apply
immediately. Changing the chunk size also re-reports the plugin latency to the
host.

Model paths and conversion settings are stored in the **plugin state**, so the
host saves and restores them per project/preset (and they travel with the
project). `Pitch`, `Speaker`, and the gains are ordinary **DAW parameters**
(automatable, host-persisted).

### Optional headless config seed

For a fresh instance with no saved settings, a TOML file can seed the initial
values (handy for automation / first run). See
[`vc-rs-vst3.example.toml`](vc-rs-vst3.example.toml). Search order:

1. `VC_RS_VST3_CONFIG` environment variable (explicit path)
2. `<os-config-dir>/vc-rs/vst3.toml` — `%APPDATA%` on Windows,
   `$XDG_CONFIG_HOME` or `~/.config` elsewhere
3. `vc-rs-vst3.toml` in the host's working directory

The seed only applies when the instance has no models set yet; once a project
has saved its settings, the state wins and the config file is ignored.
Model / backend / `chunk_ms` apply on Load / Reload from the GUI. The remaining
latency settings (`crossfade_ms`, `sola_search_ms`, `extra_convert_ms`, …) come
from the config and apply on (re)instantiation.

## Build

The default Windows package uses **Windows ML** through Windows App SDK Runtime
2.x. This keeps `onnxruntime.dll` and `DirectML.dll` out of the bundle; the app
only ships the small Windows App SDK bootstrapper DLL. CUDA and native TensorRT
packages remain available as explicit cargo feature builds.

### One-shot packaging (recommended)

[`package.ps1`](package.ps1) runs the whole distribution pipeline for a chosen
variant: `cargo xtask bundle` → (TensorRT) build the engine-builder helper →
the matching `package-<variant>.ps1` populate step → stage `vc-vst3.vst3` +
`vc-vst3.clap` + `LICENSE` + a generated `INSTALL.txt` → a versioned
`dist\vc-vst3-<variant>-v<version>-win-x64.zip`.

```powershell
pwsh crates\vc-vst3\package.ps1                                  # Windows ML (default)
pwsh crates\vc-vst3\package.ps1 -Variant cuda `
  -CudaBin "...\CUDA\v12.9\bin" -CudnnBin "...\CUDNN\v9.22\bin\12.9\x64"
pwsh crates\vc-vst3\package.ps1 -Variant tensorrt -BuilderSm sm86
```

Variant-specific options are forwarded to the populate script. Useful flags:
`-OutDir <dir>` (default `dist\`), `-SkipBuild` (reuse `target\bundled`),
`-NoZip` (populate only), `-Clean` (drop stale bundles first). For the `cuda`
and `tensorrt` variants set up the matching CUDA/TensorRT toolchain on `PATH`
first (e.g. dot-source [`scripts\activate.ps1`](../../scripts/activate.ps1)) —
the script does not modify your environment. The steps below document the
underlying cargo + populate commands the script orchestrates.

### Windows ML package (default, Windows)

```sh
cargo xtask bundle vc-vst3 --release
```

Enables `vc-core/windowsml`. Model loading bootstraps Windows App SDK Runtime
2.x on the worker thread, then loads the runtime's shared ONNX Runtime
(`onnxruntime.dll`) with ORT API 24. The default provider is `windowsml`, which
tries Windows ML catalog EPs first and falls back to DirectML, then CPU.
`windowsml-directml` and `windowsml-cpu` force those paths. Explicit catalog
providers are also accepted: `windowsml-nvtrtx`, `windowsml-qnn`,
`windowsml-openvino`, `windowsml-migraphx`, and `windowsml-vitisai`. These
explicit providers do not fallback; they fail if the requested catalog EP is not
present or ready.

End users must have **Windows App SDK Runtime 2.x** installed. After bundling,
copy the bootstrapper DLL into the bundle:

```powershell
pwsh crates\vc-vst3\package-windowsml.ps1
```

Do not bundle `onnxruntime.dll`, `DirectML.dll`, CUDA, or cuDNN DLLs for this
package. Those are provided by Windows App SDK Runtime.

### CUDA package

```sh
ORT_CUDA_VERSION=12
cargo xtask bundle vc-vst3 --release --no-default-features --features cuda
```

Enables the ONNX Runtime **CUDA execution provider** (`vc-core/cuda`). The
plugin has *no* load-time NVIDIA dependency — the CUDA EP and its DLLs load at
runtime — so it loads in a DAW even without NVIDIA libraries installed; GPU
execution then needs the CUDA runtime DLLs (see below). The repo's Cargo config
pins `ORT_CUDA_VERSION=12` so the downloaded ONNX Runtime CUDA provider matches
the CUDA 12.x DLLs copied by `package-cuda.ps1`.

### TensorRT package

```sh
cargo xtask bundle vc-vst3 --release --no-default-features --features tensorrt
```

Drops the ONNX Runtime CUDA EP entirely and runs the GPU path through the
**native TensorRT** shim (`vc-core/tensorrt`). This avoids shipping the ~2 GB
CUDA EP + cuDNN/cuBLAS/cuFFT set; instead it needs the TensorRT runtime beside
the plugin (`nvinfer_<N>.dll`, `nvinfer_plugin_<N>.dll`, the matching
`nvinfer_builder_resource_sm*_<N>.dll` for your GPU, and `cudart64_<M>.dll`).
`<N>` is the TensorRT major version (10, 11, ...) and `<M>` the paired CUDA
major (12 for TRT10, 13 for TRT11), both detected from the install at build and
packaging time.

> ⚠️ The native shim links `nvinfer_<N>.dll` at **load time**, so this package
> fails to load in a DAW unless those TensorRT DLLs resolve on the OS DLL search
> path. Windows searches the loaded module's own directory first, so placing the
> DLLs next to the plugin binary (in `Contents/<arch>/`) satisfies the implicit
> import. Selecting the `cuda` provider in this package falls back to TensorRT.

> Build the plugin package-scoped via `cargo xtask bundle vc-vst3` (not
> `cargo build --workspace`). A whole-workspace build unifies features with the
> CLI and would pull both the CUDA EP **and** the TensorRT/`nvinfer` dependency
> into the plugin.

After bundling, populate the TensorRT runtime with
[`package-tensorrt.ps1`](package-tensorrt.ps1) (see *Bundling the TensorRT
runtime* below).

Output bundles land in `target/bundled/`:

- `vc-vst3.vst3` — a bundle; the binary lives in a platform-specific
  `Contents/<arch>/` subfolder (e.g. `x86_64-win`, `x86_64-linux`, `MacOS`)
- `vc-vst3.clap`

### Bundling the CUDA runtime (self-contained GPU build, Windows)

So users don't have to install CUDA/cuDNN or edit `PATH`, the required CUDA
12.x / cuDNN 9.x DLLs can be shipped beside the plugin. The ONNX Runtime *core*
is statically linked, so only its CUDA execution-provider DLLs and their
CUDA/cuDNN dependencies are needed; the plugin makes its own folder discoverable
for bundled DLLs at startup without changing the DAW process' default DLL search
policy, and preloads the bundled DLLs on the explicit CUDA Load / Reload path
when they are present ([`src/dll_path.rs`](src/dll_path.rs)).

After bundling, run [`package-cuda.ps1`](package-cuda.ps1):

```powershell
pwsh crates\vc-vst3\package-cuda.ps1 `
  -CudaBin "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin" `
  -CudnnBin "C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64"
```

This copies the minimal set (ONNX Runtime CUDA provider DLLs + `cudart`,
`cublas`, `cublasLt`, `cufft`, and the `cudnn*64_9` libraries) plus the license
files in [`dist/licenses/`](dist/licenses) into the bundle. End users then only
need an up-to-date NVIDIA GPU **driver** — no CUDA/cuDNN install. See
[`dist/licenses/THIRD-PARTY-NOTICES.md`](dist/licenses/THIRD-PARTY-NOTICES.md)
for redistribution terms.

### Bundling the TensorRT runtime (self-contained GPU build, Windows)

For the **TensorRT package**, [`package-tensorrt.ps1`](package-tensorrt.ps1)
copies the TensorRT DLLs into the bundle. There are two dependency layers:

- **Runtime** (plugin load + engine execution): `nvinfer_<N>.dll`,
  `nvinfer_plugin_<N>.dll`, `cudart64_<M>.dll`. Always copied.
- **Engine build** (first run, on a cache miss): the ORT-free helper
  `vc-tensorrt-builder.exe` builds engines from the ONNX models, which needs
  `nvonnxparser_<N>.dll` and the `nvinfer_builder_resource_sm*_<N>.dll` matching
  the user's GPU. Copied unless `-RuntimeOnly`.

The TensorRT major `<N>` is read from the chosen install (the newest TensorRT
folder under the repo root, or `%TENSORRT_ROOT%`), and the CUDA major `<M>` is
paired automatically (TRT10 → CUDA 12, TRT11 → CUDA 13).

```powershell
# Self-contained for the target GPU's SM (e.g. sm86 = RTX 30xx, sm89 = RTX 40xx;
# `nvidia-smi --query-gpu=compute_cap --format=csv` reports it):
pwsh crates\vc-vst3\package-tensorrt.ps1 -BuilderSm sm86

# Or runtime DLLs only (smallest; engines built/cached elsewhere):
pwsh crates\vc-vst3\package-tensorrt.ps1 -RuntimeOnly
```

The builder resource DLLs are GPU-architecture specific and large (~160–640 MB
each); with no `-BuilderSm` the script bundles them all (~2.5 GB) and warns.
Because a VST3/CLAP host's process is the DAW (not the plugin), the plugin finds
the bundled helper via the `VC_RS_TENSORRT_BUILDER_HELPER` env var — set it to
the helper's path in the installed plugin folder (the script prints the exact
command). With engines prebuilt and cached, only the runtime layer is needed at
play time. End users otherwise need just an up-to-date NVIDIA GPU **driver**.

## Install

Copy the bundle into a standard plugin search path for your OS:

- VST3 — Windows: `%CommonProgramFiles%\VST3\`; macOS:
  `~/Library/Audio/Plug-Ins/VST3/`; Linux: `~/.vst3/`
- CLAP — Windows: `%CommonProgramFiles%\CLAP\`; macOS:
  `~/Library/Audio/Plug-Ins/CLAP/`; Linux: `~/.clap/`

For the default Windows ML package, install Windows App SDK Runtime 2.x and run
`package-windowsml.ps1` so `Microsoft.WindowsAppRuntime.Bootstrap.dll` is beside
the plugin binary. No ONNX Runtime, DirectML, CUDA, or cuDNN DLLs should be
copied into that bundle.

For the CUDA package, GPU execution needs the ONNX Runtime CUDA provider DLLs
and the CUDA / cuDNN runtime libraries. Two options:

- **Self-contained (recommended):** run `package-cuda.ps1` (see above) so the
  DLLs ship inside the bundle. No `PATH` setup needed — only an NVIDIA driver.
- **System install:** put the CUDA / cuDNN library directories on the OS dynamic
  library search path, or launch the DAW from a shell that already has them set.

## Licensing note

Building the **VST3** target links the Steinberg VST3 SDK bindings (GPLv3) via
nice-plug, so the resulting `.vst3` is GPLv3. The `.clap` bundle is not affected.
