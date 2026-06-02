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
- choose the **backend** (CPU / CUDA)
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

```sh
ORT_CUDA_VERSION=12
cargo xtask bundle vc-vst3 --release
```

The plugin is built **CUDA-only**: unlike the CLI it does *not* enable the ONNX
Runtime TensorRT EP or the native TensorRT shim (the `vc-core/tensorrt`
feature), because those add a load-time `nvinfer_10.dll` dependency that stops
the plugin from loading in a DAW unless TensorRT is on `PATH`. With it off, the
plugin has no load-time NVIDIA runtime dependency (the CUDA EP and its DLLs load
at runtime), so it loads even without any NVIDIA libraries installed; GPU
execution then needs the CUDA runtime DLLs (see below).

The repo's Cargo config pins `ORT_CUDA_VERSION=12` so the downloaded ONNX
Runtime CUDA provider matches the CUDA 12.x DLLs copied by `package-cuda.ps1`.

> Build the plugin package-scoped via `cargo xtask bundle vc-vst3` (not
> `cargo build --workspace`). A whole-workspace build unifies features with the
> CLI and would re-introduce the TensorRT/`nvinfer` dependency.

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

## Install

Copy the bundle into a standard plugin search path for your OS:

- VST3 — Windows: `%CommonProgramFiles%\VST3\`; macOS:
  `~/Library/Audio/Plug-Ins/VST3/`; Linux: `~/.vst3/`
- CLAP — Windows: `%CommonProgramFiles%\CLAP\`; macOS:
  `~/Library/Audio/Plug-Ins/CLAP/`; Linux: `~/.clap/`

For GPU execution the plugin needs the ONNX Runtime CUDA provider DLLs and the
CUDA / cuDNN runtime libraries. Two options:

- **Self-contained (recommended):** run `package-cuda.ps1` (see above) so the
  DLLs ship inside the bundle. No `PATH` setup needed — only an NVIDIA driver.
- **System install:** put the CUDA / cuDNN library directories on the OS dynamic
  library search path, or launch the DAW from a shell that already has them set.

## Licensing note

Building the **VST3** target links nih-plug's GPLv3 VST3 bindings, so the
resulting `.vst3` is GPLv3. The `.clap` bundle is not affected.
