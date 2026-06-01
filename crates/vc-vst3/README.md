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
  silence.

## Configuration (headless)

There is no GUI yet. Model paths and conversion defaults come from a TOML file.
See [`vc-rs-vst3.example.toml`](vc-rs-vst3.example.toml). Search order:

1. `VC_RS_VST3_CONFIG` environment variable (explicit path)
2. `<os-config-dir>/vc-rs/vst3.toml` — `%APPDATA%` on Windows,
   `$XDG_CONFIG_HOME` or `~/.config` elsewhere
3. `vc-rs-vst3.toml` in the host's working directory

`Pitch`, `Speaker`, `Input Gain`, and `Output Gain` are exposed as
host-automatable parameters and override the config values at runtime.

## Build

```sh
# GPU build (native TensorRT, matches the CLI). Requires the CUDA, cuDNN, and
# TensorRT runtime libraries on the dynamic library search path (same setup as
# building/running the CLI):
cargo xtask bundle vc-vst3 --release

# CPU / ONNX-Runtime only (fastest to build, no GPU deps): set the environment
# variable VC_RS_ENABLE_NATIVE_TENSORRT=0 first, then bundle.
cargo xtask bundle vc-vst3 --release
```

Output bundles land in `target/bundled/`:

- `vc-vst3.vst3` — a bundle; the binary lives in a platform-specific
  `Contents/<arch>/` subfolder (e.g. `x86_64-win`, `x86_64-linux`, `MacOS`)
- `vc-vst3.clap`

## Install

Copy the bundle into a standard plugin search path for your OS:

- VST3 — Windows: `%CommonProgramFiles%\VST3\`; macOS:
  `~/Library/Audio/Plug-Ins/VST3/`; Linux: `~/.vst3/`
- CLAP — Windows: `%CommonProgramFiles%\CLAP\`; macOS:
  `~/Library/Audio/Plug-Ins/CLAP/`; Linux: `~/.clap/`

At runtime the plugin's process (the DAW) must be able to load the ONNX Runtime
shared library and, when using `provider = "cuda" | "tensorrt"`, the CUDA /
cuDNN / TensorRT runtime libraries (`.dll` on Windows, `.so` on Linux,
`.dylib` on macOS). Put those library directories on the OS dynamic library
search path, or launch the DAW from a shell that already has them set.

## Licensing note

Building the **VST3** target links nih-plug's GPLv3 VST3 bindings, so the
resulting `.vst3` is GPLv3. The `.clap` bundle is not affected.
