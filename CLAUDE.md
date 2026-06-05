# CLAUDE.md

Guidance for AI assistants working in this repository. Read this together with
[`AGENTS.md`](AGENTS.md) (build-environment + real-time-safety + code-comment
rules — those apply in full and are not repeated here).

## What this project is

`vc-rs` is a Rust **RVC (Retrieval-based Voice Conversion)** voice changer. It
converts microphone input or WAV files into another voice using ONNX-format RVC
models. Two front-ends share one inference pipeline:

- **CLI** (`vc-rs.exe`, crate `vc-cli`) — real-time mic→speaker and WAV→WAV.
- **VST3 plugin** (`vc-vst3.vst3`, crate `vc-vst3`) — loads into a DAW.

The project is Windows-first (x64). Distribution targets two GPU/inference
backends: **Windows ML** (via Windows App SDK Runtime, broad GPU support incl.
DirectML) and **native TensorRT** (NVIDIA-only, self-contained runtime). A
`cuda` ONNX Runtime EP feature exists for local dev but is no longer packaged.

## Workspace layout

Cargo workspace (`resolver = "2"`); `tools/tensorrt_builder` is excluded.

| Crate | Path | Role |
| --- | --- | --- |
| `vc-core` | `crates/vc-core` | Audio-I/O-agnostic engine: RVC pipeline, DSP, SOLA/PSOLA, providers. Both front-ends depend on it. |
| `vc-cli` | `crates/vc-cli` | CLI binary `vc-rs`; owns CPAL/WASAPI device I/O and the real-time engine worker. |
| `vc-vst3` | `crates/vc-vst3` | VST3 plugin (nice-plug + egui); feeds the pipeline from the host `process()` callback. |
| `xtask` | `xtask` | `cargo xtask bundle …` plugin bundler (nice-plug-xtask). |

### `vc-core` modules

- `model_rvc` — ONNX Runtime sessions, streaming RVC state, feature extraction,
  F0 extraction, pitch prep, output shaping. Public API: `RvcPipeline` /
  `RvcPipelineConfig` (`pipeline.rs`), `inspect_model` (`inspect.rs`),
  `VoiceModel`/`PassthroughModel` (`api.rs`). Submodules: `feature`, `pitch`,
  `stream`, `shape`, `sessions`, `onnx_meta` (dependency-free ONNX reader used
  by `inspect`, works without ORT), `tensorrt` + `native_tensorrt` (TRT path).
- `dsp` — resampling, sample conversion, RMS/envelope, correlation, crossfade.
- `sola` — chunk joining via SOLA or PSOLA.
- `provider` — the shared `Provider` enum (re-exported at crate root).
- `windows_ml` — Windows ML catalog EP support (Windows + `windowsml` only).

Changes to chunk sizing, model context, smoothing, or output latency usually
cross `engine` (in `vc-cli`), `model_rvc`, `sola`, and `dsp` — review together.
See [`docs/architecture.md`](docs/architecture.md) for the full data-flow and
the realtime/worker boundary (the canonical design doc).

## Feature flags (inference backends)

Backends are compile-time features, mutually arranged so ORT and native
TensorRT never share a process:

- `vc-core/ort` — ONNX Runtime CPU core (base for `cpu`, `cuda`, `windowsml`).
- `vc-core/windowsml` — ORT loaded dynamically from Windows App SDK Runtime 2.x
  (ORT API 24) + DirectML. Keeps `onnxruntime.dll`/`DirectML.dll` out of bundles.
- `vc-core/cuda` — ORT CUDA EP (large; dev-only, unpackaged).
- `vc-core/tensorrt` — native TensorRT shim, **no ORT**. Builds/runs serialized
  engines via TensorRT's builder API (not the ORT TensorRT EP).
- `vc-core/clap` — derives `clap::ValueEnum` on shared enums (CLI only).

Front-end defaults: `vc-cli` defaults to `["windowsml","tensorrt"]` (one dev
binary covers both; default provider falls back to `cpu`). `vc-vst3` defaults to
`["windowsml"]`. Distribution packages build single-provider variants with
`--no-default-features --features windowsml|tensorrt`.

> ⚠️ Build the plugin **package-scoped**: `cargo xtask bundle vc-vst3 …`, not
> `cargo build --workspace`. A whole-workspace build unifies features and would
> pull both the CUDA EP and the TensorRT dependency into the plugin.

## Build & test

GPU build/run line is **CUDA 13 / TensorRT 11** (CUDA 12 / TRT 10 dropped). CPU
or Windows-ML-only work needs none of the NVIDIA SDKs.

The [`justfile`](justfile) is the preferred entry point — recipes wrap the long
cargo/xtask/PowerShell invocations and dot-source `activate.ps1` where the GPU
stack is needed, so per-session env setup isn't forgotten. Run `just` to list
them. (`just setup` provisions `just` itself for next time; on a fresh checkout
bootstrap it directly with `pwsh -File scripts/bootstrap.ps1`.)

```powershell
just setup              # one-time: Rust, Git, MSVC C++, just (winget scope)
just test               # full workspace tests (activates GPU stack)
just test-cpu           # fast tests, no GPU stack (VC_RS_ENABLE_NATIVE_TENSORRT=0)
just build              # dev CLI (both backends)
just bundle             # VST3, Windows ML   (just bundle tensorrt for the TRT variant)
just verify             # tests + bundle smoke test
just package            # build the shipped distribution zips
```

Underlying raw commands (what the recipes run):

```powershell
. scripts/activate.ps1          # per shell session: CUDA/cuDNN/TensorRT on PATH
cargo build --release           # CLI (vc-rs.exe)
cargo xtask bundle vc-vst3 --release                                   # VST3, Windows ML
cargo xtask bundle vc-vst3 --release --no-default-features --features tensorrt
cargo test --workspace
pwsh -File scripts/verify.ps1   # cargo test + bundle smoke test
```

> **STATUS_DLL_NOT_FOUND**: test exes link the native TensorRT shim, so the
> TensorRT `bin` must be on PATH (`activate.ps1`). To run tests without the GPU
> stack, set `VC_RS_ENABLE_NATIVE_TENSORRT=0` (or `verify.ps1 -NoNativeTensorRT`).

First-time SDK setup, packaging, and env details: [`scripts/README.md`](scripts/README.md)
and [`docs/development_ja.md`](docs/development_ja.md).

## CLI commands

`devices` (list audio devices), `inspect --model x.onnx` (ONNX I/O + metadata,
backend-independent), `windowsml-eps list|install` (Windows + windowsml only),
`engine-cache info|clear` (size/clear the shared TensorRT + WinML-TensorRT-RTX
engine cache at `%LOCALAPPDATA%\vc-rs\tensorrt-cache`), `run` (real-time
mic→speaker), `wav` (file→file, same pipeline for deterministic testing). Key params: `--provider`, `--chunk-ms`, `--extra-convert-ms`,
`--speaker-id`, `--pitch-shift`, `--input/output-gain`, `--rms-mix-rate`.
Full usage is in [`README.md`](README.md).

## VST3 plugin notes

Model/backend/chunk edits are **staged** in the egui editor and apply only on
**Load / Reload**; live params (Pitch/Speaker/gains) are DAW parameters that
apply immediately. Model paths + settings persist in plugin state (per
project/preset). The GUI lists only providers the package was built with.
Optional headless TOML seed for fresh instances (see `crates/vc-vst3/README.md`).

## Conventions & guardrails

- **Real-time safety is a hard boundary.** Audio callbacks only move samples
  through lock-free ring buffers and emit silence on underrun — no allocation,
  locks, blocking I/O, inference, or logging. All model-scale work lives on the
  worker thread. See `AGENTS.md` and `docs/architecture.md`.
- **Frame-grid / chunk alignment changes are audio-quality changes, not
  cleanup.** Content features, continuous F0 (`pitchf`), coarse pitch, and model
  output must refer to the same time window; misalignment sounds like drift or
  unstable consonants.
- **WAV mode reuses the realtime pipeline** so quality changes can be tested
  deterministically. A WAV-vs-realtime difference should trace to buffering /
  scheduling / final-tail handling, not a separate model path.
- RVC models must be **`.onnx`** (`.pth` is not supported). Models are never
  bundled; `download-models.ps1` fetches the reference ContentVec/RMVPE models.
- For the **TensorRT VST3 package**, leftover ORT provider DLLs (e.g.
  `onnxruntime_providers_cuda.dll`) in the bundle crash the windowsml plugin in
  DAWs — Windows ML bundles must not contain ORT/DirectML/CUDA DLLs.
- Leave intent/invariant/constraint comments when modifying non-trivial code
  (per `AGENTS.md`); add guardrail comments where a future refactor would be
  unsafe.

## Git

Default branch `main`; work currently on `dev`. Commit/push only when asked;
branch first if on `main`. License is MIT; bundled third-party notices are
generated during packaging (`THIRD_PARTY_NOTICES.md`).
