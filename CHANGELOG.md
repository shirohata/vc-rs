# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
with versioned release entries only, and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Version numbers come from `[workspace.package].version` in the root
[`Cargo.toml`](Cargo.toml); the packaging scripts read the same field to name the
release archives. See [`docs/distribution.md`](docs/distribution.md) for the full
versioning and publishing procedure.

## [0.5.0] - 2026-09-07

### Added

- Built-in RVC PTH-to-ONNX conversion with GUI integration.
- Support for rvc-onnx-web streaming exports, including NSF phase and noise
  inputs, on native TensorRT and Windows ML TensorRT RTX.
- Realtime worker stop causes in engine status and usable execution providers
  from the live Windows ML catalog in provider pickers.

### Changed

- Windows ML packages now require Windows App SDK Runtime 2.1 or newer and use
  ONNX Runtime API 24 through ort 2.0.0-rc.13.
- Upgraded native TensorRT to 11.2.1 with CUDA 13.3 Update 1 and isolated native
  caches by SDK version. Existing engines rebuild on first use after upgrading.
- Consolidated provider abstraction in the shared core and updated dependencies.
- Release publishing now requires the CI formatting and Clippy checks to pass.

### Fixed

- Resolved the Windows ML bootstrapper beside the VST3 plugin instead of
  depending on the DAW's DLL search policy.
- Corrected resampling timelines and preserved final WAV tails.
- Corrected NSF phase carry across overlapping windows and kept RVC latent
  noise on an absolute frame timeline.
- Bounded PTH ZIP parsing with the zip crate and updated dependencies for
  security fixes.
- Deferred audio stream error logging away from the realtime callback.

### Performance

- Reduced fixed-hop resampler latency and reused RMS scratch buffers.

### Distribution notes

- Windows binaries are not code-signed; Windows may display a security warning
  when downloading or running them.

## [0.4.0] - 2026-06-25

### Added

- Standalone GTCRN input denoising across the shared runtime, CLI, GUI,
  Windows ML packages, and native TensorRT path.
- Optional ASIO support for standalone CLI and GUI builds, with per-direction
  audio host selection and RT-priority CPAL handling.
- Microsoft Store GUI-only MSIX packaging via the Windows App Development CLI.
- Offline chunk-join sweep tooling, model-free pipeline benchmarks, and
  validation shared across frontends.

### Changed

- Reworked the standalone audio host layer to support separate input and output
  backends.
- Optimized mono channel mixing, RMS mix-gain processing, output level
  postprocessing, and shared waveform handling.
- Expanded distribution packaging and release checks for GTCRN, Store MSIX, and
  version-specific GitHub release notes.

### Fixed

- Supported Applio-style RVC exports and models with the optional `rnd`
  latent-noise input.
- Accepted `f0: true` metadata and sized model windows at the exported model's
  sample rate.
- Failed fast when the Windows ML bootstrap DLL is missing and handled catalog
  execution-provider preparation consistently.
- Fixed TensorRT-only RVC profile builds.
- Validated conversion settings consistently across frontends.

## [0.3.0] - 2026-06-15

### Added

- Live passthrough switching for standalone sessions with a complete model set.
- Named GPU device selection for supported CUDA and TensorRT providers.
- Staged model-loading progress in the GUI.
- Chunk-join diagnostics for measuring output-boundary artifacts.
- Shared conversion-pipeline architecture guidance, CI checks, cargo-deny
  policy, a check-only pre-commit rustfmt hook, and CPU hot-path benchmarks.

### Changed

- GPU Priority now applies to every backend, not just native TensorRT: it sets a
  process-wide Windows GPU scheduling priority class (in addition to the
  TensorRT CUDA stream priority on that path), so the control is shown in the GUI
  for Windows ML / CPU builds as well. High additionally opts the process out of
  CPU power throttling (EcoQoS) so inference keeps full clock when the window is
  in the background, removing the large foreground/background timing difference.
- CLI, GUI, VST3, and WAV conversion now reuse shared chunk-conversion,
  smoothing, and output-assembly components.
- Standalone realtime processing now wakes the input worker when audio arrives
  instead of polling every 2 ms.
- The GUI chunk-size control now supports values down to 40 ms.
- Updated CPAL, nice-plug, egui, rfd, toml, rubato, and compatible transitive
  dependencies.

### Fixed

- Reduced audible chunk-join artifacts at small chunk sizes.
- Restored VST3 processing correctly after plugin reload.
- Opened CPAL streams using the device's native channel count.
- Surfaced Windows ML catalog execution-provider preparation failures.
- Ensured VST3 installation stages the required runtime DLLs.

### Performance

- Removed repeated allocation and redundant work from inference, DSP, and
  SOLA/PSOLA hot paths.
- Vectorized SOLA/PSOLA offset search and reused input-side inference buffers.

## [0.2.1] - 2026-06-10

### Added

- Standalone RNNoise input denoising in the GUI and CLI.
- Configurable input noise gate before RVC feature and F0 extraction, available
  in the standalone apps and VST3 plugin.
- Optional F0 post-processing support in the core RVC pipeline.
- On-demand download of explicitly selected Windows ML catalog execution
  providers during standalone GUI and CLI model loading.
- Deterministic CPU-only A/B audio comparison tooling for regression analysis.

### Changed

- Release publishing now relies on GitHub's asset digests instead of generating
  separate SHA-256 sidecar files.

## [0.2.0] - 2026-06-07

### Added

- Standalone GUI app (`vc-gui.exe`) backed by a shared realtime runtime, shipped
  alongside the CLI in the standalone packages.
- `doctor` CLI command for runtime diagnostics.
- TensorRT GPU priority control.

### Changed

- Standalone packages now bundle the GUI together with the CLI.
- Refined GUI runtime controls and diagnostics.
- Capped the TensorRT builder at 4 max threads.
- Distribution packaging now generates exact per-binary Rust license notices.
- TensorRT packages always bundle every GPU builder resource for full
  compatibility (removed the `-BuilderSm` packaging option).

### Fixed

- Preserved silent output buffering in the realtime worker.

### Docs

- Added distribution safety guidance, versioning, and publishing procedure
  ([`docs/distribution.md`](docs/distribution.md)).
- Added a release verification/publish script (`scripts/release.ps1`) and this
  changelog.

## [0.1.0] - 2026-06-05

Initial release.

### Added

- Rust RVC (Retrieval-based Voice Conversion) voice changer with two front-ends
  sharing one inference pipeline: the `vc-rs` CLI (real-time mic→speaker and
  WAV→WAV) and the `vc-vst3` VST3 plugin.
- Two distributed inference backends: Windows ML (broad GPU support incl.
  DirectML, via the Windows App SDK Runtime) and native TensorRT (NVIDIA-only,
  self-contained runtime).
- Side-by-side VST3 variants with isolated per-variant packaging.
- One-shot distribution packaging scripts for all four Windows x64 variants.
- Auto-generated bundled third-party license notices during packaging.

[0.5.0]: https://github.com/shirohata/vc-rs/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/shirohata/vc-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/shirohata/vc-rs/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/shirohata/vc-rs/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/shirohata/vc-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shirohata/vc-rs/releases/tag/v0.1.0
