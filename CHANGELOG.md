# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Version numbers come from `[workspace.package].version` in the root
[`Cargo.toml`](Cargo.toml); the packaging scripts read the same field to name the
release archives. See [`docs/distribution.md`](docs/distribution.md) for the full
versioning and publishing procedure.

## [Unreleased]

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

[Unreleased]: https://github.com/shirohata/vc-rs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/shirohata/vc-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shirohata/vc-rs/releases/tag/v0.1.0
