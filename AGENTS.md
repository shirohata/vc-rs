# Repository guidance

`vc-rs` is a Windows-first Rust RVC voice changer with CLI, GUI, and VST3
front-ends. Keep these instructions focused on working constraints; look up
module inventories, feature defaults, and commands in their source files.

Read documentation relevant to the task:
- [`docs/architecture.md`](docs/architecture.md): conversion data flow and ownership.
- [`scripts/README.md`](scripts/README.md) and [`justfile`](justfile): setup and build tasks.
- [`crates/vc-vst3/README.md`](crates/vc-vst3/README.md): plugin behavior and host integration.
- [`docs/distribution.md`](docs/distribution.md): packaging and publishing requirements.

## Build and verification

First-time setup (winget + NVIDIA SDKs): see `scripts/README.md`. The line is
Windows, CUDA 13.3 Update 1 / TensorRT 11.2.1. Day-to-day:
- Prefer the relevant `just` recipe; run `just` to list tasks. Recipes handle
  environment setup. For raw Cargo commands, dot-source
  `. scripts/activate.ps1` and `. scripts/rustflags.ps1` in that shell
  (puts CUDA/cuDNN/TensorRT on PATH; without it test exes fail to launch with
  `STATUS_DLL_NOT_FOUND`). To run tests without the GPU stack, set
  `VC_RS_ENABLE_NATIVE_TENSORRT=0`.
- `just test-ci` runs the CI Rust test suites. `just test-cpu` disables the native
  TensorRT shim but retains workspace default features; it is not a substitute
  for the CPU-feature tests or Windows ML runtime validation.
- Select tests and format/lint checks for the behavior and files affected.
  Documentation- or comment-only edits need diff/link checks as applicable, not
  runtime tests. Validate the affected backend when runtime behavior changes,
  and audio quality/latency when the change can affect them. Report checks run
  and limitations. Do not repeat or broaden
  passing checks without a new change, failure, or unresolved concern.
- CI gates are defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
  CI cannot cover native TensorRT FFI; changes there need local SDK-enabled
  verification. Keep workspace lint policy intact; justify narrow exceptions
  at the affected item instead of relaxing the workspace rules.
- Build VST3 artifacts package-scoped (`just bundle [windowsml|tensorrt]`);
  whole-workspace feature unification is not suitable for distributed plugins.

## Windows ML checks in the Codex sandbox

- `MddBootstrapInitialize2` HRESULT `0x80670016` can be sandbox-specific;
  do not infer a missing runtime from it alone. Follow the
  [Windows ML troubleshooting procedure](docs/development_ja.md#windows-ml-troubleshooting)
  before changing runtime setup or bootstrap code; `os error 126` is a separate issue.
- For this signature, retry the same authorized diagnostic/test through
  `sandbox_permissions: require_escalated` with normal tool approval review,
  without redundant conversational confirmation. If unavailable or rejected,
  report the verification limitation and continue independent work.
- If it passes outside, record the sandbox limitation and continue release work;
  otherwise investigate the failure. CPU-only tests do not replace Windows ML validation.

## GUI and VST3 settings policy

- Expose controls needed for ordinary voice conversion. Keep algorithm choices
  and tuning settings that users do not need to adjust routinely in config
  files or the CLI. A config option does not imply a graphical control.
- Smoother selection (SOLA / PSOLA / phase vocoder) is an advanced config
  setting in GUI and VST3; do not add a smoother selector to either editor.
- Hiding a control must preserve its config key, serialization, and saved
  project/preset values. Apply defaults only when the setting is absent so
  simplifying the UI does not silently change existing users' audio.
- Document the advanced configuration path and when changes take effect,
  including VST3's distinction between fresh-instance config seeds and saved
  project state.

## Real-time audio constraints
- Avoid heap allocation, blocking I/O, and locks on the real-time audio callback path.
- Do not perform logging directly inside the audio callback unless already proven safe.
- Prefer preallocated buffers and message passing to background workers.
- When changing chunking, buffering, or latency-sensitive code, verify real-time
  safety as part of the change. Review the affected worker, model, and DSP paths
  together; frame alignment changes require audio-quality validation.

## Shared conversion pipeline
- CLI, GUI, VST3, and WAV conversion must reuse the shared conversion pipeline
  wherever their device, host, and offline-processing constraints permit.
- Keep inference, chunk conversion, smoothing, and output assembly in shared
  components rather than duplicating them in front-ends.
- Do not add a front-end-specific conversion path without documenting why the
  shared components cannot satisfy its constraints.
- Keep unavoidable differences narrowly scoped to device or host I/O,
  scheduling, buffering, latency reporting, and offline final-tail handling.
- Follow [`docs/architecture.md`](docs/architecture.md) as the canonical
  description of conversion data flow and ownership boundaries.

## Distribution safety
- Do not embed or ship machine-specific paths, developer-machine user names,
  secrets, local models, caches, logs, debug artifacts, or other local state.
- Build distributable archives only through the repository packaging scripts;
  keep backend variants isolated and include all required third-party licenses.
- Before publishing a package, follow [`docs/distribution.md`](docs/distribution.md).

## Comments for future coding agents

For non-trivial changes, explain intent, invariants, compatibility, or real-time
constraints where a future refactor could break them. Mention coupled modules
or tests when necessary. Update existing comments when they suffice; do not
add redundant comments or restate obvious code behavior.

## Git

- Inspect the current branch and working tree; preserve unrelated changes.
- Create a working branch before editing on `main`.
- Commit or push only when requested.
