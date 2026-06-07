# Distribution Safety

This document is the required checklist for packages distributed to other
users. The repository currently ships four Windows x64 variants:

- `vc-rs-cli-windowsml`
- `vc-rs-cli-tensorrt`
- `vc-vst3-windowsml`
- `vc-vst3-tensorrt`

Build release archives with the repository packaging scripts, normally:

```powershell
. scripts/activate.ps1
pwsh scripts/package-all.ps1 -BuilderSm <target-sm>
```

Do not assemble release archives manually from `target\release` or
`target\bundled`. The packaging scripts apply release flags, isolate variants,
populate runtime dependencies, and copy license material.

## Package Contents

Packages must not contain:

- Machine-specific absolute paths, developer user names, or local directory
  layouts.
- Secrets, credentials, tokens, private configuration, or environment dumps.
- Local models or weights, TensorRT engine caches, audio recordings, logs,
  temporary files, or other developer-machine state.
- Debug artifacts such as `.pdb` files, unless a separate debug-symbol package
  is intentionally produced.
- Runtime DLLs belonging to a different backend variant.

Reference ContentVec and RMVPE models are downloaded only at the user's request
by `download-models.ps1`; they are third-party files and must not be added to a
release archive.

## Backend Isolation

Build every distributed binary with one provider feature set. Do not reuse a
binary produced for another variant.

- Windows ML packages must not contain ONNX Runtime, DirectML, CUDA, cuDNN, or
  TensorRT DLLs. They contain the Windows App SDK bootstrapper and require the
  Windows App SDK Runtime on the user's machine.
- TensorRT packages must not contain ONNX Runtime provider DLLs. They contain
  the matching TensorRT and CUDA runtime DLLs; non-runtime-only packages also
  contain the engine-builder helper and the selected GPU builder resources.
- Build VST3 variants package-scoped. Do not use a whole-workspace build whose
  unified features can pull incompatible providers into the plugin.

Start packaging from clean per-variant staging directories. Stale sidecar DLLs
from another package must never survive into a release.

## Licenses

Every archive must include the repository `LICENSE` and all notices or license
texts required by bundled third-party code and redistributable DLLs.

License generation and collection must be checked for the exact package and
feature set being shipped. Treat a missing, stale, or mismatched license notice
as a release blocker, even when a packaging script emits only a warning.

In particular, verify:

- The Rust dependency notice covers the actual CLI or VST3 variant.
- Windows ML packages include the license matching the redistributed Windows
  App SDK bootstrapper.
- TensorRT packages include the applicable NVIDIA license or EULA text.

## Pre-Publish Check

Before publishing each final ZIP:

1. Build it with the appropriate `package.ps1` or `scripts/package-all.ps1`
   command without relying on an unverified `-SkipBuild` artifact.
2. Extract the ZIP into a fresh directory and inspect the actual archived
   contents, not only the staging directory.
3. Confirm required binaries, runtime DLLs, install instructions, and license
   files are present, and prohibited files are absent.
4. Search the extracted files and printable binary strings for build-machine
   paths, user names, secrets, and other local state.
5. Smoke-test the extracted CLI package. Validate the extracted VST3 package
   with the Steinberg validator and, when practical, load it in a clean DAW
   environment.
6. Test on a machine or environment that does not rely on the build machine's
   SDK paths, caches, or environment variables.
7. Generate and publish a SHA-256 checksum for the final ZIP.

Distributed binaries are currently not code-signed. Keep the user-facing
documentation explicit about the resulting Windows warning until signing is
introduced.

## Current Automation Limits

The packaging scripts provide important safeguards, including release stripping,
Rust path remapping, fresh VST3 staging, variant-specific population, and license
copying. They do not currently provide a complete publish gate.

Review these known limitations before release:

- CLI packages currently copy the committed VST3 license directory rather than
  generating a CLI-specific Rust dependency notice.
- Packaging may continue after warning that a Windows App SDK license is
  unavailable, and TensorRT license discovery is best-effort.
- Final ZIPs are not automatically scanned for secrets, local paths, prohibited
  files, or backend cross-contamination.
- `-SkipBuild` does not prove that the reused binary matches the requested
  backend variant.

Do not weaken these safeguards or dismiss these limitations without replacing
them with an equivalent or stronger automated check.
