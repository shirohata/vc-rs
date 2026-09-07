# Distribution Safety

This document is the required checklist for packages distributed to other
users. The repository currently ships four Windows x64 variants:

- `vc-rs-windowsml` (GUI + CLI)
- `vc-rs-tensorrt` (GUI + CLI)
- `vc-vst3-windowsml`
- `vc-vst3-tensorrt`

Build release archives with the repository packaging scripts, normally:

```powershell
. scripts/activate.ps1
pwsh scripts/package-all.ps1
```

Do not assemble release archives manually from `target\release` or
`target\bundled`. The packaging scripts apply release flags, isolate variants,
populate runtime dependencies, and copy license material.

Microsoft Store submission uses a narrower Windows ML GUI-only MSIX:

```powershell
pwsh scripts/package-store-msix.ps1 -BuildPackage `
  -PackageName <Partner-Center-package-identity-name> `
  -Publisher "CN=<Partner-Center-publisher>"
```

The script starts from the `app-windowsml` package output, then creates a
temporary Store staging directory containing only `vc-gui.exe`, the Windows ML
framework dependency declaration, install notes, generated MSIX assets, and the
GUI-relevant license files. It packages that stage through `winapp package`. It
must not ship the CLI, VST3 bundle, TensorRT runtime files, local models, caches,
logs, or developer-machine state.

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
  Windows App SDK Runtime 2.x (minimum 2.1) on the user's machine.
  The shared runtime uses ORT API 24; keep that Cargo API feature and the
  bootstrapper/MSIX minimum runtime requirements aligned when upgrading `ort`.
- TensorRT packages must not contain ONNX Runtime provider DLLs. They contain
  the matching TensorRT and CUDA runtime DLLs; non-runtime-only packages also
  contain the engine-builder helper and the selected GPU builder resources.
- Build VST3 variants package-scoped. Do not use a whole-workspace build whose
  unified features can pull incompatible providers into the plugin.
- Standalone packages enable the pure-Rust `rnnoise` feature. VST3 packages
  must not enable or advertise RNNoise; inspect package-scoped dependency and
  license output to catch accidental feature unification.
- Standalone packages enable `gtcrn` (the 16 kHz GTCRN input denoiser; pulls
  `realfft`/`rustfft`). Windows ML runs GTCRN through ORT CPU; TensorRT keeps
  ONNX Runtime out of the package and runs GTCRN through the native TensorRT
  shim. VST3 (either variant) must not enable or ship `gtcrn`; verify the
  package-scoped feature set and staged files contain no GTCRN code or models.
  GTCRN models are never bundled (fetched via `download-models.ps1 -Gtcrn`).

Start packaging from clean per-variant staging directories. Stale sidecar DLLs
from another package must never survive into a release.

## Licenses

Every archive must include the repository `LICENSE` and all notices or license
texts required by bundled third-party code and redistributable DLLs.

License generation and collection must be checked for the exact package and
feature set being shipped. Treat a missing, stale, or mismatched license notice
as a release blocker. Distribution packaging requires `cargo-about`; ordinary
builds and tests do not.

In particular, verify:

- The Rust dependency notice covers the actual standalone app or VST3 variant.
- Standalone packages contain separate notices for `vc-rs.exe` and `vc-gui.exe`.
- Non-runtime-only TensorRT packages contain a separate notice for the bundled
  `vc-tensorrt-builder.exe`.
- Windows ML packages include the license matching the redistributed Windows
  App SDK bootstrapper.
- TensorRT packages link the official NVIDIA TensorRT SDK License Agreement and
  include the CUDA EULA copied from the selected CUDA install.
- The [`../CHANGELOG.md`](../CHANGELOG.md) has a finalized entry for the version
  being shipped (see [Versioning](#versioning)).

## Versioning

The release version lives in exactly one place: `[workspace.package].version` in
the root [`../Cargo.toml`](../Cargo.toml). Every crate inherits it through
`version.workspace = true`, and the packaging scripts read the same field to name
the archives `vc-rs-<variant>-v<version>-win-x64.zip` (see
[`../crates/vc-cli/package.ps1`](../crates/vc-cli/package.ps1)). Do not hand-edit
versions in individual crate manifests or in the scripts.

The project follows [Semantic Versioning](https://semver.org/). While the API is
pre-1.0 (`0.x`), treat breaking changes to the CLI, VST3 parameters/state, or
package layout as a minor bump and additive changes as a patch bump.

To prepare a release:

1. Bump `[workspace.package].version` in the root `Cargo.toml`.
2. Run a build (`cargo build`) so the bumped version is written back into
   `Cargo.lock`, and commit both files together.
3. Add a finalized [`../CHANGELOG.md`](../CHANGELOG.md)
   `## [X.Y.Z] - <date>` section for the release and refresh the version
   comparison links at the bottom. This project does not keep an `Unreleased`
   section; development-only notes belong in plans, issues, or commit history
   until they are promoted into a release entry.

> The tag `v<version>` and the `v<version>` embedded in each archive name must
> match. A mismatch means the version was bumped after the binaries were built —
> rebuild.

## Pre-Publish Check

[`../scripts/release.ps1`](../scripts/release.ps1) (`just release`) automates the
mechanical gate — steps 1, 3, and 4 below — across all four ZIPs: it confirms the
canonical archives exist (or builds them with `-Build`) and scans each for
prohibited files, backend cross-contamination, missing required files, and
build-machine paths/user names leaked into our own binaries. It treats any
finding as a blocker and refuses to continue. The remaining steps (2, 5, 6) are
manual judgement/runtime checks.

Before publishing each final ZIP:

1. Build it with the appropriate `package.ps1` or `scripts/package-all.ps1`
   command without relying on an unverified `-SkipBuild` artifact. (`release.ps1
   -Build`.)
2. Extract the ZIP into a fresh directory and inspect the actual archived
   contents, not only the staging directory.
3. Confirm required binaries, runtime DLLs, install instructions, and license
   files are present, and prohibited files are absent. (Automated by `release.ps1`.)
4. Search the extracted files and printable binary strings for build-machine
   paths, user names, secrets, and other local state. (Automated by `release.ps1`;
   pass `-ScanPattern` to add project-specific strings.)
5. Smoke-test both executables in the extracted standalone app package. Validate
   the extracted VST3 package
   with the Steinberg validator and, when practical, load it in a clean DAW
   environment.
6. Test on a machine or environment that does not rely on the build machine's
   SDK paths, caches, or environment variables.

GitHub shows a SHA-256 digest for each uploaded release asset, so no separate
checksum files are produced or attached.

## Publish

Once every variant's ZIP has passed the Pre-Publish Check:

1. Confirm the version is bumped, `Cargo.lock` is updated, and the
   [`../CHANGELOG.md`](../CHANGELOG.md) entry for this version is finalized and
   committed (see [Versioning](#versioning)).
2. Run `scripts/release.ps1 -Publish` (`just release -Publish`). It re-runs the
   scan, requires a clean worktree, fetches `origin/main`, and stops unless
   `origin/main` can fast-forward to the current commit. It then creates the
   annotated tag `v<version>`, atomically pushes the current commit to `main`
   together with the tag, and creates the GitHub release with all four ZIPs
   attached (GitHub displays a SHA-256 digest per asset). A concurrent update or
   any non-fast-forward push is rejected before the release is created. The tag
   matches the `v<version>` in the archive names. Use `-Draft` to review the
   release before it goes public. Use `-Remote` or `-ReleaseBranch` only when
   intentionally publishing somewhere other than `origin/main`. The script
   extracts only this version's `CHANGELOG.md` section for the GitHub Release
   body and stops before publishing if that section is missing.
3. Confirm the not-code-signed Windows warning is stated in the release body,
   consistent with the user-facing docs.

Distributed binaries are currently not code-signed. Keep the user-facing
documentation explicit about the resulting Windows warning until signing is
introduced.

## Current Automation Limits

The packaging scripts provide important safeguards, including release stripping,
Rust path remapping, fresh VST3 staging, variant-specific population, exact
per-binary Rust license generation, and required redistributable license
collection. `release.ps1` adds the publish gate: scanning the final ZIPs before
tagging and releasing.

Review these known limitations before release:

- `release.ps1` scans our own binaries (not third-party vendor DLLs) for
  build-machine paths and the current user name, and matches prohibited files by
  name. It does not scan vendor DLLs for arbitrary secrets, so do not stage
  unexpected files into a package staging directory.
- `-SkipBuild` does not prove that the reused binary matches the requested
  backend variant.
- `package-store-msix.ps1` requires the Windows App Development CLI
  (`winget install Microsoft.WinAppCli`) and creates an unsigned MSIX unless
  `-PfxPath` is provided. The default MSIX identity is for local packaging only;
  Store submission must use the package name and publisher reserved in Partner
  Center. The `-CreateTestCertificate` path uses `winapp cert generate` and is
  for local install testing only; it must not be used for Store submission. The
  `-TrustTestRootCertificate` option is a persistent local trust decision and
  must only be used with disposable test certificates. The
  `-TrustTestMachineCertificate` option uses `winapp cert install`, writes
  machine-level trust, and must only be used on local test machines. Store
  metadata, privacy/support URLs, code-signing, and Partner Center validation
  remain release-owner responsibilities.

Do not weaken these safeguards or dismiss these limitations without replacing
them with an equivalent or stronger automated check.
