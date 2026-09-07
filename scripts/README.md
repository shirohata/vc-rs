# Build environment setup (Windows)

The recommended GPU baseline is **CUDA 13.3 Update 1 / TensorRT 11.2.1**.
Automatic discovery stays on the CUDA 13 / TensorRT 11 line. Run scripts from
the repo root with `pwsh`.

## First-time setup

1. **winget scope** — `pwsh -File scripts/bootstrap.ps1`
   Installs Rustup, Git, and the MSVC C++ build tools (VS BuildTools VCTools
   workload; needed because the `cc` crate compiles the native TensorRT shim).
   Idempotent. CMake is intentionally NOT required. Use `-Force` to repair.

2. **NVIDIA SDKs** — install CUDA Toolkit **13.3 Update 1** from
   https://developer.nvidia.com/cuda-13-3-1-download-archive. Download the
   **TensorRT 11.2.1 Windows amd64, cuda-13.3 ZIP** from
   https://developer.nvidia.com/tensorrt/download (NVIDIA Developer login required).
   Extract under `external\nvidia\`; a direct SDK root or one wrapping
   `TensorRT-*` directory is supported. Python wheels are not needed for this
   project's native C++ backend. Keep older installs when rollback is needed.
   See NVIDIA's [prerequisites](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/prerequisites.html).

   cuDNN **9.x** is optional for native TensorRT; the separate ORT CUDA backend
   may need it. Obtain it from https://developer.nvidia.com/cudnn-downloads
   (older builds: https://developer.nvidia.com/cudnn-archive).

## Per shell session

```powershell
. scripts/activate.ps1
```

Dot-source it (not a child shell) so the env applies to your session. It puts
the matched CUDA/cuDNN/TensorRT on PATH and sets `CUDA_PATH`, `CUDA_HOME`,
`TENSORRT_ROOT`, and `ORT_CUDA_VERSION`. Selection precedence is explicit
`-TensorRtRoot` / `-CudaPath`, then environment variables, then discovery.
An environment CUDA path from a different major is ignored. `-CuDnnBin` remains
an optional override.

Both Rust build scripts share SDK discovery; activation and packaging share the
PowerShell resolver. They compare all four numeric version components from
`NvInferVersion.h` (including the `TRT_*_ENTERPRISE` aliases), require complete
headers/import libraries/runtime DLLs, and auto-select only TensorRT 11.x.
An invalid explicit SDK fails instead of silently selecting another install.
The selected full version is printed during activation/build/packaging.

When upgrading, explicitly select the new SDK and CUDA once if your shell still
has the old `TENSORRT_ROOT` or `CUDA_PATH`. For example, with the extracted SDK
root assigned to `$sdkRoot`:

```powershell
. scripts/activate.ps1 -TensorRtRoot $sdkRoot -CudaPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
```

Native caches are stored under `native-trt-<major.minor.patch.build>` inside the
existing cache root. An SDK upgrade rebuilds engines and timing caches on first
use; later starts reuse them. Old caches are retained for rollback and remain
visible to the existing cache inspection/clear commands. Windows ML TensorRT RTX
cache paths are unchanged. Rebuild the app/plugin **and** builder helper for each
SDK; copying only new runtime DLLs into an old build is not an upgrade workflow.

SDK-selection regression tests need no GPU or SDK:

```powershell
pwsh -File scripts/test-tensorrt-sdk.ps1
cargo test -p vc-core --test tensorrt_sdk --no-default-features
```

## Optional: ASIO audio backend (`--features asio`)

ASIO support is an opt-in feature on `vc-cli`/`vc-gui` and is **not** part of the
default build, CI, or distribution packages. Compared to a normal build it needs
just **one** extra tool — LLVM:

- **LLVM/Clang (required)** — `asio-sys` runs `bindgen` over the ASIO SDK headers.
  Install LLVM (`winget install LLVM.LLVM`) and point bindgen at `libclang.dll`:
  `$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"`.

The other two pieces are already covered:

- **MSVC C++ build tools** — `asio-sys` compiles the SDK's C++ via `cc`; the same
  toolchain the project already requires (installed by `just setup` / bootstrap),
  so nothing extra to do.
- **Steinberg ASIO SDK** — `asio-sys` **auto-downloads** it from steinberg.net at
  build time when `CPAL_ASIO_DIR` is unset, so a plain build "just works" once LLVM
  is present. Setting `$env:CPAL_ASIO_DIR` to a local SDK copy is optional but
  recommended for offline/CI builds, to pin a version, and to make the Steinberg
  dependency explicit (see licensing note below).

Then build the standalone front-ends with the feature:

```powershell
cargo build -p vc-cli --features asio
cargo build -p vc-gui --features asio
```

> **Licensing:** ASIO is a Steinberg trademark and SDK distributed under
> Steinberg's own license. This repo never commits or redistributes the SDK — the
> auto-download fetches it directly from Steinberg. Before shipping an
> ASIO-enabled build, review Steinberg's terms (pointing `CPAL_ASIO_DIR` at an SDK
> copy you obtained yourself makes that acceptance deliberate).

ASIO devices/drivers are listed with `vc-rs devices --audio-backend asio`, and
the backend is selectable per direction (`--input-backend`/`--output-backend`);
see [`../docs/cli.md`](../docs/cli.md).

## Verify

```powershell
pwsh -File scripts/verify.ps1
```

Runs `cargo test --workspace` then `cargo xtask bundle vc-vst3`. Flags:
- `-Variant tensorrt` — build the TensorRT bundle instead of the default Windows ML one.
- `-SkipBundle` — tests only.
- `-NoNativeTensorRT` — skip the GPU stack and run tests fast.

## Local VST3 validator

For VST3 smoke tests, build Steinberg's command-line validator into the
repository:

```powershell
pwsh -File scripts/install-vst3-validator.ps1
```

This clones the VST3 SDK into `external\steinberg\vst3sdk\`, builds it in
`external\steinberg\vst3sdk-build\`, and leaves the validator at:

```powershell
external\steinberg\vst3sdk-build\bin\Release\validator.exe
```

Use it against the local bundle:

```powershell
pwsh -File scripts/validate-vst3.ps1
```

`validate-vst3.ps1` builds `target\bundled\vc-vst3.vst3` and validates it. If
the validator is missing, it first runs `install-vst3-validator.ps1`.

Useful flags:
- `-Variant tensorrt` — build/validate the TensorRT VST3 variant.
- `-DebugBuild` — validate a debug bundle instead of release.
- `-PopulateRuntime` — copy the variant runtime DLLs into the bundle before
  validation.
- `-NoInstallValidator` — fail instead of auto-building the validator.

For the validator install script itself, pass `-Update` to pull an existing SDK
checkout, or `-CleanBuild` to recreate the CMake build directory.

## Install local VST3 bundle

Copy the local bundle into the per-user Windows VST3 directory
(`%LocalAppData%\Programs\Common\VST3`):

```powershell
pwsh -File scripts/install-vst3-bundle.ps1
```

The install name is variant-specific (`vc-vst3-windowsml.vst3` or
`vc-vst3-tensorrt.vst3`) so both builds can be installed side by side. The VST3
class IDs and display names are also variant-specific.

For the usual development loop, build, validate, then install in one command:

```powershell
pwsh -File scripts/install-vst3-bundle.ps1 -BuildFirst -ValidateFirst
# or
just install-vst3
```

For the machine-wide VST3 directory (`%CommonProgramFiles%\VST3`, usually
`C:\Program Files\Common Files\VST3`), pass `-System`; that may require an
elevated PowerShell session.

For a dry run or alternate test copy, use:

```powershell
pwsh -File scripts/install-vst3-bundle.ps1 -DestinationRoot C:\tmp\VST3 -WhatIf
```

## Package the distributables

The shipped Windows distributions are four packages: `app-windowsml`,
`app-tensorrt`, `vst3-windowsml`, and `vst3-tensorrt`. The app packages contain
both `vc-gui.exe` and `vc-rs.exe`. Each crate's
`package.ps1` builds one (`-Variant windowsml|tensorrt`); `package-all.ps1`
drives all four into `dist\`:

```powershell
. scripts/activate.ps1                 # tensorrt targets need the GPU toolchain
cargo install cargo-about --features cli # one-time packaging prerequisite
pwsh scripts/package-all.ps1
```

Packaging requires `cargo-about` so each staged binary receives a notice for its
exact package and backend feature set. Ordinary builds, tests, validation, and
local install workflows do not require it.

TensorRT packages link the official NVIDIA TensorRT SDK License Agreement from
their third-party notice because NVIDIA's SDK archives do not consistently
include a standalone agreement file.

Alongside each `.zip`, a populated, ready-to-run `dist\<stem>\` folder (binary +
DLLs + licenses) is left in place for quick local testing — kept by default for
the windowsml variants and removed for tensorrt (which can be multiple GB). All
of `dist\` is gitignored.

Flags:
- `-Targets app-windowsml,vst3-windowsml` — build only a subset (e.g. the
  Windows ML pair, which needs no GPU toolchain).
- `-Asio` — build the **app** packages (`vc-rs.exe`/`vc-gui.exe`) with the ASIO
  audio backend. Needs LLVM + the Steinberg ASIO SDK on this machine (see
  *Optional: ASIO audio backend* above); the vst3 targets ignore it. e.g.
  `just package -Targets app-windowsml -Asio` or
  `just package -Asio`. Off by default so the windowsml package keeps needing no
  special toolchain.
- `cli-windowsml` and `cli-tensorrt` remain accepted as legacy aliases for the
  corresponding app targets.
- `-RuntimeOnly` / `-TensorRtBin <dir>` — forwarded to the tensorrt targets (see
  each crate's `package-tensorrt.ps1`). TensorRT packages always bundle every
  GPU builder resource for full compatibility.
- `-KeepStage` / `-CleanStage` — force keeping (e.g. tensorrt) or removing the
  ready-to-run `dist\<stem>\` folders, overriding the per-variant default.
- `-OutDir <dir>` — where the `.zip` files (and kept folders) land (default `dist\`).
- `-ContinueOnError` — keep building after a failure and report a summary.

## Build the Microsoft Store GUI MSIX

Store MSIX packaging uses the Windows App Development CLI:

```powershell
winget install Microsoft.WinAppCli
```

For Store submission as an MSIX package, use:

```powershell
pwsh scripts/package-store-msix.ps1 -BuildPackage `
  -PackageName <Partner-Center-package-identity-name> `
  -Publisher "CN=<Partner-Center-publisher>"
# or:
just store-msix -BuildPackage
```

This consumes only the `app-windowsml` package output, creates a trimmed staging
directory under `tmp\store-msix-stage`, then runs `winapp package` to write:

```text
dist\vc-rs-windowsml-gui-store-v<version>-win-x64.msix
```

The MSIX is unsigned unless `-PfxPath <cert.pfx>` is supplied. Local installation
requires a trusted certificate whose subject matches `-Publisher`. For Partner
Center submission, replace the default `-PackageName` and `-Publisher` values
with the identity reserved for the Store app.

The Store MSIX intentionally excludes `vc-rs.exe`, VST3 bundles, TensorRT runtime
files, local models, caches, logs, and development artifacts.
It declares a Windows App Runtime framework dependency by default:
`Microsoft.WindowsAppRuntime.2`, minimum version `2.1.0.0`. Store installation
can resolve that dependency; local sideloading still requires the runtime package
to be installed or supplied separately.

Useful flags:
- `-EmitOnly` — validate inputs, prepare `tmp\store-msix-stage`, and print the
  `winapp package` command without packaging.
- `-WinApp <path>` — use a specific `winapp.exe`; otherwise PATH/App Execution
  Alias lookup is used.
- `-PfxPath <cert.pfx>` / `-PfxPassword <password>` — sign the MSIX for local
  installation/testing through `winapp package --cert`.
- `-WindowsAppRuntimeDependencyName`, `-WindowsAppRuntimeDependencyMinVersion`,
  `-WindowsAppRuntimeDependencyPublisher` — override the Store framework
  dependency identity.
- `-NoWindowsAppRuntimeDependency` — omit the framework dependency, only for
  experiments.
- `-IncludeWindowsAppRuntimeBootstrap` — also copy the unpackaged-app bootstrap
  DLL into the MSIX; Store builds normally should not need this.
- `-CreateTestCertificate` — create a temporary self-signed code-signing
  certificate under `tmp\store-msix-cert` with `winapp cert generate`, set the
  MSIX publisher to that certificate subject, and sign the MSIX. Existing test
  PFX files are reused.
- `-ForceNewTestCertificate` — replace the existing local test PFX/CER instead
  of reusing it.
- `-TrustTestCertificate` — with `-CreateTestCertificate`, import the public
  certificate into `Cert:\CurrentUser\TrustedPeople`.
- `-TrustTestRootCertificate` — with `-CreateTestCertificate`, also import the
  certificate into `Cert:\CurrentUser\Root`. This is required for local MSIX
  installation with a self-signed certificate and should only be used for
  disposable local test certificates you generated yourself.
- `-TrustTestMachineCertificate` — with `-CreateTestCertificate`, also import
  the certificate into the local machine certificate stores using
  `winapp cert install`. Run from an elevated PowerShell session; use this when
  `Add-AppxPackage` still reports `0x800B0109` after the signature validates for
  the current user.

For a one-command local installable test package:

```powershell
pwsh scripts/package-store-msix.ps1 `
  -CreateTestCertificate `
  -TrustTestCertificate `
  -TrustTestRootCertificate
```

If `Get-AuthenticodeSignature` reports `Valid` but `Add-AppxPackage` still fails
with `0x800B0109`, rebuild from an elevated PowerShell session with machine-level
trust:

```powershell
pwsh scripts/package-store-msix.ps1 `
  -CreateTestCertificate `
  -TrustTestCertificate `
  -TrustTestRootCertificate `
  -TrustTestMachineCertificate
```

The generated certificate is only for local testing. Do not use it for Store
submission; Partner Center builds should pass the Store-reserved `-PackageName`
and `-Publisher`.

## A/B audio-quality comparison

To check whether a change altered the converted audio, compare two
versions/builds on the same clip through the **deterministic CPU `wav` path**
(the CPU execution provider is reproducible run-to-run, unlike the GPU EPs):

```powershell
$env:VC_RS_TEST_RVC_MODEL = 'C:\models\voice.onnx'   # RVC model is not shipped
just models                                           # fetch ContentVec / RMVPE
just compare-audio -RefA main -RefB dev -InputWav clip.wav
```

Each `-RefA` / `-RefB` may be a **git ref** (built in a temp worktree with
`--features cpu` and run), a **built `vc-rs.exe`**, or an **existing `.wav`**
(used as-is — comparing two finished outputs needs no model). Both conversions
force `--provider cpu`. The outputs are scored by `tools/audio_compare`
(max abs diff, relative RMS, log-spectral distance in dB); the script exits
non-zero if any metric exceeds its threshold (`-MaxAbs` / `-MaxRelRms` /
`-MaxLsdDb`).

Comparing against a commit older than the `cpu` feature: build that side with a
feature it has, e.g. `-FeaturesA windowsml` (it still runs with `--provider cpu`).

`tools/audio_compare` is a workspace-excluded dev tool; test it on its own with
`cargo test --manifest-path tools/audio_compare/Cargo.toml`.

## Gotcha: STATUS_DLL_NOT_FOUND

Test exes link the native TensorRT shim, so the TensorRT bin must be on PATH
(via `activate.ps1`) or they fail to launch with `STATUS_DLL_NOT_FOUND`. To run
tests without a GPU stack, set `VC_RS_ENABLE_NATIVE_TENSORRT=0` (or use
`scripts/verify.ps1 -NoNativeTensorRT`).
