# Build environment setup (Windows)

Only the **CUDA 13 / TensorRT 11** line is supported. Run scripts from the repo
root with `pwsh`.

## First-time setup

1. **winget scope** — `pwsh -File scripts/bootstrap.ps1`
   Installs Rustup, Git, and the MSVC C++ build tools (VS BuildTools VCTools
   workload; needed because the `cc` crate compiles the native TensorRT shim).
   Idempotent. CMake is intentionally NOT required. Use `-Force` to repair.

2. **Login-gated NVIDIA SDKs (manual)** — not scriptable here; downloading
   requires an NVIDIA Developer login and EULA acceptance, so do this yourself.
   - CUDA Toolkit **v13.2** — https://developer.nvidia.com/cuda-toolkit-archive
   - cuDNN **v9.x** — https://developer.nvidia.com/cudnn-downloads
     (older builds: https://developer.nvidia.com/cudnn-archive)
   - TensorRT **11** — https://developer.nvidia.com/tensorrt
     (downloads: https://developer.nvidia.com/tensorrt-download). Extract under
     `external\nvidia\`; `crates/vc-core/build.rs` auto-discovers the newest
     `TensorRT-*` folder there.

## Per shell session

```powershell
. scripts/activate.ps1
```

Dot-source it (not a child shell) so the env applies to your session. It puts
the matched CUDA/cuDNN/TensorRT on PATH and sets `CUDA_PATH`, `TENSORRT_ROOT`,
`ORT_CUDA_VERSION`. Auto-discovers paths; override with `-CudaPath` /
`-TensorRtRoot` / `-CuDnnBin`.

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
pwsh scripts/package-all.ps1 -BuilderSm sm86
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
- `cli-windowsml` and `cli-tensorrt` remain accepted as legacy aliases for the
  corresponding app targets.
- `-BuilderSm <sm..>` / `-RuntimeOnly` / `-TensorRtBin <dir>` — forwarded to the
  tensorrt targets (see each crate's `package-tensorrt.ps1`).
- `-KeepStage` / `-CleanStage` — force keeping (e.g. tensorrt) or removing the
  ready-to-run `dist\<stem>\` folders, overriding the per-variant default.
- `-OutDir <dir>` — where the `.zip` files (and kept folders) land (default `dist\`).
- `-ContinueOnError` — keep building after a failure and report a summary.

## Gotcha: STATUS_DLL_NOT_FOUND

Test exes link the native TensorRT shim, so the TensorRT bin must be on PATH
(via `activate.ps1`) or they fail to launch with `STATUS_DLL_NOT_FOUND`. To run
tests without a GPU stack, set `VC_RS_ENABLE_NATIVE_TENSORRT=0` (or use
`scripts/verify.ps1 -NoNativeTensorRT`).
