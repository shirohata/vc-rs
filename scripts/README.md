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
     the repo root; `crates/vc-core/build.rs` auto-discovers the newest
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
- `-Variant tensorrt` — build the TensorRT-only bundle instead of CUDA.
- `-SkipBundle` — tests only.
- `-NoNativeTensorRT` — skip the GPU stack and run tests fast.

## Gotcha: STATUS_DLL_NOT_FOUND

Test exes link the native TensorRT shim, so the TensorRT bin must be on PATH
(via `activate.ps1`) or they fail to launch with `STATUS_DLL_NOT_FOUND`. To run
tests without a GPU stack, set `VC_RS_ENABLE_NATIVE_TENSORRT=0` (or use
`scripts/verify.ps1 -NoNativeTensorRT`).
