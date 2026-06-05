<#
.SYNOPSIS
    Verify the vc-rs build environment by running the test suite and bundling
    the VST3 plugin.

.DESCRIPTION
    Activates the matched CUDA/cuDNN/TensorRT line (via scripts/activate.ps1),
    then runs `cargo test` and `cargo xtask bundle vc-vst3`. A clean pass means
    the toolchain, MSVC C++, and the GPU SDKs are wired up correctly.

    By default it builds the Windows ML plugin variant. Use -Variant tensorrt for
    the TensorRT-only build, or -SkipBundle to only run tests.

    Tests link the native TensorRT shim (via the CLI's `tensorrt` feature), so
    the TensorRT bin must be on PATH or the test exes fail to launch with
    STATUS_DLL_NOT_FOUND. activate.ps1 handles that. To run tests fast without a
    GPU stack, pass -NoNativeTensorRT (sets VC_RS_ENABLE_NATIVE_TENSORRT=0).

.PARAMETER Variant
    Plugin bundle variant to build: 'windowsml' (default) or 'tensorrt'.

.EXAMPLE
    pwsh -File scripts/verify.ps1
    pwsh -File scripts/verify.ps1 -Variant tensorrt
    pwsh -File scripts/verify.ps1 -SkipBundle -NoNativeTensorRT
#>

[CmdletBinding()]
param(
    [ValidateSet('windowsml', 'tensorrt')]
    [string]$Variant = 'windowsml',

    [switch]$SkipBundle,

    # Drop the native TensorRT shim link so test exes launch without GPU DLLs.
    [switch]$NoNativeTensorRT
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

# Keep absolute build-machine paths (user name etc.) out of every binary this
# builds, and keep CARGO_ENCODED_RUSTFLAGS identical to the build/test recipes so
# they share one build cache instead of rebuilding on each switch.
. (Join-Path $PSScriptRoot 'rustflags.ps1')

function Invoke-Step {
    param([string]$Label, [scriptblock]$Action)
    Write-Host ""
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed (exit $LASTEXITCODE)"
    }
}

# --- Activate the matching GPU stack ----------------------------------------
if (-not $NoNativeTensorRT) {
    . (Join-Path $PSScriptRoot "activate.ps1")
} else {
    Write-Host "== Native TensorRT shim disabled (VC_RS_ENABLE_NATIVE_TENSORRT=0) ==" -ForegroundColor Yellow
    $env:VC_RS_ENABLE_NATIVE_TENSORRT = "0"
}

Push-Location $repoRoot
try {
    # --- Tests --------------------------------------------------------------
    Invoke-Step "cargo test --workspace" { cargo test --workspace }

    # --- Bundle the plugin --------------------------------------------------
    if (-not $SkipBundle) {
        if ($Variant -eq 'tensorrt') {
            Invoke-Step "cargo xtask bundle vc-vst3 (tensorrt)" {
                cargo xtask bundle vc-vst3 --release --no-default-features --features tensorrt
            }
        } else {
            Invoke-Step "cargo xtask bundle vc-vst3 (windowsml)" {
                cargo xtask bundle vc-vst3 --release
            }
        }
    }

    Write-Host ""
    Write-Host "== verify: OK ==" -ForegroundColor Green
    if (-not $SkipBundle) {
        $bundled = Join-Path $repoRoot "target\bundled"
        Write-Host "Bundle output: $bundled" -ForegroundColor Green
        Get-ChildItem $bundled -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match 'vc-vst3' } |
            ForEach-Object { Write-Host "  $($_.Name)" }
    }
}
finally {
    Pop-Location
}
