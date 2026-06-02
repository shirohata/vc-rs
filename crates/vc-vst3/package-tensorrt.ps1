<#
.SYNOPSIS
    Copy the TensorRT runtime DLLs (and, for self-contained first-run engine
    builds, the ORT-free builder helper plus its build-time DLLs) into the built
    TensorRT plugin bundle, so the GPU build runs without a separate TensorRT
    install on the user's PATH.

.DESCRIPTION
    Run AFTER:
        cargo xtask bundle vc-vst3 --release --no-default-features --features tensorrt

    The ONNX Runtime CPU core is statically linked into the plugin and the GPU
    path runs through native TensorRT (no ONNX Runtime CUDA EP, no cuDNN/cuBLAS/
    cuFFT). The plugin binary links `nvinfer_10.dll` / `nvinfer_plugin_10.dll` /
    `cudart` at LOAD time, so those must sit next to the plugin or the DAW fails
    to load it. Windows searches a module's own directory first, so co-locating
    them in Contents\<arch>\ (and beside the .clap) satisfies the import.

    Two layers of dependency:
      * Runtime (plugin load + engine execution): nvinfer_10, nvinfer_plugin_10,
        cudart64_12. Always copied.
      * Engine build (first run, on a cache miss): the ORT-free helper
        `vc-tensorrt-builder.exe` builds engines from the ONNX models via the
        TensorRT builder, which needs nvonnxparser_10 and the
        `nvinfer_builder_resource_sm*_10.dll` matching the user's GPU. Copied
        unless -RuntimeOnly. The plugin's process is the DAW, so point it at the
        bundled helper with VC_RS_TENSORRT_BUILDER_HELPER (printed at the end).

.PARAMETER TensorRtBin
    TensorRT `bin` directory holding nvinfer_10.dll etc. Default:
    %TENSORRT_ROOT%\bin, else the bundled TensorRT folder under the repo root.

.PARAMETER CudaBin
    CUDA Toolkit bin directory (for cudart64_12.dll). Default: %CUDA_PATH%\bin.

.PARAMETER BundleDir
    Directory containing the built bundles. Default: target\bundled.

.PARAMETER BuilderExe
    Path to vc-tensorrt-builder.exe. Default: searched under target\release and
    tools\tensorrt_probe\target\release. Ignored with -RuntimeOnly.

.PARAMETER BuilderSm
    Which GPU builder-resource DLLs to bundle, by SM tag (e.g. sm89 for RTX 40xx,
    sm86 for RTX 30xx, sm75 for RTX 20xx, sm90 Hopper, sm100/sm120 Blackwell, and
    'ptx' for the JIT fallback). Default: all present (full compatibility, ~2.5 GB
    — a size warning is printed). Pass 'none' to skip them. Ignored with
    -RuntimeOnly.

.PARAMETER RuntimeOnly
    Copy only the runtime DLLs (no builder helper, parser, or builder resources).
    Use when engines are prebuilt/cached or built outside the plugin.

.EXAMPLE
    # Self-contained for an RTX 40-series (Ada / sm89) machine:
    pwsh crates\vc-vst3\package-tensorrt.ps1 -BuilderSm sm89

.EXAMPLE
    # Just the runtime DLLs (smallest; engines built elsewhere):
    pwsh crates\vc-vst3\package-tensorrt.ps1 -RuntimeOnly
#>
[CmdletBinding()]
param(
    [string]$TensorRtBin = $(if ($env:TENSORRT_ROOT) { Join-Path $env:TENSORRT_ROOT 'bin' } else { '' }),
    [string]$CudaBin = $(if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'bin' } else { '' }),
    [string]$BundleDir,
    [string]$BuilderExe,
    [string[]]$BuilderSm = @(),
    [switch]$RuntimeOnly
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$licenseSrc = Join-Path $PSScriptRoot 'dist\licenses'

if (-not $TensorRtBin) {
    $TensorRtBin = Join-Path $repoRoot 'TensorRT-10.16.1.11.Windows.amd64.cuda-12.9\TensorRT-10.16.1.11\bin'
}
if (-not $BundleDir) { $BundleDir = Join-Path $repoRoot 'target\bundled' }

function Resolve-Required([string]$path, [string]$what) {
    if (-not $path) { throw "$what is not set." }
    if (-not (Test-Path $path)) { throw "$what not found: $path" }
    return (Resolve-Path $path).Path
}

$TensorRtBin = Resolve-Required $TensorRtBin 'TensorRtBin (set TENSORRT_ROOT or pass -TensorRtBin)'
$CudaBin = Resolve-Required $CudaBin 'CudaBin (set CUDA_PATH or pass -CudaBin)'

# Runtime DLLs the plugin imports at load time and uses to deserialize/run
# engines. These are mandatory: without them the DAW cannot load the plugin.
$runtimeSources = @(
    (Join-Path $TensorRtBin 'nvinfer_10.dll'),
    (Join-Path $TensorRtBin 'nvinfer_plugin_10.dll'),
    (Join-Path $CudaBin 'cudart64_12.dll')
)

# Build-time helper + DLLs for first-run engine construction (cache miss).
$builderSources = @()
$resolvedBuilderExe = $null
if (-not $RuntimeOnly) {
    if (-not $BuilderExe) {
        $candidates = @(
            (Join-Path $repoRoot 'target\release\vc-tensorrt-builder.exe'),
            (Join-Path $repoRoot 'tools\tensorrt_probe\target\release\vc-tensorrt-builder.exe')
        )
        $BuilderExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        if (-not $BuilderExe) {
            throw @"
vc-tensorrt-builder.exe not found. Build it first, e.g.:
    cargo build --release --manifest-path tools\tensorrt_probe\Cargo.toml
or pass -BuilderExe <path>, or use -RuntimeOnly to skip the engine builder.
Searched:
$($candidates -join "`n")
"@
        }
    }
    $resolvedBuilderExe = Resolve-Required $BuilderExe 'BuilderExe (vc-tensorrt-builder.exe)'
    $builderSources += $resolvedBuilderExe
    $builderSources += (Join-Path $TensorRtBin 'nvonnxparser_10.dll')

    # Builder-resource DLLs. These are GPU-architecture specific and very large.
    $allResources = Get-ChildItem -Path $TensorRtBin -Filter 'nvinfer_builder_resource_*_10.dll'
    if ($BuilderSm -contains 'none') {
        Write-Host "Skipping builder-resource DLLs (-BuilderSm none). First-run engine builds will need them on PATH." -ForegroundColor Yellow
    }
    elseif (-not $BuilderSm -or $BuilderSm.Count -eq 0) {
        $builderSources += $allResources.FullName
        $bytes = ($allResources | Measure-Object -Property Length -Sum).Sum
        Write-Host ("WARNING: bundling ALL builder-resource DLLs ({0:N1} GB). Pass -BuilderSm <sm89,...> to ship only your GPU's resource." -f ($bytes / 1GB)) -ForegroundColor Yellow
    }
    else {
        foreach ($sm in $BuilderSm) {
            $name = "nvinfer_builder_resource_${sm}_10.dll"
            $path = Join-Path $TensorRtBin $name
            if (-not (Test-Path $path)) {
                $available = ($allResources.Name | ForEach-Object { ($_ -replace '^nvinfer_builder_resource_', '') -replace '_10\.dll$', '' }) -join ', '
                throw "Builder resource '$name' not found in $TensorRtBin. Available SM tags: $available"
            }
            $builderSources += $path
        }
    }
}

$sources = @($runtimeSources + $builderSources)
$missing = $sources | Where-Object { -not (Test-Path $_) }
if ($missing) { throw "Missing source files:`n" + ($missing -join "`n") }

# Destinations: the VST3 binary folder and the folder next to the .clap.
$dests = @()
$vst3Bin = Join-Path $BundleDir 'vc-vst3.vst3\Contents\x86_64-win'
if (Test-Path $vst3Bin) { $dests += $vst3Bin }
if (Test-Path (Join-Path $BundleDir 'vc-vst3.clap')) { $dests += $BundleDir }
if (-not $dests) {
    throw "No bundle found in $BundleDir. Run 'cargo xtask bundle vc-vst3 --release --no-default-features --features tensorrt' first."
}

foreach ($dest in $dests) {
    Write-Host "Populating $dest"
    foreach ($src in $sources) {
        Copy-Item -Path $src -Destination $dest -Force
    }

    # Licenses next to the DLLs.
    $licDest = Join-Path $dest 'licenses'
    New-Item -ItemType Directory -Force -Path $licDest | Out-Null
    if (Test-Path $licenseSrc) {
        Copy-Item -Path (Join-Path $licenseSrc '*') -Destination $licDest -Recurse -Force
    }

    # TensorRT's own license text from the local install when present.
    $trtRoot = Split-Path $TensorRtBin -Parent
    $trtLic = Get-ChildItem -Path $trtRoot -Recurse -Include 'LICENSE*', '*LICENSE*.txt', 'EULA*' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($trtLic) { Copy-Item $trtLic.FullName (Join-Path $licDest 'TensorRT-LICENSE.txt') -Force }
}

$count = $sources.Count
$total = ($sources | ForEach-Object { (Get-Item $_).Length } | Measure-Object -Sum).Sum
Write-Host ("Done: bundled {0} file(s) ({1:N1} GB) + licenses into {2} location(s)." -f $count, ($total / 1GB), $dests.Count) -ForegroundColor Green

if (-not $RuntimeOnly -and $resolvedBuilderExe) {
    $helperName = Split-Path $resolvedBuilderExe -Leaf
    Write-Host ""
    Write-Host "First-run engine builds use the bundled helper. Because a VST3/CLAP host's" -ForegroundColor Cyan
    Write-Host "process is the DAW (not the plugin), point the plugin at it via env var:" -ForegroundColor Cyan
    Write-Host "    setx VC_RS_TENSORRT_BUILDER_HELPER `"<install-dir>\$helperName`"" -ForegroundColor Cyan
    Write-Host "(set it to the helper's path inside the installed plugin folder)." -ForegroundColor Cyan
}
