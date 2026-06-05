<#
.SYNOPSIS
    Copy the TensorRT runtime DLLs (and, for self-contained first-run engine
    builds, the ORT-free builder helper plus its build-time DLLs) into the built
    TensorRT plugin bundle, so the GPU build runs without a separate TensorRT
    install on the user's PATH.

.DESCRIPTION
    Run AFTER:
        cargo xtask bundle vc-vst3 --release --no-default-features --features tensorrt

    The TensorRT-only build drops ONNX Runtime entirely (`--features tensorrt`
    pulls in no ORT) and runs the GPU path through native TensorRT (no ONNX
    Runtime CUDA EP, no cuDNN/cuBLAS/cuFFT). The plugin binary links
    `nvinfer_<N>.dll` / `nvinfer_plugin_<N>.dll` /
    `cudart` at LOAD time, so those must sit next to the plugin or the DAW fails
    to load it. Windows searches a module's own directory first, so co-locating
    them in Contents\<arch>\ satisfies the import. The TensorRT major version
    `<N>` (10, 11, ...) and the matching `cudart64_<M>.dll`
    are detected from the chosen install rather than hardcoded.

    Two layers of dependency:
      * Runtime (plugin load + engine execution): nvinfer_<N>, nvinfer_plugin_<N>,
        cudart64_<M>. Always copied.
      * Engine build (first run, on a cache miss): the ORT-free helper
        `vc-tensorrt-builder.exe` builds engines from the ONNX models via the
        TensorRT builder, which needs nvonnxparser_<N> and the
        `nvinfer_builder_resource_sm*_<N>.dll` matching the user's GPU. Copied
        unless -RuntimeOnly. The plugin finds the helper automatically because it
        is co-located with the plugin DLL (the plugin resolves it relative to its
        own module directory, not the DAW exe). VC_RS_TENSORRT_BUILDER_HELPER is
        only needed to override that path.

.PARAMETER TensorRtBin
    TensorRT `bin` directory holding nvinfer_<N>.dll etc. Default:
    %TENSORRT_ROOT%\bin, else the newest TensorRT folder under external\nvidia.

.PARAMETER CudaBin
    CUDA Toolkit bin directory (for cudart64_<M>.dll). Default: %CUDA_PATH%\bin
    when its major matches the TensorRT version, else the newest matching CUDA
    toolkit under the standard install directory.

.PARAMETER BundleDir
    Directory containing the built bundles. Default: target\bundled.

.PARAMETER BundleName
    Name of the .vst3 bundle folder inside BundleDir to populate. Default
    vc-vst3.vst3 (the raw xtask output). package.ps1 passes the variant-specific
    staged name (e.g. vc-vst3-tensorrt.vst3) so it can populate the per-variant
    staging copy instead of the shared target\bundled.

.PARAMETER BuilderExe
    Path to vc-tensorrt-builder.exe. Default: searched under target\release and
    tools\tensorrt_builder\target\release. Ignored with -RuntimeOnly.

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
    [string]$CudaBin = '',
    [string]$BundleDir,
    [string]$BundleName = 'vc-vst3.vst3',
    [string]$BuilderExe,
    [string[]]$BuilderSm = @(),
    [switch]$RuntimeOnly
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$licenseSrc = Join-Path $PSScriptRoot 'dist\licenses'

function Resolve-Required([string]$path, [string]$what) {
    if (-not $path) { throw "$what is not set." }
    if (-not (Test-Path $path)) { throw "$what not found: $path" }
    return (Resolve-Path $path).Path
}

# Detect the TensorRT major version (10, 11, ...) from nvinfer_<N>.dll in a bin dir.
function Get-NvinferMajor([string]$binDir) {
    if (-not (Test-Path $binDir)) { return $null }
    $dll = Get-ChildItem -Path $binDir -Filter 'nvinfer_*.dll' -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^nvinfer_(\d+)\.dll$' } |
        Sort-Object { [int]($_.Name -replace '^nvinfer_(\d+)\.dll$', '$1') } -Descending |
        Select-Object -First 1
    if (-not $dll) { return $null }
    return [int]($dll.Name -replace '^nvinfer_(\d+)\.dll$', '$1')
}

# Newest TensorRT bin (highest nvinfer_<N>.dll) under external\nvidia. A
# TensorRT folder is either the install root itself or wraps a single
# TensorRT-* subdir. The repo root remains as a fallback for older local trees.
function Find-NewestTensorRtBin([string]$root) {
    $best = $null; $bestMajor = -1
    $searchRoots = @(
        (Join-Path $root 'external\nvidia'),
        (Join-Path $root 'external'),
        $root
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -Unique

    foreach ($searchRoot in $searchRoots) {
        foreach ($dir in (Get-ChildItem -Path $searchRoot -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match 'TensorRT' })) {
            $candidates = @($dir.FullName)
            $candidates += (Get-ChildItem -Path $dir.FullName -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -like 'TensorRT-*' } | ForEach-Object { $_.FullName })
            foreach ($c in $candidates) {
                $major = Get-NvinferMajor (Join-Path $c 'bin')
                if ($null -ne $major -and $major -gt $bestMajor) {
                    $bestMajor = $major; $best = (Join-Path $c 'bin')
                }
            }
        }
    }
    return $best
}

# Parse the major from a CUDA toolkit dir named like v13.2.
function Get-CudaDirMajor([string]$path) {
    if ((Split-Path $path -Leaf) -match '^[vV](\d+)\.(\d+)$') { return [int]$Matches[1] }
    return $null
}

# CUDA bin matching $cudaMajor: %CUDA_PATH% when its major matches, else the
# newest matching toolkit under the standard install directory.
function Find-CudaBin([int]$cudaMajor) {
    if ($env:CUDA_PATH -and (Get-CudaDirMajor $env:CUDA_PATH) -eq $cudaMajor) {
        return (Join-Path $env:CUDA_PATH 'bin')
    }
    $base = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    $dir = Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue |
        Where-Object { (Get-CudaDirMajor $_.FullName) -eq $cudaMajor } |
        Sort-Object { [int]($_.Name -replace '^[vV]\d+\.(\d+)$', '$1') } -Descending |
        Select-Object -First 1
    if ($dir) { return (Join-Path $dir.FullName 'bin') }
    return $null
}

if (-not $TensorRtBin) {
    $TensorRtBin = Find-NewestTensorRtBin $repoRoot
    if (-not $TensorRtBin) {
        throw "No TensorRT install found under $repoRoot\external\nvidia. Set TENSORRT_ROOT or pass -TensorRtBin."
    }
}
if (-not $BundleDir) { $BundleDir = Join-Path $repoRoot 'target\bundled' }

$TensorRtBin = Resolve-Required $TensorRtBin 'TensorRtBin (set TENSORRT_ROOT or pass -TensorRtBin)'

# TensorRT major drives every versioned DLL name; CUDA major is paired per
# NVIDIA's support matrix (TRT10 -> CUDA12, TRT11 -> CUDA13).
$major = Get-NvinferMajor $TensorRtBin
if ($null -eq $major) { throw "Could not find nvinfer_<N>.dll in $TensorRtBin" }
$cudaMajor = if ($major -eq 10) { 12 } elseif ($major -eq 11) { 13 } else { $major + 2 }

if (-not $CudaBin) {
    $CudaBin = Find-CudaBin $cudaMajor
    if (-not $CudaBin) {
        throw "No CUDA $cudaMajor.x toolkit found (needed for TensorRT $major). Set CUDA_PATH or pass -CudaBin."
    }
}
$CudaBin = Resolve-Required $CudaBin 'CudaBin (set CUDA_PATH or pass -CudaBin)'

# CUDA 13 moved the redistributable runtime DLLs from <toolkit>\bin into
# <toolkit>\bin\x64; CUDA 12 keeps them directly in bin. Search both.
$cudartDll = @($CudaBin, (Join-Path $CudaBin 'x64')) |
    Where-Object { Test-Path $_ } |
    ForEach-Object { Get-ChildItem -Path $_ -Filter 'cudart64_*.dll' -ErrorAction SilentlyContinue } |
    Select-Object -First 1
if (-not $cudartDll) { throw "No cudart64_*.dll found under $CudaBin (checked .\ and .\x64)" }

# Runtime DLLs the plugin imports at load time and uses to deserialize/run
# engines. These are mandatory: without them the DAW cannot load the plugin.
$runtimeSources = @(
    (Join-Path $TensorRtBin "nvinfer_$major.dll"),
    (Join-Path $TensorRtBin "nvinfer_plugin_$major.dll"),
    $cudartDll.FullName
)

# Build-time helper + DLLs for first-run engine construction (cache miss).
$builderSources = @()
$resolvedBuilderExe = $null
if (-not $RuntimeOnly) {
    if (-not $BuilderExe) {
        $candidates = @(
            (Join-Path $repoRoot 'target\release\vc-tensorrt-builder.exe'),
            (Join-Path $repoRoot 'tools\tensorrt_builder\target\release\vc-tensorrt-builder.exe')
        )
        $BuilderExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        if (-not $BuilderExe) {
            throw @"
vc-tensorrt-builder.exe not found. Build it first, e.g.:
    cargo build --release --manifest-path tools\tensorrt_builder\Cargo.toml
or pass -BuilderExe <path>, or use -RuntimeOnly to skip the engine builder.
Searched:
$($candidates -join "`n")
"@
        }
    }
    $resolvedBuilderExe = Resolve-Required $BuilderExe 'BuilderExe (vc-tensorrt-builder.exe)'
    $builderSources += $resolvedBuilderExe
    $builderSources += (Join-Path $TensorRtBin "nvonnxparser_$major.dll")

    # Builder-resource DLLs. These are GPU-architecture specific and very large.
    $allResources = Get-ChildItem -Path $TensorRtBin -Filter "nvinfer_builder_resource_*_$major.dll"
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
            $name = "nvinfer_builder_resource_${sm}_$major.dll"
            $path = Join-Path $TensorRtBin $name
            if (-not (Test-Path $path)) {
                $available = ($allResources.Name | ForEach-Object { ($_ -replace '^nvinfer_builder_resource_', '') -replace "_$major\.dll$", '' }) -join ', '
                throw "Builder resource '$name' not found in $TensorRtBin. Available SM tags: $available"
            }
            $builderSources += $path
        }
    }
}

$sources = @($runtimeSources + $builderSources)
$missing = $sources | Where-Object { -not (Test-Path $_) }
if ($missing) { throw "Missing source files:`n" + ($missing -join "`n") }

# Destination: the VST3 binary folder.
$dests = @()
$vst3Bin = Join-Path $BundleDir "$BundleName\Contents\x86_64-win"
if (Test-Path $vst3Bin) { $dests += $vst3Bin }
if (-not $dests) {
    throw "No bundle '$BundleName' found in $BundleDir. Run 'cargo xtask bundle vc-vst3 --release --no-default-features --features tensorrt' first."
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
    Write-Host "First-run engine builds use the bundled helper ($helperName). The plugin" -ForegroundColor Cyan
    Write-Host "discovers it automatically because it sits next to the plugin DLL, so no" -ForegroundColor Cyan
    Write-Host "env var or PATH setup is required. Override only if you relocate the helper:" -ForegroundColor Cyan
    Write-Host "    setx VC_RS_TENSORRT_BUILDER_HELPER `"<path>\$helperName`"" -ForegroundColor Cyan
}
