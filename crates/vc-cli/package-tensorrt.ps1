<#
.SYNOPSIS
    Copy the TensorRT runtime DLLs (and, for self-contained first-run engine
    builds, the ORT-free builder helper plus its build-time DLLs) next to the
    vc-rs CLI executable, so the GPU build runs without a separate TensorRT
    install on the user's PATH.

.DESCRIPTION
    Run AFTER:
        cargo build --release -p vc-cli --no-default-features --features tensorrt

    The GPU path runs through native TensorRT (no ONNX Runtime CUDA EP, no
    cuDNN/cuBLAS/cuFFT). vc-rs.exe links `nvinfer_<N>.dll` / `nvinfer_plugin_<N>.dll`
    / `cudart` at LOAD time (delay-loaded), and resolves them from its own folder,
    so they must sit beside vc-rs.exe. The TensorRT major version `<N>` (10, 11,
    ...) and the matching `cudart64_<M>.dll` are detected from the chosen install.

    Two layers of dependency:
      * Runtime (engine execution): nvinfer_<N>, nvinfer_plugin_<N>, cudart64_<M>.
        Always copied.
      * Engine build (first run, on a cache miss): vc-rs links ONNX Runtime for its
        Windows ML / CPU providers, and ORT cannot share a process with the
        TensorRT builder, so engine construction is delegated to the ORT-free
        helper `vc-tensorrt-builder.exe`. The CLI auto-discovers it beside its own
        executable (no env var needed, unlike the plugin). The helper needs
        nvonnxparser_<N> and the `nvinfer_builder_resource_sm*_<N>.dll` matching
        the user's GPU. Copied unless -RuntimeOnly.

.PARAMETER DestDir
    Directory holding vc-rs.exe to populate. Default: target\release.

.PARAMETER TensorRtBin
    TensorRT `bin` directory holding nvinfer_<N>.dll etc. Default:
    %TENSORRT_ROOT%\bin, else the newest TensorRT folder under the repo root.

.PARAMETER CudaBin
    CUDA Toolkit bin directory (for cudart64_<M>.dll). Default: %CUDA_PATH%\bin
    when its major matches the TensorRT version, else the newest matching CUDA
    toolkit under the standard install directory.

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
    Use when engines are prebuilt/cached or built outside the CLI.

.EXAMPLE
    # Self-contained for an RTX 40-series (Ada / sm89) machine:
    pwsh crates\vc-cli\package-tensorrt.ps1 -BuilderSm sm89

.EXAMPLE
    # Just the runtime DLLs (smallest; engines built elsewhere):
    pwsh crates\vc-cli\package-tensorrt.ps1 -RuntimeOnly
#>
[CmdletBinding()]
param(
    [string]$DestDir,
    [string]$TensorRtBin = $(if ($env:TENSORRT_ROOT) { Join-Path $env:TENSORRT_ROOT 'bin' } else { '' }),
    [string]$CudaBin = '',
    [string]$BuilderExe,
    [string[]]$BuilderSm = @(),
    [switch]$RuntimeOnly
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$licenseSrc = Join-Path (Join-Path $PSScriptRoot '..\vc-vst3') 'dist\licenses'

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

# Newest TensorRT bin (highest nvinfer_<N>.dll) under the repo root. A TensorRT
# folder is either the install root itself or wraps a single TensorRT-* subdir.
function Find-NewestTensorRtBin([string]$root) {
    $best = $null; $bestMajor = -1
    foreach ($dir in (Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
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

if (-not $DestDir) { $DestDir = Join-Path $repoRoot 'target\release' }
$DestDir = Resolve-Required $DestDir 'DestDir'
if (-not (Test-Path (Join-Path $DestDir 'vc-rs.exe'))) {
    throw "vc-rs.exe not found in $DestDir. Build it first: cargo build --release -p vc-cli --no-default-features --features tensorrt"
}

if (-not $TensorRtBin) {
    $TensorRtBin = Find-NewestTensorRtBin $repoRoot
    if (-not $TensorRtBin) {
        throw "No TensorRT install found under $repoRoot. Set TENSORRT_ROOT or pass -TensorRtBin."
    }
}
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

# Runtime DLLs vc-rs imports at load time and uses to deserialize/run engines.
$runtimeSources = @(
    (Join-Path $TensorRtBin "nvinfer_$major.dll"),
    (Join-Path $TensorRtBin "nvinfer_plugin_$major.dll"),
    $cudartDll.FullName
)

# Build-time helper + DLLs for first-run engine construction (cache miss).
$builderSources = @()
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
    $builderSources += (Resolve-Required $BuilderExe 'BuilderExe (vc-tensorrt-builder.exe)')
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

Write-Host "Populating $DestDir"
foreach ($src in $sources) {
    Copy-Item -Path $src -Destination $DestDir -Force
}

# Licenses next to the DLLs.
$licDest = Join-Path $DestDir 'licenses'
New-Item -ItemType Directory -Force -Path $licDest | Out-Null
if (Test-Path $licenseSrc) {
    Copy-Item -Path (Join-Path $licenseSrc '*') -Destination $licDest -Recurse -Force
}

# TensorRT's own license text from the local install when present.
$trtRoot = Split-Path $TensorRtBin -Parent
$trtLic = Get-ChildItem -Path $trtRoot -Recurse -Include 'LICENSE*', '*LICENSE*.txt', 'EULA*' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($trtLic) { Copy-Item $trtLic.FullName (Join-Path $licDest 'TensorRT-LICENSE.txt') -Force }

$count = $sources.Count
$total = ($sources | ForEach-Object { (Get-Item $_).Length } | Measure-Object -Sum).Sum
Write-Host ("Done: bundled {0} file(s) ({1:N1} GB) + licenses into {2}." -f $count, ($total / 1GB), $DestDir) -ForegroundColor Green
