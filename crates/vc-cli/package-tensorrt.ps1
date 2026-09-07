<#
.SYNOPSIS
    Copy the TensorRT runtime DLLs (and, for self-contained first-run engine
    builds, the ORT-free builder helper plus its build-time DLLs) next to the
    vc-rs CLI executable, so the GPU build runs without a separate TensorRT
    install on the user's PATH.

.DESCRIPTION
    Run AFTER:
        cargo build --release -p vc-cli --no-default-features --features tensorrt,rnnoise,gtcrn

    The GPU path runs through native TensorRT (no ONNX Runtime CUDA EP, no
    cuDNN/cuBLAS/cuFFT). vc-rs.exe links `nvinfer_<N>.dll` / `nvinfer_plugin_<N>.dll`
    / `cudart` at LOAD time (delay-loaded), and resolves them from its own folder,
    so they must sit beside vc-rs.exe. The TensorRT major version `<N>` (10, 11,
    ...) and the matching `cudart64_<M>.dll` are detected from the chosen install.

    Two layers of dependency:
      * Runtime (engine execution): nvinfer_<N>, nvinfer_plugin_<N>, cudart64_<M>.
        Always copied.
      * Engine build (first run, on a cache miss): the TensorRT builder API can't
        run in a process where ONNX Runtime has already initialized, so engine
        construction is delegated to the ORT-free helper `vc-tensorrt-builder.exe`.
        (This packaged build drops ORT entirely — `--features tensorrt` pulls in
        no ONNX Runtime — but the helper is the shared, robust build path.) The
        CLI auto-discovers it beside its own executable (no env var needed, unlike
        the plugin). The helper needs nvonnxparser_<N> and the
        `nvinfer_builder_resource_sm*_<N>.dll` matching the user's GPU. Copied
        unless -RuntimeOnly.

.PARAMETER DestDir
    Directory holding vc-rs.exe to populate. Default: target\release.

.PARAMETER TensorRtBin
    TensorRT `bin` directory holding nvinfer_<N>.dll etc. Default:
    %TENSORRT_ROOT%\bin, else the newest TensorRT folder under external\nvidia.

.PARAMETER CudaBin
    CUDA Toolkit bin directory (for cudart64_<M>.dll). Default: %CUDA_PATH%\bin
    when its major matches the TensorRT version, else the newest matching CUDA
    toolkit under the standard install directory.

.PARAMETER BuilderExe
    Path to vc-tensorrt-builder.exe. Default: searched under target\release and
    tools\tensorrt_builder\target\release. Ignored with -RuntimeOnly.

.PARAMETER RuntimeOnly
    Copy only the runtime DLLs (no builder helper, parser, or builder resources).
    Use when engines are prebuilt/cached or built outside the CLI.

.EXAMPLE
    # Self-contained package (bundles all GPU builder resources, ~2.5 GB):
    pwsh crates\vc-cli\package-tensorrt.ps1

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
    [switch]$RuntimeOnly
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$licenseSrc = Join-Path $repoRoot 'scripts\licenses\static'
if (-not (Test-Path $licenseSrc)) { throw "Static license material not found: $licenseSrc" }

function Resolve-Required([string]$path, [string]$what) {
    if (-not $path) { throw "$what is not set." }
    if (-not (Test-Path $path)) { throw "$what not found: $path" }
    return (Resolve-Path $path).Path
}

function Find-LicenseText([string]$root) {
    $namePattern = '^(LICENSE|EULA)|LICENSE'
    $direct = Get-ChildItem -Path $root -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match $namePattern } |
        Select-Object -First 1
    if ($direct) { return $direct }
    return Get-ChildItem -Path $root -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match $namePattern } |
        Select-Object -First 1
}

. (Join-Path $repoRoot "scripts\tensorrt-sdk.ps1")

if (-not $DestDir) { $DestDir = Join-Path $repoRoot 'target\release' }
$DestDir = Resolve-Required $DestDir 'DestDir'
if (-not (Test-Path (Join-Path $DestDir 'vc-rs.exe'))) {
    throw "vc-rs.exe not found in $DestDir. Build it first: cargo build --release -p vc-cli --no-default-features --features tensorrt,rnnoise,gtcrn"
}

$sdkSelection = Initialize-TensorRtPackageSdk -RepoRoot $repoRoot -TensorRtBin $TensorRtBin -CudaBin $CudaBin
$TensorRtBin = $sdkSelection.TensorRtBin
$CudaBin = $sdkSelection.CudaBin
$major = $sdkSelection.Sdk.Version.Major
$cudaMajor = Get-TensorRtCudaMajor $major

# CUDA's local EULA is copied into the package. TensorRT's SDK distribution
# terms are linked from THIRD-PARTY-NOTICES.md because NVIDIA's zip packages do
# not consistently include a standalone license file.
$cudaRoot = Split-Path $CudaBin -Parent
if ((Split-Path $CudaBin -Leaf) -eq 'x64') { $cudaRoot = Split-Path $cudaRoot -Parent }
$cudaLic = Find-LicenseText $cudaRoot
if (-not $cudaLic) { throw "CUDA license/EULA not found under $cudaRoot." }

# CUDA 13 moved the redistributable runtime DLLs from <toolkit>\bin into
# <toolkit>\bin\x64; CUDA 12 keeps them directly in bin. Search both.
$cudartDll = @($CudaBin, (Join-Path $CudaBin 'x64')) |
    Where-Object { Test-Path $_ } |
    ForEach-Object { Get-ChildItem -Path $_ -Filter "cudart64_$cudaMajor.dll" -ErrorAction SilentlyContinue } |
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
    $builderSources += (Resolve-Required $BuilderExe 'BuilderExe (vc-tensorrt-builder.exe)')
    $builderSources += (Join-Path $TensorRtBin "nvonnxparser_$major.dll")

    # Builder-resource DLLs. These are GPU-architecture specific and very large;
    # distribution packages bundle every SM tag for full GPU compatibility.
    $allResources = Get-ChildItem -Path $TensorRtBin -Filter "nvinfer_builder_resource_*_$major.dll"
    if (-not $allResources) { throw "No TensorRT builder-resource DLLs found in $TensorRtBin." }
    $builderSources += $allResources.FullName
    $bytes = ($allResources | Measure-Object -Property Length -Sum).Sum
    Write-Host ("Bundling ALL builder-resource DLLs ({0:N1} GB) for full GPU compatibility." -f ($bytes / 1GB)) -ForegroundColor Cyan
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
    Copy-Item -Path (Join-Path $licenseSrc 'THIRD-PARTY-NOTICES.md') -Destination $licDest -Force
}

Copy-Item $cudaLic.FullName (Join-Path $licDest 'CUDA-EULA.txt') -Force

$count = $sources.Count
$total = ($sources | ForEach-Object { (Get-Item $_).Length } | Measure-Object -Sum).Sum
Write-Host ("Done: bundled {0} file(s) ({1:N1} GB) + licenses into {2}." -f $count, ($total / 1GB), $DestDir) -ForegroundColor Green
