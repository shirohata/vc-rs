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

.PARAMETER RuntimeOnly
    Copy only the runtime DLLs (no builder helper, parser, or builder resources).
    Use when engines are prebuilt/cached or built outside the plugin.

.EXAMPLE
    # Self-contained package (bundles all GPU builder resources, ~2.5 GB):
    pwsh crates\vc-vst3\package-tensorrt.ps1

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

if (-not $BundleDir) { $BundleDir = Join-Path $repoRoot 'target\bundled' }
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
        Copy-Item -Path (Join-Path $licenseSrc 'THIRD-PARTY-NOTICES.md') -Destination $licDest -Force
    }

    Copy-Item $cudaLic.FullName (Join-Path $licDest 'CUDA-EULA.txt') -Force
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
