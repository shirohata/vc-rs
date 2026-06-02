<#
.SYNOPSIS
    Copy the minimal ONNX Runtime CUDA execution-provider DLLs (plus their CUDA
    and cuDNN dependencies) and license files into the built plugin bundle, so
    the GPU build runs without a separate CUDA/cuDNN install on the user's PATH.

.DESCRIPTION
    Run AFTER `cargo xtask bundle vc-vst3 --release`. The ONNX Runtime core is
    statically linked into the plugin, so only the provider DLLs and the CUDA /
    cuDNN runtime DLLs are needed. They are placed next to the plugin binary
    (VST3: Contents\x86_64-win\, CLAP: beside the .clap) and the plugin adds its
    own directory discoverable for bundled DLLs at startup (see src/dll_path.rs).

.PARAMETER ProvidersDir
    Directory holding the ort-downloaded provider DLLs. Default: target\release.

.PARAMETER CudaBin
    CUDA Toolkit bin directory. Default: %CUDA_PATH%\bin.

.PARAMETER CudnnBin
    cuDNN bin directory (the folder containing cudnn64_9.dll). Default:
    $env:VC_RS_CUDNN_BIN.

.PARAMETER BundleDir
    Directory containing the built bundles. Default: target\bundled.

.PARAMETER OrtCudaVersion
    ONNX Runtime CUDA package major version. Default: ORT_CUDA_VERSION or 12.
    This package currently supports 12 only because it bundles CUDA 12.x DLLs.

.EXAMPLE
    pwsh crates\vc-vst3\package-cuda.ps1 `
        -CudaBin "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin" `
        -CudnnBin "C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64"
#>
[CmdletBinding()]
param(
    [string]$ProvidersDir,
    [string]$CudaBin = $(if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'bin' } else { '' }),
    [string]$CudnnBin = $env:VC_RS_CUDNN_BIN,
    [string]$BundleDir,
    [string]$OrtCudaVersion = $(if ($env:ORT_CUDA_VERSION) { $env:ORT_CUDA_VERSION } else { '12' })
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$licenseSrc = Join-Path $PSScriptRoot 'dist\licenses'

if (-not $ProvidersDir) { $ProvidersDir = Join-Path $repoRoot 'target\release' }
if (-not $BundleDir) { $BundleDir = Join-Path $repoRoot 'target\bundled' }

function Resolve-Required([string]$path, [string]$what) {
    if (-not $path) { throw "$what is not set." }
    if (-not (Test-Path $path)) { throw "$what not found: $path" }
    return (Resolve-Path $path).Path
}

$ProvidersDir = Resolve-Required $ProvidersDir 'ProvidersDir (ort provider DLLs)'
$CudaBin = Resolve-Required $CudaBin 'CudaBin (set CUDA_PATH or pass -CudaBin)'
$CudnnBin = Resolve-Required $CudnnBin 'CudnnBin (set VC_RS_CUDNN_BIN or pass -CudnnBin)'

function Assert-Cuda12Runtime([string]$path, [string]$what) {
    if ($path -match '(?i)(\\|/|^)(v?13(\.|\\|/)|cuda[-_ ]?13|.*_cuda13)') {
        throw "$what appears to be CUDA 13.x, but this package is built with ORT_CUDA_VERSION=12: $path"
    }
}

if ($OrtCudaVersion -ne '12') {
    throw "Unsupported OrtCudaVersion=$OrtCudaVersion. Rebuild/package VST3 with ORT_CUDA_VERSION=12 for the CUDA 12.x DLL set."
}
Assert-Cuda12Runtime $CudaBin 'CudaBin'
Assert-Cuda12Runtime $CudnnBin 'CudnnBin'

# Minimal DLL set for the ONNX Runtime CUDA execution provider.
$providerDlls = @(
    'onnxruntime_providers_shared.dll',
    'onnxruntime_providers_cuda.dll'
)
$cudaDlls = @(
    'cudart64_12.dll',
    'cublas64_12.dll',
    'cublasLt64_12.dll',
    'cufft64_11.dll'
)

# Gather the source files. cuDNN 9 splits into cudnn64_9.dll + sub-libraries
# that it dlopens, so take every cudnn*64_9.dll.
$sources = @()
foreach ($d in $providerDlls) { $sources += Join-Path $ProvidersDir $d }
foreach ($d in $cudaDlls) { $sources += Join-Path $CudaBin $d }
$sources += (Get-ChildItem -Path $CudnnBin -Filter 'cudnn*64_9.dll').FullName

$missing = $sources | Where-Object { -not (Test-Path $_) }
if ($missing) { throw "Missing source DLLs:`n" + ($missing -join "`n") }

if (-not (Get-ChildItem -Path $CudnnBin -Filter 'cudnn64_9.dll')) {
    throw "CudnnBin must contain cuDNN 9 runtime DLLs: $CudnnBin"
}

# Destinations: the VST3 binary folder and the folder next to the .clap.
$dests = @()
$vst3Bin = Join-Path $BundleDir 'vc-vst3.vst3\Contents\x86_64-win'
if (Test-Path $vst3Bin) { $dests += $vst3Bin }
if (Test-Path (Join-Path $BundleDir 'vc-vst3.clap')) { $dests += $BundleDir }
if (-not $dests) { throw "No bundle found in $BundleDir. Run 'cargo xtask bundle vc-vst3 --release' first." }

foreach ($dest in $dests) {
    Write-Host "Populating $dest"
    foreach ($src in $sources) {
        Copy-Item -Path $src -Destination $dest -Force
    }

    # Licenses next to the DLLs.
    $licDest = Join-Path $dest 'licenses'
    New-Item -ItemType Directory -Force -Path $licDest | Out-Null
    Copy-Item -Path (Join-Path $licenseSrc '*') -Destination $licDest -Recurse -Force

    # Copy NVIDIA's own license texts from the local installs when present.
    $cudaEula = Get-ChildItem -Path (Split-Path $CudaBin -Parent) -Recurse -Include 'EULA.txt', 'LICENSE' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cudaEula) { Copy-Item $cudaEula.FullName (Join-Path $licDest 'CUDA-EULA.txt') -Force }
    $cudnnLic = Get-ChildItem -Path (Split-Path $CudnnBin -Parent) -Recurse -Include 'LICENSE*', '*LICENSE*.txt' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cudnnLic) { Copy-Item $cudnnLic.FullName (Join-Path $licDest 'cuDNN-LICENSE.txt') -Force }
}

$count = $sources.Count
Write-Host "Done: bundled $count DLL(s) + licenses into $($dests.Count) location(s)." -ForegroundColor Green
