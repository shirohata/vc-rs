<#
.SYNOPSIS
    Activate the vc-rs GPU build/run environment for the current shell session.

.DESCRIPTION
    Dot-source this script to put a matched CUDA Toolkit, cuDNN, and TensorRT
    line at the FRONT of PATH and to export the env vars the build and runtime
    rely on (CUDA_PATH, TENSORRT_ROOT, ORT_CUDA_VERSION).

    The pieces are version-coupled, mirroring crates/vc-core/build.rs:
    TensorRT 11 <-> CUDA 13 (cuDNN bin\13.2). Components are auto-discovered;
    override any of them with -CudaPath / -TensorRtRoot / -CuDnnBin.

    Supersedes tmp/env.ps1 (which hardcoded paths and used a $use12 toggle).

.EXAMPLE
    . .\scripts\activate.ps1               # CUDA 13 / TensorRT 11 line
#>

[CmdletBinding()]
param(
    [string]$CudaPath,
    [string]$TensorRtRoot,
    [string]$CuDnnBin
)

# Dot-sourced: never `exit` (it would kill the caller's shell). Use `return`.
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

# Fixed CUDA 13 / TensorRT 11 line (CUDA 12 / TensorRT 10 support was dropped).
$CudaMajor = 13
$trtMajor = 11

function Add-PathFirst([string]$Dir) {
    if (-not (Test-Path -LiteralPath $Dir)) {
        Write-Warning "PATH entry not found, skipping: $Dir"
        return
    }
    $resolved = (Resolve-Path -LiteralPath $Dir).Path
    # Avoid piling up duplicates on repeated activation in the same session.
    $parts = $env:PATH -split ';' | Where-Object { $_ -and $_ -ne $resolved }
    $env:PATH = (@($resolved) + $parts) -join ';'
}

# --- Discover the CUDA Toolkit ($CudaMajor.x, newest minor) ------------------
function Find-CudaToolkit([int]$Major) {
    $base = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (-not (Test-Path $base)) { return $null }
    Get-ChildItem $base -Directory |
        Where-Object { $_.Name -match "^v$Major\.(\d+)$" } |
        Sort-Object { [int]($_.Name -replace "^v$Major\.", '') } -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}

# --- Discover cuDNN bin matching the CUDA toolkit's major.minor --------------
function Find-CuDnnBin([string]$CudaToolkit) {
    $base = "C:\Program Files\NVIDIA\CUDNN"
    if (-not (Test-Path $base) -or -not $CudaToolkit) { return $null }
    # cuDNN nests per CUDA line, e.g. CUDNN\v9.22\bin\13.2\x64.
    $cudaVer = Split-Path $CudaToolkit -Leaf       # "v13.2"
    $line = $cudaVer.TrimStart('v', 'V')           # "13.2"
    Get-ChildItem $base -Directory |
        Sort-Object Name -Descending |             # newest cuDNN first
        ForEach-Object { Join-Path $_.FullName "bin\$line\x64" } |
        Where-Object { Test-Path $_ } |
        Select-Object -First 1
}

# --- Discover the TensorRT root with nvinfer_<trtMajor> ----------------------
# Mirrors build.rs: a candidate is a dir holding include/ and lib/nvinfer_<N>.lib;
# the install may be the dir itself or a nested TensorRT-* subdir.
function Find-TensorRtRoot([int]$Major) {
    $candidates = @()
    $searchRoots = @(
        (Join-Path $repoRoot 'external\nvidia'),
        (Join-Path $repoRoot 'external'),
        $repoRoot
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -Unique

    foreach ($searchRoot in $searchRoots) {
        foreach ($dir in (Get-ChildItem $searchRoot -Directory | Where-Object { $_.Name -match '(?i)tensorrt' })) {
            $candidates += $dir.FullName
            $candidates += (Get-ChildItem $dir.FullName -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '(?i)^tensorrt-' } | Select-Object -ExpandProperty FullName)
        }
    }
    foreach ($root in $candidates) {
        if ((Test-Path (Join-Path $root "include")) -and
            (Test-Path (Join-Path $root "lib\nvinfer_$Major.lib"))) {
            return $root
        }
    }
    return $null
}

# --- Resolve (explicit override wins over discovery) ------------------------
if (-not $CudaPath)     { $CudaPath     = Find-CudaToolkit $CudaMajor }
if (-not $TensorRtRoot) { $TensorRtRoot = Find-TensorRtRoot $trtMajor }
if (-not $CuDnnBin)     { $CuDnnBin     = Find-CuDnnBin $CudaPath }

Write-Host "== Activating vc-rs env: CUDA $CudaMajor / TensorRT $trtMajor ==" -ForegroundColor Cyan

# --- CUDA -------------------------------------------------------------------
if ($CudaPath) {
    $env:CUDA_PATH = $CudaPath
    $env:ORT_CUDA_VERSION = "$CudaMajor"
    Add-PathFirst (Join-Path $CudaPath "bin")
    # CUDA 13 split native DLLs into bin\x64; harmless to add when absent.
    Add-PathFirst (Join-Path $CudaPath "bin\x64")
    Write-Host "[cuda]     $CudaPath" -ForegroundColor Green
} else {
    Write-Warning "CUDA $CudaMajor toolkit not found."
}

# --- cuDNN ------------------------------------------------------------------
if ($CuDnnBin) {
    Add-PathFirst $CuDnnBin
    Write-Host "[cudnn]    $CuDnnBin" -ForegroundColor Green
} else {
    Write-Warning "cuDNN bin for CUDA $CudaMajor not found."
}

# --- TensorRT ---------------------------------------------------------------
if ($TensorRtRoot) {
    $env:TENSORRT_ROOT = $TensorRtRoot
    Add-PathFirst (Join-Path $TensorRtRoot "bin")
    Add-PathFirst (Join-Path $TensorRtRoot "lib")
    Write-Host "[tensorrt] $TensorRtRoot" -ForegroundColor Green
} else {
    Write-Warning "TensorRT $trtMajor root not found under $repoRoot\external\nvidia."
}

Write-Host "Done. (CUDA_PATH / TENSORRT_ROOT / ORT_CUDA_VERSION set; PATH updated)" -ForegroundColor Cyan
