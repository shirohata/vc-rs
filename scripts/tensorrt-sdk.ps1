# Shared by activation and both package variants. Keep version parsing and
# candidate validation aligned with build_support/tensorrt_sdk.rs. Dot-sourcing
# this file only defines functions; it must not change the caller's environment.

function ConvertFrom-TensorRtVersionHeader([string]$Text) {
    $definitions = @{}
    foreach ($line in ($Text -split '\r?\n')) {
        if ($line -match '^\s*#\s*define\s+(\w+)\s+(\S+)') {
            $definitions[$Matches[1]] = $Matches[2]
        }
    }
    $parts = @()
    foreach ($name in @('MAJOR', 'MINOR', 'PATCH', 'BUILD')) {
        $token = "NV_TENSORRT_$name"
        $number = $null
        for ($depth = 0; $depth -lt 16; $depth++) {
            $token = $token.Trim('(', ')')
            if ($token -match '^\d+$') {
                $number = 0
                if (-not [int]::TryParse($token, [ref]$number)) { return $null }
                break
            }
            if (-not $definitions.ContainsKey($token)) { return $null }
            $token = $definitions[$token]
        }
        if ($null -eq $number) { return $null }
        $parts += $number
    }
    return [version]($parts -join '.')
}

function Get-TensorRtSdk([string]$Root) {
    $header = Join-Path $Root 'include\NvInferVersion.h'
    if (-not (Test-Path -LiteralPath $header -PathType Leaf)) { return $null }
    $version = ConvertFrom-TensorRtVersionHeader (Get-Content -LiteralPath $header -Raw)
    if ($null -eq $version) { return $null }
    foreach ($name in @('NvInfer.h', 'NvInferPlugin.h', 'NvOnnxParser.h')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Root "include\$name") -PathType Leaf)) { return $null }
    }
    foreach ($name in @('nvinfer', 'nvinfer_plugin', 'nvonnxparser')) {
        foreach ($item in @("lib\${name}_$($version.Major).lib", "bin\${name}_$($version.Major).dll")) {
            if (-not (Test-Path -LiteralPath (Join-Path $Root $item) -PathType Leaf)) { return $null }
        }
    }
    [pscustomobject]@{ Root = (Resolve-Path -LiteralPath $Root).Path; Version = $version }
}

function Resolve-TensorRtSdk {
    param([string]$RepoRoot, [string]$ExplicitRoot, [string]$EnvironmentRoot = $env:TENSORRT_ROOT)
    $selected = if ($ExplicitRoot) { $ExplicitRoot } else { $EnvironmentRoot }
    if ($selected) {
        $sdk = Get-TensorRtSdk $selected
        if (-not $sdk) { throw "Incomplete TensorRT SDK at '$selected'; expected headers, import libraries and runtime DLLs." }
        return $sdk
    }
    $best = $null
    foreach ($searchRoot in @((Join-Path $RepoRoot 'external\nvidia'), (Join-Path $RepoRoot 'external'), $RepoRoot)) {
        if (-not (Test-Path -LiteralPath $searchRoot -PathType Container)) { continue }
        foreach ($dir in (Get-ChildItem -LiteralPath $searchRoot -Directory | Sort-Object Name | Where-Object { $_.Name -match 'TensorRT' })) {
            $candidates = @($dir.FullName) + @(Get-ChildItem -LiteralPath $dir.FullName -Directory |
                Sort-Object Name | Where-Object { $_.Name -like 'TensorRT-*' } | Select-Object -ExpandProperty FullName)
            foreach ($candidate in $candidates) {
                $sdk = Get-TensorRtSdk $candidate
                if ($sdk -and $sdk.Version.Major -eq 11 -and ($null -eq $best -or $sdk.Version -gt $best.Version)) { $best = $sdk }
            }
        }
    }
    return $best
}

function Get-TensorRtCudaMajor([int]$TensorRtMajor) {
    switch ($TensorRtMajor) {
        10 { return 12 }
        11 { return 13 }
        default { throw "Unsupported TensorRT major $TensorRtMajor; update the CUDA mapping explicitly." }
    }
}

function Get-CudaDirMajor([string]$Path) {
    if ((Split-Path $Path -Leaf) -match '^[vV](\d+)\.(\d+)$') { return [int]$Matches[1] }
    return $null
}

function Resolve-TensorRtCudaRoot {
    param([int]$Major, [string]$ExplicitRoot,
        [string]$EnvironmentRoot = $env:CUDA_PATH, [string]$EnvironmentHome = $env:CUDA_HOME,
        [string]$SearchRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA')
    if ($ExplicitRoot) {
        if ((Get-CudaDirMajor $ExplicitRoot) -ne $Major) { throw "CUDA path '$ExplicitRoot' must select CUDA $Major.x." }
        return $ExplicitRoot
    }
    foreach ($candidate in @($EnvironmentRoot, $EnvironmentHome)) {
        if ($candidate -and (Get-CudaDirMajor $candidate) -eq $Major) { return $candidate }
    }
    if (-not (Test-Path -LiteralPath $SearchRoot)) { return $null }
    Get-ChildItem -LiteralPath $SearchRoot -Directory |
        Where-Object { (Get-CudaDirMajor $_.FullName) -eq $Major } |
        Sort-Object { [int]($_.Name -replace '^[vV]\d+\.', '') } -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}

function Get-CudaRootFromBin([string]$Bin) {
    $root = Split-Path $Bin -Parent
    if ((Split-Path $Bin -Leaf) -eq 'x64') { $root = Split-Path $root -Parent }
    return $root
}

# Resolve BEFORE cargo build, not just when copying DLLs. A -TensorRtBin override
# used only at staging can silently pair old headers/helper with a new runtime.
function Initialize-TensorRtPackageSdk {
    param([string]$RepoRoot, [string]$TensorRtBin, [string]$CudaBin)
    $explicitRoot = if ($TensorRtBin) { Split-Path $TensorRtBin -Parent } else { '' }
    $sdk = Resolve-TensorRtSdk -RepoRoot $RepoRoot -ExplicitRoot $explicitRoot
    if (-not $sdk) { throw 'No complete TensorRT 11 SDK found. Set TENSORRT_ROOT or pass -TensorRtBin.' }
    $cudaMajor = Get-TensorRtCudaMajor $sdk.Version.Major
    $explicitCuda = if ($CudaBin) { Get-CudaRootFromBin $CudaBin } else { '' }
    $cudaRoot = Resolve-TensorRtCudaRoot -Major $cudaMajor -ExplicitRoot $explicitCuda
    if (-not $cudaRoot) { throw "No CUDA $cudaMajor.x toolkit found. Set CUDA_PATH." }
    foreach ($item in @('include\cuda_runtime_api.h', 'lib\x64\cudart.lib')) {
        if (-not (Test-Path -LiteralPath (Join-Path $cudaRoot $item) -PathType Leaf)) { throw "Incomplete CUDA toolkit at '$cudaRoot': missing $item" }
    }
    $env:TENSORRT_ROOT = $sdk.Root
    $env:CUDA_PATH = $cudaRoot
    $env:CUDA_HOME = $cudaRoot
    $env:ORT_CUDA_VERSION = "$cudaMajor"
    $dirs = @((Join-Path $sdk.Root 'bin'), (Join-Path $cudaRoot 'bin\x64'), (Join-Path $cudaRoot 'bin')) |
        Where-Object { Test-Path -LiteralPath $_ -PathType Container }
    $env:PATH = (@($dirs) + @($env:PATH -split ';' | Where-Object { $_ -and $_ -notin $dirs })) -join ';'
    Write-Host "[tensorrt] $($sdk.Version) ($($sdk.Root)); CUDA ($cudaRoot)"
    [pscustomobject]@{ Sdk = $sdk; TensorRtBin = (Join-Path $sdk.Root 'bin'); CudaBin = (Join-Path $cudaRoot 'bin') }
}
