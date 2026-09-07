# No Pester or NVIDIA installation required. Tests the same resolver used by
# activation and packaging against temporary SDKs with intentionally misleading names.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'tensorrt-sdk.ps1')

function Assert-Equal($Actual, $Expected, [string]$Label) {
    if ($Actual -ne $Expected) { throw "$Label`: expected '$Expected', got '$Actual'" }
}
function Assert-Throws([scriptblock]$Action) {
    $threw = $false
    try { & $Action | Out-Null } catch { $threw = $true }
    if (-not $threw) { throw 'Expected an error' }
}
function New-TestSdk([string]$Relative, [string]$Version) {
    $root = Join-Path $fixtureRoot $Relative
    foreach ($dir in @('include', 'lib', 'bin')) { New-Item -ItemType Directory -Path (Join-Path $root $dir) -Force | Out-Null }
    $values = $Version.Split('.')
    $header = for ($i = 0; $i -lt 4; $i++) {
        $name = @('MAJOR', 'MINOR', 'PATCH', 'BUILD')[$i]
        "#define NV_TENSORRT_$name TRT_${name}_ENTERPRISE // alias"
        "#define TRT_${name}_ENTERPRISE $($values[$i])"
    }
    Set-Content -LiteralPath (Join-Path $root 'include\NvInferVersion.h') -Value $header
    foreach ($name in @('NvInfer.h', 'NvInferPlugin.h', 'NvOnnxParser.h')) { Set-Content -LiteralPath (Join-Path $root "include\$name") -Value '' }
    foreach ($name in @('nvinfer', 'nvinfer_plugin', 'nvonnxparser')) {
        foreach ($item in @("lib\${name}_$($values[0]).lib", "bin\${name}_$($values[0]).dll")) { Set-Content -LiteralPath (Join-Path $root $item) -Value '' }
    }
    return $root
}

$fixtureRoot = Join-Path ([IO.Path]::GetTempPath()) "vc-rs-sdk-test-$([guid]::NewGuid())"
$savedEnvironment = @{}
foreach ($name in @('TENSORRT_ROOT', 'CUDA_PATH', 'CUDA_HOME', 'ORT_CUDA_VERSION', 'PATH')) { $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name) }
try {
    $old = New-TestSdk 'external\nvidia\TensorRT-11.0' '11.0.0.114'
    $middle = New-TestSdk 'external\nvidia\TensorRT-11.1' '11.1.0.106'
    $latest = New-TestSdk 'external\nvidia\TensorRT-Enterprise\TensorRT-11.2.1' '11.2.1.2'
    Assert-Equal (Resolve-TensorRtSdk -RepoRoot $fixtureRoot -EnvironmentRoot '').Root $latest 'nested newest SDK'
    Assert-Equal (Resolve-TensorRtSdk -RepoRoot $fixtureRoot -ExplicitRoot $old -EnvironmentRoot $middle).Root $old 'explicit overrides environment'
    Assert-Equal (Resolve-TensorRtSdk -RepoRoot $fixtureRoot -EnvironmentRoot $middle).Root $middle 'environment overrides discovery'
    $numeric = New-TestSdk 'external\TensorRT-11.10' '11.10.0.2'
    New-TestSdk 'external\TensorRT-11.9' '11.9.0.999' | Out-Null
    New-TestSdk 'TensorRT-12' '12.0.0.1' | Out-Null
    Assert-Equal (Resolve-TensorRtSdk -RepoRoot $fixtureRoot -EnvironmentRoot '').Root $numeric 'numeric order and supported major'
    $patch = New-TestSdk 'external\TensorRT-patch' '11.10.1.1'
    Assert-Equal (Resolve-TensorRtSdk -RepoRoot $fixtureRoot -EnvironmentRoot '').Root $patch 'patch order'
    $build = New-TestSdk 'external\TensorRT-build' '11.10.1.10'
    Assert-Equal (Resolve-TensorRtSdk -RepoRoot $fixtureRoot -EnvironmentRoot '').Root $build 'build order'
    Remove-Item -LiteralPath (Join-Path $build 'lib\nvonnxparser_11.lib')
    Assert-Equal (Resolve-TensorRtSdk -RepoRoot $fixtureRoot -EnvironmentRoot '').Root $patch 'incomplete SDK skipped'
    Assert-Throws { Resolve-TensorRtSdk -RepoRoot $fixtureRoot -ExplicitRoot $build }
    $direct = "#define NV_TENSORRT_MAJOR 11`n#define NV_TENSORRT_MINOR (2)`n#define NV_TENSORRT_PATCH 1`n#define NV_TENSORRT_BUILD 2"
    Assert-Equal (ConvertFrom-TensorRtVersionHeader $direct) ([version]'11.2.1.2') 'direct numeric macros'
    Assert-Equal (ConvertFrom-TensorRtVersionHeader ($direct.Replace('NV_TENSORRT_BUILD 2', 'NV_TENSORRT_BUILD NV_TENSORRT_BUILD'))) $null 'cyclic macro'
    Assert-Equal (ConvertFrom-TensorRtVersionHeader ($direct.Replace('NV_TENSORRT_BUILD 2', 'NV_TENSORRT_BUILD 1+1'))) $null 'expression macro'
    Assert-Equal (ConvertFrom-TensorRtVersionHeader '') $null 'missing macros'

    $cudaRoot = Join-Path $fixtureRoot 'CUDA\v13.3'
    foreach ($item in @('include\cuda_runtime_api.h', 'lib\x64\cudart.lib', 'bin\x64\cudart64_13.dll')) {
        $file = Join-Path $cudaRoot $item
        New-Item -ItemType Directory -Path (Split-Path $file -Parent) -Force | Out-Null
        Set-Content -LiteralPath $file -Value ''
    }
    New-Item -ItemType Directory -Path (Join-Path $fixtureRoot 'CUDA\v13.2') -Force | Out-Null
    Assert-Equal (Resolve-TensorRtCudaRoot -Major 13 -EnvironmentRoot '' -EnvironmentHome '' -SearchRoot (Join-Path $fixtureRoot 'CUDA')) $cudaRoot 'CUDA numeric selection'
    Assert-Throws { Resolve-TensorRtCudaRoot -Major 13 -ExplicitRoot (Join-Path $fixtureRoot 'CUDA\v12.9') }
    $env:TENSORRT_ROOT = $old
    $selection = Initialize-TensorRtPackageSdk -RepoRoot $fixtureRoot -TensorRtBin (Join-Path $latest 'bin') -CudaBin (Join-Path $cudaRoot 'bin\x64')
    Assert-Equal $env:TENSORRT_ROOT $latest 'package build SDK'
    Assert-Equal $env:CUDA_PATH $cudaRoot 'package build CUDA'
    Assert-Equal $env:CUDA_HOME $cudaRoot 'consistent CUDA_HOME'
    Assert-Equal $selection.TensorRtBin (Join-Path $latest 'bin') 'package DLL SDK'
    Assert-Equal ($env:PATH.Split(';')[0]) (Join-Path $latest 'bin') 'runtime PATH priority'
    Write-Host 'TensorRT SDK selection tests passed.'
} finally {
    foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name]) }
    # Only remove the uniquely named fixture under the system temp directory.
    $resolvedFixture = [IO.Path]::GetFullPath($fixtureRoot)
    $tempPrefix = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $resolvedFixture.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        (Split-Path $resolvedFixture -Leaf) -notlike 'vc-rs-sdk-test-*') { throw 'Unsafe test cleanup path' }
    if (Test-Path -LiteralPath $resolvedFixture) { Remove-Item -LiteralPath $resolvedFixture -Recurse -Force }
}
