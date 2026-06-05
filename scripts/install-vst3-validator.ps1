<#
.SYNOPSIS
    Build Steinberg's VST3 validator into this repository for local plugin tests.

.DESCRIPTION
    Clones the official VST3 SDK under external\steinberg\vst3sdk, configures a
    local CMake build under external\steinberg\vst3sdk-build, and builds the
    command-line validator.

    The SDK checkout and build tree are intentionally kept under external\ so
    downloaded third-party source and generated binaries remain separate from
    repository-owned tools.

.PARAMETER Update
    If the SDK checkout already exists, pull it and update submodules before
    building.

.PARAMETER CleanBuild
    Delete and recreate the local CMake build directory before configuring.

.PARAMETER Config
    CMake build configuration. Release is the default because validator is only
    used as a host-side conformance tool, not as code we debug.

.PARAMETER Generator
    CMake generator to use. If omitted, the newest available Visual Studio
    generator reported by CMake is selected.

.EXAMPLE
    pwsh -File scripts/install-vst3-validator.ps1
    pwsh -File scripts/install-vst3-validator.ps1 -Update -CleanBuild
#>

[CmdletBinding()]
param(
    [switch]$Update,
    [switch]$CleanBuild,

    [ValidateSet('Debug', 'Release', 'RelWithDebInfo')]
    [string]$Config = 'Release',

    [string]$Generator,

    [string]$Repository = 'https://github.com/steinbergmedia/vst3sdk.git'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$steinbergRoot = Join-Path $repoRoot 'external\steinberg'
$sdkDir = Join-Path $steinbergRoot 'vst3sdk'
$buildDir = Join-Path $steinbergRoot 'vst3sdk-build'

function Test-CommandPresent([string]$Name) {
    [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Resolve-CommandPath {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [string[]]$Fallbacks = @()
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    foreach ($fallback in $Fallbacks) {
        if (Test-Path -LiteralPath $fallback) { return $fallback }
    }

    return $null
}

function Invoke-Step {
    param([string]$Label, [scriptblock]$Action)

    Write-Host ''
    Write-Host "==> $Label" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed (exit $LASTEXITCODE)"
    }
}

function Remove-LocalBuildDir([string]$Path) {
    $resolvedRoot = (Resolve-Path -LiteralPath $repoRoot).Path
    $fullPath = [System.IO.Path]::GetFullPath($Path)

    # Guardrail for future edits: this script has a destructive cleanup mode.
    # Keep deletion constrained to the repository-local CMake build directory so
    # an accidentally empty or user-supplied path cannot remove unrelated files.
    if ($fullPath -ne [System.IO.Path]::GetFullPath($buildDir)) {
        throw "Refusing to remove unexpected path: $fullPath"
    }
    if (-not $fullPath.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the repository: $fullPath"
    }

    if (Test-Path -LiteralPath $fullPath) {
        Remove-Item -LiteralPath $fullPath -Recurse -Force
    }
}

function Resolve-CmakeGenerator([string]$CmakePath, [string]$RequestedGenerator) {
    if ($RequestedGenerator) { return $RequestedGenerator }

    $help = & $CmakePath --help
    $visualStudioGenerators = $help |
        Select-String -Pattern '^\s*\*?\s*(Visual Studio \d+ \d+)\s+=' |
        ForEach-Object { $_.Matches[0].Groups[1].Value }

    if ($visualStudioGenerators) {
        return $visualStudioGenerators[0]
    }

    throw "No Visual Studio CMake generator found. Install Visual Studio Build Tools with the C++ workload."
}

if (-not (Test-CommandPresent 'git')) {
    throw "git is required. Run scripts/bootstrap.ps1 or install Git, then re-run."
}
$cmake = Resolve-CommandPath 'cmake' @(
    "${env:ProgramFiles}\CMake\bin\cmake.exe",
    "${env:ProgramFiles(x86)}\CMake\bin\cmake.exe"
)
if (-not $cmake) {
    throw "cmake is required to build the VST3 SDK validator."
}
$cmakeGenerator = Resolve-CmakeGenerator $cmake $Generator
New-Item -ItemType Directory -Force -Path $steinbergRoot | Out-Null

Write-Host "== VST3 validator local install ==" -ForegroundColor Cyan
Write-Host "SDK:   $sdkDir"
Write-Host "Build: $buildDir"
Write-Host "CMake: $cmake"
Write-Host "Gen:   $cmakeGenerator"

Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath $sdkDir)) {
        Invoke-Step 'clone VST3 SDK' {
            git clone --recursive $Repository $sdkDir
        }
    } elseif (-not (Test-Path -LiteralPath (Join-Path $sdkDir '.git'))) {
        throw "SDK path exists but is not a git checkout: $sdkDir"
    } elseif ($Update) {
        Invoke-Step 'update VST3 SDK checkout' {
            git -C $sdkDir pull --ff-only
        }
        Invoke-Step 'update VST3 SDK submodules' {
            git -C $sdkDir submodule update --init --recursive
        }
    } else {
        Write-Host ''
        Write-Host "[skip] SDK checkout already exists. Pass -Update to pull latest." -ForegroundColor Green
    }

    if ($CleanBuild) {
        Invoke-Step 'clean VST3 SDK build directory' {
            Remove-LocalBuildDir $buildDir
        }
    }

    Invoke-Step 'configure VST3 SDK validator build' {
        & $cmake -S $sdkDir -B $buildDir `
            -G $cmakeGenerator `
            -A x64 `
            -DSMTG_CREATE_PLUGIN_LINK=0
    }

    Invoke-Step "build validator ($Config)" {
        & $cmake --build $buildDir --config $Config --target validator
    }

    $validator = Join-Path $buildDir "bin\$Config\validator.exe"
    if (-not (Test-Path -LiteralPath $validator)) {
        $matches = Get-ChildItem -LiteralPath $buildDir -Recurse -Filter validator.exe -ErrorAction SilentlyContinue
        if ($matches.Count -eq 1) {
            $validator = $matches[0].FullName
        } else {
            throw "validator.exe was built but not found at the expected path: $validator"
        }
    }

    Write-Host ''
    Write-Host '== VST3 validator ready ==' -ForegroundColor Green
    Write-Host $validator -ForegroundColor Green
    Write-Host ''
    Write-Host 'Example:' -ForegroundColor Magenta
    Write-Host "  & '$validator' '.\target\bundled\vc-vst3.vst3'"
}
finally {
    Pop-Location
}
