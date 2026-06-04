<#
.SYNOPSIS
    Build the vc-vst3 bundle and run Steinberg's VST3 validator.

.DESCRIPTION
    One-shot local VST3 smoke test:
      1. Ensures the repository-local VST3 validator exists.
      2. Dot-sources scripts\activate.ps1 unless -NoActivate is passed.
      3. Builds target\bundled\vc-vst3.vst3 via cargo xtask bundle.
      4. Runs validator.exe against the fresh bundle.

    By default this validates the raw bundle produced by nice-plug-xtask. That
    is the stable conformance check: the plugin has no model config, so it stays
    silent and avoids provider/model initialization during validator's process
    tests. Use -PopulateRuntime when you specifically want to validate the
    bundle after copying the Windows ML bootstrapper or TensorRT runtime DLLs.

.PARAMETER Variant
    Plugin bundle variant to build: 'windowsml' (default) or 'tensorrt'.

.PARAMETER DebugBuild
    Build a debug bundle instead of the release bundle.

.PARAMETER PopulateRuntime
    Run the variant's runtime populate script before validation. Windows ML
    copies the Windows App SDK bootstrapper; TensorRT copies runtime DLLs only.

.PARAMETER ValidatorPath
    Path to validator.exe. Defaults to the repository-local SDK build output.

.PARAMETER NoInstallValidator
    Fail if validator.exe is missing instead of building it locally.

.PARAMETER NoActivate
    Skip dot-sourcing scripts\activate.ps1.

.EXAMPLE
    pwsh -File scripts/validate-vst3.ps1
    pwsh -File scripts/validate-vst3.ps1 -Variant tensorrt -PopulateRuntime
#>

[CmdletBinding()]
param(
    [ValidateSet('windowsml', 'tensorrt')]
    [string]$Variant = 'windowsml',

    [switch]$DebugBuild,
    [switch]$PopulateRuntime,

    [string]$ValidatorPath,

    [switch]$NoInstallValidator,
    [switch]$NoActivate,

    [ValidateSet('Debug', 'Release', 'RelWithDebInfo')]
    [string]$ValidatorConfig = 'Release'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $ValidatorPath) {
    $ValidatorPath = Join-Path $repoRoot "tools\vst3sdk-build\bin\$ValidatorConfig\validator.exe"
}
$bundle = Join-Path $repoRoot 'target\bundled\vc-vst3.vst3'

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

Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath $ValidatorPath)) {
        if ($NoInstallValidator) {
            throw "validator.exe not found: $ValidatorPath"
        }
        Invoke-Step 'install local VST3 validator' {
            & (Join-Path $PSScriptRoot 'install-vst3-validator.ps1') -Config $ValidatorConfig
        }
    }

    if (-not $NoActivate) {
        Invoke-Step 'activate build environment' {
            . (Join-Path $PSScriptRoot 'activate.ps1')
        }
    }

    [string[]]$profileArgs = @()
    if (-not $DebugBuild) {
        $profileArgs += '--release'
    }
    if ($Variant -eq 'tensorrt') {
        Invoke-Step 'cargo xtask bundle vc-vst3 (tensorrt)' {
            cargo xtask bundle vc-vst3 @profileArgs --no-default-features --features tensorrt
        }
    } else {
        Invoke-Step 'cargo xtask bundle vc-vst3 (windowsml)' {
            cargo xtask bundle vc-vst3 @profileArgs
        }
    }

    if ($PopulateRuntime) {
        if ($Variant -eq 'tensorrt') {
            Invoke-Step 'populate TensorRT runtime DLLs' {
                & (Join-Path $repoRoot 'crates\vc-vst3\package-tensorrt.ps1') -RuntimeOnly
            }
        } else {
            Invoke-Step 'populate Windows ML bootstrapper' {
                & (Join-Path $repoRoot 'crates\vc-vst3\package-windowsml.ps1')
            }
        }
    }

    if (-not (Test-Path -LiteralPath $bundle)) {
        throw "VST3 bundle was not produced: $bundle"
    }

    Invoke-Step 'run VST3 validator' {
        & $ValidatorPath $bundle
    }

    Write-Host ''
    Write-Host '== VST3 validate: OK ==' -ForegroundColor Green
    Write-Host "Bundle:    $bundle" -ForegroundColor Green
    Write-Host "Validator: $ValidatorPath" -ForegroundColor Green
}
finally {
    Pop-Location
}
