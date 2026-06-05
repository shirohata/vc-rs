<#
.SYNOPSIS
    Install the locally built vc-vst3 bundle into the standard VST3 directory.

.DESCRIPTION
    Copies target\bundled\vc-vst3.vst3 to the per-user Windows VST3 plugin
    directory (%LocalAppData%\Programs\Common\VST3 by default). Use -BuildFirst
    and -ValidateFirst for a single build/validate/install command during plugin
    development.

    Pass -System to install into %CommonProgramFiles%\VST3 instead, which may
    require an elevated PowerShell session. For test copies, pass
    -DestinationRoot.

.PARAMETER Variant
    Plugin bundle variant to build when -BuildFirst or -ValidateFirst is used:
    'windowsml' (default) or 'tensorrt'.

.PARAMETER BuildFirst
    Build target\bundled\vc-vst3.vst3 before installing. If -ValidateFirst is
    also supplied, validate-vst3.ps1 performs the build.

.PARAMETER ValidateFirst
    Build and run Steinberg's validator before installing.

.PARAMETER PopulateRuntime
    Forwarded to validate-vst3.ps1 when -ValidateFirst is used. Copies the
    variant runtime DLLs into the bundle before validation/install.

.PARAMETER DestinationRoot
    Root VST3 directory. Defaults to %LocalAppData%\Programs\Common\VST3.

.PARAMETER System
    Install to %CommonProgramFiles%\VST3 instead of the per-user VST3 directory.

.EXAMPLE
    pwsh -File scripts/install-vst3-bundle.ps1
    pwsh -File scripts/install-vst3-bundle.ps1 -BuildFirst -ValidateFirst
    pwsh -File scripts/install-vst3-bundle.ps1 -System
    pwsh -File scripts/install-vst3-bundle.ps1 -DestinationRoot C:\tmp\VST3 -WhatIf
#>

[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [ValidateSet('windowsml', 'tensorrt')]
    [string]$Variant = 'windowsml',

    [switch]$BuildFirst,
    [switch]$ValidateFirst,
    [switch]$PopulateRuntime,

    [string]$DestinationRoot,

    [switch]$System,
    [switch]$NoActivate
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

# Scrub absolute build-machine paths (user name etc.) from the installed plugin,
# matching the other recipes' CARGO_ENCODED_RUSTFLAGS (shared build cache).
. (Join-Path $PSScriptRoot 'rustflags.ps1')
$bundle = Join-Path $repoRoot 'target\bundled\vc-vst3.vst3'
$destinationName = "vc-vst3-$Variant.vst3"
if (-not $DestinationRoot) {
    if ($System) {
        if (-not $env:CommonProgramFiles) {
            throw "CommonProgramFiles is not set; pass -DestinationRoot explicitly."
        }
        $DestinationRoot = Join-Path $env:CommonProgramFiles 'VST3'
    } else {
        if (-not $env:LocalAppData) {
            throw "LocalAppData is not set; pass -DestinationRoot explicitly."
        }
        $DestinationRoot = Join-Path $env:LocalAppData 'Programs\Common\VST3'
    }
}
$destination = Join-Path $DestinationRoot $destinationName

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

function Assert-SafeBundleDestination([string]$Root, [string]$Path, [string]$ExpectedName) {
    $fullRoot = [System.IO.Path]::GetFullPath($Root)
    $fullPath = [System.IO.Path]::GetFullPath($Path)

    # Guardrail for future edits: install replaces the whole .vst3 directory so
    # stale sidecar DLLs cannot survive between backend variants. Keep deletion
    # limited to the expected bundle path inside the chosen VST3 root.
    if ((Split-Path -Leaf $fullPath) -ne $ExpectedName) {
        throw "Refusing to install to an unexpected bundle name: $fullPath"
    }
    if (-not $fullPath.StartsWith($fullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to install outside the VST3 root: $fullPath"
    }
}

Push-Location $repoRoot
try {
    if ($ValidateFirst) {
        $validateArgs = @{ Variant = $Variant }
        if ($PopulateRuntime) { $validateArgs.PopulateRuntime = $true }
        if ($NoActivate) { $validateArgs.NoActivate = $true }
        Invoke-Step 'build and validate VST3 bundle' {
            & (Join-Path $PSScriptRoot 'validate-vst3.ps1') @validateArgs
        }
    } elseif ($BuildFirst) {
        if (-not $NoActivate) {
            Invoke-Step 'activate build environment' {
                . (Join-Path $PSScriptRoot 'activate.ps1')
            }
        }

        if ($Variant -eq 'tensorrt') {
            Invoke-Step 'cargo xtask bundle vc-vst3 (tensorrt)' {
                cargo xtask bundle vc-vst3 --release --no-default-features --features tensorrt
            }
        } else {
            Invoke-Step 'cargo xtask bundle vc-vst3 (windowsml)' {
                cargo xtask bundle vc-vst3 --release
            }
        }
    }

    if (-not (Test-Path -LiteralPath $bundle)) {
        throw "VST3 bundle not found: $bundle. Run scripts/validate-vst3.ps1 or pass -BuildFirst."
    }

    Assert-SafeBundleDestination $DestinationRoot $destination $destinationName

    Invoke-Step 'install VST3 bundle' {
        if ($PSCmdlet.ShouldProcess($destination, "replace with $bundle")) {
            New-Item -ItemType Directory -Path $DestinationRoot -Force | Out-Null
            if (Test-Path -LiteralPath $destination) {
                Remove-Item -LiteralPath $destination -Recurse -Force
            }
            Copy-Item -LiteralPath $bundle -Destination $destination -Recurse -Force
        }
    }

    Write-Host ''
    Write-Host '== VST3 install: OK ==' -ForegroundColor Green
    Write-Host "Source:      $bundle" -ForegroundColor Green
    Write-Host "Destination: $destination" -ForegroundColor Green
}
finally {
    Pop-Location
}
