<#
.SYNOPSIS
    Build every distributed vc-rs Windows package in one run.

.DESCRIPTION
    The shipped Windows distributions are four packages:

        app-windowsml    crates\vc-cli\package.ps1
        app-tensorrt     crates\vc-cli\package.ps1   -Variant tensorrt
        vst3-windowsml   crates\vc-vst3\package.ps1
        vst3-tensorrt    crates\vc-vst3\package.ps1  -Variant tensorrt

    Each per-crate package.ps1 is already a full build -> populate -> zip pipeline;
    this just drives all four (or a chosen subset) so `dist\` ends up with every
    archive in one command. The per-crate scripts are called as-is — see them for
    the underlying cargo + populate steps and the .zip naming.

    Toolchain note: the two tensorrt targets compile native code and need the
    CUDA/TensorRT toolchain reachable (dot-source scripts\activate.ps1 first).
    The windowsml targets need no GPU toolchain. If your environment only has one
    stack set up, use -Targets to build just the matching pair.

.PARAMETER Targets
    Which packages to build. Default: all four. Accepts any of
    app-windowsml, app-tensorrt, vst3-windowsml, vst3-tensorrt.
    Legacy cli-windowsml / cli-tensorrt names remain accepted as aliases.

.PARAMETER OutDir
    Where to write the .zip files. Default: <repo>\dist. Forwarded to each
    package.ps1.

.PARAMETER RuntimeOnly
    (tensorrt targets) Bundle only the runtime DLLs (no engine builder).
    Forwarded to both tensorrt packages.

.PARAMETER TensorRtBin
    (tensorrt targets) TensorRT bin directory. Forwarded to both tensorrt
    packages. Default: auto-detected (see package-tensorrt.ps1).

.PARAMETER ContinueOnError
    Keep building the remaining targets if one fails, then report a summary and
    exit non-zero. Default: stop at the first failure.

.EXAMPLE
    # All four packages (TensorRT bundling every GPU resource):
    . scripts\activate.ps1
    pwsh scripts\package-all.ps1

.EXAMPLE
    # Only the Windows ML pair (no GPU toolchain needed):
    pwsh scripts\package-all.ps1 -Targets app-windowsml,vst3-windowsml

.EXAMPLE
    # Smallest TensorRT packages (engines built/cached elsewhere):
    pwsh scripts\package-all.ps1 -Targets app-tensorrt,vst3-tensorrt -RuntimeOnly
#>
[CmdletBinding()]
param(
    [ValidateSet('app-windowsml', 'app-tensorrt', 'cli-windowsml', 'cli-tensorrt', 'vst3-windowsml', 'vst3-tensorrt')]
    [string[]]$Targets = @('app-windowsml', 'app-tensorrt', 'vst3-windowsml', 'vst3-tensorrt'),
    [string]$OutDir,

    # tensorrt targets
    [switch]$RuntimeOnly,
    [string]$TensorRtBin,

    # Forwarded to every package.ps1: keep/remove the ready-to-run dist\<stem>\
    # folders. Each package.ps1 keeps its folder by default (vst3-tensorrt still
    # drops its multi-GB folder unless -KeepStage). -CleanStage drops them all.
    [switch]$KeepStage,
    [switch]$CleanStage,

    [switch]$ContinueOnError
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $OutDir) { $OutDir = Join-Path $repoRoot 'dist' }

# Map each target to its package script and variant. The tensorrt-only options
# are forwarded just to the tensorrt targets.
$appScript = Join-Path $repoRoot 'crates\vc-cli\package.ps1'
$vst3Script = Join-Path $repoRoot 'crates\vc-vst3\package.ps1'
$plan = [ordered]@{
    'app-windowsml'  = @{ Script = $appScript;  Variant = 'windowsml'; Aliases = @('cli-windowsml') }
    'app-tensorrt'   = @{ Script = $appScript;  Variant = 'tensorrt'; Aliases = @('cli-tensorrt') }
    'vst3-windowsml' = @{ Script = $vst3Script; Variant = 'windowsml' }
    'vst3-tensorrt'  = @{ Script = $vst3Script; Variant = 'tensorrt' }
}

$results = [System.Collections.Generic.List[object]]::new()

foreach ($name in $plan.Keys) {
    $spec = $plan[$name]
    $selected = $Targets -contains $name
    foreach ($alias in @($spec.Aliases)) {
        if ($Targets -contains $alias) { $selected = $true }
    }
    if (-not $selected) { continue }

    # Build the argument splat for this target's package.ps1. ($pkgArgs, not the
    # automatic $args variable.)
    $pkgArgs = @{ Variant = $spec.Variant; OutDir = $OutDir }
    if ($KeepStage) { $pkgArgs['KeepStage'] = $true }
    if ($CleanStage) { $pkgArgs['CleanStage'] = $true }
    if ($spec.Variant -eq 'tensorrt') {
        if ($RuntimeOnly) { $pkgArgs['RuntimeOnly'] = $true }
        if ($PSBoundParameters.ContainsKey('TensorRtBin')) { $pkgArgs['TensorRtBin'] = $TensorRtBin }
    }

    Write-Host ""
    Write-Host "######## $name ########" -ForegroundColor Magenta

    try {
        & $spec.Script @pkgArgs
        if ($LASTEXITCODE -ne 0) { throw "package.ps1 exited $LASTEXITCODE" }
        $results.Add([pscustomobject]@{ Target = $name; Status = 'OK' })
    }
    catch {
        $results.Add([pscustomobject]@{ Target = $name; Status = "FAILED: $($_.Exception.Message)" })
        if (-not $ContinueOnError) {
            Write-Host ""
            Write-Host "==> $name failed; stopping (pass -ContinueOnError to keep going)." -ForegroundColor Red
            $results | Format-Table -AutoSize | Out-Host
            throw
        }
        Write-Host "==> $name failed; continuing (-ContinueOnError)." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "==> package-all summary" -ForegroundColor Cyan
$results | Format-Table -AutoSize | Out-Host

if ($results | Where-Object { $_.Status -ne 'OK' }) {
    throw "One or more packages failed."
}

Write-Host ("==> All done. Archives in {0}" -f $OutDir) -ForegroundColor Green
