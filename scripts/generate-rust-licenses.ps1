<#
.SYNOPSIS
    Generate the Rust dependency license notice for one shipped binary.

.DESCRIPTION
    Packaging must call this once per distributed Rust binary with the same
    manifest and feature flags used to build that binary. Generating directly
    into the package staging directory prevents variants and products from
    overwriting or reusing each other's dependency notices.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$ManifestPath,

    [Parameter(Mandatory)]
    [string]$OutputPath,

    [string[]]$FeatureArgs = @()
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command cargo-about -ErrorAction SilentlyContinue)) {
    throw "cargo-about is required to build distribution packages. Install it with: cargo install cargo-about --features cli"
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$config = Join-Path $scriptDir 'licenses\about.toml'
$template = Join-Path $scriptDir 'licenses\about.hbs'
$outputDir = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Write-Host "==> cargo about generate: $OutputPath" -ForegroundColor Cyan
cargo about generate --manifest-path $ManifestPath -c $config --locked @FeatureArgs $template -o $OutputPath
if ($LASTEXITCODE -ne 0) {
    throw "cargo about generate failed for $ManifestPath (exit $LASTEXITCODE)."
}

if (-not (Test-Path $OutputPath)) {
    throw "cargo about generate did not create $OutputPath."
}
