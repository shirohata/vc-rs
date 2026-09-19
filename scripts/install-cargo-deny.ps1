<#
.SYNOPSIS
Install the cargo-deny version used by CI, without running the full bootstrap.
#>
[CmdletBinding()]
param([string]$CargoPath = 'cargo')

$ErrorActionPreference = 'Stop'
# Keep this version aligned with the pinned cargo-deny Action's Dockerfile in CI.
$denyVersion = '0.20.2'
$expected = "cargo-deny $denyVersion"

Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $denyCommand = Get-Command cargo-deny -ErrorAction SilentlyContinue
    if ($denyCommand) {
        $installed = & $denyCommand.Source --version
        if ($LASTEXITCODE -eq 0 -and "$installed".Trim() -eq $expected) {
            Write-Host "[skip] $expected is already installed."
            return
        }
    }

    & $CargoPath install cargo-deny --version $denyVersion --locked
    if ($LASTEXITCODE -ne 0) { throw "Installing $expected failed." }
    $installed = & $CargoPath deny --version
    if ($LASTEXITCODE -ne 0 -or "$installed".Trim() -ne $expected) {
        throw "Expected $expected, got '$installed'. Check for another cargo-deny earlier on PATH."
    }
    Write-Host "[installed] $expected"
} finally {
    Pop-Location
}
