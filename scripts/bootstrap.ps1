<#
.SYNOPSIS
    Install vc-rs build prerequisites and pinned Rust tools.

.DESCRIPTION
    Idempotently installs the build prerequisites that have first-class winget
    packages: the Rust toolchain (rustup), the MSVC C++ build tools (needed by
    the `cc` crate to compile native_tensorrt_shim.cpp), CMake, and Git.
    Also installs the repository-pinned Rust toolchain and cargo-deny version.

    This script deliberately does NOT install the NVIDIA SDKs (CUDA Toolkit,
    cuDNN, TensorRT). Those require an NVIDIA login and are vendored / placed
    separately; `crates/vc-core/build.rs` auto-discovers TensorRT under the
    workspace root and CUDA under "Program Files". After running this script,
    set up those SDKs and activate PATH via tmp/env.ps1 (see the printed notes).

.NOTES
    Safe to re-run. Each step checks for an existing install before invoking
    winget, and winget itself is idempotent for already-installed packages.

.EXAMPLE
    pwsh -File scripts/bootstrap.ps1
#>

[CmdletBinding()]
param(
    # Reinstall/repair even if the tool already appears present.
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# --- Pinned, winget-provisionable prerequisites -----------------------------
# Single source of truth for the packages this script manages. NVIDIA SDKs are
# intentionally absent (login-gated); handle those out of band.
$Packages = @(
    @{ Name = "Rustup (Rust toolchain)"; Id = "Rustlang.Rustup"; Probe = "rustup" }
    @{ Name = "Git";                     Id = "Git.Git";         Probe = "git" }
    @{ Name = "just (command runner)";   Id = "Casey.Just";      Probe = "just" }
)

function Test-CommandPresent([string]$Name) {
    [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-MsvcCppPresent {
    # The `cc` crate needs the MSVC C++ toolset + Windows SDK headers. Detect via
    # vswhere rather than a PATH probe (cl.exe is not on PATH outside a VS shell).
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $false }
    $found = & $vswhere -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property displayName 2>$null
    return [bool]$found
}

function Invoke-Winget {
    param(
        [Parameter(Mandatory)] [string]$Id,
        [string[]]$Override
    )
    $args = @(
        "install", "--exact", "--id", $Id,
        "--accept-package-agreements", "--accept-source-agreements",
        "--disable-interactivity"
    )
    if ($Override) { $args += @("--override", ($Override -join " ")) }

    Write-Host "  winget $($args -join ' ')" -ForegroundColor DarkGray
    & winget @args
    $code = $LASTEXITCODE
    # winget success / benign "already installed" / "no applicable update" codes.
    $ok = @(0, -1978335189, -1978335212, -1978334967)
    if ($code -notin $ok) {
        throw "winget install '$Id' failed (exit 0x$('{0:X8}' -f ($code -band 0xFFFFFFFF)))"
    }
}

# --- Preconditions ----------------------------------------------------------
if (-not (Test-CommandPresent "winget")) {
    throw "winget is not available. Install 'App Installer' from the Microsoft Store, then re-run."
}

Write-Host "== vc-rs build environment bootstrap (winget scope) ==" -ForegroundColor Cyan
Write-Host ""

# --- 1. Simple packages (probe -> install) ----------------------------------
foreach ($pkg in $Packages) {
    if (-not $Force -and (Test-CommandPresent $pkg.Probe)) {
        Write-Host "[skip] $($pkg.Name) already present ('$($pkg.Probe)' on PATH)." -ForegroundColor Green
        continue
    }
    Write-Host "[install] $($pkg.Name) [$($pkg.Id)]" -ForegroundColor Yellow
    Invoke-Winget -Id $pkg.Id
}

# --- 2. MSVC C++ build tools (custom workload override) ----------------------
if (-not $Force -and (Test-MsvcCppPresent)) {
    Write-Host "[skip] MSVC C++ build tools already present." -ForegroundColor Green
} else {
    Write-Host "[install] Visual Studio 2022 Build Tools (C++ workload)" -ForegroundColor Yellow
    # VCTools workload + recommended components pulls in the Windows SDK headers
    # the `cc`-compiled TensorRT shim needs. --wait so winget blocks until done.
    Invoke-Winget -Id "Microsoft.VisualStudio.2022.BuildTools" -Override @(
        "--quiet", "--wait", "--norestart",
        "--add", "Microsoft.VisualStudio.Workload.VCTools",
        "--includeRecommended"
    )
}

# --- 3. Ensure the Rust toolchain + components ------------------------------
# rustup may have just been installed into a session that doesn't yet have it on
# PATH; fall back to the default install location for this run.
$rustup = (Get-Command rustup -ErrorAction SilentlyContinue).Source
if (-not $rustup) {
    $candidate = Join-Path $env:USERPROFILE ".cargo\bin\rustup.exe"
    if (Test-Path $candidate) { $rustup = $candidate }
}
if ($rustup) {
    Write-Host "[rustup] ensuring repository-pinned toolchain + components" -ForegroundColor Yellow
    # Resolve rust-toolchain.toml even when setup was launched from elsewhere.
    Push-Location (Split-Path -Parent $PSScriptRoot)
    try {
        & $rustup toolchain install --no-self-update
        if ($LASTEXITCODE -ne 0) { throw "Installing the pinned Rust toolchain failed." }
        & (Join-Path $PSScriptRoot 'install-cargo-deny.ps1') -CargoPath (Join-Path (Split-Path -Parent $rustup) 'cargo.exe')
    } finally {
        Pop-Location
    }
} else {
    Write-Warning "rustup not found on PATH yet. Open a new terminal in the repository and run: rustup toolchain install"
}

# --- 4. Git hooks (shared, repo-local) --------------------------------------
# Point git at the tracked .githooks dir so every clone gets the pre-commit
# rustfmt check. Relative path resolves against the working-tree root. Idempotent.
$repoRoot = Split-Path -Parent $PSScriptRoot
if (Test-Path (Join-Path $repoRoot ".githooks")) {
    Write-Host "[git] setting core.hooksPath = .githooks" -ForegroundColor Yellow
    & git -C $repoRoot config core.hooksPath .githooks
} else {
    Write-Warning "Skipping git hooks setup: .githooks not found under $repoRoot"
}

# --- Summary ----------------------------------------------------------------
Write-Host ""
Write-Host "== winget-scope prerequisites done ==" -ForegroundColor Cyan
Write-Host ""
Write-Host "Still required (login-gated, not handled here):" -ForegroundColor Magenta
Write-Host "  - CUDA Toolkit  (v13.3 Update 1 for TensorRT 11.2.1)"
Write-Host "  - cuDNN v9.x"
Write-Host "  - TensorRT      (extract under external\nvidia; build.rs auto-discovers it)"
Write-Host ""
Write-Host "Then, per shell session:" -ForegroundColor Magenta
Write-Host "  . .\scripts\activate.ps1 # put CUDA/cuDNN/TensorRT bin on PATH"
Write-Host "  .\download-models.ps1    # fetch reference ONNX models (optional)"
Write-Host ""
Write-Host "Note: open a NEW terminal so freshly installed tools are on PATH." -ForegroundColor DarkYellow
