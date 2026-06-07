<#
.SYNOPSIS
    Build, populate, and zip a distributable vc-rs standalone app package.

.DESCRIPTION
    One command that produces a ready-to-ship archive for a given backend
    variant. It runs the whole pipeline:

        1. cargo build --release -p vc-cli -p vc-gui (with matching features)
        2. (tensorrt only) build the ORT-free engine builder helper if needed
        3. stage vc-rs.exe and vc-gui.exe into dist\<stem>\
        4. the matching populate script (package-windowsml|tensorrt.ps1), which
           copies the runtime DLLs + licenses next to both executables
        5. generate exact Rust dependency notices for vc-rs.exe and vc-gui.exe
        6. add LICENSE + a generated INSTALL.txt
        7. Compress-Archive into dist\vc-rs-<variant>-v<version>-win-x64.zip

    The populate scripts are reused as-is; this only orchestrates them and adds
    the build + stage + archive steps.

    Toolchain note: the tensorrt build compiles native code that needs the
    matching CUDA/TensorRT toolchain reachable (e.g. dot-source scripts\activate.ps1
    first). This script does not modify your environment; set it up before running
    so the tensorrt build links correctly. The windowsml build needs no GPU
    toolchain.

.PARAMETER Variant
    Which backend package to build: windowsml (default) or tensorrt.

.PARAMETER OutDir
    Where to write the .zip. Default: <repo>\dist.

.PARAMETER SkipBuild
    Reuse existing target\release\vc-rs.exe and vc-gui.exe instead of building.
    The stage, populate, and archive steps still run.

.PARAMETER NoZip
    Stage and populate the package but stop before creating the .zip. The
    populated, ready-to-run dist\<stem>\ folder is left in place.

.PARAMETER KeepStage
    Keep the populated dist\<stem>\ folder beside the .zip. This is the default for
    every variant, so the flag is only useful to be explicit; -CleanStage is the
    opposite.

.PARAMETER CleanStage
    Remove the populated dist\<stem>\ folder after zipping. Use this to drop the
    tensorrt folder (which can be multiple GB) when you only want the .zip.

.PARAMETER TensorRtBin
    (tensorrt) TensorRT bin directory. Forwarded to package-tensorrt.ps1.

.PARAMETER CudaBin
    (tensorrt) CUDA Toolkit bin directory. Forwarded to package-tensorrt.ps1.

.PARAMETER BuilderSm
    (tensorrt) GPU SM tags whose builder-resource DLLs to bundle (e.g. sm86).
    Forwarded to package-tensorrt.ps1.

.PARAMETER RuntimeOnly
    (tensorrt) Bundle only the runtime DLLs (no engine builder). Forwarded to
    package-tensorrt.ps1.

.PARAMETER BuilderExe
    (tensorrt) Path to vc-tensorrt-builder.exe. When omitted (and not
    -RuntimeOnly) it is built from tools\tensorrt_builder. Forwarded to
    package-tensorrt.ps1.

.PARAMETER FoundationVersion
    (windowsml) Microsoft.WindowsAppSDK.Foundation version holding the
    bootstrapper DLL. Forwarded to package-windowsml.ps1.

.PARAMETER BootstrapDll
    (windowsml) Existing bootstrapper DLL to copy. Forwarded to
    package-windowsml.ps1.

.PARAMETER WindowsAppSdkLicense
    (windowsml) License text matching -BootstrapDll. Required when passing the
    bootstrapper DLL directly.

.EXAMPLE
    # Default Windows ML package:
    pwsh crates\vc-cli\package.ps1

.EXAMPLE
    # Self-contained TensorRT package for an RTX 30xx (sm86):
    pwsh crates\vc-cli\package.ps1 -Variant tensorrt -BuilderSm sm86

.EXAMPLE
    # Smallest TensorRT package (engines built/cached elsewhere):
    pwsh crates\vc-cli\package.ps1 -Variant tensorrt -RuntimeOnly
#>
[CmdletBinding()]
param(
    [ValidateSet('windowsml', 'tensorrt')]
    [string]$Variant = 'windowsml',
    [string]$OutDir,
    [switch]$SkipBuild,
    [switch]$NoZip,
    # Keep the populated, ready-to-run dist\<stem>\ folder beside the .zip. Kept by
    # default for every variant (handy for testing the unpacked package). -CleanStage
    # drops it after zipping (useful for the multi-GB tensorrt folder).
    [switch]$KeepStage,
    [switch]$CleanStage,

    # tensorrt
    [string]$TensorRtBin,
    [string]$CudaBin,
    [string[]]$BuilderSm,
    [switch]$RuntimeOnly,
    [string]$BuilderExe,

    # windowsml
    [string]$FoundationVersion,
    [string]$BootstrapDll,
    [string]$WindowsAppSdkLicense
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$releaseDir = Join-Path $repoRoot 'target\release'
if (-not $OutDir) { $OutDir = Join-Path $repoRoot 'dist' }

# The standalone archive ships two binaries with different dependency graphs.
# Require fresh notices for both rather than copying the VST3 notice.
if (-not (Get-Command cargo-about -ErrorAction SilentlyContinue)) {
    throw "cargo-about is required to build distribution packages. Install it with: cargo install cargo-about --features cli"
}

# Strip absolute build-machine paths (user name etc.) from the shipped binaries.
# Sets CARGO_ENCODED_RUSTFLAGS, inherited by the cargo build below and the
# tensorrt builder-helper build.
. (Join-Path $repoRoot 'scripts\rustflags.ps1')

# Single-provider feature set per variant. `--no-default-features` drops the
# other backend so the binary stays lean (a tensorrt build sheds ONNX Runtime).
$buildFeatureArgs = switch ($Variant) {
    'windowsml' { @('--no-default-features', '--features', 'windowsml') }
    'tensorrt' { @('--no-default-features', '--features', 'tensorrt') }
}

Push-Location $repoRoot
try {
    # 1. Build the GUI and CLI with the same provider feature set. Keeping this
    #    in one command prevents a package from mixing backend variants.
    if (-not $SkipBuild) {
        Write-Host "==> cargo build --release -p vc-cli -p vc-gui $($buildFeatureArgs -join ' ')" -ForegroundColor Cyan
        cargo build --release -p vc-cli -p vc-gui @buildFeatureArgs
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit $LASTEXITCODE)." }

        # The TensorRT engine-builder helper is a separate ORT-free binary. Build
        # it here (unless skipped) so package-tensorrt.ps1 can bundle it.
        if ($Variant -eq 'tensorrt' -and -not $RuntimeOnly -and -not $BuilderExe) {
            $helper = Join-Path $repoRoot 'tools\tensorrt_builder\target\release\vc-tensorrt-builder.exe'
            Write-Host "==> cargo build --release (vc-tensorrt-builder helper)" -ForegroundColor Cyan
            cargo build --release --manifest-path (Join-Path $repoRoot 'tools\tensorrt_builder\Cargo.toml')
            if ($LASTEXITCODE -ne 0) { throw "building vc-tensorrt-builder failed (exit $LASTEXITCODE)." }
            if (Test-Path $helper) { $BuilderExe = $helper }
        }
    }
    else {
        Write-Host "==> Skipping build (-SkipBuild); reusing standalone executables in $releaseDir" -ForegroundColor Yellow
    }

    $exe = Join-Path $releaseDir 'vc-rs.exe'
    if (-not (Test-Path $exe)) { throw "vc-rs.exe not found in $releaseDir. Run without -SkipBuild first." }
    $guiExe = Join-Path $releaseDir 'vc-gui.exe'
    if (-not (Test-Path $guiExe)) { throw "vc-gui.exe not found in $releaseDir. Run without -SkipBuild first." }

    # 2. Stage both executables together so the populate step drops the variant
    #    DLLs beside them (and so we don't pollute target\release). The folder is
    #    named after the archive stem so a kept dir sits next to its .zip.
    $tag = $Variant
    if ($Variant -eq 'tensorrt') {
        if ($RuntimeOnly) { $tag += '-runtime' }
        elseif ($BuilderSm -and $BuilderSm.Count -gt 0 -and ($BuilderSm -notcontains 'none')) {
            $tag += '-' + ($BuilderSm -join '-')
        }
    }

    $version = '0.0.0'
    $wsToml = Get-Content (Join-Path $repoRoot 'Cargo.toml') -Raw
    if ($wsToml -match '(?ms)\[workspace\.package\].*?^\s*version\s*=\s*"([^"]+)"') {
        $version = $Matches[1]
    }
    $stem = "vc-rs-$tag-v$version-win-x64"

    $staging = Join-Path $OutDir $stem
    if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    Copy-Item $exe (Join-Path $staging 'vc-rs.exe') -Force
    Copy-Item $guiExe (Join-Path $staging 'vc-gui.exe') -Force

    # 3. Populate the staged folder with the variant's runtime DLLs + licenses by
    #    forwarding only the parameters the caller actually supplied.
    $populateScript = Join-Path $PSScriptRoot "package-$Variant.ps1"
    if (-not (Test-Path $populateScript)) { throw "Populate script not found: $populateScript" }

    $forward = @{ DestDir = $staging }
    $forwardable = switch ($Variant) {
        'windowsml' { @('FoundationVersion', 'BootstrapDll', 'WindowsAppSdkLicense') }
        'tensorrt' { @('TensorRtBin', 'CudaBin', 'BuilderSm', 'RuntimeOnly', 'BuilderExe') }
    }
    foreach ($name in $forwardable) {
        if ($PSBoundParameters.ContainsKey($name)) { $forward[$name] = $PSBoundParameters[$name] }
    }
    # BuilderExe may have been resolved above rather than passed in.
    if ($Variant -eq 'tensorrt' -and $BuilderExe -and -not $forward.ContainsKey('BuilderExe')) {
        $forward['BuilderExe'] = $BuilderExe
    }

    Write-Host "==> $((Split-Path $populateScript -Leaf))" -ForegroundColor Cyan
    & $populateScript @forward

    # vc-rs.exe and vc-gui.exe have different direct dependencies even though
    # they use the same backend feature. Keep separate notices so each generated
    # dependency graph is exact and auditable.
    $licenseDir = Join-Path $staging 'licenses'
    $licenseGenerator = Join-Path $repoRoot 'scripts\generate-rust-licenses.ps1'
    & $licenseGenerator `
        -ManifestPath (Join-Path $PSScriptRoot 'Cargo.toml') `
        -OutputPath (Join-Path $licenseDir 'THIRD-PARTY-LICENSES-vc-rs.md') `
        -FeatureArgs $buildFeatureArgs
    & $licenseGenerator `
        -ManifestPath (Join-Path $repoRoot 'crates\vc-gui\Cargo.toml') `
        -OutputPath (Join-Path $licenseDir 'THIRD-PARTY-LICENSES-vc-gui.md') `
        -FeatureArgs $buildFeatureArgs
    if ($Variant -eq 'tensorrt' -and -not $RuntimeOnly) {
        & $licenseGenerator `
            -ManifestPath (Join-Path $repoRoot 'tools\tensorrt_builder\Cargo.toml') `
            -OutputPath (Join-Path $licenseDir 'THIRD-PARTY-LICENSES-vc-tensorrt-builder.md')
    }

    # 4. License + a short install/usage note. Also ship the optional model
    #    downloader beside the executables; it fetches the shared embedder + F0 models
    #    into .\assets\ (relative to itself), which the run flags below point at.
    $license = Join-Path $repoRoot 'LICENSE'
    if (Test-Path $license) { Copy-Item $license (Join-Path $staging 'LICENSE') -Force }

    $modelDl = Join-Path $repoRoot 'download-models.ps1'
    if (Test-Path $modelDl) { Copy-Item $modelDl (Join-Path $staging 'download-models.ps1') -Force }

    $reqLine = switch ($Variant) {
        'windowsml' { '  Windows App SDK Runtime 2.x installed (provides ONNX Runtime + DirectML).' }
        'tensorrt' { '  An up-to-date NVIDIA GPU driver. TensorRT runtime DLLs are bundled — no install needed.' }
    }
    $trtNote = if ($Variant -eq 'tensorrt' -and -not $RuntimeOnly) {
        @"

First-run TensorRT engine builds use the bundled vc-tensorrt-builder.exe, which
the applications find automatically beside themselves — keep these files together.
"@
    }
    else { '' }

    $install = @"
vc-rs — RVC voice conversion standalone app ($Variant build, v$version)

Run the GUI from this folder (keep the DLLs beside both executables):
    .\vc-gui.exe

The CLI is included for diagnostics, automation, and WAV conversion:
    .\vc-rs.exe --help
    .\vc-rs.exe doctor
    .\vc-rs.exe devices

Models — get the shared embedder + F0 models (optional helper):
    pwsh .\download-models.ps1
This downloads ContentVec + RMVPE into .\assets\. You still supply your own RVC
voice model (.onnx). Then run:
    .\vc-rs.exe run --model <your-rvc-model.onnx> ``
        --embedder .\assets\content_vec_500.onnx ``
        --f0-model .\assets\rmvpe.onnx ``
        --provider $(if ($Variant -eq 'tensorrt') { 'tensorrt' } else { 'windowsml' })
The downloaded models are third-party (GPL-3.0 upstream), not covered by this
package's MIT license — see download-models.ps1.

Requirements for this ($Variant) build:
$reqLine
$trtNote
See licenses\ for third-party license texts.
"@
    Set-Content -Path (Join-Path $staging 'INSTALL.txt') -Value $install -Encoding UTF8

    if ($NoZip) {
        Write-Host "==> -NoZip: package staged in $staging (skipping archive)." -ForegroundColor Green
        return
    }

    # 5. Archive.
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $zip = Join-Path $OutDir "$stem.zip"
    if (Test-Path $zip) { Remove-Item -Force $zip }
    Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $zip -CompressionLevel Optimal

    # 6. Keep or remove the ready-to-run dir. Default: keep it for every variant
    #    (handy for testing the unpacked package). The tensorrt dir can be multi-GB,
    #    so pass -CleanStage to drop it after zipping. Flags override the default.
    if ($KeepStage -and $CleanStage) { throw "Pass only one of -KeepStage / -CleanStage." }
    $keepDir = if ($KeepStage) { $true } elseif ($CleanStage) { $false } else { $true }
    if ($keepDir) {
        Write-Host "==> Kept ready-to-run dir: $staging" -ForegroundColor Green
    }
    else {
        Remove-Item -Recurse -Force $staging
    }

    $size = (Get-Item $zip).Length
    Write-Host ("==> Done: {0} ({1:N1} MB)" -f $zip, ($size / 1MB)) -ForegroundColor Green
}
finally {
    Pop-Location
}
