<#
.SYNOPSIS
    Build, populate, and zip a distributable vc-rs CLI package end to end.

.DESCRIPTION
    One command that produces a ready-to-ship archive for a given backend
    variant. It runs the whole pipeline:

        1. cargo build --release -p vc-cli  (with the variant's features)
        2. (tensorrt only) build the ORT-free engine builder helper if needed
        3. stage vc-rs.exe into dist\_stage_<variant>\
        4. the matching populate script (package-windowsml|tensorrt.ps1), which
           copies the runtime DLLs + licenses next to the staged vc-rs.exe
        5. add LICENSE + a generated INSTALL.txt
        6. Compress-Archive into dist\vc-rs-cli-<variant>-v<version>-win-x64.zip

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
    Reuse the existing target\release\vc-rs.exe instead of running cargo build.
    The stage, populate, and archive steps still run.

.PARAMETER NoZip
    Stage and populate the package but stop before creating the .zip (useful for
    inspecting dist\_stage_<variant>).

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
    -RuntimeOnly) it is built from tools\tensorrt_probe. Forwarded to
    package-tensorrt.ps1.

.PARAMETER FoundationVersion
    (windowsml) Microsoft.WindowsAppSDK.Foundation version holding the
    bootstrapper DLL. Forwarded to package-windowsml.ps1.

.PARAMETER BootstrapDll
    (windowsml) Existing bootstrapper DLL to copy. Forwarded to
    package-windowsml.ps1.

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

    # tensorrt
    [string]$TensorRtBin,
    [string]$CudaBin,
    [string[]]$BuilderSm,
    [switch]$RuntimeOnly,
    [string]$BuilderExe,

    # windowsml
    [string]$FoundationVersion,
    [string]$BootstrapDll
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$releaseDir = Join-Path $repoRoot 'target\release'
if (-not $OutDir) { $OutDir = Join-Path $repoRoot 'dist' }

# Single-provider feature set per variant. `--no-default-features` drops the
# other backend so the binary stays lean (a tensorrt build sheds ONNX Runtime).
$buildFeatureArgs = switch ($Variant) {
    'windowsml' { @('--no-default-features', '--features', 'windowsml') }
    'tensorrt' { @('--no-default-features', '--features', 'tensorrt') }
}

Push-Location $repoRoot
try {
    # 1. Build the CLI.
    if (-not $SkipBuild) {
        Write-Host "==> cargo build --release -p vc-cli $($buildFeatureArgs -join ' ')" -ForegroundColor Cyan
        cargo build --release -p vc-cli @buildFeatureArgs
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit $LASTEXITCODE)." }

        # The TensorRT engine-builder helper is a separate ORT-free binary. Build
        # it here (unless skipped) so package-tensorrt.ps1 can bundle it.
        if ($Variant -eq 'tensorrt' -and -not $RuntimeOnly -and -not $BuilderExe) {
            $helper = Join-Path $repoRoot 'tools\tensorrt_probe\target\release\vc-tensorrt-builder.exe'
            Write-Host "==> cargo build --release (vc-tensorrt-builder helper)" -ForegroundColor Cyan
            cargo build --release --manifest-path (Join-Path $repoRoot 'tools\tensorrt_probe\Cargo.toml')
            if ($LASTEXITCODE -ne 0) { throw "building vc-tensorrt-builder failed (exit $LASTEXITCODE)." }
            if (Test-Path $helper) { $BuilderExe = $helper }
        }
    }
    else {
        Write-Host "==> Skipping build (-SkipBuild); reusing $releaseDir\vc-rs.exe" -ForegroundColor Yellow
    }

    $exe = Join-Path $releaseDir 'vc-rs.exe'
    if (-not (Test-Path $exe)) { throw "vc-rs.exe not found in $releaseDir. Run without -SkipBuild first." }

    # 2. Stage vc-rs.exe into its own folder so the populate step drops the
    #    variant DLLs beside it (and so we don't pollute target\release).
    $tag = $Variant
    if ($Variant -eq 'tensorrt') {
        if ($RuntimeOnly) { $tag += '-runtime' }
        elseif ($BuilderSm -and $BuilderSm.Count -gt 0 -and ($BuilderSm -notcontains 'none')) {
            $tag += '-' + ($BuilderSm -join '-')
        }
    }

    $staging = Join-Path $OutDir "_stage_$tag"
    if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    Copy-Item $exe (Join-Path $staging 'vc-rs.exe') -Force

    # 3. Populate the staged folder with the variant's runtime DLLs + licenses by
    #    forwarding only the parameters the caller actually supplied.
    $populateScript = Join-Path $PSScriptRoot "package-$Variant.ps1"
    if (-not (Test-Path $populateScript)) { throw "Populate script not found: $populateScript" }

    $forward = @{ DestDir = $staging }
    $forwardable = switch ($Variant) {
        'windowsml' { @('FoundationVersion', 'BootstrapDll') }
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

    # 4. License + a short install/usage note.
    $license = Join-Path $repoRoot 'LICENSE'
    if (Test-Path $license) { Copy-Item $license (Join-Path $staging 'LICENSE') -Force }

    $version = '0.0.0'
    $wsToml = Get-Content (Join-Path $repoRoot 'Cargo.toml') -Raw
    if ($wsToml -match '(?ms)\[workspace\.package\].*?^\s*version\s*=\s*"([^"]+)"') {
        $version = $Matches[1]
    }

    $reqLine = switch ($Variant) {
        'windowsml' { '  Windows App SDK Runtime 2.x installed (provides ONNX Runtime + DirectML).' }
        'tensorrt' { '  An up-to-date NVIDIA GPU driver. TensorRT runtime DLLs are bundled — no install needed.' }
    }
    $trtNote = if ($Variant -eq 'tensorrt' -and -not $RuntimeOnly) {
        @"

First-run TensorRT engine builds use the bundled vc-tensorrt-builder.exe, which
vc-rs.exe finds automatically beside itself — keep the two together.
"@
    }
    else { '' }

    $install = @"
vc-rs — RVC voice conversion CLI ($Variant build, v$version)

Run from this folder (keep the DLLs beside vc-rs.exe):
    .\vc-rs.exe --help
    .\vc-rs.exe devices
    .\vc-rs.exe run --model <model.onnx> --provider $(if ($Variant -eq 'tensorrt') { 'tensorrt' } else { 'windowsml' })

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
    $stem = "vc-rs-cli-$tag-v$version-win-x64"
    $zip = Join-Path $OutDir "$stem.zip"
    if (Test-Path $zip) { Remove-Item -Force $zip }
    Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $zip -CompressionLevel Optimal
    Remove-Item -Recurse -Force $staging

    $size = (Get-Item $zip).Length
    Write-Host ("==> Done: {0} ({1:N1} MB)" -f $zip, ($size / 1MB)) -ForegroundColor Green
}
finally {
    Pop-Location
}
