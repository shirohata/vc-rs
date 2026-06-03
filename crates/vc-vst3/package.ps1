<#
.SYNOPSIS
    Build, populate, and zip a distributable vc-vst3 plugin package end to end.

.DESCRIPTION
    One command that produces a ready-to-ship archive for a given backend
    variant. It runs the whole pipeline:

        1. cargo xtask bundle vc-vst3 --release  (with the variant's features)
        2. (tensorrt only) build the ORT-free engine builder helper if needed
        3. the matching populate script (package-windowsml|cuda|tensorrt.ps1),
           which copies the runtime DLLs + licenses into the bundle
        4. stage vc-vst3.vst3 + vc-vst3.clap + LICENSE + a generated INSTALL.txt
        5. Compress-Archive into dist\vc-vst3-<variant>-v<version>-win-x64.zip

    The populate scripts are reused as-is; this only orchestrates them and adds
    the build + archive steps. Variant-specific options are forwarded to the
    populate script (see the parameters below and the examples).

    Toolchain note: the cuda and tensorrt builds compile native code that needs
    the matching CUDA/TensorRT toolchain reachable (e.g. dot-source
    scripts\activate.ps1 first). This script does not modify your environment;
    set it up before running so the cuda (CUDA 12.x) or tensorrt (CUDA 13.x /
    TensorRT) build links correctly.

.PARAMETER Variant
    Which backend package to build: windowsml (default), cuda, or tensorrt.

.PARAMETER OutDir
    Where to write the .zip. Default: <repo>\dist.

.PARAMETER SkipBuild
    Reuse an existing target\bundled bundle instead of running cargo xtask
    bundle. The populate step and archiving still run.

.PARAMETER NoZip
    Populate the bundle but stop before creating the .zip (useful for inspecting
    target\bundled).

.PARAMETER Clean
    Remove any existing vc-vst3.vst3 / vc-vst3.clap from target\bundled before
    building, so stale DLLs from a previous variant cannot linger in the bundle.

.PARAMETER CudaBin
    (cuda) CUDA Toolkit bin directory. Forwarded to package-cuda.ps1.

.PARAMETER CudnnBin
    (cuda) cuDNN bin directory. Forwarded to package-cuda.ps1.

.PARAMETER TensorRtBin
    (tensorrt) TensorRT bin directory. Forwarded to package-tensorrt.ps1.

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
    pwsh crates\vc-vst3\package.ps1

.EXAMPLE
    # CUDA package, DLLs pulled from explicit toolkit dirs:
    pwsh crates\vc-vst3\package.ps1 -Variant cuda `
        -CudaBin "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin" `
        -CudnnBin "C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64"

.EXAMPLE
    # Self-contained TensorRT package for an RTX 30xx (sm86):
    pwsh crates\vc-vst3\package.ps1 -Variant tensorrt -BuilderSm sm86
#>
[CmdletBinding()]
param(
    [ValidateSet('windowsml', 'cuda', 'tensorrt')]
    [string]$Variant = 'windowsml',
    [string]$OutDir,
    [switch]$SkipBuild,
    [switch]$NoZip,
    [switch]$Clean,

    # cuda
    [string]$CudaBin,
    [string]$CudnnBin,

    # tensorrt
    [string]$TensorRtBin,
    [string[]]$BuilderSm,
    [switch]$RuntimeOnly,
    [string]$BuilderExe,

    # windowsml
    [string]$FoundationVersion,
    [string]$BootstrapDll
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$bundleDir = Join-Path $repoRoot 'target\bundled'
if (-not $OutDir) { $OutDir = Join-Path $repoRoot 'dist' }

# Feature flags per variant for `cargo xtask bundle`. windowsml is the default
# feature set, so it needs no extra flags.
$bundleFeatureArgs = switch ($Variant) {
    'windowsml' { @() }
    'cuda' { @('--no-default-features', '--features', 'cuda') }
    'tensorrt' { @('--no-default-features', '--features', 'tensorrt') }
}

Push-Location $repoRoot
try {
    # 1. Optionally clear the whole bundle dir so DLLs from a prior variant cannot
    #    survive — the CLAP packaging gathers every loose sidecar beside the .clap,
    #    so a stale DLL there would otherwise leak into the new package.
    if ($Clean -and (Test-Path $bundleDir)) {
        Remove-Item -Recurse -Force $bundleDir
    }

    # 2. Build the bundle.
    if (-not $SkipBuild) {
        Write-Host "==> cargo xtask bundle vc-vst3 --release $($bundleFeatureArgs -join ' ')" -ForegroundColor Cyan
        cargo xtask bundle vc-vst3 --release @bundleFeatureArgs
        if ($LASTEXITCODE -ne 0) { throw "cargo xtask bundle failed (exit $LASTEXITCODE)." }

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
        Write-Host "==> Skipping build (-SkipBuild); reusing $bundleDir" -ForegroundColor Yellow
    }

    # 3. Populate the bundle with the variant's runtime DLLs + licenses by
    #    forwarding only the parameters the caller actually supplied.
    $populateScript = Join-Path $PSScriptRoot "package-$Variant.ps1"
    if (-not (Test-Path $populateScript)) { throw "Populate script not found: $populateScript" }

    $forward = @{ BundleDir = $bundleDir }
    $forwardable = switch ($Variant) {
        'windowsml' { @('FoundationVersion', 'BootstrapDll') }
        'cuda' { @('CudaBin', 'CudnnBin') }
        'tensorrt' { @('TensorRtBin', 'BuilderSm', 'RuntimeOnly', 'BuilderExe') }
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

    # Locate the freshly populated bundles.
    $vst3 = Join-Path $bundleDir 'vc-vst3.vst3'
    $clap = Join-Path $bundleDir 'vc-vst3.clap'
    $artifacts = @($vst3, $clap | Where-Object { Test-Path $_ })
    if (-not $artifacts) { throw "No vc-vst3.vst3 / vc-vst3.clap found in $bundleDir." }

    if ($NoZip) {
        Write-Host "==> -NoZip: bundle ready in $bundleDir (skipping archive)." -ForegroundColor Green
        return
    }

    # 4. Stage the artifacts plus license + a short install note.
    $version = '0.0.0'
    $wsToml = Get-Content (Join-Path $repoRoot 'Cargo.toml') -Raw
    if ($wsToml -match '(?ms)\[workspace\.package\].*?^\s*version\s*=\s*"([^"]+)"') {
        $version = $Matches[1]
    }

    $tag = $Variant
    if ($Variant -eq 'tensorrt') {
        if ($RuntimeOnly) { $tag += '-runtime' }
        elseif ($BuilderSm -and $BuilderSm.Count -gt 0 -and ($BuilderSm -notcontains 'none')) {
            $tag += '-' + ($BuilderSm -join '-')
        }
    }
    $stem = "vc-vst3-$tag-v$version-win-x64"

    $staging = Join-Path $bundleDir "_dist_$tag"
    if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
    New-Item -ItemType Directory -Force -Path $staging | Out-Null

    # The VST3 bundle is self-contained (its sidecar DLLs live inside
    # Contents\<arch>\). The CLAP keeps its sidecar DLLs loose beside the .clap in
    # target\bundled, so ship the .clap together with those DLLs (and licenses\)
    # in its own folder — everything in the bundle dir except the .vst3 bundle and
    # our own staging dir.
    if (Test-Path $vst3) {
        Copy-Item -Path $vst3 -Destination $staging -Recurse -Force
    }
    if (Test-Path $clap) {
        $clapDir = Join-Path $staging 'vc-vst3-clap'
        New-Item -ItemType Directory -Force -Path $clapDir | Out-Null
        Get-ChildItem -Path $bundleDir |
            Where-Object { $_.Name -ne 'vc-vst3.vst3' -and $_.Name -notlike '_dist_*' } |
            ForEach-Object { Copy-Item -Path $_.FullName -Destination $clapDir -Recurse -Force }
    }

    $license = Join-Path $repoRoot 'LICENSE'
    if (Test-Path $license) { Copy-Item $license (Join-Path $staging 'LICENSE') -Force }

    $vst3Note = if (Test-Path $vst3) {
        "`nNote: the .vst3 links the Steinberg VST3 SDK bindings (GPLv3); the .clap is not affected.`n"
    }
    else { '' }
    $trtNote = if ($Variant -eq 'tensorrt' -and -not $RuntimeOnly) {
        @"

First-run TensorRT engine builds use the bundled vc-tensorrt-builder.exe. Because
a VST3/CLAP host's process is the DAW, point the plugin at it after installing:
    setx VC_RS_TENSORRT_BUILDER_HELPER "<install-dir>\vc-tensorrt-builder.exe"
"@
    }
    else { '' }

    $install = @"
vc-vst3 — RVC voice conversion plugin ($Variant build, v$version)

Install — copy into a standard plugin search path:
  VST3:  copy the  vc-vst3.vst3  bundle into
         %CommonProgramFiles%\VST3\   (e.g. C:\Program Files\Common Files\VST3)
  CLAP:  copy the whole  vc-vst3-clap\  folder into
         %CommonProgramFiles%\CLAP\   (e.g. C:\Program Files\Common Files\CLAP)
         Hosts scan CLAP subfolders, and the DLLs next to vc-vst3.clap must stay
         beside it — keep the folder intact rather than loose-copying the .clap.

Requirements for this ($Variant) build:
$(switch ($Variant) {
    'windowsml' { '  Windows App SDK Runtime 2.x installed (provides ONNX Runtime + DirectML).' }
    'cuda'      { '  An up-to-date NVIDIA GPU driver. CUDA/cuDNN DLLs are bundled — no install needed.' }
    'tensorrt'  { '  An up-to-date NVIDIA GPU driver. TensorRT runtime DLLs are bundled — no install needed.' }
})
$trtNote$vst3Note
See licenses\ inside each bundle for third-party license texts.
"@
    Set-Content -Path (Join-Path $staging 'INSTALL.txt') -Value $install -Encoding UTF8

    # 5. Archive.
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $zip = Join-Path $OutDir "$stem.zip"
    if (Test-Path $zip) { Remove-Item -Force $zip }
    Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $zip -CompressionLevel Optimal
    Remove-Item -Recurse -Force $staging

    $size = (Get-Item $zip).Length
    Write-Host ("==> Done: {0} ({1:N1} MB)" -f $zip, ($size / 1MB)) -ForegroundColor Green
    Write-Host ("    Contents: {0}" -f (($artifacts | Split-Path -Leaf) -join ', '))
}
finally {
    Pop-Location
}
