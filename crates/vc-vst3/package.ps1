<#
.SYNOPSIS
    Build, populate, and zip a distributable vc-vst3 plugin package end to end.

.DESCRIPTION
    One command that produces a ready-to-ship archive for a given backend
    variant. It runs the whole pipeline:

        1. cargo xtask bundle vc-vst3 --release  (with the variant's features)
        2. (tensorrt only) build the ORT-free engine builder helper if needed
        2b. regenerate dist\licenses\THIRD-PARTY-LICENSES.md with cargo-about so
            the bundled Rust-crate notice matches this variant's dependencies
        3. the matching populate script (package-windowsml|tensorrt.ps1),
           which copies the runtime DLLs + licenses into the bundle
        4. stage vc-vst3-<variant>.vst3 + LICENSE + a generated INSTALL.txt
        5. Compress-Archive into dist\vc-vst3-<variant>-v<version>-win-x64.zip

    The populate scripts are reused as-is; this only orchestrates them and adds
    the build + archive steps. Variant-specific options are forwarded to the
    populate script (see the parameters below and the examples).

    Toolchain note: the tensorrt build compiles native code that needs the
    matching CUDA/TensorRT toolchain reachable (e.g. dot-source
    scripts\activate.ps1 first). This script does not modify your environment;
    set it up before running so the tensorrt (CUDA 13.x / TensorRT) build links
    correctly.

.PARAMETER Variant
    Which backend package to build: windowsml (default) or tensorrt.

.PARAMETER OutDir
    Where to write the .zip. Default: <repo>\dist.

.PARAMETER SkipBuild
    Reuse an existing target\bundled bundle instead of running cargo xtask
    bundle. The populate step and archiving still run.

.PARAMETER NoZip
    Populate the staged bundle but stop before creating the .zip (useful for
    inspecting the populated dist\<stem>\ folder).

.PARAMETER Clean
    Deprecated / implied. The build now always starts from a clean target\bundled
    (and populate runs against a fresh per-variant dist\<stem>\ copy), so DLLs
    from a previous variant can no longer linger. Accepted for back-compat; it has
    no additional effect and is ignored with -SkipBuild.

.PARAMETER KeepStage
    Keep the staged, install-ready dist\<stem>\ folder beside the .zip. By default
    it is kept for windowsml and removed for tensorrt (which can be multiple GB).
    Use this to keep a tensorrt folder too.

.PARAMETER CleanStage
    Remove the staged dist\<stem>\ folder after zipping, overriding the
    default-keep for windowsml.

.PARAMETER SkipLicenseGen
    Skip running cargo-about to regenerate dist\licenses\THIRD-PARTY-LICENSES.md.
    The committed copy is shipped as-is. Generation is also skipped automatically
    (with a warning) when cargo-about is not installed.

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
    # Self-contained TensorRT package for an RTX 30xx (sm86):
    pwsh crates\vc-vst3\package.ps1 -Variant tensorrt -BuilderSm sm86
#>
[CmdletBinding()]
param(
    [ValidateSet('windowsml', 'tensorrt')]
    [string]$Variant = 'windowsml',
    [string]$OutDir,
    [switch]$SkipBuild,
    [switch]$NoZip,
    [switch]$Clean,
    # Keep the staged, install-ready dist\<stem>\ folder beside the .zip. By
    # default it is kept for windowsml and removed for tensorrt (which can be
    # multi-GB). -KeepStage forces keep; -CleanStage forces removal.
    [switch]$KeepStage,
    [switch]$CleanStage,

    # Skip regenerating dist\licenses\THIRD-PARTY-LICENSES.md with cargo-about;
    # ship the committed copy as-is.
    [switch]$SkipLicenseGen,

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
$installBundleName = "vc-vst3-$Variant.vst3"

# Feature flags per variant for `cargo xtask bundle`. windowsml is the default
# feature set, so it needs no extra flags.
$bundleFeatureArgs = switch ($Variant) {
    'windowsml' { @() }
    'tensorrt' { @('--no-default-features', '--features', 'tensorrt') }
}

Push-Location $repoRoot
try {
    # 1. Build the bundle into a CLEAN target\bundled. We wipe it first on every
    #    build (not just on -Clean): `cargo xtask bundle` always writes to the same
    #    target\bundled\vc-vst3.vst3 regardless of features and rebuilds only the
    #    .vst3 DLL, while the populate step drops loose runtime DLLs (Bootstrap /
    #    nvinfer* / cudart) beside it. Those sidecars are not build outputs, so a
    #    stale set from a previous variant — or from a prior validate/install that
    #    populated target\bundled in place — would otherwise survive into the copy
    #    we stage below. That is exactly how a windowsml package once shipped >1 GB
    #    of leftover TensorRT DLLs (which crash the windowsml plugin in DAWs).
    if (-not $SkipBuild) {
        if (Test-Path $bundleDir) { Remove-Item -Recurse -Force $bundleDir }
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

    # 2b. Regenerate the third-party Rust-crate license notice so the shipped
    #     THIRD-PARTY-LICENSES.md reflects THIS variant's dependency tree (e.g.
    #     the tensorrt build drops ORT). It lives in dist\licenses\, which the
    #     populate step copies into the bundle, so this must run first. The
    #     feature flags match the bundle build. Skipped gracefully when
    #     cargo-about is absent so packaging never hard-depends on it.
    if (-not $SkipLicenseGen) {
        if (Get-Command cargo-about -ErrorAction SilentlyContinue) {
            $aboutCfg = Join-Path $PSScriptRoot 'about.toml'
            $aboutTpl = Join-Path $PSScriptRoot 'about.hbs'
            $licOut = Join-Path $PSScriptRoot 'dist\licenses\THIRD-PARTY-LICENSES.md'
            Write-Host "==> cargo about generate ($Variant features)" -ForegroundColor Cyan
            cargo about generate --manifest-path $PSScriptRoot\Cargo.toml -c $aboutCfg --locked @bundleFeatureArgs $aboutTpl -o $licOut
            if ($LASTEXITCODE -ne 0) { throw "cargo about generate failed (exit $LASTEXITCODE)." }
        }
        else {
            Write-Warning @"
cargo-about not found; shipping the committed THIRD-PARTY-LICENSES.md as-is.
Install it to regenerate the notice: cargo install cargo-about --features cli
"@
        }
    }

    # 3. Locate the raw, DLL-only bundle that `cargo xtask bundle` produced. We do
    #    NOT populate it in place — instead we copy it into a fresh per-variant
    #    staging dir and populate that copy, so target\bundled stays the pristine
    #    build output and variants can never cross-contaminate each other.
    $rawVst3 = Join-Path $bundleDir 'vc-vst3.vst3'
    if (-not (Test-Path $rawVst3)) { throw "No vc-vst3.vst3 found in $bundleDir." }

    # Resolve version/tag/stem now so the staging dir can be built up front.
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

    # Stage into a fresh dist\<stem>\ and copy the raw bundle in under its
    # variant-specific install name (so Windows ML and TensorRT packages can live
    # in the same VST3 search path without overwriting each other). The staging
    # dir is wiped per run, so the copy starts from exactly the xtask output with
    # no leftover sidecar DLLs.
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $staging = Join-Path $OutDir $stem
    if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
    New-Item -ItemType Directory -Force -Path $staging | Out-Null

    $stagedBundle = Join-Path $staging $installBundleName
    Copy-Item -Path $rawVst3 -Destination $stagedBundle -Recurse -Force

    # 4. Populate the STAGED bundle with the variant's runtime DLLs + licenses,
    #    forwarding only the parameters the caller actually supplied. -BundleName
    #    points the populate script at the variant-named staged bundle instead of
    #    the default vc-vst3.vst3 in target\bundled.
    $populateScript = Join-Path $PSScriptRoot "package-$Variant.ps1"
    if (-not (Test-Path $populateScript)) { throw "Populate script not found: $populateScript" }

    $forward = @{ BundleDir = $staging; BundleName = $installBundleName }
    $forwardable = switch ($Variant) {
        'windowsml' { @('FoundationVersion', 'BootstrapDll') }
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

    if ($NoZip) {
        Write-Host "==> -NoZip: populated bundle ready in $staging (skipping archive)." -ForegroundColor Green
        return
    }

    $license = Join-Path $repoRoot 'LICENSE'
    if (Test-Path $license) { Copy-Item $license (Join-Path $staging 'LICENSE') -Force }

    # Ship the optional model downloader at the package root — NOT inside the
    # .vst3 bundle, which installs into %CommonProgramFiles% (admin-only). It
    # fetches the shared models into .\assets\ beside itself; point the plugin
    # GUI at those files (model paths are not auto-discovered).
    $modelDl = Join-Path $repoRoot 'download-models.ps1'
    if (Test-Path $modelDl) { Copy-Item $modelDl (Join-Path $staging 'download-models.ps1') -Force }

    $trtNote = if ($Variant -eq 'tensorrt' -and -not $RuntimeOnly) {
        @"

First-run TensorRT engine builds use the bundled vc-tensorrt-builder.exe, which
sits next to the plugin DLL. The plugin finds it automatically (resolved against
its own module directory, not the DAW exe), so no env var or PATH setup is
needed. To override the path: setx VC_RS_TENSORRT_BUILDER_HELPER "<path>\vc-tensorrt-builder.exe"
"@
    }
    else { '' }

    $install = @"
vc-vst3 — RVC voice conversion plugin ($Variant build, v$version)

Install — copy the $installBundleName bundle into a standard VST3 search path:
  %CommonProgramFiles%\VST3\   (e.g. C:\Program Files\Common Files\VST3)

Models — get the shared embedder + F0 models (optional helper):
    pwsh .\download-models.ps1
Run it from THIS folder (not from inside the installed plugin). It downloads
ContentVec + RMVPE into .\assets\. In the plugin GUI, browse to
assets\content_vec_500.onnx (embedder) and assets\rmvpe.onnx (F0), plus your own
RVC voice model (.onnx). The downloaded models are third-party (GPL-3.0 upstream),
not covered by this package's MIT license — see download-models.ps1.

Requirements for this ($Variant) build:
$(switch ($Variant) {
    'windowsml' { '  Windows App SDK Runtime 2.x installed (provides ONNX Runtime + DirectML).' }
    'tensorrt'  { '  An up-to-date NVIDIA GPU driver. TensorRT runtime DLLs are bundled — no install needed.' }
})
$trtNote
See licenses\ inside each bundle for third-party license texts.
"@
    Set-Content -Path (Join-Path $staging 'INSTALL.txt') -Value $install -Encoding UTF8

    # 5. Archive.
    $zip = Join-Path $OutDir "$stem.zip"
    if (Test-Path $zip) { Remove-Item -Force $zip }
    Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $zip -CompressionLevel Optimal

    # 6. Keep or remove the install-ready dir. Default: keep windowsml (small,
    #    handy for testing), drop tensorrt (can be multi-GB). Flags override.
    if ($KeepStage -and $CleanStage) { throw "Pass only one of -KeepStage / -CleanStage." }
    $keepDir = if ($KeepStage) { $true } elseif ($CleanStage) { $false } else { $Variant -eq 'windowsml' }
    if ($keepDir) {
        Write-Host "==> Kept install-ready dir: $staging" -ForegroundColor Green
    }
    else {
        Remove-Item -Recurse -Force $staging
    }

    $size = (Get-Item $zip).Length
    Write-Host ("==> Done: {0} ({1:N1} MB)" -f $zip, ($size / 1MB)) -ForegroundColor Green
    Write-Host "    Contents: $installBundleName"
}
finally {
    Pop-Location
}
