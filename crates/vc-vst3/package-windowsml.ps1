<#
.SYNOPSIS
    Copy the Windows App SDK bootstrapper DLL into the plugin bundle.

.DESCRIPTION
    The default Windows ML build uses Windows App SDK Runtime 2.x for
    onnxruntime.dll and DirectML.dll, so those large DLLs are not bundled. The
    app/plugin still needs Microsoft.WindowsAppRuntime.Bootstrap.dll beside the
    binary so unpackaged hosts can add the runtime framework package to the
    process package graph.

    Run AFTER `cargo xtask bundle vc-vst3 --release`.

.PARAMETER BundleDir
    Directory containing the built bundles. Default: target\bundled.

.PARAMETER BundleName
    Name of the .vst3 bundle folder inside BundleDir to populate. Default
    vc-vst3.vst3 (the raw xtask output). package.ps1 passes the variant-specific
    staged name (e.g. vc-vst3-windowsml.vst3) so it can populate the per-variant
    staging copy instead of the shared target\bundled.

.PARAMETER FoundationVersion
    Microsoft.WindowsAppSDK.Foundation NuGet version containing the bootstrapper.

.PARAMETER BootstrapDll
    Existing bootstrapper DLL to copy. When omitted, the script downloads the
    Foundation NuGet package to a temp directory and extracts the DLL.
#>
param(
    [string]$BundleDir,
    [string]$BundleName = 'vc-vst3.vst3',
    [string]$FoundationVersion = '2.0.21',
    [string]$BootstrapDll
)

$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$licenseSrc = Join-Path $PSScriptRoot 'dist\licenses'
if (-not $BundleDir) { $BundleDir = Join-Path $repoRoot 'target\bundled' }
# Don't hard-fail here if the bundle dir is missing; let the bundle check below
# report the actionable "run cargo xtask bundle first" message instead.
if (Test-Path $BundleDir) { $BundleDir = (Resolve-Path $BundleDir).Path }

# When downloading the nupkg we also grab its license.txt — the bootstrapper we
# redistribute is under Microsoft's Windows App SDK license terms, which require
# the license to travel with the redistributed DLL.
$SdkLicense = $null
if (-not $BootstrapDll) {
    $cache = Join-Path ([System.IO.Path]::GetTempPath()) "vc-rs-windowsappsdk-foundation-$FoundationVersion"
    $nupkg = Join-Path $cache "Microsoft.WindowsAppSDK.Foundation.$FoundationVersion.nupkg"
    $extract = Join-Path $cache 'nupkg'
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    if (-not (Test-Path $nupkg)) {
        $url = "https://api.nuget.org/v3-flatcontainer/microsoft.windowsappsdk.foundation/$FoundationVersion/microsoft.windowsappsdk.foundation.$FoundationVersion.nupkg"
        Invoke-WebRequest -Uri $url -OutFile $nupkg
    }
    Expand-Archive -LiteralPath $nupkg -DestinationPath $extract -Force
    $BootstrapDll = Join-Path $extract 'runtimes\win-x64\native\Microsoft.WindowsAppRuntime.Bootstrap.dll'
    $nupkgLicense = Join-Path $extract 'license.txt'
    if (Test-Path $nupkgLicense) { $SdkLicense = $nupkgLicense }
}

if (-not (Test-Path $BootstrapDll)) {
    throw "Bootstrap DLL not found: $BootstrapDll"
}

$dests = @()
$vst3Bin = Join-Path $BundleDir "$BundleName\Contents\x86_64-win"
if (Test-Path $vst3Bin) { $dests += $vst3Bin }
if (-not $dests) { throw "No bundle '$BundleName' found in $BundleDir. Run 'cargo xtask bundle vc-vst3 --release' first." }

foreach ($dest in $dests) {
    Copy-Item $BootstrapDll (Join-Path $dest 'Microsoft.WindowsAppRuntime.Bootstrap.dll') -Force

    # Third-party license texts next to the binary (onnxruntime + notices), plus
    # the Windows App SDK license for the bootstrapper we just copied.
    $licDest = Join-Path $dest 'licenses'
    New-Item -ItemType Directory -Force -Path $licDest | Out-Null
    if (Test-Path $licenseSrc) {
        Copy-Item -Path (Join-Path $licenseSrc '*') -Destination $licDest -Recurse -Force
    }
    if ($SdkLicense) {
        Copy-Item $SdkLicense (Join-Path $licDest 'WindowsAppSDK-LICENSE.txt') -Force
    }
}

if (-not $SdkLicense) {
    Write-Warning @"
Windows App SDK license text was not bundled (you passed -BootstrapDll directly,
so the nupkg license.txt was not available). The redistributed bootstrapper is
under Microsoft's Windows App SDK license terms — copy the matching license.txt
into each bundle's licenses\WindowsAppSDK-LICENSE.txt before shipping.
"@
}

Write-Host "Done: bundled Microsoft.WindowsAppRuntime.Bootstrap.dll + licenses into $($dests.Count) location(s)." -ForegroundColor Green
