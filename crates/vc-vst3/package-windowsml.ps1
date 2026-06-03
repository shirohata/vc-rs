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

.PARAMETER FoundationVersion
    Microsoft.WindowsAppSDK.Foundation NuGet version containing the bootstrapper.

.PARAMETER BootstrapDll
    Existing bootstrapper DLL to copy. When omitted, the script downloads the
    Foundation NuGet package to a temp directory and extracts the DLL.
#>
param(
    [string]$BundleDir,
    [string]$FoundationVersion = '2.0.21',
    [string]$BootstrapDll
)

$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
if (-not $BundleDir) { $BundleDir = Join-Path $repoRoot 'target\bundled' }
# Don't hard-fail here if the bundle dir is missing; let the bundle check below
# report the actionable "run cargo xtask bundle first" message instead.
if (Test-Path $BundleDir) { $BundleDir = (Resolve-Path $BundleDir).Path }

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
}

if (-not (Test-Path $BootstrapDll)) {
    throw "Bootstrap DLL not found: $BootstrapDll"
}

$dests = @()
$vst3Bin = Join-Path $BundleDir 'vc-vst3.vst3\Contents\x86_64-win'
if (Test-Path $vst3Bin) { $dests += $vst3Bin }
if (Test-Path (Join-Path $BundleDir 'vc-vst3.clap')) { $dests += $BundleDir }
if (-not $dests) { throw "No bundle found in $BundleDir. Run 'cargo xtask bundle vc-vst3 --release' first." }

foreach ($dest in $dests) {
    Copy-Item $BootstrapDll (Join-Path $dest 'Microsoft.WindowsAppRuntime.Bootstrap.dll') -Force
}

Write-Host "Done: bundled Microsoft.WindowsAppRuntime.Bootstrap.dll into $($dests.Count) location(s)." -ForegroundColor Green
