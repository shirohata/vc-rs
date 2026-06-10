<#
.SYNOPSIS
    Copy the Windows App SDK bootstrapper DLL next to the vc-rs CLI executable.

.DESCRIPTION
    The Windows ML build loads onnxruntime.dll and DirectML.dll from the Windows
    App SDK Runtime 2.x at runtime, so those large DLLs are not bundled. The
    process still needs Microsoft.WindowsAppRuntime.Bootstrap.dll beside vc-rs.exe
    so an unpackaged process can add the runtime framework package to its package
    graph (see crates/vc-core/src/windows_ml.rs; override the lookup with
    VC_RS_WINDOWSML_BOOTSTRAP_DLL).

    Run AFTER building the CLI, e.g.:
        cargo build --release -p vc-cli --no-default-features --features windowsml,rnnoise

.PARAMETER DestDir
    Directory holding vc-rs.exe to populate. Default: target\release.

.PARAMETER FoundationVersion
    Microsoft.WindowsAppSDK.Foundation NuGet version containing the bootstrapper.

.PARAMETER BootstrapDll
    Existing bootstrapper DLL to copy. When omitted, the script downloads the
    Foundation NuGet package to a temp directory and extracts the DLL.

.PARAMETER WindowsAppSdkLicense
    License text matching -BootstrapDll. Required when passing the DLL directly;
    otherwise it is extracted from the downloaded NuGet package.
#>
[CmdletBinding()]
param(
    [string]$DestDir,
    [string]$FoundationVersion = '2.0.21',
    [string]$BootstrapDll,
    [string]$WindowsAppSdkLicense
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$licenseSrc = Join-Path $repoRoot 'scripts\licenses\static'
if (-not (Test-Path $licenseSrc)) { throw "Static license material not found: $licenseSrc" }

if (-not $DestDir) { $DestDir = Join-Path $repoRoot 'target\release' }
if (-not (Test-Path $DestDir)) { throw "DestDir not found: $DestDir" }
$DestDir = (Resolve-Path $DestDir).Path
if (-not (Test-Path (Join-Path $DestDir 'vc-rs.exe'))) {
    throw "vc-rs.exe not found in $DestDir. Build it first: cargo build --release -p vc-cli --no-default-features --features windowsml,rnnoise"
}

# When downloading the nupkg we also grab its license.txt — the Windows App SDK
# redistributable (the bootstrapper we ship) is under Microsoft's proprietary
# license terms, which require the license to travel with the redistributed DLL.
$SdkLicense = $WindowsAppSdkLicense
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
if (-not $SdkLicense -or -not (Test-Path $SdkLicense)) {
    throw "Windows App SDK license not found. Omit -BootstrapDll to download it with the NuGet package, or pass -WindowsAppSdkLicense."
}

Copy-Item $BootstrapDll (Join-Path $DestDir 'Microsoft.WindowsAppRuntime.Bootstrap.dll') -Force

# Third-party license texts next to the binary.
$licDest = Join-Path $DestDir 'licenses'
New-Item -ItemType Directory -Force -Path $licDest | Out-Null
if (Test-Path $licenseSrc) {
    Copy-Item -Path (Join-Path $licenseSrc '*') -Destination $licDest -Recurse -Force
}

# Bundle the Windows App SDK license alongside the bootstrapper we redistribute.
Copy-Item $SdkLicense (Join-Path $licDest 'WindowsAppSDK-LICENSE.txt') -Force

Write-Host "Done: bundled Microsoft.WindowsAppRuntime.Bootstrap.dll + licenses into $DestDir." -ForegroundColor Green
