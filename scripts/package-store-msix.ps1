<#
.SYNOPSIS
    Build a Microsoft Store-oriented Windows ML GUI MSIX package.

.DESCRIPTION
    Creates an unpackaged-desktop/full-trust MSIX from the existing
    app-windowsml package output. The normal distribution can remain ZIP-based;
    this script prepares a smaller Store input that excludes vc-rs.exe, VST3
    files, TensorRT files, model caches, and developer-machine state.

    For Partner Center submission, pass the package identity values reserved for
    the Store app:
      -PackageName <Partner Center package identity name>
      -Publisher   <Partner Center publisher, e.g. CN=...>

    Packaging is performed with the Windows App Development CLI (`winapp`).
    The generated .msix is unsigned unless -PfxPath is provided. Store ingestion
    can sign submitted packages, but local installation requires a trusted
    signing certificate whose subject matches -Publisher.

.PARAMETER BuildPackage
    Build the app-windowsml package first via scripts\package-all.ps1.

.PARAMETER OutDir
    Directory containing package outputs and receiving the .msix.

.PARAMETER AppStageDir
    Explicit staged vc-rs-windowsml package directory. Overrides automatic lookup.

.PARAMETER WinApp
    Path to winapp.exe. If omitted, PATH/App Execution Alias lookup is used.

.PARAMETER PfxPath
    Optional code-signing certificate for local-installable MSIX output.

.PARAMETER PfxPassword
    Password for -PfxPath, if needed.

.PARAMETER CreateTestCertificate
    Create a local self-signed code-signing certificate, export it as a temporary
    PFX, set -Publisher to the certificate subject, and sign the MSIX with it.
    Intended only for local installation/testing, not Store submission.

.PARAMETER ForceNewTestCertificate
    With -CreateTestCertificate, replace any existing local test PFX/CER instead
    of reusing it.

.PARAMETER TrustTestCertificate
    With -CreateTestCertificate, import the certificate into
    Cert:\CurrentUser\TrustedPeople.

.PARAMETER TrustTestRootCertificate
    With -CreateTestCertificate, also import the certificate into the current
    user's Root store. This is required for local MSIX installation with a
    self-signed certificate, but it is a persistent trust decision; use only for
    disposable local test certificates you generated yourself.

.PARAMETER TrustTestMachineCertificate
    With -CreateTestCertificate, also import the certificate into the local
    machine Root and TrustedPeople stores. This requires an elevated PowerShell
    session and is performed through `winapp cert install`. Some Windows AppX
    deployment paths require machine-level trust even when Authenticode
    validation succeeds for the current user.

.PARAMETER PackageName
    MSIX package identity name. Replace this with the Partner Center identity
    before Store submission.

.PARAMETER WindowsAppRuntimeDependencyName
    Windows App Runtime framework package dependency name for Store MSIX builds.

.PARAMETER WindowsAppRuntimeDependencyMinVersion
    Minimum Windows App Runtime framework package version. vc-rs uses Windows ML
    APIs introduced in Windows App Runtime 2.1, so the default is 2.1.0.0.
    Keep this aligned with WINDOWS_APP_RUNTIME_MIN_VERSION in
    crates/vc-core/src/windows_ml.rs.

.PARAMETER IncludeWindowsAppRuntimeBootstrap
    Also copy Microsoft.WindowsAppRuntime.Bootstrap.dll into the MSIX. Store
    packages normally should not need this because the framework dependency is
    declared in the manifest; this is kept only as an escape hatch.

.PARAMETER Publisher
    MSIX package identity publisher. For signing, this must match the signing
    certificate subject.

.PARAMETER EmitOnly
    Validate inputs, prepare the MSIX staging directory, and print the winapp
    command without invoking it.

.EXAMPLE
    pwsh scripts\package-store-msix.ps1 -BuildPackage
#>
[CmdletBinding()]
param(
    [switch]$BuildPackage,
    [string]$OutDir,
    [string]$AppStageDir,
    [string]$WinApp,
    [string]$PfxPath,
    [string]$PfxPassword,
    [switch]$CreateTestCertificate,
    [switch]$ForceNewTestCertificate,
    [string]$TestCertificateSubject,
    [string]$TestCertificateDir,
    [string]$TestCertificatePassword = 'vc-rs-local-msix-test',
    [switch]$TrustTestCertificate,
    [switch]$TrustTestRootCertificate,
    [switch]$TrustTestMachineCertificate,
    [string]$PackageName = 'VcRs.WindowsML.GUI',
    [string]$Publisher = 'CN=vc-rs',
    [string]$PublisherDisplayName = 'vc-rs',
    [string]$DisplayName = 'vc-rs',
    [string]$Description = 'vc-rs Windows ML GUI',
    [string]$WindowsAppRuntimeDependencyName = 'Microsoft.WindowsAppRuntime.2',
    [string]$WindowsAppRuntimeDependencyPublisher = 'CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, S=Washington, C=US',
    [string]$WindowsAppRuntimeDependencyMinVersion = '2.1.0.0',
    [switch]$NoWindowsAppRuntimeDependency,
    [switch]$IncludeWindowsAppRuntimeBootstrap,
    [switch]$EmitOnly
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $OutDir) { $OutDir = Join-Path $repoRoot 'dist' }
$OutDir = [System.IO.Path]::GetFullPath($OutDir)

if ($CreateTestCertificate -and $PfxPath) {
    throw "Pass only one of -CreateTestCertificate or -PfxPath."
}
if ($TrustTestCertificate -and -not $CreateTestCertificate) {
    throw "-TrustTestCertificate requires -CreateTestCertificate."
}
if ($TrustTestRootCertificate -and -not $CreateTestCertificate) {
    throw "-TrustTestRootCertificate requires -CreateTestCertificate."
}
if ($TrustTestMachineCertificate -and -not $CreateTestCertificate) {
    throw "-TrustTestMachineCertificate requires -CreateTestCertificate."
}
if ($ForceNewTestCertificate -and -not $CreateTestCertificate) {
    throw "-ForceNewTestCertificate requires -CreateTestCertificate."
}
if ($CreateTestCertificate) {
    if (-not $TestCertificateSubject) { $TestCertificateSubject = 'CN=vc-rs-local-msix-test' }
    $Publisher = $TestCertificateSubject
}

function Get-WorkspaceVersion {
    $wsToml = Get-Content (Join-Path $repoRoot 'Cargo.toml') -Raw
    if ($wsToml -match '(?ms)\[workspace\.package\].*?^\s*version\s*=\s*"([^"]+)"') {
        return $Matches[1]
    }
    throw "Could not read [workspace.package].version from Cargo.toml."
}

function Convert-ToMsixVersion {
    param([Parameter(Mandatory)][string]$Version)

    $parts = @($Version.Split('.'))
    if ($parts.Count -gt 4) { throw "MSIX package version must have at most four numeric parts: $Version" }
    while ($parts.Count -lt 4) { $parts += '0' }
    foreach ($part in $parts) {
        $n = 0
        if (-not [int]::TryParse($part, [ref]$n) -or $n -lt 0 -or $n -gt 65535) {
            throw "MSIX package version parts must be integers from 0 to 65535: $Version"
        }
    }
    return ($parts -join '.')
}

function Assert-ChildPath {
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Child
    )

    $rootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    $childFull = [System.IO.Path]::GetFullPath($Child)
    if (-not $childFull.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to use path outside ${Root}: $Child"
    }
}

function Resolve-WinAppCli {
    param(
        [string]$ExplicitPath
    )

    if ($ExplicitPath) {
        return (Resolve-Path -LiteralPath $ExplicitPath -ErrorAction Stop).Path
    }

    $cmd = Get-Command 'winapp.exe' -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cmd = Get-Command 'winapp' -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    throw "winapp.exe was not found. Install the Windows App Development CLI with: winget install Microsoft.WinAppCli"
}

function Resolve-AppPackageStage {
    param(
        [Parameter(Mandatory)][string]$Stem,
        [string]$ExplicitDir
    )

    if ($ExplicitDir) {
        return (Resolve-Path -LiteralPath $ExplicitDir -ErrorAction Stop).Path
    }

    $stageDir = Join-Path $OutDir $Stem
    if (Test-Path -LiteralPath $stageDir) {
        return (Resolve-Path -LiteralPath $stageDir).Path
    }

    $zip = Join-Path $OutDir "$Stem.zip"
    if (-not (Test-Path -LiteralPath $zip)) {
        throw "Missing app-windowsml package stage and ZIP for $Stem. Run scripts/package-all.ps1 -Targets app-windowsml or pass -BuildPackage."
    }

    $extractDir = Join-Path $repoRoot "tmp\store-msix-input\$Stem"
    Assert-ChildPath -Root $repoRoot -Child $extractDir
    if (Test-Path -LiteralPath $extractDir) { Remove-Item -LiteralPath $extractDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
    Expand-Archive -LiteralPath $zip -DestinationPath $extractDir -Force
    return (Resolve-Path -LiteralPath $extractDir).Path
}

function New-LocalMsixTestCertificate {
    param(
        [Parameter(Mandatory)][string]$Subject,
        [Parameter(Mandatory)][string]$DestinationDir,
        [string]$Password,
        [switch]$ForceNew,
        [switch]$Trust,
        [switch]$TrustRoot,
        [switch]$TrustMachine
    )

    Assert-ChildPath -Root $repoRoot -Child $DestinationDir
    New-Item -ItemType Directory -Force -Path $DestinationDir | Out-Null

    $effectivePassword = $Password
    $safeName = ($Subject -replace '[^A-Za-z0-9_.-]', '_').Trim('_')
    if (-not $safeName) { $safeName = 'vc-rs-msix-test' }
    $pfx = Join-Path $DestinationDir "$safeName.pfx"
    $cer = Join-Path $DestinationDir "$safeName.cer"

    $winappExe = Resolve-WinAppCli -ExplicitPath $WinApp

    $cert = $null
    if ((Test-Path -LiteralPath $pfx) -and -not $ForceNew) {
        Write-Host "==> Reusing local MSIX test certificate PFX: $pfx" -ForegroundColor Cyan
        try {
            $cert = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new($pfx, $effectivePassword)
        }
        catch {
            throw "Existing test PFX could not be opened with the provided password. Pass -ForceNewTestCertificate to replace it: $pfx"
        }
        if (($Trust -or $TrustRoot -or $TrustMachine) -and -not (Test-Path -LiteralPath $cer)) {
            Export-Certificate -Cert $cert -FilePath $cer | Out-Null
        }
    }
    else {
        if ($ForceNew) {
            if (Test-Path -LiteralPath $pfx) { Remove-Item -LiteralPath $pfx -Force }
            if (Test-Path -LiteralPath $cer) { Remove-Item -LiteralPath $cer -Force }
        }

        Write-Host "==> Creating local MSIX test certificate: $Subject" -ForegroundColor Cyan
        # Keep test-certificate creation on the same CLI path as packaging. The PFX
        # remains under tmp/ so future agents do not accidentally commit a private
        # key, and --if-exists Skip preserves trust across repeated test builds.
        $ifExists = if ($ForceNew) { 'Overwrite' } else { 'Skip' }
        $certGenerateArgs = @(
            'cert', 'generate',
            '--publisher', $Subject,
            '--output', $pfx,
            '--password', $effectivePassword,
            '--valid-days', '730',
            '--if-exists', $ifExists,
            '--export-cer'
        )
        Write-Host "==> $winappExe cert generate --publisher $Subject --output $pfx --password <password elided> --valid-days 730 --if-exists $ifExists --export-cer" -ForegroundColor Cyan
        & $winappExe @certGenerateArgs
        if ($LASTEXITCODE -ne 0) { throw "winapp cert generate failed (exit $LASTEXITCODE)." }

        $cert = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new($pfx, $effectivePassword)
    }

    if ($Trust) {
        if (-not (Test-Path -LiteralPath $cer)) {
            throw "winapp did not export the expected CER file: $cer"
        }
        Import-Certificate -FilePath $cer -CertStoreLocation 'Cert:\CurrentUser\TrustedPeople' | Out-Null
        Write-Host "==> Trusted test certificate in Cert:\CurrentUser\TrustedPeople ($($cert.Thumbprint))" -ForegroundColor Cyan
    }

    if ($TrustRoot) {
        if (-not (Test-Path -LiteralPath $cer)) {
            throw "winapp did not export the expected CER file: $cer"
        }
        # MSIX install validates the full signing chain. For a self-signed local
        # test certificate, the certificate is also the root. Root trust is a
        # persistent local security decision, so callers must request it with the
        # explicit -TrustTestRootCertificate option.
        & certutil.exe -user -addstore Root $cer | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "certutil failed to trust the test certificate root (exit $LASTEXITCODE)." }
        Write-Host "==> Trusted test certificate root in Cert:\CurrentUser\Root ($($cert.Thumbprint))" -ForegroundColor Yellow
    }

    if ($TrustMachine) {
        if (-not (Test-Path -LiteralPath $cer)) {
            throw "winapp did not export the expected CER file: $cer"
        }
        # Some AppX deployment paths validate package trust through machine-level
        # stores. This requires elevation and is deliberately a separate option
        # from CurrentUser trust because it affects all users on the machine.
        & $winappExe cert install $pfx --password $effectivePassword
        if ($LASTEXITCODE -ne 0) { throw "winapp cert install failed (exit $LASTEXITCODE)." }
        Write-Host "==> Trusted test certificate in LocalMachine Root + TrustedPeople ($($cert.Thumbprint))" -ForegroundColor Yellow
    }

    [pscustomobject]@{
        PfxPath = $pfx
        Password = $effectivePassword
        Thumbprint = $cert.Thumbprint
        Subject = $Subject
    }
}

function Write-LogoPng {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][int]$Size,
        [string]$Text = 'vc'
    )

    Add-Type -AssemblyName System.Drawing
    $bitmap = [System.Drawing.Bitmap]::new($Size, $Size)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
        $graphics.Clear([System.Drawing.Color]::FromArgb(22, 28, 36))
        $brush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(77, 171, 247))
        $graphics.FillRectangle($brush, 0, [Math]::Floor($Size * 0.70), $Size, [Math]::Ceiling($Size * 0.30))
        $brush.Dispose()

        $fontSize = [Math]::Max(10, [Math]::Floor($Size * 0.34))
        $font = [System.Drawing.Font]::new('Segoe UI', $fontSize, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
        $format = [System.Drawing.StringFormat]::new()
        $format.Alignment = [System.Drawing.StringAlignment]::Center
        $format.LineAlignment = [System.Drawing.StringAlignment]::Center
        $textBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::White)
        $graphics.DrawString($Text, $font, $textBrush, [System.Drawing.RectangleF]::new(0, 0, $Size, $Size), $format)
        $textBrush.Dispose()
        $format.Dispose()
        $font.Dispose()

        $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    }
    finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

function Write-Manifest {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$MsixVersion
    )

    $runtimeDependency = if ($NoWindowsAppRuntimeDependency) {
        ''
    }
    else {
        @"
    <PackageDependency
      Name="$([System.Security.SecurityElement]::Escape($WindowsAppRuntimeDependencyName))"
      Publisher="$([System.Security.SecurityElement]::Escape($WindowsAppRuntimeDependencyPublisher))"
      MinVersion="$([System.Security.SecurityElement]::Escape($WindowsAppRuntimeDependencyMinVersion))" />
"@
    }

    $manifest = @"
<?xml version="1.0" encoding="utf-8"?>
<Package
  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
  xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
  IgnorableNamespaces="uap rescap">
  <Identity Name="$([System.Security.SecurityElement]::Escape($PackageName))"
            Publisher="$([System.Security.SecurityElement]::Escape($Publisher))"
            Version="$MsixVersion"
            ProcessorArchitecture="x64" />
  <Properties>
    <DisplayName>$([System.Security.SecurityElement]::Escape($DisplayName))</DisplayName>
    <PublisherDisplayName>$([System.Security.SecurityElement]::Escape($PublisherDisplayName))</PublisherDisplayName>
    <Logo>Assets\StoreLogo.png</Logo>
  </Properties>
  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.17763.0" MaxVersionTested="10.0.26100.0" />
$runtimeDependency
  </Dependencies>
  <Resources>
    <Resource Language="en-us" />
  </Resources>
  <Applications>
    <Application Id="App" Executable="vc-gui.exe" EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements
        DisplayName="$([System.Security.SecurityElement]::Escape($DisplayName))"
        Description="$([System.Security.SecurityElement]::Escape($Description))"
        BackgroundColor="transparent"
        Square150x150Logo="Assets\Square150x150Logo.png"
        Square44x44Logo="Assets\Square44x44Logo.png" />
    </Application>
  </Applications>
  <Capabilities>
    <rescap:Capability Name="runFullTrust" />
  </Capabilities>
</Package>
"@
    Set-Content -Path $Path -Value $manifest -Encoding UTF8
}

$version = Get-WorkspaceVersion
$msixVersion = Convert-ToMsixVersion -Version $version
$stem = "vc-rs-windowsml-v$version-win-x64"

if ($BuildPackage) {
    Write-Host "==> Building app-windowsml package input" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot 'package-all.ps1') -Targets app-windowsml -OutDir $OutDir -KeepStage
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$appStage = Resolve-AppPackageStage -Stem $stem -ExplicitDir $AppStageDir

$required = @(
    'vc-gui.exe',
    'LICENSE',
    'licenses\THIRD-PARTY-LICENSES-vc-gui.md',
    'licenses\THIRD-PARTY-NOTICES.md',
    'licenses\WindowsAppSDK-LICENSE.txt'
)
if ($IncludeWindowsAppRuntimeBootstrap) {
    $required += 'Microsoft.WindowsAppRuntime.Bootstrap.dll'
}
foreach ($relative in $required) {
    $path = Join-Path $appStage $relative
    if (-not (Test-Path -LiteralPath $path)) {
        throw "app-windowsml package is missing ${relative}: $appStage"
    }
}

$msixStage = Join-Path $repoRoot "tmp\store-msix-stage\vc-rs-windowsml-gui-v$version-win-x64"
Assert-ChildPath -Root $repoRoot -Child $msixStage
if (Test-Path -LiteralPath $msixStage) { Remove-Item -LiteralPath $msixStage -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $msixStage 'licenses') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $msixStage 'Assets') | Out-Null

Copy-Item -LiteralPath (Join-Path $appStage 'vc-gui.exe') -Destination (Join-Path $msixStage 'vc-gui.exe') -Force
if ($IncludeWindowsAppRuntimeBootstrap) {
    Copy-Item -LiteralPath (Join-Path $appStage 'Microsoft.WindowsAppRuntime.Bootstrap.dll') -Destination (Join-Path $msixStage 'Microsoft.WindowsAppRuntime.Bootstrap.dll') -Force
}
Copy-Item -LiteralPath (Join-Path $appStage 'LICENSE') -Destination (Join-Path $msixStage 'LICENSE') -Force
Copy-Item -LiteralPath (Join-Path $appStage 'licenses\THIRD-PARTY-LICENSES-vc-gui.md') -Destination (Join-Path $msixStage 'licenses\THIRD-PARTY-LICENSES-vc-gui.md') -Force
Copy-Item -LiteralPath (Join-Path $appStage 'licenses\THIRD-PARTY-NOTICES.md') -Destination (Join-Path $msixStage 'licenses\THIRD-PARTY-NOTICES.md') -Force
Copy-Item -LiteralPath (Join-Path $appStage 'licenses\WindowsAppSDK-LICENSE.txt') -Destination (Join-Path $msixStage 'licenses\WindowsAppSDK-LICENSE.txt') -Force

$optionalOnnxLicense = Join-Path $appStage 'licenses\onnxruntime.LICENSE.txt'
if (Test-Path -LiteralPath $optionalOnnxLicense) {
    Copy-Item -LiteralPath $optionalOnnxLicense -Destination (Join-Path $msixStage 'licenses\onnxruntime.LICENSE.txt') -Force
}

$installNote = @"
vc-rs Windows ML GUI (v$version)

This Microsoft Store MSIX contains only the vc-rs GUI and declares the Windows
App Runtime framework dependency needed by the Windows ML backend. It
intentionally excludes the CLI, VST3 plugin, TensorRT runtime files, local
models, caches, logs, and development artifacts.

Models are not bundled. Select your own RVC voice model and compatible embedder
/ F0 model files from the GUI.

See licenses\ for third-party license texts.
"@
Set-Content -Path (Join-Path $msixStage 'INSTALL.txt') -Value $installNote -Encoding UTF8

Write-LogoPng -Path (Join-Path $msixStage 'Assets\Square44x44Logo.png') -Size 44
Write-LogoPng -Path (Join-Path $msixStage 'Assets\Square150x150Logo.png') -Size 150
Write-LogoPng -Path (Join-Path $msixStage 'Assets\StoreLogo.png') -Size 50
Write-Manifest -Path (Join-Path $msixStage 'AppxManifest.xml') -MsixVersion $msixVersion

$winappExe = Resolve-WinAppCli -ExplicitPath $WinApp
$msix = Join-Path $OutDir "vc-rs-windowsml-gui-store-v$version-win-x64.msix"

if ($EmitOnly) {
    Write-Host "==> Store MSIX stage: $msixStage" -ForegroundColor Cyan
    Write-Host "==> winapp package command:" -ForegroundColor Cyan
    $emitArgs = @(
        'package', $msixStage,
        '--manifest', (Join-Path $msixStage 'AppxManifest.xml'),
        '--output', $msix,
        '--name', $PackageName,
        '--exe', 'vc-gui.exe'
    )
    if ($PfxPath) { $emitArgs += @('--cert', $PfxPath, '--cert-password', '<password elided>') }
    Write-Host "$winappExe $($emitArgs -join ' ')"
    Write-Host "==> Expected MSIX: $msix" -ForegroundColor Cyan
    if ($PfxPath) { Write-Host "==> Signing requested with: $PfxPath" -ForegroundColor Cyan }
    if ($CreateTestCertificate) {
        Write-Host "==> Would create local test certificate with subject: $TestCertificateSubject" -ForegroundColor Cyan
        if ($TrustTestCertificate) { Write-Host "==> Would trust it in Cert:\CurrentUser\TrustedPeople" -ForegroundColor Cyan }
        if ($TrustTestRootCertificate) { Write-Host "==> Would trust it in Cert:\CurrentUser\Root" -ForegroundColor Yellow }
        if ($TrustTestMachineCertificate) { Write-Host "==> Would trust it in LocalMachine Root + TrustedPeople (requires elevation)" -ForegroundColor Yellow }
    }
    return
}

if ($CreateTestCertificate) {
    if (-not $TestCertificateDir) {
        $TestCertificateDir = Join-Path $repoRoot 'tmp\store-msix-cert'
    }
    $testCert = New-LocalMsixTestCertificate `
        -Subject $TestCertificateSubject `
        -DestinationDir $TestCertificateDir `
        -Password $TestCertificatePassword `
        -ForceNew:$ForceNewTestCertificate `
        -Trust:$TrustTestCertificate `
        -TrustRoot:$TrustTestRootCertificate `
        -TrustMachine:$TrustTestMachineCertificate
    $PfxPath = $testCert.PfxPath
    $PfxPassword = $testCert.Password
}

$packageArgs = @(
    'package', $msixStage,
    '--manifest', (Join-Path $msixStage 'AppxManifest.xml'),
    '--output', $msix,
    '--name', $PackageName,
    '--exe', 'vc-gui.exe'
)

if ($PfxPath) {
    $resolvedPfx = (Resolve-Path -LiteralPath $PfxPath -ErrorAction Stop).Path
    $packageArgs += @('--cert', $resolvedPfx)
    if ($PfxPassword) { $packageArgs += @('--cert-password', $PfxPassword) }
}

if (Test-Path -LiteralPath $msix) { Remove-Item -LiteralPath $msix -Force }
$displayPackageArgs = @($packageArgs)
for ($i = 0; $i -lt $displayPackageArgs.Count; $i++) {
    if ($displayPackageArgs[$i] -eq '--cert-password' -and ($i + 1) -lt $displayPackageArgs.Count) {
        $displayPackageArgs[$i + 1] = '<password elided>'
    }
}
Write-Host "==> $winappExe $($displayPackageArgs -join ' ')" -ForegroundColor Cyan
& $winappExe @packageArgs
if ($LASTEXITCODE -ne 0) { throw "winapp package failed (exit $LASTEXITCODE)." }
if (-not (Test-Path -LiteralPath $msix)) { throw "MSIX build completed, but expected output was not found: $msix" }

$size = (Get-Item -LiteralPath $msix).Length
Write-Host ("==> Done: {0} ({1:N1} MB)" -f $msix, ($size / 1MB)) -ForegroundColor Green
if (-not $PfxPath) {
    Write-Host "==> Unsigned MSIX. Local installation requires signing with a trusted certificate; Store submission can use Partner Center signing." -ForegroundColor Yellow
}
elseif ($CreateTestCertificate) {
    Write-Host "==> Signed with local test certificate: $TestCertificateSubject" -ForegroundColor Yellow
    Write-Host "==> This certificate is for local testing only; use Partner Center identity/signing for Store submission." -ForegroundColor Yellow
}
