<#
.SYNOPSIS
    Check the real release scanner against small synthetic ZIPs, without SDKs.
#>
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$versionText = Get-Content -LiteralPath (Join-Path $repoRoot 'Cargo.toml') -Raw
if ($versionText -notmatch '(?ms)\[workspace\.package\].*?^\s*version\s*=\s*"([^"]+)"') {
    throw 'Workspace version missing'
}
$releaseVersion = $Matches[1]
$fixtureRoot = Join-Path $repoRoot ('target/release-scan-tests/' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $fixtureRoot -Force | Out-Null
$marker = 'VC_RS_RELEASE_SCAN_TEST'
$moduleEntry = 'vc-vst3-windowsml.vst3/Contents/x86_64-win/vc-vst3-windowsml.vst3'
$pluginZip = Join-Path $fixtureRoot "vc-vst3-windowsml-v$releaseVersion-win-x64.zip"

function Write-ZipEntry {
    param($Zip, [string]$Name, [byte[]]$Bytes)
    $entry = $Zip.CreateEntry($Name)
    $stream = $entry.Open()
    try { $stream.Write($Bytes, 0, $Bytes.Length) } finally { $stream.Dispose() }
}

foreach ($kind in @('vc-rs', 'vc-vst3')) {
    foreach ($variant in @('windowsml', 'tensorrt')) {
        $path = Join-Path $fixtureRoot "$kind-$variant-v$releaseVersion-win-x64.zip"
        $zip = [System.IO.Compression.ZipFile]::Open($path, [System.IO.Compression.ZipArchiveMode]::Create)
        try {
            $plain = [System.Text.Encoding]::UTF8.GetBytes('synthetic fixture')
            Write-ZipEntry $zip 'LICENSE' $plain
            Write-ZipEntry $zip 'licenses/THIRD-PARTY-LICENSES.md' $plain
            if ($kind -eq 'vc-rs') {
                Write-ZipEntry $zip 'vc-rs.exe' $plain
                Write-ZipEntry $zip 'vc-gui.exe' $plain
            } else {
                Write-ZipEntry $zip "vc-vst3-$variant.vst3/Contents/x86_64-win/vc-vst3-$variant.vst3" $plain
            }
        } finally { $zip.Dispose() }
    }
}

function Assert-Scan {
    param([string]$Name, [bool]$Pass, [string]$Expected)
    $log = Join-Path $fixtureRoot "$Name.log"
    # A child process isolates release.ps1's errors and nonzero exit status.
    & pwsh -NoProfile -File (Join-Path $PSScriptRoot 'release.ps1') `
        -DistDir $fixtureRoot -ScanPattern $marker *> $log
    $code = $LASTEXITCODE
    $output = Get-Content -LiteralPath $log -Raw
    if (($Pass -and $code -ne 0) -or (-not $Pass -and $code -eq 0)) {
        throw "$Name returned unexpected exit code $code; see $log"
    }
    if ($Expected -and -not $output.Contains($Expected)) {
        throw "$Name did not report the expected violation; see $log"
    }
    Write-Host "$Name passed"
}

Assert-Scan 'clean-fixtures' $true ''
foreach ($encoding in @([System.Text.Encoding]::UTF8, [System.Text.Encoding]::Unicode)) {
    $zip = [System.IO.Compression.ZipFile]::Open($pluginZip, [System.IO.Compression.ZipArchiveMode]::Update)
    try {
        $zip.GetEntry($moduleEntry).Delete()
        Write-ZipEntry $zip $moduleEntry ($encoding.GetBytes($marker))
    } finally { $zip.Dispose() }
    Assert-Scan "vst3-leak-$($encoding.WebName)" $false "leaked string '$marker' in $moduleEntry"
}
$zip = [System.IO.Compression.ZipFile]::Open($pluginZip, [System.IO.Compression.ZipArchiveMode]::Update)
try {
    $zip.GetEntry($moduleEntry).Delete()
    Write-ZipEntry $zip $moduleEntry ([System.Text.Encoding]::UTF8.GetBytes('synthetic fixture'))
} finally { $zip.Dispose() }
$appZip = Join-Path $fixtureRoot "vc-rs-windowsml-v$releaseVersion-win-x64.zip"
$zip = [System.IO.Compression.ZipFile]::Open($appZip, [System.IO.Compression.ZipArchiveMode]::Update)
try {
    Write-ZipEntry $zip 'nvinfer_11.dll' ([System.Text.Encoding]::UTF8.GetBytes('synthetic fixture'))
} finally { $zip.Dispose() }
Assert-Scan 'backend-contamination' $false 'prohibited file: nvinfer_11.dll'
Write-Host 'Release ZIP scanner: all four checks passed.'
