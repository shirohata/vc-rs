<#
.SYNOPSIS
    Turn the built distribution ZIPs into a verified, publishable release.

.DESCRIPTION
    This is the repeatable back half of the release process described in
    docs\distribution.md. It does the mechanical, judgement-free steps so a
    release comes out the same way every time:

      1. Resolve the release version from [workspace.package] in Cargo.toml.
      2. (optional, -Build) build all four ZIPs via scripts\package-all.ps1.
      3. Confirm the four canonical ZIPs for that version exist in -DistDir.
      4. Scan each ZIP for release blockers: prohibited files, backend
         cross-contamination, build-machine paths / user names leaked into our
         own binaries, and missing required files (LICENSE, notices, binaries).
      5. (optional, -Publish) re-run CI's fmt + clippy quality gate on the
         committed state, then require the remote release branch (main by
         default) to fast-forward to the current commit, then atomically push
         that branch update and the annotated v<version> tag before creating a
         GitHub release with all four ZIPs attached. GitHub shows a SHA-256
         digest for each uploaded asset, so no separate checksum files are
         produced.

    Steps 1, 3, 4 are read-only / local and run by default. Step 5 is the only
    outward-facing action and is gated behind -Publish.

    Judgement still lives with you: bumping [workspace.package].version and
    curating CHANGELOG.md are deliberately NOT automated (see docs\distribution.md
    "Versioning"). This script verifies their results rather than guessing them.

.PARAMETER Version
    Release version to operate on. Default: read from the root Cargo.toml
    [workspace.package].version (the same field package.ps1 uses to name ZIPs).

.PARAMETER DistDir
    Directory holding the built ZIPs. Default: <repo>\dist.

.PARAMETER Build
    Build the four ZIPs first via scripts\package-all.ps1 (needs the GPU
    toolchain on PATH for the tensorrt targets; dot-source scripts\activate.ps1).

.PARAMETER Publish
    Fast-forward the remote release branch to the current commit, push the
    annotated git tag, and create the GitHub release. Without this switch the
    script only verifies locally. Requires the `gh` CLI.

.PARAMETER Tag
    Tag name to create when publishing. Default: v<version>.

.PARAMETER Remote
    Git remote whose release branch and tag are pushed when publishing.
    Default: origin.

.PARAMETER ReleaseBranch
    Remote branch advanced to the current commit when publishing. Its fetched
    tip must be an ancestor of the current commit, so the update is
    fast-forward-only. Default: main.

.PARAMETER Draft
    Create the GitHub release as a draft (recommended for a final human review
    before it goes public).

.PARAMETER ScanPattern
    Extra literal strings to treat as forbidden inside our own binaries (in
    addition to the build-machine home path and the current user name).

.EXAMPLE
    # Verify the already-built ZIPs (no publish):
    pwsh -File scripts/release.ps1

.EXAMPLE
    # Build everything, verify, then publish a draft release:
    . scripts/activate.ps1
    pwsh -File scripts/release.ps1 -Build -Publish -Draft
#>
[CmdletBinding()]
param(
    [string]$Version,
    [string]$DistDir,
    [switch]$Build,
    [switch]$Publish,
    [string]$Tag,
    [string]$Remote = 'origin',
    [string]$ReleaseBranch = 'main',
    [switch]$Draft,
    [string[]]$ScanPattern = @()
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $DistDir) { $DistDir = Join-Path $repoRoot 'dist' }

Add-Type -AssemblyName System.IO.Compression.FileSystem

# ---- helpers ---------------------------------------------------------------

function Get-WorkspaceVersion {
    # Single source of truth: [workspace.package] version in the root Cargo.toml,
    # the same field package.ps1 reads to name the archives. Keep this regex in
    # sync with crates\vc-cli\package.ps1.
    $wsToml = Get-Content (Join-Path $repoRoot 'Cargo.toml') -Raw
    if ($wsToml -match '(?ms)\[workspace\.package\].*?^\s*version\s*=\s*"([^"]+)"') {
        return $Matches[1]
    }
    throw "Could not read [workspace.package].version from Cargo.toml."
}

function Get-ChangelogReleaseNotes {
    param([string]$ChangelogPath, [string]$ReleaseVersion)

    if (-not (Test-Path -LiteralPath $ChangelogPath)) {
        throw "CHANGELOG.md not found: $ChangelogPath"
    }

    # Keep-A-Changelog versions are second-level headings. Stop at the next
    # version heading so GitHub Release notes never include older releases.
    $escapedVersion = [regex]::Escape($ReleaseVersion)
    $content = Get-Content -LiteralPath $ChangelogPath -Raw
    $match = [regex]::Match(
        $content,
        "(?ms)^## \[$escapedVersion\][^\r\n]*\r?\n.*?(?=^## \[|\z)"
    )
    if (-not $match.Success) {
        throw "CHANGELOG.md has no release section for version $ReleaseVersion."
    }
    return $match.Value.Trim()
}

function Test-BytesContainText {
    # Look for $needle inside a binary as both narrow (Latin1/ASCII) and wide
    # (UTF-16LE) text — the two ways a path or name ends up embedded in a PE.
    param([byte[]]$Bytes, [string]$Needle)
    $latin1 = [System.Text.Encoding]::Latin1.GetString($Bytes)
    if ($latin1.IndexOf($Needle, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) { return $true }
    $wide = [System.Text.Encoding]::Unicode.GetString($Bytes)
    return ($wide.IndexOf($Needle, [System.StringComparison]::OrdinalIgnoreCase) -ge 0)
}

# Files that are ours and therefore must not leak build-machine paths/names.
# Third-party vendor DLLs (NVIDIA/Microsoft) legitimately carry their own build
# paths, so the deep path scan is limited to binaries we produce.
$ourBinaryGlobs = @('vc-rs.exe', 'vc-gui.exe', 'vc-tensorrt-builder.exe', 'vc-vst3*.dll')

# Prohibited anywhere in any package (docs\distribution.md "Package Contents").
$prohibitedGlobs = @('*.pdb', '*.onnx', '*.wav', '*.log', '*.tmp')

# Backend isolation (docs\distribution.md "Backend Isolation").
# windowsml carries only the Windows App SDK bootstrapper — no ORT/DML/CUDA/TRT.
$windowsmlForbiddenGlobs = @(
    'onnxruntime*.dll', 'DirectML.dll', 'nvinfer*.dll', 'nvonnxparser*.dll',
    'cudart*.dll', 'cudnn*.dll', 'cublas*.dll'
)
# tensorrt must not carry ORT (the build has no ORT); it DOES carry nvinfer etc.
$tensorrtForbiddenGlobs = @('onnxruntime*.dll')

function Classify-Zip {
    param([string]$Name)
    $kind = if ($Name -like 'vc-vst3-*') { 'vst3' } else { 'app' }
    $variant = if ($Name -like '*-tensorrt-*') { 'tensorrt' } else { 'windowsml' }
    return [pscustomobject]@{ Kind = $kind; Variant = $variant }
}

function Invoke-ZipScan {
    # Returns a list of violation strings ($empty = clean). Enumerates entries
    # without a full extract; only our own binaries are extracted to scan bytes.
    param([string]$ZipPath, [string[]]$ForbiddenStrings)

    $violations = [System.Collections.Generic.List[string]]::new()
    $cls = Classify-Zip ([System.IO.Path]::GetFileName($ZipPath))
    $forbiddenGlobs = $prohibitedGlobs + $(if ($cls.Variant -eq 'tensorrt') { $tensorrtForbiddenGlobs } else { $windowsmlForbiddenGlobs })

    $zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        $entries = $zip.Entries | Where-Object { $_.Name }  # skip directory entries
        $names = $entries.FullName

        # 1. Prohibited / wrong-backend files by name.
        foreach ($e in $entries) {
            foreach ($g in $forbiddenGlobs) {
                if ($e.Name -like $g) { $violations.Add("prohibited file: $($e.FullName)"); break }
            }
        }

        # 2. Required files.
        if (-not ($names | Where-Object { (Split-Path $_ -Leaf) -ieq 'LICENSE' })) {
            $violations.Add('missing LICENSE')
        }
        if (-not ($names | Where-Object { $_ -like '*licenses/*' -or $_ -like '*licenses\*' })) {
            $violations.Add('missing licenses/ notices directory')
        }
        if ($cls.Kind -eq 'app') {
            foreach ($req in @('vc-rs.exe', 'vc-gui.exe')) {
                if (-not ($names | Where-Object { (Split-Path $_ -Leaf) -ieq $req })) {
                    $violations.Add("missing required binary: $req")
                }
            }
        } else {
            if (-not ($names | Where-Object { $_ -like '*.vst3/*' -or $_ -like '*.vst3\*' })) {
                $violations.Add('missing .vst3 bundle')
            }
        }

        # 3. Deep string scan of our own binaries for leaked paths / names.
        foreach ($e in $entries) {
            $isOurs = $false
            foreach ($g in $ourBinaryGlobs) { if ($e.Name -like $g) { $isOurs = $true; break } }
            if (-not $isOurs) { continue }

            $ms = New-Object System.IO.MemoryStream
            $stream = $e.Open()
            try { $stream.CopyTo($ms) } finally { $stream.Dispose() }
            $bytes = $ms.ToArray()
            $ms.Dispose()

            foreach ($needle in $ForbiddenStrings) {
                if (Test-BytesContainText -Bytes $bytes -Needle $needle) {
                    $violations.Add("leaked string '$needle' in $($e.FullName)")
                }
            }
        }
    }
    finally {
        $zip.Dispose()
    }
    return $violations
}

# ---- 1. version ------------------------------------------------------------

if (-not $Version) { $Version = Get-WorkspaceVersion }
if (-not $Tag) { $Tag = "v$Version" }
Write-Host "==> Release version: $Version (tag $Tag)" -ForegroundColor Cyan

# ---- 2. optional build -----------------------------------------------------

if ($Build) {
    Write-Host "==> Building all four packages" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot 'package-all.ps1') -OutDir $DistDir
    if ($LASTEXITCODE -ne 0) { throw "package-all.ps1 failed (exit $LASTEXITCODE)" }
}

# ---- 3. locate the canonical ZIPs ------------------------------------------

$expected = @(
    "vc-rs-windowsml-v$Version-win-x64.zip",
    "vc-rs-tensorrt-v$Version-win-x64.zip",
    "vc-vst3-windowsml-v$Version-win-x64.zip",
    "vc-vst3-tensorrt-v$Version-win-x64.zip"
)
$zips = foreach ($name in $expected) {
    $p = Join-Path $DistDir $name
    if (-not (Test-Path -LiteralPath $p)) {
        throw "Expected release ZIP not found: $p`nBuild it first (e.g. pwsh -File scripts/release.ps1 -Build)."
    }
    (Resolve-Path -LiteralPath $p).Path
}
Write-Host "==> Found all four ZIPs in $DistDir" -ForegroundColor Cyan

# ---- 4. scan ---------------------------------------------------------------

# Forbidden strings inside our own binaries: the build-machine home root (paths
# should have been remapped by scripts\rustflags.ps1) and the current user name.
$forbidden = @('C:\Users\') + $(if ($env:USERNAME) { @($env:USERNAME) } else { @() }) + $ScanPattern
$forbidden = $forbidden | Where-Object { $_ } | Select-Object -Unique

$allViolations = [System.Collections.Generic.List[string]]::new()
foreach ($zip in $zips) {
    Write-Host "    scan $([System.IO.Path]::GetFileName($zip))"
    $v = Invoke-ZipScan -ZipPath $zip -ForbiddenStrings $forbidden
    foreach ($item in $v) { $allViolations.Add("$([System.IO.Path]::GetFileName($zip)): $item") }
}
if ($allViolations.Count -gt 0) {
    Write-Host ''
    Write-Host '== RELEASE BLOCKED: scan found problems ==' -ForegroundColor Red
    $allViolations | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    throw "Scan found $($allViolations.Count) release blocker(s). Nothing was published."
}
Write-Host "==> Scan clean (no prohibited files, cross-contamination, or leaked paths)" -ForegroundColor Green

# ---- 5. optional publish ---------------------------------------------------

if (-not $Publish) {
    Write-Host ''
    Write-Host "==> Verified. Re-run with -Publish to tag and create the GitHub release." -ForegroundColor Cyan
    return
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "The GitHub CLI (gh) is required for -Publish but was not found on PATH."
}

$changelog = Join-Path $repoRoot 'CHANGELOG.md'
$releaseNotes = Get-ChangelogReleaseNotes -ChangelogPath $changelog -ReleaseVersion $Version

$worktreeChanges = @(git -C $repoRoot status --porcelain --untracked-files=normal)
if ($LASTEXITCODE -ne 0) { throw "git status failed (exit $LASTEXITCODE)" }
if ($worktreeChanges.Count -gt 0) {
    throw "Publishing requires a clean worktree. Commit or remove local changes first."
}

# ---- 5a. lint + format gate ------------------------------------------------

# CI's fmt + clippy job is the quality gate the published commit must pass, but
# CI runs asynchronously after push and was never a precondition for tagging --
# v0.4.0 was cut from a commit whose clippy job had failed. Re-run the same two
# commands here (against the now-confirmed-clean worktree) so a release can never
# be published from a commit that fails them. These are compile-only checks and
# need no GPU DLLs on PATH; keep the commands identical to .github/workflows/ci.yml
# and `just lint` so the gate matches what CI enforces.
Push-Location $repoRoot
try {
    Write-Host "==> Lint gate: cargo fmt --all --check" -ForegroundColor Cyan
    cargo fmt --all --check
    if ($LASTEXITCODE -ne 0) {
        throw "rustfmt check failed; release blocked. Run 'cargo fmt --all', recommit, and retry."
    }

    Write-Host "==> Lint gate: cargo clippy --workspace --all-targets -- -D warnings" -ForegroundColor Cyan
    cargo clippy --workspace --all-targets -- -D warnings
    if ($LASTEXITCODE -ne 0) {
        throw "clippy reported warnings/errors; release blocked. Fix them, recommit, and retry."
    }
}
finally {
    Pop-Location
}
Write-Host "==> Lint gate clean (fmt + clippy)" -ForegroundColor Green

$headCommit = (git -C $repoRoot rev-parse HEAD)
if ($LASTEXITCODE -ne 0) { throw "git rev-parse HEAD failed (exit $LASTEXITCODE)" }
$headCommit = $headCommit.Trim()

# Fetch the exact remote release branch immediately before the ancestry check.
# The subsequent normal/atomic push independently rejects a race that makes the
# update non-fast-forward after this check.
$remoteBranchRef = "refs/remotes/$Remote/$ReleaseBranch"
$remoteBranchSource = "refs/heads/$ReleaseBranch"
Write-Host "==> Fetching $Remote/$ReleaseBranch for fast-forward check" -ForegroundColor Cyan
git -C $repoRoot fetch --no-tags $Remote "+${remoteBranchSource}:${remoteBranchRef}"
if ($LASTEXITCODE -ne 0) { throw "git fetch $Remote $ReleaseBranch failed (exit $LASTEXITCODE)" }

git -C $repoRoot merge-base --is-ancestor $remoteBranchRef $headCommit
if ($LASTEXITCODE -ne 0) {
    throw "$Remote/$ReleaseBranch cannot fast-forward to the current commit $headCommit. Merge or rebase first; nothing was published."
}

# Tag must point at the committed release state. Create it if missing; never move
# an existing tag silently (a moved tag would mismatch already-built binaries).
$existingTagCommit = (git -C $repoRoot rev-parse --verify --quiet "refs/tags/$Tag^{}")
if ($LASTEXITCODE -eq 0) {
    $existingTagCommit = $existingTagCommit.Trim()
    if ($existingTagCommit -ne $headCommit) {
        throw "Tag $Tag already points to $existingTagCommit, not current commit $headCommit."
    }
    Write-Host "==> Tag $Tag already exists at the current commit" -ForegroundColor Yellow
} else {
    Write-Host "==> Creating annotated tag $Tag" -ForegroundColor Cyan
    git -C $repoRoot tag -a $Tag -m "Release $Tag"
    if ($LASTEXITCODE -ne 0) { throw "git tag failed (exit $LASTEXITCODE)" }
}

Write-Host "==> Atomically pushing HEAD -> $Remote/$ReleaseBranch and tag $Tag" -ForegroundColor Cyan
git -C $repoRoot push --atomic $Remote "HEAD:${remoteBranchSource}" "refs/tags/${Tag}:refs/tags/${Tag}"
if ($LASTEXITCODE -ne 0) {
    throw "Atomic push of $Remote/$ReleaseBranch and $Tag failed (exit $LASTEXITCODE). Nothing was released."
}

# GitHub shows a SHA-256 digest for each uploaded asset, so only the ZIPs are
# attached — no separate checksum sidecars.
$assets = @($zips)

$notesFile = [System.IO.Path]::GetTempFileName()
try {
    Set-Content -LiteralPath $notesFile -Value $releaseNotes -Encoding utf8NoBOM
    $ghArgs = @('release', 'create', $Tag, '--title', $Tag, '--notes-file', $notesFile)
    if ($Draft) { $ghArgs += '--draft' }
    $ghArgs += $assets

    Write-Host "==> gh release create $Tag with CHANGELOG.md section for $Version" -ForegroundColor Cyan
    gh @ghArgs
    if ($LASTEXITCODE -ne 0) { throw "gh release create failed (exit $LASTEXITCODE)" }
}
finally {
    Remove-Item -LiteralPath $notesFile -Force -ErrorAction SilentlyContinue
}

Write-Host ''
Write-Host "== Release $Tag published ==" -ForegroundColor Green
Write-Host "Remember: binaries are not code-signed — keep the Windows-warning note in the release body." -ForegroundColor Yellow
