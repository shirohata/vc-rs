<#
.SYNOPSIS
    A/B compare the audio output of two vc-rs versions/builds on the same input,
    using the deterministic CPU `wav` conversion path.

.DESCRIPTION
    Each side (-RefA / -RefB) can be:
      * a git ref (branch / tag / commit) - checked out into a temporary
        worktree, built with `--no-default-features --features <feat>`, and run;
      * a path to a built vc-rs.exe - run directly;
      * a path to an existing .wav - used as-is (no conversion).

    Both conversions use `--provider cpu`, which is deterministic run-to-run
    (unlike the GPU EPs), so any metric difference reflects a real code/output
    change rather than inference jitter. The two outputs are then compared with
    tools/audio_compare (max abs diff, relative RMS, log-spectral distance), and
    this script exits non-zero if any metric exceeds its threshold.

    The RVC speaker model is not shipped with the repo: pass -Model or set
    $env:VC_RS_TEST_RVC_MODEL. The reference ContentVec / RMVPE models default to
    ./assets (populate them with `just models`).

.PARAMETER Features
    Cargo features used when building a git ref (default 'cpu'). Override per side
    with -FeaturesA / -FeaturesB - older refs predate the 'cpu' feature and need
    e.g. 'windowsml' (still run with --provider cpu).

.EXAMPLE
    # Compare two branches on one clip (RVC model from the env var).
    $env:VC_RS_TEST_RVC_MODEL = 'C:\models\voice.onnx'
    pwsh -File scripts/compare-audio.ps1 -RefA main -RefB dev -Input clip.wav

.EXAMPLE
    # Regression check vs a pre-'cpu'-feature commit (build it with windowsml).
    pwsh -File scripts/compare-audio.ps1 -RefA 135e2b1 -RefB HEAD -Input clip.wav -FeaturesA windowsml

.EXAMPLE
    # Compare two already-produced WAVs (no conversion, no model needed).
    pwsh -File scripts/compare-audio.ps1 -RefA old.wav -RefB new.wav
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$RefA,
    [Parameter(Mandatory)] [string]$RefB,

    # Required only when a side needs conversion (i.e. is a git ref or an exe).
    # Named -InputWav (not -Input) to avoid shadowing the automatic $Input var.
    [string]$InputWav,

    [string]$Model = $env:VC_RS_TEST_RVC_MODEL,
    [string]$Embedder = 'assets/content_vec_500.onnx',
    [string]$F0Model = 'assets/rmvpe.onnx',

    # Conversion knobs (forwarded to `vc-rs wav`). -ExtraArgs covers the rest.
    [uint32]$ChunkMs = 0,
    [double]$PitchShift = 0,
    [long]$SpeakerId = 0,
    [string[]]$ExtraArgs = @(),

    # Build features for git-ref sides.
    [string]$Features = 'cpu',
    [string]$FeaturesA,
    [string]$FeaturesB,

    # Comparator thresholds + STFT settings.
    [double]$MaxAbs = 1e-4,
    [double]$MaxRelRms = 1e-3,
    [double]$MaxLsdDb = 0.5,
    [uint32]$FftSize = 1024,
    [uint32]$Hop = 256,
    [switch]$Json,

    [string]$OutDir,
    [switch]$KeepArtifacts
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

# Match the build/test recipes' RUSTFLAGS so git-ref builds share the cache and
# do not leak build-machine paths.
. (Join-Path $PSScriptRoot 'rustflags.ps1')

if (-not $OutDir) {
    $OutDir = Join-Path ([System.IO.Path]::GetTempPath()) ("vc-rs-compare-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$worktrees = New-Object System.Collections.Generic.List[string]

function Resolve-PathMaybe([string]$p) {
    if ([string]::IsNullOrWhiteSpace($p)) { return $null }
    $rp = Resolve-Path -LiteralPath $p -ErrorAction SilentlyContinue
    if ($rp) { return $rp.Path } else { return $null }
}

function Invoke-WavConversion([string]$Exe, [string]$OutWav) {
    if (-not $InputWav) { throw "An input clip is required for conversion. Pass -InputWav <wav>." }
    $inputPath = Resolve-PathMaybe $InputWav
    if (-not $inputPath) { throw "Input WAV not found: $InputWav" }
    if (-not $Model) {
        throw "No RVC model. Pass -Model <path> or set `$env:VC_RS_TEST_RVC_MODEL. Reference models: run `just models`."
    }
    $modelPath = Resolve-PathMaybe $Model
    if (-not $modelPath) { throw "RVC model not found: $Model" }
    $embPath = Resolve-PathMaybe $Embedder
    if (-not $embPath) { throw "Embedder not found: $Embedder (run `just models` to fetch reference models)." }
    $f0Path = Resolve-PathMaybe $F0Model
    if (-not $f0Path) { throw "F0 model not found: $F0Model (run `just models` to fetch reference models)." }

    $wavArgs = @(
        'wav',
        '--provider', 'cpu',
        '--input', $inputPath,
        '--output', $OutWav,
        '--model', $modelPath,
        '--embedder', $embPath,
        '--f0-model', $f0Path,
        '--pitch-shift', $PitchShift,
        '--speaker-id', $SpeakerId
    )
    if ($ChunkMs -gt 0) { $wavArgs += @('--chunk-ms', $ChunkMs) }
    $wavArgs += $ExtraArgs

    Write-Host "    $Exe $($wavArgs -join ' ')" -ForegroundColor DarkGray
    & $Exe @wavArgs
    if ($LASTEXITCODE -ne 0) { throw "vc-rs wav failed (exit $LASTEXITCODE)" }
}

function Build-RefExe([string]$Ref, [string]$Label, [string]$Feat) {
    $wt = Join-Path $OutDir "wt-$Label"
    Write-Host "==> [$Label] git worktree add $Ref" -ForegroundColor Cyan
    git -C $repoRoot worktree add --detach $wt $Ref
    if ($LASTEXITCODE -ne 0) { throw "git worktree add failed for '$Ref'" }
    $worktrees.Add($wt)

    Write-Host "==> [$Label] cargo build -p vc-cli --no-default-features --features $Feat" -ForegroundColor Cyan
    Push-Location $wt
    try {
        cargo build --release -p vc-cli --no-default-features --features $Feat
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed for ref '$Ref' (features '$Feat')" }
    } finally {
        Pop-Location
    }
    $exe = Join-Path $wt 'target\release\vc-rs.exe'
    if (-not (Test-Path -LiteralPath $exe)) { throw "built vc-rs.exe not found at $exe" }
    return $exe
}

# Resolve one side (ref|exe|wav) to an output WAV path.
function Resolve-Side([string]$Ref, [string]$Label, [string]$Feat) {
    $resolved = Resolve-PathMaybe $Ref
    if ($resolved -and $resolved.ToLower().EndsWith('.wav')) {
        Write-Host "==> [$Label] using existing WAV: $resolved" -ForegroundColor Cyan
        return $resolved
    }
    $outWav = Join-Path $OutDir "out_$Label.wav"
    if ($resolved -and $resolved.ToLower().EndsWith('.exe')) {
        Write-Host "==> [$Label] converting with exe: $resolved" -ForegroundColor Cyan
        Invoke-WavConversion -Exe $resolved -OutWav $outWav
        return $outWav
    }
    # Otherwise treat as a git ref.
    $exe = Build-RefExe -Ref $Ref -Label $Label -Feat $Feat
    Write-Host "==> [$Label] converting clip" -ForegroundColor Cyan
    Invoke-WavConversion -Exe $exe -OutWav $outWav
    return $outWav
}

Push-Location $repoRoot
try {
    if (-not $FeaturesA) { $FeaturesA = $Features }
    if (-not $FeaturesB) { $FeaturesB = $Features }

    $wavA = Resolve-Side -Ref $RefA -Label 'a' -Feat $FeaturesA
    $wavB = Resolve-Side -Ref $RefB -Label 'b' -Feat $FeaturesB

    Write-Host "==> building comparator (tools/audio_compare)" -ForegroundColor Cyan
    cargo build --release --manifest-path tools/audio_compare/Cargo.toml
    if ($LASTEXITCODE -ne 0) { throw "failed to build audio_compare" }
    $comparator = Join-Path $repoRoot 'tools\audio_compare\target\release\audio_compare.exe'

    $cmpArgs = @(
        '--a', $wavA,
        '--b', $wavB,
        '--max-abs', $MaxAbs,
        '--max-rel-rms', $MaxRelRms,
        '--max-lsd-db', $MaxLsdDb,
        '--fft-size', $FftSize,
        '--hop', $Hop
    )
    if ($Json) { $cmpArgs += '--json' }

    Write-Host ""
    & $comparator @cmpArgs
    $script:cmpExit = $LASTEXITCODE

    if ($script:cmpExit -eq 0) {
        Write-Host "== compare-audio: PASS ==" -ForegroundColor Green
    } elseif ($script:cmpExit -eq 1) {
        Write-Host "== compare-audio: FAIL (metrics over threshold) ==" -ForegroundColor Yellow
    } else {
        throw "audio_compare errored (exit $script:cmpExit)"
    }
}
finally {
    Pop-Location
    foreach ($wt in $worktrees) {
        git -C $repoRoot worktree remove --force $wt 2>$null
    }
    if (-not $KeepArtifacts) {
        Remove-Item -Recurse -Force -LiteralPath $OutDir -ErrorAction SilentlyContinue
    } else {
        Write-Host "artifacts kept in $OutDir" -ForegroundColor DarkGray
    }
}

# Propagate the comparator's pass/fail as this script's exit code (after cleanup).
exit $script:cmpExit
