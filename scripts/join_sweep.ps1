<#
.SYNOPSIS
    Sweep chunk / crossfade / SOLA-search settings through the deterministic CPU
    `wav --join-report` path and aggregate the per-chunk seam metrics, to find
    where chunk-join quality degrades at low latency (chunk_ms < ~100 ms).

.DESCRIPTION
    For every (clip x chunk_ms x crossfade_ms x sola_search_ms x smoother)
    combination this runs `vc-rs wav --provider cpu --join-report <csv>` and then
    aggregates the CSV (crates/vc-cli/src/join_report.rs) into one summary row:

        step_ratio  : seam discontinuity vs local roughness (~1 = clean, >>1 = click)
        correlation : SOLA/PSOLA alignment score at the chosen offset (1 = best)
        capped_pct  : fraction of seams where the crossfade hit the 3/4-of-chunk cap
        fallback_pct: fraction of voiced seams where PSOLA fell back to plain SOLA

    Timing JSONs separate warmup/drain from steady-state smoothing and total
    processing. Summary timing columns are offline wall times, not device latency.

    `--provider cpu` is deterministic run-to-run, so differences reflect the
    parameters, not inference jitter. SOLA/crossfade runs in the worker-side model
    domain. Seam metrics cover the boundary only; listen to the full overlap too.

    The RVC speaker model is not shipped: pass -Model or set
    $env:VC_RS_TEST_RVC_MODEL. Reference ContentVec / RMVPE default to ./assets
    (populate with `just models`). Use a handful of representative general-speech
    clips (male + female, with silence / consonants / sustained vowels).

.EXAMPLE
    # Default low-latency grid on two clips, model from the env var.
    $env:VC_RS_TEST_RVC_MODEL = 'C:\models\voice.onnx'
    pwsh -File scripts/join_sweep.ps1 -InputWav male.wav,female.wav -KeepArtifacts

.EXAMPLE
    # Probe the crossfade cap by forcing crossfade as a fraction of each chunk.
    pwsh -File scripts/join_sweep.ps1 -InputWav clip.wav `
         -ChunkMs 30,50,70,90 -CrossfadeRatios 0.5,0.6,0.7,0.8 -SolaSearchMs 12
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string[]]$InputWav,

    [string]$Model = $env:VC_RS_TEST_RVC_MODEL,
    [string]$Embedder = 'assets/content_vec_500.onnx',
    [string]$F0Model = 'assets/rmvpe.onnx',

    # Grid. Numeric lists are [string[]] so `-ChunkMs 30,50,90` works under
    # `pwsh -File` (where a comma token would otherwise coerce to one number via
    # the thousands separator); they are split + parsed below.
    # Chunk sizes target the <100 ms regime plus a clean baseline.
    [string[]]$ChunkMs = @('30', '50', '70', '90', '200'),
    # Absolute crossfade targets (ms). 85 = "let the 3/4-of-chunk cap decide".
    [string[]]$CrossfadeMs = @('85'),
    # Optional: also add crossfade = round(chunk_ms * ratio) per chunk, to probe
    # the cap directly. Empty by default.
    [string[]]$CrossfadeRatios = @(),
    [string[]]$SolaSearchMs = @('12', '16', '20'),
    [string[]]$Smoother = @('sola', 'psola', 'phase-vocoder'),
    [ValidateRange(0, 10000)] [int]$WarmupChunks = 3,

    [double]$PitchShift = 0,
    [long]$SpeakerId = 0,

    # Prebuilt vc-rs.exe to use instead of building the cpu binary here.
    [string]$Exe,

    [string]$OutDir,
    [switch]$KeepArtifacts
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

# Match the build recipes' RUSTFLAGS so a local build shares the cache and does
# not leak build-machine paths.
. (Join-Path $PSScriptRoot 'rustflags.ps1')

if (-not $OutDir) {
    $OutDir = Join-Path ([System.IO.Path]::GetTempPath()) ("vc-rs-joinsweep-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
}
# Only clean up the fresh directory owned by this run.
if (Test-Path -LiteralPath $OutDir) { throw "OutDir must be new: $OutDir" }
$OutDir = (New-Item -ItemType Directory -Path $OutDir).FullName
$ownedOutDir = $OutDir

function Resolve-PathMaybe([string]$p) {
    if ([string]::IsNullOrWhiteSpace($p)) { return $null }
    $rp = Resolve-Path -LiteralPath $p -ErrorAction SilentlyContinue
    if ($rp) { return $rp.Path } else { return $null }
}

function Percentile([double[]]$values, [double]$p) {
    if ($values.Count -eq 0) { return 0.0 }
    $sorted = $values | Sort-Object
    # Nearest-rank: rank = ceil(p * n), clamped to [1, n].
    $rank = [math]::Ceiling($p * $sorted.Count)
    if ($rank -lt 1) { $rank = 1 }
    if ($rank -gt $sorted.Count) { $rank = $sorted.Count }
    return [double]$sorted[$rank - 1]
}

# Split each token on ',' (so a single "30,50" element expands) and parse with
# the invariant culture, so a comma is never read as a thousands separator.
function Expand-Numbers([string[]]$tokens, [bool]$asInt) {
    $out = @()
    foreach ($t in $tokens) {
        foreach ($piece in ($t -split ',')) {
            $piece = $piece.Trim()
            if ($piece -eq '') { continue }
            if ($asInt) {
                $out += [int]::Parse($piece, [System.Globalization.CultureInfo]::InvariantCulture)
            } else {
                $out += [double]::Parse($piece, [System.Globalization.CultureInfo]::InvariantCulture)
            }
        }
    }
    return $out
}

# Parse into NEW, untyped variables. The params above are [string[]], and that
# type is "sticky": assigning parsed numbers back onto the same name re-coerces
# them to strings, which then makes `"30" * "0.3"` do string repetition instead
# of arithmetic. Fresh variables keep the real Int32/Double values.
$chunkList = @(Expand-Numbers $ChunkMs $true)
$crossfadeList = @(Expand-Numbers $CrossfadeMs $true)
$searchList = @(Expand-Numbers $SolaSearchMs $true)
$ratioList = @(Expand-Numbers $CrossfadeRatios $false)
$smootherList = @($Smoother | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() })
foreach ($sm in $smootherList) {
    if ($sm -notin @('sola', 'psola', 'phase-vocoder')) { throw "Unknown smoother: $sm" }
}

# ---- Resolve inputs / models --------------------------------------------------

if (-not $Model) {
    throw "No RVC model. Pass -Model <path> or set `$env:VC_RS_TEST_RVC_MODEL. Reference models: run `just models`."
}
$modelPath = Resolve-PathMaybe $Model
if (-not $modelPath) { throw "RVC model not found: $Model" }
$embPath = Resolve-PathMaybe $Embedder
if (-not $embPath) { throw "Embedder not found: $Embedder (run `just models` to fetch reference models)." }
$f0Path = Resolve-PathMaybe $F0Model
if (-not $f0Path) { throw "F0 model not found: $F0Model (run `just models` to fetch reference models)." }

$clips = @()
# Split on ',' too, so `-InputWav a.wav,b.wav` works under `pwsh -File` (which
# passes the comma-joined value as a single token).
foreach ($w in $InputWav) {
    foreach ($piece in ($w -split ',')) {
        $piece = $piece.Trim()
        if ($piece -eq '') { continue }
        $cp = Resolve-PathMaybe $piece
        if (-not $cp) { throw "Input WAV not found: $piece" }
        $clips += $cp
    }
}

# ---- Build the cpu binary once (unless an exe was supplied) --------------------

Push-Location $repoRoot
try {
    if ($Exe) {
        $exePath = Resolve-PathMaybe $Exe
        if (-not $exePath) { throw "vc-rs.exe not found: $Exe" }
    } else {
        Write-Host "==> cargo build --release -p vc-cli --no-default-features --features cpu" -ForegroundColor Cyan
        cargo build --release -p vc-cli --no-default-features --features cpu
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
        $exePath = Join-Path $repoRoot 'target\release\vc-rs.exe'
        if (-not (Test-Path -LiteralPath $exePath)) { throw "built vc-rs.exe not found at $exePath" }
    }

    $summary = New-Object System.Collections.Generic.List[object]

    foreach ($clip in $clips) {
        $clipName = [System.IO.Path]::GetFileNameWithoutExtension($clip)
        foreach ($chunk in $chunkList) {
            # Per-chunk crossfade set: absolute values plus optional ratio-derived ones.
            $crossfades = [System.Collections.Generic.List[uint32]]::new()
            foreach ($cf in $crossfadeList) { $crossfades.Add([uint32]$cf) }
            foreach ($r in $ratioList) {
                $crossfades.Add([uint32][math]::Round($chunk * $r))
            }
            $crossfades = $crossfades | Select-Object -Unique

            foreach ($cf in $crossfades) {
                foreach ($search in $searchList) {
                    foreach ($sm in $smootherList) {
                        $tag = "{0}_c{1}_x{2}_s{3}_{4}" -f $clipName, $chunk, $cf, $search, $sm
                        $outWav = Join-Path $OutDir ("out_$tag.wav")
                        $csv = Join-Path $OutDir ("join_$tag.csv")
                        $timingJson = Join-Path $OutDir ("timing_$tag.json")

                        $wavArgs = @(
                            'wav',
                            '--provider', 'cpu',
                            '--input', $clip,
                            '--output', $outWav,
                            '--model', $modelPath,
                            '--embedder', $embPath,
                            '--f0-model', $f0Path,
                            '--chunk-ms', $chunk,
                            '--crossfade-ms', $cf,
                            '--sola-search-ms', $search,
                            '--smoother', $sm,
                            '--pitch-shift', $PitchShift,
                            '--speaker-id', $SpeakerId,
                            '--join-report', $csv,
                            '--performance-report', $timingJson,
                            '--performance-warmup-chunks', $WarmupChunks
                        )

                        Write-Host ("==> {0}" -f $tag) -ForegroundColor Cyan
                        & $exePath @wavArgs | Out-Null
                        if ($LASTEXITCODE -ne 0) { throw "vc-rs wav failed for $tag (exit $LASTEXITCODE)" }
                        if (-not (Test-Path -LiteralPath $csv)) { throw "join report not written: $csv" }
                        $timing = Get-Content -LiteralPath $timingJson -Raw | ConvertFrom-Json

                        # Aggregate the per-chunk CSV. Chunk 0 has no seam; skip it.
                        $rows = Import-Csv -LiteralPath $csv | Where-Object { [int]$_.chunk -gt 0 }
                        $n = ($rows | Measure-Object).Count
                        if ($n -eq 0) {
                            Write-Host "    (single chunk: no seams)" -ForegroundColor DarkYellow
                            continue
                        }
                        $steps = @($rows | ForEach-Object { [double]$_.step_ratio })
                        $corrs = @($rows | ForEach-Object { [double]$_.correlation })
                        $capped = ($rows | Where-Object { $_.crossfade_capped -eq 'true' } | Measure-Object).Count
                        $fallbk = ($rows | Where-Object { $_.psola_fallback -eq 'true' } | Measure-Object).Count

                        $summary.Add([pscustomobject]@{
                                clip          = $clipName
                                chunk_ms      = [int]$chunk
                                crossfade_ms  = [int]$cf
                                sola_search   = [int]$search
                                smoother      = $sm
                                seams         = $n
                                step_p50      = [math]::Round((Percentile $steps 0.50), 2)
                                step_p95      = [math]::Round((Percentile $steps 0.95), 2)
                                step_max      = [math]::Round(($steps | Measure-Object -Maximum).Maximum, 2)
                                corr_min      = [math]::Round(($corrs | Measure-Object -Minimum).Minimum, 3)
                                corr_mean     = [math]::Round((($corrs | Measure-Object -Average).Average), 3)
                                capped_pct    = [math]::Round(100.0 * $capped / $n, 0)
                                fallback_pct  = [math]::Round(100.0 * $fallbk / $n, 0)
                                timing_chunks = $timing.steady.processing_ms.count
                                smoothing_mean_ms = $timing.steady.smoothing_ms.mean_ms
                                smoothing_p95_ms = $timing.steady.smoothing_ms.p95_ms
                                processing_mean_ms = $timing.steady.processing_ms.mean_ms
                                processing_p95_ms = $timing.steady.processing_ms.p95_ms
                                deadline_misses = $timing.steady.processing_ms.deadline_misses
                            })
                    }
                }
            }
        }
    }

    if ($summary.Count -eq 0) {
        Write-Host "No seams aggregated (clips too short for the chunk sizes?)." -ForegroundColor Yellow
        return
    }

    $summaryCsv = Join-Path $OutDir 'summary.csv'
    $summary | Export-Csv -LiteralPath $summaryCsv -NoTypeInformation
    Write-Host ""
    Write-Host "== join sweep summary (sorted by step_p95 desc) ==" -ForegroundColor Green
    $summary |
        Sort-Object -Property step_p95 -Descending |
        Format-Table clip, chunk_ms, crossfade_ms, sola_search, smoother, seams,
            step_p95, corr_mean, timing_chunks, smoothing_mean_ms, smoothing_p95_ms, processing_p95_ms, deadline_misses -AutoSize |
        Out-String -Width 200 | Write-Host
    Write-Host "summary CSV: $summaryCsv" -ForegroundColor DarkGray
}
finally {
    Pop-Location
    if (-not $KeepArtifacts) {
        $resolvedOutDir = (Resolve-Path -LiteralPath $OutDir).Path
        if ($resolvedOutDir -ne $ownedOutDir) { throw 'Refusing cleanup outside the run directory' }
        Remove-Item -Recurse -Force -LiteralPath $resolvedOutDir -ErrorAction SilentlyContinue
    } else {
        Write-Host "artifacts (per-run join CSVs + wavs) kept in $OutDir" -ForegroundColor DarkGray
    }
}
