<#
.SYNOPSIS
    Scrub absolute build-machine paths out of compiled vc-rs binaries.

.DESCRIPTION
    Dot-source this before a `cargo build` / `cargo xtask bundle` whose output is
    shipped. It exports CARGO_ENCODED_RUSTFLAGS with rustc `--remap-path-prefix`
    rules so the absolute paths Rust bakes into binaries — panic locations
    (file!()), and dependency / std-library source paths — no longer reveal the
    build machine's user name or directory layout. Without this, a release exe
    contains hundreds of `C:\Users\<name>\.cargo\...` / `...\.rustup\...` strings.

    Three prefixes are remapped, all computed dynamically (no hardcoded user name,
    so it is portable across machines/contributors):
        <CARGO_HOME>   -> /cargo   (dependency crate sources under registry/, git/)
        <rustc sysroot>-> /rustc   (std/core/alloc sources)
        <repo root>    -> .        (this workspace's own crate sources)

    This is the stable-toolchain stand-in for cargo's `[profile] trim-paths`,
    which is still unstable on the pinned toolchain. It does NOT cover
    env!("CARGO_MANIFEST_DIR") (that embeds an env var value, not a remapped
    source path) — that path is cfg-gated out of release builds in
    crates/vc-core/src/model_rvc/native_tensorrt.rs instead.

    Idempotent: re-dot-sourcing replaces this script's own remap entries rather
    than stacking duplicates, and any other pre-existing rustflags are preserved.

.EXAMPLE
    . .\scripts\rustflags.ps1 ; cargo build --release
#>

$ErrorActionPreference = 'Stop'

$repoRoot  = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' }
$sysroot   = (rustc --print sysroot).Trim()

$remaps = @(
    "--remap-path-prefix=$cargoHome=/cargo"
    "--remap-path-prefix=$sysroot=/rustc"
    "--remap-path-prefix=$repoRoot=."
)

# Cargo reads CARGO_ENCODED_RUSTFLAGS (0x1f-separated) in preference to RUSTFLAGS
# and to `build.rustflags`, so fold any pre-existing flags from either source in,
# minus a previous run's remaps (the `=/cargo|=/rustc|=.` tails this script adds),
# to stay idempotent across repeated dot-sourcing within one shell session.
$sep = [char]0x1f
$existing = @()
if ($env:CARGO_ENCODED_RUSTFLAGS) { $existing = $env:CARGO_ENCODED_RUSTFLAGS -split $sep }
elseif ($env:RUSTFLAGS)           { $existing = $env:RUSTFLAGS -split ' ' | Where-Object { $_ } }
$existing = $existing | Where-Object { $_ -notmatch '^--remap-path-prefix=.*=(?:/cargo|/rustc|\.)$' }

$env:CARGO_ENCODED_RUSTFLAGS = (@($existing) + $remaps) -join $sep
Write-Host "[rustflags] remap-path-prefix set (CARGO_HOME=/cargo, sysroot=/rustc, repo=.)" -ForegroundColor DarkGray
