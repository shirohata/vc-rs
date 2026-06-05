# vc-rs build tasks. Run `just <recipe>` (or just `just` to list them).
#
# Requires `just` and PowerShell 7+ (`pwsh`). Get both from `just setup`'s
# bootstrap step — for the very first run, before `just` exists, invoke it
# directly: `pwsh -File scripts/bootstrap.ps1`.
#
# GPU recipes dot-source scripts/activate.ps1 so a matched CUDA/cuDNN/TensorRT
# line is on PATH (the native TensorRT shim links nvinfer at load time, so test
# and dev exes need it). The `*-cpu` recipes set VC_RS_ENABLE_NATIVE_TENSORRT=0
# instead, so they build and run with no GPU stack installed.

set windows-shell := ["pwsh", "-NoProfile", "-Command"]
set shell := ["pwsh", "-NoProfile", "-Command"]

# List available recipes.
default:
    @just --list

# One-time: install winget-scoped prerequisites (Rust, Git, MSVC C++, just).
setup:
    ./scripts/bootstrap.ps1

# Fetch the reference ContentVec / RMVPE ONNX models into ./assets.
models:
    ./download-models.ps1

# Fast workspace tests, no GPU stack (TensorRT shim disabled).
test-cpu:
    . ./scripts/rustflags.ps1; $env:VC_RS_ENABLE_NATIVE_TENSORRT = "0"; cargo test --workspace

# Full workspace tests with the native TensorRT shim (activates the GPU stack).
test:
    . ./scripts/activate.ps1; . ./scripts/rustflags.ps1; cargo test --workspace

# Dev CLI build — both backends in one vc-rs.exe (activates the GPU stack).
build:
    . ./scripts/activate.ps1; . ./scripts/rustflags.ps1; cargo build --release

# Single-provider CLI build: `just build-cli windowsml` (GPU-free) or `tensorrt`.
build-cli variant="windowsml":
    . ./scripts/activate.ps1; . ./scripts/rustflags.ps1; cargo build --release -p vc-cli --no-default-features --features {{variant}}

# Bundle the VST3 plugin into target/bundled: `just bundle [windowsml|tensorrt]`.
bundle variant="windowsml":
    . ./scripts/activate.ps1; . ./scripts/rustflags.ps1; if ('{{variant}}' -eq 'tensorrt') { cargo xtask bundle vc-vst3 --release --no-default-features --features tensorrt } else { cargo xtask bundle vc-vst3 --release }

# Build the VST3 plugin and run Steinberg's validator: `just validate-vst3 [windowsml|tensorrt]`.
validate-vst3 variant="windowsml":
    ./scripts/validate-vst3.ps1 -Variant {{variant}}

# Build, validate, and copy the VST3 plugin into the per-user VST3 directory.
install-vst3 variant="windowsml":
    ./scripts/install-vst3-bundle.ps1 -Variant {{variant}} -BuildFirst -ValidateFirst

# End-to-end check (tests + bundle); forwards flags, e.g. `just verify -Variant tensorrt`.
verify *args:
    ./scripts/verify.ps1 {{args}}

# Build the shipped distribution zips; forwards flags, e.g. `just package -Targets cli-windowsml,vst3-windowsml`.
package *args:
    ./scripts/package-all.ps1 {{args}}

# Format the workspace.
fmt:
    cargo fmt --all

# Clippy across the workspace (activates the GPU stack).
lint:
    . ./scripts/activate.ps1; . ./scripts/rustflags.ps1; cargo clippy --workspace --all-targets

# Remove build artifacts.
clean:
    cargo clean
