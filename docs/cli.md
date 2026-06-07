# vc-rs CLI reference (`vc-rs.exe`)

> [日本語](cli_ja.md) | English

`vc-rs.exe` is the CLI bundled with the GUI + CLI package. Everyday voice
conversion is fully covered by [`vc-gui.exe`](../README.en.md); the CLI adds the
things the **GUI does not do**:

- **Batch WAV-file conversion** (`wav`) — the GUI is real-time only.
- **Diagnostics and model inspection** (`doctor` / `devices` / `inspect`).
- **Listing and installing Windows ML execution providers (EPs)**
  (`windowsml-eps`).
- **Engine-cache management** (`engine-cache`).
- **Automation/scripting** and the finer DSP/audio parameters the GUI keeps
  pinned (WASAPI exclusive, `psola`, `--rms-mix-rate`, and more).

The GUI and CLI share the same inference pipeline, so settings you dial in from
the CLI reproduce identically in the GUI.

## Setup

1. Extract the GUI + CLI package zip (**keep the DLLs in the same folder as
   `vc-gui.exe` / `vc-rs.exe`**).
2. Open PowerShell in that folder.
3. Fetch the embedder + F0 models (below). Supply your own RVC voice-conversion
   model (`.onnx`) separately (`.pth` is not supported).

```powershell
pwsh .\download-models.ps1
```

This downloads `.\assets\content_vec_500.onnx` and `.\assets\rmvpe.onnx`.

> For requirements (Windows App SDK Runtime / NVIDIA driver) and how to pick a
> package, see [`README.en.md`](../README.en.md).

## Commands

```powershell
.\vc-rs.exe --help
```

| Command | Purpose |
| --- | --- |
| `doctor` | Diagnose runtime dependencies and device visibility needed to run |
| `devices` | List audio input/output devices |
| `inspect` | Show ONNX model inputs, outputs, and metadata (backend-independent) |
| `run` | Real-time microphone-to-speaker conversion |
| `wav` | WAV-file to WAV-file conversion (same pipeline, deterministic testing) |
| `windowsml-eps` | List/install Windows ML catalog EPs (windowsml package only) |
| `engine-cache` | Inspect/clear the GPU engine cache |

### Diagnostics

```powershell
.\vc-rs.exe doctor
```

### List devices

```powershell
.\vc-rs.exe devices
```

### Inspect a model

```powershell
.\vc-rs.exe inspect --model <your-rvc-model>.onnx
```

`inspect` is backend-independent and prints the ONNX model's inputs, outputs,
and metadata.

### Real-time conversion

```powershell
.\vc-rs.exe run --model <your-rvc-model>.onnx `
    --embedder .\assets\content_vec_500.onnx `
    --f0-model .\assets\rmvpe.onnx `
    --input "Microphone" --output "Speakers" `
    --chunk-ms 500 --extra-convert-ms 100 `
    --provider windowsml --speaker-id 0
```

Pass a substring of the names shown by `devices` to `--input`/`--output`. On the
tensorrt package use `--provider tensorrt`.

### WAV-file conversion

Not available in the GUI. Useful for batch processing and for deterministic
verification of setting changes.

```powershell
.\vc-rs.exe wav --model <your-rvc-model>.onnx `
    --embedder .\assets\content_vec_500.onnx `
    --f0-model .\assets\rmvpe.onnx `
    --input input.wav --output out.wav `
    --provider windowsml --speaker-id 0
```

## Tuning real-time settings

Balance dropouts, latency, and CPU/GPU load with `--chunk-ms` and
`--extra-convert-ms`.

- `--chunk-ms`: how much audio is processed per pass. Increase it
  (`500` → `750` → `1000`) when you hear dropouts or load spikes. Larger is more
  stable but adds input-to-output latency. GPU execution can often use smaller
  values.
- `--extra-convert-ms`: extra leading/trailing context handed to the conversion.
  Larger can be more stable but costs more. Start around `100` ms.

When tuning, the safe order is to **first find a value with no dropouts, then
lower `--chunk-ms`** to reduce latency.

## Key conversion parameters

- `--speaker-id 0`: speaker ID for multi-speaker models (default: 0).
- `--pitch-shift 0.0`: shift F0 in semitones (default: 0.0). `12.0` is one octave
  up, `-12.0` one octave down.
- `--input-gain 1.0` / `--output-gain 1.0`: input/output gain (default: 1.0).
  Raise when too quiet; raising too far clips.
- `--silence-threshold 0.0001`: threshold below which input is treated as
  silence.
- `--rms-mix-rate <0.0-1.0>`: closer to 0.0 follows the input's loudness
  dynamics, closer to 1.0 keeps the model output's loudness (default: 0.0).

Other options the GUI keeps pinned are also available from the CLI:
`--smoother sola|psola`, `--sola-search-ms`, `--crossfade-ms`,
`--rvc-output-tail-discard-ms`, `--gpu-priority normal|high`, and the WASAPI
controls (`--audio-backend wasapi`, `--wasapi-exclusive*`, `--wasapi-buffer-ms`).
See `--help` for the full list and defaults.

## Windows ML execution providers (windowsml package)

With `--provider windowsml`, the windowsml package prefers a Windows ML catalog
EP, falling back to DirectML and finally CPU. To force a specific EP use
`windowsml-nvtrtx` / `windowsml-qnn` / `windowsml-openvino` / `windowsml-migraphx`
/ `windowsml-vitisai` (no fallback — it errors if the EP is not installed/ready).

Check and install catalog EPs from the CLI:

```powershell
.\vc-rs.exe windowsml-eps list
.\vc-rs.exe windowsml-eps install            # auto-select the best EP
.\vc-rs.exe windowsml-eps install --provider nvtrtx --yes
```

## TensorRT execution (tensorrt package)

The tensorrt package runs on the **bundled TensorRT runtime**, so nothing beyond
the NVIDIA driver needs installing.

> ⚠️ TensorRT builds an engine **on first run and whenever the model or input
> shape changes**, which can make startup very slow. Subsequent runs reuse the
> cached engine and start faster.

For detailed performance characteristics see
[`tensorrt_performance_ja.md`](tensorrt_performance_ja.md).

## Engine-cache management

Engines built by TensorRT (tensorrt package) and by Windows ML TensorRT-RTX
(`windowsml-nvtrtx`) are stored under `%LOCALAPPDATA%\vc-rs\tensorrt-cache` and
shared by both backends (override the location with `VC_RS_TENSORRT_CACHE_DIR`).
Inspect the location/size and clear the cache from the CLI:

```powershell
.\vc-rs.exe engine-cache info          # location, total size, per-model breakdown
.\vc-rs.exe engine-cache clear         # delete all (with confirmation)
.\vc-rs.exe engine-cache clear --yes   # delete all without confirmation
```

The cache is regenerable derived data — deleting it just rebuilds on the next
model load (only that run is slow again).
