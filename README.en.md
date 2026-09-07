# vc-rs

> [日本語](README.md) | English

`vc-rs` is a Rust **RVC voice conversion app**. It converts microphone input or
WAV files into another voice using an ONNX-format RVC model. There are three ways
to use it:

- **GUI (`vc-gui.exe`)** — the desktop app for standalone use. **Most people only
  need this.**
- **Bundled CLI (`vc-rs.exe`)** — a command-line tool shipped with the GUI
  package, for batch WAV conversion, diagnostics, Windows ML EP management,
  automation, and other things the GUI doesn't do. See
  [`docs/cli.md`](docs/cli.md).
- **VST3 plugin (`vc-vst3.vst3`)** — a plugin you load into your DAW.

Prebuilt Windows packages are distributed. **You do not need to build from
source** — just download, extract, supply your models, and run.

> Developers who want to build from source: see
> [`docs/development_ja.md`](docs/development_ja.md). The internal design is in
> [`docs/architecture.md`](docs/architecture.md).

## Highlights

- **Native Rust implementation** — no Python / PyTorch runtime. With no GC pauses
  or interpreter overhead, worst-case processing time stays stable even in
  real time, and distribution is lightweight (the windowsml build is a few MB).
- **Glitch-resistant real-time design** — audio callbacks only move samples
  through lock-free ring buffers; heavy work like inference is isolated on a
  separate thread. Under load the callback never blocks — it drops input or emits
  silence instead.
- **Broad GPU support and a fastest mode** — Windows ML (DirectML) runs on
  non-NVIDIA GPUs too, while native TensorRT offers the fastest path on NVIDIA GPUs.
- **WAV mode shares the real-time path** — so audio-quality tuning can be verified
  deterministically.

> The conversion quality itself is essentially the same as other tools as long as
> the same RVC model is used. vc-rs's strengths are **stability for real-time use**
> (resistance to dropouts, room to tighten latency) and **being lightweight and
> easy to use**.

## Download

Get the latest version from
**[Releases](https://github.com/shirohata/vc-rs/releases)**. Packages target
Windows (x64). There are four, depending on your front-end and hardware:

| Package | Form | Backend | Target | Size | Requirements |
| --- | --- | --- | --- | --- | --- |
| `vc-rs-windowsml-…zip` | GUI + CLI | Windows ML | Most GPUs (incl. non-NVIDIA) | Small (a few MB) | Windows App SDK Runtime |
| `vc-rs-tensorrt-…zip` | GUI + CLI | TensorRT | NVIDIA GPU | Large (~1.9 GB) | Up-to-date NVIDIA driver |
| `vc-vst3-windowsml-…zip` | VST3 plugin | Windows ML | Most GPUs (incl. non-NVIDIA) | Small | Windows App SDK Runtime |
| `vc-vst3-tensorrt-…zip` | VST3 plugin | TensorRT | NVIDIA GPU | Large (~1.9 GB) | Up-to-date NVIDIA driver |

**Which one?**

- To try it first, pick a **windowsml** package. It is a small download and runs
  on non-NVIDIA GPUs too via DirectML.
- If you **have an NVIDIA GPU and want maximum speed**, pick a **tensorrt**
  package. It is a large download and the first launch is slow (engine build),
  but subsequent runs are fast.
- Use the **GUI + CLI** packages for standalone use and the **VST3** packages
  for singing/streaming in a DAW. The bundled CLI handles automation and batch
  WAV conversion.

## Requirements

### windowsml packages

- Install the **Windows App SDK Runtime (2.x, minimum 2.1)**, which provides
  ONNX Runtime and DirectML. Get the latest stable **Runtime** installer from Microsoft's
  [Windows App SDK downloads page](https://learn.microsoft.com/windows/apps/windows-app-sdk/downloads).

### tensorrt packages

- An **up-to-date NVIDIA GPU driver**. The TensorRT runtime DLLs are bundled in
  the package, so you do not need to install CUDA or TensorRT separately.

### All packages: model files

`vc-rs` does not ship models. You supply three:

1. **RVC voice conversion model** (`.onnx`) — the target voice. **Only ONNX is
   supported**; `.pth` cannot be loaded directly, but picking a `.pth` in the
   GUI's model browser opens the built-in converter (RVC v2 / F0 models),
   which writes the `.onnx` next to it. For anything else (e.g. v1), convert
   first with RVC tools or VCClient.
2. **Embedder model** (ContentVec, `content_vec_500.onnx`)
3. **F0 model** (RMVPE, `rmvpe.onnx`)

Items 2 and 3 can be fetched with the bundled `download-models.ps1` (see below).

## Usage (GUI)

1. Extract the downloaded zip (**keep the DLLs in the same folder as
   `vc-gui.exe`**).
2. Fetch the embedder and F0 models (see *Prepare models* below).
3. Launch `vc-gui.exe`.

### Prepare models

Fetch the embedder and F0 models (run from the extracted folder):

```powershell
pwsh .\download-models.ps1
```

This downloads `.\assets\content_vec_500.onnx` and `.\assets\rmvpe.onnx`. You
still supply your own RVC voice model (`.onnx`).

> These downloaded models are third-party (GPL-3.0 upstream) and are **not**
> covered by `vc-rs`'s MIT license. Review and comply with the upstream license
> before using, modifying, or redistributing them. See the notes inside
> `download-models.ps1`.

### Working in the window

1. **Models** — **Browse** for the RVC model, embedder (ContentVec), and F0
   (RMVPE) `.onnx` files.
2. **Provider** — choose the backend (windowsml package: `windowsml` /
   `windowsml-directml` / `windowsml-nvtrtx` / `windowsml-cpu` / `cpu`; tensorrt
   package: `tensorrt`). **GPU Priority** and the CUDA/TensorRT
   **GPU Device ID** are selectable too.
3. **Audio** — pick the input/output devices (**Refresh devices** to re-scan).
   Leave blank to use "System default".
4. **Engine configuration** — set **Chunk ms** / **Extra convert ms** (see
   *Tuning real-time settings*).
5. Press **Apply / Start** to apply and start. **Model / Provider / device /
   Chunk edits do not take effect until you press it.** **Stop** stops the
   engine.
6. **Live parameters** (Pitch shift / Speaker ID / Input gain / Output gain)
   apply in real time.
7. **Telemetry** shows inference time, input/output RMS, and overruns/underruns
   so you can watch for dropouts and load (inference time is color-flagged when
   it exceeds the chunk budget).

Settings are saved automatically (`%APPDATA%\vc-rs\gui.toml`) and restored on the
next launch. When all three models are loaded, **Passthrough** switches live at
the next worker chunk boundary. RVC inference stops while passthrough is active;
switching back discards stale streaming context before conversion resumes.
Model-free passthrough remains available, but that session cannot switch live
back to RVC.

Choose `off`, `noise-gate`, `rnnoise`, or `gtcrn` under **Input denoiser**.
RNNoise uses an embedded model and needs no additional download. Passthrough
applies Input gain, the selected input denoiser, and Output gain. Switching
between `off` and `noise-gate`, including the gate threshold, is live; switching
to RNNoise or GTCRN requires **Apply / Start**. These input denoisers are not
included in VST3.

Where each denoiser sits:

| Mode | Cost | Quality | Notes |
| --- | --- | --- | --- |
| Noise Gate | very low | threshold gate only | embedded |
| RNNoise | low | modest | embedded, 48 kHz |
| **GTCRN** | **low** | **good** | **standalone packages only, 16 kHz, needs a model** |

GTCRN is an ultra-light (~48K-parameter) speech-enhancement model that runs in
real time with large margin. It ships in the **standalone CLI/GUI packages**:
Windows ML runs the tiny graph on ORT CPU, while TensorRT runs it through native
TensorRT. VST3 does not include it. Its fixed delay is ~48 ms (the 16 kHz STFT
reconstruction plus the adapter FIFO). Fetch the model with
`download-models.ps1 -Gtcrn` into `assets\gtcrn\`, then point the GUI's **GTCRN
model dir** or the CLI's `--gtcrn-model <dir>` at it.

## Tuning real-time settings

Balance dropouts, latency, and CPU/GPU load with **Chunk ms** and **Extra convert
ms**.

- **Chunk ms**: how much audio is processed per pass. Increase it if you hear
  dropouts or see sustained load (`500` → `750` → `1000`). Larger is more stable
  but adds input-to-output latency. GPU execution can often use smaller values.
- **Extra convert ms**: amount of surrounding context fed to conversion. Larger
  can be more stable but costs more. Start around `100` ms.

When tuning, **first find a value with no dropouts, then lower Chunk ms** to
reduce latency. Pitch / Speaker / Input·Output gain can be adjusted anytime under
Live parameters.

## The bundled CLI (advanced)

The GUI + CLI packages bundle the `vc-rs.exe` CLI. Everyday conversion is fully
covered by the GUI, but the CLI adds things the **GUI doesn't do**:

- **Batch WAV-file conversion** (the GUI is real-time only).
- **Diagnostics and model inspection** (`doctor` / `devices` / `inspect`).
- **Listing/installing Windows ML execution providers (EPs)** and **engine-cache
  management**.
- **Automation/scripting** and the finer DSP/audio parameters the GUI keeps
  pinned.

For usage and the command list, see [`docs/cli.md`](docs/cli.md).

## Usage (VST3 plugin)

1. Extract the zip and copy `vc-vst3-windowsml.vst3` or
   `vc-vst3-tensorrt.vst3` into the standard VST3 folder:
   - Windows: `%CommonProgramFiles%\VST3\` (e.g.
     `C:\Program Files\Common Files\VST3`)
2. In the extracted folder, run `pwsh .\download-models.ps1` to fetch the
   embedder and F0 models into `.\assets\` (**run it from the extracted folder,
   not from the installed plugin location**).
3. Load the plugin in your DAW and open its editor:
   - **Browse** for the RVC model, embedder (ContentVec), and F0 (RMVPE) `.onnx`
     files.
   - Choose the **backend** (windowsml package: `windowsml` /
     `windowsml-directml` / `cpu`; tensorrt package: `tensorrt`).
   - Set the **chunk size** (ms) — larger is more stable but adds latency.
   - Press **Load / Reload** to apply. Model / backend / chunk edits do not take
     effect until you press it.
   - Pitch / Speaker / Input·Output gain apply live and are DAW parameters
     (automatable and host-saved).

Model paths and settings are saved per project/preset. For details see
[`crates/vc-vst3/README.md`](crates/vc-vst3/README.md).

## TensorRT notes (tensorrt packages)

The tensorrt packages run on the **bundled TensorRT runtime**, so no extra
install beyond the NVIDIA driver is needed.

> ⚠️ TensorRT builds an engine **on first run and whenever the model or input
> shape changes**, which can make startup very slow. Later runs reuse the engine
> cache and start faster.

For engine-cache location/size and clearing (the CLI `engine-cache` command) and
detailed performance characteristics, see [`docs/cli.md`](docs/cli.md) and
[`docs/tensorrt_performance_ja.md`](docs/tensorrt_performance_ja.md).

## Troubleshooting / FAQ

**Q. A windowsml package won't start / model loading fails.**
A. Confirm the **Windows App SDK Runtime (2.x, minimum 2.1)** is installed (see
*Requirements*). The bundled CLI's `.\vc-rs.exe doctor` diagnoses the runtime
dependencies needed to run.

**Q. Running the exe triggers a SmartScreen warning.**
A. The distributed binaries are not code-signed, so Windows may warn. Review,
then choose "More info" → "Run anyway".

**Q. The VST3 plugin crashes in my DAW.**
A. Check that no stray ONNX Runtime provider DLLs (e.g. an old
`onnxruntime_providers_cuda.dll`) ended up in the plugin folder. The windowsml
bundle must not contain ONNX Runtime / DirectML / CUDA DLLs — those come from the
system Windows App SDK Runtime. A freshly extracted zip is fine; delete any DLLs
you copied in from an older build.

**Q. A `.pth` model won't load.**
A. RVC voice models must be **`.onnx`**. In the GUI, Browse for the `.pth`
under "RVC model" to run the built-in converter (RVC v2 / F0 models only);
otherwise convert with RVC tools or VCClient first.

**Q. Real-time audio drops out or latency is high.**
A. See *Tuning real-time settings*. Raise Chunk ms until dropouts stop, then
reduce latency.

## Helper script

`download-models.ps1` is an optional helper. It downloads third-party reference
ONNX models (ContentVec / RMVPE) from
[`wok000/weights_gpl`](https://huggingface.co/wok000/weights_gpl). The downloaded
models are not part of `vc-rs` and are not covered by this repository's MIT
license (upstream is marked GPL-3.0).

## Acknowledgements

- This implementation draws on knowledge from RVC-ecosystem OSS, especially the
  design and implementation insights of Applio, VCClient, and RVC WebUI.
- Related third-party notices are collected in
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

MIT License (see [`LICENSE`](LICENSE)). For notes on external projects and model
files, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
