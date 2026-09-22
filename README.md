# vc-rs

> English | [日本語](README.ja.md)

`vc-rs` is a **Windows RVC voice changer with native TensorRT support**.
Use NVIDIA GPU inference in a standalone app or a VST3 plugin inside your DAW.
The Windows ML package also offers DirectML and experimental MIGraphX / OpenVINO
paths. Written in Rust, it needs no Python / PyTorch environment. Convert microphone
input or WAV files into another voice using an ONNX-format RVC model.

## Highlights

- **Choose an inference backend for your hardware** — native TensorRT runs
  ContentVec, RMVPE, and the RVC generator on NVIDIA GPUs. Windows ML also offers
  DirectML, MIGraphX for AMD, and OpenVINO for Intel. MIGraphX and OpenVINO are
  experimental: MIGraphX remains unverified on target hardware; OpenVINO has
  measurements and listening checks on a limited Intel configuration.
  See the support table below for conditions.
- **Standalone or inside your DAW** — convert your microphone with the GUI, or
  use VST3 with pitch/gain automation and settings saved in your DAW project.
- **No Python setup; bring your existing models** — prebuilt apps are available.
  The GUI converts supported RVC v2 / F0 PTH models to ONNX and downloads support models.
- **Audio callbacks do not wait for inference** — workers perform inference;
  callbacks move audio through lock-free ring buffers. Overload can still cause
  drops or silence, so tune the chunk size for your environment.
- **WAV conversion and automation** — the bundled CLI handles WAV files,
  diagnostics, and scripting. Shared conversion components make comparisons with
  fixed input and settings practical.

Audio quality depends on the voice model, F0 estimation, chunk settings, and
pre/post-processing. See the [inference backend guide](docs/backends.md) for
selection and tuning.

## Choose how to use it

There are three ways to use it:

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

## Download

Get the latest version from
**[Releases](https://github.com/shirohata/vc-rs/releases)**. Packages target
Windows (x64). There are four, depending on your front-end and hardware:

| Package | Form | Backend | Target | Size | Requirements |
| --- | --- | --- | --- | --- | --- |
| `vc-rs-windowsml-…zip` | GUI + CLI | Windows ML | Most GPUs (incl. non-NVIDIA) | Small | Windows App SDK Runtime |
| `vc-rs-tensorrt-…zip` | GUI + CLI | TensorRT | NVIDIA GPU | Large (runtime bundled) | Up-to-date NVIDIA driver |
| `vc-vst3-windowsml-…zip` | VST3 plugin | Windows ML | Most GPUs (incl. non-NVIDIA) | Small | Windows App SDK Runtime |
| `vc-vst3-tensorrt-…zip` | VST3 plugin | TensorRT | NVIDIA GPU | Large (runtime bundled) | Up-to-date NVIDIA driver |

**Which one?**

- To try it first, pick a **windowsml** package. It is a small download and runs
  on non-NVIDIA GPUs too via DirectML.
- If you **want native TensorRT on your NVIDIA GPU**, pick a **tensorrt**
  package. It is a large download and the first launch is slow (engine build),
  but subsequent runs are fast.
- Use the **GUI + CLI** packages for standalone use and the **VST3** packages
  for singing/streaming in a DAW. The bundled CLI handles automation and batch
  WAV conversion.

### Inference backend support

This table describes the current source. Check each release's notes for the
features included in that download. Both packages have GUI + CLI and VST3 variants.

| Backend | Target | Package | Conditions and validation status |
| --- | --- | --- | --- |
| Native TensorRT | NVIDIA GPU | tensorrt | Distributed backend. Runtime bundled; engines build on first use. [Model-only measurements](docs/tensorrt_performance_ja.md) available |
| Windows ML / DirectML | Supported NVIDIA, AMD, Intel and other GPUs | windowsml | Distributed backend. Requires Windows App SDK Runtime. Performance depends on GPU, model and settings |
| MIGraphX through Windows ML | Supported AMD GPUs | windowsml | **Experimental**. Requires a catalog EP and compatible device. Model compatibility, audio quality and performance on AMD hardware remain unverified |
| OpenVINO through Windows ML | Supported Intel CPUs / GPUs / NPUs | windowsml | **Experimental**. Requires a catalog EP. Device-type selection available; model measurements and GUI listening checks on a limited Intel configuration. GPU selection still runs RMVPE on OpenVINO CPU. NPU unverified. [Validation record](docs/openvino-model-routing_ja.md) |

MIGraphX and OpenVINO are additional EPs in the Windows ML package, not separate
ZIPs. Picker entries depend on this PC's catalog. An available EP does not prove
that the entire model runs on GPU/NPU. See the [backend guide](docs/backends.md).
ZIP sizes exclude separately acquired runtimes, support models, and generated caches.
Check the release assets for exact download sizes.

## Requirements

### windowsml packages

- Install the **Windows App SDK Runtime (2.x, minimum 2.1)**, which provides
  ONNX Runtime and DirectML. Get the latest stable **Runtime** installer from Microsoft's
  [Windows App SDK downloads page](https://learn.microsoft.com/windows/apps/windows-app-sdk/downloads).

### tensorrt packages

- An **up-to-date NVIDIA GPU driver**. The TensorRT runtime DLLs are bundled in
  the package, so you do not need to install CUDA or TensorRT separately.

### All packages: model files

`vc-rs` does not ship models. It uses three: choose your voice model and fetch the support models from the GUI.

1. **RVC voice conversion model** (`.onnx`) — the target voice. **Only ONNX is
   supported**; `.pth` cannot be loaded directly, but picking a `.pth` in the
   GUI's model browser opens the built-in converter (RVC v2 / F0 models),
   which writes the `.onnx` next to it. For anything else (e.g. v1), convert
   first with RVC tools or VCClient.
2. **Embedder model** (ContentVec, `content_vec_500.onnx`)
3. **F0 model** (RMVPE, `rmvpe.onnx`)

Items 2 and 3 are downloaded and configured together with **Download required models** in the GUI.

## Usage (GUI)

1. Extract the downloaded zip (**keep the DLLs in the same folder as
   `vc-gui.exe`**).
2. Choose a language and review the applicable app and processing-component terms.
3. Check the microphone meter and adjust **Microphone volume**. Use **Play test sound** to check the output independently, or **Hear my voice** with headphones. Noise reduction is optional.
4. Choose or drop an RVC `.onnx` / `.pth` voice model. Supported checkpoints can be converted in the app.
5. Review and prepare missing ContentVec/RMVPE files and processing components. Verified files are reused and locations are configured automatically.
   When ready, choose **Open main screen** to save and finish setup. Start voice conversion from the normal screen.

The tutorial can be skipped from audio setup onward and reopened from **Setup**. Skipping is separate from consent and setup completion. Interrupted setup resumes its saved step without automatically restarting playback or downloads. See [the tutorial design](docs/onboarding.md).
The normal screen provides voice, input/output, volume and pitch controls once. Start becomes Restart / Stop while running. Language is at the bottom left; Setup reopens the tutorial. See [normal screen behavior](docs/normal-screen.md).

### Prepare models

The GUI downloads about 741 MB directly from the upstream host (GPL-3.0), with progress, cancellation and retry. Files are checked against pinned sizes and SHA-256 hashes before use.
Models are saved in `%LOCALAPPDATA%\vc-rs\models` (falling back to `%APPDATA%\vc-rs\models`) and reused across launches and app updates.
Existing selections and support models in the `assets` folder beside the executable are also reused. You still supply your own voice model.

Alternatively, use the bundled script from the extracted folder:

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

- **Model settings**: downloaded ContentVec / RMVPE or custom paths; Browse appears for custom selections.
- **Audio devices and noise reduction**: hosts, device refresh and denoiser-specific controls.
- **Backend Details**: available backends, GPU selection/priority, Chunk ms / Extra convert ms, metrics and diagnostics.

Gain and pitch update live. Reload-scoped changes show a Restart reminder beside transport.

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
to RNNoise or GTCRN requires **Restart** while running, or **Start** when
stopped. These input denoisers are not
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
reconstruction plus the adapter FIFO). In the GUI, select **Audio devices and noise reduction → Input denoiser → gtcrn → Download GTCRN**.
The official 352 KB model (MIT) is verified and saved alongside its license in `%LOCALAPPDATA%\vc-rs\models\gtcrn`; its directory is configured automatically. Press **Restart** while running, or **Start** when stopped, to activate it.
For CLI use, `download-models.ps1 -Gtcrn` fetches it into `assets\gtcrn\`. Select that directory or an existing model directory with `--gtcrn-model <dir>`.

## Tuning real-time settings

Balance dropouts, latency, and CPU/GPU load with **Chunk ms** and **Extra convert
ms**.

- **Chunk ms**: how much audio is processed per pass. Increase it if you hear
  dropouts or see sustained load (`500` → `750` → `1000`). Larger is more stable
  but adds input-to-output latency. GPU execution can often use smaller values.
- **Extra convert ms**: amount of surrounding context fed to conversion. Larger
  can be more stable but costs more. Start around `100` ms.

When tuning, **first find a value with no dropouts, then lower Chunk ms** to
reduce latency. Pitch and input/output gain can be adjusted live on the normal
screen; **Speaker ID** is under **Model** below voice selection.

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
     `windowsml-directml` / CPU options and catalog EPs; tensorrt package: `tensorrt`).
     See the [backend guide](docs/backends.md) for experimental options.
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
