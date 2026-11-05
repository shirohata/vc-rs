# vc-vst3 — RVC voice conversion for your DAW

Use an RVC voice model as a mono or stereo effect in your DAW. The plugin uses
models in ONNX format; model files are downloaded or supplied separately.

## Install

Download a VST3 package from [Releases](https://github.com/shirohata/vc-rs/releases).
The current releases are for Windows x64.

| Package | Requirements |
| --- | --- |
| Windows ML | Windows App SDK Runtime 2.x (minimum 2.1); supports automatic backend selection, DirectML, CPU, and available catalog backends |
| TensorRT | A supported NVIDIA GPU and current NVIDIA driver; the package includes the TensorRT runtime |

Extract the ZIP, then copy the complete `vc-vst3-windowsml.vst3` or
`vc-vst3-tensorrt.vst3` folder to `%CommonProgramFiles%\VST3\`. Keep the files
inside the bundle together. Rescan plugins in your DAW if it does not appear.

See the [backend guide](../../docs/backends.md) for hardware requirements and
experimental OpenVINO / MIGraphX options.

## Set up voice conversion

1. Add the plugin to an audio track and open its editor.
2. Under **Setup → Models and conversion settings**, browse for your RVC voice
   model's `.onnx` file.
3. Browse for the **Embedder** `.onnx` file used to train that voice model.
4. Browse for the **F0 (RMVPE)** model. If you need the reference models, run
   `pwsh .\download-models.ps1` from the extracted ZIP folder. It places RMVPE
   and the reference ContentVec embedder under `assets\`; keep that folder in a
   permanent location and select the files from there.
5. Choose a **Backend** and **Chunk** size, then press **Load / Reload**.
   Wait for the status to show that conversion is running.

Choose an embedder with the weights and preprocessing used for training your
voice model. Matching channel counts alone do not establish compatibility.

The Windows ML package offers **Windows ML (automatic)**, **DirectML (GPU)**,
CPU options, and backends available in this PC's Windows ML catalog. For
**OpenVINO**, choose CPU, GPU, or NPU separately; use **Download and check** if
prompted. The TensorRT package offers **TensorRT (NVIDIA GPU)**. TensorRT's first
load can take much longer while engines are built; later loads reuse compatible
cached engines.

## Adjust the sound

| Controls | When changes take effect |
| --- | --- |
| Pitch, Speaker, Input gain, Output gain, Noise gate, Gate threshold | Immediately; these are automatable DAW parameters |
| Models, Backend, Chunk, Extra convert, Gate attack/release/floor | After **Load / Reload**; **Unapplied changes** indicates pending edits |

Set **Speaker** to an ID supported by your voice model. Use the status line to
check loading progress. If loading or conversion fails, expand **Error details**,
correct the settings or missing files, and press **Load / Reload** again.

## Save and reopen a project

The DAW saves the plugin's parameters, model paths, and conversion settings with
the project or preset. Model files are not embedded. Keep the referenced files
available at the saved locations when reopening or moving a project.

Reopened projects retain their selections; press **Load / Reload** to start
conversion. Changing a model stages the selection
until you press that button.

## Monitoring delay and offline export

The plugin reports its latency so the DAW can compensate playback timing. Live
microphone monitoring still has delay. Smaller **Chunk** values reduce input
waiting time; larger values give more context and processing headroom. Check
both audio quality and dropouts when adjusting this setting.

After the models have loaded, you can export or freeze through the DAW's offline
rendering mode. Mode and block-size changes keep the loaded models available.
A sample-rate change rebuilds them from the last applied settings; pending
editor changes still need **Load / Reload**.

## Advanced configuration

An optional TOML file supplies initial settings for a fresh instance. Start from
[`vc-rs-vst3.example.toml`](vc-rs-vst3.example.toml). The plugin searches in this
order:

1. The file named by the `VC_RS_VST3_CONFIG` environment variable.
2. `%APPDATA%\vc-rs\vst3.toml` on Windows (`$XDG_CONFIG_HOME/vc-rs/vst3.toml` or
   `~/.config/vc-rs/vst3.toml` on other platforms).
3. `vc-rs-vst3.toml` in the DAW's working directory.

Saved project/preset settings take precedence. Editing the TOML file does not
update an existing instance or replace a project's saved settings. New model and
conversion settings take effect when you press **Load / Reload**.

Chunk joining is an advanced configuration choice: `smoother = "sola"`,
`"psola"`, or `"phase-vocoder"`. There is no smoother selector in the editor.
Missing values default to phase vocoder, and explicit choices saved in a project
or preset are preserved. See the CLI's [comparison workflow](../../docs/cli.md#compare-smoother-quality-and-processing-time)
if you want to compare the methods.

## Troubleshooting

- **The plugin is missing or fails to load:** check the package requirements and
  install the complete `.vst3` folder from a fresh ZIP. Avoid mixing files from
  Windows ML and TensorRT packages.
- **Model loading fails:** check all three model paths and **Error details**.
  Windows ML runtime diagnostics are described in the
  [troubleshooting guide](../../docs/development_ja.md#windows-ml-troubleshooting).
- **TensorRT loads slowly:** allow time for the initial engine build. See
  [engine-cache management](../../docs/cli.md#engine-cache-management) for cache
  location, size, and cleanup.
- **An experimental backend does not work with your model:** consult the
  [backend guide](../../docs/backends.md) and try an appropriate supported
  backend for your hardware.

## Build from source

Follow the [build environment setup](../../scripts/README.md). Build distributable
archives with [`package.ps1`](package.ps1) from the repository root:

```powershell
cargo install cargo-about --features cli
pwsh crates/vc-vst3/package.ps1                    # Windows ML
# TensorRT: activate the NVIDIA SDK environment first.
. scripts/activate.ps1
pwsh crates/vc-vst3/package.ps1 -Variant tensorrt
```

Use package-scoped builds so the two backend variants remain separate. See the
[packaging guide](../../scripts/README.md#package-the-distributables),
[local VST3 validation](../../scripts/README.md#local-vst3-validator), and
[distribution requirements](../../docs/distribution.md) for the development and
release workflows. Internal processing is described in the
[architecture documentation](../../docs/architecture.md).

The plugin is MIT-licensed; its VST3 bindings through nice-plug are ISC-licensed.
Third-party model licenses are supplied with their downloads.
