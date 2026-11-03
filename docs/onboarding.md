# First-run tutorial

The standalone GUI guides users through language, applicable terms, audio
devices, voice model selection, and required files. Preparation is the final page.
Language initially defaults to English for new settings;
existing settings keep their language, devices, gain and denoiser.

## Navigation and persistence

Language selection is followed by terms in the selected language. The license
texts themselves remain authoritative originals. Audio setup and later pages
can be skipped; skipping never records setup completion or grants consent.
Setup on the normal screen returns to audio setup. Advanced settings remain
on the normal screen, not inside the tutorial.

Voice selection always proceeds to Preparation, including when all files are
already available. Each required support model shows its validation status;
download and runtime acquisition prompts stay hidden while availability is
being checked, and appear only when the check confirms preparation is needed.
Downloads request only missing or invalid models and reuse verified cache files.
Open main screen stays disabled during preparation and becomes available when the full
batch and runtime preparation finish. Completion never navigates automatically:
users review the result and explicitly open the unchanged normal screen. Completion
means prerequisites are ready, not that model loading or conversion has succeeded.
No conversion starts automatically and no first-use guidance is added to the normal screen.
Legacy `Listen` progress deserializes as `Prepare`, then rechecks prerequisites.

Windows ML inspection and preparation errors share installation guidance for
Windows App Runtime 2.x, minimum 2.1, matching `vc-core/src/windows_ml.rs` and
the MSIX framework dependency. The UI links to Microsoft's official
[runtime downloads](https://learn.microsoft.com/windows/apps/windows-app-sdk/downloads)
and explains how to select the stable x64 installer. It never infers absence
from an initialization error or launches an installer automatically. Existing
installations retain diagnostics and Retry. After installation users must
restart vc-rs, because the core caches bootstrap failures for the process lifetime.

`tutorial.step` records the intended destination. A missing consent temporarily
inserts the terms page without discarding that destination. Legacy
`setup_completed` / `setup_skipped` values preserve entry to the normal screen,
subject to missing terms. Missing models do not trap completed users in setup.
An interrupted tutorial rewinds only for missing prerequisites. Playback and
downloads do not restart on launch; entering the audio page measures the
microphone silently.

Explicit navigation, acceptance and completion save before changing pages.
Failed saves retain the page and report an error. The file is written and
flushed to a unique adjacent temporary file before replacing the previous
settings file. Continuous gain changes use delayed autosave; a failed final
save cancels a window-close request instead of silently losing changes.

## Device tests

`EngineController::start_device_test(DeviceTestConfig)` replaces conversion or
the prior test. Its model-free request contains audio endpoints and an output
mode, never model paths or a provider. `device_test_snapshot()` is separate from
conversion telemetry. A missing input does not prevent a tone test, and a
missing output does not prevent microphone measurement. Explicitly saved names
must still exist; diagnostics do not silently substitute another device.

Test callbacks only move samples through preallocated SPSC rings and atomics.
The test worker measures RMS and peak after input gain, before clipping and
denoising. It holds clipping indication for one second. The display is dBFS,
with yellow peak indication from -6 dBFS and red for clipping. These meters
cannot establish whether the device already clipped before capture.

The silent mode opens only input. Tone mode plays one second of 440 Hz at peak
0.1 times output gain, with 10 ms fades, independently of microphone data.
Monitor mode reuses `PassthroughProcessor`, its resampler, and optional RNNoise.
Tone and monitor are exclusive. Reconfiguration destroys the previous streams
and queues. Leaving setup, changing devices, or a stream failure stops output;
new devices never inherit queued sound. No test creates audio recordings.

New settings use input gain 1 and output gain 1; existing saved gain is kept.
The optional noise checkbox selects RNNoise/Off only when RNNoise was built.
Other saved denoisers remain selected for conversion until changed explicitly;
they are not loaded by device tests. RNNoise changes rebuild on the worker.

## Preparation and consent

The GUI presents the app license and applicable processing component terms
before device setup. `scripts/licenses/static/ONBOARDING-COMPONENTS.md` lists the
component scope and authoritative links. Consent identifiers include the text
digest. The Windows ML packaging script rejects a bootstrapper license that
differs from the GUI's embedded text; review that text and the manifest when
updating SDKs or terms. Store MSIX staging carries the same materials.

Windows ML preparation uses the core catalog registration path and cache, but
is requested before starting inference. Read-only inspection can bypass an
acquisition prompt when components are already ready. Vendor component sizes
depend on the system catalog. Native catalog preparation has no cancellation
API: leaving its page abandons UI results, not the already requested Windows
operation. A bootstrap error, including 0x80670016, is displayed as a diagnostic
and is not evidence that the runtime must be reinstalled.

ContentVec/RMVPE download is a separate explicit action with capacity and GPL
terms shown. Valid custom files are retained; only missing/invalid roles need
reference downloads. Reference cache entries are SHA-256 checked, while custom
ONNX files receive backend-free structural validation. Starting conversion on the normal screen still
checks model-role compatibility. Completed reference files are reused; partial
transfers are not resumed. Obsolete checks are associated with their original
paths, and download/selection changes invalidate readiness.

The normal screen's Model fold and Preparation's Details offer **Embedder
packages**. Select an exact vc-embedders profile, then **Import ZIP and use…**
to import its matching local archive. No public download source is configured
for the fixed 0.1.0 candidates. Merely opening setup or selecting an entry does
not acquire anything. A successful import selects the verified ONNX only if
the voice/embedder selection has remained unchanged; otherwise it just fills
the cache. Failed/cancelled jobs keep the current selection. Custom paths remain
available when switching back from a package. Identity/version is saved beside
the existing path without migrating older selections. Apply through the normal
Start/Restart operation. The package directory retains README, LICENSE and
NOTICE, and **Verify cache and use** rechecks all four files.

Use the embedder weights, layer, projection and preprocessing used for training.
SPIN/SPIN v2 are not substitutes for ContentVec-trained models. Generator and
vocoder support is an independent requirement; importing a package does not
establish backend compatibility or audio quality. CLI `embedders` commands use
the same package cache and verification path (see [the CLI reference](cli.md#embedder-packages)).

Conversion starts only through the existing normal-screen controls and uses the
shared pipeline and consent/preparation guards. There is no preview session or
preview-success state in the tutorial. Every navigation, including completion,
stops device tests before saving. New Windows ML settings keep automatic
accelerator selection. The normal screen retains its existing backend controls;
chunk size and other timing settings are unchanged. Windows ML GUI builds expose
one CPU choice; saved `cpu` selections migrate to `windowsml-cpu`.

In GTCRN-enabled builds, the setup download batch also includes the pinned
352 KB MIT-licensed GTCRN model. Its size and license are shown before download.
The worker verifies and reuses cached files; setup waits for the whole batch and
keeps failures available for retry. Acquisition does not enable noise reduction
or replace a saved custom GTCRN directory. Already-prepared setups are not forced
to download this optional model.

## Verification

Activate `scripts/activate.ps1` and `scripts/rustflags.ps1` before raw Cargo
commands. Headless interaction fixtures inject persistence and audio-test
effects; they never capture a microphone, start inference, download files or
write the user's settings.

```powershell
$env:VC_RS_ENABLE_NATIVE_TENSORRT = '0'
cargo test -p vc-gui --no-default-features --features rnnoise
cargo test -p vc-app --no-default-features --features rnnoise
cargo test -p vc-gui --no-default-features --features windowsml,rnnoise,gtcrn,ui-snapshots render_tutorial_pages -- --ignored --nocapture
```

The optional renderer writes PNGs under `target/onboarding-ui` for review. It
uses a test-only Vulkan adapter, since upstream egui_kittest's DX12 dependency
combination has incompatible Windows bindings. Production uses glow; snapshots
do not replace an interactive smoke test on the production renderer.

The explicitly ignored `live_device_test_independent_endpoints_and_stop` test
opens the default microphone and plays a quiet one-second tone without saving
audio. Run it only as part of an authorized real-device test. It checks endpoint
independence and stopping, not whether a human heard sound or wore headphones.
For audible monitoring, use headphones and verify gain/noise changes manually.
