# Normal screen

The standalone GUI has one set of transport, model, pitch and audio controls.
The body alone scrolls; transport stays above it and language/Setup scroll with the body.
Input/output are side by side when space permits and stack in narrower windows.

Model settings sit under voice selection. ContentVec and RMVPE each offer the
reference download or a custom path. Existing paths are inferred on migration;
custom paths are remembered when switching away and retained in GUI settings.
Reference downloads require consent, show size/license, and use the existing
verified cache worker. Normal rendering only checks file existence; it does not
scan model structure or hash existing files. Setup retains structural validation,
and actual role/backend compatibility is checked by engine loading at Start.

Audio connection and denoising share a disclosure below the channels.
GTCRN exposes readiness and reference download only, without a custom file picker.
Backend Details groups backend/GPU selection, existing timing controls, real engine telemetry and
error details. Build/device capabilities determine the available backend choices.
No timing defaults or shared DSP paths are changed by this layout.

Start becomes Restart and Stop while running. A queued start prevents duplicate
requests. Start does not navigate to Setup or require its cached checks/consent;
actual engine loading failures are reported on the normal screen. Downloads retain
their consent flow. A user can explicitly open Setup from the footer.
Preparation/stopping states retain Stop; the existing controller handles
queued stop after its current synchronous preparation step, not immediately during
model loading. Restart submits the existing shared Apply operation once. The GUI
keeps requested and applied settings separate until the new session revision is
running. Reload-scoped differences show beside transport; gain/pitch and other
existing live parameters do not require Restart. Passthrough retains pitch but
disables its controls, and respects the controller's live-switch capability.

Microphone, output-device and input-denoiser selections have automatic apply requests.
The GUI acknowledges only the requested fields after success, so these changes
cannot apply staged voice-model, provider or other-device edits. Failed selections keep
the previous applied values and expose an error. Same-rate device changes
retain voice-model sessions, while rate changes rebuild internally. Off/Gate
continues without an I/O restart; RNNoise/GTCRN changes restart the audio timeline
after both routes are prepared. Model acquisition stays explicit. Gate shaping,
chunk/context settings and other reload-scoped edits still require Restart.

Meters show the existing conversion telemetry as RMS dBFS, not the tutorial's
pre-denoise peak meter. Nominal content delay excludes device/queue latency.
Unavailable metrics are not fabricated. Tests replace engine starts and downloads
and do not use real audio or save user settings; see the GUI README for rendering.
