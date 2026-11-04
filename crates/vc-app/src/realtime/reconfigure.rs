//! Control-thread changes of running standalone audio devices. Inference sessions
//! cross worker lifetimes only after join; no callback owns a model or mailbox.
use super::*;
use vc_core::model_rvc::{InputDenoiserConfig, PreparedInputDenoiser};

#[cfg(test)]
thread_local! { static FAIL_NEXT_DENOISER_PREPARATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

#[derive(Clone, Debug)]
pub(super) struct DeviceChange {
    pub host: AudioHost,
    pub device: Option<String>,
}

#[derive(Clone, Debug)]
pub(super) struct DenoiserChange {
    pub mode: DenoiserMode,
    pub model_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Default)]
struct LiveChange {
    input: Option<DeviceChange>,
    output: Option<DeviceChange>,
    denoiser: Option<DenoiserChange>,
}

#[derive(Default)]
struct MailboxState {
    pending: Option<(u64, LiveChange)>,
    delivered: u64,
}

#[derive(Default)]
pub(super) struct ReconfigureMailbox {
    revision: AtomicU64,
    pending: Mutex<MailboxState>,
}

impl ReconfigureMailbox {
    pub fn request(&self, change: DeviceChange) -> Result<u64> {
        self.update(|pending| pending.input = Some(change))
    }

    pub fn request_denoiser(&self, change: DenoiserChange) -> Result<u64> {
        self.update(|pending| pending.denoiser = Some(change))
    }

    pub fn request_output(&self, host: AudioHost, device: Option<String>) -> Result<u64> {
        self.update(|pending| pending.output = Some(DeviceChange { host, device }))
    }

    fn update(&self, change: impl FnOnce(&mut LiveChange)) -> Result<u64> {
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| anyhow!("audio control stopped"))?;
        let revision = self.revision.fetch_add(1, Ordering::SeqCst) + 1;
        let mut changes = pending
            .pending
            .take()
            .map(|(_, change)| change)
            .unwrap_or_default();
        change(&mut changes);
        pending.pending = Some((revision, changes));
        Ok(revision)
    }

    pub fn cancel(&self) {
        let mut pending = self.pending.lock().unwrap();
        self.revision.fetch_add(1, Ordering::SeqCst);
        pending.pending = None;
    }

    pub fn is_current(&self, revision: u64) -> bool {
        self.revision.load(Ordering::SeqCst) == revision
    }

    fn take(&self) -> Option<(u64, LiveChange)> {
        let mut state = self.pending.lock().unwrap();
        let (revision, change) = state.pending.clone()?;
        if state.delivered == revision {
            return None;
        }
        state.delivered = revision;
        Some((revision, change))
    }
}

// A newer request merges fields even while an earlier request is in flight.
// Completed/failed requests are cleared, so a later denoiser edit cannot retry
// an already rejected device selection.
struct Completion<'a>(&'a ReconfigureMailbox, u64);
impl Drop for Completion<'_> {
    fn drop(&mut self) {
        let mut state = self.0.pending.lock().unwrap();
        if self.0.is_current(self.1) {
            state.pending = None;
        }
    }
}

struct PreparedDenoisers {
    passthrough: PassthroughProcessor,
    rvc: Option<PreparedInputDenoiser>,
}

impl PreparedDenoisers {
    fn prepare(
        config: &RealtimeConfig,
        input_rate: u32,
        output_rate: u32,
        live: &LiveParams,
    ) -> Result<Self> {
        #[cfg(test)]
        if FAIL_NEXT_DENOISER_PREPARATION.with(|fail| fail.replace(false)) {
            bail!("injected denoiser preparation failure");
        }
        #[cfg(feature = "gtcrn")]
        let backend = vc_core::denoise::GtcrnBackend::for_provider(
            config.provider,
            config.gpu_priority,
            config.gpu_device_id,
        );
        let mut passthrough = PassthroughProcessor::new(
            config.denoiser_mode,
            config.noise_gate_shaping,
            input_rate,
            output_rate,
            config.gtcrn_model_dir.clone(),
            #[cfg(feature = "gtcrn")]
            backend,
            live,
        )?;
        passthrough.enable_adaptive_output();
        #[cfg(feature = "gtcrn")]
        if let PassthroughDenoiser::Gtcrn(denoiser) = &mut passthrough.denoiser {
            let mut probe = vec![0.0; denoiser.latency_samples() + input_rate as usize / 10];
            denoiser.process_in_place(&mut probe)?;
            if probe.iter().any(|sample| !sample.is_finite()) {
                bail!("GTCRN preparation produced non-finite audio")
            }
            denoiser.reset()?;
        }
        let rvc = if config.has_complete_model_set() {
            let choice = match config.denoiser_mode {
                DenoiserMode::Off => InputDenoiserConfig::Off,
                DenoiserMode::NoiseGate => InputDenoiserConfig::NoiseGate {
                    threshold: live.noise_gate_threshold,
                    shaping: config.noise_gate_shaping,
                },
                DenoiserMode::Rnnoise => {
                    #[cfg(feature = "rnnoise")]
                    {
                        InputDenoiserConfig::Rnnoise
                    }
                    #[cfg(not(feature = "rnnoise"))]
                    {
                        bail!("RNNoise support is not enabled in this build")
                    }
                }
                DenoiserMode::Gtcrn => {
                    #[cfg(feature = "gtcrn")]
                    {
                        InputDenoiserConfig::Gtcrn {
                            model_dir: config
                                .gtcrn_model_dir
                                .clone()
                                .ok_or_else(|| anyhow!("GTCRN requires a model directory"))?,
                            backend,
                        }
                    }
                    #[cfg(not(feature = "gtcrn"))]
                    {
                        bail!("GTCRN support is not enabled in this build")
                    }
                }
            };
            Some(PreparedInputDenoiser::prepare(input_rate, choice)?)
        } else {
            None
        };
        Ok(Self { passthrough, rvc })
    }

    // Both replacements are prepared before either active route is changed.
    // The old pair stays available until stream restart commits successfully.
    fn install(self, model: &mut RuntimeModel) -> Result<Self> {
        match model {
            RuntimeModel::PassthroughOnly(passthrough) => Ok(Self {
                passthrough: std::mem::replace(passthrough, self.passthrough),
                rvc: None,
            }),
            RuntimeModel::Switchable {
                passthrough, rvc, ..
            } => {
                let previous = rvc
                    .model_mut()
                    .replace_input_denoiser(self.rvc.expect("prepared RVC route"))?;
                Ok(Self {
                    passthrough: std::mem::replace(passthrough, self.passthrough),
                    rvc: Some(previous),
                })
            }
        }
    }
}

pub(super) struct RetainedModel {
    pub model: RuntimeModel,
    pub input_rate: u32,
    pub output_rate: u32,
    pub input_chunk: usize,
    pub clock_ppm: f64,
}

impl RetainedModel {
    pub fn matches(&self, input_rate: u32, output_rate: u32, input_chunk: usize) -> bool {
        (self.input_rate, self.output_rate, self.input_chunk)
            == (input_rate, output_rate, input_chunk)
    }
}

pub(super) fn reuse_or_load(
    retained: &mut Option<RetainedModel>,
    input_rate: u32,
    output_rate: u32,
    input_chunk: usize,
    live: &LiveParams,
    load: impl FnOnce() -> Result<RuntimeModel>,
) -> Result<RuntimeModel> {
    if retained
        .as_ref()
        .is_some_and(|old| old.matches(input_rate, output_rate, input_chunk))
    {
        retained.as_mut().expect("matched").model.reset(live)?;
        Ok(retained.take().expect("matched").model)
    } else {
        // Keep the previous model until a different-rate load actually succeeds,
        // allowing a failed load to restore the old device without another load.
        let model = load()?;
        *retained = None;
        Ok(model)
    }
}

fn exact_selection(names: &[String], selected: &str) -> Result<()> {
    match names
        .iter()
        .filter(|name| name.as_str() == selected)
        .count()
    {
        1 => Ok(()),
        0 => bail!("Selected audio device is unavailable: {selected}"),
        _ => bail!("Selected audio device name is ambiguous: {selected}"),
    }
}

// CLI substring/default resolution remains compatible. GUI live changes are
// explicit selections: never replace an unavailable endpoint with an unrelated default.
fn open_selected(config: &RealtimeConfig) -> Result<RealtimeAudio> {
    if let Some(name) = &config.input_device {
        exact_selection(&audio::cpal_input_names(config.input_host)?, name)?;
    }
    if let Some(name) = &config.output_device {
        exact_selection(&audio::cpal_output_names(config.output_host)?, name)?;
    }
    let audio = RealtimeAudio::open(
        config.input_host,
        config.output_host,
        config.wasapi_input_exclusive,
        config.wasapi_output_exclusive,
        config.input_device.as_deref(),
        config.output_device.as_deref(),
        config.wasapi_buffer_ms,
    )?;
    // Recheck after open: enumeration and opening race with unplug/default changes.
    if config
        .input_device
        .as_deref()
        .is_some_and(|name| name != audio.input_name())
        || config
            .output_device
            .as_deref()
            .is_some_and(|name| name != audio.output_name())
    {
        bail!("Selected audio endpoint changed while opening it");
    }
    config.chunk_samples(audio.input_sample_rate(), audio.output_sample_rate())?;
    Ok(audio)
}

fn publish(
    status: &Mutex<EngineStatusSnapshot>,
    session: Option<&RealtimeSession>,
    session_revision: u64,
    revision: u64,
    error: Option<String>,
) {
    let mut current = status.lock().unwrap();
    if let Some(session) = session {
        *current = session.status();
    }
    current.session_revision = session_revision;
    current.reconfiguration_revision = revision;
    current.reconfiguration_error = error;
    current.input_recoverable = true;
}

// Borrow controller-owned handles individually; neither callbacks nor the
// transferable model may own this control state across worker lifetimes.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_pending(
    mailbox: &ReconfigureMailbox,
    session: &mut Option<RealtimeSession>,
    recovery: &mut Option<(RealtimeConfig, Option<RetainedModel>)>,
    telemetry: &Arc<Telemetry>,
    live: &Arc<AtomicLiveParams>,
    passthrough: &Arc<AtomicBool>,
    status: &Arc<Mutex<EngineStatusSnapshot>>,
    session_revision: u64,
) {
    let Some((revision, change)) = mailbox.take() else {
        return;
    };
    let _completion = Completion(mailbox, revision);
    let Some(old_config) = session
        .as_ref()
        .map(|s| s.config.clone())
        .or_else(|| recovery.as_ref().map(|(config, _)| config.clone()))
    else {
        let mut current = status.lock().unwrap();
        current.reconfiguration_revision = revision;
        current.reconfiguration_error =
            Some("Start conversion before changing audio devices".into());
        return;
    };
    let mut config = old_config.clone();
    let old_applied = session
        .as_ref()
        .and_then(|active| active.status.applied_runtime.clone())
        .or_else(|| status.lock().unwrap().applied_runtime.clone())
        .unwrap_or_else(|| AppliedRuntimeConfig::from_config(&old_config));
    let mut applied = old_applied.clone();
    let input_changed = change.input.is_some();
    let output_changed = change.output.is_some();
    let denoiser_requested = change.denoiser.is_some();
    if let Some(input) = change.input {
        config.input_host = input.host;
        config.input_device = input.device.clone();
        applied.input_host = input.host;
        applied.input_device = input.device;
    }
    if let Some(output) = change.output {
        config.output_host = output.host;
        config.output_device = output.device.clone();
        applied.output_host = output.host;
        applied.output_device = output.device;
    }
    if let Some(denoiser) = change.denoiser {
        config.denoiser_mode = denoiser.mode;
        if denoiser.mode == DenoiserMode::Gtcrn {
            config.gtcrn_model_dir = denoiser.model_dir;
        }
    }
    config.passthrough = passthrough.load(Ordering::Relaxed);
    applied.denoiser_mode = config.denoiser_mode;
    applied.gtcrn_model_dir = config.gtcrn_model_dir.clone();
    let denoiser_changed = old_config.denoiser_mode != config.denoiser_mode
        || (config.denoiser_mode == DenoiserMode::Gtcrn
            && old_config.gtcrn_model_dir != config.gtcrn_model_dir);
    let simple = |mode| matches!(mode, DenoiserMode::Off | DenoiserMode::NoiseGate);
    if !mailbox.is_current(revision) {
        return;
    }
    if !input_changed
        && !output_changed
        && simple(old_config.denoiser_mode)
        && simple(config.denoiser_mode)
    {
        if let Some(active) = session.as_mut() {
            // Off/Gate already share the per-chunk atomic live path. No I/O,
            // queue flush, model reset or interruption belongs in this case.
            active.config.denoiser_mode = config.denoiser_mode;
            active.status.applied_runtime = Some(applied);
            live.noise_gate_enabled.store(
                config.denoiser_mode == DenoiserMode::NoiseGate,
                Ordering::Relaxed,
            );
            publish(status, session.as_ref(), session_revision, revision, None);
            return;
        }
    }
    let audio = match open_selected(&config) {
        Ok(audio) => audio,
        Err(error) => {
            if mailbox.is_current(revision) {
                publish(
                    status,
                    session.as_ref(),
                    session_revision,
                    revision,
                    Some(format!("{error:#}")),
                );
            }
            return;
        }
    };
    if !mailbox.is_current(revision) {
        return;
    }
    let old_timing = session
        .as_ref()
        .map(|active| (active.input_rate, active.output_rate))
        .or_else(|| {
            recovery.as_ref().and_then(|(_, model)| {
                model
                    .as_ref()
                    .map(|model| (model.input_rate, model.output_rate))
            })
        });
    let prepared = if denoiser_changed
        && old_timing == Some((audio.input_sample_rate(), audio.output_sample_rate()))
    {
        set_status(status, EngineState::Starting, "Preparing input denoiser");
        match PreparedDenoisers::prepare(
            &config,
            audio.input_sample_rate(),
            audio.output_sample_rate(),
            &live.load(),
        ) {
            Ok(prepared) => Some(prepared),
            Err(error) => {
                if mailbox.is_current(revision) {
                    publish(
                        status,
                        session.as_ref(),
                        session_revision,
                        revision,
                        Some(format!("{error:#}")),
                    );
                }
                return;
            }
        }
    } else {
        None
    };
    if !mailbox.is_current(revision) {
        return;
    }
    set_status(status, EngineState::Starting, "Switching audio devices");
    let mut retained = if let Some(mut old) = session.take() {
        old.suspend()
    } else {
        recovery.take().and_then(|(_, model)| model)
    };
    // Either endpoint changes the relative device clock. Never carry a servo
    // estimate from the previous pair into the new output resampler timeline.
    if input_changed || output_changed {
        if let Some(model) = retained.as_mut() {
            model.clock_ppm = 0.0;
        }
    }
    let previous_denoisers = if let (Some(prepared), Some(model)) = (prepared, retained.as_mut()) {
        match prepared.install(&mut model.model) {
            Ok(previous) => Some(previous),
            Err(error) => {
                *recovery = Some((old_config, retained));
                set_error_message(
                    status,
                    "Choose available audio devices to resume",
                    Some(format!("{error:#}")),
                );
                publish(
                    status,
                    None,
                    session_revision,
                    revision,
                    Some(format!("{error:#}")),
                );
                return;
            }
        }
    } else {
        None
    };
    telemetry.reset();
    let result = RealtimeSession::start_opened(
        config,
        audio,
        telemetry.clone(),
        live.clone(),
        passthrough.clone(),
        status,
        &mut retained,
        Some((mailbox, revision)),
    );
    // Stop or a newer selection during open/reset must never acknowledge or
    // restart an obsolete selection. Keep sessions only for the next request.
    if !mailbox.is_current(revision) {
        if let Ok(mut obsolete) = result {
            retained = obsolete.suspend();
        }
        if let (Some(previous), Some(model)) = (previous_denoisers, retained.as_mut()) {
            let _ = previous.install(&mut model.model);
        }
        *recovery = Some((old_config, retained));
        return;
    }
    match result {
        Ok(mut new_session) => {
            // start_opened sees pinned I/O names. Publish the original selection
            // intent instead, including Default, even across denoiser restarts.
            new_session.status.applied_runtime = Some(applied);
            if denoiser_requested {
                live.noise_gate_enabled.store(
                    new_session.config.denoiser_mode == DenoiserMode::NoiseGate,
                    Ordering::Relaxed,
                );
            }
            *session = Some(new_session);
            *recovery = None;
            publish(status, session.as_ref(), session_revision, revision, None);
        }
        Err(error) => {
            if let (Some(previous), Some(model)) = (previous_denoisers, retained.as_mut()) {
                if let Err(restore_error) = previous.install(&mut model.model) {
                    *recovery = Some((old_config, retained));
                    set_error_message(
                        status,
                        "Choose available audio devices to resume",
                        Some(format!("{restore_error:#}")),
                    );
                    publish(
                        status,
                        None,
                        session_revision,
                        revision,
                        Some(format!("{error:#}; {restore_error:#}")),
                    );
                    return;
                }
            }
            // Playback may fail after the old endpoint was released. Attempt the
            // exact old endpoints; on failure retain models for another choice.
            let restore = open_selected(&old_config).and_then(|audio| {
                RealtimeSession::start_opened(
                    old_config.clone(),
                    audio,
                    telemetry.clone(),
                    live.clone(),
                    passthrough.clone(),
                    status,
                    &mut retained,
                    Some((mailbox, revision)),
                )
            });
            let detail = match restore {
                Ok(mut restored) => {
                    restored.status.applied_runtime = Some(old_applied);
                    *session = Some(restored);
                    format!("{error:#}")
                }
                Err(restore_error) => {
                    *recovery = Some((old_config, retained));
                    set_error_message(status, "Choose available audio devices to resume", None);
                    format!("{error:#}; restoring previous audio devices failed: {restore_error:#}")
                }
            };
            if mailbox.is_current(revision) {
                publish(
                    status,
                    session.as_ref(),
                    session_revision,
                    revision,
                    Some(detail),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selections_coalesce_and_stop_invalidates_inflight_work() {
        let mailbox = ReconfigureMailbox::default();
        let first = mailbox
            .request(DeviceChange {
                host: AudioHost::Wasapi,
                device: Some("A".into()),
            })
            .unwrap();
        let last = mailbox
            .request(DeviceChange {
                host: AudioHost::Wasapi,
                device: Some("B".into()),
            })
            .unwrap();
        assert!(!mailbox.is_current(first));
        let (revision, change) = mailbox.take().unwrap();
        assert_eq!(revision, last);
        assert_eq!(change.input.unwrap().device.as_deref(), Some("B"));
        mailbox.cancel();
        assert!(!mailbox.is_current(last));
        assert!(mailbox.take().is_none());
    }

    #[test]
    fn explicit_selection_rejects_missing_partial_and_duplicate_names() {
        let names = vec!["Mic A".into(), "Mic B".into()];
        assert!(exact_selection(&names, "Mic A").is_ok());
        assert!(exact_selection(&names, "Mic").is_err());
        assert!(exact_selection(&names, "Missing").is_err());
        assert!(exact_selection(&["Mic A".into(), "Mic A".into()], "Mic A").is_err());
    }

    #[test]
    fn newer_method_change_keeps_an_inflight_microphone_selection() {
        let mailbox = ReconfigureMailbox::default();
        let first = mailbox
            .request(DeviceChange {
                host: AudioHost::Wasapi,
                device: Some("new mic".into()),
            })
            .unwrap();
        let _inflight = mailbox.take().unwrap();
        let completion = Completion(&mailbox, first);
        let last = mailbox
            .request_denoiser(DenoiserChange {
                mode: DenoiserMode::Rnnoise,
                model_dir: None,
            })
            .unwrap();
        drop(completion);
        let (id, change) = mailbox.take().unwrap();
        assert_eq!(id, last);
        assert_eq!(change.input.unwrap().device.as_deref(), Some("new mic"));
        assert_eq!(change.denoiser.unwrap().mode, DenoiserMode::Rnnoise);
        drop(Completion(&mailbox, last));
        let next = mailbox
            .request_denoiser(DenoiserChange {
                mode: DenoiserMode::Off,
                model_dir: None,
            })
            .unwrap();
        let (id, change) = mailbox.take().unwrap();
        assert_eq!(id, next);
        assert!(
            change.input.is_none(),
            "completed/failed inputs must not be retried by unrelated edits"
        );
    }

    #[test]
    fn output_changes_coalesce_with_inflight_input_and_denoiser_then_clear() {
        let mailbox = ReconfigureMailbox::default();
        let first = mailbox
            .request(DeviceChange {
                host: AudioHost::Wasapi,
                device: Some("microphone".into()),
            })
            .unwrap();
        mailbox.take().unwrap();
        mailbox
            .request_output(AudioHost::Asio, Some("old speaker".into()))
            .unwrap();
        mailbox
            .request_denoiser(DenoiserChange {
                mode: DenoiserMode::NoiseGate,
                model_dir: None,
            })
            .unwrap();
        let last = mailbox.request_output(AudioHost::Wasapi, None).unwrap();
        assert!(!mailbox.is_current(first));
        let (revision, change) = mailbox.take().unwrap();
        assert_eq!(revision, last);
        assert_eq!(change.input.unwrap().device.as_deref(), Some("microphone"));
        let output = change.output.unwrap();
        assert_eq!(output.host, AudioHost::Wasapi);
        assert!(
            output.device.is_none(),
            "Default output intent must be retained"
        );
        assert_eq!(change.denoiser.unwrap().mode, DenoiserMode::NoiseGate);
        drop(Completion(&mailbox, last));
        mailbox
            .request_denoiser(DenoiserChange {
                mode: DenoiserMode::Off,
                model_dir: None,
            })
            .unwrap();
        let (_, next) = mailbox.take().unwrap();
        assert!(next.input.is_none() && next.output.is_none());
        mailbox.cancel();
        assert!(mailbox.take().is_none());
    }

    #[test]
    fn off_gate_changes_do_not_open_streams_or_reset_queues() {
        let telemetry = Arc::new(Telemetry::default());
        telemetry.chunks.store(42, Ordering::Relaxed);
        let running = Arc::new(AtomicBool::new(true));
        let config = RealtimeConfig {
            input_device: None,
            output_device: Some("fake output".into()),
            gtcrn_model_dir: Some(PathBuf::from("accepted GTCRN path")),
            passthrough: true,
            ..Default::default()
        };
        let mut session = Some(RealtimeSession {
            config: config.clone(),
            telemetry: telemetry.clone(),
            running: running.clone(),
            last_error: Arc::new(Mutex::new(None)),
            wake: Arc::new(WorkerWake::default()),
            worker: None,
            input_stream: None,
            output_stream: None,
            status: EngineStatusSnapshot {
                state: EngineState::Running,
                input_device: "fake input".into(),
                input_recoverable: true,
                applied_runtime: Some(AppliedRuntimeConfig::from_config(&config)),
                ..Default::default()
            },
            debug_input_wav: None,
            debug_output_wav: None,
            debug_input: Arc::new(Mutex::new(Vec::new())),
            debug_output: Arc::new(Mutex::new(Vec::new())),
            input_rate: 48000,
            output_rate: 48000,
        });
        let mailbox = ReconfigureMailbox::default();
        let live = Arc::new(AtomicLiveParams::new(LiveParams::default()));
        let status = Arc::new(Mutex::new(EngineStatusSnapshot::default()));
        let mut recovery = None;
        for mode in [
            DenoiserMode::NoiseGate,
            DenoiserMode::Off,
            DenoiserMode::NoiseGate,
        ] {
            let id = mailbox
                .request_denoiser(DenoiserChange {
                    mode,
                    model_dir: None,
                })
                .unwrap();
            apply_pending(
                &mailbox,
                &mut session,
                &mut recovery,
                &telemetry,
                &live,
                &Arc::new(AtomicBool::new(true)),
                &status,
                1,
            );
            assert!(status.lock().unwrap().reconfiguration_error.is_none());
            assert_eq!(status.lock().unwrap().reconfiguration_revision, id);
            assert!(Arc::ptr_eq(&session.as_ref().unwrap().running, &running));
            assert_eq!(telemetry.chunks.load(Ordering::Relaxed), 42);
            assert_eq!(
                live.load().noise_gate_enabled,
                mode == DenoiserMode::NoiseGate
            );
            let actual = status.lock().unwrap().applied_runtime.clone().unwrap();
            assert_eq!(actual.denoiser_mode, mode);
            assert!(
                actual.input_device.is_none(),
                "Default intent must not become a resolved name"
            );
        }
        // A later failed completion must publish the accepted intermediate Gate,
        // not merely repeat its request ID or report the failed GTCRN selection.
        let failed = mailbox
            .request_denoiser(DenoiserChange {
                mode: DenoiserMode::Gtcrn,
                model_dir: Some(PathBuf::from("rejected path")),
            })
            .unwrap();
        publish(
            &status,
            session.as_ref(),
            1,
            failed,
            Some("GTCRN preparation failed".into()),
        );
        let latest = status.lock().unwrap().clone();
        assert_eq!(latest.reconfiguration_revision, failed);
        assert!(latest.reconfiguration_error.is_some());
        let actual = latest.applied_runtime.unwrap();
        assert_eq!(actual.denoiser_mode, DenoiserMode::NoiseGate);
        assert!(actual.input_device.is_none());
        assert_eq!(actual.gtcrn_model_dir, config.gtcrn_model_dir);
        drop(Completion(&mailbox, failed));

        // An output-only request must not take the Off/Gate shortcut and report
        // success without opening the selected endpoint. Missing names fail
        // before any streams start, preserving the running pair and clock.
        let failed = mailbox
            .request_output(
                AudioHost::Wasapi,
                Some("__vc_rs_nonexistent_output__".into()),
            )
            .unwrap();
        apply_pending(
            &mailbox,
            &mut session,
            &mut recovery,
            &telemetry,
            &live,
            &Arc::new(AtomicBool::new(true)),
            &status,
            1,
        );
        let latest = status.lock().unwrap().clone();
        assert_eq!(latest.reconfiguration_revision, failed);
        assert!(latest.reconfiguration_error.is_some());
        assert!(Arc::ptr_eq(&session.as_ref().unwrap().running, &running));
        assert_eq!(telemetry.chunks.load(Ordering::Relaxed), 42);
        let actual = latest.applied_runtime.unwrap();
        assert_eq!(actual.output_device.as_deref(), Some("fake output"));
        assert_eq!(actual.denoiser_mode, DenoiserMode::NoiseGate);
    }

    #[cfg(feature = "rnnoise")]
    #[test]
    fn prepared_rnnoise_and_off_replace_only_denoising_and_drop_old_audio() {
        let live = LiveParams::default();
        let mut model = retained().model;
        let config = RealtimeConfig {
            passthrough: true,
            denoiser_mode: DenoiserMode::Rnnoise,
            ..Default::default()
        };
        let previous = PreparedDenoisers::prepare(&config, 48000, 48000, &live)
            .unwrap()
            .install(&mut model)
            .unwrap();
        model.reset(&live).unwrap();
        let mut out = Vec::new();
        for _ in 0..6 {
            model
                .process_chunk(&[0.4; 960], 48000, &live, true, &mut out)
                .unwrap();
        }
        previous.install(&mut model).unwrap();
        model.reset(&live).unwrap();
        model
            .process_chunk(&[0.0; 960], 48000, &live, true, &mut out)
            .unwrap();
        assert!(out.iter().all(|&sample| sample == 0.0));
        let RuntimeModel::PassthroughOnly(processor) = model else {
            panic!()
        };
        assert_eq!(processor.mode, DenoiserMode::Off);
    }

    fn retained() -> RetainedModel {
        let mut processor = PassthroughProcessor::new(
            DenoiserMode::Off,
            NoiseGateShaping::default(),
            48000,
            48000,
            None,
            #[cfg(feature = "gtcrn")]
            vc_core::denoise::GtcrnBackend::OrtCpu,
            &LiveParams::default(),
        )
        .unwrap();
        processor.enable_adaptive_output();
        RetainedModel {
            model: RuntimeModel::PassthroughOnly(processor),
            input_rate: 48000,
            output_rate: 48000,
            input_chunk: 960,
            clock_ppm: 0.0,
        }
    }

    #[test]
    fn either_endpoint_loss_retains_healthy_models_but_not_failed_inference() {
        for message in ["Input audio stream stopped", "Output audio stream stopped"] {
            for healthy in [true, false] {
                let config = RealtimeConfig {
                    input_device: Some("previous microphone".into()),
                    output_device: Some("previous speaker".into()),
                    passthrough: true,
                    chunk_ms: 20,
                    ..Default::default()
                };
                let running = Arc::new(AtomicBool::new(healthy));
                let wake = Arc::new(WorkerWake::default());
                let joined = Arc::new(AtomicBool::new(false));
                let worker = {
                    let running = running.clone();
                    let wake = wake.clone();
                    let joined = joined.clone();
                    let model = retained().model;
                    thread::spawn(move || {
                        wake.register_current();
                        while running.load(Ordering::SeqCst) {
                            thread::park_timeout(Duration::from_millis(100));
                        }
                        joined.store(true, Ordering::SeqCst);
                        model
                    })
                };
                let detail = (!healthy).then(|| "injected inference failure".to_string());
                let failed = RealtimeSession {
                    config: config.clone(),
                    telemetry: Arc::new(Telemetry::default()),
                    running,
                    last_error: Arc::new(Mutex::new(detail.clone())),
                    wake,
                    worker: Some(worker),
                    input_stream: None,
                    output_stream: None,
                    status: EngineStatusSnapshot::default(),
                    debug_input_wav: None,
                    debug_output_wav: None,
                    debug_input: Arc::new(Mutex::new(Vec::new())),
                    debug_output: Arc::new(Mutex::new(Vec::new())),
                    input_rate: 48000,
                    output_rate: 48000,
                };
                let status = Mutex::new(EngineStatusSnapshot {
                    state: EngineState::Running,
                    applied_runtime: Some(AppliedRuntimeConfig::from_config(&config)),
                    ..Default::default()
                });
                let mut recovery = None;
                handle_audio_failure(failed, &mut recovery, &status, message);
                assert!(joined.load(Ordering::SeqCst), "{message}");
                let current = status.lock().unwrap().clone();
                assert_eq!(current.state, EngineState::Error);
                assert_eq!(current.message, message);
                assert_eq!(current.detail, detail);
                assert_eq!(current.input_recoverable, healthy);
                assert_eq!(
                    current.applied_runtime,
                    Some(AppliedRuntimeConfig::from_config(&config))
                );
                if healthy {
                    let (previous, mut model) = recovery.expect("recoverable endpoint loss");
                    assert_eq!(previous.input_device, config.input_device);
                    assert_eq!(previous.output_device, config.output_device);
                    assert!(model.as_ref().unwrap().matches(48000, 48000, 960));
                    reuse_or_load(
                        &mut model,
                        48000,
                        48000,
                        960,
                        &LiveParams::default(),
                        || panic!("endpoint recovery must reuse the retained model"),
                    )
                    .unwrap();
                } else {
                    assert!(recovery.is_none(), "do not reuse failed inference");
                }
            }
        }
    }

    #[test]
    fn recycling_skips_loader_discards_audio_and_old_microphone_clock() {
        let mut old = retained();
        let mut out = Vec::new();
        let live = LiveParams::default();
        for _ in 0..8 {
            old.model.set_clock_correction(1200.0, true).unwrap();
            old.model
                .process_chunk(&[0.7; 960], 48000, &live, true, &mut out)
                .unwrap();
        }
        let mut slot = Some(old);
        let mut recycled = reuse_or_load(&mut slot, 48000, 48000, 960, &live, || {
            panic!("must not reload")
        })
        .unwrap();
        let RuntimeModel::PassthroughOnly(processor) = &recycled else {
            panic!()
        };
        assert_eq!(processor.correction_ppm, 0.0);
        assert!(processor.adaptive_resampler.is_none());
        recycled
            .process_chunk(&[0.0; 960], 48000, &live, true, &mut out)
            .unwrap();
        assert!(out.iter().all(|&sample| sample == 0.0));
        assert!(slot.is_none());
    }

    #[test]
    fn changed_rate_load_failure_preserves_previous_model_for_restore() {
        let mut old = Some(retained());
        assert!(reuse_or_load(
            &mut old,
            44100,
            48000,
            882,
            &LiveParams::default(),
            || bail!("injected load failure")
        )
        .is_err());
        assert!(old.as_ref().unwrap().matches(48000, 48000, 960));
        let restored = reuse_or_load(&mut old, 48000, 48000, 960, &LiveParams::default(), || {
            panic!("restore reloaded")
        })
        .unwrap();
        drop(restored);
    }

    #[test]
    fn changing_either_rate_or_the_input_hop_uses_the_loader_once() {
        for (input, output, hop) in [
            (44100, 48000, 882),
            (48000, 44100, 960),
            (48000, 48000, 1440),
        ] {
            let mut old = Some(retained());
            let calls = std::cell::Cell::new(0);
            let model = reuse_or_load(&mut old, input, output, hop, &LiveParams::default(), || {
                calls.set(calls.get() + 1);
                Ok(retained().model)
            })
            .unwrap();
            assert_eq!(calls.get(), 1);
            assert!(old.is_none());
            drop(model);
        }
    }

    fn diagnostic_config() -> RealtimeConfig {
        RealtimeConfig {
            model: Some(std::env::var_os("VC_RS_DIAG_MODEL").expect("model").into()),
            embedder: Some(
                std::env::var_os("VC_RS_DIAG_EMBEDDER")
                    .expect("embedder")
                    .into(),
            ),
            f0_model: Some(std::env::var_os("VC_RS_DIAG_F0").expect("F0").into()),
            provider: Provider::from_name(
                &std::env::var("VC_RS_DIAG_PROVIDER").unwrap_or("windowsml-directml".into()),
            )
            .unwrap(),
            chunk_ms: std::env::var("VC_RS_DIAG_CHUNK_MS")
                .ok()
                .map(|value| value.parse().unwrap())
                .unwrap_or(500),
            ..Default::default()
        }
    }

    #[test]
    #[ignore = "opens selected/default devices silently; requires VC_RS_DIAG_MODEL/EMBEDDER/F0"]
    fn live_microphone_switch_without_recording_or_playback() {
        let config = diagnostic_config();
        let telemetry = Arc::new(Telemetry::default());
        let live = Arc::new(AtomicLiveParams::new(LiveParams {
            input_gain: 0.0,
            output_gain: 0.0,
            ..Default::default()
        }));
        let bypass = Arc::new(AtomicBool::new(false));
        let status = Arc::new(Mutex::new(EngineStatusSnapshot::default()));
        let mut session = Some(
            RealtimeSession::start(
                config,
                telemetry.clone(),
                live.clone(),
                bypass.clone(),
                &status,
            )
            .unwrap(),
        );
        let mut recovery = None;
        let mailbox = ReconfigureMailbox::default();
        let names = audio::cpal_input_names(AudioHost::Wasapi).unwrap();
        assert!(!names.is_empty());
        let initial_loads = MODEL_LOADS.with(|count| count.get());
        for name in names.iter().take(2) {
            let rates = (
                session.as_ref().unwrap().input_rate,
                session.as_ref().unwrap().output_rate,
            );
            let id = mailbox
                .request(DeviceChange {
                    host: AudioHost::Wasapi,
                    device: Some(name.clone()),
                })
                .unwrap();
            apply_pending(
                &mailbox,
                &mut session,
                &mut recovery,
                &telemetry,
                &live,
                &bypass,
                &status,
                1,
            );
            let current = status.lock().unwrap().clone();
            assert_eq!(current.reconfiguration_revision, id);
            assert!(current.reconfiguration_error.is_none(), "{:?}", current);
            assert_eq!(&current.input_device, name);
            if rates == (current.input_sample_rate, current.output_sample_rate) {
                assert_eq!(
                    MODEL_LOADS.with(|count| count.get()),
                    initial_loads,
                    "same-rate switch reloaded models"
                );
            }
            let deadline = Instant::now() + Duration::from_secs(30);
            while telemetry.chunks.load(Ordering::Relaxed) == 0 && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(20));
            }
            assert!(
                telemetry.chunks.load(Ordering::Relaxed) > 0,
                "worker did not resume"
            );
            assert!(session.as_ref().unwrap().last_error().is_none());
            println!(
                "silent microphone switch: {} / {} -> {} Hz; model loads={}",
                name,
                current.input_sample_rate,
                current.output_sample_rate,
                MODEL_LOADS.with(|count| count.get())
            );
        }
        let old = session.as_ref().unwrap().status.input_device.clone();
        FAIL_NEXT_PLAY.with(|fail| fail.set(true));
        mailbox
            .request(DeviceChange {
                host: AudioHost::Wasapi,
                device: Some(old.clone()),
            })
            .unwrap();
        apply_pending(
            &mailbox,
            &mut session,
            &mut recovery,
            &telemetry,
            &live,
            &bypass,
            &status,
            1,
        );
        assert!(status
            .lock()
            .unwrap()
            .reconfiguration_error
            .as_ref()
            .unwrap()
            .contains("injected"));
        assert_eq!(status.lock().unwrap().state, EngineState::Running);
        assert_eq!(session.as_ref().unwrap().status.input_device, old);
        assert_eq!(MODEL_LOADS.with(|count| count.get()), initial_loads);
        mailbox
            .request(DeviceChange {
                host: AudioHost::Wasapi,
                device: Some("__vc_rs_missing_input__".into()),
            })
            .unwrap();
        apply_pending(
            &mailbox,
            &mut session,
            &mut recovery,
            &telemetry,
            &live,
            &bypass,
            &status,
            1,
        );
        assert!(status.lock().unwrap().reconfiguration_error.is_some());
        assert_eq!(session.as_ref().unwrap().status.input_device, old);
        drop(session);
    }

    #[test]
    #[ignore = "silently opens output devices; requires VC_RS_DIAG_MODEL/EMBEDDER/F0"]
    fn live_output_switch_without_recording_or_playback() {
        let telemetry = Arc::new(Telemetry::default());
        let live = Arc::new(AtomicLiveParams::new(LiveParams {
            input_gain: 0.0,
            output_gain: 0.0,
            ..Default::default()
        }));
        let bypass = Arc::new(AtomicBool::new(false));
        let status = Arc::new(Mutex::new(EngineStatusSnapshot::default()));
        let mut session = Some(
            RealtimeSession::start(
                diagnostic_config(),
                telemetry.clone(),
                live.clone(),
                bypass.clone(),
                &status,
            )
            .unwrap(),
        );
        let original = session.as_ref().unwrap().config.clone();
        let names = audio::cpal_output_names(AudioHost::Wasapi).unwrap();
        let choices: Vec<_> = ["VC_RS_DIAG_OUTPUT_A", "VC_RS_DIAG_OUTPUT_B"]
            .into_iter()
            .filter_map(|key| std::env::var(key).ok())
            .collect();
        let choices: Vec<_> = if choices.is_empty() {
            names.into_iter().take(2).collect()
        } else {
            choices
        };
        assert!(choices.len() >= 2, "Provide two outputs for the diagnostic");
        let mailbox = ReconfigureMailbox::default();
        let mut recovery = None;
        for device in choices.into_iter().map(Some).chain(std::iter::once(None)) {
            let active = session.as_ref().unwrap();
            let rates = (active.input_rate, active.output_rate);
            let before_loads = MODEL_LOADS.with(|count| count.get());
            let revision = mailbox
                .request_output(AudioHost::Wasapi, device.clone())
                .unwrap();
            apply_pending(
                &mailbox,
                &mut session,
                &mut recovery,
                &telemetry,
                &live,
                &bypass,
                &status,
                1,
            );
            let current = status.lock().unwrap().clone();
            assert_eq!(current.reconfiguration_revision, revision);
            assert!(current.reconfiguration_error.is_none(), "{current:?}");
            let active = session.as_ref().unwrap();
            assert_eq!(active.config.input_device, original.input_device);
            assert_eq!(active.config.model, original.model);
            assert_eq!(active.config.provider, original.provider);
            assert_eq!(
                current.applied_runtime.as_ref().unwrap().output_device,
                device
            );
            if let Some(name) = device {
                assert_eq!(current.output_device, name);
            }
            if rates == (current.input_sample_rate, current.output_sample_rate) {
                assert_eq!(
                    MODEL_LOADS.with(|count| count.get()),
                    before_loads,
                    "same-rate output change reloaded models"
                );
            }
            for route in [false, true] {
                bypass.store(route, Ordering::Relaxed);
                let before = telemetry.chunks.load(Ordering::Relaxed);
                let deadline = Instant::now() + Duration::from_secs(30);
                while telemetry.chunks.load(Ordering::Relaxed) <= before
                    && Instant::now() < deadline
                {
                    thread::sleep(Duration::from_millis(20));
                }
                assert!(
                    telemetry.chunks.load(Ordering::Relaxed) > before,
                    "worker did not resume"
                );
                assert!(active.last_error().is_none());
            }
            println!(
                "silent output switch: {} / {} -> {} Hz; model loads={}",
                current.output_device,
                current.input_sample_rate,
                current.output_sample_rate,
                MODEL_LOADS.with(|count| count.get())
            );
        }
        let old = session.as_ref().unwrap().status.output_device.clone();
        let before_loads = MODEL_LOADS.with(|count| count.get());
        for inject_play_failure in [true, false] {
            if inject_play_failure {
                FAIL_NEXT_PLAY.with(|fail| fail.set(true));
            }
            mailbox
                .request_output(
                    AudioHost::Wasapi,
                    Some(if inject_play_failure {
                        old.clone()
                    } else {
                        "__vc_rs_missing_output__".into()
                    }),
                )
                .unwrap();
            apply_pending(
                &mailbox,
                &mut session,
                &mut recovery,
                &telemetry,
                &live,
                &bypass,
                &status,
                1,
            );
            let current = status.lock().unwrap().clone();
            assert!(current.reconfiguration_error.is_some());
            assert_eq!(current.state, EngineState::Running);
            assert_eq!(current.output_device, old);
            assert!(
                current.applied_runtime.unwrap().output_device.is_none(),
                "Failed output change lost Default intent"
            );
            assert_eq!(MODEL_LOADS.with(|count| count.get()), before_loads);
            bypass.store(false, Ordering::Relaxed);
            let before = telemetry.chunks.load(Ordering::Relaxed);
            let deadline = Instant::now() + Duration::from_secs(30);
            while telemetry.chunks.load(Ordering::Relaxed) <= before && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(20));
            }
            assert!(
                telemetry.chunks.load(Ordering::Relaxed) > before,
                "conversion did not resume after failed output change"
            );
            assert!(session.as_ref().unwrap().last_error().is_none());
        }

        // Exercise the controller's output-loss path with real loaded sessions,
        // without unplugging hardware or recording/playing any microphone audio.
        handle_audio_failure(
            session.take().unwrap(),
            &mut recovery,
            &status,
            "Output audio stream stopped",
        );
        assert_eq!(status.lock().unwrap().state, EngineState::Error);
        assert!(status.lock().unwrap().input_recoverable);
        assert!(recovery.as_ref().unwrap().1.is_some());
        mailbox
            .request_output(AudioHost::Wasapi, Some(old))
            .unwrap();
        apply_pending(
            &mailbox,
            &mut session,
            &mut recovery,
            &telemetry,
            &live,
            &bypass,
            &status,
            1,
        );
        let current = status.lock().unwrap().clone();
        assert_eq!(current.state, EngineState::Running);
        assert!(current.reconfiguration_error.is_none(), "{current:?}");
        assert_eq!(MODEL_LOADS.with(|count| count.get()), before_loads);
        assert!(recovery.is_none());
        for route in [false, true] {
            bypass.store(route, Ordering::Relaxed);
            let before = telemetry.chunks.load(Ordering::Relaxed);
            let deadline = Instant::now() + Duration::from_secs(30);
            while telemetry.chunks.load(Ordering::Relaxed) <= before && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(20));
            }
            assert!(telemetry.chunks.load(Ordering::Relaxed) > before);
            assert!(session.as_ref().unwrap().last_error().is_none());
        }
        println!("silent output-loss recovery: both routes resumed without model reload");
    }

    #[cfg(all(feature = "rnnoise", feature = "gtcrn"))]
    #[test]
    #[ignore = "silently opens devices; requires VC_RS_DIAG_MODEL/EMBEDDER/F0/GTCRN"]
    fn live_denoiser_switch_without_recording_or_playback() {
        exercise_live_denoiser_switch(false);
    }

    #[cfg(all(feature = "rnnoise", feature = "gtcrn"))]
    #[test]
    #[ignore = "silently opens devices; injects GTCRN preparation failure with VC_RS_DIAG_* models"]
    fn live_gtcrn_preparation_failure_retains_running_routes() {
        exercise_live_denoiser_switch(true);
    }

    #[cfg(all(feature = "rnnoise", feature = "gtcrn"))]
    fn exercise_live_denoiser_switch(expect_gtcrn_failure: bool) {
        let mut config = diagnostic_config();
        config.gtcrn_model_dir = Some(
            std::env::var_os("VC_RS_DIAG_GTCRN")
                .expect("GTCRN directory")
                .into(),
        );
        let telemetry = Arc::new(Telemetry::default());
        let live = Arc::new(AtomicLiveParams::new(LiveParams {
            input_gain: 0.0,
            output_gain: 0.0,
            ..Default::default()
        }));
        let bypass = Arc::new(AtomicBool::new(false));
        let status = Arc::new(Mutex::new(EngineStatusSnapshot::default()));
        let mut session = Some(
            RealtimeSession::start(
                config.clone(),
                telemetry.clone(),
                live.clone(),
                bypass.clone(),
                &status,
            )
            .unwrap(),
        );
        let initial_loads = MODEL_LOADS.with(|count| count.get());
        let output = session.as_ref().unwrap().config.output_device.clone();
        let mailbox = ReconfigureMailbox::default();
        let mut recovery = None;
        for mode in [
            DenoiserMode::NoiseGate,
            DenoiserMode::Off,
            DenoiserMode::Rnnoise,
            DenoiserMode::Gtcrn,
            DenoiserMode::Rnnoise,
        ] {
            let old_running = session.as_ref().unwrap().running.clone();
            if expect_gtcrn_failure && mode == DenoiserMode::Gtcrn {
                FAIL_NEXT_DENOISER_PREPARATION.with(|fail| fail.set(true));
            }
            let id = mailbox
                .request_denoiser(DenoiserChange {
                    mode,
                    model_dir: config.gtcrn_model_dir.clone(),
                })
                .unwrap();
            apply_pending(
                &mailbox,
                &mut session,
                &mut recovery,
                &telemetry,
                &live,
                &bypass,
                &status,
                1,
            );
            let current = status.lock().unwrap().clone();
            assert_eq!(current.reconfiguration_revision, id);
            if let Some(error) = current
                .reconfiguration_error
                .as_ref()
                .filter(|_| mode == DenoiserMode::Gtcrn && expect_gtcrn_failure)
            {
                assert!(error.contains("injected denoiser preparation failure"));
                assert_eq!(current.state, EngineState::Running);
                assert!(Arc::ptr_eq(
                    &old_running,
                    &session.as_ref().unwrap().running
                ));
                assert_eq!(
                    session.as_ref().unwrap().config.denoiser_mode,
                    DenoiserMode::Rnnoise
                );
                assert_eq!(MODEL_LOADS.with(|count| count.get()), initial_loads);
                for route in [true, false] {
                    bypass.store(route, Ordering::Relaxed);
                    let before = telemetry.chunks.load(Ordering::Relaxed);
                    let deadline = Instant::now() + Duration::from_secs(30);
                    while telemetry.chunks.load(Ordering::Relaxed) <= before
                        && Instant::now() < deadline
                    {
                        thread::sleep(Duration::from_millis(20));
                    }
                    assert!(telemetry.chunks.load(Ordering::Relaxed) > before);
                    assert!(session.as_ref().unwrap().last_error().is_none());
                }
                let mut retained = session.as_mut().unwrap().suspend().unwrap();
                assert_rnnoise_routes(&mut retained);
                println!("GTCRN preparation failed before commit; original RNNoise worker and both routes retained, voice-model loads={initial_loads}: {error}");
                return;
            }
            assert!(current.reconfiguration_error.is_none(), "{current:?}");
            assert_eq!(
                MODEL_LOADS.with(|count| count.get()),
                initial_loads,
                "denoiser change reloaded voice models"
            );
            assert_eq!(session.as_ref().unwrap().config.output_device, output);
            assert_eq!(session.as_ref().unwrap().config.denoiser_mode, mode);
            if matches!(mode, DenoiserMode::Off | DenoiserMode::NoiseGate) {
                assert!(Arc::ptr_eq(
                    &old_running,
                    &session.as_ref().unwrap().running
                ));
            }
            for route in [false, true, false] {
                bypass.store(route, Ordering::Relaxed);
                let before = telemetry.chunks.load(Ordering::Relaxed);
                let deadline = Instant::now() + Duration::from_secs(30);
                while telemetry.chunks.load(Ordering::Relaxed) <= before
                    && Instant::now() < deadline
                {
                    thread::sleep(Duration::from_millis(20));
                }
                assert!(
                    telemetry.chunks.load(Ordering::Relaxed) > before,
                    "route failed to resume: {:?}",
                    session.as_ref().unwrap().last_error()
                );
                assert!(session.as_ref().unwrap().last_error().is_none());
            }
            println!("silent denoiser switch: {mode:?}, both routes processed; voice-model loads={initial_loads}");
        }
        assert!(
            !expect_gtcrn_failure,
            "GTCRN unexpectedly succeeded; run the all-method success diagnostic"
        );
        // Preparation failure and a later device-play failure must both keep
        // RNNoise on both paths, without rebuilding any voice-model session.
        mailbox
            .request_denoiser(DenoiserChange {
                mode: DenoiserMode::Gtcrn,
                model_dir: Some(PathBuf::from("__missing_gtcrn__")),
            })
            .unwrap();
        apply_pending(
            &mailbox,
            &mut session,
            &mut recovery,
            &telemetry,
            &live,
            &bypass,
            &status,
            1,
        );
        assert!(status.lock().unwrap().reconfiguration_error.is_some());
        assert_eq!(
            session.as_ref().unwrap().config.denoiser_mode,
            DenoiserMode::Rnnoise
        );
        FAIL_NEXT_PLAY.with(|fail| fail.set(true));
        mailbox
            .request_denoiser(DenoiserChange {
                mode: DenoiserMode::Gtcrn,
                model_dir: config.gtcrn_model_dir,
            })
            .unwrap();
        apply_pending(
            &mailbox,
            &mut session,
            &mut recovery,
            &telemetry,
            &live,
            &bypass,
            &status,
            1,
        );
        assert!(status
            .lock()
            .unwrap()
            .reconfiguration_error
            .as_ref()
            .unwrap()
            .contains("injected"));
        assert_eq!(status.lock().unwrap().state, EngineState::Running);
        assert_eq!(MODEL_LOADS.with(|count| count.get()), initial_loads);
        let mut retained = session.as_mut().unwrap().suspend().unwrap();
        assert_rnnoise_routes(&mut retained);
    }

    #[cfg(all(feature = "rnnoise", feature = "gtcrn"))]
    fn assert_rnnoise_routes(retained: &mut RetainedModel) {
        let RuntimeModel::Switchable {
            passthrough, rvc, ..
        } = &mut retained.model
        else {
            panic!()
        };
        assert_eq!(passthrough.mode, DenoiserMode::Rnnoise);
        // RNNoise has a positive input delay; the GTCRN stage would have its own
        // 16 kHz delay instead. Compare against a prepared RNNoise reference.
        let expected = vc_core::denoise::RnnoiseDenoiser::new(retained.input_rate)
            .unwrap()
            .latency_samples();
        let with_denoiser = rvc
            .model_mut()
            .input_content_delay_samples(retained.input_rate);
        let old = rvc
            .model_mut()
            .replace_input_denoiser(
                PreparedInputDenoiser::prepare(retained.input_rate, InputDenoiserConfig::Off)
                    .unwrap(),
            )
            .unwrap();
        let without_denoiser = rvc
            .model_mut()
            .input_content_delay_samples(retained.input_rate);
        assert_eq!(
            with_denoiser - without_denoiser,
            expected,
            "RVC did not restore RNNoise's exact delay"
        );
        rvc.model_mut().replace_input_denoiser(old).unwrap();
    }
}
