mod cli;
mod doctor;
mod embedders;
mod engine;
mod engine_cache;
mod join_report;
mod performance_report;
#[cfg(all(windows, feature = "windowsml"))]
mod windows_ml_eps;

use anyhow::{anyhow, Context, Result};
use cli::{Cli, Command};
use std::thread;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;
use vc_core::model_rvc;

const MODEL_COMMAND_STACK_SIZE: usize = 64 * 1024 * 1024;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::WARN.into())
                .from_env_lossy(),
        )
        .init();

    let cli = Cli::parse_args();
    match cli.command {
        Command::Doctor => doctor::run(),
        Command::Devices(args) => list_devices(args.audio_backend),
        Command::Inspect(args) => model_rvc::inspect_model(&args.model),
        Command::Embedders(args) => embedders::run(args),
        #[cfg(all(windows, feature = "windowsml"))]
        Command::WindowsMlEps(args) => windows_ml_eps::run(args),
        Command::EngineCache(args) => engine_cache::run(args),
        Command::Run(args) => run_model_command(move || engine::run_realtime(args)),
        Command::Wav(args) => run_model_command(move || engine::run_wav(args)),
    }
}

fn list_devices(selection: cli::DeviceHost) -> Result<()> {
    use vc_app::AudioHost;
    // Enumeration is uniform through cpal for every host. `All` lists the hosts
    // relevant to this platform and reports (rather than aborts on) any that are
    // unavailable, e.g. ASIO without a driver or the `asio` feature.
    let hosts: &[AudioHost] = match selection {
        cli::DeviceHost::All => platform_device_hosts(),
        cli::DeviceHost::Wasapi => &[AudioHost::Wasapi],
        cli::DeviceHost::Asio => &[AudioHost::Asio],
        cli::DeviceHost::CoreAudio => &[AudioHost::CoreAudio],
        cli::DeviceHost::Alsa => &[AudioHost::Alsa],
        cli::DeviceHost::Jack => &[AudioHost::Jack],
    };
    for (index, host) in hosts.iter().enumerate() {
        if index > 0 {
            println!();
        }
        if let Err(err) = vc_app::audio::print_cpal_devices(*host) {
            println!("{host:?} devices unavailable: {err:#}");
        }
    }
    Ok(())
}

// Hosts shown by `devices --audio-backend all` on this platform.
fn platform_device_hosts() -> &'static [vc_app::AudioHost] {
    use vc_app::AudioHost;
    #[cfg(windows)]
    {
        &[AudioHost::Wasapi, AudioHost::Asio]
    }
    #[cfg(target_os = "macos")]
    {
        &[AudioHost::CoreAudio]
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        &[AudioHost::Alsa]
    }
}

fn run_model_command(f: impl FnOnce() -> Result<()> + Send + 'static) -> Result<()> {
    // Windows ML catalog EPs can make ORT session construction consume more
    // stack than the Windows executable default. Keep this at the CLI boundary
    // so audio callbacks stay unaffected and provider-specific code need not
    // rely on process-wide linker stack settings.
    thread::Builder::new()
        .name("vc-rs-model-command".to_string())
        .stack_size(MODEL_COMMAND_STACK_SIZE)
        .spawn(f)
        .context("failed to spawn model command thread")?
        .join()
        .map_err(|_| anyhow!("model command thread panicked"))?
}
