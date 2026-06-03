mod audio;
mod cli;
mod engine;
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
        Command::Devices(args) => audio::print_devices(args.audio_backend),
        Command::Inspect(args) => model_rvc::inspect_model(&args.model),
        #[cfg(all(windows, feature = "windowsml"))]
        Command::WindowsMlEps(args) => windows_ml_eps::run(args),
        Command::Run(args) => run_model_command(move || engine::run_realtime(args)),
        Command::Wav(args) => run_model_command(move || engine::run_wav(args)),
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
