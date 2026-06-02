mod audio;
mod cli;
mod engine;

use anyhow::Result;
use cli::{Cli, Command};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;
use vc_core::model_rvc;

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
        Command::Run(args) => engine::run_realtime(args),
        Command::Wav(args) => engine::run_wav(args),
    }
}
