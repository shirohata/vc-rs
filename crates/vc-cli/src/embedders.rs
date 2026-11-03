//! Explicit acquisition commands; run/wav never acquire or switch embedders.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::cli::{EmbedderPackageArgs, EmbeddersArgs, EmbeddersCommand};

pub fn run(args: EmbeddersArgs) -> Result<()> {
    match args.command {
        EmbeddersCommand::List => {
            for model in vc_models::catalog() {
                println!(
                    "{} {} · {} · {} · {:.1} MB · {}",
                    model.id,
                    model.version,
                    model.label,
                    model.license,
                    model.archive.size as f64 / 1_000_000.0,
                    if model.url.is_some() {
                        "published source configured"
                    } else {
                        "local ZIP only; no published source configured"
                    }
                );
            }
            println!("Select the weights, layer, projection and preprocessing used to train your voice model. Matching dimensions alone do not establish compatibility.");
            Ok(())
        }
        EmbeddersCommand::Import { package, zip } => {
            let (model, root) = resolve(&package)?;
            let cancel = cancellation()?;
            let installed = vc_models::import(model, &zip, &root, &cancel, |_| {})?;
            report(installed);
            Ok(())
        }
        EmbeddersCommand::Fetch(package) => {
            let (model, root) = resolve(&package)?;
            let cancel = cancellation()?;
            let installed = vc_models::download(model, &root, &cancel, |_| {})?;
            report(installed);
            Ok(())
        }
        EmbeddersCommand::Verify(package) => {
            let (model, root) = resolve(&package)?;
            let cancel = cancellation()?;
            vc_models::verify_cached(model, &root, &cancel)?;
            println!("Verified ONNX, README, LICENSE and NOTICE.");
            println!("{}", model.model_path(&root).display());
            Ok(())
        }
    }
}

fn cancellation() -> Result<Arc<AtomicBool>> {
    let cancel = Arc::new(AtomicBool::new(false));
    let signal = cancel.clone();
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))
        .context("Cannot register the package cancellation handler")?;
    Ok(cancel)
}

fn resolve(args: &EmbedderPackageArgs) -> Result<(&'static vc_models::Model, std::path::PathBuf)> {
    let model = vc_models::find(&args.id, &args.version)?;
    let root = match &args.cache_dir {
        Some(root) => root.clone(),
        None => vc_models::cache_dir()?,
    };
    Ok((model, root))
}

fn report(installed: vc_models::Installed) {
    println!(
        "{}",
        if installed.reused {
            "Reused verified package."
        } else {
            "Imported verified package."
        }
    );
    println!("{}", installed.path.display());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use clap::Parser;

    #[test]
    fn commands_parse_explicit_identity_version_and_source() {
        let cli = Cli::try_parse_from([
            "vc-rs",
            "embedders",
            "import",
            "applio-spin",
            "--zip",
            "spin.zip",
            "--version",
            "0.1.0",
            "--cache-dir",
            "cache",
        ])
        .unwrap();
        let Command::Embedders(EmbeddersArgs {
            command: EmbeddersCommand::Import { package, zip },
        }) = cli.command
        else {
            panic!("import");
        };
        assert_eq!(package.id, "applio-spin");
        assert_eq!(package.version, "0.1.0");
        assert_eq!(zip, std::path::Path::new("spin.zip"));
        assert_eq!(package.cache_dir.unwrap(), std::path::Path::new("cache"));
        assert!(Cli::try_parse_from(["vc-rs", "embedders", "import", "applio-spin"]).is_err());
        for action in ["list", "fetch", "verify"] {
            let mut args = vec!["vc-rs", "embedders", action];
            if action != "list" {
                args.push("applio-spin");
            }
            Cli::try_parse_from(args).unwrap();
        }
    }

    #[test]
    fn unknown_identity_or_version_fails_before_source_or_cache_io() {
        for (id, version) in [("unknown", "0.1.0"), ("applio-spin", "latest")] {
            let args = EmbeddersArgs {
                command: EmbeddersCommand::Import {
                    package: EmbedderPackageArgs {
                        id: id.into(),
                        version: version.into(),
                        cache_dir: Some("never-created".into()),
                    },
                    zip: "missing.zip".into(),
                },
            };
            assert!(run(args)
                .unwrap_err()
                .to_string()
                .contains("Unknown embedder package"));
        }
    }
}
