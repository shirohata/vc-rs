use std::io::{self, Write};

use anyhow::{bail, Context, Result};
use vc_core::windows_ml::{self, CatalogExecutionProvider, CatalogProviderInfo, CatalogReadyState};

use crate::cli::{
    WindowsMlEpProvider, WindowsMlEpsArgs, WindowsMlEpsCommand, WindowsMlEpsInstallArgs,
};

pub fn run(args: WindowsMlEpsArgs) -> Result<()> {
    match args.command {
        WindowsMlEpsCommand::List => list(),
        WindowsMlEpsCommand::Install(args) => install(args),
    }
}

fn list() -> Result<()> {
    let providers = windows_ml::list_catalog_providers()?;
    if providers.is_empty() {
        println!("No Windows ML catalog execution providers are visible on this device.");
        return Ok(());
    }

    println!("Windows ML catalog execution providers:");
    for provider in &providers {
        print_provider(provider);
    }

    match windows_ml::select_best_catalog_provider(&providers) {
        Some(provider) => {
            println!(
                "\nvc-rs priority would select: {} ({})",
                provider.label(),
                provider.vc_provider_name()
            );
        }
        None => {
            println!("\nNo vc-rs-supported Windows ML catalog EP is compatible on this device.");
            println!("The Windows ML provider can still use DirectML/CPU fallback.");
        }
    }
    Ok(())
}

fn install(args: WindowsMlEpsInstallArgs) -> Result<()> {
    let providers = windows_ml::list_catalog_providers()?;
    let selected = match args.provider {
        Some(provider) => provider.into_catalog_provider(),
        None => windows_ml::select_best_catalog_provider(&providers).with_context(|| {
            "no vc-rs-supported Windows ML catalog EP is compatible on this device; use provider windowsml for DirectML/CPU fallback"
        })?,
    };
    let current = providers
        .iter()
        .find(|provider| provider.vc_provider == Some(selected));

    println!(
        "Selected Windows ML catalog EP: {} ({})",
        selected.label(),
        selected.vc_provider_name()
    );
    if let Some(provider) = current {
        println!("Current state: {}", provider.ready_state.label());
        if !provider.version.is_empty() {
            println!("Version: {}", provider.version);
        }
        if !provider.library_path.is_empty() {
            println!("Library: {}", provider.library_path);
        }
        if provider.ready_state == CatalogReadyState::Ready {
            println!("Already ready. No install action needed.");
            return Ok(());
        }
    } else {
        println!("Current state: not listed as compatible by Windows ML catalog.");
    }

    // Download/install can take minutes and may use Windows Update/Store-backed
    // services. Keep it behind an explicit CLI command plus confirmation; VST3
    // model load should only report the missing EP and point users here.
    if !args.yes {
        let action = match current.map(|provider| provider.ready_state) {
            Some(CatalogReadyState::NotPresent) => "download and install",
            Some(CatalogReadyState::NotReady) => "prepare",
            Some(CatalogReadyState::Unknown(_)) | None => "attempt to prepare",
            Some(CatalogReadyState::Ready) => unreachable!("ready returned above"),
        };
        if !confirm(&format!(
            "Proceed to {action} {}? This may take several minutes.",
            selected.label()
        ))? {
            bail!("cancelled");
        }
    }

    let installed = windows_ml::ensure_catalog_provider_ready(selected)?;
    println!("Result state: {}", installed.ready_state.label());
    if !installed.library_path.is_empty() {
        println!("Library: {}", installed.library_path);
    }
    println!("Use with: --provider {}", selected.vc_provider_name());
    Ok(())
}

fn print_provider(provider: &CatalogProviderInfo) {
    let vc_provider = provider
        .vc_provider
        .map(CatalogExecutionProvider::vc_provider_name)
        .unwrap_or("-");
    let status = match provider.ready_state {
        CatalogReadyState::Ready | CatalogReadyState::NotReady => "installed",
        CatalogReadyState::NotPresent => "not-installed",
        CatalogReadyState::Unknown(_) => "unknown",
    };
    println!(
        "  {} status={} vc-provider={}",
        provider.name, status, vc_provider
    );
    if !provider.version.is_empty() {
        println!("    version={}", provider.version);
    }
    if !provider.package_family_name.is_empty() {
        println!("    package={}", provider.package_family_name);
    }
    if !provider.library_path.is_empty() {
        println!("    library={}", provider.library_path);
    }
}

fn confirm(prompt: &str) -> Result<bool> {
    print!("{prompt} [y/N]: ");
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

impl WindowsMlEpProvider {
    fn into_catalog_provider(self) -> CatalogExecutionProvider {
        match self {
            Self::Nvtrtx => CatalogExecutionProvider::NvTensorRtRtx,
            Self::Qnn => CatalogExecutionProvider::Qnn,
            Self::Openvino => CatalogExecutionProvider::OpenVino,
            Self::Migraphx => CatalogExecutionProvider::MiGraphX,
            Self::Vitisai => CatalogExecutionProvider::VitisAi,
        }
    }
}
