use std::io::{self, Write};

use anyhow::{bail, Result};
use vc_core::model_rvc::{self, EngineCacheInfo};

use crate::cli::{EngineCacheArgs, EngineCacheClearArgs, EngineCacheCommand};

pub fn run(args: EngineCacheArgs) -> Result<()> {
    match args.command {
        EngineCacheCommand::Info => info(),
        EngineCacheCommand::Clear(args) => clear(args),
    }
}

fn info() -> Result<()> {
    let info = model_rvc::engine_cache_info()?;
    print_info(&info);
    Ok(())
}

fn clear(args: EngineCacheClearArgs) -> Result<()> {
    let info = model_rvc::engine_cache_info()?;
    if !info.exists || info.file_count == 0 {
        println!("Engine cache is already empty: {}", info.root.display());
        return Ok(());
    }

    print_info(&info);
    if !args.yes
        && !confirm(&format!(
            "Delete all {} cached engine file(s) ({})?",
            info.file_count,
            format_bytes(info.size_bytes)
        ))?
    {
        bail!("cancelled");
    }

    let cleared = model_rvc::clear_engine_cache()?;
    println!(
        "Cleared {} ({} file(s)) from {}",
        format_bytes(cleared.size_bytes),
        cleared.file_count,
        cleared.root.display()
    );
    Ok(())
}

fn print_info(info: &EngineCacheInfo) {
    println!("Engine cache: {}", info.root.display());
    println!("Override with {}=<dir>", model_rvc::ENGINE_CACHE_DIR_ENV);
    if !info.exists {
        println!("Status: not created yet (nothing cached)");
        return;
    }
    println!(
        "Total: {} across {} file(s)",
        format_bytes(info.size_bytes),
        info.file_count
    );
    if info.entries.is_empty() {
        return;
    }
    println!("Entries:");
    for entry in &info.entries {
        let kind = if entry.is_dir { "model" } else { "file" };
        println!(
            "  {:>10}  {} ({}, {} file(s))",
            format_bytes(entry.size_bytes),
            entry.name,
            kind,
            entry.file_count
        );
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

/// Render a byte count as a short binary-prefixed string (e.g. `12.3 MiB`).
fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

#[cfg(test)]
mod tests {
    use super::format_bytes;

    #[test]
    fn formats_byte_scales() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KiB");
        assert_eq!(format_bytes(1536), "1.5 KiB");
        assert_eq!(format_bytes(5 * 1024 * 1024), "5.0 MiB");
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3.0 GiB");
    }
}
