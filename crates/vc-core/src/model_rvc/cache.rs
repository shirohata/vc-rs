//! Inspection and cleanup for the on-disk engine cache.
//!
//! Both inference backends that build serialized GPU engines persist them under
//! one root (`super::tensorrt::tensor_rt_cache_root`): the native TensorRT path
//! writes `native.engine` files plus a shared `timing.cache`, and the Windows ML
//! `NvTensorRtRtx` (TensorRT-RTX) EP writes its runtime cache into the same
//! per-model/per-shape sub-directories (see `sessions.rs`). These engines are
//! rebuildable derived data — safe to delete; the next session just re-builds
//! the missing ones — so this module only needs to size the tree and remove it.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::tensorrt::{tensor_rt_cache_root, TENSORRT_CACHE_DIR_ENV};

/// Environment variable that overrides the engine cache location.
pub const ENGINE_CACHE_DIR_ENV: &str = TENSORRT_CACHE_DIR_ENV;

/// One top-level entry under the cache root: either a per-model directory
/// (holding that model's built engines across shapes/backends) or a loose file
/// such as the shared `timing.cache`.
#[derive(Debug, Clone)]
pub struct EngineCacheEntry {
    /// File or directory name directly under the cache root.
    pub name: String,
    /// Full path to the entry.
    pub path: PathBuf,
    /// `true` when the entry is a directory.
    pub is_dir: bool,
    /// Total bytes contained (recursive for directories).
    pub size_bytes: u64,
    /// Number of regular files contained (1 for a loose file).
    pub file_count: u64,
}

/// Summary of the engine cache on disk.
#[derive(Debug, Clone)]
pub struct EngineCacheInfo {
    /// Root directory holding all cached engines.
    pub root: PathBuf,
    /// `false` when the root has not been created yet (nothing cached).
    pub exists: bool,
    /// Total bytes across the whole cache tree.
    pub size_bytes: u64,
    /// Total regular files across the whole cache tree.
    pub file_count: u64,
    /// Top-level entries (per-model dirs and loose files), largest first.
    pub entries: Vec<EngineCacheEntry>,
}

/// Result of clearing the engine cache.
#[derive(Debug, Clone)]
pub struct ClearedEngineCache {
    /// Root directory that was cleared.
    pub root: PathBuf,
    /// `false` when there was nothing to clear (root did not exist).
    pub existed: bool,
    /// Bytes freed.
    pub size_bytes: u64,
    /// Regular files removed.
    pub file_count: u64,
}

/// Resolve the engine cache root (honoring `ENGINE_CACHE_DIR_ENV`).
///
/// The path is returned whether or not it exists yet; callers that need to know
/// use [`EngineCacheInfo::exists`].
pub fn engine_cache_root() -> Result<PathBuf> {
    tensor_rt_cache_root()
}

/// Walk the engine cache root and report its location, total size, and a
/// per-top-level-entry breakdown.
pub fn engine_cache_info() -> Result<EngineCacheInfo> {
    let root = engine_cache_root()?;
    if !root.exists() {
        return Ok(EngineCacheInfo {
            root,
            exists: false,
            size_bytes: 0,
            file_count: 0,
            entries: Vec::new(),
        });
    }

    let mut entries = Vec::new();
    let mut total_bytes = 0u64;
    let mut total_files = 0u64;

    let read_dir = fs::read_dir(&root)
        .with_context(|| format!("failed to read engine cache directory {}", root.display()))?;
    for entry in read_dir {
        let entry = entry.with_context(|| format!("failed to read entry in {}", root.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to stat {}", path.display()))?;
        let (size_bytes, file_count) = if file_type.is_dir() {
            dir_size(&path)?
        } else {
            (file_len(&path)?, 1)
        };
        total_bytes += size_bytes;
        total_files += file_count;
        entries.push(EngineCacheEntry {
            name: entry.file_name().to_string_lossy().into_owned(),
            path,
            is_dir: file_type.is_dir(),
            size_bytes,
            file_count,
        });
    }

    // Largest first so the size view leads with whatever is worth clearing.
    entries.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes).then(a.name.cmp(&b.name)));

    Ok(EngineCacheInfo {
        root,
        exists: true,
        size_bytes: total_bytes,
        file_count: total_files,
        entries,
    })
}

/// Delete the entire engine cache. Returns what was freed.
///
/// Engines are rebuildable derived data, so removing them only costs a one-time
/// rebuild on the next model load. The root directory itself is removed; it is
/// recreated lazily by the backends on the next build.
pub fn clear_engine_cache() -> Result<ClearedEngineCache> {
    let info = engine_cache_info()?;
    if !info.exists {
        return Ok(ClearedEngineCache {
            root: info.root,
            existed: false,
            size_bytes: 0,
            file_count: 0,
        });
    }

    fs::remove_dir_all(&info.root).with_context(|| {
        format!(
            "failed to remove engine cache directory {}",
            info.root.display()
        )
    })?;

    Ok(ClearedEngineCache {
        root: info.root,
        existed: true,
        size_bytes: info.size_bytes,
        file_count: info.file_count,
    })
}

fn file_len(path: &Path) -> Result<u64> {
    let metadata =
        fs::metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    Ok(metadata.len())
}

/// Recursively total the byte size and file count under `dir`.
fn dir_size(dir: &Path) -> Result<(u64, u64)> {
    let mut total_bytes = 0u64;
    let mut file_count = 0u64;
    let read_dir =
        fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?;
    for entry in read_dir {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to stat {}", entry.path().display()))?;
        if file_type.is_dir() {
            let (bytes, files) = dir_size(&entry.path())?;
            total_bytes += bytes;
            file_count += files;
        } else {
            total_bytes += file_len(&entry.path())?;
            file_count += 1;
        }
    }
    Ok((total_bytes, file_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dir_size_totals_nested_files() {
        let temp = std::env::temp_dir().join(format!(
            "vc-rs-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let nested = temp.join("model_abc").join("shapes");
        fs::create_dir_all(&nested).unwrap();
        fs::write(temp.join("timing.cache"), vec![0u8; 10]).unwrap();
        fs::write(nested.join("native.engine"), vec![0u8; 100]).unwrap();

        let (bytes, files) = dir_size(&temp).unwrap();
        assert_eq!(bytes, 110);
        assert_eq!(files, 2);

        fs::remove_dir_all(&temp).unwrap();
    }
}
