//! Acquisition only: never called from an audio callback or an inference chunk.
//! Catalog hashes identify exported bytes, not compatibility with a voice model's
//! training. Selection of matching weights/layers/projection/preprocessing remains
//! explicit. Catalog URLs pin published release assets and their exact bytes.
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Deserialize)]
pub struct FileSpec {
    pub name: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Output {
    pub name: String,
    pub layer: u32,
    pub projection: bool,
    pub channels: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SourceWeight {
    pub name: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Input {
    pub sample_rate: u32,
    pub dtype: String,
    pub shape: Vec<serde_json::Value>,
    pub normalization: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Model {
    pub id: String,
    pub version: String,
    pub label: String,
    pub license: String,
    pub url: Option<String>,
    pub archive: FileSpec,
    pub members: Vec<FileSpec>,
    pub source_revision: String,
    pub source_repository: String,
    pub source_revision_scope: String,
    pub application_repository: String,
    pub application_revision: String,
    pub source_weights: Vec<SourceWeight>,
    pub profile: String,
    pub outputs: Vec<Output>,
    pub input: Input,
}

/// Persist alongside the existing ONNX path. Never replaces a custom path merely
/// because an ID, a channel count, or a newer catalog entry happens to match.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Selection {
    pub id: String,
    pub version: String,
    pub onnx_sha256: String,
}

impl Model {
    pub fn onnx(&self) -> &FileSpec {
        self.members
            .iter()
            .find(|f| f.name == "model.onnx")
            .expect("catalog ONNX")
    }

    pub fn selection(&self) -> Selection {
        Selection {
            id: self.id.clone(),
            version: self.version.clone(),
            onnx_sha256: self.onnx().sha256.clone(),
        }
    }

    pub fn directory(&self, root: &Path) -> PathBuf {
        root.join("embedders")
            .join(&self.version)
            .join(&self.id)
            .join(&self.onnx().sha256)
    }

    pub fn model_path(&self, root: &Path) -> PathBuf {
        self.directory(root).join("model.onnx")
    }
}

pub fn catalog() -> &'static [Model] {
    static MODELS: OnceLock<Vec<Model>> = OnceLock::new();
    MODELS.get_or_init(|| {
        serde_json::from_str(include_str!("catalog.json")).expect("compiled embedder catalog")
    })
}

pub fn find(id: &str, version: &str) -> Result<&'static Model> {
    catalog()
        .iter()
        .find(|m| m.id == id && m.version == version)
        .with_context(|| format!("Unknown embedder package: {id} {version}"))
}

pub fn selected(selection: &Selection) -> Option<&'static Model> {
    find(&selection.id, &selection.version)
        .ok()
        .filter(|m| m.onnx().sha256 == selection.onnx_sha256)
}

pub fn for_path(root: &Path, path: &Path) -> Option<&'static Model> {
    catalog().iter().find(|m| m.model_path(root) == path)
}

pub fn cache_dir() -> Result<PathBuf> {
    std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("APPDATA"))
        .map(|p| PathBuf::from(p).join("vc-rs").join("models"))
        .context("Cannot locate the user model directory (LOCALAPPDATA / APPDATA).")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Waiting,
    Checking,
    Acquiring,
    Extracting,
}

#[derive(Clone, Copy, Debug)]
pub struct Progress {
    pub stage: Stage,
    pub bytes: u64,
    pub total: u64,
}

#[derive(Clone, Debug)]
pub struct Installed {
    pub path: PathBuf,
    pub reused: bool,
}

fn cancelled(cancel: &AtomicBool) -> Result<()> {
    ensure!(
        !cancel.load(Ordering::Relaxed),
        "Acquisition cancelled. You can retry."
    );
    Ok(())
}

fn plain_file(path: &Path) -> Result<File> {
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_file(),
        "Expected a regular file: {}",
        path.display()
    );
    Ok(File::open(path)?)
}

fn verify(path: &Path, spec: &FileSpec, cancel: &AtomicBool) -> Result<()> {
    let mut file = plain_file(path)?;
    ensure!(
        file.metadata()?.len() == spec.size,
        "Unexpected size: {}",
        spec.name
    );
    copy_checked(&mut file, &mut std::io::sink(), spec, cancel, |_| {})
}

/// All four pinned members must be present; existence/size alone is not readiness.
pub fn verify_cached(model: &Model, root: &Path, cancel: &AtomicBool) -> Result<()> {
    validate_contract(model)?;
    cancelled(cancel)?;
    let dir = model.directory(root);
    ensure!(
        fs::symlink_metadata(&dir)?.file_type().is_dir(),
        "Invalid cache directory"
    );
    ensure!(
        fs::read_dir(&dir)?.count() == model.members.len(),
        "Unexpected cache members"
    );
    for spec in &model.members {
        verify(&dir.join(&spec.name), spec, cancel)?;
    }
    Ok(())
}

fn validate_contract(model: &Model) -> Result<()> {
    let token = |s: &str| {
        !s.is_empty()
            && s.bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'.')
            && s != "."
            && s != ".."
    };
    let hash = |s: &str| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit());
    ensure!(
        token(&model.id) && token(&model.version),
        "Invalid package identity"
    );
    ensure!(
        hash(&model.archive.sha256) && model.archive.size <= 1024 * 1024 * 1024,
        "Invalid archive contract"
    );
    ensure!(model.members.len() == 4, "Invalid member contract");
    for name in ["model.onnx", "README.md", "README.ja.md", "LICENSE.txt"] {
        let matching: Vec<_> = model.members.iter().filter(|f| f.name == name).collect();
        ensure!(matching.len() == 1, "Invalid member contract");
        let spec = matching[0];
        let limit = if name == "model.onnx" {
            1024 * 1024 * 1024
        } else {
            1024 * 1024
        };
        ensure!(
            spec.size > 0 && spec.size <= limit && hash(&spec.sha256),
            "Invalid member size or hash"
        );
    }
    Ok(())
}

fn copy_checked(
    reader: &mut impl Read,
    writer: &mut impl Write,
    spec: &FileSpec,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64),
) -> Result<()> {
    let mut bytes = 0u64;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        cancelled(cancel)?;
        let n = match reader.read(&mut buffer) {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if n == 0 {
            break;
        }
        bytes = bytes
            .checked_add(n as u64)
            .context("Acquisition size overflow")?;
        ensure!(bytes <= spec.size, "Unexpected size: {}", spec.name);
        writer.write_all(&buffer[..n])?;
        hash.update(&buffer[..n]);
        progress(bytes);
    }
    ensure!(
        bytes == spec.size && format!("{:x}", hash.finalize()) == spec.sha256,
        "Checksum verification failed: {}",
        spec.name
    );
    cancelled(cancel)
}

fn unique_path(parent: &Path, prefix: &str) -> PathBuf {
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    parent.join(format!(
        ".{prefix}-{}-{time}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ))
}

struct Staging(PathBuf);
impl Drop for Staging {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn create_plain_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_dir(),
        "Invalid cache directory"
    );
    Ok(())
}

fn extract(
    model: &Model,
    archive: &Path,
    directory: &Path,
    cancel: &AtomicBool,
    progress: &mut impl FnMut(Progress),
) -> Result<()> {
    let mut zip = zip::ZipArchive::new(plain_file(archive)?)?;
    ensure!(
        zip.len() == model.members.len(),
        "Unexpected ZIP member count"
    );
    let mut seen = std::collections::HashSet::new();
    let total: u64 = model.members.iter().map(|f| f.size).sum();
    let mut completed = 0;
    for index in 0..zip.len() {
        cancelled(cancel)?;
        let mut entry = zip.by_index(index)?;
        let spec = model
            .members
            .iter()
            .find(|f| f.name.as_bytes() == entry.name_raw())
            .context("Unexpected ZIP member")?;
        ensure!(
            entry.name_raw() == spec.name.as_bytes() && seen.insert(spec.name.as_str()),
            "Duplicate or invalid ZIP member"
        );
        ensure!(
            entry.is_file()
                && !entry.is_symlink()
                && entry
                    .unix_mode()
                    .is_none_or(|mode| mode & 0o170000 == 0 || mode & 0o170000 == 0o100000),
            "ZIP links and special files are not allowed"
        );
        ensure!(
            entry.size() == spec.size,
            "Unexpected ZIP expanded size: {}",
            spec.name
        );
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(directory.join(&spec.name))
            .context("Create extracted member")?;
        copy_checked(&mut entry, &mut file, spec, cancel, |bytes| {
            progress(Progress {
                stage: Stage::Extracting,
                bytes: completed + bytes,
                total,
            })
        })?;
        file.sync_all().context("Flush extracted member")?;
        completed += spec.size;
    }
    Ok(())
}

/// The source is opened lazily, after verified-cache reuse and the process lock.
/// Local ZIPs and HTTPS bodies use exactly the same bounded staging/verification.
/// Only the complete directory is published. Never write into a custom selection.
pub fn acquire<R: Read>(
    model: &Model,
    root: &Path,
    cancel: &AtomicBool,
    mut progress: impl FnMut(Progress),
    open: impl FnOnce() -> Result<R>,
) -> Result<Installed> {
    validate_contract(model)?;
    cancelled(cancel)?;
    progress(Progress {
        stage: Stage::Checking,
        bytes: 0,
        total: 0,
    });
    if verify_cached(model, root, cancel).is_ok() {
        return Ok(Installed {
            path: model.model_path(root),
            reused: true,
        });
    }
    let parent = model
        .directory(root)
        .parent()
        .context("Invalid cache path")?
        .to_path_buf();
    create_plain_dir(root)?;
    let mut part = root.to_path_buf();
    for name in ["embedders", &model.version, &model.id] {
        part.push(name);
        create_plain_dir(&part)?;
    }
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(parent.join(".install.lock"))
        .context("Open package lock")?;
    loop {
        cancelled(cancel)?;
        match lock.try_lock() {
            Ok(()) => break,
            Err(std::fs::TryLockError::WouldBlock) => {
                progress(Progress {
                    stage: Stage::Waiting,
                    bytes: 0,
                    total: 0,
                });
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(error) => return Err(anyhow::Error::from(error).context("Lock package cache")),
        }
    }
    if verify_cached(model, root, cancel).is_ok() {
        return Ok(Installed {
            path: model.model_path(root),
            reused: true,
        });
    }
    cancelled(cancel)?;
    let staging = Staging(unique_path(&parent, "install"));
    fs::create_dir(&staging.0).context("Create package staging directory")?;
    let archive = staging.0.join("package.zip");
    let unpacked = staging.0.join("unpacked");
    fs::create_dir(&unpacked)?;
    let mut reader = open()?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&archive)?;
    copy_checked(&mut reader, &mut file, &model.archive, cancel, |bytes| {
        progress(Progress {
            stage: Stage::Acquiring,
            bytes,
            total: model.archive.size,
        })
    })?;
    file.sync_all()?;
    drop(file);
    extract(model, &archive, &unpacked, cancel, &mut progress).context("Extract package")?;
    cancelled(cancel)?;
    let target = model.directory(root);
    // A corrupt cache is replaced only after the new package has passed every
    // check. Keep the old directory for rollback if publication itself fails.
    let backup = unique_path(&parent, "previous");
    let existed = target.try_exists()?;
    if existed {
        ensure!(
            fs::symlink_metadata(&target)?.file_type().is_dir(),
            "Invalid cache directory"
        );
        fs::rename(&target, &backup)?;
    }
    if let Err(error) = fs::rename(&unpacked, &target) {
        if existed {
            let _ = fs::rename(&backup, &target);
        }
        return Err(anyhow::Error::from(error).context("Publish verified package directory"));
    }
    if existed {
        let _ = fs::remove_dir_all(backup);
    }
    Ok(Installed {
        path: model.model_path(root),
        reused: false,
    })
}

pub fn import(
    model: &Model,
    source: &Path,
    root: &Path,
    cancel: &AtomicBool,
    progress: impl FnMut(Progress),
) -> Result<Installed> {
    acquire(model, root, cancel, progress, || plain_file(source))
}

pub fn download(
    model: &Model,
    root: &Path,
    cancel: &AtomicBool,
    progress: impl FnMut(Progress),
) -> Result<Installed> {
    acquire(model, root, cancel, progress, || {
        let Some(url) = model.url.as_deref() else {
            bail!("No published download source is configured. Import the matching local ZIP instead.");
        };
        ensure!(url.starts_with("https://"), "Model downloads require HTTPS");
        let agent = ureq::Agent::config_builder()
            .https_only(true)
            .timeout_global(Some(Duration::from_secs(3600)))
            .timeout_resolve(Some(Duration::from_secs(30)))
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_send_request(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(30)))
            .tls_config(
                ureq::tls::TlsConfig::builder()
                    .provider(ureq::tls::TlsProvider::NativeTls)
                    .build(),
            )
            .build()
            .new_agent();
        Ok(agent.get(url).call()?.into_body().into_reader())
    })
}

#[cfg(test)]
mod tests;
