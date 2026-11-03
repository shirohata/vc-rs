//! First-run model acquisition, isolated from the engine and audio callbacks.
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

pub struct Model {
    pub name: &'static str,
    pub file: &'static str,
    remote: &'static str,
    pub size: u64,
    hash: &'static str,
}

// Pinned upstream revision and LFS SHA-256: update these together. Models are
// downloaded directly, never bundled or treated as MIT-licensed application code.
const REVISION: &str = "d1b578cfcdf944dd3910786f7604754c79234b36";
pub const GTCRN_INDEX: usize = 2;
pub const MODELS: [Model; 3] = [
    Model {
        name: "ContentVec",
        file: "content_vec_500.onnx",
        remote: "content-vec/contentvec-f.onnx",
        size: 378550151,
        hash: "4b31ed3d95a568fab7952de923ff7f7d3d17128ea6fce69f665509d24c3156db",
    },
    Model {
        name: "RMVPE",
        file: "rmvpe.onnx",
        remote: "rmvpe/rmvpe_20231006.onnx",
        size: 362003174,
        hash: "84f0586308e36157f75b77c8591bf636d6719c0c4ba95f8faf3df479e7566219",
    },
    Model {
        name: "GTCRN",
        file: "gtcrn/gtcrn_stream.onnx",
        remote: "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/stream/onnx_models/gtcrn.onnx",
        size: 352084,
        hash: "f648b02f2d7ff96ebcb0eec2219688a08ed12fe7e3d50f248605a90eba8cad17",
    },
];

pub fn gtcrn_available(dir: &str) -> bool {
    !dir.trim().is_empty()
        && ["gtcrn_stream.onnx", "gtcrn.onnx"]
            .iter()
            .any(|file| available(&Path::new(dir.trim()).join(file).to_string_lossy()))
}

pub fn discover_gtcrn(dir: &mut String, roots: &[PathBuf]) -> bool {
    // Match the engine's two accepted filenames, and preserve custom directories.
    if gtcrn_available(dir) {
        return false;
    }
    for root in roots {
        let candidate = root.join("gtcrn").to_string_lossy().into_owned();
        if gtcrn_available(&candidate) {
            *dir = candidate;
            return true;
        }
    }
    false
}

pub fn cache_dir() -> Result<PathBuf, String> {
    vc_models::cache_dir().map_err(|e| e.to_string())
}

pub fn available(value: &str) -> bool {
    !value.trim().is_empty() && fs::metadata(value.trim()).is_ok_and(|m| m.is_file() && m.len() > 0)
}

/// Validate the exact downloaded bytes on a background worker. Custom models
/// need structural validation instead of a hash belonging to a different file.
pub fn validate_support(path: &Path, index: usize) -> Result<(), String> {
    if index == 0 {
        if let Ok(root) = cache_dir() {
            if let Some(model) = vc_models::for_path(&root, path) {
                return vc_models::verify_cached(model, &root, &AtomicBool::new(false))
                    .map_err(|e| format!("{e:#}"));
            }
        }
    }
    if cache_dir().is_ok_and(|dir| path == dir.join(MODELS[index].file)) {
        verify_file(path, &MODELS[index])
    } else {
        vc_core::model_rvc::validate_onnx_model(path).map_err(|e| format!("{e:#}"))
    }
}

fn verify_file(path: &Path, model: &Model) -> Result<(), String> {
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() != model.size {
        return Err("Model verification failed. Please retry the download.".into());
    }
    let mut hash = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    if format!("{:x}", hash.finalize()) != model.hash {
        return Err("Model verification failed. Please retry the download.".into());
    }
    Ok(())
}

/// Preserve custom selections; also discover models fetched by the bundled script.
pub fn discover(value: &mut String, file: &str, roots: &[PathBuf]) -> bool {
    if available(value) {
        return false;
    }
    for root in roots {
        let path = root.join(file);
        if available(&path.to_string_lossy()) {
            *value = path.to_string_lossy().into_owned();
            return true;
        }
    }
    false
}

#[derive(Clone)]
pub enum State {
    Running {
        name: &'static str,
        bytes: u64,
        total: u64,
    },
    Done,
    Failed(String),
}

pub struct Download {
    pub state: Arc<Mutex<State>>,
    cancel: Arc<AtomicBool>,
}

impl Download {
    pub fn start(dir: PathBuf, indices: Vec<usize>) -> Self {
        let state = Arc::new(Mutex::new(State::Running {
            name: "Preparing",
            bytes: 0,
            total: 0,
        }));
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_state = state.clone();
        let worker_cancel = cancel.clone();
        let result = std::thread::Builder::new()
            .name("vc-model-download".into())
            .spawn(move || {
                let result = (|| {
                    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                    let agent = ureq::Agent::config_builder()
                        .https_only(true)
                        .timeout_global(Some(Duration::from_secs(3600)))
                        .timeout_resolve(Some(Duration::from_secs(30)))
                        .timeout_connect(Some(Duration::from_secs(30)))
                        .timeout_send_request(Some(Duration::from_secs(30)))
                        .timeout_recv_response(Some(Duration::from_secs(30)))
                        // recv_body is a total transfer deadline, not an idle
                        // timeout: large models must also work on slow links.
                        .tls_config(
                            ureq::tls::TlsConfig::builder()
                                .provider(ureq::tls::TlsProvider::NativeTls)
                                .build(),
                        )
                        .build()
                        .new_agent();
                    for index in indices {
                        let model = &MODELS[index];
                        if worker_cancel.load(Ordering::Relaxed) {
                            return Err("Download cancelled. You can retry.".into());
                        }
                        *worker_state.lock().unwrap() = State::Running {
                            name: model.name,
                            bytes: 0,
                            total: model.size,
                        };
                        let url = if model.remote.starts_with("https://") {
                            model.remote.to_string()
                        } else {
                            format!(
                                "https://huggingface.co/wok000/weights_gpl/resolve/{REVISION}/{}",
                                model.remote
                            )
                        };
                        let target = dir.join(model.file);
                        if verify_file(&target, model).is_ok() {
                            continue;
                        }
                        fs::create_dir_all(target.parent().unwrap()).map_err(|e| e.to_string())?;
                        if index == GTCRN_INDEX {
                            // Preserve the upstream MIT notice alongside each downloaded copy.
                            fs::write(
                                dir.join("gtcrn/LICENSE.txt"),
                                include_str!("gtcrn-license.txt"),
                            )
                            .map_err(|e| e.to_string())?;
                        }
                        let mut response = agent
                            .get(&url)
                            .call()
                            .map_err(|e| format!("{}: {e}", model.name))?;
                        install(
                            &mut response.body_mut().as_reader(),
                            &target,
                            model,
                            &worker_cancel,
                            |bytes| {
                                *worker_state.lock().unwrap() = State::Running {
                                    name: model.name,
                                    bytes,
                                    total: model.size,
                                };
                            },
                        )
                        .map_err(|e| format!("{}: {e}", model.name))?;
                    }
                    Ok::<_, String>(())
                })();
                *worker_state.lock().unwrap() = match result {
                    Ok(()) => State::Done,
                    Err(e) => State::Failed(e),
                };
            });
        if let Err(e) = result {
            *state.lock().unwrap() = State::Failed(e.to_string());
        }
        Self { state, cancel }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Drop for Download {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn install(
    reader: &mut impl Read,
    target: &Path,
    model: &Model,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64),
) -> Result<(), String> {
    // Unique, exclusively created staging files isolate concurrent app instances.
    // Only a complete hash-verified model may acquire the final filename.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let temporary = target.with_extension(format!("{}.{nonce}.download", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| e.to_string())?;
    let result = (|| {
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        let mut bytes = 0;
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err("Download cancelled. You can retry.".into());
            }
            let read = reader.read(&mut buffer).map_err(|e| e.to_string())?;
            if read == 0 {
                break;
            }
            bytes += read as u64;
            if bytes > model.size {
                return Err("Unexpected model size.".into());
            }
            file.write_all(&buffer[..read]).map_err(|e| e.to_string())?;
            hash.update(&buffer[..read]);
            progress(bytes);
        }
        if bytes != model.size || format!("{:x}", hash.finalize()) != model.hash {
            return Err("Model verification failed. Please retry the download.".into());
        }
        file.sync_all().map_err(|e| e.to_string())?;
        Ok(())
    })();
    drop(file);
    let result = result.and_then(|()| fs::rename(&temporary, target).map_err(|e| e.to_string()));
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "Downloads 741 MB from upstream; run explicitly when validating model acquisition"]
    fn download_reference_models() {
        download_models_for_test(vec![0, 1]);
    }

    #[test]
    #[ignore = "Downloads the 352 KB GTCRN model from upstream"]
    fn download_gtcrn_model() {
        download_models_for_test(vec![GTCRN_INDEX]);
    }

    fn download_models_for_test(indices: Vec<usize>) {
        let dir = std::env::temp_dir().join(format!(
            "vc-model-network-test-{}-{}",
            std::process::id(),
            indices[0]
        ));
        let download = Download::start(dir.clone(), indices.clone());
        loop {
            let state = download.state.lock().unwrap().clone();
            match state {
                State::Done => break,
                State::Failed(error) => panic!("{error}"),
                State::Running { .. } => std::thread::sleep(Duration::from_millis(100)),
            }
        }
        for index in &indices {
            let model = &MODELS[*index];
            assert_eq!(
                fs::metadata(dir.join(model.file)).unwrap().len(),
                model.size
            );
            fs::remove_file(dir.join(model.file)).unwrap();
        }
        if indices.contains(&GTCRN_INDEX) {
            assert_eq!(
                fs::read_to_string(dir.join("gtcrn/LICENSE.txt")).unwrap(),
                include_str!("gtcrn-license.txt")
            );
            fs::remove_file(dir.join("gtcrn/LICENSE.txt")).unwrap();
            fs::remove_dir(dir.join("gtcrn")).unwrap();
        }
        fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn discover_gtcrn_preserves_custom_models_and_accepts_upstream_filename() {
        let dir = std::env::temp_dir().join(format!("vc-gtcrn-discovery-{}", std::process::id()));
        fs::create_dir_all(dir.join("gtcrn")).unwrap();
        let mut selected = String::new();
        assert!(!discover_gtcrn(&mut selected, std::slice::from_ref(&dir)));
        fs::write(dir.join("gtcrn/gtcrn.onnx"), b"model").unwrap();
        assert!(discover_gtcrn(&mut selected, std::slice::from_ref(&dir)));
        assert!(gtcrn_available(&selected));
        assert!(!discover_gtcrn(&mut selected, &[]));
        fs::remove_file(dir.join("gtcrn/gtcrn.onnx")).unwrap();
        fs::write(dir.join("gtcrn/gtcrn_stream.onnx"), b"model").unwrap();
        assert!(gtcrn_available(&selected));
        fs::remove_file(dir.join("gtcrn/gtcrn_stream.onnx")).unwrap();
        fs::remove_dir(dir.join("gtcrn")).unwrap();
        fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn install_rejects_partial_corrupt_and_cancelled_downloads() {
        let dir = std::env::temp_dir().join(format!("vc-model-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("model.onnx");
        let model = Model {
            name: "test",
            file: "model.onnx",
            remote: "",
            size: 3,
            hash: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        };
        let cancel = AtomicBool::new(false);
        for data in [b"ab".as_slice(), b"bad", b"abcd"] {
            assert!(install(&mut &data[..], &target, &model, &cancel, |_| {}).is_err());
            assert!(!target.exists());
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        }
        cancel.store(true, Ordering::Relaxed);
        assert!(install(&mut &b"abc"[..], &target, &model, &cancel, |_| {}).is_err());
        cancel.store(false, Ordering::Relaxed);
        install(&mut &b"abc"[..], &target, &model, &cancel, |_| {}).unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"abc");
        assert!(verify_file(&target, &model).is_ok());
        fs::write(&target, b"bad").unwrap();
        assert!(verify_file(&target, &model).is_err());
        install(&mut &b"abc"[..], &target, &model, &cancel, |_| {}).unwrap();
        assert!(install(&mut &b"bad"[..], &target, &model, &cancel, |_| {}).is_err());
        assert_eq!(fs::read(&target).unwrap(), b"abc");
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        let mut selection = target.to_string_lossy().into_owned();
        assert!(!discover(
            &mut selection,
            "missing.onnx",
            std::slice::from_ref(&dir)
        ));
        let mut empty = String::new();
        assert!(discover(
            &mut empty,
            "model.onnx",
            std::slice::from_ref(&dir)
        ));
        assert_eq!(empty, selection);
        let mut broken = dir.join("missing.onnx").to_string_lossy().into_owned();
        assert!(discover(
            &mut broken,
            "model.onnx",
            std::slice::from_ref(&dir)
        ));
        assert_eq!(broken, selection);
        fs::remove_dir_all(dir).unwrap();
    }
}
