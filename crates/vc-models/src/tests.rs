use super::*;
use std::io::Cursor;

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = unique_path(&std::env::temp_dir(), "vc-models-test");
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn fixture() -> (Model, Vec<u8>) {
    let mut model = catalog()[0].clone();
    model.id = "fixture".into();
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for spec in &mut model.members {
        let data = spec.name.as_bytes();
        spec.size = data.len() as u64;
        spec.sha256 = digest(data);
        writer
            .start_file(
                &spec.name,
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated),
            )
            .unwrap();
        writer.write_all(data).unwrap();
    }
    let bytes = writer.finish().unwrap().into_inner();
    pin_zip(&mut model, &bytes);
    (model, bytes)
}

fn pin_zip(model: &mut Model, bytes: &[u8]) {
    model.archive.size = bytes.len() as u64;
    model.archive.sha256 = digest(bytes);
}

fn install_fixture(model: &Model, bytes: &[u8], root: &Path) -> Result<Installed> {
    acquire(
        model,
        root,
        &AtomicBool::new(false),
        |_| {},
        || Ok(Cursor::new(bytes)),
    )
}

#[test]
fn catalog_identifies_exact_profiles_and_pins_published_release_assets() {
    assert_eq!(catalog().len(), 7);
    let mut ids = std::collections::HashSet::new();
    for model in catalog() {
        validate_contract(model).unwrap();
        assert!(ids.insert(&model.id));
        assert_eq!(
            model.url.as_deref(),
            Some(
                format!(
                    "https://github.com/shirohata/rvc-embedders-onnx/releases/download/v{}/{}",
                    model.version, model.archive.name
                )
                .as_str()
            )
        );
        assert_eq!(model.input.sample_rate, 16_000);
        assert_eq!(model.input.dtype, "float32");
        assert_eq!(model.input.normalization, "none");
        assert!(model
            .outputs
            .iter()
            .any(|o| o.name == "unit12" && o.layer == 12 && !o.projection && o.channels == 768));
        assert_eq!(selected(&model.selection()).unwrap().id, model.id);
        let root = Path::new("cache");
        assert_eq!(
            for_path(root, &model.model_path(root)).unwrap().id,
            model.id
        );
    }
    let spin = find("applio-spin", "0.1.0").unwrap();
    let spin2 = find("applio-spin-v2", "0.1.0").unwrap();
    assert_ne!(
        spin.source_weights[1].sha256,
        spin2.source_weights[1].sha256
    );
    assert_eq!(spin.outputs.len(), 1);
    let mut selection = spin.selection();
    selection.onnx_sha256 = spin2.onnx().sha256.clone();
    assert!(selected(&selection).is_none());
    assert!(find("applio-spin", "latest").is_err());
}

#[test]
fn complete_install_reuses_verified_cache_without_opening_source() {
    let (model, bytes) = fixture();
    let temp = Temp::new();
    let result = install_fixture(&model, &bytes, &temp.0).unwrap();
    assert!(!result.reused);
    assert_eq!(fs::read(&result.path).unwrap(), b"model.onnx");
    assert_eq!(fs::read_dir(model.directory(&temp.0)).unwrap().count(), 4);
    let result = acquire::<Cursor<Vec<u8>>>(
        &model,
        &temp.0,
        &AtomicBool::new(false),
        |_| {},
        || panic!("cache must avoid source I/O"),
    )
    .unwrap();
    assert!(result.reused);
}

#[test]
fn failed_and_cancelled_retries_preserve_then_repair_corrupt_cache() {
    let (model, bytes) = fixture();
    let temp = Temp::new();
    let installed = install_fixture(&model, &bytes, &temp.0).unwrap();
    fs::write(&installed.path, b"broken").unwrap();
    for data in [&bytes[..bytes.len() - 1], b"corrupt ZIP".as_slice()] {
        assert!(install_fixture(&model, data, &temp.0).is_err());
        assert_eq!(fs::read(&installed.path).unwrap(), b"broken");
    }
    let cancel = AtomicBool::new(false);
    assert!(acquire(
        &model,
        &temp.0,
        &cancel,
        |p| {
            if p.stage == Stage::Acquiring {
                cancel.store(true, Ordering::Relaxed);
            }
        },
        || Ok(Cursor::new(&bytes))
    )
    .is_err());
    assert_eq!(fs::read(&installed.path).unwrap(), b"broken");
    let parent = model.directory(&temp.0).parent().unwrap().to_path_buf();
    assert!(fs::read_dir(parent).unwrap().all(|e| !e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .starts_with(".install-")));
    assert!(!install_fixture(&model, &bytes, &temp.0).unwrap().reused);
    verify_cached(&model, &temp.0, &AtomicBool::new(false)).unwrap();
}

#[test]
fn cancellation_during_extraction_never_publishes_partial_directory() {
    let (model, bytes) = fixture();
    let temp = Temp::new();
    let cancel = AtomicBool::new(false);
    assert!(acquire(
        &model,
        &temp.0,
        &cancel,
        |p| {
            if p.stage == Stage::Extracting {
                cancel.store(true, Ordering::Relaxed);
            }
        },
        || Ok(Cursor::new(&bytes))
    )
    .is_err());
    assert!(!model.directory(&temp.0).exists());
    assert!(install_fixture(&model, &bytes, &temp.0).is_ok());
}

#[test]
fn interrupted_source_is_retryable_and_never_publishes() {
    struct Broken;
    impl Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "fixture disconnect",
            ))
        }
    }
    let (model, bytes) = fixture();
    let temp = Temp::new();
    assert!(acquire(
        &model,
        &temp.0,
        &AtomicBool::new(false),
        |_| {},
        || Ok(Broken)
    )
    .is_err());
    assert!(!model.directory(&temp.0).exists());
    assert!(install_fixture(&model, &bytes, &temp.0).is_ok());
}

fn custom_zip(names: &[&str], link: bool, wrong_data: bool) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for name in names {
        if link && *name == "model.onnx" {
            writer
                .add_symlink(
                    *name,
                    "../outside",
                    zip::write::SimpleFileOptions::default(),
                )
                .unwrap();
        } else {
            writer
                .start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            writer
                .write_all(if wrong_data {
                    b"invalid"
                } else {
                    name.as_bytes()
                })
                .unwrap();
        }
    }
    writer.finish().unwrap().into_inner()
}

#[test]
fn rejects_paths_missing_extra_members_links_size_and_member_hash_mismatch() {
    let (mut model, _) = fixture();
    let temp = Temp::new();
    for names in [
        vec!["model.onnx", "README.md", "LICENSE.txt", "NOTICE.md"],
        vec!["../model.onnx", "README.md", "LICENSE.txt", "README.ja.md"],
        vec!["/model.onnx", "README.md", "LICENSE.txt", "README.ja.md"],
        vec!["C:\\model.onnx", "README.md", "LICENSE.txt", "README.ja.md"],
        vec!["sub/model.onnx", "README.md", "LICENSE.txt", "README.ja.md"],
        vec!["model.onnx", "README.md", "LICENSE.txt"],
        vec![
            "model.onnx",
            "README.md",
            "LICENSE.txt",
            "README.ja.md",
            "extra",
        ],
    ] {
        let bytes = custom_zip(&names, false, false);
        pin_zip(&mut model, &bytes);
        assert!(install_fixture(&model, &bytes, &temp.0).is_err());
        assert!(!model.directory(&temp.0).exists());
    }
    let names = ["model.onnx", "README.md", "LICENSE.txt", "README.ja.md"];
    for (link, wrong_data) in [(true, false), (false, true)] {
        let bytes = custom_zip(&names, link, wrong_data);
        pin_zip(&mut model, &bytes);
        assert!(install_fixture(&model, &bytes, &temp.0).is_err());
    }
    let bytes = custom_zip(&names, false, false);
    pin_zip(&mut model, &bytes);
    model.members[0].size += 1;
    assert!(install_fixture(&model, &bytes, &temp.0).is_err());
    model.members[0].size -= 1;
    model.members[0].sha256 = "0".repeat(64);
    assert!(install_fixture(&model, &bytes, &temp.0).is_err());
}

#[test]
fn duplicate_zip_names_are_rejected() {
    // Writer rejects duplicates itself. Mutate equal-length names in both local
    // and central records to exercise the reader rather than the fixture writer.
    let names = ["model.onnx", "README.md", "README.ja.md", "readme.md"];
    let mut bytes = custom_zip(&names, false, false);
    for offset in 0..bytes.len() - 9 {
        if &bytes[offset..offset + 9] == b"readme.md" {
            bytes[offset..offset + 9].copy_from_slice(b"README.md");
        }
    }
    let (mut model, _) = fixture();
    pin_zip(&mut model, &bytes);
    assert!(install_fixture(&model, &bytes, &Temp::new().0).is_err());
}

#[test]
fn concurrent_installers_wait_then_reuse_and_waiting_can_cancel() {
    let (model, bytes) = fixture();
    let temp = Temp::new();
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            acquire(
                &model,
                &temp.0,
                &AtomicBool::new(false),
                |_| {},
                || {
                    barrier.wait();
                    std::thread::sleep(Duration::from_millis(150));
                    Ok(Cursor::new(&bytes))
                },
            )
            .unwrap()
        });
        barrier.wait();
        let cancel = AtomicBool::new(false);
        assert!(acquire::<Cursor<Vec<u8>>>(
            &model,
            &temp.0,
            &cancel,
            |p| {
                if p.stage == Stage::Waiting {
                    cancel.store(true, Ordering::Relaxed);
                }
            },
            || panic!("locked source")
        )
        .is_err());
        let second = acquire::<Cursor<Vec<u8>>>(
            &model,
            &temp.0,
            &AtomicBool::new(false),
            |_| {},
            || panic!("source opened twice"),
        )
        .unwrap();
        assert!(!first.join().unwrap().reused);
        assert!(second.reused);
    });
}

#[test]
fn http_body_uses_the_same_checked_installer() {
    let (model, bytes) = fixture();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut request = [0; 2048];
        let _ = socket.read(&mut request).unwrap();
        write!(
            socket,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            bytes.len()
        )
        .unwrap();
        socket.write_all(&bytes).unwrap();
    });
    let temp = Temp::new();
    let result = acquire(
        &model,
        &temp.0,
        &AtomicBool::new(false),
        |_| {},
        || {
            Ok(ureq::get(format!("http://{address}/fixture.zip"))
                .call()?
                .into_body()
                .into_reader())
        },
    )
    .unwrap();
    assert!(result.path.is_file());
    server.join().unwrap();
}

#[test]
fn unpublished_download_has_no_network_fallback() {
    let temp = Temp::new();
    let mut model = catalog()[0].clone();
    model.url = None;
    assert!(download(&model, &temp.0, &AtomicBool::new(false), |_| {})
        .unwrap_err()
        .to_string()
        .contains("No published download source"));
    assert!(!model.directory(&temp.0).exists());
}

#[test]
fn published_documents_repair_a_prepublication_cache_without_changing_model_path() {
    let (mut model, bytes) = fixture();
    let temp = Temp::new();
    let installed = install_fixture(&model, &bytes, &temp.0).unwrap();
    let directory = model.directory(&temp.0);
    fs::rename(directory.join("README.ja.md"), directory.join("NOTICE.md")).unwrap();
    assert!(verify_cached(&model, &temp.0, &AtomicBool::new(false)).is_err());
    model.url = None;
    assert!(download(&model, &temp.0, &AtomicBool::new(false), |_| {}).is_err());
    assert_eq!(fs::read(&installed.path).unwrap(), b"model.onnx");
    assert!(directory.join("NOTICE.md").is_file());
    let repaired = install_fixture(&model, &bytes, &temp.0).unwrap();
    assert_eq!(repaired.path, installed.path);
    assert!(!repaired.reused);
    assert!(!directory.join("NOTICE.md").exists());
    verify_cached(&model, &temp.0, &AtomicBool::new(false)).unwrap();
}

#[test]
#[ignore = "Requires an explicitly supplied local vc-embedders ZIP; no network"]
fn import_real_local_package() {
    let source = PathBuf::from(std::env::var_os("VC_RS_EMBEDDER_ZIP").expect("VC_RS_EMBEDDER_ZIP"));
    let id = std::env::var("VC_RS_EMBEDDER_ID").expect("VC_RS_EMBEDDER_ID");
    let model = find(&id, "0.1.0").unwrap();
    let temp = Temp::new();
    import(model, &source, &temp.0, &AtomicBool::new(false), |_| {}).unwrap();
    verify_cached(model, &temp.0, &AtomicBool::new(false)).unwrap();
}
