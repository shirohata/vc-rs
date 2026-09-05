//! Offline RVC `.pth` → `.onnx` converter.
//!
//! Pure-Rust port of rvc-onnx-web
//! (https://github.com/visgotti/rvc-onnx-web), MIT License © Joseph
//! Viscardi — see `THIRD_PARTY.md`. Each module header names the TS file it
//! ports; this file corresponds to `src/converter.ts`.
//!
//! No ONNX Runtime, torch, or Python involved: the checkpoint (a ZIP of
//! pickled tensors) is parsed directly and the RVC synthesizer is rebuilt
//! node-by-node as an ONNX graph, then serialized as protobuf. The output
//! contract is exactly what `vc-core::model_rvc::onnx_meta` accepts;
//! supported models are RVC **v2 with F0** only.

mod checkpoint;
mod graph;
mod onnx;
mod pickle;
mod tensor;
mod torch;
mod zip;

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

pub use checkpoint::{parse_pth, ParsedCheckpoint, RvcConfig, RvcVersion};
pub use tensor::{Tensor, TensorData};

/// Export flavor. Both use the RVC-WebUI input names (`ds`/`pitchf`) and an
/// external posterior-noise input `rnd`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExportMode {
    /// vc-rs realtime target: additionally externalizes the NSF noise
    /// (`nsf_noise`) and carries the NSF phase across windows
    /// (`phase_in` → `streaming_nsf_phase`), with `rvc.*` metadata.
    #[default]
    Streaming,
    /// Non-streaming, RVC-WebUI-compatible graph (audio-only output; the
    /// NSF noise stays internal as a RandomNormalLike node).
    Webui,
}

#[derive(Clone, Copy, Debug)]
pub struct ConvertOptions {
    pub export_mode: ExportMode,
    /// ONNX opset (default 20, like the TS source).
    pub opset_version: i64,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        ConvertOptions {
            export_mode: ExportMode::Streaming,
            opset_version: 20,
        }
    }
}

/// Coarse progress reporting for a UI; stages are sequential.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgressStage {
    ReadArchive,
    ParseCheckpoint,
    BuildGraph,
    Serialize,
}

impl ProgressStage {
    /// Short human-readable label (English, matching the GUI language).
    pub fn label(self) -> &'static str {
        match self {
            ProgressStage::ReadArchive => "Reading checkpoint…",
            ProgressStage::ParseCheckpoint => "Parsing checkpoint…",
            ProgressStage::BuildGraph => "Building ONNX graph…",
            ProgressStage::Serialize => "Writing model…",
        }
    }
}

pub struct Conversion {
    pub onnx_bytes: Vec<u8>,
    pub sample_rate: u32,
    pub version: RvcVersion,
}

impl std::fmt::Debug for Conversion {
    /// Manual impl so debug output never dumps the multi-megabyte model.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conversion")
            .field(
                "onnx_bytes",
                &format_args!("<{} bytes>", self.onnx_bytes.len()),
            )
            .field("sample_rate", &self.sample_rate)
            .field("version", &self.version)
            .finish()
    }
}

/// Convert an in-memory `.pth` to ONNX bytes (port of `pthToOnnx`).
pub fn pth_to_onnx(
    pth_bytes: &[u8],
    options: &ConvertOptions,
    progress: &mut dyn FnMut(ProgressStage),
) -> Result<Conversion> {
    progress(ProgressStage::ParseCheckpoint);
    let checkpoint = parse_pth(pth_bytes)?;

    progress(ProgressStage::BuildGraph);
    let model = graph::synthesizer::build_model(&checkpoint, options)?;

    progress(ProgressStage::Serialize);
    let onnx_bytes = onnx::writer::serialize(&model);

    Ok(Conversion {
        onnx_bytes,
        sample_rate: checkpoint.config.sr,
        version: checkpoint.version,
    })
}

/// Convert `input` (a `.pth`) and write `<stem>.onnx` next to it.
///
/// The write is atomic (temp file + rename) so an interrupted conversion
/// never leaves a truncated `.onnx` where the engine might load it.
pub fn convert_pth_file(
    input: &Path,
    options: &ConvertOptions,
    progress: &mut dyn FnMut(ProgressStage),
) -> Result<PathBuf> {
    progress(ProgressStage::ReadArchive);
    let bytes =
        std::fs::read(input).with_context(|| format!("failed to read {}", input.display()))?;

    let conversion = pth_to_onnx(&bytes, options, progress)?;

    let output = input.with_extension("onnx");
    let file_name = output
        .file_name()
        .ok_or_else(|| anyhow!("cannot derive output name from {}", input.display()))?;
    let tmp = output.with_file_name(format!("{}.tmp", file_name.to_string_lossy()));
    std::fs::write(&tmp, &conversion.onnx_bytes)
        .with_context(|| format!("failed to write {}", tmp.display()))?;
    if let Err(err) = std::fs::rename(&tmp, &output) {
        // Windows rename fails onto an existing file only in rare sharing
        // situations; make sure the temp file doesn't linger.
        let _ = std::fs::remove_file(&tmp);
        return Err(err)
            .with_context(|| format!("failed to move output into {}", output.display()));
    }
    Ok(output)
}

/// Test-only access to the checked-in tiny checkpoint fixtures, shared with
/// vc-core's round-trip tests (dev-dependency, `test-fixtures` feature).
#[cfg(any(test, feature = "test-fixtures"))]
pub mod test_fixtures {
    pub fn tiny_v2_f0_pth() -> &'static [u8] {
        include_bytes!("../tests/fixtures/tiny_v2_f0.pth")
    }
    pub fn tiny_v1_pth() -> &'static [u8] {
        include_bytes!("../tests/fixtures/tiny_v1.pth")
    }
    pub fn tiny_v2_nof0_pth() -> &'static [u8] {
        include_bytes!("../tests/fixtures/tiny_v2_nof0.pth")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::Model;

    fn build(mode: ExportMode) -> Model {
        let checkpoint = parse_pth(test_fixtures::tiny_v2_f0_pth()).unwrap();
        let options = ConvertOptions {
            export_mode: mode,
            ..ConvertOptions::default()
        };
        graph::synthesizer::build_model(&checkpoint, &options).unwrap()
    }

    fn names(infos: &[crate::onnx::ValueInfo]) -> Vec<&str> {
        infos.iter().map(|i| i.name.as_str()).collect()
    }

    fn count_op(model: &Model, op: &str) -> usize {
        model.graph.nodes.iter().filter(|n| n.op_type == op).count()
    }

    fn metadata<'a>(model: &'a Model, key: &str) -> Option<&'a str> {
        model
            .metadata_props
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    // Mirrors rvc-onnx-web's tests/streaming-export.spec.ts assertions.
    #[test]
    fn streaming_export_contract() {
        let model = build(ExportMode::Streaming);

        assert_eq!(
            names(&model.graph.inputs),
            vec![
                "phone",
                "phone_lengths",
                "pitch",
                "pitchf",
                "ds",
                "rnd",
                "nsf_noise",
                "phase_in"
            ]
        );
        assert_eq!(
            names(&model.graph.outputs),
            vec!["audio", "streaming_nsf_phase"]
        );

        // All randomness must be externalized.
        assert_eq!(count_op(&model, "RandomNormalLike"), 0);

        assert_eq!(metadata(&model, "rvc.export_mode"), Some("streaming"));
        assert_eq!(metadata(&model, "rvc.stream_format_version"), Some("1"));
        assert_eq!(metadata(&model, "rvc.model_version"), Some("v2"));
        assert_eq!(metadata(&model, "rvc.sample_rate"), Some("40000"));
        // frame_hop == product(upsample_rates) == 10*10*2*2
        assert_eq!(metadata(&model, "rvc.frame_hop"), Some("400"));
        assert_eq!(
            metadata(&model, "rvc.streaming.random_inputs"),
            Some("rnd,nsf_noise")
        );
        assert!(metadata(&model, "rvc.streaming.phase_contract").is_some());
        assert!(metadata(&model, "rvc.streaming.limitations").is_some());
        // Both modes carry the vcclient-style JSON metadata prop.
        assert_eq!(
            metadata(&model, "metadata"),
            Some(r#"{"f0":true,"samplingRate":40000}"#)
        );

        // The phase output must come from a wrapping Mod(fmod) node.
        let phase_producer = model
            .graph
            .nodes
            .iter()
            .find(|n| n.outputs.iter().any(|o| o == "streaming_nsf_phase"))
            .expect("phase output produced");
        assert_eq!(phase_producer.op_type, "Mod");
    }

    #[test]
    fn webui_export_contract() {
        let model = build(ExportMode::Webui);

        assert_eq!(
            names(&model.graph.inputs),
            vec!["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
        );
        assert_eq!(names(&model.graph.outputs), vec!["audio"]);

        // Posterior noise is external (rnd), but the NSF noise stays as the
        // one RandomNormalLike node, matching the TS webui export.
        assert_eq!(count_op(&model, "RandomNormalLike"), 1);

        assert_eq!(metadata(&model, "rvc.export_mode"), None);
        assert_eq!(
            metadata(&model, "metadata"),
            Some(r#"{"f0":true,"samplingRate":40000}"#)
        );
    }

    #[test]
    fn every_node_input_is_produced() {
        // Topology sanity for both modes: every node input must be a graph
        // input, an initializer, or an earlier node's output (ORT enforces
        // topological order; catching it here beats an opaque load error).
        for mode in [ExportMode::Streaming, ExportMode::Webui] {
            let model = build(mode);
            let mut known: std::collections::HashSet<&str> = model
                .graph
                .inputs
                .iter()
                .map(|i| i.name.as_str())
                .chain(model.graph.initializers.iter().map(|i| i.name.as_str()))
                .collect();
            for node in &model.graph.nodes {
                for input in &node.inputs {
                    assert!(
                        input.is_empty() || known.contains(input.as_str()),
                        "node {} ({}) consumes unknown tensor {input}",
                        node.name,
                        node.op_type
                    );
                }
                for output in &node.outputs {
                    known.insert(output);
                }
            }
            for output in &model.graph.outputs {
                assert!(
                    known.contains(output.name.as_str()),
                    "graph output {} unproduced",
                    output.name
                );
            }
        }
    }

    #[test]
    fn convert_pth_file_writes_output_next_to_source() {
        let dir = std::env::temp_dir().join(format!(
            "vc-convert-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("tiny.pth");
        std::fs::write(&source, test_fixtures::tiny_v2_f0_pth()).unwrap();

        let output = convert_pth_file(&source, &ConvertOptions::default(), &mut |_| {}).unwrap();
        assert_eq!(output, dir.join("tiny.onnx"));
        let bytes = std::fs::read(&output).unwrap();
        assert_eq!(&bytes[..2], &[0x08, 0x08]);
        // No leftover temp file from the atomic write.
        assert!(!dir.join("tiny.onnx.tmp").exists());

        // Converting again overwrites cleanly.
        convert_pth_file(&source, &ConvertOptions::default(), &mut |_| {}).unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_v1_checkpoint() {
        let err = pth_to_onnx(
            test_fixtures::tiny_v1_pth(),
            &ConvertOptions::default(),
            &mut |_| {},
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("only RVC v2 checkpoints"), "{err}");
    }

    #[test]
    fn rejects_non_f0_checkpoint() {
        let err = pth_to_onnx(
            test_fixtures::tiny_v2_nof0_pth(),
            &ConvertOptions::default(),
            &mut |_| {},
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("F0-enabled"), "{err}");
    }

    #[test]
    fn end_to_end_serializes_and_reports_progress() {
        let mut stages = Vec::new();
        let conversion = pth_to_onnx(
            test_fixtures::tiny_v2_f0_pth(),
            &ConvertOptions::default(),
            &mut |stage| stages.push(stage),
        )
        .unwrap();
        assert_eq!(
            stages,
            vec![
                ProgressStage::ParseCheckpoint,
                ProgressStage::BuildGraph,
                ProgressStage::Serialize
            ]
        );
        assert_eq!(conversion.sample_rate, 40_000);
        assert_eq!(conversion.version, RvcVersion::V2);
        // ONNX protobuf starts with ir_version field: tag 0x08, value 8.
        assert_eq!(&conversion.onnx_bytes[..2], &[0x08, 0x08]);
        assert!(conversion.onnx_bytes.len() > 10_000);
    }
}
