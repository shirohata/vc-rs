use std::path::Path;

use anyhow::Result;
use tracing::info;

use super::onnx_meta::read_model_io;

#[cfg(feature = "ort")]
use super::sessions::{describe_value_type, load_session};
#[cfg(feature = "ort")]
use super::tensorrt::{ModelRole, TensorRtRunMode, TensorRtSessionPurpose};
#[cfg(feature = "ort")]
use crate::Provider;

/// CLI `inspect` command: prints a model's full I/O and metadata via ONNX
/// Runtime. The TensorRT-only build (no `ort`) falls back to the provider-neutral
/// `onnx_meta` reader below, which reports the same I/O and metadata without a
/// session.
#[cfg(feature = "ort")]
pub fn inspect_model(path: &Path) -> Result<()> {
    // Inspect is a structural ONNX query, so keep it CPU-only and provider-neutral.
    // CUDA/TensorRT load validation belongs to `run`/`wav`, where chunk-derived
    // fixed-shape profiles are available.
    let session = load_session(
        path,
        Provider::Cpu,
        ModelRole::Inspect,
        None,
        TensorRtRunMode::PinnedCpu,
        TensorRtSessionPurpose::Main,
    )?;
    println!("Model: {}", path.display());
    println!("Inputs:");
    for input in session.inputs() {
        println!("  {}: {}", input.name(), describe_value_type(input.dtype()));
    }
    println!("Outputs:");
    for output in session.outputs() {
        println!(
            "  {}: {}",
            output.name(),
            describe_value_type(output.dtype())
        );
    }
    println!("Opset version: {}", session.opset_for_domain("")?);
    if let Ok(metadata) = session.metadata() {
        println!("Metadata:");
        if let Some(name) = metadata.name() {
            println!("  name: {name}");
        }
        if let Some(producer) = metadata.producer() {
            println!("  producer: {producer}");
        }
        if let Some(description) = metadata.description() {
            println!("  description: {description}");
        }
        if let Some(domain) = metadata.domain() {
            println!("  domain: {domain}");
        }
        if let Some(graph_description) = metadata.graph_description() {
            println!("  graph_description: {graph_description}");
        }
        if let Some(version) = metadata.version() {
            println!("  version: {version}");
        }
        for key in metadata.custom_keys()? {
            if let Some(value) = metadata.custom(&key) {
                println!("  {key}: {value}");
            }
        }
    }
    Ok(())
}

/// `ort`-free fallback for the TensorRT-only build: read the structural I/O and
/// `metadata_props` directly from the ONNX protobuf (`onnx_meta`) instead of
/// opening an ORT session. Shapes and metadata are reported; ORT-only extras
/// (opset version, the producer/domain header fields) are not available here.
#[cfg(not(feature = "ort"))]
pub fn inspect_model(path: &Path) -> Result<()> {
    let io = read_model_io(path)?;
    println!("Model: {}", path.display());
    println!("Inputs:");
    for input in &io.inputs {
        println!("  {}: {}", input.name, describe_tensor(input));
    }
    println!("Outputs:");
    for output in &io.outputs {
        println!("  {}: {}", output.name, describe_tensor(output));
    }
    if !io.metadata.is_empty() {
        println!("Metadata:");
        for (key, value) in &io.metadata {
            println!("  {key}: {value}");
        }
    }
    Ok(())
}

/// Human-readable element type + shape for the `ort`-free inspect fallback.
/// `dim_value` 0 marks a symbolic axis (`onnx_meta` collapses `dim_param` to 0).
#[cfg(not(feature = "ort"))]
fn describe_tensor(tensor: &super::onnx_meta::TensorInfo) -> String {
    // ONNX TensorProto.DataType: only the types RVC models use are named.
    let elem = match tensor.elem_type {
        1 => "float32".to_string(),
        7 => "int64".to_string(),
        9 => "bool".to_string(),
        10 => "float16".to_string(),
        11 => "float64".to_string(),
        other => format!("elem_type={other}"),
    };
    let dims: Vec<String> = tensor
        .dims
        .iter()
        .map(|dim| {
            if *dim > 0 {
                dim.to_string()
            } else {
                "?".to_string()
            }
        })
        .collect();
    format!("{elem}[{}]", dims.join(", "))
}

pub(super) struct RvcModelInfo {
    pub(super) expected_feat_channels: i64,
}

pub(super) fn inspect_contentvec_input_name(
    path: &Path,
    expected_channels: i64,
    requested_output: Option<&str>,
) -> Result<String> {
    let io = read_model_io(path)?;
    let input_name = io.single_input_name()?.to_string();
    let output_name = io.select_embedder_output(expected_channels, requested_output)?;
    info!(
        "inspected ContentVec model for fixed profile: {} input={} output={}",
        path.display(),
        input_name,
        output_name
    );
    Ok(input_name)
}

pub(super) fn inspect_rvc_model(path: &Path) -> Result<RvcModelInfo> {
    let io = read_model_io(path)?;
    io.require_inputs(&["feats", "p_len", "pitch", "pitchf", "sid"])?;
    io.require_output("audio")?;
    let expected_feat_channels = io.expected_feat_channels()?;
    io.validate_rvc_metadata()?;
    Ok(RvcModelInfo {
        expected_feat_channels,
    })
}
