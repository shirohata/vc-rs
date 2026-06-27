use std::path::Path;

use anyhow::Result;
use tracing::info;

use super::onnx_meta::{read_model_io, RvcIoNames, StreamFormat};

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
    /// Generator I/O names resolved to this model's export convention; threaded
    /// to every binding site so vcclient, RVC WebUI, and third-party converter
    /// exports all load.
    pub(super) io_names: RvcIoNames,
    /// The model's native audio sample rate from metadata `samplingRate`, when
    /// recorded. `None` means unknown; the pipeline falls back to the default
    /// `RVC_SAMPLE_RATE`. Threaded so the convert/output windows are sized at the
    /// model's real rate (e.g. 32 kHz) instead of the hardcoded 48 kHz.
    pub(super) rvc_sample_rate: Option<u32>,
    /// Streaming-export descriptor (NSF phase / `nsf_noise` contract), or `None`
    /// for a conventional RVC export. Drives the streaming time state.
    pub(super) stream: Option<StreamFormat>,
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
    let io_names = io.resolve_rvc_io_names()?;
    let expected_feat_channels = io.feat_channels(&io_names.feats)?;
    io.validate_rvc_metadata()?;
    let rvc_sample_rate = io.rvc_sample_rate();
    let stream = io.stream_format()?;
    let rnd_desc = io_names
        .rnd
        .as_ref()
        .map(|rnd| format!("{}[1,{},frames]", rnd.name, rnd.channels))
        .unwrap_or_else(|| "none".to_string());
    let stream_desc = stream
        .as_ref()
        .map(|stream| {
            format!(
                "v{} frame_hop={} sample_rate={}",
                stream.version, stream.frame_hop, stream.sample_rate
            )
        })
        .unwrap_or_else(|| "none".to_string());
    info!(
        "inspected RVC model: {} inputs=[{},{},{},{},{}] rnd={} output={} feat_channels={} sample_rate={} stream={}",
        path.display(),
        io_names.feats,
        io_names.p_len,
        io_names.pitch,
        io_names.pitchf,
        io_names.sid,
        rnd_desc,
        io_names.audio,
        expected_feat_channels,
        rvc_sample_rate
            .map(|rate| rate.to_string())
            .unwrap_or_else(|| "default".to_string()),
        stream_desc
    );
    Ok(RvcModelInfo {
        expected_feat_channels,
        io_names,
        rvc_sample_rate,
        stream,
    })
}
