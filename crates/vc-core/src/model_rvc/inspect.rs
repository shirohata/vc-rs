use std::path::Path;

use anyhow::Result;
use tracing::info;

use crate::Provider;

use super::sessions::{
    describe_value_type, expected_feat_channels, load_session, require_inputs, require_output,
    select_embedder_output, single_input_name, validate_rvc_metadata,
};
use super::tensorrt::{ModelRole, TensorRtRunMode, TensorRtSessionPurpose};

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

pub(super) struct RvcModelInfo {
    pub(super) expected_feat_channels: i64,
}

pub(super) fn inspect_contentvec_input_name(
    path: &Path,
    expected_channels: i64,
    requested_output: Option<&str>,
) -> Result<String> {
    let session = load_session(
        path,
        Provider::Cpu,
        ModelRole::Inspect,
        None,
        TensorRtRunMode::PinnedCpu,
        TensorRtSessionPurpose::Main,
    )?;
    let input_name = single_input_name(&session)?;
    let output_name = select_embedder_output(&session, expected_channels, requested_output)?;
    info!(
        "inspected ContentVec model for fixed profile: {} input={} output={}",
        path.display(),
        input_name,
        output_name
    );
    Ok(input_name)
}

pub(super) fn inspect_rvc_model(path: &Path) -> Result<RvcModelInfo> {
    let session = load_session(
        path,
        Provider::Cpu,
        ModelRole::Inspect,
        None,
        TensorRtRunMode::PinnedCpu,
        TensorRtSessionPurpose::Main,
    )?;
    require_inputs(&session, &["feats", "p_len", "pitch", "pitchf", "sid"])?;
    require_output(&session, "audio")?;
    let expected_feat_channels = expected_feat_channels(&session)?;
    validate_rvc_metadata(&session)?;
    Ok(RvcModelInfo {
        expected_feat_channels,
    })
}
