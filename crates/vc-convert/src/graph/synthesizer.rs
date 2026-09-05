//! Top-level synthesizer graph assembly: inputs/outputs, posterior
//! sampling, checkpoint validation, and export metadata.
//!
//! Ported from rvc-onnx-web (https://github.com/visgotti/rvc-onnx-web),
//! MIT License — `buildSynthesizerGraph` in `src/synthesizer-builder.ts`
//! plus `buildOnnxModel` / `validateStreamingCheckpoint` /
//! `buildStreamingMetadata` in `src/onnx-builder.ts`.

use anyhow::{bail, Result};

use crate::checkpoint::{ParsedCheckpoint, RvcVersion};
use crate::onnx::{DataType, Graph, Model};
use crate::{ConvertOptions, ExportMode};

use super::nsf::DecoderOptions;
use super::{decoder, flow, text_encoder, Fixed, GraphBuilder, Sym};

/// Reject checkpoints outside the supported envelope with actionable errors.
///
/// The TS source only runs this for streaming exports; this port supports
/// exactly v2+F0 models in both modes, so it gates every conversion.
pub(crate) fn validate_checkpoint(checkpoint: &ParsedCheckpoint) -> Result<()> {
    if checkpoint.version != RvcVersion::V2 {
        bail!(
            "only RVC v2 checkpoints are supported (this file reports {})",
            checkpoint.version.as_str()
        );
    }
    if !checkpoint.use_f0 {
        bail!("only F0-enabled RVC checkpoints are supported (this file reports f0=0)");
    }
    if !checkpoint
        .weights
        .contains_key("dec.m_source.l_linear.weight")
    {
        bail!(
            "unsupported decoder: the checkpoint lacks the standard NSF HiFi-GAN \
             source module (dec.m_source.l_linear.weight)"
        );
    }
    Ok(())
}

/// Build the complete model for the requested export mode.
///
/// Both supported modes use the webui input-name set (`ds`/`pitchf`) and an
/// external posterior noise input `rnd`; streaming additionally externalizes
/// the NSF noise and carries phase across windows.
pub(crate) fn build_model(
    checkpoint: &ParsedCheckpoint,
    options: &ConvertOptions,
) -> Result<Model> {
    validate_checkpoint(checkpoint)?;

    let streaming = options.export_mode == ExportMode::Streaming;
    let config = &checkpoint.config;
    let use_f0 = checkpoint.use_f0;
    let hidden_dim = checkpoint.version.hidden_dim();

    let mut b = GraphBuilder::new(&checkpoint.weights);

    // ---- inputs -----------------------------------------------------------
    b.inputs.push(GraphBuilder::value_info(
        "phone",
        DataType::Float,
        &[Fixed(1), Sym("phone_len"), Fixed(hidden_dim as i64)],
    ));
    b.inputs.push(GraphBuilder::value_info(
        "phone_lengths",
        DataType::Int64,
        &[Sym("batch")],
    ));
    if use_f0 {
        b.inputs.push(GraphBuilder::value_info(
            "pitch",
            DataType::Int64,
            &[Fixed(1), Sym("phone_len")],
        ));
        b.inputs.push(GraphBuilder::value_info(
            "pitchf",
            DataType::Float,
            &[Fixed(1), Sym("phone_len")],
        ));
    }
    b.inputs.push(GraphBuilder::value_info(
        "ds",
        DataType::Int64,
        &[Sym("batch")],
    ));
    b.inputs.push(GraphBuilder::value_info(
        "rnd",
        DataType::Float,
        &[
            Fixed(1),
            Fixed(config.inter_channels as i64),
            Sym("phone_len"),
        ],
    ));
    if streaming {
        b.inputs.push(GraphBuilder::value_info(
            "nsf_noise",
            DataType::Float,
            &[Fixed(1), Sym("audio_len"), Fixed(1)],
        ));
        b.inputs.push(GraphBuilder::value_info(
            "phase_in",
            DataType::Float,
            &[Fixed(1), Fixed(1), Fixed(1)],
        ));
    }

    // ---- speaker embedding: g = emb_g(ds).unsqueeze(-1) --------------------
    let emb_g = b.add_weight("emb_g.weight")?;
    let g_flat = b.unique("g_flat");
    b.gather(&emb_g, "ds", &g_flat, 0);
    let unsqueeze_axes = b.add_i64("unsqueeze_axes_neg1", vec![-1], vec![1]);
    let g = b.unique("g");
    b.unsqueeze(&g_flat, &unsqueeze_axes, &g);

    // ---- text encoder ------------------------------------------------------
    let enc = text_encoder::build_text_encoder(&mut b, config, use_f0)?;

    // ---- posterior sampling: z_p = (m_p + exp(logs_p)·rnd·0.66666)·mask ----
    let exp_logs_p = b.unary_new("Exp", &enc.logs_p, "exp_logs_p");
    let noise_scale = b.add_scalar("noise_scale", 0.66666);
    let scaled_noise = b.binary_new("Mul", "rnd", &noise_scale, "scaled_noise");
    let exp_noise = b.binary_new("Mul", &exp_logs_p, &scaled_noise, "exp_noise");
    let z_p_pre_mask = b.binary_new("Add", &enc.m_p, &exp_noise, "z_p_pre_mask");
    let z_p = b.binary_new("Mul", &z_p_pre_mask, &enc.x_mask, "z_p");

    // ---- flow (reverse) ----------------------------------------------------
    let z = flow::build_residual_coupling_block(&mut b, config, &z_p, &enc.x_mask, &g, true)?;
    let z_masked = b.binary_new("Mul", &z, &enc.x_mask, "z_masked");

    // ---- decoder -----------------------------------------------------------
    let decoder_options = DecoderOptions {
        streaming,
        nsf_noise_input: streaming.then_some("nsf_noise"),
        phase_input: streaming.then_some("phase_in"),
    };
    let dec = decoder::build_decoder(
        &mut b,
        config,
        &z_masked,
        use_f0.then_some("pitchf"),
        &g,
        use_f0,
        &decoder_options,
    )?;

    // ---- outputs -----------------------------------------------------------
    b.outputs.push(GraphBuilder::value_info(
        "audio",
        DataType::Float,
        &[Sym("batch"), Fixed(1), Sym("audio_len")],
    ));
    if !b.rename_node_output(&dec.audio, "audio") {
        b.unary("Identity", &dec.audio, "audio");
    }

    if streaming {
        let Some(phase_trace) = &dec.phase_trace else {
            bail!("streaming export requires an NSF phase trace output");
        };
        if !b.rename_node_output(phase_trace, "streaming_nsf_phase") {
            b.unary("Identity", phase_trace, "streaming_nsf_phase");
        }
        b.outputs.push(GraphBuilder::value_info(
            "streaming_nsf_phase",
            DataType::Float,
            &[Fixed(1), Sym("audio_len"), Fixed(1)],
        ));
    }

    // ---- metadata ----------------------------------------------------------
    // Divergence from the TS source (which emits no metadata for webui and
    // only rvc.* keys for streaming): both modes also get the vcclient-style
    // `metadata` JSON prop. vc-core requires truthy "f0" to accept the model
    // and uses "samplingRate" to size its windows for 32k/40k models
    // (onnx_meta.rs: validate_rvc_metadata / rvc_sample_rate).
    let mut metadata_props = vec![(
        "metadata".to_owned(),
        format!(r#"{{"f0":true,"samplingRate":{}}}"#, config.sr),
    )];
    if streaming {
        metadata_props.extend(streaming_metadata(checkpoint));
    }

    Ok(Model {
        ir_version: 8,
        opset_version: options.opset_version,
        producer_name: "vc-convert".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
        metadata_props,
        graph: Graph {
            name: "RVC_Synthesizer".to_owned(),
            nodes: b.nodes,
            inputs: b.inputs,
            outputs: b.outputs,
            initializers: b.initializers,
        },
    })
}

/// Exact port of `buildStreamingMetadata` — these keys are the
/// `rvc.stream_format_version` 1 contract that vc-core's onnx_meta reads.
fn streaming_metadata(checkpoint: &ParsedCheckpoint) -> Vec<(String, String)> {
    let frame_hop = checkpoint.config.frame_hop();
    [
        ("rvc.stream_format_version", "1".to_owned()),
        ("rvc.export_mode", "streaming".to_owned()),
        ("rvc.model_version", checkpoint.version.as_str().to_owned()),
        ("rvc.sample_rate", checkpoint.config.sr.to_string()),
        ("rvc.frame_hop", frame_hop.to_string()),
        (
            "rvc.streaming.inputs",
            "phone:float32[1,phone_len,768];phone_lengths:int64[1];pitch:int64[1,phone_len];\
             pitchf:float32[1,phone_len];ds:int64[1];rnd:float32[1,inter_channels,phone_len];\
             nsf_noise:float32[1,audio_len,1];phase_in:float32[1,1,1]"
                .to_owned(),
        ),
        (
            "rvc.streaming.outputs",
            "audio:float32[1,1,audio_len];streaming_nsf_phase:float32[1,audio_len,1]".to_owned(),
        ),
        ("rvc.streaming.random_inputs", "rnd,nsf_noise".to_owned()),
        (
            "rvc.streaming.phase_contract",
            "absolute_window_start;phase_in is normalized NSF fundamental phase at the input \
             window start;streaming_nsf_phase is the normalized NSF fundamental phase for each \
             generated sample;for overlapping windows select the next phase_in from \
             streaming_nsf_phase at the next input window start"
                .to_owned(),
        ),
        (
            "rvc.streaming.limitations",
            "RVC v2;F0 enabled;standard NSF HiFi-GAN source module;decoder convolution caches \
             are not exported"
                .to_owned(),
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect()
}
