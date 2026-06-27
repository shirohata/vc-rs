//! Minimal, dependency-free reader for the ONNX (protobuf) structural metadata
//! the RVC pipeline needs: graph input/output names, their element types and
//! shapes, and the model `metadata_props`. This lets the native TensorRT build
//! inspect a model without an ONNX Runtime session, so `ort` can be dropped
//! entirely from the TensorRT-only build.
//!
//! Only the handful of fields below are decoded; every other field is skipped by
//! wire type, so the parser tolerates the rest of the (large) ModelProto schema.
//!
//! Field numbers (ONNX `onnx.proto`):
//! - `ModelProto.graph` = 7, `ModelProto.metadata_props` = 14
//! - `GraphProto.input` = 11, `GraphProto.output` = 12
//! - `ValueInfoProto.name` = 1, `ValueInfoProto.type` = 2
//! - `TypeProto.tensor_type` = 1
//! - `TypeProto.Tensor.elem_type` = 1, `TypeProto.Tensor.shape` = 2
//! - `TensorShapeProto.dim` = 1
//! - `TensorShapeProto.Dimension.dim_value` = 1 (`dim_param` = 2 → symbolic)
//! - `StringStringEntryProto.key` = 1, `StringStringEntryProto.value` = 2

use std::fs;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use tracing::info;

#[derive(Debug, Clone)]
pub(super) struct TensorInfo {
    pub(super) name: String,
    pub(super) elem_type: i32,
    /// `dim_value` per axis; `0` marks a symbolic/unknown dimension (`dim_param`).
    pub(super) dims: Vec<i64>,
}

impl TensorInfo {
    /// Final axis when it is a statically known positive size (channel count),
    /// mirroring the ORT-based `shape.last().filter(|c| *c > 0)` behaviour.
    pub(super) fn last_dim_channels(&self) -> Option<i64> {
        self.dims.last().copied().filter(|dim| *dim > 0)
    }

    /// Whether this is a typed tensor (ONNX `elem_type` 0 = UNDEFINED), used to
    /// reject non-tensor outputs the way the ORT `ValueType::Tensor` check did.
    fn is_tensor(&self) -> bool {
        self.elem_type != 0
    }

    fn describe(&self) -> String {
        format!("elem_type={} dims={:?}", self.elem_type, self.dims)
    }
}

#[derive(Debug, Default, Clone)]
pub(super) struct ModelIo {
    pub(super) inputs: Vec<TensorInfo>,
    pub(super) outputs: Vec<TensorInfo>,
    pub(super) metadata: Vec<(String, String)>,
}

/// The RVC generator's I/O tensor names, resolved to whatever aliases a given
/// model actually exports. RVC ONNX exporters disagree on names: vcclient emits
/// `feats`/`p_len`/`pitch`/`pitchf`/`sid`, RVC WebUI's own ONNX export emits
/// `phone`/`phone_lengths`/`pitch`/`pitchf`/`ds`/`rnd` (it adds the `rnd`
/// latent-noise input), and third-party `.pth`->`.onnx` converters such as
/// rvc-onnx-web emit `phone`/`phone_lengths`/`pitch`/`nsff0`/`sid`. The tensors
/// carry the same semantics, so we resolve each role once at load and bind by the
/// resolved name everywhere downstream (ORT `run`/IoBinding, TensorRT profiles,
/// the native shim).
#[derive(Debug, Clone)]
pub(super) struct RvcIoNames {
    pub(super) feats: String,
    pub(super) p_len: String,
    pub(super) pitch: String,
    pub(super) pitchf: String,
    pub(super) sid: String,
    pub(super) audio: String,
    /// Optional latent-noise input. Some RVC exporters (e.g. RVC WebUI) expose
    /// the VITS reparameterization noise `z`
    /// (`rnd`, shape `[1, inter_channels, frames]`) as a generator input instead
    /// of sampling it inside the graph. When present, the pipeline must feed a
    /// fresh `N(0, 1)` tensor of this shape each inference; `None` means the model
    /// samples its own noise and needs no extra input.
    pub(super) rnd: Option<RvcRndInput>,
    /// Optional NSF source-noise input of a streaming export
    /// (`nsf_noise`, shape `[1, audio_len, 1]`): per-output-sample `N(0, 1)` noise
    /// the decoder's NSF source would otherwise sample internally. `None` for
    /// non-streaming exports.
    pub(super) nsf_noise: Option<String>,
    /// Optional NSF fundamental-phase input of a streaming export
    /// (`phase_in`, shape `[1, 1, 1]`): the normalized phase at the window start.
    /// Paired with [`phase_out`](Self::phase_out). `None` for non-streaming.
    pub(super) phase_in: Option<String>,
    /// Optional NSF fundamental-phase output of a streaming export
    /// (`phase_out`, shape `[1, 1, 1]`): the phase after the last generated
    /// sample. `None` for non-streaming exports.
    pub(super) phase_out: Option<String>,
}

/// Streaming-export format descriptor parsed from `rvc.*` metadata. Present only
/// for rvc-onnx-web streaming exports (`rvc.export_mode == "streaming"`).
#[derive(Debug, Clone, Copy)]
pub(super) struct StreamFormat {
    /// `rvc.stream_format_version` — the phase/noise contract version (currently 1).
    pub(super) version: u32,
    /// `rvc.frame_hop` — output samples per feature frame (NSF upsampling factor).
    /// `audio_len == feature_frames * frame_hop`.
    pub(super) frame_hop: usize,
    /// `rvc.sample_rate` — the model's output sample rate, the time base for the
    /// per-frame phase advance `f0 / sample_rate * frame_hop`.
    pub(super) sample_rate: u32,
}

/// The resolved name and static channel count (`inter_channels`, the middle axis
/// of `[1, channels, frames]`) of an RVC model's latent-noise input.
#[derive(Debug, Clone)]
pub(super) struct RvcRndInput {
    pub(super) name: String,
    pub(super) channels: i64,
}

// Accepted aliases per role, canonical (vcclient) name first. Resolution picks
// the first alias the model actually exposes, so the canonical name keeps
// precedence when a model happens to expose several.
const RVC_FEATS_ALIASES: &[&str] = &["feats", "phone"];
const RVC_P_LEN_ALIASES: &[&str] = &["p_len", "phone_lengths"];
const RVC_PITCH_ALIASES: &[&str] = &["pitch"];
const RVC_PITCHF_ALIASES: &[&str] = &["pitchf", "nsff0"];
const RVC_SID_ALIASES: &[&str] = &["sid", "ds"];
const RVC_AUDIO_ALIASES: &[&str] = &["audio", "out", "output"];
// Latent-noise input is optional, so it has no canonical fallback: a model
// either exports it under one of these names or samples noise internally.
const RVC_RND_ALIASES: &[&str] = &["rnd", "z"];
// Streaming-export I/O (rvc-onnx-web). All optional; absent on non-streaming
// exports. Names are fixed by `rvc.stream_format_version` 1.
const RVC_NSF_NOISE_ALIASES: &[&str] = &["nsf_noise"];
const RVC_PHASE_IN_ALIASES: &[&str] = &["phase_in"];
const RVC_PHASE_OUT_ALIASES: &[&str] = &["phase_out"];

impl RvcIoNames {
    /// The canonical vcclient names, for tests/benchmarks that synthesize a
    /// profile without a real model to resolve against.
    #[cfg(test)]
    pub(super) fn canonical() -> Self {
        Self {
            feats: "feats".to_string(),
            p_len: "p_len".to_string(),
            pitch: "pitch".to_string(),
            pitchf: "pitchf".to_string(),
            sid: "sid".to_string(),
            audio: "audio".to_string(),
            rnd: None,
            nsf_noise: None,
            phase_in: None,
            phase_out: None,
        }
    }
}

impl ModelIo {
    pub(super) fn input(&self, name: &str) -> Option<&TensorInfo> {
        self.inputs.iter().find(|tensor| tensor.name == name)
    }

    pub(super) fn output(&self, name: &str) -> Option<&TensorInfo> {
        self.outputs.iter().find(|tensor| tensor.name == name)
    }

    pub(super) fn metadata_value(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(entry_key, _)| entry_key == key)
            .map(|(_, value)| value.as_str())
    }

    // --- structural inspection (provider-neutral; replaces the ORT-session
    // based helpers so the native TensorRT path needs no ONNX Runtime) ---

    pub(super) fn single_input_name(&self) -> Result<&str> {
        if self.inputs.len() != 1 {
            bail!("expected a single input, got {}", self.inputs.len());
        }
        Ok(self.inputs[0].name.as_str())
    }

    pub(super) fn require_inputs(&self, names: &[&str]) -> Result<()> {
        for name in names {
            if self.input(name).is_none() {
                let actual: Vec<&str> = self.inputs.iter().map(|t| t.name.as_str()).collect();
                bail!("required input '{name}' not found; model inputs are {actual:?}");
            }
        }
        Ok(())
    }

    pub(super) fn require_output(&self, name: &str) -> Result<()> {
        if self.output(name).is_none() {
            let actual: Vec<&str> = self.outputs.iter().map(|t| t.name.as_str()).collect();
            bail!("required output '{name}' not found; model outputs are {actual:?}");
        }
        Ok(())
    }

    /// Resolve every RVC generator I/O name to the alias this model exports,
    /// erroring with the model's actual names when a role cannot be matched.
    pub(super) fn resolve_rvc_io_names(&self) -> Result<RvcIoNames> {
        Ok(RvcIoNames {
            feats: self.resolve_input_alias("feats", RVC_FEATS_ALIASES)?,
            p_len: self.resolve_input_alias("p_len", RVC_P_LEN_ALIASES)?,
            pitch: self.resolve_input_alias("pitch", RVC_PITCH_ALIASES)?,
            pitchf: self.resolve_input_alias("pitchf", RVC_PITCHF_ALIASES)?,
            sid: self.resolve_input_alias("sid", RVC_SID_ALIASES)?,
            audio: self.resolve_rvc_output()?,
            rnd: self.resolve_rvc_rnd()?,
            // Streaming I/O is optional; `find_input`/`find_output` return `None`
            // for non-streaming exports, leaving those paths untouched.
            nsf_noise: self.find_input_alias(RVC_NSF_NOISE_ALIASES),
            phase_in: self.find_input_alias(RVC_PHASE_IN_ALIASES),
            phase_out: self.find_output_alias(RVC_PHASE_OUT_ALIASES),
        })
    }

    /// First matching input name among `aliases`, or `None` (optional inputs).
    fn find_input_alias(&self, aliases: &[&str]) -> Option<String> {
        aliases
            .iter()
            .find(|alias| self.input(alias).is_some())
            .map(|alias| (*alias).to_string())
    }

    /// First matching output name among `aliases`, or `None` (optional outputs).
    fn find_output_alias(&self, aliases: &[&str]) -> Option<String> {
        aliases
            .iter()
            .find(|alias| self.output(alias).is_some())
            .map(|alias| (*alias).to_string())
    }

    /// Streaming-export descriptor, or `None` for a non-streaming export. A model
    /// is streaming when `rvc.export_mode == "streaming"`; the frame hop and
    /// sample rate are then required (they set the phase/noise time base).
    pub(super) fn stream_format(&self) -> Result<Option<StreamFormat>> {
        if self.metadata_value("rvc.export_mode") != Some("streaming") {
            return Ok(None);
        }
        let version = self
            .metadata_value("rvc.stream_format_version")
            .and_then(|value| value.trim().parse::<u32>().ok())
            .ok_or_else(|| anyhow!("streaming RVC export is missing rvc.stream_format_version"))?;
        let frame_hop = self
            .metadata_value("rvc.frame_hop")
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|hop| *hop > 0)
            .ok_or_else(|| anyhow!("streaming RVC export is missing a positive rvc.frame_hop"))?;
        let sample_rate = self
            .metadata_value("rvc.sample_rate")
            .and_then(|value| value.trim().parse::<u32>().ok())
            .filter(|rate| *rate > 0)
            .ok_or_else(|| anyhow!("streaming RVC export is missing a positive rvc.sample_rate"))?;
        Ok(Some(StreamFormat {
            version,
            frame_hop,
            sample_rate,
        }))
    }

    /// Detect the optional latent-noise input. Returns `None` when the model
    /// samples its own noise (no `rnd`/`z` input). When present, the channel
    /// count (middle axis of `[1, channels, frames]`) must be statically known so
    /// the pipeline can size the `N(0, 1)` tensor it feeds each chunk.
    fn resolve_rvc_rnd(&self) -> Result<Option<RvcRndInput>> {
        for alias in RVC_RND_ALIASES {
            if let Some(input) = self.input(alias) {
                let channels = input
                    .dims
                    .get(1)
                    .copied()
                    .filter(|dim| *dim > 0)
                    .ok_or_else(|| {
                        anyhow!(
                            "RVC '{alias}' latent-noise input must expose a static channel count \
                             as its middle axis [1, channels, frames]; got {}",
                            input.describe()
                        )
                    })?;
                return Ok(Some(RvcRndInput {
                    name: (*alias).to_string(),
                    channels,
                }));
            }
        }
        Ok(None)
    }

    fn resolve_input_alias(&self, role: &str, aliases: &[&str]) -> Result<String> {
        for alias in aliases {
            if self.input(alias).is_some() {
                return Ok((*alias).to_string());
            }
        }
        let actual: Vec<&str> = self.inputs.iter().map(|t| t.name.as_str()).collect();
        bail!("RVC model has no '{role}' input (accepted names: {aliases:?}); model inputs are {actual:?}");
    }

    fn resolve_rvc_output(&self) -> Result<String> {
        for alias in RVC_AUDIO_ALIASES {
            if self.output(alias).is_some() {
                return Ok((*alias).to_string());
            }
        }
        // Exporters occasionally name the lone audio output something bespoke;
        // a single-output graph is unambiguous, so accept it.
        if self.outputs.len() == 1 {
            return Ok(self.outputs[0].name.clone());
        }
        let actual: Vec<&str> = self.outputs.iter().map(|t| t.name.as_str()).collect();
        bail!("RVC model has no audio output (accepted names: {RVC_AUDIO_ALIASES:?}); model outputs are {actual:?}");
    }

    /// Channel count (static last axis) of the resolved RVC `feats` input.
    pub(super) fn feat_channels(&self, feats_name: &str) -> Result<i64> {
        let feats = self
            .input(feats_name)
            .ok_or_else(|| anyhow!("RVC model has no '{feats_name}' input"))?;
        feats.last_dim_channels().ok_or_else(|| {
            anyhow!("RVC '{feats_name}' input does not expose a static channel count")
        })
    }

    pub(super) fn validate_rvc_metadata(&self) -> Result<()> {
        if let Some(metadata) = self.metadata_value("metadata") {
            // Exporters format the `f0` flag differently: vcclient emits the
            // spaced integer `"f0": 1` (Python `json.dumps`), while compact
            // exporters (RVC_CONVERTER / rvc-onnx-web) emit `"f0":true`. Strip
            // whitespace and accept either truthy form so all of them load;
            // f0=0 / f0=false (non-pitch models) still fail this guard. The
            // leading quote in the needle keeps `"dynamicShapes":true` etc. from
            // matching.
            let compact: String = metadata.chars().filter(|c| !c.is_whitespace()).collect();
            if !compact.contains(r#""f0":1"#) && !compact.contains(r#""f0":true"#) {
                bail!("RVC model metadata does not indicate f0=1 or f0=true: {metadata}");
            }
            info!("RVC metadata: {metadata}");
        }
        Ok(())
    }

    /// The model's native audio sample rate from the metadata `samplingRate`
    /// field, when present and parseable. RVC exporters (RVC_CONVERTER /
    /// rvc-onnx-web / RVC WebUI) record it as a JSON number; vcclient's
    /// `{"f0": 1}` blob omits it. `None` means "unknown", and the pipeline falls
    /// back to the [`RVC_SAMPLE_RATE`](super::shape::RVC_SAMPLE_RATE) default.
    /// Dependency-free string parse, matching `validate_rvc_metadata`.
    pub(super) fn rvc_sample_rate(&self) -> Option<u32> {
        // Streaming exports record the rate as a dedicated `rvc.sample_rate` key
        // rather than inside the `metadata` JSON blob; prefer it when present so a
        // 40/48 kHz streaming model is not mis-sized to the default.
        if let Some(rate) = self
            .metadata_value("rvc.sample_rate")
            .and_then(|value| value.trim().parse::<u32>().ok())
            .filter(|rate| *rate > 0)
        {
            return Some(rate);
        }
        let metadata = self.metadata_value("metadata")?;
        let compact: String = metadata.chars().filter(|c| !c.is_whitespace()).collect();
        const KEY: &str = r#""samplingRate":"#;
        let start = compact.find(KEY)? + KEY.len();
        let digits: String = compact[start..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        digits.parse::<u32>().ok().filter(|rate| *rate > 0)
    }

    /// Pick the embedder output matching `expected_channels`, honouring an
    /// explicit `requested_output` and the unit12/unit9 preference for
    /// 768/256-channel ContentVec exports.
    pub(super) fn select_embedder_output(
        &self,
        expected_channels: i64,
        requested_output: Option<&str>,
    ) -> Result<String> {
        if let Some(name) = requested_output {
            let output = self.output(name).ok_or_else(|| {
                let actual: Vec<&str> = self.outputs.iter().map(|t| t.name.as_str()).collect();
                anyhow!("requested embedder output '{name}' not found; outputs are {actual:?}")
            })?;
            validate_embedder_output_selection(
                "requested embedder output",
                output,
                expected_channels,
            )?;
            return Ok(name.to_string());
        }

        let preferred_output = match expected_channels {
            768 => Some("unit12"),
            256 => Some("unit9"),
            _ => None,
        };
        if let Some(name) = preferred_output {
            for output in &self.outputs {
                if output.name == name && output.last_dim_channels() == Some(expected_channels) {
                    return Ok(output.name.clone());
                }
            }
        }
        for output in &self.outputs {
            if output.last_dim_channels() == Some(expected_channels) {
                return Ok(output.name.clone());
            }
        }
        if self.outputs.len() == 1 {
            let output = &self.outputs[0];
            validate_embedder_output_selection(
                "single embedder output",
                output,
                expected_channels,
            )?;
            return Ok(output.name.clone());
        }
        let actual: Vec<String> = self.outputs.iter().map(|t| t.describe()).collect();
        bail!("no embedder output matches {expected_channels} channels; outputs are {actual:?}");
    }
}

fn validate_embedder_output_selection(
    label: &str,
    tensor: &TensorInfo,
    expected_channels: i64,
) -> Result<()> {
    if !tensor.is_tensor() {
        bail!(
            "{label} '{}' must be a tensor, got {}",
            tensor.name,
            tensor.describe()
        );
    }
    if let Some(channels) = tensor.last_dim_channels() {
        if channels != expected_channels {
            bail!(
                "{label} '{}' does not match expected {expected_channels} channels: {}",
                tensor.name,
                tensor.describe()
            );
        }
    }
    Ok(())
}

pub(super) fn read_model_io(path: &Path) -> Result<ModelIo> {
    let bytes =
        fs::read(path).with_context(|| format!("failed to read ONNX model {}", path.display()))?;
    parse_model(&bytes).with_context(|| format!("failed to parse ONNX model {}", path.display()))
}

// --- protobuf wire decoding -------------------------------------------------

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn eof(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut value: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            let byte = *self
                .buf
                .get(self.pos)
                .context("unexpected end of buffer while reading varint")?;
            self.pos += 1;
            if shift >= 64 {
                bail!("varint exceeds 64 bits");
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        Ok(value)
    }

    fn read_len_prefixed(&mut self) -> Result<&'a [u8]> {
        let len = usize::try_from(self.read_varint()?).context("length does not fit in usize")?;
        let end = self
            .pos
            .checked_add(len)
            .context("length-delimited field length overflows")?;
        let slice = self
            .buf
            .get(self.pos..end)
            .context("length-delimited field exceeds buffer")?;
        self.pos = end;
        Ok(slice)
    }

    fn advance(&mut self, n: usize) -> Result<()> {
        let end = self.pos.checked_add(n).context("advance overflows")?;
        if end > self.buf.len() {
            bail!("fixed-width field exceeds buffer");
        }
        self.pos = end;
        Ok(())
    }

    /// Skip a field whose value we do not decode, by its wire type.
    fn skip(&mut self, wire_type: u64) -> Result<()> {
        match wire_type {
            0 => {
                self.read_varint()?;
            }
            1 => self.advance(8)?,
            2 => {
                self.read_len_prefixed()?;
            }
            5 => self.advance(4)?,
            other => bail!("unsupported protobuf wire type {other}"),
        }
        Ok(())
    }
}

/// Run `on_field` for each `(field_number, wire_type)` in `bytes`, where
/// `on_field` consumes the value for length-delimited/varint fields it handles
/// and returns `false` to fall through to the default skip.
fn for_each_field(
    bytes: &[u8],
    mut on_field: impl FnMut(u64, u64, &mut Reader<'_>) -> Result<bool>,
) -> Result<()> {
    let mut reader = Reader::new(bytes);
    while !reader.eof() {
        let tag = reader.read_varint()?;
        let field = tag >> 3;
        let wire = tag & 0x7;
        if !on_field(field, wire, &mut reader)? {
            reader.skip(wire)?;
        }
    }
    Ok(())
}

fn parse_model(bytes: &[u8]) -> Result<ModelIo> {
    let mut io = ModelIo::default();
    for_each_field(bytes, |field, wire, reader| match (field, wire) {
        (7, 2) => {
            let graph = reader.read_len_prefixed()?;
            parse_graph(graph, &mut io)?;
            Ok(true)
        }
        (14, 2) => {
            let entry = reader.read_len_prefixed()?;
            if let Some(pair) = parse_string_entry(entry)? {
                io.metadata.push(pair);
            }
            Ok(true)
        }
        _ => Ok(false),
    })?;
    Ok(io)
}

fn parse_graph(bytes: &[u8], io: &mut ModelIo) -> Result<()> {
    for_each_field(bytes, |field, wire, reader| match (field, wire) {
        (11, 2) => {
            io.inputs
                .push(parse_value_info(reader.read_len_prefixed()?)?);
            Ok(true)
        }
        (12, 2) => {
            io.outputs
                .push(parse_value_info(reader.read_len_prefixed()?)?);
            Ok(true)
        }
        _ => Ok(false),
    })
}

fn parse_value_info(bytes: &[u8]) -> Result<TensorInfo> {
    let mut name = String::new();
    let mut elem_type = 0i32;
    let mut dims = Vec::new();
    for_each_field(bytes, |field, wire, reader| match (field, wire) {
        (1, 2) => {
            name = read_utf8(reader.read_len_prefixed()?, "value info name")?;
            Ok(true)
        }
        (2, 2) => {
            let (parsed_elem, parsed_dims) = parse_type(reader.read_len_prefixed()?)?;
            elem_type = parsed_elem;
            dims = parsed_dims;
            Ok(true)
        }
        _ => Ok(false),
    })?;
    Ok(TensorInfo {
        name,
        elem_type,
        dims,
    })
}

/// TypeProto: only `tensor_type` (field 1) is decoded; other type variants
/// (sequence/map/etc.) leave the defaults.
fn parse_type(bytes: &[u8]) -> Result<(i32, Vec<i64>)> {
    let mut result = (0i32, Vec::new());
    for_each_field(bytes, |field, wire, reader| match (field, wire) {
        (1, 2) => {
            result = parse_tensor_type(reader.read_len_prefixed()?)?;
            Ok(true)
        }
        _ => Ok(false),
    })?;
    Ok(result)
}

fn parse_tensor_type(bytes: &[u8]) -> Result<(i32, Vec<i64>)> {
    let mut elem_type = 0i32;
    let mut dims = Vec::new();
    for_each_field(bytes, |field, wire, reader| match (field, wire) {
        (1, 0) => {
            elem_type = reader.read_varint()? as i32;
            Ok(true)
        }
        (2, 2) => {
            dims = parse_shape(reader.read_len_prefixed()?)?;
            Ok(true)
        }
        _ => Ok(false),
    })?;
    Ok((elem_type, dims))
}

fn parse_shape(bytes: &[u8]) -> Result<Vec<i64>> {
    let mut dims = Vec::new();
    for_each_field(bytes, |field, wire, reader| match (field, wire) {
        (1, 2) => {
            dims.push(parse_dim(reader.read_len_prefixed()?)?);
            Ok(true)
        }
        _ => Ok(false),
    })?;
    Ok(dims)
}

/// Dimension: `dim_value` (field 1) when static, otherwise `0` (symbolic
/// `dim_param`, field 2, is read only to consume it).
fn parse_dim(bytes: &[u8]) -> Result<i64> {
    let mut value = 0i64;
    for_each_field(bytes, |field, wire, reader| match (field, wire) {
        (1, 0) => {
            value = reader.read_varint()? as i64;
            Ok(true)
        }
        _ => Ok(false),
    })?;
    Ok(value)
}

fn parse_string_entry(bytes: &[u8]) -> Result<Option<(String, String)>> {
    let mut key = None;
    let mut value = None;
    for_each_field(bytes, |field, wire, reader| match (field, wire) {
        (1, 2) => {
            key = Some(read_utf8(reader.read_len_prefixed()?, "metadata key")?);
            Ok(true)
        }
        (2, 2) => {
            value = Some(read_utf8(reader.read_len_prefixed()?, "metadata value")?);
            Ok(true)
        }
        _ => Ok(false),
    })?;
    Ok(key.zip(value))
}

fn read_utf8(bytes: &[u8], label: &str) -> Result<String> {
    String::from_utf8(bytes.to_vec()).with_context(|| format!("{label} is not valid UTF-8"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal protobuf builders so tests assemble byte-accurate ModelProtos.
    fn varint(mut value: u64, out: &mut Vec<u8>) {
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                break;
            }
        }
    }

    fn tag(field: u64, wire: u64, out: &mut Vec<u8>) {
        varint((field << 3) | wire, out);
    }

    fn len_delimited(field: u64, payload: &[u8], out: &mut Vec<u8>) {
        tag(field, 2, out);
        varint(payload.len() as u64, out);
        out.extend_from_slice(payload);
    }

    fn varint_field(field: u64, value: u64, out: &mut Vec<u8>) {
        tag(field, 0, out);
        varint(value, out);
    }

    fn dimension(dim_value: i64) -> Vec<u8> {
        let mut out = Vec::new();
        varint_field(1, dim_value as u64, &mut out);
        out
    }

    fn shape(dims: &[i64]) -> Vec<u8> {
        let mut out = Vec::new();
        for &dim in dims {
            len_delimited(1, &dimension(dim), &mut out);
        }
        out
    }

    fn tensor_type(elem_type: i32, dims: &[i64]) -> Vec<u8> {
        let mut out = Vec::new();
        varint_field(1, elem_type as u64, &mut out);
        len_delimited(2, &shape(dims), &mut out);
        out
    }

    fn type_proto(elem_type: i32, dims: &[i64]) -> Vec<u8> {
        let mut out = Vec::new();
        len_delimited(1, &tensor_type(elem_type, dims), &mut out);
        out
    }

    fn value_info(name: &str, elem_type: i32, dims: &[i64]) -> Vec<u8> {
        let mut out = Vec::new();
        len_delimited(1, name.as_bytes(), &mut out);
        len_delimited(2, &type_proto(elem_type, dims), &mut out);
        out
    }

    fn string_entry(key: &str, value: &str) -> Vec<u8> {
        let mut out = Vec::new();
        len_delimited(1, key.as_bytes(), &mut out);
        len_delimited(2, value.as_bytes(), &mut out);
        out
    }

    #[test]
    fn parses_inputs_outputs_metadata_and_skips_unknown_fields() {
        // GraphProto with two inputs and one output.
        let mut graph = Vec::new();
        len_delimited(11, &value_info("feats", 1, &[1, 100, 768]), &mut graph);
        len_delimited(11, &value_info("sid", 7, &[1]), &mut graph);
        len_delimited(12, &value_info("audio", 1, &[1, 65536]), &mut graph);
        // An unknown graph field (e.g. node, field 1) must be skipped.
        len_delimited(1, b"ignored-node-bytes", &mut graph);

        let mut model = Vec::new();
        // Unknown ModelProto scalar field (ir_version = 1) before the graph.
        varint_field(1, 9, &mut model);
        len_delimited(7, &graph, &mut model);
        len_delimited(14, &string_entry("metadata", r#"{"f0": 1}"#), &mut model);

        let io = parse_model(&model).unwrap();
        assert_eq!(io.inputs.len(), 2);
        assert_eq!(io.input("feats").unwrap().elem_type, 1);
        assert_eq!(io.input("feats").unwrap().last_dim_channels(), Some(768));
        assert_eq!(io.input("sid").unwrap().dims, vec![1]);
        assert_eq!(io.output("audio").unwrap().name, "audio");
        assert!(io
            .metadata_value("metadata")
            .unwrap()
            .contains(r#""f0": 1"#));
    }

    fn rvc_io(inputs: &[&str], outputs: &[&str]) -> ModelIo {
        ModelIo {
            inputs: inputs
                .iter()
                .map(|name| TensorInfo {
                    name: (*name).to_string(),
                    elem_type: 1,
                    dims: vec![1, 0, 768],
                })
                .collect(),
            outputs: outputs
                .iter()
                .map(|name| TensorInfo {
                    name: (*name).to_string(),
                    elem_type: 1,
                    dims: vec![1, 0],
                })
                .collect(),
            metadata: Vec::new(),
        }
    }

    fn io_with_metadata(json: &str) -> ModelIo {
        let mut io = rvc_io(&["feats", "p_len", "pitch", "pitchf", "sid"], &["audio"]);
        io.metadata = vec![("metadata".to_string(), json.to_string())];
        io
    }

    #[test]
    fn accepts_spaced_integer_and_compact_boolean_f0() {
        // vcclient spaced integer, plus the compact/spaced boolean forms emitted
        // by RVC_CONVERTER / rvc-onnx-web.
        for json in [
            r#"{"f0": 1}"#,
            r#"{"f0":1}"#,
            r#"{"f0":true,"samplingRate":32000}"#,
            r#"{"f0": true}"#,
        ] {
            assert!(
                io_with_metadata(json).validate_rvc_metadata().is_ok(),
                "should accept {json}"
            );
        }
    }

    #[test]
    fn rejects_non_pitch_f0_metadata() {
        for json in [r#"{"f0":0}"#, r#"{"f0": 0}"#, r#"{"f0":false}"#] {
            assert!(
                io_with_metadata(json).validate_rvc_metadata().is_err(),
                "should reject {json}"
            );
        }
        // Absent metadata is permitted (the guard only runs when present).
        assert!(
            rvc_io(&["feats", "p_len", "pitch", "pitchf", "sid"], &["audio"])
                .validate_rvc_metadata()
                .is_ok()
        );
    }

    #[test]
    fn parses_sampling_rate_from_metadata() {
        assert_eq!(
            io_with_metadata(r#"{"f0":true,"samplingRate":32000}"#).rvc_sample_rate(),
            Some(32_000)
        );
        // Spaced JSON is normalized before the lookup.
        assert_eq!(
            io_with_metadata(r#"{"samplingRate": 40000, "f0": 1}"#).rvc_sample_rate(),
            Some(40_000)
        );
        // No samplingRate field, or no metadata blob -> unknown (pipeline default).
        assert_eq!(io_with_metadata(r#"{"f0": 1}"#).rvc_sample_rate(), None);
        assert_eq!(
            rvc_io(&["feats", "p_len", "pitch", "pitchf", "sid"], &["audio"]).rvc_sample_rate(),
            None
        );
    }

    #[test]
    fn resolves_rvc_webui_input_names() {
        let io = rvc_io(&["feats", "p_len", "pitch", "pitchf", "sid"], &["audio"]);
        let names = io.resolve_rvc_io_names().unwrap();
        assert_eq!(names.feats, "feats");
        assert_eq!(names.p_len, "p_len");
        assert_eq!(names.pitchf, "pitchf");
        assert_eq!(names.audio, "audio");
        assert_eq!(io.feat_channels(&names.feats).unwrap(), 768);
    }

    #[test]
    fn resolves_converter_input_name_aliases() {
        // rvc-onnx-web converter export: phone/phone_lengths/nsff0 instead of the
        // vcclient feats/p_len/pitchf names.
        let io = rvc_io(
            &["phone", "phone_lengths", "pitch", "nsff0", "sid"],
            &["audio"],
        );
        let names = io.resolve_rvc_io_names().unwrap();
        assert_eq!(names.feats, "phone");
        assert_eq!(names.p_len, "phone_lengths");
        assert_eq!(names.pitch, "pitch");
        assert_eq!(names.pitchf, "nsff0");
        assert_eq!(names.sid, "sid");
        assert_eq!(io.feat_channels(&names.feats).unwrap(), 768);
    }

    #[test]
    fn resolves_single_bespoke_output_name() {
        let io = rvc_io(&["phone", "phone_lengths", "pitch", "nsff0", "sid"], &["o"]);
        assert_eq!(io.resolve_rvc_io_names().unwrap().audio, "o");
    }

    #[test]
    fn resolves_no_rnd_input_when_absent() {
        let io = rvc_io(&["feats", "p_len", "pitch", "pitchf", "sid"], &["audio"]);
        assert!(io.resolve_rvc_io_names().unwrap().rnd.is_none());
    }

    #[test]
    fn resolves_rnd_latent_noise_input() {
        // "latest" RVC export: phone/phone_lengths/pitch/pitchf/ds plus an `rnd`
        // latent-noise input shaped [1, inter_channels, frames].
        let mut io = rvc_io(
            &["phone", "phone_lengths", "pitch", "pitchf", "ds"],
            &["audio"],
        );
        io.inputs.push(TensorInfo {
            name: "rnd".to_string(),
            elem_type: 1,
            dims: vec![1, 192, 0],
        });
        let names = io.resolve_rvc_io_names().unwrap();
        assert_eq!(names.sid, "ds");
        let rnd = names.rnd.expect("rnd should resolve");
        assert_eq!(rnd.name, "rnd");
        assert_eq!(rnd.channels, 192);
    }

    #[test]
    fn rnd_without_static_channels_errors() {
        let mut io = rvc_io(&["feats", "p_len", "pitch", "pitchf", "sid"], &["audio"]);
        io.inputs.push(TensorInfo {
            name: "rnd".to_string(),
            elem_type: 1,
            dims: vec![1, 0, 0],
        });
        let err = io.resolve_rvc_io_names().unwrap_err().to_string();
        assert!(err.contains("rnd"), "{err}");
        assert!(err.contains("static channel count"), "{err}");
    }

    #[test]
    fn resolves_streaming_io_and_format() {
        // rvc-onnx-web streaming export: rnd + nsf_noise + phase_in inputs and a
        // phase_out output, with the rvc.* metadata keys.
        let mut io = rvc_io(
            &["phone", "phone_lengths", "pitch", "nsff0", "sid"],
            &["audio"],
        );
        io.inputs.push(TensorInfo {
            name: "rnd".to_string(),
            elem_type: 1,
            dims: vec![1, 192, 0],
        });
        io.inputs.push(TensorInfo {
            name: "nsf_noise".to_string(),
            elem_type: 1,
            dims: vec![1, 0, 1],
        });
        io.inputs.push(TensorInfo {
            name: "phase_in".to_string(),
            elem_type: 1,
            dims: vec![1, 1, 1],
        });
        io.outputs.push(TensorInfo {
            name: "phase_out".to_string(),
            elem_type: 1,
            dims: vec![1, 1, 1],
        });
        io.metadata = vec![
            ("rvc.export_mode".to_string(), "streaming".to_string()),
            ("rvc.stream_format_version".to_string(), "1".to_string()),
            ("rvc.frame_hop".to_string(), "480".to_string()),
            ("rvc.sample_rate".to_string(), "48000".to_string()),
        ];

        let names = io.resolve_rvc_io_names().unwrap();
        assert_eq!(names.pitchf, "nsff0");
        assert_eq!(names.nsf_noise.as_deref(), Some("nsf_noise"));
        assert_eq!(names.phase_in.as_deref(), Some("phase_in"));
        assert_eq!(names.phase_out.as_deref(), Some("phase_out"));

        let stream = io.stream_format().unwrap().expect("streaming format");
        assert_eq!(stream.version, 1);
        assert_eq!(stream.frame_hop, 480);
        assert_eq!(stream.sample_rate, 48_000);
        // Streaming exports record the rate as `rvc.sample_rate`, not a JSON blob.
        assert_eq!(io.rvc_sample_rate(), Some(48_000));
    }

    #[test]
    fn non_streaming_export_has_no_stream_format_or_extra_io() {
        let io = rvc_io(&["feats", "p_len", "pitch", "pitchf", "sid"], &["audio"]);
        let names = io.resolve_rvc_io_names().unwrap();
        assert!(names.nsf_noise.is_none());
        assert!(names.phase_in.is_none());
        assert!(names.phase_out.is_none());
        assert!(io.stream_format().unwrap().is_none());
    }

    #[test]
    fn streaming_export_missing_frame_hop_errors() {
        let mut io = rvc_io(&["feats", "p_len", "pitch", "pitchf", "sid"], &["audio"]);
        io.metadata = vec![
            ("rvc.export_mode".to_string(), "streaming".to_string()),
            ("rvc.stream_format_version".to_string(), "1".to_string()),
            ("rvc.sample_rate".to_string(), "48000".to_string()),
        ];
        let err = io.stream_format().unwrap_err().to_string();
        assert!(err.contains("rvc.frame_hop"), "{err}");
    }

    #[test]
    fn missing_rvc_input_errors_with_actual_names() {
        let io = rvc_io(&["phone", "pitch", "nsff0", "sid"], &["audio"]);
        let err = io.resolve_rvc_io_names().unwrap_err().to_string();
        assert!(err.contains("p_len"), "{err}");
        assert!(err.contains("phone"), "{err}");
    }

    #[test]
    fn symbolic_dim_reads_as_zero_channels() {
        // A value info whose last axis is a symbolic dim_param, not a dim_value.
        let mut dim = Vec::new();
        len_delimited(2, b"channels", &mut dim); // dim_param
        let mut shape_bytes = Vec::new();
        len_delimited(1, b"\x08\x01", &mut shape_bytes); // dim with dim_value=1
        len_delimited(1, &dim, &mut shape_bytes); // symbolic dim
        let tensor = {
            let mut out = Vec::new();
            varint_field(1, 1, &mut out);
            len_delimited(2, &shape_bytes, &mut out);
            out
        };
        let info = parse_value_info(&{
            let mut out = Vec::new();
            len_delimited(1, b"x", &mut out);
            len_delimited(
                2,
                &{
                    let mut t = Vec::new();
                    len_delimited(1, &tensor, &mut t);
                    t
                },
                &mut out,
            );
            out
        })
        .unwrap();
        assert_eq!(info.dims, vec![1, 0]);
        assert_eq!(info.last_dim_channels(), None);
    }
}
