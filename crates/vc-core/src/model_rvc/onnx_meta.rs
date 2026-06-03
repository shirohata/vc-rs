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

    /// RVC `feats` input channel count (the static last axis).
    pub(super) fn expected_feat_channels(&self) -> Result<i64> {
        let feats = self
            .input("feats")
            .ok_or_else(|| anyhow!("RVC model has no 'feats' input"))?;
        feats
            .last_dim_channels()
            .ok_or_else(|| anyhow!("RVC 'feats' input does not expose a static channel count"))
    }

    pub(super) fn validate_rvc_metadata(&self) -> Result<()> {
        if let Some(metadata) = self.metadata_value("metadata") {
            if !metadata.contains(r#""f0": 1"#) {
                bail!("RVC model metadata does not indicate f0=1: {metadata}");
            }
            info!("RVC metadata: {metadata}");
        }
        Ok(())
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
            validate_embedder_output_selection("requested embedder output", output, expected_channels)?;
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
            validate_embedder_output_selection("single embedder output", output, expected_channels)?;
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
        bail!("{label} '{}' must be a tensor, got {}", tensor.name, tensor.describe());
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
            io.inputs.push(parse_value_info(reader.read_len_prefixed()?)?);
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
        assert!(io.metadata_value("metadata").unwrap().contains(r#""f0": 1"#));
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
            len_delimited(2, &{ let mut t = Vec::new(); len_delimited(1, &tensor, &mut t); t }, &mut out);
            out
        })
        .unwrap();
        assert_eq!(info.dims, vec![1, 0]);
        assert_eq!(info.last_dim_channels(), None);
    }
}
