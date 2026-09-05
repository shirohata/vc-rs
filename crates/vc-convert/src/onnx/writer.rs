//! ONNX protobuf serializer (protobuf v3 wire format, no dependencies).
//!
//! Ported from rvc-onnx-web (https://github.com/visgotti/rvc-onnx-web),
//! MIT License — `src/onnx-serializer.ts`.
//!
//! The field-number constants below are the writer-side mirror of the
//! reader in `vc-core/src/model_rvc/onnx_meta.rs`; keep the two in sync
//! with onnx.proto if either grows.

use crate::tensor::TensorData;

use super::{Attr, Attribute, Dim, Graph, Initializer, Model, Node, ValueInfo};

/// Serialize a model to ONNX protobuf bytes.
pub(crate) fn serialize(model: &Model) -> Vec<u8> {
    let mut w = Vec::new();
    write_model(&mut w, model);
    w
}

// Wire types
const VARINT: u64 = 0;
const LEN: u64 = 2;
const FIXED32: u64 = 5;

fn write_varint(w: &mut Vec<u8>, mut v: u64) {
    while v > 0x7f {
        w.push((v & 0x7f) as u8 | 0x80);
        v >>= 7;
    }
    w.push(v as u8);
}

/// Negative int64 values encode as their 64-bit two's complement varint.
fn write_varint_i64(w: &mut Vec<u8>, v: i64) {
    write_varint(w, v as u64);
}

fn write_tag(w: &mut Vec<u8>, field: u64, wire: u64) {
    write_varint(w, (field << 3) | wire);
}

fn write_varint_field(w: &mut Vec<u8>, field: u64, v: i64) {
    write_tag(w, field, VARINT);
    write_varint_i64(w, v);
}

fn write_bytes_field(w: &mut Vec<u8>, field: u64, bytes: &[u8]) {
    write_tag(w, field, LEN);
    write_varint(w, bytes.len() as u64);
    w.extend_from_slice(bytes);
}

fn write_string(w: &mut Vec<u8>, field: u64, s: &str) {
    write_bytes_field(w, field, s.as_bytes());
}

fn write_float_field(w: &mut Vec<u8>, field: u64, v: f32) {
    write_tag(w, field, FIXED32);
    w.extend_from_slice(&v.to_le_bytes());
}

fn write_message(w: &mut Vec<u8>, field: u64, body: impl FnOnce(&mut Vec<u8>)) {
    let mut sub = Vec::new();
    body(&mut sub);
    write_bytes_field(w, field, &sub);
}

// ModelProto fields (onnx.proto)
mod model_fields {
    pub const IR_VERSION: u64 = 1;
    pub const PRODUCER_NAME: u64 = 2;
    pub const PRODUCER_VERSION: u64 = 3;
    pub const GRAPH: u64 = 7;
    pub const OPSET_IMPORT: u64 = 8;
    pub const METADATA_PROPS: u64 = 14;
}

fn write_model(w: &mut Vec<u8>, model: &Model) {
    write_varint_field(w, model_fields::IR_VERSION, model.ir_version);
    if !model.producer_name.is_empty() {
        write_string(w, model_fields::PRODUCER_NAME, &model.producer_name);
    }
    if !model.producer_version.is_empty() {
        write_string(w, model_fields::PRODUCER_VERSION, &model.producer_version);
    }
    // OperatorSetIdProto: domain=1 (empty = default ONNX domain), version=2
    write_message(w, model_fields::OPSET_IMPORT, |w| {
        write_varint_field(w, 2, model.opset_version);
    });
    write_message(w, model_fields::GRAPH, |w| write_graph(w, &model.graph));
    // StringStringEntryProto: key=1, value=2
    for (key, value) in &model.metadata_props {
        write_message(w, model_fields::METADATA_PROPS, |w| {
            write_string(w, 1, key);
            write_string(w, 2, value);
        });
    }
}

// GraphProto fields
mod graph_fields {
    pub const NODE: u64 = 1;
    pub const NAME: u64 = 2;
    pub const INITIALIZER: u64 = 5;
    pub const INPUT: u64 = 11;
    pub const OUTPUT: u64 = 12;
}

fn write_graph(w: &mut Vec<u8>, graph: &Graph) {
    if !graph.name.is_empty() {
        write_string(w, graph_fields::NAME, &graph.name);
    }
    for node in &graph.nodes {
        write_message(w, graph_fields::NODE, |w| write_node(w, node));
    }
    for input in &graph.inputs {
        write_message(w, graph_fields::INPUT, |w| write_value_info(w, input));
    }
    for output in &graph.outputs {
        write_message(w, graph_fields::OUTPUT, |w| write_value_info(w, output));
    }
    for init in &graph.initializers {
        write_message(w, graph_fields::INITIALIZER, |w| write_initializer(w, init));
    }
}

// NodeProto fields
mod node_fields {
    pub const INPUT: u64 = 1;
    pub const OUTPUT: u64 = 2;
    pub const NAME: u64 = 3;
    pub const OP_TYPE: u64 = 4;
    pub const ATTRIBUTE: u64 = 5;
}

fn write_node(w: &mut Vec<u8>, node: &Node) {
    for input in &node.inputs {
        write_string(w, node_fields::INPUT, input);
    }
    for output in &node.outputs {
        write_string(w, node_fields::OUTPUT, output);
    }
    if !node.name.is_empty() {
        write_string(w, node_fields::NAME, &node.name);
    }
    write_string(w, node_fields::OP_TYPE, &node.op_type);
    for attribute in &node.attributes {
        write_message(w, node_fields::ATTRIBUTE, |w| write_attribute(w, attribute));
    }
}

// AttributeProto fields
mod attr_fields {
    pub const NAME: u64 = 1;
    pub const F: u64 = 2;
    pub const I: u64 = 3;
    pub const S: u64 = 4;
    pub const INTS: u64 = 8;
    pub const TYPE: u64 = 20;
}

// AttributeProto.AttributeType values
mod attr_type {
    pub const FLOAT: i64 = 1;
    pub const INT: i64 = 2;
    pub const STRING: i64 = 3;
    pub const INTS: i64 = 7;
}

fn write_attribute(w: &mut Vec<u8>, attribute: &Attribute) {
    write_string(w, attr_fields::NAME, &attribute.name);
    match &attribute.value {
        Attr::Int(v) => {
            write_varint_field(w, attr_fields::TYPE, attr_type::INT);
            write_varint_field(w, attr_fields::I, *v);
        }
        Attr::Ints(values) => {
            write_varint_field(w, attr_fields::TYPE, attr_type::INTS);
            // The TS serializer writes these unpacked; both encodings are
            // valid proto3, keep the same bytes.
            for v in values {
                write_varint_field(w, attr_fields::INTS, *v);
            }
        }
        Attr::Float(v) => {
            write_varint_field(w, attr_fields::TYPE, attr_type::FLOAT);
            write_float_field(w, attr_fields::F, *v);
        }
        Attr::String(s) => {
            write_varint_field(w, attr_fields::TYPE, attr_type::STRING);
            write_bytes_field(w, attr_fields::S, s.as_bytes());
        }
    }
}

// ValueInfoProto: name=1, type=2. TypeProto.tensor_type=1;
// TypeProto.Tensor.elem_type=1, .shape=2; TensorShapeProto.dim=1;
// Dimension.dim_value=1, .dim_param=2.
fn write_value_info(w: &mut Vec<u8>, info: &ValueInfo) {
    write_string(w, 1, &info.name);
    write_message(w, 2, |w| {
        write_message(w, 1, |w| {
            write_varint_field(w, 1, info.elem_type as i64);
            write_message(w, 2, |w| {
                for dim in &info.shape {
                    write_message(w, 1, |w| match dim {
                        Dim::Value(v) => write_varint_field(w, 1, *v),
                        Dim::Param(p) => write_string(w, 2, p),
                    });
                }
            });
        });
    });
}

// TensorProto fields
mod tensor_fields {
    pub const DIMS: u64 = 1;
    pub const DATA_TYPE: u64 = 2;
    pub const NAME: u64 = 8;
    pub const RAW_DATA: u64 = 9;
}

fn write_initializer(w: &mut Vec<u8>, init: &Initializer) {
    write_tensor(w, &init.name, &init.tensor);
}

fn write_tensor(w: &mut Vec<u8>, name: &str, tensor: &crate::tensor::Tensor) {
    for &dim in &tensor.shape {
        write_varint_field(w, tensor_fields::DIMS, dim as i64);
    }
    write_varint_field(
        w,
        tensor_fields::DATA_TYPE,
        super::DataType::for_data(&tensor.data) as i64,
    );
    write_string(w, tensor_fields::NAME, name);
    write_bytes_field(w, tensor_fields::RAW_DATA, &tensor_bytes(&tensor.data));
}

/// Little-endian raw_data encoding (ONNX raw_data is always little-endian).
fn tensor_bytes(data: &TensorData) -> Vec<u8> {
    match data {
        TensorData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        TensorData::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        TensorData::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        TensorData::I16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        TensorData::U8(v) => v.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::DataType;
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn serializes_minimal_model_bytes() {
        // A model with just ir_version + opset + one-node graph; the byte
        // layout is verified by hand against the proto3 wire format.
        let model = Model {
            ir_version: 8,
            opset_version: 20,
            producer_name: String::new(),
            producer_version: String::new(),
            metadata_props: vec![],
            graph: Graph {
                name: "g".to_owned(),
                nodes: vec![Node {
                    op_type: "Identity".to_owned(),
                    name: String::new(),
                    inputs: vec!["x".to_owned()],
                    outputs: vec!["y".to_owned()],
                    attributes: vec![],
                }],
                inputs: vec![ValueInfo {
                    name: "x".to_owned(),
                    elem_type: DataType::Float,
                    shape: vec![Dim::Value(1), Dim::Param("n".to_owned())],
                }],
                outputs: vec![ValueInfo {
                    name: "y".to_owned(),
                    elem_type: DataType::Float,
                    shape: vec![Dim::Value(1), Dim::Param("n".to_owned())],
                }],
                initializers: vec![],
            },
        };
        let bytes = serialize(&model);

        // ir_version: field 1 varint → 0x08 0x08
        assert_eq!(&bytes[..2], &[0x08, 0x08]);
        // opset_import: field 8 LEN → tag 0x42, len 2, {field2 varint 20}
        assert_eq!(&bytes[2..6], &[0x42, 0x02, 0x10, 0x14]);
        // graph: field 7 LEN → tag 0x3A
        assert_eq!(bytes[6], 0x3A);
    }

    #[test]
    fn initializer_raw_data_is_little_endian() {
        let mut w = Vec::new();
        write_initializer(
            &mut w,
            &Initializer {
                name: "w".to_owned(),
                tensor: Tensor {
                    data: TensorData::I64(vec![-1]),
                    shape: vec![1],
                },
            },
        );
        // dims=1 → 0x08 0x01 ; data_type=7 → 0x10 0x07 ; name → 0x42 0x01 'w'
        // raw_data → 0x4A 0x08 then eight 0xFF bytes (-1 LE)
        assert_eq!(
            w,
            vec![
                0x08, 0x01, 0x10, 0x07, 0x42, 0x01, b'w', 0x4A, 0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                0xFF, 0xFF, 0xFF
            ]
        );
    }

    #[test]
    fn negative_attribute_int_uses_two_complement_varint() {
        let mut w = Vec::new();
        write_attribute(
            &mut w,
            &Attribute {
                name: "axis".to_owned(),
                value: Attr::Int(-1),
            },
        );
        // -1 as u64 = 10-byte varint 0xFF…0x01
        let tail = &w[w.len() - 10..];
        assert_eq!(
            tail,
            &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]
        );
    }
}
