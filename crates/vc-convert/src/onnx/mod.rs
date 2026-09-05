//! In-memory ONNX model representation.
//!
//! Ported from rvc-onnx-web (https://github.com/visgotti/rvc-onnx-web),
//! MIT License — the `OnnxModel`/`OnnxGraph`/`OnnxNode`… interfaces in
//! `src/types.ts`.

pub(crate) mod writer;

use crate::tensor::{Tensor, TensorData};

/// The subset of `TensorProto.DataType` the converter emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DataType {
    Float = 1,
    Uint8 = 2,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
}

impl DataType {
    pub fn for_data(data: &TensorData) -> DataType {
        match data {
            TensorData::F32(_) => DataType::Float,
            TensorData::I64(_) => DataType::Int64,
            TensorData::I32(_) => DataType::Int32,
            TensorData::I16(_) => DataType::Int16,
            TensorData::U8(_) => DataType::Uint8,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Attr {
    Int(i64),
    Ints(Vec<i64>),
    Float(f32),
    String(String),
}

#[derive(Clone, Debug)]
pub(crate) struct Attribute {
    pub name: String,
    pub value: Attr,
}

pub(crate) fn attr(name: &str, value: Attr) -> Attribute {
    Attribute {
        name: name.to_owned(),
        value,
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub op_type: String,
    pub name: String,
    /// Empty string = omitted optional input, like ONNX convention.
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<Attribute>,
}

/// One dimension of a value-info shape: fixed or symbolic.
#[derive(Clone, Debug)]
pub(crate) enum Dim {
    Value(i64),
    Param(String),
}

#[derive(Clone, Debug)]
pub(crate) struct ValueInfo {
    pub name: String,
    pub elem_type: DataType,
    pub shape: Vec<Dim>,
}

#[derive(Clone, Debug)]
pub(crate) struct Initializer {
    pub name: String,
    pub tensor: Tensor,
}

#[derive(Debug, Default)]
pub(crate) struct Graph {
    pub name: String,
    pub nodes: Vec<Node>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub initializers: Vec<Initializer>,
}

#[derive(Debug)]
pub(crate) struct Model {
    pub ir_version: i64,
    pub opset_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub metadata_props: Vec<(String, String)>,
    pub graph: Graph,
}
