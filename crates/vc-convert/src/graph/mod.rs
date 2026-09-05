//! Graph construction core: builder state + ONNX op helpers.
//!
//! Ported from rvc-onnx-web (https://github.com/visgotti/rvc-onnx-web),
//! MIT License — `src/onnx-builder.ts`. Where the TS threads `nodes`,
//! `initializers`, `addWeight`, `addInt64Const`… through every function,
//! this port centralizes that state in [`GraphBuilder`].

pub(crate) mod decoder;
pub(crate) mod flow;
pub(crate) mod nsf;
pub(crate) mod synthesizer;
pub(crate) mod text_encoder;
pub(crate) mod weight_norm;

use std::collections::{BTreeMap, HashSet};

use anyhow::{anyhow, Result};

use crate::onnx::{attr, Attr, DataType, Initializer, Node, ValueInfo};
use crate::tensor::{Tensor, TensorData};

pub(crate) struct GraphBuilder<'ckpt> {
    pub nodes: Vec<Node>,
    pub initializers: Vec<Initializer>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub weights: &'ckpt BTreeMap<String, Tensor>,
    added_initializers: HashSet<String>,
    name_counter: u64,
}

impl<'ckpt> GraphBuilder<'ckpt> {
    pub fn new(weights: &'ckpt BTreeMap<String, Tensor>) -> Self {
        GraphBuilder {
            nodes: Vec::new(),
            initializers: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            weights,
            added_initializers: HashSet::new(),
            name_counter: 0,
        }
    }

    /// `uniqueName` in the TS source.
    pub fn unique(&mut self, prefix: &str) -> String {
        let name = format!("{prefix}_{}", self.name_counter);
        self.name_counter += 1;
        name
    }

    pub fn push(
        &mut self,
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        attributes: Vec<crate::onnx::Attribute>,
    ) {
        let name = self.unique(&op_type.to_ascii_lowercase());
        self.nodes.push(Node {
            op_type: op_type.to_owned(),
            name,
            inputs: inputs.iter().map(|s| (*s).to_owned()).collect(),
            outputs: outputs.iter().map(|s| (*s).to_owned()).collect(),
            attributes,
        });
    }

    /// Redirect the most recent node writing `from` to write `to` instead
    /// (port of `renameNodeOutput`). Returns false when no node produced it.
    pub fn rename_node_output(&mut self, from: &str, to: &str) -> bool {
        for node in self.nodes.iter_mut().rev() {
            if let Some(slot) = node.outputs.iter_mut().find(|o| *o == from) {
                *slot = to.to_owned();
                return true;
            }
        }
        false
    }

    // ---- initializers ----

    /// Add a checkpoint weight as an initializer under its own name.
    pub fn add_weight(&mut self, name: &str) -> Result<String> {
        let tensor = self
            .weights
            .get(name)
            .ok_or_else(|| anyhow!("weight not found: {name}"))?;
        if self.added_initializers.insert(name.to_owned()) {
            self.initializers.push(Initializer {
                name: name.to_owned(),
                tensor: tensor.clone(),
            });
        }
        Ok(name.to_owned())
    }

    pub fn has_weight(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    pub fn weight_shape(&self, name: &str) -> Result<&[usize]> {
        self.weights
            .get(name)
            .map(|t| t.shape.as_slice())
            .ok_or_else(|| anyhow!("weight not found: {name}"))
    }

    /// Add an f32 initializer under `name` (TS `addConstant`).
    pub fn add_f32(&mut self, name: &str, data: Vec<f32>, shape: Vec<usize>) -> String {
        self.add_init(name, TensorData::F32(data), shape)
    }

    /// Scalar f32 initializer (TS `addScalar` / `addFloatConst`).
    pub fn add_scalar(&mut self, name: &str, value: f32) -> String {
        self.add_f32(name, vec![value], vec![])
    }

    /// Int64 initializer (TS `addInt64Const`).
    pub fn add_i64(&mut self, name: &str, values: Vec<i64>, shape: Vec<usize>) -> String {
        self.add_init(name, TensorData::I64(values), shape)
    }

    fn add_init(&mut self, name: &str, data: TensorData, shape: Vec<usize>) -> String {
        if self.added_initializers.insert(name.to_owned()) {
            self.initializers.push(Initializer {
                name: name.to_owned(),
                tensor: Tensor { data, shape },
            });
        }
        name.to_owned()
    }

    // ---- op helpers (explicit output names, mirroring the TS call sites) ----

    pub fn unary(&mut self, op: &str, input: &str, out: &str) {
        self.push(op, &[input], &[out], vec![]);
    }

    pub fn binary(&mut self, op: &str, a: &str, b: &str, out: &str) {
        self.push(op, &[a, b], &[out], vec![]);
    }

    /// Binary op with a builder-generated output name; the common pattern
    /// `out = uniqueName(prefix); nodes.push(op(a, b, out))`.
    pub fn binary_new(&mut self, op: &str, a: &str, b: &str, prefix: &str) -> String {
        let out = self.unique(prefix);
        self.binary(op, a, b, &out);
        out
    }

    pub fn unary_new(&mut self, op: &str, input: &str, prefix: &str) -> String {
        let out = self.unique(prefix);
        self.unary(op, input, &out);
        out
    }

    #[allow(clippy::too_many_arguments, reason = "mirrors the TS conv1d signature")]
    pub fn conv1d(
        &mut self,
        input: &str,
        weight: &str,
        bias: Option<&str>,
        out: &str,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) {
        let attrs = vec![
            attr("kernel_shape", Attr::Ints(vec![kernel_size as i64])),
            attr("strides", Attr::Ints(vec![stride as i64])),
            attr("pads", Attr::Ints(vec![padding as i64, padding as i64])),
            attr("dilations", Attr::Ints(vec![dilation as i64])),
            attr("group", Attr::Int(1)),
        ];
        match bias {
            Some(b) => self.push("Conv", &[input, weight, b], &[out], attrs),
            None => self.push("Conv", &[input, weight], &[out], attrs),
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the TS convTranspose1d signature"
    )]
    pub fn conv_transpose1d(
        &mut self,
        input: &str,
        weight: &str,
        bias: Option<&str>,
        out: &str,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) {
        let mut attrs = vec![
            attr("kernel_shape", Attr::Ints(vec![kernel_size as i64])),
            attr("strides", Attr::Ints(vec![stride as i64])),
            attr("pads", Attr::Ints(vec![padding as i64, padding as i64])),
            attr("dilations", Attr::Ints(vec![1])),
            attr("group", Attr::Int(1)),
        ];
        if output_padding != 0 {
            attrs.push(attr(
                "output_padding",
                Attr::Ints(vec![output_padding as i64]),
            ));
        }
        match bias {
            Some(b) => self.push("ConvTranspose", &[input, weight, b], &[out], attrs),
            None => self.push("ConvTranspose", &[input, weight], &[out], attrs),
        }
    }

    pub fn layer_norm(&mut self, input: &str, scale: &str, bias: &str, out: &str) {
        self.push(
            "LayerNormalization",
            &[input, scale, bias],
            &[out],
            vec![
                attr("axis", Attr::Int(-1)),
                attr("epsilon", Attr::Float(1e-5)),
            ],
        );
    }

    pub fn leaky_relu(&mut self, input: &str, out: &str, alpha: f32) {
        self.push(
            "LeakyRelu",
            &[input],
            &[out],
            vec![attr("alpha", Attr::Float(alpha))],
        );
    }

    pub fn softmax(&mut self, input: &str, out: &str, axis: i64) {
        self.push(
            "Softmax",
            &[input],
            &[out],
            vec![attr("axis", Attr::Int(axis))],
        );
    }

    pub fn transpose(&mut self, input: &str, out: &str, perm: Vec<i64>) {
        self.push(
            "Transpose",
            &[input],
            &[out],
            vec![attr("perm", Attr::Ints(perm))],
        );
    }

    pub fn reshape(&mut self, input: &str, shape: &str, out: &str) {
        self.push(
            "Reshape",
            &[input, shape],
            &[out],
            vec![attr("allowzero", Attr::Int(0))],
        );
    }

    pub fn unsqueeze(&mut self, input: &str, axes: &str, out: &str) {
        self.push("Unsqueeze", &[input, axes], &[out], vec![]);
    }

    pub fn concat(&mut self, inputs: &[&str], out: &str, axis: i64) {
        self.push(
            "Concat",
            inputs,
            &[out],
            vec![attr("axis", Attr::Int(axis))],
        );
    }

    pub fn split_sizes(&mut self, input: &str, sizes: &str, outputs: &[&str], axis: i64) {
        self.push(
            "Split",
            &[input, sizes],
            outputs,
            vec![attr("axis", Attr::Int(axis))],
        );
    }

    pub fn gather(&mut self, data: &str, indices: &str, out: &str, axis: i64) {
        self.push(
            "Gather",
            &[data, indices],
            &[out],
            vec![attr("axis", Attr::Int(axis))],
        );
    }

    pub fn slice(
        &mut self,
        input: &str,
        starts: &str,
        ends: &str,
        axes: &str,
        steps: &str,
        out: &str,
    ) {
        self.push("Slice", &[input, starts, ends, axes, steps], &[out], vec![]);
    }

    pub fn pad_constant(&mut self, input: &str, pads: &str, value: &str, out: &str) {
        self.push(
            "Pad",
            &[input, pads, value],
            &[out],
            vec![attr("mode", Attr::String("constant".to_owned()))],
        );
    }

    pub fn cast(&mut self, input: &str, out: &str, to: DataType) {
        self.push(
            "Cast",
            &[input],
            &[out],
            vec![attr("to", Attr::Int(to as i64))],
        );
    }

    pub fn range(&mut self, start: &str, limit: &str, delta: &str, out: &str) {
        self.push("Range", &[start, limit, delta], &[out], vec![]);
    }

    pub fn where_(&mut self, cond: &str, x: &str, y: &str, out: &str) {
        self.push("Where", &[cond, x, y], &[out], vec![]);
    }

    /// `Mod` with `fmod=1` (floating-point remainder).
    pub fn fmod(&mut self, a: &str, b: &str, out: &str) {
        self.push("Mod", &[a, b], &[out], vec![attr("fmod", Attr::Int(1))]);
    }

    pub fn cumsum(&mut self, input: &str, axis: &str, out: &str) {
        self.push("CumSum", &[input, axis], &[out], vec![]);
    }

    pub fn random_normal_like(&mut self, input: &str, out: &str) {
        self.push(
            "RandomNormalLike",
            &[input],
            &[out],
            vec![
                attr("mean", Attr::Float(0.0)),
                attr("scale", Attr::Float(1.0)),
                attr("dtype", Attr::Int(DataType::Float as i64)),
            ],
        );
    }

    /// Nearest-neighbor `Resize` along the time axis (empty roi input), with
    /// the exact attributes the TS emits — `asymmetric`/`floor` are
    /// load-bearing for matching torch's `interpolate(mode="nearest")`.
    pub fn resize_nearest(&mut self, input: &str, scales: &str, out: &str) {
        self.push(
            "Resize",
            &[input, "", scales],
            &[out],
            vec![
                attr("mode", Attr::String("nearest".to_owned())),
                attr(
                    "coordinate_transformation_mode",
                    Attr::String("asymmetric".to_owned()),
                ),
                attr("nearest_mode", Attr::String("floor".to_owned())),
            ],
        );
    }

    /// Value-info helper (TS `valueInfo`): dims are fixed sizes or symbolic
    /// parameter names.
    pub fn value_info(name: &str, elem_type: DataType, dims: &[DimSpec]) -> ValueInfo {
        ValueInfo {
            name: name.to_owned(),
            elem_type,
            shape: dims
                .iter()
                .map(|d| match d {
                    DimSpec::Fixed(v) => crate::onnx::Dim::Value(*v),
                    DimSpec::Sym(s) => crate::onnx::Dim::Param((*s).to_owned()),
                })
                .collect(),
        }
    }
}

/// Shape entry for declaring graph inputs/outputs.
pub(crate) enum DimSpec {
    Fixed(i64),
    Sym(&'static str),
}

pub(crate) use DimSpec::{Fixed, Sym};
