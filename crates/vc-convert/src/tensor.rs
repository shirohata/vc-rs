//! Tensor container for checkpoint weights.
//!
//! Ported from rvc-onnx-web (https://github.com/visgotti/rvc-onnx-web),
//! MIT License — the `TorchStorage`/`TensorData` types in `src/pickle.ts` /
//! `src/types.ts`.
//!
//! Like the TS source, narrow storage dtypes (float16/bfloat16/float64) are
//! widened to f32 at parse time, so a loaded [`Tensor`] only ever carries one
//! of the variants below.

use anyhow::{anyhow, Result};

/// Typed tensor payload. Mirrors the TS union
/// `Float32Array | Int32Array | Uint8Array | Int16Array | BigInt64Array`.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    I16(Vec<i16>),
    U8(Vec<u8>),
}

impl TensorData {
    pub fn len(&self) -> usize {
        match self {
            TensorData::F32(v) => v.len(),
            TensorData::I64(v) => v.len(),
            TensorData::I32(v) => v.len(),
            TensorData::I16(v) => v.len(),
            TensorData::U8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype_name(&self) -> &'static str {
        match self {
            TensorData::F32(_) => "float32",
            TensorData::I64(_) => "int64",
            TensorData::I32(_) => "int32",
            TensorData::I16(_) => "int16",
            TensorData::U8(_) => "uint8",
        }
    }

    /// Sub-range copy, used by `_rebuild_tensor_v2` to apply storage offsets.
    pub(crate) fn slice(&self, start: usize, len: usize) -> Result<TensorData> {
        let end = start
            .checked_add(len)
            .ok_or_else(|| anyhow!("tensor slice overflow"))?;
        if end > self.len() {
            return Err(anyhow!(
                "tensor slice {start}..{end} out of bounds (storage has {} elements)",
                self.len()
            ));
        }
        Ok(match self {
            TensorData::F32(v) => TensorData::F32(v[start..end].to_vec()),
            TensorData::I64(v) => TensorData::I64(v[start..end].to_vec()),
            TensorData::I32(v) => TensorData::I32(v[start..end].to_vec()),
            TensorData::I16(v) => TensorData::I16(v[start..end].to_vec()),
            TensorData::U8(v) => TensorData::U8(v[start..end].to_vec()),
        })
    }
}

/// A checkpoint weight: contiguous data plus its shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub data: TensorData,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Borrow the payload as f32, erroring for non-float tensors.
    pub fn as_f32(&self) -> Result<&[f32]> {
        match &self.data {
            TensorData::F32(v) => Ok(v),
            other => Err(anyhow!(
                "expected float32 tensor, found {}",
                other.dtype_name()
            )),
        }
    }
}
