//! PyTorch-specific unpickling: persistent-id storages, tensor
//! reconstructors, and dtype widening.
//!
//! Ported from rvc-onnx-web (https://github.com/visgotti/rvc-onnx-web),
//! MIT License — the `reduce`/`persistentLoad`/`createTypedArray` and
//! float16/bfloat16 conversion logic of `src/pickle.ts`.
//!
//! Deliberate trims from the TS source: complex, quantized, sparse, and
//! numpy payloads error instead of returning silent placeholders — an RVC
//! checkpoint never contains them, and a placeholder would only surface as
//! a confusing failure later in graph construction.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::{anyhow, bail, Result};

use crate::pickle::{PyObject, Storage, StorageResolver, Value};
use crate::tensor::{Tensor, TensorData};

/// Handle REDUCE for the reconstructors a torch checkpoint uses.
pub(crate) fn reduce(callable: &Value, args: &Value) -> Result<Value> {
    let Value::Global(global) = callable else {
        bail!(
            "REDUCE callable is {}, expected a global",
            callable.type_name()
        );
    };
    let args = match args {
        Value::Tuple(items) => items.as_ref(),
        other => bail!("REDUCE args are {}, expected a tuple", other.type_name()),
    };

    match global.full_name().as_str() {
        "torch._utils._rebuild_tensor_v2"
        | "torch._utils._rebuild_tensor_v3"
        | "torch._utils._rebuild_device_tensor_v2" => rebuild_tensor_v2(args),
        "torch._utils._rebuild_parameter" => {
            // args: (tensor, requires_grad, backward_hooks)
            Ok(args.first().cloned().unwrap_or(Value::None))
        }
        "collections.OrderedDict" | "builtins.dict" => {
            Ok(Value::Dict(Rc::new(RefCell::new(Vec::new()))))
        }
        "torch.Size" | "builtins.tuple" | "builtins.list" => Ok(args
            .first()
            .cloned()
            .unwrap_or(Value::Tuple(Rc::from(Vec::new())))),
        "_codecs.encode" => {
            // ("text", "latin-1") → bytes; used by some numpy-flavored saves
            let text = args
                .first()
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("_codecs.encode without a string argument"))?;
            let bytes: Vec<u8> = text.chars().map(|c| c as u32 as u8).collect();
            Ok(Value::Bytes(bytes.into()))
        }
        name if name.contains("sparse") || name.contains("_rebuild_qtensor") => {
            bail!("unsupported tensor kind in checkpoint: {name} (RVC checkpoints are dense)")
        }
        name if name.starts_with("numpy") => {
            bail!("numpy payloads are not supported in RVC checkpoints ({name})")
        }
        _ => Ok(Value::Object(Rc::new(RefCell::new(PyObject {
            module: global.module.clone(),
            name: global.name.clone(),
            args: args.to_vec(),
            state: None,
        })))),
    }
}

/// Handle NEWOBJ / NEWOBJ_EX.
pub(crate) fn newobj(cls: &Value, args: Value) -> Result<Value> {
    let Value::Global(global) = cls else {
        bail!("NEWOBJ class is {}, expected a global", cls.type_name());
    };
    // Storage classes constructed via NEWOBJ start empty; persistent ids
    // carry the real data.
    if global.name.contains("Storage") {
        return Ok(Value::Storage(Rc::new(Storage {
            data: TensorData::F32(Vec::new()),
        })));
    }
    let args = match args {
        Value::Tuple(items) => items.to_vec(),
        other => vec![other],
    };
    Ok(Value::Object(Rc::new(RefCell::new(PyObject {
        module: global.module.clone(),
        name: global.name.clone(),
        args,
        state: None,
    }))))
}

/// `torch._utils._rebuild_tensor_v2(storage, offset, shape, stride,
/// requires_grad, backward_hooks, [metadata])`.
fn rebuild_tensor_v2(args: &[Value]) -> Result<Value> {
    let [storage, offset, shape, stride, ..] = args else {
        bail!(
            "_rebuild_tensor_v2 expects at least 4 arguments, got {}",
            args.len()
        );
    };
    let Value::Storage(storage) = storage else {
        // TS returns null for missing storages; we surface it as None so the
        // checkpoint layer can skip the entry the same way.
        return Ok(Value::None);
    };
    let offset = usize::try_from(
        offset
            .as_int()
            .ok_or_else(|| anyhow!("_rebuild_tensor_v2: non-integer offset"))?,
    )
    .map_err(|_| anyhow!("_rebuild_tensor_v2: negative offset"))?;
    let shape = int_vec(shape).ok_or_else(|| anyhow!("_rebuild_tensor_v2: invalid shape"))?;
    let stride = int_vec(stride).ok_or_else(|| anyhow!("_rebuild_tensor_v2: invalid stride"))?;

    // The TS source ignores strides entirely; we instead refuse tensors whose
    // memory layout isn't contiguous, because reading them flat would silently
    // scramble the weight. (RVC checkpoints are always contiguous.)
    if !is_contiguous(&shape, &stride) {
        bail!(
            "non-contiguous tensor in checkpoint (shape {shape:?}, stride {stride:?}) \
             is not supported"
        );
    }

    let numel: usize = shape.iter().product::<usize>().max(1);
    let data = if offset > 0 || numel < storage.data.len() {
        storage.data.slice(offset, numel)?
    } else {
        storage.data.clone()
    };
    Ok(Value::Tensor(Rc::new(Tensor { data, shape })))
}

fn int_vec(value: &Value) -> Option<Vec<usize>> {
    let items = match value {
        Value::Tuple(items) => items.to_vec(),
        Value::List(items) => items.borrow().clone(),
        _ => return None,
    };
    items
        .iter()
        .map(|v| v.as_int().and_then(|i| usize::try_from(i).ok()))
        .collect()
}

/// Contiguity in the row-major sense: each dim's stride equals the product of
/// the later dims' sizes (size-1 dims may carry arbitrary strides).
fn is_contiguous(shape: &[usize], stride: &[usize]) -> bool {
    if stride.len() != shape.len() {
        // torch always saves matching-rank strides; treat a mismatch as
        // contiguous like the TS port (it never looks at strides).
        return stride.is_empty();
    }
    let mut expected = 1usize;
    for (&dim, &s) in shape.iter().zip(stride.iter()).rev() {
        if dim != 1 && s != expected {
            return false;
        }
        expected = expected.saturating_mul(dim.max(1));
    }
    true
}

/// Handle a persistent id: `("storage", storage_type, key, location, numel)`.
pub(crate) fn persistent_load(pid: &Value, resolver: &StorageResolver<'_>) -> Result<Value> {
    let items = match pid {
        Value::Tuple(items) => items.as_ref(),
        other => bail!("unsupported persistent id of type {}", other.type_name()),
    };
    if items.first().and_then(Value::as_str) != Some("storage") {
        bail!("unsupported persistent id (expected a torch storage tuple)");
    }
    let [_tag, storage_type, key, _location, numel, ..] = items else {
        bail!(
            "torch storage persistent id has {} elements, expected 5",
            items.len()
        );
    };
    let Value::Global(storage_type) = storage_type else {
        bail!(
            "storage type is {}, expected a global",
            storage_type.type_name()
        );
    };
    let key = key
        .as_str()
        .ok_or_else(|| anyhow!("storage key is not a string"))?;
    let numel = usize::try_from(
        numel
            .as_int()
            .ok_or_else(|| anyhow!("storage element count is not an integer"))?,
    )
    .map_err(|_| anyhow!("negative storage element count"))?;

    let dtype = dtype_from_storage_type(&storage_type.name);
    let raw = resolver(key).map_err(|e| anyhow!("storage {key} not found: {e}"))?;
    let data =
        widen_storage(dtype, &raw, numel).map_err(|e| anyhow!("storage {key} ({dtype}): {e}"))?;
    Ok(Value::Storage(Rc::new(Storage { data })))
}

/// Map a torch storage class name to a dtype string. Port of
/// `getDtypeFromStorageType` — the match order matters (e.g. "bfloat16"
/// before "float16" before "float").
fn dtype_from_storage_type(name: &str) -> &'static str {
    let name = name.to_ascii_lowercase();
    if name.contains("qint8") || name.contains("quint8") || name.contains("qint32") {
        return "quantized";
    }
    if name.contains("complex") {
        return "complex";
    }
    if name.contains("bfloat") {
        return "bfloat16";
    }
    if name.contains("half") || name.contains("float16") {
        return "float16";
    }
    if name.contains("double") || name.contains("float64") {
        return "float64";
    }
    if name.contains("float") {
        return "float32";
    }
    if name.contains("long") || name.contains("int64") {
        return "int64";
    }
    if name.contains("short") || name.contains("int16") {
        return "int16";
    }
    if name.contains("char") || name.contains("int8") {
        return "int8";
    }
    if name.contains("int") {
        return "int32";
    }
    if name.contains("byte") || name.contains("uint8") {
        return "uint8";
    }
    if name.contains("bool") {
        return "bool";
    }
    "float32"
}

/// Decode `numel` elements of `dtype` from little-endian bytes, widening
/// float16/bfloat16/float64 to f32 like the TS `createTypedArray`.
fn widen_storage(dtype: &str, raw: &[u8], numel: usize) -> Result<TensorData> {
    fn chunks<const N: usize>(
        raw: &[u8],
        numel: usize,
    ) -> Result<impl Iterator<Item = [u8; N]> + '_> {
        let needed = numel
            .checked_mul(N)
            .ok_or_else(|| anyhow!("storage size overflow"))?;
        if raw.len() < needed {
            bail!(
                "storage has {} bytes, expected at least {needed}",
                raw.len()
            );
        }
        Ok(raw[..needed]
            .chunks_exact(N)
            .map(|c| c.try_into().expect("chunk size")))
    }

    Ok(match dtype {
        "float32" => TensorData::F32(chunks::<4>(raw, numel)?.map(f32::from_le_bytes).collect()),
        "float64" => TensorData::F32(
            chunks::<8>(raw, numel)?
                .map(|b| f64::from_le_bytes(b) as f32)
                .collect(),
        ),
        "float16" => TensorData::F32(
            chunks::<2>(raw, numel)?
                .map(|b| f16_to_f32(u16::from_le_bytes(b)))
                .collect(),
        ),
        "bfloat16" => TensorData::F32(
            chunks::<2>(raw, numel)?
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(b)) << 16))
                .collect(),
        ),
        "int64" => TensorData::I64(chunks::<8>(raw, numel)?.map(i64::from_le_bytes).collect()),
        "int32" => TensorData::I32(chunks::<4>(raw, numel)?.map(i32::from_le_bytes).collect()),
        "int16" => TensorData::I16(chunks::<2>(raw, numel)?.map(i16::from_le_bytes).collect()),
        "uint8" | "bool" => TensorData::U8(chunks::<1>(raw, numel)?.map(|[b]| b).collect()),
        other => bail!("unsupported storage dtype {other} in checkpoint"),
    })
}

/// IEEE 754 half → single. Same math as the TS `convertFloat16ToFloat32`.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exponent = (h & 0x7c00) >> 10;
    let fraction = h & 0x03ff;
    match exponent {
        0 => {
            if fraction == 0 {
                sign * 0.0
            } else {
                sign * 2f32.powi(-14) * (f32::from(fraction) / 1024.0)
            }
        }
        0x1f => {
            if fraction != 0 {
                f32::NAN
            } else {
                sign * f32::INFINITY
            }
        }
        _ => sign * 2f32.powi(i32::from(exponent) - 15) * (1.0 + f32::from(fraction) / 1024.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_vectors() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95); // ~1/3
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        // Subnormal: smallest positive half = 2^-24
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    }

    #[test]
    fn bf16_and_f64_widen() {
        // bf16 of 1.5 is the top 16 bits of 0x3FC00000
        let raw = 0x3FC0u16.to_le_bytes();
        let TensorData::F32(v) = widen_storage("bfloat16", &raw, 1).unwrap() else {
            panic!();
        };
        assert_eq!(v, vec![1.5]);

        let raw = 2.5f64.to_le_bytes();
        let TensorData::F32(v) = widen_storage("float64", &raw, 1).unwrap() else {
            panic!();
        };
        assert_eq!(v, vec![2.5]);
    }

    #[test]
    fn storage_type_names_map_in_order() {
        assert_eq!(dtype_from_storage_type("FloatStorage"), "float32");
        assert_eq!(dtype_from_storage_type("HalfStorage"), "float16");
        assert_eq!(dtype_from_storage_type("BFloat16Storage"), "bfloat16");
        assert_eq!(dtype_from_storage_type("DoubleStorage"), "float64");
        assert_eq!(dtype_from_storage_type("LongStorage"), "int64");
        assert_eq!(dtype_from_storage_type("IntStorage"), "int32");
        assert_eq!(dtype_from_storage_type("ByteStorage"), "uint8");
        assert_eq!(dtype_from_storage_type("BoolStorage"), "bool");
    }

    #[test]
    fn contiguity_check() {
        assert!(is_contiguous(&[2, 3], &[3, 1]));
        assert!(is_contiguous(&[1, 3], &[99, 1])); // size-1 dim: any stride
        assert!(is_contiguous(&[], &[]));
        assert!(!is_contiguous(&[2, 3], &[1, 2])); // transposed view
    }

    #[test]
    fn rejects_quantized_storage() {
        let err = widen_storage("quantized", &[], 0).unwrap_err().to_string();
        assert!(err.contains("unsupported storage dtype"), "{err}");
    }
}
