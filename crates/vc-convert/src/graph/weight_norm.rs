//! weight_norm reconstruction and linear-layer lowering.
//!
//! Ported from rvc-onnx-web (https://github.com/visgotti/rvc-onnx-web),
//! MIT License — `precomputeNormalizedWeight` / `buildWeightNormReconstruction`
//! / `addPretransposedLinearWeight` / `buildLinearNodes` in
//! `src/synthesizer-builder.ts`.

use anyhow::{anyhow, Result};

use crate::checkpoint::weight_norm_for;
use crate::onnx::Initializer;
use crate::tensor::{Tensor, TensorData};

use super::GraphBuilder;

/// Whether `base` is stored weight-normalized (either parametrization layout).
pub(crate) fn has_weight_norm(b: &GraphBuilder<'_>, base: &str) -> bool {
    weight_norm_for(b.weights, base).is_some()
}

/// Resolve a conv weight that may be weight-normalized: returns the tensor
/// name to feed the Conv node, adding a precomputed initializer when the
/// checkpoint stores the g/v decomposition.
///
/// The normalization is folded at conversion time (no ReduceSum/Sqrt nodes):
/// `weight = g * v / ||v||` per dim-0 slice, matching PyTorch's
/// `weight_norm(…, dim=0)` for both Conv1d and ConvTranspose1d.
pub(crate) fn conv_weight(b: &mut GraphBuilder<'_>, base: &str, prefix: &str) -> Result<String> {
    let Some(params) = weight_norm_for(b.weights, base) else {
        return b.add_weight(&format!("{base}.weight"));
    };
    let g = b
        .weights
        .get(&params.weight_g)
        .expect("checked by weight_norm_for");
    let v = b
        .weights
        .get(&params.weight_v)
        .expect("checked by weight_norm_for");
    let normalized = precompute_normalized_weight(g, v)?;
    let name = b.unique(prefix);
    b.initializers.push(Initializer {
        name: name.clone(),
        tensor: Tensor {
            data: TensorData::F32(normalized),
            shape: v.shape.clone(),
        },
    });
    Ok(name)
}

fn precompute_normalized_weight(weight_g: &Tensor, weight_v: &Tensor) -> Result<Vec<f32>> {
    let g = weight_g.as_f32()?;
    let v = weight_v.as_f32()?;
    let out_channels = *weight_v
        .shape
        .first()
        .ok_or_else(|| anyhow!("weight_v has no dimensions"))?;
    if out_channels == 0 || !v.len().is_multiple_of(out_channels) || g.len() < out_channels {
        return Err(anyhow!(
            "weight_norm shape mismatch: g has {} values, v is {:?}",
            g.len(),
            weight_v.shape
        ));
    }
    let channel_size = v.len() / out_channels;
    let mut result = vec![0.0f32; v.len()];
    for (slice_out, (slice_v, &g_val)) in result
        .chunks_exact_mut(channel_size)
        .zip(v.chunks_exact(channel_size).zip(g.iter()))
    {
        // f64 accumulation differs from the TS source only below f32
        // round-off; both match torch within 1 ulp.
        let sum_sq: f64 = slice_v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let norm = (sum_sq + 1e-12).sqrt() as f32;
        let scale = g_val / norm;
        for (out, &value) in slice_out.iter_mut().zip(slice_v) {
            *out = value * scale;
        }
    }
    Ok(result)
}

/// Lower `output = input @ W^T + bias` with the PyTorch `[out, in]` weight
/// pre-transposed to `[in, out]` at conversion time, so no runtime Transpose
/// runs on a constant (port of `buildLinearNodes` + the pretranspose path).
pub(crate) fn linear(
    b: &mut GraphBuilder<'_>,
    input: &str,
    weight_name: &str,
    bias: Option<&str>,
    output: &str,
) -> Result<()> {
    let tensor = b
        .weights
        .get(weight_name)
        .ok_or_else(|| anyhow!("weight not found: {weight_name}"))?;
    if tensor.shape.len() != 2 {
        return Err(anyhow!(
            "linear weight {weight_name} has shape {:?}, expected 2-D",
            tensor.shape
        ));
    }
    let (out_features, in_features) = (tensor.shape[0], tensor.shape[1]);
    let source = tensor.as_f32()?;
    let mut transposed = vec![0.0f32; source.len()];
    for out_idx in 0..out_features {
        for in_idx in 0..in_features {
            transposed[in_idx * out_features + out_idx] = source[out_idx * in_features + in_idx];
        }
    }
    let transposed_name = b.unique("weight_transposed");
    b.initializers.push(Initializer {
        name: transposed_name.clone(),
        tensor: Tensor {
            data: TensorData::F32(transposed),
            shape: vec![in_features, out_features],
        },
    });

    match bias {
        Some(bias) => {
            let matmul_out = b.unique("matmul_out");
            b.binary("MatMul", input, &transposed_name, &matmul_out);
            b.binary("Add", &matmul_out, bias, output);
        }
        None => {
            b.binary("MatMul", input, &transposed_name, output);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn normalizes_per_dim0_slice() {
        let g = Tensor {
            data: TensorData::F32(vec![2.0, 3.0]),
            shape: vec![2, 1, 1],
        };
        let v = Tensor {
            data: TensorData::F32(vec![3.0, 4.0, 0.0, 5.0]),
            shape: vec![2, 1, 2],
        };
        let w = precompute_normalized_weight(&g, &v).unwrap();
        // Row 0: norm 5 → scale 2/5; row 1: norm 5 → scale 3/5
        assert!((w[0] - 1.2).abs() < 1e-6);
        assert!((w[1] - 1.6).abs() < 1e-6);
        assert!((w[2] - 0.0).abs() < 1e-6);
        assert!((w[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn linear_pretransposes_weight() {
        let mut weights = BTreeMap::new();
        weights.insert(
            "l.weight".to_owned(),
            Tensor {
                data: TensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                shape: vec![2, 3], // [out=2, in=3]
            },
        );
        let mut b = GraphBuilder::new(&weights);
        linear(&mut b, "x", "l.weight", None, "y").unwrap();
        assert_eq!(b.initializers.len(), 1);
        let init = &b.initializers[0];
        assert_eq!(init.tensor.shape, vec![3, 2]);
        assert_eq!(
            init.tensor.data,
            TensorData::F32(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        );
        assert_eq!(b.nodes.len(), 1);
        assert_eq!(b.nodes[0].op_type, "MatMul");
    }
}
