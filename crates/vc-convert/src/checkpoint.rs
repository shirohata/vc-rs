//! `.pth` checkpoint parsing: ZIP → pickle → structured RVC checkpoint.
//!
//! Ported from rvc-onnx-web (https://github.com/visgotti/rvc-onnx-web),
//! MIT License — `src/pth-parser.ts` (`parsePth`, `parseConfigArray`).

use std::collections::BTreeMap;

use anyhow::{anyhow, bail, Context, Result};

use crate::pickle::{Unpickler, Value};
use crate::tensor::Tensor;
use crate::zip::ZipArchive;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RvcVersion {
    V1,
    V2,
}

impl RvcVersion {
    pub fn as_str(self) -> &'static str {
        match self {
            RvcVersion::V1 => "v1",
            RvcVersion::V2 => "v2",
        }
    }

    /// ContentVec feature width the generator consumes.
    pub fn hidden_dim(self) -> usize {
        match self {
            RvcVersion::V1 => 256,
            RvcVersion::V2 => 768,
        }
    }
}

/// The 18-element RVC `config` array, structured. Field order matches
/// `parseConfigArray` in the TS source.
#[derive(Clone, Debug)]
pub struct RvcConfig {
    pub spec_channels: usize,
    pub segment_size: usize,
    pub inter_channels: usize,
    pub hidden_channels: usize,
    pub filter_channels: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub kernel_size: usize,
    pub p_dropout: f64,
    pub resblock: String,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub upsample_rates: Vec<usize>,
    pub upsample_initial_channel: usize,
    pub upsample_kernel_sizes: Vec<usize>,
    pub spk_embed_dim: usize,
    pub gin_channels: usize,
    pub sr: u32,
}

impl RvcConfig {
    /// Samples of audio produced per content frame (product of the
    /// upsample rates); the streaming metadata calls this the frame hop.
    pub fn frame_hop(&self) -> usize {
        self.upsample_rates.iter().product()
    }
}

#[derive(Debug)]
pub struct ParsedCheckpoint {
    pub config: RvcConfig,
    /// BTreeMap so iteration (and therefore initializer order in the ONNX
    /// output) is deterministic.
    pub weights: BTreeMap<String, Tensor>,
    pub use_f0: bool,
    pub version: RvcVersion,
    pub vocoder: String,
}

impl ParsedCheckpoint {
    pub fn weight(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .ok_or_else(|| anyhow!("checkpoint is missing weight {name}"))
    }
}

/// Parse a `.pth` buffer into weights + config.
pub fn parse_pth(bytes: &[u8]) -> Result<ParsedCheckpoint> {
    if bytes.len() < 4 || &bytes[..2] != b"PK" {
        bail!(
            "not a PyTorch ≥1.6 checkpoint (ZIP archive); \
             legacy pickle-only .pth files are not supported"
        );
    }
    let mut archive = ZipArchive::parse(bytes).context("failed to read .pth archive")?;

    // Locate the pickle (usually "archive/data.pkl") and derive the tensor
    // data prefix ("archive/data/").
    let pickle_name = archive
        .entry_names()
        .find(|n| n.ends_with("data.pkl"))
        .or_else(|| archive.entry_names().find(|n| n.ends_with(".pkl")))
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("no pickle file found in .pth archive"))?;
    let dir = &pickle_name[..pickle_name.rfind('/').map_or(0, |i| i + 1)];
    let data_prefix = format!("{dir}data/");

    let pickle_bytes = archive.read_pickle(&pickle_name)?;
    let mut resolver = |key: &str| {
        for candidate in [
            format!("{data_prefix}{key}"),
            format!("data/{key}"),
            format!("archive/data/{key}"),
            key.to_owned(),
        ] {
            if archive.has_entry(&candidate) {
                return archive.read(&candidate);
            }
        }
        Err(anyhow!("storage key not found in archive: {key}"))
    };
    let checkpoint = Unpickler::new(&pickle_bytes, &mut resolver)
        .load()
        .context("failed to unpickle checkpoint")?;

    let config_value = checkpoint.dict_get("config");
    let weight_value = checkpoint.dict_get("weight");
    let (Some(config_value), Some(weight_value)) = (config_value, weight_value) else {
        bail!(
            "invalid checkpoint: missing 'config' or 'weight' keys; \
             this may not be an RVC model"
        );
    };

    let mut config = parse_config_array(&config_value)?;

    let mut weights = BTreeMap::new();
    if let Value::Dict(entries) = &weight_value {
        for (key, value) in entries.borrow().iter() {
            let Some(name) = key.as_str() else { continue };
            // Rebuilds of missing storages come back as None; skip like the TS.
            if let Value::Tensor(tensor) = value {
                weights.insert(name.to_owned(), (**tensor).clone());
            }
        }
    } else {
        bail!("invalid checkpoint: 'weight' is not a dict");
    }

    let use_f0 = match checkpoint.dict_get("f0") {
        None | Some(Value::None) => true, // TS: Boolean(checkpoint.f0 ?? 1)
        Some(v) => v.as_int().map(|i| i != 0).unwrap_or(true),
    };
    let version = match checkpoint.dict_get("version") {
        Some(v) if v.as_str() == Some("v2") => RvcVersion::V2,
        _ => RvcVersion::V1,
    };
    let vocoder = checkpoint
        .dict_get("vocoder")
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_else(|| "HiFi-GAN".to_owned());

    // Speaker count comes from the actual embedding table when present.
    if let Some(emb) = weights.get("emb_g.weight") {
        if let Some(&rows) = emb.shape.first() {
            config.spk_embed_dim = rows;
        }
    }

    Ok(ParsedCheckpoint {
        config,
        weights,
        use_f0,
        version,
        vocoder,
    })
}

fn parse_config_array(value: &Value) -> Result<RvcConfig> {
    let items = match value {
        Value::List(items) => items.borrow().clone(),
        Value::Tuple(items) => items.to_vec(),
        other => bail!(
            "checkpoint config is {}, expected a list",
            other.type_name()
        ),
    };
    if items.len() < 18 {
        bail!(
            "checkpoint config has {} elements, expected 18",
            items.len()
        );
    }

    let num = |i: usize| -> Result<usize> {
        items[i]
            .as_f64()
            .map(|f| f as usize)
            .ok_or_else(|| anyhow!("config[{i}] is {}, expected a number", items[i].type_name()))
    };
    let list = |i: usize| -> Result<Vec<usize>> {
        int_list(&items[i]).ok_or_else(|| {
            anyhow!(
                "config[{i}] is {}, expected a list of ints",
                items[i].type_name()
            )
        })
    };

    Ok(RvcConfig {
        spec_channels: num(0)?,
        segment_size: num(1)?,
        inter_channels: num(2)?,
        hidden_channels: num(3)?,
        filter_channels: num(4)?,
        n_heads: num(5)?,
        n_layers: num(6)?,
        kernel_size: num(7)?,
        p_dropout: items[8]
            .as_f64()
            .ok_or_else(|| anyhow!("config[8] (p_dropout) is not a number"))?,
        resblock: items[9]
            .as_str()
            .map(str::to_owned)
            .or_else(|| items[9].as_f64().map(|f| format!("{f}")))
            .ok_or_else(|| anyhow!("config[9] (resblock) is not a string"))?,
        resblock_kernel_sizes: list(10)?,
        resblock_dilation_sizes: nested_int_list(&items[11])
            .ok_or_else(|| anyhow!("config[11] (resblock_dilation_sizes) is invalid"))?,
        upsample_rates: list(12)?,
        upsample_initial_channel: num(13)?,
        upsample_kernel_sizes: list(14)?,
        spk_embed_dim: num(15)?,
        gin_channels: num(16)?,
        sr: u32::try_from(num(17)?).map_err(|_| anyhow!("config[17] (sr) out of range"))?,
    })
}

fn int_list(value: &Value) -> Option<Vec<usize>> {
    let items = match value {
        Value::List(items) => items.borrow().clone(),
        Value::Tuple(items) => items.to_vec(),
        _ => return None,
    };
    items
        .iter()
        .map(|v| v.as_f64().map(|f| f as usize))
        .collect()
}

fn nested_int_list(value: &Value) -> Option<Vec<Vec<usize>>> {
    let items = match value {
        Value::List(items) => items.borrow().clone(),
        Value::Tuple(items) => items.to_vec(),
        _ => return None,
    };
    items.iter().map(int_list).collect()
}

// =============================================================================
// Weight-norm detection (port of detectWeightNorm in pth-parser.ts)
// =============================================================================

/// A weight-normalized layer's parameter pair.
pub(crate) struct WeightNormParams {
    /// Magnitude tensor name (`weight_g` / `…original0`).
    pub weight_g: String,
    /// Direction tensor name (`weight_v` / `…original1`).
    pub weight_v: String,
}

/// Find the weight_norm parameter pair for `base` (e.g. "enc_p.encoder.…"),
/// checking the modern parametrizations layout first, then the legacy one.
pub(crate) fn weight_norm_for(
    weights: &BTreeMap<String, Tensor>,
    base: &str,
) -> Option<WeightNormParams> {
    let modern_g = format!("{base}.parametrizations.weight.original0");
    let modern_v = format!("{base}.parametrizations.weight.original1");
    if weights.contains_key(&modern_g) && weights.contains_key(&modern_v) {
        return Some(WeightNormParams {
            weight_g: modern_g,
            weight_v: modern_v,
        });
    }
    let legacy_g = format!("{base}.weight_g");
    let legacy_v = format!("{base}.weight_v");
    if weights.contains_key(&legacy_g) && weights.contains_key(&legacy_v) {
        return Some(WeightNormParams {
            weight_g: legacy_g,
            weight_v: legacy_v,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorData;

    #[test]
    fn parses_tiny_fixture() {
        let bytes = include_bytes!("../tests/fixtures/tiny_v2_f0.pth");
        let ckpt = parse_pth(bytes).unwrap();
        assert_eq!(ckpt.version, RvcVersion::V2);
        assert!(ckpt.use_f0);
        assert_eq!(ckpt.config.sr, 40_000);
        assert_eq!(ckpt.config.upsample_rates, vec![10, 10, 2, 2]);
        assert_eq!(ckpt.config.frame_hop(), 400);
        // spk_embed_dim must be overridden by the actual embedding rows.
        let emb = ckpt.weight("emb_g.weight").unwrap();
        assert_eq!(ckpt.config.spk_embed_dim, emb.shape[0]);
        assert!(matches!(emb.data, TensorData::F32(_)));
    }

    #[test]
    fn rejects_legacy_non_zip() {
        let err = parse_pth(&[0x80, 0x02, 0x2e]).unwrap_err().to_string();
        assert!(err.contains("legacy pickle-only"), "{err}");
    }
}
