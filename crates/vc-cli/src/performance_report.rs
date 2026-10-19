//! Offline wall-clock measurements; aggregation never runs on an audio callback.
use std::{fs::OpenOptions, path::Path, time::Duration};

use anyhow::{Context, Result};
use serde_json::{json, Value};
use vc_core::model_rvc::{ChunkStats, FiniteOutput};

fn milliseconds(value: Duration) -> f64 {
    value.as_secs_f64() * 1000.0
}

fn sample(stats: &ChunkStats) -> Value {
    json!({
        "contentvec_ms": milliseconds(stats.embedder_time),
        "rmvpe_ms": milliseconds(stats.pitch_time),
        "rvc_ms": milliseconds(stats.rvc_time),
        "inference_ms": milliseconds(stats.inference_time),
        "smoothing_ms": milliseconds(stats.smoothing_time),
        "processing_ms": milliseconds(stats.processing_time),
    })
}

fn distribution(mut values: Vec<f64>, deadline: f64) -> Value {
    if values.is_empty() {
        return json!({"count": 0, "mean_ms": null, "p95_ms": null,
            "p99_ms": null, "max_ms": null, "deadline_misses": 0});
    }
    values.sort_by(f64::total_cmp);
    let percentile = |percent: usize| values[(values.len() * percent).div_ceil(100) - 1];
    json!({"count": values.len(), "mean_ms": values.iter().sum::<f64>() / values.len() as f64,
        "p95_ms": percentile(95), "p99_ms": percentile(99), "max_ms": values.last(),
        "deadline_misses": values.iter().filter(|&&v| v > deadline).count()})
}

fn summarize(stats: &[&ChunkStats], deadline: f64) -> Value {
    let mut result = serde_json::Map::new();
    for key in [
        "contentvec_ms",
        "rmvpe_ms",
        "rvc_ms",
        "inference_ms",
        "smoothing_ms",
        "processing_ms",
    ] {
        let values = stats
            .iter()
            .map(|s| sample(s)[key].as_f64().unwrap_or_default())
            .collect();
        result.insert(key.to_owned(), distribution(values, deadline));
    }
    Value::Object(result)
}

// Query inside the conversion process: another process may see the EP as
// NotReady and omit its library path even after this session loaded it.
pub fn catalog_snapshot() -> Value {
    #[cfg(all(windows, feature = "windowsml"))]
    {
        match vc_core::windows_ml::list_catalog_providers() {
            Ok(providers) => json!(providers
                .iter()
                .map(|p| json!({
                    "name": p.name, "version": p.version, "library_path": p.library_path,
                    "ready_state": p.ready_state.label(), "package": p.package_family_name,
                }))
                .collect::<Vec<_>>()),
            Err(error) => json!({"unavailable": error.to_string()}),
        }
    }
    #[cfg(not(all(windows, feature = "windowsml")))]
    json!({"unavailable": "Windows ML is not enabled in this build"})
}

pub fn build(output: &FiniteOutput, warmup: usize, deadline_ms: f64, load: Duration) -> Value {
    let input: Vec<_> = output
        .chunks
        .iter()
        .filter(|c| !c.flushing)
        .map(|c| &c.stats)
        .collect();
    let split = warmup.min(input.len());
    let flush: Vec<_> = output
        .chunks
        .iter()
        .filter(|c| c.flushing)
        .map(|c| &c.stats)
        .collect();
    json!({
        "schema_version": 1,
        "scope": "Synchronous host wall time including transfers/synchronization within each model stage; processing includes joining/resampling. Excludes WAV I/O, offline RNNoise preprocessing, audio device queues and scheduling. Deadline misses are offline budget comparisons, not observed audio underruns. Stage names denote pipeline stages, not proof of device placement.",
        "percentile_method": "nearest_rank",
        "deadline_ms": deadline_ms,
        "pipeline_load_ms": milliseconds(load),
        "warmup_input_chunks_requested": warmup,
        "preroll": output.preroll.as_ref().map(sample),
        "initial": summarize(&input[..split], deadline_ms),
        "steady": summarize(&input[split..], deadline_ms),
        "flush": summarize(&flush, deadline_ms),
        "chunks": output.chunks.iter().map(|c| json!({"index": c.index, "flushing": c.flushing, "timing": sample(&c.stats)})).collect::<Vec<_>>(),
    })
}

/// Reserve the report before loading models. Never overwrite a WAV, model, or
/// previous measurement, including when a caller aliases a path via a symlink.
pub fn create(path: &Path) -> Result<std::fs::File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("cannot create new performance report {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_and_strict_deadline() {
        let result = distribution((1..=100).map(f64::from).collect(), 95.0);
        assert_eq!(result["p95_ms"], 95.0);
        assert_eq!(result["p99_ms"], 99.0);
        assert_eq!(result["deadline_misses"], 5);
        assert_eq!(result["mean_ms"], 50.5);
        assert!(distribution(vec![], 10.0)["mean_ms"].is_null());
    }

    #[test]
    fn preroll_startup_and_flush_do_not_pollute_steady_samples() {
        use vc_core::{model_rvc::FiniteChunk, sola::JoinDiagnostics};
        let chunks = [(1000, false), (20, false), (40, false), (900, true)]
            .into_iter()
            .enumerate()
            .map(|(index, (ms, flushing))| FiniteChunk {
                index,
                flushing,
                stats: ChunkStats {
                    processing_time: Duration::from_millis(ms),
                    smoothing_time: Duration::from_millis(ms / 2),
                    ..Default::default()
                },
                join: JoinDiagnostics::default(),
                requested_crossfade_samples: 0,
                seam_sample: None,
            })
            .collect();
        let output = FiniteOutput {
            audio: vec![],
            chunks,
            preroll: Some(ChunkStats {
                processing_time: Duration::from_millis(5000),
                ..Default::default()
            }),
        };
        let report = build(&output, 1, 30.0, Duration::ZERO);
        assert_eq!(report["steady"]["processing_ms"]["mean_ms"], 30.0);
        assert_eq!(report["steady"]["smoothing_ms"]["mean_ms"], 15.0);
        assert_eq!(report["steady"]["processing_ms"]["deadline_misses"], 1);
        assert_eq!(report["initial"]["processing_ms"]["count"], 1);
        assert_eq!(report["flush"]["processing_ms"]["count"], 1);
        assert_eq!(report["preroll"]["processing_ms"], 5000.0);
        assert!(
            build(&output, 100, 30.0, Duration::ZERO)["steady"]["processing_ms"]["max_ms"]
                .is_null()
        );
        let empty = FiniteOutput {
            audio: vec![],
            chunks: vec![],
            preroll: None,
        };
        assert!(build(&empty, 0, 30.0, Duration::ZERO)["preroll"].is_null());
    }
}
