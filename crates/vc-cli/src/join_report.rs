//! Offline analysis of chunk-join quality for `vc-rs wav --join-report`.
//!
//! The finite core adapter maps each model chunk's join into the written WAV
//! after output resampling and startup-delay removal. Those positions need not
//! be multiples of the host chunk size.
//! This module measures the discontinuity at each seam and pairs it with the
//! smoother's own per-chunk decisions ([`vc_core::sola::JoinDiagnostics`]) so an
//! audible artifact can be traced to *why* the join went wrong (low correlation,
//! PSOLA fallback, crossfade capped by a short chunk, silence, ...).
//!
//! Domain note: seam metrics are measured in the **output/device domain** (the
//! written WAV). The join diagnostics (`sola_offset`, `crossfade_len`,
//! `pitch_period`) are in the **model domain** SOLA runs in, so their sample
//! counts are not directly comparable to seam sample positions.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use vc_core::dsp;
use vc_core::sola::{JoinDiagnostics, SmoothingKind};

/// Discontinuity measured across a single chunk seam, in the output domain.
#[derive(Clone, Copy, Debug)]
pub struct SeamMetrics {
    /// `|out[seam] - out[seam-1]|`: the raw step across the boundary.
    pub sample_step: f32,
    /// Median adjacent-sample step in the seam's neighbourhood, excluding the
    /// boundary itself — the "normal" local roughness to compare against.
    pub baseline_step: f32,
    /// `sample_step / baseline_step`: how far the boundary step stands out from
    /// local roughness. A clean join is ~1; a click spikes well above.
    pub step_ratio: f32,
    pub rms_before: f32,
    pub rms_after: f32,
    /// `20*log10(rms_after / rms_before)`: energy jump across the seam.
    pub energy_step_db: f32,
}

/// Measures the seam at sample index `seam` in `output`. `window` is the
/// half-width (samples) used for both the baseline roughness and the RMS
/// windows. Returns `None` when there is no boundary to measure (seam at the
/// very start or past the end).
pub fn seam_metrics(output: &[f32], seam: usize, window: usize) -> Option<SeamMetrics> {
    if seam == 0 || seam >= output.len() {
        return None;
    }
    let window = window.max(1);
    let sample_step = (output[seam] - output[seam - 1]).abs();

    // Baseline: adjacent-sample steps within each side of the seam, never
    // crossing the boundary, so it reflects ordinary local roughness.
    let left_lo = seam.saturating_sub(window);
    let mut diffs: Vec<f32> = Vec::new();
    for k in left_lo..seam.saturating_sub(1) {
        diffs.push((output[k + 1] - output[k]).abs());
    }
    let right_hi = (seam + window).min(output.len());
    for k in seam..right_hi.saturating_sub(1) {
        diffs.push((output[k + 1] - output[k]).abs());
    }
    let baseline_step = median(&mut diffs);

    let before = &output[left_lo..seam];
    let after = &output[seam..right_hi];
    let rms_before = dsp::rms(before);
    let rms_after = dsp::rms(after);

    const EPS: f32 = 1e-9;
    let step_ratio = sample_step / (baseline_step + EPS);
    let energy_step_db = 20.0 * ((rms_after + EPS) / (rms_before + EPS)).log10();

    Some(SeamMetrics {
        sample_step,
        baseline_step,
        step_ratio,
        rms_before,
        rms_after,
        energy_step_db,
    })
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// One CSV row: a chunk's seam metrics plus the smoother's join decision.
#[derive(Clone, Copy, Debug)]
pub struct ChunkRecord {
    pub chunk: usize,
    pub seam_sample: usize,
    pub seam_ms: f64,
    pub diag: JoinDiagnostics,
    /// Configured crossfade window (model domain); lets readers see capping.
    pub crossfade_target: usize,
    /// `None` for a boundary at the very start of the written clip.
    pub seam: Option<SeamMetrics>,
}

/// Accumulates per-chunk records during a WAV run and emits the CSV + summary.
pub struct JoinReport {
    sample_rate: u32,
    records: Vec<ChunkRecord>,
}

impl JoinReport {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            records: Vec::new(),
        }
    }

    /// Record a join at its actual position after finite-output delay removal.
    /// Chunk indices include zero-input drain calls because these can recover
    /// real speech that was held upstream; omitted startup joins leave gaps.
    pub fn record_at(
        &mut self,
        chunk: usize,
        output: &[f32],
        seam_sample: usize,
        diag: JoinDiagnostics,
        crossfade_target: usize,
    ) {
        // ~5 ms window each side; clamped to at least a few samples for tiny rates.
        let window = ((self.sample_rate as usize * 5) / 1000).max(4);
        let seam = seam_metrics(output, seam_sample, window);
        self.records.push(ChunkRecord {
            chunk,
            seam_sample,
            seam_ms: seam_sample as f64 * 1000.0 / self.sample_rate.max(1) as f64,
            diag,
            crossfade_target,
            seam,
        });
    }

    /// Writes the per-chunk CSV.
    pub fn write_csv(&self, path: &Path) -> Result<()> {
        let file = File::create(path)
            .with_context(|| format!("failed to create join report {}", path.display()))?;
        let mut w = BufWriter::new(file);
        writeln!(
            w,
            "chunk,seam_sample,seam_ms,kind,sola_offset,max_offset,correlation,\
             crossfade_len,crossfade_target,crossfade_capped,pitch_period,psola_fallback,\
             sample_step,baseline_step,step_ratio,rms_before,rms_after,energy_step_db"
        )?;
        for r in &self.records {
            let d = &r.diag;
            let kind = match d.kind {
                Some(SmoothingKind::Sola) => "sola",
                Some(SmoothingKind::Psola) => "psola",
                None => "none",
            };
            let capped = d.crossfade_len < r.crossfade_target;
            let pitch = d
                .pitch_period
                .map(|p| p.to_string())
                .unwrap_or_else(|| String::from(""));
            let (sample_step, baseline_step, step_ratio, rms_before, rms_after, energy_step_db) =
                match r.seam {
                    Some(s) => (
                        s.sample_step,
                        s.baseline_step,
                        s.step_ratio,
                        s.rms_before,
                        s.rms_after,
                        s.energy_step_db,
                    ),
                    None => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                };
            writeln!(
                w,
                "{chunk},{seam_sample},{seam_ms:.3},{kind},{offset},{max_offset},{corr:.6},\
                 {cf_len},{cf_target},{capped},{pitch},{fallback},\
                 {sample_step:.6},{baseline_step:.6},{step_ratio:.3},\
                 {rms_before:.6},{rms_after:.6},{energy_step_db:.3}",
                chunk = r.chunk,
                seam_sample = r.seam_sample,
                seam_ms = r.seam_ms,
                offset = d.sola_offset,
                max_offset = d.max_offset,
                corr = d.correlation,
                cf_len = d.crossfade_len,
                cf_target = r.crossfade_target,
                pitch = pitch,
                fallback = d.psola_fallback,
            )?;
        }
        w.flush()?;
        Ok(())
    }

    /// Human-readable summary (worst seams, correlation/fallback/capping rates).
    pub fn summary(&self) -> String {
        // Only joins with two retained neighbours have measurable seams.
        let seams: Vec<&ChunkRecord> = self.records.iter().filter(|r| r.seam.is_some()).collect();
        if seams.is_empty() {
            return String::from("join report: no seams (single chunk)");
        }

        let count = seams.len();
        let mut ratios: Vec<(usize, f64, f32)> = seams
            .iter()
            .map(|r| (r.chunk, r.seam_ms, r.seam.unwrap().step_ratio))
            .collect();
        ratios.sort_by(|a, b| b.2.total_cmp(&a.2));

        let corr: Vec<f32> = seams.iter().map(|r| r.diag.correlation).collect();
        let corr_min = corr.iter().copied().fold(f32::INFINITY, f32::min);
        let corr_mean = corr.iter().copied().sum::<f32>() / count as f32;
        let fallbacks = seams.iter().filter(|r| r.diag.psola_fallback).count();
        let capped = seams
            .iter()
            .filter(|r| r.diag.crossfade_len < r.crossfade_target)
            .count();

        let mut out = String::new();
        out.push_str(&format!("join report: {count} seams\n"));
        out.push_str(&format!(
            "  correlation: min {corr_min:.3}, mean {corr_mean:.3}\n"
        ));
        out.push_str(&format!(
            "  crossfade capped (chunk < crossfade window): {capped}/{count}\n"
        ));
        out.push_str(&format!("  PSOLA fallback to SOLA: {fallbacks}/{count}\n"));
        out.push_str("  worst seams by step_ratio:\n");
        for (chunk, ms, ratio) in ratios.iter().take(5) {
            out.push_str(&format!(
                "    chunk {chunk} @ {ms:.0} ms: step_ratio {ratio:.1}\n"
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_measures_delay_compensated_boundary_in_written_audio() {
        let mut output = vec![0.0; 200];
        output[73..].fill(0.5);
        let mut report = JoinReport::new(16_000);
        report.record_at(2, &output, 73, JoinDiagnostics::default(), 160);
        assert_eq!(report.records[0].seam_sample, 73);
        assert_eq!(report.records[0].seam.unwrap().sample_step, 0.5);
    }

    #[test]
    fn seam_metrics_flags_injected_step() {
        // Smooth ramp with a sharp jump injected exactly at the seam.
        let mut audio: Vec<f32> = (0..200).map(|i| (i as f32 * 0.01).sin()).collect();
        let seam = 100;
        for s in audio.iter_mut().skip(seam) {
            *s += 0.8; // discontinuity at the boundary
        }
        let m = seam_metrics(&audio, seam, 32).expect("seam");
        assert!(m.sample_step > 0.5, "step={}", m.sample_step);
        // The injected jump dwarfs the smooth ramp's local roughness.
        assert!(m.step_ratio > 10.0, "ratio={}", m.step_ratio);
    }

    #[test]
    fn seam_metrics_clean_join_is_low_ratio() {
        let audio: Vec<f32> = (0..200).map(|i| (i as f32 * 0.05).sin()).collect();
        let m = seam_metrics(&audio, 100, 32).expect("seam");
        // No injected discontinuity: boundary step is in line with neighbours.
        assert!(m.step_ratio < 3.0, "ratio={}", m.step_ratio);
    }

    #[test]
    fn seam_metrics_none_at_edges() {
        let audio = vec![0.0; 16];
        assert!(seam_metrics(&audio, 0, 4).is_none());
        assert!(seam_metrics(&audio, 16, 4).is_none());
    }

    #[test]
    fn median_handles_even_and_odd() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&mut []), 0.0);
    }
}
