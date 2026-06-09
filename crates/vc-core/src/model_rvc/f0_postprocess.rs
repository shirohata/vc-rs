//! Lightweight F0 post-processing for the RVC pipeline.
//!
//! Operates on the RVC-aligned `pitchf` (same frame grid as the ContentVec
//! features and the coarse `pitch`). The pipeline extracts RMVPE F0 with a
//! `0.0` pitch shift, so the array reaching this module is raw / natural F0:
//! range clamp, octave correction and the median filter all act on natural F0,
//! and pitch shift is applied here exactly once at the very end. Coarse `pitch`
//! is recomputed by the caller from the final `pitchf` (`coarse_pitch_into`).
//!
//! Real-time note: this runs on the worker thread, never the audio callback.
//! Scratch buffers are reused so a steady-state chunk does no heap allocation.

/// `0.0` marks an unvoiced frame throughout this module.
const UNVOICED: f32 = 0.0;

/// Octave-jump detection tolerances (ratio space).
///
/// A frame is only treated as a single-frame octave error when its neighbours
/// are themselves close (so we do not flatten a genuine pitch glide) and the
/// centre sits near an exact 2x / 0.5x of the neighbour average.
const LR_NEAR_RATIO_TOL: f32 = 0.2;
const OCTAVE_RATIO_TOL: f32 = 0.25;

#[derive(Clone, Debug)]
pub struct F0PostprocessConfig {
    pub enabled: bool,

    pub min_f0_hz: f32,
    pub max_f0_hz: f32,

    pub remove_short_voiced_islands: bool,
    pub max_voiced_island_frames: usize,

    pub fill_short_unvoiced_gaps: bool,
    pub max_unvoiced_gap_frames: usize,

    pub fix_octave_jumps: bool,

    pub median_filter: bool,
    pub median_filter_radius: usize,

    /// When true, saturate voiced frames into `min_f0_hz..=max_f0_hz` *after*
    /// pitch shift (out-of-range values are clamped to the bound, kept voiced).
    /// Deliberately different from the pre-shift invalid step, which *zeroes*
    /// out-of-range frames: zeroing a shifted-up high note would silence it.
    /// Near-existing behaviour `false` is the default.
    pub clamp_after_pitch_shift: bool,
}

impl Default for F0PostprocessConfig {
    fn default() -> Self {
        Self {
            enabled: false,

            min_f0_hz: 50.0,   // matches existing coarse f0_min
            max_f0_hz: 1100.0, // matches existing coarse f0_max

            remove_short_voiced_islands: true,
            max_voiced_island_frames: 1,

            fill_short_unvoiced_gaps: true,
            max_unvoiced_gap_frames: 2,

            fix_octave_jumps: true,

            median_filter: true,
            median_filter_radius: 1,

            clamp_after_pitch_shift: false,
        }
    }
}

pub struct F0Postprocessor {
    config: F0PostprocessConfig,
    /// Sorting window for the median filter (reused per frame).
    median_scratch: Vec<f32>,
    /// Snapshot of the array as it enters the median pass. The median reads from
    /// this copy and writes only to the output, so a corrected frame can never
    /// feed the next frame's window (which would make the filter order-dependent).
    median_input_scratch: Vec<f32>,
}

impl F0Postprocessor {
    pub fn new(config: F0PostprocessConfig) -> Self {
        Self {
            config,
            median_scratch: Vec::new(),
            median_input_scratch: Vec::new(),
        }
    }

    // Accessors for the future runtime/UI wiring (mirrors `set_pitch_shift` &c.).
    // Not used by the engine yet; that wiring is a separate task.
    #[allow(dead_code)]
    pub fn config(&self) -> &F0PostprocessConfig {
        &self.config
    }

    #[allow(dead_code)]
    pub fn set_config(&mut self, config: F0PostprocessConfig) {
        self.config = config;
    }

    /// Post-process aligned raw `pitchf` and apply pitch shift exactly once.
    ///
    /// `input_pitchf` is the RVC-aligned, *un-shifted* (natural) F0 and is never
    /// mutated; the result is written to `output`.
    ///
    /// Always call this, even when post-processing is disabled: pitch shift is
    /// applied here (RMVPE extract receives `0.0`), so skipping the call would
    /// drop the shift entirely. When disabled, the smoothing/invalid steps are
    /// skipped and only the shift (+ optional post-shift clamp) is applied,
    /// which is bit-equivalent to the previous "shift in extract" path for a
    /// static pitch shift.
    pub fn process_pitchf_into(
        &mut self,
        input_pitchf: &[f32],
        pitch_shift_semitones: f32,
        output: &mut Vec<f32>,
    ) {
        output.clear();
        output.extend_from_slice(input_pitchf);

        if self.config.enabled {
            self.remove_invalid(output);
            if self.config.remove_short_voiced_islands {
                self.remove_short_voiced_islands(output);
            }
            if self.config.fill_short_unvoiced_gaps {
                self.fill_short_unvoiced_gaps(output);
            }
            if self.config.fix_octave_jumps {
                self.fix_octave_jumps(output);
            }
            if self.config.median_filter && self.config.median_filter_radius > 0 {
                self.median_filter(output);
            }
        }

        // Pitch shift, applied exactly once for both enabled and disabled modes.
        if pitch_shift_semitones != 0.0 {
            let factor = 2.0_f32.powf(pitch_shift_semitones / 12.0);
            for f0 in output.iter_mut() {
                if *f0 > UNVOICED {
                    *f0 *= factor;
                }
            }
        }

        if self.config.clamp_after_pitch_shift {
            let (min, max) = (self.config.min_f0_hz, self.config.max_f0_hz);
            for f0 in output.iter_mut() {
                if *f0 > UNVOICED {
                    *f0 = f0.clamp(min, max);
                }
            }
        }
    }

    /// Step 3: NaN/inf, non-positive, and out-of-[min,max] frames become unvoiced.
    /// Runs on natural (pre-shift) F0.
    fn remove_invalid(&self, pitchf: &mut [f32]) {
        let (min, max) = (self.config.min_f0_hz, self.config.max_f0_hz);
        for f0 in pitchf.iter_mut() {
            if !f0.is_finite() || *f0 <= UNVOICED || *f0 < min || *f0 > max {
                *f0 = UNVOICED;
            }
        }
    }

    /// Step 4: zero voiced runs of `<= max_voiced_island_frames` that are
    /// surrounded by unvoiced on both sides. Runs touching either edge are not
    /// "islands" (their off-window context is unknown) and are kept.
    fn remove_short_voiced_islands(&self, pitchf: &mut [f32]) {
        let max_len = self.config.max_voiced_island_frames;
        let n = pitchf.len();
        let mut i = 0;
        while i < n {
            if pitchf[i] > UNVOICED {
                let start = i;
                while i < n && pitchf[i] > UNVOICED {
                    i += 1;
                }
                let end = i; // exclusive
                let touches_edge = start == 0 || end == n;
                if !touches_edge && end - start <= max_len {
                    pitchf[start..end].fill(UNVOICED);
                }
            } else {
                i += 1;
            }
        }
    }

    /// Step 5: fill unvoiced runs of `<= max_unvoiced_gap_frames` that are
    /// bounded by voiced frames on both sides, using log-F0 linear interpolation
    /// between the bounding values. Leading/trailing gaps are left unvoiced.
    fn fill_short_unvoiced_gaps(&self, pitchf: &mut [f32]) {
        let max_len = self.config.max_unvoiced_gap_frames;
        let n = pitchf.len();
        let mut i = 0;
        while i < n {
            if pitchf[i] <= UNVOICED {
                let start = i;
                while i < n && pitchf[i] <= UNVOICED {
                    i += 1;
                }
                let end = i; // exclusive; first voiced frame after the gap (or n)
                // Bounded on both sides => start > 0 (voiced at start-1) and
                // end < n (voiced at end). Both bounds are guaranteed voiced.
                if start > 0 && end < n && end - start <= max_len {
                    let log_left = pitchf[start - 1].ln();
                    let log_right = pitchf[end].ln();
                    let steps = (end - start + 1) as f32;
                    for (k, idx) in (start..end).enumerate() {
                        let t = (k + 1) as f32 / steps;
                        pitchf[idx] = (log_left + (log_right - log_left) * t).exp();
                    }
                }
            } else {
                i += 1;
            }
        }
    }

    /// Step 6: correct isolated single-frame ~2x / ~0.5x octave jumps only.
    /// Requires left/center/right all voiced and left close to right, so a
    /// genuine sustained octave change (two adjacent shifted frames) is kept.
    fn fix_octave_jumps(&self, pitchf: &mut [f32]) {
        let n = pitchf.len();
        if n < 3 {
            return;
        }
        for i in 1..n - 1 {
            let left = pitchf[i - 1];
            let center = pitchf[i];
            let right = pitchf[i + 1];
            if left <= UNVOICED || center <= UNVOICED || right <= UNVOICED {
                continue;
            }
            if (left / right - 1.0).abs() > LR_NEAR_RATIO_TOL {
                continue;
            }
            let reference = 0.5 * (left + right);
            let ratio = center / reference;
            if (ratio - 2.0).abs() <= OCTAVE_RATIO_TOL {
                pitchf[i] = center * 0.5;
            } else if (ratio - 0.5).abs() <= OCTAVE_RATIO_TOL {
                pitchf[i] = center * 2.0;
            }
        }
    }

    /// Step 7: log-F0 median filter over voiced frames only.
    ///
    /// Unvoiced (`0.0`) frames stay unvoiced and are never mixed into a window.
    /// Reads from a snapshot (`median_input_scratch`) taken before the pass so
    /// the filter is order-independent; writes results into `pitchf`.
    fn median_filter(&mut self, pitchf: &mut [f32]) {
        let radius = self.config.median_filter_radius;
        let n = pitchf.len();
        let input = &mut self.median_input_scratch;
        input.clear();
        input.extend_from_slice(pitchf);
        let window = &mut self.median_scratch;
        for i in 0..n {
            if input[i] <= UNVOICED {
                continue;
            }
            window.clear();
            let lo = i.saturating_sub(radius);
            let hi = (i + radius + 1).min(n);
            for &v in &input[lo..hi] {
                if v > UNVOICED {
                    window.push(v);
                }
            }
            // The center is always voiced and included, so the window is never
            // empty. With no voiced neighbours the median equals the center and
            // the value is left unchanged.
            window.sort_by(f32::total_cmp);
            let m = window.len();
            pitchf[i] = if m % 2 == 1 {
                window[m / 2]
            } else {
                // Even effective count: average the two middle frames in log
                // space (the "log-F0" part that actually matters for medians).
                let a = window[m / 2 - 1];
                let b = window[m / 2];
                (0.5 * (a.ln() + b.ln())).exp()
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_all_off() -> F0PostprocessConfig {
        F0PostprocessConfig {
            enabled: true,
            remove_short_voiced_islands: false,
            fill_short_unvoiced_gaps: false,
            fix_octave_jumps: false,
            median_filter: false,
            ..F0PostprocessConfig::default()
        }
    }

    fn run(cfg: F0PostprocessConfig, input: &[f32], shift: f32) -> Vec<f32> {
        let mut p = F0Postprocessor::new(cfg);
        let mut out = Vec::new();
        p.process_pitchf_into(input, shift, &mut out);
        out
    }

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-2, "expected ~{b}, got {a}");
    }

    // 1. disabled mode: no smoothing, shift applied once, input untouched.
    #[test]
    fn disabled_passthrough_and_shift() {
        let cfg = F0PostprocessConfig::default(); // enabled = false
        let input = vec![0.0, 220.0, 0.0, 440.0];

        let out = run(cfg.clone(), &input, 0.0);
        assert_eq!(out, input, "shift 0 must be identity");

        let up = run(cfg.clone(), &input, 12.0);
        approx(up[1], 440.0);
        approx(up[3], 880.0);
        assert_eq!(up[0], 0.0);
        assert_eq!(up[2], 0.0);

        let down = run(cfg, &input, -12.0);
        approx(down[1], 110.0);
        approx(down[3], 220.0);
    }

    // 2. pitch shift applied exactly once (not double): raw 220 + 12 => 440.
    #[test]
    fn pitch_shift_single_application() {
        // Even with smoothing enabled, a clean steady tone shifts once.
        let out = run(F0PostprocessConfig::default(), &[220.0], 12.0);
        approx(out[0], 440.0);
        assert!((out[0] - 880.0).abs() > 1.0, "must not double-apply");

        let out_enabled = run(cfg_all_off(), &[220.0, 220.0, 220.0], 12.0);
        for v in out_enabled {
            approx(v, 440.0);
        }
    }

    // 3. invalid / out-of-range -> unvoiced.
    #[test]
    fn invalid_removal() {
        let cfg = cfg_all_off();
        let input = vec![f32::NAN, f32::INFINITY, -10.0, 0.0, 30.0, 2000.0, 220.0];
        let out = run(cfg, &input, 0.0);
        assert_eq!(out[0], 0.0); // NaN
        assert_eq!(out[1], 0.0); // inf
        assert_eq!(out[2], 0.0); // negative
        assert_eq!(out[3], 0.0); // zero
        assert_eq!(out[4], 0.0); // below min (50)
        assert_eq!(out[5], 0.0); // above max (1100)
        approx(out[6], 220.0); // in range
    }

    // 4. short voiced island removal, edges preserved.
    #[test]
    fn short_voiced_island_removal() {
        let cfg = F0PostprocessConfig {
            enabled: true,
            remove_short_voiced_islands: true,
            fill_short_unvoiced_gaps: false,
            fix_octave_jumps: false,
            median_filter: false,
            ..F0PostprocessConfig::default()
        };
        assert_eq!(
            run(cfg.clone(), &[0.0, 0.0, 220.0, 0.0, 0.0], 0.0),
            vec![0.0; 5]
        );
        // Leading/trailing voiced runs touch the edge and are kept.
        let edge = run(cfg, &[220.0, 0.0, 0.0, 220.0], 0.0);
        approx(edge[0], 220.0);
        approx(edge[3], 220.0);
    }

    // 5. short unvoiced gap fill (log-linear), edges not filled.
    #[test]
    fn short_unvoiced_gap_fill() {
        let cfg = F0PostprocessConfig {
            enabled: true,
            remove_short_voiced_islands: false,
            fill_short_unvoiced_gaps: true,
            fix_octave_jumps: false,
            median_filter: false,
            ..F0PostprocessConfig::default()
        };
        let out = run(cfg.clone(), &[220.0, 0.0, 0.0, 240.0], 0.0);
        let (ll, lr) = (220.0_f32.ln(), 240.0_f32.ln());
        approx(out[1], (ll + (lr - ll) * (1.0 / 3.0)).exp());
        approx(out[2], (ll + (lr - ll) * (2.0 / 3.0)).exp());

        // Leading/trailing gaps stay unvoiced.
        let edge = run(cfg, &[0.0, 220.0, 240.0, 0.0], 0.0);
        assert_eq!(edge[0], 0.0);
        assert_eq!(edge[3], 0.0);
    }

    // 6. octave jump correction, only for isolated near-2x/0.5x with close sides.
    #[test]
    fn octave_jump_correction() {
        let cfg = F0PostprocessConfig {
            enabled: true,
            remove_short_voiced_islands: false,
            fill_short_unvoiced_gaps: false,
            fix_octave_jumps: true,
            median_filter: false,
            ..F0PostprocessConfig::default()
        };
        let up = run(cfg.clone(), &[220.0, 221.0, 440.0, 222.0, 221.0], 0.0);
        approx(up[2], 220.0);
        let down = run(cfg.clone(), &[220.0, 221.0, 110.0, 222.0, 221.0], 0.0);
        approx(down[2], 220.0);
        // Left and right not close => not an octave error, keep as-is.
        let glide = run(cfg, &[220.0, 440.0, 330.0], 0.0);
        approx(glide[1], 440.0);
    }

    // 7. median filter: stable 3-point, unvoiced not mixed, order-independent.
    #[test]
    fn median_filter_behaviour() {
        let cfg = F0PostprocessConfig {
            enabled: true,
            remove_short_voiced_islands: false,
            fill_short_unvoiced_gaps: false,
            fix_octave_jumps: false,
            median_filter: true,
            median_filter_radius: 1,
            ..F0PostprocessConfig::default()
        };
        // Single spike between equal neighbours -> pulled to the neighbour value.
        let out = run(cfg.clone(), &[200.0, 800.0, 200.0], 0.0);
        approx(out[1], 200.0);
        // Order independence: the corrected [1] must not feed [2]'s window.
        // input snapshot keeps [2]'s window = {800(orig),200} plus its center.
        let series = run(cfg.clone(), &[300.0, 300.0, 600.0, 300.0, 300.0], 0.0);
        approx(series[2], 300.0);
        // Unvoiced stays unvoiced and is not averaged in.
        let with_unvoiced = run(cfg, &[0.0, 220.0, 0.0], 0.0);
        assert_eq!(with_unvoiced[0], 0.0);
        assert_eq!(with_unvoiced[2], 0.0);
        approx(with_unvoiced[1], 220.0);
    }

    // post-shift clamp saturates (does not zero) out-of-range voiced frames.
    #[test]
    fn post_shift_clamp_saturates() {
        let cfg = F0PostprocessConfig {
            enabled: true,
            remove_short_voiced_islands: false,
            fill_short_unvoiced_gaps: false,
            fix_octave_jumps: false,
            median_filter: false,
            clamp_after_pitch_shift: true,
            ..F0PostprocessConfig::default()
        };
        // 1000 Hz shifted up an octave -> 2000 Hz, saturated to max (1100), kept voiced.
        let out = run(cfg, &[1000.0], 12.0);
        approx(out[0], 1100.0);
    }
}
