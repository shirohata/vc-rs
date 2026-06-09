//! Dev tool: compare two mono WAV files and report how different they are.
//!
//! Intended for A/B audio-quality checks of the deterministic CPU `wav` path
//! (see `scripts/compare-audio.ps1`). It reports both time-domain metrics
//! (max abs diff, RMS, relative RMS, correlation) and a frequency-domain metric
//! (log-spectral distance in dB), and exits non-zero when any metric exceeds the
//! given threshold, so it can gate a comparison run.
//!
//! It deliberately depends only on `hound` + `realfft` and does NOT link
//! `vc-core`/ORT: it only reads finished WAV output, so it stays fast to build
//! and free of the inference backends.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;

/// Floor added to power spectra before the log, so silent bins do not blow up the
/// log-spectral distance. ~ -100 dB relative to unit power.
const POWER_FLOOR: f32 = 1e-10;

#[derive(Parser, Debug)]
#[command(
    about = "Compare two mono WAV files with time- and frequency-domain metrics",
    long_about = "Compares B against reference A. Lengths are truncated to the shorter \
of the two. Exits non-zero when any metric exceeds its threshold."
)]
struct Args {
    /// Reference WAV (the "A" / baseline side).
    #[arg(long)]
    a: PathBuf,
    /// Comparison WAV (the "B" side).
    #[arg(long)]
    b: PathBuf,

    /// STFT window / FFT size in samples (power of two recommended).
    #[arg(long, default_value_t = 1024)]
    fft_size: usize,
    /// STFT hop size in samples.
    #[arg(long, default_value_t = 256)]
    hop: usize,

    /// Fail if max absolute sample difference exceeds this.
    #[arg(long, default_value_t = 1.0e-4)]
    max_abs: f32,
    /// Fail if relative RMS (rms_diff / rms_a) exceeds this.
    #[arg(long, default_value_t = 1.0e-3)]
    max_rel_rms: f32,
    /// Fail if log-spectral distance (dB) exceeds this.
    #[arg(long, default_value_t = 0.5)]
    max_lsd_db: f32,

    /// Emit a single JSON object instead of a human-readable report.
    #[arg(long)]
    json: bool,
}

struct Metrics {
    len_a: usize,
    len_b: usize,
    compared_len: usize,
    max_abs_diff: f32,
    rms_a: f32,
    rms_b: f32,
    rms_diff: f32,
    rel_rms: f32,
    correlation: f32,
    lsd_db: f32,
}

fn main() -> ExitCode {
    match run() {
        Ok(passed) => {
            if passed {
                ExitCode::SUCCESS
            } else {
                // Distinct code so callers can tell "ran fine but over threshold"
                // apart from "tool/IO error".
                ExitCode::from(1)
            }
        }
        Err(err) => {
            eprintln!("audio_compare: error: {err:#}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool> {
    let args = Args::parse();
    if args.fft_size == 0 || args.hop == 0 {
        anyhow::bail!("fft-size and hop must be greater than zero");
    }
    let samples_a =
        read_wav_mono(&args.a).with_context(|| format!("failed to read {}", args.a.display()))?;
    let samples_b =
        read_wav_mono(&args.b).with_context(|| format!("failed to read {}", args.b.display()))?;

    let metrics = compute_metrics(&samples_a, &samples_b, args.fft_size, args.hop);

    let passed = metrics.max_abs_diff <= args.max_abs
        && metrics.rel_rms <= args.max_rel_rms
        && metrics.lsd_db <= args.max_lsd_db;

    if args.json {
        print_json(&args, &metrics, passed);
    } else {
        print_report(&args, &metrics, passed);
    }
    Ok(passed)
}

fn compute_metrics(a: &[f32], b: &[f32], fft_size: usize, hop: usize) -> Metrics {
    let len_a = a.len();
    let len_b = b.len();
    let n = len_a.min(len_b);
    let a = &a[..n];
    let b = &b[..n];

    let mut max_abs_diff = 0.0f32;
    let mut sum_sq_a = 0.0f64;
    let mut sum_sq_b = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_ab = 0.0f64;
    for i in 0..n {
        let da = a[i];
        let db = b[i];
        let diff = (da - db).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        sum_sq_a += (da as f64) * (da as f64);
        sum_sq_b += (db as f64) * (db as f64);
        sum_sq_diff += (diff as f64) * (diff as f64);
        sum_ab += (da as f64) * (db as f64);
    }
    let rms = |sum_sq: f64| -> f32 {
        if n == 0 {
            0.0
        } else {
            (sum_sq / n as f64).sqrt() as f32
        }
    };
    let rms_a = rms(sum_sq_a);
    let rms_b = rms(sum_sq_b);
    let rms_diff = rms(sum_sq_diff);
    let rel_rms = if rms_a > 0.0 {
        rms_diff / rms_a
    } else if rms_diff == 0.0 {
        0.0
    } else {
        f32::INFINITY
    };
    // Pearson correlation about zero (signals are zero-mean audio); falls back to
    // 1.0 when both sides are silent (perfectly equal).
    let denom = (sum_sq_a.sqrt()) * (sum_sq_b.sqrt());
    let correlation = if denom > 0.0 {
        (sum_ab / denom) as f32
    } else if sum_sq_a == 0.0 && sum_sq_b == 0.0 {
        1.0
    } else {
        0.0
    };

    let lsd_db = log_spectral_distance(a, b, fft_size, hop);

    Metrics {
        len_a,
        len_b,
        compared_len: n,
        max_abs_diff,
        rms_a,
        rms_b,
        rms_diff,
        rel_rms,
        correlation,
        lsd_db,
    }
}

/// Mean (over STFT frames) root-mean-square (over frequency bins) difference of
/// the per-bin power spectra in dB — the standard log-spectral distance.
fn log_spectral_distance(a: &[f32], b: &[f32], fft_size: usize, hop: usize) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let window = hann_window(fft_size);
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(fft_size);
    let mut buf_a = r2c.make_input_vec();
    let mut buf_b = r2c.make_input_vec();
    let mut spec_a = r2c.make_output_vec();
    let mut spec_b = r2c.make_output_vec();

    let mut total = 0.0f64;
    let mut frames = 0usize;
    let mut start = 0usize;
    loop {
        fill_windowed_frame(&mut buf_a, a, start, &window);
        fill_windowed_frame(&mut buf_b, b, start, &window);
        // realfft only fails on a length mismatch, which we control here.
        r2c.process(&mut buf_a, &mut spec_a).expect("fft a");
        r2c.process(&mut buf_b, &mut spec_b).expect("fft b");

        let mut bin_sum = 0.0f64;
        for (ca, cb) in spec_a.iter().zip(spec_b.iter()) {
            let pa = power(ca) + POWER_FLOOR;
            let pb = power(cb) + POWER_FLOOR;
            // 10*log10(power) == log-magnitude in dB.
            let d = 10.0 * (pa.log10() - pb.log10()) as f64;
            bin_sum += d * d;
        }
        let per_frame = (bin_sum / spec_a.len() as f64).sqrt();
        total += per_frame;
        frames += 1;

        if start + fft_size >= n {
            break;
        }
        start += hop;
    }
    if frames == 0 {
        0.0
    } else {
        (total / frames as f64) as f32
    }
}

fn power(c: &Complex<f32>) -> f32 {
    c.re * c.re + c.im * c.im
}

fn fill_windowed_frame(dst: &mut [f32], src: &[f32], start: usize, window: &[f32]) {
    for i in 0..dst.len() {
        let s = src.get(start + i).copied().unwrap_or(0.0);
        dst[i] = s * window[i];
    }
}

fn hann_window(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    let denom = (size - 1) as f32;
    (0..size)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / denom;
            let s = x.sin();
            s * s
        })
        .collect()
}

fn read_wav_mono(path: &std::path::Path) -> Result<Vec<f32>> {
    // Mirrors vc-cli's `read_wav_mono`: integer samples are scaled by 1/32768 and
    // multi-channel frames are averaged to mono.
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let raw: Vec<i32> = reader.samples::<i32>().collect::<Result<_, _>>()?;
            let scale = match spec.bits_per_sample {
                0..=16 => 32768.0,
                24 => 8_388_608.0,
                _ => 2_147_483_648.0,
            };
            raw.chunks(channels)
                .map(|frame| {
                    frame.iter().map(|&x| x as f32 / scale).sum::<f32>() / frame.len() as f32
                })
                .collect()
        }
        hound::SampleFormat::Float => {
            let raw: Vec<f32> = reader.samples::<f32>().collect::<Result<_, _>>()?;
            raw.chunks(channels)
                .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
                .collect()
        }
    };
    Ok(samples)
}

fn print_report(args: &Args, m: &Metrics, passed: bool) {
    println!("A (reference): {}", args.a.display());
    println!("B (compared):  {}", args.b.display());
    println!(
        "samples: A={} B={} compared={}{}",
        m.len_a,
        m.len_b,
        m.compared_len,
        if m.len_a == m.len_b {
            String::new()
        } else {
            format!("  (length differs by {})", m.len_a.abs_diff(m.len_b))
        }
    );
    println!("--- time domain ---");
    println!(
        "  max_abs_diff : {:.3e}   (limit {:.3e})",
        m.max_abs_diff, args.max_abs
    );
    println!("  rms_a        : {:.6}", m.rms_a);
    println!("  rms_b        : {:.6}", m.rms_b);
    println!("  rms_diff     : {:.3e}", m.rms_diff);
    println!(
        "  rel_rms      : {:.3e}   (limit {:.3e})",
        m.rel_rms, args.max_rel_rms
    );
    println!("  correlation  : {:.6}", m.correlation);
    println!("--- frequency domain ---");
    println!(
        "  lsd_db       : {:.4} dB  (limit {:.4} dB, fft={} hop={})",
        m.lsd_db, args.max_lsd_db, args.fft_size, args.hop
    );
    println!("result: {}", if passed { "PASS" } else { "FAIL" });
}

fn print_json(args: &Args, m: &Metrics, passed: bool) {
    // Hand-rolled to avoid a serde dependency in this tiny tool.
    let f = |v: f32| {
        if v.is_finite() {
            format!("{v}")
        } else {
            "null".to_string()
        }
    };
    println!(
        "{{\"len_a\":{},\"len_b\":{},\"compared_len\":{},\"max_abs_diff\":{},\"rms_a\":{},\"rms_b\":{},\"rms_diff\":{},\"rel_rms\":{},\"correlation\":{},\"lsd_db\":{},\"thresholds\":{{\"max_abs\":{},\"max_rel_rms\":{},\"max_lsd_db\":{}}},\"passed\":{}}}",
        m.len_a,
        m.len_b,
        m.compared_len,
        f(m.max_abs_diff),
        f(m.rms_a),
        f(m.rms_b),
        f(m.rms_diff),
        f(m.rel_rms),
        f(m.correlation),
        f(m.lsd_db),
        f(args.max_abs),
        f(args.max_rel_rms),
        f(args.max_lsd_db),
        passed
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, freq: f32, sample_rate: f32, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    /// Deterministic broadband noise (LCG) so every STFT bin sits well above the
    /// power floor — needed to exercise the log-spectral distance cleanly.
    fn noise(n: usize, amp: f32) -> Vec<f32> {
        let mut state: u32 = 0x1234_5678;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let unit = (state >> 8) as f32 / (1u32 << 24) as f32; // [0,1)
                amp * (unit * 2.0 - 1.0)
            })
            .collect()
    }

    #[test]
    fn identical_signals_are_zero_distance() {
        let a = tone(8192, 220.0, 16_000.0, 0.5);
        let m = compute_metrics(&a, &a, 1024, 256);
        assert_eq!(m.max_abs_diff, 0.0);
        assert_eq!(m.rms_diff, 0.0);
        assert_eq!(m.rel_rms, 0.0);
        assert!((m.correlation - 1.0).abs() < 1e-6);
        assert!(m.lsd_db < 1e-4, "lsd_db = {}", m.lsd_db);
    }

    #[test]
    fn pure_gain_shows_constant_log_spectral_distance() {
        // B = 2*A: power ratio is 4 across all bins, so LSD == |10*log10(4)|.
        // Broadband noise keeps every bin above the power floor so the constant
        // ratio is not washed out by silent bins.
        let a = noise(8192, 0.4);
        let b: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
        let m = compute_metrics(&a, &b, 1024, 256);

        let expected_lsd = 10.0 * 4.0f32.log10(); // ~6.02 dB
        assert!(
            (m.lsd_db - expected_lsd).abs() < 0.05,
            "lsd_db = {} expected ~{}",
            m.lsd_db,
            expected_lsd
        );
        // Same shape, scaled amplitude -> still perfectly correlated.
        assert!((m.correlation - 1.0).abs() < 1e-4);
        // rel_rms = rms(|a-2a|)/rms(a) = rms(a)/rms(a) = 1.0.
        assert!((m.rel_rms - 1.0).abs() < 1e-4, "rel_rms = {}", m.rel_rms);
    }

    #[test]
    fn shorter_length_is_compared_and_reported() {
        let a = tone(5000, 200.0, 16_000.0, 0.5);
        let b = tone(4000, 200.0, 16_000.0, 0.5);
        let m = compute_metrics(&a, &b, 1024, 256);
        assert_eq!(m.compared_len, 4000);
        assert_eq!(m.len_a, 5000);
        assert_eq!(m.len_b, 4000);
    }

    #[test]
    fn silent_signals_are_equal() {
        let a = vec![0.0f32; 4096];
        let m = compute_metrics(&a, &a, 1024, 256);
        assert_eq!(m.rel_rms, 0.0);
        assert_eq!(m.correlation, 1.0);
        assert!(m.lsd_db < 1e-6);
    }
}
