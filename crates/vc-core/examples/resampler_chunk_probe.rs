//! Inspect README chunk-size alternatives without changing production defaults.
//! Reports nominal retention for 200 ms hops, not device or CPU latency. The
//! waveform check feeds rubato directly and trims only its documented delay.

#[path = "../src/dsp/fixed_hop.rs"]
mod fixed_hop;

use anyhow::Result;
use rubato::audioadapter_buffers::direct::SequentialSlice;
use rubato::{Fft, FixedSync, Resampler, WindowFunction};

fn make(from: usize, chunk: usize, sub: usize, both: bool) -> Result<Fft<f32>> {
    let fixed = if both {
        FixedSync::Both
    } else {
        FixedSync::Input
    };
    Ok(Fft::new_custom(
        from,
        16_000,
        chunk,
        sub,
        1,
        WindowFunction::BlackmanHarris2,
        fixed,
    )?)
}

fn waveform(mut fft: Fft<f32>, rate: usize) -> Result<Vec<f32>> {
    let mut signal: Vec<f32> = (0..rate)
        .map(|i| {
            let t = i as f32 / rate as f32;
            0.3 * (t * 230.0 * std::f32::consts::TAU).sin()
                + 0.1 * (t * 6700.0 * std::f32::consts::TAU).sin()
        })
        .collect();
    signal[0] += 0.5;
    signal[rate - 1] += 0.5;
    let delay = fft.output_delay();
    let mut input = vec![0.0; fft.input_frames_max()];
    let mut output = vec![0.0; fft.output_frames_max()];
    let mut raw = Vec::new();
    let mut position = 0;
    // Feeding zeros past EOF supplies filter support for the final impulse.
    // Stop by produced frames, since Input mode may emit nothing on a call.
    while raw.len() < 16_000 + delay {
        let needed = fft.input_frames_next();
        for sample in &mut input[..needed] {
            *sample = signal.get(position).copied().unwrap_or(0.0);
            position += 1;
        }
        let source = SequentialSlice::new(&input[..needed], 1, needed)?;
        let output_capacity = output.len();
        let mut destination = SequentialSlice::new_mut(&mut output, 1, output_capacity)?;
        let (_, produced) = fft.process_into_buffer(&source, &mut destination, None)?;
        raw.extend_from_slice(&output[..produced]);
    }
    Ok(raw[delay..delay + 16_000].to_vec())
}

fn main() -> Result<()> {
    println!("input_hz,setting,batch,fft_input,fft_output,filter_ms,fixed_hold_ms,cutoff_hz,cycle_hops,wave_max_error");
    for rate in [32_000, 44_100, 48_000, 96_000] {
        let reference = waveform(make(rate, 480, 1, false)?, rate)?;
        let hop = rate / 5;
        for (name, chunk, sub, both) in [
            ("current", 480, 1, false),
            ("power2_custom", 512, 1, false),
            ("power2_default_subchunks", 512, 2, false),
            ("both_same_fft", 480, 1, true),
            ("hop_unpartitioned", hop, 1, false),
            ("hop_approx480", hop, (hop / 480).max(1), false),
        ] {
            let fft = make(rate, chunk, sub, both)?;
            let contract = fixed_hop::FixedHop::new(rate, 16_000, hop, 3200)?;
            contract.validate(hop, 3200)?;
            let hold = contract.delay(
                fft.input_frames_max(),
                fft.fft_size_in(),
                fft.fft_size_out(),
                fft.output_delay(),
            )?;
            let samples = waveform(make(rate, chunk, sub, both)?, rate)?;
            let error = samples
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            // Both changes buffering only when the actual FFT sizes/window
            // agree. Exact equality covers impulses at both ends as well.
            if both {
                assert_eq!(samples, reference);
            }
            println!(
                "{rate},{name},{},{},{},{:.5},{:.5},{:.3},{},{error:.9}",
                fft.input_frames_max(),
                fft.fft_size_in(),
                fft.fft_size_out(),
                fft.output_delay() as f64 / 16.0,
                hold as f64 / 16.0,
                fft.cutoff() * rate as f32 / 2.0,
                contract.phase_period(fft.input_frames_max(), fft.fft_size_in())?,
            );
        }
    }
    Ok(())
}
