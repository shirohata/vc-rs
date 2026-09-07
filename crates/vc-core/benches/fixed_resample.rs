//! CPU cost of the actual fixed-hop adapters, excluding initialization and GPU
//! work. Compile the DSP module here to exercise crate-private types without
//! making the lifetime-fixed constructors part of vc-core's public API. Do not
//! copy their implementations into a benchmark: that would miss regressions.

#[path = "../src/dsp.rs"]
#[allow(unused_imports)] // DSP's public re-exports are not all needed here.
mod dsp;

use divan::{black_box, Bencher};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

fn signal(rate: usize) -> Vec<f32> {
    (0..rate / 5)
        .map(|i| (i as f32 * 220.0 * std::f32::consts::TAU / rate as f32).sin() * 0.4)
        .collect()
}

#[divan::bench(args = [16_000, 32_000, 44_100, 48_000, 96_000])]
fn input_200ms(bencher: Bencher, from: usize) {
    let input = signal(from);
    let mut resampler = dsp::FixedInputResampler::new(from, 16_000, from / 5, 3200).unwrap();
    let mut output = Vec::new();
    // Traverse more than the longest 200 ms input batch/FFT phase cycle before
    // timing. Reuse the same fixed-size hop and buffers throughout the run.
    for _ in 0..200 {
        output.clear();
        resampler.process_into(&input, 3200, &mut output).unwrap();
    }
    bencher.bench_local(|| {
        output.clear();
        resampler
            .process_into(black_box(&input), 3200, &mut output)
            .unwrap();
        black_box(&output);
    });
}

#[divan::bench(args = [
    (32_000, 44_100), (32_000, 48_000),
    (40_000, 44_100), (40_000, 48_000),
    (48_000, 44_100), (48_000, 48_000),
])]
fn output_200ms(bencher: Bencher, (from, to): (usize, usize)) {
    let input = signal(from);
    let mut resampler = dsp::OutputResampler::new_fixed(from, to, from / 5, to / 5).unwrap();
    let mut output = Vec::new();
    for _ in 0..200 {
        resampler
            .process_fixed(&input, to / 5, &mut output)
            .unwrap();
    }
    bencher.bench_local(|| {
        resampler
            .process_fixed(black_box(&input), to / 5, &mut output)
            .unwrap();
        black_box(&output);
    });
}
