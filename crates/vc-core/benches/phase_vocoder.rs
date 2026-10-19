//! Isolate the production crossfade: no inference, SOLA search or resampling.
//! Include the private module so benchmarking does not expand the public API.
use divan::{black_box, Bencher};

#[path = "../src/sola/phase_vocoder.rs"]
#[allow(
    dead_code,
    unused_imports,
    reason = "The included test oracle is not registered by this harness-free benchmark; production is linted separately."
)]
mod implementation;

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

// 10/20/50/75/85/200 ms overlaps at 48 kHz. Construction, FFT planning and
// initial buffer preparation are outside the timed region. Resetting the
// destination is included equally before and after optimization.
#[divan::bench(args = [480, 960, 2400, 3600, 4080, 9600])]
fn crossfade(bencher: Bencher, len: usize) {
    let previous: Vec<_> = (0..len)
        .map(|i| 0.25 * (i as f32 * 0.0288).sin() + 0.1 * (i as f32 * 0.0837).cos())
        .collect();
    let current: Vec<_> = (0..len)
        .map(|i| 0.25 * (i as f32 * 0.0291 + 0.7).sin() + 0.1 * (i as f32 * 0.084 - 0.8).cos())
        .collect();
    let mut output = current.clone();
    let mut vocoder = implementation::PhaseVocoder::new(len);
    vocoder.crossfade(&previous, &mut output);
    bencher.bench_local(|| {
        output.copy_from_slice(&current);
        vocoder.crossfade(black_box(&previous), black_box(&mut output));
        black_box(&output);
    });
}
