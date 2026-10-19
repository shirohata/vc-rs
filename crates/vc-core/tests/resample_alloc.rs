//! Isolated allocator test: only allocations inside the warmed finite-resample
//! calls are counted. No GPU runtime, logging, input generation or assertions
//! run while counting. Also guards the worker-side smoother's warmed FFT/scratch.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct CountingAllocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, size)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn warmed_reference_and_adaptive_resamplers_do_not_allocate() {
    let input: Vec<f32> = (0..4912).map(|i| (i as f32 * 0.07).sin() * 0.3).collect();
    for rate in [32_000, 40_000, 48_000] {
        let mut scratch = vc_core::dsp::ResampleMonoScratch::default();
        let mut out = Vec::new();
        scratch
            .process_into(&input, 16_000, rate, &mut out)
            .unwrap();
        ALLOCS.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::Relaxed);
        for _ in 0..20 {
            scratch
                .process_into(&input, 16_000, rate, &mut out)
                .unwrap();
        }
        TRACK.store(false, Ordering::Relaxed);
        assert_eq!(ALLOCS.load(Ordering::Relaxed), 0, "16k->{rate} allocated");
    }
    for (from, to) in [(32_000, 48_000), (44_100, 48_000), (48_000, 48_000)] {
        let hop = from / 50;
        let signal = vec![0.25; hop];
        let mut resampler = vc_core::dsp::AdaptiveOutputResampler::new(from, to, hop).unwrap();
        let mut out = Vec::with_capacity(to / 25);
        resampler.process_into(&signal, &mut out).unwrap();
        ALLOCS.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::Relaxed);
        for index in 0..50 {
            resampler
                .set_correction_ppm(index as f64 * 10.0 - 250.0)
                .unwrap();
            resampler.process_into(&signal, &mut out).unwrap();
        }
        TRACK.store(false, Ordering::Relaxed);
        assert_eq!(
            ALLOCS.load(Ordering::Relaxed),
            0,
            "adaptive {from}->{to} allocated"
        );
    }

    use vc_core::sola::{
        model_domain_chunk_smoother, prepare_model_output, ChunkSmootherConfig, SmoothingKind,
    };
    for kind in [
        SmoothingKind::Sola,
        SmoothingKind::Psola,
        SmoothingKind::PhaseVocoder,
    ] {
        let mut smoother = model_domain_chunk_smoother(ChunkSmootherConfig {
            kind,
            output_chunk_samples: 4800,
            output_sample_rate: 48_000,
            model_sample_rate: 48_000,
            crossfade_ms: 50,
            sola_search_ms: 12,
            tail_discard_ms: 10,
        });
        let input = vec![0.25; 8256];
        let pitch = vec![200.0; 18];
        let mut out = Vec::new();
        smoother.prime_model_output(&input, &pitch);
        prepare_model_output(
            &input,
            &pitch,
            48_000,
            48_000,
            4800,
            &mut smoother,
            None,
            &mut out,
        )
        .unwrap();
        ALLOCS.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::Relaxed);
        for _ in 0..5 {
            prepare_model_output(
                &input,
                &pitch,
                48_000,
                48_000,
                4800,
                &mut smoother,
                None,
                &mut out,
            )
            .unwrap();
        }
        TRACK.store(false, Ordering::Relaxed);
        assert_eq!(ALLOCS.load(Ordering::Relaxed), 0, "{kind:?} allocated");
    }
}
