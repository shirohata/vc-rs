//! Isolated allocator test: only allocations inside the warmed finite-resample
//! calls are counted. No GPU runtime, logging, input generation or assertions
//! run while counting, so the result specifically guards the RMS scratch path.
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
fn warmed_independent_reference_windows_do_not_allocate() {
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
}
