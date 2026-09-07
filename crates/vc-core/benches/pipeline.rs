//! Microbenchmarks for the `vc-core` CPU hot paths.
//!
//! These cover the pure-CPU stages a realtime chunk passes through on the
//! worker thread — resampling, RMS/volume shaping, the SOLA/PSOLA cross-fade
//! join, and the input noise gate. They deliberately avoid ONNX model inference
//! (`RvcPipeline::process`): the model stages are non-deterministic and need GPU
//! SDKs, whereas everything here is deterministic and runs with no GPU stack
//! (`cargo bench -p vc-core`). The `dsp`/`sola` public API is what the realtime
//! worker actually spends its non-inference time in, so regressions here map
//! directly to added per-chunk latency.
//!
//! Input sizes mirror real usage: chunk lengths are quoted in 48 kHz model-rate
//! samples (the RVC output domain), and resampling benches use the 48k<->16k
//! conversions the embedder/F0 front-end performs every chunk.

use std::time::Duration;

use divan::{black_box, Bencher};
use vc_core::dsp::{self, NoiseGate, RmsMixScratch, StreamingResampleMono};
use vc_core::model_rvc::{ChunkConverter, ChunkOutputConfig, ModelOutput, VoiceModel};
use vc_core::sola::SmoothingKind;
use vc_core::sola::{model_domain_chunk_smoother, prepare_model_output, ChunkSmootherConfig};

// Wrap the system allocator so divan reports alloc/dealloc counts per iteration.
// The buffer-reuse work on the worker path targets *zero* steady-state heap
// traffic; wall-clock time alone can't show that, but the AllocProfiler's
// "alloc" column makes a regression (a per-chunk Vec creeping back in) visible.
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

// Representative chunk lengths at the 48 kHz model sample rate: ~100 ms and
// ~250 ms, the range a realtime block typically falls in.
const CHUNK_48K_100MS: usize = 4_800;
const CHUNK_48K_250MS: usize = 12_000;

/// Deterministic voiced-ish test signal: a 220 Hz tone plus a quieter harmonic,
/// so SOLA cross-correlation and the RMS envelope see realistic structure
/// instead of a pure sine. Amplitude stays well inside [-1, 1].
fn synthetic_signal(len: usize, sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate as f32;
    (0..len)
        .map(|i| {
            let t = i as f32 / sr;
            0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect()
}

mod dsp_bench {
    use super::*;

    #[divan::bench(args = [CHUNK_48K_100MS, CHUNK_48K_250MS])]
    fn rms(bencher: Bencher, len: usize) {
        let signal = synthetic_signal(len, 48_000);
        bencher.bench_local(|| dsp::rms(black_box(&signal)));
    }

    #[divan::bench(args = [CHUNK_48K_100MS, CHUNK_48K_250MS])]
    fn compute_rms_envelope_into(bencher: Bencher, len: usize) {
        let signal = synthetic_signal(len, 48_000);
        let mut out = Vec::new();
        bencher.bench_local(|| {
            dsp::compute_rms_envelope_into(black_box(&signal), 48_000, &mut out);
            black_box(&out);
        });
    }

    // Volume-following mix of the model output toward the input reference RMS;
    // runs once per chunk. The rates cover both exact fast paths (0.0/0.5) and
    // the generic powf path (0.9). Scratch is reused, matching the worker.
    #[divan::bench(args = [
        (CHUNK_48K_100MS, 0.0_f32),
        (CHUNK_48K_100MS, 0.5_f32),
        (CHUNK_48K_100MS, 0.9_f32),
        (CHUNK_48K_250MS, 0.0_f32),
        (CHUNK_48K_250MS, 0.5_f32),
        (CHUNK_48K_250MS, 0.9_f32),
    ])]
    fn apply_rms_mix_with_scratch(bencher: Bencher, (len, rms_mix_rate): (usize, f32)) {
        let reference = synthetic_signal(len, 48_000);
        let mut scratch = RmsMixScratch::default();
        let template = synthetic_signal(len, 48_000);
        bencher
            .with_inputs(|| template.clone())
            .bench_local_values(|mut output| {
                dsp::apply_rms_mix_with_scratch(
                    black_box(&reference),
                    &mut output,
                    48_000,
                    rms_mix_rate,
                    &mut scratch,
                );
                black_box(output);
            });
    }

    // One-shot resampler (allocates a fresh FFT plan + output each call) for both
    // front-end directions: 48k->16k feeds the embedder/F0, 16k->48k and the
    // model->output conversions go the other way.
    #[divan::bench(args = [(48_000, 16_000), (16_000, 48_000)])]
    fn resample_mono(bencher: Bencher, (from, to): (usize, usize)) {
        let signal = synthetic_signal(from, from as u32); // ~1 s buffer
        bencher.bench_local(|| dsp::resample_mono(black_box(&signal), from, to).unwrap());
    }

    // Streaming resampler reusing its plan/scratch across chunks — the path the
    // worker actually drives. Each iteration pushes one 100 ms chunk through a
    // persistent resampler.
    #[divan::bench(args = [(48_000, 16_000), (16_000, 48_000)])]
    fn streaming_resample_chunk(bencher: Bencher, (from, to): (usize, usize)) {
        let chunk = synthetic_signal(from / 10, from as u32); // ~100 ms
        let mut resampler = StreamingResampleMono::new(from, to).unwrap();
        let mut out = Vec::new();
        // Warm the internal pending buffer so we measure steady-state cost.
        resampler.process_into(&chunk, &mut out).unwrap();
        bencher.bench_local(|| {
            out.clear();
            resampler.process_into(black_box(&chunk), &mut out).unwrap();
            black_box(&out);
        });
    }

    // SOLA offset search: normalized cross-correlation over a crossfade-sized
    // reference against a candidate with the search window appended. Sizes are
    // the CLI defaults at 48 kHz (85 ms crossfade, 12 ms search).
    #[divan::bench]
    fn sola_offset(bencher: Bencher) {
        let crossfade = dsp::chunk_samples_for_rate(48_000, 85);
        let search = dsp::chunk_samples_for_rate(48_000, 12);
        let reference = synthetic_signal(crossfade, 48_000);
        let candidate = synthetic_signal(crossfade + search, 48_000);
        bencher
            .bench_local(|| dsp::sola_offset(black_box(&candidate), black_box(&reference), search));
    }

    #[divan::bench]
    fn sola_offset_with_threshold(bencher: Bencher) {
        let crossfade = dsp::chunk_samples_for_rate(48_000, 85);
        let search = dsp::chunk_samples_for_rate(48_000, 12);
        let reference = synthetic_signal(crossfade, 48_000);
        let candidate = synthetic_signal(crossfade + search, 48_000);
        bencher.bench_local(|| {
            dsp::sola_offset_with_threshold(
                black_box(&candidate),
                black_box(&reference),
                search,
                1e-4,
            )
        });
    }

    #[divan::bench]
    fn crossfade(bencher: Bencher) {
        let len = dsp::chunk_samples_for_rate(48_000, 85);
        let prev_tail = synthetic_signal(len, 48_000);
        let template = synthetic_signal(len, 48_000);
        bencher
            .with_inputs(|| template.clone())
            .bench_local_values(|mut current| {
                dsp::crossfade(black_box(&prev_tail), &mut current);
                black_box(current);
            });
    }

    // Input denoise gate, run in place over a chunk at the input sample rate.
    #[divan::bench(args = [CHUNK_48K_100MS, CHUNK_48K_250MS])]
    fn noise_gate_process_in_place(bencher: Bencher, len: usize) {
        let template = synthetic_signal(len, 48_000);
        bencher
            .with_inputs(|| {
                (
                    NoiseGate::new(48_000.0, 0.01, 5.0, 50.0, 0.0),
                    template.clone(),
                )
            })
            .bench_local_values(|(mut gate, mut buf)| {
                gate.process_in_place(&mut buf);
                black_box(buf);
            });
    }

    #[divan::bench(args = [CHUNK_48K_100MS, CHUNK_48K_250MS])]
    fn i16_to_f32_into(bencher: Bencher, len: usize) {
        let input: Vec<i16> = (0..len).map(|i| (i as i16).wrapping_mul(37)).collect();
        let mut out = vec![0.0f32; len];
        bencher.bench_local(|| {
            dsp::i16_to_f32_into(black_box(&input), &mut out);
            black_box(&out);
        });
    }

    #[divan::bench(args = [CHUNK_48K_100MS, CHUNK_48K_250MS])]
    fn f32_to_i16_into(bencher: Bencher, len: usize) {
        let input = synthetic_signal(len, 48_000);
        let mut out = vec![0i16; len];
        bencher.bench_local(|| {
            dsp::f32_to_i16_into(black_box(&input), &mut out);
            black_box(&out);
        });
    }

    #[divan::bench(args = [
        (CHUNK_48K_100MS, 2_usize),
        (CHUNK_48K_100MS, 4_usize),
        (CHUNK_48K_100MS, 6_usize),
        (CHUNK_48K_250MS, 2_usize),
        (CHUNK_48K_250MS, 4_usize),
        (CHUNK_48K_250MS, 6_usize),
    ])]
    fn downmix_to_mono_into(bencher: Bencher, (frames, channels): (usize, usize)) {
        let input = synthetic_signal(frames * channels, 48_000);
        let mut out = vec![0.0f32; frames];
        bencher.bench_local(|| {
            dsp::downmix_to_mono_into(black_box(&input), channels, &mut out);
            black_box(&out);
        });
    }

    #[divan::bench(args = [
        (CHUNK_48K_100MS, 1_usize),
        (CHUNK_48K_100MS, 2_usize),
        (CHUNK_48K_100MS, 4_usize),
        (CHUNK_48K_100MS, 6_usize),
        (CHUNK_48K_250MS, 1_usize),
        (CHUNK_48K_250MS, 2_usize),
        (CHUNK_48K_250MS, 4_usize),
        (CHUNK_48K_250MS, 6_usize),
    ])]
    fn upmix_mono_into(bencher: Bencher, (frames, channels): (usize, usize)) {
        let mono = synthetic_signal(frames, 48_000);
        let mut out = vec![0.0f32; frames * channels];
        bencher.bench_local(|| {
            dsp::upmix_mono_into(black_box(&mono), channels, &mut out);
            black_box(&out);
        });
    }

    #[divan::bench(args = [
        (CHUNK_48K_100MS, 1.0_f32, 1.0_f32),
        (CHUNK_48K_100MS, 1.0_f32, 2.0_f32),
        (CHUNK_48K_100MS, 0.5_f32, 2.0_f32),
        (CHUNK_48K_250MS, 1.0_f32, 1.0_f32),
        (CHUNK_48K_250MS, 1.0_f32, 2.0_f32),
        (CHUNK_48K_250MS, 0.5_f32, 2.0_f32),
    ])]
    fn output_level_finalize(bencher: Bencher, (len, envelope, applied_gain): (usize, f32, f32)) {
        let template = synthetic_signal(len, 48_000);
        bencher
            .with_inputs(|| template.clone())
            .bench_local_values(|mut output| {
                dsp::clamp_scale_in_place(&mut output, envelope);
                let output_rms_before_gain = dsp::rms(&output);
                black_box(output_rms_before_gain);
                let output_rms = if (applied_gain - 1.0).abs() > f32::EPSILON {
                    dsp::apply_gain_and_rms(&mut output, applied_gain)
                } else {
                    output_rms_before_gain
                };
                black_box(output_rms);
                black_box(output);
            });
    }
}

mod sola_bench {
    use super::*;

    fn smoother_config(kind: SmoothingKind, output_chunk_samples: usize) -> ChunkSmootherConfig {
        // Mirrors the CLI's ChunkOutputConfig defaults (crates/vc-cli/src/engine.rs):
        // SOLA/PSOLA run in the 48 kHz model domain with output also at 48 kHz, so
        // no resampling masks the join cost.
        ChunkSmootherConfig {
            kind,
            output_chunk_samples,
            output_sample_rate: 48_000,
            model_sample_rate: 48_000,
            crossfade_ms: 85,
            sola_search_ms: 12,
            tail_discard_ms: 10,
        }
    }

    // Full chunk-join path: `prepare_model_output` runs the SOLA/PSOLA offset
    // search + cross-fade and fits the result to the output chunk. This is the
    // closest model-free stand-in for a realtime chunk's post-inference cost.
    // The smoother is primed once so we measure steady-state joins; the model
    // audio/pitchf and the output buffer are reused like the realtime worker.
    #[divan::bench(args = [SmoothingKind::Sola, SmoothingKind::Psola])]
    fn prepare_model_output_chunk(bencher: Bencher, kind: SmoothingKind) {
        let chunk = CHUNK_48K_100MS;
        let mut joiner = model_domain_chunk_smoother(smoother_config(kind, chunk));
        // The model output is longer than the chunk so the joiner has a search
        // window; pitchf gives a stable voiced F0 so PSOLA takes its pitch path.
        let raw_len = chunk + dsp::chunk_samples_for_rate(48_000, 97);
        let audio = synthetic_signal(raw_len, 48_000);
        let pitchf = vec![220.0f32; 256];
        joiner.prime_model_output(&audio, &pitchf);

        // Reuse the output buffer across iterations exactly like the realtime
        // worker reuses its `prepared`/`chunk_out` Vec across chunks. Warm its
        // capacity once so the AllocProfiler measures steady-state traffic: a
        // healthy same-rate join shows 0 allocs here.
        let mut out = Vec::new();
        let _ = prepare_model_output(
            &audio,
            &pitchf,
            48_000,
            48_000,
            chunk,
            &mut joiner,
            None,
            &mut out,
        )
        .unwrap();
        bencher.bench_local(|| {
            let sola_offset = prepare_model_output(
                &audio,
                &pitchf,
                48_000,
                48_000,
                chunk,
                black_box(&mut joiner),
                None,
                black_box(&mut out),
            )
            .unwrap();
            black_box(sola_offset);
        });
    }
}

mod model_free_pipeline_bench {
    use super::*;

    struct MockVoiceModel {
        audio: Vec<f32>,
        pitchf: Vec<f32>,
        output_rms: f32,
    }

    impl MockVoiceModel {
        fn new(output_samples: usize) -> Self {
            let audio = synthetic_signal(output_samples, 48_000);
            let pitchf = vec![220.0f32; (output_samples / 480).max(1)];
            let output_rms = dsp::rms(&audio);
            Self {
                audio,
                pitchf,
                output_rms,
            }
        }
    }

    impl VoiceModel for MockVoiceModel {
        fn process(
            &mut self,
            audio: &[f32],
            _sample_rate: u32,
            out_audio: &mut Vec<f32>,
            out_pitchf: &mut Vec<f32>,
        ) -> anyhow::Result<ModelOutput> {
            out_audio.clear();
            out_audio.extend_from_slice(&self.audio);
            out_pitchf.clear();
            out_pitchf.extend_from_slice(&self.pitchf);
            let input_rms = dsp::rms(audio);

            // The model sessions are intentionally replaced here: this bench
            // measures the shared chunk conversion and join boundary without
            // charging ContentVec/RMVPE/RVC generator execution or model loads.
            Ok(ModelOutput {
                sample_rate: 48_000,
                inference_time: Duration::ZERO,
                embedder_time: Duration::ZERO,
                pitch_time: Duration::ZERO,
                rvc_time: Duration::ZERO,
                input_rms,
                voiced_ratio: 1.0,
                raw_output_samples: out_audio.len(),
                output_rms: self.output_rms,
                applied_output_gain: 1.0,
                feature_frames: self.pitchf.len(),
                pitch_frames: self.pitchf.len(),
                silent: false,
                convert_size: out_audio.len(),
                out_size: out_audio.len(),
                model_input_samples: audio.len(),
                volume: input_rms,
            })
        }
    }

    fn output_config(kind: SmoothingKind, output_chunk_samples: usize) -> ChunkOutputConfig {
        ChunkOutputConfig {
            kind,
            output_sample_rate: 48_000,
            output_chunk_samples,
            crossfade_ms: 85,
            sola_search_ms: 12,
            tail_discard_ms: 10,
        }
    }

    fn model_output_samples_for(output_chunk_samples: usize) -> usize {
        output_chunk_samples + dsp::chunk_samples_for_rate(48_000, 85 + 12 + 10)
    }

    // Integrated model-free chunk path: fake model output -> ChunkConverter ->
    // SOLA/PSOLA fixed-size output. This sits above the DSP/SOLA unit benches
    // while still avoiding all model inference and GPU/runtime dependencies.
    #[divan::bench(args = [
        (SmoothingKind::Sola, CHUNK_48K_100MS),
        (SmoothingKind::Psola, CHUNK_48K_100MS),
        (SmoothingKind::Sola, CHUNK_48K_250MS),
        (SmoothingKind::Psola, CHUNK_48K_250MS),
    ])]
    fn chunk_converter_process_chunk(
        bencher: Bencher,
        (kind, output_chunk_samples): (SmoothingKind, usize),
    ) {
        let input = synthetic_signal(output_chunk_samples, 48_000);
        let model = MockVoiceModel::new(model_output_samples_for(output_chunk_samples));
        let mut converter = ChunkConverter::new(model, output_config(kind, output_chunk_samples));
        let mut out = Vec::new();

        // Prime the smoother and reusable Vec capacities so divan reports the
        // steady-state worker cost instead of startup allocation.
        converter.process_chunk(&input, 48_000, &mut out).unwrap();

        bencher.bench_local(|| {
            let stats = converter
                .process_chunk(black_box(&input), 48_000, black_box(&mut out))
                .unwrap();
            black_box((stats, converter.last_join_diagnostics(), out.len()));
        });
    }
}
