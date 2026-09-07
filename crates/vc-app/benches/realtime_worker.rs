//! Worker-boundary benchmarks for the standalone realtime path.
//!
//! These measure from "one chunk is already queued in the input ring" to "one
//! converted chunk has been pushed to the output ring". Model execution is
//! replaced with a deterministic fake that still pays input 48k->16k resampling,
//! so the numbers are a coarse non-GPU worker cost rather than RVC inference
//! latency.

use std::time::Duration;

use divan::{black_box, Bencher};
use rtrb::{Consumer, Producer, RingBuffer};
use vc_core::dsp::{self, StreamingResampleMono};
use vc_core::model_rvc::{ChunkConverter, ChunkOutputConfig, ChunkStats, ModelOutput, VoiceModel};
use vc_core::sola::SmoothingKind;

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

const INPUT_RATE: u32 = 48_000;
const MODEL_RATE: u32 = 48_000;
const EMBEDDER_RATE: u32 = 16_000;
const CROSSFADE_MS: u32 = 85;
const SOLA_SEARCH_MS: u32 = 12;
const TAIL_DISCARD_MS: u32 = 10;
const QUEUE_CHUNKS: usize = 4;

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

fn accumulate_input_chunk(
    consumer: &mut Consumer<f32>,
    input_acc: &mut Vec<f32>,
    input_chunk: usize,
) -> bool {
    let available = consumer
        .slots()
        .min(input_chunk.saturating_sub(input_acc.len()));
    if available > 0 {
        let old = input_acc.len();
        input_acc.resize(old + available, 0.0);
        if consumer.pop_entire_slice(&mut input_acc[old..]).is_err() {
            input_acc.truncate(old);
        }
    }
    input_acc.len() >= input_chunk
}

struct ResamplingFakeModel {
    resampler_16k: StreamingResampleMono,
    input_rate: u32,
    audio_16k: Vec<f32>,
    model_audio: Vec<f32>,
    model_pitchf: Vec<f32>,
    model_rms: f32,
}

impl ResamplingFakeModel {
    fn new(input_rate: u32, model_output_samples: usize) -> Self {
        let model_audio = synthetic_signal(model_output_samples, MODEL_RATE);
        let model_pitchf = vec![220.0; (model_output_samples / 480).max(1)];
        let model_rms = dsp::rms(&model_audio);
        Self {
            resampler_16k: StreamingResampleMono::new(input_rate as usize, EMBEDDER_RATE as usize)
                .unwrap(),
            input_rate,
            audio_16k: Vec::new(),
            model_audio,
            model_pitchf,
            model_rms,
        }
    }
}

impl VoiceModel for ResamplingFakeModel {
    fn process(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        out_audio: &mut Vec<f32>,
        out_pitchf: &mut Vec<f32>,
    ) -> anyhow::Result<ModelOutput> {
        if sample_rate != self.input_rate {
            self.resampler_16k =
                StreamingResampleMono::new(sample_rate as usize, EMBEDDER_RATE as usize)?;
            self.input_rate = sample_rate;
        }
        self.audio_16k.clear();
        self.resampler_16k
            .process_into(audio, &mut self.audio_16k)?;
        let input_rms = dsp::rms(&self.audio_16k);

        out_audio.clear();
        out_audio.extend_from_slice(&self.model_audio);
        out_pitchf.clear();
        out_pitchf.extend_from_slice(&self.model_pitchf);

        Ok(ModelOutput {
            sample_rate: MODEL_RATE,
            inference_time: Duration::ZERO,
            embedder_time: Duration::ZERO,
            pitch_time: Duration::ZERO,
            rvc_time: Duration::ZERO,
            input_rms,
            voiced_ratio: 1.0,
            raw_output_samples: out_audio.len(),
            output_rms: self.model_rms,
            applied_output_gain: 1.0,
            feature_frames: self.model_pitchf.len(),
            pitch_frames: self.model_pitchf.len(),
            silent: false,
            convert_size: out_audio.len(),
            out_size: out_audio.len(),
            model_input_samples: audio.len(),
            volume: input_rms,
        })
    }
}

struct WorkerBench {
    _input_producer: Producer<f32>,
    input_consumer: Consumer<f32>,
    output_producer: Producer<f32>,
    _output_consumer: Consumer<f32>,
    converter: ChunkConverter<ResamplingFakeModel>,
    input_acc: Vec<f32>,
    prepared: Vec<f32>,
    input_chunk: usize,
}

impl WorkerBench {
    fn new(kind: SmoothingKind, chunk_ms: u32, output_rate: u32) -> Self {
        let input_chunk = dsp::chunk_samples_for_rate(INPUT_RATE, chunk_ms);
        let output_chunk = dsp::chunk_samples_for_rate(output_rate, chunk_ms);
        let model_output_samples = dsp::chunk_samples_for_rate(
            MODEL_RATE,
            chunk_ms + CROSSFADE_MS + SOLA_SEARCH_MS + TAIL_DISCARD_MS,
        );
        let model = ResamplingFakeModel::new(INPUT_RATE, model_output_samples);
        let mut converter = ChunkConverter::new(
            model,
            ChunkOutputConfig {
                kind,
                output_sample_rate: output_rate,
                output_chunk_samples: output_chunk,
                crossfade_ms: CROSSFADE_MS,
                sola_search_ms: SOLA_SEARCH_MS,
                tail_discard_ms: TAIL_DISCARD_MS,
            },
        );

        let input = synthetic_signal(input_chunk, INPUT_RATE);
        let mut prepared = Vec::with_capacity(output_chunk * 2);
        converter
            .process_chunk(&input, INPUT_RATE, &mut prepared)
            .unwrap();

        let (mut input_producer, input_consumer) =
            RingBuffer::<f32>::new(input_chunk * QUEUE_CHUNKS);
        let (output_producer, output_consumer) =
            RingBuffer::<f32>::new(output_chunk * QUEUE_CHUNKS);
        let (_, input_remainder) = input_producer.push_partial_slice(&input);
        assert!(input_remainder.is_empty());

        Self {
            _input_producer: input_producer,
            input_consumer,
            output_producer,
            _output_consumer: output_consumer,
            converter,
            input_acc: Vec::with_capacity(input_chunk * 2),
            prepared,
            input_chunk,
        }
    }

    fn process_ring_to_ring(&mut self) -> ChunkStats {
        let ready = accumulate_input_chunk(
            &mut self.input_consumer,
            &mut self.input_acc,
            self.input_chunk,
        );
        assert!(ready);

        let stats = self
            .converter
            .process_chunk(
                &self.input_acc[..self.input_chunk],
                INPUT_RATE,
                &mut self.prepared,
            )
            .unwrap();
        self.input_acc.clear();

        let (_, output_remainder) = self.output_producer.push_partial_slice(&self.prepared);
        assert!(output_remainder.is_empty());
        stats
    }
}

#[divan::bench(args = [
    (SmoothingKind::Sola, 100_u32, 48_000_u32),
    (SmoothingKind::Psola, 100_u32, 48_000_u32),
    (SmoothingKind::Sola, 250_u32, 48_000_u32),
    (SmoothingKind::Psola, 250_u32, 48_000_u32),
    (SmoothingKind::Sola, 100_u32, 44_100_u32),
    (SmoothingKind::Psola, 100_u32, 44_100_u32),
    (SmoothingKind::Sola, 250_u32, 44_100_u32),
    (SmoothingKind::Psola, 250_u32, 44_100_u32),
])]
fn ring_to_ring_non_gpu(
    bencher: Bencher,
    (kind, chunk_ms, output_rate): (SmoothingKind, u32, u32),
) {
    bencher
        .with_inputs(|| WorkerBench::new(kind, chunk_ms, output_rate))
        .bench_local_values(|mut state| {
            let stats = state.process_ring_to_ring();
            black_box((stats, state.prepared.len()));
        });
}
