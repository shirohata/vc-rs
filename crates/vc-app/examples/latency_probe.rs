//! Paced, device-free measurement of the shared conversion worker (not I/O latency).
//! Usage: latency_probe MODEL EMBEDDER F0 INPUT.wav REPORT.csv SECONDS [PROVIDER]
//! Uses 200/1000 ms, SOLA 85/12/10 ms, no denoiser, pitch +12, input gain 4,
//! and RMS mix 0. Keep these identical between binaries when comparing versions.
//! Paths are supplied at runtime; never commit local models or probe recordings.

use std::{
    fs::File,
    io::Write,
    path::Path,
    thread,
    time::{Duration, Instant},
};

use anyhow::{ensure, Context, Result};
use vc_core::{
    model_rvc::{
        set_process_gpu_priority, set_process_power_throttling, ChunkConverter, ChunkOutputConfig,
        F0Config, GpuPriority, NoiseGateShaping, OutputDynamicsConfig, RvcPipeline,
        RvcPipelineConfig,
    },
    sola::SmoothingKind,
    Provider,
};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        (6..=7).contains(&args.len()),
        "usage: latency_probe MODEL EMBEDDER F0 INPUT.wav REPORT.csv SECONDS [PROVIDER]"
    );
    let seconds: u64 = args[5].parse()?;
    ensure!(seconds > 0, "duration must be positive");
    let provider = Provider::from_name(args.get(6).map_or("tensorrt", String::as_str))
        .context("unknown provider")?;
    let mut reader = hound::WavReader::open(&args[3])?;
    let spec = reader.spec();
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2_f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|s| s.map(|s| s as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    let audio: Vec<f32> = interleaved
        .chunks_exact(usize::from(spec.channels))
        .map(|f| f.iter().sum::<f32>() / f.len() as f32)
        .collect();
    ensure!(!audio.is_empty(), "input must contain audio");
    let hop = vc_core::validation::RvcChunkTiming::from_ms(200, spec.sample_rate)?;
    set_process_gpu_priority(GpuPriority::High);
    set_process_power_throttling(true);
    if let Err(err) =
        thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Max)
    {
        eprintln!("worker priority unavailable: {err}");
    }
    let model = RvcPipeline::load(RvcPipelineConfig {
        model: Path::new(&args[0]),
        embedder: Path::new(&args[1]),
        embedder_output: None,
        f0_model: Path::new(&args[2]),
        provider,
        gpu_priority: GpuPriority::High,
        gpu_device_id: 0,
        sample_rate: spec.sample_rate,
        chunk_samples: hop.input_chunk_samples,
        speaker_id: 0,
        pitch_shift: 12.0,
        f0: F0Config::default(),
        input_gain: 4.0,
        noise_gate_enabled: false,
        noise_gate_threshold: 0.01,
        noise_gate_shaping: NoiseGateShaping::default(),
        output_extra_ms: 107,
        volume_excluded_ms: 85,
        extra_convert_ms: 1000,
        output_gain: 1.0,
        output_dynamics: OutputDynamicsConfig::default(),
        progress: None,
    })?;
    let input_delay = model.input_content_delay_samples(spec.sample_rate);
    let mut converter = ChunkConverter::new(
        model,
        ChunkOutputConfig {
            kind: SmoothingKind::Sola,
            output_sample_rate: spec.sample_rate,
            output_chunk_samples: hop.input_chunk_samples,
            crossfade_ms: 85,
            sola_search_ms: 12,
            tail_discard_ms: 10,
        },
    );
    let mut input = vec![0.0; hop.input_chunk_samples];
    let mut output = Vec::with_capacity(hop.input_chunk_samples);
    let mut position = 0;
    let mut next_input = |input: &mut [f32]| {
        for sample in input {
            *sample = audio[position];
            position = (position + 1) % audio.len();
        }
    };
    // Model load/cache build and first-use allocations are outside the measurement.
    for _ in 0..20 {
        next_input(&mut input);
        converter.process_chunk(&input, spec.sample_rate, &mut output)?;
    }
    let count = usize::try_from(seconds.checked_mul(5).context("duration overflow")?)?;
    let mut rows = Vec::with_capacity(count);
    let origin = Instant::now();
    for index in 0..count {
        let scheduled = origin + Duration::from_millis(index as u64 * 200);
        thread::sleep(scheduled.saturating_duration_since(Instant::now()));
        next_input(&mut input);
        let start = Instant::now();
        let stats = converter.process_chunk(&input, spec.sample_rate, &mut output)?;
        let elapsed = start.elapsed();
        ensure!(
            output.len() == hop.input_chunk_samples,
            "output hop changed"
        );
        rows.push((
            stats.inference_time.as_micros(),
            elapsed.as_micros(),
            start.saturating_duration_since(scheduled).as_micros(),
            Instant::now() > scheduled + Duration::from_millis(200),
        ));
    }
    // File I/O and percentile sorting are deliberately after the paced loop.
    let mut report = File::create(&args[4])?;
    writeln!(
        report,
        "chunk,inference_us,processing_us,start_late_us,deadline_missed"
    )?;
    for (index, (infer, total, late, missed)) in rows.iter().enumerate() {
        writeln!(report, "{index},{infer},{total},{late},{missed}")?;
    }
    let missed = rows.iter().filter(|r| r.3).count();
    let mut times: Vec<_> = rows.iter().map(|r| r.1).collect();
    times.sort_unstable();
    let percentile = |p: usize| times[(times.len() * p).div_ceil(100).saturating_sub(1)];
    println!("device_free_worker: chunks={count} rate={} input_content_samples={input_delay} content_samples={} processing_us p50={} p95={} p99={} deadline_misses={missed}",
        spec.sample_rate, converter.output_content_delay_samples(), percentile(50), percentile(95), percentile(99));
    Ok(())
}
