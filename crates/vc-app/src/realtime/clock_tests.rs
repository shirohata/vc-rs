use super::*;
use vc_core::model_rvc::{ModelOutput, VoiceModel};
use vc_core::sola::SmoothingKind;

// Isolate delayed waveform history from neural inference variation. The
// silence metadata exactly follows the pipeline's two-silent-input contract.
struct RollingIdentity {
    history: Vec<f32>,
    candidate: usize,
    prev_silent: bool,
}

impl VoiceModel for RollingIdentity {
    fn process(
        &mut self,
        input: &[f32],
        rate: u32,
        out: &mut Vec<f32>,
        pitch: &mut Vec<f32>,
    ) -> Result<ModelOutput> {
        let is_silent = dsp::rms(input) < 0.0001;
        let silent = is_silent && self.prev_silent;
        self.prev_silent = is_silent;
        self.history.extend_from_slice(input);
        if self.history.len() > self.candidate {
            self.history.drain(..self.history.len() - self.candidate);
        }
        out.clear();
        out.resize(self.candidate - self.history.len(), 0.0);
        out.extend_from_slice(&self.history);
        pitch.clear();
        pitch.resize(self.candidate / (rate as usize / 100), 200.0);
        Ok(ModelOutput {
            sample_rate: rate,
            inference_time: Duration::ZERO,
            embedder_time: Duration::ZERO,
            pitch_time: Duration::ZERO,
            rvc_time: Duration::ZERO,
            input_rms: dsp::rms(input),
            voiced_ratio: 1.0,
            raw_output_samples: out.len(),
            output_rms: dsp::rms(out),
            applied_output_gain: 1.0,
            feature_frames: pitch.len(),
            pitch_frames: pitch.len(),
            silent,
            convert_size: self.candidate,
            out_size: out.len(),
            model_input_samples: self.history.len(),
            volume: 1.0,
        })
    }
}

#[test]
fn two_silent_inputs_preserve_the_still_voiced_joined_tail() {
    for kind in [SmoothingKind::Sola, SmoothingKind::Psola] {
        for adaptive in [false, true] {
            let model = RollingIdentity {
                history: Vec::new(),
                candidate: 2_080,
                prev_silent: true,
            };
            let config = ChunkOutputConfig {
                kind,
                output_sample_rate: 16_000,
                output_chunk_samples: 320,
                crossfade_ms: 10,
                sola_search_ms: 0,
                tail_discard_ms: 100,
            };
            let mut converter = if adaptive {
                ChunkConverter::new_adaptive(model, config)
            } else {
                ChunkConverter::new(model, config)
            };
            let mut out = Vec::new();
            for _ in 0..20 {
                converter
                    .process_chunk(&[0.5; 320], 16_000, &mut out)
                    .unwrap();
            }
            let (mut tx, mut rx) = RingBuffer::new(1_280);
            tx.push_entire_slice(&[0.5; 640]).unwrap();
            assert!(
                !converter
                    .process_chunk(&[0.0; 320], 16_000, &mut out)
                    .unwrap()
                    .silent
            );
            queue_output(&mut tx, &out, &Telemetry::default());
            rx.pop_entire_slice(&mut [0.0; 320]).unwrap();
            let stats = converter
                .process_chunk(&[0.0; 320], 16_000, &mut out)
                .unwrap();
            assert!(stats.silent);
            assert!(out.iter().all(|sample| *sample > 0.49));
            let buffered = 1_280 - tx.slots();
            assert!(buffered > 320 && tx.slots() >= out.len());
            let telemetry = Telemetry::default();
            queue_output(&mut tx, &out, &telemetry);
            assert_eq!(telemetry.snapshot().output_dropped_samples, 0);
            let mut played = vec![0.0; buffered + out.len()];
            rx.pop_entire_slice(&mut played).unwrap();
            assert_eq!(&played[buffered..], &out);
        }
    }
}

fn simulate_audio_clocks(from: u32, to: u32, drift_ppm: i64, seconds: u64) {
    let input_hop = from as usize / 50;
    let output_hop = to as usize / 50;
    let (mut tx, mut rx) = RingBuffer::new(output_hop * 4);
    let live = LiveParams::default();
    let mut processor = PassthroughProcessor::new(
        DenoiserMode::Off,
        NoiseGateShaping::default(),
        from,
        to,
        None,
        #[cfg(feature = "gtcrn")]
        vc_core::denoise::GtcrnBackend::OrtCpu,
        &live,
    )
    .unwrap();
    processor.enable_adaptive_output();
    let mut clock = crate::clock_drift::ClockDriftController::new(to);
    let input = vec![0.2; input_hop];
    let mut out = Vec::new();
    let mut input_acc = 0;
    let mut received = 0;
    let mut rendered = 0;
    let mut primed = false;
    let mut playback = vec![0.0; (to as usize / 1000) + 1];
    let telemetry = Telemetry::default();
    for tick in 1..=seconds * 1000 {
        let total = (tick as i128 * from as i128 * (1_000_000 + drift_ppm) as i128 / 1_000_000_000)
            as usize;
        input_acc += total - received;
        received = total;
        while input_acc >= input_hop {
            processor.correction_ppm = clock.update(output_hop * 4 - tx.slots(), 0.02, primed);
            processor.process_chunk(&input, &live, &mut out).unwrap();
            if !primed {
                let reserve = crate::clock_drift::startup_reserve(to, playback.len(), output_hop);
                tx.push_entire_slice(&vec![0.0; reserve]).unwrap();
                primed = true;
            }
            queue_output(&mut tx, &out, &telemetry);
            assert_eq!(
                telemetry.snapshot().output_dropped_samples,
                0,
                "{from}->{to}/{drift_ppm}, t={tick}"
            );
            input_acc -= input_hop;
        }
        let total_out = (tick * to as u64 / 1000) as usize;
        let frames = total_out - rendered;
        rendered = total_out;
        if primed {
            rx.pop_entire_slice(&mut playback[..frames])
                .unwrap_or_else(|e| panic!("{from}->{to}/{drift_ppm}, t={tick}: {e}"));
            assert!(playback[..frames].iter().all(|x| x.is_finite()));
            if tick > 1000 {
                assert!(playback[..frames].iter().all(|x| (*x - 0.2).abs() < 1e-4));
            }
        }
    }
}

#[test]
fn real_passthrough_resampler_and_rings_follow_independent_clocks() {
    for ppm in [-500, 500] {
        simulate_audio_clocks(8_000, 8_000, ppm, 120);
        simulate_audio_clocks(44_100, 48_000, ppm, 15);
        simulate_audio_clocks(48_000, 44_100, ppm, 15);
    }
}

#[test]
fn passthrough_reset_discards_filter_history_and_keeps_clock_estimate() {
    let live = LiveParams::default();
    let mut processor = PassthroughProcessor::new(
        DenoiserMode::Off,
        NoiseGateShaping::default(),
        48_000,
        48_000,
        None,
        #[cfg(feature = "gtcrn")]
        vc_core::denoise::GtcrnBackend::OrtCpu,
        &live,
    )
    .unwrap();
    processor.enable_adaptive_output();
    processor.correction_ppm = 123.0;
    let mut initial = Vec::new();
    processor
        .process_chunk(&[0.25; 960], &live, &mut initial)
        .unwrap();
    let mut out = Vec::new();
    processor
        .process_chunk(&[0.5; 960], &live, &mut out)
        .unwrap();
    processor.reset(&live).unwrap();
    assert_eq!(processor.correction_ppm, 123.0);
    processor
        .process_chunk(&[0.25; 960], &live, &mut out)
        .unwrap();
    assert_eq!(initial, out);
}
