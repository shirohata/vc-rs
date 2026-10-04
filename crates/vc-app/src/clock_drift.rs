use vc_core::dsp::MAX_CLOCK_CORRECTION_PPM;

/// Worker-only queue servo. Observe the ring at the same input-hop boundary,
/// before inference: post-inference occupancy would also measure GPU jitter.
/// The initial phase is learned rather than targeting whole output chunks,
/// which could add seconds of latency at large hop settings.
pub(crate) struct ClockDriftController {
    sample_rate: f64,
    target: Option<f64>,
    filtered_error: f64,
    integral: f64,
    ppm: f64,
}

impl ClockDriftController {
    pub(crate) fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate: sample_rate as f64,
            target: None,
            filtered_error: 0.0,
            integral: 0.0,
            ppm: 0.0,
        }
    }

    /// Preserve the clock estimate across route changes/discontinuities, but
    /// discard queue phase/error history. Never integrate overload or startup.
    pub(crate) fn rebase(&mut self) {
        self.target = None;
        self.filtered_error = 0.0;
        self.integral = -self.ppm * 1e-6;
    }

    pub(crate) fn target_samples(&self) -> usize {
        self.target
            .map_or(0, |seconds| (seconds * self.sample_rate).round() as usize)
    }

    pub(crate) fn update(&mut self, buffered: usize, hop_seconds: f64, reliable: bool) -> f64 {
        if !reliable {
            self.rebase();
            return self.ppm;
        }
        let queue_seconds = buffered as f64 / self.sample_rate;
        let Some(target) = self.target else {
            self.target = Some(queue_seconds);
            return self.ppm;
        };
        // A two-second low-pass rejects callback quantization and ordinary
        // scheduling jitter. PI gains give a ~50 s time constant without fast pitch
        // modulation. Positive backlog error decreases output/input ratio.
        let alpha = 1.0 - (-hop_seconds / 2.0).exp();
        self.filtered_error += alpha * (queue_seconds - target - self.filtered_error);
        let proposed_integral = self.integral + 0.0004 * self.filtered_error * hop_seconds;
        let requested = -(0.04 * self.filtered_error + proposed_integral) * 1e6;
        if requested.abs() <= MAX_CLOCK_CORRECTION_PPM || requested * self.filtered_error > 0.0 {
            self.integral = proposed_integral;
        }
        let requested = (-(0.04 * self.filtered_error + self.integral) * 1e6)
            .clamp(-MAX_CLOCK_CORRECTION_PPM, MAX_CLOCK_CORRECTION_PPM);
        let slew = 50.0 * hop_seconds;
        self.ppm += (requested - self.ppm).clamp(-slew, slew);
        self.ppm
    }
}

/// Short startup reserve independent of the inference hop. Two observed
/// callback blocks cover output quantization; the capacity bound leaves room
/// for the next converted hop and its maximum clock correction.
pub(crate) fn startup_reserve(rate: u32, callback_frames: usize, hop: usize) -> usize {
    (rate as usize / 50)
        .max(callback_frames.saturating_mul(2))
        .min(hop * 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Virtual device callbacks with fractional sample accounting, not rounded
    /// hops. No wall-clock waits or neural model are needed for a two-hour run.
    fn simulate(
        rate: u32,
        hop_ms: usize,
        drift_ppm: f64,
        variable_callbacks: bool,
        changing_drift: bool,
        worker_jitter: bool,
    ) {
        let hop = rate as usize * hop_ms / 1000;
        let dt = hop as f64 / rate as f64;
        let callback = rate as usize / 200;
        let reserve = startup_reserve(rate, callback, hop);
        let mut queue = reserve + hop;
        let capacity = hop * 4;
        let mut clock = ClockDriftController::new(rate);
        let mut next_input = dt / (1.0 + drift_ppm * 1e-6);
        let mut next_output = callback as f64 / rate as f64;
        let mut fraction = 0.0;
        let mut ppm = 0.0;
        let mut callback_index = 0;
        let mut min_queue = usize::MAX;
        let mut max_queue = 0;
        let mut settled_ppm_sum = 0.0;
        let mut settled_hops = 0;
        let mut input_index = 1;
        let mut previous_jitter = 0.0;
        while next_input < 7_200.0 {
            if next_input <= next_output {
                ppm = clock.update(queue, dt, true);
                if next_input > 6_600.0 {
                    settled_ppm_sum += ppm;
                    settled_hops += 1;
                }
                fraction += hop as f64 * (1.0 + ppm * 1e-6);
                let produced = fraction.floor() as usize;
                fraction -= produced as f64;
                queue += produced;
                assert!(
                    queue <= capacity,
                    "overflow: {rate}/{hop_ms}/{drift_ppm}, {queue}, {ppm}"
                );
                let current_drift = if changing_drift && next_input > 3_600.0 {
                    -drift_ppm
                } else {
                    drift_ppm
                };
                input_index += 1;
                let jitter = if worker_jitter {
                    [0.0, 0.003, 0.001, 0.005][input_index % 4]
                } else {
                    0.0
                };
                next_input += dt / (1.0 + current_drift * 1e-6) + jitter - previous_jitter;
                previous_jitter = jitter;
                max_queue = max_queue.max(queue);
            } else {
                let block = if variable_callbacks {
                    [callback / 2, callback, callback * 2, callback][callback_index % 4]
                } else {
                    callback
                };
                assert!(queue >= block, "underrun: {rate}/{hop_ms}/{drift_ppm}, t={next_output}, {queue} < {block}, ppm={ppm}");
                queue -= block;
                min_queue = min_queue.min(queue);
                callback_index += 1;
                let next_block = if variable_callbacks {
                    [callback / 2, callback, callback * 2, callback][callback_index % 4]
                } else {
                    callback
                };
                next_output += next_block as f64 / rate as f64;
            }
        }
        let final_drift = if changing_drift {
            -drift_ppm
        } else {
            drift_ppm
        };
        let expected = (1.0 / (1.0 + final_drift * 1e-6) - 1.0) * 1e6;
        // Instantaneous correction also follows callback phase; its long-term
        // mean, not a single arbitrary callback boundary, estimates the clock.
        let mean_ppm = settled_ppm_sum / settled_hops as f64;
        assert!(
            (mean_ppm - expected).abs() < 20.0,
            "convergence: {rate}/{hop_ms}/{drift_ppm}: {mean_ppm}, expected {expected}"
        );
        assert!(max_queue - min_queue < hop + rate as usize / 25);
    }

    #[test]
    fn independent_clocks_stay_bounded_for_two_virtual_hours() {
        for rate in [44_100, 48_000] {
            for hop_ms in [20, 30, 200, 2_000] {
                for ppm in [-500.0, -100.0, -50.0, 0.0, 50.0, 100.0, 500.0] {
                    simulate(rate, hop_ms, ppm, false, false, false);
                }
            }
        }
    }

    #[test]
    fn variable_callbacks_do_not_create_a_clock_error() {
        for drift in [-500.0, 0.0, 500.0] {
            simulate(48_000, 30, drift, true, false, false);
        }
    }

    #[test]
    fn changing_clocks_and_worker_jitter_remain_bounded() {
        simulate(48_000, 20, 500.0, false, true, true);
        simulate(44_100, 200, -500.0, false, true, true);
        simulate(48_000, 2_000, 100.0, false, true, true);
    }

    #[test]
    fn overload_freezes_learning_and_correction_is_bounded() {
        let mut clock = ClockDriftController::new(48_000);
        clock.update(960, 0.02, true);
        let mut ppm = 0.0;
        for _ in 0..10_000 {
            let next = clock.update(96_000, 0.02, true);
            assert!((next - ppm).abs() <= 1.000_001);
            ppm = next;
        }
        assert_eq!(ppm, -MAX_CLOCK_CORRECTION_PPM);
        for _ in 0..10_000 {
            assert_eq!(clock.update(0, 0.02, false), ppm);
        }
        assert_eq!(clock.update(900, 0.02, true), ppm);
        assert_eq!(clock.target_samples(), 900);
        assert!(clock.integral.abs() <= MAX_CLOCK_CORRECTION_PPM * 1e-6);
    }

    #[test]
    fn large_hops_do_not_add_a_whole_chunk_of_reserve() {
        assert_eq!(startup_reserve(48_000, 240, 96_000), 960);
        assert_eq!(startup_reserve(48_000, 960, 960), 1920);
    }
}
