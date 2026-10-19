use std::sync::Arc;

use realfft::{
    num_complex::{Complex32, Complex64},
    ComplexToReal, RealFftPlanner, RealToComplex,
};

// With u = t/N - 1/2 and wrapped phase delta in [-pi, pi), |delta*u| <= pi/2.
// Degree 14 bounds the complex exponential Taylor remainder below 3.3e-9
// before spectral magnitude weighting. f64 inverse FFTs/accumulation keep
// synthesis rounding small; the direct-sum tests guard this approximation.
const PHASE_ORDER: usize = 14;

/// Applio 3.6.4's phase-interpolated overlap synthesis, after vc-rs's existing
/// SOLA alignment. This is a boundary crossfade, not a time-stretching STFT.
/// See rvc/realtime/core.py::phase_vocoder in the upstream 3.6.4 tag.
///
/// Worker-only: retained FFT plans/scratch and a bounded phase expansion reduce
/// the off-bin synthesis from O(N²) to O(PHASE_ORDER * N log N), with O(N) storage.
/// An ordinary single inverse FFT would lose the interpolated phase.
pub(super) struct PhaseVocoder {
    fft: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f64>>,
    input: Vec<f32>,
    previous: Vec<Complex32>,
    current: Vec<Complex32>,
    forward_scratch: Vec<Complex32>,
    inverse_output: Vec<f64>,
    coefficients: Vec<Complex64>,
    spectrum: Vec<Complex64>,
    delta: Vec<f64>,
    scratch: Vec<Complex64>,
    fade_in: Vec<f64>,
    window: Vec<f64>,
    synthesis: Vec<f64>,
    power: Vec<f64>,
    position: Vec<f64>,
}

impl PhaseVocoder {
    pub(super) fn new(len: usize) -> Self {
        assert!(len > 1);
        let fft = RealFftPlanner::new().plan_fft_forward(len);
        let inverse = RealFftPlanner::new().plan_fft_inverse(len);
        let fade_in: Vec<_> = (0..len)
            .map(|i| {
                (std::f64::consts::FRAC_PI_2 * i as f64 / (len - 1) as f64)
                    .sin()
                    .powi(2)
            })
            .collect();
        let window = fade_in.iter().map(|&v| (v * (1.0 - v)).sqrt()).collect();
        Self {
            input: fft.make_input_vec(),
            previous: fft.make_output_vec(),
            current: fft.make_output_vec(),
            forward_scratch: fft.make_scratch_vec(),
            inverse_output: inverse.make_output_vec(),
            coefficients: inverse.make_input_vec(),
            spectrum: inverse.make_input_vec(),
            delta: vec![0.0; len / 2 + 1],
            scratch: inverse.make_scratch_vec(),
            fft,
            inverse,
            fade_in,
            window,
            synthesis: vec![0.0; len],
            power: vec![1.0; len],
            position: (0..len).map(|i| i as f64 / len as f64 - 0.5).collect(),
        }
    }

    pub(super) fn crossfade(&mut self, previous: &[f32], current: &mut [f32]) {
        let len = previous.len().min(current.len());
        if len < 2 {
            return;
        }
        // Normal converter hops have fixed overlap lengths. Short/legacy model
        // outputs may change this length; rebuild only on that worker-side edge.
        if self.input.len() != len {
            *self = Self::new(len);
        }
        for ((dst, &sample), &window) in self.input.iter_mut().zip(previous).zip(&self.window) {
            *dst = sample * window as f32;
        }
        self.fft
            .process_with_scratch(
                &mut self.input,
                &mut self.previous,
                &mut self.forward_scratch,
            )
            .expect("phase vocoder FFT buffers match the plan");
        for ((dst, &sample), &window) in self.input.iter_mut().zip(current.iter()).zip(&self.window)
        {
            *dst = sample * window as f32;
        }
        self.fft
            .process_with_scratch(
                &mut self.input,
                &mut self.current,
                &mut self.forward_scratch,
            )
            .expect("phase vocoder FFT buffers match the plan");

        self.synthesis.fill(0.0);
        self.power.fill(1.0);
        let tau = std::f64::consts::TAU;
        for (k, (&a, &b)) in self.previous.iter().zip(&self.current).enumerate() {
            // Preserve the original f32 analysis and phase branch at +/-pi.
            // Changing analysis precision can flip that branch for opposite
            // spectra and audibly select a different interpolation direction.
            let magnitude = f64::from(a.norm()) + f64::from(b.norm());
            let phase = f64::from(a.arg());
            let delta = f64::from(b.arg()) - phase;
            let delta = delta - tau * (delta / tau + 0.5).floor();
            self.delta[k] = delta;
            self.coefficients[k] = Complex64::from_polar(magnitude, phase + delta * 0.5);
        }
        // Re(Ck * exp(i*2pi*k*t/N) * exp(i*delta_k*u)) is synthesized by
        // summing u^m * IRFFT(Ck * (i*delta_k)^m/m!). The real inverse FFT
        // supplies the interior-bin factor of two. Do not discard imaginary
        // DC/Nyquist coefficients between orders: they rotate into real parts
        // in the next order. Only the scratch spectrum has constrained ends.
        for order in 0..=PHASE_ORDER {
            self.spectrum.copy_from_slice(&self.coefficients);
            self.spectrum[0].im = 0.0;
            if len.is_multiple_of(2) {
                self.spectrum[len / 2].im = 0.0;
            }
            self.inverse
                .process_with_scratch(
                    &mut self.spectrum,
                    &mut self.inverse_output,
                    &mut self.scratch,
                )
                .expect("phase vocoder inverse FFT buffers match the plan");
            for (((sum, power), &position), &sample) in self
                .synthesis
                .iter_mut()
                .zip(&mut self.power)
                .zip(&self.position)
                .zip(&self.inverse_output)
            {
                *sum += sample * *power;
                *power *= position;
            }
            if order < PHASE_ORDER {
                for (coefficient, &delta) in self.coefficients.iter_mut().zip(&self.delta) {
                    *coefficient *= Complex64::new(0.0, delta / (order + 1) as f64);
                }
            }
        }
        for i in 0..len {
            let fade_in = self.fade_in[i];
            current[i] = (f64::from(previous[i]) * (1.0 - fade_in).powi(2)
                + f64::from(current[i]) * fade_in.powi(2)
                + self.synthesis[i] * self.window[i] / len as f64) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent, literal DFT + cosine-sum oracle for the upstream formula.
    // Covers DC/Nyquist, odd lengths and phase wrapping without relying on FFT
    // normalization or the phase expansion used by the production implementation.
    fn reference(a: &[f32], b: &[f32]) -> Vec<f32> {
        let n = a.len();
        let tau = std::f64::consts::TAU;
        let fade: Vec<_> = (0..n)
            .map(|i| {
                (std::f64::consts::FRAC_PI_2 * i as f64 / (n - 1) as f64)
                    .sin()
                    .powi(2)
            })
            .collect();
        let window: Vec<_> = fade.iter().map(|&f| (f * (1.0 - f)).sqrt()).collect();
        let dft = |audio: &[f32], k: usize| -> (f64, f64) {
            audio
                .iter()
                .enumerate()
                .fold((0.0, 0.0), |(re, im), (i, &v)| {
                    let (s, c) = (-tau * k as f64 * i as f64 / n as f64).sin_cos();
                    (
                        re + f64::from(v) * window[i] * c,
                        im + f64::from(v) * window[i] * s,
                    )
                })
        };
        let mut output: Vec<_> = (0..n)
            .map(|i| f64::from(a[i]) * (1.0 - fade[i]).powi(2) + f64::from(b[i]) * fade[i].powi(2))
            .collect();
        for k in 0..=n / 2 {
            let (ar, ai) = dft(a, k);
            let (br, bi) = dft(b, k);
            let phase = ai.atan2(ar);
            let delta = bi.atan2(br) - phase;
            let delta = delta - tau * (delta / tau + 0.5).floor();
            let magnitude = (ar.hypot(ai) + br.hypot(bi))
                * if k != 0 && !(n.is_multiple_of(2) && k == n / 2) {
                    2.0
                } else {
                    1.0
                };
            for i in 0..n {
                output[i] += magnitude
                    * ((tau * k as f64 + delta) * i as f64 / n as f64 + phase).cos()
                    * window[i]
                    / n as f64;
            }
        }
        output.into_iter().map(|v| v as f32).collect()
    }

    #[test]
    fn matches_upstream_formula_for_even_odd_and_changing_lengths() {
        let mut vocoder = PhaseVocoder::new(8);
        for n in [8, 9, 31, 32, 127, 128] {
            let a: Vec<_> = (0..n)
                .map(|i| 0.3 + (i as f32 * 0.7 + 2.9).cos() * 0.2)
                .collect();
            let b: Vec<_> = (0..n)
                .map(|i| -0.1 + (i as f32 * 0.71 - 2.8).cos() * 0.25)
                .collect();
            let expected = reference(&a, &b);
            let mut actual = b.clone();
            vocoder.crossfade(&a, &mut actual);
            for (i, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (actual - expected).abs() < 2e-5,
                    "n={n} i={i}: {actual} != {expected}"
                );
            }
            assert_eq!(actual[0], a[0]);
            assert_eq!(actual[n - 1], b[n - 1]);
        }
    }

    #[test]
    fn identical_audio_and_silence_are_preserved() {
        for n in [2, 3, 63, 64, 3600, 4080] {
            let a: Vec<_> = (0..n)
                .map(|i| 0.1 + 0.2 * (i as f32 * 0.023).sin())
                .collect();
            let mut vocoder = PhaseVocoder::new(n);
            let mut actual = a.clone();
            vocoder.crossfade(&a, &mut actual);
            assert!(
                actual.iter().zip(&a).all(|(x, y)| (x - y).abs() < 2e-5),
                "n={n}"
            );
            actual.fill(0.0);
            vocoder.crossfade(&vec![0.0; n], &mut actual);
            assert!(actual.iter().all(|&x| x == 0.0));
            vocoder.crossfade(&[], &mut []);
            vocoder.crossfade(&[1.0], &mut [0.0]);
        }
    }

    #[test]
    fn fft_synthesis_matches_direct_sum_on_realistic_and_opposite_spectra() {
        let mut seed = 0x8591_eb21u32;
        let mut noise = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32 - 0.5
        };
        for n in [63, 64, 479, 480, 2401, 3600, 9600] {
            // Include DC, Nyquist, broadband transients, and opposite spectra:
            // the latter exercises the discontinuous phase-wrap branch at pi.
            let a: Vec<_> = (0..n)
                .map(|i| noise() + 0.1 + if i % 2 == 0 { 0.2 } else { -0.2 })
                .collect();
            let b: Vec<_> = a
                .iter()
                .map(|&v| {
                    if n == 64 || n == 3600 {
                        -v
                    } else {
                        noise() - 0.1
                    }
                })
                .collect();
            let mut actual = b.clone();
            let mut vocoder = PhaseVocoder::new(n);
            vocoder.crossfade(&a, &mut actual);

            // Literal off-bin cosine sum using the same f32 analysis as the
            // previous implementation, independent of the phase expansion.
            let tau = std::f64::consts::TAU;
            let mut direct = vec![0.0; n];
            for (k, (&fa, &fb)) in vocoder.previous.iter().zip(&vocoder.current).enumerate() {
                let phase = f64::from(fa.arg());
                let delta = f64::from(fb.arg()) - phase;
                let delta = delta - tau * (delta / tau + 0.5).floor();
                let paired = k != 0 && !(n.is_multiple_of(2) && k == n / 2);
                let magnitude =
                    (f64::from(fa.norm()) + f64::from(fb.norm())) * if paired { 2.0 } else { 1.0 };
                for (i, sample) in direct.iter_mut().enumerate() {
                    *sample +=
                        magnitude * ((tau * k as f64 + delta) * i as f64 / n as f64 + phase).cos();
                }
            }
            let mut max_error = 0.0f64;
            for i in 0..n {
                let fade = vocoder.fade_in[i];
                let expected = f64::from(a[i]) * (1.0 - fade).powi(2)
                    + f64::from(b[i]) * fade.powi(2)
                    + direct[i] * vocoder.window[i] / n as f64;
                max_error = max_error.max((f64::from(actual[i]) - expected).abs());
            }
            assert!(max_error < 2e-6, "n={n}: {max_error}");
        }
    }
}
