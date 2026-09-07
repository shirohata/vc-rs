use anyhow::{ensure, Context, Result};

/// A lifetime-fixed sample contract. Validate before touching signal state,
/// including for equal-rate bypass; an equal-duration *different* hop is unsafe
/// once a FIFO was primed for a particular phase cycle.
#[derive(Clone, Copy)]
pub(super) struct FixedHop {
    pub input: usize,
    pub output: usize,
}

impl FixedHop {
    pub fn new(from: usize, to: usize, input: usize, output: usize) -> Result<Self> {
        ensure!(
            from > 0 && to > 0 && input > 0 && output > 0,
            "fixed resampler rates and hops must be positive"
        );
        ensure!(
            input as u128 * to as u128 == output as u128 * from as u128,
            "fixed resampler hops must have exactly equal durations"
        );
        Ok(Self { input, output })
    }

    pub fn validate(&self, input: usize, output: usize) -> Result<()> {
        ensure!(input == self.input && output == self.output,
            "fixed resampler expects {}/{} input/output samples, received {input}/{output}; rebuild on hop changes",
            self.input, self.output);
        Ok(())
    }

    pub fn phase_period(&self, batch: usize, fft_input: usize) -> Result<usize> {
        ensure!(
            batch > 0 && fft_input > 0,
            "resampler block sizes must be positive"
        );
        let cycle = (batch / gcd(batch, fft_input))
            .checked_mul(fft_input)
            .context("resampler phase cycle overflow")?;
        Ok(cycle / gcd(cycle, self.input))
    }

    /// Exact smallest FIFO preload, preserving the existing FFT and its trim.
    /// After n hops, B-sized input batches have supplied floor(n*H/B)*B samples
    /// to F-sized FFTs, producing E(n)=floor(floor(n*H/B)*B/F)*O raw samples.
    /// The deficit n*K-E(n) repeats after lcm(B,F)/gcd(H,lcm(B,F)) hops.
    /// Add the once-trimmed filter delay D to the maximum over that FULL cycle.
    /// Startup max(E-D,0) cannot make this bound worse; every phase recurs after
    /// startup, proving minimality. Do not replace this with a first-hop probe:
    /// 44.1 kHz / 30 ms reaches later worst phases across 160 calls.
    pub fn delay(
        &self,
        batch: usize,
        fft_input: usize,
        fft_output: usize,
        filter_delay: usize,
    ) -> Result<usize> {
        ensure!(
            self.input as u128 * fft_output as u128 == self.output as u128 * fft_input as u128,
            "FFT ratio does not match fixed hop"
        );
        let period = self.phase_period(batch, fft_input)?;
        let mut deficit = 0;
        for n in 0..period {
            let input = n as u128 * self.input as u128;
            let fft_consumed =
                input / batch as u128 * batch as u128 / fft_input as u128 * fft_input as u128;
            // Work with the bounded residual, avoiding products of accumulated
            // output counts on long cycles. The rational duration is integral.
            let pending = (input - fft_consumed) * fft_output as u128;
            ensure!(
                pending.is_multiple_of(fft_input as u128),
                "nonintegral fixed-hop deficit"
            );
            deficit = deficit.max(usize::try_from(pending / fft_input as u128)?);
        }
        filter_delay
            .checked_add(deficit)
            .context("resampler delay overflow")
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
