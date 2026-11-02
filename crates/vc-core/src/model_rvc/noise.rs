//! Latent-noise generation for RVC exports that expose the VITS
//! reparameterization noise `z` (`rnd`) as a generator input instead of sampling
//! it inside the graph. Those models expect a fresh `N(0, 1)` tensor of shape
//! `[1, inter_channels, frames]` each call, exactly like the reference
//! `torch.randn(1, 192, T)`.
//!
//! Dependency-free on purpose: pulling `rand` in would add to the cargo-deny
//! allow-list for a few lines of well-understood arithmetic. SplitMix64 supplies
//! the uniform stream and Box-Muller maps it to a standard normal. The generator
//! is seeded deterministically so a stream is reproducible across runs, while
//! still advancing per chunk so successive chunks draw independent noise (what
//! the reference does). It is worker-thread state — never touched from the audio
//! callback — so the per-chunk arithmetic here is fine.

/// SplitMix64-backed standard-normal generator. Caches the second Box-Muller
/// value so two normals cost one transcendental pair.
pub(super) struct GaussianNoise {
    state: u64,
    spare: Option<f32>,
}

impl GaussianNoise {
    /// Seeded generator. A fixed seed keeps output reproducible across runs; the
    /// stream still advances per draw, so chunks see independent noise.
    pub(super) fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    /// Next SplitMix64 output (uniform `u64`).
    fn next_u64(&mut self) -> u64 {
        // SplitMix64 (Steele et al.): increment by the golden-ratio constant,
        // then avalanche. Good statistical quality for a tiny, fast PRNG.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in the half-open interval `[0, 1)` (53-bit mantissa).
    pub(super) fn next_unit(&mut self) -> f64 {
        // Top 53 bits scaled by 2^-53 lands in [0, 1); the low bound is excluded
        // below so the log in Box-Muller never sees zero.
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Next `N(0, 1)` sample.
    fn next_normal(&mut self) -> f32 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        // Box-Muller: u1 in (0, 1] avoids ln(0); u2 spans the full turn.
        let u1 = 1.0 - self.next_unit();
        let u2 = self.next_unit();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        self.spare = Some((radius * angle.sin()) as f32);
        (radius * angle.cos()) as f32
    }

    /// Fill `out` with independent `N(0, 1)` samples.
    pub(super) fn fill(&mut self, out: &mut [f32]) {
        for slot in out.iter_mut() {
            *slot = self.next_normal();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_is_deterministic_for_a_seed() {
        let mut a = GaussianNoise::new(0x1234_5678);
        let mut b = GaussianNoise::new(0x1234_5678);
        let mut buf_a = [0.0f32; 256];
        let mut buf_b = [0.0f32; 256];
        a.fill(&mut buf_a);
        b.fill(&mut buf_b);
        assert_eq!(buf_a, buf_b);
    }

    #[test]
    fn successive_fills_differ() {
        let mut gen = GaussianNoise::new(42);
        let mut first = [0.0f32; 256];
        let mut second = [0.0f32; 256];
        gen.fill(&mut first);
        gen.fill(&mut second);
        assert_ne!(first, second, "chunks must draw independent noise");
    }

    #[test]
    fn distribution_is_roughly_standard_normal() {
        let mut gen = GaussianNoise::new(7);
        let n = 100_000;
        let mut buf = vec![0.0f32; n];
        gen.fill(&mut buf);
        let mean = buf.iter().copied().sum::<f32>() / n as f32;
        let variance = buf.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.05, "mean {mean} not near 0");
        assert!(
            (variance - 1.0).abs() < 0.05,
            "variance {variance} not near 1"
        );
        // All finite: the (0, 1] guard must keep ln() away from -inf.
        assert!(buf.iter().all(|x| x.is_finite()));
    }
}
