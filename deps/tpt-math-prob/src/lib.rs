//! Local dev shim mirroring `tpt-math-prob`: a small, fast, **seedable**
//! deterministic RNG. Seedability is required by the optimisation crates'
//! reproducibility design principle (spec §4).


/// A deterministic random number generator. Implementors must produce the same
/// stream for the same seed across platforms.
pub trait Rng {
    /// Advance the state and return the next `u64`.
    fn next_u64(&mut self) -> u64;

    /// Return the next uniformly distributed `f64` in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Return a uniform `f64` in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Return a uniform `usize` in `[0, n)`.
    fn index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Standard normal sample via Box–Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
}

/// SplitMix64 — used to seed larger generators from a single `u64`.
pub fn splitmix64(mut state: u64) -> (u64, u64) {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (state, z)
}

/// `xoshiro256++` — fast, high-quality, deterministic PRNG.
#[derive(Debug, Clone)]
pub struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    pub fn new(seed: u64) -> Self {
        let (s0, _) = splitmix64(seed);
        let (s1, _) = splitmix64(s0);
        let (s2, _) = splitmix64(s1);
        let (s3, _) = splitmix64(s2);
        Self {
            s: [s0, s1, s2, s3],
        }
    }
}

impl Rng for Xoshiro256 {
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[0]
            .wrapping_add(self.s[3]))
        .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }
}

/// Convenience: build a fresh generator from any `u64` seed.
pub fn rng_from_seed(seed: u64) -> Xoshiro256 {
    Xoshiro256::new(seed)
}
