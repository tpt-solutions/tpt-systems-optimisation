//! Seedable, deterministic RNG used by every heuristic.
//!
//! Re-exported from `tpt-math-prob` so downstream users only depend on this
//! crate. The [`Rng`] trait is object-safe, so heuristics accept
//! `&mut dyn Rng` and remain fully deterministic for a fixed seed (spec §4).

pub use tpt_math_prob::sampler::{Rng, SplitMix64};

/// Build a deterministic generator from a `u64` seed using the published
/// `tpt-math-prob` `SplitMix64` RNG (the old dev shim's `Xoshiro256` is not
/// published on crates.io).
pub fn rng_from_seed(seed: u64) -> SplitMix64 {
    SplitMix64::seed_from_u64(seed)
}

/// Convenience helpers that mirror the historical `tpt-math-prob` `Rng` API
/// (`range`, `index`, `normal`) on top of the published minimal `Rng` trait.
///
/// Implemented for every `Rng` (including `dyn Rng` and `SplitMix64`) so the
/// call sites keep reading `rng.range(lo, hi)` / `rng.index(n)` / `rng.normal()`.
pub trait RngExt {
    /// Uniform `f64` in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64;
    /// Uniform `usize` in `0..n` (`0` when `n == 0`).
    fn index(&mut self, n: usize) -> usize;
    /// Standard normal `f64` (`N(0, 1)`, Box–Muller).
    fn normal(&mut self) -> f64;
}

impl<R: Rng + ?Sized> RngExt for R {
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    fn index(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() as usize) % n
        }
    }

    fn normal(&mut self) -> f64 {
        // Box–Muller transform, deterministic for a fixed RNG state.
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
}
