//! Seedable, deterministic RNG used by every heuristic.
//!
//! Re-exported from `tpt-math-prob` so downstream users only depend on this
//! crate. The [`Rng`] trait is object-safe, so heuristics accept
//! `&mut dyn Rng` and remain fully deterministic for a fixed seed (spec §4).

pub use tpt_math_prob::{rng_from_seed, Rng, Xoshiro256};
