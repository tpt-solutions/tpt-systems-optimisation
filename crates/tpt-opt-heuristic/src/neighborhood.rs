//! Neighborhood structures for simulated annealing and tabu search.
//!
//! A [`Neighborhood`] produces a neighbour of a current continuous point.
//! [`TabuNeighborhood`] additionally reports *which move* (coordinate index)
//! produced the neighbour, which tabu search needs to maintain its tabu list.

use crate::problem::Objective;
use crate::rng::{Rng, RngExt};

/// Produces neighbours of a continuous point.
pub trait Neighborhood {
    /// Return a neighbour of `current` sampled with `rng`.
    fn neighbor(&self, current: &[f64], rng: &mut dyn Rng) -> Vec<f64>;
}

/// Produces a neighbour together with the coordinate index that was moved.
///
/// Tabu search forbids recently-used moves; the returned `usize` identifies the
/// move so it can be entered into the tabu list.
pub trait TabuNeighborhood {
    /// Return `(candidate, moved_coordinate)` for a neighbour of `current`.
    fn neighbor_move(&self, current: &[f64], rng: &mut dyn Rng) -> (Vec<f64>, usize);
}

/// Gaussian (random-walk) neighbourhood: each coordinate is perturbed by a
/// zero-mean normal draw scaled by `scale * span_i` and clamped to the box.
#[derive(Debug, Clone)]
pub struct GaussianNeighborhood {
    /// Per-coordinate box bounds.
    pub bounds: Vec<(f64, f64)>,
    /// Scale fraction of the coordinate span used as the perturbation std-dev.
    pub scale: f64,
}

impl GaussianNeighborhood {
    /// Build a Gaussian neighbourhood from per-coordinate bounds and a scale.
    pub fn new(bounds: Vec<(f64, f64)>, scale: f64) -> Self {
        Self { bounds, scale }
    }

    /// Build a Gaussian neighbourhood from an [`Objective`]'s bounds.
    pub fn from_objective(objective: &dyn Objective, scale: f64) -> Self {
        let bounds: Vec<(f64, f64)> = (0..objective.dim()).map(|i| objective.bound(i)).collect();
        Self::new(bounds, scale)
    }

    fn perturb(&self, current: &[f64], rng: &mut dyn Rng, out: &mut [f64]) {
        let n = current.len().min(out.len()).min(self.bounds.len());
        for i in 0..n {
            let (lo, hi) = self.bounds[i];
            let span = if lo.is_finite() && hi.is_finite() { (hi - lo).abs() } else { 1.0 };
            let step = self.scale * span;
            let mut v = current[i] + rng.normal() * step;
            if lo.is_finite() {
                v = v.max(lo);
            }
            if hi.is_finite() {
                v = v.min(hi);
            }
            out[i] = v;
        }
    }
}

impl Neighborhood for GaussianNeighborhood {
    fn neighbor(&self, current: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        let mut out = current.to_vec();
        self.perturb(current, rng, &mut out);
        out
    }
}

/// Uniform random-restart neighbourhood: each coordinate is resampled uniformly
/// within its box. Useful as a strong-diversification operator.
#[derive(Debug, Clone)]
pub struct UniformNeighborhood {
    /// Per-coordinate box bounds.
    pub bounds: Vec<(f64, f64)>,
}

impl UniformNeighborhood {
    /// Build from per-coordinate bounds.
    pub fn new(bounds: Vec<(f64, f64)>) -> Self {
        Self { bounds }
    }

    /// Build from an [`Objective`]'s bounds.
    pub fn from_objective(objective: &dyn Objective) -> Self {
        let bounds: Vec<(f64, f64)> = (0..objective.dim()).map(|i| objective.bound(i)).collect();
        Self::new(bounds)
    }
}

impl Neighborhood for UniformNeighborhood {
    fn neighbor(&self, _current: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        self.bounds
            .iter()
            .map(|&(lo, hi)| {
                if lo.is_finite() && hi.is_finite() {
                    rng.range(lo, hi)
                } else {
                    let a = if lo.is_finite() { lo } else { -10.0 };
                    let b = if hi.is_finite() { hi } else { 10.0 };
                    rng.range(a, b)
                }
            })
            .collect()
    }
}

/// Coordinate-step neighbourhood for tabu search: a single random coordinate is
/// perturbed (as in [`GaussianNeighborhood`]); the moved coordinate index is
/// returned so it can be tabu-flagged.
#[derive(Debug, Clone)]
pub struct CoordinateNeighborhood {
    /// Per-coordinate box bounds.
    pub bounds: Vec<(f64, f64)>,
    /// Scale fraction of the coordinate span used as the perturbation std-dev.
    pub scale: f64,
}

impl CoordinateNeighborhood {
    /// Build from per-coordinate bounds and a scale.
    pub fn new(bounds: Vec<(f64, f64)>, scale: f64) -> Self {
        Self { bounds, scale }
    }

    /// Build from an [`Objective`]'s bounds.
    pub fn from_objective(objective: &dyn Objective, scale: f64) -> Self {
        let bounds: Vec<(f64, f64)> = (0..objective.dim()).map(|i| objective.bound(i)).collect();
        Self::new(bounds, scale)
    }
}

impl TabuNeighborhood for CoordinateNeighborhood {
    fn neighbor_move(&self, current: &[f64], rng: &mut dyn Rng) -> (Vec<f64>, usize) {
        let n = current.len().min(self.bounds.len());
        let i = if n == 0 { 0 } else { rng.index(n) };
        let mut out = current.to_vec();
        if i < self.bounds.len() {
            let (lo, hi) = self.bounds[i];
            let span = if lo.is_finite() && hi.is_finite() { (hi - lo).abs() } else { 1.0 };
            let step = self.scale * span;
            let mut v = current[i] + rng.normal() * step;
            if lo.is_finite() {
                v = v.max(lo);
            }
            if hi.is_finite() {
                v = v.min(hi);
            }
            out[i] = v;
        }
        (out, i)
    }
}

impl Neighborhood for CoordinateNeighborhood {
    fn neighbor(&self, current: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        self.neighbor_move(current, rng).0
    }
}

/// Neighbourhood defined by a user closure `f(current, rng) -> neighbor`.
///
/// This is the primary extension hook for *custom neighborhoods* (spec §3).
#[derive(Clone)]
pub struct NeighborhoodFn<F>(pub F);

impl<F> Neighborhood for NeighborhoodFn<F>
where
    F: Fn(&[f64], &mut dyn Rng) -> Vec<f64>,
{
    fn neighbor(&self, current: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        (self.0)(current, rng)
    }
}
