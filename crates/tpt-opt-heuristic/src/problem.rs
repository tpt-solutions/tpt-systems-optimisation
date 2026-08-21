//! Objective abstraction shared by the continuous heuristics (SA / Tabu / PSO).
//!
//! A heuristic optimizes a user objective over a `dim`-dimensional box. The
//! [`Objective`] trait exposes the dimension, per-coordinate bounds, the raw
//! objective value, and the optimisation [`Sense`]. Two ready-made
//! implementations are provided: [`ObjectiveFn`] (from a closure) and
//! [`ModelObjective`] (adapts a `tpt-opt-core` [`Model`](tpt_opt_core::Model)).

use tpt_opt_core::{Model, Sense, VarType};

use crate::rng::Rng;

/// A continuous optimisation objective minimised or maximised over a box.
pub trait Objective {
    /// Dimension of the search space.
    fn dim(&self) -> usize;

    /// Inclusive `(lower, upper)` box bound for coordinate `i`.
    ///
    /// Bounds may be infinite; callers clamp/scale gracefully in that case.
    fn bound(&self, i: usize) -> (f64, f64);

    /// Raw objective value at `x`.
    fn evaluate(&self, x: &[f64]) -> f64;

    /// Optimisation sense (default: minimise).
    fn sense(&self) -> Sense {
        Sense::Minimize
    }
}

/// Objective defined by a closure `f(x) -> f64`.
///
/// # Example
///
/// ```
/// use tpt_opt_core::Sense;
/// use tpt_opt_heuristic::{Objective, ObjectiveFn};
/// let obj = ObjectiveFn::new(2, [(-1.0, 1.0), (-1.0, 1.0)], Sense::Minimize, |x| x[0] + x[1]);
/// assert_eq!(obj.dim(), 2);
/// assert_eq!(obj.evaluate(&[0.5, -0.5]), 0.0);
/// ```
#[derive(Clone)]
pub struct ObjectiveFn<F> {
    /// Dimension.
    pub dim: usize,
    /// Per-coordinate box bounds.
    pub bounds: Vec<(f64, f64)>,
    /// Optimisation sense.
    pub sense: Sense,
    /// Objective closure.
    pub f: F,
}

impl<F> ObjectiveFn<F>
where
    F: Fn(&[f64]) -> f64,
{
    /// Build an objective from its parts.
    pub fn new(dim: usize, bounds: impl Into<Vec<(f64, f64)>>, sense: Sense, f: F) -> Self {
        Self { dim, bounds: bounds.into(), sense, f }
    }

    /// Convenience builder for a minimisation objective.
    pub fn minimize(dim: usize, f: F, bounds: impl Into<Vec<(f64, f64)>>) -> Self {
        Self::new(dim, bounds, Sense::Minimize, f)
    }

    /// Convenience builder for a maximisation objective.
    pub fn maximize(dim: usize, f: F, bounds: impl Into<Vec<(f64, f64)>>) -> Self {
        Self::new(dim, bounds, Sense::Maximize, f)
    }
}

impl<F> Objective for ObjectiveFn<F>
where
    F: Fn(&[f64]) -> f64,
{
    fn dim(&self) -> usize {
        self.dim
    }

    fn bound(&self, i: usize) -> (f64, f64) {
        self.bounds[i]
    }

    fn evaluate(&self, x: &[f64]) -> f64 {
        (self.f)(x)
    }

    fn sense(&self) -> Sense {
        self.sense
    }
}

/// Adapter turning a `tpt-opt-core` [`Model`](tpt_opt_core::Model) into a
/// continuous [`Objective`] for the heuristics.
///
/// The model's linear objective (with its [`Sense`]) is minimised / maximised;
/// variable bounds become the box, and constraint violations are penalised into
/// the objective so the search is guided toward feasibility. This is an
/// *unconstrained relaxation*: integrality and custom constraints are not
/// enforced exactly, but the structure is enough to drive a heuristic toward a
/// good continuous point.
pub struct ModelObjective {
    model: Model,
    penalty: f64,
}

impl ModelObjective {
    /// Build an adapter for `model`, using the given constraint penalty weight.
    pub fn new(model: Model) -> Self {
        Self { model, penalty: 1e3 }
    }

    /// Override the constraint-violation penalty weight.
    pub fn with_penalty(mut self, penalty: f64) -> Self {
        self.penalty = penalty;
        self
    }
}

impl Objective for ModelObjective {
    fn dim(&self) -> usize {
        self.model.num_vars
    }

    fn bound(&self, i: usize) -> (f64, f64) {
        let v = &self.model.variables[i];
        (v.bound.bound.lower, v.bound.bound.upper)
    }

    fn evaluate(&self, x: &[f64]) -> f64 {
        let mut value = self.model.objective.eval(x);
        for c in &self.model.constraints {
            let slack = c.slack(x);
            if slack < 0.0 {
                value += self.penalty * (-slack);
            }
        }
        value
    }

    fn sense(&self) -> Sense {
        self.model.objective.sense
    }
}

/// Sample a uniformly random point inside the objective's box.
///
/// Unbounded coordinates are sampled from `[-10, 10]`; one-sided-bounded
/// coordinates are sampled from the finite bound toward the finite default.
pub(crate) fn random_point(objective: &dyn Objective, rng: &mut dyn Rng) -> Vec<f64> {
    (0..objective.dim())
        .map(|i| {
            let (lo, hi) = objective.bound(i);
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

/// Classify a variable's kind (used by adapters/diagnostics).
#[allow(dead_code)]
pub(crate) fn var_kind(model: &Model, i: usize) -> VarType {
    model.variables[i].kind
}
