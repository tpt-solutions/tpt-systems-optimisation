//! Tolerance defaults (spec §4) and the configurable tolerance bundle.
//!
//! All tolerances are `Copy` scalar bundles with no heap state, so they are
//! available regardless of the `alloc` feature. Every solver crate must thread
//! [`Tolerances`] through its numeric comparisons so users can retune them in
//! one place.

/// Numeric tolerances used across the solver crates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tolerances {
    /// Integrality tolerance: a value within this of an integer is integral.
    pub integrality: f64,
    /// Feasibility tolerance: a constraint within this of its bound is satisfied.
    pub feasibility: f64,
    /// Optimality gap tolerance (absolute) for termination.
    pub optimality_gap: f64,
    /// Pivoting / degeneracy threshold for LP/NLP steps.
    pub pivoting: f64,
}

impl Tolerances {
    /// Spec §4 defaults: integrality 1e-6, feasibility 1e-6, optimality gap
    /// 1e-4, pivoting 1e-9.
    pub fn spec_default() -> Self {
        Self { integrality: 1e-6, feasibility: 1e-6, optimality_gap: 1e-4, pivoting: 1e-9 }
    }

    /// Override the integrality tolerance.
    pub fn with_integrality(mut self, eps: f64) -> Self {
        self.integrality = eps;
        self
    }

    /// Override the feasibility tolerance.
    pub fn with_feasibility(mut self, eps: f64) -> Self {
        self.feasibility = eps;
        self
    }

    /// Override the optimality gap tolerance.
    pub fn with_optimality_gap(mut self, gap: f64) -> Self {
        self.optimality_gap = gap;
        self
    }

    /// Override the pivoting tolerance.
    pub fn with_pivoting(mut self, tol: f64) -> Self {
        self.pivoting = tol;
        self
    }
}

impl Default for Tolerances {
    fn default() -> Self {
        Self::spec_default()
    }
}
