//! Result type returned by every heuristic, bridging into `tpt-opt-core`.

use tpt_opt_core::{Solution, SolverStatus};

use crate::history::ConvergenceHistory;

/// Outcome of a heuristic run.
///
/// Carries the best solution found, its objective value, the terminal status,
/// the deterministic seed used, and the full [`ConvergenceHistory`]. It can be
/// converted into a canonical [`tpt_opt_core::Solution`] via [`HeuristicResult::solution`].
#[derive(Debug, Clone, PartialEq)]
pub struct HeuristicResult {
    /// Best decision-vector found.
    pub best_x: Vec<f64>,
    /// Objective value at `best_x`.
    pub best_value: f64,
    /// Terminal status (see `tpt-opt-core`).
    pub status: SolverStatus,
    /// Number of iterations / generations executed.
    pub iterations: usize,
    /// Deterministic seed used for the run.
    pub seed: u64,
    /// Per-iteration convergence record.
    pub history: ConvergenceHistory,
}

impl HeuristicResult {
    /// Best decision-vector found.
    pub fn best(&self) -> &[f64] {
        &self.best_x
    }

    /// Objective value at the best decision-vector.
    pub fn best_value(&self) -> f64 {
        self.best_value
    }

    /// Convergence history for the run.
    pub fn history(&self) -> &ConvergenceHistory {
        &self.history
    }

    /// Convert into a canonical [`tpt_opt_core::Solution`].
    pub fn solution(&self) -> Solution {
        Solution::new(self.best_x.clone(), self.best_value, self.status)
            .with_iterations(self.iterations)
    }
}
