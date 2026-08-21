//! Convergence-history tracking shared by every heuristic.
//!
//! Each algorithm records the best-so-far (incumbent) objective value and the
//! current / per-iteration objective value, enabling convergence analysis and
//! plotting without re-running the search.

/// Per-iteration convergence record.
///
/// `incumbent[i]` is the best objective found up to and including iteration
/// `i`; `current[i]` is the objective of the search state at iteration `i`
/// (e.g. the current temperature state for SA, or the generation best for GA).
#[derive(Debug, Clone, PartialEq)]
pub struct ConvergenceHistory {
    /// Iteration indices (0-based).
    pub iterations: Vec<usize>,
    /// Best-so-far objective value at each iteration.
    pub incumbent: Vec<f64>,
    /// Current-state objective value at each iteration.
    pub current: Vec<f64>,
}

impl ConvergenceHistory {
    /// Create an empty history.
    pub fn new() -> Self {
        Self {
            iterations: Vec::new(),
            incumbent: Vec::new(),
            current: Vec::new(),
        }
    }

    /// Record one iteration.
    pub fn push(&mut self, iteration: usize, incumbent: f64, current: f64) {
        self.iterations.push(iteration);
        self.incumbent.push(incumbent);
        self.current.push(current);
    }

    /// Number of recorded iterations.
    pub fn len(&self) -> usize {
        self.iterations.len()
    }

    /// `true` if no iteration has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.iterations.is_empty()
    }

    /// Iteration indices.
    pub fn iterations(&self) -> &[usize] {
        &self.iterations
    }

    /// Best-so-far objective values, in iteration order.
    pub fn incumbent_values(&self) -> &[f64] {
        &self.incumbent
    }

    /// Current-state objective values, in iteration order.
    pub fn current_values(&self) -> &[f64] {
        &self.current
    }

    /// The final best-so-far objective value, or `None` if empty.
    pub fn best(&self) -> Option<f64> {
        self.incumbent.last().copied()
    }
}

impl Default for ConvergenceHistory {
    fn default() -> Self {
        Self::new()
    }
}
