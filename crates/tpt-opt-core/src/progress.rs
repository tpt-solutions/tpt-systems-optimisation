//! Progress-reporting contract shared across solvers.
//!
//! Solvers that support progress reporting accept a callback through their own
//! builder API (for example `tpt_opt_milp::MilpSolver::with_progress_callback`)
//! and invoke it at coarse checkpoints with a [`ProgressEvent`]. The callback
//! returns a [`ProgressAction`]: [`ProgressAction::Continue`] keeps the search
//! running, while [`ProgressAction::Abort`] asks the solver to stop as soon as
//! possible — the solve then terminates with [`crate::solver::SolverStatus::
//! TimeLimit`] and reports whatever incumbent it has found so far.
//!
//! Events are delivered from the solver's hot path, so callbacks should be
//! cheap (update a counter, refresh a progress bar, flip a flag). Long-running
//! work belongs outside the callback.

use core::time::Duration;

/// A snapshot of solver progress, delivered at coarse checkpoints.
///
/// All fields are advisory: solvers may omit values they cannot provide
/// cheaply (e.g. a depth-first search has no meaningful global bound).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgressEvent {
    /// Nodes / iterations explored so far (monotonically non-decreasing).
    pub iterations: usize,
    /// Best known feasible objective value so far, in the model's own sense
    /// (`None` until a first incumbent exists).
    pub incumbent: Option<f64>,
    /// Best proven (or pseudo-cost-estimated) dual bound on the objective, in
    /// the model's own sense (`None` when the solver cannot provide one).
    pub bound: Option<f64>,
    /// Wall-clock time elapsed since the solve started.
    pub elapsed: Duration,
}

impl ProgressEvent {
    /// Absolute optimality gap implied by this event:
    /// `|incumbent - bound|`, or `None` when either side is missing.
    pub fn absolute_gap(&self) -> Option<f64> {
        match (self.incumbent, self.bound) {
            (Some(i), Some(b)) => Some((i - b).abs()),
            _ => None,
        }
    }
}

/// What the solver should do after delivering a [`ProgressEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressAction {
    /// Keep solving.
    Continue,
    /// Stop the search as soon as possible. The solve reports
    /// [`crate::solver::SolverStatus::TimeLimit`] together with any incumbent
    /// found so far (or plain `TimeLimit`/no-solution semantics if none).
    Abort,
}

/// Callback signature: receive a [`ProgressEvent`], return the desired
/// [`ProgressAction`].
///
/// The callback may be invoked from multiple worker threads when a solver runs
/// its parallel mode, hence the `Send` bound; implementations must therefore
/// be safe to call concurrently (the owning solver serialises delivery through
/// an internal lock).
pub type ProgressCallback = dyn FnMut(&ProgressEvent) -> ProgressAction + Send;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gap_requires_both_sides() {
        let ev = ProgressEvent {
            iterations: 3,
            incumbent: Some(10.0),
            bound: Some(8.0),
            elapsed: Duration::from_secs(1),
        };
        assert_eq!(ev.absolute_gap(), Some(2.0));

        let ev = ProgressEvent { incumbent: None, ..ev };
        assert_eq!(ev.absolute_gap(), None);
        let ev = ProgressEvent { bound: None, ..ev };
        assert_eq!(ev.absolute_gap(), None);
    }
}
