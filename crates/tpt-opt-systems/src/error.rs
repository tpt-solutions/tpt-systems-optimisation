//! Unified error type wrapping solver-specific failures with algorithm context.
//!
//! The constituent solver crates report failures in their own shapes: the
//! [`Solver`](tpt_opt_core::solver::Solver) implementations return
//! [`OptError`], while some algorithm entry points
//! return terminal statuses without an error payload. [`OptimizationError`]
//! normalises both into one type that always records *which* algorithm
//! produced the failure, so callers of the umbrella API can log, match, and
//! recover without knowing the underlying crate.

use core::fmt;

use tpt_opt_core::OptError;

/// A solve failure tagged with the algorithm that produced it.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationError {
    /// A wrapped [`OptError`] raised while solving with `algorithm`.
    Solve {
        /// Name of the algorithm/solver that failed (e.g. `"branch-and-bound"`).
        algorithm: &'static str,
        /// The underlying core error.
        source: OptError,
    },
    /// An algorithm terminated with a non-solution status and no error
    /// payload (e.g. a decomposition hit its iteration limit).
    NoSolution {
        /// Name of the algorithm/solver that failed.
        algorithm: &'static str,
        /// Human-readable description of the terminal status.
        status: String,
    },
}

impl OptimizationError {
    /// Tag a core error with the algorithm that raised it.
    pub fn solve(algorithm: &'static str, source: OptError) -> Self {
        OptimizationError::Solve { algorithm, source }
    }

    /// Record a non-solution terminal status for `algorithm`.
    pub fn no_solution(algorithm: &'static str, status: impl Into<String>) -> Self {
        OptimizationError::NoSolution { algorithm, status: status.into() }
    }

    /// The algorithm that produced this failure.
    pub fn algorithm(&self) -> &'static str {
        match self {
            OptimizationError::Solve { algorithm, .. } => algorithm,
            OptimizationError::NoSolution { algorithm, .. } => algorithm,
        }
    }

    /// The wrapped core error, if this failure carries one.
    pub fn into_core(self) -> Option<OptError> {
        match self {
            OptimizationError::Solve { source, .. } => Some(source),
            OptimizationError::NoSolution { .. } => None,
        }
    }

    /// `true` if the underlying failure is infeasibility.
    pub fn is_infeasible(&self) -> bool {
        match self {
            OptimizationError::Solve { source, .. } => source.is_infeasible(),
            OptimizationError::NoSolution { status, .. } => {
                status.eq_ignore_ascii_case("infeasible")
            }
        }
    }

    /// `true` if the underlying failure is unboundedness.
    pub fn is_unbounded(&self) -> bool {
        match self {
            OptimizationError::Solve { source, .. } => source.is_unbounded(),
            OptimizationError::NoSolution { status, .. } => {
                status.eq_ignore_ascii_case("unbounded")
            }
        }
    }
}

impl fmt::Display for OptimizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizationError::Solve { algorithm, source } => {
                write!(f, "{algorithm}: {source}")
            }
            OptimizationError::NoSolution { algorithm, status } => {
                write!(f, "{algorithm}: no solution ({status})")
            }
        }
    }
}

impl std::error::Error for OptimizationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OptimizationError::Solve { source, .. } => Some(source),
            OptimizationError::NoSolution { .. } => None,
        }
    }
}

impl From<OptError> for OptimizationError {
    fn from(source: OptError) -> Self {
        OptimizationError::Solve { algorithm: "unknown", source }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_error_carries_algorithm_and_source() {
        let err = OptimizationError::solve(
            "branch-and-bound",
            OptError::invalid_model("empty objective"),
        );
        assert_eq!(err.algorithm(), "branch-and-bound");
        assert!(err.to_string().contains("branch-and-bound"));
        assert!(err.to_string().contains("empty objective"));
        assert!(err.into_core().is_some());
    }

    #[test]
    fn no_solution_error_reports_status() {
        let err = OptimizationError::no_solution("benders", "iteration limit");
        assert_eq!(err.algorithm(), "benders");
        assert!(!err.is_infeasible());
        assert!(err.into_core().is_none());
    }

    #[test]
    fn infeasibility_and_unboundedness_predicates() {
        let inf = OptimizationError::solve(
            "milp",
            OptError::infeasible(tpt_opt_core::InfeasibilityReport::new("no feasible point")),
        );
        assert!(inf.is_infeasible());
        assert!(!inf.is_unbounded());
        let tagged = OptimizationError::no_solution("network", "Infeasible");
        assert!(tagged.is_infeasible());
    }
}
