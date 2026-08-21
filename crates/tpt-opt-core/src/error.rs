//! Error types with structured infeasibility diagnostics.
//!
//! Every error variant that needs to carry a message uses an owned
//! [`alloc::string::String`], so the whole module is gated behind the `alloc`
//! feature. Under `std` the `OptError` type additionally implements
//! [`std::error::Error`].

use alloc::string::String;
use alloc::vec::Vec;

/// Diagnostic report describing *why* a model was found infeasible.
#[derive(Debug, Clone, PartialEq)]
pub struct InfeasibilityReport {
    /// Indices of constraints that cannot be simultaneously satisfied.
    pub violated_constraints: Vec<usize>,
    /// Indices of variables whose bounds conflict (e.g. `lower > upper`).
    pub conflicting_bounds: Vec<usize>,
    /// Human-readable explanation.
    pub message: String,
}

impl InfeasibilityReport {
    /// Create a report with only a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            violated_constraints: Vec::new(),
            conflicting_bounds: Vec::new(),
            message: message.into(),
        }
    }

    /// Attach a constraint index that could not be satisfied.
    pub fn with_violated(mut self, index: usize) -> Self {
        self.violated_constraints.push(index);
        self
    }

    /// Attach a variable index whose bounds are internally inconsistent.
    pub fn with_conflict(mut self, index: usize) -> Self {
        self.conflicting_bounds.push(index);
        self
    }
}

/// Primary error type returned across the optimisation workspace.
#[derive(Debug, Clone, PartialEq)]
pub enum OptError {
    /// The model has no feasible solution.
    Infeasible(InfeasibilityReport),
    /// The feasible region is unbounded in the objective direction.
    Unbounded,
    /// Numerical failure (cycling, stalling, ill-conditioning, NaN).
    NumericalIssue(String),
    /// The solver stopped because the time limit was reached.
    TimeLimit,
    /// The model was structurally invalid (bad indices, inconsistent bounds).
    InvalidModel(String),
    /// An internal solver error (e.g. unsupported feature).
    Internal(String),
}

impl OptError {
    /// Construct an [`OptError::InvalidModel`].
    pub fn invalid_model(msg: impl Into<String>) -> Self {
        OptError::InvalidModel(msg.into())
    }

    /// Construct an [`OptError::NumericalIssue`].
    pub fn numerical(msg: impl Into<String>) -> Self {
        OptError::NumericalIssue(msg.into())
    }

    /// Construct an [`OptError::Internal`].
    pub fn internal(msg: impl Into<String>) -> Self {
        OptError::Internal(msg.into())
    }

    /// Construct an [`OptError::Infeasible`] from a report.
    pub fn infeasible(report: InfeasibilityReport) -> Self {
        OptError::Infeasible(report)
    }

    /// `true` if this error represents infeasibility.
    pub fn is_infeasible(&self) -> bool {
        matches!(self, OptError::Infeasible(_))
    }

    /// `true` if this error represents unboundedness.
    pub fn is_unbounded(&self) -> bool {
        matches!(self, OptError::Unbounded)
    }
}

impl core::fmt::Display for OptError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OptError::Infeasible(r) => {
                write!(f, "infeasible: {}", r.message)?;
                if !r.violated_constraints.is_empty() {
                    write!(f, " (constraints: {:?})", r.violated_constraints)?;
                }
                if !r.conflicting_bounds.is_empty() {
                    write!(f, " (bounds: {:?})", r.conflicting_bounds)?;
                }
                Ok(())
            }
            OptError::Unbounded => write!(f, "model is unbounded"),
            OptError::NumericalIssue(m) => write!(f, "numerical issue: {m}"),
            OptError::TimeLimit => write!(f, "time limit reached"),
            OptError::InvalidModel(m) => write!(f, "invalid model: {m}"),
            OptError::Internal(m) => write!(f, "internal solver error: {m}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for OptError {}
