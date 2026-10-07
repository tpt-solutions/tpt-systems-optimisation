//! Status, configuration, and result types.
//!
//! Termination criteria default from (and can be rebuilt from)
//! [`tpt_opt_core::Tolerances`] so every solver in the workspace retunes from
//! one authoritative bundle, and [`NlsStatus::to_solver_status`] maps onto
//! the canonical [`tpt_opt_core::SolverStatus`].

use tpt_opt_core::{SolverStatus, Tolerances};

/// Outcome of a least-squares solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NlsStatus {
    /// A converged stationary point (gradient norm below `gtol`).
    Converged,
    /// Stopped after `max_iter` outer iterations without meeting a criterion.
    MaxIterations,
    /// Step size collapsed below `xtol` without stationarity.
    StepTolerance,
    /// Cost improvement stalled below `ftol`.
    CostTolerance,
    /// The linear solve failed every iteration (singular normal equations
    /// even under damping).
    NumericalIssue,
    /// Invalid problem or configuration (e.g. crossed bounds).
    InvalidProblem,
}

impl NlsStatus {
    /// Map onto the canonical `tpt-opt-core` status.
    pub fn to_solver_status(self) -> SolverStatus {
        match self {
            NlsStatus::Converged => SolverStatus::Optimal,
            NlsStatus::MaxIterations | NlsStatus::StepTolerance | NlsStatus::CostTolerance => {
                SolverStatus::TimeLimit
            }
            NlsStatus::NumericalIssue => SolverStatus::NumericalIssue,
            NlsStatus::InvalidProblem => SolverStatus::Error,
        }
    }

    /// `true` when a usable parameter vector was produced.
    pub fn has_solution(&self) -> bool {
        matches!(
            self,
            NlsStatus::Converged
                | NlsStatus::MaxIterations
                | NlsStatus::StepTolerance
                | NlsStatus::CostTolerance
        )
    }
}

/// Errors from problem construction and solving.
#[derive(Debug, Clone)]
pub enum NlsError {
    /// The problem or configuration is inconsistent.
    InvalidProblem(String),
    /// The solver terminated without a usable solution.
    SolveFailed(NlsStatus),
}

impl std::fmt::Display for NlsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NlsError::InvalidProblem(msg) => write!(f, "invalid problem: {msg}"),
            NlsError::SolveFailed(s) => write!(f, "solve failed: {s:?}"),
        }
    }
}

impl std::error::Error for NlsError {}

/// Solver configuration. Defaults are tight enough for curve fitting; use
/// [`NlsConfig::from_tolerances`] to derive them from the workspace-wide
/// [`Tolerances`] bundle instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NlsConfig {
    /// Relative cost-change tolerance (`ftol`).
    pub ftol: f64,
    /// Relative step-size tolerance (`xtol`).
    pub xtol: f64,
    /// Gradient-norm tolerance (`gtol`, infinity norm of `Jᵀ W r`).
    pub gtol: f64,
    /// Maximum outer iterations.
    pub max_iter: usize,
    /// Maximum residual/Jacobian evaluations.
    pub max_evals: usize,
    /// Initial Levenberg damping.
    pub lambda0: f64,
    /// Robust loss applied to residuals.
    pub loss: crate::loss::Loss,
    /// Jacobian computation strategy.
    pub jacobian: crate::problem::JacobianMode,
    /// Normal-equations solve strategy.
    pub linear_solver: crate::problem::LinearSolver,
}

impl NlsConfig {
    /// Defaults: `ftol = xtol = 1e-12`, `gtol = 1e-10`, 200 iterations,
    /// plain L2, autodiff Jacobians, dense linear solve.
    pub fn new() -> Self {
        Self {
            ftol: 1e-12,
            xtol: 1e-12,
            gtol: 1e-10,
            max_iter: 200,
            max_evals: 10_000,
            lambda0: 1e-3,
            loss: crate::loss::Loss::L2,
            jacobian: crate::problem::JacobianMode::Autodiff,
            linear_solver: crate::problem::LinearSolver::Dense,
        }
    }

    /// Derive termination criteria from the workspace [`Tolerances`]:
    /// `ftol ← optimality_gap`, `gtol ← feasibility`, `xtol ← pivoting`.
    /// (Spec defaults give `1e-4 / 1e-6 / 1e-9` — a looser, one-size fit
    /// than [`NlsConfig::new`].)
    pub fn from_tolerances(t: &Tolerances) -> Self {
        Self { ftol: t.optimality_gap, gtol: t.feasibility, xtol: t.pivoting, ..Self::new() }
    }

    /// Override `ftol`.
    pub fn with_ftol(mut self, v: f64) -> Self {
        self.ftol = v;
        self
    }

    /// Override `xtol`.
    pub fn with_xtol(mut self, v: f64) -> Self {
        self.xtol = v;
        self
    }

    /// Override `gtol`.
    pub fn with_gtol(mut self, v: f64) -> Self {
        self.gtol = v;
        self
    }

    /// Override `max_iter`.
    pub fn with_max_iter(mut self, v: usize) -> Self {
        self.max_iter = v;
        self
    }

    /// Set the robust loss.
    pub fn with_loss(mut self, loss: crate::loss::Loss) -> Self {
        self.loss = loss;
        self
    }

    /// Set the Jacobian mode.
    pub fn with_jacobian(mut self, mode: crate::problem::JacobianMode) -> Self {
        self.jacobian = mode;
        self
    }

    /// Set the linear solver.
    pub fn with_linear_solver(mut self, ls: crate::problem::LinearSolver) -> Self {
        self.linear_solver = ls;
        self
    }
}

impl Default for NlsConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a least-squares solve.
#[derive(Debug, Clone)]
pub struct NlsResult {
    /// Termination status.
    pub status: NlsStatus,
    /// Final parameter vector.
    pub x: Vec<f64>,
    /// Final residual vector (`m`).
    pub residual: Vec<f64>,
    /// Final cost `½ Σ ρ(rᵢ²)`.
    pub cost: f64,
    /// Outer iterations performed.
    pub iterations: usize,
    /// Residual/Jacobian evaluations performed.
    pub evaluations: usize,
    /// Infinity norm of the (loss-weighted) gradient `Jᵀ W r` at `x`.
    pub gradient_norm: f64,
    /// Human-readable termination reason (also derivable from `status`).
    pub termination: &'static str,
    /// Final Jacobian row-major `m × n` (present when a solution exists).
    pub jacobian: Option<Vec<f64>>,
}

impl NlsResult {
    /// Gaussian covariance estimate `σ̂² (JᵀJ)⁻¹` with
    /// `σ̂² = 2·cost / max(1, m − n)` — undefined (returns `None`) without a
    /// stored Jacobian, for non-L2 losses (the IRLS weighting makes the
    /// Gaussian reading approximate; the plain `(JᵀJ)⁻¹` is still returned
    /// in that case, scaled by 1 rather than `σ̂²`), or on a singular
    /// normal matrix.
    pub fn covariance(&self) -> Option<Vec<f64>> {
        let j = self.jacobian.as_ref()?;
        let n = self.x.len();
        let m = self.residual.len();
        let h = crate::dense::normal_equations(j, m, n);
        let inv = crate::dense::spd_inverse(&h, n)?;
        let dof = (m.saturating_sub(n)).max(1) as f64;
        let sigma2 = 2.0 * self.cost / dof;
        Some(inv.into_iter().map(|v| v * sigma2).collect())
    }

    /// Standard errors (sqrt of the covariance diagonal) per parameter.
    pub fn standard_errors(&self) -> Option<Vec<f64>> {
        let cov = self.covariance()?;
        let n = self.x.len();
        Some((0..n).map(|i| cov[i * n + i].max(0.0).sqrt()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_maps_to_core() {
        assert_eq!(NlsStatus::Converged.to_solver_status(), SolverStatus::Optimal);
        assert_eq!(NlsStatus::NumericalIssue.to_solver_status(), SolverStatus::NumericalIssue);
        assert_eq!(NlsStatus::InvalidProblem.to_solver_status(), SolverStatus::Error);
        assert!(NlsStatus::Converged.has_solution());
        assert!(!NlsStatus::NumericalIssue.has_solution());
    }

    #[test]
    fn config_from_tolerances() {
        let t = Tolerances::spec_default();
        let c = NlsConfig::from_tolerances(&t);
        assert_eq!(c.ftol, t.optimality_gap);
        assert_eq!(c.gtol, t.feasibility);
        assert_eq!(c.xtol, t.pivoting);
    }
}
