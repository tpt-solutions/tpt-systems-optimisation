//! Chance constraints: scenario approximations and Gaussian deterministic
//! equivalents.
//!
//! A one-sided chance constraint requires `P(a(ξ)·x ≤ b(ξ)) ≥ 1 − ε`.
//! Two tractable treatments are provided:
//!
//! - **Scenario approximation** ([`scenario_chance_model`]): sample `S`
//!   realisations, introduce a violation binary per scenario, and require
//!   at most `⌊ε·S⌋` violations — the standard sampling-and-binary MILP
//!   (a.k.a. VaR formulation).
//! - **Gaussian deterministic equivalent** ([`gaussian_chance_row`]): when
//!   `a(ξ)` is Gaussian with mean `μ` and covariance `Σ`,
//!   `μ·x + Φ⁻¹(1−ε)·√(xᵀΣx) ≤ b` is exact; since the workspace solves
//!   LP/MILPs, the quadratic term is conservatively linearised through the
//!   column-norm bound `√(xᵀΣx) ≤ Σ_j ‖L_{·j}‖₂·|x_j|` (with `L` the
//!   Cholesky factor of `Σ`), which is tight for diagonal `Σ` up to sign
//!   splitting and always yields a *feasible* (slightly conservative)
//!   linear constraint.

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::VarBound;
use tpt_opt_milp::MilpSolver;

/// Inverse standard-normal CDF (Acklam's rational approximation, |err| <
/// 1.15e-9 over the open interval).
// The published Acklam coefficients carry more digits than strictly needed
// for round-tripping; keep them verbatim for fidelity to the reference.
#[allow(clippy::excessive_precision)]
pub fn normal_quantile(p: f64) -> f64 {
    assert!((0.0..=1.0).contains(&p), "quantile argument must be in [0,1]");
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// One Gaussian chance row in conservative linearised form.
///
/// The original requirement `P(a·x ≤ b) ≥ 1 − ε` with `a ~ N(μ, Σ)` becomes
/// (exactly) `μ·x + Φ⁻¹(1−ε)·√(xᵀΣx) ≤ b`, and (conservatively, via the
/// column-norm bound) `μ·x + Σ_j prot_j·|x_j| ≤ rhs` where
/// `prot_j = Φ⁻¹(1−ε)·‖L_{·j}‖₂` and `rhs = b`.
#[derive(Debug, Clone)]
pub struct ChanceRow {
    /// Mean coefficients `μ`.
    pub mu: Vec<f64>,
    /// Protection coefficients on `|x_j|`.
    pub protection: Vec<f64>,
    /// Right-hand side `b`.
    pub rhs: f64,
}

/// Build a [`ChanceRow`] from the mean vector and the **column 2-norms of
/// the Cholesky factor** of the coefficient covariance (for a diagonal `Σ`
/// these are simply the per-coefficient standard deviations).
pub fn gaussian_chance_row(mu: Vec<f64>, chol_col_norms: Vec<f64>, b: f64, eps: f64) -> ChanceRow {
    assert_eq!(mu.len(), chol_col_norms.len());
    assert!((0.0..1.0).contains(&eps), "violation probability eps must be in (0,1)");
    let z = normal_quantile(1.0 - eps);
    ChanceRow {
        protection: chol_col_norms.into_iter().map(|n| z.max(0.0) * n).collect(),
        mu,
        rhs: b,
    }
}

/// Scenario (binary-indicator) approximation of a chance-constrained LP.
///
/// The base problem `min c·x s.t. A₀x ≤ b₀, x ∈ bounds` is augmented with
/// sampled realisations of an uncertain row: for scenario `s` the row reads
/// `a_s·x ≤ b_s`. Violation indicators `z_s ∈ {0,1}` satisfy
/// `a_s·x − b_s ≤ M·z_s` (big-M from variable bounds) and
/// `Σ_s z_s ≤ ⌊ε·S⌋`, so at most the allowed number of samples may be
/// violated.
///
/// Returns the optimal solution of the resulting MILP.
pub fn scenario_chance_model(
    cost: Vec<f64>,
    bounds: Vec<(f64, f64)>,
    base_rows: Vec<(Vec<f64>, f64)>,
    scenario_rows: Vec<(Vec<f64>, f64)>,
    eps: f64,
) -> Result<Vec<f64>, tpt_opt_core::OptError> {
    use tpt_opt_core::solver::Solver;
    let n = bounds.len();
    let s_count = scenario_rows.len();
    let allowed = ((eps * s_count as f64).floor() as usize).min(s_count);
    let mut model = Model::new(n + s_count);
    for (i, b) in bounds.iter().enumerate() {
        model.variables[i].bound = VarBound::continuous(b.0, b.1);
    }
    for v in model.variables[n..].iter_mut() {
        v.bound = VarBound::binary();
    }
    model.set_objective(Objective {
        sense: Sense::Minimize,
        indices: (0..n).filter(|&j| cost[j] != 0.0).collect(),
        coeffs: cost.iter().copied().filter(|&c| c != 0.0).collect(),
        constant: 0.0,
    });
    for (a, b) in &base_rows {
        let idx: Vec<usize> =
            a.iter().enumerate().filter(|&(_, &c)| c != 0.0).map(|(j, _)| j).collect();
        let co: Vec<f64> = a.iter().copied().filter(|&c| c != 0.0).collect();
        model.add_constraint(Constraint::le(idx, co, *b));
    }
    // Big-M per scenario: max |a_s·x − b_s| bounded by row range over bounds.
    for (s, (a, b)) in scenario_rows.iter().enumerate() {
        let mut span = 0.0f64;
        for (&c, &(lo, hi)) in a.iter().zip(bounds.iter()) {
            span += c.abs() * (hi - lo).max(0.0);
        }
        let big_m = span + b.abs() + 1.0;
        let mut idx: Vec<usize> =
            a.iter().enumerate().filter(|&(_, &c)| c != 0.0).map(|(j, _)| j).collect();
        let mut co: Vec<f64> = a.iter().copied().filter(|&c| c != 0.0).collect();
        idx.push(n + s);
        co.push(-big_m);
        model.add_constraint(Constraint::le(idx, co, *b));
    }
    // Σ z_s ≤ allowed.
    model.add_constraint(Constraint::le(
        (n..n + s_count).collect(),
        vec![1.0; s_count],
        allowed as f64,
    ));
    let mut solver = MilpSolver::new();
    let sol = solver.solve(&model)?;
    Ok(sol.primal[..n].to_vec())
}
