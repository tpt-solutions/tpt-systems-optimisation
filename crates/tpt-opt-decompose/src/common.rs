//! Shared helpers: row-sense canonicalisation and explicit dual-LP
//! construction used by the Benders machinery.
//!
//! Every recourse/block subproblem is canonicalised to the uniform form
//!
//! ```text
//! min  d·y   s.t.  A·y ≥ β − Γ·x,   y ≥ 0
//! ```
//!
//! (`Le` rows are negated, `Eq` rows split, variable upper bounds become
//! extra rows). Its dual,
//!
//! ```text
//! max  πᵀ(β − Γ·x̂)   s.t.  Aᵀπ ≤ d,  π ≥ 0
//! ```
//!
//! is then solved *explicitly* as an LP, which sidesteps any ambiguity in
//! the underlying simplex engine's dual sign conventions: whatever π comes
//! back is dual-feasible by construction, so `πᵀb(x)` is a valid global
//! under-estimator of `Q(x)` and doubles as a Benders cut.

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::VarBound;
use tpt_opt_milp::lp::{solve_lp, LpSolution};

/// Row sense for user-supplied rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowSense {
    /// `≤`
    Le,
    /// `≥`
    Ge,
    /// `=`
    Eq,
}

/// A canonicalised row `A·y ≥ β − Γ·x`.
#[derive(Debug, Clone)]
pub(crate) struct CanonRow {
    /// Coefficients on the block/recourse variables.
    pub a: Vec<f64>,
    /// Coefficients on the linking/first-stage variables (negated into the
    /// right-hand side).
    pub gamma: Vec<f64>,
    /// Constant part of the right-hand side.
    pub beta: f64,
}

/// Convert one user row into zero or more canonical `≥` rows.
pub(crate) fn canon_row(y: &[f64], x: &[f64], sense: RowSense, rhs: f64) -> Vec<CanonRow> {
    match sense {
        RowSense::Ge => vec![CanonRow { a: y.to_vec(), gamma: x.to_vec(), beta: rhs }],
        RowSense::Le => {
            vec![CanonRow {
                a: y.iter().map(|&v| -v).collect(),
                gamma: x.iter().map(|&v| -v).collect(),
                beta: -rhs,
            }]
        }
        RowSense::Eq => {
            vec![
                CanonRow { a: y.to_vec(), gamma: x.to_vec(), beta: rhs },
                CanonRow {
                    a: y.iter().map(|&v| -v).collect(),
                    gamma: x.iter().map(|&v| -v).collect(),
                    beta: -rhs,
                },
            ]
        }
    }
}

/// Solve the explicit dual LP
///
/// ```text
/// max  Σ_r (β_r − Γ_r·x̂) π_r     s.t.  Σ_r A[r][j] π_r ≤ d_j ∀j,  0 ≤ π ≤ cap
/// ```
///
/// Returns the raw [`LpSolution`] (status, optimal π in `.x`, value in
/// `.objective`). With `cap` finite the dual is always bounded; callers use
/// large caps for optimality duals and `cap = 1` for Farkas/phase-1
/// certificates.
pub(crate) fn solve_block_dual(
    a: &[Vec<f64>],
    beta: &[f64],
    gamma: &[Vec<f64>],
    d: &[f64],
    x_hat: &[f64],
    cap: f64,
) -> LpSolution {
    let m = beta.len();
    let mut model = Model::new(m);
    for v in model.variables.iter_mut() {
        v.bound = VarBound::continuous(0.0, cap);
    }
    // Dual feasibility: one row per primal (recourse) variable.
    for (j, &dj) in d.iter().enumerate() {
        let idx: Vec<usize> = (0..m).collect();
        let coeffs: Vec<f64> = (0..m).map(|r| a[r][j]).collect();
        model.add_constraint(Constraint::le(idx, coeffs, dj));
    }
    // Objective: maximise πᵀb(x̂).
    let b_hat: Vec<f64> = beta.iter().zip(gamma.iter()).map(|(&b, g)| b - dot(g, x_hat)).collect();
    let idx: Vec<usize> = (0..m).filter(|&r| b_hat[r] != 0.0).collect();
    let coeffs: Vec<f64> = idx.iter().map(|&r| b_hat[r]).collect();
    model.set_objective(Objective { sense: Sense::Maximize, indices: idx, coeffs, constant: 0.0 });
    let lb = vec![0.0; m];
    let ub = vec![cap; m];
    solve_lp(&model, &lb, &ub, tpt_opt_core::tolerance::Tolerances::spec_default())
}

/// Dot product of two equal-length slices (0 for mismatched lengths).
pub(crate) fn dot(u: &[f64], v: &[f64]) -> f64 {
    u.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum()
}
