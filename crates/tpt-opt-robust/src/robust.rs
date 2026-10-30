//! Adjustable robust optimisation: budgeted (Bertsimas–Sim Γ-robustness)
//! and ellipsoidal uncertainty sets with tractable LP reformulations.
//!
//! For an LP `min c·x s.t. â_i·x ≤ b_i ∀i, x ≥ 0` whose row coefficients
//! may deviate within symmetric boxes, the **budgeted** set allows at most
//! `Γ_i` coefficients of row `i` to deviate simultaneously. The robust
//! counterpart
//!
//! ```text
//! â_i·x + max{ Σ_j δ_ij x_j u_j : 0 ≤ u ≤ 1, Σ u ≤ Γ_i } ≤ b_i
//! ```
//!
//! has the exact linear reformulation (via LP duality)
//!
//! ```text
//! ā_i·x + Γ_i β_i + Σ_j z_ij ≤ b_i,   z_ij ≥ δ_ij x_j − β_i,   z ≥ 0, β ≥ 0
//! ```
//!
//! which this module builds directly into a [`tpt_opt_core::Model`].
//! The **ellipsoidal** variant protects against `â_i ~ N(ā_i, Σ_i)` at
//! level `κ` by replacing the support function with its column-norm upper
//! bound `√(xᵀΣx) ≤ Σ_j ‖L_{·j}‖₂ x_j` (valid for `x ≥ 0`), again yielding
//! a single linear row.

use std::vec::Vec;

use tpt_opt_conic::{ConeProgram, SocRow};
use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::VarBound;

/// A row whose coefficients are `ā + L u` with `u` in the Euclidean unit ball
/// and `Σ = L Lᵀ`, protected at level `κ` (see
/// [`ellipsoid_socp_reformulation`]).
#[derive(Debug, Clone)]
pub struct EllipsoidRow {
    /// Nominal coefficients `ā` (length `n`).
    pub nominal: Vec<f64>,
    /// Factor `L` stored as `n` rows of `m` entries: `factor[j][k] = L_{jk}`.
    pub factor: Vec<Vec<f64>>,
    /// Right-hand side `b`.
    pub rhs: f64,
}

/// Exact second-order-cone reformulation of the ellipsoidal robust LP.
///
/// For a row `(ā + L u)·x ≤ b` with `‖u‖₂ ≤ κ`, the worst case over the
/// ellipsoid is `ā·x + κ‖Lᵀx‖₂`, so the robust counterpart is
///
/// ```text
/// ‖Lᵀ x‖₂ ≤ (b − ā·x) / κ
/// ```
///
/// which is a second-order cone. The result is a [`ConeProgram`] to be solved
/// with [`tpt_opt_conic::solve_socp`]. It is less conservative than
/// [`ellipsoid_reformulation`], which bounds `‖Lᵀx‖₂` by `Σ_j ‖L_{·j}‖₂ x_j`.
///
/// `kappa` must be positive and finite. Dimensions are checked with `assert!`,
/// as in the other reformulations in this module.
pub fn ellipsoid_socp_reformulation(
    cost: Vec<f64>,
    bounds: Vec<(f64, f64)>,
    rows: Vec<EllipsoidRow>,
    kappa: f64,
) -> ConeProgram {
    assert!(kappa > 0.0 && kappa.is_finite(), "kappa must be positive and finite");
    let n = bounds.len();
    assert_eq!(cost.len(), n, "one cost per variable");
    let soc_rows = rows
        .iter()
        .map(|row| {
            assert_eq!(row.nominal.len(), n, "nominal length must match variables");
            assert_eq!(row.factor.len(), n, "factor needs one row per variable");
            let m = row.factor.first().map_or(0, Vec::len);
            assert!(row.factor.iter().all(|r| r.len() == m), "ragged factor");
            // Affine map rows: r(x) = (b − ā·x)/κ, then q_k(x) = Σ_j L_jk x_j.
            let mut q_mat = Vec::with_capacity(m + 1);
            let mut q_rhs = Vec::with_capacity(m + 1);
            q_mat.push(row.nominal.iter().map(|a| -a / kappa).collect());
            q_rhs.push(row.rhs / kappa);
            for k in 0..m {
                q_mat.push(row.factor.iter().map(|l_row| l_row[k]).collect());
                q_rhs.push(0.0);
            }
            SocRow { q_mat, q_rhs }
        })
        .collect();
    ConeProgram {
        n,
        c: cost,
        sense: Sense::Minimize,
        bounds,
        eq_a: Vec::new(),
        eq_b: Vec::new(),
        soc_rows,
        sdp_blocks: Vec::new(),
    }
}

/// One uncertain row: nominal coefficients `nominal ± deviation` per entry.
#[derive(Debug, Clone)]
pub struct UncertainRow {
    /// Nominal coefficients `ā_ij`.
    pub nominal: Vec<f64>,
    /// Maximum deviations `δ_ij ≥ 0`.
    pub deviation: Vec<f64>,
    /// Right-hand side `b_i`.
    pub rhs: f64,
}

/// Build the Bertsimas–Sim budgeted robust LP:
/// `min c·x` s.t. protected rows and `x ∈ bounds` (bounds are typically
/// `[0, u]`; the reformulation assumes `x ≥ 0`).
///
/// `gammas[i]` is the protection budget of row `i` (`Γ_i ≥ 0`; fractional
/// budgets are supported by the reformulation).
pub fn budgeted_reformulation(
    cost: Vec<f64>,
    bounds: Vec<(f64, f64)>,
    rows: Vec<UncertainRow>,
    gammas: Vec<f64>,
) -> Model {
    let n = bounds.len();
    assert_eq!(rows.len(), gammas.len(), "one gamma per row");
    // Layout: [x (n) | beta (m) | z (m*n)].
    let m = rows.len();
    let total = n + m + m * n;
    let mut model = Model::new(total);
    for (i, b) in bounds.iter().enumerate() {
        model.variables[i].bound = VarBound::continuous(b.0, b.1);
    }
    for v in model.variables[n..].iter_mut() {
        v.bound = VarBound::continuous(0.0, f64::INFINITY);
    }
    model.set_objective(Objective {
        sense: Sense::Minimize,
        indices: (0..n).filter(|&j| cost[j] != 0.0).collect(),
        coeffs: cost.iter().copied().filter(|&c| c != 0.0).collect(),
        constant: 0.0,
    });
    for (i, row) in rows.iter().enumerate() {
        let beta = n + i;
        let zbase = n + m + i * n;
        // Protected row: ā·x + Γ β + Σ z ≤ b.
        let mut idx: Vec<usize> =
            row.nominal.iter().enumerate().filter(|&(_, &c)| c != 0.0).map(|(j, _)| j).collect();
        let mut co: Vec<f64> = row.nominal.iter().copied().filter(|&c| c != 0.0).collect();
        idx.push(beta);
        co.push(gammas[i]);
        for j in 0..n {
            idx.push(zbase + j);
            co.push(1.0);
        }
        model.add_constraint(Constraint::le(idx, co, row.rhs));
        // z_ij ≥ δ_ij x_j − β_i  ⇔  δ x − β − z ≤ 0.
        for j in 0..n {
            if row.deviation[j] != 0.0 {
                model.add_constraint(Constraint::le(
                    vec![j, beta, zbase + j],
                    vec![row.deviation[j], -1.0, -1.0],
                    0.0,
                ));
            }
        }
    }
    model
}

/// Build the ellipsoidal-set robust LP: each row's coefficients are Gaussian
/// with mean `row.nominal` and covariance `Σ_i = L Lᵀ`; the row is protected
/// at level `kappa` (in standard-deviation units) using the conservative
/// column-norm linearisation `κ·Σ_j ‖L_{·j}‖₂ x_j` (exact support function
/// would require SOCP; see module docs).
///
/// `chol_col_norms[i][j]` is `‖L_{·j}‖₂` for row `i` (for a diagonal `Σ`
/// this is just the per-coefficient standard deviation).
pub fn ellipsoid_reformulation(
    cost: Vec<f64>,
    bounds: Vec<(f64, f64)>,
    rows: Vec<UncertainRow>,
    chol_col_norms: Vec<Vec<f64>>,
    kappa: f64,
) -> Model {
    let n = bounds.len();
    assert_eq!(rows.len(), chol_col_norms.len());
    let mut model = Model::new(n);
    for (i, b) in bounds.iter().enumerate() {
        model.variables[i].bound = VarBound::continuous(b.0, b.1);
    }
    model.set_objective(Objective {
        sense: Sense::Minimize,
        indices: (0..n).filter(|&j| cost[j] != 0.0).collect(),
        coeffs: cost.iter().copied().filter(|&c| c != 0.0).collect(),
        constant: 0.0,
    });
    for (row, norms) in rows.iter().zip(chol_col_norms.iter()) {
        assert_eq!(norms.len(), n);
        let mut idx: Vec<usize> =
            row.nominal.iter().enumerate().filter(|&(_, &c)| c != 0.0).map(|(j, _)| j).collect();
        let mut co: Vec<f64> = row.nominal.iter().copied().filter(|&c| c != 0.0).collect();
        for (j, &nj) in norms.iter().enumerate() {
            if nj != 0.0 {
                idx.push(j);
                co.push(kappa * nj);
            }
        }
        model.add_constraint(Constraint::le(idx, co, row.rhs));
    }
    model
}
