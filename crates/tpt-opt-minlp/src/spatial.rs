//! Spatial branch-and-bound for bilinear programs.
//!
//! [`spatial_branch_and_bound`] finds a global minimiser of
//!
//! ```text
//! minimise   cᵀx + Σₖ qₖ · x_{iₖ} · x_{jₖ}
//! subject to A_le x ≤ b_le,  A_eq x = b_eq,  lₓ ≤ x ≤ uₓ
//! ```
//!
//! with finite variable bounds. Each node of the search tree carries a box
//! `[l, u] ⊆ [lₓ, uₓ]`. Its lower bound is the optimum of an LP in which every
//! product `w = x_i·x_j` is replaced by the four McCormick envelope rows of
//! [`crate::relax::mccormick_envelope`]. That LP is solved by
//! [`tpt_opt_milp::MilpSolver`].
//!
//! - **Upper bound.** At each node the LP point `x` is checked against the
//!   linear rows. If it is feasible, its exact objective `cᵀx + Σ qₖ x_i x_j`
//!   is a valid incumbent.
//! - **Branching.** If some product is not exact at the LP point, the most
//!   violated product is selected and the wider of its two factors is bisected.
//! - **Node selection.** Best-first on the lower bound. A node is pruned when its
//!   bound is within the gap tolerance of the incumbent.
//!
//! The method is restricted to bilinear terms over finite boxes, which is the
//! factorable case where McCormick envelopes are the standard relaxation. It is
//! not a general global NLP solver.

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::{Solver, SolverStatus, VarBound};
use tpt_opt_milp::MilpSolver;

use crate::relax::{mccormick_envelope, FactorBounds};

/// One bilinear objective term `coeff · x_i · x_j`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Product {
    /// Index of the first factor.
    pub i: usize,
    /// Index of the second factor (may equal `i` for a square term).
    pub j: usize,
    /// Objective coefficient.
    pub coeff: f64,
}

/// A bilinear program over `n = c.len()` variables.
#[derive(Clone, Debug, PartialEq)]
pub struct BilinearProgram {
    /// Linear objective coefficients (minimised).
    pub c: Vec<f64>,
    /// Finite `(lower, upper)` bound for each variable.
    pub bounds: Vec<(f64, f64)>,
    /// Bilinear objective terms.
    pub products: Vec<Product>,
    /// Inequality rows `a · x ≤ b`, as `(a, b)`.
    pub le_rows: Vec<(Vec<f64>, f64)>,
    /// Equality rows `a · x = b`, as `(a, b)`.
    pub eq_rows: Vec<(Vec<f64>, f64)>,
}

/// Controls for [`spatial_branch_and_bound`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialConfig {
    /// Absolute optimality gap.
    pub abs_gap: f64,
    /// Relative optimality gap, applied to the incumbent's magnitude.
    pub rel_gap: f64,
    /// Maximum number of relaxations solved.
    pub max_nodes: usize,
    /// Tolerance for deciding that a product is exact and a row is satisfied.
    pub feas_tol: f64,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self { abs_gap: 1e-6, rel_gap: 1e-6, max_nodes: 20_000, feas_tol: 1e-7 }
    }
}

/// Termination status of [`spatial_branch_and_bound`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialStatus {
    /// The gap between incumbent and global lower bound is within tolerance.
    Optimal,
    /// The program has no feasible point.
    Infeasible,
    /// `max_nodes` relaxations were solved before the gap closed.
    NodeLimit,
    /// Some box was too small to bisect in floating point, so the search
    /// cannot certify optimality. `lower_bound` is still valid.
    Stalled,
}

/// Result of [`spatial_branch_and_bound`].
#[derive(Clone, Debug, PartialEq)]
pub struct SpatialResult {
    /// Termination status.
    pub status: SpatialStatus,
    /// Incumbent point (empty if none was found).
    pub x: Vec<f64>,
    /// Objective value of the incumbent (`+∞` if none was found).
    pub objective: f64,
    /// Global lower bound on the optimal value.
    pub lower_bound: f64,
    /// Number of node relaxations solved.
    pub nodes: usize,
}

/// Errors reported for malformed input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpatialError {
    /// Dimensions disagree, an index is out of range, a bound is infinite or
    /// crossed, or a coefficient is not finite.
    InvalidInput,
}

struct Node {
    lo: Vec<f64>,
    hi: Vec<f64>,
    bound: f64,
}

/// Find a global minimiser of a bilinear program by spatial branch-and-bound.
///
/// Returns [`SpatialError::InvalidInput`] if the dimensions disagree, a bound is
/// infinite or crossed, a product index is out of range, or a coefficient is
/// not finite.
pub fn spatial_branch_and_bound(
    prob: &BilinearProgram,
    cfg: &SpatialConfig,
) -> Result<SpatialResult, SpatialError> {
    validate(prob)?;

    let mut incumbent: Option<(Vec<f64>, f64)> = None;
    let mut open = vec![Node {
        lo: prob.bounds.iter().map(|b| b.0).collect(),
        hi: prob.bounds.iter().map(|b| b.1).collect(),
        bound: f64::NEG_INFINITY,
    }];
    let mut nodes = 0;
    // Smallest bound over every region removed from the search without being
    // split: pruned by bound, solved exactly, infeasible, or dropped because
    // it could not be bisected. Together with the open nodes this is the
    // global lower bound.
    let mut closed_bound = f64::INFINITY;
    let mut stalled = false;

    while let Some(pos) =
        open.iter().enumerate().min_by(|a, b| a.1.bound.total_cmp(&b.1.bound)).map(|(k, _)| k)
    {
        let ub = incumbent.as_ref().map_or(f64::INFINITY, |(_, f)| *f);
        if open[pos].bound >= ub - gap_tol(cfg, ub) {
            let node = open.swap_remove(pos);
            closed_bound = closed_bound.min(node.bound);
            continue;
        }
        if nodes >= cfg.max_nodes {
            let lb = lower_bound(&open).min(closed_bound);
            return Ok(finish(SpatialStatus::NodeLimit, incumbent, lb, nodes));
        }
        let node = open.swap_remove(pos);
        nodes += 1;

        let Some(relax) = solve_relaxation(prob, &node.lo, &node.hi) else {
            // Infeasible box: it contributes no feasible points.
            continue;
        };

        if linear_feasible(prob, &relax.x, cfg.feas_tol) {
            let f = objective_at(prob, &relax.x);
            if incumbent.as_ref().is_none_or(|(_, best)| f < *best) {
                incumbent = Some((relax.x.clone(), f));
            }
        }
        let ub = incumbent.as_ref().map_or(f64::INFINITY, |(_, f)| *f);
        if relax.value >= ub - gap_tol(cfg, ub) {
            closed_bound = closed_bound.min(relax.value);
            continue;
        }

        let Some(k) = most_violated(prob, &relax, cfg.feas_tol) else {
            // Every product is exact at the LP point: the box is solved, and
            // its value is the LP value (recorded as an incumbent above).
            closed_bound = closed_bound.min(relax.value);
            continue;
        };
        let p = prob.products[k];
        let pick =
            if node.hi[p.i] - node.lo[p.i] >= node.hi[p.j] - node.lo[p.j] { p.i } else { p.j };
        let mid = 0.5 * (node.lo[pick] + node.hi[pick]);
        if mid <= node.lo[pick] || mid >= node.hi[pick] {
            // Too small to bisect in floating point: this region can no longer
            // be refined, so optimality cannot be certified.
            stalled = true;
            closed_bound = closed_bound.min(relax.value);
            continue;
        }

        let mut left_hi = node.hi.clone();
        left_hi[pick] = mid;
        let mut right_lo = node.lo.clone();
        right_lo[pick] = mid;
        open.push(Node { lo: node.lo.clone(), hi: left_hi, bound: relax.value });
        open.push(Node { lo: right_lo, hi: node.hi, bound: relax.value });
    }

    let status = match (&incumbent, stalled) {
        (_, true) => SpatialStatus::Stalled,
        (Some(_), false) => SpatialStatus::Optimal,
        (None, false) => SpatialStatus::Infeasible,
    };
    Ok(finish(status, incumbent, closed_bound, nodes))
}

/// Solution of one node relaxation.
struct Relaxation {
    /// Primal `x` part of the LP solution.
    x: Vec<f64>,
    /// Value `w_k` of each product variable at the LP solution.
    w: Vec<f64>,
    /// LP optimal value (a lower bound on the node).
    value: f64,
}

/// Solve the McCormick LP over the box `[lo, hi]`. Returns `None` if the LP is
/// infeasible or the solver did not reach optimality.
fn solve_relaxation(prob: &BilinearProgram, lo: &[f64], hi: &[f64]) -> Option<Relaxation> {
    let n = prob.c.len();
    let p = prob.products.len();
    let mut model = Model::new(n + p);
    for i in 0..n {
        model.variables[i].bound = VarBound::continuous(lo[i], hi[i]);
    }
    let mut w_lo = Vec::with_capacity(p);
    let mut w_hi = Vec::with_capacity(p);
    for (k, prod) in prob.products.iter().enumerate() {
        let corners = [
            lo[prod.i] * lo[prod.j],
            lo[prod.i] * hi[prod.j],
            hi[prod.i] * lo[prod.j],
            hi[prod.i] * hi[prod.j],
        ];
        let wl = corners.iter().fold(f64::INFINITY, |m, v| m.min(*v));
        let wh = corners.iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v));
        w_lo.push(wl);
        w_hi.push(wh);
        model.variables[n + k].bound = VarBound::continuous(wl, wh);
    }

    let mut obj_idx: Vec<usize> = (0..n).collect();
    let mut obj_co: Vec<f64> = prob.c.clone();
    for (k, prod) in prob.products.iter().enumerate() {
        obj_idx.push(n + k);
        obj_co.push(prod.coeff);
    }
    model.set_objective(Objective {
        sense: Sense::Minimize,
        indices: obj_idx,
        coeffs: obj_co,
        constant: 0.0,
    });

    for (a, b) in &prob.le_rows {
        let (idx, co) = sparse_row(a);
        model.add_constraint(Constraint::le(idx, co, *b));
    }
    for (a, b) in &prob.eq_rows {
        let (idx, co) = sparse_row(a);
        model.add_constraint(Constraint::equality(idx, co, *b));
    }
    for (k, prod) in prob.products.iter().enumerate() {
        let b = FactorBounds { lx: lo[prod.i], ux: hi[prod.i], ly: lo[prod.j], uy: hi[prod.j] };
        for row in mccormick_envelope(prod.i, prod.j, n + k, b) {
            model.add_constraint(row);
        }
    }

    let sol = MilpSolver::new().with_threads(1).solve(&model).ok()?;
    if sol.status != SolverStatus::Optimal {
        return None;
    }
    let x = sol.primal[..n].to_vec();
    let w = sol.primal[n..].to_vec();
    Some(Relaxation { x, w, value: sol.objective_value })
}

/// Index of the product with the largest exactness violation at the LP point,
/// or `None` if every product is exact within `tol`.
fn most_violated(prob: &BilinearProgram, relax: &Relaxation, tol: f64) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (k, prod) in prob.products.iter().enumerate() {
        let gap = (relax.w[k] - relax.x[prod.i] * relax.x[prod.j]).abs();
        if gap > tol && best.is_none_or(|(_, g)| gap > g) {
            best = Some((k, gap));
        }
    }
    best.map(|(k, _)| k)
}

fn objective_at(prob: &BilinearProgram, x: &[f64]) -> f64 {
    let linear: f64 = prob.c.iter().zip(x).map(|(c, xi)| c * xi).sum();
    let bilinear: f64 = prob.products.iter().map(|p| p.coeff * x[p.i] * x[p.j]).sum();
    linear + bilinear
}

fn linear_feasible(prob: &BilinearProgram, x: &[f64], tol: f64) -> bool {
    let le_ok = prob
        .le_rows
        .iter()
        .all(|(a, b)| a.iter().zip(x).map(|(ai, xi)| ai * xi).sum::<f64>() <= b + tol);
    let eq_ok = prob
        .eq_rows
        .iter()
        .all(|(a, b)| (a.iter().zip(x).map(|(ai, xi)| ai * xi).sum::<f64>() - b).abs() <= tol);
    le_ok && eq_ok
}

fn sparse_row(a: &[f64]) -> (Vec<usize>, Vec<f64>) {
    a.iter().enumerate().filter(|(_, v)| **v != 0.0).map(|(i, v)| (i, *v)).unzip()
}

fn gap_tol(cfg: &SpatialConfig, ub: f64) -> f64 {
    if ub.is_finite() {
        cfg.abs_gap.max(cfg.rel_gap * ub.abs())
    } else {
        cfg.abs_gap
    }
}

/// Smallest relaxation bound among the open nodes.
fn lower_bound(open: &[Node]) -> f64 {
    open.iter().fold(f64::INFINITY, |m, nd| m.min(nd.bound))
}

fn finish(
    status: SpatialStatus,
    incumbent: Option<(Vec<f64>, f64)>,
    lower_bound: f64,
    nodes: usize,
) -> SpatialResult {
    match incumbent {
        Some((x, f)) => SpatialResult { status, x, objective: f, lower_bound, nodes },
        None => {
            SpatialResult { status, x: Vec::new(), objective: f64::INFINITY, lower_bound, nodes }
        }
    }
}

fn validate(prob: &BilinearProgram) -> Result<(), SpatialError> {
    let n = prob.c.len();
    let finite = |v: f64| v.is_finite();
    if n == 0
        || prob.bounds.len() != n
        || prob.bounds.iter().any(|&(lo, hi)| !finite(lo) || !finite(hi) || lo > hi)
        || prob.c.iter().any(|v| !finite(*v))
    {
        return Err(SpatialError::InvalidInput);
    }
    if prob.products.iter().any(|p| p.i >= n || p.j >= n || !finite(p.coeff)) {
        return Err(SpatialError::InvalidInput);
    }
    let rows_ok = |rows: &[(Vec<f64>, f64)]| {
        rows.iter().all(|(a, b)| a.len() == n && a.iter().all(|v| finite(*v)) && finite(*b))
    };
    if !rows_ok(&prob.le_rows) || !rows_ok(&prob.eq_rows) {
        return Err(SpatialError::InvalidInput);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn product(i: usize, j: usize, coeff: f64) -> Product {
        Product { i, j, coeff }
    }

    #[test]
    fn maximise_area_under_budget() {
        // min −x·y  s.t.  x + y ≤ 3,  x, y ∈ [0, 2]  ⇒  (1.5, 1.5), −2.25.
        let prob = BilinearProgram {
            c: vec![0.0, 0.0],
            bounds: vec![(0.0, 2.0), (0.0, 2.0)],
            products: vec![product(0, 1, -1.0)],
            le_rows: vec![(vec![1.0, 1.0], 3.0)],
            eq_rows: vec![],
        };
        let r = spatial_branch_and_bound(&prob, &SpatialConfig::default()).expect("valid");
        assert_eq!(r.status, SpatialStatus::Optimal, "{r:?}");
        assert!((r.objective + 2.25).abs() < 1e-5, "{r:?}");
        assert!((r.x[0] - 1.5).abs() < 1e-3 && (r.x[1] - 1.5).abs() < 1e-3, "{r:?}");
    }

    #[test]
    fn nonconvex_minimum_on_boundary() {
        // min x·y  s.t.  x + y ≥ 1,  x, y ∈ [0, 1]  ⇒  0 at (1, 0) or (0, 1).
        let prob = BilinearProgram {
            c: vec![0.0, 0.0],
            bounds: vec![(0.0, 1.0), (0.0, 1.0)],
            products: vec![product(0, 1, 1.0)],
            le_rows: vec![(vec![-1.0, -1.0], -1.0)],
            eq_rows: vec![],
        };
        let r = spatial_branch_and_bound(&prob, &SpatialConfig::default()).expect("valid");
        assert_eq!(r.status, SpatialStatus::Optimal, "{r:?}");
        assert!(r.objective.abs() < 1e-6, "{r:?}");
    }

    #[test]
    fn agrees_with_grid_search_on_mixed_program() {
        // min x·y − 2x + y  s.t.  x − y ≤ 1,  x, y ∈ [0, 3].
        let prob = BilinearProgram {
            c: vec![-2.0, 1.0],
            bounds: vec![(0.0, 3.0), (0.0, 3.0)],
            products: vec![product(0, 1, 1.0)],
            le_rows: vec![(vec![1.0, -1.0], 1.0)],
            eq_rows: vec![],
        };
        let r = spatial_branch_and_bound(&prob, &SpatialConfig::default()).expect("valid");
        assert_eq!(r.status, SpatialStatus::Optimal, "{r:?}");

        let steps = 300;
        let mut grid_best = f64::INFINITY;
        for a in 0..=steps {
            for b in 0..=steps {
                let x = 3.0 * a as f64 / steps as f64;
                let y = 3.0 * b as f64 / steps as f64;
                if x - y <= 1.0 {
                    grid_best = grid_best.min(objective_at(&prob, &[x, y]));
                }
            }
        }
        // The incumbent is within the absolute gap of the true optimum, and
        // the reported lower bound must not exceed it.
        assert!(r.objective <= grid_best + 1e-5, "B&B {r:?} vs grid {grid_best}");
        assert!(r.lower_bound <= r.objective + 1e-12, "{r:?}");
        assert!(r.objective - r.lower_bound <= 1e-5, "{r:?}");
        assert!(grid_best - r.objective < 0.05, "B&B {r:?} vs grid {grid_best}");
        assert!(linear_feasible(&prob, &r.x, 1e-7));
    }

    #[test]
    fn infeasible_program_is_reported() {
        // x + y ≤ 1 and x + y ≥ 3 cannot both hold on [0, 2]².
        let prob = BilinearProgram {
            c: vec![0.0, 0.0],
            bounds: vec![(0.0, 2.0), (0.0, 2.0)],
            products: vec![product(0, 1, 1.0)],
            le_rows: vec![(vec![1.0, 1.0], 1.0), (vec![-1.0, -1.0], -3.0)],
            eq_rows: vec![],
        };
        let r = spatial_branch_and_bound(&prob, &SpatialConfig::default()).expect("valid");
        assert_eq!(r.status, SpatialStatus::Infeasible, "{r:?}");
        assert!(r.x.is_empty());
    }

    #[test]
    fn rejects_malformed_input() {
        let base = BilinearProgram {
            c: vec![0.0, 0.0],
            bounds: vec![(0.0, 1.0), (0.0, 1.0)],
            products: vec![product(0, 1, 1.0)],
            le_rows: vec![],
            eq_rows: vec![],
        };
        let mut infinite = base.clone();
        infinite.bounds[0] = (0.0, f64::INFINITY);
        assert_eq!(
            spatial_branch_and_bound(&infinite, &SpatialConfig::default()),
            Err(SpatialError::InvalidInput)
        );
        let mut bad_index = base.clone();
        bad_index.products[0].j = 5;
        assert_eq!(
            spatial_branch_and_bound(&bad_index, &SpatialConfig::default()),
            Err(SpatialError::InvalidInput)
        );
    }
}
