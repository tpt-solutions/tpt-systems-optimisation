//! Primal two-phase simplex LP solver used as the relaxation engine for the
//! branch-and-bound MILP solver.
//!
//! Converts a [`tpt_opt_core::model::Model`] (plus per-variable bound overrides
//! used by branching) into standard equality form with non-negative
//! structural/slack/surplus/artificial variables, runs phase I (minimise
//! infeasibility) then phase II (optimise the original objective), and maps the
//! solution back to the original variables. Bland's rule is used for
//! entering/leaving selection to guarantee termination (no cycling).

use std::vec::Vec;

use tpt_opt_core::{
    model::{Model, Objective, Sense},
    tolerance::Tolerances,
};

/// Terminal status of an LP solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpStatus {
    /// Optimal solution found.
    Optimal,
    /// The primal is infeasible.
    Infeasible,
    /// The primal is unbounded.
    Unbounded,
}

/// Result of solving an LP relaxation.
#[derive(Debug, Clone)]
pub struct LpSolution {
    /// Status of the solve.
    pub status: LpStatus,
    /// Values of the *original* variables (length = model.num_vars).
    pub x: Vec<f64>,
    /// Objective value at `x`, expressed in the model's own sense.
    pub objective: f64,
    /// Dual values, one per original constraint.
    pub dual: Vec<f64>,
    /// Reduced costs, one per original variable.
    pub reduced_costs: Vec<f64>,
    /// Iteration count across both phases.
    pub iterations: usize,
}

impl LpSolution {
    fn empty(n: usize, m: usize) -> Self {
        Self {
            status: LpStatus::Infeasible,
            x: vec![0.0; n],
            objective: 0.0,
            dual: vec![0.0; m],
            reduced_costs: vec![0.0; n],
            iterations: 0,
        }
    }
}

/// Full LP solve state: the recovered solution plus the final simplex
/// tableau. Exposed so cut generators (e.g. Gomory) can augment the tableau
/// and re-optimise.
#[derive(Debug, Clone)]
pub struct LpState {
    /// The recovered solution.
    pub sol: LpSolution,
    /// Final tableau constraint matrix (m x n_cols).
    pub a: Vec<Vec<f64>>,
    /// Right-hand side after pivoting.
    pub b: Vec<f64>,
    /// Basis column index for each row.
    pub basis: Vec<usize>,
    /// Number of structural (non-slack/artificial) columns.
    pub n_struct: usize,
    /// Total number of columns.
    pub n_cols: usize,
    /// Number of constraint rows.
    pub mrows: usize,
    /// `col_to_orig[c]` = original variable index represented by structural
    /// column `c` (if any).
    pub col_to_orig: Vec<Option<usize>>,
    /// Structural cost vector (phase II).
    pub lp_cost: Vec<f64>,
    /// Per-variable lower bounds used in the solve.
    pub var_lb: Vec<f64>,
    /// Per-variable upper bounds used in the solve.
    pub var_ub: Vec<f64>,
    /// Per-original-variable recovery info: `(terms, shift)` so that
    /// `x_j = sum(coeff * y_col) + shift`. Used to map a solved tableau back to
    /// original variables after augmentation (e.g. cuts).
    pub recover: Vec<(Vec<(usize, f64)>, f64)>,
    /// Constant term of the (minimise-sense) objective.
    pub obj_constant: f64,
    /// Original model sense (used to re-sign the objective after re-optimise).
    pub sense: Sense,
}

/// Solve the LP relaxation, returning only the [`LpSolution`].
pub fn solve_lp(model: &Model, var_lb: &[f64], var_ub: &[f64], tol: &Tolerances) -> LpSolution {
    solve_lp_state(model, var_lb, var_ub, tol).sol
}

/// Solve the LP relaxation, returning the full [`LpState`].
pub fn solve_lp_state(
    model: &Model,
    var_lb: &[f64],
    var_ub: &[f64],
    tol: &Tolerances,
) -> LpState {
    let n = model.num_vars;
    let mut sol = LpSolution::empty(n, model.constraints.len());

    // ---- 1. Build standard-form variables ----------------------------------
    struct VarMap {
        terms: Vec<(usize, f64)>,
        shift: f64,
    }
    let mut varmaps: Vec<VarMap> = Vec::with_capacity(n);
    let mut lp_cost: Vec<f64> = Vec::new();
    let mut col_to_orig: Vec<Option<usize>> = Vec::new();
    let mut obj_constant = 0.0;
    let mut free_ub_rows: Vec<(usize, usize, f64)> = Vec::new();

    for j in 0..n {
        let lb = var_lb[j];
        let ub = var_ub[j];
        let cj = obj_coeff(&model.objective, j);
        let mut terms: Vec<(usize, f64)> = Vec::new();
        let mut shift = 0.0;
        if lb.is_finite() {
            let col = lp_cost.len();
            lp_cost.push(cj);
            col_to_orig.push(Some(j));
            terms.push((col, 1.0));
            shift = lb;
        } else {
            let yp = lp_cost.len();
            lp_cost.push(cj);
            col_to_orig.push(Some(j));
            let ym = lp_cost.len();
            lp_cost.push(-cj);
            col_to_orig.push(Some(j));
            terms.push((yp, 1.0));
            terms.push((ym, -1.0));
            if ub.is_finite() {
                free_ub_rows.push((yp, ym, ub));
            }
        }
        obj_constant += cj * shift;
        varmaps.push(VarMap { terms, shift });
    }
    let recover: Vec<(Vec<(usize, f64)>, f64)> =
        varmaps.iter().map(|v| (v.terms.clone(), v.shift)).collect();

    // ---- 2. Build constraint rows ------------------------------------------
    struct Row {
        lhs: Vec<(usize, f64)>,
        rhs: f64,
        kind: RowKind,
    }
    let mut rows: Vec<Row> = Vec::with_capacity(model.constraints.len() + free_ub_rows.len());
    for c in &model.constraints {
        let has_lo = c.lower > f64::NEG_INFINITY + 1.0;
        let has_hi = c.upper < f64::INFINITY - 1.0;
        if has_lo && has_hi && (c.upper - c.lower).abs() > 1e-12 {
            let mk = |target: f64| {
                let mut lhs = Vec::new();
                let mut rhs = -target;
                for (&var, &coef) in c.indices.iter().zip(c.coeffs.iter()) {
                    for &(col, coeff) in &varmaps[var].terms {
                        lhs.push((col, coef * coeff));
                    }
                    rhs -= coef * varmaps[var].shift;
                }
                lhs
            };
            rows.push(Row { lhs: mk(c.lower), rhs: -c.lower, kind: RowKind::Ge });
            rows.push(Row { lhs: mk(c.upper), rhs: -c.upper, kind: RowKind::Le });
        } else {
            let mut lhs = Vec::new();
            let mut rhs = 0.0;
            for (&var, &coef) in c.indices.iter().zip(c.coeffs.iter()) {
                for &(col, coeff) in &varmaps[var].terms {
                    lhs.push((col, coef * coeff));
                }
                rhs -= coef * varmaps[var].shift;
            }
            let kind = if has_lo && has_hi {
                RowKind::Eq
            } else if has_hi {
                RowKind::Le
            } else {
                RowKind::Ge
            };
            rhs += match kind {
                RowKind::Le => c.upper,
                RowKind::Ge => c.lower,
                RowKind::Eq => c.lower,
            };
            rows.push(Row { lhs, rhs, kind });
        }
    }
    for (yp, ym, ub) in free_ub_rows {
        let mut lhs = Vec::new();
        lhs.push((yp, 1.0));
        lhs.push((ym, -1.0));
        rows.push(Row { lhs, rhs: ub, kind: RowKind::Le });
    }

    // ---- 3. Assemble the tableau -------------------------------------------
    let n_struct = lp_cost.len();
    let mut n_slack = 0usize;
    let mut n_art = 0usize;
    for r in &rows {
        match r.kind {
            RowKind::Le => n_slack += 1,
            RowKind::Ge => {
                n_slack += 1;
                n_art += 1;
            }
            RowKind::Eq => n_art += 1,
        }
    }
    let n_cols = n_struct + n_slack + n_art;
    let mrows = rows.len();

    if mrows == 0 {
        let mut x = vec![0.0f64; n];
        for j in 0..n {
            let cj = obj_coeff(&model.objective, j);
            if cj > 0.0 {
                x[j] = if var_ub[j].is_finite() { var_ub[j] } else { 0.0 };
            } else if cj < 0.0 {
                x[j] = if var_lb[j].is_finite() { var_lb[j] } else { 0.0 };
            }
        }
        let mut obj = obj_constant;
        for j in 0..n {
            obj += obj_coeff(&model.objective, j) * x[j];
        }
        sol.status = LpStatus::Optimal;
        sol.x = x;
        sol.objective = match model.objective.sense {
            Sense::Minimize => obj,
            Sense::Maximize => -obj,
        };
        return LpState {
            sol,
            a: Vec::new(),
            b: Vec::new(),
            basis: Vec::new(),
            n_struct,
            n_cols,
            mrows: 0,
            col_to_orig,
            lp_cost,
            var_lb: var_lb.to_vec(),
            var_ub: var_ub.to_vec(),
            recover: recover.clone(),
            obj_constant,
            sense: model.objective.sense,
        };
    }

    let mut a = vec![vec![0.0f64; n_cols]; mrows];
    let mut b = vec![0.0f64; mrows];
    let mut basis = vec![0usize; mrows];
    let mut art_of_row = vec![usize::MAX; mrows];

    let mut s_cursor = n_struct;
    let mut art_cursor = n_struct + n_slack;
    for (ri, r) in rows.iter().enumerate() {
        b[ri] = r.rhs;
        for &(col, coef) in &r.lhs {
            a[ri][col] += coef;
        }
        match r.kind {
            RowKind::Le => {
                a[ri][s_cursor] = 1.0;
                basis[ri] = s_cursor;
                s_cursor += 1;
            }
            RowKind::Ge => {
                a[ri][s_cursor] = -1.0;
                a[ri][art_cursor] = 1.0;
                basis[ri] = art_cursor;
                art_of_row[ri] = art_cursor;
                s_cursor += 1;
                art_cursor += 1;
            }
            RowKind::Eq => {
                a[ri][art_cursor] = 1.0;
                basis[ri] = art_cursor;
                art_of_row[ri] = art_cursor;
                art_cursor += 1;
            }
        }
        if b[ri] < 0.0 {
            for c in 0..n_cols {
                a[ri][c] = -a[ri][c];
            }
            b[ri] = -b[ri];
        }
    }

    let mut cost = vec![0.0f64; n_cols];
    for ri in 0..mrows {
        if art_of_row[ri] != usize::MAX {
            cost[art_of_row[ri]] = 1.0;
        }
    }
    let mut iters = 0usize;
    if !run_simplex(&mut a, &mut b, &mut basis, &cost, &mut iters, n_cols, mrows, tol) {
        sol.status = LpStatus::Infeasible;
        return LpState {
            sol,
            a,
            b,
            basis,
            n_struct,
            n_cols,
            mrows,
            col_to_orig,
            lp_cost,
            var_lb: var_lb.to_vec(),
            var_ub: var_ub.to_vec(),
            recover: recover.clone(),
            obj_constant,
            sense: model.objective.sense,
        };
    }
    for ri in 0..mrows {
        if art_of_row[ri] != usize::MAX
            && basis[ri] == art_of_row[ri]
            && b[ri] > tol.feasibility
        {
            sol.status = LpStatus::Infeasible;
            return LpState {
                sol,
                a,
                b,
                basis,
                n_struct,
                n_cols,
                mrows,
                col_to_orig,
                lp_cost,
                var_lb: var_lb.to_vec(),
                var_ub: var_ub.to_vec(),
                recover: recover.clone(),
                obj_constant,
                sense: model.objective.sense,
            };
        }
    }

    let mut cost2 = vec![0.0f64; n_cols];
    for j in 0..n_struct {
        cost2[j] = lp_cost[j];
    }
    if !run_simplex(&mut a, &mut b, &mut basis, &cost2, &mut iters, n_cols, mrows, tol) {
        sol.status = LpStatus::Unbounded;
        return LpState {
            sol,
            a,
            b,
            basis,
            n_struct,
            n_cols,
            mrows,
            col_to_orig,
            lp_cost,
            var_lb: var_lb.to_vec(),
            var_ub: var_ub.to_vec(),
            recover: recover.clone(),
            obj_constant,
            sense: model.objective.sense,
        };
    }

    let mut y = vec![0.0f64; n_struct];
    for ri in 0..mrows {
        if basis[ri] < n_struct {
            y[basis[ri]] = b[ri];
        }
    }
    let mut x = vec![0.0f64; n];
    for j in 0..n {
        let mut val = varmaps[j].shift;
        for &(col, coeff) in &varmaps[j].terms {
            val += coeff * y[col];
        }
        x[j] = val;
    }
    let mut obj = obj_constant;
    for j in 0..n {
        obj += obj_coeff(&model.objective, j) * x[j];
    }
    let obj_final = match model.objective.sense {
        Sense::Minimize => obj,
        Sense::Maximize => -obj,
    };

    let w = solve_basis_transpose(&a, &basis, &cost2, mrows);
    let mut reduced = vec![0.0f64; n_struct];
    for j in 0..n_struct {
        let mut red = cost2[j];
        for ri in 0..mrows {
            red -= w[ri] * a[ri][j];
        }
        reduced[j] = red;
    }
    let mut reduced_costs = vec![0.0f64; n];
    for j in 0..n {
        let mut rc = 0.0;
        for &(col, coeff) in &varmaps[j].terms {
            rc += coeff * reduced[col];
        }
        reduced_costs[j] = rc;
    }
    let mut dual = vec![0.0f64; model.constraints.len()];
    for (i, r) in rows.iter().enumerate() {
        if i < dual.len() {
            dual[i] = -w[i]
                * match r.kind {
                    RowKind::Le => 1.0,
                    RowKind::Ge => -1.0,
                    RowKind::Eq => 1.0,
                };
        }
    }

    sol.status = LpStatus::Optimal;
    sol.x = x;
    sol.objective = obj_final;
    sol.dual = dual;
    sol.reduced_costs = reduced_costs;
    sol.iterations = iters;

    LpState {
        sol,
        a,
        b,
        basis,
        n_struct,
        n_cols,
        mrows,
        col_to_orig,
        lp_cost,
        var_lb: var_lb.to_vec(),
        var_ub: var_ub.to_vec(),
        recover,
        obj_constant,
        sense: model.objective.sense,
    }
}

fn obj_coeff(obj: &Objective, j: usize) -> f64 {
    for (&i, &c) in obj.indices.iter().zip(obj.coeffs.iter()) {
        if i == j {
            return c;
        }
    }
    0.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowKind {
    Le,
    Ge,
    Eq,
}

/// Run the simplex loop minimising `cost`. Returns `true` on optimal/feasible
/// basis, `false` if unbounded. Bland's rule for deterministic pivots.
fn run_simplex(
    a: &mut [Vec<f64>],
    b: &mut [f64],
    basis: &mut [usize],
    cost: &[f64],
    iters: &mut usize,
    n_cols: usize,
    mrows: usize,
    tol: &Tolerances,
) -> bool {
    let max_iter = 20_000;
    while *iters <= max_iter {
        *iters += 1;
        let w = solve_basis_transpose(a, basis, cost, mrows);
        let mut entering = usize::MAX;
        let mut best_red = -tol.feasibility;
        for j in 0..n_cols {
            let mut red = cost[j];
            for ri in 0..mrows {
                red -= w[ri] * a[ri][j];
            }
            if red < best_red - 1e-15 {
                best_red = red;
                entering = j;
            } else if (red < best_red + 1e-15) && entering != usize::MAX && j < entering {
                best_red = red;
                entering = j;
            }
        }
        if entering == usize::MAX {
            return true;
        }
        let mut leaving = usize::MAX;
        let mut best_ratio = f64::INFINITY;
        for ri in 0..mrows {
            let av = a[ri][entering];
            if av > tol.feasibility {
                let ratio = b[ri] / av;
                if ratio < best_ratio - 1e-12
                    || (ratio < best_ratio + 1e-12
                        && leaving != usize::MAX
                        && basis[ri] < basis[leaving])
                {
                    best_ratio = ratio;
                    leaving = ri;
                }
            }
        }
        if leaving == usize::MAX {
            return false;
        }
        let piv = a[leaving][entering];
        for c in 0..n_cols {
            a[leaving][c] /= piv;
        }
        b[leaving] /= piv;
        for ri in 0..mrows {
            if ri != leaving {
                let f = a[ri][entering];
                if f.abs() > 0.0 {
                    for c in 0..n_cols {
                        a[ri][c] -= f * a[leaving][c];
                    }
                    b[ri] -= f * b[leaving];
                }
            }
        }
        basis[leaving] = entering;
    }
    false
}

/// Solve `B^T w = cost_basis` for `w` by Gaussian elimination on `B^T`.
fn solve_basis_transpose(a: &[Vec<f64>], basis: &[usize], cost: &[f64], mrows: usize) -> Vec<f64> {
    let mut at = vec![vec![0.0f64; mrows + 1]; mrows];
    for i in 0..mrows {
        for j in 0..mrows {
            at[i][j] = a[j][basis[i]];
        }
        at[i][mrows] = cost[basis[i]];
    }
    for col in 0..mrows {
        let mut piv = col;
        let mut bestv = at[col][col].abs();
        for r in col + 1..mrows {
            if at[r][col].abs() > bestv {
                bestv = at[r][col].abs();
                piv = r;
            }
        }
        if bestv < 1e-12 {
            return vec![0.0; mrows];
        }
        at.swap(col, piv);
        let d = at[col][col];
        for j in col..=mrows {
            at[col][j] /= d;
        }
        for r in 0..mrows {
            if r != col {
                let f = at[r][col];
                if f.abs() > 0.0 {
                    for j in col..=mrows {
                        at[r][j] -= f * at[col][j];
                    }
                }
            }
        }
    }
    let mut w = vec![0.0f64; mrows];
    for i in 0..mrows {
        w[i] = at[i][mrows];
    }
    w
}

/// Re-optimise a [`LpState`] after its tableau was augmented (e.g. by a cut),
/// running phase II from the current basis. Updates `state.sol`.
pub fn reoptimize(state: &mut LpState, tol: &Tolerances) {
    let n_cols = state.n_cols;
    let mrows = state.mrows;
    let mut cost = vec![0.0f64; n_cols];
    for j in 0..state.n_struct {
        cost[j] = state.lp_cost[j];
    }
    let mut iters = 0usize;
    if !run_simplex(
        &mut state.a,
        &mut state.b,
        &mut state.basis,
        &cost,
        &mut iters,
        n_cols,
        mrows,
        tol,
    ) {
        state.sol.status = LpStatus::Unbounded;
        return;
    }
    // Recover primal (structural columns -> original variables).
    let n_struct = state.n_struct;
    let n = state.var_lb.len();
    let mut y = vec![0.0f64; n_struct];
    for ri in 0..mrows {
        if state.basis[ri] < n_struct {
            y[state.basis[ri]] = state.b[ri];
        }
    }
    let mut x = vec![0.0f64; n];
    for (j, (terms, shift)) in state.recover.iter().enumerate() {
        let mut val = *shift;
        for &(col, coeff) in terms {
            val += coeff * y[col];
        }
        x[j] = val;
    }
    let mut obj_min = state.obj_constant;
    for k in 0..n_struct {
        obj_min += state.lp_cost[k] * y[k];
    }
    let obj = match state.sense {
        Sense::Minimize => obj_min,
        Sense::Maximize => -obj_min,
    };
    state.sol.x = x;
    state.sol.objective = obj;
    state.sol.iterations += iters;
}
