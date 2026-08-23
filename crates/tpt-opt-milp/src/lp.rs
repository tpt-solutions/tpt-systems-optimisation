//! Primal two-phase simplex LP solver used as the relaxation engine for the
//! branch-and-bound MILP solver.
//!
//! Converts a [`tpt_opt_core::model::Model`] (plus per-variable bound overrides
//! used by branching) into standard equality form with non-negative
//! structural/slack/surplus/artificial variables, runs phase I (minimise
//! infeasibility) then phase II (optimise the objective), and maps the solution
//! back to the original variables. Bland's rule is used for entering/leaving
//! selection to guarantee termination (no cycling).

#![allow(clippy::too_many_arguments, clippy::manual_memcpy, clippy::needless_range_loop)]

use std::vec::Vec;

use tpt_opt_core::{
    model::{Model, Objective, Sense},
    tolerance::Tolerances,
};

/// Sparse expression over tableau columns: `(column, coefficient)` pairs.
type ColTerms = Vec<(usize, f64)>;
/// Recovery entry mapping one original variable to tableau columns:
/// `x_j = sum(coeff * y_col) + shift`.
type RecoverEntry = (ColTerms, f64);

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

/// Full LP solve state: the recovered solution plus the final simplex tableau.
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
    /// Phase-II cost vector (already sign-adjusted for the model sense).
    pub lp_cost: Vec<f64>,
    /// Raw (un-signed) cost contribution per structural column.
    pub lp_cost_raw: Vec<f64>,
    /// Per-variable lower bounds used in the solve.
    pub var_lb: Vec<f64>,
    /// Per-variable upper bounds used in the solve.
    pub var_ub: Vec<f64>,
    /// Per-original-variable recovery info: `(terms, shift)` so that
    /// `x_j = sum(coeff * y_col) + shift`.
    pub recover: Vec<RecoverEntry>,
    /// Per-column expression in original variables: `y_col =
    /// sum(coef * x_var) + const`. `None` for artificial columns, which have
    /// no original-space meaning (used by tableau-space cut generation).
    pub col_expr: Vec<Option<RecoverEntry>>,
    /// Constant term of the objective.
    pub obj_constant: f64,
    /// Original model sense (retained for cut re-optimisation context).
    #[allow(dead_code)]
    pub sense: Sense,
}

/// Solve the LP relaxation, returning only the [`LpSolution`].
pub fn solve_lp(model: &Model, var_lb: &[f64], var_ub: &[f64], tol: Tolerances) -> LpSolution {
    solve_lp_state(model, var_lb, var_ub, tol).sol
}

/// Solve the LP relaxation, returning the full [`LpState`].
pub fn solve_lp_state(model: &Model, var_lb: &[f64], var_ub: &[f64], tol: Tolerances) -> LpState {
    let n = model.num_vars;
    let mut sol = LpSolution::empty(n, model.constraints.len());
    let sense = model.objective.sense;

    // ---- 1. Build standard-form variables ----------------------------------
    struct VarMap {
        terms: Vec<(usize, f64)>,
        shift: f64,
    }
    let mut varmaps: Vec<VarMap> = Vec::with_capacity(n);
    let mut lp_cost: Vec<f64> = Vec::new();
    let mut lp_cost_raw: Vec<f64> = Vec::new();
    let mut col_to_orig: Vec<Option<usize>> = Vec::new();
    let mut struct_ubs: Vec<Option<f64>> = Vec::new();
    // The model's fixed objective offset must participate in every reported
    // objective: branch-and-bound compares node bounds against heuristic
    // incumbents, and those incumbents are scored with `Objective::eval`
    // (constant included). Starting the accumulator at the model constant
    // keeps both conventions identical (the HiGHS binding restores the same
    // constant after its solve).
    let mut obj_constant = model.objective.constant;
    let mut free_ub_rows: Vec<(usize, usize, f64)> = Vec::new();

    for j in 0..n {
        let lb = var_lb[j];
        let ub = var_ub[j];
        let cj = obj_coeff(&model.objective, j);
        let signed = match sense {
            Sense::Maximize => -cj,
            Sense::Minimize => cj,
        };
        let mut terms: Vec<(usize, f64)> = Vec::new();
        let mut shift = 0.0;
        if lb.is_finite() {
            let col = lp_cost.len();
            lp_cost.push(signed);
            lp_cost_raw.push(cj);
            col_to_orig.push(Some(j));
            struct_ubs.push(if ub.is_finite() { Some(ub - lb) } else { None });
            terms.push((col, 1.0));
            shift = lb;
        } else {
            let yp = lp_cost.len();
            lp_cost.push(signed);
            lp_cost_raw.push(cj);
            col_to_orig.push(Some(j));
            struct_ubs.push(None);
            let ym = lp_cost.len();
            lp_cost.push(-signed);
            lp_cost_raw.push(-cj);
            col_to_orig.push(Some(j));
            struct_ubs.push(None);
            terms.push((yp, 1.0));
            terms.push((ym, -1.0));
            if ub.is_finite() {
                free_ub_rows.push((yp, ym, ub));
            }
        }
        obj_constant += cj * shift;
        varmaps.push(VarMap { terms, shift });
    }
    let recover: Vec<RecoverEntry> = varmaps.iter().map(|v| (v.terms.clone(), v.shift)).collect();

    // Original-space expression for each structural column, in column order:
    // a finite-lb variable contributes one column (`y = x_j - lb_j`), a free
    // variable contributes two split columns (`yp = x_j`, `ym = -x_j`).
    let mut struct_exprs: Vec<Option<RecoverEntry>> = Vec::with_capacity(lp_cost.len());
    for (j, v) in varmaps.iter().enumerate() {
        for &(col, coef) in &v.terms {
            while struct_exprs.len() < col {
                struct_exprs.push(None);
            }
            struct_exprs.push(Some((vec![(j, coef)], v.shift)));
        }
    }

    // ---- 2. Build constraint rows ------------------------------------------
    struct Row {
        lhs: Vec<(usize, f64)>,
        rhs: f64,
        kind: RowKind,
        /// Expression of the row's slack/surplus column in original
        /// variables: `s = sum(coef * x) + const` (None for equality rows).
        slack_expr: Option<(Vec<(usize, f64)>, f64)>,
    }
    let mut rows: Vec<Row> = Vec::with_capacity(model.constraints.len() + free_ub_rows.len());
    for c in &model.constraints {
        let has_lo = c.lower > f64::NEG_INFINITY + 1.0;
        let has_hi = c.upper < f64::INFINITY - 1.0;
        if has_lo && has_hi && (c.upper - c.lower).abs() > 1e-12 {
            let mk = |_target: f64| {
                let mut lhs = Vec::new();
                for (&var, &coef) in c.indices.iter().zip(c.coeffs.iter()) {
                    for &(col, coeff) in &varmaps[var].terms {
                        lhs.push((col, coef * coeff));
                    }
                }
                lhs
            };
            let shift_sum: f64 =
                c.indices.iter().zip(c.coeffs.iter()).map(|(&v, &a)| a * varmaps[v].shift).sum();
            rows.push(Row {
                lhs: mk(c.lower),
                rhs: c.lower - shift_sum,
                kind: RowKind::Ge,
                // s = sum a*x - lower
                slack_expr: Some((
                    c.indices.iter().copied().zip(c.coeffs.iter().copied()).collect(),
                    -c.lower,
                )),
            });
            rows.push(Row {
                lhs: mk(c.upper),
                rhs: c.upper - shift_sum,
                kind: RowKind::Le,
                // s = upper - sum a*x
                slack_expr: Some((
                    c.indices.iter().copied().zip(c.coeffs.iter().map(|a| -a)).collect(),
                    c.upper,
                )),
            });
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
            let slack_expr = match kind {
                // s = upper - sum a*x
                RowKind::Le => Some((
                    c.indices.iter().copied().zip(c.coeffs.iter().map(|a| -a)).collect(),
                    c.upper,
                )),
                // s = sum a*x - lower
                RowKind::Ge => Some((
                    c.indices.iter().copied().zip(c.coeffs.iter().copied()).collect(),
                    -c.lower,
                )),
                RowKind::Eq => None,
            };
            rows.push(Row { lhs, rhs, kind, slack_expr });
        }
    }
    for (yp, ym, ub) in free_ub_rows {
        let lhs = vec![(yp, 1.0), (ym, -1.0)];
        // The original variable j satisfies x_j = yp - ym; the row slack is
        // s = ub - x_j.
        let j = col_to_orig[yp].expect("free-split column maps to an original variable");
        rows.push(Row { lhs, rhs: ub, kind: RowKind::Le, slack_expr: Some((vec![(j, -1.0)], ub)) });
    }
    // Enforce finite upper bounds on structural columns as explicit <= rows so
    // the simplex respects variable bounds (binary/integer domains, etc.).
    for (c, u) in struct_ubs.iter().enumerate() {
        if let Some(u) = u {
            // y_c = x_j - lb_j, so the row slack is s = u - x_j + lb_j.
            let j = col_to_orig[c].expect("structural column maps to an original variable");
            let lb = var_lb[j];
            rows.push(Row {
                lhs: vec![(c, 1.0)],
                rhs: *u,
                kind: RowKind::Le,
                slack_expr: Some((vec![(j, -1.0)], u + lb)),
            });
        }
    }

    // ---- 3. Assemble the tableau -------------------------------------------
    let n_struct = lp_cost.len();
    // Normalise row orientations up front so the right-hand side is
    // non-negative: a `<=` row with negative rhs flips into a `>=` row (and
    // vice versa); equality rows merely negate. Slack/surplus expressions are
    // unaffected — the flipped row's surplus equals the original row's slack.
    // Column counts must be taken from the *normalised* kinds because flipped
    // rows need an artificial column.
    let norm_kinds: Vec<RowKind> = rows
        .iter()
        .map(|r| match r.kind {
            RowKind::Eq => RowKind::Eq,
            RowKind::Le if r.rhs < 0.0 => RowKind::Ge,
            RowKind::Le => RowKind::Le,
            RowKind::Ge if r.rhs < 0.0 => RowKind::Le,
            RowKind::Ge => RowKind::Ge,
        })
        .collect();
    let mut n_slack = 0usize;
    let mut n_art = 0usize;
    for k in &norm_kinds {
        match k {
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

    // Assemble per-column original-space expressions: structural columns in
    // order, then one slack/surplus expression per Le/Ge row (the order in
    // which slack columns are allocated), then artificials (None).
    let mut col_expr: Vec<Option<RecoverEntry>> = Vec::with_capacity(n_cols);
    col_expr.extend(struct_exprs);
    for r in &rows {
        if r.kind != RowKind::Eq {
            col_expr.push(r.slack_expr.clone());
        }
    }
    col_expr.resize(n_cols, None);

    if mrows == 0 {
        // No constraint rows: every variable is independent, so the LP is
        // unbounded as soon as any variable can improve indefinitely toward
        // an infinite bound.
        for j in 0..n {
            let cj = obj_coeff(&model.objective, j);
            if cj.abs() <= 1e-12 {
                continue;
            }
            let improving_increase = match sense {
                Sense::Minimize => cj < 0.0,
                Sense::Maximize => cj > 0.0,
            };
            let unbounded =
                if improving_increase { !var_ub[j].is_finite() } else { !var_lb[j].is_finite() };
            if unbounded {
                sol.status = LpStatus::Unbounded;
                return finish_state(
                    sol,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    n_struct,
                    n_cols,
                    0,
                    col_to_orig,
                    lp_cost,
                    lp_cost_raw,
                    var_lb,
                    var_ub,
                    recover,
                    col_expr,
                    obj_constant,
                    sense,
                );
            }
        }
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
        for (j, xj) in x.iter().enumerate() {
            obj += obj_coeff(&model.objective, j) * xj;
        }
        sol.status = LpStatus::Optimal;
        sol.x = x;
        sol.objective = obj;
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
            lp_cost_raw,
            var_lb: var_lb.to_vec(),
            var_ub: var_ub.to_vec(),
            recover,
            col_expr,
            obj_constant,
            sense,
        };
    }

    let mut a = vec![vec![0.0f64; n_cols]; mrows];
    let mut b = vec![0.0f64; mrows];
    let mut basis = vec![0usize; mrows];
    let mut art_of_row = vec![usize::MAX; mrows];

    let mut s_cursor = n_struct;
    let mut art_cursor = n_struct + n_slack;
    for (ri, r) in rows.iter().enumerate() {
        let sign = if r.rhs < 0.0 { -1.0 } else { 1.0 };
        b[ri] = r.rhs * sign;
        for &(col, coef) in &r.lhs {
            a[ri][col] += coef * sign;
        }
        match norm_kinds[ri] {
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
    }

    let mut cost = vec![0.0f64; n_cols];
    for ri in 0..mrows {
        if art_of_row[ri] != usize::MAX {
            cost[art_of_row[ri]] = 1.0;
        }
    }
    let mut iters = 0usize;
    if !run_simplex(&mut a, &mut b, &mut basis, &cost, &mut iters, n_cols, mrows, n_cols, tol) {
        sol.status = LpStatus::Infeasible;
        return finish_state(
            sol,
            a,
            b,
            basis,
            n_struct,
            n_cols,
            mrows,
            col_to_orig,
            lp_cost,
            lp_cost_raw,
            var_lb,
            var_ub,
            recover,
            col_expr,
            obj_constant,
            sense,
        );
    }
    for ri in 0..mrows {
        if art_of_row[ri] != usize::MAX && basis[ri] == art_of_row[ri] && b[ri] > tol.feasibility {
            sol.status = LpStatus::Infeasible;
            return finish_state(
                sol,
                a,
                b,
                basis,
                n_struct,
                n_cols,
                mrows,
                col_to_orig,
                lp_cost,
                lp_cost_raw,
                var_lb,
                var_ub,
                recover,
                col_expr,
                obj_constant,
                sense,
            );
        }
    }

    // Drive artificials that remain basic at zero level out of the basis so
    // they cannot drift positive during phase II (they carry zero cost
    // there). A row with no real column to pivot on is redundant; its
    // artificial stays basic at zero and no entering column can change it.
    let real_cols = n_struct + n_slack;
    for ri in 0..mrows {
        if art_of_row[ri] == usize::MAX || basis[ri] != art_of_row[ri] {
            continue;
        }
        let mut target = usize::MAX;
        for j in 0..real_cols {
            if a[ri][j].abs() > 1e-9 {
                target = j;
                break;
            }
        }
        if target != usize::MAX {
            pivot_at(&mut a, &mut b, &mut basis, ri, target, n_cols, mrows);
        }
    }

    let mut cost2 = vec![0.0f64; n_cols];
    for j in 0..n_struct {
        cost2[j] = lp_cost[j];
    }
    // Artificial columns are barred from entering during phase II.
    if !run_simplex(&mut a, &mut b, &mut basis, &cost2, &mut iters, n_cols, mrows, real_cols, tol) {
        sol.status = LpStatus::Unbounded;
        return finish_state(
            sol,
            a,
            b,
            basis,
            n_struct,
            n_cols,
            mrows,
            col_to_orig,
            lp_cost,
            lp_cost_raw,
            var_lb,
            var_ub,
            recover,
            col_expr,
            obj_constant,
            sense,
        );
    }

    let row_kinds: Vec<RowKind> = norm_kinds.clone();
    let (x, obj, dual, reduced_costs) = recover_solution(
        &a,
        &b,
        &basis,
        &recover,
        &lp_cost_raw,
        &col_to_orig,
        &row_kinds,
        n,
        n_struct,
        mrows,
        &model.constraints.len(),
        obj_constant,
    );
    sol.status = LpStatus::Optimal;
    sol.x = x;
    sol.objective = obj;
    sol.dual = dual;
    sol.reduced_costs = reduced_costs;
    sol.iterations = iters;

    finish_state(
        sol,
        a,
        b,
        basis,
        n_struct,
        n_cols,
        mrows,
        col_to_orig,
        lp_cost,
        lp_cost_raw,
        var_lb,
        var_ub,
        recover,
        col_expr,
        obj_constant,
        sense,
    )
}

/// Build the final `LpState` from all components.
#[allow(clippy::too_many_arguments)]
fn finish_state(
    sol: LpSolution,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    basis: Vec<usize>,
    n_struct: usize,
    n_cols: usize,
    mrows: usize,
    col_to_orig: Vec<Option<usize>>,
    lp_cost: Vec<f64>,
    lp_cost_raw: Vec<f64>,
    var_lb: &[f64],
    var_ub: &[f64],
    recover: Vec<RecoverEntry>,
    col_expr: Vec<Option<RecoverEntry>>,
    obj_constant: f64,
    sense: Sense,
) -> LpState {
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
        lp_cost_raw,
        var_lb: var_lb.to_vec(),
        var_ub: var_ub.to_vec(),
        recover,
        col_expr,
        obj_constant,
        sense,
    }
}

/// Recover `x`, objective, duals and reduced costs from a solved tableau.
fn recover_solution(
    a: &[Vec<f64>],
    b: &[f64],
    basis: &[usize],
    recover: &[(Vec<(usize, f64)>, f64)],
    lp_cost_raw: &[f64],
    col_to_orig: &[Option<usize>],
    row_kinds: &[RowKind],
    n: usize,
    n_struct: usize,
    mrows: usize,
    n_constraints: &usize,
    obj_constant: f64,
) -> (Vec<f64>, f64, Vec<f64>, Vec<f64>) {
    let mut y = vec![0.0f64; n_struct];
    for ri in 0..mrows {
        if basis[ri] < n_struct {
            y[basis[ri]] = b[ri];
        }
    }
    let mut x = vec![0.0f64; n];
    for (j, (terms, shift)) in recover.iter().enumerate() {
        let mut val = *shift;
        for &(col, coeff) in terms {
            val += coeff * y[col];
        }
        x[j] = val;
    }
    // Objective = shifted constant plus the structural-column contributions.
    // Variables fixed at a nonzero lower bound carry their cost through
    // `obj_constant` (their column value y = x - lb is zero).
    let mut obj = obj_constant;
    for k in 0..n_struct {
        obj += lp_cost_raw[k] * y[k];
    }
    let _ = col_to_orig;
    let _ = n;
    let _ = n_constraints;

    let n_cols = if a.is_empty() { 0 } else { a[0].len() };
    let mut full_cost = vec![0.0f64; n_cols];
    for k in 0..n_struct.min(n_cols) {
        full_cost[k] = lp_cost_raw[k];
    }
    let w = solve_basis_transpose(a, basis, &full_cost, mrows);
    let mut reduced = vec![0.0f64; n_struct];
    for j in 0..n_struct {
        let mut red = lp_cost_raw[j];
        for ri in 0..mrows {
            red -= w[ri] * a[ri][j];
        }
        reduced[j] = red;
    }
    let mut reduced_costs = vec![0.0f64; n];
    for (j, (terms, _shift)) in recover.iter().enumerate() {
        let mut rc = 0.0;
        for &(col, coeff) in terms {
            rc += coeff * reduced[col];
        }
        reduced_costs[j] = rc;
    }
    let mut dual = vec![0.0f64; *n_constraints];
    for (i, &kind) in row_kinds.iter().enumerate() {
        if i < dual.len() {
            dual[i] = -w[i]
                * match kind {
                    RowKind::Le => 1.0,
                    RowKind::Ge => -1.0,
                    RowKind::Eq => 1.0,
                };
        }
    }
    (x, obj, dual, reduced_costs)
}

/// Objective coefficient of variable `j`. Sparse entries may reference the
/// same variable more than once (e.g. scalarisations that sum overlapping
/// objectives); duplicates must *accumulate*, not resolve to the first match.
fn obj_coeff(obj: &Objective, j: usize) -> f64 {
    let mut total = 0.0;
    for (&i, &c) in obj.indices.iter().zip(obj.coeffs.iter()) {
        if i == j {
            total += c;
        }
    }
    total
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowKind {
    Le,
    Ge,
    Eq,
}

/// Pivot the tableau on `(leaving_row, entering_col)` (normalise the row,
/// eliminate the entering column elsewhere, update the basis).
fn pivot_at(
    a: &mut [Vec<f64>],
    b: &mut [f64],
    basis: &mut [usize],
    leaving: usize,
    entering: usize,
    n_cols: usize,
    mrows: usize,
) {
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

/// Run the simplex loop minimising `cost`. Returns `true` on optimal/feasible
/// basis, `false` if unbounded. Bland's rule for deterministic pivots.
/// Only columns `j < enter_limit` may enter (`enter_limit == n_cols` allows
/// everything; phase II passes the structural+slack count to keep artificial
/// columns out of the basis).
fn run_simplex(
    a: &mut [Vec<f64>],
    b: &mut [f64],
    basis: &mut [usize],
    cost: &[f64],
    iters: &mut usize,
    n_cols: usize,
    mrows: usize,
    enter_limit: usize,
    tol: Tolerances,
) -> bool {
    let max_iter = 20_000;
    while *iters <= max_iter {
        *iters += 1;
        let w = solve_basis_transpose(a, basis, cost, mrows);
        let mut entering = usize::MAX;
        let mut best_red = -tol.feasibility;
        for j in 0..enter_limit.min(n_cols) {
            let mut red = cost[j];
            for ri in 0..mrows {
                red -= w[ri] * a[ri][j];
            }
            if red < best_red - 1e-15
                || (red < best_red + 1e-15 && entering != usize::MAX && j < entering)
            {
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
        pivot_at(a, b, basis, leaving, entering, n_cols, mrows);
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
pub fn reoptimize(state: &mut LpState, tol: Tolerances) {
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
        n_cols,
        tol,
    ) {
        state.sol.status = LpStatus::Unbounded;
        return;
    }
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
    let mut obj = state.obj_constant;
    for k in 0..n_struct {
        obj += state.lp_cost_raw[k] * y[k];
    }
    state.sol.x = x;
    state.sol.objective = obj;
    state.sol.iterations += iters;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_opt_core::bounds::VarBound;
    use tpt_opt_core::model::Constraint;

    fn tol() -> Tolerances {
        Tolerances::default()
    }

    #[test]
    fn le_row_with_negative_rhs() {
        // maximise x0 + x1 s.t. -x0 - x1 <= -1 (i.e. x0 + x1 >= 1), binaries.
        // The row reaches the tableau with rhs = -1 < 0 and must flip into a
        // `>=` row without corrupting the starting basis.
        let mut m = Model::new(2);
        m.set_objective(Objective::maximize(vec![0, 1], vec![1.0, 1.0]));
        m.add_constraint(Constraint::le(vec![0, 1], vec![-1.0, -1.0], -1.0));
        m.variables[0].bound = VarBound::binary();
        m.variables[1].bound = VarBound::binary();
        let lb = vec![0.0, 0.0];
        let ub = vec![1.0, 1.0];
        let sol = solve_lp(&m, &lb, &ub, tol());
        assert_eq!(sol.status, LpStatus::Optimal);
        assert!((sol.objective - 2.0).abs() < 1e-6, "obj {}", sol.objective);
    }

    #[test]
    fn ge_row_with_negative_rhs() {
        // minimise x0 s.t. -2x0 <= -3 (i.e. x0 >= 1.5), x0 in [0, 5].
        let mut m = Model::new(1);
        m.set_objective(Objective::minimize(vec![0], vec![1.0]));
        m.add_constraint(Constraint::le(vec![0], vec![-2.0], -3.0));
        m.variables[0].bound = VarBound::continuous(0.0, 5.0);
        let lb = vec![0.0];
        let ub = vec![5.0];
        let sol = solve_lp(&m, &lb, &ub, tol());
        assert_eq!(sol.status, LpStatus::Optimal);
        assert!((sol.objective - 1.5).abs() < 1e-6, "obj {}", sol.objective);
    }

    #[test]
    fn mixed_sign_rows_stay_feasible() {
        // A model mixing positive- and negative-rhs rows of both kinds.
        let mut m = Model::new(3);
        m.set_objective(Objective::maximize(vec![0, 1, 2], vec![4.0, 3.0, 2.0]));
        m.add_constraint(Constraint::le(vec![0, 1, 2], vec![2.0, 3.0, 1.0], 5.0));
        m.add_constraint(Constraint::le(vec![0, 1, 2], vec![-1.0, -1.0, -1.0], -1.0));
        m.add_constraint(Constraint::ge(vec![0, 1, 2], vec![-1.0, -2.0, -1.0], -3.0));
        for v in m.variables.iter_mut() {
            v.bound = VarBound::binary();
        }
        let lb = vec![0.0, 0.0, 0.0];
        let ub = vec![1.0, 1.0, 1.0];
        let sol = solve_lp(&m, &lb, &ub, tol());
        assert_eq!(sol.status, LpStatus::Optimal);
        // The recovered point must be feasible and its objective consistent.
        for c in &m.constraints {
            assert!(c.is_satisfied(&sol.x, 1e-6), "row violated at {:?}", sol.x);
        }
        assert!(
            (sol.objective - m.objective.eval(&sol.x)).abs() < 1e-6,
            "obj {} != eval {}",
            sol.objective,
            m.objective.eval(&sol.x)
        );
        // Brute-force integer optimum: the LP bound must dominate it.
        let mut best = f64::NEG_INFINITY;
        for mask in 0..8u32 {
            let x: Vec<f64> = (0..3).map(|i| ((mask >> i) & 1) as f64).collect();
            if m.constraints.iter().all(|c| c.is_satisfied(&x, 1e-9)) {
                best = best.max(m.objective.eval(&x));
            }
        }
        assert!(sol.objective >= best - 1e-6, "lp {} < brute {best}", sol.objective);
    }

    #[test]
    fn duplicate_objective_entries_accumulate() {
        // Sparse objectives may reference the same variable twice (weighted
        // sums of overlapping objectives do exactly that). The LP must sum
        // the duplicate coefficients, not take the first match.
        let mut m = Model::new(1);
        m.variables[0].bound = VarBound::continuous(0.0, 3.0);
        // max x0 expressed as two +1 entries: effective coefficient 2.
        m.set_objective(Objective {
            sense: Sense::Maximize,
            indices: vec![0, 0],
            coeffs: vec![1.0, 1.0],
            constant: 0.0,
        });
        let sol = solve_lp(&m, &[0.0], &[3.0], tol());
        assert_eq!(sol.status, LpStatus::Optimal);
        assert!(
            (sol.objective - 6.0).abs() < 1e-6,
            "2*x0 at x0=3 must be 6, got {}",
            sol.objective
        );

        // Cancelling duplicates: coefficient 1 + (-1) = 0 -> objective is
        // identically zero and every feasible point is optimal.
        let mut m = Model::new(1);
        m.variables[0].bound = VarBound::continuous(0.0, 3.0);
        m.set_objective(Objective {
            sense: Sense::Minimize,
            indices: vec![0, 0],
            coeffs: vec![1.0, -1.0],
            constant: 7.5,
        });
        let sol = solve_lp(&m, &[0.0], &[3.0], tol());
        assert_eq!(sol.status, LpStatus::Optimal);
        assert!(
            (sol.objective - 7.5).abs() < 1e-6,
            "zero linear part must leave only the constant, got {}",
            sol.objective
        );
    }
}
