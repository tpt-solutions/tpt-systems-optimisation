//! Continuous NLP subproblem with the integer variables fixed at given
//! values — the workhorse of both outer approximation and generalized
//! Benders decomposition.
//!
//! Fixed integer/binary variables are **substituted out**: the NLP handed to
//! the underlying solver ranges over the continuous variables only, with
//! integer values baked into the objective, constraint and bound evaluation.
//! This avoids penalty-based "fixing" rows whose residual violation would
//! otherwise dominate the feasibility judgement.

use std::vec::Vec;

use tpt_opt_core::nlp::{solve_nlp, NlpParams, NlpProblem, NlpResult, NlpStatus};

use crate::model::{ConstraintKind, MinlpModel, VarKind};

/// Solve the continuous relaxation of `model` with every integer/binary
/// variable fixed to its value in `y` (rounded). Only constraints active at
/// the fixed point are enforced.
pub fn solve_subproblem(model: &MinlpModel, y: &[f64], params: &NlpParams) -> NlpResult {
    let n = model.num_vars();
    debug_assert_eq!(y.len(), n);
    let yr: Vec<f64> = y.iter().map(|v| v.round()).collect();
    let cidx: Vec<usize> = (0..n).filter(|&i| model.vars[i] == VarKind::Continuous).collect();

    // Starting point: midpoint of each continuous variable's box.
    let x0: Vec<f64> = cidx
        .iter()
        .map(|&i| ((model.lbs[i] + model.ubs[i]) * 0.5).clamp(model.lbs[i], model.ubs[i]))
        .collect();

    let prob = ReducedProblem { model, cidx: cidx.clone(), y: yr.clone() };
    let res = solve_nlp(&prob, &x0, params);

    // Lift back to the full space: continuous values scattered into their
    // original positions, integers pinned to their fixed values.
    let mut out = yr;
    for (k, &i) in cidx.iter().enumerate() {
        out[i] = res.x[k];
    }
    NlpResult { x: out, objective: res.objective, status: res.status, iterations: res.iterations }
}

/// Adapter implementing [`NlpProblem`] over the continuous subspace of a
/// [`MinlpModel`] with integer variables pinned to `y`.
struct ReducedProblem<'a> {
    model: &'a MinlpModel,
    /// Indices of the continuous variables, in increasing order.
    cidx: Vec<usize>,
    /// Rounded fixed values for every variable (full dimension).
    y: Vec<f64>,
}

impl ReducedProblem<'_> {
    fn lift(&self, xr: &[f64]) -> Vec<f64> {
        let mut x = self.y.clone();
        for (k, &i) in self.cidx.iter().enumerate() {
            x[i] = xr[k];
        }
        x
    }

    /// Inequality rows: 2 bounds per continuous variable + active `Le`
    /// nonlinear constraints.
    fn row_count(&self) -> usize {
        let n_le = self
            .model
            .constraints
            .iter()
            .enumerate()
            .filter(|(i, c)| c.kind == ConstraintKind::Le && self.model.is_active(*i, &self.y))
            .count();
        2 * self.cidx.len() + n_le
    }

    /// Scatter a full-dimensional gradient into the reduced coordinates.
    fn reduce_grad(&self, g_full: &[f64], g_red: &mut [f64]) {
        for (k, &i) in self.cidx.iter().enumerate() {
            g_red[k] = g_full[i];
        }
    }
}

impl NlpProblem for ReducedProblem<'_> {
    fn num_vars(&self) -> usize {
        self.cidx.len()
    }

    fn objective(&self, xr: &[f64]) -> f64 {
        let x = self.lift(xr);
        self.model.eval_objective(&x)
    }

    fn objective_grad(&self, xr: &[f64], g: &mut [f64]) {
        let x = self.lift(xr);
        let gf = self.model.eval_objective_grad(&x);
        self.reduce_grad(&gf, g);
    }

    fn num_ineq(&self) -> usize {
        self.row_count()
    }

    fn ineq(&self, i: usize, xr: &[f64]) -> f64 {
        let m = self.cidx.len();
        if i < 2 * m {
            let k = i / 2;
            let v = self.cidx[k];
            return if i % 2 == 0 { self.model.lbs[v] - xr[k] } else { xr[k] - self.model.ubs[v] };
        }
        let x = self.lift(xr);
        let mut kk = i - 2 * m;
        for (ci, c) in self.model.constraints.iter().enumerate() {
            if c.kind != ConstraintKind::Le || !self.model.is_active(ci, &self.y) {
                continue;
            }
            if kk == 0 {
                return (c.f)(&x);
            }
            kk -= 1;
        }
        0.0
    }

    fn ineq_grad(&self, i: usize, xr: &[f64], row: &mut [f64]) {
        for v in row.iter_mut() {
            *v = 0.0;
        }
        let m = self.cidx.len();
        if i < 2 * m {
            let k = i / 2;
            row[k] = if i % 2 == 0 { -1.0 } else { 1.0 };
            return;
        }
        let x = self.lift(xr);
        let mut kk = i - 2 * m;
        for (ci, c) in self.model.constraints.iter().enumerate() {
            if c.kind != ConstraintKind::Le || !self.model.is_active(ci, &self.y) {
                continue;
            }
            if kk == 0 {
                let g = self.model.eval_constraint_grad(ci, &x);
                self.reduce_grad(&g, row);
                return;
            }
            kk -= 1;
        }
    }

    fn num_eq(&self) -> usize {
        self.model
            .constraints
            .iter()
            .enumerate()
            .filter(|(i, c)| c.kind == ConstraintKind::Eq && self.model.is_active(*i, &self.y))
            .count()
    }

    fn eq(&self, j: usize, xr: &[f64]) -> f64 {
        let x = self.lift(xr);
        let mut k = j;
        for (ci, c) in self.model.constraints.iter().enumerate() {
            if c.kind != ConstraintKind::Eq || !self.model.is_active(ci, &self.y) {
                continue;
            }
            if k == 0 {
                return (c.f)(&x);
            }
            k -= 1;
        }
        0.0
    }

    fn eq_grad(&self, j: usize, xr: &[f64], row: &mut [f64]) {
        for v in row.iter_mut() {
            *v = 0.0;
        }
        let x = self.lift(xr);
        let mut k = j;
        for (ci, c) in self.model.constraints.iter().enumerate() {
            if c.kind != ConstraintKind::Eq || !self.model.is_active(ci, &self.y) {
                continue;
            }
            if k == 0 {
                let g = self.model.eval_constraint_grad(ci, &x);
                self.reduce_grad(&g, row);
                return;
            }
            k -= 1;
        }
    }
}

/// Maximum constraint violation (bounds included) at `x` for the subproblem
/// defined by integer assignment `y`. Used to judge feasibility of an NLP
/// result independently of the solver's own status.
pub fn max_violation(model: &MinlpModel, y: &[f64], x: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..model.num_vars() {
        worst = worst.max((model.lbs[i] - x[i]).max(0.0));
        worst = worst.max((x[i] - model.ubs[i]).max(0.0));
        if model.vars[i] != VarKind::Continuous {
            worst = worst.max((x[i] - y[i].round()).abs());
        }
    }
    for ci in 0..model.constraints.len() {
        if !model.is_active(ci, y) {
            continue;
        }
        worst = worst.max(match model.constraints[ci].kind {
            ConstraintKind::Le => model.eval_constraint(ci, x),
            ConstraintKind::Eq => model.eval_constraint(ci, x).abs(),
        });
    }
    worst
}

/// Convenience wrapper returning `(result, feasible)` where feasibility is
/// judged by [`max_violation`].
pub fn solve_feasible(model: &MinlpModel, y: &[f64], params: &NlpParams) -> (NlpResult, bool) {
    let res = solve_subproblem(model, y, params);
    let feas = res.status == NlpStatus::Converged && max_violation(model, y, &res.x) < 1e-4;
    (res, feas)
}
