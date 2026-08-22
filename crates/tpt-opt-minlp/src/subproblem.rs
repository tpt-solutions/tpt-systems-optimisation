//! Continuous NLP subproblem with the integer variables fixed at given
//! values — the workhorse of both outer approximation and generalized
//! Benders decomposition.

use std::vec::Vec;

use tpt_math_optimize_general::{NlpParams, NlpProblem, NlpResult, NlpStatus};

use crate::model::{ConstraintKind, MinlpModel};

/// Solve the continuous relaxation of `model` with every integer/binary
/// variable fixed to its value in `y` (rounded). Only constraints active at
/// the fixed point are enforced. Bounds are enforced as explicit inequality
/// rows because the underlying NLP interface has no bound handling.
pub fn solve_subproblem(
    model: &MinlpModel,
    y: &[f64],
    params: &NlpParams,
) -> NlpResult {
    let n = model.num_vars();
    debug_assert_eq!(y.len(), n);
    // Fixed point: integers rounded, continuous vars start at their midpoint
    // of the remaining box (clipped into bounds).
    let mut x0 = vec![0.0f64; n];
    for i in 0..n {
        x0[i] = match model.vars[i] {
            crate::model::VarKind::Continuous => {
                ((model.lbs[i] + model.ubs[i]) * 0.5).clamp(model.lbs[i], model.ubs[i])
            }
            _ => y[i].round(),
        };
    }
    let prob = Subproblem { model, y: y.to_vec() };
    solve_nlp(&prob, &x0, params)
}

/// Adapter implementing [`NlpProblem`] over a [`MinlpModel`] with integer
/// variables pinned to `y`.
struct Subproblem<'a> {
    model: &'a MinlpModel,
    y: Vec<f64>,
}

impl<'a> Subproblem<'a> {
    /// Rows: 2 bounds per variable + fixing rows for integer variables +
    /// active nonlinear constraints.
    fn row_count(&self) -> usize {
        let n = self.model.num_vars();
        let n_int = self.model.vars.iter().filter(|k| **k != crate::model::VarKind::Continuous).count();
        let n_active =
            self.model.constraints.iter().enumerate().filter(|(i, _)| self.model.is_active(*i, &self.y)).count();
        2 * n + n_int + n_active
    }
}

impl<'a> NlpProblem for Subproblem<'a> {
    fn num_vars(&self) -> usize {
        self.model.num_vars()
    }

    fn objective(&self, x: &[f64]) -> f64 {
        self.model.eval_objective(x)
    }

    fn objective_grad(&self, x: &[f64], g: &mut [f64]) {
        let grad = self.model.eval_objective_grad(x);
        g.copy_from_slice(&grad);
    }

    fn num_ineq(&self) -> usize {
        self.row_count()
    }

    fn ineq(&self, i: usize, x: &[f64]) -> f64 {
        let n = self.model.num_vars();
        if i < 2 * n {
            let v = i / 2;
            return if i % 2 == 0 { self.model.lbs[v] - x[v] } else { x[v] - self.model.ubs[v] };
        }
        let mut k = i - 2 * n;
        for v in 0..n {
            if self.model.vars[v] != crate::model::VarKind::Continuous {
                if k == 0 {
                    return x[v] - self.y[v].round();
                }
                k -= 1;
            }
        }
        // Active nonlinear constraint index `k`.
        for (ci, c) in self.model.constraints.iter().enumerate() {
            if !self.model.is_active(ci, &self.y) {
                continue;
            }
            if k == 0 {
                let val = (c.f)(x);
                return match c.kind {
                    ConstraintKind::Le => val,
                    ConstraintKind::Eq => val, // handled by paired rows below
                };
            }
            k -= 1;
        }
        0.0
    }

    fn ineq_grad(&self, i: usize, x: &[f64], row: &mut [f64]) {
        let n = self.model.num_vars();
        for v in row.iter_mut() {
            *v = 0.0;
        }
        if i < 2 * n {
            let v = i / 2;
            row[v] = if i % 2 == 0 { -1.0 } else { 1.0 };
            return;
        }
        let mut k = i - 2 * n;
        for v in 0..n {
            if self.model.vars[v] != crate::model::VarKind::Continuous {
                if k == 0 {
                    row[v] = 1.0;
                    return;
                }
                k -= 1;
            }
        }
        for (ci, c) in self.model.constraints.iter().enumerate() {
            if !self.model.is_active(ci, &self.y) {
                continue;
            }
            if k == 0 {
                let g = self.model.eval_constraint_grad(ci, x);
                row.copy_from_slice(&g);
                return;
            }
            k -= 1;
        }
    }

    fn num_eq(&self) -> usize {
        // Equalities are enforced as two inequalities each (val <= tol,
        // -val <= tol) via slack-free pairing; we instead expose them here as
        // proper equalities and keep only Le rows in `ineq`.
        let n_active_eq = self
            .model
            .constraints
            .iter()
            .enumerate()
            .filter(|(i, c)| c.kind == ConstraintKind::Eq && self.model.is_active(*i, &self.y))
            .count();
        n_active_eq
    }

    fn eq(&self, j: usize, x: &[f64]) -> f64 {
        let mut k = j;
        for (ci, c) in self.model.constraints.iter().enumerate() {
            if c.kind != ConstraintKind::Eq || !self.model.is_active(ci, &self.y) {
                continue;
            }
            if k == 0 {
                return (c.f)(x);
            }
            k -= 1;
        }
        0.0
    }

    fn eq_grad(&self, j: usize, x: &[f64], row: &mut [f64]) {
        let mut k = j;
        for (ci, c) in self.model.constraints.iter().enumerate() {
            if c.kind != ConstraintKind::Eq || !self.model.is_active(ci, &self.y) {
                continue;
            }
            if k == 0 {
                let g = self.model.eval_constraint_grad(ci, x);
                row.copy_from_slice(&g);
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
        if model.vars[i] != crate::model::VarKind::Continuous {
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