//! Outer approximation (OA) for convex mixed-integer nonlinear programs.
//!
//! The classic Duran–Grossmann scheme: alternate between
//!
//! 1. a **MILP master** over all variables plus an epigraph variable `η`,
//!    containing tangent-plane linearisations of the objective and of every
//!    nonlinear constraint accumulated at visited points, and
//! 2. an **NLP subproblem** with the integer variables fixed to the master's
//!    assignment, producing a feasible incumbent and fresh tangents.
//!
//! For convex objectives/constraints the tangents are global under-
//! estimators, so the master value is a valid lower bound and the loop
//! terminates with a duality-gap certificate ([`ConvergenceCertificate`]).
//! Indicator-gated constraints contribute tangents only while their gate is
//! satisfied at the generating point.

use std::vec::Vec;

use tpt_math_optimize_general::NlpParams;
use tpt_opt_core::{
    bounds::VarBound,
    model::{Constraint, Model, Objective},
    solver::{Solver, SolverStatus},
};
use tpt_opt_milp::MilpSolver;

use crate::certificates::{CertificateHistory, ConvergenceCertificate};
use crate::model::{ConstraintKind, MinlpModel, VarKind};
use crate::subproblem::solve_subproblem;

/// Tolerances / limits for the OA loop.
#[derive(Debug, Clone)]
pub struct OaConfig {
    /// Maximum number of master/subproblem iterations.
    pub max_iter: usize,
    /// Absolute duality-gap tolerance.
    pub abs_gap: f64,
    /// Relative duality-gap tolerance.
    pub rel_gap: f64,
    /// Feasibility tolerance when judging constraint violation.
    pub feas_tol: f64,
    /// NLP subproblem parameters.
    pub nlp: NlpParams,
}

impl Default for OaConfig {
    fn default() -> Self {
        Self {
            max_iter: 60,
            abs_gap: 1e-6,
            rel_gap: 1e-6,
            feas_tol: 1e-6,
            nlp: NlpParams { tol: 1e-8, ..NlpParams::default() },
        }
    }
}

/// Outcome of an OA solve.
#[derive(Debug, Clone)]
pub struct OaResult {
    /// Terminal status.
    pub status: OaStatus,
    /// Best primal point found (full dimension), if any.
    pub x: Option<Vec<f64>>,
    /// Its objective value, if any.
    pub objective: Option<f64>,
    /// Valid lower bound at termination.
    pub lower_bound: f64,
    /// Per-iteration certificates.
    pub history: CertificateHistory,
}

/// Terminal status of the OA loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OaStatus {
    /// Duality gap closed within tolerance.
    Optimal,
    /// No feasible point was ever found.
    Infeasible,
    /// Iteration limit reached without closing the gap (best effort).
    MaxIterations,
    /// The NLP subproblem failed numerically.
    NumericalIssue,
}

/// One accumulated linear cut over the master variables:
/// `sum coef_i * x_i >= rhs` when `ge`, else `<=`.
struct Cut {
    coefs: Vec<f64>,
    rhs: f64,
    ge: bool,
}

/// Solve a convex MINLP by outer approximation.
pub fn outer_approximate(model: &MinlpModel, cfg: &OaConfig) -> OaResult {
    let n = model.num_vars();
    let mut cuts: Vec<Cut> = Vec::new();
    let mut lower = f64::NEG_INFINITY;
    let mut upper = f64::INFINITY;
    let mut best_x: Option<Vec<f64>> = None;
    let mut history = CertificateHistory::default();
    let mut status = OaStatus::MaxIterations;

    for _iter in 0..cfg.max_iter {
        // ---- Master MILP ---------------------------------------------------
        let Some(msol) = solve_master(model, &cuts) else {
            status = OaStatus::Infeasible;
            break;
        };
        if msol.status != SolverStatus::Optimal {
            status = OaStatus::Infeasible;
            break;
        }
        lower = lower.max(msol.objective_value);
        let xm: Vec<f64> = msol.primal[..n].to_vec();

        // ---- Feasibility check at the master point -------------------------
        let violated: Vec<usize> = (0..model.constraints.len())
            .filter(|&ci| model.is_active(ci, &xm))
            .filter(|&ci| model.violation(ci, &xm, cfg.feas_tol) > 0.0)
            .collect();

        if !violated.is_empty() {
            // Tangent (feasibility) cuts at the master point.
            for &ci in &violated {
                add_constraint_cuts(model, ci, &xm, &mut cuts);
            }
            history.push(ConvergenceCertificate {
                lower_bound: lower,
                upper_bound: upper,
                gap: upper - lower,
            });
            continue;
        }

        // ---- NLP subproblem with integers fixed ----------------------------
        let ys: Vec<f64> = xm
            .iter()
            .enumerate()
            .map(|(i, v)| match model.vars[i] {
                VarKind::Continuous => *v,
                _ => v.round(),
            })
            .collect();
        let res = solve_subproblem(model, &ys, &cfg.nlp);
        let feas = res.status == tpt_math_optimize_general::NlpStatus::Converged
            && crate::subproblem::max_violation(model, &ys, &res.x) < 1e-4;

        if !feas {
            if res.status == tpt_math_optimize_general::NlpStatus::Diverged {
                status = OaStatus::NumericalIssue;
                break;
            }
            // Subproblem failed to converge cleanly: fall back to tangent
            // cuts at both the master point and the returned point to keep
            // refining the relaxation.
            for pt in [&xm, &res.x] {
                for ci in 0..model.constraints.len() {
                    if model.is_active(ci, pt) {
                        add_constraint_cuts(model, ci, pt, &mut cuts);
                    }
                }
            }
            history.push(ConvergenceCertificate {
                lower_bound: lower,
                upper_bound: upper,
                gap: upper - lower,
            });
            continue;
        }

        // ---- Incumbent update + optimality cut ------------------------------
        if res.objective < upper {
            upper = res.objective;
            best_x = Some(res.x.clone());
        }
        add_objective_cut(model, &res.x, &mut cuts);
        for ci in 0..model.constraints.len() {
            if model.is_active(ci, &res.x) {
                add_constraint_cuts(model, ci, &res.x, &mut cuts);
            }
        }
        history.push(ConvergenceCertificate {
            lower_bound: lower,
            upper_bound: upper,
            gap: upper - lower,
        });

        let gap = upper - lower;
        if gap <= cfg.abs_gap || gap <= cfg.rel_gap * upper.abs().max(1.0) {
            status = OaStatus::Optimal;
            break;
        }
    }

    let objective = best_x.as_ref().map(|x| model.eval_objective(x));
    OaResult { status, x: best_x, objective, lower_bound: lower, history }
}

/// Build and solve the epigraph master MILP: variables `x_0..x_{n-1}, η`
/// with `min η`, variable domains from the model, and all accumulated cuts.
fn solve_master(model: &MinlpModel, cuts: &[Cut]) -> Option<tpt_opt_core::solver::Solution> {
    let n = model.num_vars();
    let mut m = Model::new(n + 1);
    // Objective: minimise η (index n).
    m.set_objective(Objective::minimize(vec![n], vec![1.0]));
    for i in 0..n {
        m.variables[i].bound = match model.vars[i] {
            VarKind::Continuous => VarBound::continuous(model.lbs[i], model.ubs[i]),
            VarKind::Integer => VarBound::integer(model.lbs[i], model.ubs[i]),
            VarKind::Binary => {
                if model.lbs[i] == model.ubs[i] {
                    // Fixed binary (e.g. indicator gate pinned on/off).
                    VarBound::continuous(model.lbs[i], model.ubs[i])
                } else {
                    VarBound::binary()
                }
            }
        };
    }
    // Epigraph bound keeps the master bounded before any optimality cut.
    const ETA_BIG: f64 = 1e9;
    m.variables[n].bound = VarBound::continuous(-ETA_BIG, ETA_BIG);

    for cut in cuts {
        let mut idx: Vec<usize> = Vec::with_capacity(cut.coefs.len());
        let mut coefs: Vec<f64> = Vec::with_capacity(cut.coefs.len());
        for (i, &c) in cut.coefs.iter().enumerate() {
            if c.abs() > 1e-12 {
                idx.push(i);
                coefs.push(c);
            }
        }
        if idx.is_empty() {
            continue;
        }
        let con = if cut.ge {
            Constraint::ge(idx, coefs, cut.rhs)
        } else {
            Constraint::le(idx, coefs, cut.rhs)
        };
        m.add_constraint(con);
    }
    let mut solver = MilpSolver::new();
    solver.solve(&m).ok()
}

/// Add the objective tangent `η >= f(x_k) + ∇f(x_k)·(x − x_k)` at `xk`.
fn add_objective_cut(model: &MinlpModel, xk: &[f64], cuts: &mut Vec<Cut>) {
    let n = model.num_vars();
    let fk = model.eval_objective(xk);
    let grad = model.eval_objective_grad(xk);
    // η - Σ grad_i x_i >= f_k - Σ grad_i x_k,i
    let mut coefs = vec![0.0f64; n + 1];
    let mut rhs = fk;
    for i in 0..n {
        coefs[i] = -grad[i];
        rhs -= grad[i] * xk[i];
    }
    coefs[n] = 1.0;
    cuts.push(Cut { coefs, rhs, ge: true });
}

/// Add tangent linearisations of constraint `ci` at `xk`.
///
/// - `g(x) <= 0`: `g(x_k) + ∇g(x_k)·(x − x_k) <= 0`.
/// - `h(x) = 0`: both one-sided tangents with a small slack.
fn add_constraint_cuts(model: &MinlpModel, ci: usize, xk: &[f64], cuts: &mut Vec<Cut>) {
    let n = model.num_vars();
    let vk = model.eval_constraint(ci, xk);
    let grad = model.eval_constraint_grad(ci, xk);
    let mut lin = vec![0.0f64; n];
    let mut dot_gx = 0.0f64;
    for i in 0..n {
        lin[i] = grad[i];
        dot_gx += grad[i] * xk[i];
    }
    // Tangent at xk: v(xk) + Σ grad_i (x_i − xk_i) rel 0, i.e.
    // Σ grad_i x_i rel (Σ grad_i xk_i − v(xk)).
    match model.constraints[ci].kind {
        ConstraintKind::Le => cuts.push(Cut { coefs: lin, rhs: dot_gx - vk, ge: false }),
        ConstraintKind::Eq => {
            const SLACK: f64 = 1e-7;
            // Σ grad_i x_i <= dot − vk + s   and   −Σ grad_i x_i <= s − dot + vk
            cuts.push(Cut { coefs: lin.clone(), rhs: dot_gx - vk + SLACK, ge: false });
            let neg = lin.iter().map(|c| -c).collect();
            cuts.push(Cut { coefs: neg, rhs: SLACK - dot_gx + vk, ge: false });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tol_cfg() -> OaConfig {
        OaConfig { max_iter: 80, abs_gap: 1e-5, rel_gap: 1e-5, ..OaConfig::default() }
    }

    #[test]
    fn oa_solves_convex_minlp() {
        // min x + 2y  s.t.  y >= (x-1)^2,  x ∈ [0,2] continuous, y ∈ Z[0,4].
        // Optimum: x=1, y=0, obj=1 (unique: (0,1) costs 2).
        let mut m = MinlpModel::new(2, |x| x[0] + 2.0 * x[1]);
        m.set_var(0, VarKind::Continuous, 0.0, 2.0);
        m.set_var(1, VarKind::Integer, 0.0, 4.0);
        m.add_le(
            |x| (x[0] - 1.0) * (x[0] - 1.0) - x[1],
            |x, g| {
                g[0] = 2.0 * (x[0] - 1.0);
                g[1] = -1.0;
            },
        );

        let res = outer_approximate(&m, &tol_cfg());
        assert_eq!(res.status, OaStatus::Optimal, "history: {:?}", res.history);
        let obj = res.objective.expect("incumbent");
        assert!((obj - 1.0).abs() < 1e-4, "obj {obj}");
        let x = res.x.unwrap();
        assert!((x[0] - 1.0).abs() < 1e-3, "x0 {}", x[0]);
        assert!(x[1].abs() < 1e-6, "y {}", x[1]);
        // Lower bound must be a valid bound and the final gap tiny. The
        // incumbent is only ε-feasible (violation < 1e-8), so its objective
        // may sit a hair below the true optimum that the master bounds.
        let last = res.history.last().unwrap();
        assert!(last.lower_bound <= obj + 1e-4);
        assert!(last.gap.abs() < 1e-3, "gap {}", last.gap);
    }

    #[test]
    fn oa_handles_equality_and_integer_tradeoff() {
        // min x^2 + y  s.t.  x + y = 2.5,  x ∈ [0,3] continuous, y ∈ Z[0,3].
        // y=2 → x=0.5 → obj 2.25 (optimum); y=3 → 3.25; y=1 → 2.75... wait
        // y=1 → x=1.5 → 2.25+1=3.25? (1.5^2=2.25)+1=3.25. So optimum 2.25.
        let mut m = MinlpModel::new(2, |x| x[0] * x[0] + x[1]);
        m.set_var(0, VarKind::Continuous, 0.0, 3.0);
        m.set_var(1, VarKind::Integer, 0.0, 3.0);
        m.add_eq(
            |x| x[0] + x[1] - 2.5,
            |_x, g| {
                g[0] = 1.0;
                g[1] = 1.0;
            },
        );

        let res = outer_approximate(&m, &tol_cfg());
        assert_eq!(res.status, OaStatus::Optimal, "history: {:?}", res.history);
        let obj = res.objective.expect("incumbent");
        assert!((obj - 2.25).abs() < 1e-3, "obj {obj}");
        let x = res.x.unwrap();
        assert!((x[1] - 2.0).abs() < 1e-6, "y {}", x[1]);
        assert!((x[0] + x[1] - 2.5).abs() < 1e-4, "eq violated: {:?}", x);
    }

    #[test]
    fn oa_respects_indicator_gate() {
        // min x + y  s.t.  [b=1] y >= (x-1)^2,  b binary, x cont [0,2],
        // y integer [0,4]. With b free the constraint can be switched off:
        // optimum is x=0, y=0, b=0, obj=0.
        let mut m = MinlpModel::new(3, |x| x[0] + x[1]);
        m.set_var(0, VarKind::Continuous, 0.0, 2.0);
        m.set_var(1, VarKind::Integer, 0.0, 4.0);
        m.set_var(2, VarKind::Binary, 0.0, 1.0);
        let ci = m.add_le(
            |x| (x[0] - 1.0) * (x[0] - 1.0) - x[1],
            |x, g| {
                g[0] = 2.0 * (x[0] - 1.0);
                g[1] = -1.0;
            },
        );
        m.set_indicator(ci, 2, true);

        let res = outer_approximate(&m, &tol_cfg());
        assert_eq!(res.status, OaStatus::Optimal, "history: {:?}", res.history);
        let obj = res.objective.expect("incumbent");
        assert!(obj < 1e-4, "obj {obj} (expected ~0 with gate off)");
        let x = res.x.unwrap();
        assert!(x[2].abs() < 1e-6, "gate should be off, b={}", x[2]);

        // Force the gate on: b fixed to 1 → back to the gated optimum 1.
        m.set_var(2, VarKind::Binary, 1.0, 1.0);
        let res2 = outer_approximate(&m, &tol_cfg());
        assert_eq!(res2.status, OaStatus::Optimal);
        let obj2 = res2.objective.expect("incumbent");
        assert!((obj2 - 1.0).abs() < 1e-3, "obj {obj2}");
    }
}
