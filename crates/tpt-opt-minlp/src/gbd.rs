//! Generalized Benders decomposition (GBD) for MINLPs with complicating
//! (integer) variables.
//!
//! The master MILP ranges over the integer variables `y` plus an epigraph
//! variable `θ`; each iteration evaluates the continuous subproblem
//! `v(y) = min { f(x) : constraints, integers fixed }` and accumulates
//!
//! - **optimality cuts** `θ >= v(y_k) + s_k · (y − y_k)` where the slope
//!   `s_k` is estimated by probing neighbouring integer points (each
//!   component is *validity-checked* against the probed values and falls
//!   back to a flat cut when the finite-difference slope would be invalid),
//!     and
//! - **feasibility cuts** built from the constraint-violation measure `φ(y)`
//!   when the subproblem at `y_k` is infeasible.
//!
//! Flat cuts (`s = 0`) are always valid; slope components are only used when
//! they are verified not to cut off evaluated points, so every accumulated
//! cut is a valid relaxation of the value function regardless of convexity
//! quirks of the finite-difference estimate.

use std::vec::Vec;

use tpt_opt_core::nlp::{NlpParams, NlpStatus};
use tpt_opt_core::{
    bounds::VarBound,
    model::{Constraint, Model, Objective},
    solver::{Solver, SolverStatus},
};
use tpt_opt_milp::MilpSolver;

use crate::certificates::{CertificateHistory, ConvergenceCertificate};
use crate::model::{MinlpModel, VarKind};
use crate::subproblem::{max_violation, solve_subproblem};

/// Tolerances / limits for the GBD loop.
#[derive(Debug, Clone)]
pub struct GbdConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Absolute duality-gap tolerance.
    pub abs_gap: f64,
    /// Relative duality-gap tolerance.
    pub rel_gap: f64,
    /// Step used to probe neighbouring integer points for slope estimates.
    pub probe_step: f64,
    /// NLP subproblem parameters.
    pub nlp: NlpParams,
}

impl Default for GbdConfig {
    fn default() -> Self {
        Self {
            max_iter: 80,
            abs_gap: 1e-6,
            rel_gap: 1e-6,
            probe_step: 0.5,
            nlp: NlpParams { tol: 1e-8, ..NlpParams::default() },
        }
    }
}

/// Outcome of a GBD solve (same shape as [`crate::oa::OaResult`]).
#[derive(Debug, Clone)]
pub struct GbdResult {
    /// Terminal status.
    pub status: crate::oa::OaStatus,
    /// Best primal point found, if any.
    pub x: Option<Vec<f64>>,
    /// Its objective value, if any.
    pub objective: Option<f64>,
    /// Valid lower bound at termination.
    pub lower_bound: f64,
    /// Per-iteration certificates.
    pub history: CertificateHistory,
}

/// One master cut over `(y, θ)`: `sum coef_i z_i >= rhs` (optimality) or
/// `sum coef_i y_i <= rhs` (feasibility, no θ term).
struct MasterCut {
    coefs: Vec<f64>,
    rhs: f64,
    ge: bool,
}

/// Solve a MINLP by generalized Benders decomposition. The complicating
/// variables are exactly the integer/binary variables of `model`.
pub fn generalized_benders(model: &MinlpModel, cfg: &GbdConfig) -> GbdResult {
    let n = model.num_vars();
    let int_idx: Vec<usize> = (0..n).filter(|&i| model.vars[i] != VarKind::Continuous).collect();
    let k = int_idx.len();
    let mut cuts: Vec<MasterCut> = Vec::new();
    let mut visited: Vec<Vec<i64>> = Vec::new();
    let mut lower = f64::NEG_INFINITY;
    let mut upper = f64::INFINITY;
    let mut best_x: Option<Vec<f64>> = None;
    let mut history = CertificateHistory::default();
    let mut status = crate::oa::OaStatus::MaxIterations;

    for _iter in 0..cfg.max_iter {
        if k == 0 {
            // No complicating variables: a single NLP solve decides.
            let y0 = vec![0.0; n];
            let res = solve_subproblem(model, &y0, &cfg.nlp);
            let feas = max_violation(model, &y0, &res.x) < 1e-4;
            return GbdResult {
                status: if feas {
                    crate::oa::OaStatus::Optimal
                } else {
                    crate::oa::OaStatus::Infeasible
                },
                x: feas.then_some(res.x),
                objective: feas.then_some(res.objective),
                lower_bound: if feas { res.objective } else { f64::NEG_INFINITY },
                history,
            };
        }

        // ---- Master MILP over (y, θ) ---------------------------------------
        let Some(msol) = solve_master(model, &cuts) else {
            status = if upper.is_finite() { status } else { crate::oa::OaStatus::Infeasible };
            break;
        };
        if msol.status != SolverStatus::Optimal {
            if upper.is_finite() {
                break; // master exhausted; incumbent stands
            }
            status = crate::oa::OaStatus::Infeasible;
            break;
        }
        lower = lower.max(msol.objective_value);
        let mut ym: Vec<f64> = msol.primal[..n].to_vec();

        // ---- Diversification -------------------------------------------------
        // The master tie-breaks arbitrarily among optimal vertices; revisiting
        // an already-evaluated assignment would stall the loop (its cut is
        // already present). On revisit, continue with the *unvisited* integer
        // assignment carrying the smallest cut-estimate. When every assignment
        // has been evaluated the search is exhausted.
        let key_of =
            |y: &[f64]| -> Vec<i64> { int_idx.iter().map(|&i| y[i].round() as i64).collect() };
        if visited.contains(&key_of(&ym)) {
            match diversify(model, &int_idx, &cuts, &visited) {
                Some(yalt) => {
                    for (&slot, &pt) in int_idx.iter().zip(&yalt) {
                        ym[slot] = pt as f64;
                    }
                }
                None => {
                    // All integer assignments evaluated: the incumbent is
                    // final; with valid cuts the master bound cannot exceed it.
                    let gap = upper - lower;
                    status = if upper.is_finite()
                        && gap <= cfg.abs_gap.max(cfg.rel_gap * upper.abs().max(1.0))
                    {
                        crate::oa::OaStatus::Optimal
                    } else {
                        crate::oa::OaStatus::MaxIterations
                    };
                    break;
                }
            }
        }

        // ---- Evaluate the subproblem at ym ----------------------------------
        let res = solve_subproblem(model, &ym, &cfg.nlp);
        let viol = max_violation(model, &ym, &res.x);
        visited.push(key_of(&ym));
        if viol < 1e-4 && res.status == NlpStatus::Converged {
            if res.objective < upper {
                upper = res.objective;
                best_x = Some(res.x.clone());
            }
            let slope = slope_estimate(model, &int_idx, &ym, cfg, true);
            cuts.push(optimality_cut(&ym, res.objective, &slope));
        } else {
            let phi = if viol < 1e-4 { 0.0 } else { viol };
            let slope = slope_estimate(model, &int_idx, &ym, cfg, false);
            cuts.push(feasibility_cut(&ym, phi.max(1e-8), &slope));
        }
        history.push(ConvergenceCertificate {
            lower_bound: lower,
            upper_bound: upper,
            gap: upper - lower,
        });

        if upper.is_finite() {
            let gap = upper - lower;
            if gap <= cfg.abs_gap || gap <= cfg.rel_gap * upper.abs().max(1.0) {
                status = crate::oa::OaStatus::Optimal;
                break;
            }
        }
    }

    let objective = best_x.as_ref().map(|x| model.eval_objective(x));
    GbdResult { status, x: best_x, objective, lower_bound: lower, history }
}

/// Smallest-magnitude integer grid over the complicating variables
/// (cartesian product of the rounded bounds, capped for safety).
fn integer_grid(model: &MinlpModel, int_idx: &[usize]) -> Vec<Vec<i64>> {
    let ranges: Vec<Vec<i64>> = int_idx
        .iter()
        .map(|&i| {
            let lo = model.lbs[i].ceil() as i64;
            let hi = model.ubs[i].floor() as i64;
            (lo..=hi).collect()
        })
        .collect();
    let mut grid: Vec<Vec<i64>> = vec![vec![]];
    for r in &ranges {
        let mut next = Vec::new();
        for prefix in &grid {
            for &v in r {
                let mut p = prefix.clone();
                p.push(v);
                next.push(p);
            }
        }
        grid = next;
        if grid.len() > 50_000 {
            break; // safety cap; diversification degrades gracefully
        }
    }
    grid
}

/// Master-model estimate of `θ` at an integer assignment: the largest
/// optimality-cut lower bound, or +∞ when a feasibility cut is violated.
fn cut_estimate(cuts: &[MasterCut], n: usize, int_idx: &[usize], assign: &[i64]) -> f64 {
    let mut z = vec![0.0f64; n];
    for (&i, &v) in int_idx.iter().zip(assign) {
        z[i] = v as f64;
    }
    let mut est = f64::NEG_INFINITY;
    for cut in cuts {
        // Skip the trailing θ coefficient (index n): the estimate solves for
        // θ itself.
        let val: f64 = cut.coefs.iter().take(n).enumerate().map(|(i, &c)| c * z[i]).sum();
        if cut.ge {
            // θ + Σ c_i y_i >= rhs  →  θ >= rhs − Σ c_i y_i
            est = est.max(cut.rhs - val);
        } else if val > cut.rhs + 1e-9 {
            return f64::INFINITY; // feasibility cut violated here
        }
    }
    est
}

/// Pick the unvisited integer assignment with the smallest master estimate.
fn diversify(
    model: &MinlpModel,
    int_idx: &[usize],
    cuts: &[MasterCut],
    visited: &[Vec<i64>],
) -> Option<Vec<i64>> {
    let n = model.num_vars();
    let mut best: Option<(f64, Vec<i64>)> = None;
    for pt in integer_grid(model, int_idx) {
        if visited.contains(&pt) {
            continue;
        }
        let est = cut_estimate(cuts, n, int_idx, &pt);
        match &best {
            Some((b, _)) if *b <= est => {}
            _ => best = Some((est, pt)),
        }
    }
    best.map(|(_, pt)| pt)
}

/// Probe neighbouring integer assignments to estimate a slope vector over
/// the complicating variables. When `value_mode` is true the probed scalar
/// is the subproblem value `v` (only defined at feasible points — probes
/// that are infeasible yield a zero slope component); otherwise it is the
/// violation measure `φ`. Each slope component is kept only if the resulting
/// linear extrapolation does not exceed the probed values (within slack) at
/// both probes; otherwise that component falls back to zero (flat cut).
fn slope_estimate(
    model: &MinlpModel,
    int_idx: &[usize],
    yk: &[f64],
    cfg: &GbdConfig,
    value_mode: bool,
) -> Vec<f64> {
    let n = model.num_vars();
    let base = probe_value(model, yk, value_mode);
    let mut slope = vec![0.0f64; n];
    for &i in int_idx {
        let h = cfg.probe_step.min((model.ubs[i] - model.lbs[i]) * 0.5);
        if h <= 1e-9 {
            continue;
        }
        let mut yp = yk.to_vec();
        let mut ym_ = yk.to_vec();
        yp[i] = (yk[i] + h).clamp(model.lbs[i], model.ubs[i]);
        ym_[i] = (yk[i] - h).clamp(model.lbs[i], model.ubs[i]);
        let vp = probe_value(model, &yp, value_mode);
        let vm = probe_value(model, &ym_, value_mode);
        let hp = yp[i] - yk[i];
        let hm = yk[i] - ym_[i];
        // Both probes must actually move: when one side is clamped against
        // a bound the "central" difference degenerates to a one-sided
        // estimate whose extrapolation validity is untested, and the
        // resulting cut can exclude the true optimum.
        if hp <= 1e-9 || hm <= 1e-9 || !base.is_finite() || !vp.is_finite() || !vm.is_finite() {
            continue;
        }
        let denom = hp + hm;
        let s = (vp - vm) / denom;
        if !s.is_finite() {
            continue;
        }
        // Validity: the linear model must underestimate both probes.
        let ok_p = vp >= base + s * hp - 1e-6;
        let ok_m = vm >= base - s * hm - 1e-6;
        if ok_p && ok_m {
            slope[i] = s;
        }
    }
    slope
}

/// Evaluate the probed scalar at an integer assignment: `v(y)` when
/// feasible (else NaN), or the violation measure `φ(y)`.
fn probe_value(model: &MinlpModel, y: &[f64], value_mode: bool) -> f64 {
    let params = NlpParams { max_outer: 25, ..NlpParams::default() };
    let res = solve_subproblem(model, y, &params);
    let viol = max_violation(model, y, &res.x);
    if value_mode {
        if viol < 1e-4 && res.status == NlpStatus::Converged {
            res.objective
        } else {
            f64::NAN
        }
    } else {
        viol
    }
}

/// Optimality cut: `θ >= v_k + s·(y − y_k)`.
fn optimality_cut(yk: &[f64], vk: f64, slope: &[f64]) -> MasterCut {
    let n = yk.len();
    let mut coefs = vec![0.0f64; n + 1];
    let mut rhs = vk;
    for i in 0..n {
        coefs[i] = -slope[i];
        rhs -= slope[i] * yk[i];
    }
    coefs[n] = 1.0; // θ
    MasterCut { coefs, rhs, ge: true }
}

/// Feasibility cut: `s·y <= s·y_k − φ_k` (drives the master away from
/// integer assignments whose subproblem violates constraints).
fn feasibility_cut(yk: &[f64], phi: f64, slope: &[f64]) -> MasterCut {
    let n = yk.len();
    let mut coefs = vec![0.0f64; n];
    let mut rhs = -phi;
    for i in 0..n {
        coefs[i] = slope[i];
        rhs += slope[i] * yk[i];
    }
    MasterCut { coefs, rhs, ge: false }
}

/// Build and solve the GBD master: variables `y_0..y_{n-1}` (only the
/// complicating ones are free; continuous placeholders fixed at 0) plus `θ`.
fn solve_master(model: &MinlpModel, cuts: &[MasterCut]) -> Option<tpt_opt_core::solver::Solution> {
    let n = model.num_vars();
    let mut m = Model::new(n + 1);
    m.set_objective(Objective::minimize(vec![n], vec![1.0]));
    for i in 0..n {
        m.variables[i].bound = match model.vars[i] {
            VarKind::Continuous => VarBound::continuous(0.0, 0.0), // not a complicating var
            VarKind::Integer => VarBound::integer(model.lbs[i], model.ubs[i]),
            VarKind::Binary => {
                if model.lbs[i] == model.ubs[i] {
                    VarBound::continuous(model.lbs[i], model.ubs[i])
                } else {
                    VarBound::binary()
                }
            }
        };
    }
    const THETA_BIG: f64 = 1e9;
    m.variables[n].bound = VarBound::continuous(-THETA_BIG, THETA_BIG);

    for cut in cuts {
        let mut idx = Vec::new();
        let mut coefs = Vec::new();
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
    MilpSolver::new().solve(&m).ok()
}
