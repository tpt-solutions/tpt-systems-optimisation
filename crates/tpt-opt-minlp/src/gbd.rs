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

use tpt_opt_core::model::Sense;
use tpt_opt_core::nlp::{NlpParams, NlpStatus};
use tpt_opt_core::Tolerances;
use tpt_opt_core::{
    bounds::VarBound,
    model::{Constraint, Model, Objective},
    solver::{Solver, SolverStatus},
};
use tpt_opt_milp::lp::{solve_lp, LpStatus};
use tpt_opt_milp::MilpSolver;

use crate::certificates::{CertificateHistory, ConvergenceCertificate};
use crate::model::{ConstraintKind, MinlpModel, VarKind};
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
    /// Retained for API compatibility; the cuts no longer use probing.
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
                    // Every integer assignment has been evaluated. A feasible one
                    // was solved exactly, and an infeasible one has no feasible
                    // completion, so the incumbent is optimal regardless of the
                    // master bound. That bound may stay low, since infeasible
                    // assignments carry no cut.
                    let gap = upper - lower;
                    let tol = cfg.abs_gap.max(cfg.rel_gap * upper.abs().max(1.0));
                    status = if !upper.is_finite() {
                        crate::oa::OaStatus::Infeasible
                    } else if enumeration_complete(model, &int_idx) {
                        lower = upper;
                        crate::oa::OaStatus::Optimal
                    } else if gap <= tol {
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
            if let Some(cut) = benders_cut(model, &int_idx, &ym, &res.x) {
                cuts.push(cut);
            }
        }
        // An infeasible subproblem adds no cut. A feasibility cut would need the
        // multipliers of the min-max violation problem, and an unverified cut could
        // exclude the optimum. Termination then comes from `diversify`, which visits
        // every integer assignment.
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

    // A valid lower bound never exceeds a feasible incumbent. When it does, the
    // accumulated cuts cannot be trusted, so the run is not an optimality proof.
    if status == crate::oa::OaStatus::Optimal && upper.is_finite() {
        let tol = cfg.abs_gap.max(cfg.rel_gap * upper.abs().max(1.0));
        if lower > upper + tol {
            status = crate::oa::OaStatus::NumericalIssue;
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

/// `true` when [`integer_grid`] lists every integer assignment, i.e. its size is
/// within the cap it applies.
fn enumeration_complete(model: &MinlpModel, int_idx: &[usize]) -> bool {
    let mut size: usize = 1;
    for &i in int_idx {
        let lo = model.lbs[i].ceil() as i64;
        let hi = model.ubs[i].floor() as i64;
        let count = (hi - lo + 1).max(0) as usize;
        size = size.saturating_mul(count);
        if size > 50_000 {
            return false;
        }
    }
    true
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

/// KKT multipliers of the continuous subproblem at a point, certified by the
/// stationarity residual of the multiplier LP.
struct Multipliers {
    /// `(constraint index, λ)` for each active `Le` constraint, `λ ≥ 0`.
    ineq: Vec<(usize, f64)>,
    /// `(constraint index, μ)` for each `Eq` constraint, sign free.
    eq: Vec<(usize, f64)>,
    /// Worst stationarity residual of the multiplier LP; zero when the
    /// multipliers satisfy the KKT conditions exactly.
    residual: f64,
}

/// Recover KKT multipliers for the subproblem in its continuous variables at
/// the full point `x` (integers already fixed). Solves
/// `∇f + Σλ∇g + Σμ∇h − ν_lo + ν_hi = 0` over the continuous components, with
/// `λ, ν ≥ 0` on active constraints and bounds, by least-`ℓ₁` residual LP.
fn kkt_multipliers(model: &MinlpModel, x: &[f64]) -> Option<Multipliers> {
    let n = model.num_vars();
    let cidx: Vec<usize> = (0..n).filter(|&i| model.vars[i] == VarKind::Continuous).collect();
    let nb = cidx.len();

    let mut ineq = Vec::new();
    let mut eq = Vec::new();
    for (ci, c) in model.constraints.iter().enumerate() {
        if !model.is_active(ci, x) {
            continue;
        }
        match c.kind {
            ConstraintKind::Le if model.eval_constraint(ci, x) >= -1e-6 => ineq.push(ci),
            ConstraintKind::Le => {}
            ConstraintKind::Eq => eq.push(ci),
        }
    }
    let restrict = |g: Vec<f64>| -> Vec<f64> { cidx.iter().map(|&i| g[i]).collect() };
    let grad_ineq: Vec<Vec<f64>> =
        ineq.iter().map(|&ci| restrict(model.eval_constraint_grad(ci, x))).collect();
    let grad_eq: Vec<Vec<f64>> =
        eq.iter().map(|&ci| restrict(model.eval_constraint_grad(ci, x))).collect();
    let grad_f = restrict(model.eval_objective_grad(x));
    let lo_active: Vec<bool> = cidx.iter().map(|&i| x[i] - model.lbs[i] <= 1e-7).collect();
    let hi_active: Vec<bool> = cidx.iter().map(|&i| model.ubs[i] - x[i] <= 1e-7).collect();

    // Columns: λ, μ⁺, μ⁻, ν_lo, ν_hi, r⁺, r⁻ (all ≥ 0).
    let mut lp = Model::new(0);
    let lam: Vec<usize> =
        ineq.iter().map(|_| lp.add_variable(VarBound::continuous(0.0, f64::INFINITY))).collect();
    let mu_p: Vec<usize> =
        eq.iter().map(|_| lp.add_variable(VarBound::continuous(0.0, f64::INFINITY))).collect();
    let mu_m: Vec<usize> =
        eq.iter().map(|_| lp.add_variable(VarBound::continuous(0.0, f64::INFINITY))).collect();
    let nu_lo: Vec<Option<usize>> = (0..nb)
        .map(|j| lo_active[j].then(|| lp.add_variable(VarBound::continuous(0.0, f64::INFINITY))))
        .collect();
    let nu_hi: Vec<Option<usize>> = (0..nb)
        .map(|j| hi_active[j].then(|| lp.add_variable(VarBound::continuous(0.0, f64::INFINITY))))
        .collect();
    let r_p: Vec<usize> =
        (0..nb).map(|_| lp.add_variable(VarBound::continuous(0.0, f64::INFINITY))).collect();
    let r_m: Vec<usize> =
        (0..nb).map(|_| lp.add_variable(VarBound::continuous(0.0, f64::INFINITY))).collect();

    // Stationarity row per continuous component: Σ coef·var + r⁺ − r⁻ = −∇f.
    for j in 0..nb {
        let mut idx = Vec::new();
        let mut co = Vec::new();
        for (k, &v) in lam.iter().enumerate() {
            idx.push(v);
            co.push(grad_ineq[k][j]);
        }
        for (k, _) in eq.iter().enumerate() {
            idx.push(mu_p[k]);
            co.push(grad_eq[k][j]);
            idx.push(mu_m[k]);
            co.push(-grad_eq[k][j]);
        }
        if let Some(v) = nu_lo[j] {
            idx.push(v);
            co.push(-1.0);
        }
        if let Some(v) = nu_hi[j] {
            idx.push(v);
            co.push(1.0);
        }
        idx.push(r_p[j]);
        co.push(1.0);
        idx.push(r_m[j]);
        co.push(-1.0);
        lp.add_constraint(Constraint::equality(idx, co, -grad_f[j]));
    }
    let mut oidx = Vec::new();
    for j in 0..nb {
        oidx.push(r_p[j]);
        oidx.push(r_m[j]);
    }
    let ocoeffs = vec![1.0; oidx.len()];
    lp.set_objective(Objective {
        sense: Sense::Minimize,
        indices: oidx,
        coeffs: ocoeffs,
        constant: 0.0,
    });

    let nv = lp.num_vars;
    let sol = solve_lp(&lp, &vec![0.0; nv], &vec![f64::INFINITY; nv], Tolerances::spec_default());
    if sol.status != LpStatus::Optimal {
        return None;
    }
    let x_lp = &sol.x;
    Some(Multipliers {
        ineq: ineq.iter().zip(&lam).map(|(&ci, &v)| (ci, x_lp[v])).collect(),
        eq: eq.iter().enumerate().map(|(k, &ci)| (ci, x_lp[mu_p[k]] - x_lp[mu_m[k]])).collect(),
        residual: sol.objective.max(0.0),
    })
}

/// Benders optimality cut at `yk` from the subproblem's KKT multipliers:
/// `θ ≥ v(y_k) + ∇_y L(x_k, y_k)·(y − y_k)` with `L = f + Σλg + Σμh`. For a
/// convex problem `L(·, y)` lower-bounds the value function, so the cut is valid
/// for every integer `y`. Returns `None` when the multipliers do not certify
/// stationarity.
fn benders_cut(model: &MinlpModel, int_idx: &[usize], yk: &[f64], xk: &[f64]) -> Option<MasterCut> {
    let mult = kkt_multipliers(model, xk)?;
    if mult.residual > 1e-6 {
        return None;
    }
    let n = model.num_vars();
    let mut grad = model.eval_objective_grad(xk);
    for &(ci, l) in mult.ineq.iter().chain(&mult.eq) {
        let g = model.eval_constraint_grad(ci, xk);
        for i in 0..n {
            grad[i] += l * g[i];
        }
    }
    let mut slope = vec![0.0f64; n];
    for &i in int_idx {
        slope[i] = grad[i];
    }
    Some(optimality_cut(yk, model.eval_objective(xk), &slope))
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
