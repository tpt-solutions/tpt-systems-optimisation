//! Gauss-Newton, Levenberg-Marquardt, and Powell dogleg solvers.
//!
//! All three share one linearisation helper (residual + exact-Jacobian
//! evaluation, IRLS loss weighting, normal equations) and differ only in how
//! they pick the step:
//!
//! * **Gauss-Newton** — solve `H δ = −g` (tiny Tikhonov floor for PD safety)
//!   and backtrack until the cost decreases. Fast near the solution; can
//!   diverge from poor starts.
//! * **Levenberg-Marquardt** — solve `(H + λ diag H) δ = −g` with the
//!   Nielsen gain-ratio scheme adapting λ every step. The robust default.
//! * **Powell dogleg** — trust region blending the steepest-descent (Cauchy)
//!   step and the Gauss-Newton step; handles poor starts without a damping
//!   parameter.
//!
//! Bounds are enforced by **projection**: the trial point `x + tδ` is
//! clamped into the box; the (reflective-transform) variant remains future
//! work. Robust losses are applied as IRLS: rows are reweighted by
//! `wᵢ = ρ'(rᵢ²)` each outer iteration, so each step solves a weighted
//! least-squares problem while the reported cost is `½ Σ ρ(rᵢ²)`.
//!
//! # Example
//!
//! ```rust
//! use tpt_opt_nls::{levenberg_marquardt, NlsConfig, NlsProblem};
//!
//! let problem = NlsProblem::builder(2)
//!     .residual_block(2, |x, out| {
//!         out[0] = 1.0 - x[0].clone();
//!         out[1] = 10.0 * (x[1].clone() - x[0].clone() * x[0].clone());
//!     })
//!     .build();
//! let result = levenberg_marquardt(&problem, &[-1.2, 1.0], &NlsConfig::new()).unwrap();
//! assert_eq!(result.status, tpt_opt_nls::NlsStatus::Converged);
//! assert!((result.x[0] - 1.0).abs() < 1e-8);
//! assert!((result.x[1] - 1.0).abs() < 1e-8);
//! ```

use crate::dense;
use crate::loss::Loss;
use crate::problem::{LinearSolver, NlsProblem};
use crate::sparse::{self, Ordering, SparseError, SymbolicLdl, Triplet};
use crate::types::{NlsConfig, NlsError, NlsResult, NlsStatus};

/// One linearisation of the problem at a point.
struct Linearization {
    /// Raw residual (`m`).
    r: Vec<f64>,
    /// Raw row-major Jacobian (`m × n`).
    j: Vec<f64>,
    /// IRLS weights `wᵢ = ρ'(rᵢ²)` (`m`).
    w: Vec<f64>,
    /// Weighted cost `½ Σ ρ(rᵢ²)`.
    cost: f64,
    /// Weighted normal equations `H = Jᵀ W J` (`n × n` row-major).
    h: Vec<f64>,
    /// Weighted gradient `g = Jᵀ W r` (`n`).
    g: Vec<f64>,
    /// Weighted residual `√W ⊙ r` (for gradient reporting).
    _wr: Vec<f64>,
}

/// Evaluate residuals, Jacobian, weights, normal equations, and gradient.
fn linearize(problem: &NlsProblem, x: &[f64], cfg: &NlsConfig) -> Linearization {
    let n = problem.n();
    let m = problem.n_residuals();
    let mut r = vec![0.0f64; m];
    let mut j = vec![0.0f64; m * n];
    problem.residual_and_jacobian(x, cfg.jacobian, &mut r, &mut j);
    let w: Vec<f64> = r.iter().map(|&ri| cfg.loss.weight(ri)).collect();
    let cost: f64 = r.iter().map(|&ri| cfg.loss.rho(ri * ri)).sum::<f64>() * 0.5;
    let h = weighted_normal_equations(&j, &w, m, n);
    let g: Vec<f64> =
        (0..n).map(|k| (0..m).map(|i| j[i * n + k] * w[i] * r[i]).sum::<f64>()).collect();
    let wr: Vec<f64> = r.iter().zip(&w).map(|(&ri, &wi)| ri * wi.sqrt()).collect();
    Linearization { r, j, w, cost, h, g, _wr: wr }
}

/// `H = Jᵀ W J` with per-row weights (row-major `n × n`).
fn weighted_normal_equations(j: &[f64], w: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut h = vec![0.0f64; n * n];
    for i in 0..m {
        let wi = w[i];
        if wi == 0.0 {
            continue;
        }
        let jr = &j[i * n..(i + 1) * n];
        for (a, &jra) in jr.iter().enumerate() {
            let va = jra * wi;
            if va == 0.0 {
                continue;
            }
            for (b, &jrb) in jr.iter().enumerate().skip(a) {
                h[a * n + b] += va * jrb;
            }
        }
    }
    for a in 0..n {
        for b in 0..a {
            h[a * n + b] = h[b * n + a];
        }
    }
    h
}

/// Upper-triangle triplets of `H = Jᵀ W J` (structural sparsity of `J`).
fn h_triplets(j: &[f64], w: &[f64], m: usize, n: usize) -> Vec<Triplet> {
    let mut out = Vec::new();
    for i in 0..m {
        let wi = w[i];
        if wi == 0.0 {
            continue;
        }
        let jr = &j[i * n..(i + 1) * n];
        for (a, &va) in jr.iter().enumerate() {
            if va == 0.0 {
                continue;
            }
            for (b, &vb) in jr.iter().enumerate().skip(a) {
                if vb != 0.0 {
                    out.push(Triplet::new(a, b, wi * va * vb));
                }
            }
        }
    }
    out
}

/// Diagonal of `H` with a small floor so the Marquardt matrix is positive.
fn marquardt_diagonal(h: &[f64], n: usize) -> Vec<f64> {
    (0..n).map(|k| h[k * n + k].abs().max(1e-6)).collect()
}

/// Cache for the sparse path: symbolic analysis reused across iterations,
/// re-analysed if the structural pattern shifts.
struct SparseCache {
    sym: Option<SymbolicLdl>,
    ordering: Ordering,
}

impl SparseCache {
    fn new() -> Self {
        Self { sym: None, ordering: Ordering::MinDegree }
    }

    /// Factor `H + λ·diag_H` (as triplets) and solve for `−g`; returns
    /// `None` if the factorisation fails (non-PD).
    fn solve_step(
        &mut self,
        h_trip: &[Triplet],
        diag_h: &[f64],
        lambda: f64,
        g: &[f64],
        n: usize,
    ) -> Option<Vec<f64>> {
        let mut damped = h_trip.to_vec();
        damped.extend(
            diag_h.iter().take(n).enumerate().map(|(k, &d)| Triplet::new(k, k, lambda * d)),
        );
        for attempt in 0..2 {
            if self.sym.is_none() {
                self.sym = Some(sparse::analyze(n, &damped, self.ordering));
            }
            let sym = self.sym.as_ref().expect("symbolic present");
            match sym.factorize(&damped) {
                Ok(ldl) => {
                    let minus_g: Vec<f64> = g.iter().map(|&v| -v).collect();
                    return Some(ldl.solve(&minus_g));
                }
                Err(SparseError::NotPositiveDefinite) => return None,
                // Pattern drifted (a structural zero became nonzero):
                // re-analyse once and retry.
                Err(SparseError::PatternMismatch) if attempt == 0 => {
                    self.sym = None;
                }
                Err(_) => return None,
            }
        }
        None
    }
}

/// Dense solve of `(H + λ·diag_H) δ = −g` with a Tikhonov floor; `None` on
/// failure.
fn solve_step_dense(
    h: &[f64],
    diag_h: &[f64],
    lambda: f64,
    g: &[f64],
    n: usize,
) -> Option<Vec<f64>> {
    let mut a = h.to_vec();
    for k in 0..n {
        a[k * n + k] += lambda * diag_h[k] + 1e-12;
    }
    let l = dense::cholesky_lower(&a, n)?;
    let rhs: Vec<f64> = g.iter().map(|&v| -v).collect();
    Some(dense::cho_solve_lower(&l, &rhs, n))
}

/// Shared context for one solve run.
struct Run<'a> {
    problem: &'a NlsProblem,
    cfg: &'a NlsConfig,
    evals: usize,
    sparse: SparseCache,
}

impl<'a> Run<'a> {
    fn new(problem: &'a NlsProblem, cfg: &'a NlsConfig) -> Result<Self, NlsError> {
        problem.validate()?;
        if problem.n_residuals() == 0 {
            return Err(NlsError::InvalidProblem("problem has no residuals".into()));
        }
        Ok(Self { problem, cfg, evals: 0, sparse: SparseCache::new() })
    }

    fn lin(&mut self, x: &[f64]) -> Linearization {
        self.evals += 1;
        linearize(self.problem, x, self.cfg)
    }

    /// Compute the step for damping `lambda` over the masked (free)
    /// subspace, dispatching dense/sparse.
    fn step(&mut self, lambda: f64, mask: &Masked) -> Option<Vec<f64>> {
        let n = self.problem.n();
        match (&mask.trip, self.cfg.linear_solver) {
            (Some(trip), LinearSolver::Sparse) => {
                let diag_h = marquardt_diagonal(&mask.h, n);
                self.sparse.solve_step(trip, &diag_h, lambda, &mask.g, n)
            }
            _ => {
                let diag_h = marquardt_diagonal(&mask.h, n);
                solve_step_dense(&mask.h, &diag_h, lambda, &mask.g, n)
            }
        }
    }

    fn out_of_budget(&self, iterations: usize) -> bool {
        iterations >= self.cfg.max_iter || self.evals >= self.cfg.max_evals
    }
}

/// Linearisation restricted to the free subspace: coordinates blocked at a
/// bound (sitting on the wall with the gradient pushing outward) get
/// `g_k = 0`, decoupled rows/columns, and `H_kk = 1`, so the solve produces
/// an exact zero step there and the free coordinates are uncoupled from the
/// walls. This is projected-Newton on the active set.
struct Masked {
    h: Vec<f64>,
    g: Vec<f64>,
    trip: Option<Vec<Triplet>>,
}

/// Coordinates that cannot move without increasing the cost: at a finite
/// bound with the gradient pointing outward (minimisation sense).
fn blocked_coords(problem: &NlsProblem, x: &[f64], g: &[f64]) -> Vec<bool> {
    if !problem.is_bounded() {
        return vec![false; x.len()];
    }
    (0..x.len())
        .map(|k| {
            let (lo, hi) = problem.bounds()[k];
            (x[k] <= lo && g[k] > 0.0) || (x[k] >= hi && g[k] < 0.0)
        })
        .collect()
}

fn make_masked(lin: &Linearization, blocked: &[bool], h_trip: &Option<Vec<Triplet>>) -> Masked {
    let n = blocked.len();
    let mut h = lin.h.clone();
    let mut g = lin.g.clone();
    for (k, &bl) in blocked.iter().enumerate() {
        if !bl {
            continue;
        }
        g[k] = 0.0;
        for j in 0..n {
            h[k * n + j] = 0.0;
            h[j * n + k] = 0.0;
        }
        h[k * n + k] = 1.0;
    }

    let trip = h_trip.as_ref().map(|trip| {
        let mut filtered: Vec<Triplet> =
            trip.iter().copied().filter(|t| !blocked[t.row] && !blocked[t.col]).collect();
        for (k, &bl) in blocked.iter().enumerate() {
            if bl {
                filtered.push(Triplet::new(k, k, 1.0));
            }
        }
        filtered
    });
    Masked { h, g, trip }
}

/// Robust cost `½ Σ ρ(r(x)²)` at `x + t·δ`, clamped into the bounds;
/// returns `(cost, trial_point)`. Comparing like-for-like robust costs (not
/// raw `½‖r‖²`) keeps gain ratios meaningful under M-estimator losses.
fn projected_cost(
    problem: &NlsProblem,
    x: &[f64],
    delta: &[f64],
    t: f64,
    loss: Loss,
) -> (f64, Vec<f64>) {
    let mut trial = x.to_vec();
    for (xt, d) in trial.iter_mut().zip(delta.iter()) {
        *xt += t * d;
    }
    problem.project(&mut trial);
    (problem.loss_cost(&trial, loss), trial)
}

/// Zero step components that would push a parameter outward through a bound
/// it is sitting on (projected-Newton style). Without this, a damped step
/// dominated by a blocked coordinate can dead-lock projected solvers.
fn project_step(problem: &NlsProblem, x: &[f64], delta: &mut [f64]) {
    if !problem.is_bounded() {
        return;
    }
    for (k, d) in delta.iter_mut().enumerate() {
        let (lo, hi) = problem.bounds()[k];
        if (x[k] <= lo && *d < 0.0) || (x[k] >= hi && *d > 0.0) {
            *d = 0.0;
        }
    }
}

fn infinity_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |a, &b| a.max(b.abs()))
}

fn result(
    status: NlsStatus,
    termination: &'static str,
    lin: &Linearization,
    x: Vec<f64>,
    iterations: usize,
    evals: usize,
) -> NlsResult {
    let gradient_norm = {
        let n = x.len();
        (0..n)
            .map(|k| {
                (0..lin.r.len()).map(|i| lin.j[i * n + k] * lin.w[i] * lin.r[i]).sum::<f64>().abs()
            })
            .fold(0.0f64, f64::max)
    };
    NlsResult {
        status,
        x,
        residual: lin.r.clone(),
        cost: lin.cost,
        iterations,
        evaluations: evals,
        gradient_norm,
        termination,
        jacobian: Some(lin.j.clone()),
    }
}

/// Solve with the Gauss-Newton method plus backtracking line search.
pub fn gauss_newton(
    problem: &NlsProblem,
    x0: &[f64],
    cfg: &NlsConfig,
) -> Result<NlsResult, NlsError> {
    let mut run = Run::new(problem, cfg)?;
    let mut x = x0.to_vec();
    if x.len() != problem.n() {
        return Err(NlsError::InvalidProblem("x0 length must equal parameter count".into()));
    }
    problem.project(&mut x);
    let mut lin = run.lin(&x);
    let mut blocked = blocked_coords(problem, &x, &lin.g);
    let mut mask = make_masked(&lin, &blocked, &h_triplets_for(&run, &lin));
    if infinity_norm(&mask.g) <= cfg.gtol {
        return Ok(result(
            NlsStatus::Converged,
            "gradient tolerance met at start",
            &lin,
            x,
            0,
            run.evals,
        ));
    }
    let mut iterations = 0usize;
    while iterations < cfg.max_iter {
        if run.out_of_budget(iterations) {
            return Ok(result(
                NlsStatus::MaxIterations,
                "iteration/evaluation budget",
                &lin,
                x,
                iterations,
                run.evals,
            ));
        }
        // Step with a small regularisation floor; fall back to a damped step
        // if the plain normal equations are singular.
        let mut delta = match run.step(0.0, &mask) {
            Some(d) => d,
            None => match run.step(1e-6, &mask) {
                Some(d) => d,
                None => {
                    return Ok(result(
                        NlsStatus::NumericalIssue,
                        "singular normal equations",
                        &lin,
                        x,
                        iterations,
                        run.evals,
                    ))
                }
            },
        };
        project_step(problem, &x, &mut delta);
        if infinity_norm(&delta) <= cfg.xtol * (infinity_norm(&x) + cfg.xtol) {
            return Ok(result(
                NlsStatus::StepTolerance,
                "step size tolerance",
                &lin,
                x,
                iterations,
                run.evals,
            ));
        }
        // Backtracking line search with projection.
        let mut t = 1.0;
        let mut accepted = false;
        for _ in 0..40 {
            let (c, trial) = projected_cost(problem, &x, &delta, t, cfg.loss);
            if c < lin.cost {
                let new_lin = run.lin(&trial);
                x = trial;
                lin = new_lin;
                accepted = true;
                break;
            }
            t *= 0.5;
        }
        if !accepted {
            return Ok(result(
                NlsStatus::CostTolerance,
                "line search stalled",
                &lin,
                x,
                iterations,
                run.evals,
            ));
        }
        iterations += 1;
        blocked = blocked_coords(problem, &x, &lin.g);
        mask = make_masked(&lin, &blocked, &h_triplets_for(&run, &lin));
        if infinity_norm(&mask.g) <= cfg.gtol {
            return Ok(result(
                NlsStatus::Converged,
                "gradient tolerance",
                &lin,
                x,
                iterations,
                run.evals,
            ));
        }
    }
    Ok(result(NlsStatus::MaxIterations, "max iterations", &lin, x, iterations, run.evals))
}

/// Assemble H triplets only when the sparse path is configured.
fn h_triplets_for(run: &Run<'_>, lin: &Linearization) -> Option<Vec<Triplet>> {
    match run.cfg.linear_solver {
        LinearSolver::Sparse => {
            let n = run.problem.n();
            Some(h_triplets(&lin.j, &lin.w, lin.r.len(), n))
        }
        LinearSolver::Dense => None,
    }
}

/// Solve with the Levenberg-Marquardt algorithm (Nielsen damping update).
pub fn levenberg_marquardt(
    problem: &NlsProblem,
    x0: &[f64],
    cfg: &NlsConfig,
) -> Result<NlsResult, NlsError> {
    let mut run = Run::new(problem, cfg)?;
    let mut x = x0.to_vec();
    if x.len() != problem.n() {
        return Err(NlsError::InvalidProblem("x0 length must equal parameter count".into()));
    }
    problem.project(&mut x);
    let mut lin = run.lin(&x);
    let mut blocked = blocked_coords(problem, &x, &lin.g);
    let mut mask = make_masked(&lin, &blocked, &h_triplets_for(&run, &lin));
    if infinity_norm(&mask.g) <= cfg.gtol {
        return Ok(result(
            NlsStatus::Converged,
            "gradient tolerance met at start",
            &lin,
            x,
            0,
            run.evals,
        ));
    }
    let n = problem.n();
    // Nielsen-recommended initial damping: λ₀ scaled by the average normal
    // curvature, so degenerate starts (a zero Jacobian column, e.g. Beale
    // from (1, 1)) do not produce a wild first step.
    let trace: f64 = (0..n).map(|k| lin.h[k * n + k]).sum();
    let mut lambda = (cfg.lambda0 * trace / n as f64).max(cfg.lambda0).min(1e12);
    let mut nu = 2.0;
    let mut iterations = 0usize;
    loop {
        if run.out_of_budget(iterations) {
            return Ok(result(
                NlsStatus::MaxIterations,
                "iteration/evaluation budget",
                &lin,
                x,
                iterations,
                run.evals,
            ));
        }
        // Damped step over the free subspace.
        let mut delta = match run.step(lambda, &mask) {
            Some(d) => d,
            None => {
                // Increase damping and retry within this iteration.
                lambda *= nu;
                nu *= 2.0;
                if lambda > 1e12 {
                    return Ok(result(
                        NlsStatus::NumericalIssue,
                        "damping exhausted",
                        &lin,
                        x,
                        iterations,
                        run.evals,
                    ));
                }
                continue;
            }
        };
        project_step(problem, &x, &mut delta);
        let step_norm = {
            let sq: f64 = delta.iter().map(|&v| v * v).sum();
            sq.sqrt()
        };
        if step_norm <= cfg.xtol * (infinity_norm(&x) + cfg.xtol) {
            return Ok(result(
                NlsStatus::StepTolerance,
                "step size tolerance",
                &lin,
                x,
                iterations,
                run.evals,
            ));
        }
        // Trial point and gain ratio against the free-subspace quadratic
        // model: pred = −(gᵀδ + ½ δᵀHδ) > 0 for a descent step.
        let (trial_cost, trial) = projected_cost(problem, &x, &delta, 1.0, cfg.loss);
        let gd: f64 = mask.g.iter().zip(&delta).map(|(a, b)| a * b).sum();
        let hd: f64 = {
            let mut s = 0.0;
            for (i, di) in delta.iter().enumerate() {
                for (k, dk) in delta.iter().enumerate() {
                    s += di * mask.h[i * n + k] * dk;
                }
            }
            0.5 * s
        };
        let pred = -(gd + hd);
        let rho =
            if trial_cost < lin.cost && pred > 0.0 { (lin.cost - trial_cost) / pred } else { -1.0 };
        if rho > 0.0 {
            // Accept; decrease damping (Nielsen).
            let new_lin = run.lin(&trial);
            x = trial;
            lin = new_lin;
            iterations += 1;
            lambda *= (1.0 - (2.0 * rho - 1.0).powi(3)).max(1.0 / 3.0);
            lambda = lambda.max(1e-12);
            nu = 2.0;
            blocked = blocked_coords(problem, &x, &lin.g);
            mask = make_masked(&lin, &blocked, &h_triplets_for(&run, &lin));
            if infinity_norm(&mask.g) <= cfg.gtol {
                return Ok(result(
                    NlsStatus::Converged,
                    "gradient tolerance",
                    &lin,
                    x,
                    iterations,
                    run.evals,
                ));
            }
            if pred <= cfg.ftol * lin.cost.max(1e-300) {
                return Ok(result(
                    NlsStatus::CostTolerance,
                    "cost change tolerance",
                    &lin,
                    x,
                    iterations,
                    run.evals,
                ));
            }
        } else {
            // Reject; increase damping.
            lambda *= nu;
            nu *= 2.0;
            if lambda > 1e12 {
                return Ok(result(
                    NlsStatus::CostTolerance,
                    "damping exhausted",
                    &lin,
                    x,
                    iterations,
                    run.evals,
                ));
            }
        }
    }
}

/// Solve with the Powell dogleg trust-region method (projected).
pub fn powell_dogleg(
    problem: &NlsProblem,
    x0: &[f64],
    cfg: &NlsConfig,
) -> Result<NlsResult, NlsError> {
    let mut run = Run::new(problem, cfg)?;
    let mut x = x0.to_vec();
    if x.len() != problem.n() {
        return Err(NlsError::InvalidProblem("x0 length must equal parameter count".into()));
    }
    problem.project(&mut x);
    let mut lin = run.lin(&x);
    let mut blocked = blocked_coords(problem, &x, &lin.g);
    let mut mask = make_masked(&lin, &blocked, &h_triplets_for(&run, &lin));
    if infinity_norm(&mask.g) <= cfg.gtol {
        return Ok(result(
            NlsStatus::Converged,
            "gradient tolerance met at start",
            &lin,
            x,
            0,
            run.evals,
        ));
    }
    let n = problem.n();
    let mut delta_tr = 1.0;
    let mut iterations = 0usize;
    loop {
        if run.out_of_budget(iterations) {
            return Ok(result(
                NlsStatus::MaxIterations,
                "iteration/evaluation budget",
                &lin,
                x,
                iterations,
                run.evals,
            ));
        }
        let gnorm = {
            let sq: f64 = mask.g.iter().map(|&v| v * v).sum();
            sq.sqrt()
        };
        // Cauchy / steepest-descent step within the trust region.
        let gg: f64 = mask.g.iter().map(|&v| v * v).sum();
        let hg = dense::mat_vec(&mask.h, n, n, &mask.g);
        let ghg: f64 = mask.g.iter().zip(&hg).map(|(a, b)| a * b).sum();
        let cauchy = if ghg > 1e-300 { gg / ghg } else { 1.0 };
        let mut pu: Vec<f64> = mask.g.iter().map(|&v| -cauchy * v).collect();
        let mut pu_norm = {
            let sq: f64 = pu.iter().map(|&v| v * v).sum();
            sq.sqrt()
        };
        if ghg <= 1e-300 || pu_norm <= 1e-300 {
            // Degenerate curvature: head straight down the gradient.
            let inv = if gnorm > 0.0 { delta_tr / gnorm } else { 0.0 };
            pu = mask.g.iter().map(|&v| -inv * v).collect();
            pu_norm = {
                let sq: f64 = pu.iter().map(|&v| v * v).sum();
                sq.sqrt()
            };
        }
        // Gauss-Newton step (lightly floored for PD safety).
        let pn = run.step(1e-12, &mask).unwrap_or_else(|| pu.clone());
        let pn_norm = {
            let sq: f64 = pn.iter().map(|&v| v * v).sum();
            sq.sqrt()
        };
        // Blend onto the dogleg path.
        let delta = if pn_norm <= delta_tr {
            pn
        } else if pu_norm >= delta_tr {
            let scale = delta_tr / pu_norm.max(1e-300);
            pu.iter().map(|&v| scale * v).collect()
        } else {
            let diff: Vec<f64> = pn.iter().zip(&pu).map(|(a, b)| a - b).collect();
            let diff_norm = {
                let sq: f64 = diff.iter().map(|&v| v * v).sum();
                sq.sqrt()
            };
            let t = (delta_tr - pu_norm) / diff_norm.max(1e-300);
            pu.iter().zip(&diff).map(|(a, b)| a + t * b).collect()
        };
        let mut delta = delta;
        project_step(problem, &x, &mut delta);
        let step_norm = {
            let sq: f64 = delta.iter().map(|&v| v * v).sum();
            sq.sqrt()
        };
        if step_norm <= cfg.xtol * (infinity_norm(&x) + cfg.xtol) {
            return Ok(result(
                NlsStatus::StepTolerance,
                "step size tolerance",
                &lin,
                x,
                iterations,
                run.evals,
            ));
        }
        let (trial_cost, trial) = projected_cost(problem, &x, &delta, 1.0, cfg.loss);
        let gd: f64 = mask.g.iter().zip(&delta).map(|(a, b)| a * b).sum();
        let hg_d = dense::mat_vec(&mask.h, n, n, &delta);
        let hd: f64 = delta.iter().zip(&hg_d).map(|(a, b)| a * b).sum();
        let pred = (-(gd + hd)).max(1e-300);
        let rho = (lin.cost - trial_cost) / pred;
        if rho > 1e-4 {
            let new_lin = run.lin(&trial);
            x = trial;
            lin = new_lin;
            iterations += 1;
            blocked = blocked_coords(problem, &x, &lin.g);
            mask = make_masked(&lin, &blocked, &h_triplets_for(&run, &lin));
            if rho > 0.75 && (step_norm - delta_tr).abs() < 1e-12 {
                delta_tr = (delta_tr * 2.0).min(1e12);
            }
            if infinity_norm(&mask.g) <= cfg.gtol {
                return Ok(result(
                    NlsStatus::Converged,
                    "gradient tolerance",
                    &lin,
                    x,
                    iterations,
                    run.evals,
                ));
            }
        } else {
            delta_tr *= 0.25;
            if delta_tr < 1e-14 {
                return Ok(result(
                    NlsStatus::CostTolerance,
                    "trust region collapsed",
                    &lin,
                    x,
                    iterations,
                    run.evals,
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::Loss;
    use crate::problem::{JacobianMode, NlsProblem};

    fn rosenbrock() -> NlsProblem {
        NlsProblem::builder(2)
            .residual_block(2, |x, out| {
                out[0] = 1.0 - x[0].clone();
                out[1] = 10.0 * (x[1].clone() - x[0].clone() * x[0].clone());
            })
            .build()
    }

    fn powell_singular() -> NlsProblem {
        NlsProblem::builder(4)
            .residual_block(4, |x, out| {
                out[0] = x[0].clone() + 10.0 * x[1].clone();
                out[1] = 5.0_f64.sqrt() * (x[2].clone() - x[3].clone());
                out[2] = {
                    let d = x[1].clone() - 2.0 * x[2].clone();
                    d.clone() * d
                };
                out[3] = {
                    let d = x[0].clone() - x[3].clone();
                    10.0_f64.sqrt() * d.clone() * d
                };
            })
            .build()
    }

    fn assert_near(actual: &[f64], expect: &[f64], tol: f64) {
        for (a, e) in actual.iter().zip(expect.iter()) {
            assert!((a - e).abs() < tol, "got {a}, want {e}");
        }
    }

    #[test]
    fn lm_solves_rosenbrock() {
        let p = rosenbrock();
        let res = levenberg_marquardt(&p, &[-1.2, 1.0], &NlsConfig::new()).unwrap();
        assert_eq!(res.status, NlsStatus::Converged);
        assert_near(&res.x, &[1.0, 1.0], 1e-7);
        assert!(res.cost < 1e-20);
    }

    #[test]
    fn gn_and_dogleg_solve_rosenbrock() {
        let p = rosenbrock();
        for (name, res) in [
            ("gn", gauss_newton(&p, &[0.0, 0.0], &NlsConfig::new()).unwrap()),
            ("dogleg", powell_dogleg(&p, &[-1.2, 1.0], &NlsConfig::new()).unwrap()),
        ] {
            assert!(res.status.has_solution(), "{name}: {:?} ({})", res.status, res.termination);
            assert_near(&res.x, &[1.0, 1.0], 1e-5);
        }
    }

    #[test]
    fn all_solvers_handle_powell_singular() {
        let p = powell_singular();
        for (name, res) in [
            ("lm", levenberg_marquardt(&p, &[3.0, -1.0, 0.0, 1.0], &NlsConfig::new()).unwrap()),
            ("gn", gauss_newton(&p, &[3.0, -1.0, 0.0, 1.0], &NlsConfig::new()).unwrap()),
            ("dogleg", powell_dogleg(&p, &[3.0, -1.0, 0.0, 1.0], &NlsConfig::new()).unwrap()),
        ] {
            assert!(res.status.has_solution(), "{name}: {:?}", res.status);
            assert!(res.cost < 1e-12, "{name} cost = {}", res.cost);
        }
    }

    #[test]
    fn bounds_push_optimum_to_boundary() {
        // min (1−x)² + 100(y − x²)²  s.t.  x ≥ 1.5 → optimum (1.5, 2.25).
        let p = NlsProblem::builder(2)
            .residual_block(2, |x, out| {
                out[0] = 1.0 - x[0].clone();
                out[1] = 10.0 * (x[1].clone() - x[0].clone() * x[0].clone());
            })
            .with_bound(0, 1.5, f64::INFINITY)
            .build();
        let res = levenberg_marquardt(&p, &[4.0, 4.0], &NlsConfig::new()).unwrap();
        assert!(res.status.has_solution(), "{:?}", res.status);
        assert_near(&res.x, &[1.5, 2.25], 1e-5);
        assert!((res.cost - 0.125).abs() < 1e-8, "cost = {}", res.cost);
    }

    #[test]
    fn huber_rejects_outlier() {
        // Data y = 2x + 1 with one gross outlier at x = 4. The L2 fit is
        // dragged toward the outlier; the Huber fit recovers a line close to
        // the clean data. With delta = 0.5 the exact Huber optimum balances
        // the clean-row gradient [2, 0.5] against the outlier's linear-tail
        // contribution 0.5·[4, 1], i.e. r = (−0.25, 0, 0.25, 0.5) on the
        // clean points: (a, b) = (2.25, 0.75).
        let data = [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0), (3.0, 7.0), (4.0, 100.0)];
        let build = || {
            NlsProblem::builder(2)
                .residual_block(data.len(), move |x, out| {
                    for (i, &(xi, yi)) in data.iter().enumerate() {
                        out[i] = x[0].clone() * xi + x[1].clone() - yi;
                    }
                })
                .build()
        };
        let clean = [2.0, 1.0];
        let l2 = levenberg_marquardt(&build(), &[0.0, 0.0], &NlsConfig::new()).unwrap();
        let huber = levenberg_marquardt(
            &build(),
            &[0.0, 0.0],
            &NlsConfig::new().with_loss(Loss::Huber { delta: 0.5 }),
        )
        .unwrap();
        let err = |p: &[f64]| (p[0] - clean[0]).abs() + (p[1] - clean[1]).abs();
        assert!(err(&huber.x) < err(&l2.x), "huber {:?} should beat l2 {:?}", huber.x, l2.x);
        assert_near(&huber.x, &[2.25, 0.75], 1e-3);
        // Tightening delta to 0.2 pulls the fit further from the outlier:
        // the exact optimum is (2.1, 0.9) — err 0.2 vs 0.5 at delta = 0.5.
        let tight = levenberg_marquardt(
            &build(),
            &[0.0, 0.0],
            &NlsConfig::new().with_loss(Loss::Huber { delta: 0.2 }),
        )
        .unwrap();
        assert!(err(&tight.x) < err(&huber.x), "tight huber params {:?}", tight.x);
        assert_near(&tight.x, &[2.1, 0.9], 1e-3);
    }

    #[test]
    fn sparse_and_dense_paths_agree() {
        let p = powell_singular();
        let dense = levenberg_marquardt(
            &p,
            &[3.0, -1.0, 0.0, 1.0],
            &NlsConfig::new().with_linear_solver(LinearSolver::Dense),
        )
        .unwrap();
        let sparse = levenberg_marquardt(
            &p,
            &[3.0, -1.0, 0.0, 1.0],
            &NlsConfig::new().with_linear_solver(LinearSolver::Sparse),
        )
        .unwrap();
        assert!(dense.status.has_solution() && sparse.status.has_solution());
        assert!(dense.cost < 1e-8 && sparse.cost < 1e-8);
        assert_near(&sparse.x, &dense.x, 1e-3);
    }

    #[test]
    fn fd_jacobian_mode_still_converges() {
        let p = rosenbrock();
        let res = levenberg_marquardt(
            &p,
            &[-1.2, 1.0],
            &NlsConfig::new().with_jacobian(JacobianMode::FiniteDifference),
        )
        .unwrap();
        assert!(res.status.has_solution());
        assert_near(&res.x, &[1.0, 1.0], 1e-4);
    }

    #[test]
    fn covariance_recovers_noise_scale() {
        // Linear problem y = a·x with known noise σ = 0.1: σ̂² should land
        // near 0.01 and cov(a) near σ²/Σx².
        let data: Vec<(f64, f64)> = (0..20)
            .map(|i| {
                let x = i as f64;
                (x, 3.0 * x + 0.1 * ((i * 7) % 13 - 6) as f64 / 6.0)
            })
            .collect();
        let p = NlsProblem::builder(1)
            .residual_block(data.len(), move |x, out| {
                for (i, &(xi, yi)) in data.iter().enumerate() {
                    out[i] = x[0].clone() * xi - yi;
                }
            })
            .build();
        let res = levenberg_marquardt(&p, &[0.0], &NlsConfig::new()).unwrap();
        assert_eq!(res.status, NlsStatus::Converged);
        // The deterministic "noise" is not zero-mean, so the least-squares
        // slope is biased by ≈ Σxᵢnᵢ/Σxᵢ² ≈ −4.6e-4 from 3.
        assert!((res.x[0] - 3.0).abs() < 1e-3, "slope = {}", res.x[0]);
        let se = res.standard_errors().expect("covariance");
        assert!(se[0] > 1e-4 && se[0] < 1.0, "se = {}", se[0]);
    }
}
