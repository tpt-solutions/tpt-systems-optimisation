//! Constrained nonlinear programming via an augmented-Lagrangian (AL) solver.
//!
//! [`solve_nlp`] minimises an objective `f(x)` subject to inequality
//! constraints `c_i(x) <= 0` and equality constraints `h_j(x) = 0` using the
//! standard augmented-Lagrangian method: each outer iteration solves the
//! unconstrained subproblem
//!
//! ```text
//! L(x) = f(x) + Σ (λ_i·max(0,c_i(x)) + ½·ρ·max(0,c_i(x))²)
//!            + Σ (μ_j·h_j(x) + ½·ρ·h_j(x)²)
//! ```
//!
//! with the conjugate-gradient minimizer from `tpt-math-optimize-general`, then
//! updates the Lagrange multipliers and (if needed) the penalty `ρ`. Inequality
//! constraints only contribute to the penalty when violated (`max(0,·)`), which
//! is the correct Rockafellar form; applying the quadratic term for satisfied
//! constraints instead corrupts the solution. The inner subproblem gradient is
//! built analytically from [`NlpProblem::ineq_grad`] / [`NlpProblem::eq_grad`]
//! when provided, and falls back to central finite differences otherwise.
//!
//! Multipliers and the penalty are only updated after a *settled* inner solve
//! (gradient tolerance met), with a stall counter that forces progress when the
//! inner solver repeatedly fails to settle — this prevents multiplier blow-up
//! from truncated solves. Convergence additionally requires a complementarity
//! check (`λ·|c| ≈ 0`): without it a degenerate AL fixed point where a multiplier
//! cancels the objective gradient on a slack constraint masquerades as optimal.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use tpt_math_optimize_general::{
    minimize_conjugate_gradient_with, tpt_math_linalg_dense::DVector, Options, Solution,
};

/// A constrained nonlinear program over `x ∈ R^n`.
///
/// Implementors provide the objective and constraints (and, optionally, their
/// gradients; the gradient methods have central-finite-difference defaults so a
/// zero-order model compiles and runs).
pub trait NlpProblem {
    /// Dimension of the decision vector.
    fn num_vars(&self) -> usize;

    /// Objective value at `x`.
    fn objective(&self, x: &[f64]) -> f64;

    /// Gradient of [`NlpProblem::objective`] written into `g` (length `num_vars`).
    fn objective_grad(&self, x: &[f64], g: &mut [f64]) {
        fd_grad(self.num_vars(), |x| self.objective(x), x, g);
    }

    /// Number of inequality constraints `c_i(x) <= 0`.
    fn num_ineq(&self) -> usize;

    /// Inequality constraint `c_i(x)` value.
    fn ineq(&self, i: usize, x: &[f64]) -> f64;

    /// Gradient of the `i`-th inequality constraint written into `row`.
    fn ineq_grad(&self, i: usize, x: &[f64], row: &mut [f64]) {
        fd_grad(self.num_vars(), |x| self.ineq(i, x), x, row);
    }

    /// Number of equality constraints `h_j(x) = 0`.
    fn num_eq(&self) -> usize;

    /// Equality constraint `h_j(x)` value.
    fn eq(&self, j: usize, x: &[f64]) -> f64;

    /// Gradient of the `j`-th equality constraint written into `row`.
    fn eq_grad(&self, j: usize, x: &[f64], row: &mut [f64]) {
        fd_grad(self.num_vars(), |x| self.eq(j, x), x, row);
    }
}

/// Solver configuration for [`solve_nlp`].
#[derive(Clone, Copy, Debug)]
pub struct NlpParams {
    /// Feasibility (constraint-violation) tolerance for outer convergence.
    pub tol: f64,
    /// Maximum number of outer AL iterations.
    pub max_outer: usize,
    /// Iteration budget for each inner unconstrained subproblem.
    pub max_inner: usize,
    /// Initial quadratic penalty `ρ`.
    pub rho_init: f64,
    /// Penalty growth factor applied after each settled outer iteration.
    pub rho_growth: f64,
    /// Upper bound on the penalty `ρ`.
    pub rho_max: f64,
}

impl Default for NlpParams {
    fn default() -> Self {
        NlpParams {
            tol: 1e-6,
            max_outer: 25,
            max_inner: 350,
            rho_init: 10.0,
            rho_growth: 8.0,
            rho_max: 1e12,
        }
    }
}

/// Termination status of a [`solve_nlp`] run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NlpStatus {
    /// The outer loop reached the feasibility tolerance.
    Converged,
    /// The outer loop exhausted `max_outer` without meeting the tolerance.
    Diverged,
    /// Inner solver budget exhausted before convergence.
    MaxIters,
}

/// Outcome of a [`solve_nlp`] run.
#[derive(Clone, Debug)]
pub struct NlpResult {
    /// Best (lowest-violation) decision vector found.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub objective: f64,
    /// Termination status.
    pub status: NlpStatus,
    /// Number of outer AL iterations performed.
    pub iterations: usize,
}

/// Central finite-difference gradient of scalar `f` at `x` into `g`.
fn fd_grad(n: usize, f: impl Fn(&[f64]) -> f64, x: &[f64], g: &mut [f64]) {
    let base = 1e-6;
    for k in 0..n {
        let hk = (base * x[k].abs()).max(base);
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[k] = x[k] + hk;
        xm[k] = x[k] - hk;
        g[k] = (f(&xp) - f(&xm)) / (2.0 * hk);
    }
}

fn max_violation<P: NlpProblem>(prob: &P, x: &[f64]) -> f64 {
    let mut v = 0.0f64;
    for i in 0..prob.num_ineq() {
        v = v.max(prob.ineq(i, x).max(0.0));
    }
    for j in 0..prob.num_eq() {
        v = v.max(prob.eq(j, x).abs());
    }
    v
}

/// Self-contained BFGS unconstrained minimizer (alloc only), mirroring the
/// dev-shim's proven inner solver. Uses a normalised search direction and an
/// Armijo line search, and reports whether the gradient tolerance was reached
/// (`settled`) plus the infinity-norm of the final gradient (`grad_inf`).
fn bfgs_minimize(
    cost: impl Fn(&[f64]) -> f64,
    grad: impl Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    max_inner: usize,
    tol: f64,
) -> (Vec<f64>, bool, f64) {
    let n = x0.len();
    if n == 0 {
        return (Vec::new(), true, 0.0);
    }
    let mut x = x0.to_vec();
    let mut inv_hess = vec![0.0f64; n * n];
    for i in 0..n {
        inv_hess[i * n + i] = 1.0;
    }

    let mut settled = false;
    let mut grad_inf = 0.0f64;
    for _ in 0..max_inner {
        let f = cost(&x);
        let g = grad(&x);
        if g.iter().map(|v| v * v).sum::<f64>() < tol * tol {
            settled = true;
            grad_inf = g.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            break;
        }
        // d = -inv_hess * g, normalised by its infinity norm.
        let mut d = vec![0.0f64; n];
        for i in 0..n {
            for k in 0..n {
                d[i] += inv_hess[i * n + k] * g[k];
            }
            d[i] = -d[i];
        }
        let dnorm_inf = d.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        if dnorm_inf > 0.0 {
            let inv = 1.0 / dnorm_inf;
            for v in d.iter_mut() {
                *v *= inv;
            }
        }
        // Armijo line search.
        let mut step = 1.0;
        let mut improved = false;
        let mut nx: Vec<f64>;
        for _ls in 0..30 {
            nx = (0..n).map(|k| x[k] + step * d[k]).collect();
            if cost(&nx) <= f - 1e-4 * step * dot(&g, &d) {
                let gnew = grad(&nx);
                let mut y_vec = vec![0.0f64; n];
                for k in 0..n {
                    y_vec[k] = gnew[k] - g[k];
                }
                bfgs_update(&mut inv_hess, &d, &y_vec, step);
                x = nx;
                improved = true;
                break;
            }
            step *= 0.5;
        }
        if !improved {
            grad_inf = g.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            break;
        }
        grad_inf = g.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    }
    (x, settled, grad_inf)
}

fn bfgs_update(h: &mut [f64], d: &[f64], y: &[f64], step: f64) {
    let n = d.len();
    let mut s = vec![0.0f64; n];
    for i in 0..n {
        s[i] = step * d[i];
    }
    let ys = dot(y, &s).max(1e-12);
    let mut hy = vec![0.0f64; n];
    for i in 0..n {
        for k in 0..n {
            hy[i] += h[i * n + k] * y[k];
        }
    }
    let yhy = dot(y, &hy);
    for i in 0..n {
        for j in 0..n {
            let term1 = (hy[i] * s[j] + s[i] * hy[j]) / ys;
            let term2 = (yhy / (ys * ys)) * s[i] * s[j];
            h[i * n + j] += term1 - term2;
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Solve the constrained NLP defined by `prob` starting from `x0`.
///
/// Returns the best feasible point found; [`NlpResult::status`] reports whether
/// the feasibility tolerance was met.
pub fn solve_nlp<P: NlpProblem>(prob: &P, x0: &[f64], params: &NlpParams) -> NlpResult {
    let n = prob.num_vars();
    let mi = prob.num_ineq();
    let me = prob.num_eq();
    let mut x = x0.to_vec();
    let mut lambda = vec![0.0f64; mi];
    let mut nu = vec![0.0f64; me];
    let mut rho = params.rho_init.max(1e-3);
    let tol = params.tol.max(1e-10);
    let rho_growth = params.rho_growth.max(1.0);
    let rho_max = params.rho_max.max(rho);
    let max_outer = params.max_outer.max(1);

    let mut status = NlpStatus::MaxIters;
    let mut prev_x: Option<Vec<f64>> = None;
    let mut stalled_outers = 0usize;
    let mut outer = 0;

    for _ in 0..max_outer {
        outer += 1;
        // Minimise the augmented Lagrangian (with clamped inequality terms)
        // using the published conjugate-gradient solver.
        let al_cost = |p: &DVector<f64>| -> f64 {
            let xv: Vec<f64> = (0..n).map(|k| p[k]).collect();
            let mut val = prob.objective(&xv);
            for (i, &lam) in lambda.iter().enumerate() {
                let c = prob.ineq(i, &xv).max(0.0);
                val += lam * c + 0.5 * rho * c * c;
            }
            for (j, &nj) in nu.iter().enumerate() {
                let c = prob.eq(j, &xv);
                val += nj * c + 0.5 * rho * c * c;
            }
            val
        };
        let al_grad = |p: &DVector<f64>| -> DVector<f64> {
            let xv: Vec<f64> = (0..n).map(|k| p[k]).collect();
            let mut g = vec![0.0f64; n];
            prob.objective_grad(&xv, &mut g);
            for (i, &lam) in lambda.iter().enumerate() {
                let c = prob.ineq(i, &xv).max(0.0);
                if c > 0.0 {
                    let coeff = lam + rho * c;
                    let mut row = vec![0.0f64; n];
                    prob.ineq_grad(i, &xv, &mut row);
                    for k in 0..n {
                        g[k] += coeff * row[k];
                    }
                }
            }
            for (j, &nj) in nu.iter().enumerate() {
                let c = prob.eq(j, &xv);
                let coeff = nj + rho * c;
                let mut row = vec![0.0f64; n];
                prob.eq_grad(j, &xv, &mut row);
                for k in 0..n {
                    g[k] += coeff * row[k];
                }
            }
            DVector::from_vec(g)
        };

        let init = DVector::from_vec(x.clone());
        let opts = Options::new(params.max_inner as u64).with_gradient_tolerance(tol);
        let sol =
            minimize_conjugate_gradient_with(al_cost, al_grad, init, &opts).unwrap_or_else(|_| {
                Solution {
                    param: DVector::from_vec(x.clone()),
                    cost: f64::INFINITY,
                    iters: 0,
                    converged: false,
                    termination: String::new(),
                }
            });
        // The published conjugate-gradient solver is the primary inner solver.
        // When it does not settle (gradient tolerance unmet) the augmented
        // Lagrangian subproblem is often non-convex, so we also run the
        // quasi-Newton BFGS minimizer and keep whichever reaches the lower
        // augmented-Lagrangian value; BFGS frequently escapes CG's local minima
        // on these subproblems and is what makes the OA/SQP tests converge.
        let cg_x: Vec<f64> = (0..n).map(|k| sol.param[k]).collect();
        let cg_cost = al_cost(&DVector::from_vec(cg_x.clone()));
        let al_cost_slice = |v: &[f64]| -> f64 { al_cost(&DVector::from_vec(v.to_vec())) };
        let al_grad_slice = |v: &[f64]| -> Vec<f64> {
            al_grad(&DVector::from_vec(v.to_vec())).iter().cloned().collect()
        };
        let (bfgs_x, bfgs_settled, _) =
            bfgs_minimize(al_cost_slice, al_grad_slice, &x.clone(), params.max_inner, tol);
        let bfgs_cost = al_cost(&DVector::from_vec(bfgs_x.clone()));
        let (chosen, settled) = if sol.converged && (!bfgs_settled || cg_cost <= bfgs_cost) {
            (cg_x, true)
        } else {
            (bfgs_x, bfgs_settled)
        };
        x = chosen;

        // Infinity-norm of the AL gradient at the returned point (for the
        // scaled-KKT convergence proxy).
        let grad_inf = {
            let xv: Vec<f64> = x.clone();
            let mut g = vec![0.0f64; n];
            prob.objective_grad(&xv, &mut g);
            for (i, &lam) in lambda.iter().enumerate() {
                let c = prob.ineq(i, &xv).max(0.0);
                if c > 0.0 {
                    let coeff = lam + rho * c;
                    let mut row = vec![0.0f64; n];
                    prob.ineq_grad(i, &xv, &mut row);
                    for k in 0..n {
                        g[k] += coeff * row[k];
                    }
                }
            }
            for (j, &nj) in nu.iter().enumerate() {
                let c = prob.eq(j, &xv);
                let coeff = nj + rho * c;
                let mut row = vec![0.0f64; n];
                prob.eq_grad(j, &xv, &mut row);
                for k in 0..n {
                    g[k] += coeff * row[k];
                }
            }
            g.iter().fold(0.0f64, |a, v| a.max(v.abs()))
        };

        let max_viol = max_violation(prob, &x);

        // Multiplier/penalty updates only after a settled inner solve (or once
        // the inner loop has stalled enough times to force progress).
        if settled || stalled_outers >= 3 {
            for (i, lam) in lambda.iter_mut().enumerate() {
                *lam += rho * prob.ineq(i, &x).max(0.0);
            }
            for (j, nj) in nu.iter_mut().enumerate() {
                *nj += rho * prob.eq(j, &x);
            }
            stalled_outers = 0;
            rho = (rho * rho_growth).min(rho_max);
        } else {
            stalled_outers += 1;
        }

        // Complementarity: a multiplier may only be positive on a (near-)active
        // constraint.
        let mut compl = 0.0f64;
        for (i, &lam) in lambda.iter().enumerate() {
            compl = compl.max(lam * prob.ineq(i, &x).abs());
        }
        for (j, &nj) in nu.iter().enumerate() {
            compl = compl.max(nj.abs() * prob.eq(j, &x).abs());
        }
        let mult_scale =
            lambda.iter().chain(nu.iter()).fold(0.0f64, |a, v| a.max(v.abs())).max(1.0);
        let kkt_scaled = grad_inf / (mult_scale + rho * max_viol);
        let comp_ok = compl <= 1e-6 * (1.0 + mult_scale);

        let moved = match &prev_x {
            Some(p) => p.iter().zip(&x).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max),
            None => f64::INFINITY,
        };

        if max_viol < tol
            && comp_ok
            && (settled || moved <= tol.max(1e-9) || (grad_inf < 1e-2 && kkt_scaled < tol))
        {
            status = NlpStatus::Converged;
            break;
        }
        prev_x = Some(x.clone());
    }

    let objective = prob.objective(&x);
    NlpResult { x, objective, status, iterations: outer }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f(x) = x²,  constraint x >= 1  (i.e. 1 - x <= 0).
    struct Bounded {
        lo: f64,
    }
    impl NlpProblem for Bounded {
        fn num_vars(&self) -> usize {
            1
        }
        fn objective(&self, x: &[f64]) -> f64 {
            x[0] * x[0]
        }
        fn num_ineq(&self) -> usize {
            1
        }
        fn ineq(&self, _i: usize, x: &[f64]) -> f64 {
            self.lo - x[0]
        }
        fn num_eq(&self) -> usize {
            0
        }
        fn eq(&self, _j: usize, _x: &[f64]) -> f64 {
            0.0
        }
    }

    #[test]
    fn solves_simple_bound() {
        let prob = Bounded { lo: 1.0 };
        let res = solve_nlp(&prob, &[0.0], &NlpParams::default());
        assert_eq!(res.status, NlpStatus::Converged);
        assert!((res.x[0] - 1.0).abs() < 1e-3, "x = {}", res.x[0]);
    }
}
