#![no_std]
//! Local dev shim mirroring `tpt-math-optimize-general`: a general nonlinear
//! program (NLP) solver.
//!
//! The solver is an **augmented Lagrangian (AL)** method: the constrained
//! problem is reduced to a sequence of unconstrained subproblems minimised with
//! **BFGS + Armijo line search**, with multipliers and the penalty updated in
//! an outer loop. Gradients/Jacobians default to forward finite differences.
//!
//! This is a pragmatic local-dev implementation sufficient for the small NLP
//! subproblems arising in `tpt-opt-minlp` (outer approximation) and
//! `tpt-opt-network` (OPF). It is not an industrial-scale solver.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

/// Outcome of an NLP solve.
#[derive(Debug, Clone)]
pub struct NlpResult {
    pub x: Vec<f64>,
    pub objective: f64,
    pub status: NlpStatus,
    pub iterations: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NlpStatus {
    Converged,
    MaxIterations,
    Diverged,
}

/// A nonlinear program in the form
/// ```text
/// minimise    f(x)
/// subject to   g_i(x) <= 0   for i in 0..num_ineq
///             h_j(x)  = 0   for j in 0..num_eq
/// ```
pub trait NlpProblem {
    fn num_vars(&self) -> usize;
    fn objective(&self, x: &[f64]) -> f64;
    /// Fill `g` with the gradient of the objective at `x`.
    fn objective_grad(&self, x: &[f64], g: &mut [f64]) {
        finite_diff_grad(|y| self.objective(y), x, g);
    }
    fn num_ineq(&self) -> usize {
        0
    }
    fn num_eq(&self) -> usize {
        0
    }
    /// Inequality constraint `g_i(x) <= 0`.
    fn ineq(&self, _i: usize, _x: &[f64]) -> f64 {
        0.0
    }
    /// Fill `row` with the gradient of `ineq(i, x)`.
    fn ineq_grad(&self, i: usize, x: &[f64], row: &mut [f64]) {
        let fi = |y: &[f64]| self.ineq(i, y);
        finite_diff_grad(fi, x, row);
    }
    /// Equality constraint `h_j(x) = 0`.
    fn eq(&self, _j: usize, _x: &[f64]) -> f64 {
        0.0
    }
    /// Fill `row` with the gradient of `eq(j, x)`.
    fn eq_grad(&self, j: usize, x: &[f64], row: &mut [f64]) {
        let fj = |y: &[f64]| self.eq(j, y);
        finite_diff_grad(fj, x, row);
    }
}

/// Tunable solver parameters.
#[derive(Debug, Clone)]
pub struct NlpParams {
    pub max_outer: usize,
    pub max_inner: usize,
    pub tol: f64,
    pub rho_init: f64,
    pub rho_growth: f64,
}

impl Default for NlpParams {
    fn default() -> Self {
        Self {
            max_outer: 50,
            max_inner: 200,
            tol: 1e-6,
            rho_init: 1.0,
            rho_growth: 4.0,
        }
    }
}

/// Solve a nonlinear program starting from `x0`.
pub fn solve_nlp<P: NlpProblem>(prob: &P, x0: &[f64], params: &NlpParams) -> NlpResult {
    let n = prob.num_vars();
    let m_ineq = prob.num_ineq();
    let m_eq = prob.num_eq();
    let mut x = x0.to_vec();
    let mut lambda = vec![0.0f64; m_ineq];
    let mut nu = vec![0.0f64; m_eq];
    let mut rho = params.rho_init;

    let mut status = NlpStatus::MaxIterations;
    let mut total_iters = 0;

    for _outer in 0..params.max_outer {
        // Minimise the augmented Lagrangian with BFGS.
        let (x_new, iters) = minimize_alm(prob, &x, &lambda, &nu, rho, params);
        total_iters += iters;
        x = x_new;

        // Constraint violation at the new point.
        let mut max_viol = 0.0f64;
        for i in 0..m_ineq {
            let c = prob.ineq(i, &x).max(0.0);
            max_viol = max_viol.max(c);
            lambda[i] += rho * c;
        }
        for j in 0..m_eq {
            let c = prob.eq(j, &x).abs();
            max_viol = max_viol.max(c);
            nu[j] += rho * prob.eq(j, &x);
        }

        if max_viol < params.tol {
            status = NlpStatus::Converged;
            break;
        }
        rho *= params.rho_growth;
    }

    NlpResult {
        x,
        objective: prob.objective(&x),
        status,
        iterations: total_iters,
    }
}

/// Minimise the augmented Lagrangian for fixed multipliers/penalty via BFGS.
fn minimize_alm<P: NlpProblem>(
    prob: &P,
    x0: &[f64],
    lambda: &[f64],
    nu: &[f64],
    rho: f64,
    params: &NlpParams,
) -> (Vec<f64>, usize) {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut inv_hess = identity(n);

    let al = |y: &[f64]| -> f64 {
        let mut val = prob.objective(y);
        for i in 0..lambda.len() {
            let c = prob.ineq(i, y).max(0.0);
            val += lambda[i] * c + 0.5 * rho * c * c;
        }
        for j in 0..nu.len() {
            let c = prob.eq(j, y);
            val += nu[j] * c + 0.5 * rho * c * c;
        }
        val
    };

    let mut iters = 0;
    for _ in 0..params.max_inner {
        iters += 1;
        let f = al(&x);
        // Gradient of AL.
        let mut grad = vec![0.0f64; n];
        prob.objective_grad(&x, &mut grad);
        for i in 0..lambda.len() {
            let c = prob.ineq(i, x).max(0.0);
            if c > 0.0 {
                let mut g = vec![0.0f64; n];
                prob.ineq_grad(i, &x, &mut g);
                for k in 0..n {
                    grad[k] += (lambda[i] + rho * c) * g[k];
                }
            }
        }
        for j in 0..nu.len() {
            let c = prob.eq(j, x);
            let mut g = vec![0.0f64; n];
            prob.eq_grad(j, &x, &mut g);
            for k in 0..n {
                grad[k] += (nu[j] + rho * c) * g[k];
            }
        }

        // Stop if gradient is tiny.
        if grad.iter().map(|v| v * v).sum::<f64>().sqrt() < params.tol {
            break;
        }

        // Compute search direction d = -H*grad.
        let mut d = vec![0.0f64; n];
        for i in 0..n {
            for k in 0..n {
                d[i] += inv_hess[i * n + k] * grad[k];
            }
            d[i] = -d[i];
        }

        // Armijo line search.
        let mut step = 1.0;
        let mut improved = false;
        for _ls in 0..30 {
            let nx: Vec<f64> = x.iter().zip(&d).map(|(xi, di)| xi + step * di).collect();
            if al(&nx) <= f - 1e-4 * step * dot(&grad, &d) {
                // BFGS update.
                let mut y_vec = vec![0.0f64; n];
                let mut gnew = vec![0.0f64; n];
                prob.objective_grad(&nx, &mut gnew);
                for i in 0..lambda.len() {
                    let c = prob.ineq(i, nx).max(0.0);
                    if c > 0.0 {
                        let mut g = vec![0.0f64; n];
                        prob.ineq_grad(i, &nx, &mut g);
                        for k in 0..n {
                            gnew[k] += (lambda[i] + rho * c) * g[k];
                        }
                    }
                }
                for j in 0..nu.len() {
                    let c = prob.eq(j, nx);
                    let mut g = vec![0.0f64; n];
                    prob.eq_grad(j, &nx, &mut g);
                    for k in 0..n {
                        gnew[k] += (nu[j] + rho * c) * g[k];
                    }
                }
                for k in 0..n {
                    y_vec[k] = gnew[k] - grad[k];
                }
                bfgs_update(&mut inv_hess, &d, &y_vec, step);
                x = nx;
                improved = true;
                break;
            }
            step *= 0.5;
        }
        if !improved {
            break;
        }
    }
    (x, iters)
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

fn finite_diff_grad<F: Fn(&[f64]) -> f64>(f: F, x: &[f64], g: &mut [f64]) {
    let h = 1e-6;
    let fx = f(x);
    for i in 0..x.len() {
        let mut xp = x.to_vec();
        xp[i] += h;
        g[i] = (f(&xp) - fx) / h;
    }
}

#[allow(dead_code)]
fn finite_diff_jac_row<F: Fn(&[f64]) -> f64>(f: F, x: &[f64], row: &mut [f64]) {
    finite_diff_grad(f, x, row);
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}
