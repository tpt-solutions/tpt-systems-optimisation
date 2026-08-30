//! Constrained nonlinear programming via an augmented-Lagrangian (AL) solver.
//!
//! The published `tpt-math-optimize-general` crate only ships *unconstrained*
//! minimizers (gradient descent / conjugate gradient / Newton). The optimisation
//! crates (`tpt-opt-minlp`, `tpt-opt-network` AC-OPF) were built against the old
//! dev-shim's constrained-NLP surface (`solve_nlp` / `NlpProblem` /
//! `NlpParams` / `NlpResult` / `NlpStatus`), which is vendored here so the
//! workspace stays publishable against the published `tpt-math-*` crates.
//!
//! [`solve_nlp`] minimises an objective `f(x)` subject to inequality
//! constraints `c_i(x) <= 0` and equality constraints `h_j(x) = 0` using the
//! standard augmented-Lagrangian method: each outer iteration solves the
//! unconstrained subproblem
//!
//! ```text
//! L(x) = f(x) + Σ (λ_i c_i(x) + ρ/2 c_i(x)²) + Σ (μ_j h_j(x) + ρ/2 h_j(x)²)
//! ```
//!
//! with the conjugate-gradient minimizer from `tpt-math-optimize-general`, then
//! updates the Lagrange multipliers and (if needed) the penalty `ρ`. The inner
//! subproblem gradient is built analytically from [`NlpProblem::ineq_grad`] /
//! [`NlpProblem::eq_grad`] when provided, and falls back to central finite
//! differences otherwise.

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
    /// Upper bound on the penalty `ρ`.
    pub rho_max: f64,
}

impl Default for NlpParams {
    fn default() -> Self {
        NlpParams {
            tol: 1e-7,
            max_outer: 25,
            max_inner: 400,
            rho_init: 1.0,
            rho_max: 1e8,
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
        v = v.max(prob.ineq(i, x));
    }
    for j in 0..prob.num_eq() {
        v = v.max(prob.eq(j, x).abs());
    }
    v
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
    let mut mu = vec![0.0f64; me];
    let mut rho = params.rho_init.max(1e-3);
    let tol = params.tol.max(1e-10);
    let max_outer = params.max_outer.max(1);

    let mut prev_viol = f64::INFINITY;
    let mut outer = 0;
    while outer < max_outer {
        let cost = |p: &DVector<f64>| -> f64 {
            let xv: Vec<f64> = (0..n).map(|k| p[k]).collect();
            let mut l = prob.objective(&xv);
            for (i, &lam) in lambda.iter().enumerate() {
                let c = prob.ineq(i, &xv);
                l += lam * c + 0.5 * rho * c * c;
            }
            for (j, &mj) in mu.iter().enumerate() {
                let h = prob.eq(j, &xv);
                l += mj * h + 0.5 * rho * h * h;
            }
            l
        };
        let grad = |p: &DVector<f64>| -> DVector<f64> {
            let xv: Vec<f64> = (0..n).map(|k| p[k]).collect();
            let mut g = vec![0.0f64; n];
            prob.objective_grad(&xv, &mut g);
            for (i, &lam) in lambda.iter().enumerate() {
                let c = prob.ineq(i, &xv);
                let coeff = lam + rho * c;
                let mut row = vec![0.0f64; n];
                prob.ineq_grad(i, &xv, &mut row);
                for k in 0..n {
                    g[k] += coeff * row[k];
                }
            }
            for (j, &mj) in mu.iter().enumerate() {
                let h = prob.eq(j, &xv);
                let coeff = mj + rho * h;
                let mut row = vec![0.0f64; n];
                prob.eq_grad(j, &xv, &mut row);
                for k in 0..n {
                    g[k] += coeff * row[k];
                }
            }
            DVector::from_vec(g)
        };

        let init = DVector::from_vec(x.clone());
        let init_fb = init.clone();
        let opts = Options::new(params.max_inner as u64).with_gradient_tolerance(tol * 0.1);
        let sol = minimize_conjugate_gradient_with(cost, grad, init, &opts)
            .unwrap_or_else(|_| Solution {
                param: init_fb,
                cost: f64::INFINITY,
                iters: 0,
                converged: false,
                termination: String::new(),
            });
        x = (0..n).map(|k| sol.param[k]).collect();

        let viol = max_violation(prob, &x);
        for (i, lam) in lambda.iter_mut().enumerate() {
            *lam = (*lam + rho * prob.ineq(i, &x)).max(0.0);
        }
        for (j, mj) in mu.iter_mut().enumerate() {
            *mj += rho * prob.eq(j, &x);
        }
        if viol > 0.25 * prev_viol {
            rho = (rho * 10.0).min(params.rho_max);
        }
        prev_viol = viol;
        outer += 1;
        if viol < tol {
            break;
        }
    }

    let final_viol = max_violation(prob, &x);
    let status = if final_viol < tol {
        NlpStatus::Converged
    } else if outer >= max_outer {
        NlpStatus::Diverged
    } else {
        NlpStatus::MaxIters
    };

    let objective = prob.objective(&x);
    NlpResult {
        x,
        objective,
        status,
        iterations: outer,
    }
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
