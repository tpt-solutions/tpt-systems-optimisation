//! L-BFGS with strong-Wolfe line search for smooth unconstrained
//! minimisation.
//!
//! The limited-memory BFGS inverse-Hessian approximation keeps `m`
//! `(s, y)` correction pairs and applies them with the standard two-loop
//! recursion, so memory is O(m·n) for any `n`. Gradient supply is the
//! interesting choice: [`Objective::reverse`] builds the objective against
//! the [`reverse`](crate::reverse) value graph, giving an **exact**
//! gradient in one backward pass — the pairing reverse mode was built for.
//! [`Objective::analytic`] takes a user gradient;
//! [`Objective::new`] (finite differences) is the fallback.
//!
//! # Example
//!
//! ```rust
//! use tpt_opt_nls::{lbfgs, LbfgsConfig, Objective};
//!
//! // Rosenbrock via reverse-mode autodiff: exact gradient, one pass.
//! let f = Objective::reverse(|x, out| {
//!     let r1 = 1.0 - x[0].clone();
//!     let r2 = 10.0 * (x[1].clone() - x[0].clone() * x[0].clone());
//!     *out = 0.5 * (r1.clone() * r1.clone() + r2.clone() * r2.clone());
//! });
//! let result = lbfgs(&f, &[-1.2, 1.0], &LbfgsConfig::new()).unwrap();
//! assert_eq!(result.status, tpt_opt_nls::NlsStatus::Converged);
//! assert!((result.x[0] - 1.0).abs() < 1e-5);
//! assert!((result.x[1] - 1.0).abs() < 1e-5);
//! ```

use crate::reverse::Value;
use crate::types::{NlsError, NlsStatus};

/// Objective for the gradient-based solvers.
pub enum Objective<'a> {
    /// Value only — gradient by central finite differences.
    FiniteDifference(Box<dyn Fn(&[f64]) -> f64 + 'a>),
    /// Value plus user gradient.
    Analytic(
        Box<dyn Fn(&[f64]) -> f64 + 'a>,
        Box<dyn Fn(&[f64], &mut [f64]) + 'a>,
    ),
    /// Objective written against reverse-mode values — exact gradient via
    /// one backward pass per evaluation.
    Reverse(Box<dyn Fn(&[Value], &mut Value) + 'a>),
}

impl<'a> Objective<'a> {
    /// Value-only objective (central-difference gradient).
    pub fn new(f: impl Fn(&[f64]) -> f64 + 'a) -> Self {
        Objective::FiniteDifference(Box::new(f))
    }

    /// Value + analytic gradient.
    pub fn analytic(
        f: impl Fn(&[f64]) -> f64 + 'a,
        g: impl Fn(&[f64], &mut [f64]) + 'a,
    ) -> Self {
        Objective::Analytic(Box::new(f), Box::new(g))
    }

    /// Reverse-mode objective: `build` receives input variables and writes
    /// the scalar output. Each [`Objective::eval`] call builds a fresh
    /// graph (the leaves are what change between calls).
    pub fn reverse(build: impl Fn(&[Value], &mut Value) + 'a) -> Self {
        Objective::Reverse(Box::new(build))
    }

    /// Evaluate `f(x)`, writing the gradient into `g`.
    pub fn eval(&self, x: &[f64], g: &mut [f64]) -> f64 {
        match self {
            Objective::FiniteDifference(f) => {
                let v = f(x);
                let n = x.len();
                let mut xp = x.to_vec();
                let mut xm = x.to_vec();
                for j in 0..n {
                    let h = 1e-6 * (1.0 + x[j].abs());
                    let xj = x[j];
                    xp[j] = xj + h;
                    xm[j] = xj - h;
                    g[j] = (f(&xp) - f(&xm)) / (2.0 * h);
                    xp[j] = xj;
                    xm[j] = xj;
                }
                v
            }
            Objective::Analytic(f, g_user) => {
                let v = f(x);
                g_user(x, g);
                v
            }
            Objective::Reverse(build) => {
                let leaves: Vec<Value> = x.iter().map(|&v| Value::var(v)).collect();
                let mut out = Value::var(0.0);
                build(&leaves, &mut out);
                out.backward();
                for (lv, gv) in leaves.iter().zip(g.iter_mut()) {
                    *gv = lv.grad();
                }
                out.value()
            }
        }
    }
}

/// L-BFGS configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LbfgsConfig {
    /// Gradient infinity-norm tolerance.
    pub gtol: f64,
    /// Relative function-decrease tolerance.
    pub ftol: f64,
    /// Maximum outer iterations.
    pub max_iter: usize,
    /// Number of stored `(s, y)` correction pairs.
    pub history: usize,
    /// Armijo (sufficient-decrease) constant.
    pub c1: f64,
    /// Strong-Wolfe curvature constant.
    pub c2: f64,
    /// Maximum line-search evaluations per iteration.
    pub max_line_search: usize,
}

impl LbfgsConfig {
    /// Defaults: `gtol = 1e-8`, `ftol = 1e-12`, 300 iterations, history 10,
    /// Wolfe constants 1e-4 / 0.9, 40 line-search evals.
    pub fn new() -> Self {
        Self {
            gtol: 1e-8,
            ftol: 1e-12,
            max_iter: 300,
            history: 10,
            c1: 1e-4,
            c2: 0.9,
            max_line_search: 40,
        }
    }

    /// Override the gradient tolerance.
    pub fn with_gtol(mut self, v: f64) -> Self {
        self.gtol = v;
        self
    }

    /// Override the iteration budget.
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Override the memory (correction-pair count).
    pub fn with_history(mut self, m: usize) -> Self {
        self.history = m.max(1);
        self
    }
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of an L-BFGS run.
#[derive(Debug, Clone)]
pub struct LbfgsResult {
    /// Termination status.
    pub status: NlsStatus,
    /// Final point.
    pub x: Vec<f64>,
    /// Final objective value.
    pub value: f64,
    /// Final gradient infinity norm.
    pub gradient_norm: f64,
    /// Outer iterations.
    pub iterations: usize,
    /// Objective evaluations (including line search).
    pub evaluations: usize,
}

/// Minimise `objective` from `x0` with L-BFGS + strong-Wolfe line search.
pub fn lbfgs(
    objective: &Objective<'_>,
    x0: &[f64],
    cfg: &LbfgsConfig,
) -> Result<LbfgsResult, NlsError> {
    if x0.is_empty() {
        return Err(NlsError::InvalidProblem("x0 is empty".into()));
    }
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g = vec![0.0f64; n];
    let mut evals = 0usize;
    let mut f = {
        evals += 1;
        objective.eval(&x, &mut g)
    };
    let mut gnorm = infinity_norm(&g);
    if gnorm <= cfg.gtol {
        return Ok(done(NlsStatus::Converged, x, f, gnorm, 0, evals));
    }

    // Correction pairs (bounded history).
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(cfg.history);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(cfg.history);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(cfg.history);

    let mut iterations = 0usize;
    while iterations < cfg.max_iter {
        // Two-loop recursion for the search direction.
        let mut q = g.clone();
        let mut alphas = vec![0.0f64; s_hist.len()];
        for i in (0..s_hist.len()).rev() {
            let rho = rho_hist[i];
            let a = dot(&s_hist[i], &q) * rho;
            alphas[i] = a;
            for k in 0..n {
                q[k] -= a * y_hist[i][k];
            }
        }
        // Initial scaling γ = sᵀy / yᵀy from the newest pair.
        if let Some(i) = s_hist.len().checked_sub(1) {
            let sy = dot(&s_hist[i], &y_hist[i]);
            let yy = dot(&y_hist[i], &y_hist[i]);
            let gamma = if yy > 0.0 { sy / yy } else { 1.0 };
            for v in q.iter_mut() {
                *v *= gamma;
            }
        }
        let mut direction: Vec<f64> = q.iter().map(|v| -*v).collect();
        for i in 0..s_hist.len() {
            let rho = rho_hist[i];
            let b = dot(&y_hist[i], &direction) * rho;
            for k in 0..n {
                direction[k] -= s_hist[i][k] * (alphas[i] - b);
            }
        }
        for v in direction.iter_mut() {
            *v = -*v;
        }
        if dot(&g, &direction) >= 0.0 {
            // Lost descent (numerical): reset to steepest descent.
            for (v, gv) in direction.iter_mut().zip(g.iter()) {
                *v = -gv;
            }
        }

        // Strong-Wolfe line search along `direction`.
        let dphi0 = dot(&g, &direction);
        if dphi0 >= 0.0 {
            return Ok(done(NlsStatus::NumericalIssue, x, f, gnorm, iterations, evals));
        }
        let mut trial_x = vec![0.0f64; n];
        let mut trial_g = vec![0.0f64; n];
        // Returns (phi(α), phi'(α) = ∇f(x+αd)ᵀd).
        let mut phi = |alpha: f64, sx: &mut [f64], sg: &mut [f64]| -> (f64, f64) {
            for k in 0..n {
                sx[k] = x[k] + alpha * direction[k];
            }
            evals += 1;
            let v = objective.eval(sx, sg);
            let d = dot(sg, &direction);
            (v, d)
        };
        let alpha;
        match strong_wolfe(
            &mut phi,
            f,
            dphi0,
            cfg.c1,
            cfg.c2,
            cfg.max_line_search,
            &mut trial_x,
            &mut trial_g,
        ) {
            Some(a) => alpha = a,
            None => {
                // Strong-Wolfe failed (collapsed bracket near a curved
                // valley): fall back to plain Armijo backtracking from α=1,
                // like scipy/L-BFGS-B, before giving up.
                let mut alpha_fb = 1.0f64;
                let mut accepted = false;
                for _ in 0..cfg.max_line_search {
                    let (v, _) = phi(alpha_fb, &mut trial_x, &mut trial_g);
                    if v <= f + cfg.c1 * alpha_fb * dphi0 && v < f {
                        alpha = alpha_fb;
                        accepted = true;
                        break;
                    }
                    alpha_fb *= 0.5;
                }
                if !accepted {
                    return Ok(done(NlsStatus::CostTolerance, x, f, gnorm, iterations, evals));
                }
                alpha = alpha_fb;
            }
        };

        let x_new: Vec<f64> = (0..n).map(|k| x[k] + alpha * direction[k]).collect();
        let mut g_new = vec![0.0f64; n];
        evals += 1;
        let f_new = objective.eval(&x_new, &mut g_new);
        let gnorm_new = infinity_norm(&g_new);

        // Update the correction pair (skip curvature updates that would be
        // numerically meaningless).
        let s: Vec<f64> = (0..n).map(|k| x_new[k] - x[k]).collect();
        let y: Vec<f64> = (0..n).map(|k| g_new[k] - g[k]).collect();
        let sy = dot(&s, &y);
        if sy > 1e-10 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt() {
            if s_hist.len() == cfg.history {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / sy);
        }

        let f_old = f;
        x = x_new;
        f = f_new;
        g = g_new;
        gnorm = gnorm_new;
        iterations += 1;

        if gnorm <= cfg.gtol {
            return Ok(done(NlsStatus::Converged, x, f, gnorm, iterations, evals));
        }
        if (f_old - f).abs() <= cfg.ftol * f_old.abs().max(1.0) {
            return Ok(done(NlsStatus::CostTolerance, x, f, gnorm, iterations, evals));
        }
    }
    Ok(done(NlsStatus::MaxIterations, x, f, gnorm, iterations, evals))
}

#[allow(clippy::too_many_arguments)]
fn done(
    status: NlsStatus,
    x: Vec<f64>,
    value: f64,
    gradient_norm: f64,
    iterations: usize,
    evaluations: usize,
) -> LbfgsResult {
    LbfgsResult { status, x, value, gradient_norm, iterations, evaluations }
}

fn infinity_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |a, &b| a.max(b.abs()))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Strong-Wolfe line search (Nocedal & Wright Alg. 3.5/3.6): geometric
/// bracket expansion then zoom with quadratic interpolation. `phi(alpha)`
/// returns `(φ(α), φ'(α))`; `(phi0, dphi0)` = `(φ(0), φ'(0))` (a descent
/// slope). Returns a step satisfying both Wolfe conditions.
#[allow(clippy::too_many_arguments)]
fn strong_wolfe(
    phi: &mut impl FnMut(f64, &mut [f64], &mut [f64]) -> (f64, f64),
    phi0: f64,
    dphi0: f64,
    c1: f64,
    c2: f64,
    max_evals: usize,
    sx: &mut [f64],
    sg: &mut [f64],
) -> Option<f64> {
    let mut a_prev = 0.0f64;
    let mut phi_prev = phi0;
    let mut dphi_prev = dphi0;
    let mut alpha = 1.0f64;
    for _ in 0..max_evals {
        let (phi_a, dphi_a) = phi(alpha, sx, sg);
        if phi_a > phi0 + c1 * alpha * dphi0 || (phi_prev < phi_a && a_prev > 0.0) {
            return zoom(
                phi, phi0, dphi0, c1, c2, a_prev, phi_prev, dphi_prev, alpha, phi_a, dphi_a,
                max_evals, sx, sg,
            );
        }
        if dphi_a.abs() <= -c2 * dphi0 {
            return Some(alpha);
        }
        if dphi_a >= 0.0 {
            return zoom(
                phi, phi0, dphi0, c1, c2, alpha, phi_a, dphi_a, a_prev, phi_prev, dphi_prev,
                max_evals, sx, sg,
            );
        }
        a_prev = alpha;
        phi_prev = phi_a;
        dphi_prev = dphi_a;
        alpha *= 2.0;
        if alpha > 1e10 {
            return None;
        }
    }
    None
}

/// Bisection-zoom phase (quadratic-safed bisection keeps the bracket).
#[allow(clippy::too_many_arguments)]
fn zoom(
    phi: &mut impl FnMut(f64, &mut [f64], &mut [f64]) -> (f64, f64),
    phi0: f64,
    dphi0: f64,
    c1: f64,
    c2: f64,
    mut alo: f64,
    mut phi_lo: f64,
    mut dphi_lo: f64,
    mut ahi: f64,
    mut phi_hi: f64,
    mut _dphi_hi: f64,
    max_evals: usize,
    sx: &mut [f64],
    sg: &mut [f64],
) -> Option<f64> {
    for _ in 0..max_evals {
        // Bisection with a quadratic-secant nudge (keeps the bracket valid).
        let mut aj = (alo + ahi) / 2.0;
        if aj == alo || aj == ahi {
            return None;
        }
        let d1 = dphi_lo * (ahi - alo);
        let d2 = phi_hi - phi_lo - d1;
        if d2.abs() > 1e-18 {
            let cand = alo - d1 / (2.0 * d2) * (ahi - alo);
            let (lo, hi) = (alo.min(ahi), alo.max(ahi));
            let pad = 0.1 * (hi - lo);
            if cand > lo + pad && cand < hi - pad {
                aj = cand;
            }
        }
        let (phi_j, dphi_j) = phi(aj, sx, sg);
        if phi_j > phi0 + c1 * aj * dphi0 || phi_j >= phi_lo {
            ahi = aj;
            phi_hi = phi_j;
            _dphi_hi = dphi_j;
        } else {
            if dphi_j.abs() <= -c2 * dphi0 {
                return Some(aj);
            }
            if dphi_j * (ahi - alo) >= 0.0 {
                ahi = alo;
                phi_hi = phi_lo;
                _dphi_hi = dphi_lo;
            }
            alo = aj;
            phi_lo = phi_j;
            dphi_lo = dphi_j;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosenbrock_converges_with_reverse_gradient() {
        let f = Objective::reverse(|x, out| {
            let r1 = 1.0 - x[0].clone();
            let r2 = 10.0 * (x[1].clone() - x[0].clone() * x[0].clone());
            *out = 0.5 * (r1.clone() * r1.clone() + r2.clone() * r2.clone());
        });
        let res = lbfgs(&f, &[-1.2, 1.0], &LbfgsConfig::new()).unwrap();
        assert_eq!(res.status, NlsStatus::Converged, "{:?}", res);
        assert!((res.x[0] - 1.0).abs() < 1e-5 && (res.x[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn analytic_gradient_agrees_with_reverse() {
        let f_rev = Objective::reverse(|x, out| {
            let t = x[0].clone();
            let sq = t.clone() * t.clone();
            *out = t.clone() * (-sq).exp() + (t - 1.0).powi(4);
        });
        let f_ana = Objective::analytic(
            |x: &[f64]| x[0] * (-(x[0] * x[0])).exp() + (x[0] - 1.0).powi(4),
            |x, g| {
                g[0] = (1.0 - 2.0 * x[0] * x[0]) * (-(x[0] * x[0])).exp()
                    + 4.0 * (x[0] - 1.0).powi(3)
            },
        );
        for &x0 in &[-2.0, 0.5, 3.0] {
            let mut g1 = [0.0];
            let v1 = f_rev.eval(&[x0], &mut g1);
            let mut g2 = [0.0];
            let v2 = f_ana.eval(&[x0], &mut g2);
            assert!((v1 - v2).abs() < 1e-12);
            assert!((g1[0] - g2[0]).abs() < 1e-12, "g {} vs {}", g1[0], g2[0]);
        }
    }

    #[test]
    fn fd_gradient_mode_converges() {
        let f = Objective::new(|x: &[f64]| (x[0] - 3.0).powi(2) + 5.0 * (x[1] + 1.0).powi(2));
        let res = lbfgs(&f, &[10.0, 10.0], &LbfgsConfig::new()).unwrap();
        assert!(res.status.has_solution(), "{:?}", res);
        assert!((res.x[0] - 3.0).abs() < 1e-4 && (res.x[1] + 1.0).abs() < 1e-4);
    }

    #[test]
    fn high_dim_quadratic_is_fast() {
        // Min ½‖x − a‖² for n = 200.
        let n = 200;
        let a: Vec<f64> = (0..n).map(|i| (i % 13) as f64 - 6.0).collect();
        let a_check = a.clone();
        let a = std::rc::Rc::new(a);
        let f = Objective::reverse(move |x, out| {
            let mut acc = Value::var(0.0);
            for i in 0..x.len() {
                let d = x[i].clone() - a[i];
                acc = acc + d.clone() * d;
            }
            *out = 0.5 * acc;
        });
        let x0 = vec![0.0; n];
        let res = lbfgs(&f, &x0, &LbfgsConfig::new()).unwrap();
        assert_eq!(res.status, NlsStatus::Converged, "{:?}", res);
        assert!(res.iterations < 50, "took {} iterations", res.iterations);
        for (xk, ak) in res.x.iter().zip(a_check.iter()).take(10) {
            assert!((xk - ak).abs() < 1e-5);
        }
    }

    #[test]
    fn small_memory_still_converges() {
        let f = Objective::reverse(|x, out| {
            let mut acc = Value::var(0.0);
            for i in 0..x.len() {
                acc = acc + (x[i].clone() - i as f64).powi(2);
            }
            *out = acc;
        });
        let res = lbfgs(
            &f,
            &vec![0.0; 60],
            &LbfgsConfig::new().with_history(3).with_gtol(1e-10),
        )
        .unwrap();
        assert!(res.status.has_solution(), "{:?}", res);
        for (i, v) in res.x.iter().enumerate().take(10) {
            assert!((v - i as f64).abs() < 1e-4);
        }
    }
}
