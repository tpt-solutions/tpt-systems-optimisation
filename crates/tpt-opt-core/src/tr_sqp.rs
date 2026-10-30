//! Trust-region sequential quadratic programming for general constrained NLP.
//!
//! [`trust_region_sqp`] minimises `f(x)` subject to `c_i(x) ≤ 0` and `h_j(x) = 0`
//! (the same problem form as [`crate::nlp::solve_nlp`], and the same
//! [`NlpProblem`] trait). Each iteration:
//!
//! 1. Linearises the constraints and solves the trust-region QP
//!    `min ½ dᵀB d + ∇fᵀd + μ·(Σ s_i + Σ (u_j + v_j))` subject to
//!    `c_i + ∇c_iᵀd ≤ s_i`, `h_j + ∇h_jᵀd = u_j − v_j`, `s, u, v ≥ 0` and
//!    `‖d‖∞ ≤ Δ`. The slacks make the QP always feasible, and the exact
//!    `ℓ₁` penalty `μ` keeps it consistent with the merit function.
//! 2. The QP is solved by the interior-point method in [`crate::qp`]. Its
//!    dual multipliers give the Lagrangian used for the quasi-Newton update.
//! 3. Accepts or rejects `x + d` by the ratio of actual to predicted reduction
//!    of the `ℓ₁` merit `f + μ·(Σ max(0, c_i) + Σ |h_j|)`, and adapts `Δ`.
//! 4. Updates `B` by a damped BFGS step on the Lagrangian gradient (Powell's
//!    damping keeps `B` positive definite) and raises `μ` above the multiplier
//!    magnitudes.
//!
//! Requires the `nlp` feature (for the [`NlpProblem`] trait). The QP is dense,
//! so this is intended for problems of moderate size.

use alloc::vec;
use alloc::vec::Vec;

use crate::nlp::{NlpProblem, NlpResult, NlpStatus};
use crate::qp::{solve_qp, QpOptions, QpProblem, QpStatus};

/// Controls for [`trust_region_sqp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrSqpParams {
    /// Constraint-violation tolerance (`ℓ₁` norm of violations, absolute).
    pub feas_tol: f64,
    /// Infinity-norm tolerance on the Lagrangian gradient.
    pub opt_tol: f64,
    /// Maximum number of SQP iterations.
    pub max_iters: usize,
    /// Initial trust-region radius `Δ₀`.
    pub delta0: f64,
    /// Upper limit on the trust-region radius.
    pub delta_max: f64,
}

impl Default for TrSqpParams {
    fn default() -> Self {
        Self { feas_tol: 1e-8, opt_tol: 1e-6, max_iters: 200, delta0: 1.0, delta_max: 100.0 }
    }
}

/// Function values and derivatives at one point.
struct Eval {
    f: f64,
    g: Vec<f64>,
    c: Vec<f64>,
    jc: Vec<Vec<f64>>,
    h: Vec<f64>,
    jh: Vec<Vec<f64>>,
}

fn evaluate<P: NlpProblem>(prob: &P, x: &[f64]) -> Eval {
    let n = prob.num_vars();
    let mut g = vec![0.0; n];
    prob.objective_grad(x, &mut g);
    let m = prob.num_ineq();
    let mut c = Vec::with_capacity(m);
    let mut jc = Vec::with_capacity(m);
    for i in 0..m {
        c.push(prob.ineq(i, x));
        let mut row = vec![0.0; n];
        prob.ineq_grad(i, x, &mut row);
        jc.push(row);
    }
    let p = prob.num_eq();
    let mut h = Vec::with_capacity(p);
    let mut jh = Vec::with_capacity(p);
    for j in 0..p {
        h.push(prob.eq(j, x));
        let mut row = vec![0.0; n];
        prob.eq_grad(j, x, &mut row);
        jh.push(row);
    }
    Eval { f: prob.objective(x), g, c, jc, h, jh }
}

/// `ℓ₁` constraint violation `Σ max(0, c_i) + Σ |h_j|`.
fn violation(c: &[f64], h: &[f64]) -> f64 {
    c.iter().map(|v| v.max(0.0)).sum::<f64>() + h.iter().map(|v| v.abs()).sum::<f64>()
}

/// Solve a general constrained NLP by trust-region SQP.
///
/// Returns [`NlpStatus::Converged`] when the `ℓ₁` violation is within
/// `feas_tol` and the Lagrangian gradient within `opt_tol`. Returns
/// [`NlpStatus::MaxIters`] when `max_iters` is exhausted, and
/// [`NlpStatus::Diverged`] when the trust region collapses without progress.
pub fn trust_region_sqp<P: NlpProblem>(prob: &P, x0: &[f64], params: &TrSqpParams) -> NlpResult {
    let n = prob.num_vars();
    let m = prob.num_ineq();
    let p = prob.num_eq();

    let mut x = x0.to_vec();
    let mut ev = evaluate(prob, &x);
    let mut b = identity(n);
    let mut delta = params.delta0;
    let mut mu = 10.0;
    let mut z = vec![0.0; m];
    let mut y = vec![0.0; p];
    let mut iterations = 0;

    loop {
        let viol = violation(&ev.c, &ev.h);
        let grad_l = lagrangian_grad(&ev, &z, &y, n);
        if viol <= params.feas_tol && norm_inf(&grad_l) <= params.opt_tol {
            return result(x, ev.f, NlpStatus::Converged, iterations);
        }
        if iterations >= params.max_iters {
            return result(x, ev.f, NlpStatus::MaxIters, iterations);
        }
        iterations += 1;

        let qp = build_qp(&ev, &b, mu, delta, n, m, p);
        let sol = solve_qp(&qp, &QpOptions::default());
        if sol.status != QpStatus::Optimal {
            delta *= 0.25;
            if delta < 1e-12 {
                return result(x, ev.f, NlpStatus::Diverged, iterations);
            }
            continue;
        }
        let d: Vec<f64> = sol.x[..n].to_vec();
        let z_new: Vec<f64> = sol.z[..m].to_vec();
        let y_new: Vec<f64> = sol.y.clone();

        // Predicted reduction from the linearised ℓ₁ model.
        let bd = mat_vec(&b, &d);
        let model_change = dot(&ev.g, &d)
            + 0.5 * dot(&d, &bd)
            + mu * (lin_violation(&ev, &d) - violation(&ev.c, &ev.h));
        let pred = -model_change;
        if pred <= 0.0 {
            // No predicted decrease. A zero step with violation left means μ
            // is too small to make the linearisation descend.
            if norm_inf(&d) <= 1e-14 && viol > params.feas_tol {
                mu *= 10.0;
                if !mu.is_finite() || mu > 1e14 {
                    return result(x, ev.f, NlpStatus::Diverged, iterations);
                }
            } else {
                delta *= 0.25;
                if delta < 1e-12 {
                    return result(x, ev.f, NlpStatus::Diverged, iterations);
                }
            }
            continue;
        }

        let x_new: Vec<f64> = x.iter().zip(&d).map(|(xi, di)| xi + di).collect();
        let ev_new = evaluate(prob, &x_new);
        let merit_old = ev.f + mu * violation(&ev.c, &ev.h);
        let merit_new = ev_new.f + mu * violation(&ev_new.c, &ev_new.h);
        let ratio = (merit_old - merit_new) / pred;

        // Multipliers must stay below μ for the ℓ₁ penalty to be exact.
        let lam_max = z_new.iter().chain(&y_new).fold(0.0_f64, |acc, v| acc.max(v.abs()));
        mu = mu.max(1.1 * lam_max);
        z = z_new;
        y = y_new;

        if ratio >= 0.1 {
            let grad_l_old = lagrangian_grad(&ev, &z, &y, n);
            let grad_l_new = lagrangian_grad(&ev_new, &z, &y, n);
            let yv: Vec<f64> = grad_l_new.iter().zip(&grad_l_old).map(|(a, b)| a - b).collect();
            bfgs_update(&mut b, &d, &yv);
            x = x_new;
            ev = ev_new;
        }

        if ratio < 0.25 {
            delta = (0.25 * norm_inf(&d)).max(1e-12);
        } else if ratio > 0.75 && norm_inf(&d) >= 0.99 * delta {
            delta = (2.0 * delta).min(params.delta_max);
        }
    }
}

fn result(x: Vec<f64>, objective: f64, status: NlpStatus, iterations: usize) -> NlpResult {
    NlpResult { x, objective, status, iterations }
}

/// `∇f + Σ z_i ∇c_i + Σ y_j ∇h_j`.
fn lagrangian_grad(ev: &Eval, z: &[f64], y: &[f64], n: usize) -> Vec<f64> {
    let mut out = ev.g.clone();
    for (row, zi) in ev.jc.iter().zip(z) {
        for k in 0..n {
            out[k] += zi * row[k];
        }
    }
    for (row, yj) in ev.jh.iter().zip(y) {
        for k in 0..n {
            out[k] += yj * row[k];
        }
    }
    out
}

/// Linearised violation `Σ max(0, c_i + ∇c_iᵀd) + Σ |h_j + ∇h_jᵀd|`.
fn lin_violation(ev: &Eval, d: &[f64]) -> f64 {
    let ci: f64 = ev.c.iter().zip(&ev.jc).map(|(c, row)| (c + dot(row, d)).max(0.0)).sum();
    let hj: f64 = ev.h.iter().zip(&ev.jh).map(|(h, row)| (h + dot(row, d)).abs()).sum();
    ci + hj
}

/// Dense QP for one trust-region step. Variables are ordered
/// `w = (d, s, u, v)` with lengths `n, m, p, p`.
fn build_qp(
    ev: &Eval,
    b: &[Vec<f64>],
    mu: f64,
    delta: f64,
    n: usize,
    m: usize,
    p: usize,
) -> QpProblem {
    let nv = n + m + 2 * p;
    let mut q = vec![vec![0.0; nv]; nv];
    for (qi, bi) in q.iter_mut().zip(b) {
        qi[..n].copy_from_slice(bi);
    }

    let mut c = vec![0.0; nv];
    c[..n].copy_from_slice(&ev.g);
    for ci in c.iter_mut().skip(n) {
        *ci = mu;
    }

    let mut g_rows: Vec<Vec<f64>> = Vec::new();
    let mut g_rhs: Vec<f64> = Vec::new();
    // Linearised inequalities: ∇c_iᵀd − s_i ≤ −c_i.
    for (i, (row, ci)) in ev.jc.iter().zip(&ev.c).enumerate() {
        let mut r = vec![0.0; nv];
        r[..n].copy_from_slice(row);
        r[n + i] = -1.0;
        g_rows.push(r);
        g_rhs.push(-ci);
    }
    // s_i ≥ 0, u_j ≥ 0, v_j ≥ 0.
    for k in n..nv {
        let mut r = vec![0.0; nv];
        r[k] = -1.0;
        g_rows.push(r);
        g_rhs.push(0.0);
    }
    // Trust region ‖d‖∞ ≤ Δ.
    for k in 0..n {
        let mut up = vec![0.0; nv];
        up[k] = 1.0;
        g_rows.push(up);
        g_rhs.push(delta);
        let mut down = vec![0.0; nv];
        down[k] = -1.0;
        g_rows.push(down);
        g_rhs.push(delta);
    }

    // Linearised equalities: ∇h_jᵀd − u_j + v_j = −h_j.
    let mut a_rows: Vec<Vec<f64>> = Vec::with_capacity(p);
    let mut b_rhs: Vec<f64> = Vec::with_capacity(p);
    for (j, (row, hj)) in ev.jh.iter().zip(&ev.h).enumerate() {
        let mut r = vec![0.0; nv];
        r[..n].copy_from_slice(row);
        r[n + m + j] = -1.0;
        r[n + m + p + j] = 1.0;
        a_rows.push(r);
        b_rhs.push(-hj);
    }

    QpProblem { q, c, g: g_rows, h: g_rhs, a: a_rows, b: b_rhs }
}

/// Damped BFGS update of `B` with step `s` and gradient change `y`.
///
/// Powell's damping replaces `y` by a convex combination with `B s` when the
/// curvature `sᵀy` is too small, so `B` stays positive definite.
fn bfgs_update(b: &mut [Vec<f64>], s: &[f64], y: &[f64]) {
    let bs = mat_vec(b, s);
    let sbs = dot(s, &bs);
    let sy = dot(s, y);
    if sbs <= 1e-14 {
        return;
    }
    let yd: Vec<f64> = if sy < 0.2 * sbs {
        let theta = 0.8 * sbs / (sbs - sy);
        y.iter().zip(&bs).map(|(yi, bi)| theta * yi + (1.0 - theta) * bi).collect()
    } else {
        y.to_vec()
    };
    let syd = dot(s, &yd);
    if syd <= 1e-14 {
        return;
    }
    for (i, row) in b.iter_mut().enumerate() {
        for (j, bij) in row.iter_mut().enumerate() {
            *bij += yd[i] * yd[j] / syd - bs[i] * bs[j] / sbs;
        }
    }
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect()
}

fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, x)).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm_inf(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    type GradFn = fn(&[f64], &mut [f64]);

    struct Problem {
        ineq: Vec<fn(&[f64]) -> f64>,
        ineq_g: Vec<GradFn>,
        eq: Vec<fn(&[f64]) -> f64>,
        eq_g: Vec<GradFn>,
        f: fn(&[f64]) -> f64,
        f_g: fn(&[f64], &mut [f64]),
    }

    impl NlpProblem for Problem {
        fn num_vars(&self) -> usize {
            2
        }
        fn objective(&self, x: &[f64]) -> f64 {
            (self.f)(x)
        }
        fn objective_grad(&self, x: &[f64], g: &mut [f64]) {
            (self.f_g)(x, g)
        }
        fn num_ineq(&self) -> usize {
            self.ineq.len()
        }
        fn ineq(&self, i: usize, x: &[f64]) -> f64 {
            (self.ineq[i])(x)
        }
        fn ineq_grad(&self, i: usize, x: &[f64], row: &mut [f64]) {
            (self.ineq_g[i])(x, row)
        }
        fn num_eq(&self) -> usize {
            self.eq.len()
        }
        fn eq(&self, j: usize, x: &[f64]) -> f64 {
            (self.eq[j])(x)
        }
        fn eq_grad(&self, j: usize, x: &[f64], row: &mut [f64]) {
            (self.eq_g[j])(x, row)
        }
    }

    fn f_dist(x: &[f64]) -> f64 {
        (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2)
    }
    fn f_dist_g(x: &[f64], g: &mut [f64]) {
        g[0] = 2.0 * (x[0] - 2.0);
        g[1] = 2.0 * (x[1] - 1.0);
    }

    #[test]
    fn projection_onto_disk() {
        // min (x₁−2)² + (x₂−1)²  s.t.  x₁² + x₂² ≤ 1.
        // The optimum is the projection of (2, 1) onto the unit circle.
        let prob = Problem {
            ineq: vec![|x| x[0] * x[0] + x[1] * x[1] - 1.0],
            ineq_g: vec![|x, g| {
                g[0] = 2.0 * x[0];
                g[1] = 2.0 * x[1];
            }],
            eq: vec![],
            eq_g: vec![],
            f: f_dist,
            f_g: f_dist_g,
        };
        let r = trust_region_sqp(&prob, &[0.0, 0.0], &TrSqpParams::default());
        assert_eq!(r.status, NlpStatus::Converged, "{r:?}");
        let norm = (2.0_f64.powi(2) + 1.0).sqrt();
        assert!((r.x[0] - 2.0 / norm).abs() < 1e-5, "{r:?}");
        assert!((r.x[1] - 1.0 / norm).abs() < 1e-5, "{r:?}");
    }

    #[test]
    fn equality_constrained_projection() {
        // min (x₁−2)² + (x₂−1)²  s.t.  x₁ + x₂ = 1.
        // Projection of (2, 1) onto the line x₁ + x₂ = 1: x = (1, 0).
        let prob = Problem {
            ineq: vec![],
            ineq_g: vec![],
            eq: vec![|x| x[0] + x[1] - 1.0],
            eq_g: vec![|_, g| {
                g[0] = 1.0;
                g[1] = 1.0;
            }],
            f: f_dist,
            f_g: f_dist_g,
        };
        let r = trust_region_sqp(&prob, &[0.0, 0.0], &TrSqpParams::default());
        assert_eq!(r.status, NlpStatus::Converged, "{r:?}");
        assert!((r.x[0] - 1.0).abs() < 1e-6, "{r:?}");
        assert!(r.x[1].abs() < 1e-6, "{r:?}");
    }

    #[test]
    fn rosenbrock_with_disk_constraint() {
        // min (1−x₁)² + 100(x₂−x₁²)²  s.t.  x₁² + x₂² ≤ 2.
        // Unconstrained minimiser (1, 1) is feasible, so the answer is (1, 1).
        let prob = Problem {
            ineq: vec![|x| x[0] * x[0] + x[1] * x[1] - 2.0],
            ineq_g: vec![|x, g| {
                g[0] = 2.0 * x[0];
                g[1] = 2.0 * x[1];
            }],
            eq: vec![],
            eq_g: vec![],
            f: |x| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
            f_g: |x, g| {
                let b = x[1] - x[0] * x[0];
                g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * b;
                g[1] = 200.0 * b;
            },
        };
        let params = TrSqpParams { max_iters: 500, ..TrSqpParams::default() };
        let r = trust_region_sqp(&prob, &[-1.2, 1.0], &params);
        assert!((r.x[0] - 1.0).abs() < 1e-4, "{r:?}");
        assert!((r.x[1] - 1.0).abs() < 1e-4, "{r:?}");
    }

    #[test]
    fn feasibility_is_restored_from_infeasible_start() {
        // Start far outside x₁ ≤ 1; the optimum of min (x₁−2)² on the box is 1.
        let prob = Problem {
            ineq: vec![|x| x[0] - 1.0],
            ineq_g: vec![|_, g| {
                g[0] = 1.0;
                g[1] = 0.0;
            }],
            eq: vec![],
            eq_g: vec![],
            f: |x| (x[0] - 2.0).powi(2),
            f_g: |x, g| {
                g[0] = 2.0 * (x[0] - 2.0);
                g[1] = 0.0;
            },
        };
        let r = trust_region_sqp(&prob, &[5.0, 0.0], &TrSqpParams::default());
        assert_eq!(r.status, NlpStatus::Converged, "{r:?}");
        assert!((r.x[0] - 1.0).abs() < 1e-6, "{r:?}");
    }
}
