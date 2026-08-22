//! αBB-style convex underestimators for twice-differentiable terms.
//!
//! Given a term `f` on a box `[l, u]` and a lower bound `α >= 0` on the
//! *negated* curvature of `f` (i.e. `Hessian(f) + 2αI ⪰ 0` on the box), the
//! function
//!
//! ```text
//! g(x) = f(x) − α · Σ_i (u_i − x_i)(x_i − l_i)
//! ```
//!
//! is convex on the box, because `(u−x)(x−l)` has Hessian `-2I`. Any tangent
//! of `g` is therefore a valid linear underestimator of `f` on the box —
//! exactly the kind of cut an outer-approximation master can consume.

use std::vec::Vec;

/// Build the αBB convex underestimator `g(x) = f(x) − α Σ (u_i − x_i)(x_i − l_i)`.
///
/// `grad_f` evaluates the gradient of `f`.
pub fn alphabb_underestimator(
    f: impl Fn(&[f64]) -> f64 + 'static,
    _grad_f: impl Fn(&[f64], &mut [f64]) + 'static + Clone,
    alpha: f64,
    l: Vec<f64>,
    u: Vec<f64>,
) -> impl Fn(&[f64]) -> f64 {
    move |x: &[f64]| {
        let mut penalty = 0.0;
        for i in 0..x.len() {
            penalty += (u[i] - x[i]) * (x[i] - l[i]);
        }
        f(x) - alpha * penalty
    }
}

/// Convenience: build the tangent-cut row given `f`, its gradient, the
/// curvature bound `alpha` and the box. Returns `(coefs, ge=true, rhs)` in
/// the same `(idx, coef, rhs)`-free dense form used by [`crate::oa`]: the
/// row is `Σ s_i x_i >= rhs` and guarantees `f(x) >= rhs + Σ s_i x_i` for
/// all `x` in the box whenever `Hess(f) + 2αI ⪰ 0` there.
pub fn alphabb_cut(
    f: impl Fn(&[f64]) -> f64,
    grad_f: impl Fn(&[f64], &mut [f64]),
    alpha: f64,
    l: &[f64],
    u: &[f64],
    m: &[f64],
) -> (Vec<f64>, f64) {
    let n = m.len();
    let fm = f(m);
    let mut gf = vec![0.0f64; n];
    grad_f(m, &mut gf);
    // ∇g(m) = ∇f(m) − α (u + l − 2m)
    let mut s = vec![0.0f64; n];
    let mut gm = fm;
    for i in 0..n {
        s[i] = gf[i] - alpha * (u[i] + l[i] - 2.0 * m[i]);
        gm -= alpha * (u[i] - m[i]) * (m[i] - l[i]);
    }
    // Bound: f(x) >= g(m) + s·(x − m) = (gm − s·m) + s·x
    let rhs = gm - dot(&s, m);
    (s, rhs)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concave_term_gets_valid_linear_bound() {
        // f = −x² on [−1,1]: Hess = −2 ⇒ α = 1 makes g convex (in fact
        // constant −1). The tangent cut must satisfy f(x) >= rhs + s·x.
        let f = |x: &[f64]| -x[0] * x[0];
        let grad = |x: &[f64], g: &mut [f64]| g[0] = -2.0 * x[0];
        let (s, rhs) = alphabb_cut(f, grad, 1.0, &[-1.0], &[1.0], &[0.0]);
        assert!(s[0].abs() < 1e-9, "constant underestimator has zero slope");
        assert!((rhs - (-1.0)).abs() < 1e-9, "bound should be −1, got {rhs}");
        for x in [-1.0, -0.5, 0.0, 0.3, 1.0] {
            assert!(f(&[x]) >= rhs + s[0] * x - 1e-9, "cut invalid at {x}");
        }
    }

    #[test]
    fn convex_term_alpha_zero_is_exact_tangent() {
        // f = x² on [0,4], α = 0: tangent at m=1 is 2x − 1 ≤ x² everywhere.
        let f = |x: &[f64]| x[0] * x[0];
        let grad = |x: &[f64], g: &mut [f64]| g[0] = 2.0 * x[0];
        let (s, rhs) = alphabb_cut(f, grad, 0.0, &[0.0], &[4.0], &[1.0]);
        assert!((s[0] - 2.0).abs() < 1e-12);
        assert!((rhs - (-1.0)).abs() < 1e-12);
        for x in [0.0, 0.5, 1.0, 2.5, 4.0] {
            assert!(f(&[x]) >= rhs + s[0] * x - 1e-9, "tangent invalid at {x}");
        }
    }

    #[test]
    fn underestimator_is_convexified() {
        // For f = −x² with α = 1 the underestimator is identically −1.
        let g = alphabb_underestimator(|x| -x[0] * x[0], |_x, _g| {}, 1.0, vec![-1.0], vec![1.0]);
        for x in [-1.0, 0.0, 1.0] {
            assert!((g(&[x]) - (-1.0)).abs() < 1e-12);
        }
    }
}
