//! Unconstrained minimisation: derivative-free and quasi-Newton methods.
//!
//! - [`nelder_mead`] — simplex method (Nelder & Mead 1965) with the standard
//!   reflection / expansion / contraction / shrink moves. Derivative-free.
//! - [`powell`] — conjugate-direction method (Powell 1964) with bracketed
//!   golden-section line minimisation and the direction-replacement test from
//!   Numerical Recipes. Derivative-free.
//! - [`lbfgs`] — limited-memory BFGS with a two-loop recursion and a
//!   backtracking Armijo line search. Requires a gradient.
//!
//! All three are deterministic and allocate only their working vectors. They
//! minimise `f(x)` with no constraints; for bound or general constraints use
//! [`crate::nlp`] or `tpt-opt-nls`.

use alloc::vec;
use alloc::vec::Vec;

/// Stopping and budget controls shared by the methods in this module.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UncParams {
    /// Maximum number of outer iterations.
    pub max_iters: usize,
    /// Relative objective-decrease tolerance.
    pub f_tol: f64,
    /// Step / simplex-diameter tolerance.
    pub x_tol: f64,
    /// Infinity-norm gradient tolerance ([`lbfgs`] only).
    pub g_tol: f64,
}

impl Default for UncParams {
    fn default() -> Self {
        Self { max_iters: 1000, f_tol: 1e-12, x_tol: 1e-10, g_tol: 1e-8 }
    }
}

/// Outcome of an unconstrained minimisation.
#[derive(Clone, Debug, PartialEq)]
pub struct UncResult {
    /// Best point found.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub f: f64,
    /// Outer iterations performed.
    pub iterations: usize,
    /// Objective evaluations performed (gradient evaluations are counted once
    /// per call to the combined `f`/`g` closure in [`lbfgs`]).
    pub evaluations: usize,
    /// `true` if a tolerance was met, `false` if the iteration budget ran out
    /// or the line search failed to make progress.
    pub converged: bool,
}

/// Minimise `f` with the Nelder-Mead simplex method.
///
/// The initial simplex is `x0` plus `x0 + step·eᵢ` for each coordinate `i`.
/// Terminates when both the spread of simplex objective values and the
/// simplex diameter fall below `f_tol` and `x_tol`.
///
/// # Panics
///
/// Panics if `x0` is empty.
pub fn nelder_mead<F>(mut f: F, x0: &[f64], step: f64, params: &UncParams) -> UncResult
where
    F: FnMut(&[f64]) -> f64,
{
    assert!(!x0.is_empty(), "nelder_mead needs at least one variable");
    const ALPHA: f64 = 1.0; // reflection
    const GAMMA: f64 = 2.0; // expansion
    const RHO: f64 = 0.5; // contraction
    const SIGMA: f64 = 0.5; // shrink

    let n = x0.len();
    let mut evaluations = 0;
    let mut pts: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n + 1);
    pts.push((x0.to_vec(), 0.0));
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += step;
        pts.push((v, 0.0));
    }
    for p in pts.iter_mut() {
        p.1 = f(&p.0);
        evaluations += 1;
    }

    let mut iterations = 0;
    let mut converged = false;
    while iterations < params.max_iters {
        pts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        let f_spread = pts[n].1 - pts[0].1;
        let x_spread =
            pts[1..].iter().map(|p| max_abs_diff(&p.0, &pts[0].0)).fold(0.0_f64, f64::max);
        if f_spread.abs() <= params.f_tol && x_spread <= params.x_tol {
            converged = true;
            break;
        }
        iterations += 1;

        let mut c = vec![0.0; n];
        for p in &pts[..n] {
            for (cj, xj) in c.iter_mut().zip(&p.0) {
                *cj += xj / n as f64;
            }
        }
        let worst = pts[n].0.clone();
        let f_worst = pts[n].1;
        let f_second_worst = pts[n - 1].1;
        let f_best = pts[0].1;

        let xr = lerp(&c, &worst, -ALPHA);
        let fr = f(&xr);
        evaluations += 1;

        if fr < f_best {
            let xe = lerp(&c, &xr, GAMMA);
            let fe = f(&xe);
            evaluations += 1;
            pts[n] = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else if fr < f_second_worst {
            pts[n] = (xr, fr);
        } else {
            let (xc, fc, accept) = if fr < f_worst {
                let xc = lerp(&c, &xr, RHO);
                let fc = f(&xc);
                evaluations += 1;
                (xc, fc, fc <= fr)
            } else {
                let xc = lerp(&c, &worst, RHO);
                let fc = f(&xc);
                evaluations += 1;
                (xc, fc, fc < f_worst)
            };
            if accept {
                pts[n] = (xc, fc);
            } else {
                let best = pts[0].0.clone();
                for p in pts[1..].iter_mut() {
                    p.0 = lerp(&best, &p.0, SIGMA);
                    p.1 = f(&p.0);
                    evaluations += 1;
                }
            }
        }
    }

    pts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    let (x, f_best) = pts.swap_remove(0);
    UncResult { x, f: f_best, iterations, evaluations, converged }
}

/// Minimise `f` with Powell's conjugate-direction method.
///
/// Starts from the coordinate axes as the direction set. Each sweep performs a
/// line minimisation along every direction; the extrapolated-point test then
/// decides whether the sweep's net displacement replaces the direction of
/// largest decrease. Line minimisation uses bracketing plus golden-section
/// search, so `f` should be continuous; it need not be differentiable.
///
/// # Panics
///
/// Panics if `x0` is empty.
pub fn powell<F>(mut f: F, x0: &[f64], params: &UncParams) -> UncResult
where
    F: FnMut(&[f64]) -> f64,
{
    assert!(!x0.is_empty(), "powell needs at least one variable");
    let n = x0.len();
    let mut evaluations = 0;
    let mut dirs: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut e = vec![0.0; n];
            e[i] = 1.0;
            e
        })
        .collect();

    let mut x = x0.to_vec();
    let mut fx = f(&x);
    evaluations += 1;

    let mut iterations = 0;
    let mut converged = false;
    while iterations < params.max_iters {
        iterations += 1;
        let x_start = x.clone();
        let f_start = fx;
        let mut largest_drop = 0.0_f64;
        let mut largest_idx = 0;

        for (i, d) in dirs.iter().enumerate() {
            let before = fx;
            let (xn, fn_, evals) = line_min(&mut f, &x, d, params.x_tol);
            evaluations += evals;
            x = xn;
            fx = fn_;
            if before - fx > largest_drop {
                largest_drop = before - fx;
                largest_idx = i;
            }
        }

        let decrease = f_start - fx;
        if 2.0 * decrease <= params.f_tol * (f_start.abs() + fx.abs()) + f64::MIN_POSITIVE
            || max_abs_diff(&x, &x_start) <= params.x_tol
        {
            converged = true;
            break;
        }

        // Extrapolated point along the net displacement of this sweep.
        let xe: Vec<f64> = x.iter().zip(&x_start).map(|(a, b)| 2.0 * a - b).collect();
        let fe = f(&xe);
        evaluations += 1;

        let replace = fe < f_start && {
            let t = f_start - 2.0 * fx + fe;
            let u = f_start - fx - largest_drop;
            2.0 * t * u * u < largest_drop * (f_start - fe) * (f_start - fe)
        };
        if replace {
            let d: Vec<f64> = x.iter().zip(&x_start).map(|(a, b)| a - b).collect();
            let norm = norm2(&d);
            if norm > 0.0 {
                let d: Vec<f64> = d.iter().map(|v| v / norm).collect();
                let (xn, fn_, evals) = line_min(&mut f, &x, &d, params.x_tol);
                evaluations += evals;
                x = xn;
                fx = fn_;
                let last = dirs.len() - 1;
                dirs.swap(largest_idx, last);
                dirs[last] = d;
            }
        }
    }

    UncResult { x, f: fx, iterations, evaluations, converged }
}

/// Minimise `f` with limited-memory BFGS.
///
/// `fg` evaluates the objective and writes its gradient into the second
/// argument, returning the objective value. `memory` is the number of `(s, y)`
/// curvature pairs kept (typically 5–20). The search direction is the two-loop
/// recursion applied to the gradient; a backtracking Armijo line search picks
/// the step. Stops when `‖g‖∞ ≤ g_tol`, when the relative objective decrease
/// and the step both fall below `f_tol` and `x_tol`, or when the line search
/// cannot make progress.
///
/// # Panics
///
/// Panics if `x0` is empty or `memory` is zero.
pub fn lbfgs<F>(mut fg: F, x0: &[f64], memory: usize, params: &UncParams) -> UncResult
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    assert!(!x0.is_empty(), "lbfgs needs at least one variable");
    assert!(memory > 0, "lbfgs needs a non-zero memory");
    const C1: f64 = 1e-4;
    const MAX_BACKTRACKS: usize = 60;

    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g = vec![0.0; n];
    let mut evaluations = 1;
    let mut fx = fg(&x, &mut g);

    // Curvature history, oldest first: (s, y, 1 / sᵀy).
    let mut hist: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::with_capacity(memory);
    let mut iterations = 0;
    let mut converged = false;

    while iterations < params.max_iters {
        if norm_inf(&g) <= params.g_tol {
            converged = true;
            break;
        }
        iterations += 1;

        // Two-loop recursion: d = -H·g.
        let mut q = g.clone();
        let mut alpha = vec![0.0; hist.len()];
        for (i, (s, y, rho)) in hist.iter().enumerate().rev() {
            let a = rho * dot(s, &q);
            alpha[i] = a;
            axpy(-a, y, &mut q);
        }
        let gamma = match hist.last() {
            Some((s, y, _)) => dot(s, y) / dot(y, y),
            None => 1.0 / norm2(&g).max(1.0),
        };
        let mut r: Vec<f64> = q.iter().map(|v| gamma * v).collect();
        for (i, (s, y, rho)) in hist.iter().enumerate() {
            let b = rho * dot(y, &r);
            axpy(alpha[i] - b, s, &mut r);
        }
        let mut d: Vec<f64> = r.iter().map(|v| -v).collect();
        let mut slope = dot(&d, &g);
        if slope >= 0.0 {
            // Not a descent direction: fall back to steepest descent and
            // discard the (possibly corrupt) curvature history.
            d = g.iter().map(|v| -v).collect();
            slope = -dot(&g, &g);
            hist.clear();
        }

        // Backtracking Armijo line search.
        let mut t = 1.0;
        let mut accepted = None;
        let mut xn = vec![0.0; n];
        let mut gn = vec![0.0; n];
        for _ in 0..MAX_BACKTRACKS {
            for ((xi, x0i), di) in xn.iter_mut().zip(&x).zip(&d) {
                *xi = x0i + t * di;
            }
            let fnew = fg(&xn, &mut gn);
            evaluations += 1;
            if fnew <= fx + C1 * t * slope {
                accepted = Some(fnew);
                break;
            }
            t *= 0.5;
        }
        let Some(fnew) = accepted else {
            break;
        };

        let s: Vec<f64> = xn.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = gn.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > f64::EPSILON * dot(&y, &y) {
            if hist.len() == memory {
                hist.remove(0);
            }
            hist.push((s.clone(), y, 1.0 / sy));
        }

        let f_drop = (fx - fnew).abs();
        let step_norm = norm2(&s);
        x = xn;
        g = gn;
        fx = fnew;
        if f_drop <= params.f_tol * (1.0 + fx.abs())
            && step_norm <= params.x_tol * (1.0 + norm2(&x))
        {
            converged = true;
            break;
        }
    }

    UncResult { x, f: fx, iterations, evaluations, converged }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// `a + t·(b − a)` as a new vector.
fn lerp(a: &[f64], b: &[f64], t: f64) -> Vec<f64> {
    a.iter().zip(b).map(|(ai, bi)| ai + t * (bi - ai)).collect()
}

/// Minimise `φ(t) = f(x + t·d)` over `t`, returning `(x_new, f_new, evals)`.
///
/// Brackets a minimum by doubling from `t = 0`, then refines by golden-section
/// search. The bracket is only as good as the unimodality of `φ` along `d`.
fn line_min<F>(f: &mut F, x: &[f64], d: &[f64], x_tol: f64) -> (Vec<f64>, f64, usize)
where
    F: FnMut(&[f64]) -> f64,
{
    const GOLD: f64 = 0.618_033_988_749_894_9;
    const MAX_EXPAND: usize = 60;
    const MAX_GOLD: usize = 200;

    let mut evals = 0;
    let mut phi = |t: f64, evals: &mut usize| {
        *evals += 1;
        let p: Vec<f64> = x.iter().zip(d).map(|(xi, di)| xi + t * di).collect();
        f(&p)
    };

    let f0 = phi(0.0, &mut evals);
    let f1 = phi(1.0, &mut evals);
    let (mut lo, mut hi) = if f1 < f0 {
        let mut f_prev = f1;
        let mut next = 2.0;
        for _ in 0..MAX_EXPAND {
            let f_next = phi(next, &mut evals);
            if f_next >= f_prev {
                break;
            }
            f_prev = f_next;
            next *= 2.0;
        }
        (0.0, next)
    } else {
        let fm = phi(-1.0, &mut evals);
        if fm < f0 {
            let mut f_prev = fm;
            let mut next = -2.0;
            for _ in 0..MAX_EXPAND {
                let f_next = phi(next, &mut evals);
                if f_next >= f_prev {
                    break;
                }
                f_prev = f_next;
                next *= 2.0;
            }
            (next, 0.0)
        } else {
            (-1.0, 1.0)
        }
    };

    // Golden-section refinement on [lo, hi].
    let t_tol = x_tol.max(f64::EPSILON);
    let mut c = hi - GOLD * (hi - lo);
    let mut e = lo + GOLD * (hi - lo);
    let mut fc = phi(c, &mut evals);
    let mut fe = phi(e, &mut evals);
    for _ in 0..MAX_GOLD {
        if (hi - lo).abs() <= t_tol * (1.0 + lo.abs() + hi.abs()) {
            break;
        }
        if fc < fe {
            hi = e;
            e = c;
            fe = fc;
            c = hi - GOLD * (hi - lo);
            fc = phi(c, &mut evals);
        } else {
            lo = c;
            c = e;
            fc = fe;
            e = lo + GOLD * (hi - lo);
            fe = phi(e, &mut evals);
        }
    }

    let (t_best, f_best) = if fc < fe { (c, fc) } else { (e, fe) };
    let (t_best, f_best) = if f_best < f0 { (t_best, f_best) } else { (0.0, f0) };
    let x_new: Vec<f64> = x.iter().zip(d).map(|(xi, di)| xi + t_best * di).collect();
    (x_new, f_best, evals)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn norm_inf(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

/// `y ← y + a·x`.
fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(x: &[f64]) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }

    fn rosenbrock_grad(x: &[f64], g: &mut [f64]) -> f64 {
        let b = x[1] - x[0] * x[0];
        g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * b;
        g[1] = 200.0 * b;
        rosenbrock(x)
    }

    #[test]
    fn nelder_mead_rosenbrock() {
        let p = UncParams { max_iters: 5000, ..UncParams::default() };
        let r = nelder_mead(rosenbrock, &[-1.2, 1.0], 0.1, &p);
        assert!(r.converged, "{r:?}");
        assert!((r.x[0] - 1.0).abs() < 1e-4, "{r:?}");
        assert!((r.x[1] - 1.0).abs() < 1e-4, "{r:?}");
    }

    #[test]
    fn nelder_mead_shifted_quadratic() {
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + 2.0 * (x[1] + 1.0).powi(2) + 5.0;
        let r = nelder_mead(f, &[0.0, 0.0], 1.0, &UncParams::default());
        assert!(r.converged);
        assert!((r.x[0] - 3.0).abs() < 1e-6);
        assert!((r.x[1] + 1.0).abs() < 1e-6);
        assert!((r.f - 5.0).abs() < 1e-10);
    }

    #[test]
    fn powell_quadratic_coupled() {
        // Coupled quadratic: coordinate descent alone would zig-zag, so this
        // exercises the direction-replacement step.
        let f = |x: &[f64]| {
            let u = x[0] - 1.0;
            let v = x[1] - 2.0;
            u * u + 4.0 * v * v + 1.8 * u * v
        };
        let r = powell(f, &[0.0, 0.0], &UncParams::default());
        assert!(r.converged, "{r:?}");
        // Minimiser is c = (1, 2): the gradient vanishes there.
        assert!((r.x[0] - 1.0).abs() < 1e-6, "{r:?}");
        assert!((r.x[1] - 2.0).abs() < 1e-6, "{r:?}");
    }

    #[test]
    fn powell_rosenbrock_makes_progress() {
        let start = rosenbrock(&[-1.2, 1.0]);
        let p = UncParams { max_iters: 500, ..UncParams::default() };
        let r = powell(rosenbrock, &[-1.2, 1.0], &p);
        assert!(r.f < 1e-3 * start, "{r:?}");
        assert!(rosenbrock(&r.x) == r.f);
    }

    #[test]
    fn lbfgs_rosenbrock() {
        let r = lbfgs(rosenbrock_grad, &[-1.2, 1.0], 8, &UncParams::default());
        assert!(r.converged, "{r:?}");
        assert!((r.x[0] - 1.0).abs() < 1e-5, "{r:?}");
        assert!((r.x[1] - 1.0).abs() < 1e-5, "{r:?}");
    }

    #[test]
    fn lbfgs_high_dimensional_ill_conditioned_quadratic() {
        // f(x) = Σ i·(xᵢ − i)² with condition number n = 40.
        let n = 40;
        let fg = |x: &[f64], g: &mut [f64]| {
            let mut f = 0.0;
            for i in 0..n {
                let w = (i + 1) as f64;
                let d = x[i] - w;
                f += w * d * d;
                g[i] = 2.0 * w * d;
            }
            f
        };
        let p = UncParams { max_iters: 500, g_tol: 1e-10, ..UncParams::default() };
        let r = lbfgs(fg, &vec![0.0; n], 10, &p);
        assert!(r.converged, "{r:?}");
        for (i, xi) in r.x.iter().enumerate() {
            assert!((xi - (i + 1) as f64).abs() < 1e-6, "coord {i}: {xi}");
        }
    }

    #[test]
    fn lbfgs_is_deterministic_and_monotone() {
        let a = lbfgs(rosenbrock_grad, &[-1.2, 1.0], 5, &UncParams::default());
        let b = lbfgs(rosenbrock_grad, &[-1.2, 1.0], 5, &UncParams::default());
        assert_eq!(a, b);
        assert!(a.f <= rosenbrock(&[-1.2, 1.0]));
    }
}
