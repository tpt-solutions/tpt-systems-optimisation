//! Primal-dual interior-point method for convex quadratic programs.
//!
//! [`solve_qp`] minimises `½ xᵀ Q x + cᵀ x` subject to `G x ≤ h` and `A x = b`,
//! with `Q` symmetric positive semidefinite. The method is Mehrotra's
//! predictor-corrector: each iteration solves one reduced KKT system for the
//! Newton direction (the inequality slacks and duals are eliminated), picks a
//! centring parameter from the affine-scaling step, and takes a common step
//! length that keeps slacks and duals strictly positive.
//!
//! Bounds `l ≤ x ≤ u` are expressed as rows of `G`. The linear algebra is dense
//! (an LU factorisation with partial pivoting of the `(n + p)`-square KKT
//! matrix), so this is intended for moderate sizes; it is a building block for
//! trust-region SQP and conic solvers, not a large-scale sparse QP code.
//!
//! The solver does not produce infeasibility certificates. When the problem is
//! infeasible or unbounded the iterates typically fail to converge and the
//! status is [`QpStatus::MaxIterations`] or [`QpStatus::Numerical`].

use alloc::vec;
use alloc::vec::Vec;

/// A convex quadratic program in dense form.
///
/// `q` is `n × n`, `c` has length `n`, `g` has `m` rows of length `n` with
/// `h` of length `m`, and `a` has `p` rows of length `n` with `b` of length `p`.
#[derive(Clone, Debug, PartialEq)]
pub struct QpProblem {
    /// Hessian `Q` (symmetric positive semidefinite).
    pub q: Vec<Vec<f64>>,
    /// Linear objective `c`.
    pub c: Vec<f64>,
    /// Inequality matrix `G` in `G x ≤ h`.
    pub g: Vec<Vec<f64>>,
    /// Inequality right-hand side `h`.
    pub h: Vec<f64>,
    /// Equality matrix `A` in `A x = b`.
    pub a: Vec<Vec<f64>>,
    /// Equality right-hand side `b`.
    pub b: Vec<f64>,
}

/// Options for [`solve_qp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QpOptions {
    /// Relative tolerance on the dual residual, primal residuals and gap.
    pub tol: f64,
    /// Maximum number of interior-point iterations.
    pub max_iter: usize,
}

impl Default for QpOptions {
    fn default() -> Self {
        Self { tol: 1e-9, max_iter: 100 }
    }
}

/// Termination status of [`solve_qp`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QpStatus {
    /// Dual, primal and complementarity residuals are within tolerance.
    Optimal,
    /// The iteration budget ran out before convergence.
    MaxIterations,
    /// The KKT system became singular or non-finite.
    Numerical,
    /// The problem dimensions are inconsistent.
    InvalidInput,
}

/// Result of [`solve_qp`].
#[derive(Clone, Debug, PartialEq)]
pub struct QpSolution {
    /// Primal solution `x`.
    pub x: Vec<f64>,
    /// Objective value `½ xᵀ Q x + cᵀ x`.
    pub objective: f64,
    /// Multipliers `y` of the equality constraints.
    pub y: Vec<f64>,
    /// Multipliers `z ≥ 0` of the inequality constraints.
    pub z: Vec<f64>,
    /// Termination status.
    pub status: QpStatus,
    /// Interior-point iterations performed.
    pub iterations: usize,
}

/// Solve a convex quadratic program with a primal-dual interior-point method.
///
/// Starts from `x = 0`, slacks `s = max(h − G x, 1)` and unit inequality duals.
/// Stops when the dual residual, both primal residuals and the complementarity
/// gap `sᵀz` are each below `tol` relative to the data scale.
pub fn solve_qp(prob: &QpProblem, opts: &QpOptions) -> QpSolution {
    let n = prob.c.len();
    let m = prob.h.len();
    let p = prob.b.len();
    if !dims_ok(prob, n, m, p) {
        return failed(QpStatus::InvalidInput, n, p, m, 0);
    }

    let mut x = vec![0.0; n];
    let mut y = vec![0.0; p];
    let mut z = vec![1.0; m];
    let gx0 = mat_vec(&prob.g, &x);
    let mut s: Vec<f64> = (0..m).map(|i| (prob.h[i] - gx0[i]).max(1.0)).collect();

    let scale_c = norm_inf(&prob.c).max(1.0);
    let scale_b = norm_inf(&prob.b).max(1.0);
    let scale_h = norm_inf(&prob.h).max(1.0);

    let mut iterations = 0;
    loop {
        let res = Residuals::compute(prob, &x, &y, &z, &s);
        let obj = objective(prob, &x);
        let gap = dot(&s, &z);
        let converged = norm_inf(&res.rc) <= opts.tol * scale_c
            && norm_inf(&res.rb) <= opts.tol * scale_b
            && norm_inf(&res.rh) <= opts.tol * scale_h
            && gap <= opts.tol * (1.0 + obj.abs());
        if converged {
            return QpSolution { x, objective: obj, y, z, status: QpStatus::Optimal, iterations };
        }
        if iterations >= opts.max_iter {
            return QpSolution {
                x,
                objective: obj,
                y,
                z,
                status: QpStatus::MaxIterations,
                iterations,
            };
        }
        iterations += 1;

        let mu = if m > 0 { gap / m as f64 } else { 0.0 };

        // Predictor (affine-scaling) direction.
        let sz: Vec<f64> = s.iter().zip(&z).map(|(a, b)| a * b).collect();
        let Some(aff) = newton(prob, &x, &z, &s, &res, &sz) else {
            return failed(QpStatus::Numerical, n, p, m, iterations);
        };
        let a_aff = step_len(&s, &aff.ds, &z, &aff.dz);
        let mu_aff = if m > 0 {
            let mut acc = 0.0;
            for i in 0..m {
                acc += (s[i] + a_aff * aff.ds[i]) * (z[i] + a_aff * aff.dz[i]);
            }
            acc / m as f64
        } else {
            0.0
        };
        let sigma = if mu > 0.0 { (mu_aff / mu).powi(3).clamp(0.0, 1.0) } else { 0.0 };

        // Corrector: centring plus second-order term from the predictor.
        let rcomp: Vec<f64> = (0..m).map(|i| sz[i] - sigma * mu + aff.ds[i] * aff.dz[i]).collect();
        let Some(dir) = newton(prob, &x, &z, &s, &res, &rcomp) else {
            return failed(QpStatus::Numerical, n, p, m, iterations);
        };
        let alpha = (0.995 * step_len(&s, &dir.ds, &z, &dir.dz)).min(1.0);

        for (xi, di) in x.iter_mut().zip(&dir.dx) {
            *xi += alpha * di;
        }
        for (yi, di) in y.iter_mut().zip(&dir.dy) {
            *yi += alpha * di;
        }
        for (si, di) in s.iter_mut().zip(&dir.ds) {
            *si += alpha * di;
        }
        for (zi, di) in z.iter_mut().zip(&dir.dz) {
            *zi += alpha * di;
        }
        if !x.iter().chain(&y).chain(&s).chain(&z).all(|v| v.is_finite()) {
            return failed(QpStatus::Numerical, n, p, m, iterations);
        }
    }
}

/// Residuals of the KKT conditions at the current iterate.
struct Residuals {
    /// Stationarity: `Q x + c + Gᵀ z + Aᵀ y`.
    rc: Vec<f64>,
    /// Equality feasibility: `A x − b`.
    rb: Vec<f64>,
    /// Inequality feasibility: `G x + s − h`.
    rh: Vec<f64>,
}

impl Residuals {
    fn compute(prob: &QpProblem, x: &[f64], y: &[f64], z: &[f64], s: &[f64]) -> Self {
        let qx = mat_vec(&prob.q, x);
        let gtz = mat_t_vec(&prob.g, z, x.len());
        let aty = mat_t_vec(&prob.a, y, x.len());
        let rc = (0..x.len()).map(|j| qx[j] + prob.c[j] + gtz[j] + aty[j]).collect();
        let ax = mat_vec(&prob.a, x);
        let rb = (0..prob.b.len()).map(|j| ax[j] - prob.b[j]).collect();
        let gx = mat_vec(&prob.g, x);
        let rh = (0..prob.h.len()).map(|i| gx[i] + s[i] - prob.h[i]).collect();
        Self { rc, rb, rh }
    }
}

/// Newton direction for one complementarity target `r_comp` (so that
/// `Z ds + S dz = −r_comp`).
struct Direction {
    dx: Vec<f64>,
    dy: Vec<f64>,
    ds: Vec<f64>,
    dz: Vec<f64>,
}

fn newton(
    prob: &QpProblem,
    x: &[f64],
    z: &[f64],
    s: &[f64],
    res: &Residuals,
    r_comp: &[f64],
) -> Option<Direction> {
    let n = x.len();
    let p = prob.b.len();
    let m = prob.h.len();

    // H = Q + Gᵀ diag(z/s) G and r1 = −rc + Gᵀ S⁻¹ (r_comp − Z rh).
    let mut kkt = vec![vec![0.0; n + p]; n + p];
    for (kkt_row, q_row) in kkt.iter_mut().zip(&prob.q) {
        kkt_row[..n].copy_from_slice(q_row);
    }
    let mut rhs = vec![0.0; n + p];
    let mut weighted = vec![0.0; m];
    for i in 0..m {
        let d = z[i] / s[i];
        let row = &prob.g[i];
        for (a, ga) in row.iter().enumerate() {
            if *ga == 0.0 {
                continue;
            }
            for (b, gb) in row.iter().enumerate() {
                kkt[a][b] += d * ga * gb;
            }
        }
        weighted[i] = (r_comp[i] - z[i] * res.rh[i]) / s[i];
    }
    let gtw = mat_t_vec(&prob.g, &weighted, n);
    for j in 0..n {
        rhs[j] = -res.rc[j] + gtw[j];
    }
    for (j, row) in prob.a.iter().enumerate() {
        for (i, aji) in row.iter().enumerate() {
            kkt[i][n + j] = *aji;
            kkt[n + j][i] = *aji;
        }
        rhs[n + j] = -res.rb[j];
    }

    let sol = lu_solve(kkt, rhs)?;
    let dx = sol[..n].to_vec();
    let dy = sol[n..].to_vec();
    let gdx = mat_vec(&prob.g, &dx);
    let ds: Vec<f64> = (0..m).map(|i| -res.rh[i] - gdx[i]).collect();
    let dz: Vec<f64> = (0..m).map(|i| (-r_comp[i] - z[i] * ds[i]) / s[i]).collect();
    if !dx.iter().chain(&dy).chain(&ds).chain(&dz).all(|v| v.is_finite()) {
        return None;
    }
    Some(Direction { dx, dy, ds, dz })
}

/// Largest step in `[0, ∞)` keeping `s + α ds ≥ 0` and `z + α dz ≥ 0`,
/// capped at 1, returned as the minimum of the two limits.
fn step_len(s: &[f64], ds: &[f64], z: &[f64], dz: &[f64]) -> f64 {
    let lim = |v: &[f64], dv: &[f64]| {
        v.iter()
            .zip(dv)
            .filter(|(_, d)| **d < 0.0)
            .map(|(vi, di)| -vi / di)
            .fold(f64::INFINITY, f64::min)
    };
    lim(s, ds).min(lim(z, dz)).min(1.0)
}

/// Solve the dense square system `K v = rhs` by LU factorisation with partial
/// pivoting.
///
/// Returns `None` if a pivot is numerically zero (the matrix is singular to
/// working precision). Shared by the interior-point solvers in this workspace.
pub fn lu_solve(mut k: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    let scale = k.iter().flat_map(|row| row.iter()).fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
    for col in 0..n {
        let pivot_row = (col..n).max_by(|&a, &b| k[a][col].abs().total_cmp(&k[b][col].abs()))?;
        if k[pivot_row][col].abs() <= 1e-14 * scale {
            return None;
        }
        k.swap(col, pivot_row);
        rhs.swap(col, pivot_row);
        let pivot = k[col][col];
        for row in col + 1..n {
            let factor = k[row][col] / pivot;
            if factor == 0.0 {
                continue;
            }
            let (upper, lower) = k.split_at_mut(row);
            let src = &upper[col];
            let dst = &mut lower[0];
            for (d, s) in dst[col..].iter_mut().zip(&src[col..]) {
                *d -= factor * s;
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut v = vec![0.0; n];
    for row in (0..n).rev() {
        let mut acc = rhs[row];
        acc -= k[row][row + 1..n].iter().zip(&v[row + 1..n]).map(|(kij, vj)| kij * vj).sum::<f64>();
        v[row] = acc / k[row][row];
    }
    Some(v)
}

fn dims_ok(prob: &QpProblem, n: usize, m: usize, p: usize) -> bool {
    prob.q.len() == n
        && prob.q.iter().all(|row| row.len() == n)
        && prob.g.len() == m
        && prob.g.iter().all(|row| row.len() == n)
        && prob.a.len() == p
        && prob.a.iter().all(|row| row.len() == n)
}

fn failed(status: QpStatus, n: usize, p: usize, m: usize, iterations: usize) -> QpSolution {
    QpSolution {
        x: vec![0.0; n],
        objective: f64::NAN,
        y: vec![0.0; p],
        z: vec![0.0; m],
        status,
        iterations,
    }
}

fn objective(prob: &QpProblem, x: &[f64]) -> f64 {
    0.5 * dot(x, &mat_vec(&prob.q, x)) + dot(&prob.c, x)
}

fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, x)).collect()
}

/// `Aᵀ v` for `A` with rows of length `n`.
fn mat_t_vec(a: &[Vec<f64>], v: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for (row, vi) in a.iter().zip(v) {
        for (o, aij) in out.iter_mut().zip(row) {
            *o += aij * vi;
        }
    }
    out
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

    fn problem(q: Vec<Vec<f64>>, c: Vec<f64>) -> QpProblem {
        QpProblem { q, c, g: vec![], h: vec![], a: vec![], b: vec![] }
    }

    /// Check KKT conditions of a returned solution directly from the data.
    fn assert_kkt(prob: &QpProblem, sol: &QpSolution, tol: f64) {
        assert_eq!(sol.status, QpStatus::Optimal, "{sol:?}");
        let res = Residuals::compute(
            prob,
            &sol.x,
            &sol.y,
            &sol.z,
            &(0..prob.h.len()).map(|i| prob.h[i] - dot(&prob.g[i], &sol.x)).collect::<Vec<_>>(),
        );
        assert!(norm_inf(&res.rc) < tol, "stationarity {:?}", res.rc);
        assert!(norm_inf(&res.rb) < tol, "equality {:?}", res.rb);
        for i in 0..prob.h.len() {
            let slack = prob.h[i] - dot(&prob.g[i], &sol.x);
            assert!(slack > -tol, "infeasible row {i}");
            assert!(sol.z[i] > -tol, "negative dual {i}");
            assert!(slack * sol.z[i] < tol, "complementarity {i}");
        }
    }

    #[test]
    fn unconstrained_minimiser() {
        // ½‖x − (1, −2)‖² = ½ xᵀ I x − (1, −2)·x + const.
        let prob = problem(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![-1.0, 2.0]);
        let sol = solve_qp(&prob, &QpOptions::default());
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-8);
        assert!((sol.x[1] + 2.0).abs() < 1e-8);
    }

    #[test]
    fn box_projection() {
        // min ½‖x − (2, −1)‖² s.t. 0 ≤ x ≤ 1  ⇒  x = (1, 0).
        let mut prob = problem(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![-2.0, 1.0]);
        prob.g = vec![vec![1.0, 0.0], vec![-1.0, 0.0], vec![0.0, 1.0], vec![0.0, -1.0]];
        prob.h = vec![1.0, 0.0, 1.0, 0.0];
        let sol = solve_qp(&prob, &QpOptions::default());
        assert_kkt(&prob, &sol, 1e-7);
        assert!((sol.x[0] - 1.0).abs() < 1e-7, "{sol:?}");
        assert!(sol.x[1].abs() < 1e-7, "{sol:?}");
    }

    #[test]
    fn equality_constrained() {
        // min x² + y² s.t. x + y = 1  ⇒  (½, ½), objective ½.
        let mut prob = problem(vec![vec![2.0, 0.0], vec![0.0, 2.0]], vec![0.0, 0.0]);
        prob.a = vec![vec![1.0, 1.0]];
        prob.b = vec![1.0];
        let sol = solve_qp(&prob, &QpOptions::default());
        assert_kkt(&prob, &sol, 1e-7);
        assert!((sol.x[0] - 0.5).abs() < 1e-8);
        assert!((sol.x[1] - 0.5).abs() < 1e-8);
        assert!((sol.objective - 0.5).abs() < 1e-8);
    }

    #[test]
    fn mixed_constraints_satisfy_kkt() {
        // Coupled Hessian, nonnegativity, a budget row and an equality.
        let mut prob = problem(
            vec![vec![4.0, 1.0, 0.0], vec![1.0, 2.0, 0.5], vec![0.0, 0.5, 3.0]],
            vec![-8.0, -3.0, -3.0],
        );
        prob.g = vec![
            vec![-1.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0],
            vec![0.0, 0.0, -1.0],
            vec![1.0, 1.0, 1.0],
        ];
        prob.h = vec![0.0, 0.0, 0.0, 2.0];
        prob.a = vec![vec![1.0, -1.0, 0.0]];
        prob.b = vec![0.2];
        let sol = solve_qp(&prob, &QpOptions::default());
        assert_kkt(&prob, &sol, 1e-7);
    }

    #[test]
    fn redundant_equality_is_handled_or_reported() {
        // Two identical equality rows make the KKT matrix singular in the
        // constraint block; the solver must not return a wrong "Optimal".
        let mut prob = problem(vec![vec![2.0, 0.0], vec![0.0, 2.0]], vec![0.0, 0.0]);
        prob.a = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        prob.b = vec![1.0, 1.0];
        let sol = solve_qp(&prob, &QpOptions::default());
        match sol.status {
            QpStatus::Optimal => {
                assert!((sol.x[0] + sol.x[1] - 1.0).abs() < 1e-7);
                assert!((sol.objective - 0.5).abs() < 1e-7);
            }
            QpStatus::Numerical | QpStatus::MaxIterations => {}
            QpStatus::InvalidInput => panic!("valid input rejected"),
        }
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let prob = QpProblem {
            q: vec![vec![1.0]],
            c: vec![0.0, 0.0],
            g: vec![],
            h: vec![],
            a: vec![],
            b: vec![],
        };
        let sol = solve_qp(&prob, &QpOptions::default());
        assert_eq!(sol.status, QpStatus::InvalidInput);
    }

    #[test]
    fn lp_limit_with_zero_hessian() {
        // min −x − y s.t. x + 2y ≤ 4, 3x + y ≤ 6, x, y ≥ 0 ⇒ (1.6, 1.2).
        let mut prob = problem(vec![vec![0.0, 0.0], vec![0.0, 0.0]], vec![-1.0, -1.0]);
        prob.g = vec![vec![1.0, 2.0], vec![3.0, 1.0], vec![-1.0, 0.0], vec![0.0, -1.0]];
        prob.h = vec![4.0, 6.0, 0.0, 0.0];
        let sol = solve_qp(&prob, &QpOptions::default());
        assert_kkt(&prob, &sol, 1e-6);
        assert!((sol.x[0] - 1.6).abs() < 1e-6, "{sol:?}");
        assert!((sol.x[1] - 1.2).abs() < 1e-6, "{sol:?}");
    }
}
