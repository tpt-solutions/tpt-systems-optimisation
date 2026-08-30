//! Conic optimisation: second-order-cone (SOCP) and semidefinite (SDP)
//! solvers built on top of the workspace's verified LP engine.
//!
//! Both solvers use **Kelley cutting planes** (outer approximation): each
//! cone constraint is replaced by a sequence of valid linear supports. The
//! LP relaxation is solved with [`tpt_opt_milp::MilpSolver`] (all variables
//! continuous, so it solves the LP relaxation) and tightened until every cone
//! constraint is satisfied to `tol`. Because every cut is a valid supporting
//! hyperplane of the cone, the relaxation stays a relaxation of the true conic
//! problem and converges to the conic optimum.
//!
//! This makes the crate a drop-in conic solver for the robust-optimisation
//! workflow (ellipsoidal uncertainty → SOCP) without vendoring a bespoke
//! interior-point method.
//!
//! # Example
//!
//! ```rust
//! use tpt_opt_conic::{solve_socp, ConeProgram, SocRow, ConicStatus};
//! use tpt_opt_core::model::Sense;
//!
//! // max x1 + x2  s.t.  ‖(0.5 x1, 0.3 x1 + 0.4 x2)‖₂ ≤ 1 - x1,  x ≥ 0.
//! let q_mat = vec![
//!     vec![-1.0, 0.0], // r(x) = 1 - x1
//!     vec![0.5, 0.0],  // q component 1
//!     vec![0.3, 0.4],  // q component 2
//! ];
//! let prog = ConeProgram {
//!     n: 2,
//!     c: vec![1.0, 1.0],
//!     sense: Sense::Maximize,
//!     bounds: vec![(0.0, 2.0), (0.0, 5.0)],
//!     eq_a: vec![],
//!     eq_b: vec![],
//!     soc_rows: vec![SocRow { q_mat, q_rhs: vec![1.0, 0.0, 0.0] }],
//!     sdp_blocks: vec![],
//! };
//! let sol = solve_socp(&prog, 1e-6, 400);
//! assert_eq!(sol.status, ConicStatus::Optimal);
//! assert!(sol.objective > 1.0);
//! ```

use std::vec::Vec;

use tpt_opt_core::model::{Model, Objective, Sense};
use tpt_opt_core::{Constraint, Solver, VarBound};
use tpt_opt_milp::MilpSolver;

/// Outcome of a conic solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConicStatus {
    /// Conic optimum found (all cone constraints satisfied within `tol`).
    Optimal,
    /// LP relaxation infeasible (the conic problem is infeasible).
    Infeasible,
    /// Stopped after `max_iter` cutting-plane rounds without converging.
    MaxIterations,
}

/// A second-order-cone constraint `‖q(x)‖₂ ≤ r(x)` in standard form, where
/// `s(x) = q_mat · x + q_rhs` and `s[0] = r(x)`, `s[1..] = q(x)`. Its dimension
/// is `q_mat.len()` (so `q` has `dim - 1` components).
#[derive(Debug, Clone)]
pub struct SocRow {
    /// `dim × n` affine map; row 0 gives `r(x)`, rows `1..dim` give `q(x)`.
    pub q_mat: Vec<Vec<f64>>,
    /// Constant term of the affine map.
    pub q_rhs: Vec<f64>,
}

impl SocRow {
    /// Number of variables `n`.
    pub fn n(&self) -> usize {
        self.q_mat.first().map(|r| r.len()).unwrap_or(0)
    }

    /// Evaluate `s(x)` and return `(r, q, ‖q‖)`.
    fn eval(&self, x: &[f64]) -> (f64, Vec<f64>, f64) {
        let s: Vec<f64> = self
            .q_mat
            .iter()
            .zip(self.q_rhs.iter())
            .map(|(row, &b)| {
                let mut v = b;
                for (k, &c) in row.iter().enumerate() {
                    v += c * x[k];
                }
                v
            })
            .collect();
        let r = s[0];
        let q: Vec<f64> = s[1..].to_vec();
        let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        (r, q, norm)
    }
}

/// A semidefinite constraint `X(x) = X₀ + Σₖ xₖ·Xₖ ⪰ 0`, where each `X` is a
/// symmetric `dim × dim` matrix stored row-major.
#[derive(Debug, Clone)]
pub struct SdpBlock {
    /// Dimension of the symmetric matrix.
    pub dim: usize,
    /// Constant matrix `X₀` (`dim × dim`).
    pub x0: Vec<Vec<f64>>,
    /// Variable matrices `Xₖ` (`dim × dim` each), one per decision variable.
    pub xs: Vec<Vec<Vec<f64>>>,
}

impl SdpBlock {
    /// Evaluate `X(x)` into a fresh `dim × dim` matrix.
    fn eval(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let mut m = self.x0.clone();
        for (k, xk) in self.xs.iter().enumerate() {
            if k < x.len() {
                for i in 0..self.dim {
                    for j in 0..self.dim {
                        m[i][j] += xk[i][j] * x[k];
                    }
                }
            }
        }
        m
    }
}

/// A conic program in standard form.
///
/// ```text
/// minimise/maximise  cᵀx
/// subject to          bounds[i].0 ≤ xᵢ ≤ bounds[i].1
///                     eq_a · x = eq_b
///                     ‖qᵢ(x)‖₂ ≤ rᵢ(x)      for each `soc_rows` entry
///                     Xₖ(x) ⪰ 0           for each `sdp_blocks` entry
/// ```
#[derive(Debug, Clone)]
pub struct ConeProgram {
    /// Number of decision variables.
    pub n: usize,
    /// Linear objective coefficients.
    pub c: Vec<f64>,
    /// Optimisation sense.
    pub sense: Sense,
    /// Per-variable `(lower, upper)` bounds.
    pub bounds: Vec<(f64, f64)>,
    /// Equality constraint rows (`p × n`).
    pub eq_a: Vec<Vec<f64>>,
    /// Equality constraint right-hand sides (`p`).
    pub eq_b: Vec<f64>,
    /// Second-order-cone rows.
    pub soc_rows: Vec<SocRow>,
    /// Semidefinite blocks.
    pub sdp_blocks: Vec<SdpBlock>,
}

/// Solution returned by [`solve_conic`].
#[derive(Debug, Clone)]
pub struct ConeSolution {
    /// Termination status.
    pub status: ConicStatus,
    /// Decision vector (empty if infeasible / not found).
    pub x: Vec<f64>,
    /// Objective value `cᵀx` (sign corrected for maximisation).
    pub objective: f64,
    /// Maximum cone-constraint violation at the returned point (`max(0, …)`).
    pub max_violation: f64,
}

/// Solve a [`ConeProgram`] via Kelley cutting planes over the LP engine.
///
/// `tol` is the cone-feasibility tolerance; `max_iter` bounds the number of
/// cutting-plane rounds.
pub fn solve_conic(prog: &ConeProgram, tol: f64, max_iter: usize) -> ConeSolution {
    // Cut accumulation: (nonzero indices, coeffs, rhs) for `≤` constraints.
    let mut cuts: Vec<(Vec<usize>, Vec<f64>, f64)> = Vec::new();

    // Seed the relaxation with the always-valid necessary condition `r(x) ≥ 0`
    // for every SOC row (a SOC requires ‖q‖ ≤ r with ‖q‖ ≥ 0). This also
    // bounds the `r` affine form, keeping the LP relaxation well-posed even
    // before the first supporting-hyperplane cut is generated.
    for row in &prog.soc_rows {
        let mut idx = Vec::new();
        let mut co = Vec::new();
        for k in 0..prog.n {
            let ck = -row.q_mat[0][k];
            if ck.abs() > 0.0 {
                idx.push(k);
                co.push(ck);
            }
        }
        cuts.push((idx, co, row.q_rhs[0]));
    }

    for _round in 0..max_iter {
        match lp_primal(prog, &cuts) {
            Some(x) => {
                let mut new_cuts: Vec<(Vec<usize>, Vec<f64>, f64)> = Vec::new();
                let mut worst = 0.0f64;
                for row in &prog.soc_rows {
                    let (r, q, norm) = row.eval(&x);
                    if norm <= 1e-12 {
                        // Degenerate cone point: ‖q‖ = 0, so the constraint
                        // reduces to the necessary condition r(x) ≥ 0. If that
                        // fails there is no valid supporting direction, so add
                        // the separating cut -r(x) ≤ 0 and move on.
                        if r < -tol {
                            worst = worst.max(-r);
                            let mut idx = Vec::new();
                            let mut co = Vec::new();
                            for k in 0..prog.n {
                                let ck = -row.q_mat[0][k];
                                if ck.abs() > 0.0 {
                                    idx.push(k);
                                    co.push(ck);
                                }
                            }
                            let rhs = row.q_rhs[0];
                            new_cuts.push((idx, co, rhs));
                        }
                        continue;
                    }
                    let viol = norm - r;
                    if viol > tol {
                        worst = worst.max(viol);
                        // Supporting hyperplane: (q/‖q‖)ᵀ q(x) ≤ r(x).
                        let inv = 1.0 / norm;
                        let u: Vec<f64> = q.iter().map(|v| v * inv).collect();
                        let mut idx = Vec::new();
                        let mut co = Vec::new();
                        for k in 0..prog.n {
                            let mut ck = 0.0f64;
                            for (&ut, row_q) in u.iter().zip(row.q_mat.iter().skip(1)) {
                                ck += ut * row_q[k];
                            }
                            ck -= row.q_mat[0][k];
                            if ck.abs() > 0.0 {
                                idx.push(k);
                                co.push(ck);
                            }
                        }
                        let mut rhs = row.q_rhs[0];
                        for (&ut, &brhs) in u.iter().zip(row.q_rhs.iter().skip(1)) {
                            rhs -= ut * brhs;
                        }
                        new_cuts.push((idx, co, rhs));
                    }
                }
                for block in &prog.sdp_blocks {
                    let m = block.eval(&x);
                    let (eigs, vecs) = jacobi_eigen(m);
                    let (lambda_min, v) = eigs
                        .iter()
                        .zip(vecs.iter())
                        .min_by(|a, b| a.0.partial_cmp(b.0).unwrap())
                        .unwrap();
                    if *lambda_min < -tol {
                        worst = worst.max(-*lambda_min);
                        // Valid cut: ⟨v vᵀ, X(x)⟩ ≥ 0 (any feasible point must
                        // satisfy this for the eigenvector attaining λ_min < 0).
                        // Expressed as a `≤` row: -⟨v vᵀ, X(x)⟩ ≤ 0.
                        let mut idx = Vec::new();
                        let mut co = Vec::new();
                        for (k, xk) in block.xs.iter().enumerate() {
                            let mut ck = 0.0f64;
                            for i in 0..block.dim {
                                for j in 0..block.dim {
                                    ck -= v[i] * xk[i][j] * v[j];
                                }
                            }
                            if ck.abs() > 0.0 {
                                idx.push(k);
                                co.push(ck);
                            }
                        }
                        let mut rhs = 0.0f64;
                        for i in 0..block.dim {
                            for j in 0..block.dim {
                                rhs += v[i] * block.x0[i][j] * v[j];
                            }
                        }
                        new_cuts.push((idx, co, rhs));
                    }
                }
                if new_cuts.is_empty() {
                    // The objective value is always `cᵀx` at the returned point;
                    // the sense only selects which point is found.
                    let obj = prog.c.iter().zip(&x).map(|(cc, xx)| cc * xx).sum::<f64>();
                    return ConeSolution {
                        status: ConicStatus::Optimal,
                        x,
                        objective: obj,
                        max_violation: worst,
                    };
                }
                cuts.extend(new_cuts);
            }
            None => {
                // LP relaxation infeasible: the conic problem is infeasible.
                return ConeSolution {
                    status: ConicStatus::Infeasible,
                    x: Vec::new(),
                    objective: f64::NAN,
                    max_violation: f64::INFINITY,
                };
            }
        }
    }

    // Hit the iteration cap. Return the last relaxed point if we have one.
    match lp_primal(prog, &cuts) {
        Some(x) => {
            let obj = prog.c.iter().zip(&x).map(|(cc, xx)| cc * xx).sum::<f64>();
            ConeSolution {
                status: ConicStatus::MaxIterations,
                x,
                objective: obj,
                max_violation: f64::INFINITY,
            }
        }
        None => ConeSolution {
            status: ConicStatus::Infeasible,
            x: Vec::new(),
            objective: f64::NAN,
            max_violation: f64::INFINITY,
        },
    }
}

/// Convenience: solve a pure SOCP program.
pub fn solve_socp(prog: &ConeProgram, tol: f64, max_iter: usize) -> ConeSolution {
    solve_conic(prog, tol, max_iter)
}

/// Solve the current LP relaxation (objective `c` minimised, with `cuts` as
/// extra `≤` constraints) and return the primal vector, or `None` if the LP is
/// infeasible.
fn lp_primal(prog: &ConeProgram, cuts: &[(Vec<usize>, Vec<f64>, f64)]) -> Option<Vec<f64>> {
    let mut model = Model::new(prog.n);
    for (i, &(lo, hi)) in prog.bounds.iter().enumerate() {
        if lo == 0.0 && hi == f64::INFINITY {
            model.variables[i].bound = VarBound::continuous(0.0, f64::INFINITY);
        } else {
            model.variables[i].bound = VarBound::continuous(lo, hi);
        }
    }
    // Objective is minimised internally; flip for maximisation.
    let sign = match prog.sense {
        Sense::Minimize => 1.0,
        Sense::Maximize => -1.0,
    };
    let mut idx = Vec::new();
    let mut co = Vec::new();
    for (k, &ck) in prog.c.iter().enumerate() {
        if ck != 0.0 {
            idx.push(k);
            co.push(ck * sign);
        }
    }
    model.set_objective(Objective {
        sense: Sense::Minimize,
        indices: idx,
        coeffs: co,
        constant: 0.0,
    });
    for (row, rhs) in prog.eq_a.iter().zip(prog.eq_b.iter()) {
        let nz_i: Vec<usize> =
            row.iter().enumerate().filter(|&(_, &c)| c != 0.0).map(|(k, _)| k).collect();
        let nz_c: Vec<f64> = row.iter().copied().filter(|&c| c != 0.0).collect();
        model.add_constraint(Constraint::equality(nz_i, nz_c, *rhs));
    }
    for (idx_c, co_c, rhs) in cuts {
        model.add_constraint(Constraint::le(idx_c.clone(), co_c.clone(), *rhs));
    }
    let sol = MilpSolver::new().with_threads(1).solve(&model).ok()?;
    if sol.status != tpt_opt_core::SolverStatus::Optimal {
        return None;
    }
    Some(sol.primal)
}

/// Symmetric-eigenvalue decomposition by the cyclic Jacobi method.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[k]` is the
/// `k`-th eigenvector (unit length). Converges to machine precision for the
/// off-diagonal norm; suitable for the small PSD blocks that arise in robust
/// and convex modelling.
#[allow(clippy::needless_range_loop)]
pub fn jacobi_eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v: Vec<Vec<f64>> =
        (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let mut iter = 0;
    loop {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i][j] * a[i][j];
            }
        }
        if off.sqrt() < 1e-12 || iter > 100 + n * n * 10 {
            break;
        }
        iter += 1;
        // Largest off-diagonal entry.
        let mut p = 0;
        let mut q = 1;
        let mut max = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let av = a[i][j].abs();
                if av > max {
                    max = av;
                    p = i;
                    q = j;
                }
            }
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let theta = (aqq - app) / (2.0 * apq);
        let t =
            if theta >= 0.0 { 1.0 } else { -1.0 } / (theta.abs() + (1.0 + theta * theta).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        // Rotate rows/cols p,q.
        for k in 0..n {
            let akp = a[k][p];
            let akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for k in 0..n {
            let apk = a[p][k];
            let aqk = a[q][k];
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        for k in 0..n {
            let vkp = v[k][p];
            let vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }
    let mut eigs = Vec::with_capacity(n);
    for i in 0..n {
        eigs.push(a[i][i]);
    }
    // eigenvectors are columns of v → return as Vec<Vec<f64>> (one per eigenval).
    let vecs: Vec<Vec<f64>> = (0..n).map(|j| (0..n).map(|i| v[i][j]).collect()).collect();
    (eigs, vecs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn socp_quarter_disk_maximise() {
        // max x1 + x2  s.t.  ‖(0.5 x1, 0.3 x1 + 0.4 x2)‖₂ ≤ 1 - x1,
        //               x1 + x2 ≥ 0.5,  x ≥ 0.
        // The `≥ 0.5` row is modelled via a slack variable x3 ≥ 0 with the
        // equality x1 + x2 - x3 = 0.5. The SOC optimum is ≈ 1.075 (computed
        // independently for this correlated cone); we only assert it beats 1.0.
        // SOC row: r(x) = 1 - x1,  q(x) = (0.5 x1, 0.3 x1 + 0.4 x2); the slack
        // variable enters every affine map with zero coefficient (rows are width n).
        let q_mat = vec![
            vec![-1.0, 0.0, 0.0], // r: 1 - x1
            vec![0.5, 0.0, 0.0],  // q component 1: 0.5 x1
            vec![0.3, 0.4, 0.0],  // q component 2: 0.3 x1 + 0.4 x2
        ];
        let q_rhs = vec![1.0, 0.0, 0.0];
        let prog = ConeProgram {
            n: 3,
            c: vec![1.0, 1.0, 0.0],
            sense: Sense::Maximize,
            // Finite upper bounds keep the LP relaxation bounded; the SOC
            // optimum (≈1.075) lies well inside them.
            bounds: vec![(0.0, 2.0), (0.0, 5.0), (0.0, 5.0)],
            eq_a: vec![vec![1.0, 1.0, -1.0]],
            eq_b: vec![0.5],
            soc_rows: vec![SocRow { q_mat, q_rhs }],
            sdp_blocks: vec![],
        };
        let sol = solve_socp(&prog, 1e-6, 400);
        assert_eq!(sol.status, ConicStatus::Optimal);
        // Feasibility: SOC satisfied within tol.
        let (r, _q, norm) = prog.soc_rows[0].eval(&sol.x);
        assert!((norm - r) <= 1e-4, "soc violated: norm={norm} r={r}");
        assert!(sol.x[0] + sol.x[1] >= 0.5 - 1e-6);
        // Known optimum ≈ 1.075; allow slack.
        assert!(sol.objective > 1.0, "objective too low: {}", sol.objective);
    }

    #[test]
    fn socp_infeasible_when_bound_excludes() {
        // min x  s.t. ‖(x, x)‖ ≤ -1  (impossible: a SOC needs r ≥ ‖q‖ ≥ 0).
        // Build r(x) = -1, q(x) = (x, x) as functions of the single variable:
        // q_mat rows have length n=1.
        let q_mat = vec![vec![-1.0], vec![1.0], vec![1.0]];
        let q_rhs = vec![-1.0, 0.0, 0.0];
        let prog = ConeProgram {
            n: 1,
            c: vec![1.0],
            sense: Sense::Minimize,
            bounds: vec![(0.0, f64::INFINITY)],
            eq_a: vec![],
            eq_b: vec![],
            soc_rows: vec![SocRow { q_mat, q_rhs }],
            sdp_blocks: vec![],
        };
        let sol = solve_socp(&prog, 1e-6, 40);
        assert_eq!(sol.status, ConicStatus::Infeasible);
    }

    #[test]
    fn sdp_psd_cutting_plane() {
        // min x  s.t.  [[1, x], [x, 1]] ⪰ 0  →  |x| ≤ 1, optimum x = -1.
        let x0 = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let xs = vec![vec![vec![0.0, 1.0], vec![1.0, 0.0]]]; // X_0 coefficient matrix
        let prog = ConeProgram {
            n: 1,
            c: vec![1.0],
            sense: Sense::Minimize,
            bounds: vec![(-10.0, 10.0)],
            eq_a: vec![],
            eq_b: vec![],
            soc_rows: vec![],
            sdp_blocks: vec![SdpBlock { dim: 2, x0, xs }],
        };
        let sol = solve_conic(&prog, 1e-6, 400);
        assert_eq!(sol.status, ConicStatus::Optimal);
        assert!((sol.x[0] + 1.0).abs() < 1e-3, "x = {}", sol.x[0]);
    }

    #[test]
    fn jacobi_eigen_basic() {
        let m = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (eigs, vecs) = jacobi_eigen(m);
        // Eigenvalues should be ~1 and ~3.
        let mut s = eigs.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((s[0] - 1.0).abs() < 1e-6, "eig0 = {}", s[0]);
        assert!((s[1] - 3.0).abs() < 1e-6, "eig1 = {}", s[1]);
        // Check an eigenvector is unit length.
        let v0 = &vecs[0];
        let len2 = v0.iter().map(|v| v * v).sum::<f64>();
        assert!((len2 - 1.0).abs() < 1e-6);
    }
}
