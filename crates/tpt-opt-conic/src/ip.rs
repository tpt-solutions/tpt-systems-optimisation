//! Interior-point SOCP solver: a log-barrier path-following method.
//!
//! [`solve_socp_ip`] solves a second-order-cone program with no SDP blocks by
//! the barrier method. Three steps make the method self-contained:
//!
//! 1. **Equality elimination.** The affine hull of `eq_a · x = eq_b` is
//!    parameterised exactly as `x = x_p + N z` (reduced row echelon form), so
//!    the barrier method works on `z` with no equality multipliers and no
//!    infeasible-start logic.
//! 2. **Phase I.** Every linear and cone constraint is relaxed by a common
//!    slack `s` and `s` is minimised. Starting from `z = 0` with a large `s`
//!    makes the start strictly feasible; the phase stops as soon as `s < 0`,
//!    which yields a strictly feasible point for the original constraints. If
//!    `s` cannot be driven below zero the program is reported as infeasible.
//! 3. **Phase II.** The log-barrier `φ(z) = −Σ ln dᵢ(z) − Σ ln(F₀² − ‖F₁‖²)`
//!    is followed with `t` increasing geometrically; each centring step is a
//!    damped Newton iteration. The barrier parameter of each Lorentz cone is 2,
//!    so the duality-gap bound at a centre is `m / t` with `m` the total
//!    barrier count.
//!
//! The Newton systems are dense and solved with the LU helper from
//! `tpt-opt-core`. Problems whose feasible set has empty interior (for example
//! a cone collapsed to a point) are reported as infeasible.

use std::vec;
use std::vec::Vec;

use tpt_opt_core::model::Sense;
use tpt_opt_core::qp::lu_solve;

use crate::{ConeProgram, ConeSolution, ConicStatus, SocRow};

/// Barrier parameter growth factor between centring steps.
const MU: f64 = 10.0;
/// Newton-decrement stopping rule: `λ²/2 ≤ NEWTON_TOL` ends a centring step.
const NEWTON_TOL: f64 = 1e-10;
/// Newton iterations allowed per centring step.
const MAX_NEWTON: usize = 80;

/// Affine map `w ↦ M w + m`, with `M` stored as rows of length `dim`.
#[derive(Clone, Debug)]
struct Affine {
    rows: Vec<Vec<f64>>,
    off: Vec<f64>,
}

impl Affine {
    fn eval(&self, w: &[f64]) -> Vec<f64> {
        self.rows.iter().zip(&self.off).map(|(row, o)| dot(row, w) + o).collect()
    }
}

/// Barrier domain in `dim` variables: `a·w + b > 0` for each linear row and
/// `F₀(w) > ‖F₁(w)‖` for each cone.
struct Domain {
    dim: usize,
    lin: Vec<(Vec<f64>, f64)>,
    cones: Vec<Affine>,
}

impl Domain {
    fn strictly_inside(&self, w: &[f64]) -> bool {
        self.lin.iter().all(|(a, b)| dot(a, w) + b > 0.0)
            && self.cones.iter().all(|cone| {
                let f = cone.eval(w);
                f[0] > 0.0 && f[0] * f[0] - dot(&f[1..], &f[1..]) > 0.0
            })
    }

    /// Total barrier parameter `m`: one per linear row, two per cone.
    fn barrier_count(&self) -> usize {
        self.lin.len() + 2 * self.cones.len()
    }

    /// Barrier value `φ(w)`; only meaningful strictly inside the domain.
    fn phi(&self, w: &[f64]) -> f64 {
        let lin: f64 = self.lin.iter().map(|(a, b)| -(dot(a, w) + b).ln()).sum();
        let cone: f64 = self
            .cones
            .iter()
            .map(|cone| {
                let f = cone.eval(w);
                -(f[0] * f[0] - dot(&f[1..], &f[1..])).ln()
            })
            .sum();
        lin + cone
    }

    /// Gradient and Hessian of `φ` at `w` (strictly inside the domain).
    fn grad_hess(&self, w: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
        let d = self.dim;
        let mut g = vec![0.0; d];
        let mut h = vec![vec![0.0; d]; d];

        for (a, b) in &self.lin {
            let slack = dot(a, w) + b;
            for i in 0..d {
                g[i] -= a[i] / slack;
            }
            add_outer(&mut h, a, a, 1.0 / (slack * slack));
        }

        for cone in &self.cones {
            let f = cone.eval(w);
            let f0 = f[0] * f[0] - dot(&f[1..], &f[1..]);
            // ∇f = 2 F₀ M₀ − 2 Σ_j F_j M_j and ∇²f = 2 M₀ᵀM₀ − 2 Σ_j M_jᵀM_j.
            let mut grad_f = vec![0.0; d];
            let mut hess_f = vec![vec![0.0; d]; d];
            for (r, row) in cone.rows.iter().enumerate() {
                let (coef, sign) = if r == 0 { (2.0 * f[0], 1.0) } else { (-2.0 * f[r], -1.0) };
                for i in 0..d {
                    grad_f[i] += coef * row[i];
                }
                add_outer(&mut hess_f, row, row, 2.0 * sign);
            }
            for i in 0..d {
                g[i] -= grad_f[i] / f0;
                for j in 0..d {
                    h[i][j] += -hess_f[i][j] / f0 + grad_f[i] * grad_f[j] / (f0 * f0);
                }
            }
        }
        (g, h)
    }
}

/// Barrier objective `t·cᵀw + φ(w)`.
fn psi(dom: &Domain, w: &[f64], t: f64, c: &[f64]) -> f64 {
    t * dot(c, w) + dom.phi(w)
}

/// One centring step: damped Newton on `t·cᵀw + φ(w)` from `w`. Returns
/// `false` if the Newton system became singular, the line search stalled, or
/// the iteration budget ran out.
fn centre(dom: &Domain, w: &mut Vec<f64>, t: f64, c: &[f64]) -> bool {
    for _ in 0..MAX_NEWTON {
        let (g_phi, h) = dom.grad_hess(w);
        let g: Vec<f64> = g_phi.iter().zip(c).map(|(gi, ci)| gi + t * ci).collect();
        if norm_inf(&g) == 0.0 {
            return true;
        }
        let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
        let Some(dx) = lu_solve(h, neg_g) else {
            return false;
        };
        let slope = dot(&g, &dx);
        if -slope / 2.0 <= NEWTON_TOL {
            return true;
        }
        let psi0 = psi(dom, w, t, c);
        let mut alpha = 1.0;
        loop {
            let cand: Vec<f64> = w.iter().zip(&dx).map(|(wi, di)| wi + alpha * di).collect();
            if dom.strictly_inside(&cand) && psi(dom, &cand, t, c) <= psi0 + 0.25 * alpha * slope {
                *w = cand;
                break;
            }
            alpha *= 0.5;
            if alpha < 1e-14 {
                return false;
            }
        }
    }
    false
}

/// Outcome of a barrier-method run.
enum BarrierOutcome {
    /// Duality-gap bound `m / t` reached `tol`.
    Converged(Vec<f64>),
    /// The `stop` predicate fired after a centring step.
    Stopped(Vec<f64>),
    /// Centring failed or the outer budget ran out.
    Failed,
}

/// Follow the central path of `min t·cᵀw + φ(w)` from a strictly feasible
/// `w0` until `m / t ≤ tol`, checking `stop` after each centring step.
fn barrier_method(
    dom: &Domain,
    w0: Vec<f64>,
    c: &[f64],
    tol: f64,
    max_outer: usize,
    stop: &dyn Fn(&[f64]) -> bool,
) -> BarrierOutcome {
    let m = dom.barrier_count() as f64;
    let mut w = w0;
    let mut t = 1.0;
    for _ in 0..max_outer {
        if !centre(dom, &mut w, t, c) {
            return BarrierOutcome::Failed;
        }
        if stop(&w) {
            return BarrierOutcome::Stopped(w);
        }
        if m / t <= tol {
            return BarrierOutcome::Converged(w);
        }
        t *= MU;
    }
    BarrierOutcome::Failed
}

/// Affine hull of `A x = b` in RREF: returns `(x_p, N, k)` with every solution
/// `x = x_p + N z`, `z ∈ ℝᵏ`, or `None` if the system is inconsistent.
fn affine_hull(a: &[Vec<f64>], b: &[f64], n: usize) -> Option<(Vec<f64>, Vec<Vec<f64>>, usize)> {
    let p = a.len();
    let scale = a.iter().flat_map(|row| row.iter()).fold(1.0_f64, |m, v| m.max(v.abs()));
    let mut r: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(row, bi)| {
            let mut v = row.clone();
            v.resize(n, 0.0);
            v.push(*bi);
            v
        })
        .collect();

    let mut pivots: Vec<usize> = Vec::new();
    let mut row = 0;
    for col in 0..n {
        if row >= p {
            break;
        }
        let best =
            (row..p).max_by(|&i, &j| r[i][col].abs().total_cmp(&r[j][col].abs())).unwrap_or(row);
        if r[best][col].abs() <= 1e-12 * scale {
            continue;
        }
        r.swap(row, best);
        let pv = r[row][col];
        for v in r[row].iter_mut() {
            *v /= pv;
        }
        let pivot_row = r[row].clone();
        for (i, ri) in r.iter_mut().enumerate() {
            if i == row {
                continue;
            }
            let f = ri[col];
            if f != 0.0 {
                for (v, pj) in ri.iter_mut().zip(&pivot_row) {
                    *v -= f * pj;
                }
            }
        }
        pivots.push(col);
        row += 1;
    }
    for ri in &r[row..] {
        if ri[n].abs() > 1e-9 * scale {
            return None;
        }
    }

    let mut x_p = vec![0.0; n];
    for (i, &pc) in pivots.iter().enumerate() {
        x_p[pc] = r[i][n];
    }
    let free: Vec<usize> = (0..n).filter(|c| !pivots.contains(c)).collect();
    let k = free.len();
    let mut nmat = vec![vec![0.0; k]; n];
    for (j, &f) in free.iter().enumerate() {
        nmat[f][j] = 1.0;
        for (i, &pc) in pivots.iter().enumerate() {
            nmat[pc][j] = -r[i][f];
        }
    }
    Some((x_p, nmat, k))
}

/// Solve a second-order-cone program (no SDP blocks) by the interior-point
/// barrier method.
///
/// `tol` bounds the barrier duality gap `m / t` at the returned point; `max_iter`
/// caps the number of barrier (outer) iterations. The returned
/// [`ConeSolution::max_violation`] is the worst cone violation at the returned
/// point; bound and equality satisfaction hold by construction.
pub fn solve_socp_ip(prog: &ConeProgram, tol: f64, max_iter: usize) -> ConeSolution {
    let n = prog.n;
    let Some((x_p, nmat, k)) = affine_hull(&prog.eq_a, &prog.eq_b, n) else {
        return infeasible();
    };

    // Work in z with x = x_p + N z; minimise the internal objective.
    let sign = match prog.sense {
        Sense::Maximize => -1.0,
        Sense::Minimize => 1.0,
    };
    let c_int: Vec<f64> = prog.c.iter().map(|ci| sign * ci).collect();
    let cz: Vec<f64> = (0..k).map(|j| (0..n).map(|i| nmat[i][j] * c_int[i]).sum()).collect();

    let mut lin = Vec::new();
    for (i, &(lo, hi)) in prog.bounds.iter().enumerate() {
        if lo.is_finite() {
            lin.push((nmat[i].clone(), x_p[i] - lo));
        }
        if hi.is_finite() {
            lin.push((nmat[i].iter().map(|v| -v).collect(), hi - x_p[i]));
        }
    }
    let cones: Vec<Affine> =
        prog.soc_rows.iter().map(|soc| cone_in_z(soc, &x_p, &nmat, k)).collect();
    let phase2 = Domain { dim: k, lin, cones };

    // Phase I: relax every constraint by a slack s (variable index k) and
    // minimise s until it is negative.
    let w_start = if phase2.barrier_count() == 0 {
        Some(vec![0.0; k])
    } else {
        let phase1 = relaxed(&phase2);
        let zero = vec![0.0; k + 1];
        let mut s0 = 1.0_f64;
        for (_, b) in &phase2.lin {
            s0 = s0.max(1.0 - b);
        }
        for cone in &phase2.cones {
            let f = cone.eval(&vec![0.0; k]);
            s0 = s0.max(dot(&f[1..], &f[1..]).sqrt() - f[0] + 1.0);
        }
        let mut start = zero;
        start[k] = s0;
        let mut c1 = vec![0.0; k + 1];
        c1[k] = 1.0;
        match barrier_method(&phase1, start, &c1, tol, max_iter, &|w| w[k] < 0.0) {
            BarrierOutcome::Stopped(w) => Some(w[..k].to_vec()),
            BarrierOutcome::Converged(w) => {
                if w[k] >= 0.0 {
                    return infeasible();
                }
                Some(w[..k].to_vec())
            }
            BarrierOutcome::Failed => return max_iterations(),
        }
    };
    let Some(z0) = w_start else {
        return max_iterations();
    };

    // Phase II.
    match barrier_method(&phase2, z0, &cz, tol, max_iter, &|_| false) {
        BarrierOutcome::Converged(z) | BarrierOutcome::Stopped(z) => {
            let x: Vec<f64> =
                (0..n).map(|i| x_p[i] + (0..k).map(|j| nmat[i][j] * z[j]).sum::<f64>()).collect();
            let objective = dot(&prog.c, &x);
            let max_violation = prog
                .soc_rows
                .iter()
                .map(|soc| {
                    let (r, _q, norm) = soc.eval(&x);
                    (norm - r).max(0.0)
                })
                .fold(0.0_f64, f64::max);
            ConeSolution { status: ConicStatus::Optimal, x, objective, max_violation }
        }
        BarrierOutcome::Failed => max_iterations(),
    }
}

/// Build the cone `F(z) = M N z + (M x_p + m)` in the reduced variables.
fn cone_in_z(soc: &SocRow, x_p: &[f64], nmat: &[Vec<f64>], k: usize) -> Affine {
    let rows = soc
        .q_mat
        .iter()
        .map(|row| {
            (0..k).map(|j| row.iter().enumerate().map(|(i, m)| m * nmat[i][j]).sum()).collect()
        })
        .collect();
    let off = soc.q_mat.iter().zip(&soc.q_rhs).map(|(row, m0)| dot(row, x_p) + m0).collect();
    Affine { rows, off }
}

/// Phase-I domain: every constraint `g(w) ≥ 0` becomes `g(w) + s ≥ 0`, with
/// the slack `s` stored at index `dim` of the augmented variable.
fn relaxed(dom: &Domain) -> Domain {
    let dim = dom.dim + 1;
    let lin = dom
        .lin
        .iter()
        .map(|(a, b)| {
            let mut a2 = a.clone();
            a2.push(1.0);
            (a2, *b)
        })
        .collect();
    let cones = dom
        .cones
        .iter()
        .map(|cone| {
            let rows = cone
                .rows
                .iter()
                .enumerate()
                .map(|(r, row)| {
                    let mut row2 = row.clone();
                    row2.push(if r == 0 { 1.0 } else { 0.0 });
                    row2
                })
                .collect();
            Affine { rows, off: cone.off.clone() }
        })
        .collect();
    Domain { dim, lin, cones }
}

fn infeasible() -> ConeSolution {
    ConeSolution {
        status: ConicStatus::Infeasible,
        x: Vec::new(),
        objective: f64::NAN,
        max_violation: f64::INFINITY,
    }
}

fn max_iterations() -> ConeSolution {
    ConeSolution {
        status: ConicStatus::MaxIterations,
        x: Vec::new(),
        objective: f64::NAN,
        max_violation: f64::INFINITY,
    }
}

/// `h += scale · u vᵀ` (symmetrised by the caller's choice of `u = v`).
fn add_outer(h: &mut [Vec<f64>], u: &[f64], v: &[f64], scale: f64) {
    for (hi, ui) in h.iter_mut().zip(u) {
        for (hij, vj) in hi.iter_mut().zip(v) {
            *hij += scale * ui * vj;
        }
    }
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

    #[test]
    fn unit_disk_linear_objective() {
        // min −x₁  s.t.  x₁² + x₂² ≤ 1  ⇒  x = (1, 0), objective −1.
        let prog = ConeProgram {
            n: 2,
            c: vec![-1.0, 0.0],
            sense: Sense::Minimize,
            bounds: vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
            eq_a: vec![],
            eq_b: vec![],
            soc_rows: vec![SocRow {
                q_mat: vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
                q_rhs: vec![1.0, 0.0, 0.0],
            }],
            sdp_blocks: vec![],
        };
        let sol = solve_socp_ip(&prog, 1e-8, 60);
        assert_eq!(sol.status, ConicStatus::Optimal, "{sol:?}");
        assert!((sol.x[0] - 1.0).abs() < 1e-5, "{sol:?}");
        assert!(sol.x[1].abs() < 1e-5, "{sol:?}");
        assert!((sol.objective + 1.0).abs() < 1e-5, "{sol:?}");
    }

    #[test]
    fn equality_with_cone_closed_form() {
        // min x₁  s.t.  x₁ + x₂ = ½,  x₁² + x₂² ≤ 1.
        // Substituting x₂ = ½ − x₁ gives 2x₁² − x₁ − ¾ ≤ 0, so
        // x₁* = (1 − √7)/4 ≈ −0.411438.
        let prog = ConeProgram {
            n: 2,
            c: vec![1.0, 0.0],
            sense: Sense::Minimize,
            bounds: vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
            eq_a: vec![vec![1.0, 1.0]],
            eq_b: vec![0.5],
            soc_rows: vec![SocRow {
                q_mat: vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
                q_rhs: vec![1.0, 0.0, 0.0],
            }],
            sdp_blocks: vec![],
        };
        let sol = solve_socp_ip(&prog, 1e-8, 80);
        assert_eq!(sol.status, ConicStatus::Optimal, "{sol:?}");
        let expected = (1.0 - 7.0_f64.sqrt()) / 4.0;
        assert!((sol.x[0] - expected).abs() < 1e-5, "{sol:?}");
        assert!((sol.x[0] + sol.x[1] - 0.5).abs() < 1e-8, "{sol:?}");
    }

    #[test]
    fn empty_interior_is_reported_infeasible() {
        // ‖(x)‖ ≤ 0 with x ≥ 1 has no feasible point.
        let prog = ConeProgram {
            n: 1,
            c: vec![1.0],
            sense: Sense::Minimize,
            bounds: vec![(1.0, f64::INFINITY)],
            eq_a: vec![],
            eq_b: vec![],
            soc_rows: vec![SocRow { q_mat: vec![vec![0.0], vec![1.0]], q_rhs: vec![0.0, 0.0] }],
            sdp_blocks: vec![],
        };
        let sol = solve_socp_ip(&prog, 1e-8, 60);
        assert_eq!(sol.status, ConicStatus::Infeasible, "{sol:?}");
    }

    #[test]
    fn inconsistent_equalities_are_infeasible() {
        let prog = ConeProgram {
            n: 1,
            c: vec![1.0],
            sense: Sense::Minimize,
            bounds: vec![(f64::NEG_INFINITY, f64::INFINITY)],
            eq_a: vec![vec![1.0], vec![1.0]],
            eq_b: vec![1.0, 2.0],
            soc_rows: vec![],
            sdp_blocks: vec![],
        };
        let sol = solve_socp_ip(&prog, 1e-8, 60);
        assert_eq!(sol.status, ConicStatus::Infeasible, "{sol:?}");
    }

    #[test]
    fn affine_hull_parameterises_solutions() {
        // x₁ + x₂ + x₃ = 3 has a 2-dimensional solution set.
        let (x_p, nmat, k) =
            affine_hull(&[vec![1.0, 1.0, 1.0]], &[3.0], 3).expect("consistent system");
        assert_eq!(k, 2);
        let z = [0.7, -0.4];
        let x: Vec<f64> = (0..3).map(|i| x_p[i] + nmat[i][0] * z[0] + nmat[i][1] * z[1]).collect();
        assert!((x.iter().sum::<f64>() - 3.0).abs() < 1e-12);
    }
}
