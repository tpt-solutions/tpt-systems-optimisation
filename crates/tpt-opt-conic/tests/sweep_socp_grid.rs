//! Randomised check of the SOCP solver against a dense grid search.
//!
//! Each instance has two variables in a box and one or two second-order-cone
//! constraints `‖q(x)‖₂ ≤ r(x)` with affine `q` and `r`. A grid over the box gives
//! an inner approximation of the feasible set, so its best value is an upper
//! bound on the optimum for minimisation. The solver must return a point that is
//! feasible, and whose value is no worse than the grid's by more than the grid's
//! resolution allows.

use tpt_opt_conic::{solve_socp, ConeProgram, ConicStatus, SocRow};
use tpt_opt_core::model::Sense;

const GRID: usize = 300;
const BOX: f64 = 2.0;

/// Deterministic generator, so any failure reproduces exactly.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn real(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.unit() * (hi - lo)
    }
}

fn random_program(rng: &mut Lcg) -> ConeProgram {
    let c = vec![rng.real(-3.0, 3.0), rng.real(-3.0, 3.0)];
    let n_soc = if rng.unit() < 0.5 { 1 } else { 2 };
    let soc_rows = (0..n_soc)
        .map(|_| {
            // Row 0 is r(x) = r0 + r·x; rows 1..=m are q(x) = Q x + q0.
            let m = 1 + (rng.unit() * 2.0) as usize;
            let mut q_mat = Vec::new();
            let mut q_rhs = Vec::new();
            q_mat.push(vec![rng.real(-0.5, 0.5), rng.real(-0.5, 0.5)]);
            q_rhs.push(rng.real(0.8, 3.0));
            for _ in 0..m {
                q_mat.push(vec![rng.real(-1.0, 1.0), rng.real(-1.0, 1.0)]);
                q_rhs.push(rng.real(-0.5, 0.5));
            }
            SocRow { q_mat, q_rhs }
        })
        .collect();
    ConeProgram {
        n: 2,
        c,
        sense: Sense::Minimize,
        bounds: vec![(-BOX, BOX), (-BOX, BOX)],
        eq_a: Vec::new(),
        eq_b: Vec::new(),
        soc_rows,
        sdp_blocks: Vec::new(),
    }
}

fn affine(row: &[f64], rhs: f64, x: &[f64]) -> f64 {
    rhs + row.iter().zip(x).map(|(a, v)| a * v).sum::<f64>()
}

/// Largest violation of any cone constraint `‖q(x)‖ − r(x)` at `x`.
fn violation(p: &ConeProgram, x: &[f64]) -> f64 {
    p.soc_rows
        .iter()
        .map(|row| {
            let r = affine(&row.q_mat[0], row.q_rhs[0], x);
            let norm: f64 = row.q_mat[1..]
                .iter()
                .zip(&row.q_rhs[1..])
                .map(|(qr, &qh)| affine(qr, qh, x).powi(2))
                .sum::<f64>()
                .sqrt();
            (norm - r).max(0.0)
        })
        .fold(0.0, f64::max)
}

/// Best grid objective over feasible grid points, or `None` if none is feasible.
fn grid_best(p: &ConeProgram) -> Option<f64> {
    let mut best: Option<f64> = None;
    for i in 0..=GRID {
        for j in 0..=GRID {
            let x = [
                -BOX + 2.0 * BOX * i as f64 / GRID as f64,
                -BOX + 2.0 * BOX * j as f64 / GRID as f64,
            ];
            if violation(p, &x) <= 1e-9 {
                let v = p.c[0] * x[0] + p.c[1] * x[1];
                best = Some(best.map_or(v, |b: f64| b.min(v)));
            }
        }
    }
    best
}

#[test]
fn socp_solver_agrees_with_grid_search() {
    let mut rng = Lcg(71);
    let mut compared = 0;
    let mut grid_feasible = 0;
    for case in 0..150 {
        let p = random_program(&mut rng);
        let sol = solve_socp(&p, 1e-8, 400);
        let grid = grid_best(&p);
        if let Some(g) = grid {
            grid_feasible += 1;
            // Grid spacing bounds how far the true optimum can sit below the grid's best.
            let step = 2.0 * BOX / GRID as f64;
            let slack = step * (p.c[0].abs() + p.c[1].abs()) + 1e-3;
            assert_eq!(sol.status, ConicStatus::Optimal, "case {case}: {sol:?}");
            assert!(
                sol.max_violation < 1e-5,
                "case {case}: returned point violates a cone by {}",
                sol.max_violation
            );
            assert!(
                sol.objective <= g + 1e-6,
                "case {case}: solver {} is worse than the grid {g}",
                sol.objective
            );
            assert!(
                sol.objective >= g - slack - 1e-3 * (1.0 + g.abs()) - 0.5,
                "case {case}: solver {} is implausibly far below the grid {g}",
                sol.objective
            );
            compared += 1;
        } else if sol.status == ConicStatus::Optimal {
            // The grid found nothing, but the solver claims a feasible optimum: it must be feasible.
            assert!(sol.max_violation < 1e-5, "case {case}: infeasible point reported optimal");
        }
    }
    assert!(
        compared > 40,
        "only {compared} instances were compared ({grid_feasible} grid-feasible)"
    );
}
