//! Soundness sweep for Lagrangian relaxation.
//!
//! Relaxing one coupling row `g·x ≤ b` of a minimisation with multiplier
//! `λ ≥ 0` gives `L(λ) = min_{x ∈ X} c·x + λ(g·x − b)`, which lower-bounds the
//! optimum for every `λ ≥ 0`. Here `X` is the integer box of each block, so the
//! inner minimisation is enumerated exactly. The best bound reported by the
//! subgradient method must never exceed the monolithic MILP optimum.

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::{Solver, SolverStatus, VarBound};
use tpt_opt_decompose::lagrangian::lagrangian_subgradient;
use tpt_opt_decompose::DualConfig;
use tpt_opt_milp::MilpSolver;

const UPPER: i64 = 3;

/// Deterministic generator, so any failure reproduces exactly.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.unit() * (hi - lo + 1) as f64) as i64
    }
}

/// Two blocks of integer variables in `[0, UPPER]`, one coupling row `≤ b`.
struct Instance {
    cost: Vec<Vec<f64>>,
    coupling: Vec<Vec<f64>>,
    b: f64,
}

fn random_instance(rng: &mut Lcg) -> Instance {
    let mut cost = Vec::new();
    let mut coupling = Vec::new();
    for _ in 0..2 {
        let nv = rng.int(1, 2) as usize;
        cost.push((0..nv).map(|_| rng.int(-3, 3) as f64).collect());
        coupling.push((0..nv).map(|_| rng.int(0, 3) as f64).collect());
    }
    let b = rng.int(0, 8) as f64;
    Instance { cost, coupling, b }
}

/// Every integer point of a block's box.
fn block_points(nv: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![]];
    for _ in 0..nv {
        out = out
            .into_iter()
            .flat_map(|p| {
                (0..=UPPER).map(move |v| {
                    let mut q = p.clone();
                    q.push(v as f64);
                    q
                })
            })
            .collect();
    }
    out
}

fn monolithic(inst: &Instance) -> Option<f64> {
    let mut m = Model::new(0);
    let mut idx = Vec::new();
    let mut coeffs = Vec::new();
    let mut obj_idx = Vec::new();
    let mut obj_coeffs = Vec::new();
    for (k, c) in inst.cost.iter().enumerate() {
        for (j, &cj) in c.iter().enumerate() {
            let v = m.add_variable(VarBound::integer(0.0, UPPER as f64));
            idx.push(v);
            coeffs.push(inst.coupling[k][j]);
            obj_idx.push(v);
            obj_coeffs.push(cj);
        }
    }
    m.add_constraint(Constraint::le(idx, coeffs, inst.b));
    m.set_objective(Objective {
        sense: Sense::Minimize,
        indices: obj_idx,
        coeffs: obj_coeffs,
        constant: 0.0,
    });
    match MilpSolver::new().solve(&m) {
        Ok(s) if s.status == SolverStatus::Optimal => Some(s.objective_value),
        Ok(s) if s.status == SolverStatus::Infeasible => None,
        other => panic!("monolithic reference failed: {other:?}"),
    }
}

/// `L(λ)` and the subgradient `g·x* − b` at the inner minimiser. Each block is a
/// box, so the inner minimum is taken over every joint choice of block points.
fn dual_oracle(inst: &Instance, lambda: f64) -> (f64, f64) {
    let points: Vec<Vec<Vec<f64>>> = inst.cost.iter().map(|c| block_points(c.len())).collect();
    let mut best: Option<(f64, f64)> = None;
    for p0 in &points[0] {
        for p1 in &points[1] {
            let mut cost = 0.0;
            let mut g = 0.0;
            for (k, p) in [p0, p1].into_iter().enumerate() {
                cost += inst.cost[k].iter().zip(p).map(|(c, x)| c * x).sum::<f64>();
                g += inst.coupling[k].iter().zip(p).map(|(a, x)| a * x).sum::<f64>();
            }
            let lag = cost + lambda * (g - inst.b);
            if best.is_none_or(|(b, _)| lag < b) {
                best = Some((lag, g - inst.b));
            }
        }
    }
    best.expect("every block has at least one point")
}

#[test]
fn lagrangian_bound_never_exceeds_the_optimum() {
    let mut rng = Lcg(61);
    let mut checked = 0;
    for case in 0..200 {
        let inst = random_instance(&mut rng);
        let Some(opt) = monolithic(&inst) else { continue };
        checked += 1;
        let config = DualConfig { max_iterations: 200, ..DualConfig::default() };
        let result = lagrangian_subgradient(
            vec![0.0],
            &config,
            |lam: &[f64]| dual_oracle(&inst, lam[0].max(0.0)).0,
            |lam: &[f64]| vec![dual_oracle(&inst, lam[0].max(0.0)).1],
        );
        assert!(
            result.value <= opt + 1e-6,
            "case {case}: Lagrangian bound {} exceeds the optimum {opt}",
            result.value
        );
    }
    assert!(checked > 50, "only {checked} feasible instances");
}
