//! Randomised check of Benders decomposition against the same two-stage problem
//! solved as one monolithic MILP.
//!
//! The monolithic model has the first-stage variables, plus each scenario's
//! recourse variables with the scenario weight folded into their cost. Benders
//! must return the same optimal value. When the monolithic model is infeasible,
//! Benders must not report an optimum.

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::{Solver, SolverStatus, VarBound};
use tpt_opt_decompose::{BendersProblem, BendersSolver, BlockRow, RecourseBlock, RowSense};
use tpt_opt_milp::MilpSolver;

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

fn random_problem(rng: &mut Lcg) -> BendersProblem {
    let nx = rng.int(1, 2) as usize;
    let first_integer: Vec<bool> = (0..nx).map(|_| rng.int(0, 1) == 1).collect();
    let first_bounds: Vec<(f64, f64)> = (0..nx).map(|_| (0.0, rng.int(2, 4) as f64)).collect();
    let first_cost: Vec<f64> = (0..nx).map(|_| rng.int(0, 3) as f64).collect();

    let n_blocks = rng.int(1, 3) as usize;
    let mut blocks = Vec::new();
    for _ in 0..n_blocks {
        let ny = rng.int(1, 2) as usize;
        // Non-negative recourse costs with finite upper bounds keep each block
        // bounded below, so the problem is bounded whenever it is feasible.
        let cost: Vec<f64> = (0..ny).map(|_| rng.int(1, 5) as f64).collect();
        let y_upper: Vec<f64> = (0..ny).map(|_| rng.int(2, 6) as f64).collect();
        let mut rows = Vec::new();
        for _ in 0..rng.int(1, 3) {
            let y: Vec<f64> = (0..ny).map(|_| rng.int(-2, 3) as f64).collect();
            let x: Vec<f64> = (0..nx).map(|_| rng.int(-2, 2) as f64).collect();
            let sense = match rng.int(0, 2) {
                0 => RowSense::Le,
                1 => RowSense::Ge,
                _ => RowSense::Eq,
            };
            let rhs = rng.int(-3, 6) as f64;
            rows.push(BlockRow { y, x, sense, rhs });
        }
        blocks.push(RecourseBlock { cost, rows, y_upper });
    }
    let weights: Vec<f64> = (0..n_blocks).map(|_| rng.int(1, 3) as f64).collect();
    BendersProblem { first_cost, first_bounds, first_integer, blocks, weights }
}

/// The same problem as one MILP. Returns `None` when it is infeasible.
fn monolithic(p: &BendersProblem) -> Option<f64> {
    let total: f64 = p.weights.iter().sum();
    let mut m = Model::new(0);
    let mut x_idx = Vec::new();
    for (j, &(lo, hi)) in p.first_bounds.iter().enumerate() {
        let bound = if p.first_integer[j] {
            VarBound::integer(lo, hi)
        } else {
            VarBound::continuous(lo, hi)
        };
        x_idx.push(m.add_variable(bound));
    }

    let mut objective_idx = x_idx.clone();
    let mut objective_coeffs: Vec<f64> = p.first_cost.clone();
    let constant = 0.0;
    for (b, block) in p.blocks.iter().enumerate() {
        let w = p.weights[b] / total;
        let y_idx: Vec<usize> =
            block.y_upper.iter().map(|&u| m.add_variable(VarBound::continuous(0.0, u))).collect();
        for (k, &c) in block.cost.iter().enumerate() {
            objective_idx.push(y_idx[k]);
            objective_coeffs.push(w * c);
        }
        for row in &block.rows {
            let mut indices = Vec::new();
            let mut coeffs = Vec::new();
            for (k, &a) in row.y.iter().enumerate() {
                indices.push(y_idx[k]);
                coeffs.push(a);
            }
            for (j, &a) in row.x.iter().enumerate() {
                indices.push(x_idx[j]);
                coeffs.push(a);
            }
            let con = match row.sense {
                RowSense::Le => Constraint::le(indices, coeffs, row.rhs),
                RowSense::Ge => Constraint::ge(indices, coeffs, row.rhs),
                RowSense::Eq => Constraint::equality(indices, coeffs, row.rhs),
            };
            m.add_constraint(con);
        }
    }
    m.set_objective(Objective {
        sense: Sense::Minimize,
        indices: objective_idx,
        coeffs: objective_coeffs,
        constant,
    });
    match MilpSolver::new().solve(&m) {
        Ok(s) if s.status == SolverStatus::Optimal => Some(s.objective_value),
        Ok(s) if s.status == SolverStatus::Infeasible => None,
        other => panic!("monolithic reference failed: {other:?}"),
    }
}

#[test]
fn benders_matches_the_monolithic_optimum() {
    let mut rng = Lcg(31);
    let (mut optimal, mut infeasible) = (0, 0);
    for case in 0..400 {
        let problem = random_problem(&mut rng);
        let reference = monolithic(&problem);
        let got = BendersSolver::new(&problem).solve();
        match reference {
            Some(best) => {
                optimal += 1;
                let res = got.unwrap_or_else(|e| panic!("case {case}: Benders failed: {e:?}"));
                assert_eq!(res.status, SolverStatus::Optimal, "case {case}: status");
                assert!(
                    (res.objective - best).abs() < 1e-5,
                    "case {case}: Benders {} vs monolithic {best}",
                    res.objective
                );
            }
            None => {
                infeasible += 1;
                let ok = match &got {
                    Err(_) => true,
                    Ok(res) => res.status != SolverStatus::Optimal,
                };
                assert!(
                    ok,
                    "case {case}: monolithic is infeasible but Benders reported an optimum"
                );
            }
        }
    }
    assert!(optimal > 40 && infeasible > 5, "optimal {optimal}, infeasible {infeasible}");
}
