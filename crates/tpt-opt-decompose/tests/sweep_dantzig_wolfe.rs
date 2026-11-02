//! Randomised check of Dantzig–Wolfe decomposition against the same block
//! program solved as one monolithic LP.
//!
//! Every block variable carries an explicit upper bound, so each block is a
//! bounded polytope and column generation converges. The monolithic LP has the
//! same feasible set, so the two optima must agree. Where the monolithic LP is
//! infeasible, Dantzig–Wolfe must not report an optimum.

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::{Solver, SolverStatus, VarBound};
use tpt_opt_decompose::{DantzigWolfe, DwBlock, DwLocalRow, DwProblem, RowSense};
use tpt_opt_milp::MilpSolver;

const UPPER: f64 = 5.0;

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

fn sense(rng: &mut Lcg) -> RowSense {
    match rng.int(0, 2) {
        0 => RowSense::Le,
        1 => RowSense::Ge,
        _ => RowSense::Eq,
    }
}

fn random_problem(rng: &mut Lcg) -> DwProblem {
    let n_blocks = rng.int(2, 3) as usize;
    let n_coupling = rng.int(1, 2) as usize;
    let mut blocks = Vec::new();
    for _ in 0..n_blocks {
        let nv = rng.int(1, 3) as usize;
        let cost: Vec<f64> = (0..nv).map(|_| rng.int(-3, 3) as f64).collect();
        let coupling: Vec<Vec<f64>> =
            (0..n_coupling).map(|_| (0..nv).map(|_| rng.int(-2, 3) as f64).collect()).collect();
        // Explicit upper bounds make every block a polytope, as the monolithic model has.
        let mut local_rows: Vec<DwLocalRow> = (0..nv)
            .map(|j| {
                let mut coeffs = vec![0.0; nv];
                coeffs[j] = 1.0;
                DwLocalRow { coeffs, sense: RowSense::Le, rhs: UPPER }
            })
            .collect();
        for _ in 0..rng.int(0, 2) {
            let coeffs: Vec<f64> = (0..nv).map(|_| rng.int(-2, 3) as f64).collect();
            let rhs = rng.int(-2, 8) as f64;
            local_rows.push(DwLocalRow { coeffs, sense: sense(rng), rhs });
        }
        blocks.push(DwBlock { cost, coupling, local_rows });
    }
    let coupling_rhs: Vec<f64> = (0..n_coupling).map(|_| rng.int(-2, 10) as f64).collect();
    let coupling_sense: Vec<RowSense> = (0..n_coupling).map(|_| sense(rng)).collect();
    DwProblem { coupling_rhs, coupling_sense, blocks }
}

/// The same program as one LP. Returns `None` when infeasible.
fn monolithic(p: &DwProblem) -> Option<f64> {
    let mut m = Model::new(0);
    let mut offsets = Vec::new();
    for block in &p.blocks {
        let start = m.num_vars;
        for _ in &block.cost {
            m.add_variable(VarBound::continuous(0.0, UPPER));
        }
        offsets.push(start);
    }
    let mut obj_idx = Vec::new();
    let mut obj_coeffs = Vec::new();
    for (k, block) in p.blocks.iter().enumerate() {
        for (j, &c) in block.cost.iter().enumerate() {
            obj_idx.push(offsets[k] + j);
            obj_coeffs.push(c);
        }
        for row in &block.local_rows {
            let idx: Vec<usize> = (0..row.coeffs.len()).map(|j| offsets[k] + j).collect();
            m.add_constraint(make(idx, row.coeffs.clone(), row.sense, row.rhs));
        }
    }
    for (r, &rhs) in p.coupling_rhs.iter().enumerate() {
        let mut idx = Vec::new();
        let mut coeffs = Vec::new();
        for (k, block) in p.blocks.iter().enumerate() {
            for j in 0..block.cost.len() {
                idx.push(offsets[k] + j);
                coeffs.push(block.coupling[r][j]);
            }
        }
        m.add_constraint(make(idx, coeffs, p.coupling_sense[r], rhs));
    }
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

fn make(indices: Vec<usize>, coeffs: Vec<f64>, sense: RowSense, rhs: f64) -> Constraint {
    match sense {
        RowSense::Le => Constraint::le(indices, coeffs, rhs),
        RowSense::Ge => Constraint::ge(indices, coeffs, rhs),
        RowSense::Eq => Constraint::equality(indices, coeffs, rhs),
    }
}

#[test]
fn dantzig_wolfe_matches_the_monolithic_lp() {
    let mut rng = Lcg(41);
    let (mut optimal, mut infeasible) = (0, 0);
    for case in 0..400 {
        let problem = random_problem(&mut rng);
        let reference = monolithic(&problem);
        let got = DantzigWolfe::new(&problem).solve();
        match reference {
            Some(best) => {
                optimal += 1;
                let res =
                    got.unwrap_or_else(|e| panic!("case {case}: Dantzig–Wolfe failed: {e:?}"));
                assert_eq!(res.status, SolverStatus::Optimal, "case {case}: status");
                assert!(
                    (res.objective - best).abs() < 1e-5,
                    "case {case}: Dantzig–Wolfe {} vs monolithic {best}",
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
                    "case {case}: monolithic is infeasible but Dantzig–Wolfe reported an optimum"
                );
            }
        }
    }
    assert!(optimal > 40 && infeasible > 5, "optimal {optimal}, infeasible {infeasible}");
}
