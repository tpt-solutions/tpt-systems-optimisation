//! Randomised check of branch-and-price against the same program solved as one
//! monolithic MILP.
//!
//! Each block is a box with integer bounds and no other local rows, so every
//! block polytope is integral and its LP extreme points are integer points. That
//! is the condition under which pricing with the LP pricer is exact, and it gives
//! branch-and-price the same integer feasible set as the monolithic model.

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::OptError;
use tpt_opt_core::{Solver, SolverStatus, VarBound};
use tpt_opt_decompose::{BranchAndPrice, Column, DwBlock, DwLocalRow, DwProblem, Pricer, RowSense};
use tpt_opt_milp::MilpSolver;

const UPPER: i64 = 4;

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
    let blocks: Vec<DwBlock> = (0..n_blocks)
        .map(|_| {
            let nv = rng.int(1, 2) as usize;
            let cost: Vec<f64> = (0..nv).map(|_| rng.int(-3, 3) as f64).collect();
            let coupling: Vec<Vec<f64>> =
                (0..n_coupling).map(|_| (0..nv).map(|_| rng.int(-2, 3) as f64).collect()).collect();
            // Upper bounds as rows make each block a bounded box, as the monolithic model has.
            let local_rows = (0..nv)
                .map(|j| {
                    let mut coeffs = vec![0.0; nv];
                    coeffs[j] = 1.0;
                    DwLocalRow { coeffs, sense: RowSense::Le, rhs: UPPER as f64 }
                })
                .collect();
            DwBlock { cost, coupling, local_rows }
        })
        .collect();
    let coupling_rhs: Vec<f64> = (0..n_coupling).map(|_| rng.int(-2, 10) as f64).collect();
    let coupling_sense: Vec<RowSense> = (0..n_coupling).map(|_| sense(rng)).collect();
    DwProblem { coupling_rhs, coupling_sense, blocks }
}

/// Monolithic MILP with integer block variables in `[0, UPPER]`. `None` when
/// infeasible.
fn monolithic(p: &DwProblem) -> Option<f64> {
    let mut m = Model::new(0);
    let mut offsets = Vec::new();
    for block in &p.blocks {
        offsets.push(m.num_vars);
        for _ in &block.cost {
            m.add_variable(VarBound::integer(0.0, UPPER as f64));
        }
    }
    let mut obj_idx = Vec::new();
    let mut obj_coeffs = Vec::new();
    for (k, block) in p.blocks.iter().enumerate() {
        for (j, &c) in block.cost.iter().enumerate() {
            obj_idx.push(offsets[k] + j);
            obj_coeffs.push(c);
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
        m.add_constraint(match p.coupling_sense[r] {
            RowSense::Le => Constraint::le(idx, coeffs, rhs),
            RowSense::Ge => Constraint::ge(idx, coeffs, rhs),
            RowSense::Eq => Constraint::equality(idx, coeffs, rhs),
        });
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

/// Branch-and-price here solves the integer master over the columns that column
/// generation produced, so it is not a complete branch-and-price. It is sound:
/// a reported optimum is a feasible integer solution, so it can never beat the
/// true optimum. It is not always exact, because the optimal combination may use
/// columns that never enter the pool. The sweep checks soundness on every case
/// and reports how often the pool reaches the exact optimum.
#[test]
fn branch_and_price_is_sound_on_integer_blocks() {
    let mut rng = Lcg(51);
    let (mut feasible, mut exact, mut pool_errors) = (0, 0, 0);
    for case in 0..300 {
        let problem = random_problem(&mut rng);
        let reference = monolithic(&problem);
        let got = BranchAndPrice::new(&problem, EnumPricer { problem: &problem }).solve();
        let Ok(res) = got else {
            pool_errors += 1;
            continue;
        };
        match (res.status, reference) {
            (SolverStatus::Optimal, Some(best)) => {
                feasible += 1;
                // Soundness: the reported solution is feasible and never better than the optimum.
                assert!(
                    res.objective >= best - 1e-6,
                    "case {case}: {} beats the optimum {best}",
                    res.objective
                );
                let mut total = 0.0;
                for (k, block) in problem.blocks.iter().enumerate() {
                    let point = &res.points[k];
                    assert_eq!(point.len(), block.cost.len(), "case {case}: block {k} point");
                    total += dot(&block.cost, point);
                }
                assert!(
                    (total - res.objective).abs() < 1e-6,
                    "case {case}: points do not reproduce the objective"
                );
                for (r, &rhs) in problem.coupling_rhs.iter().enumerate() {
                    let lhs: f64 = problem
                        .blocks
                        .iter()
                        .enumerate()
                        .map(|(k, b)| dot(&b.coupling[r], &res.points[k]))
                        .sum();
                    let ok = match problem.coupling_sense[r] {
                        RowSense::Le => lhs <= rhs + 1e-6,
                        RowSense::Ge => lhs >= rhs - 1e-6,
                        RowSense::Eq => (lhs - rhs).abs() <= 1e-6,
                    };
                    assert!(ok, "case {case}: coupling row {r} violated ({lhs} vs {rhs})");
                }
                if (res.objective - best).abs() < 1e-6 {
                    exact += 1;
                }
            }
            (SolverStatus::Infeasible, Some(_)) => {
                panic!("case {case}: LP-feasible problem reported infeasible")
            }
            (SolverStatus::Optimal, None) => panic!(
                "case {case}: monolithic is infeasible but branch-and-price reported a solution"
            ),
            _ => {}
        }
    }
    println!("branch-and-price: {feasible} sound optima, {exact} exact, {pool_errors} pool-limited errors");
    assert!(feasible > 30, "only {feasible} optimal cases were checked");
}

/// Exact integer pricer: enumerates every integer point of a block's box and
/// returns the most negative reduced-cost column. This is the kind of oracle
/// branch-and-price needs for integer blocks, since the LP pricer only generates
/// box vertices.
struct EnumPricer<'a> {
    problem: &'a DwProblem,
}

impl Pricer for EnumPricer<'_> {
    fn price(
        &mut self,
        block: usize,
        pi: &[f64],
        sigma_k: f64,
    ) -> Result<Option<Column>, OptError> {
        let b = &self.problem.blocks[block];
        let nv = b.cost.len();
        let mut y = vec![0i64; nv];
        let mut best: Option<(f64, Vec<f64>)> = None;
        loop {
            let point: Vec<f64> = y.iter().map(|&v| v as f64).collect();
            let mut rc = dot(&b.cost, &point) - sigma_k;
            for (r, row) in b.coupling.iter().enumerate() {
                rc -= pi[r] * dot(row, &point);
            }
            if best.as_ref().is_none_or(|(b_rc, _)| rc < *b_rc) {
                best = Some((rc, point));
            }
            // Advance the mixed-radix counter over [0, UPPER]^nv.
            let mut k = 0;
            loop {
                if k == nv {
                    let found = best.filter(|(rc, _)| *rc < -1e-9).map(|(_, p)| p);
                    return Ok(found.map(|point| Column {
                        block,
                        cost: dot(&b.cost, &point),
                        coeffs: b.coupling.iter().map(|row| dot(row, &point)).collect(),
                        point,
                    }));
                }
                if y[k] < UPPER {
                    y[k] += 1;
                    break;
                }
                y[k] = 0;
                k += 1;
            }
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
