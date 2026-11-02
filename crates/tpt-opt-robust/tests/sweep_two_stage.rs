//! Randomised check of the two-stage extensive form against Benders decomposition.
//!
//! `TwoStageProblem::solve` builds the deterministic equivalent and solves it
//! monolithically. Benders decomposition solves the same problem by a different
//! route: one recourse block per scenario, weighted by its probability. The two
//! optima must agree. Where Benders reports infeasibility, the robust crate must
//! not report a solution.

use tpt_opt_core::SolverStatus;
use tpt_opt_decompose::{
    BendersProblem, BendersSolver, BlockRow, RecourseBlock, RowSense as BRowSense,
};
use tpt_opt_robust::scenario::{RowSense, Scenario, StageData, StageRow, TwoStageProblem};

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

fn random_problem(rng: &mut Lcg) -> TwoStageProblem {
    let n1 = rng.int(1, 2) as usize;
    let n2 = rng.int(1, 2) as usize;
    let first_cost: Vec<f64> = (0..n1).map(|_| rng.int(0, 3) as f64).collect();
    let first_bounds = vec![(0.0, 4.0); n1];
    let second_bounds = vec![(0.0, 5.0); n2];
    let n_scen = rng.int(2, 3) as usize;
    let scenarios = (0..n_scen)
        .map(|_| {
            let probability = rng.int(1, 3) as f64;
            // Positive recourse costs keep every scenario bounded below.
            let cost: Vec<f64> = (0..n2).map(|_| rng.int(1, 5) as f64).collect();
            let rows = (0..rng.int(1, 3))
                .map(|_| StageRow {
                    w: (0..n1).map(|_| rng.int(-2, 2) as f64).collect(),
                    t: (0..n2).map(|_| rng.int(-2, 3) as f64).collect(),
                    h: rng.int(-2, 6) as f64,
                    sense: sense(rng),
                })
                .collect();
            (Scenario { probability, data: vec![] }, StageData { cost, rows })
        })
        .collect();
    TwoStageProblem { first_cost, first_bounds, second_bounds, scenarios }
}

/// The same problem as Benders blocks: one per scenario, weighted by probability.
fn as_benders(p: &TwoStageProblem) -> BendersProblem {
    let blocks = p
        .scenarios
        .iter()
        .map(|(_, data)| RecourseBlock {
            cost: data.cost.clone(),
            rows: data
                .rows
                .iter()
                .map(|r| BlockRow {
                    y: r.t.clone(),
                    x: r.w.clone(),
                    sense: match r.sense {
                        RowSense::Le => BRowSense::Le,
                        RowSense::Ge => BRowSense::Ge,
                        RowSense::Eq => BRowSense::Eq,
                    },
                    rhs: r.h,
                })
                .collect(),
            y_upper: p.second_bounds.iter().map(|b| b.1).collect(),
        })
        .collect();
    BendersProblem {
        first_cost: p.first_cost.clone(),
        first_bounds: p.first_bounds.clone(),
        first_integer: vec![false; p.first_cost.len()],
        blocks,
        weights: p.scenarios.iter().map(|(s, _)| s.probability).collect(),
    }
}

#[test]
fn extensive_form_matches_benders() {
    let mut rng = Lcg(81);
    let (mut optimal, mut infeasible) = (0, 0);
    for case in 0..300 {
        let problem = random_problem(&mut rng);
        let extensive = problem.solve();
        let benders = BendersSolver::new(&as_benders(&problem)).solve();
        match extensive {
            Ok(sol) => {
                optimal += 1;
                let res = benders.unwrap_or_else(|e| panic!("case {case}: Benders failed: {e:?}"));
                assert_eq!(res.status, SolverStatus::Optimal, "case {case}: Benders status");
                assert!(
                    (res.objective - sol.objective).abs() < 1e-5,
                    "case {case}: Benders {} vs extensive form {}",
                    res.objective,
                    sol.objective
                );
            }
            Err(_) => {
                infeasible += 1;
                let ok = match &benders {
                    Err(_) => true,
                    Ok(res) => res.status != SolverStatus::Optimal,
                };
                assert!(
                    ok,
                    "case {case}: extensive form has no solution but Benders reports an optimum"
                );
            }
        }
    }
    assert!(optimal > 40, "only {optimal} feasible instances (infeasible {infeasible})");
}

/// Scenario 1 forces `x = 0.5` (from `2x = 1`) and then `y = 7` (from
/// `-2x + y = 6`), which exceeds the recourse bound 5. The problem is infeasible,
/// so the solve must fail rather than return the values of an infeasible point.
#[test]
fn infeasible_two_stage_problem_is_an_error() {
    use tpt_opt_robust::scenario::StageRow;
    let problem = TwoStageProblem {
        first_cost: vec![0.0],
        first_bounds: vec![(0.0, 4.0)],
        second_bounds: vec![(0.0, 5.0)],
        scenarios: vec![(
            Scenario { probability: 1.0, data: vec![] },
            StageData {
                cost: vec![4.0],
                rows: vec![
                    StageRow { w: vec![2.0], t: vec![0.0], h: 1.0, sense: RowSense::Eq },
                    StageRow { w: vec![-2.0], t: vec![1.0], h: 6.0, sense: RowSense::Eq },
                ],
            },
        )],
    };
    assert!(problem.solve().is_err(), "an infeasible problem must not return a solution");
}
