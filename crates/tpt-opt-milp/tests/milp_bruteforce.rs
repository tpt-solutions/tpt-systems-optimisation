//! Randomised correctness check of the MILP solver against exhaustive
//! enumeration.
//!
//! Each case is a small pure-integer program with bounded variables and `<=`,
//! `>=`, and `=` rows. Enumerating every integer point of the box gives the true
//! feasibility and optimum. The solver must agree on the status and the optimal
//! value, and its returned point must be feasible. The same check is repeated
//! with several thread counts and seeds, because the parallel and heuristic
//! paths are separate code.

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::{Solver, SolverStatus, VarBound};
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

struct Case {
    model: Model,
    /// Integer box per variable, for enumeration.
    boxes: Vec<(i64, i64)>,
}

fn random_case(rng: &mut Lcg) -> Case {
    let n = rng.int(2, 4) as usize;
    let mut model = Model::new(0);
    let mut boxes = Vec::new();
    for _ in 0..n {
        let lo = rng.int(-2, 1);
        let hi = lo + rng.int(1, 4);
        boxes.push((lo, hi));
        model.add_variable(VarBound::integer(lo as f64, hi as f64));
    }
    for _ in 0..rng.int(1, 3) {
        let idx: Vec<usize> = (0..n).collect();
        let coeffs: Vec<f64> = (0..n).map(|_| rng.int(-3, 3) as f64).collect();
        let rhs = rng.int(-4, 8) as f64;
        let con = match rng.int(0, 2) {
            0 => Constraint::le(idx, coeffs, rhs),
            1 => Constraint::ge(idx, coeffs, rhs),
            _ => Constraint::equality(idx, coeffs, rhs),
        };
        model.add_constraint(con);
    }
    let coeffs: Vec<f64> = (0..n).map(|_| rng.int(-3, 3) as f64).collect();
    let sense = if rng.int(0, 1) == 0 { Sense::Minimize } else { Sense::Maximize };
    model.set_objective(Objective {
        sense,
        indices: (0..n).collect(),
        coeffs,
        constant: rng.int(-2, 2) as f64,
    });
    Case { model, boxes }
}

/// Best objective over every feasible integer point, or `None` if there is none.
fn enumerate(case: &Case) -> Option<f64> {
    let m = &case.model;
    let mut best: Option<f64> = None;
    // Start every counter at its lower bound so only in-box points are visited.
    let mut x: Vec<i64> = case.boxes.iter().map(|&(lo, _)| lo).collect();
    loop {
        let point: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        if m.constraints.iter().all(|c| c.slack(&point) >= -1e-9) {
            let value = m.objective.eval(&point);
            best = Some(match (best, m.objective.sense) {
                (None, _) => value,
                (Some(b), Sense::Minimize) => b.min(value),
                (Some(b), Sense::Maximize) => b.max(value),
            });
        }
        // Advance the mixed-radix counter over the box.
        let mut k = 0;
        loop {
            if k == x.len() {
                return best;
            }
            let (lo, hi) = case.boxes[k];
            if x[k] < hi {
                x[k] += 1;
                break;
            }
            x[k] = lo;
            k += 1;
        }
    }
}

fn check(seed: u64, cases: usize, make_solver: impl Fn() -> MilpSolver) {
    let mut rng = Lcg(seed);
    let mut optimal = 0;
    let mut infeasible = 0;
    for case_no in 0..cases {
        let case = random_case(&mut rng);
        let truth = enumerate(&case);
        let sol = make_solver().solve(&case.model).expect("solve");
        match truth {
            Some(best) => {
                optimal += 1;
                assert_eq!(sol.status, SolverStatus::Optimal, "seed {seed} case {case_no}: status");
                assert!(
                    (sol.objective_value - best).abs() < 1e-6,
                    "seed {seed} case {case_no}: objective {} vs enumerated {best}",
                    sol.objective_value
                );
                let point: Vec<f64> = sol.primal.clone();
                assert!(
                    case.model.constraints.iter().all(|c| c.slack(&point) >= -1e-6),
                    "seed {seed} case {case_no}: returned point violates a row"
                );
            }
            None => {
                infeasible += 1;
                assert_eq!(
                    sol.status,
                    SolverStatus::Infeasible,
                    "seed {seed} case {case_no}: status"
                );
            }
        }
    }
    assert!(
        optimal > cases / 10 && infeasible > cases / 20,
        "optimal {optimal}, infeasible {infeasible}"
    );
}

#[test]
fn sequential_solver_matches_enumeration() {
    check(11, 600, MilpSolver::new);
}

#[test]
fn seeded_solver_matches_enumeration() {
    check(12, 300, || MilpSolver::new().with_seed(99));
}

#[test]
fn parallel_solver_matches_enumeration() {
    check(13, 300, || MilpSolver::new().with_threads(4));
}

#[test]
fn cut_loop_matches_enumeration() {
    check(14, 300, || MilpSolver::new().with_cuts(true));
}
