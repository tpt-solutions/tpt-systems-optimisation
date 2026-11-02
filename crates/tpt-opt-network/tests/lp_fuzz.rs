//! Randomised cross-check of the network crate's LP engine against the MILP
//! engine of `tpt-opt-milp` on small LPs with boxed, possibly negative bounds,
//! and `<=`, `>=`, and `=` rows.
//!
//! For every case where the reference proves optimality, the network solver
//! must match its objective and return a feasible point. Where the reference
//! proves infeasibility, the network solver must return an error.

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::{Solver, SolverStatus, VarBound};
use tpt_opt_milp::MilpSolver;
use tpt_opt_network::LpSolver;

/// Small deterministic generator (LCG), so failures reproduce exactly.
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

fn random_model(rng: &mut Lcg) -> Model {
    let n = rng.int(2, 5) as usize;
    let mut m = Model::new(0);
    for _ in 0..n {
        let lo = rng.int(-3, 2) as f64;
        let hi = lo + rng.int(0, 8) as f64;
        m.add_variable(VarBound::continuous(lo, hi));
    }
    for _ in 0..rng.int(1, 4) {
        let idx: Vec<usize> = (0..n).collect();
        let coeffs: Vec<f64> = (0..n).map(|_| rng.int(-3, 3) as f64).collect();
        let rhs = rng.int(-4, 10) as f64;
        let con = match rng.int(0, 2) {
            0 => Constraint::le(idx, coeffs, rhs),
            1 => Constraint::ge(idx, coeffs, rhs),
            _ => Constraint::equality(idx, coeffs, rhs),
        };
        m.add_constraint(con);
    }
    let coeffs: Vec<f64> = (0..n).map(|_| rng.int(-3, 3) as f64).collect();
    m.set_objective(Objective {
        sense: Sense::Minimize,
        indices: (0..n).collect(),
        coeffs,
        constant: 0.0,
    });
    m
}

/// Largest violation of any bound or constraint at `x` (zero when feasible).
fn violation(m: &Model, x: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for (j, v) in m.variables.iter().enumerate() {
        worst = worst.max(v.bound.bound.lower - x[j]).max(x[j] - v.bound.bound.upper);
    }
    for c in &m.constraints {
        worst = worst.max(-c.slack(x));
    }
    worst
}

#[test]
fn network_lp_agrees_with_the_reference_on_random_lps() {
    let mut rng = Lcg(7);
    let mut optimal = 0;
    let mut infeasible = 0;
    for case in 0..3000 {
        let model = random_model(&mut rng);
        let reference = MilpSolver::new().solve(&model).expect("reference solve");
        let mine = LpSolver::new().solve(&model);
        match reference.status {
            SolverStatus::Optimal => {
                optimal += 1;
                let sol = mine.unwrap_or_else(|e| panic!("case {case}: network failed: {e:?}"));
                assert!(
                    (sol.objective_value - reference.objective_value).abs() < 1e-6,
                    "case {case}: objective {} vs reference {}",
                    sol.objective_value,
                    reference.objective_value
                );
                let viol = violation(&model, &sol.primal);
                assert!(viol < 1e-6, "case {case}: returned point violates a limit by {viol}");
            }
            SolverStatus::Infeasible => {
                infeasible += 1;
                assert!(
                    mine.is_err(),
                    "case {case}: reference is infeasible but network returned a solution"
                );
            }
            _ => {}
        }
    }
    // The generator must exercise both outcomes, or the test proves little.
    assert!(optimal > 500 && infeasible > 50, "optimal {optimal}, infeasible {infeasible}");
}
