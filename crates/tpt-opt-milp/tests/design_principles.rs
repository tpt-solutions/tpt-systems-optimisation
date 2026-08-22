//! Design-principle verification tests (spec §4 / todo.md cross-cutting checklist).
//!
//! Covers four principles end-to-end against the branch-and-bound solver:
//!
//! - **Extensibility**: a user-defined [`CustomConstraint`] drives an outer
//!   approximation repair loop around the MILP solver.
//! - **Numerical Stability**: badly-scaled and near-degenerate models still
//!   solve to the correct optimum.
//! - **Parallelism**: parallel tree search reproduces the sequential result
//!   exactly for a fixed seed.
//! - **Reproducibility / fuzz invariants**: randomly generated (but seeded and
//!   therefore deterministic) MILPs satisfy feasibility, integrality,
//!   objective-consistency and brute-force-optimality invariants.

use tpt_opt_core::custom::CustomConstraint;
use tpt_opt_core::{
    bounds::VarBound,
    model::{Constraint, Model, Objective},
    solver::{Solver, SolverStatus},
};
use tpt_opt_milp::MilpSolver;

// ---------------------------------------------------------------------------
// Extensibility: CustomConstraint end-to-end
// ---------------------------------------------------------------------------

/// Convex quadratic ball constraint `x^2 + y^2 <= r^2` expressed through the
/// core extensibility hook instead of the sparse linear matrix.
struct BallConstraint {
    radius_sq: f64,
}

impl CustomConstraint for BallConstraint {
    fn arity(&self) -> usize {
        2
    }

    fn evaluate(&self, x: &[f64]) -> f64 {
        // Violation: g(x) = x^2 + y^2 - r^2 (> 0 means violated).
        x[0] * x[0] + x[1] * x[1] - self.radius_sq
    }

    fn gradient(&self, x: &[f64], grad: &mut [f64]) {
        grad[0] = 2.0 * x[0];
        grad[1] = 2.0 * x[1];
    }
}

#[test]
fn custom_constraint_drives_outer_approximation_loop() {
    // Maximise x + y over the box [0, 10]^2 intersected with the ball
    // x^2 + y^2 <= 25. The ball is NOT part of the linear model; it lives in
    // a CustomConstraint. An outer-approximation loop solves the MILP, checks
    // the custom constraint at the returned point, and adds the supporting
    // hyperplane cut when violated, until the point is feasible for the ball.
    //
    // Analytic optimum: max x + y on the radius-5 circle in the first quadrant
    // is 5 * sqrt(2) ~= 7.0710678.
    const R_SQ: f64 = 25.0;
    let target = 5.0 * std::f64::consts::SQRT_2;

    let mut model = Model::new(2);
    model.set_objective(Objective::maximize(vec![0, 1], vec![1.0, 1.0]));
    for i in 0..2 {
        model.variables[i].bound = VarBound::continuous(0.0, 10.0);
    }

    let custom = BallConstraint { radius_sq: R_SQ };
    let mut solver = MilpSolver::new();
    let mut sol = solver.solve(&model).expect("box problem always solvable");

    let mut x = [sol.primal_value(0).unwrap(), sol.primal_value(1).unwrap()];
    let mut rounds = 0;
    while custom.is_violated(&x, 1e-9) {
        rounds += 1;
        assert!(rounds <= 50, "outer approximation failed to converge");
        // Supporting-hyperplane (OA) cut at the violated point x:
        //   g(x_hat) + grad g(x_hat) . (z - x_hat) <= 0
        //   with g(z) = z0^2 + z1^2 - r^2 and grad g = (2*z0, 2*z1)
        // => 2*x_hat*z0 + 2*y_hat*z1 <= x_hat^2 + y_hat^2 + r^2
        let rhs = R_SQ + x[0] * x[0] + x[1] * x[1];
        model.add_constraint(Constraint::le(vec![0, 1], vec![2.0 * x[0], 2.0 * x[1]], rhs));
        sol = solver.solve(&model).expect("OA master stays feasible");
        x = [sol.primal_value(0).unwrap(), sol.primal_value(1).unwrap()];
    }

    assert_eq!(sol.status, SolverStatus::Optimal);
    assert!(
        (sol.objective_value - target).abs() < 1e-3,
        "OA loop should reach 5*sqrt(2), got {}",
        sol.objective_value
    );
    assert!(!custom.is_violated(&x, 1e-6), "final point must satisfy the ball");
}

#[test]
fn custom_constraint_validates_solution_directly() {
    // Simpler end-to-end pattern: solve a plain MILP, then use the custom
    // constraint's default `is_violated` to accept/reject the solution.
    let mut m = Model::new(2);
    m.set_objective(Objective::maximize(vec![0, 1], vec![3.0, 2.0]));
    m.add_constraint(Constraint::le(vec![0, 1], vec![1.0, 1.0], 4.0));
    m.variables[0].bound = VarBound::integer(0.0, 4.0);
    m.variables[1].bound = VarBound::integer(0.0, 4.0);

    let sol = MilpSolver::new().solve(&m).unwrap();
    let point = [sol.primal_value(0).unwrap(), sol.primal_value(1).unwrap()];
    // Optimum of max 3x+2y s.t. x+y<=4, integers in [0,4]^2 is (4, 0):
    // norm^2 = 16, comfortably inside a radius-10 budget.
    let budget = BudgetConstraint { limit: 100.0 };
    assert!(!budget.is_violated(&point, 1e-6));
    // A tighter budget (radius 1) correctly flags the same point.
    let tight = BudgetConstraint { limit: 1.0 };
    assert!(tight.is_violated(&point, 1e-6));
    assert!((sol.objective_value - 12.0).abs() < 1e-6);
}

/// Diagonal ellipse-ish budget: x^2 + y^2 <= limit (arity-2 demo).
struct BudgetConstraint {
    limit: f64,
}

impl CustomConstraint for BudgetConstraint {
    fn arity(&self) -> usize {
        2
    }
    fn evaluate(&self, x: &[f64]) -> f64 {
        x[0] * x[0] + x[1] * x[1] - self.limit
    }
    fn gradient(&self, x: &[f64], grad: &mut [f64]) {
        grad[0] = 2.0 * x[0];
        grad[1] = 2.0 * x[1];
    }
}

// ---------------------------------------------------------------------------
// Numerical stability
// ---------------------------------------------------------------------------

#[test]
fn badly_scaled_knapsack_keeps_optimum() {
    // Unique-optimum knapsack: weights [5,4,6,3], values [10,7,12,5], cap 11.
    // Best subset {0, 2}: weight 11, value 22. Rescale weights by 1e-6 and
    // values by 1e6: the optimum must be unchanged up to scaling.
    let w = [5.0, 4.0, 6.0, 3.0];
    let v = [10.0, 7.0, 12.0, 5.0];

    let build = |wscale: f64, vscale: f64, cap: f64| {
        let mut m = Model::new(4);
        m.set_objective(Objective::maximize(
            (0..4).collect(),
            v.iter().map(|&x| x * vscale).collect(),
        ));
        m.add_constraint(Constraint::le(
            (0..4).collect(),
            w.iter().map(|&x| x * wscale).collect(),
            cap,
        ));
        for i in 0..4 {
            m.variables[i].bound = VarBound::binary();
        }
        m
    };

    let base = MilpSolver::new().solve(&build(1.0, 1.0, 11.0)).unwrap();
    let tiny_w = MilpSolver::new().solve(&build(1e-6, 1.0, 11e-6)).unwrap();
    let huge_v = MilpSolver::new().solve(&build(1.0, 1e6, 11.0)).unwrap();

    assert_eq!(base.status, SolverStatus::Optimal);
    assert_eq!(tiny_w.status, SolverStatus::Optimal);
    assert_eq!(huge_v.status, SolverStatus::Optimal);
    assert!((base.objective_value - 22.0).abs() < 1e-6, "base obj {}", base.objective_value);
    assert!(
        (tiny_w.objective_value - 22.0).abs() < 1e-6,
        "tiny-weight obj {}",
        tiny_w.objective_value
    );
    assert!(
        (huge_v.objective_value - 22e6).abs() < 1e0,
        "huge-value obj {}",
        huge_v.objective_value
    );
}

#[test]
fn mixed_magnitude_row_solves_correctly() {
    // One row mixing 1e-6 and 1e6 magnitude coefficients plus a duplicate
    // near-degenerate equality at two scales. Correct optimum: x = 1, y = 0,
    // objective 1e6.
    let mut m = Model::new(2);
    m.set_objective(Objective::maximize(vec![0, 1], vec![1e6, 1.0]));
    // 1e-6 x + 1e6 y <= 1  forces y = 0 and allows x up to 1e6, but the box
    // caps x at 1.
    m.add_constraint(Constraint::le(vec![0, 1], vec![1e-6, 1e6], 1.0));
    m.add_constraint(Constraint::le(vec![0, 1], vec![1.0, 1.0], 1.0));
    m.variables[0].bound = VarBound::continuous(0.0, 1.0);
    m.variables[1].bound = VarBound::continuous(0.0, 1.0);

    let sol = MilpSolver::new().solve(&m).unwrap();
    assert_eq!(sol.status, SolverStatus::Optimal);
    assert!((sol.objective_value - 1e6).abs() < 1e-3, "obj {}", sol.objective_value);
    assert!(sol.primal_value(1).unwrap() < 1e-9, "y must be 0");
}

#[test]
fn degenerate_duplicate_rows_stay_optimal() {
    // min -(x + y) with the same equality present at two wildly different
    // scalings: x + y = 1 and 1e6*(x + y) = 1e6. Must return 1, not cycle or
    // report a numerical failure.
    let mut m = Model::new(2);
    m.set_objective(Objective::maximize(vec![0, 1], vec![1.0, 1.0]));
    m.add_constraint(Constraint::equality(vec![0, 1], vec![1.0, 1.0], 1.0));
    m.add_constraint(Constraint::equality(vec![0, 1], vec![1e6, 1e6], 1e6));
    m.variables[0].bound = VarBound::continuous(0.0, 1.0);
    m.variables[1].bound = VarBound::continuous(0.0, 1.0);

    let sol = MilpSolver::new().solve(&m).unwrap();
    assert_eq!(sol.status, SolverStatus::Optimal);
    assert!((sol.objective_value - 1.0).abs() < 1e-6, "obj {}", sol.objective_value);
}

// ---------------------------------------------------------------------------
// Parallelism: parallel result == sequential result for a fixed seed
// ---------------------------------------------------------------------------

/// Deterministic xorshift64* RNG for reproducible instance generation.
struct XorShift(u64);

impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

fn random_knapsack(seed: u64, items: usize) -> Model {
    let mut rng = XorShift(seed);
    let mut m = Model::new(items);
    let mut weights = Vec::with_capacity(items);
    let mut values = Vec::with_capacity(items);
    for _ in 0..items {
        weights.push(rng.below(100) as f64 + 1.0);
        values.push(rng.below(100) as f64 + 1.0);
    }
    let total: f64 = weights.iter().sum();
    let cap = total * 0.5;
    m.set_objective(Objective::maximize((0..items).collect(), values));
    m.add_constraint(Constraint::le((0..items).collect(), weights, cap));
    for i in 0..items {
        m.variables[i].bound = VarBound::binary();
    }
    m
}

#[test]
fn parallel_tree_search_matches_sequential_exactly() {
    let model = random_knapsack(0xDEAD_BEEF, 28);
    let seq = MilpSolver::new().with_seed(42).with_threads(1).solve(&model).unwrap();
    let par = MilpSolver::new().with_seed(42).with_threads(4).solve(&model).unwrap();

    assert_eq!(seq.status, SolverStatus::Optimal);
    assert_eq!(par.status, SolverStatus::Optimal);
    assert!(
        (seq.objective_value - par.objective_value).abs() < 1e-9,
        "sequential {} vs parallel {}",
        seq.objective_value,
        par.objective_value
    );
    // Seed-deterministic work distribution: identical primal vectors too.
    assert_eq!(seq.primal, par.primal, "primal vectors must match across thread counts");
}

#[test]
fn repeated_sequential_solves_are_bit_identical() {
    let model = random_knapsack(0x1234_5678, 24);
    let a = MilpSolver::new().with_seed(7).solve(&model).unwrap();
    let b = MilpSolver::new().with_seed(7).solve(&model).unwrap();
    assert_eq!(a.objective_value, b.objective_value);
    assert_eq!(a.primal, b.primal);
}

// ---------------------------------------------------------------------------
// Fuzz-style invariant testing on random MILPs
// ---------------------------------------------------------------------------

/// Generate a random binary program that is feasible *by construction*
/// (a planted witness x* satisfies every row), solve it, and verify the
/// solver's invariants:
///
/// 1. terminal status is Optimal (bounded + feasible by construction),
/// 2. the returned primal respects variable bounds and integrality,
/// 3. every constraint row is satisfied within tolerance,
/// 4. the reported objective equals the objective recomputed from the primal,
/// 5. the reported objective equals the brute-force optimum over all 2^n
///    assignments (small n only).
#[test]
fn fuzz_random_binary_programs_satisfy_invariants() {
    let mut rng = XorShift(0xF00D_CAFE);
    for trial in 0..40u32 {
        let n = 5 + (rng.below(6) as usize); // 5..=10 variables
        let rows = 2 + (rng.below(5) as usize); // 2..=6 rows

        // Planted witness.
        let witness: Vec<f64> = (0..n).map(|_| rng.below(2) as f64).collect();

        let mut m = Model::new(n);
        let obj_coeffs: Vec<f64> = (0..n).map(|_| (rng.below(21) as f64) - 10.0).collect();
        m.set_objective(Objective::maximize((0..n).collect(), obj_coeffs.clone()));

        for _ in 0..rows {
            let coeffs: Vec<f64> = (0..n).map(|_| (rng.below(11) as f64) - 5.0).collect();
            let lhs: f64 = coeffs.iter().zip(&witness).map(|(c, x)| c * x).sum();
            // Slack keeps the witness strictly inside the row.
            let slack = rng.below(6) as f64;
            if rng.below(2) == 0 {
                m.add_constraint(Constraint::le((0..n).collect(), coeffs.clone(), lhs + slack));
            } else {
                m.add_constraint(Constraint::ge((0..n).collect(), coeffs.clone(), lhs - slack));
            }
        }
        for i in 0..n {
            m.variables[i].bound = VarBound::binary();
        }

        let sol = MilpSolver::new()
            .with_seed(trial as u64)
            .solve(&m)
            .expect("solver must not error on a well-formed model");

        // Invariant 1: bounded + feasible => Optimal.
        assert_eq!(sol.status, SolverStatus::Optimal, "trial {trial}: {:?}", sol.status);

        // Invariant 2: bounds + integrality.
        for (i, &v) in sol.primal.iter().enumerate() {
            assert!((-1e-6..=1.0 + 1e-6).contains(&v), "trial {trial}: var {i} out of box: {v}");
            let rounded = v.round();
            assert!((v - rounded).abs() < 1e-6, "trial {trial}: var {i} not integral: {v}");
        }

        // Invariant 3: every row satisfied.
        for (ri, row) in m.constraints.iter().enumerate() {
            assert!(row.is_satisfied(&sol.primal, 1e-6), "trial {trial}: row {ri} violated");
        }

        // Invariant 4: objective consistency.
        let recomputed: f64 = obj_coeffs.iter().zip(&sol.primal).map(|(c, x)| c * x).sum();
        assert!(
            (recomputed - sol.objective_value).abs() < 1e-6,
            "trial {trial}: reported {} vs recomputed {}",
            sol.objective_value,
            recomputed
        );

        // Invariant 5: brute-force optimality (n <= 10 => at most 1024 combos).
        let mut best = f64::NEG_INFINITY;
        for mask in 0u32..(1u32 << n) {
            let assign: Vec<f64> = (0..n).map(|i| ((mask >> i) & 1) as f64).collect();
            if m.constraints.iter().all(|row| row.is_satisfied(&assign, 1e-9)) {
                let val: f64 = obj_coeffs.iter().zip(&assign).map(|(c, x)| c * x).sum();
                if val > best {
                    best = val;
                }
            }
        }
        assert!(
            (sol.objective_value - best).abs() < 1e-6,
            "trial {trial}: reported {} vs brute-force optimum {}",
            sol.objective_value,
            best
        );
    }
}
