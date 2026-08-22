//! Cross-solver validation (feature-gated): the bundled branch-and-bound
//! engine vs. the external HiGHS binding on shared small MILP instances.
//!
//! Run with `cargo test -p tpt-opt-milp --features highs`. Building this test
//! compiles the HiGHS C++ sources via `highs-sys`, which requires a C++ build
//! toolchain (cmake + MSVC/gcc/clang).

#![cfg(feature = "highs")]

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::solver::Solver;
use tpt_opt_core::VarBound;
use tpt_opt_milp::{HighsSolver, MilpSolver};

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// 0/1 knapsack: max 10a + 13b + 7c + 4d s.t. 5a + 7b + 4c + 3d ≤ 14.
/// Optimum: b=c=d=1 (weight 14) → 24.
fn knapsack() -> Model {
    let mut model = Model::new(0);
    for _ in 0..4 {
        model.add_variable(VarBound::binary());
    }
    model.set_objective(Objective {
        sense: Sense::Maximize,
        indices: vec![0, 1, 2, 3],
        coeffs: vec![10.0, 13.0, 7.0, 4.0],
        constant: 0.0,
    });
    model.add_constraint(Constraint::le(vec![0, 1, 2, 3], vec![5.0, 7.0, 4.0, 3.0], 14.0));
    model
}

/// Covering: min 2x + 3y + 4z s.t. x + y ≥ 2, y + z ≥ 3, x,y,z ∈ Z≥0.
/// Optimum: y=3 alone covers both rows → 9.
fn covering() -> Model {
    let mut model = Model::new(0);
    for _ in 0..3 {
        model.add_variable(VarBound::integer(0.0, f64::INFINITY));
    }
    model.set_objective(Objective {
        sense: Sense::Minimize,
        indices: vec![0, 1, 2],
        coeffs: vec![2.0, 3.0, 4.0],
        constant: 0.0,
    });
    model.add_constraint(Constraint::ge(vec![0, 1], vec![1.0, 1.0], 2.0));
    model.add_constraint(Constraint::ge(vec![1, 2], vec![1.0, 1.0], 3.0));
    model
}

/// Mixed continuous/integer with an equality row and negative bounds:
/// min x − 2y s.t. x + y = 4.5, x ∈ [−2, 6] cont., y ∈ [0, 5] integer.
/// Optimum: y = 5, x = −0.5 → −10.5.
fn mixed_equality() -> Model {
    let mut model = Model::new(0);
    model.add_variable(VarBound::continuous(-2.0, 6.0));
    model.add_variable(VarBound::integer(0.0, 5.0));
    model.set_objective(Objective {
        sense: Sense::Minimize,
        indices: vec![0, 1],
        coeffs: vec![1.0, -2.0],
        constant: 0.0,
    });
    model.add_constraint(Constraint::equality(vec![0, 1], vec![1.0, 1.0], 4.5));
    model
}

#[test]
fn highs_matches_inhouse_knapsack() {
    let model = knapsack();
    let inhouse = MilpSolver::new().solve(&model).unwrap();
    let highs = HighsSolver::new().solve(&model).unwrap();
    assert_eq!(inhouse.status, tpt_opt_core::SolverStatus::Optimal);
    assert_eq!(highs.status, tpt_opt_core::SolverStatus::Optimal);
    assert!(
        approx(inhouse.objective_value, highs.objective_value, 1e-6),
        "inhouse {} vs highs {}",
        inhouse.objective_value,
        highs.objective_value
    );
    assert!(approx(highs.objective_value, 24.0, 1e-6), "highs obj {}", highs.objective_value);
}

#[test]
fn highs_matches_inhouse_covering() {
    let model = covering();
    let inhouse = MilpSolver::new().solve(&model).unwrap();
    let highs = HighsSolver::new().solve(&model).unwrap();
    assert_eq!(inhouse.status, tpt_opt_core::SolverStatus::Optimal);
    assert_eq!(highs.status, tpt_opt_core::SolverStatus::Optimal);
    assert!(
        approx(inhouse.objective_value, highs.objective_value, 1e-6),
        "inhouse {} vs highs {}",
        inhouse.objective_value,
        highs.objective_value
    );
    assert!(approx(highs.objective_value, 9.0, 1e-6), "highs obj {}", highs.objective_value);
}

#[test]
fn highs_matches_inhouse_mixed_equality() {
    let model = mixed_equality();
    let inhouse = MilpSolver::new().solve(&model).unwrap();
    let highs = HighsSolver::new().solve(&model).unwrap();
    assert_eq!(inhouse.status, tpt_opt_core::SolverStatus::Optimal);
    assert_eq!(highs.status, tpt_opt_core::SolverStatus::Optimal);
    assert!(
        approx(inhouse.objective_value, highs.objective_value, 1e-6),
        "inhouse {} vs highs {}",
        inhouse.objective_value,
        highs.objective_value
    );
    assert!(approx(highs.objective_value, -10.5, 1e-6), "highs obj {}", highs.objective_value);
}

#[test]
fn highs_agrees_on_infeasible() {
    // x ≤ 1, x ≥ 3 over one integer variable.
    let mut model = Model::new(0);
    model.add_variable(VarBound::integer(0.0, 10.0));
    model.set_objective(Objective::minimize(vec![0], vec![1.0]));
    model.add_constraint(Constraint::le(vec![0], vec![1.0], 1.0));
    model.add_constraint(Constraint::ge(vec![0], vec![1.0], 3.0));
    let inhouse = MilpSolver::new().solve(&model).unwrap();
    let highs = HighsSolver::new().solve(&model).unwrap();
    assert_eq!(inhouse.status, tpt_opt_core::SolverStatus::Infeasible);
    assert_eq!(highs.status, tpt_opt_core::SolverStatus::Infeasible);
}

#[test]
fn highs_agrees_on_unbounded_lp() {
    // max x over free x → unbounded.
    let mut model = Model::new(0);
    model.add_variable(VarBound::continuous(f64::NEG_INFINITY, f64::INFINITY));
    model.set_objective(Objective::maximize(vec![0], vec![1.0]));
    let inhouse = MilpSolver::new().solve(&model).unwrap();
    let highs = HighsSolver::new().solve(&model).unwrap();
    assert_eq!(inhouse.status, tpt_opt_core::SolverStatus::Unbounded);
    assert_eq!(highs.status, tpt_opt_core::SolverStatus::Unbounded);
}
