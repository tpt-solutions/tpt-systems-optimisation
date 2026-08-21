//! Integration tests for the `tpt-opt-milp` branch-and-bound solver.

use tpt_opt_core::{
    bounds::VarBound,
    model::{Constraint, Model, Objective},
    solver::{Solver, SolverStatus},
};
use tpt_opt_milp::MilpSolver;

#[test]
fn binary_knapsack_like() {
    // maximise x + y  s.t. 2x + y <= 1.5,  x, y binary
    let mut m = Model::new(2);
    m.set_objective(Objective::maximize(vec![0, 1], vec![1.0, 1.0]));
    m.add_constraint(Constraint::le(vec![0, 1], vec![2.0, 1.0], 1.5));
    m.variables[0].bound = VarBound::binary();
    m.variables[1].bound = VarBound::binary();

    let mut solver = MilpSolver::new();
    let sol = solver.solve(&m).unwrap();
    assert_eq!(sol.status, SolverStatus::Optimal);
    assert!((sol.objective_value - 1.0).abs() < 1e-6, "obj={}", sol.objective_value);
    // one of the binaries is 1, the other 0
    let ones =
        sol.primal_value(0).unwrap().round() as i32 + sol.primal_value(1).unwrap().round() as i32;
    assert_eq!(ones, 1);
}

#[test]
fn small_milp_optimum() {
    // maximise 3x + 2y  s.t.  x + y <= 4,  x <= 2,  x integer >=0, y continuous>=0
    // optimum at x=2, y=2 -> 10
    let mut m = Model::new(2);
    m.set_objective(Objective::maximize(vec![0, 1], vec![3.0, 2.0]));
    m.add_constraint(Constraint::le(vec![0, 1], vec![1.0, 1.0], 4.0));
    m.add_constraint(Constraint::le(vec![0], vec![1.0], 2.0));
    m.variables[0].bound = VarBound::integer(0.0, 100.0);
    m.variables[1].bound = VarBound::continuous(0.0, f64::INFINITY);

    let mut solver = MilpSolver::new().with_branching(tpt_opt_milp::BranchingRule::PseudoCost);
    let sol = solver.solve(&m).unwrap();
    assert_eq!(sol.status, SolverStatus::Optimal);
    assert!((sol.objective_value - 10.0).abs() < 1e-4, "obj={}", sol.objective_value);
    assert!((sol.primal_value(0).unwrap() - 2.0).abs() < 1e-4);
}

#[test]
fn infeasible_model() {
    // x >= 2 and x <= 1 with x binary -> infeasible
    let mut m = Model::new(1);
    m.set_objective(Objective::maximize(vec![0], vec![1.0]));
    m.add_constraint(Constraint::ge(vec![0], vec![1.0], 2.0));
    m.add_constraint(Constraint::le(vec![0], vec![1.0], 1.0));
    m.variables[0].bound = VarBound::binary();

    let mut solver = MilpSolver::new();
    let sol = solver.solve(&m).unwrap();
    assert_eq!(sol.status, SolverStatus::Infeasible);
}

#[test]
fn deterministic_seed() {
    let mut m = Model::new(3);
    m.set_objective(Objective::maximize(vec![0, 1, 2], vec![2.0, 3.0, 1.0]));
    m.add_constraint(Constraint::le(vec![0, 1, 2], vec![1.0, 1.0, 1.0], 2.0));
    for i in 0..3 {
        m.variables[i].bound = VarBound::binary();
    }
    let a = MilpSolver::new().with_seed(7).solve(&m).unwrap();
    let b = MilpSolver::new().with_seed(7).solve(&m).unwrap();
    assert_eq!(a.objective_value, b.objective_value);
}
