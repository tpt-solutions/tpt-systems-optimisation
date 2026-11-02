//! Regression tests for the network crate's two-phase simplex.
//!
//! Each case once exposed a defect: an entering-column search that could pick
//! the right-hand side (a panic on any infeasible LP), artificial columns
//! re-entering in phase II (a wrong optimum), and an empty equality row
//! misreported as unbounded.

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::{OptError, Solver, SolverStatus, VarBound};
use tpt_opt_network::LpSolver;

fn minimise(model: &mut Model, index: usize) {
    model.set_objective(Objective {
        sense: Sense::Minimize,
        indices: vec![index],
        coeffs: vec![1.0],
        constant: 0.0,
    });
}

#[test]
fn fixed_variable_is_respected_not_ignored() {
    // x is fixed at 0, so x + y = 5 forces y = 5 and the optimum is 5.
    let mut m = Model::new(0);
    let x = m.add_variable(VarBound::continuous(0.0, 0.0));
    let y = m.add_variable(VarBound::continuous(0.0, 10.0));
    m.add_constraint(Constraint::equality(vec![x, y], vec![1.0, 1.0], 5.0));
    minimise(&mut m, y);
    let sol = LpSolver::new().solve(&m).unwrap();
    assert_eq!(sol.status, SolverStatus::Optimal);
    assert!((sol.objective_value - 5.0).abs() < 1e-9, "got {}", sol.objective_value);
}

#[test]
fn infeasible_equality_returns_an_infeasibility_error_without_panicking() {
    let mut m = Model::new(0);
    let x = m.add_variable(VarBound::continuous(0.0, 10.0));
    m.add_constraint(Constraint::equality(vec![x], vec![1.0], 20.0));
    minimise(&mut m, x);
    let mut solver = LpSolver::new();
    let err = solver.solve(&m).unwrap_err();
    assert!(matches!(err, OptError::Infeasible(_)), "{err:?}");
    assert_eq!(solver.status(), SolverStatus::Infeasible);
}

#[test]
fn infeasible_pair_of_equalities_is_detected() {
    let mut m = Model::new(0);
    let x = m.add_variable(VarBound::continuous(0.0, 10.0));
    let y = m.add_variable(VarBound::continuous(0.0, 10.0));
    m.add_constraint(Constraint::equality(vec![x, y], vec![1.0, 1.0], 25.0));
    m.add_constraint(Constraint::equality(vec![x, y], vec![1.0, -1.0], 0.0));
    minimise(&mut m, x);
    let mut solver = LpSolver::new();
    assert!(solver.solve(&m).is_err());
    assert_eq!(solver.status(), SolverStatus::Infeasible);
}

#[test]
fn free_variable_with_an_empty_equality_row_is_solved() {
    let mut m = Model::new(0);
    let x = m.add_variable(VarBound::continuous(f64::NEG_INFINITY, f64::INFINITY));
    m.add_constraint(Constraint::equality(vec![x], vec![1.0], 3.0));
    m.add_constraint(Constraint::equality(vec![], vec![], 0.0));
    minimise(&mut m, x);
    let sol = LpSolver::new().solve(&m).unwrap();
    assert!((sol.objective_value - 3.0).abs() < 1e-9, "got {}", sol.objective_value);
}
