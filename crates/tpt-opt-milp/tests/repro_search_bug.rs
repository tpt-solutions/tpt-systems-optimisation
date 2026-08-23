//! Regression test for the premature-termination search bug from todo.md
//! "Open Risks": node bounds (LP objectives) excluded the model's objective
//! constant while heuristic incumbents included it, so any positive constant
//! on a maximisation model made every node look dominated and the tree was
//! pruned after a single node.

use tpt_opt_core::bounds::VarBound;
use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::Solver;
use tpt_opt_milp::MilpSolver;

/// The exact instance parsed by `mps_parse_handwritten_features`:
///
/// Maximize 3X + 4Y + Z + 5 (constant)
///   s.t. 2X + 3Y + 4Z <= 9
///        Z + W >= 1
///        2 <= W <= 5
///        X, Y integer in [0, 10]; Z in [0, 2.5]; W free
///
/// W carries no objective coefficient. True optimum: X=3, Y=1, Z=0 -> 18.
#[test]
fn objective_constant_is_respected_by_the_tree_search() {
    let mut m = Model::new(4);
    m.set_objective(Objective {
        sense: Sense::Maximize,
        indices: vec![0, 1, 2],
        coeffs: vec![3.0, 4.0, 1.0],
        constant: 5.0,
    });
    m.add_constraint(Constraint::le(vec![0, 1, 2], vec![2.0, 3.0, 4.0], 9.0));
    m.add_constraint(Constraint::ge(vec![2, 3], vec![1.0, 1.0], 1.0));
    m.add_constraint(Constraint::new(vec![3], vec![1.0], 2.0, 5.0).unwrap());
    m.variables[0].bound = VarBound::integer(0.0, 10.0);
    m.variables[1].bound = VarBound::integer(0.0, 10.0);
    m.variables[2].bound = VarBound::continuous(0.0, 2.5);
    m.variables[3].bound = VarBound::continuous(f64::NEG_INFINITY, f64::INFINITY);

    let sol = MilpSolver::new().solve(&m).unwrap();
    assert_eq!(sol.status, tpt_opt_core::solver::SolverStatus::Optimal);
    assert!(
        (sol.objective_value - 18.0).abs() < 1e-6,
        "expected optimum 18, got {} at {:?}",
        sol.objective_value,
        sol.primal
    );
    // The reported objective must agree with the primal point.
    assert!(
        (m.objective.eval(&sol.primal) - sol.objective_value).abs() < 1e-6,
        "objective {} inconsistent with primal {:?}",
        sol.objective_value,
        sol.primal
    );
}

/// Same shape, minimisation with a negative constant: the constant must not
/// mask improving subtrees either.
#[test]
fn objective_constant_minimisation_finds_true_optimum() {
    // Minimize -3X - 4Y + 100 s.t. 2X + 3Y <= 9, X,Y integer in [0, 10].
    // True optimum: X=3, Y=1 -> -13 + 100 = 87.
    let mut m = Model::new(2);
    m.set_objective(Objective {
        sense: Sense::Minimize,
        indices: vec![0, 1],
        coeffs: vec![-3.0, -4.0],
        constant: 100.0,
    });
    m.add_constraint(Constraint::le(vec![0, 1], vec![2.0, 3.0], 9.0));
    m.variables[0].bound = VarBound::integer(0.0, 10.0);
    m.variables[1].bound = VarBound::integer(0.0, 10.0);

    let sol = MilpSolver::new().solve(&m).unwrap();
    assert_eq!(sol.status, tpt_opt_core::solver::SolverStatus::Optimal);
    assert!(
        (sol.objective_value - 87.0).abs() < 1e-6,
        "expected optimum 87, got {} at {:?}",
        sol.objective_value,
        sol.primal
    );
}
