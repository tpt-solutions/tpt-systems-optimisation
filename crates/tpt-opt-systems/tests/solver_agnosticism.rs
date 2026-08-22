//! Solver Agnosticism verification (spec §4 / todo.md cross-cutting checklist).
//!
//! Every solver family exposed by the umbrella must be usable through the
//! *same* `Solver<Model>` contract — one generic driver function, instantiated
//! for each solver type. This test fails to compile if any solver drifts from
//! the `solve` / `set_parameter` / `warm_start` / `status` / `solution`
//! signature.

#![allow(unused_imports)]
use tpt_opt_systems::core::model::{Constraint, Model, Objective};
use tpt_opt_systems::core::solver::{Solution, SolveParameters, Solver, SolverStatus, WarmStart};
use tpt_opt_systems::core::{OptError, VarBound};

// With no solver features enabled there is nothing to drive; the helpers
// below are compiled only when at least one backend is present.
#[cfg(any(feature = "milp", feature = "network", feature = "heuristic"))]
/// A tiny box-bounded LP every exact solver can chew on:
/// minimise -x - y s.t. x + y <= 1, x, y in [0, 1]  (optimum -1).
fn sample_model() -> Model {
    let mut m = Model::new(2);
    m.set_objective(Objective::minimize(vec![0, 1], vec![-1.0, -1.0]));
    m.add_constraint(Constraint::le(vec![0, 1], vec![1.0, 1.0], 1.0));
    m.variables[0].bound = VarBound::continuous(0.0, 1.0);
    m.variables[1].bound = VarBound::continuous(0.0, 1.0);
    m
}

/// Generic driver exercising the full agnosticism surface through the trait.
#[cfg(any(feature = "milp", feature = "network", feature = "heuristic"))]
fn drive<S: Solver<Model>>(solver: &mut S) -> Result<Solution, OptError> {
    let model = sample_model();
    solver.set_parameter(&SolveParameters::defaults().with_seed(7))?;
    solver.warm_start(WarmStart::empty())?;
    let sol = solver.solve(&model)?;
    // status()/solution() accessors must reflect the last solve.
    let _ = solver.status();
    let _ = solver.solution();
    Ok(sol)
}

#[test]
#[cfg(feature = "milp")]
fn milp_solver_implements_the_contract() {
    let mut s = tpt_opt_milp::MilpSolver::new();
    let sol = drive(&mut s).expect("solve must succeed");
    assert_eq!(sol.status, SolverStatus::Optimal);
    assert!((sol.objective_value - (-1.0)).abs() < 1e-6);
}

#[test]
#[cfg(feature = "network")]
fn network_lp_solver_implements_the_contract() {
    let mut s = tpt_opt_network::LpSolver::new();
    let sol = drive(&mut s).expect("solve must succeed");
    assert_eq!(sol.status, SolverStatus::Optimal);
    assert!((sol.objective_value - (-1.0)).abs() < 1e-6);
}

#[test]
#[cfg(feature = "heuristic")]
fn heuristic_solvers_implement_the_contract() {
    use tpt_opt_heuristic::{ParticleSwarmOptimization, SimulatedAnnealing, TabuSearch};
    // Heuristics are anytime: without a target they terminate with TimeLimit,
    // so require a finite objective and a clean trait round-trip only.
    // The Solver<Model> impls evaluate the canonical model's objective, not
    // the placeholder closure passed to `new`.
    let placeholder =
        tpt_opt_heuristic::ObjectiveFn::minimize(2, |x| x[0] + x[1], [(0.0, 1.0), (0.0, 1.0)]);
    for obj in [
        drive(&mut SimulatedAnnealing::new(placeholder.clone()).with_iterations(200))
            .expect("SA solve")
            .objective_value,
        drive(&mut TabuSearch::new(placeholder.clone()).with_iterations(200))
            .expect("tabu solve")
            .objective_value,
        drive(&mut ParticleSwarmOptimization::new(placeholder).with_iterations(60))
            .expect("PSO solve")
            .objective_value,
    ] {
        assert!(obj.is_finite(), "heuristic returned non-finite objective {obj}");
    }
}
