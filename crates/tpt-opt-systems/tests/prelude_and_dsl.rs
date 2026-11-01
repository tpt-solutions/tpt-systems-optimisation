//! The prelude, the expression DSL, and automatic classification working
//! together through the umbrella crate's public surface only.

use tpt_opt_systems::prelude::*;

#[test]
fn dsl_builds_a_classifiable_linear_program() {
    let mut b = ModelBuilder::new();
    let x = b.continuous("x", 0.0, 10.0);
    let y = b.continuous("y", 0.0, 10.0);
    b.constrain((x + y).le(8.0));
    b.constrain((x - y).ge(-2.0));
    b.maximize(3.0 * x + 2.0 * y);
    let model = b.build().unwrap();

    assert_eq!(classify(&model), ProblemClass::LinearProgram);
    assert_eq!(model.num_constraints(), 2);
}

#[test]
fn dsl_builds_a_classifiable_mixed_integer_program() {
    let mut b = ModelBuilder::new();
    let n = b.integer("n", 0.0, 9.0);
    b.minimize(n);
    let model = b.build().unwrap();
    assert_eq!(classify(&model), ProblemClass::MixedInteger);
}

#[cfg(feature = "milp")]
#[test]
fn dsl_model_solves_end_to_end_through_solve_auto() {
    // Max 3x + 2y with x + y <= 8 and x - y >= -2, x, y in [0, 10].
    // Feasible vertices: (0,0)->0, (8,0)->24 (optimum), (3,5)->19, (0,2)->4.
    let mut b = ModelBuilder::new();
    let x = b.continuous("x", 0.0, 10.0);
    let y = b.continuous("y", 0.0, 10.0);
    b.constrain((x + y).le(8.0));
    b.constrain((x - y).ge(-2.0));
    b.maximize(3.0 * x + 2.0 * y);
    let model = b.build().unwrap();

    let sol = solve_auto(&model).unwrap();
    assert!((sol.objective_value - 24.0).abs() < 1e-6, "obj {}", sol.objective_value);
}

#[cfg(feature = "milp")]
#[test]
fn dsl_model_solves_through_the_milp_solver_front_end() {
    let mut b = ModelBuilder::new();
    let x = b.integer("x", 0.0, 10.0);
    b.constrain(x.ge(3.5));
    b.minimize(2.0 * x + 1.0);
    let model = b.build().unwrap();

    let sol = MilpSolver::new().solve(&model).unwrap();
    assert!((sol.objective_value - 9.0).abs() < 1e-6, "obj {}", sol.objective_value);
}
