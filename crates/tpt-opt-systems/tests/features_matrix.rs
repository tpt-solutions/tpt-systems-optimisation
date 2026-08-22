//! Feature-matrix integration tests: the gated re-export surface is
//! exercised under its own feature flags (milp + network).

#![cfg(all(feature = "milp", feature = "network"))]

use tpt_opt_core::{solver::Solver, SolverStatus};
use tpt_opt_systems::{convert::network_flow_to_milp, MilpBuilder, MilpSolver, NetworkFlowBuilder};

#[test]
fn milp_builder_solves_knapsack() {
    // max 10a + 13b + 7c + 4d s.t. 5a + 7b + 4c + 3d <= 14 → 24.
    let mut b = MilpBuilder::new(0);
    let a = b.add_binary();
    let c = b.add_binary();
    let d = b.add_binary();
    let e = b.add_binary();
    let sol = b
        .le(&[a, c, d, e], &[5.0, 7.0, 4.0, 3.0], 14.0)
        .maximize(&[a, c, d, e], &[10.0, 13.0, 7.0, 4.0])
        .with_seed(42)
        .solve()
        .unwrap();
    assert_eq!(sol.status, SolverStatus::Optimal);
    assert!((sol.objective_value - 24.0).abs() < 1e-6);
}

#[test]
fn milp_builder_build_model_is_reusable() {
    let mut b = MilpBuilder::new(0);
    let x = b.add_integer(0.0, 5.0);
    let model = b.ge(&[x], &[1.0], 2.0).minimize(&[x], &[1.0]).build_model();
    assert_eq!(model.num_vars, 1);
    let sol = MilpSolver::new().solve(&model).unwrap();
    assert!((sol.objective_value - 2.0).abs() < 1e-6);
}

#[test]
fn network_builder_routes_minimum_cost() {
    let mut flow = NetworkFlowBuilder::new(4);
    flow.add_edge(0, 1, 3.0, 2.0);
    flow.add_edge(0, 2, 2.0, 2.0);
    flow.add_edge(1, 3, 2.0, 3.0);
    flow.add_edge(2, 3, 3.0, 1.0);
    flow.supply(0, 4.0);
    flow.demand(3, 4.0);
    let result = flow.solve().unwrap();
    assert!(result.status.has_solution());
    assert!((result.total_cost - 16.0).abs() < 1e-6);
}

#[test]
fn flow_to_milp_matches_specialised_solver() {
    use tpt_opt_systems::{graph::Edge, network};

    let mut g = tpt_opt_systems::graph::Graph::new(3);
    g.add_edge(Edge::new(0, 1, 4.0, 1.0));
    g.add_edge(Edge::new(1, 2, 4.0, 1.0));
    g.add_edge(Edge::new(0, 2, 1.0, 5.0));
    let balances = [3.0, 0.0, -3.0];

    let specialised = network::min_cost_flow(&g, &balances);
    let model = network_flow_to_milp(&g, &balances);
    let via_milp = MilpSolver::new().solve(&model).unwrap();

    assert!(specialised.status.has_solution());
    assert!(via_milp.status.has_solution());
    assert!(
        (specialised.total_cost - via_milp.objective_value).abs() < 1e-6,
        "specialised {} vs milp {}",
        specialised.total_cost,
        via_milp.objective_value
    );
}

#[test]
fn optimization_error_wraps_solver_failure_with_context() {
    // An invalid model (constraint referencing a nonexistent variable) is
    // rejected by validation and surfaces as a tagged error. (An infeasible
    // but well-formed model returns Ok with an Infeasible status instead,
    // per the Solver contract.)
    let err = MilpBuilder::new(0).le(&[0], &[1.0], 1.0).minimize(&[], &[]).solve().unwrap_err();
    assert_eq!(err.algorithm(), "branch-and-bound");
    assert!(err.to_string().contains("branch-and-bound"));
    assert!(err.into_core().is_some());
}

#[test]
fn infeasible_model_reports_status_not_error() {
    let mut b = MilpBuilder::new(0);
    let x = b.add_integer(0.0, 1.0);
    let sol = b.le(&[x], &[1.0], 0.5).ge(&[x], &[1.0], 1.0).minimize(&[x], &[1.0]).solve().unwrap();
    assert_eq!(sol.status, SolverStatus::Infeasible);
}
