//! Solve the same min-cost-flow instance two ways: with the specialised
//! network solver and by lowering the instance to a canonical MILP via
//! `convert::network_flow_to_milp` — demonstrating that any MILP backend can
//! serve as a flow solver.
//!
//! Instance: 4-node diamond, 16 units from node 0 to node 3.

use tpt_opt_systems::core::Solver;
use tpt_opt_systems::{graph, milp, network_flow_to_milp, NetworkFlowBuilder};

fn main() {
    // --- specialised solver -------------------------------------------------
    let mut flow = NetworkFlowBuilder::new(4);
    let e01 = flow.add_edge(0, 1, 10.0, 1.0);
    let e02 = flow.add_edge(0, 2, 8.0, 2.0);
    let e13 = flow.add_edge(1, 3, 10.0, 1.0);
    let e23 = flow.add_edge(2, 3, 8.0, 1.0);
    flow.supply(0, 16.0);
    flow.demand(3, 16.0);
    let specialised = flow.solve().expect("instance is feasible");
    println!("specialised min-cost-flow: cost = {}", specialised.total_cost);

    // --- lowered to a MILP --------------------------------------------------
    let mut g = graph::Graph::new(4);
    g.add_edge(graph::Edge::new(0, 1, 10.0, 1.0));
    g.add_edge(graph::Edge::new(0, 2, 8.0, 2.0));
    g.add_edge(graph::Edge::new(1, 3, 10.0, 1.0));
    g.add_edge(graph::Edge::new(2, 3, 8.0, 1.0));
    let balances = [16.0, 0.0, 0.0, -16.0];
    let model = network_flow_to_milp(&g, &balances);
    let mut solver = milp::MilpSolver::new();
    let sol = solver.solve(&model).expect("MILP lowering should solve");
    println!("MILP-lowered min-cost-flow: cost = {}", sol.objective_value);

    assert!((specialised.total_cost - sol.objective_value).abs() < 1e-6);
    let _ = (e01, e02, e13, e23); // edge indices kept for reference
}