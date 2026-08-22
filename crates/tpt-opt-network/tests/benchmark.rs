//! Hand-crafted benchmark instances for the network-flow solvers.
//!
//! Serves as the Phase 3 integration test: a min-cost-flow instance with a
//! known unique optimum solved by **both** algorithms (successive shortest
//! path and network simplex), cross-validated against each other and against
//! the analytic optimum, with flow-conservation and capacity checks.

use tpt_math_graph::{Edge, Graph};
use tpt_opt_core::solver::SolverStatus;
use tpt_opt_network::{min_cost_flow, network_simplex};

/// Build the benchmark instance:
///
/// ```text
/// supplies: node0 = +4, node3 = -4 (nodes 1, 2 tranship)
///
///   0 --cap 3, cost 2--> 1
///   0 --cap 2, cost 2--> 2
///   1 --cap 1, cost 1--> 2
///   1 --cap 2, cost 3--> 3
///   2 --cap 3, cost 1--> 3
/// ```
///
/// Optimal routing (unique): 2 units 0→1→{1 unit 1→2→3, 1 unit 1→3} and
/// 2 units 0→2→3, total cost **15**.
fn build_instance() -> Graph {
    let mut g = Graph::new(4);
    g.add_edge(Edge::new(0, 1, 3.0, 2.0));
    g.add_edge(Edge::new(0, 2, 2.0, 2.0));
    g.add_edge(Edge::new(1, 2, 1.0, 1.0));
    g.add_edge(Edge::new(1, 3, 2.0, 3.0));
    g.add_edge(Edge::new(2, 3, 3.0, 1.0));
    g
}

const SUPPLIES: [f64; 4] = [4.0, 0.0, 0.0, -4.0];
const OPTIMUM: f64 = 15.0;

fn check_conservation(graph: &Graph, flow: &[f64], eps: f64) {
    let mut net = vec![0.0f64; graph.num_nodes()];
    for (e, &f) in graph.edges().iter().zip(flow.iter()) {
        net[e.from] += f;
        net[e.to] -= f;
        assert!(f >= -eps && f <= e.capacity + eps, "capacity violated on edge {}", e.from);
    }
    for (n, &s) in SUPPLIES.iter().enumerate() {
        assert!((net[n] - s).abs() < eps, "conservation violated at node {n}");
    }
}

#[test]
fn shortest_path_matches_analytic_optimum() {
    let g = build_instance();
    let r = min_cost_flow(&g, &SUPPLIES);
    assert_eq!(r.status, SolverStatus::Optimal);
    assert!((r.total_cost - OPTIMUM).abs() < 1e-6, "cost {} != {OPTIMUM}", r.total_cost);
    check_conservation(&g, &r.flow, 1e-6);
}

#[test]
fn simplex_matches_analytic_optimum() {
    let g = build_instance();
    let r = network_simplex(&g, &SUPPLIES);
    assert_eq!(r.status, SolverStatus::Optimal);
    assert!((r.total_cost - OPTIMUM).abs() < 1e-6, "cost {} != {OPTIMUM}", r.total_cost);
    check_conservation(&g, &r.flow, 1e-6);
}

#[test]
fn algorithms_agree_on_optimum_and_cost() {
    let g = build_instance();
    let a = min_cost_flow(&g, &SUPPLIES);
    let b = network_simplex(&g, &SUPPLIES);
    assert_eq!(a.status, b.status);
    assert!((a.total_cost - b.total_cost).abs() < 1e-6);
}

/// Infeasible variant: demand exceeds what the cut out of node 0 can carry
/// (both arcs leaving node 0 shrunk to 1 unit each).
#[test]
fn infeasible_variant_reports_infeasible() {
    let mut g = Graph::new(4);
    g.add_edge(Edge::new(0, 1, 1.0, 2.0));
    g.add_edge(Edge::new(0, 2, 1.0, 2.0));
    g.add_edge(Edge::new(1, 3, 2.0, 3.0));
    g.add_edge(Edge::new(2, 3, 3.0, 1.0));
    let r = min_cost_flow(&g, &SUPPLIES);
    assert_ne!(r.status, SolverStatus::Optimal, "over-demanded instance must not be optimal");
}
