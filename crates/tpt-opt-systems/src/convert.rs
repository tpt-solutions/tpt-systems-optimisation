//! Format conversion utilities between solver families.
//!
//! The flagship conversion is [`network_flow_to_milp`]: a min-cost-flow
//! instance (graph + node balances) becomes a canonical linear
//! [`Model`] — one continuous variable per edge, capacity bounds,
//! one flow-conservation equality row per node, and a linear cost
//! objective. This lets any MILP backend (the bundled branch-and-bound
//! engine or an external binding such as HiGHS) solve network instances
//! that lack a specialised algorithm.

use tpt_opt_core::graph::Graph;
use tpt_opt_core::{Constraint, Model, Objective};

/// Convert a min-cost-flow instance into a canonical MILP model.
///
/// For each edge `e = (u → v)` with capacity `cap_e` and cost `c_e` the
/// resulting model has a continuous variable `f_e ∈ [0, cap_e]`. Each node
/// `w` contributes the conservation row
/// `sum_{e: u=w} f_e − sum_{e: v=w} f_e == balance_w`, and the objective is
/// `min sum c_e · f_e`.
///
/// # Example
///
/// ```
/// use tpt_opt_core::graph::{Edge, Graph};
/// use tpt_opt_core::solver::Solver;
/// use tpt_opt_systems::{convert::network_flow_to_milp, MilpSolver};
///
/// let mut g = Graph::new(2);
/// g.add_edge(Edge::new(0, 1, 5.0, 2.0));
/// let model = network_flow_to_milp(&g, &[3.0, -3.0]);
/// let sol = MilpSolver::new().solve(&model).unwrap();
/// assert!((sol.objective_value - 6.0).abs() < 1e-6);
/// assert!((sol.primal[0] - 3.0).abs() < 1e-6);
/// ```
pub fn network_flow_to_milp(graph: &Graph, balances: &[f64]) -> Model {
    let edges = graph.edges();
    let n = graph.num_nodes();
    debug_assert_eq!(balances.len(), n, "one balance per node");

    let mut model = Model::with_name(edges.len(), "min-cost-flow");
    for _ in 0..edges.len() {
        model.add_variable(tpt_opt_core::VarBound::continuous(0.0, f64::INFINITY));
    }
    // Capacity bounds.
    for (e, edge) in edges.iter().enumerate() {
        model.variables[e].bound = tpt_opt_core::VarBound::continuous(0.0, edge.capacity);
    }

    // Flow conservation: outflow − inflow == balance at every node.
    for (w, &balance) in balances.iter().enumerate().take(n) {
        let mut indices = Vec::new();
        let mut coeffs = Vec::new();
        for (e, edge) in edges.iter().enumerate() {
            if edge.from == w {
                indices.push(e);
                coeffs.push(1.0);
            } else if edge.to == w {
                indices.push(e);
                coeffs.push(-1.0);
            }
        }
        model.add_constraint(Constraint::equality(indices, coeffs, balance));
    }

    // Linear cost objective.
    let indices: Vec<usize> = (0..edges.len()).collect();
    let coeffs: Vec<f64> = edges.iter().map(|e| e.cost).collect();
    model.set_objective(Objective::minimize(indices, coeffs));
    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_opt_core::graph::Edge;
    use tpt_opt_core::solver::Solver;

    #[test]
    fn conversion_matches_specialised_solver() {
        // 4-node diamond from the network crate's doc example.
        let mut g = Graph::new(4);
        g.add_edge(Edge::new(0, 1, 3.0, 2.0));
        g.add_edge(Edge::new(0, 2, 2.0, 2.0));
        g.add_edge(Edge::new(1, 3, 2.0, 3.0));
        g.add_edge(Edge::new(2, 3, 3.0, 1.0));
        let balances = [4.0, 0.0, 0.0, -4.0];

        let specialised = tpt_opt_network::min_cost_flow(&g, &balances);
        let model = network_flow_to_milp(&g, &balances);
        let milp = tpt_opt_milp::MilpSolver::new().solve(&model).unwrap();

        assert!(specialised.status.has_solution());
        assert!(milp.status.has_solution());
        assert!(
            (specialised.total_cost - milp.objective_value).abs() < 1e-6,
            "specialised {} vs milp {}",
            specialised.total_cost,
            milp.objective_value
        );
        assert!((milp.objective_value - 16.0).abs() < 1e-6);
    }

    #[test]
    fn infeasible_when_capacity_insufficient() {
        // Demand exceeds total outgoing capacity of the source.
        let mut g = Graph::new(2);
        g.add_edge(Edge::new(0, 1, 1.0, 1.0));
        let model = network_flow_to_milp(&g, &[5.0, -5.0]);
        let sol = tpt_opt_milp::MilpSolver::new().solve(&model).unwrap();
        assert_eq!(sol.status, tpt_opt_core::SolverStatus::Infeasible);
    }
}
