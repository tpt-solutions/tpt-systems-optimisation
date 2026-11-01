//! Route 3 units from node 0 to node 3 at minimum cost.
//!
//! Two routes exist: a cheap one with capacity 2 (0→1→3, cost 2 per unit) and an
//! expensive one with capacity 5 (0→2→3, cost 6 per unit). The optimum sends
//! 2 units cheaply and 1 unit expensively: 2·2 + 1·6 = 10.
//!
//! Run with: `cargo run -p tpt-opt-systems --example network_min_cost_flow --features network`

use tpt_opt_systems::NetworkFlowBuilder;

fn main() {
    let mut flow = NetworkFlowBuilder::new(4);
    flow.add_edge(0, 1, 2.0, 1.0);
    flow.add_edge(1, 3, 2.0, 1.0);
    flow.add_edge(0, 2, 5.0, 3.0);
    flow.add_edge(2, 3, 5.0, 3.0);
    flow.supply(0, 3.0);
    flow.demand(3, 3.0);

    let routed = flow.solve().expect("the network is feasible");
    println!("status: {:?}", routed.status);
    println!("total cost: {}", routed.total_cost);
    println!("edge flows: {:?}", routed.flow);

    assert!((routed.total_cost - 10.0).abs() < 1e-6);
}
