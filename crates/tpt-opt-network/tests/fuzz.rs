//! Fuzz testing for the network solvers: seeded random min-cost-flow
//! instances with verified invariants.
//!
//! For each seed we generate a random graph with a random supply/demand
//! pair and check, whenever the solver reports a solution:
//!
//! 1. *Feasibility*: flows lie in `[0, capacity]` and node balances equal
//!    the requested supplies.
//! 2. *Cost consistency*: `total_cost` equals `sum flow_e * cost_e`.
//! 3. *Cross-algorithm agreement*: successive-shortest-path and
//!    network-simplex agree on status and optimal cost.

use tpt_math_graph::{Edge, Graph};
use tpt_opt_core::SolverStatus;
use tpt_opt_network::{min_cost_flow, network_simplex};

/// Tiny deterministic xorshift RNG so failures are reproducible by seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn build_instance(rng: &mut Rng) -> (Graph, Vec<f64>) {
    let n = 3 + rng.below(5) as usize; // 3..=7 nodes
    let mut g = Graph::new(n);
    let m = n + rng.below((n + 2) as u64) as usize;
    for _ in 0..m {
        let from = rng.below(n as u64) as usize;
        let to = rng.below(n as u64) as usize;
        if from == to {
            continue;
        }
        let cap = 1.0 + rng.below(10) as f64;
        let cost = 0.5 + rng.below(10) as f64 * 0.5;
        g.add_edge(Edge::new(from, to, cap, cost));
    }

    // Random source/sink with a random demand; roughly half of these are
    // routable, so both feasible and infeasible paths get exercised.
    let s = rng.below(n as u64) as usize;
    let mut t = rng.below(n as u64) as usize;
    if t == s {
        t = (t + 1) % n;
    }
    let demand = 1.0 + rng.below(8) as f64;
    let mut supplies = vec![0.0; n];
    supplies[s] = demand;
    supplies[t] = -demand;
    (g, supplies)
}

#[test]
fn fuzz_random_flows_feasible_and_consistent() {
    for seed in 1u64..=200 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17) | 1);
        let (g, supplies) = build_instance(&mut rng);
        let caps: Vec<f64> = g.edges().iter().map(|e| e.capacity).collect();
        let costs: Vec<f64> = g.edges().iter().map(|e| e.cost).collect();

        let ssp = min_cost_flow(&g, &supplies);
        let ns = network_simplex(&g, &supplies);

        // Cross-algorithm agreement on the terminal status class.
        assert_eq!(
            ssp.status.has_solution(),
            ns.status.has_solution(),
            "seed {seed}: solvers disagree on feasibility"
        );

        if !ssp.status.has_solution() {
            assert_eq!(ssp.status, SolverStatus::Infeasible);
            continue;
        }

        // Feasibility: capacities respected.
        for (f, c) in ssp.flow.iter().zip(&caps) {
            assert!(*f >= -1e-6 && *f <= c + 1e-6, "seed {seed}: flow {f} outside [0,{c}]");
        }

        // Conservation: net outflow equals supply at every node.
        for (b, s) in ssp.node_balance.iter().zip(&supplies) {
            assert!((b - s).abs() < 1e-6, "seed {seed}: balance {b} != supply {s}");
        }

        // Cost consistency.
        let recomputed: f64 = ssp.flow.iter().zip(&costs).map(|(f, c)| f * c).sum();
        assert!(
            (recomputed - ssp.total_cost).abs() < 1e-6,
            "seed {seed}: reported cost {} != recomputed {recomputed}",
            ssp.total_cost
        );

        // Both algorithms must land on the same optimal cost.
        assert!(
            (ssp.total_cost - ns.total_cost).abs() < 1e-6,
            "seed {seed}: SSP cost {} != simplex cost {}",
            ssp.total_cost,
            ns.total_cost
        );
    }
}
