# tpt-opt-systems

Umbrella crate for the `tpt-systems-optimisation` workspace (spec.txt §3).
Depend on this single crate instead of tracking the individual solver crates;
enable exactly the solver families you need through flat Cargo features.

## Feature matrix

| Feature       | Crate               | Contents |
|---------------|---------------------|----------|
| *(none)*      | `tpt-opt-core`      | Canonical `Model`, `Solver` trait, bounds, tolerances, errors (always available) |
| `milp`        | `tpt-opt-milp`      | Branch-and-bound/cut MILP: clique/cover/MIR/Gomory/lift-and-project cuts, primal heuristics, SOS/indicator/piecewise-linear modelling |
| `minlp`       | `tpt-opt-minlp`     | Outer approximation, generalized Benders decomposition, SQP branch-and-bound, McCormick/αBB relaxations, logical & complementarity constraints |
| `network`     | `tpt-opt-network`   | Min-cost flow (SSP + network simplex), Hungarian assignment, DC/AC/security-constrained OPF, graph preprocessing, dynamic networks |
| `cp`          | `tpt-opt-cp`        | Constraint programming: propagation, AllDifferent/Cumulative/Circuit/Regular/Element globals, conflict-directed backjumping |
| `heuristic`   | `tpt-opt-heuristic` | Simulated annealing, genetic algorithms, tabu search, particle swarm optimisation |
| `multi`       | `tpt-opt-multi`     | NSGA-II/III, Pareto fronts, hypervolume, knee points/trade-offs, linear scalarisation |
| `robust`      | `tpt-opt-robust`    | Two-/multi-stage stochastic programming, SAA, VSS/EVPI, chance constraints, Bertsimas–Sim budgeted robustness, distributionally robust optimisation |
| `decompose`   | `tpt-opt-decompose` | Benders decomposition, Dantzig–Wolfe + column generation, branch-and-price, Lagrangian relaxation, structure detection |
| `all-solvers` | *(meta-feature)*    | Enables every solver family at once |

With **no features** enabled the crate exposes only the always-on core
surface (`tpt_opt_systems::core`, plus flat re-exports of the core types), so
an empty-feature build compiles without any solver backend.

## What the umbrella adds

* **Unified errors** — `OptimizationError` tags every failure with the
  algorithm that produced it and preserves the underlying
  `tpt_opt_core::OptError` when one exists.
* **Convenience builders** — `MilpBuilder` (`milp`) for fluent model assembly
  + solve in one call; `NetworkFlowBuilder` (`network`) for graph assembly +
  min-cost flow in one call.
* **Format conversion** — `convert::network_flow_to_milp` (`network` + `milp`)
  lowers a min-cost-flow instance to a canonical MILP so any MILP backend can
  solve it.

## Example

```toml
[dependencies]
tpt-opt-systems = { version = "0.1", features = ["milp", "network"] }
```

```rust
use tpt_opt_systems::{MilpBuilder, NetworkFlowBuilder};

// A tiny MILP through the builder…
let mut b = MilpBuilder::new(0);
let x = b.add_integer(0.0, 10.0);
let sol = b.ge(&[x], &[1.0], 3.0).minimize(&[x], &[2.0]).solve().unwrap();
assert!((sol.objective_value - 6.0).abs() < 1e-6);

// …and a min-cost flow through its builder.
let mut flow = NetworkFlowBuilder::new(2);
flow.add_edge(0, 1, 5.0, 2.0);
flow.supply(0, 3.0);
flow.demand(1, 3.0);
let routed = flow.solve().unwrap();
assert!(routed.status.has_solution());
```

## License

Licensed under either of MIT or Apache-2.0 at your option.