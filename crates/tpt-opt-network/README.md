# tpt-opt-network

Network-flow and graph-based optimisation for the `tpt-systems-optimisation`
workspace (spec.txt §3 "tpt-opt-network"). Pure Rust, no `unsafe`, `std`.

## Algorithms

| Algorithm | Function / type | Notes |
|-----------|----------------|-------|
| **Min-cost flow** | `min_cost_flow` | Successive shortest path with Johnson potentials (primary). |
| **Network simplex** | `network_simplex` | Negative-cycle cancelling. |
| **Assignment** | `hungarian` / `hungarian_maximize` | Hungarian (Kuhn–Munkres) for the square assignment problem. |
| **DC-OPF** | `dc_opf` | Linearised OPF solved as an LP via the in-crate two-phase simplex. |
| **AC-OPF** | `ac_opf` | Polar-coordinate OPF solved as an NLP (augmented Lagrangian + BFGS via `tpt-math-optimize-general`). |
| **SC-OPF** | `sc_opf` | Base-case DC-OPF plus N-1 contingency re-solves. |
| **Dynamic networks** | `DynamicNetwork` | Period-by-period min-cost flow with a warm-start hint. |
| **Graph preprocessing** | `has_cycle`, `bridges`, `biconnected_components` | Cycle detection, bridge identification, biconnected decomposition. |

## Reproducibility & numerics (spec §4)

All comparisons thread `tpt_opt_core::Tolerances` through the solvers. Numerical
failures surface as `tpt_opt_core::SolverStatus::NumericalIssue` rather than
silently wrong results. The in-crate LP/OPF solvers are deterministic for a
fixed input.

## Tying into `tpt-opt-core`

* The LP solver (`LpSolver`) implements `tpt_opt_core::Solver<Model>`.
* Results use `tpt_opt_core::Solution` / `tpt_opt_core::SolverStatus`.
* OPF inputs are described by `Network`, `Bus`, `Generator`, `Line`.

## Example

```rust
use tpt_opt_network::{dc_opf, Network, Bus, Generator, Line};

let net = Network {
    buses: vec![
        Bus { id: 0, is_slack: true, demand_p: 0.0, demand_q: 0.0, v_min: 0.9, v_max: 1.1 },
        Bus { id: 1, is_slack: false, demand_p: 100.0, demand_q: 0.0, v_min: 0.9, v_max: 1.1 },
    ],
    generators: vec![
        Generator { bus: 0, p_min: 0.0, p_max: 200.0, c0: 0.0, c1: 1.0, c2: 0.0 },
    ],
    lines: vec![
        Line { from: 0, to: 1, reactance: 0.1, capacity: 150.0 },
    ],
};

let res = dc_opf(&net);
assert_eq!(res.status, tpt_opt_core::SolverStatus::Optimal);
assert!((res.total_cost - 100.0).abs() < 1e-6);
```

## License

Licensed under either of MIT or Apache-2.0 at your option.
