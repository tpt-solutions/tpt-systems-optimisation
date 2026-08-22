# tpt-opt-decompose

Large-scale decomposition methods for linear and mixed-integer programs in the
`tpt-systems-optimisation` workspace: Benders decomposition, Dantzig–Wolfe
decomposition with column generation, branch-and-price, Lagrangian relaxation,
and automatic decomposable-structure detection — all built on
[`tpt-opt-core`](../tpt-opt-core) models and solved with the bundled LP/MILP
engine ([`tpt-opt-milp`](../tpt-opt-milp)).

Part of [TPT Solutions](https://github.com/tpt-solutions)' optimisation stack,
bridging `tpt-math` (pure mathematics) and the Tier 2 domain repositories.
See the workspace `spec.txt` for the overall design.

## Features

| Module | What it provides |
| --- | --- |
| `benders` | Two-stage Benders over (mixed-)integer first stages: explicit dual-LP cut generation, Farkas feasibility cuts, Magnanti–Wong Pareto-optimal cuts (`with_pareto_cuts`), and trust-region / level-set stabilisation certified by a final unrestricted master solve. |
| `dantzig_wolfe` | Dantzig–Wolfe decomposition of block-angular programs: restricted-master column pool with dedup/capacity (`RmpPool`), big-M artificial seeding, per-block pricing LPs, and λ-based solution reconstruction. |
| `branch_price` | Branch-and-price: depth-first branch-and-bound on integer master variables with embedded column generation and a pluggable `Pricer` trait (continuous-LP default `LpPricer`; integer knapsack pricing for cutting-stock-style masters), plus dual-neutral cleanup pricing at integer nodes. |
| `lagrangian` | Subgradient ascent (Polyak / diminishing steps), a cutting-plane bundle/level method, and surrogate-relaxation search over multiplier space. |
| `structure` | Bipartite row–column connectivity analysis that detects independent blocks, linking rows/columns, and recommends a decomposition strategy (`detect_structure`). |

Every component builds plain `tpt_opt_core::Model`s internally, so any solver
implementing the core `Solver` trait can be substituted for the bundled
engine.

## Quick start

Two-stage capacity problem solved by Benders (integer first stage, one
recourse block):

```rust
use tpt_opt_decompose::{BendersProblem, BendersSolver, BlockRow, RecourseBlock, RowSense};

// Capacity x ∈ [0, 4] (integer) at cost 2/unit; demand 3 served by recourse
// y ≥ 3 − x at cost 5/unit ⇒ optimum x = 3, y = 0, objective 6.
let problem = BendersProblem {
    first_cost: vec![2.0],
    first_bounds: vec![(0.0, 4.0)],
    first_integer: vec![true],
    blocks: vec![RecourseBlock {
        cost: vec![5.0],
        rows: vec![BlockRow { y: vec![1.0], x: vec![1.0], sense: RowSense::Ge, rhs: 3.0 }],
        y_upper: vec![f64::INFINITY],
    }],
    weights: vec![1.0],
};
let result = BendersSolver::new(&problem).solve().unwrap();
assert!((result.objective - 6.0).abs() < 1e-6);
```

Detecting block-angular structure and getting a strategy recommendation:

```rust
use tpt_opt_core::model::{Constraint, Model};
use tpt_opt_decompose::detect_structure;

let mut model = Model::new(4);
model.add_constraint(Constraint::ge(vec![0, 1], vec![1.0, 1.0], 1.0)); // block A
model.add_constraint(Constraint::ge(vec![2, 3], vec![1.0, 1.0], 1.0)); // block B
model.add_constraint(Constraint::le(vec![0, 2], vec![1.0, 1.0], 3.0)); // linking row
let report = detect_structure(&model);
assert_eq!(report.num_components, 2);
assert_eq!(report.linking_rows, vec![2]);
```

## Status

Implemented and tested against hand-computed optima (Benders capacity /
multi-scenario / feasibility-cut instances, cutting stock via branch-and-price
cross-validated against a monolithic pattern MILP, analytic Lagrangian duals,
structure-detection reports). Not yet published to crates.io.

## License

Dual-licensed under MIT or Apache-2.0 — see `LICENSE-MIT` and
`LICENSE-APACHE` in the repository root.