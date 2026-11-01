# Getting started

Add the umbrella crate with only the solver families you need:

```toml
[dependencies]
tpt-opt-systems = { version = "0.1", features = ["milp", "network"] }
```

Enable `all-solvers` to get every family at once, and `serde` for JSON model
import and export.

## Your first model

The expression DSL builds a model from named variables. Operators and
relations lower to the canonical `Model` that every solver consumes.

```rust
use tpt_opt_systems::prelude::*;

let mut b = ModelBuilder::new();
let x = b.continuous("x", 0.0, 10.0);
let y = b.integer("y", 0.0, 5.0);
b.constrain((2.0 * x + 3.0 * y).le(12.0));
b.maximize(x + 4.0 * y);
let model = b.build()?;

let sol = solve_auto(&model)?; // needs the `milp` feature
println!("objective = {}", sol.objective_value);
```

`classify(&model)` reports whether the model is a linear program or a
mixed-integer program. `solve_auto` sends both to the MILP backend, whose root
node is the LP relaxation.

## Feature flags

| Feature | Enables |
|---------|---------|
| `milp` | branch-and-cut MILP, `MilpBuilder`, MPS and LP readers |
| `network` | min-cost flow, assignment, OPF, `NetworkFlowBuilder` |
| `minlp` | outer approximation, Benders, spatial branch-and-bound |
| `cp` | constraint programming with global constraints |
| `heuristic` | annealing, genetic algorithm, tabu, particle swarm |
| `multi` | NSGA-II/III, hypervolume, scalarisation |
| `robust` | two-stage and multi-stage stochastic programs, SAA, chance constraints |
| `decompose` | Benders, Dantzig–Wolfe, branch-and-price, Lagrangian relaxation |
| `conic` | second-order-cone and semidefinite programs |
| `nls` | Gauss–Newton, Levenberg–Marquardt, and dogleg least squares |
| `factorgraph` | pose graphs and bundle adjustment (implies `nls`) |
| `bayes` | Gaussian-process Bayesian optimisation |
| `serde` | JSON import and export of models |

## Running the examples

Each example names the feature it needs:

```text
cargo run -p tpt-opt-systems --example network_min_cost_flow --features network
cargo run -p tpt-opt-systems --example nls_exponential_fit --features nls
cargo run -p tpt-opt-systems --example conic_socp --features conic
cargo run -p tpt-opt-systems --example mps_loading --features milp
cargo run -p tpt-opt-systems --example bayes_minimise --features bayes
cargo run -p tpt-opt-systems --example factorgraph_square_loop --features factorgraph
```
