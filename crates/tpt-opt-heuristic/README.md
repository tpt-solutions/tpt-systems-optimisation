# tpt-opt-heuristic

Metaheuristic optimization algorithms for the `tpt-systems-optimisation`
workspace (spec.txt §3 "tpt-opt-heuristic"). Pure Rust, no `unsafe`, `std`.

## Algorithms

| Algorithm | Type | Notes |
|-----------|------|-------|
| **Simulated Annealing** | `SimulatedAnnealing` | Geometric / adaptive / reheating cooling; custom `Neighborhood`. |
| **Genetic Algorithms** | `GeneticAlgorithm<Vec<f64>>` / `GeneticAlgorithm<Vec<usize>>` | Continuous & permutation genomes; single/two-point/uniform/order crossover; bit-flip/flip/swap/inversion/scramble mutation; tournament/roulette/rank selection. |
| **Tabu Search** | `TabuSearch` | Adaptive tenure, aspiration, diversification. |
| **Particle Swarm** | `ParticleSwarmOptimization` | Inertia adaptation; global / ring / Von Neumann topologies. |

## Reproducibility (spec §4)

Every heuristic accepts a deterministic seed via `with_seed(seed)`. Two runs
with the same seed produce byte-identical results. All randomness flows through
the seedable `tpt-math-prob` RNG.

## Tying into `tpt-opt-core`

* Results are reported through `HeuristicResult`, convertible to a
  `tpt_opt_core::Solution` via `HeuristicResult::solution`.
* Invalid configurations return `tpt_opt_core::OptError`.
* The terminal condition is summarized with `tpt_opt_core::SolverStatus`.

### Solver agnosticism (spec §4)

Every heuristic implements `tpt_opt_core::solver::Solver<Model>` — the same
`solve` / `set_parameter` / `warm_start` / `status` / `solution` contract as
the MILP, LP, and CP backends. When driven through the trait, the solver
evaluates the canonical model's objective (the closure passed to `new` is
temporarily replaced by a view of the model), so one generic driver can run
any backend interchangeably:

| Solver | Notes |
|--------|-------|
| `SimulatedAnnealing` | Full trait support incl. warm start and seed parameter. |
| `TabuSearch` | Full trait support incl. warm start and seed parameter. |
| `ParticleSwarmOptimization` | Full trait support incl. warm start and seed parameter. |
| `GeneticAlgorithm<Vec<f64>>` | Continuous genome; population-based, so warm-start hints are accepted and ignored. |

## Example

```rust
use tpt_opt_heuristic::{SimulatedAnnealing, ObjectiveFn, CoolingSchedule};

let obj = ObjectiveFn::minimize(2, |x| x[0]*x[0] + x[1]*x[1], [(-5.0, 5.0), (-5.0, 5.0)]);
let mut sa = SimulatedAnnealing::new(obj).with_seed(7);
let res = sa.solve().unwrap();
assert!(res.best_value < 1.0);
```

## License

Licensed under either of MIT or Apache-2.0 at your option.
