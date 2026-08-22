# tpt-opt-robust

Optimisation under uncertainty for the `tpt-systems-optimisation` workspace:
scenario-based stochastic programming, sample average approximation,
adjustable robust optimisation, chance constraints, and distributionally
robust optimisation — all reduced to MILPs solved by
[`tpt-opt-milp`](../tpt-opt-milp) over [`tpt-opt-core`](../tpt-opt-core)
models.

Part of [TPT Solutions](https://github.com/tpt-solutions)' optimisation stack,
bridging `tpt-math` (pure mathematics) and the Tier 2 domain repositories.
See the workspace `spec.txt` for the overall design.

## Features

| Module | What it provides |
| --- | --- |
| `scenario` | Two-stage extensive forms (`TwoStageProblem`) and multi-stage scenario trees with prefix-merged non-anticipativity (`multi_stage_model`). |
| `saa` | Generic sample average approximation: replication-based statistical lower bounds, validation-based upper bounds, and optimality-gap confidence intervals (`SaaSolver`). |
| `value` | Value of the stochastic solution (VSS) and expected value of perfect information (EVPI) from the recourse / wait-and-see / expected-value solutions. |
| `chance` | Scenario (binary-indicator VaR) approximations and Gaussian deterministic equivalents (exact for diagonal covariance; conservative column-norm linearisation in general). |
| `robust` | Adjustable robust optimisation: Bertsimas–Sim budgeted (Γ-robustness) coefficient uncertainty via its exact LP reformulation, plus a conservative ellipsoidal-set reformulation. |
| `dro` | Distributionally robust optimisation: box/moment ambiguity sets (closed-form worst case + cutting-plane decision solver) and Wasserstein-ball worst-case evaluation for linear losses. |

Every MILP-backed component builds a plain `tpt_opt_core::Model`, so any
solver implementing the core `Solver` trait can be substituted for the
bundled branch-and-bound engine. All sampling is driven by the seedable
`tpt_math_prob::Xoshiro256` RNG for reproducibility.

## Quick start

Two-stage news-vendor problem with a known analytic optimum:

```rust
use tpt_opt_robust::scenario::{RowSense, Scenario, StageData, StageRow, TwoStageProblem};

// min 2x + 8·E[(d − x)⁺] over demands {2, 4} w.p. 1/2 each → RP* = 8 at x* = 4.
let mk = |d: f64| StageData {
    cost: vec![8.0],
    rows: vec![StageRow { w: vec![-1.0], t: vec![-1.0], h: -d, sense: RowSense::Le }],
};
let problem = TwoStageProblem {
    first_cost: vec![2.0],
    first_bounds: vec![(0.0, 10.0)],
    second_bounds: vec![(0.0, 10.0)],
    scenarios: vec![
        (Scenario { probability: 0.5, data: vec![] }, mk(2.0)),
        (Scenario { probability: 0.5, data: vec![] }, mk(4.0)),
    ],
};
let sol = problem.solve().unwrap();
assert!((sol.objective - 8.0).abs() < 1e-6);
assert!((sol.x[0] - 4.0).abs() < 1e-6);
```

Γ-robust (Bertsimas–Sim) protection on an uncertain row:

```rust
use tpt_opt_robust::{budgeted_reformulation, UncertainRow};

// min −x₁ − x₂ s.t. (â₁x₁ + â₂x₂ ≤ 1), â_j ∈ [0.8, 1.2], at most Γ = 1 deviates.
let model = budgeted_reformulation(
    vec![-1.0, -1.0],
    vec![(0.0, 10.0), (0.0, 10.0)],
    vec![UncertainRow { nominal: vec![1.0, 1.0], deviation: vec![0.2, 0.2], rhs: 1.0 }],
    vec![1.0],
);
```

## Status

Implemented and tested against hand-computed optima (news-vendor VSS/EVPI,
VaR budgets, Gaussian protection levels, Bertsimas–Sim interpolation,
Wasserstein margins). Not yet published to crates.io.

## License

Dual-licensed under MIT or Apache-2.0 — see `LICENSE-MIT` and
`LICENSE-APACHE` in the repository root.