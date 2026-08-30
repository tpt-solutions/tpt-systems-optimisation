# tpt-opt-conic

Conic (second-order-cone and semidefinite) optimisation for the
`tpt-systems-optimisation` workspace. Provides SOCP/SDP solvers built on top
of the workspace's verified LP engine, so robust- and convex-modelling code
can drop in a conic solver without vendoring a bespoke interior-point method.

Pure Rust, no `unsafe`, `std` by default (no_std-compatible core via
`tpt-opt-core`).

## What is implemented

* **Kelley cutting planes (outer approximation)** over the canonical LP engine
  (`solve_conic` / `solve_socp`). Every cone constraint is replaced by a
  sequence of valid supporting hyperplanes; because each cut is a supporting
  hyperplane of the cone, the relaxation stays a relaxation of the true conic
  problem and converges to the conic optimum.
* **Second-order-cone (SOCP) rows** — `‖q(x)‖₂ ≤ r(x)` in standard affine
  form (`SocRow`), with a zero-norm guard that separates via the necessary
  condition `r(x) ≥ 0` when the cone direction degenerates.
* **Semidefinite (SDP) blocks** — `X(x) = X₀ + Σₖ xₖ·Xₖ ⪰ 0` (`SdpBlock`),
  separated by the eigenvector cut `⟨v vᵀ, X(x)⟩ ≥ 0` at the most-negative
  eigenvalue (symmetric eigendecomposition by the cyclic Jacobi method).
* **Necessary-condition seeding** — every SOC row seeds the relaxation with
  `r(x) ≥ 0` up front so the LP relaxation stays bounded and well-posed
  before the first supporting-hyperplane cut is generated.

## Tying into `tpt-opt-core`

The crate consumes the canonical `tpt_opt_core::Model` through its LP engine
(`tpt_opt_milp::MilpSolver`); cone programs are expressed in the standalone
`ConeProgram` form (variables, linear objective, equality rows, SOC rows, SDP
blocks). Numerical tolerances and the iteration cap are passed to
`solve_conic` / `solve_socp`.

## Example

```rust
use tpt_opt_conic::{solve_socp, ConeProgram, SocRow};
use tpt_opt_core::model::Sense;

// max x1 + x2  s.t.  ‖(0.5 x1, 0.3 x1 + 0.4 x2)‖₂ ≤ 1 - x1,  x ≥ 0.
let q_mat = vec![
    vec![-1.0, 0.0], // r(x) = 1 - x1
    vec![0.5, 0.0],  // q component 1
    vec![0.3, 0.4],  // q component 2
];
let prog = ConeProgram {
    n: 2,
    c: vec![1.0, 1.0],
    sense: Sense::Maximize,
    bounds: vec![(0.0, 2.0), (0.0, 5.0)],
    eq_a: vec![],
    eq_b: vec![],
    soc_rows: vec![SocRow { q_mat, q_rhs: vec![1.0, 0.0, 0.0] }],
    sdp_blocks: vec![],
};
let sol = solve_socp(&prog, 1e-6, 400);
assert_eq!(sol.status, tpt_opt_conic::ConicStatus::Optimal);
assert!(sol.objective > 1.0);
```

## License

Licensed under either of MIT or Apache-2.0 at your option.
