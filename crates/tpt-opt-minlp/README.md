# tpt-opt-minlp

Mixed-integer nonlinear programming (MINLP) for the
`tpt-systems-optimisation` workspace (spec.txt §3 "tpt-opt-minlp"). Pure Rust,
no `unsafe`, deterministic by construction.

## What is implemented

* **Model representation** (`model.rs`) — boxed-closure objective and
  constraints (`g(x) <= 0`, `h(x) = 0`) with optional analytic gradients
  (central finite differences as fallback), plus **indicator-gated**
  constraints: a nonlinear consequent enforced only while its binary switch
  equals a given value.
* **Outer approximation** (`oa.rs`, Duran–Grossmann) — MILP epigraph master
  accumulating tangent planes ↔ NLP subproblems with the integers fixed.
  Converges with duality-gap certificates on convex instances; non-convex
  runs are flagged via their certificate history rather than silently
  trusted.
* **Generalized Benders decomposition** (`gbd.rs`) — master over complicating
  integer variables with validity-checked slope cuts (each finite-difference
  slope component is verified against probed neighbours before use) and
  violation-based feasibility cuts; grid diversification prevents master
  tie-break stalling.
* **SQP branch-and-bound** (`sqp.rs`) — for possibly non-convex MINLPs:
  multi-start NLP relaxations per node, most-fractional branching, bound
  pruning. Bounds are heuristic on non-convex problems (documented).
* **Convex relaxations** — McCormick envelopes for bilinear products
  (`relax.rs`) and αBB convex underestimators / tangent cuts for
  twice-differentiable terms with a curvature bound (`alphabb.rs`).
* **Logical constraints** (`logical.rs`) — AND/OR/XOR/cardinality/implication
  compiled to linear rows over binaries.
* **Complementarity constraints** (`complementarity.rs`) — big-M
  linearisation of `x·y = 0` pairs.
* **Convergence certificates** (`certificates.rs`) — per-iteration lower /
  upper bounds and gap tracking.

The continuous subproblems are solved by the augmented-Lagrangian solver in
`tpt-math-optimize-general` through the shared adapter in `subproblem.rs`
(fixed integers are substituted out, not penalised).

## Status

The crate builds clean (fmt/clippy `-D warnings`) and its tests pass: unit
tests per module plus an integration benchmark (`tests/benchmark.rs`)
cross-validating OA vs. GBD on a convex instance and SQP branch-and-bound vs.
integer enumeration on a non-convex one. A full MINLPLib corpus runner remains
future work (see `todo.md` Phase 4).

## Example

```rust
use tpt_opt_minlp::{model::{MinlpModel, VarKind}, oa::{outer_approximate, OaConfig}};

// min x + y  s.t.  y >= (x−1)²,  x ∈ [0,2] continuous, y ∈ ℤ ∩ [0,4].
let mut m = MinlpModel::new(2, |x| x[0] + x[1]);
m.set_var(0, VarKind::Continuous, 0.0, 2.0);
m.set_var(1, VarKind::Integer, 0.0, 4.0);
m.add_le(
    |x| (x[0] - 1.0) * (x[0] - 1.0) - x[1],
    |x, g| { g[0] = 2.0 * (x[0] - 1.0); g[1] = -1.0; },
);
let res = outer_approximate(&m, &OaConfig::default());
assert_eq!(res.status, tpt_opt_minlp::oa::OaStatus::Optimal);
assert!((res.objective.unwrap() - 1.0).abs() < 1e-3);
```

## License

Licensed under either of MIT or Apache-2.0 at your option.