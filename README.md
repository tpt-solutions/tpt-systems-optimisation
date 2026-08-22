# tpt-systems-optimisation

A workspace of pure-Rust solvers for **large-scale, network-aware, and discrete
mathematical optimisation**, bridging the gap between the pure-math building
blocks in [`tpt-math`](https://github.com/tpt-solutions/tpt-math) and the Tier 2
domain repositories (energy, transportation, process, construction, earth,
materials, medical, electronics).

Every solver is implemented in pure Rust with **no external solver
dependencies** by default. The trait design allows plugging in commercial
solvers (HiGHS — MIT licensed — is wired behind an opt-in, non-default feature)
for benchmarking or production use without changing model code.

> Full specification: [`spec.txt`](./spec.txt). Build checklist: [`todo.md`](./todo.md).

## Crate map

| Crate | Role |
|-------|------|
| `tpt-opt-core` | Canonical problem representation + `Solver` trait contract (no_std + alloc) |
| `tpt-opt-milp` | Mixed-Integer Linear Programming (branch-and-bound / branch-and-cut) |
| `tpt-opt-network` | Network flow + optimal power flow (network simplex, OPF) |
| `tpt-opt-minlp` | Mixed-Integer Nonlinear Programming (OA, GBD, SQP) |
| `tpt-opt-cp` | Constraint programming engine (propagation, global constraints) |
| `tpt-opt-heuristic` | Metaheuristics (SA, GA, tabu, PSO) |
| `tpt-opt-multi` | Multi-objective / Pareto optimisation (NSGA-II, ε-constraint) |
| `tpt-opt-robust` | Robust & stochastic optimisation under uncertainty |
| `tpt-opt-decompose` | Large-scale decomposition (Benders, Dantzig-Wolfe, Lagrangian) |
| `tpt-opt-systems` | Feature-gated umbrella re-exporting all of the above |

## Build order

`core → milp → network → minlp → cp → heuristic → multi → robust → decompose → systems`

## Tier 2 consumption examples

Each Tier 2 repo depends on `tpt-opt-systems` with only the features it needs:

| Consumer | Features |
|----------|----------|
| `tpt-energy` | `milp`, `network`, `robust` |
| `tpt-transportation` | `milp`, `cp`, `multi`, `heuristic` |
| `tpt-process` | `minlp`, `decompose`, `multi` |
| `tpt-construction` | `cp`, `milp`, `multi` |
| `tpt-earth` | `network`, `robust` |
| `tpt-materials` | `multi`, `heuristic` |
| `tpt-medical` | `milp`, `cp` |
| `tpt-electronics` | `milp`, `network` |

## Quick start

Depend on the umbrella crate with just the solver families you need:

```toml
[dependencies]
tpt-opt-systems = { version = "0.1", features = ["milp", "network"] }
```

```rust
use tpt_opt_systems::{MilpBuilder, NetworkFlowBuilder};

// A tiny MILP through the fluent builder…
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

Runnable cross-crate programs live in [`examples/`](./examples) (excluded from
the workspace so their all-features dependency never affects packaging):

```text
cargo run --manifest-path examples/Cargo.toml --example milp_knapsack
cargo run --manifest-path examples/Cargo.toml --example flow_to_milp
cargo run --manifest-path examples/Cargo.toml --example heuristic_pareto
```

## Tooling

All task logic lives in the `xtask` crate; both `cargo` and `just` front it:

```text
cargo xtask fmt | clippy | test | deny | no-std | check | all   # or:
just fmt / clippy / test / deny / no-std / check / all
```

CI (`.github/workflows/ci.yml`) runs rustfmt, clippy `-D warnings`, tests +
doctests via cargo-nextest, cargo-deny, a no_std cross-check for
`tpt-opt-core`, a feature-powerset sweep of the umbrella (cargo-hack), the
eight Tier 2 feature combinations, an MSRV build at Rust 1.84, informational
cargo-semver-checks, bench compile-smoke, and a non-blocking MILP performance
report.

## Testing

Beyond per-crate unit/doctests, the workspace carries cross-cutting suites:

- **Design principles** (`tpt-opt-milp/tests/design_principles.rs`,
  `tpt-opt-cp/tests/custom_constraint.rs`,
  `tpt-opt-systems/tests/solver_agnosticism.rs`) — degeneracy/bad-scaling
  robustness, custom-constraint extensibility, seeded determinism,
  parallel-vs-sequential equality.
- **Fuzz suites** (`tests/fuzz.rs` in cp/network/minlp plus the MILP fuzz in
  `design_principles.rs`) — seeded random instances with verified invariants
  (soundness, completeness vs brute force, feasibility, cost consistency).
- **Cross-validation** (`tpt-opt-milp/tests/highs_cross_validation.rs`) —
  bundled engine vs HiGHS behind the opt-in `highs` feature.
- **Performance report** (`tpt-opt-milp/tests/perf_regression.rs`) —
  ignored-by-default timing report over fixed instances; run with
  `cargo test -p tpt-opt-milp --test perf_regression -- --ignored --nocapture`.

## Changelog convention

Changelogs are **per crate** (`crates/<name>/CHANGELOG.md`, Keep-a-Changelog
format); there is intentionally no root changelog.

## License

Licensed under either of [MIT](./LICENSE-MIT) or [Apache-2.0](./LICENSE-APACHE)
at your option.

### Upstream math dependencies (local dev)

The crates in `deps/` are local development shims mirroring the API surface of
the sibling `tpt-math` workspace (`tpt-math-linalg`, `tpt-math-graph`,
`tpt-math-prob`, …). They are wired as **path dependencies** for local builds
and will be swapped to version dependencies once `tpt-math` itself publishes.
They are excluded from the published optimisation crates.
