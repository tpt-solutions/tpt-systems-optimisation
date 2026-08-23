# tpt-systems-optimisation

[![CI](https://github.com/tpt-solutions/tpt-systems-optimisation/actions/workflows/ci.yml/badge.svg)](https://github.com/tpt-solutions/tpt-systems-optimisation/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue.svg)](./LICENSE-MIT)
[![MSRV 1.84](https://img.shields.io/badge/MSRV-1.84-orange.svg)](./rust-toolchain.toml)
[![crates.io](https://img.shields.io/crates/v/tpt-opt-systems.svg)](https://crates.io/crates/tpt-opt-systems)
[![docs.rs](https://docs.rs/tpt-opt-systems/badge.svg)](https://docs.rs/tpt-opt-systems)

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

| Crate | Role | Docs |
|-------|------|------|
| [`tpt-opt-core`](./crates/tpt-opt-core) | Canonical problem representation + `Solver` trait contract (no_std + alloc) | [docs.rs](https://docs.rs/tpt-opt-core) |
| [`tpt-opt-milp`](./crates/tpt-opt-milp) | Mixed-Integer Linear Programming (branch-and-bound / branch-and-cut) | [docs.rs](https://docs.rs/tpt-opt-milp) |
| [`tpt-opt-network`](./crates/tpt-opt-network) | Network flow + optimal power flow (network simplex, OPF) | [docs.rs](https://docs.rs/tpt-opt-network) |
| [`tpt-opt-minlp`](./crates/tpt-opt-minlp) | Mixed-Integer Nonlinear Programming (OA, GBD, SQP) | [docs.rs](https://docs.rs/tpt-opt-minlp) |
| [`tpt-opt-cp`](./crates/tpt-opt-cp) | Constraint programming engine (propagation, global constraints) | [docs.rs](https://docs.rs/tpt-opt-cp) |
| [`tpt-opt-heuristic`](./crates/tpt-opt-heuristic) | Metaheuristics (SA, GA, tabu, PSO) | [docs.rs](https://docs.rs/tpt-opt-heuristic) |
| [`tpt-opt-multi`](./crates/tpt-opt-multi) | Multi-objective / Pareto optimisation (NSGA-II/III, ε-constraint, Tchebycheff) | [docs.rs](https://docs.rs/tpt-opt-multi) |
| [`tpt-opt-robust`](./crates/tpt-opt-robust) | Robust & stochastic optimisation under uncertainty | [docs.rs](https://docs.rs/tpt-opt-robust) |
| [`tpt-opt-decompose`](./crates/tpt-opt-decompose) | Large-scale decomposition (Benders, Dantzig-Wolfe, Lagrangian) | [docs.rs](https://docs.rs/tpt-opt-decompose) |
| [`tpt-opt-systems`](./crates/tpt-opt-systems) | Feature-gated umbrella re-exporting all of the above | [docs.rs](https://docs.rs/tpt-opt-systems) |
| [`tpt-opt-cli`](./crates/tpt-opt-cli) | Command-line front end: solve an MPS/LP file from the shell | [docs.rs](https://docs.rs/tpt-opt-cli) |

(The crates.io/docs.rs badges resolve once the crates are published; links are
stable beforehand.)

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

### Domain-flavoured snippets

One taste of the API surface each consumer would lean on (feature sets as in
the table above):

```rust
// tpt-energy — DC-OPF over a transmission network (`network`)
let dispatch = tpt_opt_systems::network::opf::dc_opf(&grid)?;
println!("total generation cost: {}", dispatch.objective_value);

// tpt-energy — two-stage unit commitment under wind scenarios (`robust`)
let uc = tpt_opt_systems::robust::scenario::TwoStageProblem::solve(&first_stage, &scenarios)?;
println!("VSS = {}", tpt_opt_systems::robust::value::vss(&rp, &ws, &eev)?);

// tpt-transportation — vehicle assignment as CP with `cumulative` (`cp`)
let mut model = tpt_opt_systems::cp::Model::new();
model.add(tpt_opt_systems::cp::Cumulative::new(starts, durations, demand, fleet))?;

// tpt-transportation — depot selection MILP through the builder (`milp`)
let sol = tpt_opt_systems::MilpBuilder::new(0).add_binary().le(&rows, &coeffs, budget).minimize(&cost_idx, &cost).solve()?;

// tpt-process — reactor-network synthesis by outer approximation (`minlp`)
let flowsheet = tpt_opt_systems::minlp::OaSolver::new().with_gap(1e-4).solve(&synthesis_model)?;

// tpt-process — Benders decomposition of a multi-period planning LP (`decompose`)
let plan = tpt_opt_systems::decompose::BendersSolver::new().solve(&master, &subproblems)?;

// tpt-construction — schedule feasibility with `alldifferent` + precedence (`cp`)
let schedule = tpt_opt_systems::cp::solve(&crew_model)?.next()?;

// tpt-earth — min-cost flow for haulage routing (`network`)
let routed = tpt_opt_systems::NetworkFlowBuilder::new(nodes).add_edge(u, v, cap, cost).supply(src, q).demand(dst, q).solve()?;

// tpt-materials — alloy blend on the Pareto front (`multi`)
let front = tpt_opt_systems::multi::nsga2::Nsga2::new(blend_problem).with_seed(42).run();

// tpt-medical — ward staffing MILP with fairness rows (`milp`)
let roster = tpt_opt_systems::MilpBuilder::new(0).add_integer(0.0, staff).ge(&cov_idx, &cov, demand).minimize(&w_idx, &w).solve()?;

// tpt-electronics — placement overlap check via `alldifferent` (`cp`) or
// thermal-budget MILP (`milp`); both families share the core `Model`.
```

(Signatures are illustrative; consult each crate's docs.rs page for exact APIs.)

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

Benchmark corpora for integration tests are fetched **outside** every crate
directory (so they never enter a packaged `.crate` file):

```text
cargo xtask fetch-fixtures            # download Netlib LP + MIPLIB classics into tests/fixtures/
cargo xtask fetch-fixtures --list     # show the curated manifest
cargo xtask fetch-fixtures netlib     # subset by suite or instance name
```

New Tier 2 consumer crates can be scaffolded from the bundled template:

```sh
cargo install cargo-generate
cargo generate --git https://github.com/tpt-solutions/tpt-systems-optimisation --path template
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

### In-house vs HiGHS at a glance

The cross-validation suite solves five shared instances with both engines and
asserts agreement on status *and* objective:

| Instance | Type | Bundled B&B | HiGHS | Agreement |
|----------|------|-------------|-------|-----------|
| knapsack (24 items) | maximisation | optimal | optimal | ✔ same optimum |
| covering (20×25) | set cover | optimal | optimal | ✔ same optimum |
| mixed continuous/integer equality | minimisation | optimal | optimal | ✔ −10.5 both |
| infeasible variant | feasibility | `Infeasible` | `Infeasible` | ✔ |
| unbounded variant | feasibility | `Unbounded` | `Unbounded` | ✔ |

For raw speed comparisons, reproduce locally rather than trusting published
numbers (hardware-dependent): build with `--features tpt-opt-milp/highs`
(needs cmake + a C++ toolchain) and time both engines over the fixture corpus
(`cargo xtask fetch-fixtures`). As a correctness anchor, the bundled LP engine
solves the classic Netlib instance `afiro` to its published optimum
−464.753142857… (see `tests/fixtures/netlib/afiro.mps`).

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
