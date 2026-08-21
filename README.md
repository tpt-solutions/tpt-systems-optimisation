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

## License

Licensed under either of [MIT](./LICENSE-MIT) or [Apache-2.0](./LICENSE-APACHE)
at your option.

### Upstream math dependencies (local dev)

The crates in `deps/` are local development shims mirroring the API surface of
the sibling `tpt-math` workspace (`tpt-math-linalg`, `tpt-math-graph`,
`tpt-math-prob`, …). They are wired as **path dependencies** for local builds
and will be swapped to version dependencies once `tpt-math` itself publishes.
They are excluded from the published optimisation crates.
