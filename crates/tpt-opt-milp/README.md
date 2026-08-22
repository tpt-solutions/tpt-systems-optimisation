# tpt-opt-milp

Mixed-integer linear programming (MILP) solver for the
`tpt-systems-optimisation` workspace (spec.txt §3 "tpt-opt-milp"). Pure Rust,
no `unsafe`, `std` with an `alloc`/no_std-compatible core.

## What is implemented

* **Branch-and-bound / branch-and-cut core** (`milp.rs`) — root-node cut passes,
  primal heuristics at the nodes, and deterministic parallel tree search via
  `.with_threads(n > 1)`.
* **Cutting planes** — model-space clique, cover (with exact lifting) and MIR
  cuts (`cuts.rs`); tableau-space Gomory mixed-integer and lift-and-project
  intersection cuts (`gomory.rs`).
* **Primal heuristics** — rounding, feasibility pump, RINS and local branching.
* **Branching** — most-fractional, pseudo-cost, and limited strong branching.
* **Node selection** — best-bound, depth-first, and best-estimate.
* **Modelling extras** — SOS1/SOS2 sets, indicator constraints, and
  piecewise-linear objectives.
* **Determinism** — `.with_seed(...)` makes branching and heuristics
  reproducible for a fixed seed.

## External solver binding (feature `highs`)

An optional, non-default `highs` feature wires [HiGHS](https://highs.dev)
(MIT-licensed, via the `highs`/`highs-sys` crates) as an alternate
`tpt_opt_core::Solver` implementation (`HighsSolver`) for benchmarking and
cross-validation against the bundled engine:

```toml
tpt-opt-milp = { version = "0.1", features = ["highs"] }
```

> **Build-toolchain requirement:** enabling `highs` compiles the HiGHS C++
> sources from scratch. This requires **cmake** plus a C++ compiler toolchain
> (**MSVC** on Windows, gcc/clang on Linux/macOS) at build time. The default
> feature set stays pure Rust with no such requirement. A host `libclang`
> is also needed by bindgen when generating the FFI declarations.
>
> Cross-solver validation tests comparing `MilpSolver` against `HighsSolver`
> on shared small MILP instances live in
> `tests/highs_cross_validation.rs` (run with
> `cargo test -p tpt-opt-milp --features highs`).

## Status

The crate builds and its hand-crafted MILP examples in `tests/milp_api.rs`
solve to optimality; the full Phase 2 checklist in `todo.md` is complete.

## Tying into `tpt-opt-core`

The solver consumes the canonical `tpt_opt_core::Model` and reports results
through `tpt_opt_core::Solution` / `tpt_opt_core::SolverStatus`, with
numerical tolerances configurable via `tpt_opt_core::Tolerances`.

## Example

```rust
use tpt_opt_milp::MilpSolver;
use tpt_opt_core::{Model, Objective, Constraint, VarBound, Sense};

let mut model = Model::new(2);
let x = model.add_variable(VarBound::binary());
let y = model.add_variable(VarBound::integer(0.0, 10.0));
model.set_objective(Objective::minimize(vec![x, y], vec![1.0, 1.0]));
model.add_constraint(Constraint::le(vec![x, y], vec![1.0, 1.0], 3.0));

let mut solver = MilpSolver::new();
let sol = solver.solve(&model).unwrap();
assert!(sol.status.has_solution());
```

## License

Licensed under either of MIT or Apache-2.0 at your option.
