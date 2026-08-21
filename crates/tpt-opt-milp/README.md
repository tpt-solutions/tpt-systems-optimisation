# tpt-opt-milp

Mixed-integer linear programming (MILP) solver for the
`tpt-systems-optimisation` workspace (spec.txt §3 "tpt-opt-milp"). Pure Rust,
no `unsafe`, `std` with an `alloc`/no_std-compatible core.

## What is implemented

* **Branch-and-bound / branch-and-cut core** (`milp.rs`) — a root-node Gomory
  mixed-integer cut pass plus depth-first (LIFO) diving.
* **Cutting planes** — Gomory mixed-integer cuts (`cuts.rs`). Clique, cover,
  MIR and lift-and-project cuts are not yet implemented.
* **Primal heuristics** — rounding and a feasibility pump (`try_rounding` /
  `try_feasibility_pump`). RINS and local branching are not yet implemented.
* **Branching** — most-fractional and a best-effort pseudo-cost estimate; strong
  branching not yet implemented.
* **Node selection** — depth-first (LIFO) diving; best-bound / best-estimate not
  yet implemented.
* **Determinism** — `.with_seed(...)` makes branching and heuristics
  reproducible for a fixed seed.

External MILP solvers can be plugged in behind the `tpt_opt_core::Solver`
trait; an optional, non-default `highs` feature (HiGHS, MIT) is planned for
benchmarking/production use (see spec §4 "Solver Agnosticism").

## Status

The crate builds and its hand-crafted MILP examples in `tests/milp_api.rs`
solve to optimality. It is **not yet** feature-complete with respect to the
full spec checklist (see `todo.md` Phase 2): several cut families, primal
heuristics, node-selection strategies, SOS/indicator/piecewise-linear support,
and parallel tree search remain to be implemented.

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
