# Changelog

All notable changes to `tpt-opt-milp` are documented here. This crate follows
the workspace convention of per-crate changelogs (Keep a Changelog format),
starting from the initial `0.1.0` scaffold.

## [0.1.0] - Unreleased

### Added
- Branch-and-bound / branch-and-cut MILP core (`milp.rs`) with a root-node
  Gomory mixed-integer cut pass.
- Gomory mixed-integer cut generation (`cuts.rs`).
- Primal heuristics: rounding and feasibility pump.
- Most-fractional and pseudo-cost branching, depth-first (LIFO) node diving.
- Deterministic `.with_seed(...)` for reproducible branching/heuristics.
- `tests/milp_api.rs` with small hand-crafted MILP examples solved to optimality.

### Not yet implemented (see `todo.md` Phase 2)
- Clique / cover / MIR / lift-and-project cuts.
- RINS and local-branching heuristics.
- Best-bound / best-estimate node selection.
- Strong branching, SOS1/SOS2, indicator constraints, piecewise-linear
  objectives, parallel tree search.
- Optional non-default `highs` feature (HiGHS binding).
