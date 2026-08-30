# Changelog

All notable changes to `tpt-opt-milp` are documented here. This crate follows
the workspace convention of per-crate changelogs (Keep a Changelog format),
starting from the initial `0.1.0` scaffold.

## [0.1.0] - Unreleased

### Added
- Branch-and-bound / branch-and-cut MILP core (`milp.rs`) with root-node cut
  passes re-solved over configurable rounds (`with_parallel_cuts`).
- Cut families: Gomory mixed-integer cuts and lift-and-project intersection
  cuts in tableau space (`gomory.rs`); clique, cover, and MIR cuts in model
  space (`cuts.rs`).
- Primal heuristics: rounding (plus seeded randomised trials), feasibility
  pump, RINS, and local branching — applied at the root and periodically
  during search.
- Node selection strategies: best-bound, best-estimate (pseudo-cost
  degradation), depth-first (`NodeSelection`).
- Branching rules: most-fractional, pseudo-cost product score, limited strong
  branching with pseudo-cost refinement (`BranchingRule::StrongBranching`).
- Special ordered sets SOS1/SOS2 (`sos.rs`) with specialised branching and
  member-range fixing; indicator constraints via big-M expansion from variable
  bounds (`indicator.rs`); piecewise-linear objectives via lambda
  reformulation + SOS2 (`piecewise.rs`).
- Deterministic parallel tree search (`solve_parallel`,
  breadth-partitioned scoped-thread subtree assignment) plus concurrent root
  cut rounds; results are seed-deterministic regardless of thread count.
- Deterministic `.with_seed(...)`, `.with_threads(...)`, and
  `.with_parallel_cuts(...)` configuration.
- Two-phase simplex LP engine (`lp.rs`) with bound-shifted objective recovery,
  negative-RHS row flipping, unboundedness detection, and correct handling of
  the model's objective constant in every node bound.
- Progress-callback API: `MilpSolver::with_progress_callback` delivering
  `tpt_opt_core::progress::ProgressEvent` checkpoints (iterations, incumbent,
  dual bound, elapsed time) with cooperative `ProgressAction::Abort`
  early termination, in both sequential and parallel modes.
- Free-format MPS reader/writer and CPLEX-LP reader/writer (`format.rs`):
  INTORG/INTEND markers, OBJSENSE MAX, UP/LO/FX/FR/MI/PL/BV/LI/UI/SC bound
  cards, ±1e30 infinity sentinels, double-bounded rows, General/Binary
  sections, comments, wrapped rows; round-trip tests preserve sense,
  integrality, and optimum.
- Optional non-default `highs` feature: `HighsSolver` binding translating the
  canonical `Model` to HiGHS' column-wise form, mapping terminal statuses,
  applying the parameter bundle, restoring the objective constant, and
  supporting warm-start primal hints (requires cmake + a C++ toolchain).
- Integration tests: MIPLIB p0033 solved to the published optimum 3089
  (sequential and 4-thread); benchmark-corpus suite over Netlib LP classics
  (afiro, adlittle, e226) and MIPLIB MILP classics (flugpl, gt2, egout,
  bell5 soundness) fetched out-of-band via `cargo xtask fetch-fixtures`;
  cross-validation against HiGHS on five shared instances (feature-gated);
  design-principle suites for seeded determinism, parallel-vs-sequential
  equality, numerical robustness, custom-constraint extensibility, and fuzzed
  invariants; ignored-by-default performance report (`perf_regression.rs`).

### Fixed
- Objective-convention mismatch that let maximisation models with a positive
  objective constant be pruned prematurely (node bounds now include the
  constant, matching incumbent scoring); regression-tested.
- Zero-constraint LP shortcut declared Optimal without checking improvement
  toward an infinite bound; unbounded LPs are now detected correctly.

## [Unreleased]

### Not yet implemented (see `todo.md`)
- Tree-wide cut management beyond the root passes.
- Work-stealing parallel tree search pool (current scheme is deterministic
  breadth-partitioned subtree assignment).