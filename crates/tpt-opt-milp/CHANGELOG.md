# Changelog

All notable changes to `tpt-opt-milp` are documented here. This crate follows
the workspace convention of per-crate changelogs (Keep a Changelog format),
starting from the initial `0.1.0` scaffold.

## [Unreleased]

### Added
- `find_iis` (`iis.rs`): irreducible infeasible subsystem by the deletion
  filter over linear constraints, with variable bounds and custom constraints
  held fixed.
- `presolve_model` (`presolve.rs`): empty-row removal, singleton-row bound
  tightening (integer bounds rounded inward), duplicate-term merging, and a
  `PresolveReport`. Variable indices are preserved.
- `read_opb` and `write_opb` (`opb.rs`): pseudo-Boolean OPB reader and writer
  for binary models with integer coefficients; the writer rejects anything OPB
  cannot express.
- `read_lp` now accepts objective constants, and `write_lp` writes them.

### Changed
- `read_lp` rejects trailing tokens in Bounds entries and constraint right-hand
  sides instead of silently dropping them.
- Removed the unused `sense` field from `LpState`.
- `Instant` comes from `web-time`, so the solver's time limit works on
  `wasm32-unknown-unknown`.

### Fixed
- Duals (`lp.rs`): the dual of each row was the cost of the basic variable in
  that row rather than the simplex multiplier, and its sign depended on the row
  kind, so duals of `<=` and `=` rows were negated. Reduced costs are now taken
  from the basis costs against the tableau. Checked by strong duality on 3000
  random LPs (`tests/lp_duality.rs`) and a regression for each row kind and sense.
  Column generation (`tpt-opt-decompose`) depended on these duals.
- Root cuts (`gomory.rs`, `lp.rs`): the tableau-space column expression of a
  shifted integer column had the wrong sign (`y = x - lb` was recorded as
  `y = x + lb`). Gomory and lift-and-project cuts were invalid for variables with
  nonzero lower bounds, and `with_cuts(true)` could report a feasible model as
  infeasible. Found by a brute-force check against enumeration
  (`tests/milp_bruteforce.rs`), which passes on 20 000 random models.
- LP relaxation (`lp.rs`): phase I missed a positive artificial sitting in a
  row other than its own, so an infeasible LP could report `Optimal`. Phase II
  let a zero-valued artificial grow past a negative entry, which could break an
  equality. Both are fixed, and a random cross-check against the network LP
  agrees on 3000 cases.

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