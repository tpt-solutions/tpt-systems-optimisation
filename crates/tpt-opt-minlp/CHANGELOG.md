# Changelog

All notable changes to `tpt-opt-minlp` are documented here. This crate follows
the workspace convention of per-crate changelogs (Keep a Changelog format),
starting from the initial `0.1.0`.

## [Unreleased]

### Fixed
- `outer_approximate` no longer repeats the same master point until `max_iter` when
  the fixed-integer NLP fails to converge on a point that already satisfies every
  constraint. Such a point is now used as the feasible subproblem solution and
  contributes an objective cut, so the loop makes progress.

### Changed
- `generalized_benders` optimality cuts now come from the KKT multipliers of the
  continuous subproblem (recovered by a least-residual LP and certified by its
  residual), giving `θ ≥ f(x_k,y_k) + ∇_y(f + Σλg + Σμh)·(y − y_k)`. The previous
  finite-difference cuts were not valid and could cut off the optimum. Infeasible
  subproblems add no cut and are enumerated instead, and an exhausted integer
  enumeration proves optimality. The cut's validity argument covers convex problems
  only.
- `generalized_benders` reports `NumericalIssue` rather than `Optimal` when its lower
  bound exceeds the incumbent. `GbdConfig::probe_step` is retained for API
  compatibility and no longer affects the cuts.

### Added
- `tests/sweep_convex_minlp.rs`: convex MINLP instances against enumeration, with
  OA, GBD and SQP-B&B all required to reach the optimum.

## [0.1.0] - Unreleased

### Added
- `MinlpModel` representation with boxed-closure objective/constraints,
  optional analytic gradients (finite-difference fallback), variable domains,
  and indicator-gated nonlinear constraints (`model.rs`).
- Outer approximation (Duran–Grossmann) with MILP epigraph master, NLP
  subproblems over fixed integers, duality-gap termination and certificate
  history (`oa.rs`).
- Generalized Benders decomposition with validity-checked slope cuts,
  feasibility cuts from the violation measure, and grid diversification on
  master revisits (`gbd.rs`).
- SQP-style branch-and-bound for non-convex MINLPs: multi-start NLP node
  relaxations, most-fractional branching, bound pruning (`sqp.rs`).
- Continuous-subproblem adapter substituting fixed integers out of the NLP
  (`subproblem.rs`).
- McCormick envelopes for bilinear terms (`relax.rs`); αBB underestimators
  and tangent cuts (`alphabb.rs`).
- Logical constraints AND/OR/XOR/cardinality/implication → linear rows
  (`logical.rs`); big-M complementarity linearisation (`complementarity.rs`).
- Convergence certificates with gap tracking (`certificates.rs`).
- Integration benchmark cross-validating OA vs. GBD (convex instance) and SQP
  B&B vs. integer enumeration (non-convex instance) (`tests/benchmark.rs`).

### Added (Phase 13e)

- `spatial` module: `spatial_branch_and_bound` for bilinear programs
  `min cᵀx + Σ qₖ x_i x_j` over finite boxes with linear rows. Node bounds come
  from the McCormick LP (`relax::mccormick_envelope`) solved by
  `tpt_opt_milp::MilpSolver`; incumbents are exact objective values at feasible
  LP points; branching bisects the wider factor of the most violated product.
  Reports `Optimal`, `Infeasible`, `NodeLimit` or `Stalled` (a box too small to
  bisect in floating point, where optimality cannot be certified). Restricted to
  factorable bilinear terms; not a general global NLP solver.

### Not yet implemented (see `todo.md` Phase 4)
- MINLPLib corpus runner (external fixtures pipeline).
- Extended convex-relaxation coverage beyond bilinear/twice-differentiable
  terms (e.g. general factorable-program relaxations).