# Changelog

All notable changes to `tpt-opt-minlp` are documented here. This crate follows
the workspace convention of per-crate changelogs (Keep a Changelog format),
starting from the initial `0.1.0`.

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

### Not yet implemented (see `todo.md` Phase 4)
- MINLPLib corpus runner (external fixtures pipeline).
- Extended convex-relaxation coverage beyond bilinear/twice-differentiable
  terms (e.g. general factorable-program relaxations).