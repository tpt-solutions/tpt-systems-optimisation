# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Cross-cutting verification suite for the spec's design principles:
  - `tpt-opt-milp/tests/design_principles.rs`: degeneracy handling, badly
    scaled rows, custom-constraint outer-approximation, fuzzed binary
    programs, bit-identical sequential re-solves, and parallel-tree-search
    determinism.
  - `tpt-opt-cp/tests/custom_constraint.rs`: end-to-end extensibility demo —
    a user-defined `SumOfSquaresLe` constraint implementing the public
    `Constraint` trait, propagated and checked by the CP engine.
  - `tpt-opt-systems/tests/solver_agnosticism.rs`: one generic driver proves
    `MilpSolver`, `LpSolver`, `SimulatedAnnealing`, `TabuSearch`, and
    `ParticleSwarmOptimization` all honour the same `Solver<Model>` contract
    (`solve` / `set_parameter` / `warm_start` / `status` / `solution`).
- `Solver<Model>` implementations for the remaining heuristic solvers
  (`TabuSearch`, `ParticleSwarmOptimization`, continuous `GeneticAlgorithm`),
  completing solver agnosticism for the heuristic family.

### Fixed

- Umbrella feature gating: the `builders` module was compiled only under
  `milp`, but its graph-dependent half (`NetworkFlowBuilder`) is required by
  `network`-only builds — combinations like `["network", "robust"]` failed
  with an unresolved-import error, as did any combo enabling `milp` without
  `network` (the module unconditionally imported `tpt_math_graph`). The
  module is now gated on `any(milp, network)` with each builder gated
  individually. Found by the Tier 2 consumption matrix.
- `tpt-opt-network` simplex: `reduced_costs` returned the *negated* reduced
  cost, so the entering rule ascended instead of descended and phase II could
  terminate prematurely at a non-optimal vertex (e.g. reporting 0 instead of
  the true optimum on `min -x-y s.t. x+y<=1`). Reduced costs are now computed
  with the correct sign convention and basic columns price out to zero.
- `tpt_opt_core::custom::CustomConstraint::is_violated` now checks the exact
  user predicate directly instead of delegating to the linearisation's
  gradient cut, so feasibility verdicts match the declared constraint even
  when the linearisation is loose at the candidate point.

### Changed

- CI: the `no-std` job no longer invokes a non-existent `cargo xtask`; it
  checks `tpt-opt-core` for `thumbv6m-none-eabi` both with and without the
  `alloc` feature.

## [0.1.0] - 2026-08-22

### Added

- Umbrella crate wiring the whole `tpt-systems-optimisation` solver suite
  behind one dependency with one flat feature per family: `milp`, `minlp`,
  `network`, `cp`, `heuristic`, `multi`, `robust`, `decompose`, plus the
  `all-solvers` meta-feature.
- Always-on core surface (`tpt_opt_systems::core` and flat core re-exports);
  a no-features build compiles without any solver backend.
- Unified `OptimizationError` tagging failures with the producing algorithm
  and preserving the underlying `tpt_opt_core::OptError`.
- `MilpBuilder` (feature `milp`): fluent canonical-model assembly solved by
  the bundled branch-and-bound engine.
- `NetworkFlowBuilder` (feature `network`): fluent min-cost-flow assembly.
- `convert::network_flow_to_milp` (features `network` + `milp`): lowers a
  min-cost-flow instance into a canonical MILP model for any MILP backend.