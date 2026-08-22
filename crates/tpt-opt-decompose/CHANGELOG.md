# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Benders decomposition (`BendersSolver`) for two-stage problems with
  (mixed-)integer first-stage variables: explicit dual-LP cut generation,
  Farkas feasibility cuts, and gap-certified convergence.
- Magnanti–Wong Pareto-optimal cut generation (`with_pareto_cuts`) against a
  user-supplied core point.
- Stabilisation techniques: infinity-norm trust region and level-set
  restriction, both certified by a final unrestricted master solve.
- Dantzig–Wolfe decomposition (`DantzigWolfe`) for block-angular programs:
  big-M artificial seeding, per-block pricing LPs, and λ-based solution
  reconstruction.
- Restricted master problem management (`RmpPool`): near-duplicate column
  rejection and an optional capacity cap.
- Branch-and-price (`BranchAndPrice`): depth-first branch-and-bound over
  integer master variables with embedded column generation, a pluggable
  `Pricer` trait (continuous-LP default `LpPricer`), and dual-neutral cleanup
  pricing at integer nodes.
- Lagrangian relaxation: subgradient ascent (Polyak / diminishing steps,
  `lagrangian_subgradient`), cutting-plane bundle/level method
  (`lagrangian_bundle_level`), and surrogate-relaxation search
  (`surrogate_search`).
- Automatic decomposable-structure detection (`detect_structure`): bipartite
  row–column connectivity analysis identifying independent blocks, linking
  rows/columns, and recommending a decomposition strategy.
- Integration test suite validating every framework against hand-computed
  optima (Benders capacity/multi-scenario/feasibility-cut instances, cutting
  stock via branch-and-price cross-validated against a monolithic pattern
  MILP, analytic Lagrangian duals, structure-detection reports).