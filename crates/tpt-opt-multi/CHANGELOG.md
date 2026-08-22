# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Pareto dominance, front extraction, and the additive epsilon indicator
  (`dominance`).
- Hypervolume: exact 2-D and WFG algorithm for N-D (`hypervolume`).
- Objective normalisation from samples (`normalizer::ObjectiveNormalizer`).
- Seeded NSGA-II with configurable population/generations (`nsga2`).
- NSGA-III (`nsga3`): Das–Dennis structured reference directions
  (`das_dennis`), custom reference-direction preference articulation
  (`Nsga3::with_reference_directions`), ASF-based normalisation with
  hyperplane intercepts, and deterministic niche-based environmental
  selection.
- Decision-making utilities (`decision`): knee-point detection (chord
  distance for 2-D convex fronts with L2 fallback; ASF knee for M ≥ 3),
  envelope-based trade-off ratio matrices (`tradeoff_ratios`), and
  deterministic k-means clustering of front members (`cluster_solutions`).
- Weighted-sum and ε-constraint scalarisations backed by `tpt-opt-milp`
  (`scalarize`).
