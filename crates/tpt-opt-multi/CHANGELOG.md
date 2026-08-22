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
- Weighted-sum and ε-constraint scalarisations backed by `tpt-opt-milp`
  (`scalarize`).