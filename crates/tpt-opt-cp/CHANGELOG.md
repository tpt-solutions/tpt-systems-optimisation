# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Integer domains with value-set operations (`domain::Domain`).
- Propagation fixpoint over per-constraint filters (`model::fixpoint`).
- Constraints: `Linear`, `AllDifferent`, `Cumulative`, `Element`, `Table`,
  and reification via `Reified`.
- First-fail backtracking search: `solver::solve` and `solver::solutions`.
- Unit tests including an n-queens model built from `AllDifferent`,
  `Linear`, and reified diagonal constraints.