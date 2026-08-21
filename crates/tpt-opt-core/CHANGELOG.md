# Changelog

All notable changes to this crate are documented here, per
[Keep a Changelog](https://keepachangelog.com/). This project adheres to
semantic versioning; the initial `0.1.0` baseline is pre-1.0 and may have
breaking changes between minor versions.

## [0.1.0] - Unreleased

### Added
- Canonical linear problem representation: `Model`, `Variable`, `Constraint`,
  `Objective`, `Sense`.
- Variable bound kinds: `VarBound`, `VarType` (continuous / integer / binary /
  semi-continuous) with `Bound` intervals.
- Configurable numeric tolerances (`Tolerances`) with spec §4 defaults.
- Solver-agnosticism contract: `Solver`, `SolveParameters`, `Solution`,
  `SolverStatus`, `WarmStart`, `Verbosity`.
- Structured error types with infeasibility diagnostics (`OptError`,
  `InfeasibilityReport`).
- Sparse constraint-matrix assembly (`model_to_csr`, `model_to_csc`,
  `ConstraintMatrix`) compatible with `tpt-math-linalg`.
- Extensibility hook `CustomConstraint` (`evaluate` / `gradient` /
  `is_violated`).
- `no_std` support with optional `alloc` (default `std`).
- Cargo-publish metadata, README, and CHANGELOG.
