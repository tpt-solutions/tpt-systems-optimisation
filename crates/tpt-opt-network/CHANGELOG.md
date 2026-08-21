# Changelog

All notable changes to `tpt-opt-network` are documented here. This crate follows
the workspace convention of per-crate changelogs (Keep a Changelog format),
starting from the initial `0.1.0` scaffold.

## [0.1.0] - Unreleased

### Added
- Min-cost flow: successive shortest path (`min_cost_flow`) and network simplex
  (`network_simplex`).
- Hungarian algorithm for assignment / matching (`assignment.rs`).
- Optimal power flow: DC-OPF (`dc_opf`, LP), AC-OPF (`ac_opf`, polar NLP via
  `tpt-math-optimize-general`), and SC-OPF (`sc_opf`, N-1 contingency analysis)
  in `opf.rs`.
- Dynamic networks (`dynamic.rs`): period-by-period min-cost flow with a
  warm-start hint.
- Graph preprocessing: cycle detection, bridge identification, biconnected
  component decomposition (`graph_preprocess.rs`).
- In-crate two-phase simplex LP solver (`lp.rs`) implementing
  `tpt_opt_core::Solver<Model>`, used by DC-OPF / SC-OPF.
- Dependency on `tpt-math-optimize-general` (required by `ac_opf`).

### Fixed
- Previously the crate did not compile: `lib.rs` declared `mod opf` /
  `mod dynamic` without source files, and `lp.rs` / `min_cost_flow.rs` had
  borrow-check errors. Both modules were implemented and the borrow errors
  corrected.
