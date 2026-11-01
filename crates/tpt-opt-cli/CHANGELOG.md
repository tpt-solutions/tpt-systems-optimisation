# Changelog

All notable changes to this crate are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Initial release: MPS/CPLEX-LP model ingestion via `tpt-opt-milp::format`,
  solving through `MilpSolver` with time-limit/thread/seed/cut options,
  `--export` format conversion, quiet mode, and stable exit codes.
### Changed

- Subcommands: `solve` (the default, so `tpt-opt-cli model.mps` still works),
  `info`, `classify`, `convert`, `iis`, and `presolve`.
- `--format json` output for `solve`, `info`, `classify`, `iis`, and `presolve`.
- `--gap`, `--presolve`, `--log` (progress events on stderr), `--out`
  (solution values to a file), and `--solver milp|highs`.
- `highs` cargo feature enabling `--solver highs` (needs a C++ toolchain).
