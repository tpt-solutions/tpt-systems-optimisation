# Changelog

All notable changes to this crate are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Initial release: MPS/CPLEX-LP model ingestion via `tpt-opt-milp::format`,
  solving through `MilpSolver` with time-limit/thread/seed/cut options,
  `--export` format conversion, quiet mode, and stable exit codes.