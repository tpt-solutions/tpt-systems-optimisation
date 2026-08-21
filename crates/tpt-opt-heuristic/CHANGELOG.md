# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to semantic versioning.

## [0.1.0] - Unreleased

### Added
- Simulated annealing with geometric, adaptive, and reheating cooling schedules
  and configurable `Neighborhood` trait objects / closures.
- Genetic algorithms over continuous (`Vec<f64>`) and permutation (`Vec<usize>`)
  genomes, with single-point / two-point / uniform / order-based crossover,
  bit-flip / flip / swap / inversion / scramble mutation, and tournament /
  roulette / rank selection.
- Tabu search with adaptive tenure, aspiration criteria, and
  diversification / intensification.
- Particle swarm optimization with inertia-weight adaptation and global / local
  ring / Von Neumann topologies.
- Deterministic seeding via `tpt-math-prob`'s `Xoshiro256` (`with_seed`).
- Convergence-history tracking retrievable per run.
- Bridge to `tpt-opt-core`: results as `Solution`, config errors as `OptError`,
  status as `SolverStatus`; `SimulatedAnnealing` implements `Solver<Model>`.
