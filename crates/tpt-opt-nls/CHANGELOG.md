# Changelog

All notable changes to `tpt-opt-nls` are documented here. This crate follows
the workspace convention of per-crate changelogs (Keep a Changelog format),
starting from the initial `0.1.0` scaffold.

## [0.1.0] - Unreleased

### Added
- `NlsProblem` builder with residual blocks written as closures over forward
  mode dual numbers; exact Jacobians in one seeded evaluation, with central
  finite differences as a fallback (`JacobianMode`).
- Gauss-Newton (backtracking line search), Levenberg-Marquardt (Nielsen
  gain-ratio damping), and Powell dogleg (trust region) solvers sharing one
  linearisation core.
- Box constraints by projection with projected-Newton active-set step
  masking.
- Robust losses (Huber, Cauchy, Tukey, soft-L1) applied as IRLS weights with
  like-for-like robust cost acceptance.
- Standalone reverse-mode autodiff value graph (`reverse`): operator-style
  expressions, one-backward-pass gradients, iterative topological sort and
  iterative drop for deep graphs.
- Sparse factorisation stack (`sparse`): greedy minimum-degree ordering
  (exact-degree AMD core) over the quotient graph, exact-fill symbolic
  analysis, and up-looking sparse `LDLᵀ` with permutation-aware triangular
  solves; scatter-add triplet input with one-sided (mirror-ignoring) value
  semantics. Used by `LinearSolver::Sparse` with automatic re-analysis when
  the structural pattern shifts.
- Dense helpers (`dense`): LU with partial pivoting, Cholesky, SPD inverse,
  (weighted) normal equations.
- Gaussian covariance estimate `σ̂²(JᵀJ)⁻¹` and standard errors on
  `NlsResult`.
- `NlsConfig::from_tolerances` mapping the workspace `Tolerances` bundle onto
  solver termination criteria; `NlsStatus::to_solver_status` mapping onto the
  canonical `SolverStatus`.
- Tests: Rosenbrock and Powell singular across all three solvers, Huber
  optimum verified analytically, bound-constrained Rosenbrock at the wall,
  sparse-vs-dense agreement, grid-graph fill reduction, random-SPD solves
  under both orderings, reverse-mode gradient/Jacobian checks against central
  differences, and a 50 000-deep graph safety test.
