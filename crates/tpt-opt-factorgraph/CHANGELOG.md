# Changelog

All notable changes to `tpt-opt-factorgraph` are documented here. This crate
follows the workspace convention of per-crate changelogs (Keep a Changelog
format), starting from the initial `0.1.0` scaffold.

## [0.1.0] - Unreleased

### Added
- `Manifold` variable types: Euclidean(n), SO(2), SE(2), SO(3) (quaternion,
  scalar last), SE(3) (`(t, q)`), with compose/inverse/retract/local, exact
  SE(2) V-matrix exp/log, SE(3) twist exp/log (V and V⁻¹), and short-arc
  quaternion log.
- `Factor` trait plus standard factors: prior, between (group-element
  measurements), 2-D range-bearing, and pinhole reprojection.
- `KeyedValues` insertion-ordered variable storage and the `FactorGraph`
  container with per-factor noise models, robust kernels, and
  linearisation caches.
- Noise models: unit, isotropic, diagonal, full Gaussian via square-root
  information; robust kernels (Huber/Cauchy/Tukey/soft-L1 from
  `tpt-opt-nls`) applied as IRLS weights on whitened residuals.
- Levenberg-Marquardt solver with numeric on-manifold Jacobians (central
  differences through `retract`), block-sparse assembly, minimum-degree
  fill-reducing ordering, and sparse LDLᵀ inner steps.
- `solve_schur`: Schur-complement elimination of a point set (damped point
  blocks inverted densely, reduced camera system solved sparsely, points
  recovered by back-substitution).
- `solve_incremental`: iSAM-style selective re-linearisation with cached
  Jacobian blocks and first-order residual correction between solves.
- `marginal_covariance`: per-key Gaussian covariance blocks from sparse
  unit-column solves on the information matrix.
- `set_factor_robust` for the two-stage robust workflow.
- Tests: 2-D Manhattan pose graph (clean + corrupted-closure robust), SE(3)
  chain with loop closure, bundle adjustment, Schur-vs-batch and
  incremental-vs-batch agreement, analytic marginal covariance, and
  manifold round-trip property tests.
