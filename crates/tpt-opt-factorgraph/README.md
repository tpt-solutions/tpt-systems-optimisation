# tpt-opt-factorgraph

Factor-graph least squares for SLAM-style problems, for the
`tpt-systems-optimisation` workspace. Pure Rust, no `unsafe`, no external
dependencies beyond `tpt-opt-core` and `tpt-opt-nls`.

## What is implemented

* **Manifold variables** (`manifold`) — Euclidean(n), SO(2), SE(2), SO(3)
  (unit quaternions, scalar last), SE(3) (`(t, q)` storage), with
  `compose` / `inverse` / `retract` / `local` and exact exponential/log maps
  (SE(2) V-matrix, SE(3) V and V⁻¹, quaternion exp/log with short-arc log).
* **Factors** — `PriorFactor`, `BetweenFactor` (odometry / loop closure,
  group-element measurements), `RangeBearingFactor` (SE(2) → landmark), and
  pinhole `ReprojectionFactor` (SE(3) camera + Euclidean(3) point). Custom
  factors implement the `Factor` trait (keys + dim + residual).
* **Noise models** (`noise`) — unit, isotropic, diagonal, and full
  Gaussian via a square-root information matrix; optional robust kernels
  (Huber/Cauchy/Tukey/soft-L1 reused from `tpt-opt-nls`) applied as IRLS
  weights on the whitened residual.
* **Solver** — Levenberg-Marquardt with numeric **on-manifold** Jacobians
  (central differences through `retract`), block-sparse normal equations,
  and a fill-reducing-ordered (minimum-degree) sparse LDLᵀ inner solve from
  `tpt-opt-nls::sparse`. Entry points: `solve` (batch), `solve_schur`
  (Schur-complement elimination of a chosen point set with camera
  back-substitution), and `solve_incremental` (iSAM-style: only factors
  adjacent to changed keys are re-linearised; cached blocks ride along with
  first-order residual corrections).
* **Uncertainty** — `marginal::marginal_covariance`: the whitened Hessian
  is the Gaussian information matrix; per-key covariance blocks come from
  sparse solves against unit columns.
* **Two-stage robust workflow** — `FactorGraph::set_factor_robust` enables
  kernels after an L2 warm start (IRLS weights are meaningless from a
  drifted cold start; the tests demonstrate the pattern).

## Quick start

See the crate-level doctest and `tests/slam.rs` for a 2-D Manhattan-world
pose graph, an SE(3) chain, and a small bundle adjustment solved three ways
(batch, Schur, incremental).

## Status

`0.1.0` — new crate (Phase 13c of the workspace todo). A full Bayes-tree
iSAM2 with fluid reordering, analytic manifold Jacobians, and IMU
preintegration remain future work in the workspace todo.
