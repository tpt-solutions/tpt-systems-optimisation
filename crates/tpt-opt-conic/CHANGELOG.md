# Changelog

All notable changes to `tpt-opt-conic` are documented here. This crate follows
the workspace convention of per-crate changelogs (Keep a Changelog format),
starting from the initial `0.1.0` scaffold.

## [0.1.0] - Unreleased

### Added
- Conic-program form `ConeProgram` with decision variables, linear objective,
  equality rows, second-order-cone rows (`SocRow`), and semidefinite blocks
  (`SdpBlock`).
- Kelley cutting-plane (outer-approximation) solver `solve_conic` /
  `solve_socp` over the canonical LP engine (`tpt_opt_milp::MilpSolver`),
  returning a `ConeSolution` with status, primal point, objective, and maximum
  cone-constraint violation.
- SOCP supporting-hyperplane cuts `r(x) ≥ (q/‖q‖)ᵀ q(x)` with a zero-norm guard
  that separates via the necessary condition `r(x) ≥ 0`.
- SDP eigenvector cuts `⟨v vᵀ, X(x)⟩ ≥ 0` at the most-negative eigenvalue, with
  symmetric eigendecomposition by the cyclic Jacobi method (`jacobi_eigen`).
- Necessary-condition seeding `r(x) ≥ 0` for every SOC row so the LP
  relaxation stays bounded before the first cut.
- Status enum `ConicStatus` (Optimal / Infeasible / MaxIterations).
- Unit tests: a maximisation SOCP quarter-disk instance, an infeasible SOC, an
  SDP PSD-cutting-plane instance (`[[1, x], [x, 1]] ⪰ 0` → `|x| ≤ 1`), and a
  Jacobi eigendecomposition sanity check.

### Fixed
- `SocRow::eval` now includes the `q_rhs` constant term, so `r(x)` and `q(x)`
  are evaluated correctly (previously the constant was dropped, breaking
  feasibility and cut generation).
- SDP cut sign corrected to the valid `⟨v vᵀ, X(x)⟩ ≥ 0` (was inverted to `≤ 0`,
  which excluded the true feasible region).
- Reported objective value is always `cᵀx` (the previous maximisation negation
  returned the negated value).
- API binding against current `tpt-opt-core` / `tpt-opt-milp`:
  `Constraint::equality`/`Constraint::le`, `Solver` trait in scope for `solve`.

## [Unreleased]

### Not yet implemented (see `todo.md`)
- Trust-region / analytic-centre regularisation to reduce cutting-plane
  tailing-off (the method converges but can need many rounds on stiff cones).
- Direct SDP block modelling of robust ellipsoidal uncertainty sets feeding
  `tpt-opt-robust` (the crate already builds and solves the SDP relaxation).
