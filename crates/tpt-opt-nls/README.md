# tpt-opt-nls

Nonlinear least squares for the `tpt-systems-optimisation` workspace. Pure
Rust, no `unsafe`, zero external dependencies beyond `tpt-opt-core`.

## What is implemented

* **Solvers** — Gauss-Newton with backtracking line search, Levenberg–Marquardt
  with the Nielsen gain-ratio damping update, and Powell dogleg trust region.
  All three handle box constraints by projection with projected-Newton
  (active-set) step masking, and all three apply robust losses via IRLS.
* **Exact Jacobians** — residual blocks are closures over forward-mode dual
  numbers (`dual` module): one seeded evaluation produces the residual and its
  exact Jacobian rows. Finite differences remain available as a fallback
  (`JacobianMode::FiniteDifference`).
* **Reverse-mode autodiff** — a standalone value-graph (`reverse` module)
  giving scalar-output gradients in one backward pass; the right tool when
  parameters dwarf outputs. Iterative topological sort and iterative drop
  make deep graphs safe.
* **Robust losses** — Huber, Cauchy, Tukey biweight, soft-L1 (`loss` module),
  applied as IRLS row weights re-evaluated each outer iteration.
* **Sparse normal equations** — minimum-degree (AMD-core) fill-reducing
  ordering plus exact-fill up-looking sparse `LDLᵀ` (`sparse` module), used
  automatically when `LinearSolver::Sparse` is configured. Dense Cholesky/LU
  helpers live in `dense`.
* **Uncertainty** — Gaussian covariance `σ̂²(JᵀJ)⁻¹` and per-parameter
  standard errors from the final Jacobian.
* **Workspace integration** — termination criteria can be derived from
  `tpt_opt_core::Tolerances` (`NlsConfig::from_tolerances`); statuses map onto
  `tpt_opt_core::SolverStatus` (`NlsStatus::to_solver_status`).

## Quick start

```rust
use tpt_opt_nls::{levenberg_marquardt, NlsConfig, NlsProblem};

// Fit y = a·e^{b·t}.
let t = [0.0, 0.5, 1.0, 1.5];
let y = [2.0f64, 1.2131, 0.7358, 0.4463];
let problem = NlsProblem::builder(2)
    .residual_block(t.len(), move |x, out| {
        for i in 0..t.len() {
            out[i] = x[0].clone() * (x[1].clone() * t[i]).exp() - y[i];
        }
    })
    .build();
let result = levenberg_marquardt(&problem, &[1.0, 0.0], &NlsConfig::new()).unwrap();
assert_eq!(result.status, tpt_opt_nls::NlsStatus::Converged);
```

## Status

`0.1.0` — new crate (Phase 13b of the workspace todo). Reflective trust
regions for bounds, sparse Jacobian storage end-to-end, and second-order
autodiff are documented as future work in the workspace todo.
