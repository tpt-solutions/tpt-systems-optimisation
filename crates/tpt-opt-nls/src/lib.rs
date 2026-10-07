//! Nonlinear least squares for the `tpt-systems-optimisation` workspace:
//! Gauss-Newton, Levenberg-Marquardt, and Powell dogleg solvers with exact
//! forward-mode autodiff Jacobians, a tape-free reverse-mode graph, robust
//! M-estimator losses, box constraints, and dense or fill-reducing-ordered
//! sparse normal equations.
//!
//! Residual blocks are written once as closures over dual numbers and get
//! an **exact** Jacobian in a single seeded evaluation — no finite-difference
//! truncation error. Status and tolerances integrate with
//! `tpt_opt_core` ([`NlsConfig::from_tolerances`],
//! [`NlsStatus::to_solver_status`]).
//!
//! The supporting math that the published `tpt-math-*` crates do not ship
//! (forward/reverse autodiff, dense Cholesky/LU, sparse minimum-degree
//! ordering + LDLᵀ) lives in this crate's [`dual`], [`reverse`], [`dense`],
//! and [`sparse`] modules and is reused by `tpt-opt-factorgraph`.
//!
//! # Example
//!
//! Fit `y = a·e^{b·t}` to data generated with `a = 2, b = −1`:
//!
//! ```rust
//! use tpt_opt_nls::{levenberg_marquardt, NlsConfig, NlsProblem};
//!
//! let t = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
//! let y: Vec<f64> = t.iter().map(|&ti| 2.0 * f64::exp(-ti)).collect(); // = 2·e^{−t}
//! let problem = NlsProblem::builder(2)
//!     .residual_block(t.len(), move |x, out| {
//!         for i in 0..t.len() {
//!             out[i] = x[0].clone() * (x[1].clone() * t[i]).exp() - y[i];
//!         }
//!     })
//!     .build();
//! let result = levenberg_marquardt(&problem, &[1.0, 0.0], &NlsConfig::new()).unwrap();
//! assert_eq!(result.status, tpt_opt_nls::NlsStatus::Converged);
//! assert!((result.x[0] - 2.0).abs() < 1e-6, "a = {}", result.x[0]);
//! assert!((result.x[1] + 1.0).abs() < 1e-6, "b = {}", result.x[1]);
//! ```

pub mod dense;
pub mod dual;
pub mod lbfgs;
pub mod loss;
pub mod problem;
pub mod reverse;
pub mod solvers;
pub mod sparse;
pub mod types;

pub use lbfgs::{lbfgs, LbfgsConfig, LbfgsResult, Objective};
pub use loss::Loss;
pub use problem::{JacobianMode, LinearSolver, NlsProblem, ResidualFn};
pub use solvers::{gauss_newton, levenberg_marquardt, powell_dogleg};
pub use sparse::{Ordering, SparseError, SymbolicLdl, Triplet};
pub use types::{NlsConfig, NlsError, NlsResult, NlsStatus};
