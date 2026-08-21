//! Canonical optimisation problem representation and the `Solver` trait contract.
//!
//! `tpt-opt-core` is the foundation layer of the `tpt-systems-optimisation`
//! workspace. It defines a *linear* canonical form — [`model::Model`],
//! [`model::Variable`], [`model::Constraint`], [`model::Objective`] — along with
//! variable bound kinds ([`bounds`]), numeric tolerances ([`tolerance`]),
//! structured errors with infeasibility diagnostics ([`error`]), the solver
//! agnosticism contract ([`solver`]), sparse constraint-matrix assembly
//! ([`matrix`]), and the extensibility hook for user constraints ([`custom`]).
//!
//! The crate is `no_std` compatible with an optional `alloc` feature (enabled by
//! the default `std` feature). Heap-backed types ([`error::OptError`],
//! [`solver::Solution`], …) are gated behind `alloc`; scalar types
//! ([`tolerance::Tolerances`], [`bounds::VarBound`], [`solver::SolveParameters`])
//! are always available.
//!
//! # Design principles (spec §4)
//!
//! - **Solver Agnosticism** — every solver crate implements
//!   [`solver::Solver<M>`] with a consistent `solve` / `set_parameter` /
//!   `warm_start` / `status` / `solution` signature.
//! - **Reproducibility** — [`solver::SolveParameters::with_seed`] threads a
//!   deterministic seed through branching and heuristics.
//! - **Numerical Stability** — all tolerances live in
//!   [`tolerance::Tolerances`] and are configurable.
//! - **Extensibility** — [`custom::CustomConstraint`] lets users plug novel
//!   constraints into any solver.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod bounds;
pub mod custom;
pub mod error;
pub mod matrix;
pub mod model;
pub mod solver;
pub mod tolerance;

pub use bounds::{Bound, VarBound, VarType};
pub use custom::CustomConstraint;
pub use error::{InfeasibilityReport, OptError};
pub use matrix::{model_to_csc, model_to_csr, ConstraintMatrix};
pub use model::{Constraint, Model, Objective, Sense, Variable};
pub use solver::{Solution, SolveParameters, Solver, SolverStatus, Verbosity, WarmStart};
pub use tolerance::Tolerances;
