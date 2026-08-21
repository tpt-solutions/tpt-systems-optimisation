//! Multi-objective / Pareto optimisation.
//!
//! Provides the building blocks for multi-objective optimisation:
//!
//! - Pareto dominance, Pareto-front extraction and the additive epsilon indicator
//!   ([`dominance`]).
//! - Objective normalisation for disparate scales ([`normalizer`]).
//! - Hypervolume computation (exact 2-D and the WFG algorithm for N-D)
//!   ([`hypervolume`]).
//! - [`nsga2`] — a self-contained NSGA-II evolutionary multi-objective solver.
//! - Linear scalarisation helpers ([`scalarize`]) building a
//!   [`tpt_opt_core::Model`] solved with [`tpt_opt_milp::MilpSolver`].

pub mod dominance;
pub mod hypervolume;
pub mod normalizer;
pub mod nsga2;
pub mod scalarize;

pub use dominance::{dominates, epsilon_indicator, pareto_front};
pub use hypervolume::hypervolume;
pub use normalizer::ObjectiveNormalizer;
pub use nsga2::{Nsga2, Nsga2Config};
pub use scalarize::{LinearMultiObjective, Scalarization};
