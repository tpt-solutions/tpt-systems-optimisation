//! `tpt-opt-decompose`: large-scale decomposition methods for linear and
//! mixed-integer programs.
//!
//! The crate exposes four families of algorithms, all built on
//! [`tpt_opt_core::Model`] and solved with the bundled LP/MILP engine
//! ([`tpt_opt_milp`]):
//!
//! - **Benders decomposition** ([`benders`]) for two-stage problems with
//!   (mixed-)integer first-stage variables: explicit dual-LP cut
//!   generation, Farkas feasibility cuts, optional Magnanti–Wong
//!   Pareto-optimal cuts, and trust-region / level-set stabilisation.
//! - **Dantzig–Wolfe decomposition & column generation** ([`dantzig_wolfe`])
//!   for block-angular programs: restricted-master management
//!   ([`dantzig_wolfe::RmpPool`]), big-M artificial seeding, and pricing
//!   over block polyhedra.
//! - **Branch-and-price** ([`branch_price`]): column generation embedded in
//!   depth-first branch-and-bound over integer master variables, with a
//!   pluggable [`branch_price::Pricer`] (integer knapsack pricing for
//!   cutting-stock-style masters) and a continuous-LP default.
//! - **Lagrangian relaxation** ([`lagrangian`]): subgradient ascent
//!   (Polyak / diminishing steps), a cutting-plane bundle/level method, and
//!   surrogate-relaxation search.
//! - **Structure detection** ([`structure`]): bipartite row–column analysis
//!   that finds independent blocks, linking rows/columns, and recommends a
//!   strategy.
//!
//! # Example
//!
//! Solve a two-stage problem by Benders: choose capacity `x ∈ [0, 4]`
//! (integer) at cost 2 per unit, then serve demand `d = 3` at cost 5 per
//! unit of unmet-capacity penalty `y ≥ d − x`:
//!
//! ```
//! use tpt_opt_decompose::{BendersProblem, BendersSolver, BlockRow, RecourseBlock, RowSense};
//!
//! let problem = BendersProblem {
//!     first_cost: vec![2.0],
//!     first_bounds: vec![(0.0, 4.0)],
//!     first_integer: vec![true],
//!     blocks: vec![RecourseBlock {
//!         cost: vec![5.0],
//!         rows: vec![BlockRow { y: vec![1.0], x: vec![1.0], sense: RowSense::Ge, rhs: 3.0 }],
//!         y_upper: vec![f64::INFINITY],
//!     }],
//!     weights: vec![1.0],
//! };
//! let result = BendersSolver::new(&problem).solve().unwrap();
//! assert!((result.objective - 2.0 * 3.0).abs() < 1e-6); // x = 3, y = 0
//! assert_eq!(result.status, tpt_opt_core::SolverStatus::Optimal);
//! ```

pub mod benders;
pub mod branch_price;
pub mod common;
pub mod dantzig_wolfe;
pub mod lagrangian;
pub mod structure;

pub use benders::{
    BendersProblem, BendersResult, BendersSolver, BlockRow, RecourseBlock, Stabilization,
};
pub use branch_price::{BpResult, BranchAndPrice, LpPricer, Pricer};
pub use common::RowSense;
pub use dantzig_wolfe::{Column, DantzigWolfe, DwBlock, DwLocalRow, DwProblem, DwResult, RmpPool};
pub use lagrangian::{DualConfig, DualResult};
pub use structure::{detect_structure, Strategy, StructureReport};
