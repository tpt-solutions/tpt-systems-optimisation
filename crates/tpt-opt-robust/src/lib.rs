//! Optimisation under uncertainty.
//!
//! `tpt-opt-robust` collects the standard decision-making frameworks for
//! problems whose data is uncertain:
//!
//! - **Scenario-based stochastic programming** — two-stage and multi-stage
//!   extensive forms with non-anticipativity via scenario-tree prefix
//!   merging, solved as MILPs with [`tpt_opt_milp::MilpSolver`]
//!   ([`scenario`]).
//! - **Sample average approximation (SAA)** — replication-based statistical
//!   lower/upper bounds and optimality-gap confidence intervals
//!   ([`saa`]).
//! - **Value of the stochastic solution** — VSS and EVPI computed from the
//!   recourse, wait-and-see and expected-value solutions ([`value`]).
//! - **Chance constraints** — scenario (binary-indicator) approximations and
//!   Gaussian deterministic equivalents (exact for diagonal covariance,
//!   conservative column-norm reformulation in general) ([`chance`]).
//! - **Adjustable robust optimisation** — Bertsimas–Sim budgeted
//!   (Γ-robustness) coefficient uncertainty with an exact LP reformulation,
//!   plus a conservative ellipsoidal-set reformulation ([`robust`]).
//! - **Distributionally robust optimisation** — box/moment ambiguity sets
//!   (closed-form worst case and a cutting-plane decision solver) and
//!   Wasserstein-ball worst-case evaluation for linear losses ([`dro`]).
//!
//! All MILP-backed components build [`tpt_opt_core::Model`] instances and
//! solve them with [`tpt_opt_milp::MilpSolver`]; sampling flows through the
//! seedable [`tpt_math_prob::Xoshiro256`] RNG for reproducibility (spec §4).
//!
//! # Example
//!
//! Two-stage news-vendor problem with known analytic optimum:
//!
//! ```
//! use tpt_opt_robust::scenario::{RowSense, Scenario, StageData, StageRow, TwoStageProblem};
//!
//! // min 2x + 8·E[(d − x)⁺] over demands {2, 4} w.p. 1/2 each → RP* = 8 at x* = 4.
//! let mk = |d: f64| StageData {
//!     cost: vec![8.0],
//!     rows: vec![StageRow { w: vec![-1.0], t: vec![-1.0], h: -d, sense: RowSense::Le }],
//! };
//! let problem = TwoStageProblem {
//!     first_cost: vec![2.0],
//!     first_bounds: vec![(0.0, 10.0)],
//!     second_bounds: vec![(0.0, 10.0)],
//!     scenarios: vec![
//!         (Scenario { probability: 0.5, data: vec![] }, mk(2.0)),
//!         (Scenario { probability: 0.5, data: vec![] }, mk(4.0)),
//!     ],
//! };
//! let sol = problem.solve().unwrap();
//! assert!((sol.objective - 8.0).abs() < 1e-6);
//! assert!((sol.x[0] - 4.0).abs() < 1e-6);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod chance;
pub mod dro;
pub mod robust;
pub mod saa;
pub mod scenario;
pub mod value;

pub use chance::{gaussian_chance_row, normal_quantile, scenario_chance_model};
pub use dro::{worst_case_box_expectation, worst_case_linear_wasserstein, DroCuttingPlane};
pub use robust::{budgeted_reformulation, ellipsoid_reformulation, UncertainRow};
pub use saa::{SaaConfig, SaaResult, SaaSolver};
pub use scenario::{
    multi_stage_model, solve_two_stage, Scenario, StageData, TwoStageProblem, TwoStageSolution,
};
pub use value::{evpi, vss, ValueMetrics};
