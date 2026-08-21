//! Network-flow and graph-based optimisation algorithms.
//!
//! `tpt-opt-network` provides self-contained solvers for:
//!
//! - **Min-cost flow** — successive shortest path (primary) and a
//!   negative-cycle-canceling *network simplex* variant, both operating on the
//!   directed [`tpt_math_graph::Graph`] with per-edge capacity/cost.
//! - **Assignment / matching** — the Hungarian (Kuhn–Munkres) algorithm for the
//!   square assignment problem.
//! - **Optimal power flow** — DC-OPF (linearised, solved as an LP via an
//!   internal two-phase simplex implementing [`tpt_opt_core::solver::Solver`]),
//!   AC-OPF (polar coordinates, solved as a nonlinear program via
//!   `tpt_math_optimize_general::solve_nlp`), and security-constrained OPF
//!   (DC-OPF with N-1 contingency constraints using Line Outage Distribution
//!   Factors).
//! - **Graph preprocessing** — cycle detection, bridge identification and
//!   biconnected-component decomposition.
//! - **Dynamic networks** — period-by-period min-cost flow with warm-starting
//!   from the previous period's flow.
//!
//! All numerical comparisons thread [`tpt_opt_core::tolerance::Tolerances`]
//! through the solvers, and numerical failures surface as
//! [`tpt_opt_core::solver::SolverStatus::NumericalIssue`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod assignment;
pub mod dynamic;
pub mod graph_preprocess;
pub mod min_cost_flow;
pub mod opf;

mod lp;

pub use assignment::{hungarian, AssignmentResult};
pub use dynamic::{DynamicNetwork, DynamicNetworkResult};
pub use graph_preprocess::{biconnected_components, bridges, has_cycle};
pub use min_cost_flow::{min_cost_flow, network_simplex, MinCostFlowResult, NetworkFlow};
pub use opf::{
    ac_opf, dc_opf, sc_opf, Bus, DcOpfResult, Generator, Line, Network, ScOpfResult,
};

pub use lp::LpSolver;
