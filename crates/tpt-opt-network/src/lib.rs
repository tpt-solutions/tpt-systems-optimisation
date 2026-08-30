//! Network-flow and graph-based optimisation algorithms.
//!
//! `tpt-opt-network` provides self-contained solvers for:
//!
//! - **Min-cost flow** — successive shortest path (primary) and a
//!   negative-cycle-canceling *network simplex* variant, both operating on the
//!   directed [`tpt_opt_core::graph::Graph`] with per-edge capacity/cost.
//! - **Assignment / matching** — the Hungarian (Kuhn–Munkres) algorithm for the
//!   square assignment problem.
//! - **Optimal power flow** — DC-OPF (linearised, solved as an LP via an
//!   internal two-phase simplex implementing [`tpt_opt_core::solver::Solver`]),
//!   AC-OPF (polar coordinates, solved as a nonlinear program via
//!   `tpt_opt_core::nlp::solve_nlp`), and security-constrained OPF
//!   (DC-OPF with N-1 contingency constraints using Line Outage Distribution
//!   Factors).
//! - **Graph preprocessing** — cycle detection, bridge identification,
//!   biconnected-component decomposition, and a series-parallel reduction
//!   check (K₄-minor-free detection via parallel/series/dangling reductions).
//! - **Dynamic networks** — period-by-period min-cost flow with warm-starting
//!   from the previous period's flow.
//!
//! All numerical comparisons thread [`tpt_opt_core::tolerance::Tolerances`]
//! through the solvers, and numerical failures surface as
//! [`tpt_opt_core::solver::SolverStatus::NumericalIssue`].
//!
//! # Example
//!
//! Route 4 units from node 0 to node 3 at minimum cost:
//!
//! ```
//! use tpt_opt_core::graph::{Edge, Graph};
//! use tpt_opt_core::solver::SolverStatus;
//! use tpt_opt_network::min_cost_flow;
//!
//! let mut g = Graph::new(4);
//! g.add_edge(Edge::new(0, 1, 3.0, 2.0)); // cap, cost
//! g.add_edge(Edge::new(0, 2, 2.0, 2.0));
//! g.add_edge(Edge::new(1, 3, 2.0, 3.0));
//! g.add_edge(Edge::new(2, 3, 3.0, 1.0));
//!
//! let result = min_cost_flow(&g, &[4.0, 0.0, 0.0, -4.0]);
//! assert_eq!(result.status, SolverStatus::Optimal);
//! // Optimal routing: 2 units via 0->1->3 and 2 units via 0->2->3.
//! assert!((result.total_cost - 16.0).abs() < 1e-6);
//! ```

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
pub use graph_preprocess::{
    biconnected_components, bridges, has_cycle, series_parallel_check, SeriesParallelReport,
};
pub use min_cost_flow::{min_cost_flow, network_simplex, MinCostFlowResult, NetworkFlow};
pub use opf::{
    ac_opf, dc_opf, sc_opf, AcOpfResult, Bus, DcOpfResult, Generator, Line, Network, ScOpfResult,
};

pub use lp::LpSolver;
