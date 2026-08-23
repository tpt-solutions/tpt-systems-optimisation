//! # tpt-opt-systems — the umbrella crate for the tpt optimisation suite
//!
//! One dependency, one flat feature per solver family. Everything is
//! re-exported from a single namespace so downstream code depends only on
//! `tpt-opt-systems` instead of tracking eight individual crates.
//!
//! ## Feature matrix
//!
//! | Feature      | Crate                 | Contents |
//! |--------------|-----------------------|----------|
//! | *(none)*     | `tpt-opt-core`        | Canonical [`Model`], `Solver` trait, bounds, tolerances, errors (always available) |
//! | `milp`       | `tpt-opt-milp`        | Branch-and-bound/cut MILP: clique/cover/MIR/Gomory/lift-and-project cuts, heuristics, SOS/indicator/piecewise |
//! | `minlp`      | `tpt-opt-minlp`       | Outer approximation, generalized Benders, SQP branch-and-bound, McCormick/αBB relaxations, logical constraints |
//! | `network`    | `tpt-opt-network`     | Min-cost flow, network simplex, Hungarian assignment, DC/AC/SC-OPF, graph preprocessing, dynamic networks |
//! | `cp`         | `tpt-opt-cp`          | Constraint programming: propagation, AllDifferent/Cumulative/Circuit/Regular/Element globals, CBJ search |
//! | `heuristic`  | `tpt-opt-heuristic`   | Simulated annealing, genetic algorithms, tabu search, particle swarm |
//! | `multi`      | `tpt-opt-multi`       | NSGA-II/III, Pareto fronts, hypervolume, knee points, linear scalarisation |
//! | `robust`     | `tpt-opt-robust`      | Two-/multi-stage stochastic programming, SAA, VSS/EVPI, chance constraints, Bertsimas–Sim, DRO |
//! | `decompose`  | `tpt-opt-decompose`   | Benders, Dantzig–Wolfe + column generation, branch-and-price, Lagrangian relaxation, structure detection |
//! | `all-solvers`| *(meta)*              | Enables every solver family above |
//!
//! With **no features** the crate exposes only the always-on core surface
//! ([`core`], plus the flat core re-exports below), so an empty-feature build
//! compiles without any solver backend.
//!
//! ## Unified error handling
//!
//! Failures from every family are normalised into [`OptimizationError`],
//! which tags the producing algorithm and preserves the underlying
//! [`OptError`] when one exists.
//!
//! ## Convenience builders
//!
//! - `MilpBuilder` (`milp`) — fluent model assembly + solve in one call.
//! - `NetworkFlowBuilder` (`network`) — graph assembly + min-cost flow.
//!
//! ## Format conversion
//!
//! `convert::network_flow_to_milp` (`network` + `milp`) lowers a
//! min-cost-flow instance to a canonical MILP so any MILP backend can solve
//! it.
//!
//! ## Example
//!
//! ```toml
//! [dependencies]
//! tpt-opt-systems = { version = "0.1", features = ["milp", "network"] }
//! ```
//!
//! Each block below is feature-gated so the example also compiles (as a
//! no-op) on an empty-feature build.
//!
//! ```rust,no_run
//! #[cfg(feature = "milp")]
//! {
//!     // A tiny MILP through the builder…
//!     use tpt_opt_systems::MilpBuilder;
//!     let mut b = MilpBuilder::new(0);
//!     let x = b.add_integer(0.0, 10.0);
//!     let sol = b.ge(&[x], &[1.0], 3.0).minimize(&[x], &[2.0]).solve().unwrap();
//!     assert!((sol.objective_value - 6.0).abs() < 1e-6);
//! }
//!
//! #[cfg(feature = "network")]
//! {
//!     // …and a min-cost flow through its builder.
//!     use tpt_opt_systems::NetworkFlowBuilder;
//!     let mut flow = NetworkFlowBuilder::new(2);
//!     flow.add_edge(0, 1, 5.0, 2.0);
//!     flow.supply(0, 3.0);
//!     flow.demand(1, 3.0);
//!     let routed = flow.solve().unwrap();
//!     assert!(routed.status.has_solution());
//! }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

pub mod error;

#[cfg(any(feature = "milp", feature = "network"))]
pub mod builders;
#[cfg(all(feature = "network", feature = "milp"))]
pub mod convert;

pub use error::OptimizationError;

// ---- Always-on core surface ----------------------------------------------
/// The canonical problem representation and solver contract
/// (`tpt-opt-core`), available with no features enabled.
pub use tpt_opt_core as core;

pub use tpt_opt_core::{
    Bound, Constraint, InfeasibilityReport, Model, Objective, OptError, Sense, Solution,
    SolveParameters, SolverStatus, Tolerances, VarBound, VarType, Variable,
};

// ---- Per-family re-exports ------------------------------------------------

#[cfg(feature = "milp")]
pub use builders::MilpBuilder;
/// Mixed-integer linear programming (feature `milp`).
#[cfg(feature = "milp")]
pub use tpt_opt_milp as milp;
#[cfg(feature = "milp")]
pub use tpt_opt_milp::{
    add_clique_cuts, add_cover_cuts, add_gomory_cuts, add_lift_and_project_cuts, add_mir_cuts,
    solve_lp, BranchingRule, IndicatorConstraint, LpSolution, LpState, LpStatus, MilpSolver,
    NodeSelection, PiecewiseObjective, SosSet, SosType, Trigger,
};

/// Mixed-integer nonlinear programming (feature `minlp`).
#[cfg(feature = "minlp")]
pub use tpt_opt_minlp as minlp;
#[cfg(feature = "minlp")]
pub use tpt_opt_minlp::{MinlpModel, OaConfig, OaResult, OaStatus};

#[cfg(feature = "network")]
pub use builders::NetworkFlowBuilder;
/// Network flows, assignment and OPF (feature `network`).
#[cfg(all(feature = "network", feature = "milp"))]
pub use convert::network_flow_to_milp;
#[cfg(feature = "network")]
pub use tpt_math_graph as graph;
#[cfg(feature = "network")]
pub use tpt_opt_network as network;
#[cfg(feature = "network")]
pub use tpt_opt_network::{
    ac_opf, biconnected_components, bridges, dc_opf, has_cycle, hungarian, min_cost_flow,
    network_simplex, sc_opf, series_parallel_check, AcOpfResult, AssignmentResult, Bus,
    DcOpfResult, DynamicNetwork, DynamicNetworkResult, Generator, Line, MinCostFlowResult, Network,
    ScOpfResult, SeriesParallelReport,
};

/// Constraint programming (feature `cp`).
#[cfg(feature = "cp")]
pub use tpt_opt_cp as cp;
#[cfg(feature = "cp")]
pub use tpt_opt_cp::{model as cp_model, solver as cp_solver};

/// Metaheuristics (feature `heuristic`).
#[cfg(feature = "heuristic")]
pub use tpt_opt_heuristic as heuristic;
#[cfg(feature = "heuristic")]
pub use tpt_opt_heuristic::{
    CoolingSchedule, GeneticAlgorithm, HeuristicResult, ObjectiveFn, ParticleSwarmOptimization,
    SimulatedAnnealing, TabuSearch,
};

/// Multi-objective / Pareto optimisation (feature `multi`).
#[cfg(feature = "multi")]
pub use tpt_opt_multi as multi;
#[cfg(feature = "multi")]
pub use tpt_opt_multi::{
    cluster_solutions, das_dennis, dominates, epsilon_indicator, hypervolume, knee_point,
    pareto_front, tradeoff_ratios, LinearMultiObjective, Nsga2, Nsga2Config, Nsga3, Nsga3Config,
    ObjectiveNormalizer, Scalarization,
};

/// Optimisation under uncertainty (feature `robust`).
#[cfg(feature = "robust")]
pub use tpt_opt_robust as robust;
#[cfg(feature = "robust")]
pub use tpt_opt_robust::{
    budgeted_reformulation, evpi, gaussian_chance_row, multi_stage_model, normal_quantile,
    scenario_chance_model, solve_two_stage, vss, worst_case_box_expectation,
    worst_case_linear_wasserstein, DroCuttingPlane, SaaConfig, SaaResult, SaaSolver, Scenario,
    StageData, TwoStageProblem, TwoStageSolution, UncertainRow, ValueMetrics,
};

/// Large-scale decomposition methods (feature `decompose`).
#[cfg(feature = "decompose")]
pub use tpt_opt_decompose as decompose;
#[cfg(feature = "decompose")]
pub use tpt_opt_decompose::{
    detect_structure, BendersProblem, BendersResult, BendersSolver, BlockRow, BpResult,
    BranchAndPrice, Column, DantzigWolfe, DualConfig, DualResult, DwBlock, DwLocalRow, DwProblem,
    DwResult, LpPricer, Pricer, RecourseBlock, RmpPool, RowSense, Stabilization, Strategy,
    StructureReport,
};
