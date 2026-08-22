//! `tpt-opt-milp`: a branch-and-bound / branch-and-cut mixed-integer linear
//! programming solver.
//!
//! The crate provides [`MilpSolver`], which implements
//! [`tpt_opt_core::solver::Solver<Model>`] and solves mixed-integer linear
//! programs by branch-and-bound over an LP relaxation ([`lp`]).
//!
//! Highlights:
//!
//! - Branching rules: most-fractional, pseudo-cost, and limited strong
//!   branching ([`milp::BranchingRule`]).
//! - Node selection: best-bound, depth-first, and best-estimate
//!   ([`milp::NodeSelection`]).
//! - Primal heuristics: rounding, feasibility pump, RINS, local branching.
//! - Cut suite: model-space clique cuts, cover inequalities with exact
//!   lifting, and mixed-integer rounding (MIR) cuts ([`cuts`]); plus
//!   tableau-space Gomory mixed-integer cuts and lift-and-project
//!   intersection cuts ([`gomory`]).
//! - Modelling extras: SOS1/SOS2 sets ([`sos`]), indicator constraints
//!   ([`indicator`]), and piecewise-linear objectives ([`piecewise`]).
//! - Deterministic parallel tree search via `.with_threads(n > 1)`.

pub mod cuts;
pub mod gomory;
pub mod indicator;
pub mod lp;
pub mod milp;
pub mod piecewise;
pub mod sos;

pub use cuts::{add_clique_cuts, add_cover_cuts, add_mir_cuts};
pub use gomory::{
    add_gomory_cuts, add_lift_and_project_cuts, gomory_cuts, lift_and_project_cuts, TableauCut,
};
pub use indicator::{IndicatorConstraint, Trigger};
pub use lp::{solve_lp, solve_lp_state, LpSolution, LpState, LpStatus};
pub use milp::{BranchingRule, MilpSolver, NodeSelection};
pub use piecewise::PiecewiseObjective;
pub use sos::{SosSet, SosType};
