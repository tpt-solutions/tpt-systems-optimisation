//! `tpt-opt-milp`: a branch-and-bound / branch-and-cut mixed-integer linear
//! programming solver.
//!
//! The crate provides [`MilpSolver`], which implements
//! [`tpt_opt_core::solver::Solver<Model>`] and solves mixed-integer linear
//! programs by branch-and-bound over an LP relaxation
//! ([`lp`]). It includes most-fractional / pseudo-cost branching, best-bound /
//! depth-first node selection, rounding and feasibility-pump primal heuristics,
//! and optional Gomory mixed-integer root cuts ([`cuts`]).

pub mod cuts;
pub mod lp;
pub mod milp;

pub use cuts::add_gomory_cuts;
pub use lp::{solve_lp, solve_lp_state, LpSolution, LpState, LpStatus};
pub use milp::{BranchingRule, MilpSolver, NodeSelection};
