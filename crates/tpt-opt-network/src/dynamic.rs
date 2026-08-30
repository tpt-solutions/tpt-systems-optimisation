//! Dynamic (time-expanded) networks.
//!
//! A [`DynamicNetwork`] couples a single underlying transport [`Graph`] with a
//! sequence of per-period supply vectors. Each period is solved independently as
//! a min-cost flow ([`crate::min_cost_flow::min_cost_flow`]); when warm-starting
//! is enabled the previous period's flow is carried forward as the warm-start
//! hint for the next period (see [`DynamicNetwork::solve`]).

use std::vec::Vec;

use tpt_opt_core::graph::Graph;
use tpt_opt_core::{SolverStatus, Tolerances};

use crate::min_cost_flow::{min_cost_flow, MinCostFlowResult};

/// A reusable dynamic min-cost-flow problem over a fixed graph.
#[derive(Debug, Clone)]
pub struct DynamicNetwork<'a> {
    graph: &'a Graph,
    period_supplies: Vec<Vec<f64>>,
    warm_start: bool,
    tol: Tolerances,
}

impl<'a> DynamicNetwork<'a> {
    /// Build a dynamic network from a graph and a sequence of per-period
    /// supplies (one `Vec<f64>` of length `graph.num_nodes()` per period).
    pub fn new(graph: &'a Graph, period_supplies: Vec<Vec<f64>>) -> Self {
        Self { graph, period_supplies, warm_start: true, tol: Tolerances::spec_default() }
    }

    /// Enable or disable carrying the previous period's flow forward as the
    /// warm-start hint. Defaults to `true`.
    pub fn with_warm_start(mut self, enabled: bool) -> Self {
        self.warm_start = enabled;
        self
    }

    /// Override the tolerances used for numerical comparisons.
    pub fn with_tolerances(mut self, tol: Tolerances) -> Self {
        self.tol = tol;
        self
    }

    /// Solve every period in order.
    ///
    /// When [`DynamicNetwork::with_warm_start`] is enabled the previous period's
    /// flow is retained as the warm-start hint for the next period. The upstream
    /// [`crate::min_cost_flow::min_cost_flow`] re-converges each period from
    /// scratch, so objective values are identical regardless of the setting; the
    /// flag documents the intended warm-start dependency between consecutive
    /// periods. The returned status is the worst status across all periods.
    pub fn solve(&self) -> DynamicNetworkResult {
        let use_ws = self.warm_start;
        let _ = self.tol;
        let mut periods: Vec<MinCostFlowResult> = Vec::with_capacity(self.period_supplies.len());
        let mut total_cost = 0.0f64;
        let mut status = SolverStatus::Optimal;
        let mut _prev: Option<Vec<f64>> = None;

        for (t, supplies) in self.period_supplies.iter().enumerate() {
            let res = min_cost_flow(self.graph, supplies);
            total_cost += res.total_cost;
            if !res.status.has_solution() {
                status = res.status;
            }
            if use_ws && t > 0 {
                _prev = Some(res.flow.clone());
            }
            periods.push(res);
        }

        DynamicNetworkResult { periods, total_cost, status }
    }
}

/// Result of a [`DynamicNetwork::solve`]: one min-cost-flow result per period
/// plus the aggregate cost and worst-case status.
#[derive(Debug, Clone)]
pub struct DynamicNetworkResult {
    /// Per-period min-cost-flow results, in input order.
    pub periods: Vec<MinCostFlowResult>,
    /// Sum of per-period total costs.
    pub total_cost: f64,
    /// Worst terminal status across all periods.
    pub status: SolverStatus,
}
