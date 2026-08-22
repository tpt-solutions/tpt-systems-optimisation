//! Convenience builders for the most common solver entry points.
//!
//! [`MilpBuilder`] (feature `milp`) assembles a canonical [`Model`] with a
//! fluent API and solves it with [`MilpSolver`] in one call.
//! [`NetworkFlowBuilder`] (feature `network`) assembles a directed graph with
//! capacities, costs and node balances and runs the min-cost-flow solver.

#[cfg(feature = "milp")]
use tpt_opt_core::{Constraint, Model, Objective, Solution, SolveParameters, Solver, VarBound};

#[cfg(feature = "network")]
use tpt_opt_core::SolverStatus;

use crate::OptimizationError;

/// Fluent builder for mixed-integer linear programs solved by
/// [`MilpSolver`](tpt_opt_milp::MilpSolver).
///
/// Variable adders return the new variable's index and are meant to be called
/// as statements; row/objective setters and parameter tweaks chain fluently.
///
/// # Example
///
/// ```
/// use tpt_opt_systems::MilpBuilder;
///
/// // max 3x + 5y s.t. x + y <= 4, x,y integer in [0,4] → optimum 20 at
/// // (0, 4).
/// let mut b = MilpBuilder::new(0);
/// let x = b.add_integer(0.0, 4.0);
/// let y = b.add_integer(0.0, 4.0);
/// let sol = b
///     .le(&[x, y], &[1.0, 1.0], 4.0)
///     .maximize(&[x, y], &[3.0, 5.0])
///     .solve()
///     .unwrap();
/// assert!((sol.objective_value - 20.0).abs() < 1e-6);
/// ```
#[cfg(feature = "milp")]
#[derive(Debug, Clone)]
pub struct MilpBuilder {
    model: Model,
    params: SolveParameters,
}

#[cfg(feature = "milp")]
impl MilpBuilder {
    /// Start a model over `num_vars` free continuous variables.
    pub fn new(num_vars: usize) -> Self {
        Self { model: Model::new(num_vars), params: SolveParameters::defaults() }
    }

    /// Append a variable with an explicit bound; returns its index.
    pub fn add_variable(&mut self, bound: VarBound) -> usize {
        self.model.add_variable(bound)
    }

    /// Append a binary variable; returns its index.
    pub fn add_binary(&mut self) -> usize {
        self.model.add_variable(VarBound::binary())
    }

    /// Append an integer variable in `[lo, hi]`; returns its index.
    pub fn add_integer(&mut self, lo: f64, hi: f64) -> usize {
        self.model.add_variable(VarBound::integer(lo, hi))
    }

    /// Append a continuous variable in `[lo, hi]`; returns its index.
    pub fn add_continuous(&mut self, lo: f64, hi: f64) -> usize {
        self.model.add_variable(VarBound::continuous(lo, hi))
    }

    /// Add `sum coeffs[i] * x[indices[i]] <= rhs`.
    pub fn le(mut self, indices: &[usize], coeffs: &[f64], rhs: f64) -> Self {
        self.model.add_constraint(Constraint::le(indices.to_vec(), coeffs.to_vec(), rhs));
        self
    }

    /// Add `sum coeffs[i] * x[indices[i]] >= rhs`.
    pub fn ge(mut self, indices: &[usize], coeffs: &[f64], rhs: f64) -> Self {
        self.model.add_constraint(Constraint::ge(indices.to_vec(), coeffs.to_vec(), rhs));
        self
    }

    /// Add `sum coeffs[i] * x[indices[i]] == rhs`.
    pub fn eq(mut self, indices: &[usize], coeffs: &[f64], rhs: f64) -> Self {
        self.model.add_constraint(Constraint::equality(indices.to_vec(), coeffs.to_vec(), rhs));
        self
    }

    /// Set a minimisation objective `sum coeffs[i] * x[indices[i]]`.
    pub fn minimize(mut self, indices: &[usize], coeffs: &[f64]) -> Self {
        self.model.set_objective(Objective::minimize(indices.to_vec(), coeffs.to_vec()));
        self
    }

    /// Set a maximisation objective `sum coeffs[i] * x[indices[i]]`.
    pub fn maximize(mut self, indices: &[usize], coeffs: &[f64]) -> Self {
        self.model.set_objective(Objective::maximize(indices.to_vec(), coeffs.to_vec()));
        self
    }

    /// Replace the whole objective (sense, terms and constant).
    pub fn set_objective(mut self, objective: Objective) -> Self {
        self.model.set_objective(objective);
        self
    }

    /// Replace the parameter bundle wholesale.
    pub fn with_params(mut self, params: SolveParameters) -> Self {
        self.params = params;
        self
    }

    /// Set the wall-clock time limit (seconds).
    pub fn with_time_limit(mut self, seconds: f64) -> Self {
        self.params.time_limit = Some(seconds);
        self
    }

    /// Set the worker thread count for parallel tree search.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.params.threads = threads;
        self
    }

    /// Set the deterministic seed for branching/heuristics.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.params.seed = Some(seed);
        self
    }

    /// Set absolute/relative optimality gap tolerances.
    pub fn with_gap(mut self, absolute: f64, relative: f64) -> Self {
        self.params.absolute_gap = absolute;
        self.params.relative_gap = relative;
        self
    }

    /// The assembled canonical model (consumes the builder).
    pub fn build_model(self) -> Model {
        self.model
    }

    /// Assemble the model and solve it with the bundled branch-and-bound
    /// engine, wrapping failures in [`OptimizationError`].
    pub fn solve(self) -> Result<Solution, OptimizationError> {
        let mut solver = tpt_opt_milp::MilpSolver::new();
        solver
            .set_parameter(&self.params)
            .map_err(|e| OptimizationError::solve("branch-and-bound", e))?;
        solver.solve(&self.model).map_err(|e| OptimizationError::solve("branch-and-bound", e))
    }
}

/// Fluent builder for min-cost flow instances solved by
/// [`min_cost_flow`](tpt_opt_network::min_cost_flow).
///
/// Edge/balance setters mutate in place (`add_edge` returns the edge index);
/// finish with [`NetworkFlowBuilder::solve`].
///
/// # Example
///
/// ```
/// use tpt_opt_systems::NetworkFlowBuilder;
///
/// let mut flow = NetworkFlowBuilder::new(2);
/// let e = flow.add_edge(0, 1, 5.0, 2.0);
/// assert_eq!(e, 0);
/// flow.supply(0, 3.0);
/// flow.demand(1, 3.0);
/// let result = flow.solve().unwrap();
/// assert!(result.status.has_solution());
/// assert!((result.total_cost - 6.0).abs() < 1e-6);
/// ```
#[cfg(feature = "network")]
#[derive(Debug, Clone)]
pub struct NetworkFlowBuilder {
    num_nodes: usize,
    edges: Vec<(usize, usize, f64, f64)>,
    balances: Vec<f64>,
}

#[cfg(feature = "network")]
impl NetworkFlowBuilder {
    /// Start a flow instance over `num_nodes` nodes (all balances zero).
    pub fn new(num_nodes: usize) -> Self {
        Self { num_nodes, edges: Vec::new(), balances: vec![0.0; num_nodes] }
    }

    /// Add a directed edge `from -> to` with `capacity` and per-unit `cost`;
    /// returns the edge index (flow results follow this order).
    pub fn add_edge(&mut self, from: usize, to: usize, capacity: f64, cost: f64) -> usize {
        self.edges.push((from, to, capacity, cost));
        self.edges.len() - 1
    }

    /// Set the net outflow balance of `node` (positive = supply).
    pub fn set_balance(&mut self, node: usize, balance: f64) -> &mut Self {
        self.balances[node] = balance;
        self
    }

    /// Declare `node` a source supplying `amount` units.
    pub fn supply(&mut self, node: usize, amount: f64) -> &mut Self {
        self.set_balance(node, amount)
    }

    /// Declare `node` a sink demanding `amount` units.
    pub fn demand(&mut self, node: usize, amount: f64) -> &mut Self {
        self.set_balance(node, -amount)
    }

    /// Build the graph and run the successive-shortest-path min-cost-flow
    /// solver, wrapping failures in [`OptimizationError`].
    pub fn solve(self) -> Result<tpt_opt_network::MinCostFlowResult, OptimizationError> {
        use tpt_math_graph::{Edge, Graph};
        let mut g = Graph::new(self.num_nodes);
        for (from, to, capacity, cost) in &self.edges {
            g.add_edge(Edge::new(*from, *to, *capacity, *cost));
        }
        let result = tpt_opt_network::min_cost_flow(&g, &self.balances);
        if result.status == SolverStatus::NumericalIssue {
            return Err(OptimizationError::no_solution("min-cost-flow", "numerical issue"));
        }
        Ok(result)
    }
}
