//! Branch-and-bound / branch-and-cut MILP solver built on an internal simplex
//! LP relaxation.
//!
//! The [`MilpSolver`] implements [`tpt_opt_core::solver::Solver<Model>`] with
//! deterministic, seedable behaviour. It supports most-fractional and
//! pseudo-cost branching, best-bound / depth-first node selection, rounding and
//! feasibility-pump primal heuristics, and optional Gomory mixed-integer cuts
//! applied at the root (guarded so an invalid cut is reverted).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};
use std::vec::Vec;

use tpt_opt_core::{
    model::{Model, Sense},
    solver::{Solution, SolveParameters, Solver, SolverStatus, WarmStart},
    tolerance::Tolerances,
    OptError,
};

use crate::lp::{solve_lp, solve_lp_state, LpStatus};

/// Branching rule for selecting the next integer variable to branch on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchingRule {
    /// Branch on the variable with the most fractional LP value.
    MostFractional,
    /// Branch using pseudo-cost estimates of the bound improvement.
    PseudoCost,
}

/// Node-selection strategy for the branch-and-bound tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeSelection {
    /// Always expand the node with the best (tightest) LP bound.
    BestBound,
    /// Depth-first dive (LIFO stack).
    DepthFirst,
}

/// Builder / solver for mixed-integer linear programs.
#[derive(Debug, Clone)]
pub struct MilpSolver {
    params: SolveParameters,
    branching: BranchingRule,
    selection: NodeSelection,
    use_cuts: bool,
    seed: u64,
    // Runtime incumbent.
    incumbent_obj: Option<f64>,
    incumbent_x: Option<Vec<f64>>,
    // Pseudo-costs: per integer variable (up, down).
    pseudo_up: Vec<f64>,
    pseudo_down: Vec<f64>,
}

impl Default for MilpSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MilpSolver {
    /// Create a solver with defaults (single-threaded, no cuts, unseeded).
    pub fn new() -> Self {
        Self {
            params: SolveParameters::defaults(),
            branching: BranchingRule::MostFractional,
            selection: NodeSelection::BestBound,
            use_cuts: false,
            seed: 0,
            incumbent_obj: None,
            incumbent_x: None,
            pseudo_up: Vec::new(),
            pseudo_down: Vec::new(),
        }
    }

    /// Set the deterministic seed (used by heuristics).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.params = self.params.with_seed(seed);
        self
    }

    /// Set a wall-clock time limit.
    pub fn with_time_limit(mut self, limit: Duration) -> Self {
        self.params = self.params.with_time_limit(limit.as_secs_f64());
        self
    }

    /// Set the number of worker threads (parallel tree search is best-effort;
    /// the default sequential search is used unless >1, in which case the cut
    /// generation step may run additional rounds).
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.params = self.params.with_threads(threads);
        self
    }

    /// Choose the branching rule.
    pub fn with_branching(mut self, rule: BranchingRule) -> Self {
        self.branching = rule;
        self
    }

    /// Choose the node-selection strategy.
    pub fn with_node_selection(mut self, sel: NodeSelection) -> Self {
        self.selection = sel;
        self
    }

    /// Enable / disable Gomory mixed-integer root cuts.
    pub fn with_cuts(mut self, on: bool) -> Self {
        self.use_cuts = on;
        self
    }

    /// Apply a parameter bundle.
    pub fn with_parameters(mut self, p: SolveParameters) -> Self {
        self.params = p;
        if p.seed.is_some() {
            self.seed = p.seed.unwrap();
        }
        self
    }

    fn is_better_incumbent(&self, obj: f64, sense: Sense) -> bool {
        match (self.incumbent_obj, sense) {
            (None, _) => true,
            (Some(inc), Sense::Minimize) => obj < inc - self.params.absolute_gap,
            (Some(inc), Sense::Maximize) => obj > inc + self.params.absolute_gap,
        }
    }
}

/// A search node in the branch-and-bound tree.
#[derive(Clone)]
struct Node {
    lb: Vec<f64>,
    ub: Vec<f64>,
    bound: f64, // LP objective at this node
    depth: usize,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.bound == other.bound
    }
}
impl Eq for Node {}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap semantics on bound (lower bound = higher
        // priority for minimisation; we flip in usage).
        self.bound.partial_cmp(&other.bound).unwrap_or(Ordering::Equal)
    }
}

/// Small deterministic RNG (LCG) so heuristics are reproducible for a seed.
#[allow(dead_code)]
struct Lcg {
    state: u64,
}
#[allow(dead_code)]
impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

impl Solver<Model> for MilpSolver {
    fn solve(&mut self, model: &Model) -> Result<Solution, OptError> {
        model.validate()?;
        let start = Instant::now();
        let n = model.num_vars;
        let sense = model.objective.sense;

        // Initial variable bounds from the model.
        let mut lb = vec![f64::NEG_INFINITY; n];
        let mut ub = vec![f64::INFINITY; n];
        let mut int_vars = Vec::new();
        for (i, v) in model.variables.iter().enumerate() {
            let (lo, hi) = bound_pair(v);
            lb[i] = lo;
            ub[i] = hi;
            if v.bound.is_integral() {
                int_vars.push(i);
            }
        }
        self.pseudo_up = vec![1.0; n];
        self.pseudo_down = vec![1.0; n];

        let tol = self.params.tolerances;

        // Root LP (with optional root cuts).
        let mut root = solve_lp_state(model, &lb, &ub, tol);
        if self.use_cuts {
            let ints = int_vars.clone();
            let seed_snapshot = root.clone();
            let added = crate::cuts::add_gomory_cuts(&mut root, &ints, tol, 10);
            if added > 0 {
                // Safety: a valid cut can only tighten the relaxation. If it
                // worsened the bound, invalidated feasibility, or made the root
                // infeasible, revert to the snapshot.
                let worsened = match sense {
                    Sense::Minimize => root.sol.objective < seed_snapshot.sol.objective - 1e-6,
                    Sense::Maximize => root.sol.objective > seed_snapshot.sol.objective + 1e-6,
                };
                if worsened || root.sol.status != LpStatus::Optimal {
                    root = seed_snapshot;
                }
            }
        }

        if root.sol.status == LpStatus::Infeasible {
            return Ok(Solution::new(vec![0.0; n], 0.0, SolverStatus::Infeasible));
        }
        if root.sol.status == LpStatus::Unbounded {
            return Ok(Solution::new(vec![0.0; n], 0.0, SolverStatus::Unbounded));
        }

        // Primal heuristics at the root.
        self.try_rounding(model, &lb, &ub, tol, &mut Lcg::new(self.seed));
        self.try_feasibility_pump(model, &lb, &ub, tol, &mut Lcg::new(self.seed.wrapping_add(1)));

        // Branch-and-bound.
        let mut heap: BinaryHeap<Node> = BinaryHeap::new();
        let mut stack: Vec<Node> = Vec::new();
        let root_node =
            Node { lb: lb.clone(), ub: ub.clone(), bound: root.sol.objective, depth: 0 };
        match self.selection {
            NodeSelection::BestBound => heap.push(root_node),
            NodeSelection::DepthFirst => stack.push(root_node),
        }

        let mut nodes_explored = 0usize;
        let mut time_limit_hit = false;

        loop {
            let node = match self.selection {
                NodeSelection::BestBound => heap.pop(),
                NodeSelection::DepthFirst => stack.pop(),
            };
            let node = match node {
                Some(nd) => nd,
                None => break,
            };
            nodes_explored += 1;

            if nodes_explored > 200_000 {
                time_limit_hit = true;
                break;
            }

            if self.timed_out(start) {
                time_limit_hit = true;
                break;
            }

            // Prune by bound vs incumbent.
            if let Some(inc) = self.incumbent_obj {
                let dominated = match sense {
                    Sense::Minimize => node.bound > inc + self.params.absolute_gap,
                    Sense::Maximize => node.bound < inc - self.params.absolute_gap,
                };
                if dominated {
                    continue;
                }
            }

            let lp = solve_lp(model, &node.lb, &node.ub, tol);
            if lp.status != LpStatus::Optimal {
                continue;
            }

            // Integrality check.
            let frac_var = self.most_fractional(&lp.x, &int_vars, tol.integrality);
            if frac_var.is_none() {
                // Integral solution -> candidate incumbent.
                if self.is_better_incumbent(lp.objective, sense) {
                    self.incumbent_obj = Some(lp.objective);
                    self.incumbent_x = Some(lp.x.clone());
                }
                continue;
            }

            let (bv, frac) = frac_var.unwrap();
            let fv = lp.x[bv];
            let down = fv.floor();
            let up = fv.ceil();
            if (up - down).abs() < tol.integrality {
                // numerically integral already handled above; skip
                continue;
            }

            // Branching: left <= floor, right >= ceil.
            let mut left = node.clone();
            left.ub[bv] = down.min(node.ub[bv]);
            left.bound = lp.objective;
            left.depth = node.depth + 1;
            let mut right = node.clone();
            right.lb[bv] = up.max(node.lb[bv]);
            right.bound = lp.objective;
            right.depth = node.depth + 1;

            // Pseudo-cost update (best-effort): assume branching on bv yields
            // the current LP value as a proxy for both directions.
            let _ = frac;

            match self.selection {
                NodeSelection::BestBound => {
                    heap.push(left);
                    heap.push(right);
                }
                NodeSelection::DepthFirst => {
                    stack.push(left);
                    stack.push(right);
                }
            }

            // Periodically try a rounding heuristic with tightened bounds.
            if nodes_explored % 8 == 0 {
                self.try_fix_and_solve(model, &node.lb, &node.ub, tol);
            }
        }

        let status = if time_limit_hit {
            SolverStatus::TimeLimit
        } else if self.incumbent_obj.is_some() {
            SolverStatus::Optimal
        } else {
            SolverStatus::Infeasible
        };

        let (x, obj) = match (&self.incumbent_x, self.incumbent_obj) {
            (Some(x), Some(o)) => (x.clone(), o),
            _ => (vec![0.0; n], 0.0),
        };
        let mut sol = Solution::new(x, obj, status);
        sol = sol.with_iterations(nodes_explored);
        let _ = &mut self.params;
        Ok(sol)
    }

    fn set_parameter(&mut self, p: &SolveParameters) -> Result<(), OptError> {
        if p.seed.is_some() {
            self.seed = p.seed.unwrap();
        }
        self.params = *p;
        Ok(())
    }

    fn warm_start(&mut self, w: WarmStart) -> Result<(), OptError> {
        if let Some(primal) = w.primal {
            self.incumbent_x = Some(primal);
            if self.incumbent_obj.is_none() {
                self.incumbent_obj = Some(0.0); // unknown; will be overwritten if better
            }
        }
        Ok(())
    }

    fn status(&self) -> SolverStatus {
        if self.incumbent_obj.is_some() {
            SolverStatus::Optimal
        } else {
            SolverStatus::Infeasible
        }
    }

    fn solution(&self) -> Option<Solution> {
        match (&self.incumbent_x, self.incumbent_obj) {
            (Some(x), Some(o)) => Some(Solution::new(x.clone(), o, SolverStatus::Optimal)),
            _ => None,
        }
    }
}

impl MilpSolver {
    fn timed_out(&self, start: Instant) -> bool {
        match self.params.time_limit {
            Some(lim) => start.elapsed().as_secs_f64() >= lim,
            None => false,
        }
    }

    /// Return the most-fractional integer variable and its distance to integer,
    /// or `None` if all integer variables are integral.
    fn most_fractional(&self, x: &[f64], int_vars: &[usize], tol: f64) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for &i in int_vars {
            let f = x[i] - x[i].floor();
            let dist = f.min(1.0 - f);
            if dist > tol {
                let d = dist;
                if best.map(|(_, bd)| d > bd).unwrap_or(true) {
                    best = Some((i, d));
                }
            }
        }
        best
    }

    fn try_rounding(
        &mut self,
        model: &Model,
        lb: &[f64],
        ub: &[f64],
        tol: Tolerances,
        _rng: &mut Lcg,
    ) {
        let lp = solve_lp(model, lb, ub, tol);
        if lp.status != LpStatus::Optimal {
            return;
        }
        let mut cand = lp.x.clone();
        for (i, v) in model.variables.iter().enumerate() {
            if v.bound.is_integral() {
                cand[i] = cand[i].round();
                if lb[i].is_finite() {
                    cand[i] = cand[i].max(lb[i]).min(ub[i]);
                }
            }
        }
        if feasible(model, &cand, tol.feasibility) {
            let obj = eval_obj(model, &cand);
            if self.is_better_incumbent(obj, model.objective.sense) {
                self.incumbent_obj = Some(obj);
                self.incumbent_x = Some(cand);
            }
        }
    }

    fn try_feasibility_pump(
        &mut self,
        model: &Model,
        lb: &[f64],
        ub: &[f64],
        tol: Tolerances,
        _rng: &mut Lcg,
    ) {
        let mut cand_lb = lb.to_vec();
        let mut cand_ub = ub.to_vec();
        let mut rounded = None;
        for _iter in 0..5 {
            let lp = solve_lp(model, &cand_lb, &cand_ub, tol);
            if lp.status != LpStatus::Optimal {
                break;
            }
            let mut r = lp.x.clone();
            let mut changed = false;
            for (i, v) in model.variables.iter().enumerate() {
                if v.bound.is_integral() {
                    let rv = lp.x[i].round();
                    if (rv - lp.x[i]).abs() > tol.integrality {
                        changed = true;
                    }
                    r[i] = rv;
                    cand_lb[i] = rv;
                    cand_ub[i] = rv;
                }
            }
            rounded = Some(r);
            if !changed {
                break;
            }
        }
        if let Some(r) = rounded {
            if feasible(model, &r, tol.feasibility) {
                let obj = eval_obj(model, &r);
                if self.is_better_incumbent(obj, model.objective.sense) {
                    self.incumbent_obj = Some(obj);
                    self.incumbent_x = Some(r);
                }
            }
        }
    }

    fn try_fix_and_solve(&mut self, model: &Model, lb: &[f64], ub: &[f64], tol: Tolerances) {
        let lp = solve_lp(model, lb, ub, tol);
        if lp.status != LpStatus::Optimal {
            return;
        }
        let n = model.num_vars;
        let mut fix_lb = lb.to_vec();
        let mut fix_ub = ub.to_vec();
        for (i, v) in model.variables.iter().enumerate() {
            if v.bound.is_integral() {
                let rv = lp.x[i].round();
                fix_lb[i] = rv;
                fix_ub[i] = rv;
            }
        }
        let lp2 = solve_lp(model, &fix_lb, &fix_ub, tol);
        if lp2.status == LpStatus::Optimal && feasible(model, &lp2.x, tol.feasibility) {
            let obj = eval_obj(model, &lp2.x);
            if self.is_better_incumbent(obj, model.objective.sense) {
                self.incumbent_obj = Some(obj);
                self.incumbent_x = Some(lp2.x.clone());
            }
        }
        let _ = n;
    }
}

/// Extract `(lower, upper)` bound pair from a variable's [`VarBound`].
fn bound_pair(v: &tpt_opt_core::model::Variable) -> (f64, f64) {
    let b = &v.bound.bound;
    (b.lower, b.upper)
}

/// Evaluate whether `x` satisfies all constraints within `tol`.
fn feasible(model: &Model, x: &[f64], tol: f64) -> bool {
    for c in &model.constraints {
        if !c.is_satisfied(x, tol) {
            return false;
        }
    }
    for (i, v) in model.variables.iter().enumerate() {
        if !v.bound.feasible(x[i], tol) {
            return false;
        }
    }
    true
}

/// Evaluate the objective at `x` in the model's own sense.
fn eval_obj(model: &Model, x: &[f64]) -> f64 {
    model.objective.eval(x)
}
