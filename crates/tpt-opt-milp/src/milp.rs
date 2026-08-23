//! Branch-and-bound / branch-and-cut MILP solver built on an internal simplex
//! LP relaxation.
//!
//! The [`MilpSolver`] implements [`tpt_opt_core::solver::Solver<Model>`] with
//! deterministic, seedable behaviour:
//!
//! - **Branching**: most-fractional, pseudo-cost, and limited strong branching
//!   ([`BranchingRule`]).
//! - **Node selection**: best-bound, depth-first diving, and best-estimate
//!   ([`NodeSelection`]).
//! - **Primal heuristics**: rounding, feasibility pump, RINS, and local
//!   branching (root + periodic re-application).
//! - **Cuts** ([`crate::cuts`]): Gomory mixed-integer, lift-and-project
//!   intersection cuts (tableau space), plus clique, cover, and MIR cuts
//!   (model space), all behind `.with_cuts(true)`.
//! - **Modelling extras**: SOS1/SOS2 sets ([`crate::sos`]), indicator
//!   constraints ([`crate::indicator`]), and piecewise-linear objectives
//!   ([`crate::piecewise`]).
//! - **Parallelism**: `.with_threads(n > 1)` partitions the tree
//!   deterministically across scoped worker threads (see [`MilpSolver::solve`]).
//!
//! All randomness derives from the configured seed, so a fixed seed reproduces
//! the same search — including the deterministic work distribution used in
//! parallel mode.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::vec::Vec;

use tpt_opt_core::{
    model::{Constraint, Model, Sense},
    progress::{ProgressAction, ProgressCallback, ProgressEvent},
    solver::{Solution, SolveParameters, Solver, SolverStatus, WarmStart},
    tolerance::Tolerances,
    OptError,
};

use crate::indicator::IndicatorConstraint;
use crate::lp::{solve_lp, solve_lp_state, LpStatus};
use crate::piecewise::PiecewiseObjective;
use crate::sos::SosSet;

/// Branching rule for selecting the next integer variable to branch on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchingRule {
    /// Branch on the variable with the most fractional LP value.
    MostFractional,
    /// Branch using pseudo-cost estimates of the bound improvement.
    PseudoCost,
    /// Strong branching: tentatively solve the child LPs of the
    /// `candidates` most-fractional variables and branch on the best
    /// improvement product. Costs `2 × candidates` LP solves per node.
    StrongBranching {
        /// Number of fractional candidates to evaluate per node.
        candidates: usize,
    },
}

/// Node-selection strategy for the branch-and-bound tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeSelection {
    /// Always expand the node with the best (tightest) LP bound.
    BestBound,
    /// Depth-first dive (LIFO stack).
    DepthFirst,
    /// Expand the node with the best *estimated* integer objective
    /// (LP bound plus pseudo-cost degradation of fractional variables).
    BestEstimate,
}

/// Builder / solver for mixed-integer linear programs.
pub struct MilpSolver {
    params: SolveParameters,
    branching: BranchingRule,
    selection: NodeSelection,
    use_cuts: bool,
    cut_rounds: usize,
    seed: u64,
    node_limit: Option<usize>,
    nested: bool,
    sos_sets: Vec<SosSet>,
    indicators: Vec<IndicatorConstraint>,
    piecewise: Option<PiecewiseObjective>,
    /// Optional progress callback (see [`MilpSolver::with_progress_callback`]).
    /// Shared through an `Arc<Mutex<..>>` so parallel workers can emit events;
    /// *not* inherited by clones (nested heuristic sub-solves stay silent).
    progress: Option<Arc<Mutex<Box<ProgressCallback>>>>,
    /// Set when the progress callback (or an internal limit) requests an
    /// early stop; shared with clones so nested solves shut down promptly.
    abort_flag: Arc<AtomicBool>,
    // Runtime state (result of the last solve).
    incumbent_obj: Option<f64>,
    incumbent_x: Option<Vec<f64>>,
    last_status: SolverStatus,
    last_nodes: usize,
    pending_warm: Option<Vec<f64>>,
}

impl Clone for MilpSolver {
    fn clone(&self) -> Self {
        Self {
            params: self.params,
            branching: self.branching,
            selection: self.selection,
            use_cuts: self.use_cuts,
            cut_rounds: self.cut_rounds,
            seed: self.seed,
            node_limit: self.node_limit,
            nested: self.nested,
            sos_sets: self.sos_sets.clone(),
            indicators: self.indicators.clone(),
            piecewise: self.piecewise.clone(),
            // Deliberately dropped: nested sub-solves must not spam the
            // user's callback (or abort it mid-heuristic).
            progress: None,
            abort_flag: Arc::clone(&self.abort_flag),
            incumbent_obj: self.incumbent_obj,
            incumbent_x: self.incumbent_x.clone(),
            last_status: self.last_status,
            last_nodes: self.last_nodes,
            pending_warm: self.pending_warm.clone(),
        }
    }
}

impl core::fmt::Debug for MilpSolver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MilpSolver")
            .field("params", &self.params)
            .field("branching", &self.branching)
            .field("selection", &self.selection)
            .field("use_cuts", &self.use_cuts)
            .field("cut_rounds", &self.cut_rounds)
            .field("seed", &self.seed)
            .field("node_limit", &self.node_limit)
            .field("nested", &self.nested)
            .field("sos_sets", &self.sos_sets)
            .field("indicators", &self.indicators)
            .field("piecewise", &self.piecewise)
            .field("has_progress_callback", &self.progress.is_some())
            .field("last_status", &self.last_status)
            .field("last_nodes", &self.last_nodes)
            .finish_non_exhaustive()
    }
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
            cut_rounds: 1,
            seed: 0,
            node_limit: None,
            nested: false,
            sos_sets: Vec::new(),
            indicators: Vec::new(),
            piecewise: None,
            progress: None,
            abort_flag: Arc::new(AtomicBool::new(false)),
            incumbent_obj: None,
            incumbent_x: None,
            last_status: SolverStatus::Error,
            last_nodes: 0,
            pending_warm: None,
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

    /// Set the number of worker threads. Values `> 1` enable the
    /// deterministic parallel tree search (see [module docs](self)).
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.params = self.params.with_threads(threads.max(1));
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

    /// Enable / disable the full cut suite (Gomory, lift-and-project, clique,
    /// cover, MIR) at the root node.
    pub fn with_cuts(mut self, on: bool) -> Self {
        self.use_cuts = on;
        self
    }

    /// Number of root cut rounds (each round generates cuts and re-solves the
    /// root relaxation). Only meaningful with `.with_cuts(true)`.
    pub fn with_parallel_cuts(mut self, rounds: usize) -> Self {
        self.cut_rounds = rounds.max(1);
        self
    }

    /// Cap the number of explored nodes; the solve then reports
    /// [`SolverStatus::TimeLimit`] with the best incumbent found.
    pub fn with_node_limit(mut self, nodes: usize) -> Self {
        self.node_limit = Some(nodes);
        self
    }

    /// Register a progress callback invoked at coarse checkpoints (root LP,
    /// root heuristics, every 16 explored nodes, and on each new incumbent).
    ///
    /// Returning [`ProgressAction::Abort`] from the callback stops the search
    /// as soon as possible — including in parallel mode and inside nested
    /// heuristic sub-solves — and the solve reports
    /// [`SolverStatus::TimeLimit`] with the best incumbent found so far.
    /// Events are serialised through an internal lock, so a `FnMut` callback
    /// is safe even when worker threads emit concurrently.
    pub fn with_progress_callback(mut self, cb: Box<ProgressCallback>) -> Self {
        self.progress = Some(Arc::new(Mutex::new(cb)));
        self
    }

    /// Attach an SOS1/SOS2 set over model variables.
    pub fn add_sos(&mut self, set: SosSet) {
        self.sos_sets.push(set);
    }

    /// Builder-style variant of [`MilpSolver::add_sos`].
    pub fn with_sos(mut self, set: SosSet) -> Self {
        self.sos_sets.push(set);
        self
    }

    /// Attach an indicator constraint ("if binary y = trigger then row").
    pub fn add_indicator(&mut self, ind: IndicatorConstraint) {
        self.indicators.push(ind);
    }

    /// Builder-style variant of [`MilpSolver::add_indicator`].
    pub fn with_indicator(mut self, ind: IndicatorConstraint) -> Self {
        self.indicators.push(ind);
        self
    }

    /// Attach a piecewise-linear objective term (reformulated with SOS2).
    ///
    /// # Errors
    /// Returns [`OptError::InvalidModel`] if the breakpoints are invalid.
    pub fn set_piecewise(&mut self, pw: PiecewiseObjective) -> Result<(), OptError> {
        // Validate eagerly by checking breakpoint count/distinctness through a
        // dry-run against a 0-variable model is not possible; validate shape.
        if pw.breakpoints.len() < 2 {
            return Err(OptError::invalid_model(
                "piecewise objective needs at least two breakpoints",
            ));
        }
        self.piecewise = Some(pw);
        Ok(())
    }

    /// Apply a parameter bundle.
    pub fn with_parameters(mut self, p: SolveParameters) -> Self {
        self.params = p;
        if p.seed.is_some() {
            self.seed = p.seed.unwrap();
        }
        self
    }

    /// Mark this solver as nested (used internally by RINS / local branching
    /// sub-solves to prevent recursive heuristic explosion).
    fn into_nested(mut self) -> Self {
        self.nested = true;
        self
    }
}

// ---------------------------------------------------------------------------
// Search machinery (shared by sequential and parallel modes)
// ---------------------------------------------------------------------------

/// A search node in the branch-and-bound tree.
#[derive(Clone)]
struct Node {
    lb: Vec<f64>,
    ub: Vec<f64>,
    /// LP objective of the parent at creation time (the node's inherited
    /// bound); refreshed with the node's own LP value once solved. Always the
    /// *raw* objective in the model's own sense (used for pruning).
    bound: f64,
    /// Heap priority (sign-adjusted per sense; includes the best-estimate
    /// degradation term under [`NodeSelection::BestEstimate`]).
    key: f64,
    depth: usize,
    /// Active `(sos_set_index, lo, hi)` member ranges; members outside their
    /// range are fixed to zero via bounds.
    sos_ranges: Vec<(usize, usize, usize)>,
    /// Parent branching info for pseudo-cost updates:
    /// `(variable, direction (0 = down, 1 = up), fractional distance)`.
    parent: Option<(usize, u8, f64)>,
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
        // BinaryHeap is a max-heap; `key` is pre-negated for minimisation
        // problems so popping always yields the best-first node.
        self.key.partial_cmp(&other.key).unwrap_or(Ordering::Equal)
    }
}

/// Best-known feasible solution accumulated during a search.
#[derive(Debug, Default, Clone)]
struct Incumbent {
    obj: Option<f64>,
    x: Option<Vec<f64>>,
}

impl Incumbent {
    fn consider(&mut self, obj: f64, x: Vec<f64>, sense: Sense, gap: f64) -> bool {
        let better = match (self.obj, sense) {
            (None, _) => true,
            (Some(cur), Sense::Minimize) => obj < cur - gap,
            (Some(cur), Sense::Maximize) => obj > cur + gap,
        };
        if better {
            self.obj = Some(obj);
            self.x = Some(x);
        }
        better
    }
}

/// Per-worker search state returned by [`MilpSolver::run_search`].
#[derive(Debug, Default)]
struct SearchOutcome {
    inc: Incumbent,
    nodes_explored: usize,
    timed_out: bool,
    pseudo_up: Vec<f64>,
    pseudo_down: Vec<f64>,
}

/// Small deterministic RNG (LCG) so heuristics are reproducible for a seed.
struct Lcg {
    state: u64,
}

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

impl MilpSolver {
    /// Run the branch-and-bound loop over `initial` nodes. `&self` so it can
    /// be shared across scoped threads; all mutation is local to `outcome`.
    #[allow(clippy::too_many_arguments)]
    fn run_search(
        &self,
        model: &Model,
        sense: Sense,
        int_vars: &[usize],
        sos: &[SosSet],
        initial: Vec<Node>,
        inc0: Incumbent,
        start: Instant,
    ) -> SearchOutcome {
        let n = model.num_vars;
        let tol = self.params.tolerances;
        let mut outcome = SearchOutcome {
            inc: inc0,
            pseudo_up: vec![1.0; n],
            pseudo_down: vec![1.0; n],
            ..SearchOutcome::default()
        };

        let mut heap: BinaryHeap<Node> = BinaryHeap::new();
        let mut stack: Vec<Node> = Vec::new();
        for nd in initial {
            match self.selection {
                NodeSelection::BestBound | NodeSelection::BestEstimate => heap.push(nd),
                NodeSelection::DepthFirst => stack.push(nd),
            }
        }

        loop {
            let node = match self.selection {
                NodeSelection::BestBound | NodeSelection::BestEstimate => heap.pop(),
                NodeSelection::DepthFirst => stack.pop(),
            };
            let mut node = match node {
                Some(nd) => nd,
                None => break,
            };
            outcome.nodes_explored += 1;

            if self.timed_out(start)
                || self.limit_reached(outcome.nodes_explored)
                || self.abort_flag.load(AtomicOrdering::Relaxed)
            {
                outcome.timed_out = true;
                break;
            }

            // Prune by bound vs incumbent.
            if let Some(inc) = outcome.inc.obj {
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

            // Pseudo-cost update from the parent branching decision.
            if let Some((var, dir, dist)) = node.parent {
                if dist > 1e-9 {
                    let observed = ((lp.objective - node.bound) / dist).max(0.0);
                    if observed.is_finite() {
                        let slot = if dir == 0 {
                            &mut outcome.pseudo_down
                        } else {
                            &mut outcome.pseudo_up
                        };
                        slot[var] = 0.5 * slot[var] + 0.5 * observed;
                    }
                }
            }
            node.bound = lp.objective;

            // Bound-based prune again with the node's own LP value.
            if let Some(inc) = outcome.inc.obj {
                let dominated = match sense {
                    Sense::Minimize => node.bound > inc + self.params.absolute_gap,
                    Sense::Maximize => node.bound < inc - self.params.absolute_gap,
                };
                if dominated {
                    continue;
                }
            }

            // 1. SOS violations branch before integrality.
            if let Some((si, lo, mid, hi)) = find_sos_branch(sos, &lp.x, &node, tol.integrality) {
                let kl = child_key(
                    self.selection,
                    sense,
                    lp.objective,
                    &lp.x,
                    int_vars,
                    &outcome.pseudo_up,
                    &outcome.pseudo_down,
                );
                let kr = kl;
                push_children(
                    self.selection,
                    sense,
                    sos,
                    node,
                    lp.objective,
                    kl,
                    kr,
                    Some((si, lo, mid, hi)),
                    &mut heap,
                    &mut stack,
                );
                continue;
            }

            // 2. Integrality check.
            let frac = self.fractional_candidates(&lp.x, int_vars, tol.integrality);
            if frac.is_empty() {
                // Integral solution -> candidate incumbent (SOS already OK).
                if feasible_with_sos(sos, model, &lp.x, &node, tol.feasibility) {
                    let improved = outcome.inc.consider(
                        lp.objective,
                        lp.x.clone(),
                        sense,
                        self.params.absolute_gap,
                    );
                    if improved
                        && !self.emit_progress(
                            outcome.nodes_explored,
                            outcome.inc.obj,
                            frontier_bound(&heap, sense),
                            start,
                        )
                    {
                        outcome.timed_out = true;
                        break;
                    }
                }
                continue;
            }

            // 3. Choose the branching variable.
            let bv = match self.branching {
                BranchingRule::StrongBranching { candidates } => {
                    let (v, obs) = self.strong_branch(
                        model,
                        sense,
                        &node.lb,
                        &node.ub,
                        parent_obj_of(&node),
                        &lp.x,
                        &frac,
                        candidates,
                        tol,
                    );
                    for (var, d, u) in obs {
                        if d.is_finite() {
                            outcome.pseudo_down[var] =
                                0.5 * outcome.pseudo_down[var] + 0.5 * d.max(0.0);
                        }
                        if u.is_finite() {
                            outcome.pseudo_up[var] =
                                0.5 * outcome.pseudo_up[var] + 0.5 * u.max(0.0);
                        }
                    }
                    v
                }
                _ => self.choose_branch_var(&lp.x, &frac, &outcome),
            };

            // 4. Periodic primal heuristics.
            if outcome.nodes_explored % 16 == 0 {
                heur_rounding(
                    model,
                    sense,
                    &lp.x,
                    &node.lb,
                    &node.ub,
                    tol,
                    &mut outcome.inc,
                    self.params.absolute_gap,
                    &mut Lcg::new(self.seed.wrapping_add(outcome.nodes_explored as u64)),
                );
            }
            if outcome.nodes_explored % 64 == 0 && !self.nested {
                let snap = outcome.inc.clone();
                heur_rins(self, model, sense, &lp.x, &snap, start, &mut outcome.inc);
            }

            // Periodic progress emission (same cadence as the heuristics so a
            // callback sees fresh incumbents within at most 16 nodes).
            if outcome.nodes_explored % 16 == 0
                && !self.emit_progress(
                    outcome.nodes_explored,
                    outcome.inc.obj,
                    frontier_bound(&heap, sense),
                    start,
                )
            {
                outcome.timed_out = true;
                break;
            }

            // 5. Create children.
            let fv = lp.x[bv];
            let down = fv.floor();
            let up = fv.ceil();
            let key = child_key(
                self.selection,
                sense,
                lp.objective,
                &lp.x,
                int_vars,
                &outcome.pseudo_up,
                &outcome.pseudo_down,
            );
            let mut left = node.clone();
            left.ub[bv] = down.min(node.ub[bv]);
            left.bound = lp.objective;
            left.key = key;
            left.depth = node.depth + 1;
            left.parent = Some((bv, 0, fv - down));
            let mut right = node.clone();
            right.lb[bv] = up.max(node.lb[bv]);
            right.bound = lp.objective;
            right.key = key;
            right.depth = node.depth + 1;
            right.parent = Some((bv, 1, up - fv));

            match self.selection {
                NodeSelection::BestBound | NodeSelection::BestEstimate => {
                    heap.push(left);
                    heap.push(right);
                }
                NodeSelection::DepthFirst => {
                    stack.push(left);
                    stack.push(right);
                }
            }
        }

        outcome
    }

    fn timed_out(&self, start: Instant) -> bool {
        match self.params.time_limit {
            Some(lim) => start.elapsed().as_secs_f64() >= lim,
            None => false,
        }
    }

    fn limit_reached(&self, nodes: usize) -> bool {
        self.node_limit.is_some_and(|lim| nodes >= lim)
    }

    /// Deliver a [`ProgressEvent`] to the registered callback, if any.
    /// Returns `false` when the callback requested an abort (the shared flag
    /// is set first, so parallel workers and nested sub-solves stop too).
    /// A poisoned callback lock is recovered rather than propagated: losing
    /// one event must not fail an otherwise healthy solve.
    fn emit_progress(
        &self,
        iterations: usize,
        incumbent: Option<f64>,
        bound: Option<f64>,
        start: Instant,
    ) -> bool {
        let Some(cb) = &self.progress else { return true };
        let ev = ProgressEvent { iterations, incumbent, bound, elapsed: start.elapsed() };
        let action = cb.lock().unwrap_or_else(|poisoned| poisoned.into_inner())(&ev);
        if action == ProgressAction::Abort {
            self.abort_flag.store(true, AtomicOrdering::Relaxed);
            return false;
        }
        true
    }

    /// Fractional integer variables sorted by decreasing fractionality
    /// distance (most fractional first).
    fn fractional_candidates(&self, x: &[f64], int_vars: &[usize], tol: f64) -> Vec<usize> {
        let mut cands: Vec<(usize, f64)> = int_vars
            .iter()
            .copied()
            .filter_map(|i| {
                let f = x[i] - x[i].floor();
                let dist = f.min(1.0 - f);
                (dist > tol).then_some((i, dist))
            })
            .collect();
        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        cands.into_iter().map(|(i, _)| i).collect()
    }

    /// Choose the branching variable according to the configured rule.
    fn choose_branch_var(&self, x: &[f64], cands: &[usize], outcome: &SearchOutcome) -> usize {
        match self.branching {
            BranchingRule::MostFractional => cands[0],
            BranchingRule::PseudoCost => {
                let mut best = cands[0];
                let mut best_score = f64::NEG_INFINITY;
                for &i in cands.iter().take(8) {
                    let f = x[i] - x[i].floor();
                    let down = (f * outcome.pseudo_down[i]).max(1e-6);
                    let up = ((1.0 - f) * outcome.pseudo_up[i]).max(1e-6);
                    let score = down * up;
                    if score > best_score {
                        best_score = score;
                        best = i;
                    }
                }
                best
            }
            BranchingRule::StrongBranching { candidates: _ } => {
                // Full strong branching runs in `strong_branch`; here we fall
                // back to the pseudo-cost product when LP context is absent.
                cands[0]
            }
        }
    }

    /// Strong branching: evaluate child LPs for up to `k` fractional
    /// candidates and branch on the best improvement product. Returns the
    /// chosen variable plus per-candidate `(down, up)` degradation-per-unit
    /// observations for pseudo-cost refinement.
    #[allow(clippy::too_many_arguments)]
    fn strong_branch(
        &self,
        model: &Model,
        sense: Sense,
        lb: &[f64],
        ub: &[f64],
        parent_obj: f64,
        x: &[f64],
        cands: &[usize],
        k: usize,
        tol: Tolerances,
    ) -> (usize, Vec<(usize, f64, f64)>) {
        const CAP: f64 = 1e12;
        let eps = 1e-6;
        let mut observations = Vec::new();
        let mut best = cands[0];
        let mut best_score = f64::NEG_INFINITY;
        for &v in cands.iter().take(k.max(1)) {
            let fv = x[v];
            let down_v = fv.floor();
            let up_v = fv.ceil();

            let lb_d = lb.to_vec();
            let mut ub_d = ub.to_vec();
            ub_d[v] = down_v.min(ub[v]);
            let lp_d = solve_lp(model, &lb_d, &ub_d, tol);

            let mut lb_u = lb.to_vec();
            let ub_u = ub.to_vec();
            lb_u[v] = up_v.max(lb[v]);
            let lp_u = solve_lp(model, &lb_u, &ub_u, tol);

            let imp = |lp: &crate::lp::LpSolution| match lp.status {
                LpStatus::Optimal => match sense {
                    Sense::Minimize => lp.objective - parent_obj,
                    Sense::Maximize => parent_obj - lp.objective,
                },
                LpStatus::Infeasible => CAP, // child dead: maximal improvement
                LpStatus::Unbounded => -CAP,
            };
            let d_imp = imp(&lp_d).clamp(-CAP, CAP);
            let u_imp = imp(&lp_u).clamp(-CAP, CAP);
            observations.push((v, d_imp / (fv - down_v).max(1e-9), u_imp / (up_v - fv).max(1e-9)));
            let score = d_imp.max(eps) * u_imp.max(eps);
            if score > best_score {
                best_score = score;
                best = v;
            }
        }
        (best, observations)
    }
}

/// The bound a node inherited from its parent (raw objective).
fn parent_obj_of(node: &Node) -> f64 {
    node.bound
}

/// Best remaining frontier bound for progress reporting: the heap top's key
/// converted back to the raw-objective convention (`None` under depth-first
/// selection, where no global bound is maintained). Under
/// [`NodeSelection::BestEstimate`] the value includes pseudo-cost degradation,
/// i.e. it is an estimate rather than a proven bound.
fn frontier_bound(heap: &BinaryHeap<Node>, sense: Sense) -> Option<f64> {
    heap.peek().map(|n| match sense {
        Sense::Minimize => -n.key,
        Sense::Maximize => n.key,
    })
}

/// Heap priority for a freshly created child node. For [`NodeSelection::
/// BestEstimate`] the LP bound is adjusted by the pseudo-cost degradation of
/// remaining fractional variables; the result is sign-adjusted so the max-heap
/// pops best-first under both senses.
fn child_key(
    selection: NodeSelection,
    sense: Sense,
    bound: f64,
    x: &[f64],
    int_vars: &[usize],
    pseudo_up: &[f64],
    pseudo_down: &[f64],
) -> f64 {
    let est = match selection {
        NodeSelection::BestEstimate => {
            let mut deg = 0.0;
            for &i in int_vars {
                let f = x[i] - x[i].floor();
                if f > 1e-9 && (1.0 - f) > 1e-9 {
                    deg += (f * pseudo_down[i]).min((1.0 - f) * pseudo_up[i]);
                }
            }
            match sense {
                Sense::Minimize => bound + deg,
                Sense::Maximize => bound - deg,
            }
        }
        _ => bound,
    };
    match sense {
        Sense::Minimize => -est,
        Sense::Maximize => est,
    }
}

/// Find a violated SOS set that can still be branched (its active member
/// range has >= 2 members). Returns `(set_index, lo, mid, hi)` for the split.
fn find_sos_branch(
    sos: &[SosSet],
    x: &[f64],
    node: &Node,
    tol: f64,
) -> Option<(usize, usize, usize, usize)> {
    for (si, set) in sos.iter().enumerate() {
        let (lo, hi) = node
            .sos_ranges
            .iter()
            .find(|&&(idx, _, _)| idx == si)
            .map(|&(_, lo, hi)| (lo, hi))
            .unwrap_or((0, set.len()));
        if hi - lo < 2 {
            continue;
        }
        // Violation check restricted to the active range.
        let sub = SosSet {
            kind: set.kind,
            vars: set.vars[lo..hi].to_vec(),
            weights: set.weights[lo..hi].to_vec(),
        };
        if !sub.is_satisfied(x, tol) {
            let mid = lo + (hi - lo) / 2;
            return Some((si, lo, mid, hi));
        }
    }
    None
}

/// Push the two children implied by an SOS split into the frontier.
#[allow(clippy::too_many_arguments)]
fn push_children(
    selection: NodeSelection,
    _sense: Sense,
    sos: &[SosSet],
    node: Node,
    parent_obj: f64,
    key_left: f64,
    key_right: f64,
    sos_split: Option<(usize, usize, usize, usize)>,
    heap: &mut BinaryHeap<Node>,
    stack: &mut Vec<Node>,
) {
    let mut left = node.clone();
    left.bound = parent_obj;
    left.key = key_left;
    left.depth = node.depth + 1;
    let mut right = node.clone();
    right.bound = parent_obj;
    right.key = key_right;
    right.depth = node.depth + 1;
    if let Some((si, lo, mid, hi)) = sos_split {
        left.sos_ranges.retain(|&(idx, _, _)| idx != si);
        left.sos_ranges.push((si, lo, mid));
        right.sos_ranges.retain(|&(idx, _, _)| idx != si);
        right.sos_ranges.push((si, mid, hi));
        // Fix excluded members to zero via bounds.
        let set = &sos[si];
        for (k, &v) in set.vars.iter().enumerate() {
            if !(lo..mid).contains(&k) {
                left.lb[v] = 0.0;
                left.ub[v] = 0.0;
            }
            if !(mid..hi).contains(&k) {
                right.lb[v] = 0.0;
                right.ub[v] = 0.0;
            }
        }
    }
    match selection {
        NodeSelection::BestBound | NodeSelection::BestEstimate => {
            heap.push(left);
            heap.push(right);
        }
        NodeSelection::DepthFirst => {
            stack.push(left);
            stack.push(right);
        }
    }
}

/// Feasibility check including SOS sets and node SOS ranges.
fn feasible_with_sos(sos: &[SosSet], model: &Model, x: &[f64], node: &Node, tol: f64) -> bool {
    if !feasible(model, x, tol) {
        return false;
    }
    for (si, set) in sos.iter().enumerate() {
        let (lo, hi) = node
            .sos_ranges
            .iter()
            .find(|&&(idx, _, _)| idx == si)
            .map(|&(_, lo, hi)| (lo, hi))
            .unwrap_or((0, set.len()));
        let sub = SosSet {
            kind: set.kind,
            vars: set.vars[lo..hi].to_vec(),
            weights: set.weights[lo..hi].to_vec(),
        };
        if !sub.is_satisfied(x, tol) {
            return false;
        }
    }
    true
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

// ---------------------------------------------------------------------------
// Primal heuristics
// ---------------------------------------------------------------------------

/// Randomised + deterministic rounding of the LP relaxation.
#[allow(clippy::too_many_arguments)]
fn heur_rounding(
    model: &Model,
    sense: Sense,
    x_lp: &[f64],
    lb: &[f64],
    ub: &[f64],
    tol: Tolerances,
    inc: &mut Incumbent,
    gap: f64,
    rng: &mut Lcg,
) {
    // Deterministic nearest rounding first.
    let mut cand = x_lp.to_vec();
    for (i, v) in model.variables.iter().enumerate() {
        if v.bound.is_integral() {
            cand[i] = x_lp[i].round();
            if lb[i].is_finite() || ub[i].is_finite() {
                cand[i] = cand[i].max(lb[i]).min(ub[i]);
            }
        }
    }
    if feasible(model, &cand, tol.feasibility) {
        inc.consider(eval_obj(model, &cand), cand, sense, gap);
        return;
    }
    // A few seeded randomised rounding trials.
    for _ in 0..4 {
        let mut trial = x_lp.to_vec();
        for (i, v) in model.variables.iter().enumerate() {
            if v.bound.is_integral() {
                let f = x_lp[i] - x_lp[i].floor();
                let go_up = rng.f64() < f;
                let r = if go_up { x_lp[i].ceil() } else { x_lp[i].floor() };
                trial[i] = r.max(lb[i]).min(ub[i]);
            }
        }
        if feasible(model, &trial, tol.feasibility) {
            inc.consider(eval_obj(model, &trial), trial, sense, gap);
        }
    }
}

/// Feasibility pump: alternate rounding and LP projection onto the rounded
/// integral assignment.
fn heur_feasibility_pump(
    model: &Model,
    sense: Sense,
    lb: &[f64],
    ub: &[f64],
    tol: Tolerances,
    inc: &mut Incumbent,
    gap: f64,
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
            inc.consider(eval_obj(model, &r), r, sense, gap);
        }
    }
}

/// RINS: Relaxation Induced Neighborhood Search. Fix the integer variables
/// where the LP relaxation and the incumbent agree; re-optimise the remaining
/// (smaller) MILP with a capped inner solver.
fn heur_rins(
    outer: &MilpSolver,
    model: &Model,
    sense: Sense,
    x_lp: &[f64],
    inc: &Incumbent,
    start: Instant,
    out: &mut Incumbent,
) {
    let Some(x_inc) = &inc.x else { return };
    let tol = outer.params.tolerances;
    let mut sub = model.clone();
    for (i, v) in model.variables.iter().enumerate() {
        if v.bound.is_integral() && (x_lp[i] - x_inc[i]).abs() <= tol.integrality {
            let val = x_inc[i].round();
            sub.variables[i].bound.bound.lower = val.max(v.bound.bound.lower);
            sub.variables[i].bound.bound.upper = val.min(v.bound.bound.upper);
        }
    }
    let remaining = model
        .variables
        .iter()
        .enumerate()
        .filter(|(i, v)| v.bound.is_integral() && (x_lp[*i] - x_inc[*i]).abs() > tol.integrality)
        .count();
    if remaining == 0 {
        return;
    }
    let time_left =
        outer.params.time_limit.map(|tl| (tl - start.elapsed().as_secs_f64()).max(0.05));
    let mut inner = MilpSolver::new()
        .into_nested()
        .with_node_selection(NodeSelection::DepthFirst)
        .with_node_limit(400);
    if let Some(tl) = time_left {
        inner = inner.with_time_limit(Duration::from_secs_f64(tl.min(2.0)));
    }
    if let Ok(sol) = inner.solve(&sub) {
        if sol.status.has_solution() && feasible(model, &sol.primal, tol.feasibility) {
            out.consider(
                eval_obj(model, &sol.primal),
                sol.primal.clone(),
                sense,
                outer.params.absolute_gap,
            );
        }
    }
}

/// Local branching: add `||x - x_inc||_1 <= k` around the incumbent and
/// re-optimise with a capped inner solver.
fn heur_local_branching(
    outer: &MilpSolver,
    model: &Model,
    sense: Sense,
    inc: &Incumbent,
    start: Instant,
    out: &mut Incumbent,
) {
    let Some(x_inc) = &inc.x else { return };
    let tol = outer.params.tolerances;
    let ints: Vec<usize> = model
        .variables
        .iter()
        .enumerate()
        .filter(|(_, v)| v.bound.is_integral())
        .map(|(i, _)| i)
        .collect();
    if ints.is_empty() {
        return;
    }
    let k = (ints.len() / 5).clamp(2, 10) as f64;
    let mut sub = model.clone();
    // Σ_{x_inc=1} (1 - x_i) + Σ_{x_inc=0} x_i <= k
    let mut idx = Vec::new();
    let mut coefs = Vec::new();
    let mut rhs = k;
    for &i in &ints {
        if x_inc[i].round() >= 0.5 {
            idx.push(i);
            coefs.push(-1.0);
            rhs -= 1.0;
        } else {
            idx.push(i);
            coefs.push(1.0);
        }
    }
    sub.add_constraint(Constraint::le(idx, coefs, rhs));

    let time_left =
        outer.params.time_limit.map(|tl| (tl - start.elapsed().as_secs_f64()).max(0.05));
    let mut inner = MilpSolver::new()
        .into_nested()
        .with_node_selection(NodeSelection::DepthFirst)
        .with_node_limit(400);
    if let Some(tl) = time_left {
        inner = inner.with_time_limit(Duration::from_secs_f64(tl.min(2.0)));
    }
    if let Ok(sol) = inner.solve(&sub) {
        if sol.status.has_solution() && feasible(model, &sol.primal, tol.feasibility) {
            out.consider(
                eval_obj(model, &sol.primal),
                sol.primal.clone(),
                sense,
                outer.params.absolute_gap,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Solver trait implementation
// ---------------------------------------------------------------------------

impl Solver<Model> for MilpSolver {
    fn solve(&mut self, model: &Model) -> Result<Solution, OptError> {
        model.validate()?;
        let start = Instant::now();
        let sense = model.objective.sense;
        let orig_n = model.num_vars;

        // A fresh solve clears any abort left over from a previous one.
        self.abort_flag.store(false, AtomicOrdering::Relaxed);

        // ---- Build the working model --------------------------------------
        let mut work = model.clone();

        // Indicator constraints expand to big-M rows derived from bounds.
        if !self.indicators.is_empty() {
            let inds = self.indicators.clone();
            for ind in &inds {
                let rows = ind.expand(&work).ok_or_else(|| {
                    OptError::invalid_model(
                        "indicator row cannot be big-M expanded: tighten variable bounds",
                    )
                })?;
                for r in rows {
                    work.add_constraint(r);
                }
            }
        }

        // Piecewise-linear objective reformulation (appends lambda variables).
        let mut extra_sos: Vec<SosSet> = Vec::new();
        if let Some(pw) = &self.piecewise {
            let (aug, sos, _) = pw.augment(&work)?;
            work = aug;
            extra_sos.push(sos);
        }

        let n = work.num_vars;
        let mut lb = vec![f64::NEG_INFINITY; n];
        let mut ub = vec![f64::INFINITY; n];
        let mut int_vars = Vec::new();
        let mut binaries = Vec::new();
        for (i, v) in work.variables.iter().enumerate() {
            let (lo, hi) = bound_pair(v);
            lb[i] = lo;
            ub[i] = hi;
            if v.bound.is_integral() {
                int_vars.push(i);
                if v.is_binary() {
                    binaries.push(i);
                }
            }
        }

        // ---- Warm start ----------------------------------------------------
        let mut inc = Incumbent::default();
        if let Some(wx) = &self.pending_warm {
            if wx.len() == orig_n {
                let mut full = wx.clone();
                full.resize(n, 0.0);
                if feasible(&work, &full, self.params.tolerances.feasibility) {
                    inc.consider(eval_obj(&work, &full), full, sense, self.params.absolute_gap);
                }
            }
            self.pending_warm = None;
        }

        // ---- Root cuts -------------------------------------------------------
        // Model-space families first (clique / cover / MIR), then
        // `cut_rounds` rounds of tableau-space cuts (Gomory mixed-integer +
        // lift-and-project intersection cuts) derived from the root
        // relaxation and appended to the working model so every node LP
        // benefits from them.
        if self.use_cuts {
            crate::cuts::add_clique_cuts(&mut work, &binaries, 20);
            crate::cuts::add_cover_cuts(&mut work, &binaries, 20);
            crate::cuts::add_mir_cuts(&mut work, &int_vars, 20);
            let tol0 = self.params.tolerances;
            for _ in 0..self.cut_rounds {
                let st = solve_lp_state(&work, &lb, &ub, tol0);
                if st.sol.status != LpStatus::Optimal {
                    break;
                }
                let g = crate::gomory::add_gomory_cuts(&mut work, &st, &int_vars, 20);
                let l = crate::gomory::add_lift_and_project_cuts(
                    &mut work, &st, &binaries, &int_vars, 20,
                );
                if g + l == 0 {
                    break; // no new cuts: further rounds cannot help
                }
            }
        }

        // ---- Root LP ---------------------------------------------------------
        let tol = self.params.tolerances;
        let root = solve_lp_state(&work, &lb, &ub, tol);

        if root.sol.status == LpStatus::Infeasible {
            self.last_status = SolverStatus::Infeasible;
            self.last_nodes = 0;
            return Ok(Solution::new(vec![0.0; orig_n], 0.0, SolverStatus::Infeasible));
        }
        if root.sol.status == LpStatus::Unbounded {
            self.last_status = SolverStatus::Unbounded;
            self.last_nodes = 0;
            return Ok(Solution::new(vec![0.0; orig_n], 0.0, SolverStatus::Unbounded));
        }

        // ---- Root LP progress checkpoint --------------------------------------
        self.emit_progress(0, inc.obj, Some(root.sol.objective), start);

        // ---- Root heuristics -------------------------------------------------
        heur_rounding(
            &work,
            sense,
            &root.sol.x,
            &lb,
            &ub,
            tol,
            &mut inc,
            self.params.absolute_gap,
            &mut Lcg::new(self.seed),
        );
        heur_feasibility_pump(&work, sense, &lb, &ub, tol, &mut inc, self.params.absolute_gap);
        if !self.nested {
            {
                let snap0 = inc.clone();
                heur_local_branching(self, &work, sense, &snap0, start, &mut inc);
            }
        }

        // ---- Root-heuristic progress checkpoint -------------------------------
        // An abort here skips the tree search entirely; the finalise step then
        // reports TimeLimit with whatever warm-start/root incumbents exist.
        self.emit_progress(0, inc.obj, Some(root.sol.objective), start);

        // ---- Initial node -----------------------------------------------------
        let root_key = match sense {
            Sense::Minimize => -root.sol.objective,
            Sense::Maximize => root.sol.objective,
        };
        let root_node = Node {
            lb,
            ub,
            bound: root.sol.objective,
            key: root_key,
            depth: 0,
            sos_ranges: Vec::new(),
            parent: None,
        };

        // All SOS sets: user-attached plus any generated by the piecewise
        // reformulation.
        let mut all_sos = self.sos_sets.clone();
        all_sos.extend(extra_sos);

        // ---- Dispatch: sequential or deterministic parallel -------------------
        let threads = self.params.threads.max(1);
        let aborted_at_root = self.abort_flag.load(AtomicOrdering::Relaxed);
        let outcome = if aborted_at_root {
            SearchOutcome { inc, timed_out: true, ..SearchOutcome::default() }
        } else if threads > 1 && !self.nested {
            self.solve_parallel(&work, sense, &int_vars, &all_sos, root_node, start, inc, threads)
        } else {
            self.run_search(&work, sense, &int_vars, &all_sos, vec![root_node], inc, start)
        };

        // ---- Finalise ----------------------------------------------------------
        self.last_nodes = outcome.nodes_explored;
        let status = if outcome.timed_out || self.limit_reached(outcome.nodes_explored) {
            SolverStatus::TimeLimit
        } else if outcome.inc.obj.is_some() {
            SolverStatus::Optimal
        } else {
            SolverStatus::Infeasible
        };
        self.last_status = status;
        self.incumbent_obj = outcome.inc.obj;
        self.incumbent_x = outcome.inc.x.as_ref().map(|x| x[..orig_n].to_vec());

        let (x, obj) = match (&outcome.inc.x, outcome.inc.obj) {
            (Some(x), Some(o)) => (x[..orig_n].to_vec(), o),
            _ => (vec![0.0; orig_n], 0.0),
        };
        let mut sol = Solution::new(x, obj, status);
        sol = sol.with_iterations(outcome.nodes_explored);
        sol = sol.with_solve_time(start.elapsed().as_secs_f64());
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
            self.pending_warm = Some(primal);
        }
        Ok(())
    }

    fn status(&self) -> SolverStatus {
        self.last_status
    }

    fn solution(&self) -> Option<Solution> {
        match (&self.incumbent_x, self.incumbent_obj) {
            (Some(x), Some(o)) => Some(
                Solution::new(x.clone(), o, SolverStatus::Optimal).with_iterations(self.last_nodes),
            ),
            _ => None,
        }
    }
}

impl MilpSolver {
    /// Deterministic parallel search: breadth-expand the root until at least
    /// `threads` subtrees exist, assign subtrees round-robin to scoped worker
    /// threads, and merge the outcomes (best objective; ties broken by the
    /// lexicographically smallest primal vector). The partition and each
    /// worker's traversal are fully deterministic, so results do not depend
    /// on thread scheduling.
    #[allow(clippy::too_many_arguments)]
    fn solve_parallel(
        &self,
        work: &Model,
        sense: Sense,
        int_vars: &[usize],
        sos: &[SosSet],
        root_node: Node,
        start: Instant,
        mut inc: Incumbent,
        threads: usize,
    ) -> SearchOutcome {
        // Breadth-first expand the frontier until >= threads nodes. Integral
        // or infeasible expansion results are dropped rather than pushed, so
        // the frontier can drain below `threads` (or empty entirely) — stop
        // as soon as that happens instead of popping from an empty vector.
        let mut frontier: Vec<Node> = vec![root_node];
        let tol = self.params.tolerances;
        while !frontier.is_empty() && frontier.len() < threads {
            // Take the shallowest node (FIFO) and expand it.
            let node = frontier.remove(0);
            let lp = solve_lp(work, &node.lb, &node.ub, tol);
            if lp.status != LpStatus::Optimal {
                continue;
            }
            if let Some((si, lo, mid, hi)) = find_sos_branch(sos, &lp.x, &node, tol.integrality) {
                let neutral = vec![1.0; work.num_vars];
                let kl = child_key(
                    self.selection,
                    sense,
                    lp.objective,
                    &lp.x,
                    int_vars,
                    &neutral,
                    &neutral,
                );
                let mut h = BinaryHeap::new();
                let mut s = Vec::new();
                push_children(
                    self.selection,
                    sense,
                    sos,
                    node,
                    lp.objective,
                    kl,
                    kl,
                    Some((si, lo, mid, hi)),
                    &mut h,
                    &mut s,
                );
                frontier.extend(h.into_vec());
                frontier.extend(s);
                continue;
            }
            let cands = self.fractional_candidates(&lp.x, int_vars, tol.integrality);
            if cands.is_empty() {
                if feasible_with_sos(sos, work, &lp.x, &node, tol.feasibility) {
                    inc.consider(lp.objective, lp.x.clone(), sense, self.params.absolute_gap);
                }
                continue;
            }
            let bv = cands[0];
            let fv = lp.x[bv];
            let neutral = vec![1.0; work.num_vars];
            let key =
                child_key(self.selection, sense, lp.objective, &lp.x, int_vars, &neutral, &neutral);
            let mut left = node.clone();
            left.ub[bv] = fv.floor().min(node.ub[bv]);
            left.bound = lp.objective;
            left.key = key;
            left.depth = node.depth + 1;
            left.parent = Some((bv, 0, fv - fv.floor()));
            let mut right = node.clone();
            right.lb[bv] = fv.ceil().max(node.lb[bv]);
            right.bound = lp.objective;
            right.key = key;
            right.depth = node.depth + 1;
            right.parent = Some((bv, 1, fv.ceil() - fv));
            frontier.push(left);
            frontier.push(right);
            if frontier.len() > 256 {
                break; // safety valve
            }
        }

        // Round-robin assignment: worker w gets frontier[w], frontier[w+T], ...
        let mut outcomes: Vec<SearchOutcome> = Vec::new();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for w in 0..threads {
                let seeds: Vec<Node> = frontier.iter().skip(w).step_by(threads).cloned().collect();
                let inc_w = if w == 0 { inc.clone() } else { Incumbent::default() };
                handles.push(scope.spawn(move || {
                    self.run_search(work, sense, int_vars, sos, seeds, inc_w, start)
                }));
            }
            for h in handles {
                if let Ok(o) = h.join() {
                    outcomes.push(o);
                }
            }
        });

        // Merge: best objective wins; ties broken by lexicographically
        // smallest primal vector for determinism.
        let mut merged = SearchOutcome {
            pseudo_up: outcomes.first().map(|o| o.pseudo_up.clone()).unwrap_or_default(),
            pseudo_down: outcomes.first().map(|o| o.pseudo_down.clone()).unwrap_or_default(),
            ..SearchOutcome::default()
        };
        merged.inc = inc;
        for o in &outcomes {
            merged.nodes_explored += o.nodes_explored;
            merged.timed_out |= o.timed_out;
            if let (Some(obj), Some(x)) = (o.inc.obj, &o.inc.x) {
                let replace = match merged.inc.obj {
                    None => true,
                    Some(cur) if (obj - cur).abs() <= self.params.absolute_gap => {
                        // Tie: prefer lexicographically smaller primal.
                        x < merged.inc.x.as_ref().unwrap()
                    }
                    Some(cur) => match sense {
                        Sense::Minimize => obj < cur,
                        Sense::Maximize => obj > cur,
                    },
                };
                if replace {
                    merged.inc.obj = Some(obj);
                    merged.inc.x = Some(x.clone());
                }
            }
        }
        merged
    }
}
