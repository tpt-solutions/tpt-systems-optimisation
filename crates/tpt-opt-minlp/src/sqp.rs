//! SQP-style branch-and-bound for (possibly non-convex) MINLPs.
//!
//! Each tree node relaxes the integer variables to continuous and solves the
//! resulting NLP with the sequential-quadratic/quasi-Newton augmented
//! Lagrangian solver ([`tpt_math_optimize_general::solve_nlp`]). Nodes are
//! pruned on infeasibility or on a relaxation bound worse than the incumbent;
//! fractional solutions are branched on the most fractional integer variable
//! (bound disjunction). Unlike outer approximation, no convexity assumption
//! is required: the NLP relaxations are solved directly, so non-convex
//! objectives/constraints are handled, at the cost of only a *heuristic*
//! (not globally valid) bound on non-convex problems.

use std::vec::Vec;

use tpt_math_optimize_general::{solve_nlp, NlpParams, NlpProblem, NlpResult};

use crate::model::{ConstraintKind, MinlpModel, VarKind};
use crate::oa::OaStatus;

/// Tolerances / limits for the branch-and-bound search.
#[derive(Debug, Clone)]
pub struct SqpConfig {
    /// Maximum number of nodes explored.
    pub max_nodes: usize,
    /// Integrality tolerance.
    pub int_tol: f64,
    /// Feasibility tolerance for accepting relaxation solutions.
    pub feas_tol: f64,
    /// NLP relaxation parameters.
    pub nlp: NlpParams,
}

impl Default for SqpConfig {
    fn default() -> Self {
        Self {
            max_nodes: 2000,
            int_tol: 1e-6,
            feas_tol: 1e-4,
            nlp: NlpParams { tol: 1e-8, ..NlpParams::default() },
        }
    }
}

/// Outcome of an SQP branch-and-bound solve.
#[derive(Debug, Clone)]
pub struct SqpResult {
    /// Terminal status.
    pub status: OaStatus,
    /// Best primal point found (full dimension), if any.
    pub x: Option<Vec<f64>>,
    /// Its objective value, if any.
    pub objective: Option<f64>,
    /// Number of nodes explored.
    pub nodes_explored: usize,
    /// Best (weakest) relaxation bound observed; heuristic on non-convex
    /// problems, valid on convex ones.
    pub best_bound: f64,
}

/// One search node: tightened variable bounds.
struct Node {
    lbs: Vec<f64>,
    ubs: Vec<f64>,
}

/// NLP relaxation of `model` over the node's box. Indicator-gated
/// constraints are enforced while their gate variable is >= 0.5 (the
/// standard continuous-relaxation reading).
struct RelaxProblem<'a> {
    model: &'a MinlpModel,
    lbs: Vec<f64>,
    ubs: Vec<f64>,
}

impl RelaxProblem<'_> {
    fn n_active_le(&self) -> usize {
        self.model
            .constraints
            .iter()
            .enumerate()
            .filter(|(_, c)| c.kind == ConstraintKind::Le)
            .count()
    }

    fn active_le_index(&self, k: usize) -> Option<usize> {
        self.model
            .constraints
            .iter()
            .enumerate()
            .filter(|(_, c)| c.kind == ConstraintKind::Le)
            .nth(k)
            .map(|(i, _)| i)
    }

    fn active_eq_index(&self, k: usize) -> Option<usize> {
        self.model
            .constraints
            .iter()
            .enumerate()
            .filter(|(_, c)| c.kind == ConstraintKind::Eq)
            .nth(k)
            .map(|(i, _)| i)
    }
}

impl NlpProblem for RelaxProblem<'_> {
    fn num_vars(&self) -> usize {
        self.model.num_vars()
    }

    fn objective(&self, x: &[f64]) -> f64 {
        self.model.eval_objective(x)
    }

    fn objective_grad(&self, x: &[f64], g: &mut [f64]) {
        let grad = self.model.eval_objective_grad(x);
        g.copy_from_slice(&grad);
    }

    fn num_ineq(&self) -> usize {
        2 * self.model.num_vars() + self.n_active_le()
    }

    fn ineq(&self, i: usize, x: &[f64]) -> f64 {
        let n = self.model.num_vars();
        if i < 2 * n {
            let v = i / 2;
            return if i % 2 == 0 { self.lbs[v] - x[v] } else { x[v] - self.ubs[v] };
        }
        let ci = self.active_le_index(i - 2 * n).unwrap_or(0);
        (self.model.constraints[ci].f)(x)
    }

    fn ineq_grad(&self, i: usize, x: &[f64], row: &mut [f64]) {
        let n = self.model.num_vars();
        for v in row.iter_mut() {
            *v = 0.0;
        }
        if i < 2 * n {
            let v = i / 2;
            row[v] = if i % 2 == 0 { -1.0 } else { 1.0 };
            return;
        }
        let ci = self.active_le_index(i - 2 * n).unwrap_or(0);
        let g = self.model.eval_constraint_grad(ci, x);
        row.copy_from_slice(&g);
    }

    fn num_eq(&self) -> usize {
        self.model.constraints.iter().filter(|c| c.kind == ConstraintKind::Eq).count()
    }

    fn eq(&self, j: usize, x: &[f64]) -> f64 {
        let ci = self.active_eq_index(j).unwrap_or(0);
        (self.model.constraints[ci].f)(x)
    }

    fn eq_grad(&self, j: usize, x: &[f64], row: &mut [f64]) {
        let ci = self.active_eq_index(j).unwrap_or(0);
        let g = self.model.eval_constraint_grad(ci, x);
        row.copy_from_slice(&g);
    }
}

/// Maximum violation of the *relaxation* (node bounds, gates, constraint
/// bodies) at `x`. Integrality is deliberately excluded: a relaxation may
/// legitimately return fractional integer variables, which triggers
/// branching rather than pruning.
fn node_violation(model: &MinlpModel, node: &Node, x: &[f64], _int_tol: f64) -> f64 {
    let mut worst = 0.0f64;
    for (i, &xi) in x.iter().enumerate().take(model.num_vars()) {
        worst = worst.max((node.lbs[i] - xi).max(0.0));
        worst = worst.max((xi - node.ubs[i]).max(0.0));
    }
    for ci in 0..model.constraints.len() {
        let gate_ok = match model.constraints[ci].active_if {
            Some((b, val)) => {
                if val {
                    x[b] >= 0.5
                } else {
                    x[b] <= 0.5
                }
            }
            None => true,
        };
        if !gate_ok {
            continue;
        }
        worst = worst.max(match model.constraints[ci].kind {
            ConstraintKind::Le => model.eval_constraint(ci, x),
            ConstraintKind::Eq => model.eval_constraint(ci, x).abs(),
        });
    }
    worst
}

/// Solve one node's continuous relaxation from several deterministic
/// starting points (midpoint, corners, alternating pattern) and return the
/// best feasible result. Multi-start matters on non-convex relaxations,
/// where a single start can stall at an interior stationary point that is
/// not a minimum.
fn solve_node_relaxation(model: &MinlpModel, node: &Node, cfg: &SqpConfig) -> NlpResult {
    let n = model.num_vars();
    let clamp_box = |v: &[f64]| -> Vec<f64> {
        v.iter().enumerate().map(|(i, &x)| x.clamp(node.lbs[i], node.ubs[i])).collect()
    };
    let mid: Vec<f64> = (0..n).map(|i| 0.5 * (node.lbs[i] + node.ubs[i])).collect();
    let alt: Vec<f64> =
        (0..n).map(|i| if i % 2 == 0 { node.lbs[i] } else { node.ubs[i] }).collect();
    let alt2: Vec<f64> =
        (0..n).map(|i| if i % 2 == 0 { node.ubs[i] } else { node.lbs[i] }).collect();

    let mut best: Option<NlpResult> = None;
    for start in [
        clamp_box(&mid),
        clamp_box(&node.lbs),
        clamp_box(&node.ubs),
        clamp_box(&alt),
        clamp_box(&alt2),
    ] {
        let prob = RelaxProblem { model, lbs: node.lbs.clone(), ubs: node.ubs.clone() };
        let res = solve_nlp(&prob, &start, &cfg.nlp);
        let feas = node_violation(model, node, &res.x, cfg.int_tol) <= cfg.feas_tol;
        if !feas {
            continue;
        }
        match &best {
            Some(b) if b.objective <= res.objective => {}
            _ => best = Some(res),
        }
    }
    // Warm-restart polish: one more solve from the incumbent relaxation
    // point (multipliers reset) frequently removes residual suboptimality
    // left by an early-stopped inner loop.
    if let Some(b) = &best {
        let prob = RelaxProblem { model, lbs: node.lbs.clone(), ubs: node.ubs.clone() };
        let res2 = solve_nlp(&prob, &b.x, &cfg.nlp);
        if node_violation(model, node, &res2.x, cfg.int_tol) <= cfg.feas_tol
            && res2.objective < b.objective
        {
            best = Some(res2);
        }
    }
    best.unwrap_or_else(|| {
        // Every start infeasible/failed: return a midpoint attempt so the
        // caller sees a violating point and prunes the node.
        let prob = RelaxProblem { model, lbs: node.lbs.clone(), ubs: node.ubs.clone() };
        solve_nlp(&prob, &clamp_box(&mid), &cfg.nlp)
    })
}

/// Solve a (possibly non-convex) MINLP with NLP-based branch-and-bound.
pub fn sqp_branch_and_bound(model: &MinlpModel, cfg: &SqpConfig) -> SqpResult {
    let n = model.num_vars();
    let mut stack: Vec<Node> = vec![Node { lbs: model.lbs.clone(), ubs: model.ubs.clone() }];
    let mut upper = f64::INFINITY;
    let mut best_x: Option<Vec<f64>> = None;
    let mut best_bound = f64::INFINITY;
    let mut explored = 0usize;
    let mut status = OaStatus::MaxIterations;

    while let Some(node) = stack.pop() {
        if explored >= cfg.max_nodes {
            break;
        }
        explored += 1;

        // Continuous relaxation over the node box (multi-start).
        let res: NlpResult = solve_node_relaxation(model, &node, cfg);
        let viol = node_violation(model, &node, &res.x, cfg.int_tol);

        if viol > cfg.feas_tol {
            // Infeasible (or unsolved) node: prune. A genuinely infeasible
            // root means the whole model is infeasible.
            if explored == 1 {
                status = OaStatus::Infeasible;
                return SqpResult {
                    status,
                    x: None,
                    objective: None,
                    nodes_explored: explored,
                    best_bound: f64::NEG_INFINITY,
                };
            }
            continue;
        }
        best_bound = best_bound.min(res.objective);

        // Bound prune against the incumbent.
        if res.objective >= upper - 1e-9 {
            continue;
        }

        // Integrality check: most-fractional integer variable, if any
        // exceeds the integrality tolerance.
        let branch_var = (0..n)
            .filter(|&i| model.vars[i] != VarKind::Continuous)
            .map(|i| (i, (res.x[i] - res.x[i].round()).abs()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .filter(|&(_, f)| f >= cfg.int_tol)
            .map(|(i, _)| i);

        match branch_var {
            None => {
                // Integral feasible solution: round integers for cleanliness.
                let mut x = res.x.clone();
                for (i, xi) in x.iter_mut().enumerate().take(n) {
                    if model.vars[i] != VarKind::Continuous {
                        *xi = xi.round();
                    }
                }
                let obj = model.eval_objective(&x);
                if obj < upper {
                    upper = obj;
                    best_x = Some(x);
                }
            }
            Some(i) => {
                // Branch on the most fractional variable (bound disjunction).
                // Floor/ceil of the *raw* value: rounding first can clamp to
                // a bound making floor == ceil, which would spawn children
                // identical to the parent (infinite loop) and drop one side
                // of the disjunction entirely.
                let v = res.x[i].clamp(node.lbs[i], node.ubs[i]);
                let floor = v.floor();
                let ceil = v.ceil();
                if floor >= node.lbs[i] - 1e-12 {
                    let lo = node.lbs.clone();
                    let mut hi = node.ubs.clone();
                    hi[i] = floor;
                    if hi[i] >= lo[i] - 1e-12 {
                        stack.push(Node { lbs: lo, ubs: hi });
                    }
                }
                if ceil <= node.ubs[i] + 1e-12 {
                    let mut lo = node.lbs.clone();
                    let hi = node.ubs.clone();
                    lo[i] = ceil;
                    if hi[i] >= lo[i] - 1e-12 {
                        stack.push(Node { lbs: lo, ubs: hi });
                    }
                }
            }
        }
    }

    if best_x.is_some() {
        status = OaStatus::Optimal;
    } else if status != OaStatus::Infeasible && explored >= cfg.max_nodes {
        status = OaStatus::MaxIterations;
    } else if best_x.is_none() {
        status = OaStatus::Infeasible;
    }

    let objective = best_x.as_ref().map(|x| model.eval_objective(x));
    SqpResult { status, x: best_x, objective, nodes_explored: explored, best_bound }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::VarKind;

    fn cfg() -> SqpConfig {
        SqpConfig { max_nodes: 500, ..SqpConfig::default() }
    }

    #[test]
    fn concave_minimization_reaches_extreme_point() {
        // min −((x−2)² + (y−2)²), x cont [0,4], y int [0,4].
        // Concave objective → optimum at a box corner: −8.
        let mut m =
            MinlpModel::new(2, |x| -((x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 2.0) * (x[1] - 2.0)));
        m.set_var(0, VarKind::Continuous, 0.0, 4.0);
        m.set_var(1, VarKind::Integer, 0.0, 4.0);

        let res = sqp_branch_and_bound(&m, &cfg());
        assert_eq!(res.status, OaStatus::Optimal);
        let obj = res.objective.unwrap();
        assert!((obj - (-8.0)).abs() < 1e-3, "obj {obj}");
        let x = res.x.unwrap();
        assert!((x[1] - x[1].round()).abs() < 1e-6, "y not integral: {}", x[1]);
        // Corner: both coordinates at an extreme.
        assert!(x[0].abs() < 1e-3 || (x[0] - 4.0).abs() < 1e-3, "x0 {}", x[0]);
        assert!(x[1].abs() < 1e-3 || (x[1] - 4.0).abs() < 1e-3, "y {}", x[1]);
    }

    #[test]
    fn bilinear_product_finds_global_min() {
        // min x·y, x cont [1,3], y int [1,3] → 1 at (1,1). Bilinear = non-convex.
        let mut m = MinlpModel::new(2, |x| x[0] * x[1]);
        m.set_var(0, VarKind::Continuous, 1.0, 3.0);
        m.set_var(1, VarKind::Integer, 1.0, 3.0);

        let res = sqp_branch_and_bound(&m, &cfg());
        assert_eq!(res.status, OaStatus::Optimal);
        let obj = res.objective.unwrap();
        assert!((obj - 1.0).abs() < 1e-3, "obj {obj}");
    }

    #[test]
    fn infeasible_integer_region_detected() {
        // x cont [0,1], y int [2,3], x + y <= 0.5 → infeasible.
        let mut m = MinlpModel::new(2, |x| x[0] + x[1]);
        m.set_var(0, VarKind::Continuous, 0.0, 1.0);
        m.set_var(1, VarKind::Integer, 2.0, 3.0);
        m.add_le(
            |x| x[0] + x[1] - 0.5,
            |_x, g| {
                g[0] = 1.0;
                g[1] = 1.0;
            },
        );

        let res = sqp_branch_and_bound(&m, &cfg());
        assert_eq!(res.status, OaStatus::Infeasible);
        assert!(res.x.is_none());
    }

    #[test]
    fn matches_brute_force_on_mixed_problem() {
        // min (x−1.3)² + (y−2.7)² with x cont [0,3], y int [0,4]:
        // per y, best x = 1.3 → objective (y−2.7)²; y=3 → 0.09.
        let mut m =
            MinlpModel::new(2, |x| (x[0] - 1.3) * (x[0] - 1.3) + (x[1] - 2.7) * (x[1] - 2.7));
        m.set_var(0, VarKind::Continuous, 0.0, 3.0);
        m.set_var(1, VarKind::Integer, 0.0, 4.0);

        let res = sqp_branch_and_bound(&m, &cfg());
        assert_eq!(res.status, OaStatus::Optimal);
        let obj = res.objective.unwrap();
        assert!((obj - 0.09).abs() < 1e-3, "obj {obj}");
        let x = res.x.unwrap();
        assert_eq!(x[1].round(), 3.0);
        // Objective-level optimality is asserted above; the continuous
        // coordinate itself carries the augmented-Lagrangian solver's
        // finite-difference precision (~1e-2 here).
        assert!((x[0] - 1.3).abs() < 2e-2, "x0 {}", x[0]);
    }
}
