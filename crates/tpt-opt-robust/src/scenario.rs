//! Scenario-based stochastic programming: two-stage and multi-stage
//! extensive forms.
//!
//! A **two-stage** program chooses first-stage decisions `x` here-and-now,
//! then observes a scenario `s` and takes recourse `y_s`:
//!
//! ```text
//! min  c1·x + Σ_s p_s · (c2_s·y_s)
//! s.t. x ∈ X,  W_s·x + T_s·y_s  ⋈  h_s   ∀s,  y_s ∈ Y
//! ```
//!
//! The extensive form is a single (MILP-)LP solved with
//! [`tpt_opt_milp::MilpSolver`].
//!
//! A **multi-stage** program generalises to T stages with one decision block
//! per *node* of the scenario tree; non-anticipativity is enforced by
//! creating variables per unique tree node (paths sharing a prefix share
//! their earlier-stage variables), so the formulation stays compact.

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::solver::Solver;
use tpt_opt_core::{OptError, VarBound};
use tpt_opt_milp::MilpSolver;

/// One scenario: a probability weight and an arbitrary data payload
/// (interpretation is up to the surrounding problem).
#[derive(Debug, Clone)]
pub struct Scenario {
    /// Probability of the scenario (need not be normalised on input; the
    /// extensive form normalises internally).
    pub probability: f64,
    /// Scenario data (e.g. realised demands, prices).
    pub data: Vec<f64>,
}

/// Second-stage data for one scenario.
///
/// Constraints are `W·x + T·y ⋈ h` with sense taken from `sense`
/// (`f64::NAN`-free: use `0.0` for `≤`, `1.0` for `≥`, `2.0` for `=` via
/// [`RowSense`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowSense {
    /// `≤`
    Le,
    /// `≥`
    Ge,
    /// `=`
    Eq,
}

/// A single recourse row for one scenario: `W-row·x + T-row·y ⋈ h`.
#[derive(Debug, Clone)]
pub struct StageRow {
    /// Coefficients on the first-stage variables `x` (length `n1`).
    pub w: Vec<f64>,
    /// Coefficients on the recourse variables `y_s` (length `n2`).
    pub t: Vec<f64>,
    /// Right-hand side.
    pub h: f64,
    /// Row sense.
    pub sense: RowSense,
}

/// Per-scenario second-stage description.
#[derive(Debug, Clone)]
pub struct StageData {
    /// Recourse objective coefficients `c2_s` (length `n2`).
    pub cost: Vec<f64>,
    /// Recourse rows.
    pub rows: Vec<StageRow>,
}

/// A two-stage stochastic linear program.
#[derive(Debug, Clone)]
pub struct TwoStageProblem {
    /// First-stage cost `c1` (length `n1`).
    pub first_cost: Vec<f64>,
    /// First-stage variable bounds `(lo, hi)`.
    pub first_bounds: Vec<(f64, f64)>,
    /// Recourse variable bounds `(lo, hi)` shared by all scenarios.
    pub second_bounds: Vec<(f64, f64)>,
    /// Scenarios with their second-stage data.
    pub scenarios: Vec<(Scenario, StageData)>,
}

/// Result of solving a two-stage program.
#[derive(Debug, Clone)]
pub struct TwoStageSolution {
    /// Optimal first-stage decision.
    pub x: Vec<f64>,
    /// Optimal objective (first-stage cost + expected recourse cost).
    pub objective: f64,
    /// Per-scenario recourse decisions (aligned with `scenarios`).
    pub recourse: Vec<Vec<f64>>,
}

impl TwoStageProblem {
    /// Build the deterministic-equivalent extensive-form model.
    ///
    /// Variable layout: `[x (n1) | y_0 (n2) | y_1 (n2) | ...]`.
    pub fn extensive_form(&self) -> Model {
        let n1 = self.first_bounds.len();
        let n2 = self.second_bounds.len();
        let total_p: f64 = self.scenarios.iter().map(|(s, _)| s.probability).sum();
        assert!(total_p > 0.0, "scenario probabilities must sum to a positive value");
        let n = n1 + n2 * self.scenarios.len();
        let mut model = Model::new(n);
        for (i, b) in self.first_bounds.iter().enumerate() {
            model.variables[i].bound = VarBound::continuous(b.0, b.1);
        }
        for s in 0..self.scenarios.len() {
            for (j, b) in self.second_bounds.iter().enumerate() {
                model.variables[n1 + s * n2 + j].bound = VarBound::continuous(b.0, b.1);
            }
        }
        // Objective: c1·x + Σ p̂_s c2_s·y_s (normalised probabilities).
        let mut idx = Vec::with_capacity(n);
        let mut coeffs = Vec::with_capacity(n);
        for (j, &c) in self.first_cost.iter().enumerate() {
            if c != 0.0 {
                idx.push(j);
                coeffs.push(c);
            }
        }
        let constant = 0.0f64;
        for (s, (scen, data)) in self.scenarios.iter().enumerate() {
            let p = scen.probability / total_p;
            for (j, &c) in data.cost.iter().enumerate() {
                if c != 0.0 {
                    idx.push(n1 + s * n2 + j);
                    coeffs.push(p * c);
                }
            }
        }
        model.set_objective(Objective { sense: Sense::Minimize, indices: idx, coeffs, constant });
        // Rows: W_s·x + T_s·y_s ⋈ h_s.
        for (s, (_, data)) in self.scenarios.iter().enumerate() {
            for row in &data.rows {
                let mut ridx = Vec::new();
                let mut rcoeffs = Vec::new();
                for (j, &w) in row.w.iter().enumerate() {
                    if w != 0.0 {
                        ridx.push(j);
                        rcoeffs.push(w);
                    }
                }
                for (j, &t) in row.t.iter().enumerate() {
                    if t != 0.0 {
                        ridx.push(n1 + s * n2 + j);
                        rcoeffs.push(t);
                    }
                }
                let con = match row.sense {
                    RowSense::Le => Constraint::le(ridx, rcoeffs, row.h),
                    RowSense::Ge => Constraint::ge(ridx, rcoeffs, row.h),
                    RowSense::Eq => Constraint::equality(ridx, rcoeffs, row.h),
                };
                model.add_constraint(con);
            }
        }
        model
    }

    /// Solve the extensive form with [`MilpSolver`].
    pub fn solve(&self) -> Result<TwoStageSolution, OptError> {
        let model = self.extensive_form();
        let mut solver = MilpSolver::new();
        let sol = solver.solve(&model)?;
        let n1 = self.first_bounds.len();
        let n2 = self.second_bounds.len();
        let recourse: Vec<Vec<f64>> = (0..self.scenarios.len())
            .map(|s| sol.primal[n1 + s * n2..n1 + (s + 1) * n2].to_vec())
            .collect();
        Ok(TwoStageSolution {
            x: sol.primal[..n1].to_vec(),
            objective: sol.objective_value,
            recourse,
        })
    }
}

/// Convenience free function mirroring [`TwoStageProblem::solve`].
pub fn solve_two_stage(problem: &TwoStageProblem) -> Result<TwoStageSolution, OptError> {
    problem.solve()
}

/// A multi-stage scenario tree given as full paths.
///
/// Each path is a sequence of per-stage data payloads; paths sharing a
/// prefix share their earlier-stage decision variables (non-anticipativity).
/// Stage `t` decisions at a node are `n_t` continuous variables with the
/// node's bounds; inter-stage linking constraints are supplied per path as
/// rows over the *path's* stage variables.
#[derive(Debug, Clone)]
pub struct ScenarioPath {
    /// Probability of reaching the leaf (normalised internally).
    pub probability: f64,
    /// Per-stage data payload (length = number of stages).
    pub data: Vec<Vec<f64>>,
}

/// Per-stage node bookkeeping: `(node id, variable bounds)` pairs.
type StageNodes = Vec<(usize, Vec<(f64, f64)>)>;

/// Build a multi-stage extensive-form model with prefix-merged
/// non-anticipativity.
///
/// The caller describes the model through callbacks:
///
/// - `stage_vars(t, data_t) -> (count, bounds)`: how many decision variables
///   stage `t` has at a node whose stage-`t` data is `data_t`, with bounds.
/// - `stage_cost(t, data_t) -> Vec<f64>`: objective coefficients for those
///   variables (length must match `stage_vars` count).
/// - `linking(t, data_path_prefix) -> Vec<(coeffs_by_stage_var, rhs, sense)>`:
///   rows coupling stage `t` variables to earlier stages at a node,
///   expressed as coefficient vectors over the *concatenated* stage-variable
///   block of the node (stages 0..=t).
///
/// Because callbacks receive the node's data prefix, the resulting model is
/// fully general over scenario trees.
pub fn multi_stage_model<FV, FC, FL>(
    paths: &[ScenarioPath],
    num_stages: usize,
    mut stage_vars: FV,
    mut stage_cost: FC,
    mut linking: FL,
) -> Result<Model, OptError>
where
    FV: FnMut(usize, &[f64]) -> Vec<(f64, f64)>,
    FC: FnMut(usize, &[f64]) -> Vec<f64>,
    FL: FnMut(usize, &[Vec<f64>]) -> Vec<(Vec<f64>, f64, RowSense)>,
{
    let total_p: f64 = paths.iter().map(|p| p.probability).sum();
    assert!(total_p > 0.0, "path probabilities must sum to a positive value");
    assert!(paths.iter().all(|p| p.data.len() == num_stages), "path data length != num_stages");

    // Build the node tree keyed by data-prefix equality; nodes are keyed by
    // their data prefix (Vec<Vec<f64>>).
    let mut node_keys: Vec<Vec<Vec<f64>>> = Vec::new(); // node id -> data prefix
    let mut node_stage: Vec<usize> = Vec::new();
    let mut node_children: Vec<Vec<(Vec<f64>, usize)>> = Vec::new();
    let mut node_prob: Vec<f64> = Vec::new();

    // Node 0 is the root (stage 0, empty prefix). Walking a path creates or
    // reuses one child per stage, keyed by that stage's data within the
    // parent; every visited node accumulates the path's probability so
    // non-leaf nodes carry their correct marginal weight.
    node_keys.push(Vec::new());
    node_stage.push(0);
    node_children.push(Vec::new());
    node_prob.push(0.0);
    for path in paths {
        let mut cur = 0usize;
        node_prob[0] += path.probability;
        for t in 0..num_stages {
            let marker = path.data[t].clone();
            let next = node_children[cur].iter().find(|(m, _)| *m == marker).map(|&(_, id)| id);
            cur = match next {
                Some(id) => id,
                None => {
                    let mut prefix = node_keys[cur].clone();
                    prefix.push(marker.clone());
                    node_keys.push(prefix);
                    node_stage.push(t + 1);
                    node_children.push(Vec::new());
                    node_prob.push(0.0);
                    let id = node_keys.len() - 1;
                    node_children[cur].push((marker, id));
                    id
                }
            };
            node_prob[cur] += path.probability;
        }
    }

    // Create variables per node: stage-t decisions live on nodes with
    // `node_stage == t`, using the node's own (last) data element.
    let mut node_var_start = vec![usize::MAX; node_keys.len()];
    let mut n_total = 0usize;
    let mut stage_counts: Vec<StageNodes> = vec![Vec::new(); num_stages + 1];
    for id in 0..node_keys.len() {
        let t = node_stage[id];
        if t == 0 {
            continue; // root carries no decisions
        }
        let data: &[f64] = node_keys[id].last().map(|v| v.as_slice()).unwrap_or(&[]);
        let bounds = stage_vars(t, data);
        node_var_start[id] = n_total;
        stage_counts[t].push((id, bounds.clone()));
        n_total += bounds.len();
    }

    let mut model = Model::new(n_total);
    let mut obj_idx: Vec<usize> = Vec::new();
    let mut obj_coeffs: Vec<f64> = Vec::new();
    for (t, nodes) in stage_counts.iter().enumerate().skip(1) {
        for &(id, ref bounds) in nodes {
            let start = node_var_start[id];
            for (k, b) in bounds.iter().enumerate() {
                model.variables[start + k].bound = VarBound::continuous(b.0, b.1);
            }
            let data: &[f64] = node_keys[id].last().map(|v| v.as_slice()).unwrap_or(&[]);
            let cost = stage_cost(t, data);
            assert_eq!(cost.len(), bounds.len(), "stage cost length != var count");
            // Weight by the node's accumulated probability.
            let p = node_prob[id] / total_p;
            for (k, &c) in cost.iter().enumerate() {
                if c != 0.0 {
                    obj_idx.push(start + k);
                    obj_coeffs.push(p * c);
                }
            }
        }
    }
    model.set_objective(Objective {
        sense: Sense::Minimize,
        indices: obj_idx,
        coeffs: obj_coeffs,
        constant: 0.0,
    });

    // Linking rows per node over its own concatenated prefix block.
    for id in 0..node_keys.len() {
        let t = node_stage[id];
        if t == 0 || node_var_start[id] == usize::MAX {
            continue;
        }
        let rows = linking(t, &node_keys[id]);
        for (mut coeffs, rhs, sense) in rows {
            // Coefficients run over the node's stage 1..=t variable blocks,
            // last segment first; map them by walking up the prefix chain.
            let mut ridx = Vec::new();
            let mut rcoeffs = Vec::new();
            let mut cursor = id;
            for stage in (1..=t).rev() {
                let cnt = {
                    let data: &[f64] =
                        node_keys[cursor].last().map(|v| v.as_slice()).unwrap_or(&[]);
                    stage_vars(stage, data).len()
                };
                assert!(coeffs.len() >= cnt, "linking row shorter than the stage block");
                let start = node_var_start[cursor];
                let seg = &coeffs[coeffs.len() - cnt..];
                for (k, &c) in seg.iter().enumerate() {
                    if c != 0.0 {
                        ridx.push(start + k);
                        rcoeffs.push(c);
                    }
                }
                coeffs.truncate(coeffs.len() - cnt);
                // Walk to the parent for the next-earlier stage block.
                if stage > 1 {
                    let mut parent = None;
                    for (pid, kids) in node_children.iter().enumerate() {
                        if kids.iter().any(|&(_, cid)| cid == cursor) {
                            parent = Some(pid);
                            break;
                        }
                    }
                    cursor = parent.unwrap_or(0);
                }
            }
            let con = match sense {
                RowSense::Le => Constraint::le(ridx, rcoeffs, rhs),
                RowSense::Ge => Constraint::ge(ridx, rcoeffs, rhs),
                RowSense::Eq => Constraint::equality(ridx, rcoeffs, rhs),
            };
            model.add_constraint(con);
        }
    }
    Ok(model)
}
