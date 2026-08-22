//! Benders decomposition for two-stage problems with (mixed-)integer
//! first-stage variables.
//!
//! The problem class is
//!
//! ```text
//! min  c·x + Σ_k w_k · Q_k(x)
//! s.t. x ∈ X  (bounds, optional integrality),
//!      Q_k(x) = min { d_k·y : A_k y ≥ β_k − Γ_k x, y ≥ 0 }
//! ```
//!
//! Each subproblem is canonicalised to uniform `≥` form ([`crate::common`])
//! and its **dual LP is built explicitly**, so every dual point π returned
//! is feasible by construction and `πᵀ(β − Γx)` is a valid global affine
//! under-estimator of `Q_k`:
//!
//! - *Optimality cut*: `θ_k ≥ πᵀβ − (Γᵀπ)·x`.
//! - *Feasibility cut* (phase-1 value > 0): the phase-1 optimal π satisfies
//!   `Aᵀπ ≤ 0`, hence any recourse-feasible `x` obeys `πᵀβ − (Γᵀπ)·x ≤ 0`.
//!
//! Extras: **Magnanti–Wong Pareto-optimal cuts** (re-optimise the dual
//! against a core point subject to staying optimal at `x̂`) and
//! **stabilisation** via an infinity-norm trust region or a level-set row
//! (both certified by a final unrestricted master solve).

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::{OptError, SolverStatus, VarBound};
use tpt_opt_milp::lp::{solve_lp, LpStatus};
use tpt_opt_milp::MilpSolver;

use crate::common::{canon_row, dot, solve_block_dual, CanonRow, RowSense};

/// One recourse row: `y-row · y + x-row · x ⋈ rhs`.
#[derive(Debug, Clone)]
pub struct BlockRow {
    /// Coefficients on the recourse variables.
    pub y: Vec<f64>,
    /// Coefficients on the first-stage variables.
    pub x: Vec<f64>,
    /// Row sense.
    pub sense: RowSense,
    /// Right-hand side.
    pub rhs: f64,
}

/// One recourse subproblem: `min d·y` over rows with `0 ≤ y ≤ u`
/// (`u_j = ∞` allowed).
#[derive(Debug, Clone)]
pub struct RecourseBlock {
    /// Recourse cost vector `d`.
    pub cost: Vec<f64>,
    /// Recourse rows.
    pub rows: Vec<BlockRow>,
    /// Upper bounds on the recourse variables (`f64::INFINITY` = unbounded).
    pub y_upper: Vec<f64>,
}

/// Stabilisation strategy for the master sequence.
#[derive(Debug, Clone, Copy)]
pub enum Stabilization {
    /// Plain Benders (default).
    None,
    /// Restrict each iterate to `‖x − centre‖∞ ≤ Δ`; expand on improvement,
    /// shrink otherwise. A final unrestricted solve certifies the bound.
    TrustRegion {
        /// Initial trust-region radius.
        initial_delta: f64,
        /// Maximum radius (the unrestricted problem).
        max_delta: f64,
    },
    /// Add `c·x ≤ level` with `level = UB − margin·|UB|`; relax to `UB` when
    /// the restricted master becomes infeasible. Certified by a final
    /// unrestricted solve.
    LevelSet {
        /// Margin fraction below the incumbent upper bound.
        margin_fraction: f64,
    },
}

/// A two-stage Benders problem.
#[derive(Debug, Clone)]
pub struct BendersProblem {
    /// First-stage cost `c`.
    pub first_cost: Vec<f64>,
    /// First-stage bounds `(lo, hi)` — must be finite.
    pub first_bounds: Vec<(f64, f64)>,
    /// Integrality flags for the first-stage variables.
    pub first_integer: Vec<bool>,
    /// Recourse blocks (scenarios / independent subproblems).
    pub blocks: Vec<RecourseBlock>,
    /// Expectation weights per block (normalised internally).
    pub weights: Vec<f64>,
}

/// Outcome of a Benders run.
#[derive(Debug, Clone)]
pub struct BendersResult {
    /// Best first-stage decision found.
    pub x: Vec<f64>,
    /// Recourse decisions at `x` (per block).
    pub recourse: Vec<Vec<f64>>,
    /// Best (incumbent) objective value.
    pub objective: f64,
    /// Final master lower bound.
    pub lower_bound: f64,
    /// `objective − lower_bound` at termination.
    pub gap: f64,
    /// Number of master iterations performed.
    pub iterations: usize,
    /// [`SolverStatus::Optimal`] when the gap closed; [`SolverStatus::TimeLimit`]
    /// when the iteration cap hit first.
    pub status: SolverStatus,
}

/// Per-block evaluation outcome at a trial point.
enum BlockEval {
    /// Recourse feasible: optimal value and an optimal dual π.
    Feasible { q: f64, pi: Vec<f64> },
    /// Recourse infeasible: Farkas certificate π from the phase-1 dual
    /// (satisfies `Aᵀπ ≤ 0`, `πᵀb(x̂) > 0`).
    Infeasible { farkas_pi: Vec<f64> },
}

/// Configurable Benders driver over [`BendersProblem`].
pub struct BendersSolver<'a> {
    problem: &'a BendersProblem,
    tolerance: f64,
    max_iterations: usize,
    theta_lower: Vec<f64>,
    stabilization: Stabilization,
    pareto_core: Option<Vec<f64>>,
}

impl<'a> BendersSolver<'a> {
    /// Create a solver with default settings (`tolerance = 1e-6`,
    /// `max_iterations = 200`, `theta_lower = 0` per block).
    pub fn new(problem: &'a BendersProblem) -> Self {
        let k = problem.blocks.len();
        Self {
            problem,
            tolerance: 1e-6,
            max_iterations: 200,
            theta_lower: vec![0.0; k],
            stabilization: Stabilization::None,
            pareto_core: None,
        }
    }

    /// Optimality gap tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Iteration cap.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Valid lower bounds for the epigraph variables `θ_k` (default `0`;
    /// set negative if recourse costs can be negative).
    pub fn with_theta_lower(mut self, theta_lower: Vec<f64>) -> Self {
        assert_eq!(theta_lower.len(), self.problem.blocks.len());
        self.theta_lower = theta_lower;
        self
    }

    /// Enable stabilisation.
    pub fn with_stabilization(mut self, s: Stabilization) -> Self {
        self.stabilization = s;
        self
    }

    /// Enable Magnanti–Wong Pareto cuts against the given core point
    /// (should lie in the relative interior of `X`).
    pub fn with_pareto_cuts(mut self, core_point: Vec<f64>) -> Self {
        assert_eq!(core_point.len(), self.problem.first_bounds.len());
        self.pareto_core = Some(core_point);
        self
    }

    fn canonical_block(&self, k: usize) -> (Vec<CanonRow>, usize) {
        let block = &self.problem.blocks[k];
        let n_y = block.cost.len();
        let mut rows: Vec<CanonRow> = Vec::new();
        for row in &block.rows {
            rows.extend(canon_row(&row.y, &row.x, row.sense, row.rhs));
        }
        // Variable upper bounds become extra ≥ rows: −y_j ≥ −u_j.
        for (j, &u) in block.y_upper.iter().enumerate() {
            if u.is_finite() {
                let mut a = vec![0.0; n_y];
                a[j] = -1.0;
                let gamma = vec![0.0; self.problem.first_bounds.len()];
                rows.push(CanonRow { a, gamma, beta: -u });
            }
        }
        (rows, n_y)
    }

    /// Evaluate all blocks at `x̂`.
    fn evaluate_blocks(&self, x_hat: &[f64]) -> Result<Vec<BlockEval>, OptError> {
        let mut out = Vec::with_capacity(self.problem.blocks.len());
        for k in 0..self.problem.blocks.len() {
            let (rows, n_y) = self.canonical_block(k);
            let a_mat: Vec<Vec<f64>> = rows.iter().map(|r| r.a.clone()).collect();
            let gamma: Vec<Vec<f64>> = rows.iter().map(|r| r.gamma.clone()).collect();
            let beta: Vec<f64> = rows.iter().map(|r| r.beta).collect();

            // Phase-1 dual: max πᵀb(x̂) s.t. Aᵀπ ≤ 0 (y-cols), π ≤ 1.
            let d_phase1 = vec![0.0; n_y];
            let p1 = solve_block_dual(&a_mat, &beta, &gamma, &d_phase1, x_hat, 1.0);
            if p1.status == LpStatus::Infeasible {
                return Err(OptError::invalid_model("recourse dual infeasible"));
            }
            if p1.objective > 1e-7 {
                out.push(BlockEval::Infeasible { farkas_pi: p1.x });
                continue;
            }

            // Feasible: optimality dual (bounded because the primal is).
            let opt =
                solve_block_dual(&a_mat, &beta, &gamma, &self.problem.blocks[k].cost, x_hat, 1e12);
            match opt.status {
                LpStatus::Optimal => out.push(BlockEval::Feasible { q: opt.objective, pi: opt.x }),
                _ => return Err(OptError::invalid_model("recourse unbounded or dual failed")),
            }
        }
        Ok(out)
    }

    /// Run the full Benders loop.
    pub fn solve(self) -> Result<BendersResult, OptError> {
        use tpt_opt_core::solver::Solver;
        let n1 = self.problem.first_bounds.len();
        let k_count = self.problem.blocks.len();
        assert_eq!(self.problem.weights.len(), k_count);
        let total_w: f64 = self.problem.weights.iter().sum();
        assert!(total_w > 0.0, "block weights must sum to a positive value");
        assert!(
            self.problem.first_bounds.iter().all(|(l, u)| l.is_finite() && u.is_finite()),
            "first-stage bounds must be finite"
        );

        // Pre-canonicalise blocks once.
        let canon: Vec<(Vec<CanonRow>, usize)> =
            (0..k_count).map(|k| self.canonical_block(k)).collect();

        // Cut storage: optimality cuts (block, α, g) meaning θ_k ≥ α + g·x;
        // feasibility cuts (α, g) meaning α + g·x ≤ 0.
        let mut opt_cuts: Vec<(usize, f64, Vec<f64>)> = Vec::new();
        let mut feas_cuts: Vec<(f64, Vec<f64>)> = Vec::new();

        let mut best_ub = f64::INFINITY;
        let mut best_x = vec![0.0; n1];
        let mut best_recourse: Vec<Vec<f64>> = vec![Vec::new(); k_count];
        let mut lb = f64::NEG_INFINITY;
        let mut iterations = 0usize;

        // Stabilisation state.
        let mut tr_delta = match self.stabilization {
            Stabilization::TrustRegion { initial_delta, .. } => initial_delta,
            _ => f64::INFINITY,
        };
        let tr_max = match self.stabilization {
            Stabilization::TrustRegion { max_delta, .. } => max_delta,
            _ => f64::INFINITY,
        };
        let mut level: Option<f64> = None;

        let status = loop {
            if iterations >= self.max_iterations {
                break SolverStatus::TimeLimit;
            }
            iterations += 1;

            // ---- Build and solve the master -------------------------------
            let mut model = Model::new(n1 + k_count);
            for (j, &(l, u)) in self.problem.first_bounds.iter().enumerate() {
                model.variables[j].bound = if self.problem.first_integer[j] {
                    VarBound::integer(l, u)
                } else {
                    VarBound::continuous(l, u)
                };
            }
            for (kk, tl) in self.theta_lower.iter().enumerate() {
                model.variables[n1 + kk].bound = VarBound::continuous(*tl, f64::INFINITY);
            }
            let mut oidx: Vec<usize> = Vec::new();
            let mut ocoeffs: Vec<f64> = Vec::new();
            for (j, &c) in self.problem.first_cost.iter().enumerate() {
                if c != 0.0 {
                    oidx.push(j);
                    ocoeffs.push(c);
                }
            }
            for (kk, &w) in self.problem.weights.iter().enumerate() {
                let wn = w / total_w;
                if wn != 0.0 {
                    oidx.push(n1 + kk);
                    ocoeffs.push(wn);
                }
            }
            model.set_objective(Objective {
                sense: Sense::Minimize,
                indices: oidx,
                coeffs: ocoeffs,
                constant: 0.0,
            });
            // Optimality cuts: θ_k − g·x ≥ α.
            for &(k, alpha, ref g) in &opt_cuts {
                let mut idx = vec![n1 + k];
                let mut co = vec![1.0];
                for (j, &gj) in g.iter().enumerate() {
                    if gj != 0.0 {
                        idx.push(j);
                        co.push(-gj);
                    }
                }
                model.add_constraint(Constraint::ge(idx, co, alpha));
            }
            // Feasibility cuts: g·x ≤ −α.
            for &(alpha, ref g) in &feas_cuts {
                let idx: Vec<usize> =
                    g.iter().enumerate().filter(|&(_, &v)| v != 0.0).map(|(j, _)| j).collect();
                let co: Vec<f64> = g.iter().copied().filter(|&v| v != 0.0).collect();
                model.add_constraint(Constraint::le(idx, co, -alpha));
            }
            // Stabilisation restrictions (tracked so we can certify later).
            let mut restricted = false;
            if tr_delta.is_finite() {
                restricted = true;
                for (j, (&xj, &(bl, bu))) in
                    best_x.iter().zip(self.problem.first_bounds.iter()).enumerate()
                {
                    let lo = (xj - tr_delta).max(bl);
                    let hi = (xj + tr_delta).min(bu);
                    model.variables[j].bound = if self.problem.first_integer[j] {
                        VarBound::integer(lo, hi)
                    } else {
                        VarBound::continuous(lo, hi)
                    };
                }
            }
            if let Some(lvl) = level {
                restricted = true;
                let idx: Vec<usize> =
                    (0..n1).filter(|&j| self.problem.first_cost[j] != 0.0).collect();
                let co: Vec<f64> = idx.iter().map(|&j| self.problem.first_cost[j]).collect();
                model.add_constraint(Constraint::le(idx, co, lvl));
            }

            let mut solver = MilpSolver::new();
            let sol = solver.solve(&model)?;
            match sol.status {
                SolverStatus::Infeasible => {
                    if restricted && level.is_some() {
                        // Level set too aggressive → relax to the incumbent.
                        level = Some(best_ub);
                        continue;
                    }
                    return Err(OptError::infeasible(
                        tpt_opt_core::error::InfeasibilityReport::new("Benders master infeasible"),
                    ));
                }
                SolverStatus::Unbounded => {
                    return Err(OptError::invalid_model("Benders master unbounded"));
                }
                _ => {}
            }
            lb = lb.max(sol.objective_value);
            let x_hat: Vec<f64> = sol.primal[..n1].to_vec();

            // ---- Evaluate subproblems -------------------------------------
            let evals = self.evaluate_blocks(&x_hat)?;
            let mut all_feasible = true;
            let mut ub_here = dot(&self.problem.first_cost, &x_hat);
            for (k, ev) in evals.iter().enumerate() {
                let (rows, n_y) = &canon[k];
                match ev {
                    BlockEval::Infeasible { farkas_pi, .. } => {
                        all_feasible = false;
                        // Farkas cut: any recourse-feasible x obeys
                        // πᵀβ − (Γᵀπ)·x ≤ 0, stored as α + g·x ≤ 0 with
                        // α = πᵀβ and g = −Γᵀπ.
                        let beta_v: Vec<f64> = rows.iter().map(|r| r.beta).collect();
                        let gamma_m: Vec<Vec<f64>> = rows.iter().map(|r| r.gamma.clone()).collect();
                        let alpha = dot(farkas_pi, &beta_v);
                        let g: Vec<f64> = (0..n1)
                            .map(|j| {
                                -gamma_m
                                    .iter()
                                    .enumerate()
                                    .map(|(r, gr)| farkas_pi[r] * gr[j])
                                    .sum::<f64>()
                            })
                            .collect();
                        feas_cuts.push((alpha, g));
                    }
                    BlockEval::Feasible { q, pi } => {
                        ub_here += self.problem.weights[k] / total_w * q;
                        // Optional Magnanti–Wong refinement against the core.
                        let pi_use = match (&self.pareto_core, n_y) {
                            (Some(core), _) if core != &x_hat => {
                                let a_mat: Vec<Vec<f64>> =
                                    rows.iter().map(|r| r.a.clone()).collect();
                                let gamma_m: Vec<Vec<f64>> =
                                    rows.iter().map(|r| r.gamma.clone()).collect();
                                let beta_v: Vec<f64> = rows.iter().map(|r| r.beta).collect();
                                pareto_dual(
                                    &a_mat,
                                    &beta_v,
                                    &gamma_m,
                                    &self.problem.blocks[k].cost,
                                    &x_hat,
                                    core,
                                    *q,
                                )?
                                .unwrap_or_else(|| pi.clone())
                            }
                            _ => pi.clone(),
                        };
                        best_recourse[k] =
                            primal_recourse(rows, *n_y, &self.problem.blocks[k].cost, &x_hat)?;
                        // Optimality cut θ_k ≥ πᵀβ − (Γᵀπ)·x, stored as
                        // α + g·x with α = πᵀβ and g = −Γᵀπ.
                        let beta_v: Vec<f64> = rows.iter().map(|r| r.beta).collect();
                        let gamma_m: Vec<Vec<f64>> = rows.iter().map(|r| r.gamma.clone()).collect();
                        let alpha = dot(&pi_use, &beta_v);
                        let g: Vec<f64> = (0..n1)
                            .map(|j| {
                                -gamma_m
                                    .iter()
                                    .enumerate()
                                    .map(|(r, gr)| pi_use[r] * gr[j])
                                    .sum::<f64>()
                            })
                            .collect();
                        opt_cuts.push((k, alpha, g));
                    }
                }
            }

            if all_feasible && ub_here < best_ub {
                best_ub = ub_here;
                best_x = x_hat.clone();
                if matches!(self.stabilization, Stabilization::TrustRegion { .. }) {
                    tr_delta = (tr_delta * 2.0).min(tr_max);
                }
                if let Stabilization::LevelSet { margin_fraction } = self.stabilization {
                    level = Some(best_ub - margin_fraction * best_ub.abs().max(1.0));
                }
            } else if matches!(self.stabilization, Stabilization::TrustRegion { .. }) {
                tr_delta = (tr_delta / 2.0).max(1e-6);
            }

            // Convergence: gap closed on an UNRESTRICTED master certifies
            // global optimality; under stabilisation, drop the restriction
            // and require one more confirming iteration. The finite-UB guard
            // matters: with no incumbent yet, `inf − lb ≤ inf·tol` would
            // otherwise compare true (`inf ≤ inf`).
            let gap_closed =
                best_ub.is_finite() && best_ub - lb <= self.tolerance * (1.0 + best_ub.abs());
            if gap_closed && !restricted {
                break SolverStatus::Optimal;
            }
            if gap_closed && restricted {
                tr_delta = f64::INFINITY;
                level = None;
            }
        };

        Ok(BendersResult {
            x: best_x,
            recourse: best_recourse,
            objective: best_ub,
            lower_bound: lb,
            gap: best_ub - lb,
            iterations,
            status,
        })
    }
}

/// Solve the Magnanti–Wong secondary LP: maximise `πᵀb(x_core)` subject to
/// dual feasibility and near-optimality at `x̂`. Returns `None` if the LP
/// fails (caller falls back to the primary dual).
fn pareto_dual(
    a: &[Vec<f64>],
    beta: &[f64],
    gamma: &[Vec<f64>],
    d: &[f64],
    x_hat: &[f64],
    core: &[f64],
    q_hat: f64,
) -> Result<Option<Vec<f64>>, OptError> {
    let m = beta.len();
    let mut model = Model::new(m);
    for v in model.variables.iter_mut() {
        v.bound = VarBound::continuous(0.0, 1e12);
    }
    for (j, &dj) in d.iter().enumerate() {
        let idx: Vec<usize> = (0..m).collect();
        let coeffs: Vec<f64> = (0..m).map(|r| a[r][j]).collect();
        model.add_constraint(Constraint::le(idx, coeffs, dj));
    }
    // Near-optimality at x̂: πᵀb(x̂) ≥ Q(x̂) − ε.
    let b_hat: Vec<f64> = beta.iter().zip(gamma.iter()).map(|(&b, g)| b - dot(g, x_hat)).collect();
    let idx: Vec<usize> = (0..m).filter(|&r| b_hat[r] != 0.0).collect();
    let coeffs: Vec<f64> = idx.iter().map(|&r| b_hat[r]).collect();
    model.add_constraint(Constraint::ge(idx, coeffs, q_hat - 1e-7));
    // Maximise πᵀb(core).
    let b_core: Vec<f64> = beta.iter().zip(gamma.iter()).map(|(&b, g)| b - dot(g, core)).collect();
    let oidx: Vec<usize> = (0..m).filter(|&r| b_core[r] != 0.0).collect();
    let ocoeffs: Vec<f64> = oidx.iter().map(|&r| b_core[r]).collect();
    model.set_objective(Objective {
        sense: Sense::Maximize,
        indices: oidx,
        coeffs: ocoeffs,
        constant: 0.0,
    });
    let lb = vec![0.0; m];
    let ub = vec![1e12; m];
    let sol = solve_lp(&model, &lb, &ub, tpt_opt_core::tolerance::Tolerances::spec_default());
    Ok(match sol.status {
        LpStatus::Optimal => Some(sol.x),
        _ => None,
    })
}

/// Re-solve the primal recourse LP at `x̂` to recover a `y` point for
/// reporting (feasibility already established).
fn primal_recourse(
    rows: &[CanonRow],
    n_y: usize,
    cost: &[f64],
    x_hat: &[f64],
) -> Result<Vec<f64>, OptError> {
    let mut model = Model::new(n_y);
    for v in model.variables.iter_mut() {
        v.bound = VarBound::continuous(0.0, f64::INFINITY);
    }
    for r in rows {
        let idx: Vec<usize> =
            r.a.iter().enumerate().filter(|&(_, &c)| c != 0.0).map(|(j, _)| j).collect();
        let co: Vec<f64> = r.a.iter().copied().filter(|&c| c != 0.0).collect();
        let rhs = r.beta - dot(&r.gamma, x_hat);
        model.add_constraint(Constraint::ge(idx, co, rhs));
    }
    let oidx: Vec<usize> = (0..n_y).filter(|&j| cost[j] != 0.0).collect();
    let ocoeffs: Vec<f64> = oidx.iter().map(|&j| cost[j]).collect();
    model.set_objective(Objective {
        sense: Sense::Minimize,
        indices: oidx,
        coeffs: ocoeffs,
        constant: 0.0,
    });
    let lb = vec![0.0; n_y];
    let ub = vec![f64::INFINITY; n_y];
    let sol = solve_lp(&model, &lb, &ub, tpt_opt_core::tolerance::Tolerances::spec_default());
    Ok(sol.x)
}
