//! Lagrangian relaxation: subgradient optimisation, a stabilised
//! cutting-plane **bundle** method for the dual master, and surrogate
//! relaxation helpers.
//!
//! The dual function `L(λ) = min_{x∈X} c·x + λᵀ(Ax − b)` is concave and
//! piecewise linear; the caller supplies oracles returning `L(λ)` and any
//! subgradient `∂L/∂λ` (at a primal minimiser `x̂` the subgradient is
//! `Ax̂ − b`). Both drivers *maximise* `L` over `λ ≥ 0`.

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::VarBound;
use tpt_opt_milp::lp::{solve_lp, LpStatus};

/// Configuration shared by the dual optimisers.
#[derive(Debug, Clone)]
pub struct DualConfig {
    /// Iteration cap.
    pub max_iterations: usize,
    /// Initial step size (diminishing rule) / initial trust-region radius.
    pub initial_step: f64,
    /// Known dual target for Polyak steps (`None` ⇒ diminishing steps).
    pub target: Option<f64>,
    /// Convergence tolerance on the dual gap / master gap.
    pub tolerance: f64,
}

impl Default for DualConfig {
    fn default() -> Self {
        Self { max_iterations: 500, initial_step: 1.0, target: None, tolerance: 1e-6 }
    }
}

/// Outcome of a dual optimisation run.
#[derive(Debug, Clone)]
pub struct DualResult {
    /// Best multiplier vector found.
    pub lambda: Vec<f64>,
    /// Best dual value `L(λ)` found.
    pub value: f64,
    /// Evaluated values per iteration.
    pub history: Vec<f64>,
}

/// Classic subgradient ascent on the concave dual with either Polyak steps
/// (when [`DualConfig::target`] is set) or diminishing `t₀/√k` steps.
///
/// * `evaluate(λ)` returns `L(λ)`; `subgradient(λ)` returns any subgradient
///   `∂L/∂λ`.
pub fn lagrangian_subgradient<F, G>(
    lambda0: Vec<f64>,
    config: &DualConfig,
    mut evaluate: F,
    mut subgradient: G,
) -> DualResult
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    let mut lambda = lambda0.iter().map(|&v| v.max(0.0)).collect::<Vec<f64>>();
    let mut best_val = f64::NEG_INFINITY;
    let mut best_lambda = lambda.clone();
    let mut history = Vec::new();

    for k in 0..config.max_iterations {
        let val = evaluate(&lambda);
        history.push(val);
        if val > best_val {
            best_val = val;
            best_lambda = lambda.clone();
        }
        let g = subgradient(&lambda);
        let norm_sq: f64 = g.iter().map(|&v| v * v).sum();
        if norm_sq <= 1e-18 {
            break;
        }
        let step = match config.target {
            Some(target) => config.initial_step * (target - val) / norm_sq,
            None => config.initial_step / ((k + 1) as f64).sqrt(),
        };
        for (lj, gj) in lambda.iter_mut().zip(&g) {
            *lj = (*lj + step * gj).max(0.0);
        }
        if matches!(config.target, Some(t) if best_val >= t - config.tolerance) {
            break;
        }
    }

    DualResult { lambda: best_lambda, value: best_val, history }
}

/// Stabilised cutting-plane **bundle** method for the concave dual.
///
/// All evaluated affine cuts `η ≤ L_k + g_kᵀ(λ − λ_k)` are kept; each round
/// solves the cut master maximising `η` subject to an L1 trust region
/// `‖λ − centre‖₁ ≤ Δ`. On improvement the centre moves and `Δ` grows;
/// otherwise `Δ` shrinks around the best point. Termination requires the
/// master gap to close at an *interior* solution (a step pinned to the
/// trust-region boundary expands `Δ` instead of stopping, since the model
/// still wants to move).
pub fn lagrangian_bundle_level<F, G>(
    lambda0: Vec<f64>,
    config: &DualConfig,
    mut evaluate: F,
    mut subgradient: G,
) -> Result<DualResult, tpt_opt_core::OptError>
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    let mut centre = lambda0.iter().map(|&v| v.max(0.0)).collect::<Vec<f64>>();
    let mut cuts: Vec<(Vec<f64>, f64, Vec<f64>)> = Vec::new(); // (λ_k, L_k, g_k)
    let mut best_val = f64::NEG_INFINITY;
    let mut best_lambda = centre.clone();
    let mut history = Vec::new();
    let mut delta = config.initial_step.max(1e-9);

    for _ in 0..config.max_iterations {
        // Evaluate at the current centre; add its cut.
        let val = evaluate(&centre);
        let g = subgradient(&centre);
        history.push(val);
        if val > best_val {
            best_val = val;
            best_lambda = centre.clone();
        }
        cuts.push((centre.clone(), val, g));

        // Cut master: max η s.t. η ≤ L_k + g_kᵀ(λ − λ_k), ‖λ−centre‖₁ ≤ Δ.
        let (eta_star, lambda_star, at_boundary) = solve_bundle_master(&cuts, &centre, delta)?;
        let gap_closed = eta_star - best_val <= config.tolerance * (1.0 + best_val.abs());
        if gap_closed && !at_boundary {
            break;
        }
        // Adaptation: expand when the step is pinned to the boundary (the
        // model wants to travel further), shrink when progress stalls.
        if at_boundary {
            delta *= 2.0;
        } else if gap_closed || val <= best_val - config.tolerance {
            delta = (delta / 2.0).max(1e-9);
        }
        // Move the centre to the master point (bundle methods re-centre on
        // the candidate even without improvement — the cut pool retains all
        // information).
        centre = lambda_star;
    }

    Ok(DualResult { lambda: best_lambda, value: best_val, history })
}

/// Solve the stabilised cut master. Variables: `λ ≥ 0`, slacks `s ≥ 0`
/// (L1 deviations), free `η`. Rows: one cut row per bundle element plus two
/// deviation rows per coordinate plus the trust-region budget `Σs ≤ Δ`.
/// Returns `(η*, λ*, at_boundary)` where `at_boundary` indicates the chosen
/// λ sits on the trust-region sphere (`Σs ≈ Δ`).
fn solve_bundle_master(
    cuts: &[(Vec<f64>, f64, Vec<f64>)],
    centre: &[f64],
    delta: f64,
) -> Result<(f64, Vec<f64>, bool), tpt_opt_core::OptError> {
    let n = centre.len();
    let num_vars = 2 * n + 1; // λ_0..n-1, s_0..n-1, η
    let mut model = Model::new(num_vars);
    for j in 0..n {
        model.variables[j].bound = VarBound::continuous(0.0, f64::INFINITY);
        model.variables[n + j].bound = VarBound::continuous(0.0, f64::INFINITY);
    }
    model.variables[2 * n].bound = VarBound::continuous(f64::NEG_INFINITY, f64::INFINITY);

    // Cut rows: η − g_kᵀλ ≤ L_k − g_kᵀλ_k.
    for (lam_k, lk, gk) in cuts {
        let mut idx = vec![2 * n];
        let mut co = vec![1.0];
        for (j, &gj) in gk.iter().enumerate() {
            if gj != 0.0 {
                idx.push(j);
                co.push(-gj);
            }
        }
        let rhs = lk - dot(gk, lam_k);
        model.add_constraint(Constraint::le(idx, co, rhs));
    }
    // Deviation rows: |λ_j − centre_j| ≤ s_j.
    for (j, &cj) in centre.iter().enumerate() {
        model.add_constraint(Constraint::le(vec![j, n + j], vec![1.0, -1.0], cj));
        model.add_constraint(Constraint::le(vec![j, n + j], vec![-1.0, -1.0], -cj));
    }
    // Trust-region budget: Σ s_j ≤ Δ.
    let idx: Vec<usize> = (n..2 * n).collect();
    model.add_constraint(Constraint::le(idx, vec![1.0; n], delta));
    // Objective: maximise η.
    model.set_objective(Objective {
        sense: Sense::Maximize,
        indices: vec![2 * n],
        coeffs: vec![1.0],
        constant: 0.0,
    });

    let lb = vec![0.0; num_vars]
        .into_iter()
        .enumerate()
        .map(|(i, v)| if i == 2 * n { f64::NEG_INFINITY } else { v })
        .collect::<Vec<f64>>();
    let ub = vec![f64::INFINITY; num_vars];
    let sol = solve_lp(&model, &lb, &ub, tpt_opt_core::tolerance::Tolerances::spec_default());
    match sol.status {
        LpStatus::Optimal => {
            let slack_sum: f64 = sol.x[n..2 * n].iter().sum();
            let at_boundary = slack_sum >= delta - 1e-7 * (1.0 + delta);
            Ok((sol.x[2 * n], sol.x[..n].to_vec(), at_boundary))
        }
        _ => Err(tpt_opt_core::OptError::invalid_model("bundle master failed")),
    }
}

fn dot(u: &[f64], v: &[f64]) -> f64 {
    u.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum()
}

/// Coordinate-ascent search for a good surrogate multiplier vector `μ ≥ 0`:
/// repeatedly sweep coordinates, trying multiplicative perturbations and
/// keeping improvements. Deterministic.
///
/// * `evaluate(μ)` returns the surrogate relaxation value `S(μ)`
///   (maximised here).
pub fn surrogate_search<F>(n_mu: usize, config: &DualConfig, mut evaluate: F) -> (Vec<f64>, f64)
where
    F: FnMut(&[f64]) -> f64,
{
    let mut mu = vec![1.0; n_mu];
    let mut best = evaluate(&mu);
    for _ in 0..config.max_iterations.min(50) {
        let mut improved = false;
        for j in 0..n_mu {
            for &factor in &[0.5f64, 0.8, 1.25, 2.0] {
                let old = mu[j];
                mu[j] = (old * factor).max(0.0);
                let val = evaluate(&mu);
                if val > best + 1e-12 {
                    best = val;
                    improved = true;
                } else {
                    mu[j] = old;
                }
            }
        }
        if !improved {
            break;
        }
    }
    (mu, best)
}
