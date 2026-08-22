//! Lagrangian relaxation: subgradient optimisation, a cutting-plane
//! **bundle/level** method for the dual master, and surrogate relaxation
//! helpers.
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
    /// Initial step size (diminishing rule) / Polyak scale.
    pub initial_step: f64,
    /// Known dual target for Polyak steps (`None` ⇒ diminishing steps).
    pub target: Option<f64>,
    /// Convergence tolerance on the dual gap / level gap.
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

/// Cutting-plane **bundle/level** method: maintain every evaluated affine
/// cut of the concave dual, alternate between solving the cut master for an
/// upper bound `η*` and a proximity step to the level
/// `τ = LB + α(η* − LB)` (α = ½). Every master is an LP.
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
    let mut lambda = lambda0.iter().map(|&v| v.max(0.0)).collect::<Vec<f64>>();
    let mut cuts: Vec<(Vec<f64>, f64, Vec<f64>)> = Vec::new(); // (λ_k, L_k, g_k)
    let mut best_val = f64::NEG_INFINITY;
    let mut best_lambda = lambda.clone();
    let mut history = Vec::new();

    for _ in 0..config.max_iterations {
        let val = evaluate(&lambda);
        let g = subgradient(&lambda);
        history.push(val);
        if val > best_val {
            best_val = val;
            best_lambda = lambda.clone();
        }
        cuts.push((lambda.clone(), val, g));

        // Envelope master: max η over the accumulated cuts.
        let (eta_star, _) =
            solve_envelope(&cuts, None, None)?.ok_or_else(|| {
                tpt_opt_core::OptError::invalid_model("bundle master unbounded")
            })?;
        if eta_star - best_val <= config.tolerance {
            break;
        }
        // Level point: closest λ (L1) to the current iterate with envelope ≥ τ.
        let tau = best_val + 0.5 * (eta_star - best_val);
        match solve_envelope(&cuts, Some(tau), Some(&lambda))? {
            Some((_, next)) => lambda = next,
            None => break,
        }
    }

    Ok(DualResult { lambda: best_lambda, value: best_val, history })
}

/// Solve the cut-master LP over cuts `(λ_k, L_k, g_k)`.
///
/// * `level = None`: maximise `η` subject to `η ≤ L_k + g_kᵀ(λ − λ_k)`
///   for every cut and `λ ≥ 0`; returns `(η*, λ*)`.
/// * `level = Some(τ)` with `centre`: minimise `‖λ − centre‖₁` subject to
///   `η ≥ τ` plus the same cut rows; returns `(η, λ)`.
fn solve_envelope(
    cuts: &[(Vec<f64>, f64, Vec<f64>)],
    level: Option<f64>,
    centre: Option<&[f64]>,
) -> Result<Option<(f64, Vec<f64>)>, tpt_opt_core::OptError> {
    let n = cuts[0].0.len();
    let has_level = level.is_some();
    let num_vars = n + 1 + usize::from(has_level) * n;
    let mut model = Model::new(num_vars);
    for j in 0..n {
        model.variables[j].bound = VarBound::continuous(0.0, f64::INFINITY);
    }
    model.variables[n].bound = VarBound::continuous(f64::NEG_INFINITY, f64::INFINITY);
    if has_level {
        for j in 0..n {
            model.variables[n + 1 + j].bound = VarBound::continuous(0.0, f64::INFINITY);
        }
    }
    // Cut rows: η − g_kᵀλ ≤ L_k − g_kᵀλ_k.
    for (lam_k, lk, gk) in cuts {
        let mut idx = vec![n];
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
    if has_level {
        // Envelope floor: η ≥ τ.
        model.add_constraint(Constraint::ge(vec![n], vec![1.0], level.unwrap_or(0.0)));
        // L1 proximity: |λ_j − centre_j| ≤ s_j via two rows each.
        let centre = centre.unwrap_or(&[]);
        for j in 0..n {
            let cj = centre.get(j).copied().unwrap_or(0.0);
            model.add_constraint(Constraint::le(vec![j, n + 1 + j], vec![1.0, -1.0], cj));
            model.add_constraint(Constraint::le(vec![j, n + 1 + j], vec![-1.0, -1.0], -cj));
        }
        let idx: Vec<usize> = (n + 1..num_vars).collect();
        model.set_objective(Objective {
            sense: Sense::Minimize,
            indices: idx,
            coeffs: vec![1.0; n],
            constant: 0.0,
        });
    } else {
        model.set_objective(Objective {
            sense: Sense::Maximize,
            indices: vec![n],
            coeffs: vec![1.0],
            constant: 0.0,
        });
    }
    let lb = vec![0.0; num_vars]
        .into_iter()
        .enumerate()
        .map(|(i, v)| if i == n && !has_level { f64::NEG_INFINITY } else { v })
        .collect::<Vec<f64>>();
    let ub = vec![f64::INFINITY; num_vars];
    let sol = solve_lp(&model, &lb, &ub, tpt_opt_core::tolerance::Tolerances::spec_default());
    Ok(match sol.status {
        LpStatus::Optimal => Some((sol.x[n], sol.x[..n].to_vec())),
        _ => None,
    })
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