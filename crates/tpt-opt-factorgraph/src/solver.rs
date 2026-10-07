//! Sparse Gauss-Newton / Levenberg-Marquardt on a factor graph.
//!
//! Linearisation is numeric but **on-manifold**: each factor's Jacobian
//! block w.r.t. a key is central differences of the residual under tangent
//! perturbations (`retract`), so rotations stay on SO(2)/SO(3) and poses on
//! SE(2)/SE(3) while stepping. Blocks are whitened by the factor's noise
//! model and IRLS-scaled by its robust kernel, then assembled into a
//! block-sparse normal system solved by the fill-reducing-ordered sparse
//! `LDLᵀ` from `tpt-opt-nls` (or by a Schur complement when a point set is
//! given).
//!
//! Two entry points:
//!
//! * [`solve`] — batch: every factor re-linearised each iteration;
//! * [`solve_incremental`] — iSAM-style: only factors adjacent to the
//!   changed keys (or without a cache) are re-linearised; the remaining
//!   cached blocks are kept and their cached residuals are first-order
//!   corrected after every accepted step (`r̃ ← r̃ + J̃ δ`). A full
//!   Bayes-tree with fluid reordering remains future work.

use std::collections::BTreeMap;

use tpt_opt_nls::sparse::SymbolicLdl;
use tpt_opt_nls::NlsStatus;

use crate::factor::{Key, VarView};
use crate::graph::FactorGraph;
use crate::linear::BlockSystem;

/// Solver configuration (termination criteria mirror `tpt-opt-nls`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FactorSolverConfig {
    /// Relative cost-change tolerance.
    pub ftol: f64,
    /// Relative step-size tolerance.
    pub xtol: f64,
    /// Gradient (infinity-norm) tolerance on the weighted system.
    pub gtol: f64,
    /// Maximum outer iterations.
    pub max_iter: usize,
    /// Initial Levenberg damping.
    pub lambda0: f64,
    /// Finite-difference step for the on-manifold numeric Jacobian.
    pub fd_step: f64,
    /// Print per-iteration diagnostics to stderr.
    pub verbose: bool,
}

impl FactorSolverConfig {
    /// Defaults: `ftol = xtol = 1e-12`, `gtol = 1e-10`, 100 iterations,
    /// `lambda0 = 1e-3`, `fd_step = 1e-6`.
    pub fn new() -> Self {
        Self {
            ftol: 1e-12,
            xtol: 1e-12,
            gtol: 1e-10,
            max_iter: 100,
            lambda0: 1e-3,
            fd_step: 1e-6,
            verbose: false,
        }
    }

    /// Override the iteration budget.
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Override the gradient tolerance.
    pub fn with_gtol(mut self, v: f64) -> Self {
        self.gtol = v;
        self
    }
}

impl Default for FactorSolverConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a graph solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveOutcome {
    /// Termination status (reuses the `tpt-opt-nls` status vocabulary).
    pub status: NlsStatus,
    /// Outer iterations performed.
    pub iterations: usize,
    /// Factors re-linearised in total.
    pub linearizations: usize,
    /// Final IRLS-weighted cost.
    pub cost: f64,
    /// Final gradient infinity norm (weighted).
    pub gradient_norm: f64,
}

/// One factor's whitened, robust-scaled linearisation.
#[derive(Debug, Clone)]
pub struct FactorLin {
    /// Slot index (graph order).
    pub slot: usize,
    /// Whitened, robust-scaled residual (dim).
    pub r: Vec<f64>,
    /// `(key, dim × dof block)` per factor key.
    pub blocks: Vec<(Key, Vec<f64>)>,
}

/// Linearise one factor numerically on the manifold.
fn linearize_factor(
    graph: &FactorGraph,
    slot: usize,
    fd_step: f64,
) -> Result<FactorLin, crate::graph::GraphError> {
    let entry = &graph.factors().nth(slot).expect("slot exists");
    let keys: Vec<Key> = entry.factor.keys().to_vec();
    let views = graph.views(&keys)?;
    let dim = entry.factor.dim();

    // Unwhitened residual at the current point.
    let mut r0 = vec![0.0; dim];
    entry.factor.evaluate(&views, &mut r0);
    let rw = entry.noise.whiten(&r0);
    let weight = entry.noise.robust_weight(&rw, entry.robust);
    let scale = weight.sqrt();
    let r_scaled: Vec<f64> = rw.iter().map(|v| v * scale).collect();

    // Numeric on-manifold Jacobian per key: central differences through
    // retract (unwhitened), whitened afterwards (whitening is linear).
    let mut blocks = Vec::with_capacity(keys.len());
    for (i, view) in views.iter().enumerate() {
        let dof = view.dof();
        let mut j = vec![0.0; dim * dof];
        let mut rp = vec![0.0; dim];
        let mut rm = vec![0.0; dim];
        for d in 0..dof {
            let mut e = vec![0.0; dof];
            e[d] = fd_step;
            let xp = view.kind.retract(view.data, &e);
            let em: Vec<f64> = e.iter().map(|v| -v).collect();
            let xm = view.kind.retract(view.data, &em);
            let mut views_p: Vec<VarView<'_>> = views.to_vec();
            views_p[i] = VarView { key: view.key, kind: view.kind, data: &xp };
            let mut views_m: Vec<VarView<'_>> = views.to_vec();
            views_m[i] = VarView { key: view.key, kind: view.kind, data: &xm };
            entry.factor.evaluate(&views_p, &mut rp);
            entry.factor.evaluate(&views_m, &mut rm);
            for row in 0..dim {
                j[row * dof + d] = (rp[row] - rm[row]) / (2.0 * fd_step);
            }
        }
        let jw = entry.noise.whiten_jacobian(&j, dim, dof);
        let js: Vec<f64> = jw.iter().map(|v| v * scale).collect();
        blocks.push((keys[i], js));
    }

    Ok(FactorLin { slot, r: r_scaled, blocks })
}

/// Assemble the block system from a set of factor linearisations.
fn assemble(graph: &FactorGraph, lins: &[FactorLin]) -> BlockSystem {
    let n_keys = graph.values.len();
    let mut offsets = Vec::with_capacity(n_keys);
    let mut off = 0usize;
    for &key in graph.values.keys() {
        let dof = graph.values.kind(key).map(|m| m.dof()).unwrap_or(0);
        offsets.push((key, off, dof));
        off += dof;
    }
    let mut sys = BlockSystem {
        slots: offsets.clone(),
        h: BTreeMap::new(),
        g: vec![0.0; off],
        cost: 0.0,
        n: off,
    };
    for lin in lins {
        let dim = lin.r.len();
        let slot_of_key: std::collections::HashMap<Key, usize> =
            offsets.iter().enumerate().map(|(s, &(k, _, _))| (k, s)).collect();
        // Gradient contribution.
        for (key, j) in &lin.blocks {
            let s = slot_of_key[key];
            let (o, dof) = (offsets[s].1, offsets[s].2);
            for c in 0..dof {
                sys.g[o + c] += (0..dim).map(|r| j[r * dof + c] * lin.r[r]).sum::<f64>();
            }
        }
        // Cost contribution (½ r̃ᵀ r̃ already IRLS-weighted).
        sys.cost += 0.5 * lin.r.iter().map(|v| v * v).sum::<f64>();
        // Hessian blocks for all key pairs.
        for (i, (ki, ji)) in lin.blocks.iter().enumerate() {
            let si = slot_of_key[ki];
            let (oi, di) = (offsets[si].1, offsets[si].2);
            for (kj, jk) in lin.blocks.iter().skip(i) {
                let sj = slot_of_key[kj];
                let (ok, dk) = (offsets[sj].1, offsets[sj].2);
                let (lo_s, hi_s, lo_d, hi_d, transpose) =
                    if oi <= ok { (si, sj, di, dk, false) } else { (sj, si, dk, di, true) };
                let _ = (oi, ok);
                // block = J_iᵀ J_k (di × dk), optionally transposed into
                // the sorted-slot orientation.
                let entry = sys.h.entry((lo_s, hi_s)).or_insert_with(|| vec![0.0; lo_d * hi_d]);
                for a in 0..lo_d {
                    for b in 0..hi_d {
                        let v = if !transpose {
                            (0..dim).map(|r| ji[r * di + a] * jk[r * dk + b]).sum::<f64>()
                        } else {
                            (0..dim).map(|r| jk[r * dk + a] * ji[r * di + b]).sum::<f64>()
                        };
                        entry[a * hi_d + b] += v;
                    }
                }
            }
        }
    }
    sys
}

/// Freshly linearise every factor (no caches consulted or written).
pub fn linearize_all(
    graph: &FactorGraph,
    fd_step: f64,
) -> Result<Vec<FactorLin>, crate::graph::GraphError> {
    (0..graph.n_factors()).map(|slot| linearize_factor(graph, slot, fd_step)).collect()
}

/// Assemble the block system from factor linearisations (public for the
/// marginal-covariance module and tests).
pub fn assemble_pub(graph: &FactorGraph, lins: &[FactorLin]) -> BlockSystem {
    assemble(graph, lins)
}

/// Batch LM solve eliminating the given slots via a Schur complement in
/// the inner step (camera variables solved sparsely, points by
/// back-substitution). Eliminate slot indices refer to graph-key order.
pub fn solve_schur(
    graph: &mut FactorGraph,
    cfg: &FactorSolverConfig,
    eliminate_keys: &[Key],
) -> Result<SolveOutcome, crate::graph::GraphError> {
    let key_order: Vec<Key> = graph.values.keys().to_vec();
    let slots: Vec<usize> =
        eliminate_keys.iter().filter_map(|k| key_order.iter().position(|&a| a == *k)).collect();
    run_lm_inner(graph, cfg, None, &slots)
}

/// Shared LM driver. `linearize` decides batch vs incremental; schur_slots
/// routes the inner step through the Schur complement when non-empty.
fn run_lm_inner(
    graph: &mut FactorGraph,
    cfg: &FactorSolverConfig,
    relinearize: Option<&[Key]>,
    schur_slots: &[usize],
) -> Result<SolveOutcome, crate::graph::GraphError> {
    run_lm(graph, cfg, relinearize, schur_slots)
}

/// Shared LM driver (core).
fn run_lm(
    graph: &mut FactorGraph,
    cfg: &FactorSolverConfig,
    relinearize: Option<&[Key]>,
    schur_slots: &[usize],
) -> Result<SolveOutcome, crate::graph::GraphError> {
    // --- Gather factor linearisations (fresh or cached). ---
    let mut lins: Vec<FactorLin> = Vec::with_capacity(graph.n_factors());
    let mut count = 0usize;
    for slot in 0..graph.n_factors() {
        let cached = graph.factors().nth(slot).and_then(|f| f.cache.clone());
        let stale = match relinearize {
            None => true,
            Some(changed) => graph.stale_factor_indices(changed).contains(&slot),
        };
        if stale || cached.is_none() {
            let lin = linearize_factor(graph, slot, cfg.fd_step)?;
            count += 1;
            // Store in the cache.
            if let Some(f) = graph.factors_mut().get_mut(slot) {
                f.cache = Some(crate::graph::FactorCache {
                    r: lin.r.clone(),
                    j_blocks: lin.blocks.iter().map(|(k, j)| (*k, j.clone())).collect(),
                });
            }
            lins.push(lin);
        } else if let Some(c) = cached {
            lins.push(FactorLin { slot, r: c.r, blocks: c.j_blocks });
        }
    }
    let mut sys = assemble(graph, &lins);
    let mut gnorm = infinity_norm(&sys.g);
    if gnorm <= cfg.gtol {
        return Ok(SolveOutcome {
            status: NlsStatus::Converged,
            iterations: 0,
            linearizations: count,
            cost: sys.cost,
            gradient_norm: gnorm,
        });
    }

    // --- LM loop. ---
    let trace: f64 = (0..sys.slots.len())
        .map(|s| {
            let (_, dof) = sys.layout(s);
            sys.h.get(&(s, s)).map(|b| (0..dof).map(|r| b[r * dof + r]).sum::<f64>()).unwrap_or(0.0)
        })
        .sum::<f64>();
    let mut lambda = cfg.lambda0.max(1e-12) * (trace / sys.n as f64).max(1.0);
    let mut nu = 2.0;
    let mut iterations = 0usize;
    let mut sym: Option<SymbolicLdl> = None;
    loop {
        if iterations >= cfg.max_iter {
            return Ok(SolveOutcome {
                status: NlsStatus::MaxIterations,
                iterations,
                linearizations: count,
                cost: sys.cost,
                gradient_norm: gnorm,
            });
        }
        let maybe_delta = if schur_slots.is_empty() {
            sys.solve_step(lambda, &mut sym)
        } else {
            sys.solve_step_schur(schur_slots, lambda, &mut sym)
        };
        let Some(delta) = maybe_delta else {
            lambda *= nu;
            nu *= 2.0;
            if lambda > 1e12 {
                return Ok(SolveOutcome {
                    status: NlsStatus::NumericalIssue,
                    iterations,
                    linearizations: count,
                    cost: sys.cost,
                    gradient_norm: gnorm,
                });
            }
            continue;
        };
        let step_sq: f64 = delta.iter().map(|v| v * v).sum();
        let x_sq: f64 = graph
            .values
            .keys()
            .iter()
            .filter_map(|&k| graph.values.get(k).ok())
            .flat_map(|v| v.iter().map(|x| x * x))
            .sum();
        if step_sq.sqrt() <= cfg.xtol * (x_sq.sqrt() + cfg.xtol) {
            return Ok(SolveOutcome {
                status: NlsStatus::StepTolerance,
                iterations,
                linearizations: count,
                cost: sys.cost,
                gradient_norm: gnorm,
            });
        }
        // Predicted decrease on the undamped quadratic model.
        let gd: f64 = sys.g.iter().zip(&delta).map(|(a, b)| a * b).sum();
        let hd: f64 = {
            // H is stored by upper-triangle blocks: cross pairs enter
            // δᵀHδ twice.
            let mut s = 0.0;
            for (&(si, sj), block) in &sys.h {
                let (oi, di) = sys.layout(si);
                let (oj, dj) = sys.layout(sj);
                let w = if si == sj { 1.0 } else { 2.0 };
                for a in 0..di {
                    for b in 0..dj {
                        s += w * delta[oi + a] * block[a * dj + b] * delta[oj + b];
                    }
                }
            }
            0.5 * s
        };
        let pred = -(gd + hd);
        // Trial: retract every key by its delta slice, evaluate robust cost.
        let keys: Vec<Key> = graph.values.keys().to_vec();
        let mut new_values: Vec<Vec<f64>> = Vec::with_capacity(keys.len());
        for &k in &keys {
            let kind = graph.values.kind(k)?;
            let data = graph.values.get(k)?.to_vec();
            let (o, dof) = sys
                .slots
                .iter()
                .find(|(kk, _, _)| *kk == k)
                .map(|(_, o, d)| (*o, *d))
                .unwrap_or((usize::MAX, 0));
            if o == usize::MAX {
                new_values.push(data);
                continue;
            }
            new_values.push(kind.retract(&data, &delta[o..o + dof]));
        }
        let trial_cost = {
            // Swap values in, compute weighted cost factor-by-factor, swap back.
            let old: Vec<Vec<f64>> =
                keys.iter().map(|&k| graph.values.get(k).unwrap().to_vec()).collect();
            for (i, &k) in keys.iter().enumerate() {
                let _ = graph.values.set(k, new_values[i].clone());
            }
            let mut cost = 0.0;
            for slot in 0..graph.n_factors() {
                let entry = &graph.factors().nth(slot).unwrap();
                let views = graph.views(entry.factor.keys())?;
                let dim = entry.factor.dim();
                let mut r = vec![0.0; dim];
                entry.factor.evaluate(&views, &mut r);
                let rw = entry.noise.whiten(&r);
                let w = entry.noise.robust_weight(&rw, entry.robust);
                cost += 0.5 * w * rw.iter().map(|v| v * v).sum::<f64>();
            }
            for (i, &k) in keys.iter().enumerate() {
                let _ = graph.values.set(k, old[i].clone());
            }
            cost
        };
        let rho =
            if trial_cost < sys.cost && pred > 0.0 { (sys.cost - trial_cost) / pred } else { -1.0 };
        if cfg.verbose {
            eprintln!(
                "[lm] it {iterations} lambda {lambda:.3e} cost {:.6e} trial {trial_cost:.6e} pred {pred:.3e} rho {rho:.3} gnorm {gnorm:.3e} step {:.3e}",
                sys.cost,
                step_sq.sqrt()
            );
        }
        if rho > 0.0 {
            // Accept: apply the retraction, then refresh linearisations.
            for (i, &k) in keys.iter().enumerate() {
                graph.values.set(k, new_values[i].clone())?;
            }
            // First-order cache correction for factors we will NOT
            // re-linearise (incremental mode only).
            if let Some(changed) = relinearize {
                let changed = changed.to_vec();
                for slot in 0..graph.n_factors() {
                    let touches_changed = {
                        let f = graph.factors().nth(slot).unwrap();
                        f.factor.keys().iter().any(|k| changed.contains(k))
                    };
                    if touches_changed {
                        if let Some(f) = graph.factors_mut().get_mut(slot) {
                            f.cache = None;
                        }
                        continue;
                    }
                    // r̃ ← r̃ + Σ J̃ δ.
                    let mut cache = graph.factors().nth(slot).unwrap().cache.clone();
                    if let Some(c) = cache.as_mut() {
                        for (k, j) in &c.j_blocks {
                            let (o, dof) = sys
                                .slots
                                .iter()
                                .find(|(kk, _, _)| kk == k)
                                .map(|(_, o, d)| (*o, *d))
                                .unwrap_or((usize::MAX, 0));
                            if o == usize::MAX {
                                continue;
                            }
                            let dim = c.r.len();
                            for r in 0..dim {
                                c.r[r] +=
                                    (0..dof).map(|cc| j[r * dof + cc] * delta[o + cc]).sum::<f64>();
                            }
                        }
                    }
                    if let Some(f) = graph.factors_mut().get_mut(slot) {
                        f.cache = cache;
                    }
                }
            }
            iterations += 1;
            // Re-linearise and reassemble.
            let mut lins = Vec::with_capacity(graph.n_factors());
            for slot in 0..graph.n_factors() {
                let cached = graph.factors().nth(slot).and_then(|f| f.cache.clone());
                match cached {
                    Some(c) if relinearize.is_some() => {
                        lins.push(FactorLin { slot, r: c.r, blocks: c.j_blocks })
                    }
                    _ => {
                        let lin = linearize_factor(graph, slot, cfg.fd_step)?;
                        count += 1;
                        if let Some(f) = graph.factors_mut().get_mut(slot) {
                            f.cache = Some(crate::graph::FactorCache {
                                r: lin.r.clone(),
                                j_blocks: lin.blocks.iter().map(|(k, j)| (*k, j.clone())).collect(),
                            });
                        }
                        lins.push(lin);
                    }
                }
            }
            sys = assemble(graph, &lins);
            gnorm = infinity_norm(&sys.g);
            lambda *= (1.0 - (2.0 * rho - 1.0).powi(3)).max(1.0 / 3.0);
            lambda = lambda.max(1e-12);
            nu = 2.0;
            if gnorm <= cfg.gtol {
                return Ok(SolveOutcome {
                    status: NlsStatus::Converged,
                    iterations,
                    linearizations: count,
                    cost: sys.cost,
                    gradient_norm: gnorm,
                });
            }
            if pred <= cfg.ftol * sys.cost.max(1e-300) {
                return Ok(SolveOutcome {
                    status: NlsStatus::CostTolerance,
                    iterations,
                    linearizations: count,
                    cost: sys.cost,
                    gradient_norm: gnorm,
                });
            }
        } else {
            lambda *= nu;
            nu *= 2.0;
            if lambda > 1e12 {
                return Ok(SolveOutcome {
                    status: NlsStatus::CostTolerance,
                    iterations,
                    linearizations: count,
                    cost: sys.cost,
                    gradient_norm: gnorm,
                });
            }
        }
    }
}

fn infinity_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |a, &b| a.max(b.abs()))
}

/// Batch solve: Levenberg-Marquardt with full re-linearisation every
/// iteration and a fill-reducing-ordered sparse LDLᵀ inner solve.
pub fn solve(
    graph: &mut FactorGraph,
    cfg: &FactorSolverConfig,
) -> Result<SolveOutcome, crate::graph::GraphError> {
    run_lm_inner(graph, cfg, None, &[])
}

/// Incremental (iSAM-style) solve: only factors adjacent to `changed`
/// (or uncached) are re-linearised; cached blocks ride along with
/// first-order residual corrections. `graph.solve(...)` (batch) resets all
/// caches.
pub fn solve_incremental(
    graph: &mut FactorGraph,
    cfg: &FactorSolverConfig,
    changed: &[Key],
) -> Result<SolveOutcome, crate::graph::GraphError> {
    run_lm_inner(graph, cfg, Some(changed), &[])
}
