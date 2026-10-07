//! The factor graph container: keyed variables plus boxed factors with
//! per-factor noise models, robust kernels, and linearisation caches for
//! incremental (iSAM-style) updates.

use tpt_opt_nls::Loss;

use crate::factor::{Factor, Key, VarView};
use crate::manifold_values::{KeyError, KeyedValues};
use crate::noise::NoiseModel;

/// Errors from graph operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// A referenced key is missing from the values.
    MissingKey(Key),
    /// Key/value bookkeeping failed.
    KeyError(KeyError),
    /// The factor's declared dimension disagrees with its keys/manifold.
    BadFactor(&'static str),
}

impl From<KeyError> for GraphError {
    fn from(e: KeyError) -> Self {
        GraphError::KeyError(e)
    }
}

/// Cached whitened linearisation of one factor (used by incremental
/// solves).
#[derive(Debug, Clone)]
pub struct FactorCache {
    /// Whitened, robust-scaled residual.
    pub r: Vec<f64>,
    /// Whitened, robust-scaled Jacobian block per factor key (dim × dof).
    pub j_blocks: Vec<(Key, Vec<f64>)>,
}

/// One factor with its noise model and optional robust kernel.
#[derive(Debug)]
pub struct FactorEntry {
    /// The factor itself.
    pub factor: Box<dyn Factor>,
    /// Gaussian whitening applied to the residual.
    pub noise: NoiseModel,
    /// Optional robust kernel (IRLS weight on the whitened residual).
    pub robust: Option<Loss>,
    /// Cached linearisation from the most recent solve, for incremental
    /// updates (`None` → re-linearise).
    pub(crate) cache: Option<FactorCache>,
}

/// A factor graph: variables (`KeyedValues`) plus ordered factors.
#[derive(Debug, Default)]
pub struct FactorGraph {
    /// Variable values by key.
    pub values: KeyedValues,
    factors: Vec<FactorEntry>,
}

impl FactorGraph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a variable.
    pub fn add_variable(
        &mut self,
        key: Key,
        kind: crate::manifold::Manifold,
        data: Vec<f64>,
    ) -> Result<(), GraphError> {
        self.values.insert(key, kind, data)?;
        Ok(())
    }

    /// Add a factor with its noise model and optional robust kernel.
    /// The factor's keys must all exist.
    pub fn add_factor(
        &mut self,
        factor: Box<dyn Factor>,
        noise: NoiseModel,
        robust: Option<Loss>,
    ) -> Result<(), GraphError> {
        for &k in factor.keys() {
            if !self.values.contains(k) {
                return Err(GraphError::MissingKey(k));
            }
        }
        self.factors.push(FactorEntry { factor, noise, robust, cache: None });
        Ok(())
    }

    /// Number of factors.
    pub fn n_factors(&self) -> usize {
        self.factors.len()
    }

    /// Factors (read-only).
    pub fn factors(&self) -> impl Iterator<Item = &FactorEntry> {
        self.factors.iter()
    }

    /// Overwrite a variable's value; caches touching that key are dropped
    /// so the next incremental solve re-linearises the affected factors.
    pub fn set_value(&mut self, key: Key, data: Vec<f64>) -> Result<(), GraphError> {
        self.values.set(key, data)?;
        for f in self.factors.iter_mut() {
            if f.factor.keys().contains(&key) {
                f.cache = None;
            }
        }
        Ok(())
    }

    /// Build variable views for a factor's keys; `Err` on a missing key.
    pub fn views(&self, keys: &[Key]) -> Result<Vec<VarView<'_>>, GraphError> {
        keys.iter()
            .map(|&k| {
                let kind = self.values.kind(k)?;
                let data = self.values.get(k)?;
                Ok(VarView { key: k, kind, data })
            })
            .collect()
    }

    /// Total plain whitened error `½ Σ ‖r̃‖²` (robust kernels do not enter
    /// this metric). Recomputes from scratch.
    pub fn error(&self) -> Result<f64, GraphError> {
        let mut total = 0.0;
        for entry in &self.factors {
            let keys = entry.factor.keys();
            let views = self.views(keys)?;
            let dim = entry.factor.dim();
            let mut r = vec![0.0; dim];
            entry.factor.evaluate(&views, &mut r);
            let rw = entry.noise.whiten(&r);
            total += 0.5 * rw.iter().map(|v| v * v).sum::<f64>();
        }
        Ok(total)
    }

    /// Access to the factor list for the solver (linearise + cache update).
    pub(crate) fn factors_mut(&mut self) -> &mut Vec<FactorEntry> {
        &mut self.factors
    }

    /// Replace the robust kernel of factor `index` (graph order), dropping
    /// its linearisation cache. The typical robust workflow is two-stage:
    /// batch-solve without kernels, then enable them and re-solve from the
    /// warm start (IRLS from a cold, drifted start down-weights the very
    /// constraints that would correct it).
    pub fn set_factor_robust(
        &mut self,
        index: usize,
        robust: Option<Loss>,
    ) -> Result<(), GraphError> {
        let entry = self.factors.get_mut(index).ok_or(GraphError::MissingKey(index))?;
        entry.robust = robust;
        entry.cache = None;
        Ok(())
    }

    /// Slots (graph order) of factors whose keys intersect `changed`, plus
    /// every factor with no cache.
    pub(crate) fn stale_factor_indices(&self, changed: &[Key]) -> Vec<usize> {
        self.factors
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                f.cache.is_none() || f.factor.keys().iter().any(|k| changed.contains(k))
            })
            .map(|(i, _)| i)
            .collect()
    }
}
