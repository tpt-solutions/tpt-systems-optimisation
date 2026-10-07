//! Factor trait and variable views.

use crate::manifold::Manifold;

/// A variable key (unique per factor graph).
pub type Key = usize;

/// A read-only view of one variable's value plus its manifold type.
#[derive(Debug, Clone, Copy)]
pub struct VarView<'a> {
    /// Variable key.
    pub key: Key,
    /// Which manifold the storage should be interpreted on.
    pub kind: Manifold,
    /// Storage slice (`kind.dim()` long).
    pub data: &'a [f64],
}

impl VarView<'_> {
    /// The variable's tangent-space dimension.
    pub fn dof(&self) -> usize {
        self.kind.dof()
    }
}

/// A factor: a cost term over a fixed set of keys, `r(v_{k₁}, …, v_{k_m})`.
///
/// Implementations write the **unwhitened** residual into `out`; noise
/// models and robust kernels are applied by the solver (see
/// [`crate::noise::NoiseModel`]).
pub trait Factor: std::fmt::Debug + Send + Sync {
    /// The keys this factor connects, in a fixed order matching the
    /// Jacobian blocks.
    fn keys(&self) -> &[Key];

    /// Dimension of the (unwhitened) residual.
    fn dim(&self) -> usize;

    /// Evaluate the residual at the given variable views (order matches
    /// [`Factor::keys`]).
    fn evaluate(&self, vars: &[VarView<'_>], out: &mut [f64]);

    /// Human-readable name for diagnostics.
    fn name(&self) -> &'static str {
        "Factor"
    }
}
