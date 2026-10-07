//! Standard factors: prior, between, range-bearing, reprojection.

use crate::factor::{Factor, Key, VarView};
use crate::manifold::{wrap_angle, Manifold};

/// Prior factor: penalise deviation of `x` from a fixed value `prior`,
/// measured in tangent coordinates: `r = local(prior, x)`, so `x = prior`
/// gives zero.
///
/// The prior value uses the manifold's storage layout; the residual
/// dimension equals the manifold's tangent dimension.
#[derive(Debug, Clone)]
pub struct PriorFactor {
    key: [Key; 1],
    prior: Vec<f64>,
    kind: Manifold,
}

impl PriorFactor {
    /// A prior on `key` at `prior` (storage layout of `kind`).
    pub fn new(key: Key, kind: Manifold, prior: Vec<f64>) -> Self {
        Self { key: [key], prior, kind }
    }
}

impl Factor for PriorFactor {
    fn keys(&self) -> &[Key] {
        &self.key
    }

    fn dim(&self) -> usize {
        self.kind.dof()
    }

    fn evaluate(&self, vars: &[VarView<'_>], out: &mut [f64]) {
        let v = &vars[0];
        let r = v.kind.local(&self.prior, v.data);
        out.copy_from_slice(&r[..out.len()]);
    }

    fn name(&self) -> &'static str {
        "Prior"
    }
}

/// Between factor (odometry / loop closure): `r = local(measured,
/// local(x₁, x₂))` — the tangent mismatch between the measured relative
/// transform `x₁⁻¹ ∘ x₂` and the measurement.
///
/// `measured` uses the same storage layout as the variables.
#[derive(Debug, Clone)]
pub struct BetweenFactor {
    key: [Key; 2],
    measured: Vec<f64>,
    kind: Manifold,
}

impl BetweenFactor {
    /// A between measurement on two same-type variables.
    pub fn new(key1: Key, key2: Key, kind: Manifold, measured: Vec<f64>) -> Self {
        Self { key: [key1, key2], measured, kind }
    }
}

impl Factor for BetweenFactor {
    fn keys(&self) -> &[Key] {
        &self.key
    }

    fn dim(&self) -> usize {
        self.kind.dof()
    }

    fn evaluate(&self, vars: &[VarView<'_>], out: &mut [f64]) {
        let (a, b) = (&vars[0], &vars[1]);
        // Group-level compare: x₁⁻¹∘x₂ is a group element (storage layout),
        // then `local(measured, ·)` maps the mismatch to tangent coords.
        let rel = a.kind.compose(&a.kind.inverse(a.data), b.data);
        let r = a.kind.local(&self.measured, &rel);
        out.copy_from_slice(&r[..out.len()]);
    }

    fn name(&self) -> &'static str {
        "Between"
    }
}

/// 2-D range-bearing measurement from an SE(2) pose to an Euclidean(2)
/// point landmark: `r = [wrap(bearing_pred − bearing_meas),
/// range_pred − range_meas]` (bearing first, then range), with the bearing
/// measured from the body-frame +x axis: `bearing = atan2(dy_body, dx_body)`.
#[derive(Debug, Clone)]
pub struct RangeBearingFactor {
    key: [Key; 2],
    /// Measured `(bearing, range)`.
    measured: [f64; 2],
}

impl RangeBearingFactor {
    /// A range-bearing measurement `[bearing, range]`.
    pub fn new(pose_key: Key, point_key: Key, measured: [f64; 2]) -> Self {
        Self { key: [pose_key, point_key], measured }
    }
}

impl Factor for RangeBearingFactor {
    fn keys(&self) -> &[Key] {
        &self.key
    }

    fn dim(&self) -> usize {
        2
    }

    fn evaluate(&self, vars: &[VarView<'_>], out: &mut [f64]) {
        let pose = &vars[0];
        let point = &vars[1];
        // Pose is SE(2) (x, y, θ); point is Euclidean(2).
        let (px, py, theta) = (pose.data[0], pose.data[1], pose.data[2]);
        let (lx, ly) = (point.data[0], point.data[1]);
        let (c, s) = (theta.cos(), theta.sin());
        let dx = c * (lx - px) + s * (ly - py);
        let dy = -s * (lx - px) + c * (ly - py);
        out[0] = wrap_angle(dy.atan2(dx) - self.measured[0]);
        out[1] = (dx * dx + dy * dy).sqrt() - self.measured[1];
    }

    fn name(&self) -> &'static str {
        "RangeBearing"
    }
}

/// Pinhole reprojection factor: SE(3) camera pose (world→camera, i.e. the
/// projection is `R p_w + t`) plus an Euclidean(3) landmark, projecting to
/// `[u, v]` pixels with intrinsics `(f_x, f_y, c_x, c_y)`:
/// `u = f_x·X/Z + c_x`, `v = f_y·Y/Z + c_y`.
#[derive(Debug, Clone)]
pub struct ReprojectionFactor {
    key: [Key; 2],
    /// Measured pixel `[u, v]`.
    measured: [f64; 2],
    /// Focal lengths `(f_x, f_y)`.
    pub focal: [f64; 2],
    /// Principal point `(c_x, c_y)`.
    pub principal: [f64; 2],
}

impl ReprojectionFactor {
    /// A reprojection measurement.
    pub fn new(
        pose_key: Key,
        point_key: Key,
        measured: [f64; 2],
        focal: [f64; 2],
        principal: [f64; 2],
    ) -> Self {
        Self { key: [pose_key, point_key], measured, focal, principal }
    }
}

impl Factor for ReprojectionFactor {
    fn keys(&self) -> &[Key] {
        &self.key
    }

    fn dim(&self) -> usize {
        2
    }

    fn evaluate(&self, vars: &[VarView<'_>], out: &mut [f64]) {
        let pose = &vars[0];
        let point = &vars[1];
        // Pose is SE(3) (t, q): world→camera, p_cam = R p_w + t.
        let t = [pose.data[0], pose.data[1], pose.data[2]];
        let q = [pose.data[3], pose.data[4], pose.data[5], pose.data[6]];
        let p_w = [point.data[0], point.data[1], point.data[2]];
        let r = crate::manifold::quat_rotate(&q, &p_w);
        let z = (r[2] + t[2]).max(1e-9);
        out[0] = self.focal[0] * (r[0] + t[0]) / z + self.principal[0] - self.measured[0];
        out[1] = self.focal[1] * (r[1] + t[1]) / z + self.principal[1] - self.measured[1];
    }

    fn name(&self) -> &'static str {
        "Reprojection"
    }
}
