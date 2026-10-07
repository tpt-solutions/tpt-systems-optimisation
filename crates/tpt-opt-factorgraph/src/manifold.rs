//! Manifold variable types: Euclidean vectors, SO(2), SE(2), SO(3), SE(3).
//!
//! Variables live in plain `f64` storage (`Vec<f64>`); each [`Manifold`]
//! knows how to interpret that storage as a Lie-group element and provides
//! the operations the solvers need:
//!
//! * `dof` — tangent-space dimension (the optimisation dimension);
//! * `retract(x, δ)` — `x ⊕ δ`: move `x` by a tangent step (right
//!   multiplication by the exponential of δ);
//! * `local(a, b)` — `b ⊖ a`: the tangent vector that carries `a` to `b`;
//!   inverse of `retract` in the second argument.
//!
//! These two satisfy `retract(a, local(a, b)) = b`, which is all the
//! numeric Jacobian machinery requires. Storage conventions:
//!
//! | manifold   | storage dim | tangent dim | storage layout |
//! |------------|-------------|-------------|----------------|
//! | Euclidean  | n           | n           | plain vector   |
//! | SO(2)      | 1           | 1           | angle θ        |
//! | SE(2)      | 3           | 3           | (x, y, θ)      |
//! | SO(3)      | 4           | 3           | quaternion (x, y, z, w) — scalar last |
//! | SE(3)      | 7           | 6           | (t_x, t_y, t_z, q_x, q_y, q_z, q_w) |
//!
//! SE(3) tangents are (v; ω) with translation first, rotation second; the
//! exponential/log maps use the standard V-matrix formulation.

use std::f64::consts::PI;

/// The closed set of manifold types a variable can live on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Manifold {
    /// `ℝⁿ` with the flat metric; `n` is the storage/tangent dimension.
    Euclidean(usize),
    /// Planar rotations, stored as the angle θ.
    So2,
    /// Planar rigid transforms, stored as `(x, y, θ)`.
    Se2,
    /// 3-D rotations, stored as unit quaternions `(x, y, z, w)`.
    So3,
    /// 3-D rigid transforms, stored as `(t, q)` — translation then quaternion.
    Se3,
}

/// Wrap an angle into (−π, π].
pub fn wrap_angle(a: f64) -> f64 {
    let mut x = (a + PI) % (2.0 * PI);
    if x <= 0.0 {
        x += 2.0 * PI;
    }
    x - PI
}

/// Quaternion product `a ∘ b` (apply b first, then a); inputs (x, y, z, w).
pub fn quat_mul(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    let (ax, ay, az, aw) = (a[0], a[1], a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// Rotate a 3-vector by a unit quaternion ( sandwich-free direct formula).
pub fn quat_rotate(q: &[f64; 4], v: &[f64; 3]) -> [f64; 3] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let [vx, vy, vz] = *v;
    // v + 2 q⃗ × (q⃗ × v + w v)
    let (ux, uy, uz) =
        (w * vx + y * vz - z * vy, w * vy + z * vx - x * vz, w * vz + x * vy - y * vx);
    [vx + 2.0 * (y * uz - z * uy), vy + 2.0 * (z * ux - x * uz), vz + 2.0 * (x * uy - y * ux)]
}

/// SO(3) exponential: rotation vector `ω = θ·axis` → unit quaternion.
pub fn quat_exp(w: &[f64; 3]) -> [f64; 4] {
    let norm = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    if norm < 1e-12 {
        // First-order: q ≈ [ω/2, 1], normalised.
        let mut q = [w[0] * 0.5, w[1] * 0.5, w[2] * 0.5, 1.0];
        let n = (q.iter().map(|v| v * v).sum::<f64>()).sqrt();
        for v in &mut q {
            *v /= n;
        }
        return q;
    }
    let s = (norm * 0.5).sin() / norm;
    [w[0] * s, w[1] * s, w[2] * s, (norm * 0.5).cos()]
}

/// SO(3) logarithm: unit quaternion → rotation vector.
pub fn quat_log(q: &[f64; 4]) -> [f64; 3] {
    let mut q = *q;
    let n = (q.iter().map(|v| v * v).sum::<f64>()).sqrt();
    for v in &mut q {
        *v /= n;
    }
    if q[3] < 0.0 {
        // Take the short arc.
        for v in &mut q {
            *v = -*v;
        }
    }
    let vnorm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
    if vnorm < 1e-12 {
        // θ ≈ 2‖v‖, axis ill-defined: ω ≈ 2v.
        return [2.0 * q[0], 2.0 * q[1], 2.0 * q[2]];
    }
    let theta = 2.0 * vnorm.atan2(q[3]);
    let s = theta / vnorm;
    [q[0] * s, q[1] * s, q[2] * s]
}

/// SE(2) logarithm: pose `(t_x, t_y, θ)` → tangent `(v_x, v_y, ω)`
/// (inverse of [`se2_exp`]'s V-matrix: `V⁻¹ = adj(V)/det(V)`).
pub fn se2_log(t: &[f64; 3]) -> [f64; 3] {
    let (tx, ty, w) = (t[0], t[1], t[2]);
    if w.abs() < 1e-10 {
        return [tx, ty, w];
    }
    let s = w.sin() / w;
    let c = (1.0 - w.cos()) / w;
    let det = s * s + c * c;
    [(s * tx + c * ty) / det, (-c * tx + s * ty) / det, w]
}

/// SE(2) exponential: tangent `(v_x, v_y, ω)` → pose `(t_x, t_y, θ)`.
pub fn se2_exp(v: &[f64; 3]) -> [f64; 3] {
    let (vx, vy, w) = (v[0], v[1], v[2]);
    if w.abs() < 1e-10 {
        return [vx, vy, w];
    }
    let (s, c) = (w.sin(), w.cos());
    let a = s / w;
    let b = (1.0 - c) / w;
    [a * vx - b * vy, b * vx + a * vy, w]
}

/// SE(3) exponential: twist `(v; ω)` → `(t, q)`.
pub fn se3_exp(twist: &[f64; 6]) -> ([f64; 3], [f64; 4]) {
    let v = [twist[0], twist[1], twist[2]];
    let w = [twist[3], twist[4], twist[5]];
    let q = quat_exp(&w);
    let theta = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    let t = if theta < 1e-10 {
        v
    } else {
        // t = (V v) with V = sin(θ)/θ I + (1−cosθ)/θ² [ω] + (1−sin(θ)/θ)/θ² ωωᵀ.
        let (s, c) = (theta.sin(), theta.cos());
        let a = s / theta;
        let b = (1.0 - c) / (theta * theta);
        let f = (1.0 - a) / (theta * theta);
        let dot = w[0] * v[0] + w[1] * v[1] + w[2] * v[2];
        [
            a * v[0] + b * (w[1] * v[2] - w[2] * v[1]) + f * w[0] * dot,
            a * v[1] + b * (w[2] * v[0] - w[0] * v[2]) + f * w[1] * dot,
            a * v[2] + b * (w[0] * v[1] - w[1] * v[0]) + f * w[2] * dot,
        ]
    };
    (t, q)
}

/// SE(3) logarithm: `(t, q)` → twist `(v; ω)`.
pub fn se3_log(t: &[f64; 3], q: &[f64; 4]) -> [f64; 6] {
    let w = quat_log(q);
    let theta = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    let v = if theta < 1e-10 {
        *t
    } else {
        // V⁻¹ = I − ½[ω] + (1/θ² − (1+cosθ)/(2θ sinθ))·[ω]², and
        // [ω]²t = ω(ωᵀt) − θ²t.
        let (s, c) = (theta.sin(), theta.cos());
        let k = 1.0 / (theta * theta) - (1.0 + c) / (2.0 * theta * s);
        let cr = [w[1] * t[2] - w[2] * t[1], w[2] * t[0] - w[0] * t[2], w[0] * t[1] - w[1] * t[0]];
        let dot = w[0] * t[0] + w[1] * t[1] + w[2] * t[2];
        [
            t[0] - 0.5 * cr[0] + k * (w[0] * dot - theta * theta * t[0]),
            t[1] - 0.5 * cr[1] + k * (w[1] * dot - theta * theta * t[1]),
            t[2] - 0.5 * cr[2] + k * (w[2] * dot - theta * theta * t[2]),
        ]
    };
    [v[0], v[1], v[2], w[0], w[1], w[2]]
}

impl Manifold {
    /// Tangent-space dimension.
    pub fn dof(&self) -> usize {
        match self {
            Manifold::Euclidean(n) => *n,
            Manifold::So2 => 1,
            Manifold::Se2 => 3,
            Manifold::So3 => 3,
            Manifold::Se3 => 6,
        }
    }

    /// Storage dimension.
    pub fn dim(&self) -> usize {
        match self {
            Manifold::Euclidean(n) => *n,
            Manifold::So2 => 1,
            Manifold::Se2 => 3,
            Manifold::So3 => 4,
            Manifold::Se3 => 7,
        }
    }

    /// A canonical identity element for this manifold.
    pub fn identity(&self) -> Vec<f64> {
        match self {
            Manifold::Euclidean(n) => vec![0.0; *n],
            Manifold::So2 => vec![0.0],
            Manifold::Se2 => vec![0.0; 3],
            Manifold::So3 => vec![0.0, 0.0, 0.0, 1.0],
            Manifold::Se3 => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Group composition `a ∘ b` (b first, then a).
    pub fn compose(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        match self {
            Manifold::Euclidean(_) => a.iter().zip(b.iter()).map(|(x, y)| x + y).collect(),
            Manifold::So2 => vec![wrap_angle(a[0] + b[0])],
            Manifold::Se2 => {
                let (c, s) = (a[2].cos(), a[2].sin());
                vec![
                    a[0] + c * b[0] - s * b[1],
                    a[1] + s * b[0] + c * b[1],
                    wrap_angle(a[2] + b[2]),
                ]
            }
            Manifold::So3 => {
                let qa = [a[0], a[1], a[2], a[3]];
                let qb = [b[0], b[1], b[2], b[3]];
                quat_mul(&qa, &qb).to_vec()
            }
            Manifold::Se3 => {
                let ta = [a[0], a[1], a[2]];
                let tb = [b[0], b[1], b[2]];
                let qa = [a[3], a[4], a[5], a[6]];
                let qb = [b[3], b[4], b[5], b[6]];
                let rotated = quat_rotate(&qa, &tb);
                let q = quat_mul(&qa, &qb);
                vec![
                    ta[0] + rotated[0],
                    ta[1] + rotated[1],
                    ta[2] + rotated[2],
                    q[0],
                    q[1],
                    q[2],
                    q[3],
                ]
            }
        }
    }

    /// Group inverse.
    pub fn inverse(&self, a: &[f64]) -> Vec<f64> {
        match self {
            Manifold::Euclidean(n) => a.iter().map(|v| -v).take(*n).collect(),
            Manifold::So2 => vec![wrap_angle(-a[0])],
            Manifold::Se2 => {
                let (c, s) = (a[2].cos(), a[2].sin());
                vec![-c * a[0] - s * a[1], s * a[0] - c * a[1], wrap_angle(-a[2])]
            }
            Manifold::So3 => vec![-a[0], -a[1], -a[2], a[3]],
            Manifold::Se3 => {
                // T⁻¹ = (−Rᵀt, q*).
                let q = [a[3], a[4], a[5], a[6]];
                let qc = [-q[0], -q[1], -q[2], q[3]];
                let t = [a[0], a[1], a[2]];
                let r = quat_rotate(&qc, &t); // Rᵀt = R(q*)t
                vec![-r[0], -r[1], -r[2], qc[0], qc[1], qc[2], qc[3]]
            }
        }
    }

    /// Retraction `x ⊕ δ`: compose `x` with `exp(δ)`.
    pub fn retract(&self, x: &[f64], delta: &[f64]) -> Vec<f64> {
        match self {
            Manifold::Euclidean(_) => {
                let mut out = x.to_vec();
                for (o, d) in out.iter_mut().zip(delta.iter()) {
                    *o += d;
                }
                out
            }
            Manifold::So2 => vec![wrap_angle(x[0] + delta[0])],
            Manifold::Se2 => {
                let step = se2_exp(&[delta[0], delta[1], delta[2]]);
                self.compose(x, &step)
            }
            Manifold::So3 => {
                let step = quat_exp(&[delta[0], delta[1], delta[2]]);
                self.compose(x, &step)
            }
            Manifold::Se3 => {
                let mut dw = [0.0f64; 6];
                dw.copy_from_slice(delta);
                let (t, q) = se3_exp(&dw);
                let step = [t[0], t[1], t[2], q[0], q[1], q[2], q[3]];
                self.compose(x, &step)
            }
        }
    }

    /// Local coordinates `b ⊖ a` with `retract(a, local(a, b)) = b`.
    pub fn local(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        match self {
            Manifold::Euclidean(_) => b.iter().zip(a.iter()).map(|(x, y)| x - y).collect(),
            Manifold::So2 => vec![wrap_angle(b[0] - a[0])],
            Manifold::Se2 => {
                let rel = self.compose(&self.inverse(a), b);
                se2_log(&[rel[0], rel[1], rel[2]]).to_vec()
            }
            Manifold::So3 => {
                let qa = [a[0], a[1], a[2], a[3]];
                let qb = [b[0], b[1], b[2], b[3]];
                let rel = quat_mul(&[-qa[0], -qa[1], -qa[2], qa[3]], &qb);
                quat_log(&rel).to_vec()
            }
            Manifold::Se3 => {
                let inv = self.inverse(a);
                let rel = self.compose(&inv, b);
                let t = [rel[0], rel[1], rel[2]];
                let q = [rel[3], rel[4], rel[5], rel[6]];
                se3_log(&t, &q).to_vec()
            }
        }
    }

    /// Normalise storage in place semantics (e.g. quaternion unit length);
    /// returns a normalised copy.
    pub fn normalized(&self, x: &[f64]) -> Vec<f64> {
        match self {
            Manifold::So3 => {
                let n = (x.iter().map(|v| v * v).sum::<f64>()).sqrt();
                if n < 1e-300 {
                    return self.identity();
                }
                x.iter().map(|v| v / n).collect()
            }
            Manifold::Se3 => {
                let mut out = x.to_vec();
                let n = (out[3..7].iter().map(|v| v * v).sum::<f64>()).sqrt();
                if n >= 1e-300 {
                    for v in out[3..7].iter_mut() {
                        *v /= n;
                    }
                } else {
                    out[3..7].copy_from_slice(&[0.0, 0.0, 0.0, 1.0]);
                }
                out
            }
            _ => x.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn so3_exp_log_roundtrip() {
        // ‖ω‖ ≤ π: beyond π the quaternion has two preimages and log picks
        // the short arc, so the roundtrip is only the identity there.
        for w in [[0.1, 0.2, 0.3], [0.5, -0.5, 1.2], [1e-9, -1e-9, 1e-9], [1.5, 0.1, -2.0]] {
            let q = quat_exp(&w);
            let w2 = quat_log(&q);
            assert!(close(&w2, &w, 1e-9), "{w:?} -> {q:?} -> {w2:?}");
        }
    }

    #[test]
    fn se2_exp_matches_composition() {
        // exp(a) ∘ exp(b) has local coords ≈ b + a only for commuting parts;
        // instead check exp then a known value: exp((1, 0, π/2)) should be
        // (2/π·1·... t = (sin(ω)/ω·vx, (1−cosω)/ω·vx, ω)) = (2/π, 2/π, π/2).
        let e = se2_exp(&[1.0, 0.0, std::f64::consts::FRAC_PI_2]);
        let expect =
            [2.0 / std::f64::consts::PI, 2.0 / std::f64::consts::PI, std::f64::consts::FRAC_PI_2];
        assert!(close(&e, &expect, 1e-12));
    }

    #[test]
    fn retract_local_roundtrip_all_manifolds() {
        let cases: Vec<(Manifold, Vec<f64>, Vec<f64>)> = vec![
            (Manifold::Euclidean(3), vec![1.0, -2.0, 3.0], vec![0.5, 0.25, -0.5]),
            (Manifold::So2, vec![1.0], vec![0.7]),
            (Manifold::Se2, vec![1.0, 2.0, 0.3], vec![0.5, -0.4, 0.9]),
            (Manifold::So3, vec![0.0, 0.0, 0.0, 1.0], vec![0.2, -0.3, 0.5]),
            (
                Manifold::Se3,
                vec![1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                vec![0.3, 0.1, -0.7, 0.2, -0.1, 0.4],
            ),
        ];
        for (m, x, delta) in cases {
            let xr = m.retract(&x, &delta);
            let back = m.local(&x, &xr);
            assert!(close(&back, &delta, 1e-9), "{m:?}: {back:?} vs {delta:?}");
            // local identity: local(x, x) = 0.
            assert!(m.local(&x, &x).iter().all(|v| v.abs() < 1e-12));
            // inverse twice is identity.
            let inv2 = m.inverse(&m.inverse(&x));
            let x = m.normalized(&x);
            assert!(close(&inv2, &x, 1e-12), "{m:?}");
        }
    }

    #[test]
    fn se3_compose_inverse_chain() {
        let m = Manifold::Se3;
        let a = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0];
        let step = se3_exp(&[0.5, -0.2, 0.3, 0.1, 0.2, 0.3]);
        let b = m.compose(
            &a,
            &[step.0[0], step.0[1], step.0[2], step.1[0], step.1[1], step.1[2], step.1[3]],
        );
        let b_inv = m.inverse(&b);
        let identity = m.compose(&b, &b_inv);
        assert!(close(&identity, &m.identity(), 1e-12), "T T⁻¹ = I, got {identity:?}");
    }

    #[test]
    fn wrap_angle_behavior() {
        assert!(wrap_angle(std::f64::consts::PI).abs() <= std::f64::consts::PI);
        assert!((wrap_angle(3.0 * std::f64::consts::PI) - std::f64::consts::PI).abs() < 1e-12);
        assert!((wrap_angle(-0.3) + 0.3).abs() < 1e-15);
    }
}
