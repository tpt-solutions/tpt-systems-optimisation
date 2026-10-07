//! Robust loss functions (M-estimators) applied via IRLS row scaling.
//!
//! Every loss `ρ(u)` is a function of the squared residual `u = r²`,
//! convex-in-`u` monotone with `ρ(0) = 0`, matching the conventions of
//! scipy/DCP-style least squares. Solvers reweight each residual row by
//! `w = ρ'(r²)` (an IRLS step, re-evaluated every outer iteration), so the
//! weighted problem is a standard (linearised) least-squares problem while
//! the *effective* cost `Σ ρ(rᵢ²)` is robust to outliers.
//!
//! # Example
//!
//! ```rust
//! use tpt_opt_nls::loss::Loss;
//!
//! // Huber behaves like L2 near zero and grows linearly in |r| beyond δ.
//! let huber = Loss::Huber { delta: 1.0 };
//! assert!((huber.rho(0.25) - 0.25).abs() < 1e-12); // r = 0.5 → r²
//! assert!(huber.rho(100.0) < 100.0);               // linear tail
//! assert!((Loss::L2.rho(3.0) - 3.0).abs() < 1e-12);
//! ```

/// Robust loss applied to squared residuals.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Loss {
    /// Plain least squares: `ρ(u) = u`.
    #[default]
    L2,
    /// Huber: quadratic for `u ≤ δ²`, linear beyond. `δ > 0`.
    Huber {
        /// Transition point in residual units.
        delta: f64,
    },
    /// Cauchy: `ρ(u) = (δ²/2)·ln(1 + u/δ²)` — heavy redescending influence.
    Cauchy {
        /// Scale in residual units.
        delta: f64,
    },
    /// Tukey biweight: influence vanishes entirely beyond `δ`. `δ > 0`.
    Tukey {
        /// Cutoff in residual units.
        delta: f64,
    },
    /// Soft L1: `ρ(u) = δ²(√(1 + u/δ²) − 1)`.
    SoftL1 {
        /// Scale in residual units.
        delta: f64,
    },
}

impl Loss {
    /// `ρ(u)` for squared residual `u`.
    pub fn rho(&self, u: f64) -> f64 {
        match *self {
            Loss::L2 => u,
            Loss::Huber { delta } => {
                let d2 = delta * delta;
                if u <= d2 {
                    u
                } else {
                    2.0 * delta * u.sqrt() - d2
                }
            }
            Loss::Cauchy { delta } => {
                let d2 = delta * delta;
                0.5 * d2 * (1.0 + u / d2).ln()
            }
            Loss::Tukey { delta } => {
                let d2 = delta * delta;
                if u <= d2 {
                    let t = 1.0 - u / d2;
                    d2 / 6.0 * (1.0 - t * t * t)
                } else {
                    d2 / 6.0
                }
            }
            Loss::SoftL1 { delta } => {
                let d2 = delta * delta;
                d2 * ((1.0 + u / d2).sqrt() - 1.0)
            }
        }
    }

    /// `ρ'(u)` — the IRLS weight applied to residual row `√u`.
    pub fn drho(&self, u: f64) -> f64 {
        match *self {
            Loss::L2 => 1.0,
            Loss::Huber { delta } => {
                let d2 = delta * delta;
                if u <= d2 {
                    1.0
                } else {
                    let r = u.sqrt();
                    if r > 0.0 {
                        delta / r
                    } else {
                        1.0
                    }
                }
            }
            Loss::Cauchy { delta } => {
                let d2 = delta * delta;
                1.0 / (1.0 + u / d2)
            }
            Loss::Tukey { delta } => {
                let d2 = delta * delta;
                if u <= d2 {
                    let t = 1.0 - u / d2;
                    t * t
                } else {
                    0.0
                }
            }
            Loss::SoftL1 { delta } => {
                let d2 = delta * delta;
                1.0 / (1.0 + u / d2).sqrt()
            }
        }
    }

    /// IRLS weight for signed residual `r`: `w(r) = ρ'(r²)` with the `r → 0`
    /// limit taken analytically (never 0/0).
    pub fn weight(&self, r: f64) -> f64 {
        let u = r * r;
        if u <= f64::EPSILON {
            return self.drho(0.0);
        }
        self.drho(u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_is_identity() {
        assert_eq!(Loss::L2.rho(4.0), 4.0);
        assert_eq!(Loss::L2.drho(4.0), 1.0);
    }

    #[test]
    fn huber_quadratic_then_linear() {
        let h = Loss::Huber { delta: 1.0 };
        assert!((h.rho(0.49) - 0.49).abs() < 1e-12);
        // r = 2 → u = 4 → ρ = 2·1·2 − 1 = 3.
        assert!((h.rho(4.0) - 3.0).abs() < 1e-12);
        assert!((h.drho(0.25) - 1.0).abs() < 1e-12);
        assert!((h.drho(4.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn tukey_zeroes_influence_beyond_delta() {
        let t = Loss::Tukey { delta: 1.0 };
        assert_eq!(t.drho(4.0), 0.0);
        assert!((t.drho(0.25) - 0.5625).abs() < 1e-12);
        // Saturating cost.
        assert!((t.rho(100.0) - t.rho(1.0)).abs() < 1e-12);
    }

    #[test]
    fn cauchy_softl1_continuity_and_bounds() {
        for l in [Loss::Cauchy { delta: 0.8 }, Loss::SoftL1 { delta: 0.8 }] {
            // ρ(0) = 0, ρ' (0) = 1.
            assert!(l.rho(0.0).abs() < 1e-15);
            assert!((l.drho(0.0) - 1.0).abs() < 1e-15);
            // Sub-quadratic growth: ρ(u)/u decreasing.
            assert!(l.rho(1e4) / 1e4 < l.rho(1.0));
        }
    }

    #[test]
    fn weight_handles_zero_residual() {
        let h = Loss::Huber { delta: 1.0 };
        assert_eq!(h.weight(0.0), 1.0);
        assert_eq!(Loss::Tukey { delta: 1.0 }.weight(0.0), 1.0);
    }
}
