//! Gaussian noise models and robust kernel wiring.

use tpt_opt_nls::Loss;

/// Whitening model for a factor's residual: `r̃ = √W r` with `√W = sqrt
/// information` (inverse square root covariance). Robust kernels are
/// applied to the *whitened* residual as an IRLS row scale, matching
/// GTSAM-style semantics.
#[derive(Debug, Clone)]
pub enum NoiseModel {
    /// Identity whitening (unit information).
    Unit,
    /// `σ·I` covariance: whitening divides by `sigma`.
    Isotropic {
        /// Standard deviation per coordinate.
        sigma: f64,
    },
    /// Diagonal covariance with per-coordinate standard deviations.
    Diagonal {
        /// Standard deviation per coordinate.
        sigmas: Vec<f64>,
    },
    /// Full covariance via its **square-root information** matrix
    /// `S` (row-major, `dim × dim`) with `SᵀS = Σ⁻¹`.
    Gaussian {
        /// Row-major square-root information matrix.
        sqrt_info: Vec<f64>,
    },
}

impl NoiseModel {
    /// Whiten a residual in place semantics: returns `S r`.
    pub fn whiten(&self, r: &[f64]) -> Vec<f64> {
        match self {
            NoiseModel::Unit => r.to_vec(),
            NoiseModel::Isotropic { sigma } => r.iter().map(|v| v / sigma).collect(),
            NoiseModel::Diagonal { sigmas } => {
                r.iter().zip(sigmas.iter()).map(|(v, s)| v / s).collect()
            }
            NoiseModel::Gaussian { sqrt_info } => {
                let dim = r.len();
                (0..dim)
                    .map(|i| (0..dim).map(|j| sqrt_info[i * dim + j] * r[j]).sum::<f64>())
                    .collect()
            }
        }
    }

    /// Whiten the rows of a residual's Jacobian block (`dim × dof`,
    /// row-major): each output row is the corresponding row of `S J`.
    pub fn whiten_jacobian(&self, j: &[f64], dim: usize, dof: usize) -> Vec<f64> {
        match self {
            NoiseModel::Unit => j.to_vec(),
            NoiseModel::Isotropic { sigma } => j.iter().map(|v| v / sigma).collect(),
            NoiseModel::Diagonal { sigmas } => {
                let mut out = j.to_vec();
                for (i, row) in out.chunks_mut(dof).enumerate() {
                    let s = sigmas[i];
                    for v in row.iter_mut() {
                        *v /= s;
                    }
                }
                out
            }
            NoiseModel::Gaussian { sqrt_info } => {
                let mut out = vec![0.0; dim * dof];
                for (i, row) in out.chunks_mut(dof).enumerate() {
                    for (jj, v) in row.iter_mut().enumerate() {
                        *v = (0..dim).map(|k| sqrt_info[i * dim + k] * j[k * dof + jj]).sum();
                    }
                }
                out
            }
        }
    }

    /// IRLS weight from the robust kernel applied to the whitened residual
    /// (`1.0` without a kernel).
    pub fn robust_weight(&self, whitened_r: &[f64], robust: Option<Loss>) -> f64 {
        match robust {
            None => 1.0,
            Some(loss) => {
                let sq: f64 = whitened_r.iter().map(|v| v * v).sum();
                loss.weight(sq.sqrt())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotropic_whitening_scales() {
        let n = NoiseModel::Isotropic { sigma: 2.0 };
        assert_eq!(n.whiten(&[2.0, 4.0]), vec![1.0, 2.0]);
        assert_eq!(n.whiten_jacobian(&[2.0, 4.0, 6.0, 8.0], 2, 2), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn gaussian_whitening_is_matrix_product() {
        // S = [[2, 0], [1, 1]]: S·[3, 1] = [6, 4].
        let n = NoiseModel::Gaussian { sqrt_info: vec![2.0, 0.0, 1.0, 1.0] };
        assert_eq!(n.whiten(&[3.0, 1.0]), vec![6.0, 4.0]);
        // Jacobian (2×2) rows mix identically.
        let j = n.whiten_jacobian(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        assert_eq!(j, vec![2.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn robust_weight_downweights_large_residuals() {
        let huber = Loss::Huber { delta: 1.0 };
        let n = NoiseModel::Unit;
        assert_eq!(n.robust_weight(&[0.1, 0.0], Some(huber)), 1.0);
        let w = n.robust_weight(&[10.0, 0.0], Some(huber));
        assert!((w - 0.1).abs() < 1e-12);
        assert_eq!(n.robust_weight(&[10.0], None), 1.0);
    }
}
