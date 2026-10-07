//! Dense linear algebra helpers for the least-squares solvers.
//!
//! Everything is hand-rolled, row-major, `f64`, and panic-free: failures
//! (singular matrices, non-positive-definite Cholesky) are reported as
//! `Result`/`bool` rather than aborting the caller. Sufficient for the
//! normal-equation systems (`n × n`) that arise in nonlinear least squares.

/// Result of a dense LU factorisation with partial pivoting.
#[derive(Debug, Clone)]
pub struct LuFactor {
    /// Row-major `n × n` combined L (unit diagonal) and U factors.
    pub lu: Vec<f64>,
    /// Pivot permutation: `pivots[k]` is the original row moved to position `k`.
    pub pivots: Vec<usize>,
    /// `false` when a zero pivot was hit (matrix singular to working precision).
    pub singular: bool,
}

/// LU-factor a row-major `n × n` matrix with partial pivoting.
pub fn lu_factor(a: &[f64], n: usize) -> LuFactor {
    let mut lu = a.to_vec();
    let mut pivots: Vec<usize> = (0..n).collect();
    let mut singular = false;
    for k in 0..n {
        // Column pivot search.
        let (mut p, mut max) = (k, lu[k * n + k].abs());
        for i in (k + 1)..n {
            let v = lu[i * n + k].abs();
            if v > max {
                max = v;
                p = i;
            }
        }
        if max <= f64::EPSILON * (1.0 + k as f64) {
            singular = true;
            continue;
        }
        if p != k {
            pivots.swap(k, p);
            for j in 0..n {
                lu.swap(k * n + j, p * n + j);
            }
        }
        let piv = lu[k * n + k];
        for i in (k + 1)..n {
            let m = lu[i * n + k] / piv;
            lu[i * n + k] = m;
            if m != 0.0 {
                for j in (k + 1)..n {
                    lu[i * n + j] -= m * lu[k * n + j];
                }
            }
        }
    }
    LuFactor { lu, pivots, singular }
}

/// Solve `A x = b` in place (b replaced by x) given an [`LuFactor`].
///
/// Returns `false` if the factorisation was singular; `b` is left untouched in
/// that case.
pub fn lu_solve_in_place(f: &LuFactor, b: &mut [f64]) -> bool {
    let n = f.pivots.len();
    if f.singular || b.len() != n {
        return false;
    }
    let mut x: Vec<f64> = (0..n).map(|k| b[f.pivots[k]]).collect();
    // Forward substitution (unit lower).
    for i in 0..n {
        let mut s = x[i];
        for (j, xj) in x.iter().enumerate().take(i) {
            s -= f.lu[i * n + j] * xj;
        }
        x[i] = s;
    }
    // Back substitution (upper).
    for i in (0..n).rev() {
        let mut s = x[i];
        for (j, xj) in x.iter().enumerate().skip(i + 1) {
            s -= f.lu[i * n + j] * xj;
        }
        let d = f.lu[i * n + i];
        x[i] = if d.abs() <= f64::MIN_POSITIVE { 0.0 } else { s / d };
    }
    b.copy_from_slice(&x);
    true
}

/// Cholesky-factor a symmetric positive-definite row-major `n × n` matrix
/// (only the lower triangle is read). Returns the lower factor `L` with
/// `A = L Lᵀ`, or `None` if a non-positive pivot is encountered.
pub fn cholesky_lower(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Solve `A x = b` given the Cholesky lower factor of `A = L Lᵀ`.
/// `x` and `b` may alias.
pub fn cho_solve_lower(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    // L y = b.
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * x[j];
        }
        x[i] = s / l[i * n + i];
    }
    // Lᵀ x = y.
    for i in (0..n).rev() {
        let mut s = x[i];
        for (j, xj) in x.iter().enumerate().take(n).skip(i + 1) {
            s -= l[j * n + i] * xj;
        }
        x[i] = s / l[i * n + i];
    }
    x
}

/// Invert a symmetric positive-definite row-major `n × n` matrix via
/// Cholesky. Returns `None` if not positive definite.
pub fn spd_inverse(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let l = cholesky_lower(a, n)?;
    let mut inv = vec![0.0; n * n];
    for c in 0..n {
        // Solve A x = e_c.
        let mut e = vec![0.0; n];
        e[c] = 1.0;
        let x = cho_solve_lower(&l, &e, n);
        for r in 0..n {
            inv[r * n + c] = x[r];
        }
    }
    Some(inv)
}

/// Row-major matrix-vector product `y = A x` for a row-major `m × n` matrix.
pub fn mat_vec(a: &[f64], m: usize, n: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; m];
    for (i, yv) in y.iter_mut().enumerate() {
        let row = &a[i * n..(i + 1) * n];
        *yv = row.iter().zip(x.iter()).map(|(&a, &x)| a * x).sum();
    }
    y
}

/// Compute `H = Jᵀ J` for a row-major `m × n` Jacobian into a row-major
/// symmetric `n × n` matrix (only the full symmetric result is produced; the
/// caller may use just one triangle).
pub fn normal_equations(j: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut h = vec![0.0; n * n];
    for row in 0..m {
        let jr = &j[row * n..(row + 1) * n];
        for i in 0..n {
            let jiv = jr[i];
            if jiv == 0.0 {
                continue;
            }
            for k in i..n {
                h[i * n + k] += jiv * jr[k];
            }
        }
    }
    for i in 0..n {
        for k in 0..i {
            h[i * n + k] = h[k * n + i];
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_solves_general_system() {
        // A = [[2,1],[1,3]], b = [3,5] → x = [4/5, 7/5].
        let a = [2.0, 1.0, 1.0, 3.0];
        let f = lu_factor(&a, 2);
        assert!(!f.singular);
        let mut b = vec![3.0, 5.0];
        assert!(lu_solve_in_place(&f, &mut b));
        assert!((b[0] - 0.8).abs() < 1e-12);
        assert!((b[1] - 1.4).abs() < 1e-12);
    }

    #[test]
    fn lu_detects_singular() {
        let a = vec![1.0, 2.0, 2.0, 4.0];
        let f = lu_factor(&a, 2);
        assert!(f.singular);
    }

    #[test]
    fn cholesky_roundtrip() {
        let a = [4.0, 2.0, 2.0, 3.0];
        let l = cholesky_lower(&a, 2).expect("spd");
        let prod = [l[0] * l[0], l[0] * l[2], l[0] * l[2], l[2] * l[2] + l[3] * l[3]];
        for k in 0..4 {
            assert!((prod[k] - a[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn spd_inverse_matches_identity_product() {
        let a = [6.0, 2.0, 2.0, 4.0];
        let inv = spd_inverse(&a, 2).expect("spd");
        let id = [
            a[0] * inv[0] + a[1] * inv[2],
            a[0] * inv[1] + a[1] * inv[3],
            a[2] * inv[0] + a[3] * inv[2],
            a[2] * inv[1] + a[3] * inv[3],
        ];
        assert!((id[0] - 1.0).abs() < 1e-12 && (id[3] - 1.0).abs() < 1e-12);
        assert!(id[1].abs() < 1e-12 && id[2].abs() < 1e-12);
    }

    #[test]
    fn normal_equations_matches_transpose_product() {
        let j = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let h = normal_equations(&j, 2, 3);
        // JᵀJ = [[17, 22, 27], [22, 29, 36], [27, 36, 45]].
        let expect = [17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0];
        for k in 0..9 {
            assert!((h[k] - expect[k]).abs() < 1e-12, "h[{k}] = {}", h[k]);
        }
    }
}
