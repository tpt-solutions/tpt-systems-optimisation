//! Block-structured normal equations for a factor graph: assembly from
//! Jacobian blocks, sparse LDLᵀ steps, and Schur-complement elimination.

use std::collections::BTreeMap;

use tpt_opt_nls::dense;
use tpt_opt_nls::sparse::{self, Ordering, SparseError, SymbolicLdl, Triplet};

use crate::factor::Key;

/// Assembled (whitened, IRLS-weighted) normal equations of one
/// linearisation: `H δ = −g` with `H = Σ JᵀJ`, `g = Σ Jᵀr`.
#[derive(Debug, Clone, Default)]
pub struct BlockSystem {
    /// Per slot in graph-key insertion order: `(key, tangent offset, dof)`.
    pub slots: Vec<(Key, usize, usize)>,
    /// Upper-triangle Hessian blocks keyed by sorted slot pair; block is
    /// row-major `dof_i × dof_j`.
    pub h: BTreeMap<(usize, usize), Vec<f64>>,
    /// Gradient, length = total tangent dim.
    pub g: Vec<f64>,
    /// IRLS-weighted cost `½ Σ w ‖r̃‖²` at the linearisation point.
    pub cost: f64,
    /// Total tangent dimension.
    pub n: usize,
}

/// Row-major transpose of a `rows × cols` block.
fn transpose_block(b: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = b[r * cols + c];
        }
    }
    out
}

/// `a (ra × ca) · b (rb × cb)` with `ca == rb`.
fn mat_mul(a: &[f64], ra: usize, ca: usize, b: &[f64], cb: usize) -> Vec<f64> {
    let mut out = vec![0.0; ra * cb];
    for i in 0..ra {
        for k in 0..ca {
            let aik = a[i * ca + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..cb {
                out[i * cb + j] += aik * b[k * cb + j];
            }
        }
    }
    out
}

/// `a (rows × cols) · x`.
fn mat_vec(a: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    (0..rows).map(|i| (0..cols).map(|j| a[i * cols + j] * x[j]).sum()).collect()
}

impl BlockSystem {
    /// Offset/dof of a slot.
    pub fn layout(&self, s: usize) -> (usize, usize) {
        (self.slots[s].1, self.slots[s].2)
    }

    /// Flatten to upper-triangle triplets.
    pub fn to_triplets(&self) -> Vec<Triplet> {
        let mut out = Vec::new();
        for (&(si, sj), block) in &self.h {
            let (oi, di) = self.layout(si);
            let (oj, dj) = self.layout(sj);
            for r in 0..di {
                let c0 = if si == sj { r } else { 0 };
                for c in c0..dj {
                    let v = block[r * dj + c];
                    if v != 0.0 {
                        out.push(Triplet::new(oi + r, oj + c, v));
                    }
                }
            }
        }
        out
    }

    /// Marquardt diagonal with a small floor.
    fn diag(&self) -> Vec<f64> {
        let mut d = vec![1e-6; self.n];
        for (&(s, s2), block) in &self.h {
            if s != s2 {
                continue;
            }
            let (o, dof) = self.layout(s);
            for r in 0..dof {
                d[o + r] = block[r * dof + r].abs().max(1e-6);
            }
        }
        d
    }

    /// Solve `(H + λ·diag) δ = −g` via sparse LDLᵀ with `sym` (analysed
    /// on first use, reused across calls; re-analysed once if the
    /// structural pattern drifts). Returns `None` when the factorisation
    /// fails (non-PD).
    pub fn solve_step(&self, lambda: f64, sym: &mut Option<SymbolicLdl>) -> Option<Vec<f64>> {
        let diag = self.diag();
        let mut triplets = self.to_triplets();
        triplets.extend(
            diag.iter().take(self.n).enumerate().map(|(k, &d)| Triplet::new(k, k, lambda * d)),
        );
        for attempt in 0..2 {
            if sym.is_none() {
                *sym = Some(sparse::analyze(self.n, &triplets, Ordering::MinDegree));
            }
            let analysed = sym.as_ref().expect("symbolic present");
            match analysed.factorize(&triplets) {
                Ok(ldl) => {
                    let rhs: Vec<f64> = self.g.iter().map(|&v| -v).collect();
                    return Some(ldl.solve(&rhs));
                }
                Err(SparseError::NotPositiveDefinite) => return None,
                Err(SparseError::PatternMismatch) if attempt == 0 => *sym = None,
                Err(_) => return None,
            }
        }
        None
    }

    /// Schur-complement step eliminating the given slots ("points"):
    /// build and sparsely solve the reduced camera system, then
    /// back-substitute `δ_p = −A_pp⁻¹ (g_p + Σ_c A_pc δ_c)`.
    pub fn solve_step_schur(
        &self,
        eliminate: &[usize],
        lambda: f64,
        sym: &mut Option<SymbolicLdl>,
    ) -> Option<Vec<f64>> {
        let points: std::collections::BTreeSet<usize> = eliminate.iter().copied().collect();
        // Damped, inverted point blocks.
        let mut point_inv: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        for &p in eliminate {
            let block = self.h.get(&(p, p))?;
            let (o, dof) = self.layout(p);
            let mut a = block.clone();
            let dg = self.diag();
            for r in 0..dof {
                a[r * dof + r] += lambda * dg[o + r];
            }
            point_inv.insert(p, dense::spd_inverse(&a, dof)?);
        }

        // Collect, per point, its adjacent (camera slot, A_cp) pairs with
        // A_cp normalised to dof_c × dof_p.
        let mut adj: BTreeMap<usize, Vec<(usize, Vec<f64>)>> = BTreeMap::new();
        for (&(si, sj), block) in &self.h {
            if si == sj {
                continue;
            }
            if points.contains(&si) && !points.contains(&sj) {
                let (dc, dp) = (self.layout(sj).1, self.layout(si).1);
                // stored (point, camera) with point < camera: block is
                // A_pc (dofp × dofc); A_cp is its transpose.
                adj.entry(si).or_default().push((sj, transpose_block(block, dp, dc)));
            } else if !points.contains(&si) && points.contains(&sj) {
                // stored (camera, point): block is A_cp directly.
                adj.entry(sj).or_default().push((si, block.clone()));
            }
        }

        // Reduced camera system: copy camera-camera blocks, then subtract
        // A_cp A_pp⁻¹ A_pc' per point.
        let mut reduced: BTreeMap<(usize, usize), Vec<f64>> = BTreeMap::new();
        for (&(si, sj), block) in &self.h {
            if !points.contains(&si) && !points.contains(&sj) {
                reduced.insert((si, sj), block.clone());
            }
        }
        let mut reduced_g = self.g.clone();

        for (&p, cam_blocks) in adj.iter() {
            let inv = &point_inv[&p];
            let (_, dp) = self.layout(p);
            // M_c = A_cp · A_pp⁻¹   (dof_c × dof_p); raw A_cp kept for the
            // pair term A_pc2 = A_c2pᵀ.
            let raw: BTreeMap<usize, Vec<f64>> =
                cam_blocks.iter().map(|&(c, ref a)| (c, a.clone())).collect();
            let mats: Vec<(usize, Vec<f64>)> = cam_blocks
                .iter()
                .map(|&(c, ref a_cp)| {
                    let (_, dc) = self.layout(c);
                    (c, mat_mul(a_cp, dc, dp, inv, dp))
                })
                .collect();
            let (op, _) = self.layout(p);
            let gp = &self.g[op..op + dp];
            for &(c, ref m) in &mats {
                let (oc, dc) = self.layout(c);
                let corr = mat_vec(m, dc, dp, gp);
                for (i, v) in corr.iter().enumerate() {
                    reduced_g[oc + i] -= v;
                }
            }
            // H'[c1, c2] -= A_c1p A_pp⁻¹ A_pc2 = M_1 · A_pc2, where
            // A_pc2 = A_c2pᵀ.
            for (i, &(c1, ref m1)) in mats.iter().enumerate() {
                let (_, d1) = self.layout(c1);
                for &(c2, _) in mats.iter().skip(i) {
                    let (_, d2) = self.layout(c2);
                    let a_c2p = &raw[&c2];
                    let a_pc2 = transpose_block(a_c2p, d2, dp);
                    let contrib = mat_mul(m1, d1, dp, &a_pc2, d2);
                    let (lo, hi) = if c1 <= c2 { (c1, c2) } else { (c2, c1) };
                    let entry = reduced.entry((lo, hi)).or_insert_with(|| {
                        let (_, dd1) = self.layout(lo);
                        let (_, dd2) = self.layout(hi);
                        vec![0.0; dd1 * dd2]
                    });
                    if c1 <= c2 {
                        for (k, v) in contrib.iter().enumerate() {
                            entry[k] -= v;
                        }
                    } else {
                        let t = transpose_block(&contrib, d1, d2);
                        for (k, v) in t.iter().enumerate() {
                            entry[k] -= v;
                        }
                    }
                }
            }
        }

        // Solve the reduced camera system (offsets are preserved, so the
        // step vector is indexed by the original offsets directly).
        let dim_c = self.n - points.iter().map(|&p| self.layout(p).1).sum::<usize>();
        let reduced_sys = BlockSystem {
            slots: (0..self.slots.len())
                .filter(|s| !points.contains(s))
                .map(|s| self.slots[s])
                .collect(),
            h: reduced,
            g: reduced_g,
            cost: self.cost,
            n: dim_c,
        };
        let cam_step = reduced_sys.solve_step(lambda, sym)?;

        // The reduced solve is indexed by the original offsets (values, not
        // a dense renumbering), so a positional copy is correct; point
        // offsets stay zero until back-substitution.
        let mut delta = vec![0.0; self.n];
        delta[..cam_step.len()].copy_from_slice(&cam_step);
        // Back-substitute points.
        for (&p, inv) in &point_inv {
            let (op, dp) = self.layout(p);
            let mut rhs = self.g[op..op + dp].to_vec();
            for &(c, ref a_cp) in adj.get(&p).into_iter().flatten() {
                let (oc, dc) = self.layout(c);
                // A_pc = A_cpᵀ · δ_c.
                let a_pc = transpose_block(a_cp, dc, dp);
                let prod = mat_vec(&a_pc, dp, dc, &delta[oc..oc + dc]);
                for (i, v) in prod.iter().enumerate() {
                    rhs[i] += v;
                }
            }
            let dpv = mat_vec(inv, dp, dp, &rhs);
            for (i, v) in dpv.iter().enumerate() {
                delta[op + i] -= v;
            }
        }
        Some(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the 2-variable linear system H = [[4,1],[1,3]], g = [1,2] and
    /// check both solve paths against the direct solve.
    fn two_by_two() -> BlockSystem {
        BlockSystem {
            slots: vec![(0, 0, 1), (1, 1, 1)],
            h: BTreeMap::from([((0, 0), vec![4.0]), ((0, 1), vec![1.0]), ((1, 1), vec![3.0])]),
            g: vec![1.0, 2.0],
            cost: 0.0,
            n: 2,
        }
    }

    #[test]
    fn sparse_step_matches_direct_solve() {
        let sys = two_by_two();
        let mut sym = None;
        let d = sys.solve_step(0.0, &mut sym).expect("step");
        // [[4,1],[1,3]] δ = −[1,2] → δ = (−1/11, −7/11).
        assert!((d[0] + 1.0 / 11.0).abs() < 1e-12);
        assert!((d[1] + 7.0 / 11.0).abs() < 1e-12);
    }

    #[test]
    fn schur_step_matches_direct_solve() {
        let sys = two_by_two();
        let mut sym = None;
        let d = sys.solve_step_schur(&[1], 0.0, &mut sym).expect("schur step");
        assert!((d[0] + 1.0 / 11.0).abs() < 1e-12, "d = {d:?}");
        assert!((d[1] + 7.0 / 11.0).abs() < 1e-12);
    }

    #[test]
    fn transpose_and_matmul_helpers() {
        let b = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
        let t = transpose_block(&b, 2, 2);
        assert_eq!(t, vec![1.0, 3.0, 2.0, 4.0]);
        let prod = mat_mul(&b, 2, 2, &b, 2);
        assert_eq!(prod, vec![7.0, 10.0, 15.0, 22.0]);
        let v = mat_vec(&b, 2, 2, &[1.0, 1.0]);
        assert_eq!(v, vec![3.0, 7.0]);
    }
}
