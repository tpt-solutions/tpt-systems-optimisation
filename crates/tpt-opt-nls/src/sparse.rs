//! Sparse symmetric factorisation: fill-reducing ordering + LDLᵀ.
//!
//! The published `tpt-math-linalg-sparse` crate provides storage formats and
//! iterative solvers only, so the direct sparse factorisations needed by the
//! sparse normal-equations path live here (see the Phase 13a decision in the
//! workspace todo: upstream `tpt-math` is published and frozen, so the new
//! math ships inside its consumer, `tpt-opt-nls`).
//!
//! Pipeline:
//! 1. [`analyze`] symmetrises the sparsity pattern of a symmetric matrix
//!    given as [`Triplet`]s and computes a fill-reducing ordering
//!    ([`Ordering::ApproxMinDegree`], the default: quotient-graph minimum
//!    degree with approximate external degrees, as in AMD; the exact-degree
//!    [`Ordering::MinDegree`] and the identity [`Ordering::Natural`] are also
//!    available). Along the way it runs the elimination
//!    game once, which yields the **exact filled pattern** of `L` — the
//!    column counts and row patterns fall out directly, no elimination-tree
//!    reach machinery required.
//! 2. [`SymbolicLdl::factorize`] performs an up-looking numeric
//!    factorisation `P A Pᵀ = L D Lᵀ` (unit lower-triangular `L` by columns,
//!    diagonal `D`). The matrix must be positive definite (as `JᵀJ + λI`
//!    always is for `λ > 0`); a non-positive pivot is reported, never
//!    divided through.
//! 3. [`Ldl::solve`] permutes, forward-solves, scales, and back-solves.
//!
//! # Example
//!
//! ```rust
//! use tpt_opt_nls::sparse::{analyze, Ordering, Triplet};
//!
//! // A = [[4, 1, 0], [1, 5, 2], [0, 2, 6]] — SPD.
//! let triplets = vec![
//!     Triplet::new(0, 0, 4.0), Triplet::new(1, 1, 5.0), Triplet::new(2, 2, 6.0),
//!     Triplet::new(0, 1, 1.0), Triplet::new(1, 0, 1.0),
//!     Triplet::new(1, 2, 2.0), Triplet::new(2, 1, 2.0),
//! ];
//! let sym = analyze(3, &triplets, Ordering::Natural);
//! let ldl = sym.factorize(&triplets).expect("spd");
//! let x = ldl.solve(&[1.0, 2.0, 3.0]);
//! // A x = b check.
//! let ax = [
//!     4.0 * x[0] + x[1],
//!     x[0] + 5.0 * x[1] + 2.0 * x[2],
//!     2.0 * x[1] + 6.0 * x[2],
//! ];
//! assert!((ax[0] - 1.0).abs() < 1e-12);
//! assert!((ax[1] - 2.0).abs() < 1e-12);
//! assert!((ax[2] - 3.0).abs() < 1e-12);
//! ```

use std::collections::BTreeSet;

/// Errors from sparse factorisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseError {
    /// A non-positive pivot was met — the matrix is not positive definite
    /// (increase the LM damping and retry).
    NotPositiveDefinite,
    /// The triplet list contains an out-of-range index.
    IndexOutOfBounds,
    /// A triplet coordinate has no slot in the analysed pattern — the
    /// numeric pattern grew since `analyze` (callers should re-analyse).
    PatternMismatch,
}

/// A sparse matrix entry (scatter-add semantics on duplicate coordinates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triplet {
    /// Row index.
    pub row: usize,
    /// Column index.
    pub col: usize,
    /// Value.
    pub value: f64,
}

impl Triplet {
    /// A new triplet entry.
    pub fn new(row: usize, col: usize, value: f64) -> Self {
        Self { row, col, value }
    }
}

/// Fill-reducing ordering strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ordering {
    /// Identity permutation.
    Natural,
    /// Greedy minimum degree on the quotient graph (exact-degree core of
    /// AMD; deterministic tie-break by lowest index).
    MinDegree,
    /// Quotient-graph minimum degree with **approximate external degrees**
    /// (Amestoy–Davis–Duff). Each pivot updates its neighbours' degrees from
    /// element boundary sizes instead of forming the elimination clique, so it
    /// is far cheaper than [`Ordering::MinDegree`] on large patterns.
    ApproxMinDegree,
}

/// Symbolic analysis of a symmetric matrix: permutation + exact filled
/// pattern of the Cholesky factor.
#[derive(Debug, Clone)]
pub struct SymbolicLdl {
    /// Dimension.
    pub n: usize,
    /// `perm[k]` = original index of the k-th permuted position.
    pub perm: Vec<usize>,
    /// `iperm[i]` = permuted position of original index `i`.
    pub iperm: Vec<usize>,
    /// Number of nonzeros in the strict lower triangle of `L`.
    pub lnz: usize,
    /// Column pointers for `L`'s strict lower columns (length `n + 1`).
    pub l_col_ptr: Vec<usize>,
    /// Strict-lower row indices of `L`, grouped by column (ascending within
    /// each column).
    pub l_rows: Vec<usize>,
    /// Per-permuted-row entries of `L`'s **full** row pattern `(j)` with
    /// `j < k` (original pattern *and* fill), ascending — the positions the
    /// up-looking numeric pass must visit.
    pub t_rows: Vec<Vec<usize>>,
    /// Per-permuted-row original-pattern entries `(j, slot)` with `j < k` —
    /// the numeric pattern of row `k` of `A` and where its values land.
    a_rows: Vec<Vec<(usize, usize)>>,
    /// Total number of unique off-diagonal slots.
    n_slots: usize,
    /// Map from unordered original `(row, col)` pair to its value slot.
    slot_lookup: std::collections::HashMap<(usize, usize), usize>,
    /// Per-slot: `true` when the value is read from `row < col` triplets
    /// (upper side), `false` for `row > col`. Callers commonly list the
    /// symmetric mirror of every entry; reading one side only prevents
    /// double-counting under scatter-add.
    slot_side: std::collections::HashMap<(usize, usize), bool>,
}

/// Numeric LDLᵀ factor from [`SymbolicLdl::factorize`].
#[derive(Debug, Clone)]
pub struct Ldl {
    sym_n: usize,
    perm: Vec<usize>,
    /// Column pointers into `l_rows`/`l_cols`... (`n + 1` entries).
    l_col_ptr: Vec<usize>,
    /// Strict-lower row index per stored entry.
    l_rows: Vec<usize>,
    /// Stored value per entry (parallel to `l_rows`).
    l_vals: Vec<f64>,
    /// Diagonal `D`.
    d: Vec<f64>,
}

impl SymbolicLdl {
    /// Number of nonzeros in `L` (including diagonal).
    pub fn nnz_l(&self) -> usize {
        self.lnz + self.n
    }
}

/// Quotient-graph minimum degree with approximate external degrees.
///
/// Eliminated pivots become *elements* whose boundary `le[e]` is the set of
/// live variables they touch. A variable's degree is estimated from its direct
/// neighbours and the element boundaries that remain after the pivot, using
/// the AMD bound `min(n−k−1, d_old + |Lp|−1, |A_i| + |Lp|−1 + Σ_e |L_e \ Lp|)`.
/// Ties go to the lowest index, so the order is deterministic.
fn approx_min_degree(n: usize, adj: &[BTreeSet<usize>]) -> Vec<usize> {
    let mut adj_v: Vec<BTreeSet<usize>> = adj.to_vec();
    let mut adj_e: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    let mut le: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut absorbed = vec![false; n];
    let mut eliminated = vec![false; n];
    let mut deg: Vec<usize> = adj_v.iter().map(BTreeSet::len).collect();
    let mut in_lp = vec![false; n];
    // Per-element weight `|L_e \ Lp|`, valid when `stamp[e]` equals the step.
    let mut w = vec![0usize; n];
    let mut stamp = vec![usize::MAX; n];
    let mut order = Vec::with_capacity(n);

    for step in 0..n {
        // Pivot: smallest approximate degree among live variables.
        let mut p = usize::MAX;
        for v in 0..n {
            if !eliminated[v] && (p == usize::MAX || deg[v] < deg[p]) {
                p = v;
            }
        }

        // Lp = live neighbours of p, directly or through its adjacent elements.
        let mut lp: Vec<usize> = adj_v[p].iter().copied().collect();
        for &e in &adj_e[p] {
            lp.extend(le[e].iter().copied());
        }
        lp.sort_unstable();
        lp.dedup();
        lp.retain(|&j| j != p);
        for &j in &lp {
            in_lp[j] = true;
        }

        // Elements adjacent to p are absorbed into the new element p.
        let absorbed_now: Vec<usize> = adj_e[p].iter().copied().collect();
        for &e in &absorbed_now {
            absorbed[e] = true;
            le[e] = Vec::new();
        }

        // Weights |L_e \ Lp| for the surviving elements touching Lp.
        for &i in &lp {
            adj_e[i].retain(|&e| !absorbed[e]);
            for &e in &adj_e[i] {
                if stamp[e] != step {
                    stamp[e] = step;
                    w[e] = le[e].len();
                }
                w[e] -= 1;
            }
        }

        // Structural update: Lp becomes the boundary of element p.
        for &i in &lp {
            adj_v[i].retain(|&j| j != p && !in_lp[j]);
            adj_e[i].insert(p);
        }
        le[p] = lp.clone();
        eliminated[p] = true;
        adj_v[p].clear();
        adj_e[p].clear();

        // Approximate external degrees of the boundary variables.
        let remaining = n - step - 1;
        for &i in &lp {
            let ext: usize = adj_e[i].iter().filter(|&&e| e != p).map(|&e| w[e]).sum();
            let bound_clique = deg[i] + lp.len() - 1;
            let bound_elements = adj_v[i].len() + lp.len() - 1 + ext;
            deg[i] = remaining.min(bound_clique).min(bound_elements);
        }
        for &j in &lp {
            in_lp[j] = false;
        }
        order.push(p);
    }
    order
}

/// Run the symbolic analysis (ordering + exact fill) for a symmetric matrix.
///
/// `triplets` may list only the upper triangle, only the lower, or both —
/// the pattern is symmetrised. Duplicates are scatter-added at factorise
/// time; here only coordinates matter.
pub fn analyze(n: usize, triplets: &[Triplet], ordering: Ordering) -> SymbolicLdl {
    // Symmetrised adjacency (no diagonal).
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for t in triplets {
        if t.row >= n || t.col >= n || t.row == t.col {
            continue;
        }
        adj[t.row].insert(t.col);
        adj[t.col].insert(t.row);
    }

    // Greedy minimum-degree order (or identity).
    let order: Vec<usize> = match ordering {
        Ordering::Natural => (0..n).collect(),
        Ordering::ApproxMinDegree => approx_min_degree(n, &adj),
        Ordering::MinDegree => {
            let mut eliminated = vec![false; n];
            let mut order = Vec::with_capacity(n);
            for _ in 0..n {
                let mut best = usize::MAX;
                let mut best_deg = usize::MAX;
                for v in 0..n {
                    if eliminated[v] {
                        continue;
                    }
                    let d = adj[v].len();
                    if d < best_deg {
                        best_deg = d;
                        best = v;
                    }
                }
                // Merge the neighbourhood into a clique (elimination game).
                let nbrs: Vec<usize> = adj[best].iter().copied().collect();
                for (a_idx, &a) in nbrs.iter().enumerate() {
                    adj[a].remove(&best);
                    for &b in nbrs.iter().skip(a_idx + 1) {
                        adj[a].insert(b);
                        adj[b].insert(a);
                    }
                }
                adj[best].clear();
                eliminated[best] = true;
                order.push(best);
            }
            order
        }
    };

    // Re-run the elimination game in the chosen order, recording the exact
    // filled pattern: when eliminating v, its current neighbourhood is
    // column v of L (strict lower part, stored as (row > col) pairs).
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for t in triplets {
        if t.row >= n || t.col >= n || t.row == t.col {
            continue;
        }
        adj[t.row].insert(t.col);
        adj[t.col].insert(t.row);
    }
    let mut filled_cols: Vec<Vec<usize>> = vec![Vec::new(); n]; // per ORIGINAL col
    for &v in &order {
        let nbrs: Vec<usize> = adj[v].iter().copied().collect();
        for (a_idx, &a) in nbrs.iter().enumerate() {
            adj[a].remove(&v);
            for &b in nbrs.iter().skip(a_idx + 1) {
                adj[a].insert(b);
                adj[b].insert(a);
            }
        }
        adj[v].clear();
        filled_cols[v] = nbrs; // rows of L(:, v), all ≠ v
    }

    // Map to permuted indices and build L's structure.
    let mut iperm = vec![0usize; n];
    for (k, &v) in order.iter().enumerate() {
        iperm[v] = k;
    }
    // Each filled entry {col, row} maps to permuted positions (p, q). The
    // elimination runs in permuted order, so every neighbour of `col` still
    // alive at its elimination has q = iperm[row] > p = iperm[col] — the
    // entry is strict-lower (row q, col p).
    let mut lower: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for col in 0..n {
        let p = iperm[col];
        for &row in &filled_cols[col] {
            let q = iperm[row];
            lower[p].insert(q);
        }
    }
    let mut l_col_ptr = vec![0usize; n + 1];
    for (j, col) in lower.iter().enumerate() {
        l_col_ptr[j + 1] = l_col_ptr[j] + col.len();
    }
    let lnz = l_col_ptr[n];
    let mut l_rows = Vec::with_capacity(lnz);
    for col in &lower {
        l_rows.extend(col.iter().copied());
    }

    // Numeric pattern of permuted row k: entries (j < k, slot) — from the
    // ORIGINAL triplet values, deduplicated by unordered original pair.
    let mut slot_of: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    let mut slot_side: std::collections::HashMap<(usize, usize), bool> =
        std::collections::HashMap::new();
    let mut a_rows: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for t in triplets {
        if t.row >= n || t.col >= n || t.row == t.col {
            continue;
        }
        let key = if t.row < t.col { (t.row, t.col) } else { (t.col, t.row) };
        let next = slot_of.len();
        let slot = *slot_of.entry(key).or_insert(next);
        slot_side.entry(key).or_insert(t.row < t.col);
        let (p, q) = (iperm[t.row], iperm[t.col]);
        let (k, j) = if p > q { (p, q) } else { (q, p) };
        let row_entry = (j, slot);
        if !a_rows[k].contains(&row_entry) {
            a_rows[k].push(row_entry);
        }
    }
    for row in a_rows.iter_mut() {
        row.sort_unstable();
        row.dedup();
    }
    let n_slots = slot_of.len();

    // Full row pattern of L: transpose of `lower` (the exact filled
    // structure). Row k of L must visit every column j < k with L_kj ≠ 0,
    // fill positions included.
    let mut t_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (j, rows) in lower.iter().enumerate() {
        for &i in rows {
            t_rows[i].push(j);
        }
    }

    for row in t_rows.iter_mut() {
        row.sort_unstable();
    }

    SymbolicLdl {
        n,
        perm: order,
        iperm,
        lnz,
        l_col_ptr,
        l_rows,
        t_rows,
        a_rows,
        n_slots,
        slot_lookup: slot_of,
        slot_side,
    }
}

impl SymbolicLdl {
    /// Numeric up-looking `LDLᵀ` factorisation of the matrix given by
    /// `triplets` (same coordinates as passed to [`analyze`]; duplicates
    /// scatter-add). Returns [`SparseError::NotPositiveDefinite`] when a
    /// non-positive pivot is met.
    pub fn factorize(&self, triplets: &[Triplet]) -> Result<Ldl, SparseError> {
        let n = self.n;
        let mut diag = vec![0.0f64; n];
        let mut a_vals = vec![0.0f64; self.n_slots];
        for t in triplets {
            if t.row >= n || t.col >= n {
                return Err(SparseError::IndexOutOfBounds);
            }
            if t.row == t.col {
                diag[self.iperm[t.row]] += t.value;
            } else {
                let upper = t.row < t.col;
                let key = if upper { (t.row, t.col) } else { (t.col, t.row) };
                match self.slot_lookup.get(&key) {
                    // Mirrored duplicates (the other triangle) are ignored:
                    // one side already carries the symmetric value.
                    Some(&slot) if self.slot_side.get(&key) == Some(&upper) => {
                        a_vals[slot] += t.value;
                    }
                    Some(&_) => {}
                    None => return Err(SparseError::PatternMismatch),
                }
            }
        }

        let nnz = self.l_col_ptr[n];
        let mut l_rows = vec![0usize; nnz];
        let mut l_vals = vec![0.0f64; nnz];
        l_rows.copy_from_slice(&self.l_rows);
        let mut col_end = self.l_col_ptr.clone(); // running end of each column
        let mut d = vec![0.0f64; n];
        let mut y = vec![0.0f64; n];

        for k in 0..n {
            // y[k] = a_kk, y[j] = a_kj for j in row-k pattern.
            y[k] = diag[k];
            for &(j, slot) in &self.a_rows[k] {
                y[j] = a_vals[slot];
            }
            // Process the FULL row pattern of L (original + fill) in
            // ascending order: the filled graph makes row k of L a clique,
            // so every column-distribution target below k lies inside it.
            // y is pre-seeded at the original-A positions; fill positions
            // start at zero.
            let mut dk = y[k];
            for &j in &self.t_rows[k] {
                let yj = y[j];
                y[j] = 0.0;
                // Distribute column j of L (entries so far: rows in (j, k)).
                let (cs, ce) = (self.l_col_ptr[j], col_end[j]);
                for p in cs..ce {
                    y[l_rows[p]] -= l_vals[p] * yj;
                }
                if d[j] <= 0.0 {
                    return Err(SparseError::NotPositiveDefinite);
                }
                // Append row k to column j. Every symbolic entry is written
                // (zeros included) so l_vals stays aligned with the
                // pre-filled l_rows pattern.
                let lkj = yj / d[j];
                let pos = col_end[j];
                l_vals[pos] = lkj;
                col_end[j] = pos + 1;
                dk -= lkj * yj;
            }
            if dk <= 0.0 || !dk.is_finite() {
                return Err(SparseError::NotPositiveDefinite);
            }
            d[k] = dk;
            y[k] = 0.0;
        }

        Ok(Ldl {
            sym_n: n,
            perm: self.perm.clone(),
            l_col_ptr: self.l_col_ptr.clone(),
            l_rows,
            l_vals,
            d,
        })
    }
}

impl Ldl {
    /// Solve `A x = b` for the factored (permuted) matrix: permute, unit-L
    /// forward solve, diagonal scale, unit-Lᵀ back solve, un-permute.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.sym_n;
        // Permute: x[p] = b[perm[p]].
        let mut x: Vec<f64> = (0..n).map(|p| b[self.perm[p]]).collect();
        // Forward: unit lower triangular L y = x (column-oriented).
        for j in 0..n {
            let xj = x[j];
            if xj == 0.0 {
                continue;
            }
            for p in self.l_col_ptr[j]..self.l_col_ptr[j + 1] {
                x[self.l_rows[p]] -= self.l_vals[p] * xj;
            }
        }
        // Diagonal: z = D⁻¹ y.
        for (xj, dj) in x.iter_mut().zip(self.d.iter()) {
            *xj /= dj;
        }
        // Backward: unit lower transposed Lᵀ w = z (row-oriented).
        for j in (0..n).rev() {
            let mut s = x[j];
            for p in self.l_col_ptr[j]..self.l_col_ptr[j + 1] {
                s -= self.l_vals[p] * x[self.l_rows[p]];
            }
            x[j] = s;
        }
        // Un-permute: out[perm[p]] = x[p].
        let mut out = vec![0.0f64; n];
        for p in 0..n {
            out[self.perm[p]] = x[p];
        }
        out
    }

    /// Dimension of the factored system.
    pub fn n(&self) -> usize {
        self.sym_n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_solve(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let f = crate::dense::lu_factor(a, n);
        assert!(!f.singular);
        let mut x = b.to_vec();
        assert!(crate::dense::lu_solve_in_place(&f, &mut x));
        x
    }

    fn mat_from_triplets(n: usize, triplets: &[Triplet]) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for t in triplets {
            a[t.row * n + t.col] += t.value;
            if t.row != t.col {
                a[t.col * n + t.row] += t.value;
            }
        }
        a
    }

    /// Random SPD matrix: A = M Mᵀ + 0.5 I with a banded-ish random M,
    /// generated from a fixed LCG so the test is deterministic.
    fn random_spd(n: usize, seed: u64) -> (Vec<Triplet>, Vec<f64>) {
        let mut s = seed;
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (1u64 << 31) as f64 - 1.0
        };
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                if i - j < 6 {
                    m[i * n + j] = next();
                }
            }
        }
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push(Triplet::new(i, i, 0.5));
        }
        // Each unordered entry (k <= i) exactly once — mat_from_triplets
        // mirrors off-diagonals.
        for i in 0..n {
            for j in 0..n {
                let v = m[i * n + j];
                if v == 0.0 {
                    continue;
                }
                for k in 0..=i {
                    let w = m[k * n + j];
                    if w == 0.0 {
                        continue;
                    }
                    triplets.push(Triplet::new(k, i, v * w));
                }
            }
        }
        let dense = mat_from_triplets(n, &triplets);
        (triplets, dense)
    }

    #[test]
    fn solves_small_spd() {
        // One entry per unordered pair (listing both sides is also
        // supported — the mirrored side is ignored on read).
        let triplets = vec![
            Triplet::new(0, 0, 4.0),
            Triplet::new(1, 1, 5.0),
            Triplet::new(2, 2, 6.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 2, 2.0),
        ];
        let dense = mat_from_triplets(3, &triplets);
        for ordering in [Ordering::Natural, Ordering::MinDegree] {
            let sym = analyze(3, &triplets, ordering);
            let ldl = sym.factorize(&triplets).expect("spd");
            let b = [1.0, -2.0, 3.5];
            let x = ldl.solve(&b);
            let expect = dense_solve(&dense, &b, 3);
            for k in 0..3 {
                assert!((x[k] - expect[k]).abs() < 1e-10, "{ordering:?} x[{k}]");
            }
        }
    }

    #[test]
    fn solves_random_spd_both_orderings() {
        for seed in [7u64, 42, 1234] {
            let (triplets, dense) = random_spd(40, seed);
            let b: Vec<f64> =
                (0..40).map(|i| ((i * 37 + seed as usize) % 11) as f64 - 5.0).collect();
            let expect = dense_solve(&dense, &b, 40);
            for ordering in [Ordering::Natural, Ordering::MinDegree] {
                let sym = analyze(40, &triplets, ordering);
                let ldl = sym.factorize(&triplets).expect("spd");
                let x = ldl.solve(&b);
                let err: f64 =
                    x.iter().zip(&expect).map(|(a, e)| (a - e).abs()).fold(0.0, f64::max);
                assert!(err < 1e-8, "seed {seed} {ordering:?} err {err}");
            }
        }
    }

    #[test]
    fn detects_indefinite() {
        let triplets = [Triplet::new(0, 0, -1.0), Triplet::new(1, 1, 1.0)];
        let sym = analyze(2, &triplets, Ordering::Natural);
        assert!(matches!(sym.factorize(&triplets), Err(SparseError::NotPositiveDefinite)));
    }

    #[test]
    fn min_degree_does_not_exceed_natural_fill_on_grid() {
        // 5×5 2-D grid graph + diagonal dominance: a classic fill case where
        // minimum degree beats natural ordering.
        let side = 5;
        let n = side * side;
        let mut triplets: Vec<Triplet> = (0..n).map(|i| Triplet::new(i, i, 10.0)).collect();
        let mut edge = |a: usize, b: usize| {
            // Single unordered entry; mat_from_triplets mirrors it.
            triplets.push(Triplet::new(a.min(b), b.max(a), -1.0));
        };
        for r in 0..side {
            for c in 0..side {
                let v = r * side + c;
                if c + 1 < side {
                    edge(v, v + 1);
                }
                if r + 1 < side {
                    edge(v, v + side);
                }
            }
        }
        let nat = analyze(n, &triplets, Ordering::Natural);
        let amd = analyze(n, &triplets, Ordering::MinDegree);
        assert!(
            amd.lnz <= nat.lnz,
            "min-degree fill {} should not exceed natural fill {}",
            amd.lnz,
            nat.lnz
        );
        // Both orderings must still solve identically.
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
        let dense = mat_from_triplets(n, &triplets);
        let expect = dense_solve(&dense, &b, n);
        for sym in [&nat, &amd] {
            let x = sym.factorize(&triplets).expect("spd").solve(&b);
            let err: f64 = x.iter().zip(&expect).map(|(a, e)| (a - e).abs()).fold(0.0, f64::max);
            assert!(err < 1e-8);
        }
    }

    /// 2-D grid with unit off-diagonal couplings and diagonal dominance.
    fn grid_triplets(side: usize) -> (usize, Vec<Triplet>) {
        let n = side * side;
        let mut triplets: Vec<Triplet> = (0..n).map(|i| Triplet::new(i, i, 10.0)).collect();
        for r in 0..side {
            for c in 0..side {
                let v = r * side + c;
                if c + 1 < side {
                    triplets.push(Triplet::new(v, v + 1, -1.0));
                }
                if r + 1 < side {
                    triplets.push(Triplet::new(v, v + side, -1.0));
                }
            }
        }
        (n, triplets)
    }

    #[test]
    fn approx_min_degree_solves_and_reduces_fill_on_grid() {
        let (n, triplets) = grid_triplets(5);
        let nat = analyze(n, &triplets, Ordering::Natural);
        let amd = analyze(n, &triplets, Ordering::ApproxMinDegree);
        assert!(amd.lnz <= nat.lnz, "approx min-degree fill {} vs natural {}", amd.lnz, nat.lnz);

        let mut seen = vec![false; n];
        for &p in &amd.perm {
            assert!(!seen[p]);
            seen[p] = true;
        }
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
        let dense = mat_from_triplets(n, &triplets);
        let expect = dense_solve(&dense, &b, n);
        let x = amd.factorize(&triplets).expect("spd").solve(&b);
        let err: f64 = x.iter().zip(&expect).map(|(a, e)| (a - e).abs()).fold(0.0, f64::max);
        assert!(err < 1e-8);
    }

    #[test]
    fn approx_min_degree_fill_is_close_to_exact_min_degree_on_larger_grid() {
        // The approximate degrees are upper bounds, so the fill may differ from the
        // exact-degree ordering, but it must stay in the same range.
        let (n, triplets) = grid_triplets(16);
        let exact = analyze(n, &triplets, Ordering::MinDegree);
        let approx = analyze(n, &triplets, Ordering::ApproxMinDegree);
        let natural = analyze(n, &triplets, Ordering::Natural);
        assert!(approx.lnz < natural.lnz);
        assert!(
            approx.lnz as f64 <= 1.5 * exact.lnz as f64,
            "approx fill {} vs exact min-degree fill {}",
            approx.lnz,
            exact.lnz
        );
    }

    #[test]
    fn permutation_is_bijective() {
        let triplets = vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 2.0),
            Triplet::new(0, 2, 1.0),
            Triplet::new(2, 0, 1.0),
        ];
        let sym = analyze(3, &triplets, Ordering::MinDegree);
        let mut seen = [false; 3];
        for &p in &sym.perm {
            assert!(p < 3 && !seen[p]);
            seen[p] = true;
        }
        for i in 0..3 {
            assert_eq!(sym.perm[sym.iperm[i]], i);
        }
    }
}
