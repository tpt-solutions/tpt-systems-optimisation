//! The Hungarian (Kuhn–Munkres) algorithm for the assignment problem.
//!
//! Given a square cost matrix `C`, it finds a permutation `π` minimising
//! `sum_i C[i][π(i)]` (a minimum-cost perfect matching / assignment).
//! Rectangular matrices are padded to square with zero-cost dummy rows/columns.

use std::vec::Vec;

/// Result of an assignment problem.
#[derive(Debug, Clone)]
pub struct AssignmentResult {
    /// Total cost of the optimal assignment.
    pub total_cost: f64,
    /// `assignment[i]` is the column assigned to row `i`.
    pub assignment: Vec<usize>,
}

/// Solve the assignment problem (minimisation) on `cost`.
///
/// `cost` must be non-empty; rectangular matrices are padded to square.
pub fn hungarian(cost: &[Vec<f64>]) -> AssignmentResult {
    let rows = cost.len();
    let cols = cost.first().map_or(0, |r| r.len());
    let n = rows.max(cols);

    // Build a square (n+1) x (n+1) 1-indexed matrix, padding with zeros.
    let mut a: Vec<Vec<f64>> = vec![vec![0.0; n + 1]; n + 1];
    for i in 0..rows {
        for j in 0..cols {
            a[i + 1][j + 1] = cost[i][j];
        }
    }

    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1]; // p[j] = row matched to column j
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let c = a[i0][j] - u[i0] - v[j];
                    if c < minv[j] {
                        minv[j] = c;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![0usize; n];
    let mut total = 0.0f64;
    for (j, &pj) in p.iter().enumerate() {
        if j == 0 {
            continue;
        }
        if pj != 0 {
            let r = pj - 1;
            assignment[r] = j - 1;
            // Only count real (non-padded) entries.
            if r < rows && (j - 1) < cols {
                total += cost[r][j - 1];
            }
        }
    }

    AssignmentResult { total_cost: total, assignment }
}

/// Solve the assignment problem (maximisation) by negating costs.
pub fn hungarian_maximize(cost: &[Vec<f64>]) -> AssignmentResult {
    let rows = cost.len();
    let _cols = cost.first().map_or(0, |r| r.len());
    let mut neg: Vec<Vec<f64>> = Vec::with_capacity(rows);
    for r in cost {
        neg.push(r.iter().map(|&x| -x).collect());
    }
    let mut res = hungarian(&neg);
    res.total_cost = -res.total_cost;
    res
}
