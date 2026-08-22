//! Decision-making utilities over a Pareto front: knee-point detection,
//! trade-off analysis and solution clustering.
//!
//! These helpers take a set of objective vectors (minimisation, lower is
//! better) — typically a Pareto front — and distil it into actionable
//! information: which point is the "best compromise" ([`knee_point`]), what
//! each point costs in trade-off terms ([`tradeoff_ratios`]), and how the
//! front organises into distinct compromise regions ([`cluster_solutions`]).

use std::vec::Vec;

use tpt_math_prob::{Rng, Xoshiro256};

/// Normalise each objective to [0, 1] using the front's min/max.
fn normalize(objs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = objs.first().map(|o| o.len()).unwrap_or(0);
    if objs.is_empty() || m == 0 {
        return Vec::new();
    }
    let lo: Vec<f64> =
        (0..m).map(|j| objs.iter().map(|o| o[j]).fold(f64::INFINITY, f64::min)).collect();
    let hi: Vec<f64> =
        (0..m).map(|j| objs.iter().map(|o| o[j]).fold(f64::NEG_INFINITY, f64::max)).collect();
    objs.iter()
        .map(|o| {
            (0..m)
                .map(|j| {
                    let span = hi[j] - lo[j];
                    if span > 1e-12 {
                        (o[j] - lo[j]) / span
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

/// Knee-point detection: the index of the front member that maximises the
/// normalised distance from the chord (hyperplane) joining the axis-optimal
/// extreme points — the classic "best compromise" heuristic for convex
/// fronts.
///
/// For 2-objective fronts this is maximum perpendicular distance from the
/// chord between the two axis optima. For M ≥ 3 objectives it uses the
/// achievement-scalarising-function knee: the point minimising the maximum
/// normalised objective (the most balanced trade-off). Returns `None` for
/// an empty front. Ties resolve to the lowest index (deterministic).
pub fn knee_point(objs: &[Vec<f64>]) -> Option<usize> {
    if objs.is_empty() {
        return None;
    }
    let norm = normalize(objs);
    let m = norm[0].len();
    if m == 2 && norm.len() >= 2 {
        // Extremes: min f1 and min f2 (ties -> lowest index).
        let e0 = (0..norm.len())
            .min_by(|&a, &b| {
                norm[a][0].partial_cmp(&norm[b][0]).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        let e1 = (0..norm.len())
            .min_by(|&a, &b| {
                norm[a][1].partial_cmp(&norm[b][1]).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        let (ax, ay) = (norm[e0][0], norm[e0][1]);
        let (bx, by) = (norm[e1][0], norm[e1][1]);
        let (dx, dy) = (bx - ax, by - ay);
        let len2 = dx * dx + dy * dy;
        let scored: Vec<(usize, f64)> = norm
            .iter()
            .enumerate()
            .map(|(i, p)| {
                // Perpendicular distance to the infinite chord line.
                let d = if len2 > 1e-18 {
                    ((p[0] - ax) * dy - (p[1] - ay) * dx).abs() / len2.sqrt()
                } else {
                    0.0
                };
                (i, d)
            })
            .collect();
        let max_d = scored.iter().map(|&(_, d)| d).fold(f64::NEG_INFINITY, f64::max);
        if max_d > 1e-9 {
            // Genuinely convex front: classic maximum-chord-distance knee.
            return scored
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i);
        }
        // Degenerate (e.g. linear) front: fall back to the point closest to
        // the ideal corner (L2 scalarisation).
        return norm
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.iter().map(|c| c * c).sum::<f64>()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);
    }
    // M ≥ 3: ASF knee — minimise the largest normalised objective.
    norm.iter()
        .enumerate()
        .map(|(i, p)| (i, p.iter().cloned().fold(f64::NEG_INFINITY, f64::max)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
}

/// Trade-off analysis at solution `idx`: an `m × m` matrix where entry
/// `(a, b)` estimates how much of objective `b`'s remaining range must be
/// sacrificed per unit of objective `a` gained, measured against the front's
/// envelope:
///
/// ```text
/// T[a][b] = (max_b − f_b[idx]) / (f_a[idx] − min_a)
/// ```
///
/// i.e. moving from `idx` toward the best-known value of `a` costs
/// `(max_b − f_b)` of objective `b`. Diagonal entries are 0; degenerate
/// divisions (zero denominator or zero numerator) yield 0. All quantities
/// are in normalised units so objectives on disparate scales compare fairly.
pub fn tradeoff_ratios(objs: &[Vec<f64>], idx: usize) -> Vec<Vec<f64>> {
    let m = objs.first().map(|o| o.len()).unwrap_or(0);
    assert!(idx < objs.len(), "solution index out of range");
    let mut out = vec![vec![0.0f64; m]; m];
    if m == 0 {
        return out;
    }
    let lo: Vec<f64> =
        (0..m).map(|j| objs.iter().map(|o| o[j]).fold(f64::INFINITY, f64::min)).collect();
    let hi: Vec<f64> =
        (0..m).map(|j| objs.iter().map(|o| o[j]).fold(f64::NEG_INFINITY, f64::max)).collect();
    for a in 0..m {
        for (b, &hb) in hi.iter().enumerate() {
            if a == b {
                continue;
            }
            let gain_a = objs[idx][a] - lo[a]; // sacrifice in a (≥ 0)
            let cost_b = hb - objs[idx][b]; // potential improvement in b
            out[a][b] = if gain_a > 1e-12 && cost_b > 1e-12 { cost_b / gain_a } else { 0.0 };
        }
    }
    out
}

/// Cluster front members into `k` groups by k-means on the normalised
/// objective vectors. Initialisation is deterministic (spread seeding driven
/// by a fixed-seed RNG), assignments use Lloyd iterations until stable or
/// `max_iter` rounds elapse. Returns one cluster id per input point.
///
/// Panics if `k == 0` or `k > objs.len()`.
pub fn cluster_solutions(objs: &[Vec<f64>], k: usize) -> Vec<usize> {
    assert!(k >= 1, "need at least one cluster");
    assert!(k <= objs.len(), "more clusters than points");
    if k == objs.len() {
        return (0..objs.len()).collect();
    }
    let norm = normalize(objs);
    let dim = norm.first().map(|p| p.len()).unwrap_or(0);
    if dim == 0 {
        return vec![0; objs.len()];
    }

    // Deterministic spread initialisation: first centroid = point closest to
    // the ideal corner; subsequent centroids = points farthest from their
    // nearest existing centroid (max-min), tie-broken by index via jittered
    // keys from a fixed-seed RNG.
    let mut rng = Xoshiro256::new(0x5EED_0000_0000_0001u64);
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut first = 0usize;
    let mut best_d2 = f64::NEG_INFINITY;
    for (i, p) in norm.iter().enumerate() {
        let d2: f64 = p.iter().map(|c| c * c).sum::<f64>() + rng.next_f64() * 1e-9;
        if d2 > best_d2 {
            best_d2 = d2;
            first = i;
        }
    }
    centroids.push(norm[first].clone());
    while centroids.len() < k {
        let mut far = 0usize;
        let mut best_min = f64::NEG_INFINITY;
        for (i, p) in norm.iter().enumerate() {
            let dmin = centroids
                .iter()
                .map(|c| c.iter().zip(p.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>())
                .fold(f64::INFINITY, f64::min)
                + rng.next_f64() * 1e-9;
            if dmin > best_min {
                best_min = dmin;
                far = i;
            }
        }
        centroids.push(norm[far].clone());
    }

    // Lloyd iterations.
    let mut assign = vec![0usize; norm.len()];
    for _ in 0..100 {
        let mut changed = false;
        for (i, p) in norm.iter().enumerate() {
            let nearest = centroids
                .iter()
                .enumerate()
                .map(|(ci, c)| {
                    (ci, c.iter().zip(p.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>())
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(ci, _)| ci)
                .unwrap_or(0);
            if assign[i] != nearest {
                assign[i] = nearest;
                changed = true;
            }
        }
        // Recompute centroids (empty clusters keep their previous centre).
        for (ci, cent) in centroids.iter_mut().enumerate().take(k) {
            let members: Vec<&Vec<f64>> =
                norm.iter().enumerate().filter(|&(i, _)| assign[i] == ci).map(|(_, p)| p).collect();
            if !members.is_empty() {
                for j in 0..dim {
                    cent[j] = members.iter().map(|p| p[j]).sum::<f64>() / members.len() as f64;
                }
            }
        }
        if !changed {
            break;
        }
    }
    assign
}
