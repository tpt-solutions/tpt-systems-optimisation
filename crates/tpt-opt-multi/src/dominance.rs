//! Pareto dominance, Pareto-front extraction and the additive epsilon indicator.

/// Returns `true` if objective vector `a` **Pareto-dominates** `b`: `a` is no
/// worse than `b` on every objective and strictly better on at least one.
///
/// Both vectors must have equal length; the lower value is assumed better
/// (minimisation). For maximisation, negate the objectives before comparing.
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let mut strictly_better = false;
    for i in 0..a.len() {
        if a[i] > b[i] + 1e-9 {
            return false;
        }
        if a[i] < b[i] - 1e-9 {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Extract the non-dominated (Pareto-optimal) subset of `points`, returning the
/// indices into `points` of the members of the Pareto front, in input order.
pub fn pareto_front(points: &[Vec<f64>]) -> Vec<usize> {
    let mut front = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let dominated = points.iter().any(|q| q != p && dominates(q, p));
        if !dominated {
            front.push(i);
        }
    }
    front
}

/// Additive epsilon indicator `I_ε(a, b)`: the smallest `ε` such that every point
/// in `b` is `ε`-dominated by some point in `a`. A value of `0.0` (or negative)
/// means `a` weakly dominates `b`. Lower is better.
pub fn epsilon_indicator(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut worst = f64::NEG_INFINITY;
    for bp in b {
        let mut best = f64::INFINITY;
        for ap in a {
            let mut eps = f64::NEG_INFINITY;
            for k in 0..ap.len() {
                eps = eps.max(ap[k] - bp[k]);
            }
            best = best.min(eps);
        }
        worst = worst.max(best);
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominance_basics() {
        assert!(dominates(&[1.0, 1.0], &[2.0, 2.0]));
        assert!(!dominates(&[2.0, 1.0], &[1.0, 2.0]));
        assert!(!dominates(&[1.0, 2.0], &[1.0, 2.0]));
    }

    #[test]
    fn front_of_triangle() {
        // Minimise (x, y) s.t. x + y <= 1, x,y in [0,1]: the front is the segment.
        let pts = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0], vec![0.7, 0.7]];
        let mut f = pareto_front(&pts);
        f.sort_unstable();
        assert_eq!(f, vec![0, 1, 2]);
    }

    #[test]
    fn epsilon_indicator_zero_when_covering() {
        let a = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![vec![0.5, 0.5]];
        assert!(epsilon_indicator(&a, &b) <= 1e-9);
    }
}
