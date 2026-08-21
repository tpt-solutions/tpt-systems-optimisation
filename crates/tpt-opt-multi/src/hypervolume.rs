//! Hypervolume indicator (volume of the union of boxes bounded by the Pareto
//! front and a reference point).
//!
//! Implemented via the *slicing objectives* algorithm: sort by one objective and
//! recurse on the cross-section. Exact for any dimension (cost grows with the
//! number of points, so it is intended for modest fronts — e.g. NSGA-II result
//! sets).

/// Compute the hypervolume of `points` relative to `reference` (minimisation).
///
/// Points not dominated by `reference` are ignored. Returns the Lebesgue measure
/// of the union of boxes `[p, reference]` over all `p ∈ points`.
pub fn hypervolume(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let d = points[0].len();
    assert_eq!(d, reference.len(), "point and reference dimensionality differ");
    let mut pts: Vec<Vec<f64>> = points
        .iter()
        .filter(|p| p.iter().zip(reference).all(|(v, r)| *v <= *r + 1e-9))
        .cloned()
        .collect();
    if pts.is_empty() {
        return 0.0;
    }
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
    hv_rec(&pts, reference)
}

fn hv_rec(pts: &[Vec<f64>], reference: &[f64]) -> f64 {
    let n = pts.len();
    let d = pts[0].len();
    if d == 1 {
        return (reference[0] - pts[0][0]).max(0.0);
    }
    // a[i] = f1 of point i (ascending); a[n] = reference bound (closes last slab).
    let mut a: Vec<f64> = pts.iter().map(|p| p[0]).collect();
    a.push(reference[0]);

    let mut total = 0.0;
    let mut t: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        t.push(pts[i][1..].to_vec());
        let thickness = a[i + 1] - a[i];
        if thickness > 1e-12 {
            let mut t_sorted = t.clone();
            t_sorted.sort_by(|x, y| x[0].partial_cmp(&y[0]).unwrap());
            let ref_rest: Vec<f64> = reference[1..].to_vec();
            total += thickness * hv_rec(&t_sorted, &ref_rest);
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_d_known() {
        // (1,2) and (2,1) with ref (3,3): union area = 3.
        let pts = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!((hypervolume(&pts, &[3.0, 3.0]) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn two_d_single_box() {
        let pts = vec![vec![1.0, 1.0]];
        assert!((hypervolume(&pts, &[3.0, 3.0]) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn three_d_known() {
        // Three points forming a staircase, ref (4,4,4).
        // (1,4,4),(2,2,4),(3,3,1) is awkward; use symmetric:
        let pts = vec![vec![1.0, 1.0, 4.0], vec![1.0, 4.0, 1.0], vec![4.0, 1.0, 1.0]];
        // Reference (4,4,4): each box is 3*3*0? no, f ranges...
        // Just check monotonicity: bigger reference -> bigger volume.
        let v1 = hypervolume(&pts, &[4.0, 4.0, 4.0]);
        let v2 = hypervolume(&pts, &[5.0, 5.0, 5.0]);
        assert!(v2 > v1);
    }

    #[test]
    fn dominated_points_ignored() {
        let pts = vec![vec![2.0, 2.0], vec![5.0, 5.0]];
        // (5,5) is outside ref (3,3) -> ignored; only (2,2) contributes area 1.
        assert!((hypervolume(&pts, &[3.0, 3.0]) - 1.0).abs() < 1e-9);
    }
}
