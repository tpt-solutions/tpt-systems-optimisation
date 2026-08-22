//! Special ordered sets (SOS) for the MILP solver.
//!
//! Two kinds are supported:
//!
//! - [`SosType::Sos1`] — *at most one* variable in the set may be non-zero.
//!   Typical use: modelling discrete choices / piecewise selections.
//! - [`SosType::Sos2`] — *at most two adjacent* variables (with respect to the
//!   given weight ordering) may be non-zero. Typical use: convex-combination
//!   interpolation over ordered breakpoints (piecewise-linear modelling).
//!
//! Sets are enforced by specialised branching in the branch-and-bound tree
//! ([`crate::MilpSolver`]): a violated SOS set is branched by splitting its
//! member list into two contiguous halves (for SOS2 the split respects the
//! weight order), which finitely converges because each branch strictly
//! shrinks the largest violable adjacency span.

use std::vec::Vec;

/// Kind of special ordered set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SosType {
    /// At most one member may be non-zero.
    Sos1,
    /// At most two *adjacent* members (in weight order) may be non-zero.
    Sos2,
}

/// A special ordered set over model variable indices with reference weights.
///
/// Weights impose the ordering used by SOS2 adjacency (and give SOS1 sets a
/// canonical order for deterministic branching). Variable indices refer to the
/// model solved by [`crate::MilpSolver`].
#[derive(Debug, Clone, PartialEq)]
pub struct SosSet {
    /// SOS1 or SOS2.
    pub kind: SosType,
    /// Member variable indices, stored in ascending weight order.
    pub vars: Vec<usize>,
    /// Reference weights (strictly increasing after construction).
    pub weights: Vec<f64>,
}

impl SosSet {
    /// Build an SOS set from `(variable, weight)` pairs. The pairs are sorted
    /// by weight so `vars` ends up in ascending weight order.
    ///
    /// # Panics
    /// Panics if `pairs` is empty or contains duplicate weights (the ordering
    /// must be strict for SOS2 semantics).
    pub fn new(kind: SosType, mut pairs: Vec<(usize, f64)>) -> Self {
        assert!(!pairs.is_empty(), "SOS set must have at least one member");
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("SOS weights must not be NaN"));
        for w in pairs.windows(2) {
            assert!(
                w[0].1 < w[1].1,
                "SOS weights must be strictly increasing (got {} twice)",
                w[0].1
            );
        }
        Self {
            kind,
            vars: pairs.iter().map(|&(v, _)| v).collect(),
            weights: pairs.iter().map(|&(_, w)| w).collect(),
        }
    }

    /// Number of members.
    pub fn len(&self) -> usize {
        self.vars.len()
    }

    /// `true` if the set has no members.
    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    /// Check whether `x` satisfies this set within `tol`.
    pub fn is_satisfied(&self, x: &[f64], tol: f64) -> bool {
        match self.kind {
            SosType::Sos1 => {
                let nonzeros = self.vars.iter().filter(|&&v| x[v].abs() > tol).count();
                nonzeros <= 1
            }
            SosType::Sos2 => {
                let nz: Vec<usize> = self
                    .vars
                    .iter()
                    .enumerate()
                    .filter(|&(_, &v)| x[v].abs() > tol)
                    .map(|(i, _)| i)
                    .collect();
                match nz.len() {
                    0 | 1 => true,
                    2 => nz[1] == nz[0] + 1, // adjacent in weight order
                    _ => false,
                }
            }
        }
    }

    /// Split the set into two halves for branching. Returns `(left, right)`
    /// member-index ranges `[0, mid)` and `[mid, len)`; each half becomes a
    /// tighter sub-set in the child nodes. For SOS1 with a single member there
    /// is nothing to branch on (`None`).
    pub fn split(&self) -> Option<(usize, usize)> {
        if self.vars.len() < 2 {
            return None;
        }
        Some((0, self.vars.len() / 2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sos_new_sorts_by_weight() {
        let s = SosSet::new(SosType::Sos2, vec![(3, 30.0), (1, 10.0), (2, 20.0)]);
        assert_eq!(s.vars, vec![1, 2, 3]);
        assert_eq!(s.weights, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn sos1_satisfaction() {
        let s = SosSet::new(SosType::Sos1, vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        let x = [0.0, 5.0, 0.0];
        assert!(s.is_satisfied(&x, 1e-9));
        let bad = [1.0, 0.0, 2.0];
        assert!(!s.is_satisfied(&bad, 1e-9));
    }

    #[test]
    fn sos2_adjacency() {
        let s = SosSet::new(SosType::Sos2, vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        // Adjacent pair (indices 0 and 1) is fine.
        assert!(s.is_satisfied(&[0.5, 0.5, 0.0], 1e-9));
        // Non-adjacent pair (0 and 2) violates.
        assert!(!s.is_satisfied(&[0.5, 0.0, 0.5], 1e-9));
        // Three non-zeros violate.
        assert!(!s.is_satisfied(&[0.3, 0.3, 0.3], 1e-9));
    }

    #[test]
    fn sos_split_halves() {
        let s = SosSet::new(SosType::Sos1, vec![(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)]);
        assert_eq!(s.split(), Some((0, 2)));
        let single = SosSet::new(SosType::Sos1, vec![(0, 1.0)]);
        assert_eq!(single.split(), None);
    }

    #[test]
    #[should_panic(expected = "strictly increasing")]
    fn sos_rejects_duplicate_weights() {
        let _ = SosSet::new(SosType::Sos2, vec![(0, 1.0), (1, 1.0)]);
    }
}
