//! Convergence certificates: per-iteration bounds and duality-gap tracking.

use std::vec::Vec;

/// A convergence certificate for one iteration of a decomposition loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConvergenceCertificate {
    /// Valid lower bound on the optimal objective (master problem value).
    pub lower_bound: f64,
    /// Best known feasible objective (upper bound).
    pub upper_bound: f64,
    /// Absolute duality gap `upper - lower`.
    pub gap: f64,
}

impl ConvergenceCertificate {
    /// Relative gap `|gap| / max(1, |upper|)`.
    pub fn relative_gap(&self) -> f64 {
        self.gap.abs() / self.upper_bound.abs().max(1.0)
    }

    /// Whether the certificate proves optimality within tolerances.
    pub fn is_optimal(&self, abs_tol: f64, rel_tol: f64) -> bool {
        self.gap <= abs_tol || self.relative_gap() <= rel_tol
    }
}

/// History of certificates produced by a solve.
#[derive(Debug, Clone, Default)]
pub struct CertificateHistory {
    certs: Vec<ConvergenceCertificate>,
}

impl CertificateHistory {
    /// Record one certificate.
    pub fn push(&mut self, cert: ConvergenceCertificate) {
        self.certs.push(cert);
    }

    /// All recorded certificates, in order.
    pub fn iter(&self) -> impl Iterator<Item = &ConvergenceCertificate> {
        self.certs.iter()
    }

    /// Number of recorded iterations.
    pub fn len(&self) -> usize {
        self.certs.len()
    }

    /// Whether any certificate was recorded.
    pub fn is_empty(&self) -> bool {
        self.certs.is_empty()
    }

    /// The final certificate, if any.
    pub fn last(&self) -> Option<&ConvergenceCertificate> {
        self.certs.last()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gap_and_optimality_checks() {
        let c = ConvergenceCertificate { lower_bound: 2.2, upper_bound: 2.25, gap: 0.05 };
        assert!((c.relative_gap() - 0.05 / 2.25).abs() < 1e-12);
        assert!(!c.is_optimal(1e-6, 1e-6));
        assert!(c.is_optimal(0.1, 1e-6));
        let mut h = CertificateHistory::default();
        h.push(c);
        assert_eq!(h.len(), 1);
        assert_eq!(h.last().unwrap().gap, 0.05);
    }
}