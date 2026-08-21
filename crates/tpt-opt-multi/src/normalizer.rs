//! Min–max objective normalisation for disparate scales.

/// Normalises objective vectors to a common `[0, 1]` box using per-objective
/// min/max bounds. Values outside the observed range are clamped.
#[derive(Debug, Clone)]
pub struct ObjectiveNormalizer {
    mins: Vec<f64>,
    maxs: Vec<f64>,
}

impl ObjectiveNormalizer {
    /// Build a normalizer from observed objective vectors (one per row).
    pub fn from_samples(samples: &[Vec<f64>]) -> Self {
        assert!(!samples.is_empty());
        let m = samples[0].len();
        let mut mins = vec![f64::INFINITY; m];
        let mut maxs = vec![f64::NEG_INFINITY; m];
        for s in samples {
            for (k, &v) in s.iter().enumerate() {
                if v < mins[k] {
                    mins[k] = v;
                }
                if v > maxs[k] {
                    maxs[k] = v;
                }
            }
        }
        Self { mins, maxs }
    }

    /// Number of objectives.
    pub fn dim(&self) -> usize {
        self.mins.len()
    }

    /// Normalise a single objective vector into `[0, 1]` per dimension.
    pub fn normalize(&self, v: &[f64]) -> Vec<f64> {
        v.iter()
            .enumerate()
            .map(|(k, &x)| {
                let span = self.maxs[k] - self.mins[k];
                if span <= 1e-12 {
                    0.0
                } else {
                    ((x - self.mins[k]) / span).clamp(0.0, 1.0)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_into_unit_box() {
        let n = ObjectiveNormalizer::from_samples(&[
            vec![0.0, 10.0],
            vec![2.0, 20.0],
            vec![1.0, 15.0],
        ]);
        let z = n.normalize(&[0.0, 10.0]);
        assert!((z[0]).abs() < 1e-9);
        assert!((z[1]).abs() < 1e-9);
        let o = n.normalize(&[2.0, 20.0]);
        assert!((o[0] - 1.0).abs() < 1e-9);
        assert!((o[1] - 1.0).abs() < 1e-9);
    }
}
