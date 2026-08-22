//! Indicator constraints: "if binary `y = v` then a linear row must hold".
//!
//! An [`IndicatorConstraint`] couples a binary indicator variable to an
//! ordinary linear row. Unlike a plain big-M formulation written by hand, the
//! solver derives the tightest big-M *from the current variable bounds* at
//! solve time, so the relaxation stays as strong as the bounds allow and the
//! row is completely ignored when the indicator is off.
//!
//! Semantics:
//!
//! ```text
//! y = trigger  =>  lower <= sum_i coeffs[i] * x[indices[i]] <= upper
//! ```
//!
//! When `y` takes the opposite value the row is not enforced.

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model};

/// Direction of the indicator implication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// The row is enforced when the binary equals 1.
    One,
    /// The row is enforced when the binary equals 0.
    Zero,
}

/// An indicator constraint over model variables (see [module docs](self)).
#[derive(Debug, Clone, PartialEq)]
pub struct IndicatorConstraint {
    /// Index of the binary indicator variable.
    pub indicator: usize,
    /// Which value of the indicator activates the row.
    pub trigger: Trigger,
    /// Row variable indices.
    pub indices: Vec<usize>,
    /// Row coefficients.
    pub coeffs: Vec<f64>,
    /// Row lower bound (`-inf` for one-sided rows).
    pub lower: f64,
    /// Row upper bound (`+inf` for one-sided rows).
    pub upper: f64,
}

impl IndicatorConstraint {
    /// "if `y == 1` then `sum coeffs[i]*x[i] <= upper`".
    pub fn if_le(indicator: usize, indices: Vec<usize>, coeffs: Vec<f64>, upper: f64) -> Self {
        Self { indicator, trigger: Trigger::One, indices, coeffs, lower: f64::NEG_INFINITY, upper }
    }

    /// "if `y == 1` then `sum coeffs[i]*x[i] >= lower`".
    pub fn if_ge(indicator: usize, indices: Vec<usize>, coeffs: Vec<f64>, lower: f64) -> Self {
        Self { indicator, trigger: Trigger::One, indices, coeffs, lower, upper: f64::INFINITY }
    }

    /// "if `y == 0` then `lower <= sum coeffs[i]*x[i] <= upper`" (ranged).
    pub fn when_zero_ranged(
        indicator: usize,
        indices: Vec<usize>,
        coeffs: Vec<f64>,
        lower: f64,
        upper: f64,
    ) -> Self {
        Self { indicator, trigger: Trigger::Zero, indices, coeffs, lower, upper }
    }

    /// Expand into concrete linear [`Constraint`]s using big-M coefficients
    /// derived from the *global* variable bounds of `model`.
    ///
    /// For each finite side of the row one row is emitted, with the smallest
    /// `M` that makes it vacuous when the indicator is inactive:
    ///
    /// ```text
    /// Trigger::One:  a'x + M*y <= U + M   (M = max(0, hi - U))
    ///                a'x - M*y >= L - M   (M = max(0, L - lo))
    /// Trigger::Zero: a'x - M*y <= U       (M = max(0, hi - U))
    ///                a'x + M*y >= L       (M = max(0, L - lo))
    /// ```
    ///
    /// where `[lo, hi]` is the activity range of `a'x` over the variable
    /// bounds. If any row variable is unbounded the expansion is impossible
    /// and `None` is returned — callers should tighten variable bounds.
    pub fn expand(&self, model: &Model) -> Option<Vec<Constraint>> {
        // Activity range of the row over the global variable bounds.
        let mut lo = 0.0f64;
        let mut hi = 0.0f64;
        for (&v, &c) in self.indices.iter().zip(self.coeffs.iter()) {
            let b = &model.variables[v].bound.bound;
            let (vl, vu) = (b.lower, b.upper);
            if !vl.is_finite() || !vu.is_finite() {
                return None;
            }
            if c >= 0.0 {
                lo += c * vl;
                hi += c * vu;
            } else {
                lo += c * vu;
                hi += c * vl;
            }
        }

        let mut rows = Vec::new();
        if self.upper.is_finite() {
            let m = (hi - self.upper).max(0.0);
            let (yc, rhs) = match self.trigger {
                Trigger::One => (m, self.upper + m),
                Trigger::Zero => (-m, self.upper),
            };
            let mut idx = self.indices.clone();
            let mut coefs = self.coeffs.clone();
            idx.push(self.indicator);
            coefs.push(yc);
            rows.push(Constraint::le(idx, coefs, rhs));
        }
        if self.lower.is_finite() {
            let m = (self.lower - lo).max(0.0);
            let (yc, rhs) = match self.trigger {
                Trigger::One => (-m, self.lower - m),
                Trigger::Zero => (m, self.lower),
            };
            let mut idx = self.indices.clone();
            let mut coefs = self.coeffs.clone();
            idx.push(self.indicator);
            coefs.push(yc);
            rows.push(Constraint::ge(idx, coefs, rhs));
        }
        Some(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_opt_core::bounds::VarBound;

    fn small_model() -> Model {
        let mut m = Model::new(3);
        m.variables[0].bound = VarBound::binary();
        m.variables[1].bound = VarBound::continuous(0.0, 10.0);
        m.variables[2].bound = VarBound::continuous(-5.0, 5.0);
        m
    }

    #[test]
    fn expansion_enforces_row_when_active() {
        let ind = IndicatorConstraint::if_le(0, vec![1], vec![1.0], 4.0);
        let rows = ind.expand(&small_model()).expect("expandable");
        assert_eq!(rows.len(), 1);
        // y=1 forces x1 <= 4.
        let x = [1.0, 7.0, 0.0];
        assert!(!rows.iter().all(|r| r.is_satisfied(&x, 1e-9)));
        // y=0 leaves x1 free up to its own bound.
        let x = [0.0, 9.0, 0.0];
        assert!(rows.iter().all(|r| r.is_satisfied(&x, 1e-9)));
        // y=1 with x1 = 3 satisfies.
        let x = [1.0, 3.0, 0.0];
        assert!(rows.iter().all(|r| r.is_satisfied(&x, 1e-9)));
    }

    #[test]
    fn zero_trigger_expansion() {
        let ind = IndicatorConstraint::when_zero_ranged(0, vec![2], vec![1.0], -1.0, 1.0);
        let rows = ind.expand(&small_model()).expect("expandable");
        assert_eq!(rows.len(), 2);
        // y=0 forces -1 <= x2 <= 1.
        let x = [0.0, 0.0, 3.0];
        assert!(!rows.iter().all(|r| r.is_satisfied(&x, 1e-9)));
        // y=1 frees x2 within [-5, 5].
        let x = [1.0, 0.0, 4.0];
        assert!(rows.iter().all(|r| r.is_satisfied(&x, 1e-9)));
    }

    #[test]
    fn unbounded_row_is_not_expandable() {
        let mut m = small_model();
        m.variables[1].bound = VarBound::continuous(0.0, f64::INFINITY);
        let ind = IndicatorConstraint::if_le(0, vec![1], vec![1.0], 4.0);
        assert!(ind.expand(&m).is_none());
    }
}
