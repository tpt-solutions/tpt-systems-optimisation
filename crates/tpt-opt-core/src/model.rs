//! Canonical problem representation: variables, constraints, objectives, model.
//!
//! The [`Model`] is a *linear* canonical form (the shared backbone every solver
//! crate understands). Non-linear extensions (e.g. [`crate::custom::CustomConstraint`])
//! attach on top without disturbing the linear spine.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::bounds::{Bound, VarBound, VarType};
use crate::error::{InfeasibilityReport, OptError};

/// Optimisation sense of an [`Objective`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Sense {
    /// Minimise the objective.
    Minimize,
    /// Maximise the objective.
    Maximize,
}

/// A decision variable: an index plus its [`VarBound`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Variable {
    /// Stable index within the model's variable vector.
    pub index: usize,
    /// Bound + kind describing the variable.
    pub bound: VarBound,
    /// Convenience mirror of `bound.kind`.
    pub kind: VarType,
}

impl Variable {
    /// Create a variable at `index` with the given bound.
    pub fn new(index: usize, bound: VarBound) -> Self {
        let kind = bound.kind;
        Self { index, bound, kind }
    }

    /// `true` if this variable is binary.
    pub fn is_binary(&self) -> bool {
        self.kind == VarType::Binary
    }

    /// `true` if this variable must be integral.
    pub fn is_integer(&self) -> bool {
        self.bound.is_integral()
    }
}

/// A single sparse linear constraint: `lower <= sum_i coeffs[i] * x[indices[i]] <= upper`.
///
/// Serde representation: the row bounds are `Option<f64>` — `None` encodes
/// the corresponding ±infinity (as used by [`Constraint::le`]/[`Constraint::ge`]),
/// so one-sided rows round-trip through formats without non-finite float
/// support (JSON).
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    /// Variable indices referenced by this constraint (sparse pattern).
    pub indices: Vec<usize>,
    /// Matching coefficients for `indices`.
    pub coeffs: Vec<f64>,
    /// Lower bound of the constraint range.
    pub lower: f64,
    /// Upper bound of the constraint range.
    pub upper: f64,
    /// `true` when this row is backed by a [`crate::custom::CustomConstraint`].
    pub is_custom: bool,
}

impl Constraint {
    /// Build a ranged constraint, validating index/coeff alignment.
    pub fn new(
        indices: Vec<usize>,
        coeffs: Vec<f64>,
        lower: f64,
        upper: f64,
    ) -> Result<Self, OptError> {
        if indices.len() != coeffs.len() {
            return Err(OptError::invalid_model(
                "constraint indices and coeffs must have equal length",
            ));
        }
        if lower > upper {
            return Err(OptError::invalid_model("constraint lower bound exceeds upper bound"));
        }
        Ok(Self { indices, coeffs, lower, upper, is_custom: false })
    }

    /// An equality constraint `sum coeffs[i]*x[i] == rhs`.
    pub fn equality(indices: Vec<usize>, coeffs: Vec<f64>, rhs: f64) -> Self {
        Self { indices, coeffs, lower: rhs, upper: rhs, is_custom: false }
    }

    /// A less-or-equal constraint `sum coeffs[i]*x[i] <= upper`.
    pub fn le(indices: Vec<usize>, coeffs: Vec<f64>, upper: f64) -> Self {
        Self { indices, coeffs, lower: f64::NEG_INFINITY, upper, is_custom: false }
    }

    /// A greater-or-equal constraint `sum coeffs[i]*x[i] >= lower`.
    pub fn ge(indices: Vec<usize>, coeffs: Vec<f64>, lower: f64) -> Self {
        Self { indices, coeffs, lower, upper: f64::INFINITY, is_custom: false }
    }

    /// Evaluate the left-hand side at `x`.
    pub fn eval(&self, x: &[f64]) -> f64 {
        let mut s = 0.0;
        for (&i, &c) in self.indices.iter().zip(self.coeffs.iter()) {
            s += c * x[i];
        }
        s
    }

    /// `true` if the constraint is satisfied at `x` within `tol`.
    pub fn is_satisfied(&self, x: &[f64], tol: f64) -> bool {
        let lhs = self.eval(x);
        lhs >= self.lower - tol && lhs <= self.upper + tol
    }

    /// Slack of the constraint at `x`: negative means violated.
    pub fn slack(&self, x: &[f64]) -> f64 {
        let lhs = self.eval(x);
        let above = lhs - self.upper;
        let below = self.lower - lhs;
        if above > 0.0 {
            -above
        } else if below > 0.0 {
            -below
        } else {
            above.max(below)
        }
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::Constraint;
    use alloc::vec::Vec;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Wire form: `None` row bounds encode the corresponding ±infinity.
    #[derive(Serialize, Deserialize)]
    struct Repr {
        indices: Vec<usize>,
        coeffs: Vec<f64>,
        lower: Option<f64>,
        upper: Option<f64>,
        is_custom: bool,
    }

    impl Serialize for Constraint {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            Repr {
                indices: self.indices.clone(),
                coeffs: self.coeffs.clone(),
                lower: self.lower.is_finite().then_some(self.lower),
                upper: self.upper.is_finite().then_some(self.upper),
                is_custom: self.is_custom,
            }
            .serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Constraint {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let r = Repr::deserialize(deserializer)?;
            Ok(Constraint {
                indices: r.indices,
                coeffs: r.coeffs,
                lower: r.lower.unwrap_or(f64::NEG_INFINITY),
                upper: r.upper.unwrap_or(f64::INFINITY),
                is_custom: r.is_custom,
            })
        }
    }
}

/// A linear objective: `sense . (constant + sum_i coeffs[i]*x[indices[i]])`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Objective {
    /// Minimise or maximise.
    pub sense: Sense,
    /// Variable indices referenced by the objective.
    pub indices: Vec<usize>,
    /// Matching objective coefficients.
    pub coeffs: Vec<f64>,
    /// Constant offset added to the linear term.
    pub constant: f64,
}

impl Objective {
    /// Minimise `constant + sum coeffs[i]*x[i]`.
    pub fn minimize(indices: Vec<usize>, coeffs: Vec<f64>) -> Self {
        Self { sense: Sense::Minimize, indices, coeffs, constant: 0.0 }
    }

    /// Maximise `constant + sum coeffs[i]*x[i]`.
    pub fn maximize(indices: Vec<usize>, coeffs: Vec<f64>) -> Self {
        Self { sense: Sense::Maximize, indices, coeffs, constant: 0.0 }
    }

    /// Evaluate the objective at `x` (as a minimisation value; maximisation
    /// callers negate as needed).
    pub fn eval(&self, x: &[f64]) -> f64 {
        let mut s = self.constant;
        for (&i, &c) in self.indices.iter().zip(self.coeffs.iter()) {
            s += c * x[i];
        }
        s
    }
}

/// The canonical optimisation model: variables, linear constraints, objective.
///
/// With the optional `serde` feature the whole model round-trips through any
/// serde format — infinite variable bounds and one-sided row bounds are
/// encoded as `null` on the wire (see [`Constraint`] and
/// [`crate::bounds::Bound`]) — enabling warm-start caching and reproducible
/// bug reports.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Model {
    /// Total number of variables (including any not explicitly added).
    pub num_vars: usize,
    /// Variables, indexed by `index`.
    pub variables: Vec<Variable>,
    /// Linear (and custom-tagged) constraints.
    pub constraints: Vec<Constraint>,
    /// Objective function.
    pub objective: Objective,
    /// Optional human-readable name.
    pub name: Option<String>,
}

impl Model {
    /// Create an empty model over `num_vars` variables (all free continuous).
    pub fn new(num_vars: usize) -> Self {
        let variables = (0..num_vars)
            .map(|i| {
                Variable::new(
                    i,
                    VarBound::continuous(Bound::UNBOUNDED_LOWER, Bound::UNBOUNDED_UPPER),
                )
            })
            .collect();
        Self {
            num_vars,
            variables,
            constraints: Vec::new(),
            objective: Objective::minimize(Vec::new(), Vec::new()),
            name: None,
        }
    }

    /// Create an empty model with a name.
    pub fn with_name(num_vars: usize, name: impl Into<String>) -> Self {
        let mut m = Self::new(num_vars);
        m.name = Some(name.into());
        m
    }

    /// Number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Append a variable with the given bound; returns its assigned index.
    pub fn add_variable(&mut self, bound: VarBound) -> usize {
        let index = self.num_vars;
        self.variables.push(Variable::new(index, bound));
        self.num_vars += 1;
        index
    }

    /// Append a constraint, returning its index.
    pub fn add_constraint(&mut self, c: Constraint) -> usize {
        self.constraints.push(c);
        self.constraints.len() - 1
    }

    /// Replace the objective.
    pub fn set_objective(&mut self, o: Objective) {
        self.objective = o;
    }

    /// Validate structural integrity (index ranges, bound sanity).
    pub fn validate(&self) -> Result<(), OptError> {
        for (vi, v) in self.variables.iter().enumerate() {
            if v.index != vi {
                return Err(OptError::invalid_model(format!(
                    "variable at position {vi} has mismatched index {}",
                    v.index
                )));
            }
            if v.bound.bound.lower > v.bound.bound.upper {
                return Err(OptError::infeasible(
                    InfeasibilityReport::new("variable bound lower exceeds upper")
                        .with_conflict(vi),
                ));
            }
        }
        let nv = self.num_vars;
        for (ci, c) in self.constraints.iter().enumerate() {
            if c.indices.len() != c.coeffs.len() {
                return Err(OptError::invalid_model(format!(
                    "constraint {ci} has mismatched indices/coeffs lengths"
                )));
            }
            for &i in &c.indices {
                if i >= nv {
                    return Err(OptError::invalid_model(format!(
                        "constraint {ci} references out-of-range variable {i}"
                    )));
                }
            }
            for &i in &self.objective.indices {
                if i >= nv {
                    return Err(OptError::invalid_model(format!(
                        "objective references out-of-range variable {i}"
                    )));
                }
            }
        }
        Ok(())
    }
}
