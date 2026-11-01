//! Expression DSL for building [`Model`]s with named variables and operator
//! syntax.
//!
//! [`ModelBuilder`] creates [`Var`] handles, [`LinExpr`] values combine them
//! with `+ - *`, and [`LinExpr::le`] / [`LinExpr::ge`] / [`LinExpr::eq`] turn an
//! expression into a [`LinRelation`] ready for [`ModelBuilder::constrain`].
//! Everything lowers to the canonical [`Model`] that every solver consumes.
//!
//! ```
//! use tpt_opt_core::expr::ModelBuilder;
//!
//! let mut b = ModelBuilder::new();
//! let x = b.continuous("x", 0.0, 10.0);
//! let y = b.integer("y", 0.0, 5.0);
//! b.constrain((2.0 * x + 3.0 * y).le(12.0));
//! b.maximize(x + 4.0 * y);
//! let model = b.build().unwrap();
//!
//! assert_eq!(model.num_vars, 2);
//! assert_eq!(model.num_constraints(), 1);
//! assert_eq!(model.objective.coeffs, vec![1.0, 4.0]);
//! ```

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::{Add, Mul, Neg, Sub};

use crate::bounds::VarBound;
use crate::error::OptError;
use crate::model::{Constraint, Model, Objective};

/// Handle to a variable created by a [`ModelBuilder`].
///
/// A `Var` is only meaningful for the builder that created it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Var {
    index: usize,
}

impl Var {
    /// Position of this variable in the built model.
    pub fn index(self) -> usize {
        self.index
    }

    /// `self ≤ rhs` as a relation.
    pub fn le(self, rhs: f64) -> LinRelation {
        LinExpr::from(self).le(rhs)
    }

    /// `self ≥ rhs` as a relation.
    pub fn ge(self, rhs: f64) -> LinRelation {
        LinExpr::from(self).ge(rhs)
    }

    /// `self = rhs` as a relation.
    pub fn eq(self, rhs: f64) -> LinRelation {
        LinExpr::from(self).eq(rhs)
    }
}

/// A linear expression `constant + Σ coeff·var`.
///
/// Repeated variables are merged, and terms are kept in variable order so the
/// lowered model is deterministic.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LinExpr {
    terms: BTreeMap<usize, f64>,
    constant: f64,
}

impl LinExpr {
    /// The zero expression.
    pub fn zero() -> Self {
        Self::default()
    }

    /// A constant expression.
    pub fn constant(value: f64) -> Self {
        Self { terms: BTreeMap::new(), constant: value }
    }

    /// Constant term of the expression.
    pub fn constant_term(&self) -> f64 {
        self.constant
    }

    /// `self ≤ rhs`.
    pub fn le(self, rhs: f64) -> LinRelation {
        LinRelation::new(self, f64::NEG_INFINITY, rhs)
    }

    /// `self ≥ rhs`.
    pub fn ge(self, rhs: f64) -> LinRelation {
        LinRelation::new(self, rhs, f64::INFINITY)
    }

    /// `self = rhs`.
    pub fn eq(self, rhs: f64) -> LinRelation {
        LinRelation::new(self, rhs, rhs)
    }

    fn scale(mut self, k: f64) -> Self {
        for c in self.terms.values_mut() {
            *c *= k;
        }
        self.constant *= k;
        self
    }

    fn add_scaled(&mut self, other: &LinExpr, k: f64) {
        for (&j, &c) in &other.terms {
            *self.terms.entry(j).or_insert(0.0) += k * c;
        }
        self.constant += k * other.constant;
    }
}

/// A relation `lower ≤ expr ≤ upper` with the expression's constant already
/// moved onto the bounds. Build one with [`LinExpr::le`], [`LinExpr::ge`],
/// [`LinExpr::eq`], or the same methods on [`Var`].
#[derive(Debug, Clone, PartialEq)]
pub struct LinRelation {
    expr: LinExpr,
    lower: f64,
    upper: f64,
}

impl LinRelation {
    fn new(expr: LinExpr, lower: f64, upper: f64) -> Self {
        let shift = expr.constant;
        let expr = LinExpr { constant: 0.0, ..expr };
        Self { expr, lower: lower - shift, upper: upper - shift }
    }
}

/// Builder for a [`Model`] with named variables.
#[derive(Debug, Clone, Default)]
pub struct ModelBuilder {
    names: Vec<String>,
    bounds: Vec<VarBound>,
    constraints: Vec<LinRelation>,
    objective: Option<(bool, LinExpr)>,
}

impl ModelBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a variable with an explicit [`VarBound`]. Its name is used by
    /// [`Self::name`] and is not otherwise interpreted.
    pub fn variable(&mut self, name: impl Into<String>, bound: VarBound) -> Var {
        let index = self.names.len();
        self.names.push(name.into());
        self.bounds.push(bound);
        Var { index }
    }

    /// Add a continuous variable in `[lower, upper]`.
    pub fn continuous(&mut self, name: impl Into<String>, lower: f64, upper: f64) -> Var {
        self.variable(name, VarBound::continuous(lower, upper))
    }

    /// Add an integer variable in `[lower, upper]`.
    pub fn integer(&mut self, name: impl Into<String>, lower: f64, upper: f64) -> Var {
        self.variable(name, VarBound::integer(lower, upper))
    }

    /// Add a binary variable.
    pub fn binary(&mut self, name: impl Into<String>) -> Var {
        self.variable(name, VarBound::binary())
    }

    /// Name the variable was created with.
    pub fn name(&self, var: Var) -> &str {
        &self.names[var.index]
    }

    /// Add a constraint.
    pub fn constrain(&mut self, relation: LinRelation) {
        self.constraints.push(relation);
    }

    /// Set a minimisation objective, replacing any previous one.
    pub fn minimize(&mut self, expr: impl Into<LinExpr>) {
        self.objective = Some((false, expr.into()));
    }

    /// Set a maximisation objective, replacing any previous one.
    pub fn maximize(&mut self, expr: impl Into<LinExpr>) {
        self.objective = Some((true, expr.into()));
    }

    /// Lower the accumulated variables, constraints, and objective to a
    /// [`Model`], validating the result.
    ///
    /// An unset objective is treated as the zero minimisation objective.
    pub fn build(self) -> Result<Model, OptError> {
        let mut model = Model::new(self.names.len());
        for (j, bound) in self.bounds.iter().enumerate() {
            model.variables[j] = crate::model::Variable::new(j, *bound);
        }
        for relation in &self.constraints {
            let (indices, coeffs) = split_terms(&relation.expr);
            model.add_constraint(Constraint::new(indices, coeffs, relation.lower, relation.upper)?);
        }
        let (maximise, expr) = self.objective.unwrap_or((false, LinExpr::zero()));
        let (indices, coeffs) = split_terms(&expr);
        let mut objective = if maximise {
            Objective::maximize(indices, coeffs)
        } else {
            Objective::minimize(indices, coeffs)
        };
        objective.constant = expr.constant;
        model.set_objective(objective);
        model.validate()?;
        Ok(model)
    }
}

fn split_terms(expr: &LinExpr) -> (Vec<usize>, Vec<f64>) {
    expr.terms.iter().filter(|&(_, &c)| c != 0.0).map(|(&j, &c)| (j, c)).unzip()
}

impl From<Var> for LinExpr {
    fn from(v: Var) -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(v.index, 1.0);
        Self { terms, constant: 0.0 }
    }
}

impl From<f64> for LinExpr {
    fn from(value: f64) -> Self {
        Self::constant(value)
    }
}

impl Neg for Var {
    type Output = LinExpr;
    fn neg(self) -> LinExpr {
        LinExpr::from(self).scale(-1.0)
    }
}

impl Neg for LinExpr {
    type Output = LinExpr;
    fn neg(self) -> LinExpr {
        self.scale(-1.0)
    }
}

impl Add for LinExpr {
    type Output = LinExpr;
    fn add(mut self, rhs: LinExpr) -> LinExpr {
        self.add_scaled(&rhs, 1.0);
        self
    }
}

impl Sub for LinExpr {
    type Output = LinExpr;
    fn sub(mut self, rhs: LinExpr) -> LinExpr {
        self.add_scaled(&rhs, -1.0);
        self
    }
}

impl Mul<f64> for LinExpr {
    type Output = LinExpr;
    fn mul(self, k: f64) -> LinExpr {
        self.scale(k)
    }
}

impl Mul<LinExpr> for f64 {
    type Output = LinExpr;
    fn mul(self, expr: LinExpr) -> LinExpr {
        expr.scale(self)
    }
}

impl Add<f64> for LinExpr {
    type Output = LinExpr;
    fn add(mut self, rhs: f64) -> LinExpr {
        self.constant += rhs;
        self
    }
}

impl Sub<f64> for LinExpr {
    type Output = LinExpr;
    fn sub(mut self, rhs: f64) -> LinExpr {
        self.constant -= rhs;
        self
    }
}

// Each `Var` operator delegates to the `LinExpr` form so the arithmetic lives in one place.
macro_rules! var_binary {
    ($($trait:ident::$method:ident),*) => {$(
        impl $trait<Var> for Var {
            type Output = LinExpr;
            fn $method(self, rhs: Var) -> LinExpr {
                $trait::$method(LinExpr::from(self), LinExpr::from(rhs))
            }
        }
        impl $trait<LinExpr> for Var {
            type Output = LinExpr;
            fn $method(self, rhs: LinExpr) -> LinExpr {
                $trait::$method(LinExpr::from(self), rhs)
            }
        }
        impl $trait<Var> for LinExpr {
            type Output = LinExpr;
            fn $method(self, rhs: Var) -> LinExpr {
                $trait::$method(self, LinExpr::from(rhs))
            }
        }
        impl $trait<f64> for Var {
            type Output = LinExpr;
            fn $method(self, rhs: f64) -> LinExpr {
                $trait::$method(LinExpr::from(self), rhs)
            }
        }
    )*};
}

var_binary!(Add::add, Sub::sub);

impl Add<Var> for f64 {
    type Output = LinExpr;
    fn add(self, v: Var) -> LinExpr {
        LinExpr::from(v) + self
    }
}

impl Sub<Var> for f64 {
    type Output = LinExpr;
    fn sub(self, v: Var) -> LinExpr {
        LinExpr::constant(self) - LinExpr::from(v)
    }
}

impl Add<LinExpr> for f64 {
    type Output = LinExpr;
    fn add(self, expr: LinExpr) -> LinExpr {
        expr + self
    }
}

impl Sub<LinExpr> for f64 {
    type Output = LinExpr;
    fn sub(self, expr: LinExpr) -> LinExpr {
        LinExpr::constant(self) - expr
    }
}

impl Mul<f64> for Var {
    type Output = LinExpr;
    fn mul(self, k: f64) -> LinExpr {
        LinExpr::from(self).scale(k)
    }
}

impl Mul<Var> for f64 {
    type Output = LinExpr;
    fn mul(self, v: Var) -> LinExpr {
        LinExpr::from(v).scale(self)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn repeated_variables_merge_and_zero_terms_drop() {
        let mut b = ModelBuilder::new();
        let x = b.continuous("x", 0.0, 1.0);
        let y = b.continuous("y", 0.0, 1.0);
        b.minimize(x + x - 2.0 * x + 3.0 * y);
        let m = b.build().unwrap();
        // x cancels out entirely; only y remains.
        assert_eq!(m.objective.indices, vec![1]);
        assert_eq!(m.objective.coeffs, vec![3.0]);
    }

    #[test]
    fn constant_terms_move_onto_relation_bounds() {
        let mut b = ModelBuilder::new();
        let x = b.continuous("x", 0.0, 10.0);
        b.constrain((x + 2.0).le(7.0));
        b.constrain((x - 1.0).ge(0.5));
        let m = b.build().unwrap();
        assert_eq!(m.constraints[0].upper, 5.0);
        assert_eq!(m.constraints[0].lower, f64::NEG_INFINITY);
        assert_eq!(m.constraints[1].lower, 1.5);
    }

    #[test]
    fn equality_and_objective_constant_lower() {
        let mut b = ModelBuilder::new();
        let x = b.binary("x");
        let y = b.binary("y");
        b.constrain((x + y).eq(1.0));
        b.maximize(3.0 * x + 2.0 * y + 4.0);
        let m = b.build().unwrap();
        assert_eq!(m.constraints[0].lower, 1.0);
        assert_eq!(m.constraints[0].upper, 1.0);
        assert_eq!(m.objective.constant, 4.0);
        assert_eq!(m.objective.sense, crate::model::Sense::Maximize);
    }

    #[test]
    fn names_are_kept_and_indices_follow_creation_order() {
        let mut b = ModelBuilder::new();
        let a = b.continuous("alpha", 0.0, 1.0);
        let z = b.integer("zeta", 0.0, 3.0);
        assert_eq!(b.name(a), "alpha");
        assert_eq!(b.name(z), "zeta");
        assert_eq!(z.index(), 1);
    }

    #[test]
    fn crossed_relation_bounds_are_rejected_at_build() {
        let mut b = ModelBuilder::new();
        let x = b.continuous("x", 0.0, 1.0);
        b.constrain(LinRelation { expr: LinExpr::from(x), lower: 3.0, upper: 2.0 });
        assert!(b.build().is_err());
    }

    #[test]
    fn scalar_on_the_left_of_variable_operators() {
        let mut b = ModelBuilder::new();
        let x = b.continuous("x", 0.0, 1.0);
        b.minimize(2.0 - x);
        let m = b.build().unwrap();
        assert_eq!(m.objective.constant, 2.0);
        assert_eq!(m.objective.coeffs, vec![-1.0]);
    }
}
