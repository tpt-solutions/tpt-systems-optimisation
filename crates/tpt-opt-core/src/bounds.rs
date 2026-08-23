//! Variable bound types and variable kinds.
//!
//! A [`VarBound`] pairs a [`VarType`] (continuous / integer / binary /
//! semi-continuous) with a [`Bound`] interval. Bounds are represented with
//! infinities for open ends, matching the convention used by the canonical
//! [`crate::model::Model`] and by external solver bindings.
//!
//! With the optional `serde` feature these types implement
//! `Serialize`/`Deserialize`. Open ends (±[`f64::INFINITY`]) are mapped to
//! `null` on the wire and back, so models with unbounded variables round-trip
//! losslessly through every serde format — including self-describing text
//! formats like JSON that cannot represent non-finite floats natively.

use tpt_math_numeric::is_integer;

/// A variable kind, orthogonal to its numeric bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VarType {
    /// Free / bounded continuous variable.
    Continuous,
    /// Integer-valued variable.
    Integer,
    /// Binary variable restricted to `{0, 1}`.
    Binary,
    /// Either zero or within `[lower, upper]`.
    SemiContinuous,
}

/// An interval `[lower, upper]` over the reals, with infinities for open ends.
///
/// Serde representation: each end is `Option<f64>` — `Some(v)` for a finite
/// bound, `None` for the corresponding infinity — so formats without
/// non-finite float support (JSON) stay lossless.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bound {
    /// Lower bound; [`f64::NEG_INFINITY`] when unbounded below.
    pub lower: f64,
    /// Upper bound; [`f64::INFINITY`] when unbounded above.
    pub upper: f64,
}

impl Bound {
    /// Sentinel for an unbounded-below end.
    pub const UNBOUNDED_LOWER: f64 = f64::NEG_INFINITY;
    /// Sentinel for an unbounded-above end.
    pub const UNBOUNDED_UPPER: f64 = f64::INFINITY;

    /// A fully free bound `(-inf, +inf)`.
    pub fn free() -> Self {
        Self { lower: Self::UNBOUNDED_LOWER, upper: Self::UNBOUNDED_UPPER }
    }

    /// A two-sided bound `[lower, upper]`.
    pub fn boxed(lower: f64, upper: f64) -> Self {
        Self { lower, upper }
    }

    /// A lower-bounded (semi-infinite) bound `[lower, +inf)`.
    pub fn lower(lower: f64) -> Self {
        Self { lower, upper: Self::UNBOUNDED_UPPER }
    }

    /// An upper-bounded (semi-infinite) bound `(-inf, upper]`.
    pub fn upper(upper: f64) -> Self {
        Self { lower: Self::UNBOUNDED_LOWER, upper }
    }

    /// `true` if `x` lies within the interval (within `tol` at the ends).
    pub fn contains(&self, x: f64, tol: f64) -> bool {
        x >= self.lower - tol && x <= self.upper + tol
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::Bound;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Wire form: `None` encodes the corresponding ±infinity end.
    #[derive(Serialize, Deserialize)]
    struct Repr {
        lower: Option<f64>,
        upper: Option<f64>,
    }

    impl Serialize for Bound {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let repr = Repr {
                lower: self.lower.is_finite().then_some(self.lower),
                upper: self.upper.is_finite().then_some(self.upper),
            };
            repr.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Bound {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let repr = Repr::deserialize(deserializer)?;
            Ok(Bound {
                lower: repr.lower.unwrap_or(f64::NEG_INFINITY),
                upper: repr.upper.unwrap_or(f64::INFINITY),
            })
        }
    }
}

/// The combined kind + bounds describing a single decision variable.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VarBound {
    /// The variable kind.
    pub kind: VarType,
    /// The numeric bound interval.
    pub bound: Bound,
}

impl VarBound {
    /// Continuous variable in `[lower, upper]`.
    pub fn continuous(lower: f64, upper: f64) -> Self {
        Self { kind: VarType::Continuous, bound: Bound::boxed(lower, upper) }
    }

    /// Integer variable in `[lower, upper]` (inclusive integer endpoints).
    pub fn integer(lower: f64, upper: f64) -> Self {
        Self { kind: VarType::Integer, bound: Bound::boxed(lower, upper) }
    }

    /// Binary variable: integer in `[0, 1]`.
    pub fn binary() -> Self {
        Self { kind: VarType::Binary, bound: Bound::boxed(0.0, 1.0) }
    }

    /// Semi-continuous variable: either `0` or within `[lower, upper]`.
    pub fn semi_continuous(lower: f64, upper: f64) -> Self {
        Self { kind: VarType::SemiContinuous, bound: Bound::boxed(lower, upper) }
    }

    /// `true` if the variable must take integral values.
    pub fn is_integral(&self) -> bool {
        matches!(self.kind, VarType::Integer | VarType::Binary)
    }

    /// `true` if `x` satisfies both the bound interval and integrality.
    pub fn feasible(&self, x: f64, tol: f64) -> bool {
        match self.kind {
            VarType::Continuous => self.bound.contains(x, tol),
            VarType::Integer | VarType::Binary => self.bound.contains(x, tol) && is_integer(x, tol),
            VarType::SemiContinuous => {
                // Semi-continuous: either zero, or inside the box and integral.
                if x.abs() <= tol {
                    true
                } else {
                    self.bound.contains(x, tol) && is_integer(x, tol)
                }
            }
        }
    }
}
