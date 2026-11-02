//! `f64` math routed through one place so the crate builds without `std`.
//!
//! `core` has no transcendental or rounding methods on `f64`, so under
//! `no_std` these forward to `libm`. With the `std` feature they delegate to
//! the inherent methods, so std builds are unchanged.

// Without `alloc` only `round` (used by `bounds`) is reachable, so the rest
// of each backend is dead code in that configuration.
#[cfg(feature = "std")]
#[allow(dead_code)]
mod backend {
    #[inline]
    pub(crate) fn sqrt(x: f64) -> f64 {
        x.sqrt()
    }
    #[inline]
    pub(crate) fn ln(x: f64) -> f64 {
        x.ln()
    }
    #[inline]
    pub(crate) fn exp(x: f64) -> f64 {
        x.exp()
    }
    #[inline]
    pub(crate) fn sin(x: f64) -> f64 {
        x.sin()
    }
    #[inline]
    pub(crate) fn cos(x: f64) -> f64 {
        x.cos()
    }
    #[inline]
    pub(crate) fn floor(x: f64) -> f64 {
        x.floor()
    }
    #[inline]
    pub(crate) fn round(x: f64) -> f64 {
        x.round()
    }
    #[inline]
    pub(crate) fn powf(x: f64, y: f64) -> f64 {
        x.powf(y)
    }
    #[inline]
    pub(crate) fn powi(x: f64, n: i32) -> f64 {
        x.powi(n)
    }
}

#[cfg(not(feature = "std"))]
#[allow(dead_code)]
mod backend {
    #[inline]
    pub(crate) fn sqrt(x: f64) -> f64 {
        libm::sqrt(x)
    }
    #[inline]
    pub(crate) fn ln(x: f64) -> f64 {
        libm::log(x)
    }
    #[inline]
    pub(crate) fn exp(x: f64) -> f64 {
        libm::exp(x)
    }
    #[inline]
    pub(crate) fn sin(x: f64) -> f64 {
        libm::sin(x)
    }
    #[inline]
    pub(crate) fn cos(x: f64) -> f64 {
        libm::cos(x)
    }
    #[inline]
    pub(crate) fn floor(x: f64) -> f64 {
        libm::floor(x)
    }
    #[inline]
    pub(crate) fn round(x: f64) -> f64 {
        libm::round(x)
    }
    #[inline]
    pub(crate) fn powf(x: f64, y: f64) -> f64 {
        libm::pow(x, y)
    }
    /// `libm` has no integer-power routine; `powi` is `pow` with an exact
    /// integer exponent, which is exact for the small exponents used here.
    #[inline]
    pub(crate) fn powi(x: f64, n: i32) -> f64 {
        libm::pow(x, n as f64)
    }
}

#[allow(unused_imports)]
pub(crate) use backend::{cos, exp, floor, ln, powf, powi, round, sin, sqrt};
