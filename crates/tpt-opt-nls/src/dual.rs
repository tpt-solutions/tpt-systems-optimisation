//! Forward-mode automatic differentiation with dual numbers.
//!
//! A [`Dual`] carries a value plus a gradient vector with respect to `n`
//! inputs. Residual blocks are written once, generic over `Dual`, and get an
//! **exact** Jacobian in a single seeded evaluation — no finite-difference
//! truncation error, at the same O(1) evaluation cost per operation as
//! central differences but without the extra passes.
//!
//! A `Dual` whose gradient is *empty* behaves as a plain scalar: the same
//! closure evaluated in plain mode produces residual values only, so problem
//! evaluation never duplicates user code.
//!
//! # Example
//!
//! ```rust
//! use tpt_opt_nls::dual::Dual;
//!
//! let n = 2;
//! let x = [Dual::seeded(3.0, 0, n), Dual::seeded(4.0, 1, n)];
//! let mut out = [Dual::zero(n); 1];
//! // r(x) = x₀·x₁ + sin(x₀)
//! out[0] = x[0].clone() * x[1].clone() + x[0].clone().sin();
//! // ∂r/∂x₀ = x₁ + cos(x₀) = 4 + cos 3, ∂r/∂x₁ = x₀ = 3.
//! assert!((out[0].grad[0] - (4.0 + 3.0_f64.cos())).abs() < 1e-12);
//! assert!((out[0].grad[1] - 3.0).abs() < 1e-12);
//! ```

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A dual number: `value` plus first derivatives with respect to `n` inputs.
///
/// An **empty** gradient means "plain value mode": arithmetic degenerates to
/// scalar arithmetic and gradients are not propagated. This lets one closure
/// serve both residual and Jacobian evaluation.
#[derive(Clone, PartialEq)]
pub struct Dual {
    /// Function value.
    pub value: f64,
    /// Partial derivatives (length `n`, or empty in plain mode).
    pub grad: Vec<f64>,
}

impl Dual {
    /// A zero dual number carrying an `n`-length gradient.
    pub fn zero(n: usize) -> Self {
        Self { value: 0.0, grad: vec![0.0; n] }
    }

    /// A constant (zero gradient) dual with `n`-length gradient bookkeeping.
    pub fn constant(value: f64, n: usize) -> Self {
        Self { value, grad: vec![0.0; n] }
    }

    /// A plain scalar: value only, no gradient propagation.
    pub fn plain(value: f64) -> Self {
        Self { value, grad: Vec::new() }
    }

    /// The `i`-th input variable of `n`: value `v`, gradient `e_i`.
    pub fn seeded(value: f64, i: usize, n: usize) -> Self {
        let mut grad = vec![0.0; n];
        if i < n {
            grad[i] = 1.0;
        }
        Self { value, grad }
    }

    /// `true` if this dual carries no gradient (plain mode).
    pub fn is_plain(&self) -> bool {
        self.grad.is_empty()
    }

    /// Square root.
    pub fn sqrt(&self) -> Self {
        let v = self.value.sqrt();
        Self { value: v, grad: self.scale_grad(0.5 / v) }
    }

    /// Natural exponential.
    pub fn exp(&self) -> Self {
        let v = self.value.exp();
        Self { value: v, grad: self.scale_grad(v) }
    }

    /// Natural logarithm (undefined for non-positive values).
    pub fn ln(&self) -> Self {
        let inv = 1.0 / self.value;
        Self { value: self.value.ln(), grad: self.scale_grad(inv) }
    }

    /// Sine.
    pub fn sin(&self) -> Self {
        Self { value: self.value.sin(), grad: self.scale_grad(self.value.cos()) }
    }

    /// Cosine.
    pub fn cos(&self) -> Self {
        Self { value: self.value.cos(), grad: self.scale_grad(-self.value.sin()) }
    }

    /// Tangent.
    pub fn tan(&self) -> Self {
        let c = self.value.cos();
        Self { value: self.value.tan(), grad: self.scale_grad(1.0 / (c * c)) }
    }

    /// Hyperbolic sine.
    pub fn sinh(&self) -> Self {
        let v = self.value.sinh();
        Self { value: v, grad: self.scale_grad(self.value.cosh()) }
    }

    /// Hyperbolic cosine.
    pub fn cosh(&self) -> Self {
        let v = self.value.cosh();
        Self { value: v, grad: self.scale_grad(self.value.sinh()) }
    }

    /// Arc tangent of `self / other`, derivative `dθ = (x d y − y d x) / (x² + y²)`.
    pub fn atan2(&self, other: &Dual) -> Dual {
        let denom = self.value * self.value + other.value * other.value;
        let d = self.binary_grad(other, other.value / denom, -self.value / denom);
        Dual { value: self.value.atan2(other.value), grad: d }
    }

    /// Raise to a floating-point power: `d(x^p) = p·x^(p−1) dx + ln(x)·x^p dp`.
    pub fn powf(&self, p: &Dual) -> Dual {
        let v = self.value.powf(p.value);
        let d_self = p.value * self.value.powf(p.value - 1.0);
        let d_p = if self.value > 0.0 { v * self.value.ln() } else { 0.0 };
        Dual { value: v, grad: self.binary_grad(p, d_self, d_p) }
    }

    /// Raise to an integer power (constant exponent).
    pub fn powi(&self, p: i32) -> Dual {
        if p == 0 {
            return Dual::constant(1.0, self.grad.len());
        }
        let v = self.value.powi(p);
        Self { value: v, grad: self.scale_grad(p as f64 * self.value.powi(p - 1)) }
    }

    /// Absolute value; the derivative at 0 is taken as 0 (subgradient).
    pub fn abs(&self) -> Dual {
        let s = if self.value >= 0.0 { 1.0 } else { -1.0 };
        Dual { value: self.value.abs(), grad: self.scale_grad(s) }
    }

    /// Hypotenuse `sqrt(x² + y²)` with a numerically stable derivative.
    pub fn hypot(&self, other: &Dual) -> Dual {
        let v = self.value.hypot(other.value);
        let d = self.binary_grad(other, self.value / v, other.value / v);
        Dual { value: v, grad: d }
    }

    fn scale_grad(&self, s: f64) -> Vec<f64> {
        self.grad.iter().map(|&g| g * s).collect()
    }

    fn binary_grad(&self, other: &Dual, ds: f64, do_: f64) -> Vec<f64> {
        if self.is_plain() {
            return Vec::new();
        }
        let n = self.grad.len();
        let mut g = self.scale_grad(ds);
        if !other.is_plain() {
            for (i, &og) in other.grad.iter().take(n).enumerate() {
                g[i] += do_ * og;
            }
        }
        g
    }

    fn zip_grad(a: &Dual, b: &Dual, fa: f64, fb: f64) -> Vec<f64> {
        if a.is_plain() && b.is_plain() {
            return Vec::new();
        }
        let n = a.grad.len().max(b.grad.len());
        let mut g = vec![0.0; n];
        for (i, gv) in g.iter_mut().enumerate() {
            *gv = fa * a.grad.get(i).copied().unwrap_or(0.0)
                + fb * b.grad.get(i).copied().unwrap_or(0.0);
        }
        g
    }
}

impl fmt::Debug for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_plain() {
            write!(f, "Dual(plain {})", self.value)
        } else {
            write!(f, "Dual({}, grad {:?})", self.value, self.grad)
        }
    }
}

impl From<f64> for Dual {
    fn from(v: f64) -> Self {
        Dual::plain(v)
    }
}

impl Add for Dual {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        Dual { value: self.value + rhs.value, grad: Self::zip_grad(&self, &rhs, 1.0, 1.0) }
    }
}

impl Sub for Dual {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        Dual { value: self.value - rhs.value, grad: Self::zip_grad(&self, &rhs, 1.0, -1.0) }
    }
}

impl Mul for Dual {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        Dual {
            value: self.value * rhs.value,
            grad: Self::zip_grad(&self, &rhs, rhs.value, self.value),
        }
    }
}

impl Div for Dual {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        let inv = 1.0 / rhs.value;
        let q = self.value * inv;
        Dual { value: q, grad: Self::zip_grad(&self, &rhs, inv, -q * inv) }
    }
}

impl Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual { value: -self.value, grad: self.scale_grad(-1.0) }
    }
}

impl AddAssign for Dual {
    fn add_assign(&mut self, rhs: Dual) {
        if self.grad.is_empty() && !rhs.grad.is_empty() {
            // Promote plain accumulator to gradient-carrying.
            *self = Dual { value: self.value + rhs.value, grad: rhs.scale_grad(1.0) };
            return;
        }
        self.value += rhs.value;
        if !self.grad.is_empty() {
            for (gv, &rg) in self.grad.iter_mut().zip(rhs.grad.iter()) {
                *gv += rg;
            }
        }
    }
}

impl SubAssign for Dual {
    fn sub_assign(&mut self, rhs: Dual) {
        self.value -= rhs.value;
        if !self.grad.is_empty() {
            for (gv, &rg) in self.grad.iter_mut().zip(rhs.grad.iter()) {
                *gv -= rg;
            }
        }
    }
}

impl MulAssign<f64> for Dual {
    fn mul_assign(&mut self, s: f64) {
        self.value *= s;
        for gv in self.grad.iter_mut() {
            *gv *= s;
        }
    }
}

impl DivAssign<f64> for Dual {
    fn div_assign(&mut self, s: f64) {
        let inv = 1.0 / s;
        self.value *= inv;
        for gv in self.grad.iter_mut() {
            *gv *= inv;
        }
    }
}

macro_rules! impl_scalar_binop {
    ($trait:ident, $meth:ident, $dual_f:expr, $val_f:ident) => {
        impl std::ops::$trait<f64> for Dual {
            type Output = Dual;
            fn $meth(self, s: f64) -> Dual {
                let other = Dual::constant(s, self.grad.len());
                std::ops::$trait::$meth(self, other)
            }
        }
        impl std::ops::$trait<Dual> for f64 {
            type Output = Dual;
            fn $meth(self, d: Dual) -> Dual {
                std::ops::$trait::$meth(Dual::constant(self, d.grad.len()), d)
            }
        }
    };
}

impl_scalar_binop!(Add, add, add, add);
impl_scalar_binop!(Sub, sub, sub, sub);
impl_scalar_binop!(Mul, mul, mul, mul);
impl_scalar_binop!(Div, div, div, div);

/// Evaluate a residual block in plain mode (values only).
pub fn eval_plain(f: &dyn Fn(&[Dual], &mut [Dual]), x: &[f64], out: &mut [f64]) {
    let xd: Vec<Dual> = x.iter().map(|&v| Dual::plain(v)).collect();
    let mut od: Vec<Dual> = (0..out.len()).map(|_| Dual::plain(0.0)).collect();
    f(&xd, &mut od);
    for (o, d) in out.iter_mut().zip(od.iter()) {
        *o = d.value;
    }
}

/// Evaluate a residual block with seeded inputs, returning the residual values
/// and the row-major Jacobian (`out.len() × x.len()`).
pub fn eval_jacobian(
    f: &dyn Fn(&[Dual], &mut [Dual]),
    x: &[f64],
    out_val: &mut [f64],
    jac: &mut [f64],
) {
    let n = x.len();
    let m = out_val.len();
    let xd: Vec<Dual> = (0..n).map(|i| Dual::seeded(x[i], i, n)).collect();
    let mut od: Vec<Dual> = (0..m).map(|_| Dual::zero(n)).collect();
    f(&xd, &mut od);
    for (i, d) in od.iter().enumerate() {
        out_val[i] = d.value;
        jac[i * n..(i + 1) * n].copy_from_slice(&d.grad);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-5;

    fn central(f: impl Fn(f64) -> f64, x: f64) -> f64 {
        (f(x + EPS) - f(x - EPS)) / (2.0 * EPS)
    }

    #[test]
    fn arithmetic_chain_rule() {
        let n = 1;
        for &x in &[0.5, 1.3, 2.7] {
            let d = Dual::seeded(x, 0, n);
            // r = x·sin(x) + exp(x/2) / (x² + 1)
            let r = d.clone() * d.sin() + (d.clone() * 0.5).exp() / (d.clone() * d.clone() + 1.0);
            let expect = central(|t| t * t.sin() + (t / 2.0).exp() / (t * t + 1.0), x);
            assert!((r.grad[0] - expect).abs() < 1e-7, "x={x} {} vs {expect}", r.grad[0]);
        }
    }

    #[test]
    fn math_functions_chain_rule() {
        let n = 1;
        for &x in &[0.4, 1.1, 2.2] {
            let d = Dual::seeded(x, 0, n);
            let r = d.sqrt().ln() + d.powi(3) - d.powf(&Dual::plain(1.5))
                + d.abs().hypot(&Dual::constant(1.0, n));
            let expect =
                central(|t| (t.sqrt().ln()) + t.powi(3) - t.powf(1.5) + t.abs().hypot(1.0), x);
            assert!((r.grad[0] - expect).abs() < 1e-6);
        }
    }

    #[test]
    fn trig_atan2() {
        let n = 2;
        let x = [Dual::seeded(1.0, 0, n), Dual::seeded(2.0, 1, n)];
        let a = x[0].atan2(&x[1]);
        assert!((a.value - 1.0_f64.atan2(2.0)).abs() < 1e-15);
        // ∂θ/∂x = y/(x²+y²) = 2/5, ∂θ/∂y = −x/(x²+y²) = −1/5.
        assert!((a.grad[0] - 0.4).abs() < 1e-12);
        assert!((a.grad[1] + 0.2).abs() < 1e-12);
    }

    #[test]
    fn plain_mode_no_gradient() {
        let d = Dual::plain(3.0) * Dual::plain(2.0);
        assert!(d.is_plain());
        assert!((d.value - 6.0).abs() < 1e-15);
    }

    #[test]
    fn assign_ops_accumulate() {
        let n = 2;
        let mut acc = Dual::zero(n);
        acc += Dual::seeded(1.0, 0, n);
        acc += Dual::seeded(2.0, 1, n) * Dual::plain(3.0);
        assert!((acc.value - 7.0).abs() < 1e-15);
        assert!((acc.grad[0] - 1.0).abs() < 1e-15);
        assert!((acc.grad[1] - 3.0).abs() < 1e-15);
        acc *= 2.0;
        assert!((acc.grad[0] - 2.0).abs() < 1e-15);
        acc /= 2.0;
        assert!((acc.grad[1] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn eval_jacobian_matches_fd() {
        // r(x) = [x0·x1, x0² − x1, exp(x0)·x1]
        let f = |x: &[Dual], out: &mut [Dual]| {
            out[0] = x[0].clone() * x[1].clone();
            out[1] = x[0].clone() * x[0].clone() - x[1].clone();
            out[2] = x[0].exp() * x[1].clone();
        };
        let x = [1.3, 0.7];
        let mut val = [0.0; 3];
        let mut jac = [0.0; 6];
        eval_jacobian(&f, &x, &mut val, &mut jac);
        assert!((val[0] - 1.3 * 0.7).abs() < 1e-12);
        // Row 0: [0.7, 1.3]; row 1: [2.6, −1]; row 2: [e^1.3·0.7, e^1.3].
        assert!((jac[0] - 0.7).abs() < 1e-12 && (jac[1] - 1.3).abs() < 1e-12);
        assert!((jac[2] - 2.6).abs() < 1e-12 && (jac[3] + 1.0).abs() < 1e-12);
        let e = 1.3_f64.exp();
        assert!((jac[4] - e * 0.7).abs() < 1e-12 && (jac[5] - e).abs() < 1e-12);
    }

    #[test]
    fn tan_sinh_cosh_derivatives() {
        let n = 1;
        let x = 0.6;
        let d = Dual::seeded(x, 0, n);
        let r = d.tan() + d.sinh() * d.cosh();
        let expect = central(|t| t.tan() + t.sinh() * t.cosh(), x);
        assert!((r.grad[0] - expect).abs() < 1e-7);
    }
}
