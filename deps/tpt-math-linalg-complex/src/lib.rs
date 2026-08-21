//! Local dev shim mirroring `tpt-math-linalg-complex`: a minimal complex
//! scalar type used by OPF (polar/rectangular) formulations.

/// A complex number with `f64` real and imaginary parts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
    pub fn norm_sqr(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    pub fn norm(&self) -> f64 {
        self.norm_sqr().sqrt()
    }
    pub fn conj(&self) -> Self {
        Self::new(self.re, -self.im)
    }
    pub fn arg(&self) -> f64 {
        self.im.atan2(self.re)
    }
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }
}

impl core::ops::Add for Complex {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}

impl core::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}

impl core::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}

impl core::ops::Div for Complex {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let d = o.norm_sqr();
        Self::new((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)
    }
}
