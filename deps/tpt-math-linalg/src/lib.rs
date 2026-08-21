#![no_std]
//! Local development shim mirroring the `tpt-math-linalg` API surface used by
//! the optimisation crates: sparse CSR/CSC matrices plus a few dense helpers.
//!
//! This is intentionally minimal — it exists so the `tpt-opt-*` crates compile
//! and test locally before the real `tpt-math` workspace is published. Once
//! `tpt-math` publishes, swap this path dependency for a version dependency.

extern crate alloc;

mod csc;
mod csr;

pub use csc::CscMatrix;
pub use csr::CsrMatrix;

/// A (row, col, value) triplet used to build sparse matrices incrementally.
#[derive(Debug, Clone, PartialEq)]
pub struct Triplet<T> {
    pub row: usize,
    pub col: usize,
    pub value: T,
}

impl<T> Triplet<T> {
    pub fn new(row: usize, col: usize, value: T) -> Self {
        Self { row, col, value }
    }
}

/// Dense vector dot product.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        s += x * y;
    }
    s
}

/// In-place scaling of a dense vector by `alpha`.
pub fn scale_inplace(v: &mut [f64], alpha: f64) {
    for x in v.iter_mut() {
        *x *= alpha;
    }
}

/// Returns `true` if `a` and `b` are element-wise equal within `tol`.
pub fn all_close(a: &[f64], b: &[f64], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
}
