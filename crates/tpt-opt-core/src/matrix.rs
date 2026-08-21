//! Sparse constraint-matrix conversions compatible with `tpt-math-linalg`.
//!
//! The constraint matrix is assembled from a [`crate::model::Model`]'s linear
//! rows into the `CsrMatrix` / `CscMatrix` types exposed by `tpt-math-linalg`,
//! so downstream solver crates (and external bindings such as HiGHS) can ingest
//! the canonical form without re-implementing sparse assembly.

use alloc::vec;
use alloc::vec::Vec;

use tpt_math_linalg::{CscMatrix, CsrMatrix, Triplet};

use crate::model::Model;

/// Build the CSR constraint matrix for a model's linear constraints.
///
/// Row `i` corresponds to constraint `i`; column `j` corresponds to variable
/// `j`. Custom-tagged rows are skipped (they are not representable as a fixed
/// sparse row and are handled by the solver's custom-constraint path).
pub fn model_to_csr(model: &Model) -> CsrMatrix<f64> {
    let mut triplets = Vec::with_capacity(model.constraints.iter().map(|c| c.indices.len()).sum());
    for (r, c) in model.constraints.iter().enumerate() {
        if c.is_custom {
            continue;
        }
        for (&i, &v) in c.indices.iter().zip(c.coeffs.iter()) {
            triplets.push(Triplet::new(r, i, v));
        }
    }
    CsrMatrix::from_triplets(model.constraints.len(), model.num_vars, &triplets)
}

/// Build the CSC constraint matrix for a model's linear constraints.
///
/// Convenience wrapper: assembles CSR then transposes to CSC, reusing the
/// `tpt-math-linalg` transpose routine.
pub fn model_to_csc(model: &Model) -> CscMatrix<f64> {
    let csr = model_to_csr(model);
    let mut triplets = Vec::with_capacity(csr.values().len());
    for r in 0..csr.nrows() {
        let start = csr.row_ptr()[r];
        let end = csr.row_ptr()[r + 1];
        for k in start..end {
            triplets.push(Triplet::new(csr.col_ind()[k], r, csr.values()[k]));
        }
    }
    CscMatrix::from_triplets(csr.nrows(), csr.ncols(), &triplets)
}

/// Holds both the CSR and CSC views of a model's constraint matrix.
#[derive(Debug, Clone)]
pub struct ConstraintMatrix {
    /// Row-major view (row = constraint).
    pub csr: CsrMatrix<f64>,
    /// Column-major view (column = variable).
    pub csc: CscMatrix<f64>,
}

impl ConstraintMatrix {
    /// Assemble both views from a model.
    pub fn from_model(model: &Model) -> Self {
        let csr = model_to_csr(model);
        let mut triplets = vec![];
        for r in 0..csr.nrows() {
            let start = csr.row_ptr()[r];
            let end = csr.row_ptr()[r + 1];
            for k in start..end {
                triplets.push(Triplet::new(csr.col_ind()[k], r, csr.values()[k]));
            }
        }
        let csc = CscMatrix::from_triplets(csr.nrows(), csr.ncols(), &triplets);
        Self { csr, csc }
    }
}
