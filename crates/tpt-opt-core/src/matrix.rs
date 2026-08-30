//! Sparse constraint-matrix conversions compatible with `tpt-math-linalg-sparse`.
//!
//! The constraint matrix is assembled from a [`crate::model::Model`]'s linear
//! rows into the `CsrMatrix` / `CscMatrix` types exposed by
//! `tpt-math-linalg-sparse`, so downstream solver crates (and external bindings
//! such as HiGHS) can ingest the canonical form without re-implementing sparse
//! assembly.

use tpt_math_linalg_sparse::{CooMatrix, CscMatrix, CsrMatrix};

use crate::model::Model;

/// Build the CSR constraint matrix for a model's linear constraints.
///
/// Row `i` corresponds to constraint `i`; column `j` corresponds to variable
/// `j`. Custom-tagged rows are skipped (they are not representable as a fixed
/// sparse row and are handled by the solver's custom-constraint path).
pub fn model_to_csr(model: &Model) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(model.constraints.len(), model.num_vars);
    for (r, c) in model.constraints.iter().enumerate() {
        if c.is_custom {
            continue;
        }
        for (&i, &v) in c.indices.iter().zip(c.coeffs.iter()) {
            coo.push(r, i, v);
        }
    }
    coo.to_csr()
}

/// Build the CSC constraint matrix for a model's linear constraints.
///
/// Convenience wrapper: assembles CSR then converts to CSC via the
/// `tpt-math-linalg-sparse` conversion routines.
pub fn model_to_csc(model: &Model) -> CscMatrix<f64> {
    let csr = model_to_csr(model);
    let mut coo = CooMatrix::new(csr.nrows(), csr.ncols());
    for (r, c, v) in csr.iter() {
        coo.push(r, c, *v);
    }
    coo.to_csc()
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
        let mut coo = CooMatrix::new(model.constraints.len(), model.num_vars);
        for (r, c) in model.constraints.iter().enumerate() {
            if c.is_custom {
                continue;
            }
            for (&i, &v) in c.indices.iter().zip(c.coeffs.iter()) {
                coo.push(r, i, v);
            }
        }
        let csr = coo.to_csr();
        let csc = coo.to_csc();
        Self { csr, csc }
    }
}
