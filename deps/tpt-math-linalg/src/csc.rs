use super::Triplet;
use alloc::vec;
use alloc::vec::Vec;

/// Compressed Sparse Column matrix.
///
/// Layout is compatible with the `tpt-math-linalg` CSC contract: `col_ptr` has
/// length `ncols + 1`, `row_ind`/`values` have length `col_ptr[ncols]`.
#[derive(Debug, Clone, PartialEq)]
pub struct CscMatrix<T> {
    nrows: usize,
    ncols: usize,
    col_ptr: Vec<usize>,
    row_ind: Vec<usize>,
    values: Vec<T>,
}

impl<T: Copy + Default> CscMatrix<T> {
    pub fn from_triplets(nrows: usize, ncols: usize, triplets: &[Triplet<T>]) -> Self {
        let mut col_ptr = vec![0usize; ncols + 1];
        for t in triplets {
            col_ptr[t.col + 1] += 1;
        }
        for j in 1..=ncols {
            col_ptr[j] += col_ptr[j - 1];
        }
        let mut row_ind = vec![0usize; triplets.len()];
        let mut values = vec![T::default(); triplets.len()];
        let mut cursor = col_ptr.clone();
        for t in triplets {
            let slot = cursor[t.col];
            row_ind[slot] = t.row;
            values[slot] = t.value;
            cursor[t.col] += 1;
        }
        Self { nrows, ncols, col_ptr, row_ind, values }
    }
}

impl CscMatrix<f64> {
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }
    pub fn row_ind(&self) -> &[usize] {
        &self.row_ind
    }
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}
