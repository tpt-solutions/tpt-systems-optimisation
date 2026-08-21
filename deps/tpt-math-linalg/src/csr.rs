use super::Triplet;
use alloc::vec;
use alloc::vec::Vec;

/// Compressed Sparse Row matrix.
///
/// Layout is compatible with the `tpt-math-linalg` CSR contract: `row_ptr` has
/// length `nrows + 1`, `col_ind`/`values` have length `row_ptr[nrows]`, and the
/// entries of row `i` occupy `row_ptr[i]..row_ptr[i+1]` in sorted column order.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix<T> {
    nrows: usize,
    ncols: usize,
    row_ptr: Vec<usize>,
    col_ind: Vec<usize>,
    values: Vec<T>,
}

impl<T: Copy + Default> CsrMatrix<T> {
    /// Build a CSR matrix from unsorted/sorted triplets.
    pub fn from_triplets(nrows: usize, ncols: usize, triplets: &[Triplet<T>]) -> Self {
        let mut row_ptr = vec![0usize; nrows + 1];
        for t in triplets {
            row_ptr[t.row + 1] += 1;
        }
        for i in 1..=nrows {
            row_ptr[i] += row_ptr[i - 1];
        }
        let mut col_ind = vec![0usize; triplets.len()];
        let mut values = vec![T::default(); triplets.len()];
        let mut cursor = row_ptr.clone();
        for t in triplets {
            let slot = cursor[t.row];
            col_ind[slot] = t.col;
            values[slot] = t.value;
            cursor[t.row] += 1;
        }
        Self {
            nrows,
            ncols,
            row_ptr,
            col_ind,
            values,
        }
    }
}

impl CsrMatrix<f64> {
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }
    pub fn col_ind(&self) -> &[usize] {
        &self.col_ind
    }
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Transpose into a CSC-compatible representation returned as a new CSR.
    pub fn transpose(&self) -> CsrMatrix<f64> {
        let mut triplets = Vec::with_capacity(self.values.len());
        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for k in start..end {
                triplets.push(Triplet::new(self.col_ind[k], i, self.values[k]));
            }
        }
        CsrMatrix::from_triplets(self.ncols, self.nrows, &triplets)
    }
}
