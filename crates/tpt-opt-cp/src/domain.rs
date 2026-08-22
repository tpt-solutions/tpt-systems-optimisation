//! Finite integer domains.

/// A finite set of non-negative integer values (kept sorted, unique).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Domain {
    values: Vec<usize>,
    min: usize,
    max: usize,
}

impl Domain {
    /// Domain over the inclusive integer range `[lo, hi]`.
    pub fn new(lo: usize, hi: usize) -> Self {
        let values: Vec<usize> = (lo..=hi).collect();
        let min = *values.first().unwrap_or(&0);
        let max = *values.last().unwrap_or(&0);
        Self { values, min, max }
    }

    /// Domain from an explicit (unsorted, possibly non-contiguous) value list.
    pub fn from_values(mut vals: Vec<usize>) -> Self {
        vals.sort_unstable();
        vals.dedup();
        let min = *vals.first().unwrap_or(&0);
        let max = *vals.last().unwrap_or(&0);
        Self { values: vals, min, max }
    }

    /// Number of allowed values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// `true` if the domain is empty (no feasible value) — a conflict.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// `true` if exactly one value remains.
    pub fn is_singleton(&self) -> bool {
        self.values.len() == 1
    }

    /// The single remaining value (panics if not a singleton).
    pub fn value(&self) -> usize {
        assert_eq!(self.values.len(), 1, "domain is not a singleton");
        self.values[0]
    }

    /// All allowed values.
    pub fn values(&self) -> &[usize] {
        &self.values
    }

    /// Smallest allowed value.
    pub fn min(&self) -> usize {
        self.min
    }

    /// Largest allowed value.
    pub fn max(&self) -> usize {
        self.max
    }

    /// Whether `v` is still allowed.
    pub fn contains(&self, v: usize) -> bool {
        self.values.binary_search(&v).is_ok()
    }

    /// Remove `v`; returns `true` if it was present.
    pub fn remove(&mut self, v: usize) -> bool {
        if let Ok(pos) = self.values.binary_search(&v) {
            self.values.remove(pos);
            if let Some(f) = self.values.first() {
                self.min = *f;
            }
            if let Some(l) = self.values.last() {
                self.max = *l;
            }
            true
        } else {
            false
        }
    }

    /// Restrict the domain to a single value; returns `true` if changed.
    pub fn assign(&mut self, v: usize) -> bool {
        if !self.contains(v) {
            self.values.clear();
            false
        } else if self.values.len() == 1 {
            true
        } else {
            self.values = vec![v];
            self.min = v;
            self.max = v;
            true
        }
    }

    /// Keep only values satisfying `pred`; returns `true` if any were removed.
    pub fn retain<F: Fn(usize) -> bool>(&mut self, pred: F) -> bool {
        let before = self.values.len();
        self.values.retain(|&v| pred(v));
        if let Some(f) = self.values.first() {
            self.min = *f;
        }
        if let Some(l) = self.values.last() {
            self.max = *l;
        }
        self.values.len() != before
    }
}
