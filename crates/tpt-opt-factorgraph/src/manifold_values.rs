//! Keyed variable storage for a factor graph.

use crate::factor::Key;
use crate::manifold::Manifold;

/// Errors from keyed-value operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyError {
    /// The key is not present.
    Missing(Key),
    /// The key is already present.
    Duplicate(Key),
    /// The data slice length does not match the manifold's storage dim.
    BadDimension { expected: usize, got: usize },
}

/// An insertion-ordered map from [`Key`] to `(Manifold, storage)`.
///
/// Deterministic ordering matters: the solver assigns tangent-space
/// offsets in insertion order, so results are reproducible run to run.
#[derive(Debug, Clone, Default)]
pub struct KeyedValues {
    keys: Vec<Key>,
    kinds: Vec<Manifold>,
    data: Vec<Vec<f64>>,
    index: std::collections::HashMap<Key, usize>,
}

impl KeyedValues {
    /// An empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a variable with `kind` and `data` (`kind.dim()` long).
    pub fn insert(&mut self, key: Key, kind: Manifold, data: Vec<f64>) -> Result<(), KeyError> {
        if data.len() != kind.dim() {
            return Err(KeyError::BadDimension { expected: kind.dim(), got: data.len() });
        }
        if self.index.contains_key(&key) {
            return Err(KeyError::Duplicate(key));
        }
        self.index.insert(key, self.keys.len());
        self.keys.push(key);
        self.kinds.push(kind);
        self.data.push(kind.normalized(&data));
        Ok(())
    }

    /// `true` if the key is present.
    pub fn contains(&self, key: Key) -> bool {
        self.index.contains_key(&key)
    }

    /// Keys in insertion order.
    pub fn keys(&self) -> &[Key] {
        &self.keys
    }

    /// Variable count.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Manifold kind of a key.
    pub fn kind(&self, key: Key) -> Result<Manifold, KeyError> {
        self.index.get(&key).map(|&i| self.kinds[i]).ok_or(KeyError::Missing(key))
    }

    /// Storage slice of a key.
    pub fn get(&self, key: Key) -> Result<&[f64], KeyError> {
        self.index.get(&key).map(|&i| self.data[i].as_slice()).ok_or(KeyError::Missing(key))
    }

    /// Mutable storage slice of a key.
    pub fn get_mut(&mut self, key: Key) -> Result<&mut Vec<f64>, KeyError> {
        self.index.get(&key).map(|&i| &mut self.data[i]).ok_or(KeyError::Missing(key))
    }

    /// Overwrite a key's storage (length-checked, normalised).
    pub fn set(&mut self, key: Key, data: Vec<f64>) -> Result<(), KeyError> {
        let i = *self.index.get(&key).ok_or(KeyError::Missing(key))?;
        if data.len() != self.kinds[i].dim() {
            return Err(KeyError::BadDimension { expected: self.kinds[i].dim(), got: data.len() });
        }
        self.data[i] = self.kinds[i].normalized(&data);
        Ok(())
    }
}
