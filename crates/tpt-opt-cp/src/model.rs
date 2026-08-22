//! The constraint-programming model: variables, domains and constraints.

use std::vec::Vec;

use crate::constraints::{Constraint, Inconsistency};
use crate::domain::Domain;

/// Comparison relation used by [`Linear`](crate::constraints::Linear) constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    /// `≤`
    Le,
    /// `≥`
    Ge,
    /// `=`
    Eq,
}

/// A variable handle (its index in the model).
pub type Variable = usize;

/// A constraint-programming model.
pub struct CpModel {
    /// Per-variable domains, indexed by [`Variable`].
    pub domains: Vec<Domain>,
    constraints: Vec<Box<dyn Constraint>>,
}

impl CpModel {
    /// Create an empty model.
    pub fn new() -> Self {
        Self { domains: Vec::new(), constraints: Vec::new() }
    }

    /// Add an integer variable over `[lo, hi]`; returns its index.
    pub fn add_var(&mut self, lo: usize, hi: usize) -> Variable {
        let v = self.domains.len();
        self.domains.push(Domain::new(lo, hi));
        v
    }

    /// Add an integer variable with an explicit value set; returns its index.
    pub fn add_var_values(&mut self, vals: Vec<usize>) -> Variable {
        let v = self.domains.len();
        self.domains.push(Domain::from_values(vals));
        v
    }

    /// Add a constraint to the model.
    pub fn add_constraint(&mut self, c: Box<dyn Constraint>) {
        self.constraints.push(c);
    }

    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.domains.len()
    }

    /// The constraints.
    pub fn constraints(&self) -> &[Box<dyn Constraint>] {
        &self.constraints
    }
}

impl Default for CpModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Run propagation to a fixpoint; returns `Err` if a domain empties.
pub(crate) fn fixpoint(
    doms: &mut [Domain],
    cons: &[Box<dyn Constraint>],
) -> Result<(), Inconsistency> {
    loop {
        let before: Vec<usize> = doms.iter().map(|d| d.len()).collect();
        for c in cons {
            c.propagate(doms)?;
        }
        let after: Vec<usize> = doms.iter().map(|d| d.len()).collect();
        if before == after {
            break;
        }
        if doms.iter().any(|d| d.is_empty()) {
            return Err(Inconsistency);
        }
    }
    Ok(())
}
