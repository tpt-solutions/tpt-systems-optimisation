//! Marginal covariance extraction from the factor graph.
//!
//! The Hessian `H = Σ JᵀJ` of the whitened system *is* the Gaussian
//! information matrix of the linearised posterior, so the marginal
//! covariance of a key is the matching diagonal block of `H⁻¹`. We compute
//! it by factorising `H` once (fill-reducing-ordered sparse `LDLᵀ`) and
//! solving `H X = E` for the key's tangent unit columns.

use tpt_opt_nls::sparse::{self, Ordering};

use crate::factor::Key;
use crate::graph::{FactorGraph, GraphError};

/// Errors from marginal-covariance extraction.
#[derive(Debug, Clone, PartialEq)]
pub enum MarginalError {
    /// Graph traversal failed.
    Graph(GraphError),
    /// The Hessian is singular (an under-constrained variable): increase
    /// information or add priors.
    Singular,
}

impl From<GraphError> for MarginalError {
    fn from(e: GraphError) -> Self {
        MarginalError::Graph(e)
    }
}

/// Marginal covariance `H⁻¹` block (row-major `dof × dof`) for `key` at
/// the current values. A small Tikhonov floor (`1e-9 · diag`) keeps the
/// factorisation well-posed for exactly-determined systems.
pub fn marginal_covariance(
    graph: &FactorGraph,
    key: Key,
    fd_step: f64,
) -> Result<Vec<f64>, MarginalError> {
    let lins = crate::solver::linearize_all(graph, fd_step)?;
    let sys = crate::solver::assemble_pub(graph, &lins);
    let mut triplets = sys.to_triplets();
    // Tikhonov floor on the diagonal.
    for s in 0..sys.slots.len() {
        let (o, dof) = sys.layout(s);
        for r in 0..dof {
            triplets.push(tpt_opt_nls::sparse::Triplet::new(o + r, o + r, 1e-9));
        }
    }
    let sym = sparse::analyze(sys.n, &triplets, Ordering::MinDegree);
    let ldl = sym.factorize(&triplets).map_err(|_| MarginalError::Singular)?;
    let (o, dof) = sys
        .slots
        .iter()
        .find(|&&(k, _, _)| k == key)
        .map(|&(_, o, d)| (o, d))
        .ok_or(MarginalError::Graph(GraphError::MissingKey(key)))?;
    let mut block = vec![0.0; dof * dof];
    for c in 0..dof {
        let mut e = vec![0.0; sys.n];
        e[o + c] = 1.0;
        let x = ldl.solve(&e);
        for r in 0..dof {
            block[r * dof + c] = x[o + r];
        }
    }
    Ok(block)
}
