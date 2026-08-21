//! Backtracking search with propagation.

use std::vec::Vec;

use crate::constraints::Constraint;
use crate::domain::Domain;
use crate::model::{fixpoint, CpModel};

/// A (partial or complete) solution: one value per variable.
#[derive(Debug, Clone)]
pub struct CpSolution {
    /// Assignment indexed by variable handle.
    pub assignment: Vec<usize>,
}

/// Find one solution to `model`, or `None` if infeasible.
pub fn solve(model: &CpModel) -> Option<CpSolution> {
    let mut doms = model.domains.clone();
    if fixpoint(&mut doms, model.constraints()).is_err() {
        return None;
    }
    search(&mut doms, model.constraints()).map(|a| CpSolution { assignment: a })
}

/// Enumerate up to `limit` solutions.
pub fn solutions(model: &CpModel, limit: usize) -> Vec<CpSolution> {
    let mut out = Vec::new();
    let mut doms = model.domains.clone();
    if fixpoint(&mut doms, model.constraints()).is_err() {
        return out;
    }
    collect(&mut doms, model.constraints(), limit, &mut out);
    out
}

fn search(doms: &mut [Domain], cons: &[Box<dyn Constraint>]) -> Option<Vec<usize>> {
    if fixpoint(doms, cons).is_err() {
        return None;
    }
    // All variables fixed?
    if doms.iter().all(|d| d.is_singleton()) {
        let assign: Vec<usize> = doms.iter().map(|d| d.value()).collect();
        if cons.iter().all(|c| c.check(&assign)) {
            return Some(assign);
        }
        return None;
    }
    // First-fail: smallest domain > 1.
    let var = doms
        .iter()
        .enumerate()
        .filter(|(_, d)| d.len() > 1)
        .min_by_key(|(_, d)| d.len())
        .map(|(i, _)| i)
        .unwrap();
    let candidates: Vec<usize> = doms[var].values().to_vec();
    for v in candidates {
        let mut nd = doms.to_vec();
        nd[var].assign(v);
        if let Some(sol) = search(&mut nd, cons) {
            return Some(sol);
        }
    }
    None
}

fn collect(
    doms: &mut [Domain],
    cons: &[Box<dyn Constraint>],
    limit: usize,
    out: &mut Vec<CpSolution>,
) {
    if out.len() >= limit {
        return;
    }
    if fixpoint(doms, cons).is_err() {
        return;
    }
    if doms.iter().all(|d| d.is_singleton()) {
        let assign: Vec<usize> = doms.iter().map(|d| d.value()).collect();
        if cons.iter().all(|c| c.check(&assign)) {
            out.push(CpSolution {
                assignment: assign,
            });
        }
        return;
    }
    let var = doms
        .iter()
        .enumerate()
        .filter(|(_, d)| d.len() > 1)
        .min_by_key(|(_, d)| d.len())
        .map(|(i, _)| i)
        .unwrap();
    let candidates: Vec<usize> = doms[var].values().to_vec();
    for v in candidates {
        if out.len() >= limit {
            return;
        }
        let mut nd = doms.to_vec();
        nd[var].assign(v);
        collect(&mut nd, cons, limit, out);
    }
}
