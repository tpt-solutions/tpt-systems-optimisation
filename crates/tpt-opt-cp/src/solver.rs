//! Backtracking search with propagation.
//!
//! The one-solution search uses **conflict-directed backjumping** (CBJ):
//! each dead end reports the set of past decisions actually implicated in
//! the failure, and the search jumps directly to the deepest implicated
//! variable instead of stepping back one level. Failed decision prefixes
//! are additionally recorded as **no-goods** (bounded in count and arity)
//! so equivalent prefixes encountered later are pruned immediately.
//! Enumeration keeps plain depth-first search — every leaf must be visited
//! anyway, so backjumping cannot skip solutions.

use std::cell::RefCell;
use std::vec::Vec;

use crate::constraints::Constraint;
use crate::domain::Domain;
use crate::model::{fixpoint, fixpoint_report, CpModel};

/// A (partial or complete) solution: one value per variable.
#[derive(Debug, Clone)]
pub struct CpSolution {
    /// Assignment indexed by variable handle.
    pub assignment: Vec<usize>,
}

/// Bound on recorded no-goods (memory guard).
const MAX_NOGOODS: usize = 512;
/// Maximum number of decisions in a recorded no-good.
const MAX_NOGOOD_ARITY: usize = 10;

/// Search context shared across the recursion.
struct Ctx<'a> {
    cons: &'a [Box<dyn Constraint>],
    /// Static conflict neighbourhoods: `neighbors[v]` lists the variables
    /// sharing at least one constraint with `v` (sorted, unique).
    neighbors: Vec<Vec<usize>>,
    /// Recorded failed decision sets; each entry is a set of `(var, value)`
    /// pairs that jointly lead to failure. Interior mutability lets deep
    /// recursion record while sharing the context immutably.
    nogoods: RefCell<Vec<Vec<(usize, usize)>>>,
}

impl Ctx<'_> {
    /// Does the current decision path contain a recorded no-good? Returns
    /// the variables of the first matching no-good (a valid conflict set).
    fn hit_nogood(&self, path: &[(usize, usize)]) -> Option<Vec<usize>> {
        for ng in self.nogoods.borrow().iter() {
            if ng.iter().all(|&(v, val)| path.iter().any(|&(pv, pval)| pv == v && pval == val)) {
                return Some(ng.iter().map(|&(v, _)| v).collect());
            }
        }
        None
    }

    /// Record a failed decision set as a no-good (bounded in count/arity).
    fn record_nogood(&self, conflict: &[usize], path: &[(usize, usize)]) {
        let mut store = self.nogoods.borrow_mut();
        if store.len() >= MAX_NOGOODS || conflict.is_empty() || conflict.len() > MAX_NOGOOD_ARITY {
            return;
        }
        let mut ng: Vec<(usize, usize)> = conflict
            .iter()
            .filter_map(|&w| path.iter().find(|&&(pw, _)| pw == w).copied())
            .collect();
        ng.sort_unstable();
        ng.dedup();
        if !ng.is_empty() && !store.contains(&ng) {
            store.push(ng);
        }
    }
}

/// Find one solution to `model`, or `None` if infeasible.
pub fn solve(model: &CpModel) -> Option<CpSolution> {
    let cons = model.constraints();
    let mut doms = model.domains.clone();
    if fixpoint(&mut doms, cons).is_err() {
        return None;
    }
    let ctx = Ctx { cons, neighbors: build_neighbors(cons), nogoods: RefCell::new(Vec::new()) };
    let mut path: Vec<(usize, usize)> = Vec::new();
    match dfs(&mut doms, &mut path, &ctx) {
        Ok(a) => Some(CpSolution { assignment: a }),
        Err(_) => None,
    }
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

/// Static conflict graph from constraint scopes.
fn build_neighbors(cons: &[Box<dyn Constraint>]) -> Vec<Vec<usize>> {
    let n = cons.iter().flat_map(|c| c.vars().iter().copied()).max().unwrap_or(0) + 1;
    let mut adj = vec![Vec::new(); n];
    for c in cons {
        let vs = c.vars();
        for &a in vs {
            if a >= n {
                continue;
            }
            for &b in vs {
                if b != a && !adj[a].contains(&b) {
                    adj[a].push(b);
                }
            }
        }
    }
    for list in &mut adj {
        list.sort_unstable();
    }
    adj
}

/// Conflict-directed backjumping DFS.
///
/// Returns `Ok(assignment)` on success. On exhaustion returns
/// `Err(conflict_set)` — the set of *decision-path* variables whose current
/// values are implicated in the failure. A caller branching on variable `u`
/// treats a returned set not containing `u` as "this subtree's failure is
/// independent of my choice": it stops trying its remaining values and
/// forwards the set upward (the backjump).
fn dfs(
    doms: &mut [Domain],
    path: &mut Vec<(usize, usize)>,
    ctx: &Ctx,
) -> Result<Vec<usize>, Vec<usize>> {
    // No-good pruning: this exact decision prefix already failed elsewhere.
    if let Some(vars) = ctx.hit_nogood(path) {
        return Err(vars);
    }

    // Propagate; attribute any wipeout to the failing constraint's scope.
    if let Err(ci) = fixpoint_report(doms, ctx.cons) {
        let mut set: Vec<usize> = ctx.cons[ci]
            .vars()
            .iter()
            .copied()
            .filter(|&w| path.iter().any(|&(pw, _)| pw == w))
            .collect();
        set.sort_unstable();
        set.dedup();
        if set.is_empty() {
            // Failing scope touches only propagation-forced singletons;
            // conservatively blame the whole path.
            set.extend(path.iter().map(|&(w, _)| w));
        }
        return Err(set);
    }

    // All fixed? Verify completely.
    if doms.iter().all(|d| d.is_singleton()) {
        let assign: Vec<usize> = doms.iter().map(|d| d.value()).collect();
        let failing: Vec<usize> = ctx
            .cons
            .iter()
            .filter(|c| !c.check(&assign))
            .flat_map(|c| c.vars().iter().copied())
            .collect();
        if failing.is_empty() {
            return Ok(assign);
        }
        let mut set: Vec<usize> =
            failing.into_iter().filter(|&w| path.iter().any(|&(pw, _)| pw == w)).collect();
        set.sort_unstable();
        set.dedup();
        if set.is_empty() {
            set.extend(path.iter().map(|&(w, _)| w));
        }
        return Err(set);
    }

    // First-fail: smallest domain > 1.
    let u = doms
        .iter()
        .enumerate()
        .filter(|(_, d)| d.len() > 1)
        .min_by_key(|(_, d)| d.len())
        .map(|(i, _)| i)
        .unwrap();

    let candidates: Vec<usize> = doms[u].values().to_vec();
    let mut merged: Vec<usize> = Vec::new();
    for v in candidates {
        let mut nd = doms.to_vec();
        nd[u].assign(v);
        path.push((u, v));
        let r = dfs(&mut nd, path, ctx);
        path.pop();
        match r {
            Ok(a) => return Ok(a),
            Err(cset) => {
                if !cset.contains(&u) {
                    // Failure independent of u's choice: backjump past u.
                    return Err(cset);
                }
                ctx.record_nogood(&cset, path);
                merged.extend(cset.into_iter().filter(|&w| w != u));
            }
        }
    }

    // Every value of u failed. Union the accumulated conflicts with u's
    // static neighbours on the path (guards against under-attribution).
    let mut out = merged;
    for &w in &ctx.neighbors[u] {
        if path.iter().any(|&(pw, _)| pw == w) && !out.contains(&w) {
            out.push(w);
        }
    }
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        out.extend(path.iter().map(|&(w, _)| w));
    }
    Err(out)
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
            out.push(CpSolution { assignment: assign });
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
