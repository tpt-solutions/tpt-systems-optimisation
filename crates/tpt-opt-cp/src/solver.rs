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
//!
//! Variable ordering is configurable: [`VariableSelection::FirstFail`] (the
//! default, smallest remaining domain), [`VariableSelection::Impact`]
//! (highest constraint-degree first — most likely to trigger early
//! propagation) and [`VariableSelection::Activity`] (VSIDS-style: the variable
//! implicated in the most failures so far, so the search fails fast on the
//! contentious decisions).

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

/// Variable-selection strategy for the backtracking search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VariableSelection {
    /// Smallest remaining domain first (the classic first-fail heuristic).
    #[default]
    FirstFail,
    /// Highest constraint-degree first: branch on the most-connected variable
    /// to maximise early propagation.
    Impact,
    /// VSIDS-style activity: branch on the variable most often implicated in
    /// recent failures, so the search fails fast on contentious decisions.
    Activity,
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
    /// Constraint-degree of each variable (number of constraints touching it).
    degree: Vec<usize>,
    /// Variable-selection strategy.
    selection: VariableSelection,
    /// Activity counts (VSIDS), updated on every failure.
    activity: RefCell<Vec<usize>>,
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

    /// Bump activity counts for every variable in a failure's conflict set.
    fn bump(&self, set: &[usize]) {
        let mut act = self.activity.borrow_mut();
        for &w in set {
            act[w] += 1;
        }
    }

    /// Choose the next variable to branch on (domain size > 1).
    fn choose_var(&self, doms: &[Domain]) -> usize {
        let candidates: Vec<usize> = (0..doms.len()).filter(|&i| doms[i].len() > 1).collect();
        match self.selection {
            VariableSelection::FirstFail => {
                candidates.into_iter().min_by_key(|&i| doms[i].len()).unwrap()
            }
            VariableSelection::Impact => {
                candidates.into_iter().max_by_key(|&i| self.degree[i]).unwrap()
            }
            VariableSelection::Activity => {
                let act = self.activity.borrow();
                candidates
                    .into_iter()
                    .max_by(|&a, &b| {
                        act[a].cmp(&act[b]).then_with(|| doms[a].len().cmp(&doms[b].len()))
                    })
                    .unwrap()
            }
        }
    }
}

/// Find one solution to `model` with the default (first-fail) ordering.
pub fn solve(model: &CpModel) -> Option<CpSolution> {
    solve_with(model, VariableSelection::FirstFail)
}

/// Find one solution to `model` using `selection` for variable ordering.
pub fn solve_with(model: &CpModel, selection: VariableSelection) -> Option<CpSolution> {
    let cons = model.constraints();
    let mut doms = model.domains.clone();
    if fixpoint(&mut doms, cons).is_err() {
        return None;
    }
    let ctx = build_ctx(cons, selection);
    let mut path: Vec<(usize, usize)> = Vec::new();
    match dfs(&mut doms, &mut path, &ctx) {
        Ok(a) => Some(CpSolution { assignment: a }),
        Err(_) => None,
    }
}

/// Enumerate up to `limit` solutions with the default (first-fail) ordering.
pub fn solutions(model: &CpModel, limit: usize) -> Vec<CpSolution> {
    solutions_with(model, limit, VariableSelection::FirstFail)
}

/// Enumerate up to `limit` solutions using `selection` for variable ordering.
pub fn solutions_with(
    model: &CpModel,
    limit: usize,
    selection: VariableSelection,
) -> Vec<CpSolution> {
    let mut out = Vec::new();
    let mut doms = model.domains.clone();
    if fixpoint(&mut doms, model.constraints()).is_err() {
        return out;
    }
    let ctx = build_ctx(model.constraints(), selection);
    collect(&mut doms, &ctx, limit, &mut out);
    out
}

/// Build the search context (neighbourhoods, degrees, activity, no-goods).
fn build_ctx(cons: &[Box<dyn Constraint>], selection: VariableSelection) -> Ctx<'_> {
    let n = cons.iter().flat_map(|c| c.vars().iter().copied()).max().unwrap_or(0) + 1;
    let mut neighbors = vec![Vec::new(); n];
    let mut degree = vec![0usize; n];
    for c in cons {
        let vs = c.vars();
        for &a in vs {
            if a >= n {
                continue;
            }
            degree[a] += 1;
            for &b in vs {
                if b != a && !neighbors[a].contains(&b) {
                    neighbors[a].push(b);
                }
            }
        }
    }
    for list in &mut neighbors {
        list.sort_unstable();
    }
    Ctx {
        cons,
        neighbors,
        degree,
        selection,
        activity: RefCell::new(vec![0usize; n]),
        nogoods: RefCell::new(Vec::new()),
    }
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
        ctx.bump(&vars);
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
        ctx.bump(&set);
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
        ctx.bump(&set);
        return Err(set);
    }

    let u = ctx.choose_var(doms);

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
    ctx.bump(&out);
    Err(out)
}

fn collect(doms: &mut [Domain], ctx: &Ctx, limit: usize, out: &mut Vec<CpSolution>) {
    if out.len() >= limit {
        return;
    }
    if fixpoint(doms, ctx.cons).is_err() {
        return;
    }
    if doms.iter().all(|d| d.is_singleton()) {
        let assign: Vec<usize> = doms.iter().map(|d| d.value()).collect();
        if ctx.cons.iter().all(|c| c.check(&assign)) {
            out.push(CpSolution { assignment: assign });
        }
        return;
    }
    let var = ctx.choose_var(doms);
    let candidates: Vec<usize> = doms[var].values().to_vec();
    for v in candidates {
        if out.len() >= limit {
            return;
        }
        let mut nd = doms.to_vec();
        nd[var].assign(v);
        collect(&mut nd, ctx, limit, out);
    }
}
