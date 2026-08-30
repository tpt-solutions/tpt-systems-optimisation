//! The constraint-programming model: variables, domains and constraints.
//!
//! Propagation to a fixpoint uses the **AC-3 worklist algorithm** (Mackworth,
//! 1977), generalised from binary arcs `(x_i, x_j)` to constraint-scope
//! hyperarcs: a constraint is re-filtered only when some variable it watches
//! has actually changed since its last revision. For binary constraints this
//! reduces exactly to classic AC-3; for global constraints it is the standard
//! generalised arc-consistency (GAC) worklist.

use std::collections::VecDeque;
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

impl CpModel {
    /// Preprocess the model in place with **AC-3**: propagate every
    /// constraint to a fixpoint using the worklist algorithm, removing values
    /// that cannot appear in any solution *before* search starts.
    ///
    /// Returns the propagation statistics, or `Err(`[`Inconsistency`]) when a
    /// domain wipes out — the model is unsatisfiable and search can be
    /// skipped entirely. This is exactly the root-node propagation
    /// [`solve`](crate::solver::solve) performs, exposed so callers can
    /// detect unsatisfiability cheaply, inspect forced assignments
    /// (`domain.is_singleton()`), or measure propagation effort via
    /// [`Ac3Stats`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_opt_cp::constraints::AllDifferent;
    /// use tpt_opt_cp::model::CpModel;
    ///
    /// let mut m = CpModel::new();
    /// let x = m.add_var_values(vec![0]);      // fixed
    /// let y = m.add_var(0, 1);
    /// let z = m.add_var(0, 2);
    /// m.add_constraint(Box::new(AllDifferent::new(vec![x, y, z])));
    ///
    /// let stats = m.ac3().expect("propagation succeeds");
    /// assert_eq!(m.domains[y].value(), 1);    // forced by x = 0
    /// assert_eq!(m.domains[z].value(), 2);    // then forced by y = 1
    /// assert_eq!(stats.removals, 3);
    /// ```
    pub fn ac3(&mut self) -> Result<Ac3Stats, Inconsistency> {
        fixpoint_ac3(&mut self.domains, &self.constraints).map_err(|_| Inconsistency)
    }

    /// Preprocess the model in place with **AC-4** (Mohr & Henderson, 1986):
    /// maintain, for every `(variable, value)` pair in each constraint's scope,
    /// a support witness; whenever a value is removed, every constraint that
    /// watched it re-checks its own `(variable, value)` pairs and prunes those
    /// that lost their last support. This reaches the same domain-consistent
    /// fixpoint as [`ac3`](CpModel::ac3) but with tighter incremental pruning
    /// on constraints that expose a cheap support test; native `propagate`
    /// filters run alongside the support oracle so global constraints still get
    /// their dedicated GAC each revision.
    ///
    /// Returns the propagation statistics, or `Err(`[`Inconsistency`]) when a
    /// domain wipes out.
    pub fn ac4(&mut self) -> Result<Ac3Stats, Inconsistency> {
        fixpoint_ac4(&mut self.domains, &self.constraints).map_err(|_| Inconsistency)
    }
}

/// Statistics from one AC-3 propagation run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Ac3Stats {
    /// Number of constraint revisions performed (filter invocations). A
    /// no-change run revises every constraint exactly once; each subsequent
    /// revision is triggered by an actual domain change in the constraint's
    /// scope.
    pub revisions: usize,
    /// Total number of values removed from domains across all revisions.
    pub removals: usize,
}

/// Run AC-3 propagation to a fixpoint; returns `Err` if a domain empties.
pub(crate) fn fixpoint(
    doms: &mut [Domain],
    cons: &[Box<dyn Constraint>],
) -> Result<(), Inconsistency> {
    fixpoint_report(doms, cons).map_err(|_| Inconsistency)
}

/// Run AC-3 propagation to a fixpoint, reporting the index of the constraint
/// whose filter emptied a domain (used by conflict-directed backjumping to
/// attribute failures). Returns `Err(index)` on wipeout.
///
/// Worklist discipline: initially every constraint is queued. Popping a
/// constraint runs its filter once; whenever a variable in its scope loses
/// values, every constraint watching that variable is re-enqueued (including
/// the reviser itself — see the loop comment). The loop ends when the queue
/// drains with no further pruning — a fixpoint.
/// Constraints are assumed to prune only within their own scope (the contract
/// every `Constraint` impl in this crate follows).
pub(crate) fn fixpoint_ac3(
    doms: &mut [Domain],
    cons: &[Box<dyn Constraint>],
) -> Result<Ac3Stats, usize> {
    // Variable -> constraints whose scope contains it (watch lists).
    let mut var_cons: Vec<Vec<usize>> = vec![Vec::new(); doms.len()];
    for (ci, c) in cons.iter().enumerate() {
        for &v in c.vars() {
            if v < doms.len() && !var_cons[v].contains(&ci) {
                var_cons[v].push(ci);
            }
        }
    }

    let mut queue: VecDeque<usize> = (0..cons.len()).collect();
    let mut queued = vec![true; cons.len()];
    let mut stats = Ac3Stats::default();

    while let Some(ci) = queue.pop_front() {
        queued[ci] = false;
        let scope = cons[ci].vars().to_vec();
        let before: Vec<usize> = scope.iter().map(|&v| doms[v].len()).collect();

        stats.revisions += 1;
        cons[ci].propagate(doms).map_err(|_| ci)?;

        for (k, &v) in scope.iter().enumerate() {
            if doms[v].is_empty() {
                return Err(ci);
            }
            if doms[v].len() != before[k] {
                stats.removals += before[k] - doms[v].len();
                // Re-enqueue every constraint watching this variable —
                // including `ci` itself: unlike a binary AC-3 `revise` (a
                // complete support check), a global filter is not guaranteed
                // to reach its own fixpoint in one call, so keep revising it
                // until a pass makes no change.
                for &cj in &var_cons[v] {
                    if !queued[cj] {
                        queued[cj] = true;
                        queue.push_back(cj);
                    }
                }
            }
        }
    }
    Ok(stats)
}

/// Run **AC-4** propagation to a fixpoint; returns `Err(index)` if a domain
/// empties. Combines the support-maintenance algorithm with native `propagate`
/// filters: each time a variable loses a value, every constraint watching it is
/// (a) re-filtered through its own `propagate` GAC and (b) re-checked for
/// `(variable, value)` pairs that lost their last support via
/// [`Constraint::supported`]; pruned values are enqueued for further
/// propagation. The result is a domain-consistent fixpoint.
pub(crate) fn fixpoint_ac4(
    doms: &mut [Domain],
    cons: &[Box<dyn Constraint>],
) -> Result<Ac3Stats, usize> {
    let mut var_cons: Vec<Vec<usize>> = vec![Vec::new(); doms.len()];
    for (ci, c) in cons.iter().enumerate() {
        for &v in c.vars() {
            if v < doms.len() && !var_cons[v].contains(&ci) {
                var_cons[v].push(ci);
            }
        }
    }

    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    let mut stats = Ac3Stats::default();

    // Initial revision: native GAC pass, then support pruning for every pair.
    for (ci, c) in cons.iter().enumerate() {
        stats.revisions += 1;
        if c.propagate(doms).is_err() {
            return Err(ci);
        }
        for &v in c.vars() {
            if doms[v].is_empty() {
                return Err(ci);
            }
        }
    }
    for ci in 0..cons.len() {
        prune_unsupported(ci, doms, cons, &mut queue, &mut stats);
        if doms.iter().any(|d| d.is_empty()) {
            // Locate the offending constraint for the error index.
            for (j, c) in cons.iter().enumerate() {
                if c.vars().iter().any(|&v| doms[v].is_empty()) {
                    return Err(j);
                }
            }
        }
    }

    while let Some((rv, _rval)) = queue.pop_front() {
        for &cj in &var_cons[rv] {
            stats.revisions += 1;
            if cons[cj].propagate(doms).is_err() {
                return Err(cj);
            }
            prune_unsupported(cj, doms, cons, &mut queue, &mut stats);
            if doms.iter().any(|d| d.is_empty()) {
                for (j, c) in cons.iter().enumerate() {
                    if c.vars().iter().any(|&v| doms[v].is_empty()) {
                        return Err(j);
                    }
                }
            }
        }
    }
    Ok(stats)
}

/// Remove every `(v, val)` in `cons[ci]`'s scope that has no support, enqueue
/// each removed pair, and count the removals in `stats`.
fn prune_unsupported(
    ci: usize,
    doms: &mut [Domain],
    cons: &[Box<dyn Constraint>],
    queue: &mut VecDeque<(usize, usize)>,
    stats: &mut Ac3Stats,
) {
    let scope = cons[ci].vars().to_vec();
    for &v in &scope {
        let vals = doms[v].values().to_vec();
        for val in vals {
            if !cons[ci].supported(v, val, doms) && doms[v].remove(val) {
                stats.removals += 1;
                queue.push_back((v, val));
            }
        }
    }
}

/// Compatibility wrapper preserving the historical round-robin signature:
/// delegates to [`fixpoint_ac3`] and discards the statistics. (The AC-3
/// worklist reaches the identical fixpoint while doing strictly less work on
/// sparse constraint graphs.)
pub(crate) fn fixpoint_report(
    doms: &mut [Domain],
    cons: &[Box<dyn Constraint>],
) -> Result<(), usize> {
    fixpoint_ac3(doms, cons).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{AllDifferent, Linear};

    #[test]
    fn ac3_forces_chain_and_counts_removals() {
        // x fixed to 0; AllDifferent over {x, y, z} forces y = 1 (loses 0),
        // which in turn forces z = 2 (loses 0, then 1).
        let mut m = CpModel::new();
        let x = m.add_var_values(vec![0]);
        let y = m.add_var(0, 1);
        let z = m.add_var(0, 2);
        m.add_constraint(Box::new(AllDifferent::new(vec![x, y, z])));

        let stats = m.ac3().expect("feasible");
        assert_eq!(m.domains[x].value(), 0);
        assert_eq!(m.domains[y].value(), 1);
        assert_eq!(m.domains[z].value(), 2);
        assert_eq!(stats.removals, 3, "y loses 0; z loses 0 then 1");
        assert!(stats.revisions >= 3, "filter needs self-revisions to fixpoint");
    }

    #[test]
    fn ac3_detects_wipeout_as_inconsistency() {
        let mut m = CpModel::new();
        let x = m.add_var_values(vec![0]);
        let y = m.add_var_values(vec![0]);
        m.add_constraint(Box::new(AllDifferent::new(vec![x, y])));
        assert!(m.ac3().is_err(), "two vars fixed to the same value");
    }

    #[test]
    fn ac3_on_loose_model_revises_each_constraint_once() {
        // x + y <= 18 with domains [0,9]: every combination is feasible, so
        // nothing is pruned and the single queued revision suffices.
        let mut m = CpModel::new();
        let x = m.add_var(0, 9);
        let y = m.add_var(0, 9);
        m.add_constraint(Box::new(Linear::new(vec![(x, 1), (y, 1)], Relation::Le, 18)));
        let stats = m.ac3().expect("feasible");
        assert_eq!(stats.removals, 0);
        assert_eq!(stats.revisions, 1);
        assert_eq!(m.domains[x].len(), 10);
        assert_eq!(m.domains[y].len(), 10);
    }

    #[test]
    fn ac3_fixpoint_matches_naive_round_robin() {
        // Cross-check the worklist result against brute-force repeated
        // full passes on a mixed model with propagation chains.
        let build = || {
            let mut m = CpModel::new();
            let a = m.add_var(0, 3);
            let b = m.add_var(0, 3);
            let c = m.add_var(0, 3);
            m.add_constraint(Box::new(Linear::new(vec![(a, 1), (b, 1)], Relation::Eq, 4)));
            m.add_constraint(Box::new(Linear::new(vec![(b, 1), (c, 2)], Relation::Le, 5)));
            m.add_constraint(Box::new(AllDifferent::new(vec![a, b, c])));
            m
        };
        let mut m = build();
        m.ac3().expect("feasible");

        // Naive: repeat full passes until nothing changes.
        let mut n = build();
        loop {
            let before: Vec<usize> = n.domains.iter().map(|d| d.len()).collect();
            let cons = &n.constraints;
            for c in cons.iter() {
                let _ = c.propagate(&mut n.domains);
            }
            let after: Vec<usize> = n.domains.iter().map(|d| d.len()).collect();
            if before == after {
                break;
            }
        }
        assert_eq!(m.domains, n.domains, "AC-3 must reach the same fixpoint");
    }

    #[test]
    fn ac3_preprocessing_keeps_solve_results_identical() {
        // Preprocessing must not change what search returns.
        let build = || {
            let mut m = CpModel::new();
            let a = m.add_var(0, 4);
            let b = m.add_var(0, 4);
            let c = m.add_var(0, 4);
            m.add_constraint(Box::new(Linear::new(vec![(a, 2), (b, 1)], Relation::Eq, 6)));
            m.add_constraint(Box::new(AllDifferent::new(vec![a, b, c])));
            m
        };
        let raw = build();
        let mut pre = build();
        assert!(pre.ac3().is_ok());
        let s_raw = crate::solver::solve(&raw).expect("feasible");
        let s_pre = crate::solver::solve(&pre).expect("feasible");
        assert_eq!(s_raw.assignment, s_pre.assignment);
    }
}
