//! Constraint definitions and their propagation rules.

use std::vec::Vec;

use crate::domain::Domain;
use crate::model::Relation;

/// A propagation failure (empty domain reached).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inconsistency;

/// A constraint over a subset of the model's variables.
pub trait Constraint {
    /// Variables this constraint references.
    fn vars(&self) -> &[usize];
    /// Narrow the given domains; returns `Err(Inconsistency)` if infeasible.
    fn propagate(&self, doms: &mut [Domain]) -> Result<(), Inconsistency>;
    /// Full check on a complete assignment (used after search fixes everything).
    fn check(&self, assign: &[usize]) -> bool;
}

fn unique_vars(xs: &[(usize, i64)]) -> Vec<usize> {
    let mut v: Vec<usize> = xs.iter().map(|&(x, _)| x).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Linear constraint `Σ coeffs_i * x_i  rel  rhs` over integer variables.
pub struct Linear {
    terms: Vec<(usize, i64)>,
    rel: Relation,
    rhs: i64,
    vars: Vec<usize>,
}

impl Linear {
    /// Build `Σ coeffs_i * x_i  rel  rhs`.
    pub fn new(terms: Vec<(usize, i64)>, rel: Relation, rhs: i64) -> Self {
        let vars = unique_vars(&terms);
        Self { terms, rel, rhs, vars }
    }
}

impl Constraint for Linear {
    fn vars(&self) -> &[usize] {
        &self.vars
    }

    fn propagate(&self, doms: &mut [Domain]) -> Result<(), Inconsistency> {
        for &(k, ck) in &self.terms {
            let before: Vec<usize> = doms[k].values().to_vec();
            for &v in &before {
                let iv = v as i64;
                let mut lo = 0i64;
                let mut hi = 0i64;
                for &(i, ci) in &self.terms {
                    if i == k {
                        continue;
                    }
                    let d = &doms[i];
                    if ci >= 0 {
                        lo += ci * d.min() as i64;
                        hi += ci * d.max() as i64;
                    } else {
                        lo += ci * d.max() as i64;
                        hi += ci * d.min() as i64;
                    }
                }
                let total_lo = lo + ck * iv;
                let total_hi = hi + ck * iv;
                let feasible = match self.rel {
                    Relation::Le => total_lo <= self.rhs,
                    Relation::Ge => total_hi >= self.rhs,
                    Relation::Eq => total_lo <= self.rhs && self.rhs <= total_hi,
                };
                if !feasible {
                    doms[k].remove(v);
                    if doms[k].is_empty() {
                        return Err(Inconsistency);
                    }
                }
            }
        }
        Ok(())
    }

    fn check(&self, assign: &[usize]) -> bool {
        let mut total = 0i64;
        for &(v, c) in &self.terms {
            total += c * assign[v] as i64;
        }
        match self.rel {
            Relation::Le => total <= self.rhs,
            Relation::Ge => total >= self.rhs,
            Relation::Eq => total == self.rhs,
        }
    }
}

/// All-different constraint: the listed variables take distinct values.
pub struct AllDifferent {
    vars: Vec<usize>,
}

impl AllDifferent {
    /// Build from a variable list.
    pub fn new(vars: Vec<usize>) -> Self {
        Self { vars }
    }
}

impl Constraint for AllDifferent {
    fn vars(&self) -> &[usize] {
        &self.vars
    }

    fn propagate(&self, doms: &mut [Domain]) -> Result<(), Inconsistency> {
        let singletons: Vec<(usize, usize)> = self
            .vars
            .iter()
            .filter(|&&v| doms[v].is_singleton())
            .map(|&v| (v, doms[v].value()))
            .collect();
        for &(sv, val) in &singletons {
            for &v in &self.vars {
                if v != sv && doms[v].remove(val) && doms[v].is_empty() {
                    return Err(Inconsistency);
                }
            }
        }
        Ok(())
    }

    fn check(&self, assign: &[usize]) -> bool {
        let mut seen = std::collections::HashSet::new();
        for &v in &self.vars {
            if !seen.insert(assign[v]) {
                return false;
            }
        }
        true
    }
}

/// A task for the cumulative (resource-constrained) constraint.
#[derive(Debug, Clone)]
pub struct Task {
    /// Variable index of the task's start time.
    pub start: usize,
    /// Fixed duration.
    pub duration: usize,
    /// Fixed resource demand.
    pub demand: usize,
}

/// Cumulative (resource-constrained scheduling) constraint.
pub struct Cumulative {
    tasks: Vec<Task>,
    capacity: usize,
    vars: Vec<usize>,
}

impl Cumulative {
    /// Build from tasks and a renewable resource capacity.
    pub fn new(tasks: Vec<Task>, capacity: usize) -> Self {
        let vars = tasks.iter().map(|t| t.start).collect();
        Self { tasks, capacity, vars }
    }
}

impl Constraint for Cumulative {
    fn vars(&self) -> &[usize] {
        &self.vars
    }

    fn propagate(&self, doms: &mut [Domain]) -> Result<(), Inconsistency> {
        // Sound (conservative) pruning: a start is infeasible only if this task's
        // own demand alone already exceeds capacity. Deeper time-table pruning is
        // enforced via the final `check` during search.
        for t in &self.tasks {
            let before: Vec<usize> = doms[t.start].values().to_vec();
            for &s in &before {
                if t.demand > self.capacity {
                    doms[t.start].remove(s);
                    if doms[t.start].is_empty() {
                        return Err(Inconsistency);
                    }
                }
            }
        }
        Ok(())
    }

    fn check(&self, assign: &[usize]) -> bool {
        let mut max_time = 0usize;
        for t in &self.tasks {
            max_time = max_time.max(assign[t.start] + t.duration);
        }
        for time in 0..max_time {
            let mut load = 0usize;
            for t in &self.tasks {
                let s = assign[t.start];
                if time >= s && time < s + t.duration {
                    load += t.demand;
                }
            }
            if load > self.capacity {
                return false;
            }
        }
        true
    }
}

/// Element constraint `array[index] = value`.
pub struct Element {
    array: Vec<usize>,
    index: usize,
    value: usize,
    vars: Vec<usize>,
}

impl Element {
    /// Build `array[index] = value`.
    pub fn new(array: Vec<usize>, index: usize, value: usize) -> Self {
        Self { array, index, value, vars: vec![index, value] }
    }
}

impl Constraint for Element {
    fn vars(&self) -> &[usize] {
        &self.vars
    }

    fn propagate(&self, doms: &mut [Domain]) -> Result<(), Inconsistency> {
        let value_domain: Vec<usize> = doms[self.value].values().to_vec();
        let allowed_vals: Vec<usize> =
            doms[self.index].values().iter().filter_map(|&i| self.array.get(i).copied()).collect();
        if doms[self.value].retain(|v| allowed_vals.contains(&v)) && doms[self.value].is_empty() {
            return Err(Inconsistency);
        }
        if doms[self.index]
            .retain(|i| self.array.get(i).is_some_and(|&a| value_domain.contains(&a)))
            && doms[self.index].is_empty()
        {
            return Err(Inconsistency);
        }
        Ok(())
    }

    fn check(&self, assign: &[usize]) -> bool {
        let i = assign[self.index];
        self.array.get(i).is_some_and(|&a| a == assign[self.value])
    }
}

/// Table constraint: tuples of allowed simultaneous assignments.
pub struct Table {
    vars: Vec<usize>,
    tuples: Vec<Vec<usize>>,
}

impl Table {
    /// Build from the ordered variable list and allowed tuples.
    pub fn new(vars: Vec<usize>, tuples: Vec<Vec<usize>>) -> Self {
        Self { vars, tuples }
    }
}

impl Constraint for Table {
    fn vars(&self) -> &[usize] {
        &self.vars
    }

    fn propagate(&self, doms: &mut [Domain]) -> Result<(), Inconsistency> {
        for (pos, &v) in self.vars.iter().enumerate() {
            let before: Vec<usize> = doms[v].values().to_vec();
            for val in before {
                let supported = self.tuples.iter().any(|t| {
                    t[pos] == val
                        && t.iter().enumerate().all(|(j, &tv)| doms[self.vars[j]].contains(tv))
                });
                if !supported {
                    doms[v].remove(val);
                    if doms[v].is_empty() {
                        return Err(Inconsistency);
                    }
                }
            }
        }
        Ok(())
    }

    fn check(&self, assign: &[usize]) -> bool {
        let vals: Vec<usize> = self.vars.iter().map(|&v| assign[v]).collect();
        self.tuples.iter().any(|t| *t == vals)
    }
}

/// Regular (automaton-based sequence) constraint: the sequence of values
/// taken by `vars` must drive a deterministic finite automaton from
/// `initial` to one of `accepting`.
///
/// States are numbered `0..num_states`; transitions are `(from, symbol,
/// to)` triples where `symbol` is a variable *value*. Propagation uses
/// forward/backward reachability: a value survives at position `i` only if
/// some transition supports it between a forward-reachable state and a
/// backward-co-reachable state (domain consistency for the sequence).
pub struct Regular {
    vars: Vec<usize>,
    /// `(from_state, symbol, to_state)` triples.
    transitions: Vec<(usize, usize, usize)>,
    initial: usize,
    accepting: Vec<usize>,
}

impl Regular {
    /// Build a regular constraint over `vars` with the given DFA.
    ///
    /// Panics if `initial` or any accepting state is out of range, or a
    /// transition references an unknown state.
    pub fn new(
        vars: Vec<usize>,
        transitions: Vec<(usize, usize, usize)>,
        initial: usize,
        accepting: Vec<usize>,
        num_states: usize,
    ) -> Self {
        assert!(initial < num_states, "initial state out of range");
        assert!(accepting.iter().all(|&s| s < num_states), "accepting state out of range");
        assert!(
            transitions.iter().all(|&(f, _, t)| f < num_states && t < num_states),
            "transition state out of range"
        );
        Self { vars, transitions, initial, accepting }
    }

    fn forward(&self, doms: &[Domain]) -> Vec<Vec<usize>> {
        // F[i] = states reachable after consuming positions 0..i.
        let n = self.vars.len();
        let mut f = vec![Vec::new(); n + 1];
        f[0].push(self.initial);
        for i in 0..n {
            let mut next: Vec<usize> = Vec::new();
            for &(from, sym, to) in &self.transitions {
                if f[i].contains(&from) && doms[self.vars[i]].contains(sym) && !next.contains(&to) {
                    next.push(to);
                }
            }
            f[i + 1] = next;
        }
        f
    }

    fn backward(&self, doms: &[Domain]) -> Vec<Vec<usize>> {
        // B[i] = states from which some accepting state is reachable using
        // positions i..n.
        let n = self.vars.len();
        let mut b = vec![Vec::new(); n + 1];
        b[n] = self.accepting.clone();
        for i in (0..n).rev() {
            let mut prev: Vec<usize> = Vec::new();
            for &(from, sym, to) in &self.transitions {
                if b[i + 1].contains(&to)
                    && doms[self.vars[i]].contains(sym)
                    && !prev.contains(&from)
                {
                    prev.push(from);
                }
            }
            b[i] = prev;
        }
        b
    }
}

impl Constraint for Regular {
    fn vars(&self) -> &[usize] {
        &self.vars
    }

    fn propagate(&self, doms: &mut [Domain]) -> Result<(), Inconsistency> {
        let f = self.forward(doms);
        // Quick reject: no accepting state reachable at all.
        if f[self.vars.len()].iter().all(|s| !self.accepting.contains(s)) {
            return Err(Inconsistency);
        }
        let b = self.backward(doms);
        let n = self.vars.len();
        for i in 0..n {
            let mut supported = Vec::new();
            for &(from, sym, to) in &self.transitions {
                if f[i].contains(&from) && b[i + 1].contains(&to) && !supported.contains(&sym) {
                    supported.push(sym);
                }
            }
            let changed = doms[self.vars[i]].retain(|v| supported.contains(&v));
            if changed && doms[self.vars[i]].is_empty() {
                return Err(Inconsistency);
            }
        }
        Ok(())
    }

    fn check(&self, assign: &[usize]) -> bool {
        let mut state = self.initial;
        for &v in &self.vars {
            let sym = assign[v];
            match self.transitions.iter().find(|&&(f, s, _)| f == state && s == sym) {
                Some(&(_, _, to)) => state = to,
                None => return false,
            }
        }
        self.accepting.contains(&state)
    }
}

/// Circuit constraint: `vars[i] = j` means "node `i`'s successor is node
/// `j`". Enforces that the successors form a single Hamiltonian cycle over
/// all `n` nodes: every value distinct (a permutation), no self-loops, and
/// no sub-cycle shorter than `n`.
///
/// Propagation removes self-loops, applies all-different singleton
/// reasoning, prunes successor choices that would close a premature cycle,
/// and fails when some node loses every possible predecessor.
pub struct Circuit {
    vars: Vec<usize>,
}

impl Circuit {
    /// Build over the successor variables of nodes `0..vars.len()`.
    pub fn new(vars: Vec<usize>) -> Self {
        Self { vars }
    }

    /// Successor map implied by currently-fixed variables (`None` if the
    /// variable is not yet a singleton).
    fn fixed_successors(&self, doms: &[Domain]) -> Vec<Option<usize>> {
        self.vars.iter().map(|&v| doms[v].is_singleton().then(|| doms[v].value())).collect()
    }
}

impl Constraint for Circuit {
    fn vars(&self) -> &[usize] {
        &self.vars
    }

    fn propagate(&self, doms: &mut [Domain]) -> Result<(), Inconsistency> {
        let n = self.vars.len();

        // 1. No self-loops.
        for (i, &v) in self.vars.iter().enumerate() {
            if doms[v].remove(i) && doms[v].is_empty() {
                return Err(Inconsistency);
            }
        }

        // 2. All-different singleton propagation.
        let singles: Vec<(usize, usize)> = self
            .vars
            .iter()
            .enumerate()
            .filter(|&(_, &v)| doms[v].is_singleton())
            .map(|(i, &v)| (i, doms[v].value()))
            .collect();
        for &(si, val) in &singles {
            for (i, &v) in self.vars.iter().enumerate() {
                if i != si && doms[v].remove(val) && doms[v].is_empty() {
                    return Err(Inconsistency);
                }
            }
        }

        // 3. Every node needs at least one possible predecessor.
        for node in 0..n {
            let supported =
                self.vars.iter().enumerate().any(|(i, &v)| i != node && doms[v].contains(node));
            if !supported {
                return Err(Inconsistency);
            }
        }

        // 4. Closed sub-cycle among already-fixed successors: if following
        //    the fixed chain from a fixed node returns to it without
        //    covering all n nodes, no completion can merge the cycles.
        let succ_fixed = self.fixed_successors(doms);
        for i in 0..n {
            if succ_fixed[i].is_none() {
                continue;
            }
            let mut cur = i;
            let mut visited = vec![false; n];
            let mut count = 0usize;
            loop {
                if visited[cur] {
                    break; // returned to a node on this walk: cycle closed
                }
                visited[cur] = true;
                count += 1;
                match succ_fixed[cur] {
                    Some(next) => cur = next,
                    None => break,
                }
            }
            if visited[i] && count > 0 && cur == i && count < n {
                return Err(Inconsistency);
            }
        }

        // 5. Premature-cycle pruning: assigning x_i = j closes a cycle
        //    through i; reject unless that cycle would span all n nodes.
        let succ = self.fixed_successors(doms);
        for (i, &v) in self.vars.iter().enumerate() {
            if doms[v].is_singleton() {
                continue;
            }
            let candidates: Vec<usize> = doms[v].values().to_vec();
            for &j in &candidates {
                // Walk the fixed-successor chain starting at j.
                let mut cur = j;
                let mut visited = vec![false; n];
                let mut steps = 0usize;
                let mut reaches_i = false;
                let mut closes_early = false;
                loop {
                    if cur == i {
                        reaches_i = true;
                        break;
                    }
                    if visited[cur] {
                        // Closed a cycle that does not pass through i.
                        closes_early = true;
                        break;
                    }
                    visited[cur] = true;
                    steps += 1;
                    match succ[cur] {
                        Some(next) => cur = next,
                        None => break, // chain ends at an unfixed variable
                    }
                }
                let premature = reaches_i && steps < n - 1;
                if premature || closes_early {
                    doms[v].remove(j);
                    if doms[v].is_empty() {
                        return Err(Inconsistency);
                    }
                }
            }
        }
        Ok(())
    }

    fn check(&self, assign: &[usize]) -> bool {
        let n = self.vars.len();
        // Permutation without self-loops.
        let mut seen = vec![false; n];
        for (i, &v) in self.vars.iter().enumerate() {
            let j = assign[v];
            if j >= n || j == i || seen[j] {
                return false;
            }
            seen[j] = true;
        }
        // Single cycle covering every node: walk n successors from node 0
        // and require every node to be visited exactly once en route.
        let mut cur = 0usize;
        let mut visited = vec![false; n];
        for _ in 0..n {
            if visited[cur] {
                return false;
            }
            visited[cur] = true;
            cur = assign[self.vars[cur]];
        }
        cur == 0 && visited.iter().all(|&b| b)
    }
}

/// Reified constraint `b <-> inner`.
pub struct Reified {
    inner: Box<dyn Constraint>,
    bvar: usize,
    vars: Vec<usize>,
}

impl Reified {
    /// Build a reification of `inner` governed by boolean variable `bvar`.
    pub fn new(inner: Box<dyn Constraint>, bvar: usize) -> Self {
        let mut vars = inner.vars().to_vec();
        vars.push(bvar);
        Self { inner, bvar, vars }
    }
}

impl Constraint for Reified {
    fn vars(&self) -> &[usize] {
        &self.vars
    }

    fn propagate(&self, doms: &mut [Domain]) -> Result<(), Inconsistency> {
        if doms[self.bvar].is_singleton() {
            let b = doms[self.bvar].value() == 1;
            if b {
                self.inner.propagate(doms)?;
            }
        }
        Ok(())
    }

    fn check(&self, assign: &[usize]) -> bool {
        let b = assign[self.bvar] == 1;
        if b {
            self.inner.check(assign)
        } else {
            true
        }
    }
}
