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
        Self {
            terms,
            rel,
            rhs,
            vars,
        }
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
        Self {
            tasks,
            capacity,
            vars,
        }
    }
}

impl Constraint for Cumulative {
    fn vars(&self) -> &[usize] {
        &self.vars
    }

    fn propagate(&self, doms: &mut [Domain]) -> Result<(), Inconsistency> {
        for t in &self.tasks {
            let before: Vec<usize> = doms[t.start].values().to_vec();
            for &s in &before {
                let end = s + t.duration;
                let mut ok = true;
                for time in s..end {
                    let mut load = 0usize;
                    for ot in &self.tasks {
                        let lo = doms[ot.start].min();
                        let hi_start = doms[ot.start].max() + ot.duration;
                        if hi_start > time && lo <= time {
                            load += ot.demand;
                        }
                    }
                    if load > self.capacity {
                        ok = false;
                        break;
                    }
                }
                if !ok {
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
        Self {
            array,
            index,
            value,
            vars: vec![index, value],
        }
    }
}

impl Constraint for Element {
    fn vars(&self) -> &[usize] {
        &self.vars
    }

    fn propagate(&self, doms: &mut [Domain]) -> Result<(), Inconsistency> {
        let value_domain: Vec<usize> = doms[self.value].values().to_vec();
        let allowed_vals: Vec<usize> = doms[self.index]
            .values()
            .iter()
            .filter_map(|&i| self.array.get(i).copied())
            .collect();
        if doms[self.value].retain(|v| allowed_vals.contains(&v)) && doms[self.value].is_empty() {
            return Err(Inconsistency);
        }
        if doms[self.index].retain(|i| {
            self.array.get(i).is_some_and(|&a| value_domain.contains(&a))
        }) && doms[self.index].is_empty()
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
                        && t.iter()
                            .enumerate()
                            .all(|(j, &tv)| doms[self.vars[j]].contains(tv))
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
        Self {
            inner,
            bvar,
            vars,
        }
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
