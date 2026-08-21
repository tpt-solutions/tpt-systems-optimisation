//! Genetic algorithms (GA) over continuous and permutation genomes.
//!
//! The GA is generic over a [`Genome`] `G`. Two genomes are provided out of the
//! box: `Vec<f64>` (continuous) and `Vec<usize>` (permutation). Operators are
//! selected by [`CrossoverKind`] / [`MutationKind`] / [`SelectionKind`] and are
//! implemented as generic functions over the [`Gene`] trait, so custom genomes
//! can be plugged in by implementing `Gene` + `Genome`.
//!
//! Determinism: all randomness flows through a seeded [`Xoshiro256`](crate::Xoshiro256).

use std::cmp::Ordering;

use tpt_math_prob::Xoshiro256;
use tpt_opt_core::{OptError, Sense};

use crate::history::ConvergenceHistory;
use crate::problem::Objective;
use crate::result::HeuristicResult;
use crate::rng::Rng;

/// A single gene in a [`Genome`].
///
/// Implemented for `f64` (continuous) and `usize` (permutation indices). A gene
/// knows how to be randomly initialised and how to undergo `flip` / `bit_flip`
/// mutation within `[lo, hi]`.
pub trait Gene: Copy + Clone + PartialEq + std::fmt::Debug {
    /// Random gene within `[lo, hi]`.
    fn random(rng: &mut dyn Rng, lo: f64, hi: f64) -> Self;
    /// Perturbing mutation (Gaussian-style for `f64`, rounded step for `usize`).
    fn flip(&self, rng: &mut dyn Rng, lo: f64, hi: f64) -> Self;
    /// Binary toggle between the two endpoints `lo` / `hi`.
    fn bit_flip(&self, _rng: &mut dyn Rng, lo: f64, hi: f64) -> Self;
    /// Lossless conversion into an `f64` for history/result reporting.
    fn to_f64(&self) -> f64;
}

impl Gene for f64 {
    fn random(rng: &mut dyn Rng, lo: f64, hi: f64) -> Self {
        rng.range(lo, hi)
    }
    fn flip(&self, rng: &mut dyn Rng, lo: f64, hi: f64) -> Self {
        let span = (hi - lo).abs().max(1e-3);
        let step = span * 0.1;
        let v = self + rng.normal() * step;
        if lo.is_finite() {
            v.max(lo)
        } else {
            v
        }
        .min(if hi.is_finite() { hi } else { v })
    }
    fn bit_flip(&self, _rng: &mut dyn Rng, lo: f64, hi: f64) -> Self {
        if *self < (lo + hi) / 2.0 {
            hi
        } else {
            lo
        }
    }
    fn to_f64(&self) -> f64 {
        *self
    }
}

impl Gene for usize {
    fn random(rng: &mut dyn Rng, lo: f64, hi: f64) -> Self {
        // `hi` is treated as an exclusive upper bound (permutation indices 0..n).
        let n = ((hi - lo).round()) as usize;
        (lo as usize) + rng.index(n.max(1))
    }
    fn flip(&self, rng: &mut dyn Rng, lo: f64, hi: f64) -> Self {
        let span = (hi - lo).abs().max(1.0);
        let step = span * 0.1;
        let v = (*self as f64 + rng.normal() * step).round();
        let lo_i = lo as usize;
        let hi_i = (hi as usize).saturating_sub(1).max(lo_i);
        let v = if v < lo_i as f64 {
            lo_i as f64
        } else {
            v
        };
        let v = if v > hi_i as f64 { hi_i as f64 } else { v };
        v as usize
    }
    fn bit_flip(&self, _rng: &mut dyn Rng, lo: f64, hi: f64) -> Self {
        let lo_i = lo as usize;
        let hi_i = (hi as usize).saturating_sub(1).max(lo_i);
        if *self == lo_i {
            hi_i
        } else {
            lo_i
        }
    }
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

/// A genome operated on by the GA. `Vec<T: Gene>` implements this for both
/// continuous and permutation representations.
pub trait Genome: Clone + PartialEq + std::fmt::Debug {
    /// Number of genes.
    fn len(&self) -> usize;
    /// `true` if the genome is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Random genome sampled under `setup`.
    fn random(rng: &mut dyn Rng, setup: &GaSetup) -> Self;
    /// Apply the configured mutation operator in place.
    fn mutate(&mut self, rng: &mut dyn Rng, setup: &GaSetup);
    /// Produce two children from two parents.
    fn crossover(a: &Self, b: &Self, rng: &mut dyn Rng, setup: &GaSetup) -> (Self, Self);
    /// Convert into an `f64` vector for reporting.
    fn as_f64(&self) -> Vec<f64>;
}

impl<T: Gene + Clone + std::fmt::Debug> Genome for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }
    fn random(rng: &mut dyn Rng, setup: &GaSetup) -> Self {
        (0..setup.dim)
            .map(|i| {
                let (lo, hi) = setup.bounds[i];
                T::random(rng, lo, hi)
            })
            .collect()
    }
    fn mutate(&mut self, rng: &mut dyn Rng, setup: &GaSetup) {
        if rng.next_f64() < setup.mutation_rate {
            mutate(setup.mutation, self, rng, &setup.bounds);
        }
    }
    fn crossover(a: &Self, b: &Self, rng: &mut dyn Rng, setup: &GaSetup) -> (Self, Self) {
        if rng.next_f64() < setup.crossover_rate {
            crossover(setup.crossover, a, b, rng, setup.dim)
        } else {
            (a.clone(), b.clone())
        }
    }
    fn as_f64(&self) -> Vec<f64> {
        self.iter().map(|g| g.to_f64()).collect()
    }
}

/// Crossover operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossoverKind {
    /// One random cut point.
    SinglePoint,
    /// Two random cut points.
    TwoPoint,
    /// Per-gene uniform choice of parent.
    Uniform,
    /// Position-based (order) crossover for permutations.
    OrderBased,
}

/// Mutation operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationKind {
    /// Toggle a gene between its two endpoints (0/1 for binary).
    BitFlip,
    /// Perturb a gene (Gaussian-style).
    Flip,
    /// Swap two genes.
    Swap,
    /// Reverse a sub-segment.
    Inversion,
    /// Shuffle a sub-segment.
    Scramble,
}

/// Parent-selection operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionKind {
    /// Tournament of `k` randomly drawn individuals; keep the fittest.
    Tournament(usize),
    /// Fitness-proportional roulette-wheel selection.
    Roulette,
    /// Linear-rank-proportional selection.
    Rank,
}

/// Static configuration bundle for a GA run.
#[derive(Debug, Clone)]
pub struct GaSetup {
    /// Number of genes.
    pub dim: usize,
    /// Per-gene `[lo, hi]` bounds used for random init and mutation.
    pub bounds: Vec<(f64, f64)>,
    /// Crossover operator.
    pub crossover: CrossoverKind,
    /// Mutation operator.
    pub mutation: MutationKind,
    /// Probability of applying crossover to a parent pair.
    pub crossover_rate: f64,
    /// Probability of applying mutation to a child.
    pub mutation_rate: f64,
}

fn two_cuts(n: usize, rng: &mut dyn Rng) -> (usize, usize) {
    if n <= 1 {
        return (0, 0);
    }
    let a = rng.index(n);
    let b = rng.index(n);
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Single-point crossover.
pub fn single_point<T: Gene + Clone>(
    a: &[T],
    b: &[T],
    rng: &mut dyn Rng,
) -> (Vec<T>, Vec<T>) {
    let n = a.len().min(b.len());
    let c = if n <= 1 { 0 } else { rng.index(n) };
    let mut c1 = Vec::with_capacity(n);
    let mut c2 = Vec::with_capacity(n);
    for i in 0..n {
        if i < c {
            c1.push(a[i]);
            c2.push(b[i]);
        } else {
            c1.push(b[i]);
            c2.push(a[i]);
        }
    }
    (c1, c2)
}

/// Two-point crossover.
pub fn two_point<T: Gene + Clone>(a: &[T], b: &[T], rng: &mut dyn Rng) -> (Vec<T>, Vec<T>) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (a.to_vec(), b.to_vec());
    }
    let (s, e) = two_cuts(n, rng);
    let mut c1 = Vec::with_capacity(n);
    let mut c2 = Vec::with_capacity(n);
    for i in 0..n {
        if i >= s && i <= e {
            c1.push(a[i]);
            c2.push(b[i]);
        } else {
            c1.push(b[i]);
            c2.push(a[i]);
        }
    }
    (c1, c2)
}

/// Uniform crossover.
pub fn uniform<T: Gene + Clone>(a: &[T], b: &[T], rng: &mut dyn Rng) -> (Vec<T>, Vec<T>) {
    let n = a.len().min(b.len());
    let mut c1 = Vec::with_capacity(n);
    let mut c2 = Vec::with_capacity(n);
    for i in 0..n {
        if rng.next_f64() < 0.5 {
            c1.push(a[i]);
            c2.push(b[i]);
        } else {
            c1.push(b[i]);
            c2.push(a[i]);
        }
    }
    (c1, c2)
}

/// Position-based (order) crossover. Keeps a random subset of `a`'s positions,
/// fills the rest from `b` in order. Valid for both representations.
pub fn order_based<T: Gene + Clone>(
    a: &[T],
    b: &[T],
    rng: &mut dyn Rng,
    _n_exclusive: usize,
) -> (Vec<T>, Vec<T>) {
    let n = a.len().min(b.len());
    if n <= 1 {
        return (a.to_vec(), b.to_vec());
    }
    let (s, e) = two_cuts(n, rng);
    let mut c1 = vec![None; n];
    let mut c2 = vec![None; n];
    for i in s..=e {
        c1[i] = Some(a[i]);
    }
    for i in s..=e {
        c2[i] = Some(b[i]);
    }
    fill_remaining(&mut c1, b, &mut 0);
    fill_remaining(&mut c2, a, &mut 0);
    let c1 = c1.into_iter().map(|x| x.unwrap_or_else(|| a[0])).collect();
    let c2 = c2.into_iter().map(|x| x.unwrap_or_else(|| b[0])).collect();
    (c1, c2)
}

fn fill_remaining<T: Gene + Clone>(child: &mut [Option<T>], donor: &[T], bi: &mut usize) {
    let n = child.len().min(donor.len());
    for i in 0..n {
        if child[i].is_some() {
            continue;
        }
        while *bi < donor.len() {
            let v = donor[*bi];
            *bi += 1;
            if !child.iter().any(|x| x == &Some(v)) {
                child[i] = Some(v);
                break;
            }
        }
        if child[i].is_none() {
            // Fallback: reuse a donor value at this index (keeps validity).
            child[i] = Some(donor[i % donor.len()]);
        }
    }
}

/// Apply `kind` mutation to `genes` in place.
pub fn mutate<T: Gene + Clone>(
    kind: MutationKind,
    genes: &mut [T],
    rng: &mut dyn Rng,
    bounds: &[(f64, f64)],
) {
    let n = genes.len().min(bounds.len());
    if n == 0 {
        return;
    }
    match kind {
        MutationKind::BitFlip => {
            let i = rng.index(n);
            let (lo, hi) = bounds[i];
            genes[i] = genes[i].bit_flip(rng, lo, hi);
        }
        MutationKind::Flip => {
            let i = rng.index(n);
            let (lo, hi) = bounds[i];
            genes[i] = genes[i].flip(rng, lo, hi);
        }
        MutationKind::Swap => {
            if n >= 2 {
                let i = rng.index(n);
                let j = rng.index(n);
                genes.swap(i, j);
            }
        }
        MutationKind::Inversion => {
            let (s, e) = two_cuts(n, rng);
            genes[s..=e].reverse();
        }
        MutationKind::Scramble => {
            let (s, e) = two_cuts(n, rng);
            let mut seg: Vec<T> = genes[s..=e].to_vec();
            for k in (1..seg.len()).rev() {
                let j = rng.index(k + 1);
                seg.swap(k, j);
            }
            genes[s..=e].copy_from_slice(&seg);
        }
    }
}

/// Apply a crossover operator, returning two children.
pub fn crossover<T: Gene + Clone>(
    kind: CrossoverKind,
    a: &[T],
    b: &[T],
    rng: &mut dyn Rng,
    n_exclusive: usize,
) -> (Vec<T>, Vec<T>) {
    match kind {
        CrossoverKind::SinglePoint => single_point(a, b, rng),
        CrossoverKind::TwoPoint => two_point(a, b, rng),
        CrossoverKind::Uniform => uniform(a, b, rng),
        CrossoverKind::OrderBased => order_based(a, b, rng, n_exclusive),
    }
}

/// Select a parent index from `fitness` using `kind`.
///
/// Returns an index in `0..fitness.len()`. Fitness is assumed maximisation
/// (higher is better); GA converts raw objectives to fitness before calling.
pub fn select_index(fitness: &[f64], kind: SelectionKind, rng: &mut dyn Rng) -> usize {
    let n = fitness.len();
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 0;
    }
    match kind {
        SelectionKind::Tournament(k) => {
            let k = k.max(1).min(n);
            let mut best = rng.index(n);
            for _ in 1..k {
                let c = rng.index(n);
                if fitness[c] > fitness[best] {
                    best = c;
                }
            }
            best
        }
        SelectionKind::Roulette => {
            let min = fitness.iter().cloned().fold(f64::INFINITY, f64::min);
            let adj: Vec<f64> = fitness.iter().map(|&f| f - min + 1e-9).collect();
            let total: f64 = adj.iter().sum();
            let mut r = rng.next_f64() * total;
            for (i, w) in adj.iter().enumerate() {
                r -= *w;
                if r <= 0.0 {
                    return i;
                }
            }
            n - 1
        }
        SelectionKind::Rank => {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| {
                fitness[a]
                    .partial_cmp(&fitness[b])
                    .unwrap_or(Ordering::Equal)
            });
            let weights: Vec<f64> = idx.iter().enumerate().map(|(rank, _)| (rank + 1) as f64).collect();
            let total: f64 = weights.iter().sum();
            let mut r = rng.next_f64() * total;
            for (i, w) in weights.iter().enumerate() {
                r -= *w;
                if r <= 0.0 {
                    return idx[i];
                }
            }
            idx[n - 1]
        }
    }
}

/// Genetic algorithm over genome `G`.
pub struct GeneticAlgorithm<G: Genome> {
    objective: Box<dyn Fn(&G) -> f64>,
    sense: Sense,
    setup: GaSetup,
    population_size: usize,
    generations: usize,
    elite_count: usize,
    selection: SelectionKind,
    target: Option<f64>,
    seed: u64,
    rng: Xoshiro256,
    history: ConvergenceHistory,
}

impl<G: Genome> GeneticAlgorithm<G> {
    /// Build a GA with an objective closure `f(genome) -> raw_value`, a sense,
    /// dimension, and per-gene bounds.
    pub fn new(
        objective: impl Fn(&G) -> f64 + 'static,
        sense: Sense,
        dim: usize,
        bounds: Vec<(f64, f64)>,
    ) -> Self {
        Self {
            objective: Box::new(objective),
            sense,
            setup: GaSetup {
                dim,
                bounds,
                crossover: CrossoverKind::Uniform,
                mutation: MutationKind::Flip,
                crossover_rate: 0.9,
                mutation_rate: 0.1,
            },
            population_size: 50,
            generations: 100,
            elite_count: 2,
            selection: SelectionKind::Tournament(3),
            target: None,
            seed: 0,
            rng: Xoshiro256::new(0),
            history: ConvergenceHistory::new(),
        }
    }

    /// Set population size.
    pub fn population_size(mut self, n: usize) -> Self {
        self.population_size = n;
        self
    }

    /// Set number of generations.
    pub fn generations(mut self, n: usize) -> Self {
        self.generations = n;
        self
    }

    /// Set number of elite genomes carried unchanged each generation.
    pub fn elite_count(mut self, n: usize) -> Self {
        self.elite_count = n;
        self
    }

    /// Set the parent-selection operator.
    pub fn selection(mut self, kind: SelectionKind) -> Self {
        self.selection = kind;
        self
    }

    /// Set the crossover operator.
    pub fn crossover(mut self, kind: CrossoverKind) -> Self {
        self.setup.crossover = kind;
        self
    }

    /// Set the mutation operator.
    pub fn mutation(mut self, kind: MutationKind) -> Self {
        self.setup.mutation = kind;
        self
    }

    /// Set the crossover application probability.
    pub fn crossover_rate(mut self, rate: f64) -> Self {
        self.setup.crossover_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set the mutation application probability.
    pub fn mutation_rate(mut self, rate: f64) -> Self {
        self.setup.mutation_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Stop early (reporting [`SolverStatus::Optimal`](tpt_opt_core::SolverStatus::Optimal))
    /// once the incumbent reaches `target`.
    pub fn with_target(mut self, target: f64) -> Self {
        self.target = Some(target);
        self
    }

    /// Set the deterministic seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.rng = Xoshiro256::new(seed);
        self
    }

    /// Convergence history from the last [`solve`](Self::solve).
    pub fn history(&self) -> &ConvergenceHistory {
        &self.history
    }

    /// Run the genetic algorithm.
    pub fn solve(&mut self) -> Result<HeuristicResult, OptError> {
        if self.population_size == 0 {
            return Err(OptError::invalid_model("population size must be > 0"));
        }
        if self.generations == 0 {
            return Err(OptError::invalid_model("generations must be > 0"));
        }
        if self.setup.dim == 0 {
            return Err(OptError::invalid_model("genome dimension is zero"));
        }
        if self.setup.bounds.len() != self.setup.dim {
            return Err(OptError::invalid_model("bounds length must equal dimension"));
        }

        let better = |a: f64, b: f64| match self.sense {
            Sense::Minimize => a < b,
            Sense::Maximize => a > b,
        };

        let mut population: Vec<G> = (0..self.population_size)
            .map(|_| G::random(&mut self.rng, &self.setup))
            .collect();
        let mut best_genome: Option<G> = None;
        let mut best_val = match self.sense {
            Sense::Minimize => f64::INFINITY,
            Sense::Maximize => f64::NEG_INFINITY,
        };

        let mut history = ConvergenceHistory::new();

        for gen in 0..self.generations {
            let raw: Vec<f64> = population.iter().map(|g| (self.objective)(g)).collect();
            for (i, g) in population.iter().enumerate() {
                if better(raw[i], best_val) {
                    best_val = raw[i];
                    best_genome = Some(g.clone());
                }
            }
            let fitness = self.fitnesses(&raw);

            let mut next: Vec<G> = Vec::with_capacity(self.population_size);
            if self.elite_count > 0 {
                let mut order: Vec<usize> = (0..population.len()).collect();
                order.sort_by(|&a, &b| fitness[b].partial_cmp(&fitness[a]).unwrap_or(Ordering::Equal));
                for &i in order.iter().take(self.elite_count.min(population.len())) {
                    next.push(population[i].clone());
                }
            }
            while next.len() < self.population_size {
                let pa = select_index(&fitness, self.selection, &mut self.rng);
                let pb = select_index(&fitness, self.selection, &mut self.rng);
                let (mut c1, mut c2) =
                    G::crossover(&population[pa], &population[pb], &mut self.rng, &self.setup);
                c1.mutate(&mut self.rng, &self.setup);
                c2.mutate(&mut self.rng, &self.setup);
                next.push(c1);
                if next.len() < self.population_size {
                    next.push(c2);
                }
            }
            population = next;

            let gen_best = raw
                .iter()
                .copied()
                .fold(f64::NAN, |a, b| if a.is_nan() || better(b, a) { b } else { a });
            history.push(gen, best_val, gen_best);

            if let Some(t) = self.target {
                if better(best_val, t) || best_val == t {
                    break;
                }
            }
        }

        let best = best_genome.expect("population non-empty");
        let best_x = best.as_f64();
        let status = match (self.target, self.sense) {
            (Some(t), Sense::Minimize) if best_val <= t => tpt_opt_core::SolverStatus::Optimal,
            (Some(t), Sense::Maximize) if best_val >= t => tpt_opt_core::SolverStatus::Optimal,
            _ => tpt_opt_core::SolverStatus::TimeLimit,
        };
        Ok(HeuristicResult {
            best_x,
            best_value: best_val,
            status,
            iterations: self.generations,
            seed: self.seed,
            history,
        })
    }

    fn fitnesses(&self, raw: &[f64]) -> Vec<f64> {
        let mut f: Vec<f64> = raw
            .iter()
            .map(|&v| match self.sense {
                Sense::Minimize => -v,
                Sense::Maximize => v,
            })
            .collect();
        let min = f.iter().cloned().fold(f64::INFINITY, f64::min);
        for v in f.iter_mut() {
            *v = *v - min + 1e-9;
        }
        f
    }
}

impl GeneticAlgorithm<Vec<f64>> {
    /// Build a continuous GA from an [`Objective`](crate::Objective).
    pub fn for_objective(objective: impl Objective + 'static) -> Self {
        let dim = objective.dim();
        let bounds: Vec<(f64, f64)> = (0..dim).map(|i| objective.bound(i)).collect();
        let sense = objective.sense();
        GeneticAlgorithm::new(
            move |g: &Vec<f64>| objective.evaluate(g),
            sense,
            dim,
            bounds,
        )
    }
}

impl GeneticAlgorithm<Vec<usize>> {
    /// Build a permutation GA: genes are indices `0..n`, evaluated by `f`.
    pub fn for_permutation(
        n: usize,
        f: impl Fn(&Vec<usize>) -> f64 + 'static,
        sense: Sense,
    ) -> Self {
        let bounds = vec![(0.0, n as f64); n];
        GeneticAlgorithm::new(f, sense, n, bounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ObjectiveFn;

    #[test]
    fn crossover_operators_valid() {
        let a = vec![0usize, 1, 2, 3, 4];
        let b = vec![4usize, 3, 2, 1, 0];
        let mut rng = Xoshiro256::new(1);
        for kind in [
            CrossoverKind::SinglePoint,
            CrossoverKind::TwoPoint,
            CrossoverKind::Uniform,
            CrossoverKind::OrderBased,
        ] {
            let (c1, c2) = crossover(kind, &a, &b, &mut rng, 5);
            assert_eq!(c1.len(), 5);
            assert_eq!(c2.len(), 5);
            // order-based preserves gene multiset
            if kind == CrossoverKind::OrderBased {
                assert_eq!(c1.len(), a.len());
            }
        }
    }

    #[test]
    fn mutation_operators_valid() {
        let bounds = vec![(0.0, 1.0); 6];
        let mut rng = Xoshiro256::new(2);
        for kind in [
            MutationKind::BitFlip,
            MutationKind::Flip,
            MutationKind::Swap,
            MutationKind::Inversion,
            MutationKind::Scramble,
        ] {
            let mut genes = vec![0.0f64, 0.2, 0.4, 0.6, 0.8, 1.0];
            mutate(kind, &mut genes, &mut rng, &bounds);
            assert_eq!(genes.len(), 6);
            for g in &genes {
                assert!((*g >= -1e-9) && (*g <= 1.0 + 1e-9));
            }
        }
    }

    #[test]
    fn selection_indices_in_range() {
        let fitness = vec![1.0, 3.0, 2.0, 0.5, 4.0];
        let mut rng = Xoshiro256::new(3);
        for kind in [
            SelectionKind::Tournament(3),
            SelectionKind::Roulette,
            SelectionKind::Rank,
        ] {
            for _ in 0..200 {
                let i = select_index(&fitness, kind, &mut rng);
                assert!(i < fitness.len());
            }
        }
    }

    #[test]
    fn determinism_same_seed() {
        let obj = ObjectiveFn::minimize(4, |x| x.iter().map(|v| v * v).sum::<f64>(), [(-2.0, 2.0); 4]);
        let build = || {
            GeneticAlgorithm::for_objective(obj.clone())
                .population_size(40)
                .generations(60)
                .with_seed(77)
        };
        let a = build().solve().unwrap();
        let b = build().solve().unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn ga_improves_on_sphere() {
        let obj = ObjectiveFn::minimize(4, |x| x.iter().map(|v| v * v).sum::<f64>(), [(-2.0, 2.0); 4]);
        let mut ga = GeneticAlgorithm::for_objective(obj)
            .population_size(60)
            .generations(120)
            .crossover(CrossoverKind::Uniform)
            .mutation(MutationKind::Flip)
            .with_seed(5);
        let res = ga.solve().unwrap();
        assert!(res.best_value < 1.0, "got {}", res.best_value);
    }

    #[test]
    fn ga_onemax_permutation() {
        // Maximise the sum of a permutation of 0..n  => best is descending.
        let n = 8;
        let mut ga = GeneticAlgorithm::for_permutation(n, |p| p.iter().sum::<usize>() as f64, Sense::Maximize)
            .population_size(50)
            .generations(80)
            .crossover(CrossoverKind::OrderBased)
            .mutation(MutationKind::Swap)
            .with_seed(11);
        let res = ga.solve().unwrap();
        assert!(res.best_value >= (n * (n - 1) / 2) as f64 - 1e-9);
    }
}
