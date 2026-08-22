//! NSGA-II — a fast elitist non-dominated sorting genetic algorithm for
//! multi-objective optimisation (Deb et al., 2002).
//!
//! Self-contained: it only needs an objective function and per-variable bounds.
//! All randomness flows through a seedable [`tpt_math_prob::Xoshiro256`] so that
//! runs with equal seeds are reproducible (spec §4).

use std::vec::Vec;

use tpt_math_prob::{Rng, Xoshiro256};

/// Configuration for [`Nsga2`].
#[derive(Debug, Clone)]
pub struct Nsga2Config {
    /// Population size (must be even; rounded up if needed).
    pub population: usize,
    /// Number of generations.
    pub generations: usize,
    /// Simulated-binary-crossover distribution index `η_c`.
    pub eta_c: f64,
    /// Polynomial-mutation distribution index `η_m`.
    pub eta_m: f64,
    /// Crossover probability.
    pub crossover_prob: f64,
    /// Mutation probability (per variable).
    pub mutation_prob: f64,
    /// Deterministic seed.
    pub seed: u64,
}

impl Default for Nsga2Config {
    fn default() -> Self {
        Self {
            population: 50,
            generations: 100,
            eta_c: 15.0,
            eta_m: 20.0,
            crossover_prob: 0.9,
            mutation_prob: 0.1,
            seed: 0,
        }
    }
}

#[derive(Clone)]
struct Individual {
    x: Vec<f64>,
    f: Vec<f64>,
    rank: usize,
    crowding: f64,
}

/// NSGA-II optimiser over continuous decision vectors.
pub struct Nsga2 {
    bounds: Vec<(f64, f64)>,
    #[allow(clippy::type_complexity)]
    objective: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    config: Nsga2Config,
}

impl Nsga2 {
    /// Create a solver for `n` decision variables with the given `bounds` and
    /// objective function (returns one value per objective; lower is better).
    pub fn new<F>(bounds: Vec<(f64, f64)>, objective: F) -> Self
    where
        F: Fn(&[f64]) -> Vec<f64> + 'static,
    {
        Self { bounds, objective: Box::new(objective), config: Nsga2Config::default() }
    }

    /// Override the configuration.
    pub fn with_config(mut self, config: Nsga2Config) -> Self {
        self.config = config;
        self
    }

    /// Override the deterministic seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Run the algorithm and return the final population's individuals as
    /// `(decision_vector, objective_vector)` pairs.
    pub fn solve(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        let mut rng = Xoshiro256::new(self.config.seed);
        let pop = self.config.population.max(4) | 1; // ensure odd? keep even-friendly
        let pop = pop + (pop % 2);

        let mut individuals: Vec<Individual> = (0..pop)
            .map(|_| {
                let x: Vec<f64> = self.bounds.iter().map(|&(lo, hi)| rng.range(lo, hi)).collect();
                let f = (self.objective)(&x);
                Individual { x, f, rank: 0, crowding: 0.0 }
            })
            .collect();

        self.assign_fronts(&mut individuals);

        for _gen in 0..self.config.generations {
            let mut offspring = self.make_offspring(&individuals, &mut rng);
            for ind in &mut offspring {
                ind.f = (self.objective)(&ind.x);
            }
            let mut combined = individuals.clone();
            combined.append(&mut offspring);
            self.assign_fronts(&mut combined);

            individuals = self.select_next(&combined, pop);
        }

        individuals.into_iter().map(|ind| (ind.x, ind.f)).collect()
    }

    /// Return only the non-dominated (Pareto) solutions from the final population.
    pub fn pareto_front(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        let all = self.solve();
        let objs: Vec<Vec<f64>> = all.iter().map(|(_, f)| f.clone()).collect();
        crate::dominance::pareto_front(&objs).into_iter().map(|i| all[i].clone()).collect()
    }

    fn make_offspring(&self, pop: &[Individual], rng: &mut Xoshiro256) -> Vec<Individual> {
        let mut children = Vec::with_capacity(pop.len());
        while children.len() < pop.len() {
            let p1 = self.tournament(pop, rng);
            let p2 = self.tournament(pop, rng);
            let (mut c1, mut c2) = if rng.next_f64() < self.config.crossover_prob {
                self.sbx(&p1.x, &p2.x, rng)
            } else {
                (p1.x.clone(), p2.x.clone())
            };
            self.mutate(&mut c1, rng);
            self.mutate(&mut c2, rng);
            children.push(Individual { x: c1, f: Vec::new(), rank: 0, crowding: 0.0 });
            if children.len() < pop.len() {
                children.push(Individual { x: c2, f: Vec::new(), rank: 0, crowding: 0.0 });
            }
        }
        children
    }

    fn tournament<'a>(&self, pop: &'a [Individual], rng: &mut Xoshiro256) -> &'a Individual {
        let a = &pop[rng.index(pop.len())];
        let b = &pop[rng.index(pop.len())];
        better(a, b)
    }

    fn sbx(&self, pa: &[f64], pb: &[f64], rng: &mut Xoshiro256) -> (Vec<f64>, Vec<f64>) {
        let eta = self.config.eta_c;
        let mut c1 = vec![0.0f64; pa.len()];
        let mut c2 = vec![0.0f64; pa.len()];
        for i in 0..pa.len() {
            let (lo, hi) = self.bounds[i];
            let (mut u, mut v) = (pa[i], pb[i]);
            if (u - v).abs() < 1e-12 {
                c1[i] = u;
                c2[i] = v;
                continue;
            }
            if u < v {
                std::mem::swap(&mut u, &mut v);
            }
            let beta = u / v;
            let r = rng.next_f64();
            let beta_q = if r <= 0.5 {
                (2.0 * r).powf(1.0 / (eta + 1.0))
            } else {
                (1.0 / (2.0 * (1.0 - r))).powf(1.0 / (eta + 1.0))
            };
            let _ = beta;
            let (new_u, new_v) =
                (0.5 * ((u + v) - beta_q * (u - v)), 0.5 * ((u + v) + beta_q * (u - v)));
            c1[i] = new_u.clamp(lo, hi);
            c2[i] = new_v.clamp(lo, hi);
        }
        (c1, c2)
    }

    fn mutate(&self, x: &mut [f64], rng: &mut Xoshiro256) {
        let eta = self.config.eta_m;
        for (xi, b) in x.iter_mut().zip(self.bounds.iter()) {
            if rng.next_f64() < self.config.mutation_prob {
                let (lo, hi) = *b;
                let span = hi - lo;
                let r = rng.next_f64();
                let delta = if r < 0.5 {
                    (2.0 * r).powf(1.0 / (eta + 1.0)) - 1.0
                } else {
                    1.0 - (2.0 * (1.0 - r)).powf(1.0 / (eta + 1.0))
                };
                *xi = (*xi + delta * span).clamp(lo, hi);
            }
        }
    }

    fn assign_fronts(&self, pop: &mut [Individual]) {
        let n = pop.len();
        let mut dominated: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut dom_count = vec![0usize; n];
        let mut fronts: Vec<Vec<usize>> = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if crate::dominates(&pop[i].f, &pop[j].f) {
                    dominated[i].push(j);
                } else if crate::dominates(&pop[j].f, &pop[i].f) {
                    dom_count[i] += 1;
                }
            }
            if dom_count[i] == 0 {
                pop[i].rank = 0;
                if fronts.is_empty() {
                    fronts.push(Vec::new());
                }
                fronts[0].push(i);
            }
        }

        let mut fi = 0;
        while !fronts[fi].is_empty() {
            let mut next: Vec<usize> = Vec::new();
            for &i in &fronts[fi] {
                for &j in &dominated[i] {
                    dom_count[j] -= 1;
                    if dom_count[j] == 0 {
                        pop[j].rank = fi + 1;
                        next.push(j);
                    }
                }
            }
            fi += 1;
            fronts.push(next);
        }
        fronts.pop(); // last empty

        for front in &fronts {
            self.crowding_distance(front, pop);
        }
    }

    fn crowding_distance(&self, front: &[usize], pop: &mut [Individual]) {
        let m = pop[front[0]].f.len();
        for &i in front {
            pop[i].crowding = 0.0;
        }
        for k in 0..m {
            let mut order: Vec<usize> = front.to_vec();
            order.sort_by(|&a, &b| {
                pop[a].f[k].partial_cmp(&pop[b].f[k]).unwrap_or(std::cmp::Ordering::Equal)
            });
            pop[order[0]].crowding = f64::INFINITY;
            pop[*order.last().unwrap()].crowding = f64::INFINITY;
            let fmin = pop[order[0]].f[k];
            let fmax = pop[order[order.len() - 1]].f[k];
            let span = if fmax - fmin > 1e-12 { fmax - fmin } else { 1.0 };
            for w in 1..order.len() - 1 {
                let prev = pop[order[w - 1]].f[k];
                let next = pop[order[w + 1]].f[k];
                pop[order[w]].crowding += (next - prev) / span;
            }
        }
    }

    fn select_next(&self, combined: &[Individual], pop: usize) -> Vec<Individual> {
        let mut result = Vec::with_capacity(pop);
        let mut front_idx = 0;
        let fronts = self.fronts_of(combined);
        while result.len() + fronts[front_idx].len() <= pop {
            for &i in &fronts[front_idx] {
                result.push(combined[i].clone());
            }
            front_idx += 1;
            if front_idx >= fronts.len() {
                break;
            }
        }
        if result.len() < pop && front_idx < fronts.len() {
            let mut last = fronts[front_idx].clone();
            last.sort_by(|&a, &b| {
                combined[b]
                    .crowding
                    .partial_cmp(&combined[a].crowding)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &i in &last {
                if result.len() >= pop {
                    break;
                }
                result.push(combined[i].clone());
            }
        }
        result
    }

    fn fronts_of(&self, pop: &[Individual]) -> Vec<Vec<usize>> {
        let max_rank = pop.iter().map(|i| i.rank).max().unwrap_or(0);
        let mut fronts: Vec<Vec<usize>> = vec![Vec::new(); max_rank + 1];
        for (i, ind) in pop.iter().enumerate() {
            fronts[ind.rank].push(i);
        }
        fronts
    }
}

fn better<'a>(a: &'a Individual, b: &'a Individual) -> &'a Individual {
    if a.rank != b.rank {
        if a.rank < b.rank {
            a
        } else {
            b
        }
    } else if a.crowding > b.crowding {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_pareto_front_of_zdt1_like() {
        // f1 = x, f2 = 1 - sqrt(f1) on x in [0,1]: convex front.
        let solver = Nsga2::new(vec![(0.0, 1.0)], |x| {
            let f1 = x[0];
            let f2 = 1.0 - f1.sqrt();
            vec![f1, f2]
        })
        .with_config(Nsga2Config {
            population: 40,
            generations: 60,
            seed: 1,
            ..Default::default()
        });
        let front = solver.pareto_front();
        assert!(front.len() >= 5, "expected a spread of Pareto points");
        // All returned points should be mutually non-dominated.
        for i in 0..front.len() {
            for j in 0..front.len() {
                if i != j {
                    assert!(!crate::dominates(&front[i].1, &front[j].1));
                }
            }
        }
    }
}
