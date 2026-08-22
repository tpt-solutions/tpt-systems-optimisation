//! NSGA-III — many-objective optimisation with reference-point preference
//! articulation (Deb & Jain, 2014).
//!
//! Structured [`das_dennis`] reference directions on the unit simplex let the
//! caller articulate *where* on the Pareto front to concentrate the search:
//! more divisions → a denser, more uniform front; fewer divisions (or a
//! custom direction set via [`Nsga3::with_reference_directions`]) → a sparser
//! front concentrated along the supplied directions. Selection combines fast
//! non-dominated sorting with **niching**: solutions filling under-populated
//! reference niches are preferred, which maintains diversity in 3+ objectives
//! where crowding distance degenerates.
//!
//! Self-contained like [`crate::nsga2`]: an objective function plus per-
//! variable bounds; all randomness flows through a seedable
//! [`tpt_math_prob::Xoshiro256`] for reproducibility (spec §4).

use std::vec::Vec;

use tpt_math_prob::{Rng, Xoshiro256};

/// Configuration for [`Nsga3`].
#[derive(Debug, Clone)]
pub struct Nsga3Config {
    /// Population size.
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
    /// Das–Dennis simplex divisions per axis (`p` in Deb & Jain). The
    /// resulting direction count is C(M + p − 1, p − 1) for M objectives.
    pub divisions: usize,
    /// Deterministic seed.
    pub seed: u64,
}

impl Default for Nsga3Config {
    fn default() -> Self {
        Self {
            population: 60,
            generations: 120,
            eta_c: 15.0,
            eta_m: 20.0,
            crossover_prob: 0.9,
            mutation_prob: 0.1,
            divisions: 4,
            seed: 0,
        }
    }
}

/// Generate Das–Dennis reference directions on the unit simplex: all points
/// with non-negative coordinates summing to 1 on the lattice `1/p`.
///
/// For M objectives and p divisions this yields C(M + p − 1, p − 1)
/// directions, deterministically ordered lexicographically.
pub fn das_dennis(divisions: usize, m: usize) -> Vec<Vec<f64>> {
    assert!(m >= 2, "need at least two objectives");
    assert!(divisions >= 1, "need at least one division");
    let mut out = Vec::new();
    let mut cur = vec![0.0f64; m];
    fn recurse(
        idx: usize,
        remaining: usize,
        m: usize,
        cur: &mut Vec<f64>,
        out: &mut Vec<Vec<f64>>,
    ) {
        if idx + 1 == m {
            cur[idx] = remaining as f64;
            out.push(cur.clone());
            return;
        }
        for k in 0..=remaining {
            cur[idx] = k as f64;
            recurse(idx + 1, remaining - k, m, cur, out);
        }
    }
    recurse(0, divisions, m, &mut cur, &mut out);
    for dir in &mut out {
        for c in dir.iter_mut() {
            *c /= divisions as f64;
        }
    }
    out
}

#[derive(Clone)]
struct Individual {
    x: Vec<f64>,
    f: Vec<f64>,
    rank: usize,
}

/// NSGA-III optimiser over continuous decision vectors.
pub struct Nsga3 {
    bounds: Vec<(f64, f64)>,
    #[allow(clippy::type_complexity)]
    objective: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    config: Nsga3Config,
    /// Optional caller-supplied reference directions (overrides `divisions`).
    custom_directions: Option<Vec<Vec<f64>>>,
}

impl Nsga3 {
    /// Create a solver for `n` decision variables with the given `bounds`
    /// and objective function (one value per objective; lower is better).
    pub fn new<F>(bounds: Vec<(f64, f64)>, objective: F) -> Self
    where
        F: Fn(&[f64]) -> Vec<f64> + 'static,
    {
        Self {
            bounds,
            objective: Box::new(objective),
            config: Nsga3Config::default(),
            custom_directions: None,
        }
    }

    /// Override the configuration.
    pub fn with_config(mut self, config: Nsga3Config) -> Self {
        self.config = config;
        self
    }

    /// Override the deterministic seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Supply explicit unit-simplex reference directions instead of the
    /// structured Das–Dennis lattice — the mechanism for concentrating the
    /// search in preferred regions of the front.
    ///
    /// Directions need not be normalised; they are rescaled to sum to 1.
    pub fn with_reference_directions(mut self, dirs: Vec<Vec<f64>>) -> Self {
        assert!(!dirs.is_empty(), "at least one reference direction required");
        let m = dirs[0].len();
        assert!(dirs.iter().all(|d| d.len() == m), "direction dimension mismatch");
        let mut norm: Vec<Vec<f64>> = dirs
            .into_iter()
            .map(|d| {
                let s: f64 = d.iter().sum();
                assert!(s > 0.0, "reference direction must have positive mass");
                d.into_iter().map(|c| c / s).collect()
            })
            .collect();
        norm.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.custom_directions = Some(norm);
        self
    }

    /// Run the algorithm; returns the final population as
    /// `(decision_vector, objective_vector)` pairs.
    pub fn solve(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        let mut rng = Xoshiro256::new(self.config.seed);
        let pop = self.config.population.max(4);
        let m = (self.objective)(&vec![0.0; self.bounds.len()]).len().max(2);

        let mut individuals: Vec<Individual> = (0..pop)
            .map(|_| {
                let x: Vec<f64> = self.bounds.iter().map(|&(lo, hi)| rng.range(lo, hi)).collect();
                let f = (self.objective)(&x);
                Individual { x, f, rank: 0 }
            })
            .collect();

        for _gen in 0..self.config.generations {
            let mut offspring = self.make_offspring(&individuals, &mut rng);
            for ind in &mut offspring {
                ind.f = (self.objective)(&ind.x);
            }
            let mut combined = individuals.clone();
            combined.append(&mut offspring);
            individuals = self.environmental_select(&mut combined, pop, m);
        }

        individuals.into_iter().map(|ind| (ind.x, ind.f)).collect()
    }

    /// Non-dominated solutions of the final population.
    pub fn pareto_front(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        let all = self.solve();
        let objs: Vec<Vec<f64>> = all.iter().map(|(_, f)| f.clone()).collect();
        crate::dominance::pareto_front(&objs).into_iter().map(|i| all[i].clone()).collect()
    }

    /// Environmental selection: non-dominated sorting, then niching on the
    /// boundary front using reference directions.
    fn environmental_select(
        &self,
        combined: &mut [Individual],
        pop: usize,
        m: usize,
    ) -> Vec<Individual> {
        assign_fronts(combined);
        let fronts = group_by_rank(combined);

        let mut chosen: Vec<usize> = Vec::new();
        for front in &fronts {
            if chosen.len() + front.len() <= pop {
                chosen.extend(front.iter().copied());
            } else {
                // Niching fill of the last (boundary) front.
                let dirs = match &self.custom_directions {
                    Some(d) => d.clone(),
                    None => das_dennis(self.config.divisions, m),
                };
                let remaining = pop - chosen.len();
                let candidates: Vec<usize> =
                    front.iter().copied().filter(|i| !chosen.contains(i)).collect();
                let picked = niche_fill(combined, &chosen, &candidates, &dirs, remaining);
                chosen.extend(picked);
            }
            if chosen.len() >= pop {
                break;
            }
        }
        chosen.into_iter().map(|i| combined[i].clone()).collect()
    }

    fn make_offspring(&self, inds: &[Individual], rng: &mut Xoshiro256) -> Vec<Individual> {
        let mut offspring = Vec::with_capacity(inds.len());
        while offspring.len() < inds.len() {
            let p1 = tournament(inds, rng);
            let p2 = tournament(inds, rng);
            let mut c1 = p1.x.clone();
            let mut c2 = p2.x.clone();
            if rng.next_f64() < self.config.crossover_prob {
                sbx(&mut c1, &mut c2, self.config.eta_c, &self.bounds, rng);
            }
            for child in [&mut c1, &mut c2] {
                mutate(child, self.config.eta_m, self.config.mutation_prob, &self.bounds, rng);
            }
            offspring.push(Individual { x: c1, f: Vec::new(), rank: 0 });
            if offspring.len() < inds.len() {
                offspring.push(Individual { x: c2, f: Vec::new(), rank: 0 });
            }
        }
        offspring
    }
}

/// Fast non-dominated sorting over `inds` (sets `rank`).
fn assign_fronts(inds: &mut [Individual]) {
    let n = inds.len();
    let objs: Vec<&Vec<f64>> = inds.iter().map(|i| &i.f).collect();
    let mut domination_count = vec![0usize; n];
    let mut dominated: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (objs[i], objs[j]);
            let i_dom_j = a.iter().zip(b.iter()).all(|(x, y)| x <= y)
                && a.iter().zip(b.iter()).any(|(x, y)| x < y);
            let j_dom_i = b.iter().zip(a.iter()).all(|(x, y)| x <= y)
                && b.iter().zip(a.iter()).any(|(x, y)| x < y);
            if i_dom_j {
                dominated[i].push(j);
                domination_count[j] += 1;
            } else if j_dom_i {
                dominated[j].push(i);
                domination_count[i] += 1;
            }
        }
    }
    let mut current: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();
    let mut rank = 0usize;
    while !current.is_empty() {
        let mut next = Vec::new();
        for &i in &current {
            inds[i].rank = rank;
            for &j in &dominated[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next.push(j);
                }
            }
        }
        current = next;
        rank += 1;
    }
}

/// Group indices by assigned rank (front order preserved).
fn group_by_rank(inds: &[Individual]) -> Vec<Vec<usize>> {
    let max_rank = inds.iter().map(|i| i.rank).max().unwrap_or(0);
    let mut groups = vec![Vec::new(); max_rank + 1];
    for (i, ind) in inds.iter().enumerate() {
        groups[ind.rank].push(i);
    }
    groups.retain(|g| !g.is_empty());
    groups
}

/// Binary tournament by rank (ties broken by index — deterministic).
fn tournament<'a>(inds: &'a [Individual], rng: &mut Xoshiro256) -> &'a Individual {
    let a = (rng.next_u64() as usize) % inds.len();
    let b = (rng.next_u64() as usize) % inds.len();
    let (ia, ib) = (&inds[a], &inds[b]);
    if ia.rank != ib.rank {
        if ia.rank < ib.rank {
            ia
        } else {
            ib
        }
    } else if a <= b {
        ia
    } else {
        ib
    }
}

/// Simulated binary crossover (SBX), in place.
fn sbx(c1: &mut [f64], c2: &mut [f64], eta: f64, bounds: &[(f64, f64)], rng: &mut Xoshiro256) {
    for k in 0..c1.len() {
        if rng.next_f64() > 0.5 {
            continue;
        }
        let (lo, hi) = bounds[k];
        let (x1, x2) = (c1[k], c2[k]);
        if (x2 - x1).abs() < 1e-12 {
            continue;
        }
        let u = rng.range(0.0, 1.0);
        let beta = if u <= 0.5 {
            (2.0 * u).powf(1.0 / (eta + 1.0))
        } else {
            (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
        };
        let v1 = 0.5 * ((1.0 + beta) * x1 + (1.0 - beta) * x2);
        let v2 = 0.5 * ((1.0 - beta) * x1 + (1.0 + beta) * x2);
        c1[k] = v1.clamp(lo, hi);
        c2[k] = v2.clamp(lo, hi);
    }
}

/// Polynomial mutation, in place.
fn mutate(x: &mut [f64], eta: f64, prob: f64, bounds: &[(f64, f64)], rng: &mut Xoshiro256) {
    for (k, v) in x.iter_mut().enumerate() {
        if rng.next_f64() > prob {
            continue;
        }
        let (lo, hi) = bounds[k];
        let u = rng.range(0.0, 1.0);
        let delta = if u < 0.5 {
            (2.0 * u).powf(1.0 / (eta + 1.0)) - 1.0
        } else {
            1.0 - (2.0 * (1.0 - u)).powf(1.0 / (eta + 1.0))
        };
        *v = (*v + delta * (hi - lo)).clamp(lo, hi);
    }
}

/// Normalise objectives into the reference space: subtract the ideal point
/// and divide by axis intercepts estimated from the extreme (axis-optimal)
/// points via ASF weighting; falls back to the population max when the
/// intercept hyperplane is degenerate.
fn normalize(objs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = objs.first().map(|o| o.len()).unwrap_or(0);
    if m == 0 || objs.is_empty() {
        return Vec::new();
    }
    let ideal: Vec<f64> =
        (0..m).map(|j| objs.iter().map(|o| o[j]).fold(f64::INFINITY, f64::min)).collect();
    // Extreme points: minimise the achievement scalarising function with a
    // heavy weight on axis j.
    let mut extremes: Vec<Vec<f64>> = Vec::with_capacity(m);
    for j in 0..m {
        let mut best = 0usize;
        let mut best_asf = f64::INFINITY;
        for (i, o) in objs.iter().enumerate() {
            let asf = (0..m)
                .map(|k| {
                    let w = if k == j { 1e6 } else { 1.0 };
                    w * (o[k] - ideal[k])
                })
                .fold(f64::NEG_INFINITY, f64::max);
            if asf < best_asf {
                best_asf = asf;
                best = i;
            }
        }
        extremes.push(objs[best].clone());
    }
    // Intercepts from the hyperplane through the extremes (Gaussian
    // elimination on the M×M system); fall back to max-per-axis.
    let mut intercepts = solve_intercepts(&extremes, &ideal).unwrap_or_else(|| {
        (0..m)
            .map(|j| objs.iter().map(|o| o[j]).fold(f64::NEG_INFINITY, f64::max) - ideal[j])
            .collect::<Vec<f64>>()
    });
    for a in intercepts.iter_mut() {
        if !a.is_finite() || *a <= 1e-12 {
            *a = 1.0;
        }
    }
    objs.iter().map(|o| (0..m).map(|j| (o[j] - ideal[j]) / intercepts[j]).collect()).collect()
}

/// Solve the M×M linear system `E · a = 1` (intercepts of the hyperplane
/// through extreme rows E); `None` if singular.
fn solve_intercepts(extremes: &[Vec<f64>], ideal: &[f64]) -> Option<Vec<f64>> {
    let m = extremes.len();
    let mut a = vec![vec![0.0f64; m]; m];
    for (i, e) in extremes.iter().enumerate() {
        for j in 0..m {
            a[i][j] = e[j] - ideal[j];
        }
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..m {
        let pivot = (col..m)
            .fold(col, |best, r| if a[r][col].abs() > a[best][col].abs() { r } else { best });
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(pivot, col);
        for r in 0..m {
            if r != col && a[r][col].abs() > 1e-15 {
                let factor = a[r][col] / a[col][col];
                for c in col..m {
                    a[r][c] -= factor * a[col][c];
                }
            }
        }
    }
    let mut x = vec![0.0f64; m];
    for i in 0..m {
        x[i] = 1.0 / a[i][i];
    }
    Some(x)
}

/// Perpendicular distance from point `p` to the ray spanned by unit-ish
/// direction `w` (through the origin).
fn perpendicular_distance(p: &[f64], w: &[f64]) -> f64 {
    let dot: f64 = p.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
    let wn: f64 = w.iter().map(|c| c * c).sum::<f64>().sqrt();
    let proj = if wn > 0.0 { dot / wn } else { 0.0 };
    p.iter().zip(w.iter()).map(|(a, b)| a - proj * b / wn).map(|d| d * d).sum::<f64>().sqrt()
}

/// Associate each candidate with its nearest reference direction; then pick
/// `remaining` solutions preferring under-populated niches (deterministic:
/// smallest perpendicular distance, ties by candidate index).
fn niche_fill(
    inds: &[Individual],
    already_chosen: &[usize],
    candidates: &[usize],
    dirs: &[Vec<f64>],
    remaining: usize,
) -> Vec<usize> {
    let normalized = normalize(
        &already_chosen
            .iter()
            .chain(candidates.iter())
            .map(|&i| inds[i].f.clone())
            .collect::<Vec<_>>(),
    );
    let offset = already_chosen.len();
    // Niche counts include the already-chosen members.
    let mut counts = vec![0usize; dirs.len()];
    let mut assoc: Vec<(usize, f64)> = Vec::with_capacity(candidates.len()); // (niche, dist)
    for (k, _) in candidates.iter().enumerate() {
        let p = &normalized[offset + k];
        let (niche, dist) = dirs
            .iter()
            .enumerate()
            .map(|(ni, w)| (ni, perpendicular_distance(p, w)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, f64::INFINITY));
        counts[niche] += 1;
        assoc.push((niche, dist));
    }
    for &idx in already_chosen {
        let p = &normalized[already_chosen.iter().position(|&i| i == idx).unwrap_or(0)];
        let (niche, _) = dirs
            .iter()
            .enumerate()
            .map(|(ni, w)| (ni, perpendicular_distance(p, w)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, f64::INFINITY));
        counts[niche] += 1;
    }

    let mut picked = Vec::with_capacity(remaining);
    let mut taken = vec![false; candidates.len()];
    // Iterate niche occupancy levels 1, 2, ... picking one solution per
    // eligible niche until the slot budget is exhausted.
    'levels: for level in 1..=candidates.len() + 1 {
        for (ni, _) in dirs.iter().enumerate() {
            if picked.len() >= remaining {
                break 'levels;
            }
            if counts[ni] != level {
                continue;
            }
            // Best unpicked candidate in this niche.
            let mut best: Option<(usize, f64)> = None;
            for (k, &(niche, dist)) in assoc.iter().enumerate() {
                if niche == ni && !taken[k] && best.map_or(true, |(_, bd)| dist < bd) {
                    best = Some((k, dist));
                }
            }
            if let Some((k, _)) = best {
                taken[k] = true;
                counts[ni] += 1;
                picked.push(candidates[k]);
            }
        }
        if taken.iter().all(|&t| t) {
            break;
        }
    }
    // Budget not met (degenerate association): fill by index order.
    for (k, _) in assoc.iter().enumerate() {
        if picked.len() >= remaining {
            break;
        }
        if !taken[k] {
            taken[k] = true;
            picked.push(candidates[k]);
        }
    }
    picked
}
