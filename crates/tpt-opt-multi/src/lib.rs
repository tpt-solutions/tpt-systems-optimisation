//! Multi-objective / Pareto optimisation.
//!
//! Provides the building blocks for multi-objective optimisation:
//!
//! - Pareto dominance, Pareto-front extraction and the additive epsilon indicator
//!   ([`dominance`]).
//! - Objective normalisation for disparate scales ([`normalizer`]).
//! - Hypervolume computation (exact 2-D and the WFG algorithm for N-D)
//!   ([`hypervolume()`]).
//! - [`nsga2`] — a self-contained NSGA-II evolutionary multi-objective solver.
//! - [`nsga3`] — NSGA-III with Das–Dennis reference directions and niching
//!   selection for many-objective / preference-articulated search.
//! - [`decision`] — knee-point detection, trade-off ratios and deterministic
//!   k-means clustering over a front.
//! - Linear scalarisation helpers ([`scalarize`]) building a
//!   [`tpt_opt_core::Model`] solved with [`tpt_opt_milp::MilpSolver`].

pub mod decision;
pub mod dominance;
pub mod hypervolume;
pub mod normalizer;
pub mod nsga2;
pub mod nsga3;
pub mod scalarize;

pub use decision::{cluster_solutions, knee_point, tradeoff_ratios};
pub use dominance::{dominates, epsilon_indicator, pareto_front};
pub use hypervolume::hypervolume;
pub use normalizer::ObjectiveNormalizer;
pub use nsga2::{Nsga2, Nsga2Config};
pub use nsga3::{das_dennis, Nsga3, Nsga3Config};
pub use scalarize::{LinearMultiObjective, Scalarization};

#[cfg(test)]
mod nsga3_tests {
    use crate::nsga3::{das_dennis, Nsga3, Nsga3Config};

    #[test]
    fn das_dennis_counts_and_sums() {
        // M=3, p=4 -> C(6, 2) = 15 directions; each sums to 1.
        let dirs = das_dennis(4, 3);
        assert_eq!(dirs.len(), 15);
        for d in &dirs {
            assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(d.iter().all(|&c| c >= 0.0));
        }
        // M=2, p=2 -> (0,0.5),(0,1),(0.5,0),(1,0) lexicographic.
        let dirs2 = das_dennis(2, 2);
        assert_eq!(dirs2.len(), 3);
    }

    #[test]
    fn nsga3_finds_front_on_convex_problem() {
        // f1 = x^2, f2 = (1-x)^2 on [0,1]: Pareto front is the whole range;
        // the final population must be non-dominated and spread.
        let solver = Nsga3::new(vec![(0.0, 1.0)], |x| vec![x[0] * x[0], (1.0 - x[0]).powi(2)])
            .with_config(Nsga3Config {
                population: 40,
                generations: 60,
                divisions: 4,
                ..Default::default()
            })
            .with_seed(42);
        let front = solver.pareto_front();
        assert!(!front.is_empty());
        let xs: Vec<f64> = front.iter().map(|(x, _)| x[0]).collect();
        let spread = xs.iter().cloned().fold(f64::MIN, f64::max)
            - xs.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread > 0.5, "front should span most of [0,1], got {spread}");
    }

    #[test]
    fn nsga3_custom_directions_concentrate_search() {
        // Most reference directions favour low f1: niching keeps roughly one
        // solution per direction, so the population concentrates there.
        let dirs = vec![
            vec![0.95, 0.05],
            vec![0.9, 0.1],
            vec![0.85, 0.15],
            vec![0.8, 0.2],
            vec![0.75, 0.25],
            vec![0.7, 0.3],
            vec![0.65, 0.35],
            vec![0.6, 0.4],
            vec![0.2, 0.8],
            vec![0.05, 0.95],
        ];
        let solver = Nsga3::new(vec![(0.0, 1.0)], |x| vec![x[0], 1.0 - x[0]])
            .with_config(Nsga3Config { population: 20, generations: 50, ..Default::default() })
            .with_reference_directions(dirs)
            .with_seed(7);
        let pop = solver.solve();
        assert_eq!(pop.len(), 20);
        let mean_f1: f64 = pop.iter().map(|(_, f)| f[0]).sum::<f64>() / pop.len() as f64;
        assert!(mean_f1 < 0.55, "biased directions should favour low f1, got {mean_f1}");
    }

    #[test]
    fn nsga3_deterministic_same_seed() {
        let mk = || {
            Nsga3::new(vec![(0.0, 1.0), (0.0, 1.0)], |x| {
                vec![(x[0] - 0.3).powi(2) + x[1], (x[1] - 0.6).powi(2) + x[0]]
            })
            .with_config(Nsga3Config {
                population: 20,
                generations: 15,
                divisions: 3,
                ..Default::default()
            })
            .with_seed(123)
        };
        let a = mk().solve();
        let b = mk().solve();
        assert_eq!(a, b);
    }
}

#[cfg(test)]
mod decision_tests {
    use crate::decision::{cluster_solutions, knee_point, tradeoff_ratios};

    #[test]
    fn knee_point_picks_middle_of_convex_front() {
        // Front along f1 + f2 = 1: the knee is the balanced middle point.
        let objs: Vec<Vec<f64>> =
            (0..=10).map(|i| vec![i as f64 / 10.0, 1.0 - i as f64 / 10.0]).collect();
        let knee = knee_point(&objs).expect("non-empty");
        assert_eq!(knee, 5);
        assert!(knee_point(&[]).is_none());
    }

    #[test]
    fn tradeoff_ratios_envelope_semantics() {
        // Front: (0,1), (0.5,0.5), (1,0).
        let objs = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let t = tradeoff_ratios(&objs, 1); // middle point
                                           // T[0][1]: gain in f1 costs (max_f2 - f2)/gain = (1-0.5)/0.5 = 1.
        assert!((t[0][1] - 1.0).abs() < 1e-9);
        assert!((t[1][0] - 1.0).abs() < 1e-9);
        // At the extreme point (idx 0), no further f1 gain possible.
        let t0 = tradeoff_ratios(&objs, 0);
        assert_eq!(t0[0][1], 0.0);
    }

    #[test]
    fn clustering_separates_two_groups() {
        // Two well-separated groups of points.
        let mut objs: Vec<Vec<f64>> =
            (0..5).map(|i| vec![i as f64 * 0.01, i as f64 * 0.01]).collect();
        objs.extend((0..5).map(|i| vec![0.9 + i as f64 * 0.01, 0.9 + i as f64 * 0.01]));
        let assign = cluster_solutions(&objs, 2);
        let g0: Vec<usize> =
            assign.iter().enumerate().filter(|(_, &c)| c == 0).map(|(i, _)| i).collect();
        let g1: Vec<usize> =
            assign.iter().enumerate().filter(|(_, &c)| c == 1).map(|(i, _)| i).collect();
        // Each cluster must be one contiguous group (no mixing).
        let all_in_one = |g: &[usize]| g.iter().all(|&i| i < 5) || g.iter().all(|&i| i >= 5);
        assert!(all_in_one(&g0) && all_in_one(&g1), "clusters mixed: {assign:?}");
        // Deterministic across calls.
        assert_eq!(assign, cluster_solutions(&objs, 2));
    }
}
