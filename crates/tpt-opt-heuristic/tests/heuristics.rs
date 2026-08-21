//! Integration tests covering the spec's required test matrix.

use tpt_opt_heuristic::{
    CoolingSchedule, CrossoverKind, GeneticAlgorithm, MutationKind, ObjectiveFn, ParticleSwarmOptimization,
    SelectionKind, SimulatedAnnealing, TabuSearch, Topology, InertiaSchedule,
};

#[test]
fn determinism_same_seed_all_heuristics() {
    let sphere = || {
        ObjectiveFn::minimize(3, |x| x.iter().map(|v| v * v).sum::<f64>(), [(-3.0, 3.0); 3])
    };

    let sa_a = SimulatedAnnealing::new(sphere()).with_seed(1).with_iterations(300).solve().unwrap();
    let sa_b = SimulatedAnnealing::new(sphere()).with_seed(1).with_iterations(300).solve().unwrap();
    assert_eq!(sa_a, sa_b);

    let ga_a = GeneticAlgorithm::for_objective(sphere())
        .population_size(30)
        .generations(40)
        .with_seed(1)
        .solve()
        .unwrap();
    let ga_b = GeneticAlgorithm::for_objective(sphere())
        .population_size(30)
        .generations(40)
        .with_seed(1)
        .solve()
        .unwrap();
    assert_eq!(ga_a, ga_b);

    let ts_a = TabuSearch::new(sphere()).with_seed(1).with_iterations(300).solve().unwrap();
    let ts_b = TabuSearch::new(sphere()).with_seed(1).with_iterations(300).solve().unwrap();
    assert_eq!(ts_a, ts_b);

    let pso_a = ParticleSwarmOptimization::new(sphere())
        .with_seed(1)
        .with_iterations(300)
        .solve()
        .unwrap();
    let pso_b = ParticleSwarmOptimization::new(sphere())
        .with_seed(1)
        .with_iterations(300)
        .solve()
        .unwrap();
    assert_eq!(pso_a, pso_b);
}

#[test]
fn sa_minimizes_sphere() {
    let obj = ObjectiveFn::minimize(5, |x| x.iter().map(|v| v * v).sum::<f64>(), [(-5.0, 5.0); 5]);
    let mut sa = SimulatedAnnealing::new(obj)
        .with_seed(3)
        .with_cooling(CoolingSchedule::geometric(20.0, 0.97))
        .with_iterations(5000);
    let res = sa.solve().unwrap();
    assert!(res.best_value < 0.5, "got {}", res.best_value);
}

#[test]
fn ga_improves_on_simple_problem() {
    // Maximize sum of a 10-vector in [0,1]; optimum is 10.
    let obj = ObjectiveFn::maximize(10, |x| x.iter().sum::<f64>(), [(0.0, 1.0); 10]);
    let mut ga = GeneticAlgorithm::for_objective(obj)
        .population_size(50)
        .generations(100)
        .crossover(CrossoverKind::Uniform)
        .mutation(MutationKind::Flip)
        .with_target(9.9)
        .with_seed(4);
    let res = ga.solve().unwrap();
    assert!(res.best_value > 9.0, "got {}", res.best_value);
    assert_eq!(res.status, tpt_opt_core::SolverStatus::Optimal);
}

#[test]
fn ga_onemax_permutation() {
    let n = 10;
    let mut ga = GeneticAlgorithm::for_permutation(n, |p| p.iter().sum::<usize>() as f64, tpt_opt_core::Sense::Maximize)
        .population_size(60)
        .generations(100)
        .crossover(CrossoverKind::OrderBased)
        .mutation(MutationKind::Swap)
        .with_target((n * (n - 1) / 2) as f64)
        .with_seed(6);
    let res = ga.solve().unwrap();
    assert!(res.best_value >= (n * (n - 1) / 2) as f64 - 1e-9);
}

#[test]
fn tabu_reaches_optimum_small() {
    let obj = ObjectiveFn::minimize(
        2,
        |x| (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2),
        [(0.0, 6.0), (-4.0, 0.0)],
    );
    let mut ts = TabuSearch::new(obj).with_seed(7).with_iterations(2000).with_sample_size(30);
    let res = ts.solve().unwrap();
    assert!(res.best_value < 0.1, "got {}", res.best_value);
}

#[test]
fn pso_converges_sphere() {
    let obj = ObjectiveFn::minimize(4, |x| x.iter().map(|v| v * v).sum::<f64>(), [(-2.0, 2.0); 4]);
    let mut pso = ParticleSwarmOptimization::new(obj)
        .with_seed(8)
        .with_swarm_size(60)
        .with_iterations(1000)
        .with_inertia(InertiaSchedule::linear(0.9, 0.3))
        .with_topology(Topology::Ring);
    let res = pso.solve().unwrap();
    assert!(res.best_value < 0.1, "got {}", res.best_value);
}

#[test]
fn operator_unit_validity() {
    use tpt_opt_heuristic::{crossover, mutate, select_index, Gene};
    let mut rng = tpt_opt_heuristic::rng_from_seed(11);
    let bounds = vec![(0.0, 1.0); 8];
    let a: Vec<f64> = (0..8).map(|_| 0.5).collect();
    let b: Vec<f64> = (0..8).map(|_| 0.1).collect();
    for kind in [
        CrossoverKind::SinglePoint,
        CrossoverKind::TwoPoint,
        CrossoverKind::Uniform,
    ] {
        let (c1, c2) = crossover(kind, &a, &b, &mut rng, 8);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);
    }
    for m in [
        MutationKind::BitFlip,
        MutationKind::Flip,
        MutationKind::Swap,
        MutationKind::Inversion,
        MutationKind::Scramble,
    ] {
        let mut g = a.clone();
        mutate(m, &mut g, &mut rng, &bounds);
        assert_eq!(g.len(), 8);
    }
    let fitness = vec![0.2, 0.9, 0.4, 0.1, 0.7];
    for s in [
        SelectionKind::Tournament(3),
        SelectionKind::Roulette,
        SelectionKind::Rank,
    ] {
        for _ in 0..100 {
            let i = select_index(&fitness, s, &mut rng);
            assert!(i < fitness.len());
        }
    }
    // Genome trait is exercised by GA; ensure gene conversion works.
    let _: f64 = 0.0f64.to_f64();
}
