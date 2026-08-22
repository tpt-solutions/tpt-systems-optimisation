//! Two families working together: a seeded simulated-annealing run on a
//! non-convex function, then NSGA-II approximating the Pareto front of a
//! two-objective problem — both through the umbrella crate.

use tpt_opt_systems::{heuristic, multi, Sense};

fn main() {
    // --- simulated annealing -------------------------------------------------
    // Minimise the Rastrigin-style bowl f(x) = sum(x_i^2 + 0.5*cos(3*x_i)).
    let objective =
        heuristic::ObjectiveFn::minimize(4, |x| {
            x.iter().map(|v| v * v + 0.5 * (3.0 * v).cos()).sum::<f64>()
        }, [(-2.0, 2.0); 4]);
    let mut sa = heuristic::SimulatedAnnealing::new(objective)
        .with_seed(42)
        .with_iterations(20_000);
    let res = sa.solve().expect("SA should run");
    println!("SA best value:   {:.6} at x ~ {:?}", res.best_value, res.best_x);

    // --- NSGA-II Pareto front -------------------------------------------------
    // f1 = x^2, f2 = (1-x)^2 over [0,1]: the whole range is Pareto-optimal.
    let nsga = multi::Nsga2::new(vec![(0.0, 1.0)], |x| vec![x[0] * x[0], (1.0 - x[0]).powi(2)])
        .with_config(multi::Nsga2Config { population: 40, generations: 60, ..Default::default() })
        .with_seed(7);
    let mut front = nsga.pareto_front();
    front.sort_by(|a, b| a.0[0].partial_cmp(&b.0[0]).unwrap());
    println!("NSGA-II Pareto front ({} points):", front.len());
    for (x, f) in &front {
        println!("  x = {:.3}  ->  (f1, f2) = ({:.4}, {:.4})", x[0], f[0], f[1]);
    }
    assert!(!front.is_empty());
    let _ = Sense::Minimize; // sense plumbing exercised by both solvers above
}