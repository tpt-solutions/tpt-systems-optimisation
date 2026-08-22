//! 0/1 knapsack through the umbrella's `MilpBuilder`.
//!
//! max 3x + 5y + 4z + 2w  s.t. 4x + 7y + 5z + 3w <= 10, all binary.
//! Two optimal selections exist, both with value 7: {x, z} (weight 9) and
//! {y, w} (weight 10). The solver returns one of them.

use tpt_opt_systems::MilpBuilder;

fn main() {
    let values = [3.0, 5.0, 4.0, 2.0];
    let weights = [4.0, 7.0, 5.0, 3.0];
    let capacity = 10.0;

    let mut b = MilpBuilder::new(0);
    let vars: Vec<usize> = (0..4).map(|_| b.add_binary()).collect();
    let sol = b
        .le(&vars, &weights, capacity)
        .maximize(&vars, &values)
        .solve()
        .expect("knapsack should solve");

    // Verify feasibility independently.
    let total_weight: f64 =
        vars.iter().zip(&weights).map(|(&i, w)| sol.primal[i] * w).sum();
    assert!(total_weight <= capacity + 1e-6);

    let chosen: Vec<usize> =
        vars.iter().enumerate().filter(|&(i, _)| sol.primal[i] > 0.5).map(|(i, _)| i).collect();
    println!("selected items: {chosen:?}");
    println!("total weight:   {total_weight}");
    println!("objective:      {}", sol.objective_value);
    assert!((sol.objective_value - 7.0).abs() < 1e-6, "optimum is 7");
}