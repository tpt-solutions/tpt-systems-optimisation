//! Integration tests for the progress-callback API
//! (`MilpSolver::with_progress_callback` + `tpt_opt_core::progress`).

use std::sync::{Arc, Mutex};

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::progress::{ProgressAction, ProgressEvent};
use tpt_opt_core::solver::{Solver, SolverStatus, WarmStart};
use tpt_opt_milp::MilpSolver;

/// Binary knapsack: max 5a+4b+7c+3d+6e s.t. 3a+2b+5c+4d+3e <= 10.
/// Optimum 17 (items b+c+e fill the capacity exactly; verified by
/// enumeration — every other feasible subset scores at most 16).
fn knapsack() -> Model {
    let mut m = Model::new(5);
    for v in m.variables.iter_mut() {
        v.bound = tpt_opt_core::VarBound::binary();
    }
    m.add_constraint(Constraint::le(vec![0, 1, 2, 3, 4], vec![3.0, 2.0, 5.0, 4.0, 3.0], 10.0));
    m.set_objective(Objective {
        sense: Sense::Maximize,
        indices: vec![0, 1, 2, 3, 4],
        coeffs: vec![5.0, 4.0, 7.0, 3.0, 6.0],
        constant: 0.0,
    });
    m
}

/// Deterministic pseudo-random (weights, values) for a 24-item knapsack.
fn knapsack_data() -> (Vec<f64>, Vec<f64>) {
    let n = 24;
    let mut s: u64 = 42;
    let next = |s: &mut u64| {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 33) % 100
    };
    let mut w = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        w.push((next(&mut s) % 40 + 1) as f64);
        v.push((next(&mut s) % 90 + 1) as f64);
    }
    (w, v)
}

/// A 24-item knapsack whose search reliably spans more than two progress
/// events (used by the mid-search abort test).
fn bigger_knapsack() -> Model {
    let (w, v) = knapsack_data();
    let n = w.len();
    let mut m = Model::new(n);
    for var in m.variables.iter_mut() {
        var.bound = tpt_opt_core::VarBound::binary();
    }
    m.add_constraint(Constraint::le((0..n).collect(), w.clone(), 250.0));
    m.set_objective(Objective {
        sense: Sense::Maximize,
        indices: (0..n).collect(),
        coeffs: v,
        constant: 0.0,
    });
    m
}

/// Greedy ratio-ordered feasible point for `bigger_knapsack`'s data — used as
/// a warm start so an incumbent exists regardless of heuristic luck.
fn greedy_point() -> Vec<f64> {
    let (w, v) = knapsack_data();
    let mut order: Vec<usize> = (0..w.len()).collect();
    order.sort_by(|&a, &b| {
        (v[b] / w[b]).partial_cmp(&(v[a] / w[a])).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut x = vec![0.0; w.len()];
    let mut cap = 250.0;
    for &i in &order {
        if w[i] <= cap {
            x[i] = 1.0;
            cap -= w[i];
        }
    }
    x
}

#[test]
fn callback_receives_events_and_solve_stays_optimal() {
    let events: Arc<Mutex<Vec<ProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let mut solver = MilpSolver::new().with_progress_callback(Box::new(move |ev| {
        sink.lock().unwrap().push(*ev);
        ProgressAction::Continue
    }));

    let sol = solver.solve(&knapsack()).expect("solve");
    assert_eq!(sol.status, SolverStatus::Optimal);
    assert!((sol.objective_value - 17.0).abs() < 1e-6, "knapsack optimum is 17");

    let events = events.lock().unwrap();
    assert!(!events.is_empty(), "at least the root checkpoints must fire");
    // Iteration counts are monotonically non-decreasing.
    for w in events.windows(2) {
        assert!(w[1].iterations >= w[0].iterations);
    }
    // The final event must carry the optimal incumbent.
    let last = events.last().unwrap();
    assert_eq!(last.incumbent, Some(17.0));
    // Root checkpoints report the LP bound.
    assert!(events[0].bound.is_some());
}

#[test]
fn abort_on_first_event_reports_time_limit() {
    let mut solver =
        MilpSolver::new().with_progress_callback(Box::new(|_ev| ProgressAction::Abort));
    let sol = solver.solve(&knapsack()).expect("solve");
    // Aborted before/inside the search: never Optimal.
    assert_eq!(sol.status, SolverStatus::TimeLimit);
    assert!(solver.status() == SolverStatus::TimeLimit);
}

#[test]
fn abort_mid_search_reports_time_limit_with_incumbent() {
    // Abort once the search passes 16 explored nodes: deterministic trigger
    // independent of how quickly the tiny root phase completes.
    let mut solver = MilpSolver::new().with_progress_callback(Box::new(|ev| {
        if ev.iterations >= 16 {
            ProgressAction::Abort
        } else {
            ProgressAction::Continue
        }
    }));
    // Warm-start with a greedy feasible point so the aborted solve is
    // guaranteed to report an incumbent even if every heuristic misses.
    let warm = greedy_point();
    let (_, vals) = knapsack_data();
    let warm_value: f64 = warm.iter().zip(vals.iter()).map(|(&xi, &vi)| xi * vi).sum();
    solver.warm_start(WarmStart::primal(warm)).expect("warm start");

    let sol = solver.solve(&bigger_knapsack()).expect("solve");
    assert_eq!(sol.status, SolverStatus::TimeLimit);
    // The warm-start incumbent must survive the abort, and the reported
    // point must be integral.
    assert!(
        sol.objective_value >= warm_value - 1e-6,
        "incumbent {} must be at least the warm start {warm_value}",
        sol.objective_value
    );
    assert!(sol.primal.iter().all(|&v| (v - v.round()).abs() < 1e-6));
}

#[test]
fn no_callback_behaves_identically() {
    let mut solver = MilpSolver::new();
    let sol = solver.solve(&knapsack()).expect("solve");
    assert_eq!(sol.status, SolverStatus::Optimal);
    assert!((sol.objective_value - 17.0).abs() < 1e-6);
}

#[test]
fn parallel_mode_emits_and_abort_propagates() {
    let events: Arc<Mutex<Vec<ProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let mut solver =
        MilpSolver::new().with_threads(4).with_progress_callback(Box::new(move |ev| {
            sink.lock().unwrap().push(*ev);
            ProgressAction::Continue
        }));
    let sol = solver.solve(&knapsack()).expect("parallel solve");
    assert_eq!(sol.status, SolverStatus::Optimal);
    assert!((sol.objective_value - 17.0).abs() < 1e-6);
    assert!(!events.lock().unwrap().is_empty());

    // Abort in parallel mode: every worker observes the shared flag.
    let mut solver = MilpSolver::new()
        .with_threads(4)
        .with_progress_callback(Box::new(|_ev| ProgressAction::Abort));
    let sol = solver.solve(&knapsack()).expect("parallel solve");
    assert_eq!(sol.status, SolverStatus::TimeLimit);
}
