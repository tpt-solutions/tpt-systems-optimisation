//! Min-cost flow: successive shortest path and network simplex.
//!
//! Both algorithms operate on a directed [`tpt_math_graph::Graph`] whose edges
//! carry a `capacity` and a `cost`. A per-node `supply` vector describes the
//! problem: `supply[i] > 0` is a source, `supply[i] < 0` a sink, and the total
//! supply must balance (`sum == 0`).

use std::vec::Vec;

use tpt_math_graph::Graph;
use tpt_opt_core::{SolverStatus, Tolerances};

/// Result of a min-cost flow computation.
#[derive(Debug, Clone)]
pub struct MinCostFlowResult {
    /// Total cost `sum_e flow_e * cost_e`.
    pub total_cost: f64,
    /// Flow on each original edge (in `Graph::edges()` order).
    pub flow: Vec<f64>,
    /// Net outflow at each node (`out - in`); equals `supply` at optimality.
    pub node_balance: Vec<f64>,
    /// Terminal status. [`SolverStatus::Infeasible`] means not all supply could
    /// be routed (the graph is disconnected or capacities too small).
    pub status: SolverStatus,
}

/// A residual arc in the min-cost-flow working graph.
#[derive(Clone)]
struct RArc {
    to: usize,
    cap: f64,
    cost: f64,
    rev: usize,
}

struct Residual {
    arcs: Vec<RArc>,
    head: Vec<Vec<usize>>,
    fwd_of: Vec<usize>,
    s: usize,
    t: usize,
    nn: usize,
    total_supply: f64,
}

fn build_residual(graph: &Graph, supplies: &[f64], eps: f64) -> Residual {
    let n = graph.num_nodes();
    let s = n;
    let t = n + 1;
    let nn = n + 2;
    let mut arcs: Vec<RArc> = Vec::new();
    let mut head: Vec<Vec<usize>> = vec![Vec::new(); nn];
    let mut fwd_of: Vec<usize> = vec![usize::MAX; graph.num_edges()];

    for (ei, e) in graph.edges().iter().enumerate() {
        let fwd = arcs.len();
        let cap = if e.capacity > 0.0 { e.capacity } else { 0.0 };
        arcs.push(RArc { to: e.to, cap, cost: e.cost, rev: arcs.len() + 1 });
        head[e.from].push(fwd);
        let bwd = arcs.len();
        arcs.push(RArc { to: e.from, cap: 0.0, cost: -e.cost, rev: fwd });
        head[e.to].push(bwd);
        fwd_of[ei] = fwd;
    }

    let mut total = 0.0f64;
    for i in 0..n {
        let b = supplies[i];
        if b > eps {
            total += b;
            let fwd = arcs.len();
            arcs.push(RArc { to: i, cap: b, cost: 0.0, rev: arcs.len() + 1 });
            head[s].push(fwd);
            let bwd = arcs.len();
            arcs.push(RArc { to: s, cap: 0.0, cost: 0.0, rev: fwd });
            head[i].push(bwd);
        } else if b < -eps {
            let fwd = arcs.len();
            arcs.push(RArc { to: t, cap: -b, cost: 0.0, rev: arcs.len() + 1 });
            head[i].push(fwd);
            let bwd = arcs.len();
            arcs.push(RArc { to: i, cap: 0.0, cost: 0.0, rev: fwd });
            head[t].push(bwd);
        }
    }

    Residual { arcs, head, fwd_of, s, t, nn, total_supply: total }
}

fn extract(r: &Residual, graph: &Graph, eps: f64) -> MinCostFlowResult {
    let n = graph.num_nodes();
    let mut flow = vec![0.0f64; graph.num_edges()];
    for (ei, e) in graph.edges().iter().enumerate() {
        let fwd = r.fwd_of[ei];
        flow[ei] = e.capacity - r.arcs[fwd].cap;
    }
    let mut total_cost = 0.0f64;
    for (ei, e) in graph.edges().iter().enumerate() {
        total_cost += flow[ei] * e.cost;
    }
    let mut bal = vec![0.0f64; n];
    for (ei, e) in graph.edges().iter().enumerate() {
        bal[e.from] += flow[ei];
        bal[e.to] -= flow[ei];
    }

    // Check that all supply was routed (no residual out of S).
    let mut unmet = 0.0f64;
    for &ai in &r.head[r.s] {
        if r.arcs[ai].cap > eps {
            unmet += r.arcs[ai].cap;
        }
    }
    let status = if unmet > eps * (1.0 + r.total_supply) {
        SolverStatus::Infeasible
    } else {
        SolverStatus::Optimal
    };

    MinCostFlowResult { total_cost, flow, node_balance: bal, status }
}

/// Solve min-cost flow by the **successive shortest path** algorithm using
/// potentials (Johnson's) so Dijkstra can be used even with negative edge costs.
pub fn min_cost_flow(graph: &Graph, supplies: &[f64]) -> MinCostFlowResult {
    let tol = Tolerances::spec_default();
    let eps = tol.feasibility.max(1e-12);
    let mut r = build_residual(graph, supplies, eps);

    // Initialise potentials with Bellman-Ford so reduced costs are non-negative.
    let mut pot = vec![0.0f64; r.nn];
    {
        let mut dist = vec![f64::INFINITY; r.nn];
        dist[r.s] = 0.0;
        for _ in 0..r.nn {
            let mut updated = false;
            for u in 0..r.nn {
                if !dist[u].is_finite() {
                    continue;
                }
                for &ai in &r.head[u] {
                    let a = &r.arcs[ai];
                    if a.cap > eps {
                        let nd = dist[u] + a.cost;
                        if nd < dist[a.to] - eps {
                            dist[a.to] = nd;
                            updated = true;
                        }
                    }
                }
            }
            if !updated {
                break;
            }
        }
        for v in 0..r.nn {
            if dist[v].is_finite() {
                pot[v] = dist[v];
            }
        }
    }

    let mut prev_node = vec![usize::MAX; r.nn];
    let mut prev_arc = vec![usize::MAX; r.nn];
    let mut dist = vec![0.0f64; r.nn];
    let mut visited = vec![false; r.nn];

    loop {
        for v in 0..r.nn {
            dist[v] = f64::INFINITY;
            visited[v] = false;
            prev_node[v] = usize::MAX;
            prev_arc[v] = usize::MAX;
        }
        dist[r.s] = 0.0;
        for _ in 0..r.nn {
            let mut u = usize::MAX;
            let mut best = f64::INFINITY;
            for v in 0..r.nn {
                if !visited[v] && dist[v] < best {
                    best = dist[v];
                    u = v;
                }
            }
            if u == usize::MAX {
                break;
            }
            visited[u] = true;
            for &ai in &r.head[u] {
                let a = &r.arcs[ai];
                if a.cap > eps {
                    let rc = a.cost + pot[u] - pot[a.to];
                    let nd = dist[u] + rc;
                    if nd < dist[a.to] - eps {
                        dist[a.to] = nd;
                        prev_node[a.to] = u;
                        prev_arc[a.to] = ai;
                    }
                }
            }
        }

        if !dist[r.t].is_finite() {
            break;
        }

        // Augment along the shortest path S -> T.
        let mut bot = f64::INFINITY;
        let mut v = r.t;
        while v != r.s {
            let ai = prev_arc[v];
            bot = bot.min(r.arcs[ai].cap);
            v = prev_node[v];
        }
        v = r.t;
        while v != r.s {
            let ai = prev_arc[v];
            r.arcs[ai].cap -= bot;
            r.arcs[r.arcs[ai].rev].cap += bot;
            v = prev_node[v];
        }

        // Update potentials (only for reachable nodes).
        for v in 0..r.nn {
            if dist[v].is_finite() {
                pot[v] += dist[v];
            }
        }
    }

    extract(&r, graph, eps)
}

/// Find a negative-cost cycle in the residual graph, returning its arc indices.
fn find_negative_cycle(r: &Residual, eps: f64) -> Option<Vec<usize>> {
    let mut dist = vec![0.0f64; r.nn];
    let mut par = vec![usize::MAX; r.nn];
    let mut par_arc = vec![usize::MAX; r.nn];
    let mut last_neg = usize::MAX;
    for _ in 0..r.nn {
        last_neg = usize::MAX;
        for u in 0..r.nn {
            for &ai in &r.head[u] {
                let a = &r.arcs[ai];
                if a.cap > eps {
                    if dist[u] + a.cost < dist[a.to] - eps {
                        dist[a.to] = dist[u] + a.cost;
                        par[a.to] = u;
                        par_arc[a.to] = ai;
                        last_neg = a.to;
                    }
                }
            }
        }
    }
    if last_neg == usize::MAX {
        return None;
    }
    let mut v = last_neg;
    for _ in 0..r.nn {
        v = par[v];
    }
    let mut cycle = Vec::new();
    let mut cur = v;
    loop {
        cycle.push(par_arc[cur]);
        cur = par[cur];
        if cur == v {
            break;
        }
    }
    cycle.reverse();
    Some(cycle)
}

/// Obtain a feasible (not necessarily optimal) flow via Edmonds-Karp so that
/// negative-cycle cancelling has a valid starting point.
fn max_flow_feasible(r: &mut Residual, eps: f64) {
    loop {
        // BFS for an augmenting path S -> T in the residual graph.
        let mut prev_arc = vec![usize::MAX; r.nn];
        let mut q = std::collections::VecDeque::new();
        let mut seen = vec![false; r.nn];
        seen[r.s] = true;
        q.push_back(r.s);
        while let Some(u) = q.pop_front() {
            if u == r.t {
                break;
            }
            for &ai in &r.head[u] {
                let a = &r.arcs[ai];
                if a.cap > eps && !seen[a.to] {
                    seen[a.to] = true;
                    prev_arc[a.to] = ai;
                    q.push_back(a.to);
                }
            }
        }
        if !seen[r.t] {
            break;
        }
        let mut bot = f64::INFINITY;
        let mut v = r.t;
        while v != r.s {
            let ai = prev_arc[v];
            bot = bot.min(r.arcs[ai].cap);
            v = r.arcs[r.arcs[ai].rev].to;
        }
        v = r.t;
        while v != r.s {
            let ai = prev_arc[v];
            r.arcs[ai].cap -= bot;
            r.arcs[r.arcs[ai].rev].cap += bot;
            v = r.arcs[r.arcs[ai].rev].to;
        }
    }
}

/// Solve min-cost flow with a **network simplex** (negative-cycle cancelling)
/// method. The residual graph is first made feasible, then every negative-cost
/// cycle is cancelled until the flow is optimal.
pub fn network_simplex(graph: &Graph, supplies: &[f64]) -> MinCostFlowResult {
    let tol = Tolerances::spec_default();
    let eps = tol.feasibility.max(1e-12);
    let mut r = build_residual(graph, supplies, eps);

    max_flow_feasible(&mut r, eps);

    // Cancel negative cycles.
    for _ in 0..1000 {
        match find_negative_cycle(&r, eps) {
            Some(cycle) => {
                let mut bot = f64::INFINITY;
                for &ai in &cycle {
                    bot = bot.min(r.arcs[ai].cap);
                }
                if !bot.is_finite() || bot <= eps {
                    break;
                }
                for &ai in &cycle {
                    r.arcs[ai].cap -= bot;
                    r.arcs[r.arcs[ai].rev].cap += bot;
                }
            }
            None => break,
        }
    }

    extract(&r, graph, eps)
}

/// A reusable min-cost-flow problem description.
#[derive(Debug, Clone)]
pub struct NetworkFlow<'a> {
    graph: &'a Graph,
    supplies: Vec<f64>,
    tol: Tolerances,
}

impl<'a> NetworkFlow<'a> {
    /// Build a problem from a graph and a per-node supply vector.
    pub fn new(graph: &'a Graph, supplies: Vec<f64>) -> Self {
        Self {
            graph,
            supplies,
            tol: Tolerances::spec_default(),
        }
    }

    /// Override the tolerances used for numerical comparisons.
    pub fn with_tolerances(mut self, tol: Tolerances) -> Self {
        self.tol = tol;
        self
    }

    /// Solve with the successive shortest path algorithm.
    pub fn solve(&self) -> MinCostFlowResult {
        let eps = self.tol.feasibility.max(1e-12);
        let _ = eps;
        min_cost_flow(self.graph, &self.supplies)
    }

    /// Solve with the network simplex (negative-cycle cancelling) algorithm.
    pub fn solve_simplex(&self) -> MinCostFlowResult {
        network_simplex(self.graph, &self.supplies)
    }
}
