//! Graph preprocessing utilities operating on an undirected view of a directed
//! [`tpt_opt_core::graph::Graph`].
//!
//! Each directed edge `(u, v)` is treated as a single undirected connection
//! between `u` and `v`. Parallel directed edges between the same pair are
//! recognised as multiple undirected edges (so they are never bridges).

use std::vec::Vec;

use tpt_opt_core::graph::Graph;

/// Returns `true` if the directed graph contains a (directed) cycle, delegating
/// to [`tpt_opt_core::graph::Graph::has_cycle`].
pub fn has_cycle(graph: &Graph) -> bool {
    graph.has_cycle()
}

/// Undirected adjacency entry: `(to_node, edge_index)`.
type Adj = Vec<Vec<(usize, usize)>>;

fn undirected_adjacency(graph: &Graph) -> Adj {
    let n = graph.num_nodes();
    let mut adj: Adj = vec![Vec::new(); n];
    for (e, edge) in graph.edges().iter().enumerate() {
        adj[edge.from].push((edge.to, e));
        adj[edge.to].push((edge.from, e));
    }
    adj
}

/// Identify bridges (cut edges) in the undirected view of `graph`.
///
/// Returns the undirected edges as `(min(u, v), max(u, v))` node pairs whose
/// removal increases the number of connected components. Graphs handled by this
/// crate are small, so a recursive Tarjan DFS is used for clarity.
pub fn bridges(graph: &Graph) -> Vec<(usize, usize)> {
    let adj = undirected_adjacency(graph);
    let n = graph.num_nodes();
    let mut tin = vec![-1i64; n];
    let mut low = vec![0i64; n];
    let mut timer: i64 = 0;
    let mut result: Vec<(usize, usize)> = Vec::new();
    for start in 0..n {
        if tin[start] == -1 {
            bridges_dfs(start, -1, &adj, &mut tin, &mut low, &mut timer, &mut result);
        }
    }
    result
}

fn bridges_dfs(
    u: usize,
    p_edge: i64,
    adj: &Adj,
    tin: &mut [i64],
    low: &mut [i64],
    timer: &mut i64,
    result: &mut Vec<(usize, usize)>,
) {
    tin[u] = *timer;
    low[u] = *timer;
    *timer += 1;
    for &(v, e) in adj[u].iter() {
        if e as i64 == p_edge {
            continue;
        }
        if tin[v] != -1 {
            low[u] = low[u].min(tin[v]);
        } else {
            bridges_dfs(v, e as i64, adj, tin, low, timer, result);
            low[u] = low[u].min(low[v]);
            if low[v] > tin[u] {
                let a = u.min(v);
                let b = u.max(v);
                result.push((a, b));
            }
        }
    }
}

/// Result of biconnected component decomposition.
#[derive(Debug, Clone)]
pub struct Biconnected {
    /// Each component is a list of original (directed) edge indices.
    pub components: Vec<Vec<usize>>,
    /// Articulation points (cut vertices).
    pub articulation_points: Vec<usize>,
}

/// Decompose the undirected view of `graph` into biconnected components and
/// identify articulation points (Tarjan's algorithm).
pub fn biconnected_components(graph: &Graph) -> Biconnected {
    let n = graph.num_nodes();
    let adj = undirected_adjacency(graph);
    let mut tin = vec![-1i64; n];
    let mut low = vec![0i64; n];
    let mut timer: i64 = 0;
    let mut stack: Vec<usize> = Vec::new();
    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut artic: Vec<usize> = Vec::new();

    for start in 0..n {
        if tin[start] == -1 {
            bicon_dfs(
                start,
                -1,
                &adj,
                &mut tin,
                &mut low,
                &mut timer,
                &mut stack,
                &mut components,
                &mut artic,
            );
        }
    }
    components.retain(|c| !c.is_empty());
    artic.sort_unstable();
    artic.dedup();
    Biconnected { components, articulation_points: artic }
}

/// Outcome of a series-parallel reduction check.
#[derive(Debug, Clone)]
pub struct SeriesParallelReport {
    /// `true` when every connected component reduces to a single edge (or
    /// vanishes entirely) — i.e. the undirected view has no K₄ minor and
    /// admits an O(n + m) dynamic-programming treatment for many flow/OPF
    /// subproblems.
    pub is_series_parallel: bool,
    /// Number of degree-2 vertex contractions performed.
    pub series_reductions: usize,
    /// Number of parallel-edge merges performed.
    pub parallel_reductions: usize,
    /// Number of dangling (degree-1) edge prunes performed.
    pub dangling_prunes: usize,
    /// Undirected edges `(min, max)` left after the reduction fixpoint.
    pub remaining_edges: Vec<(usize, usize)>,
}

/// Check whether the undirected view of `graph` is **series-parallel** by
/// running the classic reduction system to a fixpoint:
///
/// - *parallel reduction*: merge two edges spanning the same node pair;
/// - *series reduction*: contract a degree-2 vertex into a single edge
///   between its two neighbours;
/// - *dangling prune*: drop the edge of a degree-1 vertex.
///
/// A connected graph is series-parallel (equivalently K₄-minor-free /
/// treewidth ≤ 2) exactly when this system reduces it to one edge or to
/// nothing; disconnected graphs qualify when every component does.
pub fn series_parallel_check(graph: &Graph) -> SeriesParallelReport {
    let n = graph.num_nodes();
    let adj = undirected_adjacency(graph);

    // Connected components over the undirected view.
    let mut comp = vec![usize::MAX; n];
    let mut components: Vec<Vec<usize>> = Vec::new();
    for start in 0..n {
        if comp[start] != usize::MAX {
            continue;
        }
        let id = components.len();
        let mut stack = vec![start];
        comp[start] = id;
        let mut members = Vec::new();
        while let Some(u) = stack.pop() {
            members.push(u);
            for &(v, _) in adj[u].iter() {
                if comp[v] == usize::MAX {
                    comp[v] = id;
                    stack.push(v);
                }
            }
        }
        components.push(members);
    }

    // Per-component edge multisets.
    let mut per_component: Vec<Vec<(usize, usize)>> = vec![Vec::new(); components.len()];
    for edge in graph.edges().iter() {
        if edge.from == edge.to {
            continue; // self-loops never aid reduction; ignore them
        }
        let (a, b) = (edge.from.min(edge.to), edge.from.max(edge.to));
        per_component[comp[edge.from]].push((a, b));
    }

    let mut report = SeriesParallelReport {
        is_series_parallel: true,
        series_reductions: 0,
        parallel_reductions: 0,
        dangling_prunes: 0,
        remaining_edges: Vec::new(),
    };

    for mut edges in per_component {
        reduce_component(&mut edges, &mut report);
        if edges.len() > 1 {
            report.is_series_parallel = false;
        }
        report.remaining_edges.extend(edges);
    }
    report.remaining_edges.sort_unstable();
    report.remaining_edges.dedup();
    report
}

/// Run parallel / dangling / series reductions on one component's edge list
/// until a fixpoint, accumulating counts into `report`.
fn reduce_component(edges: &mut Vec<(usize, usize)>, report: &mut SeriesParallelReport) {
    loop {
        let mut changed = false;

        // Parallel reduction: dedupe identical pairs.
        edges.sort_unstable();
        let before = edges.len();
        edges.dedup();
        report.parallel_reductions += before - edges.len();
        changed |= before != edges.len();

        // Degree bookkeeping.
        let mut degree: std::collections::BTreeMap<usize, usize> = Default::default();
        for &(a, b) in edges.iter() {
            *degree.entry(a).or_insert(0) += 1;
            *degree.entry(b).or_insert(0) += 1;
        }

        // Dangling prune: remove the single edge at a degree-1 vertex.
        if let Some((&v, _)) = degree.iter().find(|(_, &d)| d == 1) {
            if let Some(pos) = edges.iter().position(|&(a, b)| a == v || b == v) {
                edges.swap_remove(pos);
                report.dangling_prunes += 1;
                continue; // edge count strictly decreased; loop again
            }
        }

        // Series reduction: contract a degree-2 vertex.
        let two = degree.iter().find(|(_, &d)| d == 2).map(|(&v, _)| v);
        if let Some(v) = two {
            let incident: Vec<usize> = edges
                .iter()
                .enumerate()
                .filter(|(_, &(a, b))| a == v || b == v)
                .map(|(i, _)| i)
                .collect();
            debug_assert_eq!(incident.len(), 2);
            let (i, j) = (incident[0], incident[1]);
            let other = |e: (usize, usize)| if e.0 == v { e.1 } else { e.0 };
            let (a, b) = (other(edges[i]), other(edges[j]));
            edges.swap_remove(j);
            edges.swap_remove(i);
            if a != b {
                edges.push((a.min(b), a.max(b)));
            }
            report.series_reductions += 1;
            changed = true;
        }

        if !changed {
            return;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn bicon_dfs(
    u: usize,
    p_edge: i64,
    adj: &Adj,
    tin: &mut [i64],
    low: &mut [i64],
    timer: &mut i64,
    stack: &mut Vec<usize>,
    components: &mut Vec<Vec<usize>>,
    artic: &mut Vec<usize>,
) {
    tin[u] = *timer;
    low[u] = *timer;
    *timer += 1;
    let mut children = 0;
    for &(v, e) in adj[u].iter() {
        if e as i64 == p_edge {
            continue;
        }
        if tin[v] != -1 {
            low[u] = low[u].min(tin[v]);
            if tin[v] < tin[u] {
                stack.push(e);
            }
        } else {
            stack.push(e);
            children += 1;
            bicon_dfs(v, e as i64, adj, tin, low, timer, stack, components, artic);
            low[u] = low[u].min(low[v]);
            if (p_edge == -1 && children > 1) || (p_edge != -1 && low[v] >= tin[u]) {
                if !artic.contains(&u) {
                    artic.push(u);
                }
                let mut comp = Vec::new();
                loop {
                    let x = stack.pop().expect("non-empty edge stack");
                    comp.push(x);
                    if x == e {
                        break;
                    }
                }
                components.push(comp);
            }
        }
    }
}

#[cfg(test)]
mod sp_tests {
    use super::*;
    use tpt_opt_core::graph::Edge;

    fn graph(n: usize, edges: &[(usize, usize)]) -> Graph {
        let mut g = Graph::new(n);
        for &(a, b) in edges {
            g.add_edge(Edge::new(a, b, 1.0, 1.0));
        }
        g
    }

    #[test]
    fn path_reduces_by_pruning() {
        let r = series_parallel_check(&graph(4, &[(0, 1), (1, 2), (2, 3)]));
        assert!(r.is_series_parallel);
        assert_eq!(r.dangling_prunes, 3);
        assert_eq!(r.series_reductions, 0);
        assert!(r.remaining_edges.is_empty());
    }

    #[test]
    fn tree_is_series_parallel() {
        let tree = graph(5, &[(0, 1), (0, 2), (2, 3), (2, 4)]);
        assert!(series_parallel_check(&tree).is_series_parallel);
    }

    #[test]
    fn triangle_reduces_via_series_then_parallel() {
        let r = series_parallel_check(&graph(3, &[(0, 1), (1, 2), (2, 0)]));
        assert!(r.is_series_parallel);
        assert_eq!(r.series_reductions, 1);
        assert_eq!(r.parallel_reductions, 1);
    }

    #[test]
    fn bowtie_two_triangles_sharing_a_vertex_is_sp() {
        let bowtie = graph(5, &[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]);
        assert!(series_parallel_check(&bowtie).is_series_parallel);
    }

    #[test]
    fn k4_is_rejected() {
        let k4 = graph(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
        let r = series_parallel_check(&k4);
        assert!(!r.is_series_parallel);
        assert_eq!(r.remaining_edges.len(), 6);
        assert_eq!(r.series_reductions, 0);
    }

    #[test]
    fn k4_minus_an_edge_is_sp() {
        let g = graph(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]);
        assert!(series_parallel_check(&g).is_series_parallel);
    }

    #[test]
    fn parallel_edges_merge() {
        let r = series_parallel_check(&graph(2, &[(0, 1), (0, 1), (1, 0)]));
        assert!(r.is_series_parallel);
        assert_eq!(r.parallel_reductions, 2);
    }

    #[test]
    fn disconnected_components_judged_independently() {
        // Triangle (SP) + K4 (not): the whole graph is not SP.
        let mut mixed = Graph::new(7);
        for &(a, b) in &[(0, 1), (1, 2), (2, 0)] {
            mixed.add_edge(Edge::new(a, b, 1.0, 1.0));
        }
        for &(a, b) in &[(3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)] {
            mixed.add_edge(Edge::new(a, b, 1.0, 1.0));
        }
        assert!(!series_parallel_check(&mixed).is_series_parallel);
        // Two disjoint triangles: SP overall.
        let mut two = Graph::new(6);
        for &(a, b) in &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)] {
            two.add_edge(Edge::new(a, b, 1.0, 1.0));
        }
        assert!(series_parallel_check(&two).is_series_parallel);
    }
}
