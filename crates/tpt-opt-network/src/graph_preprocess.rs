//! Graph preprocessing utilities operating on an undirected view of a directed
//! [`tpt_math_graph::Graph`].
//!
//! Each directed edge `(u, v)` is treated as a single undirected connection
//! between `u` and `v`. Parallel directed edges between the same pair are
//! recognised as multiple undirected edges (so they are never bridges).

use std::vec::Vec;

use tpt_math_graph::Graph;

/// Returns `true` if the directed graph contains a (directed) cycle, delegating
/// to [`tpt_math_graph::Graph::has_cycle`].
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
