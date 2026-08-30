//! A small, dependency-free directed graph used by the network-flow and OPF
//! solvers (and re-exported by the umbrella crate).
//!
//! This is a local stand-in for the historical `tpt-math-graph` `Graph` API
//! (`num_nodes` / `edges` / `has_cycle` / `Edge { from, to, capacity, cost }`):
//! the published `tpt-math-graph` crate re-exports `petgraph` directly and no
//! longer ships this convenience surface, so it is vendored here to keep the
//! optimisation crates publishable against the published `tpt-math-*` crates.

use crate::alloc::vec;
use crate::alloc::vec::Vec;

/// A directed edge carrying a `capacity` and a `cost` (min-cost-flow / OPF
/// semantics).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    /// Tail node.
    pub from: usize,
    /// Head node.
    pub to: usize,
    /// Flow capacity (non-negative).
    pub capacity: f64,
    /// Per-unit flow cost.
    pub cost: f64,
}

impl Edge {
    /// Construct an edge from `from` to `to` with the given `capacity`/`cost`.
    pub fn new(from: usize, to: usize, capacity: f64, cost: f64) -> Self {
        Edge {
            from,
            to,
            capacity,
            cost,
        }
    }
}

/// Directed graph with `n` nodes and a list of [`Edge`]s.
///
/// Nodes are integers `0..n`; edges are stored in insertion order and can be
/// visited via [`Graph::edges`]. `has_cycle` performs a directed cycle test.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    n: usize,
    edges: Vec<Edge>,
    /// Outgoing edge indices per node (directed), used for `has_cycle`.
    out: Vec<Vec<usize>>,
}

impl Graph {
    /// Create a graph with `n` nodes and no edges.
    pub fn new(n: usize) -> Self {
        Graph {
            n,
            edges: Vec::new(),
            out: vec![Vec::new(); n],
        }
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Append an edge, returning its index in [`Graph::edges`].
    pub fn add_edge(&mut self, e: Edge) -> usize {
        let idx = self.edges.len();
        if e.from < self.n {
            self.out[e.from].push(idx);
        }
        self.edges.push(e);
        idx
    }

    /// All edges in insertion order.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// `true` if the directed graph contains a (directed) cycle.
    pub fn has_cycle(&self) -> bool {
        // Iterative DFS with white/gray/black colouring.
        let mut color = vec![0u8; self.n]; // 0 = white, 1 = gray, 2 = black
        for start in 0..self.n {
            if color[start] != 0 {
                continue;
            }
            let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
            color[start] = 1;
            while let Some(&(u, ci)) = stack.last() {
                if ci < self.out[u].len() {
                    // Advance the neighbour cursor before descending.
                    stack.last_mut().unwrap().1 += 1;
                    let v = self.edges[self.out[u][ci]].to;
                    if color[v] == 1 {
                        return true;
                    } else if color[v] == 0 {
                        color[v] = 1;
                        stack.push((v, 0));
                    }
                } else {
                    color[u] = 2;
                    stack.pop();
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_cycle_on_path() {
        let mut g = Graph::new(3);
        g.add_edge(Edge::new(0, 1, 1.0, 1.0));
        g.add_edge(Edge::new(1, 2, 1.0, 1.0));
        assert!(!g.has_cycle());
        assert_eq!(g.num_edges(), 2);
    }

    #[test]
    fn detects_directed_cycle() {
        let mut g = Graph::new(3);
        g.add_edge(Edge::new(0, 1, 1.0, 1.0));
        g.add_edge(Edge::new(1, 2, 1.0, 1.0));
        g.add_edge(Edge::new(2, 0, 1.0, 1.0));
        assert!(g.has_cycle());
    }

    #[test]
    fn edges_preserved_in_order() {
        let mut g = Graph::new(2);
        g.add_edge(Edge::new(0, 1, 3.0, 2.0));
        g.add_edge(Edge::new(1, 0, 1.0, 4.0));
        assert_eq!(g.edges()[0].capacity, 3.0);
        assert_eq!(g.edges()[1].cost, 4.0);
    }
}
