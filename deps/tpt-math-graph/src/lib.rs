#![no_std]
//! Local dev shim mirroring `tpt-math-graph`: a directed graph with
//! weighted edges, used by `tpt-opt-network` for network-flow and OPF problems.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

/// A directed edge with an optional capacity and cost (for flow problems).
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub capacity: f64,
    pub cost: f64,
}

impl Edge {
    pub fn new(from: usize, to: usize, capacity: f64, cost: f64) -> Self {
        Self { from, to, capacity, cost }
    }
}

/// A directed graph with node count and a list of edges.
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    num_nodes: usize,
    edges: Vec<Edge>,
}

impl Graph {
    pub fn new(num_nodes: usize) -> Self {
        Self { num_nodes, edges: Vec::new() }
    }

    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Build a CSR-style adjacency: for each node, the list of outgoing edge
    /// indices. Useful for traversal algorithms.
    pub fn outgoing(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.num_nodes];
        for (i, e) in self.edges.iter().enumerate() {
            out[e.from].push(i);
        }
        out
    }

    /// Detect whether the graph contains a cycle (DFS over directed edges).
    pub fn has_cycle(&self) -> bool {
        let out = self.outgoing();
        let mut state = vec![0u8; self.num_nodes]; // 0=unvisited,1=instack,2=done
        fn dfs(u: usize, out: &[Vec<usize>], edges: &[Edge], state: &mut [u8]) -> bool {
            state[u] = 1;
            for &ei in &out[u] {
                let v = edges[ei].to;
                match state[v] {
                    1 => return true,
                    0 => {
                        if dfs(v, out, edges, state) {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
            state[u] = 2;
            false
        }
        for u in 0..self.num_nodes {
            if state[u] == 0 && dfs(u, &out, &self.edges, &mut state) {
                return true;
            }
        }
        false
    }
}
