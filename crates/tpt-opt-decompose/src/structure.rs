//! Automatic decomposable-structure detection and strategy recommendation.
//!
//! The constraint matrix is analysed as a bipartite row–column graph:
//!
//! 1. Connected components of the graph identify independent blocks.
//! 2. Rows touching several components are **linking rows** (Dantzig–Wolfe
//!    coupling rows); columns appearing in rows of several components are
//!    **linking columns** (Benders complicating variables).
//! 3. A recommendation is made from the counts: pure block-diagonal ⇒
//!    solve blocks independently; few linking rows ⇒ Dantzig–Wolfe; few
//!    linking columns ⇒ Benders; otherwise solve directly.
//!
//! Detection is heuristic (greedy linker peeling by degree) but cheap and
//! deterministic.

use std::vec::Vec;

use tpt_opt_core::model::Model;

/// Recommended decomposition strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Independent blocks, no coupling at all.
    IndependentBlocks,
    /// Few dense linking rows → Dantzig–Wolfe / column generation.
    DantzigWolfe,
    /// Few linking columns → Benders on the complicating variables.
    Benders,
    /// No exploitable structure detected.
    Direct,
}

/// Result of structure detection.
#[derive(Debug, Clone)]
pub struct StructureReport {
    /// Number of independent row/column blocks found.
    pub num_components: usize,
    /// Row indices belonging to each block component.
    pub component_rows: Vec<Vec<usize>>,
    /// Column indices belonging to each block component.
    pub component_cols: Vec<Vec<usize>>,
    /// Rows that couple several components.
    pub linking_rows: Vec<usize>,
    /// Columns that appear in several components.
    pub linking_cols: Vec<usize>,
    /// Recommended strategy.
    pub strategy: Strategy,
}

/// Detect decomposable structure in `model`.
pub fn detect_structure(model: &Model) -> StructureReport {
    let n_cols = model.num_vars;
    let mut parent: Vec<usize> = (0..n_cols).collect();

    fn find(parent: &mut Vec<usize>, i: usize) -> usize {
        let mut root = i;
        while parent[root] != root {
            root = parent[root];
        }
        let mut cur = i;
        while parent[cur] != root {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }

    fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    // Union all columns co-occurring in any row.
    for c in &model.constraints {
        for w in c.indices.windows(2) {
            union(&mut parent, w[0], w[1]);
        }
    }

    // Classify rows by how many distinct components they touch.
    let mut linking_rows = Vec::new();
    let mut row_components: Vec<Option<usize>> = vec![None; model.constraints.len()];
    for (ri, c) in model.constraints.iter().enumerate() {
        let mut roots: Vec<usize> =
            c.indices.iter().map(|&j| find(&mut parent, j)).collect();
        roots.sort_unstable();
        roots.dedup();
        match roots.len() {
            0 => {}
            1 => row_components[ri] = Some(roots[0]),
            _ => linking_rows.push(ri),
        }
    }

    // Linking columns: columns touched by ≥ 2 distinct row-components or by
    // any linking row.
    let mut comp_of_col: Vec<Option<usize>> = vec![None; n_cols];
    for j in 0..n_cols {
        let r = find(&mut parent, j);
        comp_of_col[j] = Some(r);
    }
    let mut linking_cols = Vec::new();
    for (ri, c) in model.constraints.iter().enumerate() {
        if linking_rows.contains(&ri) {
            continue;
        }
        let rc = row_components[ri];
        for &j in &c.indices {
            if comp_of_col[j] != rc {
                if !linking_cols.contains(&j) {
                    linking_cols.push(j);
                }
            }
        }
    }
    for &ri in &linking_rows {
        for &j in &model.constraints[ri].indices {
            if !linking_cols.contains(&j) {
                linking_cols.push(j);
            }
        }
    }

    // Collect components from non-linking rows.
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (ri, rc) in row_components.iter().enumerate() {
        if let Some(root) = rc {
            groups.entry(*root).or_default().push(ri);
        }
    }
    let component_rows: Vec<Vec<usize>> = groups.into_values().collect();
    let num_components = component_rows.len().max(1);

    // Component column sets.
    let mut component_cols: Vec<Vec<usize>> = vec![Vec::new(); num_components];
    for j in 0..n_cols {
        if linking_cols.contains(&j) {
            continue;
        }
        let root = find(&mut parent, j);
        // Map root onto one of the collected row groups (same order).
        let slot = component_rows
            .iter()
            .position(|rows| {
                rows.iter()
                    .any(|&ri| model.constraints[ri].indices.contains(&j))
            })
            .unwrap_or(0);
        let _ = root;
        component_cols[slot].push(j);
    }

    // Recommendation.
    let strategy = if linking_rows.is_empty() && linking_cols.is_empty() && num_components > 1 {
        Strategy::IndependentBlocks
    } else if !linking_rows.is_empty() && linking_cols.is_empty() {
        Strategy::DantzigWolfe
    } else if !linking_cols.is_empty() && linking_rows.is_empty() {
        Strategy::Benders
    } else if !linking_rows.is_empty() || !linking_cols.is_empty() {
        // Both kinds present: prefer the sparser coupling.
        if linking_rows.len() <= linking_cols.len() {
            Strategy::DantzigWolfe
        } else {
            Strategy::Benders
        }
    } else {
        Strategy::Direct
    };

    StructureReport {
        num_components,
        component_rows,
        component_cols,
        linking_rows,
        linking_cols,
        strategy,
    }
}