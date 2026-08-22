//! Automatic decomposable-structure detection and strategy recommendation.
//!
//! The constraint matrix is analysed with a greedy union-find over columns:
//!
//! 1. Rows are processed smallest-span-first. A row merges its columns'
//!    components unless it bridges **two or more already-established
//!    groups** (groups that already absorbed a processed row) — such a row
//!    is a **linking row** and is left out of the partition. Processing
//!    tight rows first lets block-local rows form groups before wide
//!    coupling rows arrive.
//! 2. Columns touched by any linking row are **linking columns**
//!    (Benders-style complicating variables); every other column belongs
//!    to exactly one block.
//! 3. A recommendation follows from the counts: pure block-diagonal ⇒
//!    solve blocks independently; linking rows only ⇒ Dantzig–Wolfe;
//!    linking columns only ⇒ Benders; both ⇒ the sparser coupling wins;
//!    neither ⇒ solve directly.
//!
//! This is a heuristic: pathological orderings/overlaps can misclassify a
//! bridge row, but it matches the canonical block-angular shapes reliably.

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
    /// Columns that appear in several components (or in linking rows).
    pub linking_cols: Vec<usize>,
    /// Recommended strategy.
    pub strategy: Strategy,
}

/// Detect decomposable structure in `model`.
pub fn detect_structure(model: &Model) -> StructureReport {
    let n_cols = model.num_vars;
    let mut parent: Vec<usize> = (0..n_cols.max(1)).collect();

    fn find(parent: &mut [usize], i: usize) -> usize {
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

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    // Greedy classification, smallest rows first: a row becomes a linking
    // row only when it bridges ≥2 groups that already contain processed
    // rows; otherwise it merges into (or forms) a single group.
    let mut order: Vec<usize> = (0..model.constraints.len()).collect();
    order.sort_by_key(|&ri| model.constraints[ri].indices.len());
    let mut linking_rows = Vec::new();
    let mut row_component: Vec<Option<usize>> = vec![None; model.constraints.len()];
    let mut group_has_row: Vec<bool> = vec![false; n_cols.max(1)];
    for &ri in &order {
        let c = &model.constraints[ri];
        if c.indices.is_empty() {
            continue;
        }
        let mut roots: Vec<usize> = c.indices.iter().map(|&j| find(&mut parent, j)).collect();
        roots.sort_unstable();
        roots.dedup();
        let established = roots.iter().filter(|&&r| group_has_row[r]).count();
        if established >= 2 {
            linking_rows.push(ri);
        } else {
            for w in c.indices.windows(2) {
                union(&mut parent, w[0], w[1]);
            }
            let root = find(&mut parent, c.indices[0]);
            row_component[ri] = Some(root);
            group_has_row[root] = true;
        }
    }

    // Linking columns: any column appearing in a linking row. (A column
    // shared by two blocks without a linking row is impossible: the first
    // row bridging the blocks would have been classified as linking.)
    let mut linking_cols = Vec::new();
    for &ri in &linking_rows {
        for &j in &model.constraints[ri].indices {
            if !linking_cols.contains(&j) {
                linking_cols.push(j);
            }
        }
    }

    // Group non-linking rows by component root (stable order of first
    // appearance).
    let mut root_order: Vec<usize> = Vec::new();
    for root in row_component.iter().flatten() {
        if !root_order.contains(root) {
            root_order.push(*root);
        }
    }
    let num_components = root_order.len().max(1);
    let mut component_rows = vec![Vec::new(); num_components];
    for (ri, rc) in row_component.iter().enumerate() {
        if let Some(root) = rc {
            let slot = root_order.iter().position(|&r| r == *root).unwrap_or(0);
            component_rows[slot].push(ri);
        }
    }

    // Component columns: non-linking columns grouped by their root.
    let mut component_cols = vec![Vec::new(); num_components];
    for j in 0..n_cols {
        if linking_cols.contains(&j) {
            continue;
        }
        let root = find(&mut parent, j);
        if let Some(slot) = root_order.iter().position(|&r| r == root) {
            component_cols[slot].push(j);
        }
    }

    // Recommendation.
    let strategy = if linking_rows.is_empty() && linking_cols.is_empty() {
        if num_components > 1 {
            Strategy::IndependentBlocks
        } else {
            Strategy::Direct
        }
    } else if !linking_rows.is_empty() && linking_cols.is_empty() {
        Strategy::DantzigWolfe
    } else if linking_rows.is_empty() && !linking_cols.is_empty() {
        Strategy::Benders
    } else if linking_rows.len() <= linking_cols.len() {
        Strategy::DantzigWolfe
    } else {
        Strategy::Benders
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
