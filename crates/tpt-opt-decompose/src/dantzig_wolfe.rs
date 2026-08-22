//! Dantzig–Wolfe decomposition and generic column generation.
//!
//! The problem class is block-angular:
//!
//! ```text
//! min  Σ_k c_k·y_k
//! s.t. Σ_k A_{0,k} y_k ⋈ b_0          (coupling rows)
//!      G_k y_k ≥ g_k   ∀k             (block-local rows)
//!      y_k ≥ 0         ∀k
//! ```
//!
//! Each block's polyhedron `P_k` is replaced by a convex combination of its
//! extreme points ("columns"); the **restricted master problem** (RMP)
//! selects the combination weights `λ` and is re-solved while a **pricing**
//! pass searches each block for a column of negative reduced cost
//! (`c̃·y − σ_k < 0` under the RMP duals `(π, σ)`). When no block prices
//! out, the RMP value equals the true optimum (LP duality).
//!
//! The RMP is seeded with one feasible point per block (phase-1) plus
//! big-M artificials on the coupling rows, so it is always feasible and
//! artificials are priced out naturally. [`RmpPool`] manages the growing
//! column set (dedup + capacity).

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::{OptError, SolverStatus};
use tpt_opt_milp::lp::{solve_lp, LpStatus};

use crate::common::{canon_row, dot, CanonRow, RowSense};

/// One block-local row: `coeffs · y ⋈ rhs`.
#[derive(Debug, Clone)]
pub struct DwLocalRow {
    /// Coefficients over this block's variables.
    pub coeffs: Vec<f64>,
    /// Row sense.
    pub sense: RowSense,
    /// Right-hand side.
    pub rhs: f64,
}

/// One independent block.
#[derive(Debug, Clone)]
pub struct DwBlock {
    /// Block objective coefficients `c_k`.
    pub cost: Vec<f64>,
    /// Coupling-row coefficients: `coupling[r][j]` is the coefficient of
    /// this block's variable `j` in coupling row `r`.
    pub coupling: Vec<Vec<f64>>,
    /// Block-local rows.
    pub local_rows: Vec<DwLocalRow>,
}

/// A block-angular program (all variables implicitly `≥ 0`).
#[derive(Debug, Clone)]
pub struct DwProblem {
    /// Coupling right-hand sides `b_0`.
    pub coupling_rhs: Vec<f64>,
    /// Coupling row senses.
    pub coupling_sense: Vec<RowSense>,
    /// Independent blocks.
    pub blocks: Vec<DwBlock>,
}

/// One generated column: an extreme point of some block's polyhedron.
#[derive(Debug, Clone)]
pub struct Column {
    /// Owning block index.
    pub block: usize,
    /// Original-space objective contribution `c_k · y`.
    pub cost: f64,
    /// Coupling-row coefficients `A_{0,k} y`.
    pub coeffs: Vec<f64>,
    /// The extreme point itself (for solution reconstruction).
    pub point: Vec<f64>,
}

/// Restricted master problem column-pool management: near-duplicate
/// rejection and an optional capacity cap.
#[derive(Debug, Clone)]
pub struct RmpPool {
    columns: Vec<Column>,
    dedup_tol: f64,
    max_columns: Option<usize>,
}

impl Default for RmpPool {
    fn default() -> Self {
        Self { columns: Vec::new(), dedup_tol: 1e-9, max_columns: None }
    }
}

impl RmpPool {
    /// An empty pool with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the near-duplicate tolerance (points closer than this in every
    /// coordinate are considered the same column).
    pub fn with_dedup_tol(mut self, tol: f64) -> Self {
        self.dedup_tol = tol;
        self
    }

    /// Cap the number of stored columns (oldest are never evicted; once
    /// full, insertion fails).
    pub fn with_max_columns(mut self, max: usize) -> Self {
        self.max_columns = Some(max);
        self
    }

    /// Stored columns.
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Insert a column unless a near-duplicate exists or the pool is full;
    /// returns whether the column was added.
    pub fn try_insert(&mut self, col: Column) -> bool {
        let dup = self.columns.iter().any(|c| {
            c.block == col.block
                && c.point.len() == col.point.len()
                && c.point.iter().zip(&col.point).all(|(&a, &b)| (a - b).abs() <= self.dedup_tol)
        });
        if dup {
            return false;
        }
        if let Some(cap) = self.max_columns {
            if self.columns.len() >= cap {
                return false;
            }
        }
        self.columns.push(col);
        true
    }
}

/// Outcome of a Dantzig–Wolfe run.
#[derive(Debug, Clone)]
pub struct DwResult {
    /// Optimal objective (or best bound if terminated early).
    pub objective: f64,
    /// Reconstructed block solutions at the optimum.
    pub points: Vec<Vec<f64>>,
    /// Final λ weights aligned with [`DwResult::columns`].
    pub lambda: Vec<f64>,
    /// Final column set.
    pub columns: Vec<Column>,
    /// Number of pricing rounds performed.
    pub iterations: usize,
    /// [`SolverStatus::Optimal`] when pricing proved optimality;
    /// [`SolverStatus::TimeLimit`] on iteration cap;
    /// [`SolverStatus::Infeasible`] when artificials remain in the basis.
    pub status: SolverStatus,
}

/// Configurable Dantzig–Wolfe driver.
pub struct DantzigWolfe<'a> {
    problem: &'a DwProblem,
    tolerance: f64,
    max_iterations: usize,
    big_m: f64,
    pool: RmpPool,
}

impl<'a> DantzigWolfe<'a> {
    /// Create a driver with defaults (`tolerance = 1e-7`,
    /// `max_iterations = 500`, `big_m = 1e9`).
    pub fn new(problem: &'a DwProblem) -> Self {
        Self { problem, tolerance: 1e-7, max_iterations: 500, big_m: 1e9, pool: RmpPool::new() }
    }

    /// Pricing/optimality tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Iteration cap.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Big-M cost of the coupling-row artificial columns.
    pub fn with_big_m(mut self, big_m: f64) -> Self {
        self.big_m = big_m;
        self
    }

    /// Customise the restricted-master pool (dedup tolerance, capacity).
    pub fn with_pool(mut self, pool: RmpPool) -> Self {
        self.pool = pool;
        self
    }

    fn canon_block(&self, k: usize) -> Vec<CanonRow> {
        let block = &self.problem.blocks[k];
        let n_y = block.cost.len();
        let mut rows: Vec<CanonRow> = Vec::new();
        for r in &block.local_rows {
            rows.extend(canon_row(&r.coeffs, &[], r.sense, r.rhs));
        }
        // Non-negativity is implicit; nothing else to add.
        let _ = n_y;
        rows
    }

    /// Find any feasible point of block `k` via phase-1
    /// (`min Σs : G y + s ≥ g`).
    fn seed_point(&self, k: usize) -> Result<Vec<f64>, OptError> {
        let rows = self.canon_block(k);
        let n_y = self.problem.blocks[k].cost.len();
        let m = rows.len();
        let mut model = Model::new(n_y + m);
        // Rows: G y + s ≥ g.
        for (r, cr) in rows.iter().enumerate() {
            let mut idx: Vec<usize> =
                cr.a.iter().enumerate().filter(|&(_, &c)| c != 0.0).map(|(j, _)| j).collect();
            let mut co: Vec<f64> = cr.a.iter().copied().filter(|&c| c != 0.0).collect();
            idx.push(n_y + r);
            co.push(1.0);
            model.add_constraint(Constraint::ge(idx, co, cr.beta));
        }
        // min Σ s.
        let idx: Vec<usize> = (n_y..n_y + m).collect();
        let co = vec![1.0; m];
        model.set_objective(Objective {
            sense: Sense::Minimize,
            indices: idx,
            coeffs: co,
            constant: 0.0,
        });
        let lb = vec![0.0; n_y + m];
        let ub = vec![f64::INFINITY; n_y + m];
        let sol = solve_lp(&model, &lb, &ub, tpt_opt_core::tolerance::Tolerances::spec_default());
        match sol.status {
            LpStatus::Optimal => Ok(sol.x[..n_y].to_vec()),
            _ => Err(OptError::invalid_model("block polyhedron empty or unbounded")),
        }
    }

    /// Price block `k` under RMP duals: minimise
    /// `(c_k − A₀ᵀπ)·y` over `P_k`; returns `(reduced_cost, column)` when
    /// the reduced cost is below `-tolerance`.
    fn price_block(
        &self,
        k: usize,
        pi: &[f64],
        sigma_k: f64,
    ) -> Result<Option<(f64, Column)>, OptError> {
        let block = &self.problem.blocks[k];
        let n_y = block.cost.len();
        let mut price_cost = vec![0.0f64; n_y];
        for j in 0..n_y {
            price_cost[j] = block.cost[j]
                - pi.iter().zip(block.coupling.iter()).map(|(&p, row)| p * row[j]).sum::<f64>();
        }
        let rows = self.canon_block(k);
        let mut model = Model::new(n_y);
        for v in model.variables.iter_mut() {
            v.bound = tpt_opt_core::VarBound::continuous(0.0, f64::INFINITY);
        }
        for cr in &rows {
            let idx: Vec<usize> =
                cr.a.iter().enumerate().filter(|&(_, &c)| c != 0.0).map(|(j, _)| j).collect();
            let co: Vec<f64> = cr.a.iter().copied().filter(|&c| c != 0.0).collect();
            model.add_constraint(Constraint::ge(idx, co, cr.beta));
        }
        let oidx: Vec<usize> = (0..n_y).filter(|&j| price_cost[j] != 0.0).collect();
        let ocoeffs: Vec<f64> = oidx.iter().map(|&j| price_cost[j]).collect();
        model.set_objective(Objective {
            sense: Sense::Minimize,
            indices: oidx,
            coeffs: ocoeffs,
            constant: 0.0,
        });
        let lb = vec![0.0; n_y];
        let ub = vec![f64::INFINITY; n_y];
        let sol = solve_lp(&model, &lb, &ub, tpt_opt_core::tolerance::Tolerances::spec_default());
        if sol.status != LpStatus::Optimal {
            return Err(OptError::invalid_model("pricing LP failed"));
        }
        let rc = sol.objective - sigma_k;
        if rc >= -self.tolerance {
            return Ok(None);
        }
        let y = sol.x;
        let coeffs: Vec<f64> = block.coupling.iter().map(|row| dot(row, &y)).collect();
        let cost = dot(&block.cost, &y);
        Ok(Some((rc, Column { block: k, cost, coeffs, point: y })))
    }

    /// Run the column-generation loop.
    pub fn solve(mut self) -> Result<DwResult, OptError> {
        let m_couple = self.problem.coupling_rhs.len();
        let k_count = self.problem.blocks.len();

        // Seed: one feasible point per block + big-M artificials.
        for k in 0..k_count {
            let y0 = self.seed_point(k)?;
            let block = &self.problem.blocks[k];
            let coeffs: Vec<f64> = block.coupling.iter().map(|row| dot(row, &y0)).collect();
            self.pool.try_insert(Column {
                block: k,
                cost: dot(&block.cost, &y0),
                coeffs,
                point: y0,
            });
        }

        let mut iterations = 0usize;
        let (objective, lambda, status) = loop {
            if iterations >= self.max_iterations {
                break (f64::INFINITY, Vec::new(), SolverStatus::TimeLimit);
            }
            iterations += 1;

            // ---- Build the RMP -------------------------------------------
            let ncols = self.pool.columns().len();
            let mut model = Model::new(ncols + 2 * m_couple);
            for v in model.variables.iter_mut() {
                v.bound = tpt_opt_core::VarBound::continuous(0.0, f64::INFINITY);
            }
            // Artificial columns: ±e_r with big-M cost (absorb any violation).
            for r in 0..m_couple {
                model.variables[ncols + 2 * r].bound =
                    tpt_opt_core::VarBound::continuous(0.0, f64::INFINITY);
                model.variables[ncols + 2 * r + 1].bound =
                    tpt_opt_core::VarBound::continuous(0.0, f64::INFINITY);
            }
            let mut oidx: Vec<usize> = Vec::new();
            let mut ocoeffs: Vec<f64> = Vec::new();
            for (c, col) in self.pool.columns().iter().enumerate() {
                if col.cost != 0.0 {
                    oidx.push(c);
                    ocoeffs.push(col.cost);
                }
            }
            for a in 0..(2 * m_couple) {
                oidx.push(ncols + a);
                ocoeffs.push(self.big_m);
            }
            model.set_objective(Objective {
                sense: Sense::Minimize,
                indices: oidx,
                coeffs: ocoeffs,
                constant: 0.0,
            });
            // Coupling rows over columns (+ artificials).
            for r in 0..m_couple {
                let mut idx: Vec<usize> = Vec::new();
                let mut co: Vec<f64> = Vec::new();
                for (c, col) in self.pool.columns().iter().enumerate() {
                    let a_rc = col.coeffs.get(r).copied().unwrap_or(0.0);
                    if a_rc != 0.0 {
                        idx.push(c);
                        co.push(a_rc);
                    }
                }
                idx.push(ncols + 2 * r);
                co.push(1.0);
                idx.push(ncols + 2 * r + 1);
                co.push(-1.0);
                let con = match self.problem.coupling_sense[r] {
                    RowSense::Le => Constraint::le(idx, co, self.problem.coupling_rhs[r]),
                    RowSense::Ge => Constraint::ge(idx, co, self.problem.coupling_rhs[r]),
                    RowSense::Eq => Constraint::equality(idx, co, self.problem.coupling_rhs[r]),
                };
                model.add_constraint(con);
            }
            // Convexity rows: Σ_{c ∈ k} λ_c = 1.
            for k in 0..k_count {
                let idx: Vec<usize> =
                    (0..ncols).filter(|&c| self.pool.columns()[c].block == k).collect();
                let co = vec![1.0; idx.len()];
                model.add_constraint(Constraint::equality(idx, co, 1.0));
            }

            let lb = vec![0.0; ncols + 2 * m_couple];
            let ub = vec![f64::INFINITY; ncols + 2 * m_couple];
            let sol =
                solve_lp(&model, &lb, &ub, tpt_opt_core::tolerance::Tolerances::spec_default());
            if sol.status != LpStatus::Optimal {
                return Err(OptError::invalid_model("restricted master LP failed"));
            }
            let duals = sol.dual.clone();
            let pi = &duals[..m_couple];
            let sigma = &duals[m_couple..m_couple + k_count];

            // ---- Pricing ---------------------------------------------------
            let mut added = false;
            for (k, &sk) in sigma.iter().enumerate() {
                if let Some((_rc, col)) = self.price_block(k, pi, sk)? {
                    added |= self.pool.try_insert(col);
                }
            }
            // Artificial usage with no fresh column ⇒ the coupling demand
            // cannot be met by any block polyhedron point: infeasible.
            // (Pricing may keep re-proposing an existing column; only newly
            // inserted columns can restore feasibility.)
            let art_use: f64 = sol.x[ncols..].iter().sum();
            if art_use > 1e-6 && !added {
                break (sol.objective, sol.x[..ncols].to_vec(), SolverStatus::Infeasible);
            }
            if !added {
                // No negative reduced costs anywhere: optimal.
                break (sol.objective, sol.x[..ncols].to_vec(), SolverStatus::Optimal);
            }
        };

        // Reconstruct block points from λ.
        let mut points = vec![Vec::new(); k_count];
        for (lam, col) in lambda.iter().zip(self.pool.columns()) {
            if *lam > 1e-12 {
                if points[col.block].is_empty() {
                    points[col.block] = vec![0.0; col.point.len()];
                }
                for (pj, &pv) in points[col.block].iter_mut().zip(&col.point) {
                    *pj += lam * pv;
                }
            }
        }

        Ok(DwResult {
            objective,
            points,
            lambda,
            columns: self.pool.columns().to_vec(),
            iterations,
            status,
        })
    }
}
