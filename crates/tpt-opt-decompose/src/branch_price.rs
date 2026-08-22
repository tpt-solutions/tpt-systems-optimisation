//! Branch-and-price: column generation embedded in branch-and-bound over
//! the master variables.
//!
//! The master is the same restricted master used by
//! [`crate::dantzig_wolfe`] (coupling rows + optional convexity rows), but
//! the λ variables are declared **integer**. Pricing is pluggable through
//! the [`Pricer`] trait so integer pricing subproblems (e.g. knapsacks for
//! cutting stock) can be supplied; [`LpPricer`] provides the continuous-LP
//! default mirroring plain Dantzig–Wolfe.
//!
//! Branching is performed on fractional **master** variables (`λ_c ≤ ⌊v⌋`
//! vs `λ_c ≥ ⌈v⌉`, depth-first). This is valid because pricing can
//! regenerate any column the branching rules exclude from the current pool;
//! note that, as with any master-variable branching scheme, integrality of
//! the *extensive-form* solution follows from the master being solved to
//! integrality over a complete-enough column set.

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::{OptError, SolverStatus};
use tpt_opt_milp::lp::{solve_lp, LpStatus};

use crate::common::{canon_row, dot, CanonRow, RowSense};
use crate::dantzig_wolfe::{Column, DwProblem, RmpPool};

/// Pricing oracle: given the current RMP duals, return the most negative
/// reduced-cost column for `block`, or `None` if none exists.
pub trait Pricer {
    /// Price one block under duals `(π, σ_k)`; reduced cost convention
    /// matches [`crate::dantzig_wolfe`].
    fn price(
        &mut self,
        block: usize,
        pi: &[f64],
        sigma_k: f64,
    ) -> Result<Option<Column>, OptError>;
}

/// Continuous-LP pricer over a [`DwProblem`]'s block polyhedra (the same
/// pricing LP plain Dantzig–Wolfe uses).
pub struct LpPricer<'a> {
    problem: &'a DwProblem,
}

impl<'a> LpPricer<'a> {
    /// Build a pricer over the given problem.
    pub fn new(problem: &'a DwProblem) -> Self {
        Self { problem }
    }

    fn canon_block(&self, k: usize) -> Vec<CanonRow> {
        self.problem.blocks[k]
            .local_rows
            .iter()
            .flat_map(|r| canon_row(&r.coeffs, &[], r.sense, r.rhs))
            .collect()
    }
}

impl Pricer for LpPricer<'_> {
    fn price(
        &mut self,
        k: usize,
        pi: &[f64],
        sigma_k: f64,
    ) -> Result<Option<Column>, OptError> {
        let block = &self.problem.blocks[k];
        let n_y = block.cost.len();
        let mut price_cost = vec![0.0f64; n_y];
        for j in 0..n_y {
            price_cost[j] =
                block.cost[j] - pi.iter().zip(block.coupling.iter()).map(|(&p, row)| p * row[j]).sum::<f64>();
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
        if rc >= -1e-7 {
            return Ok(None);
        }
        let y = sol.x;
        Ok(Some(Column {
            block: k,
            cost: dot(&block.cost, &y),
            coeffs: block.coupling.iter().map(|row| dot(row, &y)).collect(),
            point: y,
        }))
    }
}

/// Outcome of a branch-and-price run.
#[derive(Debug, Clone)]
pub struct BpResult {
    /// Best integer objective found.
    pub objective: f64,
    /// Reconstructed block solutions at the incumbent.
    pub points: Vec<Vec<f64>>,
    /// Final column pool size.
    pub columns: usize,
    /// Nodes explored.
    pub nodes: usize,
    /// [`SolverStatus::Optimal`] when the tree was exhausted/pruned to
    /// proven optimality; [`SolverStatus::TimeLimit`] on node cap.
    pub status: SolverStatus,
}

/// Branch-and-price driver.
pub struct BranchAndPrice<'a, P: Pricer> {
    problem: &'a DwProblem,
    pricer: P,
    tolerance: f64,
    max_nodes: usize,
    big_m: f64,
    convexity: bool,
}

impl<'a, P: Pricer> BranchAndPrice<'a, P> {
    /// Create a driver over `problem` with the given pricer.
    pub fn new(problem: &'a DwProblem, pricer: P) -> Self {
        Self {
            problem,
            pricer,
            tolerance: 1e-6,
            max_nodes: 2000,
            big_m: 1e9,
            convexity: true,
        }
    }

    /// Optimality tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Node cap.
    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes;
        self
    }

    /// Big-M for coupling-row artificials.
    pub fn with_big_m(mut self, big_m: f64) -> Self {
        self.big_m = big_m;
        self
    }

    /// Disable per-block convexity rows (set-partitioning style masters
    /// such as cutting stock where blocks have unbounded multiplicity).
    pub fn with_convexity(mut self, convexity: bool) -> Self {
        self.convexity = convexity;
        self
    }

    fn seed_pool(&self, pool: &mut RmpPool) -> Result<(), OptError> {
        for k in 0..self.problem.blocks.len() {
            // Phase-1 feasible point.
            let block = &self.problem.blocks[k];
            let n_y = block.cost.len();
            let rows: Vec<CanonRow> = block
                .local_rows
                .iter()
                .flat_map(|r| canon_row(&r.coeffs, &[], r.sense, r.rhs))
                .collect();
            let m = rows.len();
            let mut model = Model::new(n_y + m);
            for (r, cr) in rows.iter().enumerate() {
                let mut idx: Vec<usize> =
                    cr.a.iter().enumerate().filter(|&(_, &c)| c != 0.0).map(|(j, _)| j).collect();
                let mut co: Vec<f64> = cr.a.iter().copied().filter(|&c| c != 0.0).collect();
                idx.push(n_y + r);
                co.push(1.0);
                model.add_constraint(Constraint::ge(idx, co, cr.beta));
            }
            let idx: Vec<usize> = (n_y..n_y + m).collect();
            model.set_objective(Objective {
                sense: Sense::Minimize,
                indices: idx,
                coeffs: vec![1.0; m],
                constant: 0.0,
            });
            let lb = vec![0.0; n_y + m];
            let ub = vec![f64::INFINITY; n_y + m];
            let sol = solve_lp(&model, &lb, &ub, tpt_opt_core::tolerance::Tolerances::spec_default());
            if sol.status != LpStatus::Optimal {
                return Err(OptError::invalid_model("block polyhedron empty"));
            }
            let y0 = sol.x[..n_y].to_vec();
            pool.try_insert(Column {
                block: k,
                cost: dot(&block.cost, &y0),
                coeffs: block.coupling.iter().map(|row| dot(row, &y0)).collect(),
                point: y0,
            });
        }
        Ok(())
    }

    /// Column generation at a node until no negative reduced costs remain.
    /// Returns `(lp_objective, lp_lambda)`.
    fn generate_columns(
        &mut self,
        pool: &mut RmpPool,
        restrictions: &[(usize, f64, f64)],
    ) -> Result<(f64, Vec<f64>), OptError> {
        let m_couple = self.problem.coupling_rhs.len();
        let k_count = self.problem.blocks.len();
        loop {
            let ncols = pool.columns().len();
            let mut model = Model::new(ncols + 2 * m_couple);
            for (i, v) in model.variables.iter_mut().enumerate() {
                let (lo, hi) = if i < ncols {
                    restrictions
                        .iter()
                        .find(|&&(c, _, _)| c == i)
                        .map_or((0.0, f64::INFINITY), |&(_, l, u)| (l, u))
                } else {
                    (0.0, f64::INFINITY)
                };
                v.bound = tpt_opt_core::VarBound::continuous(lo, hi);
            }
            let mut oidx: Vec<usize> = Vec::new();
            let mut ocoeffs: Vec<f64> = Vec::new();
            for (c, col) in pool.columns().iter().enumerate() {
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
            for r in 0..m_couple {
                let mut idx: Vec<usize> = Vec::new();
                let mut co: Vec<f64> = Vec::new();
                for (c, col) in pool.columns().iter().enumerate() {
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
            if self.convexity {
                for k in 0..k_count {
                    let idx: Vec<usize> =
                        (0..ncols).filter(|&c| pool.columns()[c].block == k).collect();
                    if idx.is_empty() {
                        continue;
                    }
                    let co = vec![1.0; idx.len()];
                    model.add_constraint(Constraint::equality(idx, co, 1.0));
                }
            }
            let lb = vec![0.0; ncols + 2 * m_couple];
            let ub = vec![f64::INFINITY; ncols + 2 * m_couple];
            let sol =
                solve_lp(&model, &lb, &ub, tpt_opt_core::tolerance::Tolerances::spec_default());
            if sol.status != LpStatus::Optimal {
                return Err(OptError::invalid_model("node RMP failed"));
            }
            let duals = sol.dual.clone();
            let pi = &duals[..m_couple];
            let sigma: Vec<f64> = if self.convexity {
                duals[m_couple..m_couple + k_count].to_vec()
            } else {
                vec![0.0; k_count]
            };

            let mut added = false;
            for k in 0..k_count {
                if let Some(col) = self.pricer.price(k, pi, sigma[k])? {
                    added |= pool.try_insert(col);
                }
            }
            if !added {
                return Ok((sol.objective, sol.primal[..ncols].to_vec()));
            }
        }
    }

    /// Run branch-and-price.
    pub fn solve(self) -> Result<BpResult, OptError> {
        let k_count = self.problem.blocks.len();
        let mut pool = RmpPool::new();
        self.seed_pool(&mut pool)?;

        let mut best_obj = f64::INFINITY;
        let mut best_points = vec![Vec::new(); k_count];
        let mut nodes = 0usize;
        let mut proven = true;

        // DFS stack of restriction sets.
        let mut stack: Vec<Vec<(usize, f64, f64)>> = vec![Vec::new()];
        while let Some(restrictions) = stack.pop() {
            nodes += 1;
            if nodes > self.max_nodes {
                proven = false;
                break;
            }
            let (lp_obj, lambda) = self.generate_columns(&mut pool, &restrictions)?;
            if lp_obj >= best_obj - self.tolerance {
                continue; // bound prune
            }
            // Integral λ ⇒ incumbent candidate.
            let integral =
                lambda.iter().all(|&v| (v - v.round()).abs() <= 1e-6);
            if integral {
                if lp_obj < best_obj {
                    best_obj = lp_obj;
                    best_points = vec![Vec::new(); k_count];
                    for (lam, col) in lambda.iter().zip(pool.columns()) {
                        if *lam > 0.5 {
                            best_points[col.block] = col.point.clone();
                        }
                    }
                }
                continue;
            }
            // Branch on the most fractional λ.
            let mut branch = (usize::MAX, 0.5f64);
            for (c, &v) in lambda.iter().enumerate() {
                let frac = (v - v.floor()).abs();
                if frac > 1e-6 && frac < 1.0 - 1e-6 && (frac - 0.5).abs() < (branch.1 - 0.5).abs()
                {
                    branch = (c, v);
                }
            }
            if branch.0 == usize::MAX {
                continue;
            }
            let (c, v) = branch;
            let mut left = restrictions.clone();
            left.retain(|&(cc, _, _)| cc != c);
            left.push((c, 0.0, v.floor()));
            let mut right = restrictions.clone();
            right.retain(|&(cc, _, _)| cc != c);
            right.push((c, v.ceil(), f64::INFINITY));
            stack.push(left);
            stack.push(right);
        }

        Ok(BpResult {
            objective: best_obj,
            points: best_points,
            columns: pool.columns().len(),
            nodes,
            status: if proven { SolverStatus::Optimal } else { SolverStatus::TimeLimit },
        })
    }
}

