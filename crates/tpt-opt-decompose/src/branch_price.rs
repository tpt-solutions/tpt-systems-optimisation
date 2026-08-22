//! Branch-and-price via the classic **price-and-branch** scheme.
//!
//! Column generation runs first (over the same restricted master as
//! [`crate::dantzig_wolfe`], with a pluggable [`Pricer`] — integer knapsack
//! pricing for cutting-stock-style masters, or the continuous-LP
//! [`LpPricer`]); once pricing proves no negative reduced costs remain, the
//! accumulated column pool is solved as a single **integer** master with
//! the bundled MILP engine. This is the standard practical variant of
//! branch-and-price: it forgoes per-node regeneration in exchange for
//! robustness, and is exact whenever the final pool contains an optimal
//! integer combination (as it does for most set-partitioning/covering
//! masters after CG converges).

use std::vec::Vec;

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::{OptError, SolverStatus, VarBound};
use tpt_opt_milp::lp::{solve_lp, LpStatus};
use tpt_opt_milp::MilpSolver;

use crate::common::{canon_row, dot, CanonRow, RowSense};
use crate::dantzig_wolfe::{Column, DwProblem, RmpPool};

/// Pricing oracle: given the current RMP duals, return the most negative
/// reduced-cost column for `block`, or `None` if none exists.
pub trait Pricer {
    /// Price one block under duals `(π, σ_k)`; reduced cost convention
    /// matches [`crate::dantzig_wolfe`].
    fn price(&mut self, block: usize, pi: &[f64], sigma_k: f64)
        -> Result<Option<Column>, OptError>;

    /// Return *all* improving columns (`rc < −tol`) for the block. Used for
    /// the per-round generation pass; enumerating pricers (knapsack
    /// solvers) should override this to dump several columns at once.
    /// Defaults to the single best column from [`Pricer::price`].
    fn price_batch(
        &mut self,
        block: usize,
        pi: &[f64],
        sigma_k: f64,
    ) -> Result<Vec<Column>, OptError> {
        Ok(self.price(block, pi, sigma_k)?.into_iter().collect())
    }

    /// One-shot **cleanup** pass invoked when regular pricing finds no
    /// column with `rc < −tol`. May return columns with reduced cost up to
    /// a small *positive* slack: such dual-neutral columns are irrelevant
    /// to the LP but frequently pivotal for the integer master (classic
    /// cutting-stock tailing-off). Default: no cleanup columns.
    fn price_cleanup(
        &mut self,
        _block: usize,
        _pi: &[f64],
        _sigma_k: f64,
    ) -> Result<Vec<Column>, OptError> {
        Ok(Vec::new())
    }
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
    fn price(&mut self, k: usize, pi: &[f64], sigma_k: f64) -> Result<Option<Column>, OptError> {
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
            v.bound = VarBound::continuous(0.0, f64::INFINITY);
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
    /// Pricing rounds performed during column generation.
    pub pricing_rounds: usize,
    /// [`SolverStatus::Optimal`] when the integer master was solved to
    /// proven optimality over the pool; [`SolverStatus::Infeasible`] when
    /// the coupling demand cannot be met by any generated column.
    pub status: SolverStatus,
}

/// Branch-and-price driver.
pub struct BranchAndPrice<'a, P: Pricer> {
    problem: &'a DwProblem,
    pricer: P,
    tolerance: f64,
    max_pricing_rounds: usize,
    big_m: f64,
    convexity: bool,
}

impl<'a, P: Pricer> BranchAndPrice<'a, P> {
    /// Create a driver over `problem` with the given pricer.
    pub fn new(problem: &'a DwProblem, pricer: P) -> Self {
        Self {
            problem,
            pricer,
            tolerance: 1e-7,
            max_pricing_rounds: 500,
            big_m: 1e9,
            convexity: true,
        }
    }

    /// Pricing/optimality tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Cap on column-generation rounds.
    pub fn with_max_pricing_rounds(mut self, max_rounds: usize) -> Self {
        self.max_pricing_rounds = max_rounds;
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
            // Phase-1 feasible point of the block polyhedron.
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
            let sol =
                solve_lp(&model, &lb, &ub, tpt_opt_core::tolerance::Tolerances::spec_default());
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

    /// Column generation until no negative reduced costs remain (or the
    /// round cap / infeasibility is hit). Returns the final LP objective,
    /// λ, feasibility flag, and the number of pricing rounds performed.
    fn generate_columns(
        &mut self,
        pool: &mut RmpPool,
    ) -> Result<(f64, Vec<f64>, bool, usize), OptError> {
        let m_couple = self.problem.coupling_rhs.len();
        let k_count = self.problem.blocks.len();
        let mut rounds = 0usize;
        loop {
            let ncols = pool.columns().len();
            let mut model = Model::new(ncols + 2 * m_couple);
            for v in model.variables.iter_mut() {
                v.bound = VarBound::continuous(0.0, f64::INFINITY);
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
                return Err(OptError::invalid_model("restricted master LP failed"));
            }
            let duals = sol.dual.clone();
            let pi = &duals[..m_couple];
            let sigma: Vec<f64> = if self.convexity {
                duals[m_couple..m_couple + k_count].to_vec()
            } else {
                vec![0.0; k_count]
            };

            let mut added = false;
            for (k, &sk) in sigma.iter().enumerate() {
                for col in self.pricer.price_batch(k, pi, sk)? {
                    added |= pool.try_insert(col);
                }
            }
            if !added {
                // Converged for the LP: harvest dual-neutral columns once,
                // then stop.
                for (k, &sk) in sigma.iter().enumerate() {
                    for col in self.pricer.price_cleanup(k, pi, sk)? {
                        added |= pool.try_insert(col);
                    }
                }
            }
            let art_use: f64 = sol.x[ncols..].iter().sum();
            if art_use > 1e-6 && !added {
                // Coupling demand unmeetable by any block point.
                return Ok((sol.objective, sol.x[..ncols].to_vec(), false, rounds));
            }
            if !added || rounds >= self.max_pricing_rounds {
                return Ok((sol.objective, sol.x[..ncols].to_vec(), true, rounds));
            }
            rounds += 1;
        }
    }

    /// Run price-and-branch.
    pub fn solve(mut self) -> Result<BpResult, OptError> {
        use tpt_opt_core::solver::Solver;
        let k_count = self.problem.blocks.len();
        let m_couple = self.problem.coupling_rhs.len();
        let mut pool = RmpPool::new();
        self.seed_pool(&mut pool)?;

        let (_, _, feasible, rounds) = self.generate_columns(&mut pool)?;
        if !feasible {
            return Ok(BpResult {
                objective: f64::INFINITY,
                points: vec![Vec::new(); k_count],
                columns: pool.columns().len(),
                pricing_rounds: rounds,
                status: SolverStatus::Infeasible,
            });
        }

        // Integer master over the generated pool.
        let ncols = pool.columns().len();
        let mut model = Model::new(ncols);
        for v in model.variables.iter_mut() {
            v.bound = VarBound::integer(0.0, f64::INFINITY);
        }
        let oidx: Vec<usize> = (0..ncols).filter(|&c| pool.columns()[c].cost != 0.0).collect();
        let ocoeffs: Vec<f64> = oidx.iter().map(|&c| pool.columns()[c].cost).collect();
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
        let sol = MilpSolver::new().solve(&model)?;
        let status = sol.status;
        let mut points = vec![Vec::new(); k_count];
        if status == SolverStatus::Optimal {
            for (c, col) in pool.columns().iter().enumerate() {
                if sol.primal[c] > 0.5 {
                    points[col.block] = col.point.clone();
                }
            }
        }
        Ok(BpResult {
            objective: sol.objective_value,
            points,
            columns: ncols,
            pricing_rounds: rounds,
            status,
        })
    }
}
