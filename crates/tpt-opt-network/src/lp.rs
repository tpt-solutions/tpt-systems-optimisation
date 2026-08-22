//! A small, self-contained two-phase simplex LP solver.
//!
//! This is intentionally minimal (dense tableau, Bland's rule to avoid cycling)
//! and exists so that `tpt-opt-network` can solve DC-OPF and SC-OPF without an
//! external LP dependency. It implements [`tpt_opt_core::solver::Solver`] over
//! the canonical [`tpt_opt_core::model::Model`], so DC-OPF is expressed as a
//! `Model` and handed to this solver.

use std::vec::Vec;

use tpt_opt_core::model::Model;
use tpt_opt_core::solver::{Solution, SolveParameters, Solver, SolverStatus};
use tpt_opt_core::{OptError, Sense};

/// Column metadata for the standard-form tableau.
struct Col {
    /// Phase-II (real objective) cost of this column.
    phase2: f64,
    /// `true` for the artificial variables used to find an initial basis.
    artificial: bool,
}

/// A single standard-form row before tabulation.
struct RowSpec {
    coeffs: Vec<(usize, f64)>,
    rhs: f64,
    /// Column that is basic in this row in the initial canonical basis.
    basic: usize,
}

/// Result of converting a [`Model`] into standard form.
struct StandardForm {
    cols: Vec<Col>,
    rows: Vec<RowSpec>,
    /// For each original variable: `(constant, [(new_col, coeff)])` expression.
    orig_expr: Vec<(f64, Vec<(usize, f64)>)>,
    /// Constant accumulated into the real objective from substitutions.
    obj_const: f64,
    /// Real objective sign: `+1.0` for minimise, `-1.0` for maximise.
    sign: f64,
}

/// An LP solver over the canonical [`Model`].
///
/// Implements [`tpt_opt_core::solver::Solver`] so that formulations such as
/// DC-OPF (which build a `Model`) can be solved uniformly.
pub struct LpSolver {
    params: SolveParameters,
    status: SolverStatus,
    last: Option<Solution>,
}

impl LpSolver {
    /// Create a new solver with default parameters.
    pub fn new() -> Self {
        Self { params: SolveParameters::defaults(), status: SolverStatus::Error, last: None }
    }

    /// Solve a pre-built standard form, returning the primal and objective.
    fn solve_standard(&mut self, sf: &StandardForm) -> Result<Solution, OptError> {
        let m = sf.rows.len();
        let ncols = sf.cols.len();
        let rhs = ncols;

        if m == 0 || ncols == 0 {
            // Degenerate: nothing to solve.
            let primal = vec![0.0; sf.orig_expr.len()];
            return Ok(Solution::new(primal, sf.obj_const, SolverStatus::Optimal));
        }

        // Build the tableau.
        let mut tab: Vec<Vec<f64>> = vec![vec![0.0; ncols + 1]; m];
        let mut basis: Vec<usize> = vec![0; m];
        for (i, row) in sf.rows.iter().enumerate() {
            for &(c, v) in &row.coeffs {
                tab[i][c] = v;
            }
            tab[i][rhs] = row.rhs;
            basis[i] = row.basic;
        }

        let eps = self.params.tolerances.feasibility.max(1e-9);
        let mut c1 = vec![0.0f64; ncols];
        let mut c2 = vec![0.0f64; ncols];
        for (j, col) in sf.cols.iter().enumerate() {
            c1[j] = if col.artificial { 1.0 } else { 0.0 };
            c2[j] = col.phase2;
        }

        // Phase I.
        let phase1 = run_simplex(&mut tab, &mut basis, &c1, m, ncols, eps);
        if phase1 != SolverStatus::Optimal {
            self.status = phase1;
            return Err(OptError::numerical("phase I did not terminate"));
        }
        let phase1_obj = objective_value(&tab, &basis, &c1, m, ncols);
        if phase1_obj > eps {
            self.status = SolverStatus::Infeasible;
            return Err(OptError::infeasible(tpt_opt_core::InfeasibilityReport::new(
                "LP is infeasible (phase I > 0)",
            )));
        }

        // Try to pivot remaining basic artificials out of the basis.
        for i in 0..m {
            if sf.cols[basis[i]].artificial {
                if tab[i][rhs] > eps {
                    self.status = SolverStatus::Infeasible;
                    return Err(OptError::infeasible(tpt_opt_core::InfeasibilityReport::new(
                        "LP is infeasible (artificial remains basic at positive level)",
                    )));
                }
                // Find a non-artificial non-basic column to pivot in.
                let mut found = None;
                for j in 0..ncols {
                    if sf.cols[j].artificial {
                        continue;
                    }
                    if tab[i][j].abs() > eps {
                        found = Some(j);
                        break;
                    }
                }
                if let Some(j) = found {
                    pivot(&mut tab, &mut basis, i, j, m, ncols);
                }
            }
        }

        // Phase II.
        let phase2 = run_simplex(&mut tab, &mut basis, &c2, m, ncols, eps);
        self.status = phase2;
        if phase2 == SolverStatus::Unbounded {
            return Err(OptError::Unbounded);
        }
        if phase2 != SolverStatus::Optimal {
            return Err(OptError::numerical("phase II did not converge"));
        }

        if !tab.iter().all(|r| r.iter().all(|v| v.is_finite())) {
            self.status = SolverStatus::NumericalIssue;
            return Err(OptError::numerical("non-finite tableau entries"));
        }

        // Recover the primal solution in the substituted variables.
        let mut xnew = vec![0.0f64; ncols];
        for i in 0..m {
            xnew[basis[i]] = tab[i][rhs];
        }

        // Map back to the original variables.
        let mut primal = vec![0.0f64; sf.orig_expr.len()];
        for (v, (con, expr)) in sf.orig_expr.iter().enumerate() {
            let mut val = *con;
            for &(c, coeff) in expr {
                val += coeff * xnew[c];
            }
            primal[v] = val;
        }

        let tableau_obj = objective_value(&tab, &basis, &c2, m, ncols);
        let obj = sf.sign * tableau_obj + sf.obj_const;

        let sol = Solution::new(primal, obj, SolverStatus::Optimal);
        Ok(sol)
    }
}

impl Default for LpSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver<Model> for LpSolver {
    fn solve(&mut self, model: &Model) -> Result<Solution, OptError> {
        let sf = to_standard_form(model)?;
        let sol = self.solve_standard(&sf)?;
        self.last = Some(sol.clone());
        Ok(sol)
    }

    fn set_parameter(&mut self, param: &SolveParameters) -> Result<(), OptError> {
        self.params = *param;
        Ok(())
    }

    fn warm_start(&mut self, _warm: tpt_opt_core::solver::WarmStart) -> Result<(), OptError> {
        // The simplex does not currently exploit a supplied basis; the request
        // is accepted (best-effort) so the contract is honoured.
        Ok(())
    }

    fn status(&self) -> SolverStatus {
        self.status
    }

    fn solution(&self) -> Option<Solution> {
        self.last.clone()
    }
}

/// Convert a [`Model`] into a standard form suitable for the two-phase simplex.
fn to_standard_form(model: &Model) -> Result<StandardForm, OptError> {
    let n = model.num_vars;
    if model.variables.len() != n {
        return Err(OptError::invalid_model("variable count mismatch in model"));
    }

    // Original objective coefficients (before sign flip).
    let mut orig_obj = vec![0.0f64; n];
    for (&i, &c) in model.objective.indices.iter().zip(model.objective.coeffs.iter()) {
        if i >= n {
            return Err(OptError::invalid_model("objective references out-of-range variable"));
        }
        orig_obj[i] += c;
    }
    let sign = match model.objective.sense {
        Sense::Minimize => 1.0,
        Sense::Maximize => -1.0,
    };

    let mut cols: Vec<Col> = Vec::new();
    let mut orig_expr: Vec<(f64, Vec<(usize, f64)>)> = Vec::with_capacity(n);
    let mut bound_rows: Vec<(Vec<(usize, f64)>, f64)> = Vec::new();
    let mut obj_const = model.objective.constant;

    for (var, &c) in model.variables.iter().zip(&orig_obj) {
        let lb = var.bound.bound.lower;
        let ub = var.bound.bound.upper;
        let phase2 = sign * c;

        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();

        if lb_finite && ub_finite {
            let y = cols.len();
            cols.push(Col { phase2, artificial: false });
            orig_expr.push((lb, vec![(y, 1.0)]));
            obj_const += c * lb;
            let span = ub - lb;
            bound_rows.push((vec![(y, 1.0)], span));
        } else if lb_finite && !ub_finite {
            let y = cols.len();
            cols.push(Col { phase2, artificial: false });
            orig_expr.push((lb, vec![(y, 1.0)]));
            obj_const += c * lb;
        } else if !lb_finite && ub_finite {
            let yp = cols.len();
            cols.push(Col { phase2, artificial: false });
            let ym = cols.len();
            cols.push(Col { phase2: -phase2, artificial: false });
            bound_rows.push((vec![(yp, 1.0), (ym, -1.0)], ub));
            orig_expr.push((0.0, vec![(yp, 1.0), (ym, -1.0)]));
        } else {
            let yp = cols.len();
            cols.push(Col { phase2, artificial: false });
            let ym = cols.len();
            cols.push(Col { phase2: -phase2, artificial: false });
            orig_expr.push((0.0, vec![(yp, 1.0), (ym, -1.0)]));
        }
    }

    let mut rows: Vec<RowSpec> = Vec::new();

    // Variable bound rows (all are <= constraints with a slack basic var).
    for (coeffs, rhs) in &bound_rows {
        let mut c = coeffs.clone();
        cols.push(Col { phase2: 0.0, artificial: false });
        // slack appears in the deps graph but we register it as a new column.
        let slack = cols.len() - 1;
        c.push((slack, 1.0));
        rows.push(RowSpec { coeffs: c, rhs: *rhs, basic: slack });
    }

    // Model constraints.
    for con in &model.constraints {
        if con.indices.len() != con.coeffs.len() {
            return Err(OptError::invalid_model("constraint indices/coeffs length mismatch"));
        }
        // Build the base coefficient vector in the substituted space.
        let mut base: Vec<(usize, f64)> = Vec::new();
        let mut const_term = 0.0f64;
        for (&ov, &coef) in con.indices.iter().zip(con.coeffs.iter()) {
            if ov >= n {
                return Err(OptError::invalid_model("constraint references out-of-range variable"));
            }
            let (con_val, expr) = &orig_expr[ov];
            const_term += coef * *con_val;
            for &(nc, f) in expr {
                base.push((nc, coef * f));
            }
        }

        let lo = con.lower;
        let hi = con.upper;
        let lo_finite = lo.is_finite();
        let hi_finite = hi.is_finite();

        if lo_finite {
            // >= lo : A x - s + a = (lo - const), surplus s and artificial a.
            let mut c = base.clone();
            let surplus = cols.len();
            cols.push(Col { phase2: 0.0, artificial: false });
            c.push((surplus, -1.0));
            let artificial = cols.len();
            cols.push(Col { phase2: 0.0, artificial: true });
            c.push((artificial, 1.0));
            rows.push(RowSpec { coeffs: c, rhs: lo - const_term, basic: artificial });
        }

        if hi_finite {
            let equality = lo_finite && (hi - lo).abs() <= 1e-12;
            if !equality {
                // <= hi : A x + s = (hi - const), slack s basic.
                let mut c = base.clone();
                let slack = cols.len();
                cols.push(Col { phase2: 0.0, artificial: false });
                c.push((slack, 1.0));
                rows.push(RowSpec { coeffs: c, rhs: hi - const_term, basic: slack });
            }
        }
    }

    Ok(StandardForm { cols, rows, orig_expr, obj_const, sign })
}

/// Reduced-cost row for the current basis, used by the simplex loop.
///
/// Returns the *true* reduced costs `d_j = c_j - c_B^T B^{-1} A_j`, so the
/// caller's entering rule (`d_j < -eps`, Bland's rule) descends the objective
/// for minimisation.
fn reduced_costs(
    tab: &[Vec<f64>],
    basis: &[usize],
    c: &[f64],
    m: usize,
    _ncols: usize,
) -> Vec<f64> {
    let mut r = vec![0.0f64; tab[0].len()];
    let n = r.len() - 1;
    r[..n].copy_from_slice(&c[..n]);
    for i in 0..m {
        let cbj = c[basis[i]];
        if cbj != 0.0 {
            for (k, &tabik) in tab[i].iter().enumerate() {
                r[k] -= cbj * tabik;
            }
        }
    }
    // The basic columns must price out to exactly zero.
    for i in 0..m {
        r[basis[i]] = 0.0;
    }
    r
}

/// The current objective value `c_B^T b` for cost vector `c`.
fn objective_value(tab: &[Vec<f64>], basis: &[usize], c: &[f64], m: usize, _ncols: usize) -> f64 {
    let rhs = tab[0].len() - 1;
    let mut val = 0.0f64;
    for i in 0..m {
        val += c[basis[i]] * tab[i][rhs];
    }
    val
}

/// Pivot the tableau so that column `entering` becomes basic in row `leaving`.
fn pivot(
    tab: &mut [Vec<f64>],
    basis: &mut [usize],
    leaving: usize,
    entering: usize,
    m: usize,
    ncols: usize,
) {
    let rhs = ncols;
    let piv = tab[leaving][entering];
    for k in 0..=rhs {
        tab[leaving][k] /= piv;
    }
    for i in 0..m {
        if i != leaving {
            let f = tab[i][entering];
            if f != 0.0 {
                for k in 0..=rhs {
                    tab[i][k] -= f * tab[leaving][k];
                }
            }
        }
    }
    basis[leaving] = entering;
}

/// Run the simplex to optimality / unboundedness using Bland's rule.
fn run_simplex(
    tab: &mut [Vec<f64>],
    basis: &mut [usize],
    c: &[f64],
    m: usize,
    ncols: usize,
    eps: f64,
) -> SolverStatus {
    let rhs = ncols;
    for _iter in 0..(10_000 * (m + ncols) + 1000) {
        let r = reduced_costs(tab, basis, c, m, ncols);
        // Entering: smallest index with negative reduced cost (Bland).
        let enter = match r.iter().position(|&rc| rc < -eps) {
            Some(j) => j,
            None => return SolverStatus::Optimal,
        };

        // Ratio test.
        let mut leave = None;
        let mut best = f64::INFINITY;
        for i in 0..m {
            let a = tab[i][enter];
            if a > eps {
                let ratio = tab[i][rhs] / a;
                let better = ratio < best - eps
                    || ((ratio - best).abs() <= eps && leave.is_some_and(|l| basis[i] < basis[l]));
                if better {
                    best = ratio;
                    leave = Some(i);
                }
            }
        }
        let leave = match leave {
            Some(i) => i,
            None => return SolverStatus::Unbounded,
        };
        pivot(tab, basis, leave, enter, m, ncols);
    }
    SolverStatus::Error
}
