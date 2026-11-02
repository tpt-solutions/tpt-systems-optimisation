//! Python bindings: a MILP `Problem` solved by `tpt-opt-milp`, and simulated
//! annealing over a Python callable via `tpt-opt-heuristic`.

use std::cell::RefCell;
use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::{Solver, SolverStatus, VarBound};
use tpt_opt_heuristic::{ObjectiveFn, SimulatedAnnealing};
use tpt_opt_milp::{read_lp, read_mps, write_lp, write_mps, MilpSolver};
use tpt_opt_multi::scalarize::LinearTerm;
use tpt_opt_multi::{LinearMultiObjective, Nsga2, Nsga2Config};

fn status_name(s: SolverStatus) -> &'static str {
    match s {
        SolverStatus::Optimal => "optimal",
        SolverStatus::Infeasible => "infeasible",
        SolverStatus::Unbounded => "unbounded",
        SolverStatus::TimeLimit => "time_limit",
        SolverStatus::NumericalIssue => "numerical_issue",
        SolverStatus::Error => "error",
    }
}

fn value_error<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Outcome of [`Problem::solve`].
#[pyclass(name = "Solution", module = "tpt_opt")]
#[derive(Clone)]
struct PySolution {
    /// `"optimal"`, `"infeasible"`, `"unbounded"`, `"time_limit"`, and so on.
    #[pyo3(get)]
    status: String,
    /// Objective value at `primal`, including the objective constant.
    #[pyo3(get)]
    objective: f64,
    /// Variable values, in the order the variables were added.
    #[pyo3(get)]
    primal: Vec<f64>,
}

#[pymethods]
impl PySolution {
    fn __repr__(&self) -> String {
        format!("Solution(status={:?}, objective={})", self.status, self.objective)
    }
}

/// A linear or mixed-integer program: variables, linear constraints, and an
/// objective. Variables are indexed in the order they are added.
#[pyclass(name = "Problem", module = "tpt_opt")]
#[derive(Clone)]
struct PyProblem {
    model: Model,
}

#[pymethods]
impl PyProblem {
    #[new]
    fn new() -> Self {
        Self { model: Model::new(0) }
    }

    /// Add a variable and return its index. `kind` is `"continuous"`,
    /// `"integer"`, or `"binary"` (binary ignores the bounds and uses `[0, 1]`).
    #[pyo3(signature = (lower=0.0, upper=f64::INFINITY, kind="continuous"))]
    fn add_variable(&mut self, lower: f64, upper: f64, kind: &str) -> PyResult<usize> {
        let bound = match kind {
            "continuous" => VarBound::continuous(lower, upper),
            "integer" => VarBound::integer(lower, upper),
            "binary" => VarBound::binary(),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown variable kind {other:?} (continuous, integer, or binary)"
                )))
            }
        };
        Ok(self.model.add_variable(bound))
    }

    /// Add `lower <= sum(coeff * x[index]) <= upper` from `(index, coeff)` terms.
    #[pyo3(signature = (terms, lower=f64::NEG_INFINITY, upper=f64::INFINITY))]
    fn add_constraint(&mut self, terms: Vec<(usize, f64)>, lower: f64, upper: f64) -> PyResult<usize> {
        let (indices, coeffs) = self.split_terms(&terms)?;
        let con = Constraint::new(indices, coeffs, lower, upper).map_err(value_error)?;
        Ok(self.model.add_constraint(con))
    }

    /// Set the objective: `constant + sum(coeff * x[index])`, minimised or maximised.
    #[pyo3(signature = (terms, sense="minimize", constant=0.0))]
    fn set_objective(&mut self, terms: Vec<(usize, f64)>, sense: &str, constant: f64) -> PyResult<()> {
        let (indices, coeffs) = self.split_terms(&terms)?;
        let sense = match sense {
            "minimize" => Sense::Minimize,
            "maximize" => Sense::Maximize,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sense {other:?} (minimize or maximize)"
                )))
            }
        };
        self.model.set_objective(Objective { sense, indices, coeffs, constant });
        Ok(())
    }

    /// Solve with the MILP solver. `threads > 1` enables the deterministic
    /// parallel tree search; `seed` fixes heuristics and branching.
    #[pyo3(signature = (time_limit=None, threads=1, seed=None))]
    fn solve(&self, time_limit: Option<f64>, threads: usize, seed: Option<u64>) -> PyResult<PySolution> {
        let mut solver = MilpSolver::new().with_threads(threads.max(1));
        if let Some(secs) = time_limit {
            if !(secs > 0.0 && secs.is_finite()) {
                return Err(PyValueError::new_err("time_limit must be a positive number"));
            }
            solver = solver.with_time_limit(Duration::from_secs_f64(secs));
        }
        if let Some(seed) = seed {
            solver = solver.with_seed(seed);
        }
        let sol = solver.solve(&self.model).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PySolution {
            status: status_name(sol.status).to_string(),
            objective: sol.objective_value,
            primal: sol.primal,
        })
    }

    /// Parse free-format MPS text.
    #[staticmethod]
    fn from_mps(text: &str) -> PyResult<Self> {
        Ok(Self { model: read_mps(text).map_err(value_error)? })
    }

    /// Parse CPLEX-LP text.
    #[staticmethod]
    fn from_lp(text: &str) -> PyResult<Self> {
        Ok(Self { model: read_lp(text).map_err(value_error)? })
    }

    /// Render as free-format MPS text.
    fn to_mps(&self) -> String {
        write_mps(&self.model)
    }

    /// Render as CPLEX-LP text.
    fn to_lp(&self) -> String {
        write_lp(&self.model)
    }

    #[getter]
    fn num_vars(&self) -> usize {
        self.model.num_vars
    }

    #[getter]
    fn num_constraints(&self) -> usize {
        self.model.num_constraints()
    }
}

impl PyProblem {
    fn split_terms(&self, terms: &[(usize, f64)]) -> PyResult<(Vec<usize>, Vec<f64>)> {
        let mut indices = Vec::with_capacity(terms.len());
        let mut coeffs = Vec::with_capacity(terms.len());
        for &(index, coeff) in terms {
            if index >= self.model.num_vars {
                return Err(PyValueError::new_err(format!(
                    "variable index {index} is out of range (problem has {} variables)",
                    self.model.num_vars
                )));
            }
            indices.push(index);
            coeffs.push(coeff);
        }
        Ok((indices, coeffs))
    }
}

/// Minimise `f(x)` over the box `bounds` by simulated annealing.
///
/// `f` receives a list of floats and returns a float. An exception raised by
/// `f` stops the search and is re-raised. Returns `(best_x, best_value)`.
#[pyfunction]
#[pyo3(signature = (f, bounds, seed=0, iterations=10_000))]
fn simulated_annealing(
    f: Py<PyAny>,
    bounds: Vec<(f64, f64)>,
    seed: u64,
    iterations: usize,
) -> PyResult<(Vec<f64>, f64)> {
    let dim = bounds.len();
    if dim == 0 {
        return Err(PyValueError::new_err("bounds must not be empty"));
    }
    // The objective must be 'static, so the callable and the first Python error
    // are shared through reference counts. The error is re-raised after the
    // search, since the objective cannot return a Python exception directly.
    let f = std::rc::Rc::new(f);
    let error: std::rc::Rc<RefCell<Option<PyErr>>> = std::rc::Rc::new(RefCell::new(None));
    let objective = {
        let f = f.clone();
        let error = error.clone();
        ObjectiveFn::minimize(
            dim,
            move |x: &[f64]| {
                if error.borrow().is_some() {
                    return f64::INFINITY;
                }
                Python::with_gil(|py| {
                    let result = f.call1(py, (x.to_vec(),)).and_then(|v| v.extract::<f64>(py));
                    match result {
                        Ok(value) => value,
                        Err(e) => {
                            *error.borrow_mut() = Some(e);
                            f64::INFINITY
                        }
                    }
                })
            },
            bounds,
        )
    };
    let result = SimulatedAnnealing::new(objective)
        .with_seed(seed)
        .with_iterations(iterations)
        .solve();
    if let Some(e) = error.borrow_mut().take() {
        return Err(e);
    }
    let result = result.map_err(value_error)?;
    Ok((result.best_x, result.best_value))
}

/// A multi-objective problem whose objectives are linear functions of one
/// shared box-bounded decision vector.
///
/// Objectives are `(coefficients, constant)` pairs. Each one contributes
/// `constant + sum(coefficients[j] * x[j])`, and all of them are minimised.
#[pyclass(name = "LinearMultiObjective", module = "tpt_opt")]
#[derive(Clone)]
struct PyMulti {
    prob: LinearMultiObjective,
}

#[pymethods]
impl PyMulti {
    #[new]
    fn new(objectives: Vec<(Vec<f64>, f64)>, bounds: Vec<(f64, f64)>) -> PyResult<Self> {
        if objectives.is_empty() {
            return Err(PyValueError::new_err("at least one objective is required"));
        }
        for (coeffs, _) in &objectives {
            if coeffs.len() != bounds.len() {
                return Err(PyValueError::new_err(format!(
                    "objective has {} coefficients but there are {} variables",
                    coeffs.len(),
                    bounds.len()
                )));
            }
        }
        let terms = objectives
            .into_iter()
            .map(|(coeffs, constant)| LinearTerm { coeffs, constant })
            .collect();
        Ok(Self { prob: LinearMultiObjective::new(terms, bounds) })
    }

    /// Minimise `sum(w_i * f_i(x))`. Returns `(x, f(x))`.
    fn weighted_sum(&self, weights: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let x = self.prob.solve_weighted_sum(&weights).map_err(value_error)?;
        let f = self.prob.evaluate(&x);
        Ok((x, f))
    }

    /// Minimise objective `primary` subject to `f_i(x) <= eps[i]` for every
    /// other `i` (`eps[primary]` is ignored). Returns `(x, f(x))`.
    fn epsilon_constraint(&self, primary: usize, eps: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if primary >= self.prob.num_objectives() {
            return Err(PyValueError::new_err("primary objective index is out of range"));
        }
        let (x, _) = self.prob.solve_epsilon_constraint(primary, &eps).map_err(value_error)?;
        let f = self.prob.evaluate(&x);
        Ok((x, f))
    }

    /// Each objective's best and worst value over the Pareto extremes:
    /// `(ideal, nadir)`.
    fn payoff_table(&self) -> PyResult<(Vec<f64>, Vec<f64>)> {
        self.prob.payoff_table().map_err(value_error)
    }

    /// Approximate the Pareto front with NSGA-II. Returns non-dominated
    /// `(x, f(x))` pairs.
    #[pyo3(signature = (population=100, generations=200, seed=0))]
    fn nsga2(&self, population: usize, generations: usize, seed: u64) -> Vec<(Vec<f64>, Vec<f64>)> {
        let prob = self.prob.clone();
        let config = Nsga2Config { population, generations, ..Nsga2Config::default() };
        Nsga2::new(prob.bounds().to_vec(), move |x: &[f64]| prob.evaluate(x))
            .with_config(config)
            .with_seed(seed)
            .pareto_front()
    }
}

#[pymodule]
fn tpt_opt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProblem>()?;
    m.add_class::<PySolution>()?;
    m.add_class::<PyMulti>()?;
    m.add_function(wrap_pyfunction!(simulated_annealing, m)?)?;
    Ok(())
}
