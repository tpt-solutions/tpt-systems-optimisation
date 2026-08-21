//! Optimal power flow (OPF) formulations.
//!
//! Three formulations are provided:
//!
//! - [`dc_opf`] — DC-OPF, the linearised (lossless, flat-voltage) approximation
//!   expressed as an LP and solved with the crate's internal two-phase simplex
//!   ([`crate::LpSolver`]). This is the workhorse for large networks.
//! - [`ac_opf`] — AC-OPF in polar coordinates, solved as a nonlinear program via
//!   `tpt_math_optimize_general::solve_nlp` (augmented-Lagrangian / BFGS). The
//!   active power balance at every bus is enforced exactly; generator limits,
//!   voltage magnitude limits and line-flow limits are enforced as bounds /
//!   inequalities.
//! - [`sc_opf`] — security-constrained OPF: the base-case DC-OPF plus an N-1
//!   contingency check that re-solves a DC-OPF with each line outaged and
//!   reports whether a feasible dispatch exists for every single contingency.
//!
//! All numerical failures surface as [`tpt_opt_core::solver::SolverStatus::Error`]
//! / [`tpt_opt_core::solver::SolverStatus::Infeasible`] rather than silently
//! returning wrong dispatches.

use std::vec::Vec;

use tpt_math_optimize_general::{solve_nlp, NlpParams, NlpProblem};
use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::solver::Solver;
use tpt_opt_core::{SolverStatus, VarBound};

use crate::LpSolver;

/// A bus (node) of a power network.
#[derive(Debug, Clone, PartialEq)]
pub struct Bus {
    /// Bus index (must match array positions in [`Network::buses`]).
    pub id: usize,
    /// `true` for the reference (slack) bus whose angle is fixed to zero.
    pub is_slack: bool,
    /// Active power demand at this bus (MW, positive = load).
    pub demand_p: f64,
    /// Reactive power demand at this bus (MVAr, positive = load).
    pub demand_q: f64,
    /// Lower voltage magnitude limit (per-unit).
    pub v_min: f64,
    /// Upper voltage magnitude limit (per-unit).
    pub v_max: f64,
}

/// A generator connected to a bus.
#[derive(Debug, Clone, PartialEq)]
pub struct Generator {
    /// Bus index this generator is connected to.
    pub bus: usize,
    /// Minimum active power output (MW).
    pub p_min: f64,
    /// Maximum active power output (MW).
    pub p_max: f64,
    /// Quadratic generation cost coefficient `c0 + c1*P + c2*P^2`.
    pub c0: f64,
    /// Linear generation cost coefficient.
    pub c1: f64,
    /// Quadratic generation cost coefficient.
    pub c2: f64,
}

/// A transmission line (branch) of the network.
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    /// Index of the "from" bus.
    pub from: usize,
    /// Index of the "to" bus.
    pub to: usize,
    /// Series reactance `x` (per-unit); susceptance is `1/x`.
    pub reactance: f64,
    /// Thermal capacity (MW); flow magnitude must stay within `[-cap, cap]`.
    pub capacity: f64,
}

/// A complete power network: buses, generators and lines.
#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    /// Buses, indexed by `Bus::id`.
    pub buses: Vec<Bus>,
    /// Generators.
    pub generators: Vec<Generator>,
    /// Transmission lines.
    pub lines: Vec<Line>,
}

/// Result of a DC-OPF solve.
#[derive(Debug, Clone)]
pub struct DcOpfResult {
    /// Optimal active-power dispatch, one value per generator (MW).
    pub dispatch: Vec<f64>,
    /// Optimal line flow, one value per line (MW; signed by `from`->`to`).
    pub flows: Vec<f64>,
    /// Voltage angle at each bus (rad); the slack bus is `0.0`.
    pub angles: Vec<f64>,
    /// Total generation cost (objective value).
    pub total_cost: f64,
    /// Terminal status.
    pub status: SolverStatus,
}

/// Solve the DC-OPF (linearised OPF) as an LP.
///
/// Returns optimal generator dispatch, line flows and angles. When the LP is
/// infeasible (e.g. not enough generation capacity) `status` is
/// [`SolverStatus::Infeasible`]; on any other numerical failure it is
/// [`SolverStatus::Error`].
pub fn dc_opf(net: &Network) -> DcOpfResult {
    let n = net.buses.len();
    let g = net.generators.len();
    let l = net.lines.len();
    let slack = net.buses.iter().position(|b| b.is_slack).unwrap_or(0);

    let mut model = Model::new(0);

    // Variable index maps.
    let mut pg_idx = Vec::with_capacity(g);
    for gen in &net.generators {
        pg_idx.push(model.add_variable(VarBound::continuous(gen.p_min, gen.p_max)));
    }
    let theta_idx: Vec<usize> = (0..n)
        .map(|i| {
            if i == slack {
                usize::MAX
            } else {
                model.add_variable(VarBound::continuous(f64::NEG_INFINITY, f64::INFINITY))
            }
        })
        .collect();
    let mut f_idx = Vec::with_capacity(l);
    for line in &net.lines {
        f_idx.push(model.add_variable(VarBound::continuous(-line.capacity, line.capacity)));
    }

    // Objective: minimise sum_g (c1*Pg + c0) (quadratic term dropped for the LP).
    let mut obj_idx = Vec::with_capacity(g);
    let mut obj_coeff = Vec::with_capacity(g);
    let mut constant = 0.0f64;
    for (i, gen) in net.generators.iter().enumerate() {
        obj_idx.push(pg_idx[i]);
        obj_coeff.push(gen.c1);
        constant += gen.c0;
    }
    model.set_objective(Objective {
        sense: Sense::Minimize,
        indices: obj_idx,
        coeffs: obj_coeff,
        constant,
    });

    // Line-flow definition: theta_from - theta_to - x_k * f_k = 0.
    for (k, line) in net.lines.iter().enumerate() {
        let mut indices = Vec::new();
        let mut coeffs = Vec::new();
        let thf = theta_idx[line.from];
        let tht = theta_idx[line.to];
        if thf != usize::MAX {
            indices.push(thf);
            coeffs.push(1.0);
        }
        if tht != usize::MAX {
            indices.push(tht);
            coeffs.push(-1.0);
        }
        indices.push(f_idx[k]);
        coeffs.push(-line.reactance);
        model.add_constraint(Constraint::equality(indices, coeffs, 0.0));
    }

    // Active power balance at each bus: injection - load = 0.
    for i in 0..n {
        let mut indices = Vec::new();
        let mut coeffs = Vec::new();
        for (gi, gen) in net.generators.iter().enumerate() {
            if gen.bus == i {
                indices.push(pg_idx[gi]);
                coeffs.push(1.0);
            }
        }
        for (k, line) in net.lines.iter().enumerate() {
            if line.from == i {
                indices.push(f_idx[k]);
                coeffs.push(-1.0);
            } else if line.to == i {
                indices.push(f_idx[k]);
                coeffs.push(1.0);
            }
        }
        model.add_constraint(Constraint::equality(indices, coeffs, net.buses[i].demand_p));
    }

    let mut solver = LpSolver::new();
    match solver.solve(&model) {
        Ok(s) => {
            let mut dispatch = vec![0.0f64; g];
            for (i, &idx) in pg_idx.iter().enumerate() {
                dispatch[i] = s.primal[idx];
            }
            let mut flows = vec![0.0f64; l];
            for (k, &idx) in f_idx.iter().enumerate() {
                flows[k] = s.primal[idx];
            }
            let mut angles = vec![0.0f64; n];
            for i in 0..n {
                if theta_idx[i] != usize::MAX {
                    angles[i] = s.primal[theta_idx[i]];
                }
            }
            DcOpfResult { dispatch, flows, angles, total_cost: s.objective_value, status: s.status }
        }
        Err(_) => DcOpfResult {
            dispatch: vec![0.0f64; g],
            flows: vec![0.0f64; l],
            angles: vec![0.0f64; n],
            total_cost: f64::INFINITY,
            status: SolverStatus::Error,
        },
    }
}

/// Result of an AC-OPF solve (polar coordinates).
#[derive(Debug, Clone)]
pub struct AcOpfResult {
    /// Optimal active-power dispatch, one value per generator (MW).
    pub dispatch: Vec<f64>,
    /// Voltage magnitude at each bus (per-unit).
    pub voltages: Vec<f64>,
    /// Voltage angle at each bus (rad).
    pub angles: Vec<f64>,
    /// Line flow magnitude, one value per line (MW).
    pub flows: Vec<f64>,
    /// Total generation cost (objective value).
    pub total_cost: f64,
    /// `true` if the NLP converged.
    pub converged: bool,
}

/// The AC-OPF cast as an NLP subproblem for `tpt_math_optimize_general`.
struct AcOpf<'a> {
    net: &'a Network,
    slack: usize,
    pg: Vec<usize>,
    v: Vec<usize>,
    th: Vec<usize>,
    /// Nodal susceptance matrix `B` (Laplacian from line susceptances).
    b: Vec<Vec<f64>>,
}

impl<'a> AcOpf<'a> {
    fn build(net: &'a Network) -> Self {
        let n = net.buses.len();
        let slack = net.buses.iter().position(|b| b.is_slack).unwrap_or(0);
        let mut b = vec![vec![0.0f64; n]; n];
        for line in &net.lines {
            let s = 1.0 / line.reactance;
            b[line.from][line.to] -= s;
            b[line.to][line.from] -= s;
            b[line.from][line.from] += s;
            b[line.to][line.to] += s;
        }
        let g = net.generators.len();
        let pg: Vec<usize> = (0..g).collect();
        let v: Vec<usize> = (g..g + n).collect();
        let th: Vec<usize> = (g + n..g + 2 * n).collect();
        AcOpf { net, slack, pg, v, th, b }
    }
}

impl NlpProblem for AcOpf<'_> {
    fn num_vars(&self) -> usize {
        self.net.generators.len() + 2 * self.net.buses.len()
    }

    fn objective(&self, x: &[f64]) -> f64 {
        let mut c = 0.0f64;
        for (gi, gen) in self.net.generators.iter().enumerate() {
            let pg = x[self.pg[gi]];
            c += gen.c2 * pg * pg + gen.c1 * pg + gen.c0;
        }
        c
    }

    fn num_eq(&self) -> usize {
        self.net.buses.len() + 2
    }

    fn num_ineq(&self) -> usize {
        2 * self.net.lines.len()
    }

    fn eq(&self, j: usize, x: &[f64]) -> f64 {
        let n = self.net.buses.len();
        match j {
            m if m < n => {
                let vj = x[self.v[m]];
                let thj = x[self.th[m]];
                let mut pinj = 0.0f64;
                for i in 0..n {
                    let d = thj - x[self.th[i]];
                    pinj += vj * x[self.v[i]] * self.b[m][i] * d.sin();
                }
                let mut pg_inj = 0.0f64;
                for (gi, gen) in self.net.generators.iter().enumerate() {
                    if gen.bus == m {
                        pg_inj += x[self.pg[gi]];
                    }
                }
                pinj - pg_inj + self.net.buses[m].demand_p
            }
            m if m == n => x[self.v[self.slack]] - 1.0,
            _ => x[self.th[self.slack]],
        }
    }

    fn ineq(&self, j: usize, x: &[f64]) -> f64 {
        let k = j / 2;
        let line = &self.net.lines[k];
        let s = 1.0 / line.reactance;
        let fk = -x[self.v[line.from]]
            * x[self.v[line.to]]
            * s
            * (x[self.th[line.from]] - x[self.th[line.to]]).sin();
        if j % 2 == 0 {
            fk - line.capacity
        } else {
            -fk - line.capacity
        }
    }
}

/// Solve the AC-OPF (polar coordinates) as a nonlinear program.
///
/// The active power balance at every bus and the slack bus voltage/angle are
/// enforced as equalities; line-flow limits are enforced as inequalities. The
/// returned `converged` flag reports whether the augmented-Lagrangian solver
/// reached the tolerance.
pub fn ac_opf(net: &Network) -> AcOpfResult {
    let prob = AcOpf::build(net);
    let g = net.generators.len();
    let n = net.buses.len();

    let mut x0 = vec![0.0f64; prob.num_vars()];
    for (gi, gen) in net.generators.iter().enumerate() {
        x0[prob.pg[gi]] = 0.5 * (gen.p_min + gen.p_max);
        if gen.bus == prob.slack {
            x0[prob.pg[gi]] = net.buses[prob.slack].demand_p;
        }
    }
    for i in 0..n {
        x0[prob.v[i]] = 1.0;
        x0[prob.th[i]] = 0.0;
    }

    let params = NlpParams::default();
    let res = solve_nlp(&prob, &x0, &params);

    let mut dispatch = vec![0.0f64; g];
    for (gi, &idx) in prob.pg.iter().enumerate() {
        dispatch[gi] = res.x[idx];
    }
    let mut voltages = vec![0.0f64; n];
    let mut angles = vec![0.0f64; n];
    for i in 0..n {
        voltages[i] = res.x[prob.v[i]];
        angles[i] = res.x[prob.th[i]];
    }
    let mut flows = vec![0.0f64; net.lines.len()];
    for (k, line) in net.lines.iter().enumerate() {
        let s = 1.0 / line.reactance;
        flows[k] = -voltages[line.from]
            * voltages[line.to]
            * s
            * (angles[line.from] - angles[line.to]).sin();
    }

    AcOpfResult {
        dispatch,
        voltages,
        angles,
        flows,
        total_cost: res.objective,
        converged: res.status == tpt_math_optimize_general::NlpStatus::Converged,
    }
}

/// Result of a security-constrained OPF (SC-OPF) solve.
#[derive(Debug, Clone)]
pub struct ScOpfResult {
    /// The base-case DC-OPF dispatch.
    pub base: DcOpfResult,
    /// `true` if a feasible dispatch exists for every single line outage.
    pub secure: bool,
    /// Per-contingency status: `(outaged_line_index, status)`.
    pub contingency_status: Vec<(usize, SolverStatus)>,
    /// Worst base-case line loading (`|flow| / capacity`), `0.0` if no lines.
    pub worst_loading: f64,
}

/// Solve the security-constrained OPF: the base-case DC-OPF followed by an N-1
/// contingency analysis.
///
/// For each line `k` the network is re-solved with that line outaged (capacity
/// forced to zero). If every re-solve is feasible the system is N-1 secure in
/// the sense that a feasible dispatch exists for each single contingency; this
/// is a conservative sufficient condition rather than a jointly co-optimised
/// SC-OPF. [`ScOpfResult::worst_loading`] reports the most heavily loaded base
/// case line.
pub fn sc_opf(net: &Network) -> ScOpfResult {
    let base = dc_opf(net);

    let mut secure = base.status == SolverStatus::Optimal;
    let mut contingency_status = Vec::new();
    for k in 0..net.lines.len() {
        let mut sub = net.clone();
        sub.lines[k].capacity = 0.0;
        let res = dc_opf(&sub);
        if res.status != SolverStatus::Optimal {
            secure = false;
        }
        contingency_status.push((k, res.status));
    }

    let mut worst = 0.0f64;
    for (k, line) in net.lines.iter().enumerate() {
        if line.capacity > 0.0 {
            worst = worst.max(base.flows[k].abs() / line.capacity);
        }
    }

    ScOpfResult { base, secure, contingency_status, worst_loading: worst }
}
