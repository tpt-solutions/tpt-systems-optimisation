#![no_std]
// Numeric linear-algebra loops below index arrays by the loop counter; this is
// intentional and clearer than iterator rewrites for dense matrix code.
#![allow(clippy::needless_range_loop)]
//! Local dev shim mirroring `tpt-math-optimize-convex`: a convex quadratic
//! program solver used for relaxation master problems in `tpt-opt-minlp` and
//! `tpt-opt-decompose`.
//!
//! It delegates to the augmented-Lagrangian NLP solver in
//! `tpt-math-optimize-general`, which is exact for convex QP.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use tpt_math_optimize_general::{solve_nlp, NlpParams, NlpProblem, NlpResult, NlpStatus};

/// A convex quadratic program
/// ```text
/// minimise    1/2 xᵀ P x + qᵀ x
/// subject to  A_ineq x <= b_ineq
///             A_eq x    = b_eq
/// ```
#[derive(Debug, Clone)]
pub struct ConvexQp {
    n: usize,
    p: Vec<f64>,
    q: Vec<f64>,
    a_ineq: Vec<f64>,
    b_ineq: Vec<f64>,
    a_eq: Vec<f64>,
    b_eq: Vec<f64>,
    m_ineq: usize,
    m_eq: usize,
}

impl ConvexQp {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            p: vec![0.0; n * n],
            q: vec![0.0; n],
            a_ineq: Vec::new(),
            b_ineq: Vec::new(),
            a_eq: Vec::new(),
            b_eq: Vec::new(),
            m_ineq: 0,
            m_eq: 0,
        }
    }

    pub fn set_objective(&mut self, p: Vec<f64>, q: Vec<f64>) {
        self.p = p;
        self.q = q;
    }

    pub fn add_ineq(&mut self, a_row: Vec<f64>, b: f64) {
        self.a_ineq.extend(a_row);
        self.b_ineq.push(b);
        self.m_ineq += 1;
    }

    pub fn add_eq(&mut self, a_row: Vec<f64>, b: f64) {
        self.a_eq.extend(a_row);
        self.b_eq.push(b);
        self.m_eq += 1;
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn solve(&self, x0: &[f64]) -> QpResult {
        let prob = QpView { qp: self };
        let res = solve_nlp(&prob, x0, &NlpParams::default());
        QpResult {
            x: res.x,
            objective: res.objective,
            status: res.status,
            iterations: res.iterations,
        }
    }
}

struct QpView<'a> {
    qp: &'a ConvexQp,
}

impl NlpProblem for QpView<'_> {
    fn num_vars(&self) -> usize {
        self.qp.n
    }
    fn objective(&self, x: &[f64]) -> f64 {
        let mut s = 0.0f64;
        for i in 0..self.qp.n {
            let mut pi = 0.0f64;
            for j in 0..self.qp.n {
                pi += self.qp.p[i * self.qp.n + j] * x[j];
            }
            s += 0.5 * pi * x[i];
        }
        for i in 0..self.qp.n {
            s += self.qp.q[i] * x[i];
        }
        s
    }
    fn objective_grad(&self, x: &[f64], g: &mut [f64]) {
        for i in 0..self.qp.n {
            let mut pi = 0.0f64;
            for j in 0..self.qp.n {
                pi += self.qp.p[i * self.qp.n + j] * x[j];
            }
            g[i] = pi + self.qp.q[i];
        }
    }
    fn num_ineq(&self) -> usize {
        self.qp.m_ineq
    }
    fn ineq(&self, i: usize, x: &[f64]) -> f64 {
        let mut s = 0.0f64;
        for j in 0..self.qp.n {
            s += self.qp.a_ineq[i * self.qp.n + j] * x[j];
        }
        s - self.qp.b_ineq[i]
    }
    fn ineq_grad(&self, i: usize, _x: &[f64], row: &mut [f64]) {
        for j in 0..self.qp.n {
            row[j] = self.qp.a_ineq[i * self.qp.n + j];
        }
    }
    fn num_eq(&self) -> usize {
        self.qp.m_eq
    }
    fn eq(&self, j: usize, x: &[f64]) -> f64 {
        let mut s = 0.0f64;
        for k in 0..self.qp.n {
            s += self.qp.a_eq[j * self.qp.n + k] * x[k];
        }
        s - self.qp.b_eq[j]
    }
    fn eq_grad(&self, j: usize, _x: &[f64], row: &mut [f64]) {
        for k in 0..self.qp.n {
            row[k] = self.qp.a_eq[j * self.qp.n + k];
        }
    }
}

/// Result of solving a [`ConvexQp`].
#[derive(Debug, Clone)]
pub struct QpResult {
    pub x: Vec<f64>,
    pub objective: f64,
    pub status: NlpStatus,
    pub iterations: usize,
}

#[allow(unused_imports)]
use NlpResult as _;
