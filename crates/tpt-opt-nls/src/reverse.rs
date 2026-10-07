//! Reverse-mode automatic differentiation on a value graph.
//!
//! Where forward mode ([`crate::dual`]) costs O(n) gradient work per
//! arithmetic operation, reverse mode records the computation graph once and
//! propagates adjoints backwards, giving the **gradient of one scalar output
//! in a single backward pass** regardless of the number of inputs. This is
//! the right tool when the parameter count dwarfs the output count — e.g.
//! the scalar cost `½‖r(x)‖²` of a least-squares problem over thousands of
//! parameters, or matrix-free Gauss–Newton systems driven by cost gradients.
//!
//! A [`Value`] is a reference-counted node carrying its value, its adjoint
//! (after a backward pass), and its parents with the partial derivatives
//! ∂node/∂parent evaluated eagerly at forward time. Expressions read like
//! scalar code thanks to the standard operator impls.
//!
//! Graphs are built per evaluation point (leaves are inputs, one
//! [`Value::backward`] per output); this keeps the implementation
//! allocation-simple and deterministic.
//!
//! # Example
//!
//! ```rust
//! use tpt_opt_nls::reverse::Value;
//!
//! // Rosenbrock cost ½((1−x)² + 10(y−x²)²) at (0, 0): gradient (−1, 0).
//! let x = Value::var(0.0);
//! let y = Value::var(0.0);
//! let r1 = 1.0 - x.clone();
//! let r2 = 10.0 * (y.clone() - x.clone() * x.clone());
//! let cost = 0.5 * (r1.clone() * r1.clone() + r2.clone() * r2.clone());
//! cost.backward();
//! assert!((x.grad() + 1.0).abs() < 1e-12);
//! assert!(y.grad().abs() < 1e-12);
//! ```

use std::cell::Cell;
use std::collections::HashSet;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

struct Node {
    value: f64,
    /// Adjoint ∂output/∂this, valid after a [`Value::backward`] pass.
    grad: Cell<f64>,
    /// `(parent, ∂node/∂parent)` pairs with partials frozen at forward time.
    parents: Vec<(Rc<Node>, f64)>,
}

impl Drop for Node {
    fn drop(&mut self) {
        // Iteratively dismantle the parent chain: recursive drop glue over a
        // deep tape (50k+ nodes) would overflow the stack.
        let mut stack: Vec<Rc<Node>> = self.parents.drain(..).map(|(p, _)| p).collect();
        while let Some(mut node) = stack.pop() {
            if let Some(inner) = Rc::get_mut(&mut node) {
                // Last reference: absorb its parents into the work list.
                // (Shared nodes are dismantled by their final owner.)
                stack.extend(inner.parents.drain(..).map(|(p, _)| p));
            }
        }
    }
}

/// A differentiable scalar (a node in the value graph). Cheap to clone.
#[derive(Clone)]
pub struct Value(Rc<Node>);

impl Value {
    fn node(value: f64, parents: Vec<(Rc<Node>, f64)>) -> Self {
        Value(Rc::new(Node { value, grad: Cell::new(0.0), parents }))
    }

    /// A leaf input variable (or constant) with the given value.
    pub fn var(value: f64) -> Self {
        Value::node(value, Vec::new())
    }

    /// Current forward value.
    pub fn value(&self) -> f64 {
        self.0.value
    }

    /// Adjoint accumulated by the most recent [`Value::backward`] pass.
    pub fn grad(&self) -> f64 {
        self.0.grad.get()
    }

    /// Number of direct parents (0 for leaves).
    pub fn arity(&self) -> usize {
        self.0.parents.len()
    }

    /// Run the backward pass from this node: every reachable node's adjoint
    /// is reset, this node's adjoint is seeded with 1, and adjoints are
    /// propagated along the topological order (children before parents).
    /// Afterwards [`Value::grad`] holds ∂self/∂node for every leaf used to
    /// build the expression.
    pub fn backward(&self) {
        let order = topo_order(self);
        for n in &order {
            n.0.grad.set(0.0);
        }
        self.0.grad.set(1.0);
        for v in order.iter().rev() {
            let g = v.0.grad.get();
            if g == 0.0 {
                continue;
            }
            for (parent, partial) in &v.0.parents {
                parent.grad.set(parent.grad.get() + g * partial);
            }
        }
    }

    /// Jacobian of an `m`-output function: one backward pass per output.
    /// `leaves` are the inputs, `outs` the outputs; returns the row-major
    /// `outs.len() × leaves.len()` Jacobian.
    pub fn jacobian(leaves: &[Value], outs: &[Value]) -> Vec<f64> {
        let mut jac = vec![0.0; outs.len() * leaves.len()];
        for (row, o) in outs.iter().enumerate() {
            o.backward();
            for (col, l) in leaves.iter().enumerate() {
                jac[row * leaves.len() + col] = l.grad();
            }
        }
        jac
    }

    /// Square root (∂ at 0 taken as 0).
    pub fn sqrt(&self) -> Self {
        let v = self.0.value.sqrt();
        let p = if v > 0.0 { 0.5 / v } else { 0.0 };
        Value::node(v, vec![(Rc::clone(&self.0), p)])
    }

    /// Natural exponential.
    pub fn exp(&self) -> Self {
        let v = self.0.value.exp();
        Value::node(v, vec![(Rc::clone(&self.0), v)])
    }

    /// Natural logarithm.
    pub fn ln(&self) -> Self {
        Value::node(self.0.value.ln(), vec![(Rc::clone(&self.0), 1.0 / self.0.value)])
    }

    /// Sine.
    pub fn sin(&self) -> Self {
        Value::node(self.0.value.sin(), vec![(Rc::clone(&self.0), self.0.value.cos())])
    }

    /// Cosine.
    pub fn cos(&self) -> Self {
        Value::node(self.0.value.cos(), vec![(Rc::clone(&self.0), -self.0.value.sin())])
    }

    /// Tangent.
    pub fn tan(&self) -> Self {
        let c = self.0.value.cos();
        Value::node(self.0.value.tan(), vec![(Rc::clone(&self.0), 1.0 / (c * c))])
    }

    /// Hyperbolic sine.
    pub fn sinh(&self) -> Self {
        Value::node(self.0.value.sinh(), vec![(Rc::clone(&self.0), self.0.value.cosh())])
    }

    /// Hyperbolic cosine.
    pub fn cosh(&self) -> Self {
        Value::node(self.0.value.cosh(), vec![(Rc::clone(&self.0), self.0.value.sinh())])
    }

    /// Absolute value (subgradient 0 at zero).
    pub fn abs(&self) -> Self {
        let s = if self.0.value >= 0.0 { 1.0 } else { -1.0 };
        Value::node(self.0.value.abs(), vec![(Rc::clone(&self.0), s)])
    }

    /// Two-argument arc tangent (derivative w.r.t. both arguments).
    pub fn atan2(&self, other: &Value) -> Value {
        let d = self.0.value * self.0.value + other.0.value * other.0.value;
        Value::node(
            self.0.value.atan2(other.0.value),
            vec![(Rc::clone(&self.0), other.0.value / d), (Rc::clone(&other.0), -self.0.value / d)],
        )
    }

    /// Hypotenuse `sqrt(x² + y²)` with a stable derivative.
    pub fn hypot(&self, other: &Value) -> Value {
        let h = self.0.value.hypot(other.0.value);
        Value::node(
            h,
            vec![(Rc::clone(&self.0), self.0.value / h), (Rc::clone(&other.0), other.0.value / h)],
        )
    }

    /// Raise to a constant integer power.
    pub fn powi(&self, p: i32) -> Value {
        if p == 0 {
            return Value::node(1.0, vec![(Rc::clone(&self.0), 0.0)]);
        }
        let v = self.0.value.powi(p);
        Value::node(v, vec![(Rc::clone(&self.0), p as f64 * self.0.value.powi(p - 1))])
    }

    /// Raise to a differentiable power: both operands receive adjoints.
    pub fn powf(&self, e: &Value) -> Value {
        let (b, p) = (self.0.value, e.0.value);
        let v = b.powf(p);
        let pb = if b > 0.0 { v * p / b } else { 0.0 };
        let pe = if b > 0.0 { v * b.ln() } else { 0.0 };
        Value::node(v, vec![(Rc::clone(&self.0), pb), (Rc::clone(&e.0), pe)])
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value({})", self.0.value)
    }
}

/// Iterative topological order (children before parents) — explicit stack so
/// deep graphs cannot overflow the call stack.
fn topo_order(root: &Value) -> Vec<Value> {
    let mut order: Vec<Value> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut stack: Vec<(Rc<Node>, usize)> = vec![(Rc::clone(&root.0), 0)];
    visited.insert(Rc::as_ptr(&root.0) as usize);
    while let Some(top) = stack.last_mut() {
        let idx = top.1;
        if idx < top.0.parents.len() {
            top.1 += 1;
            let child = Rc::clone(&top.0.parents[idx].0);
            if visited.insert(Rc::as_ptr(&child) as usize) {
                stack.push((child, 0));
            }
        } else {
            let (node, _) = stack.pop().expect("stack non-empty");
            order.push(Value(node));
        }
    }
    order
}

impl Add for Value {
    type Output = Value;
    fn add(self, rhs: Value) -> Value {
        Value::node(
            self.0.value + rhs.0.value,
            vec![(Rc::clone(&self.0), 1.0), (Rc::clone(&rhs.0), 1.0)],
        )
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Value) -> Value {
        Value::node(
            self.0.value - rhs.0.value,
            vec![(Rc::clone(&self.0), 1.0), (Rc::clone(&rhs.0), -1.0)],
        )
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Value) -> Value {
        Value::node(
            self.0.value * rhs.0.value,
            vec![(Rc::clone(&self.0), rhs.0.value), (Rc::clone(&rhs.0), self.0.value)],
        )
    }
}

impl Div for Value {
    type Output = Value;
    fn div(self, rhs: Value) -> Value {
        let q = self.0.value / rhs.0.value;
        Value::node(
            q,
            vec![(Rc::clone(&self.0), 1.0 / rhs.0.value), (Rc::clone(&rhs.0), -q / rhs.0.value)],
        )
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        Value::node(-self.0.value, vec![(Rc::clone(&self.0), -1.0)])
    }
}

impl Add<f64> for Value {
    type Output = Value;
    fn add(self, s: f64) -> Value {
        self + Value::var(s)
    }
}

impl Sub<f64> for Value {
    type Output = Value;
    fn sub(self, s: f64) -> Value {
        self - Value::var(s)
    }
}

impl Mul<f64> for Value {
    type Output = Value;
    fn mul(self, s: f64) -> Value {
        self * Value::var(s)
    }
}

impl Div<f64> for Value {
    type Output = Value;
    fn div(self, s: f64) -> Value {
        self / Value::var(s)
    }
}

impl Add<Value> for f64 {
    type Output = Value;
    fn add(self, v: Value) -> Value {
        Value::var(self) + v
    }
}

impl Sub<Value> for f64 {
    type Output = Value;
    fn sub(self, v: Value) -> Value {
        Value::var(self) - v
    }
}

impl Mul<Value> for f64 {
    type Output = Value;
    fn mul(self, v: Value) -> Value {
        Value::var(self) * v
    }
}

impl Div<Value> for f64 {
    type Output = Value;
    fn div(self, v: Value) -> Value {
        Value::var(self) / v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-7;

    fn central(f: impl Fn(f64) -> f64, x: f64) -> f64 {
        (f(x + EPS) - f(x - EPS)) / (2.0 * EPS)
    }

    #[test]
    fn rosenbrock_cost_gradient() {
        // ½((1−x)² + 10(y−x²)²): ∇ at (0,0) = (−1, 0), at (−1, 1) = (−4, 20).
        let build = |x0: f64, y0: f64| {
            let x = Value::var(x0);
            let y = Value::var(y0);
            let r1 = 1.0 - x.clone();
            let r2 = 10.0 * (y.clone() - x.clone() * x.clone());
            let cost = 0.5 * (r1.clone() * r1.clone() + r2.clone() * r2.clone());
            (x, y, cost)
        };
        let (x, y, cost) = build(0.0, 0.0);
        cost.backward();
        assert!((x.grad() + 1.0).abs() < 1e-12, "gx = {}", x.grad());
        assert!(y.grad().abs() < 1e-12, "gy = {}", y.grad());
        // With the ½ factor this cost is half the classic Rosenbrock, so the
        // gradient at (−1, 1) is (−2, 0): r₁ = 2, r₂ = 0 there.
        let (x, y, cost) = build(-1.0, 1.0);
        cost.backward();
        assert!((x.grad() + 2.0).abs() < 1e-12, "gx = {}", x.grad());
        assert!(y.grad().abs() < 1e-12, "gy = {}", y.grad());
    }

    #[test]
    fn diamond_graph_accumulates() {
        // f = x·x + x·3 (x used twice): f' = 2x + 3.
        let x = Value::var(2.0);
        let f = x.clone() * x.clone() + x.clone() * 3.0;
        f.backward();
        assert!((f.value() - 10.0).abs() < 1e-12);
        assert!((x.grad() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn unary_ops_match_central_differences() {
        for &t in &[0.4, 1.1, 2.2] {
            let v = Value::var(t);
            let f = v.sqrt().ln() + v.clone().powi(3) - v.clone().powf(&Value::var(1.5))
                + v.clone().abs().hypot(&Value::var(1.0))
                + v.sin() * v.cos()
                + v.clone().tan()
                + v.clone().sinh() * v.clone().cosh()
                + v.exp() * Value::var(0.5);
            let g0 = {
                f.backward();
                v.grad()
            };
            let expect = central(
                |x| {
                    x.sqrt().ln() + x.powi(3) - x.powf(1.5)
                        + x.abs().hypot(1.0)
                        + x.sin() * x.cos()
                        + x.tan()
                        + x.sinh() * x.cosh()
                        + x.exp() * 0.5
                },
                t,
            );
            assert!((g0 - expect).abs() < 1e-5, "t={t} got {g0} want {expect}");
        }
    }

    #[test]
    fn division_and_atan2() {
        let a = Value::var(1.0);
        let b = Value::var(2.0);
        let f = a.clone() / b.clone() + a.clone().atan2(&b.clone());
        f.backward();
        // ∂(a/b)/∂a = 1/b = 0.5; ∂atan2(a,b)/∂a = b/(a²+b²) = 0.4 → 0.9.
        assert!((a.grad() - 0.9).abs() < 1e-12);
        // ∂(a/b)/∂b = −a/b² = −0.25; ∂atan2/∂b = −a/(a²+b²) = −0.2 → −0.45.
        assert!((b.grad() + 0.45).abs() < 1e-12);
    }

    #[test]
    fn jacobian_of_vector_function() {
        // r(x, y) = (x·y, x² − y, e^x·y).
        let x = Value::var(1.3);
        let y = Value::var(0.7);
        let outs = vec![
            x.clone() * y.clone(),
            x.clone() * x.clone() - y.clone(),
            x.clone().exp() * y.clone(),
        ];
        let jac = Value::jacobian(&[x.clone(), y.clone()], &outs);
        let e = 1.3_f64.exp();
        let expect = [0.7, 1.3, 2.6, -1.0, e * 0.7, e];
        for k in 0..6 {
            assert!((jac[k] - expect[k]).abs() < 1e-12, "jac[{k}] = {}", jac[k]);
        }
    }

    #[test]
    fn deep_chain_no_stack_overflow() {
        // 50 000-deep chain: iterative topo must handle it.
        let mut v = Value::var(1.01);
        for _ in 0..50_000 {
            v = v * 1.0000001 + 1e-9;
        }
        v.backward();
        // f' = 1.0000001^50000 — just check it finished and is finite/positive.
        assert!(v.grad().is_finite() && v.grad() > 0.0);
    }
}
