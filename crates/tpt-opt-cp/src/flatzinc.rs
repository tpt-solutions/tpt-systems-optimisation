//! A FlatZinc subset: parse a flattened constraint model and solve it with the
//! CP engine.
//!
//! Supported statements:
//!
//! - `var lo..hi: x;`, `var {a, b, c}: x;`, `var bool: x;`, and fixed
//!   `var int: x = v;`, each with optional `:: annotation` text;
//! - `array [1..n] of var ...: a :: annotation = [x, y, ...];` and
//!   `array [1..n] of int: c = [1, 2, ...];`, which alias scalars or constants;
//! - `int: n = v;` named constants;
//! - `constraint int_lin_le(c, x, rhs);`, `int_lin_eq`, `int_lin_lt`,
//!   `int_eq(a, b)`, `int_le`, `int_lt`, and `all_different_int(xs)`;
//! - `solve satisfy;`, `solve minimize x;`, `solve maximize x;`.
//!
//! Variables are shifted so the CP domains start at zero. The shift is common to
//! every variable, so `all_different` and the linear constraints keep their
//! meaning. Optimisation enumerates every solution, which is exact but only for
//! models with few solutions. Above [`MAX_ENUMERATED`] the solver refuses to
//! answer, rather than report a truncated optimum.
//!
//! Anything outside this subset is an error, not silently ignored.

use std::collections::HashMap;
use std::fmt;
use std::vec::Vec;

use crate::constraints::{AllDifferent, Linear};
use crate::model::{CpModel, Relation};
use crate::solver::{solutions, solve};

/// Largest number of solutions enumerated for an optimisation goal.
pub const MAX_ENUMERATED: usize = 100_000;

/// A parse or solve failure, with a message naming the offending construct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatZincError(pub String);

impl fmt::Display for FlatZincError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FlatZinc: {}", self.0)
    }
}

impl std::error::Error for FlatZincError {}

fn err<T>(msg: impl Into<String>) -> Result<T, FlatZincError> {
    Err(FlatZincError(msg.into()))
}

/// The solve goal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Goal {
    /// Find any solution.
    Satisfy,
    /// Minimise the named variable.
    Minimize(String),
    /// Maximise the named variable.
    Maximize(String),
}

/// One solution: every named scalar variable with its value.
pub type Assignment = Vec<(String, i64)>;

/// A parsed FlatZinc model, ready to solve.
pub struct FlatZincProgram {
    model: CpModel,
    /// Scalar variables in declaration order: `(name, CP variable index)`.
    names: Vec<(String, usize)>,
    /// The shift applied to every variable: original value = CP value + shift.
    shift: i64,
    goal: Goal,
}

impl FlatZincProgram {
    /// The solve goal.
    pub fn goal(&self) -> &Goal {
        &self.goal
    }

    /// Solve the model.
    ///
    /// Returns `Ok(None)` when there is no solution. For a satisfaction goal the
    /// first solution found is returned. For an optimisation goal every solution
    /// is enumerated and the best one returned.
    pub fn solve(&self) -> Result<Option<Assignment>, FlatZincError> {
        let to_assignment = |values: &[usize]| -> Assignment {
            self.names
                .iter()
                .map(|(name, idx)| (name.clone(), values[*idx] as i64 + self.shift))
                .collect()
        };
        match &self.goal {
            Goal::Satisfy => Ok(solve(&self.model).map(|s| to_assignment(&s.assignment))),
            Goal::Minimize(name) | Goal::Maximize(name) => {
                let idx = self.index_of(name)?;
                let all = solutions(&self.model, MAX_ENUMERATED + 1);
                if all.len() > MAX_ENUMERATED {
                    return err(format!(
                        "more than {MAX_ENUMERATED} solutions; optimisation by enumeration is not feasible"
                    ));
                }
                let minimise = matches!(self.goal, Goal::Minimize(_));
                let best = all.iter().min_by_key(|s| {
                    let v = s.assignment[idx] as i64;
                    if minimise {
                        v
                    } else {
                        -v
                    }
                });
                Ok(best.map(|s| to_assignment(&s.assignment)))
            }
        }
    }

    fn index_of(&self, name: &str) -> Result<usize, FlatZincError> {
        self.names
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, idx)| *idx)
            .ok_or_else(|| FlatZincError(format!("objective variable {name} is not declared")))
    }
}

/// Parse FlatZinc text into a solvable [`FlatZincProgram`].
pub fn parse(text: &str) -> Result<FlatZincProgram, FlatZincError> {
    let tokens = tokenize(text)?;
    let statements: Vec<&[Tok]> =
        tokens.split(|t| *t == Tok::Sym(';')).filter(|s| !s.is_empty()).collect();

    let mut decls = Declarations::default();
    let mut constraints: Vec<Stmt> = Vec::new();
    let mut goal = Goal::Satisfy;
    for stmt in statements {
        match stmt.first() {
            Some(Tok::Ident(k)) if k == "var" => decls.var(stmt)?,
            Some(Tok::Ident(k)) if k == "array" => decls.array(stmt)?,
            Some(Tok::Ident(k)) if k == "int" || k == "bool" => decls.constant(stmt)?,
            Some(Tok::Ident(k)) if k == "constraint" => {
                constraints.push(Stmt::Constraint(stmt.to_vec()))
            }
            Some(Tok::Ident(k)) if k == "solve" => goal = parse_goal(stmt)?,
            Some(Tok::Ident(k)) if k == "output" => {}
            Some(Tok::Ident(k)) if k == "predicate" || k == "function" => {
                return err(format!("{k} declarations are not supported"));
            }
            _ => return err(format!("unsupported statement starting with {:?}", stmt[0])),
        }
    }

    // Common shift: the smallest declared lower bound, so every domain starts at 0.
    let shift = decls.scalars.iter().map(|s| s.lo).min().unwrap_or(0);
    let mut model = CpModel::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut names = Vec::new();
    for s in &decls.scalars {
        let idx = match &s.domain {
            Domain::Range(lo, hi) => model.add_var((lo - shift) as usize, (hi - shift) as usize),
            Domain::Set(values) => {
                model.add_var_values(values.iter().map(|v| (v - shift) as usize).collect())
            }
        };
        index.insert(s.name.clone(), idx);
        names.push((s.name.clone(), idx));
    }

    let resolver = Resolver { index: &index, decls: &decls };
    for stmt in &constraints {
        let Stmt::Constraint(tokens) = stmt;
        let con = Constraint::parse(tokens, &resolver)?;
        con.add_to(&mut model, shift)?;
    }

    if let Goal::Minimize(name) | Goal::Maximize(name) = &goal {
        if !index.contains_key(name) {
            return err(format!("objective variable {name} is not declared"));
        }
    }
    Ok(FlatZincProgram { model, names, shift, goal })
}

// ---- tokens ----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Ident(String),
    Int(i64),
    /// `..`, `::`, or a single punctuation character.
    Sym(char),
    DotDot,
    DoubleColon,
}

fn tokenize(text: &str) -> Result<Vec<Tok>, FlatZincError> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '%' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push(Tok::Ident(chars[start..i].iter().collect()));
        } else if c.is_ascii_digit()
            || (c == '-' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit()))
        {
            let start = i;
            i += 1;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let text: String = chars[start..i].iter().collect();
            let value = text
                .parse::<i64>()
                .map_err(|_| FlatZincError(format!("integer {text} is out of range")))?;
            out.push(Tok::Int(value));
        } else if c == '.' && chars.get(i + 1) == Some(&'.') {
            out.push(Tok::DotDot);
            i += 2;
        } else if c == ':' && chars.get(i + 1) == Some(&':') {
            out.push(Tok::DoubleColon);
            i += 2;
        } else if "[](){},:;=<".contains(c) {
            out.push(Tok::Sym(c));
            i += 1;
        } else {
            return err(format!("unexpected character {c:?}"));
        }
    }
    Ok(out)
}

// ---- declarations ----------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum Domain {
    Range(i64, i64),
    Set(Vec<i64>),
}

impl Domain {
    fn lo(&self) -> i64 {
        match self {
            Domain::Range(lo, _) => *lo,
            Domain::Set(values) => values.iter().copied().min().unwrap_or(0),
        }
    }
}

#[derive(Debug, Clone)]
struct Scalar {
    name: String,
    domain: Domain,
    lo: i64,
}

#[derive(Debug, Default)]
struct Declarations {
    scalars: Vec<Scalar>,
    /// Named integer constants (`int: n = 3;`).
    consts: HashMap<String, i64>,
    /// Arrays: name -> items (variable names or integer literals).
    arrays: HashMap<String, Vec<Item>>,
}

/// A named constant or literal, possibly a scalar variable.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Item {
    Name(String),
    Int(i64),
}

impl Declarations {
    fn var(&mut self, stmt: &[Tok]) -> Result<(), FlatZincError> {
        let mut p = Cursor::new(stmt, 1);
        let domain = p.domain()?;
        p.expect_sym(':')?;
        let name = p.ident()?;
        p.skip_annotation();
        if p.eat_sym('=') {
            let value = p.int()?;
            self.scalars.push(Scalar { name, lo: value, domain: Domain::Range(value, value) });
        } else {
            let domain =
                domain.ok_or_else(|| FlatZincError(format!("variable {name} has no domain")))?;
            let lo = domain.lo();
            self.scalars.push(Scalar { name, lo, domain });
        }
        Ok(())
    }

    fn array(&mut self, stmt: &[Tok]) -> Result<(), FlatZincError> {
        // array [1..n] of <var|int> <domain>: name :: ann = [items];
        let mut p = Cursor::new(stmt, 1);
        p.expect_sym('[')?;
        while !p.eat_sym(']') {
            p.advance()?;
        }
        p.keyword("of")?;
        p.eat_keyword("var");
        // The element domain is ignored; elements are declared scalars or constants.
        let _ = p.domain()?;
        p.expect_sym(':')?;
        let name = p.ident()?;
        p.skip_annotation();
        p.expect_sym('=')?;
        p.expect_sym('[')?;
        let mut items = Vec::new();
        while !p.eat_sym(']') {
            items.push(p.item()?);
            p.eat_sym(',');
        }
        self.arrays.insert(name, items);
        Ok(())
    }

    fn constant(&mut self, stmt: &[Tok]) -> Result<(), FlatZincError> {
        // int: n = v;   or   bool: b = true;  (only integer constants are kept)
        let mut p = Cursor::new(stmt, 1);
        p.expect_sym(':')?;
        let name = p.ident()?;
        p.expect_sym('=')?;
        let value = p.int()?;
        self.consts.insert(name, value);
        Ok(())
    }
}

// ---- constraints and solve -------------------------------------------------

enum Stmt {
    Constraint(Vec<Tok>),
}

/// What a constraint argument can refer to.
struct Resolver<'a> {
    index: &'a HashMap<String, usize>,
    decls: &'a Declarations,
}

impl Resolver<'_> {
    /// Resolve an item to a CP variable, or to an integer constant.
    fn item(&self, item: &Item) -> Result<Term, FlatZincError> {
        match item {
            Item::Int(v) => Ok(Term::Const(*v)),
            Item::Name(n) => {
                if let Some(idx) = self.index.get(n) {
                    Ok(Term::Var(*idx))
                } else if let Some(v) = self.decls.consts.get(n) {
                    Ok(Term::Const(*v))
                } else {
                    err(format!("unknown name {n}"))
                }
            }
        }
    }

    fn list(&self, name: &str) -> Result<Vec<Term>, FlatZincError> {
        let items = self
            .decls
            .arrays
            .get(name)
            .ok_or_else(|| FlatZincError(format!("unknown array {name}")))?;
        items.iter().map(|i| self.item(i)).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Term {
    Var(usize),
    Const(i64),
}

/// A constraint argument: a list, or a single item.
enum Arg {
    List(Vec<Item>),
    Scalar(Item),
}

/// A supported, resolved constraint ready to post.
enum Constraint {
    /// `Σ c_i · t_i  rel  rhs`, with each term a variable or a constant.
    Linear {
        terms: Vec<(Term, i64)>,
        rel: Relation,
        rhs: i64,
    },
    AllDifferent(Vec<usize>),
}

impl Constraint {
    fn parse(tokens: &[Tok], r: &Resolver<'_>) -> Result<Self, FlatZincError> {
        // constraint <name>( <args> )
        let mut p = Cursor::new(tokens, 1);
        let name = p.ident()?;
        p.expect_sym('(')?;
        let mut args: Vec<Arg> = Vec::new();
        while !p.eat_sym(')') {
            if p.eat_sym('[') {
                let mut items = Vec::new();
                while !p.eat_sym(']') {
                    items.push(p.item()?);
                    p.eat_sym(',');
                }
                args.push(Arg::List(items));
            } else {
                args.push(Arg::Scalar(p.item()?));
            }
            p.eat_sym(',');
        }
        p.skip_annotation();

        let list = |arg: &Arg| -> Result<Vec<Term>, FlatZincError> {
            match arg {
                Arg::List(items) => items.iter().map(|i| r.item(i)).collect(),
                Arg::Scalar(Item::Name(n)) => r.list(n),
                Arg::Scalar(_) => err(format!("{name} expects an array")),
            }
        };
        let scalar = |arg: &Arg| -> Result<Term, FlatZincError> {
            match arg {
                Arg::Scalar(item) => r.item(item),
                Arg::List(_) => err(format!("{name} expects a scalar")),
            }
        };

        match name.as_str() {
            "int_lin_le" | "int_lin_eq" | "int_lin_lt" => {
                if args.len() != 3 {
                    return err(format!("{name} takes three arguments"));
                }
                let coeffs = list(&args[0])?;
                let vars = list(&args[1])?;
                let rhs = match scalar(&args[2])? {
                    Term::Const(v) => v,
                    Term::Var(_) => return err(format!("{name} needs a constant right-hand side")),
                };
                if coeffs.len() != vars.len() {
                    return err(format!("{name}: coefficient and variable counts differ"));
                }
                let mut terms = Vec::with_capacity(vars.len());
                for (c, v) in coeffs.iter().zip(&vars) {
                    match (c, v) {
                        (Term::Const(c), v) => terms.push((*v, *c)),
                        (Term::Var(_), _) => {
                            return err(format!("{name}: coefficients must be constants"))
                        }
                    }
                }
                let (rel, rhs) = match name.as_str() {
                    "int_lin_le" => (Relation::Le, rhs),
                    "int_lin_lt" => (Relation::Le, rhs - 1),
                    _ => (Relation::Eq, rhs),
                };
                Ok(Constraint::Linear { terms, rel, rhs })
            }
            "int_eq" | "int_le" | "int_lt" => {
                if args.len() != 2 {
                    return err(format!("{name} takes two arguments"));
                }
                let a = scalar(&args[0])?;
                let b = scalar(&args[1])?;
                // a - b  rel  0   (or -1 for the strict form)
                let (rel, rhs) = match name.as_str() {
                    "int_eq" => (Relation::Eq, 0),
                    "int_le" => (Relation::Le, 0),
                    _ => (Relation::Le, -1),
                };
                Ok(Constraint::Linear { terms: vec![(a, 1), (b, -1)], rel, rhs })
            }
            "all_different_int" => {
                if args.len() != 1 {
                    return err("all_different_int takes one array");
                }
                let vars = list(&args[0])?
                    .into_iter()
                    .map(|t| match t {
                        Term::Var(idx) => Ok(idx),
                        Term::Const(_) => err("all_different_int over constants is not supported"),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Constraint::AllDifferent(vars))
            }
            other => err(format!("unsupported constraint {other}")),
        }
    }

    fn add_to(self, model: &mut CpModel, shift: i64) -> Result<(), FlatZincError> {
        match self {
            Constraint::Linear { terms, rel, rhs } => {
                // Variables carry the shift: x = y + shift, so Σ c·x = Σ c·y + shift·Σ c,
                // and constants are the original values.
                let mut cp_terms = Vec::new();
                let mut var_coeff_sum = 0i64;
                let mut const_sum = 0i64;
                for (t, c) in terms {
                    match t {
                        Term::Var(idx) => {
                            cp_terms.push((idx, c));
                            var_coeff_sum += c;
                        }
                        Term::Const(v) => const_sum += c * v,
                    }
                }
                let rhs = rhs
                    .checked_sub(shift.checked_mul(var_coeff_sum).ok_or_else(overflow)?)
                    .and_then(|r| r.checked_sub(const_sum))
                    .ok_or_else(overflow)?;
                model.add_constraint(Box::new(Linear::new(cp_terms, rel, rhs)));
                Ok(())
            }
            Constraint::AllDifferent(vars) => {
                model.add_constraint(Box::new(AllDifferent::new(vars)));
                Ok(())
            }
        }
    }
}

fn overflow() -> FlatZincError {
    FlatZincError("integer overflow in a constraint's right-hand side".into())
}

fn parse_goal(stmt: &[Tok]) -> Result<Goal, FlatZincError> {
    let mut p = Cursor::new(stmt, 1);
    let kind = p.ident()?;
    match kind.as_str() {
        "satisfy" => Ok(Goal::Satisfy),
        "minimize" => Ok(Goal::Minimize(p.ident()?)),
        "maximize" => Ok(Goal::Maximize(p.ident()?)),
        other => err(format!("unsupported solve goal {other}")),
    }
}

// ---- token cursor ----------------------------------------------------------

struct Cursor<'a> {
    tokens: &'a [Tok],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(tokens: &'a [Tok], pos: usize) -> Self {
        Self { tokens, pos }
    }

    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Result<Tok, FlatZincError> {
        let t = self
            .tokens
            .get(self.pos)
            .cloned()
            .ok_or_else(|| FlatZincError("unexpected end of statement".into()))?;
        self.pos += 1;
        Ok(t)
    }

    fn eat_sym(&mut self, c: char) -> bool {
        if self.peek() == Some(&Tok::Sym(c)) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_sym(&mut self, c: char) -> Result<(), FlatZincError> {
        if self.eat_sym(c) {
            Ok(())
        } else {
            err(format!("expected {c:?}, found {:?}", self.peek()))
        }
    }

    fn ident(&mut self) -> Result<String, FlatZincError> {
        match self.advance()? {
            Tok::Ident(s) => Ok(s),
            other => err(format!("expected a name, found {other:?}")),
        }
    }

    fn int(&mut self) -> Result<i64, FlatZincError> {
        match self.advance()? {
            Tok::Int(v) => Ok(v),
            other => err(format!("expected an integer, found {other:?}")),
        }
    }

    fn keyword(&mut self, word: &str) -> Result<(), FlatZincError> {
        match self.advance()? {
            Tok::Ident(s) if s == word => Ok(()),
            other => err(format!("expected {word}, found {other:?}")),
        }
    }

    fn eat_keyword(&mut self, word: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Ident(s)) if s == word) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// A variable domain: `int`, `bool`, `lo..hi`, or `{a, b, ...}`. `int` and
    /// `bool` without bounds are rejected, because the CP engine needs finite
    /// domains. Returns `None` for a bare `int` (to be given an `=` value).
    fn domain(&mut self) -> Result<Option<Domain>, FlatZincError> {
        match self.peek().cloned() {
            Some(Tok::Ident(s)) if s == "int" => {
                self.pos += 1;
                if self.peek() == Some(&Tok::Sym(':')) {
                    Ok(None)
                } else {
                    err("unbounded integer domains are not supported; give a range")
                }
            }
            Some(Tok::Ident(s)) if s == "bool" => {
                self.pos += 1;
                Ok(Some(Domain::Range(0, 1)))
            }
            Some(Tok::Int(lo)) => {
                self.pos += 1;
                if self.peek() != Some(&Tok::DotDot) {
                    return err("expected .. in a range domain");
                }
                self.pos += 1;
                let hi = self.int()?;
                if hi < lo {
                    return err(format!("empty domain {lo}..{hi}"));
                }
                Ok(Some(Domain::Range(lo, hi)))
            }
            Some(Tok::Sym('{')) => {
                self.pos += 1;
                let mut values = Vec::new();
                while !self.eat_sym('}') {
                    values.push(self.int()?);
                    self.eat_sym(',');
                }
                if values.is_empty() {
                    return err("empty set domain");
                }
                Ok(Some(Domain::Set(values)))
            }
            other => err(format!("unsupported variable domain starting with {other:?}")),
        }
    }

    fn item(&mut self) -> Result<Item, FlatZincError> {
        match self.advance()? {
            Tok::Int(v) => Ok(Item::Int(v)),
            Tok::Ident(s) => Ok(Item::Name(s)),
            other => err(format!("expected a name or integer, found {other:?}")),
        }
    }

    /// Skip a `:: annotation` (with any parenthesised arguments) if present.
    fn skip_annotation(&mut self) {
        if self.peek() != Some(&Tok::DoubleColon) {
            return;
        }
        self.pos += 1;
        let mut depth = 0i32;
        while let Some(t) = self.peek() {
            match t {
                Tok::Sym('(') | Tok::Sym('[') => depth += 1,
                Tok::Sym(')') | Tok::Sym(']') => depth -= 1,
                Tok::Sym('=') | Tok::Sym(';') if depth == 0 => break,
                _ => {}
            }
            self.pos += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_handles_ranges_negatives_and_comments() {
        let toks = tokenize("var -2..3: x; % a comment\n").unwrap();
        assert_eq!(
            toks,
            vec![
                Tok::Ident("var".into()),
                Tok::Int(-2),
                Tok::DotDot,
                Tok::Int(3),
                Tok::Sym(':'),
                Tok::Ident("x".into()),
                Tok::Sym(';'),
            ]
        );
    }

    #[test]
    fn satisfaction_with_linear_and_alldifferent() {
        let text = "\
var 1..3: x :: output_var;
var 1..3: y :: output_var;
var 1..3: z :: output_var;
array [1..3] of var int: a :: output_array([1..3]) = [x, y, z];
constraint all_different_int(a);
constraint int_lin_eq([1, 1, 1], [x, y, z], 6);
constraint int_lt(x, y);
solve satisfy;
";
        let program = parse(text).unwrap();
        let sol = program.solve().unwrap().expect("solvable");
        let get = |n: &str| sol.iter().find(|(name, _)| name == n).unwrap().1;
        assert_eq!(get("x") + get("y") + get("z"), 6);
        assert!(get("x") < get("y"));
        assert_ne!(get("x"), get("z"));
        assert_ne!(get("y"), get("z"));
    }

    #[test]
    fn negative_domains_are_shifted_correctly() {
        let text = "\
var -3..-1: x;
var -3..-1: y;
constraint int_lin_le([1, 1], [x, y], -5);
solve maximize x;
";
        let program = parse(text).unwrap();
        let sol = program.solve().unwrap().expect("solvable");
        let get = |n: &str| sol.iter().find(|(name, _)| name == n).unwrap().1;
        // x + y ≤ -5 with both in [-3, -1]: the largest x is -2 (y = -3).
        assert_eq!(get("x"), -2);
        assert!(get("x") + get("y") <= -5);
    }

    #[test]
    fn optimisation_returns_the_best_solution() {
        let text = "\
var 0..4: x;
var 0..4: y;
constraint int_lin_le([1, 1], [x, y], 5);
solve minimize y;
";
        let program = parse(text).unwrap();
        let sol = program.solve().unwrap().expect("solvable");
        let y = sol.iter().find(|(n, _)| n == "y").unwrap().1;
        assert_eq!(y, 0);
    }

    #[test]
    fn infeasible_model_returns_none() {
        let text = "var 0..2: x; constraint int_lin_eq([1], [x], 5); solve satisfy;";
        assert_eq!(parse(text).unwrap().solve().unwrap(), None);
    }

    #[test]
    fn unsupported_constructs_are_errors_not_silent() {
        assert!(parse("var int: x; solve satisfy;").is_err());
        assert!(parse("var 0..1: x; constraint int_ne(x, x); solve satisfy;").is_err());
        assert!(parse("predicate foo(var int: x);").is_err());
        assert!(parse("var 0..1: x; solve minimize q;").is_err());
    }
}
