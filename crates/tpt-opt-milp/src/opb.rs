//! OPB (pseudo-Boolean) reader and writer.
//!
//! OPB is the PB-competition format for linear constraints over 0/1 variables
//! `x1 … xn`. The supported subset is:
//!
//! - an optional `* #variable= n #constraint= m` header and `*` comment lines;
//! - one objective line, `min:` or `max:`, ending in `;`;
//! - one constraint per line: terms, one of `>=`, `<=`, `=` (also `=<`, `=>`),
//!   an integer right-hand side, and a closing `;`;
//! - terms as whitespace-separated `±coefficient xIndex` pairs, with integer
//!   coefficients.
//!
//! Every variable is binary. Negated literals (`~x`) and non-integer
//! coefficients are rejected rather than approximated. The writer applies the
//! same restrictions, so a model must be integral and binary to be written.

use tpt_opt_core::bounds::{VarBound, VarType};
use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::OptError;

/// Parse OPB text into a [`Model`] with binary variables.
///
/// Variables are created up to the largest index mentioned, or up to the header
/// count if that is larger.
pub fn read_opb(input: &str) -> Result<Model, OptError> {
    let mut header_vars = 0usize;
    let mut objective: Option<(Sense, Terms)> = None;
    let mut rows: Vec<(Terms, Op, i64)> = Vec::new();
    let mut max_index = 0usize;

    for raw in input.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('*') {
            if let Some(n) = header_count(rest, "#variable=") {
                header_vars = n;
            }
            continue;
        }
        let body = line.strip_suffix(';').unwrap_or(line).trim();
        let lower = body.to_ascii_lowercase();

        if lower.starts_with("min:") || lower.starts_with("max:") {
            if objective.is_some() {
                return err("OPB: more than one objective line");
            }
            let sense = if lower.starts_with("min:") { Sense::Minimize } else { Sense::Maximize };
            let terms = parse_terms(&body[4..], &mut max_index)?;
            objective = Some((sense, terms));
            continue;
        }

        let (lhs, op, rhs) = split_relation(body)?;
        let terms = parse_terms(lhs, &mut max_index)?;
        rows.push((terms, op, rhs));
    }

    let (sense, obj_terms) = objective.unwrap_or((Sense::Minimize, Vec::new()));
    let n = header_vars.max(max_index);
    let mut model = Model::new(n);
    for j in 0..n {
        model.variables[j] = tpt_opt_core::model::Variable::new(j, VarBound::binary());
    }

    let (indices, coeffs) = to_vectors(&obj_terms);
    model.set_objective(match sense {
        Sense::Minimize => Objective::minimize(indices, coeffs),
        Sense::Maximize => Objective::maximize(indices, coeffs),
    });

    for (terms, op, rhs) in rows {
        let (indices, coeffs) = to_vectors(&terms);
        let rhs = rhs as f64;
        let (lower, upper) = match op {
            Op::Ge => (rhs, f64::INFINITY),
            Op::Le => (f64::NEG_INFINITY, rhs),
            Op::Eq => (rhs, rhs),
        };
        model.add_constraint(Constraint::new(indices, coeffs, lower, upper)?);
    }
    model.validate()?;
    Ok(model)
}

/// Serialise a binary, integral [`Model`] as OPB text.
///
/// Returns an error when a variable is not binary, a coefficient or bound is
/// not an integer, the objective has a constant term, or a constraint is
/// unbounded on both sides.
pub fn write_opb(model: &Model) -> Result<String, OptError> {
    for v in &model.variables {
        let binary_kind = matches!(v.kind, VarType::Binary | VarType::Integer);
        if !binary_kind || v.bound.bound.lower != 0.0 || v.bound.bound.upper != 1.0 {
            return err("OPB writer: every variable must be binary (0/1 integer)");
        }
    }
    if model.objective.constant != 0.0 {
        return err("OPB writer: objective constant cannot be represented");
    }

    let mut out = String::new();
    out.push_str(&format!(
        "* #variable= {} #constraint= {}\n",
        model.num_vars,
        model.num_constraints()
    ));
    let obj_terms = from_model_terms(&model.objective.indices, &model.objective.coeffs)?;
    let keyword = match model.objective.sense {
        Sense::Minimize => "min:",
        Sense::Maximize => "max:",
    };
    out.push_str(&format!("{keyword} {};\n", render_terms(&obj_terms)));

    for c in &model.constraints {
        if c.is_custom {
            return err("OPB writer: custom constraints cannot be written");
        }
        let terms = from_model_terms(&c.indices, &c.coeffs)?;
        let body = render_terms(&terms);
        if c.lower == c.upper {
            out.push_str(&format!("{body} = {};\n", int_value(c.lower)?));
            continue;
        }
        if c.lower.is_finite() {
            out.push_str(&format!("{body} >= {};\n", int_value(c.lower)?));
        }
        if c.upper.is_finite() {
            out.push_str(&format!("{body} <= {};\n", int_value(c.upper)?));
        }
        if !c.lower.is_finite() && !c.upper.is_finite() {
            return err("OPB writer: a constraint must be bounded on at least one side");
        }
    }
    Ok(out)
}

/// `(coefficient, zero-based variable index)` pairs.
type Terms = Vec<(i64, usize)>;

#[derive(Clone, Copy)]
enum Op {
    Ge,
    Le,
    Eq,
}

fn err<T>(msg: &str) -> Result<T, OptError> {
    Err(OptError::invalid_model(msg))
}

/// Read `#variable=` / `#constraint=`-style counts from a header comment body.
fn header_count(rest: &str, key: &str) -> Option<usize> {
    let pos = rest.find(key)?;
    rest[pos + key.len()..].split_whitespace().next()?.parse().ok()
}

/// Find the relation operator and parse the integer right-hand side.
fn split_relation(body: &str) -> Result<(&str, Op, i64), OptError> {
    for (token, op) in
        [(">=", Op::Ge), ("=>", Op::Ge), ("<=", Op::Le), ("=<", Op::Le), ("=", Op::Eq)]
    {
        if let Some(pos) = body.find(token) {
            let rhs_text = body[pos + token.len()..].trim();
            let rhs = rhs_text.parse::<i64>().map_err(|_| {
                OptError::invalid_model(format!("OPB: bad right-hand side `{rhs_text}`"))
            })?;
            return Ok((body[..pos].trim(), op, rhs));
        }
    }
    err("OPB: constraint has no relation operator")
}

/// Parse whitespace-separated `±coeff xIndex` pairs, noting the largest index.
fn parse_terms(text: &str, max_index: &mut usize) -> Result<Terms, OptError> {
    let mut terms = Vec::new();
    let mut pending: Option<i64> = None;
    for token in text.split_whitespace() {
        if let Some(index) = token.strip_prefix('x').or_else(|| token.strip_prefix('X')) {
            let idx: usize = index
                .parse()
                .map_err(|_| OptError::invalid_model(format!("OPB: bad variable `{token}`")))?;
            if idx == 0 {
                return err("OPB: variables are numbered from x1");
            }
            let coeff = pending.take().unwrap_or(1);
            *max_index = (*max_index).max(idx);
            terms.push((coeff, idx - 1));
        } else if token.starts_with('~') {
            return err("OPB: negated literals are not supported");
        } else {
            let coeff: i64 = token
                .parse()
                .map_err(|_| OptError::invalid_model(format!("OPB: bad coefficient `{token}`")))?;
            if pending.is_some() {
                return err("OPB: coefficient without a variable");
            }
            pending = Some(coeff);
        }
    }
    if pending.is_some() {
        return err("OPB: trailing coefficient without a variable");
    }
    Ok(terms)
}

fn to_vectors(terms: &[(i64, usize)]) -> (Vec<usize>, Vec<f64>) {
    terms.iter().map(|&(c, j)| (j, c as f64)).unzip()
}

fn from_model_terms(indices: &[usize], coeffs: &[f64]) -> Result<Terms, OptError> {
    indices
        .iter()
        .zip(coeffs)
        .filter(|&(_, &c)| c != 0.0)
        .map(|(&j, &c)| {
            if c.fract() != 0.0 || !c.is_finite() {
                err("OPB writer: coefficients must be integers")
            } else {
                Ok((c as i64, j))
            }
        })
        .collect()
}

fn render_terms(terms: &[(i64, usize)]) -> String {
    if terms.is_empty() {
        return "0 x1".into();
    }
    terms.iter().map(|&(c, j)| format!("{c:+} x{}", j + 1)).collect::<Vec<_>>().join(" ")
}

fn int_value(v: f64) -> Result<i64, OptError> {
    if v.fract() == 0.0 && v.is_finite() {
        Ok(v as i64)
    } else {
        err("OPB writer: right-hand sides must be integers")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_opt_core::expr::ModelBuilder;
    use tpt_opt_core::Solver;

    const SAMPLE: &str = "\
* #variable= 3 #constraint= 2
min: 3 x1 +4 x2 -2 x3 ;
+1 x1 +1 x2 +1 x3 >= 1 ;
+2 x1 -1 x3 <= 1 ;
";

    #[test]
    fn reads_header_objective_and_constraints() {
        let m = read_opb(SAMPLE).unwrap();
        assert_eq!(m.num_vars, 3);
        assert_eq!(m.objective.sense, Sense::Minimize);
        assert_eq!(m.objective.coeffs, vec![3.0, 4.0, -2.0]);
        assert_eq!(m.num_constraints(), 2);
        assert_eq!(m.constraints[0].lower, 1.0);
        assert_eq!(m.constraints[1].upper, 1.0);
        assert!(m.variables.iter().all(|v| v.is_binary()));
    }

    #[test]
    fn written_model_reads_back_with_the_same_optimum() {
        let m = read_opb(SAMPLE).unwrap();
        let text = write_opb(&m).unwrap();
        let back = read_opb(&text).unwrap();
        let a = crate::MilpSolver::new().solve(&m).unwrap().objective_value;
        let b = crate::MilpSolver::new().solve(&back).unwrap().objective_value;
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }

    #[test]
    fn maximise_with_equality_round_trips() {
        let mut b = ModelBuilder::new();
        let x = b.binary("x");
        let y = b.binary("y");
        b.constrain((x + y).eq(1.0));
        b.maximize(2.0 * x + 3.0 * y);
        let m = b.build().unwrap();
        let text = write_opb(&m).unwrap();
        assert!(text.contains("max:"));
        assert!(text.contains("= 1;"));
        let back = read_opb(&text).unwrap();
        assert_eq!(back.constraints[0].lower, 1.0);
        assert_eq!(back.constraints[0].upper, 1.0);
        assert_eq!(back.objective.sense, Sense::Maximize);
    }

    #[test]
    fn rejects_unsupported_input() {
        assert!(read_opb("min: 1.5 x1 ;\n").is_err());
        assert!(read_opb("min: ~x1 ;\n").is_err());
        assert!(read_opb("min: x0 ;\n").is_err());
        assert!(read_opb("min: x1 ;\n+1 x1 >= 1.5 ;\n").is_err());
        assert!(read_opb("min: x1 ;\n+1 x1 ;\n").is_err());
    }

    #[test]
    fn writer_rejects_what_opb_cannot_express() {
        let mut b = ModelBuilder::new();
        let x = b.continuous("x", 0.0, 1.0);
        b.minimize(x);
        assert!(write_opb(&b.build().unwrap()).is_err());

        let mut b = ModelBuilder::new();
        let x = b.binary("x");
        b.minimize(0.5 * x);
        assert!(write_opb(&b.build().unwrap()).is_err());

        let mut b = ModelBuilder::new();
        let x = b.binary("x");
        b.minimize(x + 2.0);
        assert!(write_opb(&b.build().unwrap()).is_err());
    }
}
