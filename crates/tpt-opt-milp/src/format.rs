//! MPS and CPLEX-LP file-format import/export for canonical [`Model`]s.
//!
//! Two free-format readers and two writers are provided:
//!
//! - [`read_mps`] / [`write_mps`] — the classic MPS interchange format
//!   (free-field variant: fields separated by whitespace rather than fixed
//!   columns). Supports `ROWS`/`COLUMNS`/`RHS`/`RANGES`/`BOUNDS`, integer
//!   markers (`INTORG`/`INTEND`), `OBJSENSE MAX`, and the common bound cards
//!   (`UP LO FX FR MI PL BV LI UI SC`). Values with magnitude ≥ 1e30 are
//!   treated as infinities per MPS convention; an RHS entry on the objective
//!   row is interpreted as the *negated* objective constant.
//! - [`read_lp`] / [`write_lp`] — the CPLEX-style LP format with `Minimize`
//!   / `Maximize`, `Subject To` (including double-bounded rows
//!   `lo <= expr <= hi`), `Bounds`, `General`, and `Binary` sections,
//!   `\` comments, and wrapped constraint lines.
//!
//! # Fidelity notes
//!
//! - Variable and row names are generated on export (`x1..xn`, `c1..cm`,
//!   objective `OBJ`) because [`Model`] does not carry per-variable names;
//!   imported names are likewise discarded.
//! - The LP format has no objective-constant syntax, so constants are dropped
//!   by [`write_lp`] (MPS round-trips them exactly).
//! - Semi-continuous variables export to MPS via the `SC` bound card when the
//!   lower limit is positive, and degrade to plain continuous variables in the
//!   LP writer (the LP format has no semi-continuous syntax).
//!
//! ```rust
//! use tpt_opt_milp::format::{read_mps, solve_parsed};
//!
//! // A tiny knapsack written as free-format MPS …
//! let mps = "\
//! NAME          KNAP
//! OBJSENSE
//!     MAX
//! ROWS
//!  N  OBJ
//!  L  CAP
//! COLUMNS
//!     MARKER                 'MARKER'                 'INTORG'
//!     X1        OBJ          5.0             CAP       2.0
//!     X2        OBJ          4.0             CAP       3.0
//!     MARKER                 'MARKER'                 'INTEND'
//! RHS
//!     RHS       CAP          4.0
//! ENDATA
//! ";
//! let model = read_mps(mps).unwrap();
//! let solution = solve_parsed(&model).unwrap();
//! assert_eq!(solution.objective_value, 10.0); // take x1 twice
//! ```

use std::collections::HashMap;
use std::fmt::Write as _;
use std::string::String;
use std::vec::Vec;

use tpt_opt_core::bounds::{Bound, VarBound, VarType};
use tpt_opt_core::error::OptError;
use tpt_opt_core::model::{Constraint, Model, Objective, Sense};
use tpt_opt_core::solver::Solver;

/// Magnitude at/above which an MPS numeric field means "infinity".
const INF_SENTINEL: f64 = 1e30;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn fmt_num(v: f64) -> String {
    format!("{v}")
}

fn err<T>(msg: impl Into<String>) -> Result<T, OptError> {
    Err(OptError::invalid_model(msg))
}

/// How a canonical row maps onto MPS row types.
enum RowClass {
    /// `lower == upper`.
    Equality,
    /// Only an upper bound is finite.
    Le,
    /// Only a lower bound is finite.
    Ge,
    /// Both sides finite and distinct — exported as an L row plus RANGES.
    Ranged,
}

fn classify(c: &Constraint) -> RowClass {
    if c.lower == c.upper {
        RowClass::Equality
    } else if c.lower == f64::NEG_INFINITY {
        RowClass::Le
    } else if c.upper == f64::INFINITY {
        RowClass::Ge
    } else {
        RowClass::Ranged
    }
}

// ---------------------------------------------------------------------------
// MPS writer
// ---------------------------------------------------------------------------

/// Serialise `model` as free-format MPS text.
///
/// Maximisation models are emitted with an `OBJSENSE MAX` section so the
/// sense survives a round-trip. See the [module docs](self) for fidelity
/// notes on names, constants, and semi-continuous variables.
pub fn write_mps(model: &Model) -> String {
    let mut s = String::new();
    match &model.name {
        Some(name) => {
            let _ = writeln!(s, "NAME          {name}");
        }
        None => s.push_str("NAME\n"),
    }
    if model.objective.sense == Sense::Maximize {
        s.push_str("OBJSENSE\n    MAX\n");
    }

    // ---- ROWS -------------------------------------------------------------
    s.push_str("ROWS\n N  OBJ\n");
    for (i, c) in model.constraints.iter().enumerate() {
        let tag = match classify(c) {
            RowClass::Equality => "E",
            RowClass::Le | RowClass::Ranged => "L",
            RowClass::Ge => "G",
        };
        let _ = writeln!(s, " {tag}  c{}", i + 1);
    }

    // ---- COLUMNS ----------------------------------------------------------
    // Per-column entries: (row name, coefficient), objective first.
    let mut columns: Vec<Vec<(String, f64)>> = vec![Vec::new(); model.num_vars];
    for (&j, &coef) in model.objective.indices.iter().zip(&model.objective.coeffs) {
        columns[j].push(("OBJ".to_string(), coef));
    }
    for (i, c) in model.constraints.iter().enumerate() {
        let row = format!("c{}", i + 1);
        for (&j, &coef) in c.indices.iter().zip(&c.coeffs) {
            columns[j].push((row.clone(), coef));
        }
    }

    s.push_str("COLUMNS\n");
    let mut in_int_block = false;
    for (j, col) in columns.iter().enumerate() {
        let integral = matches!(
            model.variables.get(j).map(|v| v.bound.kind),
            Some(VarType::Integer | VarType::Binary)
        );
        if integral && !in_int_block {
            let _ = writeln!(s, "    MARKER                 'MARKER'                 'INTORG'");
            in_int_block = true;
        } else if !integral && in_int_block {
            let _ = writeln!(s, "    MARKER                 'MARKER'                 'INTEND'");
            in_int_block = false;
        }
        let name = format!("x{}", j + 1);
        for (row, coef) in col {
            let _ = writeln!(s, "    {name:<9} {row:<9} {}", fmt_num(*coef));
        }
    }
    if in_int_block {
        let _ = writeln!(s, "    MARKER                 'MARKER'                 'INTEND'");
    }

    // ---- RHS --------------------------------------------------------------
    s.push_str("RHS\n");
    if model.objective.constant != 0.0 {
        // MPS convention: an RHS entry on the objective row negates the constant.
        let _ = writeln!(s, "    RHS       OBJ       {}", fmt_num(-model.objective.constant));
    }
    for (i, c) in model.constraints.iter().enumerate() {
        let rhs = match classify(c) {
            RowClass::Equality | RowClass::Ge => c.lower,
            RowClass::Le | RowClass::Ranged => c.upper,
        };
        let _ = writeln!(s, "    RHS       c{:<7} {}", i + 1, fmt_num(rhs));
    }

    // ---- RANGES -----------------------------------------------------------
    let ranged: Vec<(usize, f64)> = model
        .constraints
        .iter()
        .enumerate()
        .filter_map(|(i, c)| match classify(c) {
            RowClass::Ranged => Some((i + 1, c.upper - c.lower)),
            _ => None,
        })
        .collect();
    if !ranged.is_empty() {
        s.push_str("RANGES\n");
        for (row, range) in ranged {
            let _ = writeln!(s, "    RNG       c{:<7} {}", row, fmt_num(range));
        }
    }

    // ---- BOUNDS -----------------------------------------------------------
    let mut bounds_section = String::new();
    for (j, v) in model.variables.iter().enumerate() {
        let name = format!("x{}", j + 1);
        let (lo, up) = (v.bound.bound.lower, v.bound.bound.upper);
        let mut cards: Vec<(&str, Option<f64>)> = Vec::new();
        match v.bound.kind {
            VarType::Binary => cards.push(("BV", None)),
            VarType::SemiContinuous if lo > 0.0 && up == f64::INFINITY => {
                cards.push(("SC", Some(lo)));
            }
            _ => {
                if lo == up {
                    cards.push(("FX", Some(lo)));
                } else if lo == f64::NEG_INFINITY && up == f64::INFINITY {
                    cards.push(("FR", None));
                } else if lo == f64::NEG_INFINITY {
                    cards.push(("MI", None));
                    if up != f64::INFINITY {
                        cards.push(("UP", Some(up)));
                    }
                } else {
                    if lo != 0.0 {
                        cards.push(("LO", Some(lo)));
                    }
                    if up != f64::INFINITY {
                        cards.push(("UP", Some(up)));
                    }
                }
            }
        }
        for (card, value) in cards {
            match value {
                Some(v) => {
                    let _ =
                        writeln!(bounds_section, " {card:<2} BND       {name:<9} {}", fmt_num(v));
                }
                None => {
                    let _ = writeln!(bounds_section, " {card:<2} BND       {name}");
                }
            }
        }
    }
    if !bounds_section.is_empty() {
        s.push_str("BOUNDS\n");
        s.push_str(&bounds_section);
    }

    s.push_str("ENDATA\n");
    s
}

// ---------------------------------------------------------------------------
// MPS reader
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum MpsRowKind {
    Objective,
    Free,
    Le,
    Ge,
    Equality,
}

struct RawRow {
    kind: MpsRowKind,
    coeffs: HashMap<usize, f64>,
    rhs: Option<f64>,
    range: Option<f64>,
}

impl RawRow {
    fn new(kind: MpsRowKind) -> Self {
        Self { kind, coeffs: HashMap::new(), rhs: None, range: None }
    }
}

/// Parse one MPS number, mapping ±1e30 sentinels to infinities.
fn parse_mps_number(token: &str) -> Option<f64> {
    let v: f64 = token.parse().ok()?;
    Some(if v >= INF_SENTINEL {
        f64::INFINITY
    } else if v <= -INF_SENTINEL {
        f64::NEG_INFINITY
    } else {
        v
    })
}

/// Parse free-format MPS text into a canonical [`Model`].
///
/// See the [module docs](self) for the supported feature set and conventions.
pub fn read_mps(input: &str) -> Result<Model, OptError> {
    let mut rows: Vec<RawRow> = Vec::new();
    let mut row_index: HashMap<String, usize> = HashMap::new();
    let mut obj_row: Option<usize> = None;

    let mut col_names: Vec<String> = Vec::new();
    let mut col_index: HashMap<String, usize> = HashMap::new();
    let mut var_bounds: Vec<VarBound> = Vec::new();

    let mut model_name: Option<String> = None;
    let mut maximize = false;
    let mut pending_objsense = false;
    let mut in_int_block = false;
    let mut section = "";

    for raw_line in input.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('*') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();

        // Section headers start in column 1 (no leading whitespace); every
        // indented line is a data record for the active section. This keeps
        // data whose first field repeats a keyword (e.g. an RHS vector named
        // `RHS`) from being mistaken for a header.
        let indented = raw_line.starts_with(' ') || raw_line.starts_with('\t');
        if !indented {
            match tokens[0] {
                "NAME" => {
                    section = "NAME";
                    if tokens.len() > 1 {
                        model_name = Some(tokens[1..].join("_"));
                    }
                    continue;
                }
                "OBJSENSE" => {
                    section = "OBJSENSE";
                    if tokens.len() > 1 {
                        maximize =
                            matches!(tokens[1].to_ascii_uppercase().as_str(), "MAX" | "MAXIMIZE");
                    } else {
                        pending_objsense = true;
                    }
                    continue;
                }
                "ROWS" => {
                    section = "ROWS";
                    continue;
                }
                "COLUMNS" => {
                    section = "COLUMNS";
                    continue;
                }
                "RHS" => {
                    section = "RHS";
                    continue;
                }
                "RANGES" => {
                    section = "RANGES";
                    continue;
                }
                "BOUNDS" => {
                    section = "BOUNDS";
                    continue;
                }
                "ENDATA" => break,
                _ => {}
            }
        }

        if pending_objsense {
            maximize = matches!(tokens[0].to_ascii_uppercase().as_str(), "MAX" | "MAXIMIZE");
            pending_objsense = false;
            continue;
        }

        match section {
            // Indented continuation lines for these headers carry no data we
            // use (the OBJSENSE value line was consumed above).
            "NAME" | "OBJSENSE" => continue,
            "ROWS" => {
                if tokens.len() < 2 {
                    return err(format!("malformed ROWS entry: `{line}`"));
                }
                let kind = match tokens[0].to_ascii_uppercase().as_str() {
                    "N" => {
                        if obj_row.is_none() {
                            obj_row = Some(rows.len());
                            MpsRowKind::Objective
                        } else {
                            MpsRowKind::Free
                        }
                    }
                    "L" => MpsRowKind::Le,
                    "G" => MpsRowKind::Ge,
                    "E" => MpsRowKind::Equality,
                    other => return err(format!("unknown ROWS type `{other}`")),
                };
                let name = tokens[1].to_string();
                if row_index.contains_key(&name) {
                    return err(format!("duplicate row name `{name}`"));
                }
                row_index.insert(name, rows.len());
                rows.push(RawRow::new(kind));
            }
            "COLUMNS" => {
                // Integer marker lines carry a quoted 'MARKER' tag.
                if tokens.iter().any(|t| t.trim_matches('\'').eq_ignore_ascii_case("MARKER")) {
                    if tokens.iter().any(|t| t.trim_matches('\'').eq_ignore_ascii_case("INTORG")) {
                        in_int_block = true;
                    } else if tokens
                        .iter()
                        .any(|t| t.trim_matches('\'').eq_ignore_ascii_case("INTEND"))
                    {
                        in_int_block = false;
                    }
                    continue;
                }
                if tokens.len() < 3 || tokens.len() % 2 == 0 {
                    return err(format!("malformed COLUMNS entry: `{line}`"));
                }
                let col_name = tokens[0];
                let j = match col_index.get(col_name) {
                    Some(&j) => j,
                    None => {
                        let j = col_names.len();
                        col_names.push(col_name.to_string());
                        col_index.insert(col_name.to_string(), j);
                        var_bounds.push(if in_int_block {
                            VarBound::integer(0.0, f64::INFINITY)
                        } else {
                            VarBound::continuous(0.0, f64::INFINITY)
                        });
                        j
                    }
                };
                for pair in tokens[1..].chunks(2) {
                    let row_name = pair[0];
                    let value = parse_mps_number(pair[1]).ok_or_else(|| {
                        OptError::invalid_model(format!("bad number `{}`", pair[1]))
                    })?;
                    let &ri = row_index.get(row_name).ok_or_else(|| {
                        OptError::invalid_model(format!(
                            "COLUMNS references unknown row `{row_name}`"
                        ))
                    })?;
                    *rows[ri].coeffs.entry(j).or_insert(0.0) += value;
                }
            }
            "RHS" | "RANGES" => {
                if tokens.len() < 3 || tokens.len() % 2 == 0 {
                    return err(format!("malformed {section} entry: `{line}`"));
                }
                for pair in tokens[1..].chunks(2) {
                    let row_name = pair[0];
                    let value = parse_mps_number(pair[1]).ok_or_else(|| {
                        OptError::invalid_model(format!("bad number `{}`", pair[1]))
                    })?;
                    let &ri = row_index.get(row_name).ok_or_else(|| {
                        OptError::invalid_model(format!(
                            "{section} references unknown row `{row_name}`"
                        ))
                    })?;
                    if section == "RHS" {
                        if Some(ri) == obj_row {
                            // Objective-row RHS is the negated constant.
                            rows[ri].rhs = Some(-value);
                        } else {
                            rows[ri].rhs = Some(value);
                        }
                    } else {
                        rows[ri].range = Some(value);
                    }
                }
            }
            "BOUNDS" => {
                apply_bound_card(&tokens, &mut col_index, &mut var_bounds, line)?;
            }
            other => {
                return err(format!("data outside any known section near `{other}`"));
            }
        }
    }

    finish_mps_model(rows, obj_row, col_names.len(), var_bounds, model_name, maximize)
}

fn apply_bound_card(
    tokens: &[&str],
    col_index: &mut HashMap<String, usize>,
    var_bounds: &mut [VarBound],
    line: &str,
) -> Result<(), OptError> {
    let card = tokens[0].to_ascii_uppercase();
    let valued = matches!(card.as_str(), "UP" | "LO" | "FX" | "LI" | "UI" | "SC");
    if tokens.len() < 3 || (valued && tokens.len() < 4) {
        return err(format!("malformed BOUNDS entry: `{line}`"));
    }
    let col_name = tokens[2];
    let &j = col_index.get(col_name).ok_or_else(|| {
        OptError::invalid_model(format!("BOUNDS references unknown column `{col_name}`"))
    })?;
    let value = if valued {
        Some(parse_mps_number(tokens[3]).ok_or_else(|| {
            OptError::invalid_model(format!("bad number `{}` in BOUNDS", tokens[3]))
        })?)
    } else {
        None
    };

    let b = &mut var_bounds[j];
    match card.as_str() {
        "UP" => {
            let up = value.unwrap_or(f64::INFINITY);
            // Classic convention: a negative UP on an integer column with the
            // default lower bound of 0 moves the lower bound to -inf.
            if b.kind != VarType::Continuous && up < 0.0 && b.bound.lower == 0.0 {
                b.bound.lower = f64::NEG_INFINITY;
            }
            b.bound.upper = up;
        }
        "LO" => b.bound.lower = value.unwrap_or(f64::NEG_INFINITY),
        "FX" => {
            let v = value.unwrap_or(0.0);
            b.bound.lower = v;
            b.bound.upper = v;
        }
        "FR" => b.bound = Bound::free(),
        "MI" => b.bound.lower = f64::NEG_INFINITY,
        "PL" => b.bound.upper = f64::INFINITY,
        "BV" => {
            b.kind = VarType::Binary;
            b.bound = Bound::boxed(0.0, 1.0);
        }
        "LI" => {
            b.kind = VarType::Integer;
            b.bound.lower = value.unwrap_or(f64::NEG_INFINITY);
        }
        "UI" => {
            b.kind = VarType::Integer;
            b.bound.upper = value.unwrap_or(f64::INFINITY);
        }
        "SC" => {
            b.kind = VarType::SemiContinuous;
            b.bound.lower = value.unwrap_or(0.0);
            if b.bound.upper == 0.0 {
                b.bound.upper = f64::INFINITY;
            }
        }
        other => return err(format!("unsupported BOUNDS card `{other}`")),
    }
    Ok(())
}

fn finish_mps_model(
    rows: Vec<RawRow>,
    obj_row: Option<usize>,
    num_vars: usize,
    var_bounds: Vec<VarBound>,
    model_name: Option<String>,
    maximize: bool,
) -> Result<Model, OptError> {
    let mut model = Model::new(num_vars);
    model.name = model_name;
    for (j, b) in var_bounds.iter().enumerate() {
        model.variables[j] = tpt_opt_core::model::Variable::new(j, *b);
    }

    // Objective from the (first) N row.
    let mut indices: Vec<usize> = Vec::new();
    let mut coeffs: Vec<f64> = Vec::new();
    let mut constant = 0.0;
    if let Some(ri) = obj_row {
        let obj = &rows[ri];
        let mut entries: Vec<(usize, f64)> = obj.coeffs.iter().map(|(&j, &c)| (j, c)).collect();
        entries.sort_unstable_by_key(|&(j, _)| j);
        for (j, c) in entries {
            indices.push(j);
            coeffs.push(c);
        }
        constant = obj.rhs.unwrap_or(0.0);
    }
    model.objective = Objective {
        sense: if maximize { Sense::Maximize } else { Sense::Minimize },
        indices,
        coeffs,
        constant,
    };

    // Constraints from every non-objective, non-free row, in file order.
    for (ri, row) in rows.iter().enumerate() {
        if Some(ri) == obj_row || row.kind == MpsRowKind::Free {
            continue;
        }
        let rhs = row.rhs.unwrap_or(0.0);
        let (lower, upper) = match row.kind {
            MpsRowKind::Le => match row.range {
                Some(r) => (rhs - r.abs(), rhs),
                None => (f64::NEG_INFINITY, rhs),
            },
            MpsRowKind::Ge => match row.range {
                Some(r) => (rhs, rhs + r.abs()),
                None => (rhs, f64::INFINITY),
            },
            MpsRowKind::Equality => match row.range {
                Some(r) if r > 0.0 => (rhs, rhs + r),
                Some(r) if r < 0.0 => (rhs + r, rhs),
                _ => (rhs, rhs),
            },
            MpsRowKind::Objective | MpsRowKind::Free => continue,
        };
        let mut entries: Vec<(usize, f64)> = row.coeffs.iter().map(|(&j, &c)| (j, c)).collect();
        entries.sort_unstable_by_key(|&(j, _)| j);
        let (indices, coeffs): (Vec<usize>, Vec<f64>) = entries.into_iter().unzip();
        model.add_constraint(Constraint { indices, coeffs, lower, upper, is_custom: false });
    }

    model.validate()?;
    Ok(model)
}

// ---------------------------------------------------------------------------
// LP writer
// ---------------------------------------------------------------------------

/// Render one linear expression as LP-format terms (no leading/trailing space).
fn lp_terms(indices: &[usize], coeffs: &[f64]) -> String {
    let mut s = String::new();
    let mut first = true;
    for (&j, &c) in indices.iter().zip(coeffs) {
        if c == 0.0 {
            continue;
        }
        let mag = c.abs();
        let coeff_str = if mag == 1.0 { String::new() } else { format!("{} ", fmt_num(mag)) };
        if first {
            if c < 0.0 {
                s.push('-');
            }
            first = false;
        } else if c < 0.0 {
            s.push_str(" - ");
        } else {
            s.push_str(" + ");
        }
        let _ = write!(s, "{coeff_str}x{}", j + 1);
    }
    if first {
        s.push('0'); // empty expression placeholder
    }
    s
}

/// Serialise `model` as CPLEX-style LP text.
///
/// The objective constant cannot be represented in LP format and is dropped;
/// use [`write_mps`] when the constant must survive. See the
/// [module docs](self) for other fidelity notes.
pub fn write_lp(model: &Model) -> String {
    let mut s = String::new();
    s.push_str("\\ Exported by tpt-opt-milp\n");
    s.push_str(if model.objective.sense == Sense::Maximize { "Maximize\n" } else { "Minimize\n" });
    let _ = writeln!(s, " obj: {}", lp_terms(&model.objective.indices, &model.objective.coeffs));

    s.push_str("Subject To\n");
    for (i, c) in model.constraints.iter().enumerate() {
        let terms = lp_terms(&c.indices, &c.coeffs);
        if c.lower == c.upper {
            let _ = writeln!(s, " c{}: {} = {}", i + 1, terms, fmt_num(c.lower));
        } else if c.lower == f64::NEG_INFINITY {
            let _ = writeln!(s, " c{}: {} <= {}", i + 1, terms, fmt_num(c.upper));
        } else if c.upper == f64::INFINITY {
            let _ = writeln!(s, " c{}: {} >= {}", i + 1, terms, fmt_num(c.lower));
        } else {
            let _ = writeln!(
                s,
                " c{}: {} <= {} <= {}",
                i + 1,
                fmt_num(c.lower),
                terms,
                fmt_num(c.upper)
            );
        }
    }

    let mut bounds_lines = String::new();
    let mut general_names: Vec<String> = Vec::new();
    let mut binary_names: Vec<String> = Vec::new();
    for (j, v) in model.variables.iter().enumerate() {
        let name = format!("x{}", j + 1);
        match v.bound.kind {
            VarType::Binary => binary_names.push(name),
            VarType::Integer => general_names.push(name),
            VarType::SemiContinuous | VarType::Continuous => {
                // No LP-format representation for semi-continuous: degrade to
                // plain continuous bounds.
                push_continuous_bound(&mut bounds_lines, &name, v.bound.bound);
            }
        }
    }

    if !bounds_lines.is_empty() {
        s.push_str("Bounds\n");
        s.push_str(&bounds_lines);
    }
    if !general_names.is_empty() {
        s.push_str("General\n ");
        s.push_str(&general_names.join(" "));
        s.push('\n');
    }
    if !binary_names.is_empty() {
        s.push_str("Binary\n ");
        s.push_str(&binary_names.join(" "));
        s.push('\n');
    }
    s.push_str("End\n");
    s
}

fn push_continuous_bound(out: &mut String, name: &str, b: Bound) {
    let default = b.lower == 0.0 && b.upper == f64::INFINITY;
    if default {
        return;
    }
    if b.lower == b.upper {
        let _ = writeln!(out, " {name} = {}", fmt_num(b.lower));
    } else if b.lower == f64::NEG_INFINITY && b.upper == f64::INFINITY {
        let _ = writeln!(out, " {name} free");
    } else if b.lower == f64::NEG_INFINITY {
        let _ = writeln!(out, " {name} <= {}", fmt_num(b.upper));
    } else if b.upper == f64::INFINITY {
        let _ = writeln!(out, " {name} >= {}", fmt_num(b.lower));
    } else {
        let _ = writeln!(out, " {} <= {name} <= {}", fmt_num(b.lower), fmt_num(b.upper));
    }
}

// ---------------------------------------------------------------------------
// LP reader
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum LpMode {
    Header,
    Objective,
    Constraints,
    Bounds,
    General,
    Binary,
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Name(String),
    Plus,
    Minus,
    Colon,
    Le,
    Ge,
    Eq,
}

fn tokenize(line: &str) -> Result<Vec<Tok>, OptError> {
    let chars: Vec<char> = line.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '+' => {
                toks.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                toks.push(Tok::Minus);
                i += 1;
            }
            ':' => {
                toks.push(Tok::Colon);
                i += 1;
            }
            '<' => {
                toks.push(Tok::Le); // '<' and '<=' both mean <=
                i += 1;
                if i < chars.len() && chars[i] == '=' {
                    i += 1;
                }
            }
            '>' => {
                toks.push(Tok::Ge); // '>' and '>=' both mean >=
                i += 1;
                if i < chars.len() && chars[i] == '=' {
                    i += 1;
                }
            }
            '=' => {
                // '=', '==', '=<' and '=>' variants.
                if i + 1 < chars.len() && chars[i + 1] == '<' {
                    toks.push(Tok::Le);
                    i += 2;
                } else if i + 1 < chars.len() && chars[i + 1] == '>' {
                    toks.push(Tok::Ge);
                    i += 2;
                } else {
                    toks.push(Tok::Eq);
                    i += 1;
                    if i < chars.len() && chars[i] == '=' {
                        i += 1;
                    }
                }
            }
            _ => {
                let start = i;
                let is_num_start = c.is_ascii_digit()
                    || (c == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit());
                if is_num_start {
                    i = scan_number(&chars, i)?;
                    let text: String = chars[start..i].iter().collect();
                    let v: f64 = text.parse().map_err(|_| {
                        OptError::invalid_model(format!("LP format: bad number `{text}`"))
                    })?;
                    toks.push(Tok::Num(v));
                    continue;
                }
                // Name token.
                while i < chars.len()
                    && (chars[i].is_alphanumeric()
                        || matches!(chars[i], '_' | '.' | '$' | '#' | '!' | '?' | '@'))
                {
                    i += 1;
                }
                if i == start {
                    return err(format!("LP format: unexpected character `{c}`"));
                }
                let text: String = chars[start..i].iter().collect();
                toks.push(Tok::Name(text));
            }
        }
    }
    Ok(toks)
}

/// Scan a numeric literal starting at `start`, handling fractions and
/// scientific exponents (including the exponent's sign). Returns the index
/// just past the number.
fn scan_number(chars: &[char], start: usize) -> Result<usize, OptError> {
    let mut i = start;
    while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
        i += 1;
    }
    if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
        i += 1;
        if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
            i += 1;
        }
        let digits_start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if i == digits_start {
            return err("LP format: exponent missing digits");
        }
    }
    Ok(i)
}

/// A parsed linear expression over variable names.
struct Expr {
    terms: Vec<(f64, String)>,
}

impl Expr {
    fn empty() -> Self {
        Self { terms: Vec::new() }
    }
}

struct LpBuilder {
    names: Vec<String>,
    index: HashMap<String, usize>,
    bounds: Vec<VarBound>,
    sense: Sense,
    obj: Expr,
    /// Completed constraints: (optional name, expr, lower, upper).
    constraints: Vec<(Option<String>, Expr, f64, f64)>,
}

impl LpBuilder {
    fn new() -> Self {
        Self {
            names: Vec::new(),
            index: HashMap::new(),
            bounds: Vec::new(),
            sense: Sense::Minimize,
            obj: Expr::empty(),
            constraints: Vec::new(),
        }
    }

    fn get_or_create(&mut self, name: &str) -> usize {
        if let Some(&j) = self.index.get(name) {
            return j;
        }
        let j = self.names.len();
        self.names.push(name.to_string());
        self.index.insert(name.to_string(), j);
        self.bounds.push(VarBound::continuous(0.0, f64::INFINITY));
        j
    }

    fn set_kind(&mut self, name: &str, kind: VarType) {
        let j = self.get_or_create(name);
        let lo = self.bounds[j].bound.lower;
        let up = self.bounds[j].bound.upper;
        let (lo, up) = if kind == VarType::Binary { (0.0, 1.0) } else { (lo, up) };
        self.bounds[j] = VarBound { kind, bound: Bound::boxed(lo, up) };
    }

    fn build(self) -> Result<Model, OptError> {
        let n = self.names.len();
        let mut model = Model::new(n);
        for (j, b) in self.bounds.iter().enumerate() {
            model.variables[j] = tpt_opt_core::model::Variable::new(j, *b);
        }
        let mut obj_indices = Vec::new();
        let mut obj_coeffs = Vec::new();
        for (c, term) in &self.obj.terms {
            let j = self.index.get(term).copied().ok_or_else(|| {
                OptError::invalid_model(format!("objective references unknown variable `{term}`"))
            })?;
            obj_indices.push(j);
            obj_coeffs.push(*c);
        }
        model.objective = Objective {
            sense: self.sense,
            indices: obj_indices,
            coeffs: obj_coeffs,
            constant: 0.0,
        };
        for (_, expr, lower, upper) in &self.constraints {
            let mut indices = Vec::new();
            let mut coeffs = Vec::new();
            for (c, term) in &expr.terms {
                let j = self.index.get(term).copied().ok_or_else(|| {
                    OptError::invalid_model(format!(
                        "constraint references unknown variable `{term}`"
                    ))
                })?;
                indices.push(j);
                coeffs.push(*c);
            }
            model.add_constraint(Constraint {
                indices,
                coeffs,
                lower: *lower,
                upper: *upper,
                is_custom: false,
            });
        }
        model.validate()?;
        Ok(model)
    }
}

/// Parse CPLEX-style LP text into a canonical [`Model`].
///
/// See the [module docs](self) for the supported grammar subset.
pub fn read_lp(input: &str) -> Result<Model, OptError> {
    let mut b = LpBuilder::new();
    let mut mode = LpMode::Header;
    // Pending constraint carried across wrapped lines: (name, expr).
    let mut pending: (Option<String>, Expr) = (None, Expr::empty());

    for raw_line in input.lines() {
        // Strip '\' comments, then trim.
        let line = raw_line.split('\\').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let toks = tokenize(line)?;
        if toks.is_empty() {
            continue;
        }

        // Section keyword detection on the first token. A keyword directly
        // followed by ':' is a row/objective name prefix, not a section.
        let kw = match &toks[0] {
            Tok::Name(n) => n.to_ascii_lowercase(),
            _ => String::new(),
        };
        let followed_by_colon = matches!(toks.get(1), Some(Tok::Colon));
        if kw == "end" && !followed_by_colon {
            break;
        }
        let keyword_mode = if followed_by_colon {
            None
        } else {
            match kw.as_str() {
                "minimize" | "minimise" | "min" => Some((LpMode::Objective, Sense::Minimize)),
                "maximize" | "maximise" | "max" => Some((LpMode::Objective, Sense::Maximize)),
                "subject" | "such" | "st" | "st." | "s.t." => Some((LpMode::Constraints, b.sense)),
                "bounds" | "bound" => Some((LpMode::Bounds, b.sense)),
                "general" | "generals" | "gen" | "integer" | "integers" | "int" => {
                    Some((LpMode::General, b.sense))
                }
                "binary" | "binaries" | "bin" => Some((LpMode::Binary, b.sense)),
                _ => None,
            }
        };
        if let Some((m, sense)) = keyword_mode {
            // "subject to" / "such that" consume the second word.
            let mut skip = 1usize;
            if kw == "subject" || kw == "such" {
                match toks.get(1) {
                    Some(Tok::Name(n))
                        if matches!(n.to_ascii_lowercase().as_str(), "to" | "that") =>
                    {
                        skip = 2;
                    }
                    _ => return err("LP format: malformed section header"),
                }
            }
            if m == LpMode::Objective {
                b.sense = sense;
            }
            mode = m;
            parse_rest(&mut b, &mut pending, &toks[skip..], mode)?;
            continue;
        }

        parse_rest(&mut b, &mut pending, &toks, mode)?;
    }

    if !pending.1.terms.is_empty() {
        return err("LP format: constraint ends without a relational operator");
    }
    if mode == LpMode::Header {
        return err("LP format: no Minimize/Maximize header found");
    }
    b.build()
}

/// Parse one line's remaining tokens according to the active section mode,
/// carrying unfinished constraints across lines.
fn parse_rest(
    b: &mut LpBuilder,
    pending: &mut (Option<String>, Expr),
    toks: &[Tok],
    mode: LpMode,
) -> Result<(), OptError> {
    // A section-header line arrives here with its keyword already stripped;
    // nothing further to parse.
    if toks.is_empty() {
        return Ok(());
    }
    match mode {
        LpMode::Header => Ok(()),
        LpMode::Objective => {
            let mut rest = toks;
            // Optional "objname:" prefix.
            if rest.len() >= 2 && matches!(&rest[1], Tok::Colon) {
                if let Tok::Name(_) = &rest[0] {
                    rest = &rest[2..];
                }
            }
            append_terms(&mut b.obj, rest, true)
        }
        LpMode::Constraints => {
            let mut rest = toks;
            // Optional "rowname:" prefix starts a fresh constraint.
            if rest.len() >= 2 && matches!(&rest[1], Tok::Colon) {
                if let Tok::Name(n) = &rest[0] {
                    if !pending.1.terms.is_empty() {
                        return err("LP format: previous constraint never completed");
                    }
                    pending.0 = Some(n.clone());
                    rest = &rest[2..];
                }
            }
            parse_constraint_tail(pending, rest).map(|finished| {
                if let Some((name, expr, lo, hi)) = finished {
                    // Register every referenced name so variable indices
                    // follow first-reference (file) order even for
                    // variables that appear only in constraints.
                    for (_, term) in &expr.terms {
                        b.get_or_create(term);
                    }
                    b.constraints.push((name, expr, lo, hi));
                }
            })
        }
        LpMode::Bounds => parse_bounds_line(b, toks),
        LpMode::General => {
            for t in toks {
                if let Tok::Name(n) = t {
                    b.set_kind(n, VarType::Integer);
                } else {
                    return err("LP format: General section expects variable names");
                }
            }
            Ok(())
        }
        LpMode::Binary => {
            for t in toks {
                if let Tok::Name(n) = t {
                    b.set_kind(n, VarType::Binary);
                } else {
                    return err("LP format: Binary section expects variable names");
                }
            }
            Ok(())
        }
    }
}

/// Append `[sign] [coeff] name` terms to `expr`. When `objective` is set, a
/// trailing bare number is rejected (no objective constants in LP format).
fn append_terms(expr: &mut Expr, toks: &[Tok], objective: bool) -> Result<(), OptError> {
    let mut i = 0;
    while i < toks.len() {
        let mut sign = 1.0;
        if matches!(toks[i], Tok::Plus) {
            i += 1;
        } else if matches!(toks[i], Tok::Minus) {
            sign = -1.0;
            i += 1;
        }
        let mut coeff = 1.0;
        let mut saw_coeff = false;
        if i < toks.len() {
            if let Tok::Num(v) = toks[i] {
                coeff = v;
                saw_coeff = true;
                i += 1;
            }
        }
        if i < toks.len() {
            if let Tok::Name(n) = &toks[i] {
                expr.terms.push((sign * coeff, n.clone()));
                i += 1;
                continue;
            }
        }
        if saw_coeff {
            let msg = if objective {
                "LP format: objective constants are not supported"
            } else {
                "LP format: coefficient without a variable name"
            };
            return err(msg);
        }
        // Anything else (a relational operator or bound number) belongs to the caller.
        break;
    }
    Ok(())
}

/// A completed constraint: optional row name, expression, lower/upper.
type CompletedConstraint = (Option<String>, Expr, f64, f64);

/// Continue parsing a (possibly wrapped) constraint fragment. Returns
/// `Some(completed_constraint)` when the fragment terminated with a
/// relational operator, or `None` when more terms follow on the next line.
fn parse_constraint_tail(
    pending: &mut (Option<String>, Expr),
    toks: &[Tok],
) -> Result<Option<CompletedConstraint>, OptError> {
    // Leading `lo <=` marks a double-bounded row: peel it off first.
    let mut rest = toks;
    let mut forced_lower: Option<f64> = None;
    if matches!(rest.first(), Some(Tok::Num(_))) && matches!(rest.get(1), Some(Tok::Le)) {
        forced_lower = match rest[0] {
            Tok::Num(v) => Some(v),
            _ => unreachable!("checked above"),
        };
        rest = &rest[2..];
    }

    // Find where the expression terms end and a relational part begins.
    let mut split = rest.len();
    for (k, t) in rest.iter().enumerate() {
        if matches!(t, Tok::Le | Tok::Ge | Tok::Eq) {
            split = k;
            break;
        }
    }
    append_terms(&mut pending.1, &rest[..split], false)?;

    if split == rest.len() {
        if forced_lower.is_some() {
            return err("LP format: double-bounded row missing its upper side");
        }
        return Ok(None); // wrapped line: more terms on the next line
    }

    // Relational part.
    let rel = &rest[split];
    let after = &rest[split + 1..];
    let completed = match rel {
        Tok::Eq => {
            let rhs = expect_number(after, "equality RHS")?;
            Some((rhs, rhs))
        }
        Tok::Le => {
            let hi = expect_number(after, "<= RHS")?;
            Some((forced_lower.unwrap_or(f64::NEG_INFINITY), hi))
        }
        Tok::Ge => {
            if forced_lower.is_some() {
                return err("LP format: unexpected lower bound before '>=' row");
            }
            let lo = expect_number(after, ">= RHS")?;
            Some((lo, f64::INFINITY))
        }
        _ => unreachable!("relational token checked above"),
    };
    let expr = std::mem::replace(pending, (None, Expr::empty())).1;
    Ok(completed.map(|(lower, upper)| (pending.0.take(), expr, lower, upper)))
}

fn expect_number(toks: &[Tok], what: &str) -> Result<f64, OptError> {
    match toks.first() {
        Some(Tok::Num(v)) => Ok(*v),
        Some(Tok::Plus) => match toks.get(1) {
            Some(Tok::Name(n)) if is_inf_name(n) => Ok(f64::INFINITY),
            _ => err(format!("LP format: expected number for {what}")),
        },
        Some(Tok::Minus) => match toks.get(1) {
            Some(Tok::Name(n)) if is_inf_name(n) => Ok(f64::NEG_INFINITY),
            _ => err(format!("LP format: expected number for {what}")),
        },
        Some(Tok::Name(n)) if is_inf_name(n) => Ok(f64::INFINITY),
        _ => err(format!("LP format: expected number for {what}")),
    }
}

fn is_inf_name(n: &str) -> bool {
    matches!(n.to_ascii_lowercase().as_str(), "inf" | "infinity")
}

/// Parse one Bounds-section line.
fn parse_bounds_line(b: &mut LpBuilder, toks: &[Tok]) -> Result<(), OptError> {
    // `free x y ...`
    if let Some(Tok::Name(kw)) = toks.first() {
        if kw.eq_ignore_ascii_case("free") {
            for t in &toks[1..] {
                if let Tok::Name(n) = t {
                    let j = b.get_or_create(n);
                    b.bounds[j] = VarBound { kind: VarType::Continuous, bound: Bound::free() };
                } else {
                    return err("LP format: `free` expects variable names");
                }
            }
            return Ok(());
        }
    }

    // Numeric-leading forms: `lo <= x [<= hi]` and reversed relations.
    if let Some(Tok::Num(_)) = toks.first() {
        if toks.len() >= 3 {
            let lo = match toks[0] {
                Tok::Num(v) => v,
                _ => unreachable!("checked above"),
            };
            match &toks[1] {
                Tok::Le => {
                    if let Tok::Name(n) = &toks[2] {
                        let j = b.get_or_create(n);
                        let up = if toks.len() >= 5 && matches!(&toks[3], Tok::Le) {
                            expect_number(&toks[4..], "upper bound")?
                        } else {
                            f64::INFINITY
                        };
                        b.bounds[j].bound = Bound::boxed(lo, up);
                        return Ok(());
                    }
                }
                Tok::Ge => {
                    if let Tok::Name(n) = &toks[2] {
                        let j = b.get_or_create(n);
                        b.bounds[j].bound.upper = lo; // `hi >= x` means x <= hi
                        return Ok(());
                    }
                }
                _ => {}
            }
        }
        return err("LP format: malformed Bounds entry");
    }

    // Name-leading forms.
    let name = match toks.first() {
        Some(Tok::Name(n)) => n.clone(),
        _ => return err("LP format: malformed Bounds entry"),
    };
    if toks.len() == 1 {
        return err(format!("LP format: bare variable `{name}` in Bounds"));
    }
    match &toks[1] {
        Tok::Name(kw) if kw.eq_ignore_ascii_case("free") => {
            let j = b.get_or_create(&name);
            b.bounds[j] = VarBound { kind: VarType::Continuous, bound: Bound::free() };
            Ok(())
        }
        Tok::Le => {
            let up = signed_number(&toks[2..])?;
            let j = b.get_or_create(&name);
            b.bounds[j].bound.upper = up;
            if up < 0.0 {
                b.bounds[j].bound.lower = f64::NEG_INFINITY;
            }
            Ok(())
        }
        Tok::Ge => {
            let lo = signed_number(&toks[2..])?;
            let j = b.get_or_create(&name);
            b.bounds[j].bound.lower = lo;
            Ok(())
        }
        Tok::Eq => {
            let v = signed_number(&toks[2..])?;
            let j = b.get_or_create(&name);
            b.bounds[j].bound = Bound::boxed(v, v);
            Ok(())
        }
        _ => err("LP format: malformed Bounds entry"),
    }
}

/// Parse a possibly-signed number or infinity from the front of `toks`.
fn signed_number(toks: &[Tok]) -> Result<f64, OptError> {
    match toks.first() {
        Some(Tok::Num(v)) => Ok(*v),
        Some(Tok::Plus) => match toks.get(1) {
            Some(Tok::Name(n)) if is_inf_name(n) => Ok(f64::INFINITY),
            _ => err("LP format: expected a number"),
        },
        Some(Tok::Minus) => match toks.get(1) {
            Some(Tok::Name(n)) if is_inf_name(n) => Ok(f64::NEG_INFINITY),
            Some(Tok::Num(v)) => Ok(-*v),
            _ => err("LP format: expected a number"),
        },
        Some(Tok::Name(n)) if is_inf_name(n) => Ok(f64::INFINITY),
        _ => err("LP format: expected a number"),
    }
}

// ---------------------------------------------------------------------------
// Convenience solver hook
// ---------------------------------------------------------------------------

/// Solve a freshly parsed [`Model`] with the default MILP configuration.
///
/// Provided so callers of [`read_mps`] / [`read_lp`] can go straight from
/// text to solution in one step.
pub fn solve_parsed(model: &Model) -> Result<tpt_opt_core::solver::Solution, OptError> {
    crate::MilpSolver::new().solve(model)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_opt_core::solver::SolverStatus;

    /// Classic knapsack: max 3x+4y+5z+6w s.t. 2x+3y+4z+5w <= 9, all binary.
    /// Optimum 12 (x + y + z, weight exactly 9).
    fn knapsack() -> Model {
        let mut m = Model::with_name(4, "KNAP");
        for j in 0..4 {
            m.variables[j] = tpt_opt_core::model::Variable::new(j, VarBound::binary());
        }
        m.set_objective(Objective::maximize(vec![0, 1, 2, 3], vec![3.0, 4.0, 5.0, 6.0]));
        m.add_constraint(Constraint::le(vec![0, 1, 2, 3], vec![2.0, 3.0, 4.0, 5.0], 9.0));
        m
    }

    fn opt_value(model: &Model) -> f64 {
        let sol = crate::MilpSolver::new().solve(model).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        sol.objective_value
    }

    // ----- MPS -------------------------------------------------------------

    #[test]
    fn mps_roundtrip_knapsack_preserves_optimum() {
        let original = knapsack();
        let text = write_mps(&original);
        let parsed = read_mps(&text).unwrap();
        assert_eq!(parsed.objective.sense, Sense::Maximize);
        assert_eq!(parsed.num_vars, 4);
        assert!(parsed.variables.iter().all(|v| v.kind == VarType::Binary));
        let a = opt_value(&original);
        let b = opt_value(&parsed);
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        assert!((a - 12.0).abs() < 1e-9, "expected knapsack optimum 12, got {a}");
    }

    #[test]
    fn mps_parse_handwritten_features() {
        let text = "\
* comment line
NAME          TESTPROB
OBJSENSE
    MAX
ROWS
 N  COST
 L  CAP
 G  MINX
 E  BAL
 N  FREEROW
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X         COST         3.0             CAP       2.0
    Y         COST         4.0             CAP       3.0
    MARKER                 'MARKER'                 'INTEND'
    Z         COST         1.0             CAP       4.0
    Z         MINX         1.0
    W         BAL          1.0             MINX      1.0
    W         FREEROW      9.9
RHS
    RHS       CAP          9.0             MINX      1.0
    RHS       BAL          2.0             COST      -5.0
RANGES
    RNG       BAL          3.0
BOUNDS
 UP BND       Z            2.5
 FR BND       W
 UI BND       Y            10
 UI BND       X            10
ENDATA
";
        let m = read_mps(text).unwrap();
        assert_eq!(m.num_vars, 4);
        assert_eq!(m.objective.sense, Sense::Maximize);
        // Integrality from markers; Z/W continuous.
        assert_eq!(m.variables[0].kind, VarType::Integer);
        assert_eq!(m.variables[1].kind, VarType::Integer);
        assert_eq!(m.variables[2].kind, VarType::Continuous);
        assert_eq!(m.variables[3].kind, VarType::Continuous);
        // Bound cards applied.
        assert_eq!(m.variables[2].bound.bound.upper, 2.5);
        assert_eq!(m.variables[3].bound.bound.lower, f64::NEG_INFINITY);
        assert_eq!(m.variables[3].bound.bound.upper, f64::INFINITY);
        assert_eq!(m.variables[1].bound.bound.upper, 10.0);
        // Rows: CAP <= 9, MINX >= 1, BAL ranged [2, 5]; free row dropped.
        assert_eq!(m.num_constraints(), 3);
        let bal = &m.constraints[2];
        assert_eq!(bal.lower, 2.0);
        assert_eq!(bal.upper, 5.0);
        // Objective constant: RHS -5 on COST => +5.
        assert_eq!(m.objective.constant, 5.0);
        // CAP row collects X, Y, Z.
        assert_eq!(m.constraints[0].indices.len(), 3);

        // Solve end-to-end: the true optimum is 18 at X=3, Y=1, Z=0 (W is
        // free within [2, 5] and carries no objective coefficient). This
        // instance previously triggered a search bug where node bounds
        // excluded the objective constant and the tree was pruned after one
        // node (returning 17); see the regression test in
        // `tests/repro_search_bug.rs`.
        let sol = crate::MilpSolver::new().solve(&m).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!((sol.objective_value - 18.0).abs() < 1e-6, "obj {}", sol.objective_value);
        assert!(
            (m.objective.eval(&sol.primal) - sol.objective_value).abs() < 1e-6,
            "objective inconsistent with primal {:?}",
            sol.primal
        );
    }

    #[test]
    fn mps_minimize_is_the_default_sense() {
        let text = "\
NAME          TINY
ROWS
 N  OBJ
 G  R1
COLUMNS
    X         OBJ          1.0             R1        1.0
RHS
    RHS       R1           4.0
ENDATA
";
        let m = read_mps(text).unwrap();
        assert_eq!(m.objective.sense, Sense::Minimize);
        let sol = crate::MilpSolver::new().solve(&m).unwrap();
        assert!((sol.objective_value - 4.0).abs() < 1e-9);
    }

    #[test]
    fn mps_rejects_unknown_row_reference() {
        let text = "\
ROWS
 N  OBJ
COLUMNS
    X         NOPE         1.0
ENDATA
";
        assert!(read_mps(text).is_err());
    }

    #[test]
    fn mps_rejects_bad_number() {
        let text = "\
ROWS
 N  OBJ
COLUMNS
    X         OBJ          abc
ENDATA
";
        assert!(read_mps(text).is_err());
    }

    #[test]
    fn mps_ranged_e_row_and_free_column() {
        let text = "\
ROWS
 N  OBJ
 E  DBL
COLUMNS
    X         OBJ          1.0             DBL       1.0
RHS
    RHS       DBL          2.0
RANGES
    RNG       DBL          -3.0
BOUNDS
 MI BND       X
 PL BND       X
ENDATA
";
        let m = read_mps(text).unwrap();
        // Negative range on an E row: [rhs + r, rhs] = [-1, 2].
        assert_eq!(m.constraints[0].lower, -1.0);
        assert_eq!(m.constraints[0].upper, 2.0);
        // MI + PL leaves X free; min x subject to -1 <= x <= 2 → -1.
        let sol = crate::MilpSolver::new().solve(&m).unwrap();
        assert!((sol.objective_value - (-1.0)).abs() < 1e-9);
    }

    // ----- LP --------------------------------------------------------------

    #[test]
    fn lp_roundtrip_knapsack_is_exact() {
        let original = knapsack();
        let text = write_lp(&original);
        let parsed = read_lp(&text).unwrap();
        assert_eq!(parsed.objective.sense, Sense::Maximize);
        assert_eq!(parsed.num_vars, 4);
        assert!(parsed.variables.iter().all(|v| v.kind == VarType::Binary));
        assert_eq!(parsed.num_constraints(), 1);
        let a = opt_value(&original);
        let b = opt_value(&parsed);
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }

    #[test]
    fn lp_parse_handwritten_features() {
        let text = "\\ small mixed test
Maximize
 obj: 3 x + 4 y + z
Subject To
 cap: 2 x + 3 y + 4 z <= 9
 lo: z >= 1
 dbl: 2 <= w + z <= 5
 alt: x + y =< 3
 wrap: 2 x
   + 2 y <= 8
Bounds
 z <= 2.5
 w free
General
 x y
End
";
        let m = read_lp(text).unwrap();
        assert_eq!(m.objective.sense, Sense::Maximize);
        assert_eq!(m.num_vars, 4);
        assert_eq!(m.variables[0].kind, VarType::Integer);
        assert_eq!(m.variables[1].kind, VarType::Integer);
        assert_eq!(m.variables[2].kind, VarType::Continuous);
        assert_eq!(m.variables[2].bound.bound.upper, 2.5);
        assert_eq!(m.variables[3].bound.bound.lower, f64::NEG_INFINITY);
        assert_eq!(m.num_constraints(), 5);
        // Double-bounded row captured both sides.
        assert_eq!(m.constraints[2].lower, 2.0);
        assert_eq!(m.constraints[2].upper, 5.0);
        // Wrapped constraint accumulated both fragments.
        assert_eq!(m.constraints[4].indices.len(), 2);

        // Solve end-to-end: with z >= 1 the cap forces 2x + 3y <= 5, so the
        // best integer point is x=1, y=1 with z=1 -> objective 8 (w is free
        // within [1, 4] and carries no objective coefficient).
        let sol = crate::MilpSolver::new().solve(&m).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!((sol.objective_value - 8.0).abs() < 1e-6, "obj {}", sol.objective_value);
    }

    #[test]
    fn lp_accepts_operator_variants_and_inf_bounds() {
        let text = "\
Minimize
 o: x + y
Subject To
 r1: x + y >= 2
 r2: x - y => 0
Bounds
 x >= -inf
 y <= +inf
End
";
        let m = read_lp(text).unwrap();
        assert_eq!(m.variables[0].bound.bound.lower, f64::NEG_INFINITY);
        assert_eq!(m.variables[1].bound.bound.upper, f64::INFINITY);
        let sol = crate::MilpSolver::new().solve(&m).unwrap();
        assert!((sol.objective_value - 2.0).abs() < 1e-9);
    }

    #[test]
    fn lp_fixed_bound_and_equality_row() {
        let text = "\
Minimize
 obj: x
Subject To
 e: x = 3
Bounds
 x = 3
End
";
        let m = read_lp(text).unwrap();
        assert_eq!(m.constraints[0].lower, 3.0);
        assert_eq!(m.constraints[0].upper, 3.0);
        let sol = crate::MilpSolver::new().solve(&m).unwrap();
        assert!((sol.objective_value - 3.0).abs() < 1e-9);
    }

    #[test]
    fn lp_rejects_objective_constant() {
        let text = "\
Minimize
 obj: 2 x + 5
Subject To
 r: x >= 1
End
";
        assert!(read_lp(text).is_err());
    }

    // ----- Cross-format ----------------------------------------------------

    #[test]
    fn cross_format_mps_to_lp_agrees() {
        let original = knapsack();
        let via_mps = read_mps(&write_mps(&original)).unwrap();
        let via_both = read_lp(&write_lp(&via_mps)).unwrap();
        let a = opt_value(&original);
        let b = opt_value(&via_both);
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }

    #[test]
    fn solve_parsed_helper_works() {
        let m = read_mps(
            "\
ROWS
 N  OBJ
 G  C
COLUMNS
    X         OBJ          1.0             C         1.0
RHS
    RHS       C            7.0
ENDATA
",
        )
        .unwrap();
        let sol = solve_parsed(&m).unwrap();
        assert!((sol.objective_value - 7.0).abs() < 1e-9);
    }
}
