//! C4.5 / C5.0 Rule-Set Extraction and Pruning.
//!
//! Transforms hierarchical decision tree branches into flat, modular rule sets:
//! 1. Rule Extraction: Each root-to-leaf path produces an initial rule (conjunction of splits).
//! 2. Path Simplification: Multiple tests on the same feature are combined into tight intervals.
//! 3. Rule Pruning: Preconditions are greedily pruned using Quinlan's pessimistic error
//!    rate upper bounds with confidence factor CF (default 0.25).
//! 4. Rule-Set Inference: Evaluates rules by highest-confidence vote, falling back to a default rule.

use std::fmt;

/// Relational operator in a rule condition.
#[derive(Clone, Debug, PartialEq)]
pub enum ConditionOp {
    LessOrEqual,
    GreaterThan,
    InSet(Vec<f64>),
    NotInSet(Vec<f64>),
    IsMissing,
}

impl fmt::Display for ConditionOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LessOrEqual => write!(f, "<="),
            Self::GreaterThan => write!(f, ">"),
            Self::InSet(vals) => write!(f, "in {:?}", vals),
            Self::NotInSet(vals) => write!(f, "not in {:?}", vals),
            Self::IsMissing => write!(f, "is missing"),
        }
    }
}

/// A single atomic predicate in a rule (e.g., `cibil_score <= 650.0`).
#[derive(Clone, Debug, PartialEq)]
pub struct RuleCondition {
    pub col: usize,
    pub col_name: String,
    pub op: ConditionOp,
    pub threshold: f64,
}

impl RuleCondition {
    /// Evaluate whether this condition is satisfied by a feature vector.
    pub fn matches(&self, row: &[Option<f64>]) -> bool {
        let val = row.get(self.col).copied().flatten();
        match (&self.op, val) {
            (ConditionOp::IsMissing, None) => true,
            (ConditionOp::IsMissing, Some(_)) => false,
            (_, None) => false,
            (ConditionOp::LessOrEqual, Some(v)) => v <= self.threshold,
            (ConditionOp::GreaterThan, Some(v)) => v > self.threshold,
            (ConditionOp::InSet(set), Some(v)) => set.iter().any(|&item| (item - v).abs() < 1e-9),
            (ConditionOp::NotInSet(set), Some(v)) => !set.iter().any(|&item| (item - v).abs() < 1e-9),
        }
    }
}

impl fmt::Display for RuleCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.op {
            ConditionOp::IsMissing => write!(f, "{} is missing", self.col_name),
            ConditionOp::InSet(vals) => write!(f, "{} in {:?}", self.col_name, vals),
            ConditionOp::NotInSet(vals) => write!(f, "{} not in {:?}", self.col_name, vals),
            ConditionOp::LessOrEqual => write!(f, "{} <= {:.4}", self.col_name, self.threshold),
            ConditionOp::GreaterThan => write!(f, "{} > {:.4}", self.col_name, self.threshold),
        }
    }
}

/// A single IF-THEN rule extracted from tree branches.
#[derive(Clone, Debug)]
pub struct Rule {
    pub id: usize,
    pub conditions: Vec<RuleCondition>,
    pub prediction: f64,
    pub cover: usize,
    pub errors: usize,
    pub confidence: f64,
    pub leaf_id: usize,
}

impl Rule {
    /// Test if all preconditions match a row.
    pub fn matches(&self, row: &[Option<f64>]) -> bool {
        self.conditions.iter().all(|c| c.matches(row))
    }

    /// Simplify conditions on the same column (tighten intervals, remove redundant bounds).
    pub fn simplify_conditions(&mut self) {
        let mut simplified: Vec<RuleCondition> = Vec::new();

        // Group conditions by column
        let mut cols: Vec<usize> = self.conditions.iter().map(|c| c.col).collect();
        cols.sort_unstable();
        cols.dedup();

        for col in cols {
            let col_conds: Vec<&RuleCondition> = self.conditions.iter().filter(|c| c.col == col).collect();
            if col_conds.len() == 1 {
                simplified.push(col_conds[0].clone());
                continue;
            }

            // Find tightest bounds for numeric conditions
            let mut max_gt: Option<f64> = None;
            let mut min_le: Option<f64> = None;
            let mut other_conds: Vec<RuleCondition> = Vec::new();

            for c in col_conds {
                match c.op {
                    ConditionOp::GreaterThan => {
                        max_gt = Some(max_gt.map_or(c.threshold, |v| v.max(c.threshold)));
                    }
                    ConditionOp::LessOrEqual => {
                        min_le = Some(min_le.map_or(c.threshold, |v| v.min(c.threshold)));
                    }
                    _ => other_conds.push(c.clone()),
                }
            }

            let name = self
                .conditions
                .iter()
                .find(|c| c.col == col)
                .map(|c| c.col_name.clone())
                .unwrap_or_else(|| format!("x{}", col));

            if let Some(gt) = max_gt {
                simplified.push(RuleCondition {
                    col,
                    col_name: name.clone(),
                    op: ConditionOp::GreaterThan,
                    threshold: gt,
                });
            }
            if let Some(le) = min_le {
                simplified.push(RuleCondition {
                    col,
                    col_name: name,
                    op: ConditionOp::LessOrEqual,
                    threshold: le,
                });
            }
            simplified.extend(other_conds);
        }

        self.conditions = simplified;
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Rule {} (cover: {}, errors: {}, confidence: {:.1}%):",
            self.id,
            self.cover,
            self.errors,
            self.confidence * 100.0
        )?;
        if self.conditions.is_empty() {
            writeln!(f, "  IF (always true)")?;
        } else {
            for (i, c) in self.conditions.iter().enumerate() {
                if i == 0 {
                    writeln!(f, "  IF {}", c)?;
                } else {
                    writeln!(f, "     AND {}", c)?;
                }
            }
        }
        write!(f, "  THEN prediction = {:.4}", self.prediction)
    }
}

/// An ordered collection of rules with fallback default behavior.
#[derive(Clone, Debug)]
pub struct RuleSet {
    pub rules: Vec<Rule>,
    pub default_prediction: f64,
    pub default_confidence: f64,
    pub col_names: Vec<String>,
}

impl RuleSet {
    /// Number of active rules.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Predict a single row. Evaluates matching rules by highest confidence;
    /// falls back to default prediction if no rule fires.
    pub fn predict_row(&self, row: &[Option<f64>]) -> f64 {
        let mut best_rule: Option<&Rule> = None;
        for rule in &self.rules {
            if rule.matches(row) {
                match best_rule {
                    None => best_rule = Some(rule),
                    Some(b) if rule.confidence > b.confidence => best_rule = Some(rule),
                    _ => {}
                }
            }
        }
        best_rule
            .map(|r| r.prediction)
            .unwrap_or(self.default_prediction)
    }

    /// Predict a batch of rows.
    pub fn predict(&self, x: &[Vec<Option<f64>>]) -> Vec<f64> {
        x.iter().map(|row| self.predict_row(row)).collect()
    }

    /// Format the rule set as a human-readable text report.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "RuleSet ({} rules, default prediction = {:.4}, default confidence = {:.1}%):\n\n",
            self.rules.len(),
            self.default_prediction,
            self.default_confidence * 100.0
        ));
        for rule in &self.rules {
            out.push_str(&format!("{}\n\n", rule));
        }
        out.push_str(&format!(
            "Default Rule:\n  THEN prediction = {:.4} (confidence: {:.1}%)\n",
            self.default_prediction,
            self.default_confidence * 100.0
        ));
        out
    }
}

impl fmt::Display for RuleSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

/// Compute Quinlan's pessimistic error upper bound using binomial normal approximation:
/// e = number of errors, n = total cases, z = standard normal deviate corresponding to CF.
pub fn pessimistic_error_bound(errors: usize, n: usize, z: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let e = errors as f64;
    let n_f = n as f64;
    let f = e / n_f;
    let z2 = z * z;

    let numerator = f + (z2 / (2.0 * n_f)) + z * ((f * (1.0 - f) / n_f) + (z2 / (4.0 * n_f * n_f))).sqrt();
    let denominator = 1.0 + (z2 / n_f);
    (numerator / denominator).clamp(0.0, 1.0)
}

/// Convert confidence factor CF (e.g. 0.25) to standard normal z quantile.
pub fn cf_to_z(cf: f64) -> f64 {
    // Tabulated two-tailed standard normal quantiles for typical Quinlan CF values
    if cf <= 0.01 {
        2.326
    } else if cf <= 0.05 {
        1.645
    } else if cf <= 0.10 {
        1.282
    } else if cf <= 0.25 {
        0.674 // Quinlan's classic default CF=25%
    } else if cf <= 0.50 {
        0.000
    } else {
        0.674
    }
}
