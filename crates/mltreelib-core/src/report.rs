//! Unified Model Report Model.
//!
//! Provides a standardized, structured report across all mltreelib estimators:
//! trees, solvers, anomaly detectors, clustering models, rule miners, and
//! component analysis.
//!
//! Each report consists of four required sections and one optional inference section:
//! 1. Objective: What the model optimizes and final loss.
//! 2. Learner: Mathematical family, representation, structure, and capabilities.
//! 3. Controller: Tuned knobs, budgets, seed, and missing value policy.
//! 4. Diagnostics: Convergence, fit time, quality metrics, and warnings.
//! 5. Inference (optional): Coefficient table with standard errors, t/z stats,
//!    p-values, and confidence intervals when mathematically valid.

use std::collections::BTreeMap;
use std::fmt;
use serde::{Deserialize, Serialize};

/// Objective optimization direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizationDirection {
    Minimize,
    Maximize,
    Reconstruct,
}

impl fmt::Display for OptimizationDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Minimize => write!(f, "minimize"),
            Self::Maximize => write!(f, "maximize"),
            Self::Reconstruct => write!(f, "reconstruct"),
        }
    }
}

/// Objective section of the model report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveSection {
    pub name: String,
    pub math_form: Option<String>,
    pub direction: OptimizationDirection,
    pub final_value: Option<f64>,
    pub components: Vec<(String, f64)>,
    pub units: Option<String>,
}

/// Learner section describing model family, structure, and capacity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearnerSection {
    pub family: String,
    pub kind: String,
    pub representation: String,
    pub structure: BTreeMap<String, String>,
    pub nominal_complexity: usize,
    pub effective_complexity: Option<f64>,
    pub capabilities: Vec<String>,
}

/// Controller section recording configuration, budgets, and settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControllerSection {
    pub kind: String,
    pub config: BTreeMap<String, String>,
    pub seed: Option<u64>,
    pub iteration_budget: Option<usize>,
    pub evaluation_budget: Option<usize>,
    pub missing_policy: String,
    pub fallback_reason: Option<String>,
}

/// Diagnostics section with convergence, quality metrics, and warnings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticsSection {
    pub converged: Option<bool>,
    pub stop_reason: Option<String>,
    pub fit_time_sec: Option<f64>,
    pub eval_count: Option<usize>,
    pub metrics: BTreeMap<String, f64>,
    pub warnings: Vec<String>,
}

/// Single coefficient row in the inference section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoefficientInference {
    pub name: String,
    pub estimate: f64,
    pub std_err: f64,
    pub stat_value: f64,
    pub p_value: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

/// Inference section when classical inference is mathematically valid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InferenceSection {
    pub valid: bool,
    pub unavailability_reason: Option<String>,
    pub coefficients: Vec<CoefficientInference>,
    pub df_residual: Option<usize>,
    pub r_squared: Option<f64>,
    pub adj_r_squared: Option<f64>,
    pub f_statistic: Option<f64>,
    pub f_p_value: Option<f64>,
}

/// Complete Unified Model Report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelReport {
    pub title: String,
    pub version: u32,
    pub objective: ObjectiveSection,
    pub learner: LearnerSection,
    pub controller: ControllerSection,
    pub diagnostics: DiagnosticsSection,
    pub inference: Option<InferenceSection>,
}

impl ModelReport {
    pub fn new(
        title: impl Into<String>,
        objective: ObjectiveSection,
        learner: LearnerSection,
        controller: ControllerSection,
        diagnostics: DiagnosticsSection,
        inference: Option<InferenceSection>,
    ) -> Self {
        Self {
            title: title.into(),
            version: 1,
            objective,
            learner,
            controller,
            diagnostics,
            inference,
        }
    }

    /// Render report as formatted R/statsmodels-style text.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        let bar = "=".repeat(78);
        let sub = "-".repeat(78);

        out.push_str(&bar);
        out.push('\n');
        out.push_str(&format!("Model Report: {}\n", self.title));
        out.push_str(&bar);
        out.push('\n');

        // 1. Objective
        out.push_str("1. OBJECTIVE\n");
        out.push_str(&sub);
        out.push('\n');
        out.push_str(&format!("  Target Goal:       {}\n", self.objective.name));
        if let Some(form) = &self.objective.math_form {
            out.push_str(&format!("  Form:              {}\n", form));
        }
        out.push_str(&format!("  Direction:         {}\n", self.objective.direction));
        if let Some(val) = self.objective.final_value {
            let u = self.objective.units.as_deref().unwrap_or("");
            out.push_str(format!("  Final Loss/Value:  {:.6} {}\n", val, u).trim_end());
            out.push('\n');
        }
        for (k, v) in &self.objective.components {
            out.push_str(&format!("    - {}: {:.6}\n", k, v));
        }
        out.push('\n');

        // 2. Learner
        out.push_str("2. LEARNER\n");
        out.push_str(&sub);
        out.push('\n');
        out.push_str(&format!("  Family:            {}\n", self.learner.family));
        out.push_str(&format!("  Method:            {}\n", self.learner.kind));
        out.push_str(&format!("  Representation:    {}\n", self.learner.representation));
        out.push_str(&format!("  Nominal Complexity: {}\n", self.learner.nominal_complexity));
        if let Some(eff) = self.learner.effective_complexity {
            out.push_str(&format!("  Effective Complexity: {:.4}\n", eff));
        }
        if !self.learner.structure.is_empty() {
            out.push_str("  Structure:\n");
            for (k, v) in &self.learner.structure {
                out.push_str(&format!("    - {}: {}\n", k, v));
            }
        }
        if !self.learner.capabilities.is_empty() {
            out.push_str(&format!("  Capabilities:      {}\n", self.learner.capabilities.join(", ")));
        }
        out.push('\n');

        // 3. Controller
        out.push_str("3. CONTROLLER\n");
        out.push_str(&sub);
        out.push('\n');
        out.push_str(&format!("  Selected Kind:     {}\n", self.controller.kind));
        if let Some(seed) = self.controller.seed {
            out.push_str(&format!("  Random Seed:       {}\n", seed));
        }
        out.push_str(&format!("  Missing Policy:    {}\n", self.controller.missing_policy));
        if let Some(b) = self.controller.iteration_budget {
            out.push_str(&format!("  Iteration Budget:  {}\n", b));
        }
        if let Some(b) = self.controller.evaluation_budget {
            out.push_str(&format!("  Evaluation Budget: {}\n", b));
        }
        if let Some(fb) = &self.controller.fallback_reason {
            out.push_str(&format!("  Fallback Reason:   {}\n", fb));
        }
        if !self.controller.config.is_empty() {
            out.push_str("  Knobs:\n");
            for (k, v) in &self.controller.config {
                out.push_str(&format!("    - {}: {}\n", k, v));
            }
        }
        out.push('\n');

        // 4. Diagnostics
        out.push_str("4. DIAGNOSTICS\n");
        out.push_str(&sub);
        out.push('\n');
        if let Some(conv) = self.diagnostics.converged {
            out.push_str(&format!("  Converged:         {}\n", if conv { "Yes" } else { "No" }));
        }
        if let Some(reason) = &self.diagnostics.stop_reason {
            out.push_str(&format!("  Stop Reason:       {}\n", reason));
        }
        if let Some(t) = self.diagnostics.fit_time_sec {
            out.push_str(&format!("  Fit Time:          {:.4}s\n", t));
        }
        if let Some(c) = self.diagnostics.eval_count {
            out.push_str(&format!("  Evaluations:       {}\n", c));
        }
        if !self.diagnostics.metrics.is_empty() {
            out.push_str("  Quality Metrics:\n");
            for (k, v) in &self.diagnostics.metrics {
                out.push_str(&format!("    - {:<20}: {:.6}\n", k, v));
            }
        }
        if !self.diagnostics.warnings.is_empty() {
            out.push_str("  Warnings:\n");
            for w in &self.diagnostics.warnings {
                out.push_str(&format!("    [WARNING] {}\n", w));
            }
        }
        out.push('\n');

        // 5. Inference
        if let Some(inf) = &self.inference {
            out.push_str("5. STATISTICAL INFERENCE\n");
            out.push_str(&sub);
            out.push('\n');
            if !inf.valid {
                let reason = inf.unavailability_reason.as_deref().unwrap_or("Inference not supported for this estimator family.");
                out.push_str(&format!("  Inference Unavailable: {}\n\n", reason));
            } else {
                if let (Some(r2), Some(ar2)) = (inf.r_squared, inf.adj_r_squared) {
                    out.push_str(&format!("  R-squared: {:.4} | Adj R-squared: {:.4}\n", r2, ar2));
                }
                if let (Some(f), Some(p)) = (inf.f_statistic, inf.f_p_value) {
                    out.push_str(&format!("  F-statistic: {:.4} (p = {:.4e})\n", f, p));
                }
                if !inf.coefficients.is_empty() {
                    out.push_str("\n  Coefficients:\n");
                    out.push_str(&format!(
                        "  {:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}\n",
                        "Term", "Estimate", "Std.Err", "Stat", "p-value", "CI Lower", "CI Upper"
                    ));
                    out.push_str(&format!("  {}\n", "-".repeat(74)));
                    for c in &inf.coefficients {
                        let p_str = if c.p_value < 1e-4 {
                            format!("{:.2e}", c.p_value)
                        } else {
                            format!("{:.4}", c.p_value)
                        };
                        out.push_str(&format!(
                            "  {:<20} {:>10.4} {:>10.4} {:>10.4} {:>10} {:>10.4} {:>10.4}\n",
                            c.name, c.estimate, c.std_err, c.stat_value, p_str, c.ci_lower, c.ci_upper
                        ));
                    }
                }
                out.push('\n');
            }
        }

        out.push_str(&bar);
        out.push('\n');
        out
    }

    /// Render report as formatted Markdown.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("# Model Report: {}\n\n", self.title));

        // 1. Objective
        out.push_str("## 1. Objective\n\n");
        out.push_str(&format!("- **Goal:** {}\n", self.objective.name));
        if let Some(form) = &self.objective.math_form {
            out.push_str(&format!("- **Mathematical Form:** `{}`\n", form));
        }
        out.push_str(&format!("- **Direction:** `{}`\n", self.objective.direction));
        if let Some(val) = self.objective.final_value {
            let u = self.objective.units.as_deref().unwrap_or("");
            out.push_str(&format!("- **Final Objective Value:** `{:.6}` {}\n", val, u));
        }
        for (k, v) in &self.objective.components {
            out.push_str(&format!("  - **{}**: `{:.6}`\n", k, v));
        }
        out.push('\n');

        // 2. Learner
        out.push_str("## 2. Learner\n\n");
        out.push_str(&format!("- **Family:** {}\n", self.learner.family));
        out.push_str(&format!("- **Method Kind:** `{}`\n", self.learner.kind));
        out.push_str(&format!("- **Representation:** {}\n", self.learner.representation));
        out.push_str(&format!("- **Nominal Complexity:** {}\n", self.learner.nominal_complexity));
        if let Some(eff) = self.learner.effective_complexity {
            out.push_str(&format!("- **Effective Complexity:** `{:.4}`\n", eff));
        }
        if !self.learner.structure.is_empty() {
            out.push_str("- **Structure:**\n");
            for (k, v) in &self.learner.structure {
                out.push_str(&format!("  - {}: `{}`\n", k, v));
            }
        }
        if !self.learner.capabilities.is_empty() {
            out.push_str(&format!("- **Capabilities:** {}\n", self.learner.capabilities.join(", ")));
        }
        out.push('\n');

        // 3. Controller
        out.push_str("## 3. Controller\n\n");
        out.push_str(&format!("- **Method Kind:** `{}`\n", self.controller.kind));
        if let Some(seed) = self.controller.seed {
            out.push_str(&format!("- **Random Seed:** `{}`\n", seed));
        }
        out.push_str(&format!("- **Missing Value Policy:** `{}`\n", self.controller.missing_policy));
        if let Some(b) = self.controller.iteration_budget {
            out.push_str(&format!("- **Iteration Budget:** `{}`\n", b));
        }
        if let Some(b) = self.controller.evaluation_budget {
            out.push_str(&format!("- **Evaluation Budget:** `{}`\n", b));
        }
        if let Some(fb) = &self.controller.fallback_reason {
            out.push_str(&format!("- **Fallback Reason:** {}\n", fb));
        }
        if !self.controller.config.is_empty() {
            out.push_str("- **Configuration Knobs:**\n");
            for (k, v) in &self.controller.config {
                out.push_str(&format!("  - {}: `{}`\n", k, v));
            }
        }
        out.push('\n');

        // 4. Diagnostics
        out.push_str("## 4. Diagnostics\n\n");
        if let Some(conv) = self.diagnostics.converged {
            out.push_str(&format!("- **Converged:** {}\n", if conv { "Yes" } else { "No" }));
        }
        if let Some(reason) = &self.diagnostics.stop_reason {
            out.push_str(&format!("- **Stop Reason:** {}\n", reason));
        }
        if let Some(t) = self.diagnostics.fit_time_sec {
            out.push_str(&format!("- **Fit Time:** `{:.4}s`\n", t));
        }
        if let Some(c) = self.diagnostics.eval_count {
            out.push_str(&format!("- **Evaluations:** `{}`\n", c));
        }
        if !self.diagnostics.metrics.is_empty() {
            out.push_str("- **Quality Metrics:**\n");
            for (k, v) in &self.diagnostics.metrics {
                out.push_str(&format!("  - {}: `{:.6}`\n", k, v));
            }
        }
        if !self.diagnostics.warnings.is_empty() {
            out.push_str("\n> [!WARNING]\n");
            for w in &self.diagnostics.warnings {
                out.push_str(&format!("> - {}\n", w));
            }
        }
        out.push('\n');

        // 5. Inference
        if let Some(inf) = &self.inference {
            out.push_str("## 5. Statistical Inference\n\n");
            if !inf.valid {
                let reason = inf.unavailability_reason.as_deref().unwrap_or("Classical inference is mathematically unavailable for this estimator.");
                out.push_str(&format!("> [!NOTE]\n> **Inference Unavailable:** {}\n\n", reason));
            } else {
                if let (Some(r2), Some(ar2)) = (inf.r_squared, inf.adj_r_squared) {
                    out.push_str(&format!("- **R²:** `{:.4}` | **Adjusted R²:** `{:.4}`\n", r2, ar2));
                }
                if let (Some(f), Some(p)) = (inf.f_statistic, inf.f_p_value) {
                    out.push_str(&format!("- **F-statistic:** `{:.4}` (p = `{:.4e}`)\n", f, p));
                }
                if !inf.coefficients.is_empty() {
                    out.push_str("\n| Term | Estimate | Std.Err | Statistic | p-value | 95% CI Lower | 95% CI Upper |\n");
                    out.push_str("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n");
                    for c in &inf.coefficients {
                        let p_str = if c.p_value < 1e-4 {
                            format!("{:.2e}", c.p_value)
                        } else {
                            format!("{:.4}", c.p_value)
                        };
                        out.push_str(&format!(
                            "| `{}` | {:.4} | {:.4} | {:.4} | {} | {:.4} | {:.4} |\n",
                            c.name, c.estimate, c.std_err, c.stat_value, p_str, c.ci_lower, c.ci_upper
                        ));
                    }
                }
                out.push('\n');
            }
        }

        out
    }

    /// Serialize report to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize report from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}
