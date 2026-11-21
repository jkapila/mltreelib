//! Statistical hypothesis testing and goodness-of-fit summaries.
//!
//! Provides R `summary(glm)` / `summary(lm)` and Python statsmodels parity:
//! coefficient standard errors, z/t-values, p-values, deviances, AIC, BIC,
//! and R-compatible formatted console output.

use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

/// Single parameter coefficient statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct CoeffStats {
    pub name: String,
    pub estimate: f64,
    pub std_err: f64,
    pub stat_value: f64,
    pub p_value: f64,
}

/// Goodness-of-fit and hypothesis testing summary for linear and GLM models.
#[derive(Clone, Debug, PartialEq)]
pub struct GlmSummary {
    pub is_gaussian: bool,
    pub n_obs: usize,
    pub df_null: usize,
    pub df_residual: usize,
    pub null_deviance: f64,
    pub residual_deviance: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub dispersion: f64,
    pub r_squared: Option<f64>,
    pub adj_r_squared: Option<f64>,
    pub f_statistic: Option<f64>,
    pub f_p_value: Option<f64>,
    pub coefficients: Vec<CoeffStats>,
}

impl GlmSummary {
    /// Compute p-value for a z-statistic under standard Normal distribution.
    pub fn p_value_z(z: f64) -> f64 {
        if !z.is_finite() {
            return 1.0;
        }
        let norm = match Normal::new(0.0, 1.0) {
            Ok(n) => n,
            Err(_) => return 1.0,
        };
        let p = 2.0 * (1.0 - norm.cdf(z.abs()));
        p.clamp(0.0, 1.0)
    }

    /// Compute p-value for a t-statistic under Student's t distribution with `df` degrees of freedom.
    pub fn p_value_t(t: f64, df: usize) -> f64 {
        if !t.is_finite() || df == 0 {
            return 1.0;
        }
        let t_dist = match StudentsT::new(0.0, 1.0, df as f64) {
            Ok(dist) => dist,
            Err(_) => return Self::p_value_z(t),
        };
        let p = 2.0 * (1.0 - t_dist.cdf(t.abs()));
        p.clamp(0.0, 1.0)
    }

    /// Formats as an R `summary(glm)` / statsmodels style report.
    pub fn to_r_format(&self, title: &str) -> String {
        let mut out = String::new();
        out.push_str("======================================================================\n");
        out.push_str(&format!("{}\n", title));
        out.push_str("----------------------------------------------------------------------\n");

        if self.is_gaussian {
            let sigma = self.dispersion.sqrt();
            out.push_str(&format!(
                "Residual standard error: {:.4} on {} degrees of freedom\n",
                sigma, self.df_residual
            ));
            if let (Some(r2), Some(adj_r2)) = (self.r_squared, self.adj_r_squared) {
                out.push_str(&format!(
                    "Multiple R-squared:  {:.4},  Adjusted R-squared:  {:.4}\n",
                    r2, adj_r2
                ));
            }
            if let (Some(f_stat), Some(f_p)) = (self.f_statistic, self.f_p_value) {
                let p_df = self.coefficients.len().saturating_sub(1).max(1);
                let f_p_str = if f_p < 2.2e-16 {
                    "< 2.2e-16".to_string()
                } else {
                    format!("{:.4e}", f_p)
                };
                out.push_str(&format!(
                    "F-statistic: {:.2} on {} and {} DF,  p-value: {}\n",
                    f_stat, p_df, self.df_residual, f_p_str
                ));
            }
        } else {
            out.push_str(&format!(
                "Null deviance:     {:.4}  on {}  degrees of freedom\n",
                self.null_deviance, self.df_null
            ));
            out.push_str(&format!(
                "Residual deviance: {:.4}  on {}  degrees of freedom\n",
                self.residual_deviance, self.df_residual
            ));
            out.push_str(&format!("AIC: {:.4},  BIC: {:.4}\n", self.aic, self.bic));
        }

        out.push_str("\nCoefficients:\n");
        let stat_col_name = if self.is_gaussian {
            "t value"
        } else {
            "z value"
        };
        let p_col_name = if self.is_gaussian {
            "Pr(>|t|)"
        } else {
            "Pr(>|z|)"
        };
        out.push_str(&format!(
            "{:<20} {:>12} {:>12} {:>10} {:>12} \n",
            "", "Estimate", "Std. Error", stat_col_name, p_col_name
        ));
        out.push_str("----------------------------------------------------------------------\n");

        for c in &self.coefficients {
            let p_str = if c.p_value < 2.2e-16 {
                "< 2.2e-16".to_string()
            } else if c.p_value < 0.001 {
                format!("{:.4e}", c.p_value)
            } else {
                format!("{:.4}", c.p_value)
            };
            let stars = if c.p_value < 0.001 {
                "***"
            } else if c.p_value < 0.01 {
                "**"
            } else if c.p_value < 0.05 {
                "*"
            } else if c.p_value < 0.1 {
                "."
            } else {
                " "
            };
            out.push_str(&format!(
                "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>12} {:<3}\n",
                c.name, c.estimate, c.std_err, c.stat_value, p_str, stars
            ));
        }
        out.push_str("----------------------------------------------------------------------\n");
        out.push_str("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n");
        out.push_str("======================================================================\n");
        out
    }
}
