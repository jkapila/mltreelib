//! Statistical stopping gate (the `split_test` knob): the scan's chosen
//! split must significantly beat the pooled node model — a Chow
//! structural-break F test — else the node stays a leaf.

use crate::model_tree::Grower;
use crate::stats::f_sf;
/// Split significance gate (statistical stopping).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SplitTest {
    /// No gate — every candidate split behaves as before (default: every
    /// knob is default-off until a measured round decides).
    None,
    /// Chow structural-break F test: the chosen split must produce two
    /// child leaf models significantly better than the pooled node model
    /// (`p < split_alpha`), else the node stays a leaf. Regression
    /// lineage: the MOB/ctree family's significance stopping, as a
    /// post-selection gate on the scan's best split.
    Chow,
    /// Model-based recursive partitioning (MOB / Zeileis et al. 2008):
    /// evaluates cumulative empirical fluctuation processes of score residuals
    /// along candidate variables. Selects the variable with minimum p-value
    /// and tests whether parameter instability is statistically significant
    /// (Bonferroni-adjusted p < split_alpha), bypassing candidate-level refits.
    Mob,
    /// Conditional Inference Trees (CTree / Hothorn, Hornik & Zeileis 2006):
    /// separates variable selection from splitpoint selection using permutation
    /// rank hypothesis testing. Tests independence between response and covariates,
    /// selecting the variable with minimum adjusted p-value. Strictly unbiased
    /// against variable scale and cardinality (solves the cardinality trap).
    Ctree,
}

/// Statistical gate result recorded on the node's training-log profile.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct GateNote {
    #[serde(with = "crate::persistence::float")]
    pub f: f64,
    #[serde(with = "crate::persistence::float")]
    pub p: f64,
    /// Effective significance level (`split_alpha`, Bonferroni-divided).
    #[serde(with = "crate::persistence::float")]
    pub alpha: f64,
    pub passed: bool,
    pub test_name: &'static str,
    /// Per-variable test diagnostics: (col_idx, test_statistic, adj_p_value)
    #[serde(with = "crate::persistence::float")]
    pub var_p_values: Vec<(usize, f64, f64)>,
}

impl<'de> serde::Deserialize<'de> for GateNote {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct State {
            #[serde(with = "crate::persistence::float")]
            f: f64,
            #[serde(with = "crate::persistence::float")]
            p: f64,
            #[serde(with = "crate::persistence::float")]
            alpha: f64,
            passed: bool,
            test_name: String,
            #[serde(with = "crate::persistence::float")]
            var_p_values: Vec<(usize, f64, f64)>,
        }

        let state = State::deserialize(d)?;
        let test_name = match state.test_name.as_str() {
            "chow" => "chow",
            "ctree" => "ctree",
            "mob" => "mob",
            _ => return Err(serde::de::Error::custom("unknown model split test")),
        };
        Ok(Self {
            f: state.f,
            p: state.p,
            alpha: state.alpha,
            passed: state.passed,
            test_name,
            var_p_values: state.var_p_values,
        })
    }
}

impl<'a> Grower<'a> {
    /// Chow structural-break F test on a proposed split: the two child
    /// leaf models (same design, the child path) against the pooled node
    /// model. `None` = not computable (df <= 0, non-finite residuals) —
    /// the split is then allowed (the gate is permissive when it cannot
    /// speak). Degrees of freedom use row counts; weights enter the fits
    /// and sums of squares (documented approximation).
    pub(crate) fn chow_test(
        &self,
        rows: &[usize],
        lrows: &[usize],
        rrows: &[usize],
        path: &[usize],
        col: usize,
    ) -> Option<(f64, f64)> {
        let mut cpath = path.to_vec();
        cpath.push(col);
        let (rss_l, k) = self.node_rss(lrows, &cpath);
        let (rss_r, k2) = self.node_rss(rrows, &cpath);
        let (rss_p, _) = self.node_rss(rows, &cpath);
        if k != k2 || !(rss_l.is_finite() && rss_r.is_finite() && rss_p.is_finite()) {
            return None;
        }
        let n = rows.len() as f64;
        let kk = k as f64;
        let df2 = n - 2.0 * kk;
        if df2 <= 0.0 {
            return None;
        }
        let rss_s = rss_l + rss_r;
        if rss_s <= 0.0 {
            // Perfect child fits: infinitely significant.
            return Some((f64::INFINITY, 0.0));
        }
        let f = ((rss_p - rss_s) / kk) / (rss_s / df2);
        if !f.is_finite() || f <= 0.0 {
            // No (or numerically negative) improvement: no evidence.
            return Some((0.0, 1.0));
        }
        Some((f, f_sf(f, k as u32, df2 as u32)))
    }

}
