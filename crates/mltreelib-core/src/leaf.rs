//! Leaves: what a leaf model is ([`LeafKind`]), which columns it may use
//! ([`LeafCols`]), and fitting / evaluating it on a node's rows.

use crate::family::LeafFamily;
use crate::model_tree::Grower;
use crate::node_stats::{
    design_row, fit_glm, fit_linear, Gram, HeteroscedasticModel, LeafModel, LeafStats,
    LinearModel, SoftmaxModel, SurvivalStats,
};
use crate::stats::{huber_location, w_median, w_quantile};

/// Which columns a leaf's linear model may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafCols {
    /// Every usable predictor column.
    All,
    /// Only the columns used in splits along the root→leaf path.
    PathUsed,
}

/// What the leaf model is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LeafKind {
    /// Linear model over the chosen columns (the model tree).
    Linear,
    /// Plain mean — the CART/rpart leaf. The split criterion then *is*
    /// rpart's deviance criterion.
    Mean,
    /// Weighted median — L1-optimal constant, outlier-robust.
    Median,
    /// Weighted quantile at tau (0.5 = median; any tau in (0, 1)).
    Quantile(f64),
    /// Huber M-location (quadratic near the center, linear in the tails).
    Huber,
}

/// Parameter controls for fitting leaf models independently of tree topology.
#[derive(Clone, Debug, PartialEq)]
pub struct LeafControl {
    /// Leaf model kind (Linear, Mean, Median, Quantile(t), Huber).
    pub leaf: LeafKind,
    /// Distribution / link function for the leaf model.
    pub family: LeafFamily,
    /// Which predictor columns linear leaves may use (All vs PathUsed).
    pub leaf_cols: LeafCols,
    /// Ridge penalty on leaf coefficients (intercept unpenalized).
    pub l2: f64,
    /// Lasso penalty on leaf coefficients (intercept unpenalized).
    pub l1: f64,
    /// Keep each leaf's t-digest for quantile / interval prediction.
    pub leaf_digest: bool,
    /// Monotone constraints (-1, 0, +1) per feature column.
    pub monotone: Vec<i8>,
    /// Specific column subset for leaf models. If Some, overrides leaf_cols policy.
    pub leaf_features: Option<Vec<usize>>,
    /// Family-specific extra parameters (e.g. tweedie_p, tobit_threshold, etc.).
    pub extra: std::collections::HashMap<String, f64>,
    /// Optional event indicators (1=event, 0=censored) for survival leaves.
    pub events: Option<Vec<f64>>,
    /// Number of distinct classes for softmax targets.
    pub n_classes: Option<usize>,
}

impl Default for LeafControl {
    fn default() -> Self {
        Self {
            leaf: LeafKind::Linear,
            family: LeafFamily::Gaussian,
            leaf_cols: LeafCols::All,
            l2: 1e-3,
            l1: 0.0,
            leaf_digest: true,
            monotone: Vec::new(),
            leaf_features: None,
            extra: std::collections::HashMap::new(),
            events: None,
            n_classes: None,
        }
    }
}

impl From<&crate::model_tree::ModelTreeParams> for LeafControl {
    fn from(p: &crate::model_tree::ModelTreeParams) -> Self {
        Self {
            leaf: p.leaf,
            family: p.leaf_family,
            leaf_cols: p.leaf_cols,
            l2: p.l2,
            l1: p.l1,
            leaf_digest: p.leaf_digest,
            monotone: p.monotone.clone(),
            leaf_features: None,
            extra: std::collections::HashMap::new(),
            events: p.events.clone(),
            n_classes: p.n_classes,
        }
    }
}

impl LeafControl {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_leaf(mut self, leaf: LeafKind) -> Self {
        self.leaf = leaf;
        self
    }

    pub fn with_family(mut self, family: LeafFamily) -> Self {
        self.family = family;
        self
    }

    pub fn with_leaf_cols(mut self, leaf_cols: LeafCols) -> Self {
        self.leaf_cols = leaf_cols;
        self
    }

    pub fn with_l2(mut self, l2: f64) -> Self {
        self.l2 = l2;
        self
    }

    pub fn with_l1(mut self, l1: f64) -> Self {
        self.l1 = l1;
        self
    }

    pub fn with_digest(mut self, leaf_digest: bool) -> Self {
        self.leaf_digest = leaf_digest;
        self
    }

    pub fn with_monotone(mut self, monotone: Vec<i8>) -> Self {
        self.monotone = monotone;
        self
    }

    pub fn with_leaf_features(mut self, features: Vec<usize>) -> Self {
        self.leaf_features = Some(features);
        self
    }

    pub fn with_param(mut self, key: impl Into<String>, val: f64) -> Self {
        self.extra.insert(key.into(), val);
        self
    }
}

/// Helper to select active columns for a leaf under the given column policy.
pub(crate) fn chosen_cols_policy(
    leaf_cols: LeafCols,
    leaf_features: Option<&[usize]>,
    active: &[usize],
    path: &[usize],
) -> Vec<usize> {
    if let Some(feats) = leaf_features {
        let mut p: Vec<usize> = feats
            .iter()
            .copied()
            .filter(|c| active.contains(c))
            .collect();
        p.sort_unstable();
        p.dedup();
        p
    } else {
        match leaf_cols {
            LeafCols::All => active.to_vec(),
            LeafCols::PathUsed => {
                let mut p: Vec<usize> = path
                    .iter()
                    .copied()
                    .filter(|c| active.contains(c))
                    .collect();
                p.sort_unstable();
                p.dedup();
                p
            }
        }
    }
}

/// Helper to determine whether any missing values exist and total design dimension.
pub(crate) fn compute_leaf_design(
    cols: &[Vec<Option<f64>>],
    rows: &[usize],
    chosen: &[usize],
) -> (bool, usize) {
    let any_missing = rows
        .iter()
        .any(|&r| chosen.iter().any(|&c| cols[c][r].is_none()));
    let dim = 1 + chosen.len() + if any_missing { chosen.len() } else { 0 };
    (any_missing, dim)
}

/// Fit a leaf model on the specified rows and columns using `control`.
pub fn fit_leaf_model(
    cols: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    rows: &[usize],
    path: &[usize],
    active: &[usize],
    control: &LeafControl,
) -> LeafModel {
    let stats = if control.leaf_digest {
        let yv: Vec<f64> = rows.iter().map(|&r| y[r]).collect();
        let wv: Vec<f64> = rows.iter().map(|&r| w[r]).collect();
        LeafStats::from_targets(&yv, &wv)
    } else {
        LeafStats::from_rows(y, w, rows)
    };
    let family = control.family;
    if family == LeafFamily::Softmax {
        let n_classes = control.n_classes.unwrap_or_else(|| {
            rows.iter()
                .filter(|&&r| w[r] > 0.0)
                .map(|&r| (y[r] as usize) + 1)
                .max()
                .unwrap_or(2)
                .max(2)
        });
        let mut probs = vec![0.0; n_classes];
        let mut total_w = 0.0;
        for &r in rows {
            if w[r] > 0.0 {
                let c = y[r] as usize;
                if c < n_classes {
                    probs[c] += w[r];
                    total_w += w[r];
                }
            }
        }
        if total_w > 0.0 {
            for p in &mut probs {
                *p /= total_w;
            }
        } else {
            probs.fill(1.0 / n_classes as f64);
        }
        let best_c = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(c, _)| c as f64)
            .unwrap_or(0.0);

        let linear_models = if control.leaf == LeafKind::Linear {
            let chosen = chosen_cols_policy(control.leaf_cols, control.leaf_features.as_deref(), active, path);
            let (any_missing, _) = compute_leaf_design(cols, rows, &chosen);
            let mut models = Vec::with_capacity(n_classes);
            for c in 0..n_classes {
                let mut binary_y: Vec<f64> = Vec::with_capacity(y.len());
                for (_r, &val) in y.iter().enumerate() {
                    binary_y.push(if val as usize == c { 1.0 } else { 0.0 });
                }
                let mean_c = probs[c].clamp(1e-6, 1.0 - 1e-6);
                let fitted = fit_glm(
                    cols,
                    &binary_y,
                    w,
                    rows,
                    &chosen,
                    any_missing,
                    &control.monotone,
                    control.l1,
                    control.l2,
                    LeafFamily::Binomial,
                    mean_c,
                );
                let lm = fitted.unwrap_or_else(|| LinearModel {
                    coef: vec![0.0; chosen.len()],
                    na_coef: if any_missing { vec![0.0; chosen.len()] } else { Vec::new() },
                    cols: chosen.clone(),
                    intercept: mean_c.ln() - (1.0 - mean_c).ln(),
                });
                models.push(lm);
            }
            Some(models)
        } else {
            None
        };

        return LeafModel {
            stats,
            linear: None,
            center: Some(best_c),
            family,
            softmax: Some(SoftmaxModel {
                n_classes,
                probs,
                linear_models,
            }),
            survival: None,
            heteroscedastic: None,
        };
    }

    if family == LeafFamily::Survival {
        let events = control.events.as_ref();
        let mut records: Vec<(f64, f64, f64)> = rows
            .iter()
            .filter(|&&r| w[r] > 0.0 && y[r] >= 0.0 && y[r].is_finite())
            .map(|&r| {
                let e = events.and_then(|evs| evs.get(r)).copied().unwrap_or(1.0);
                (y[r], e, w[r])
            })
            .collect();
        records.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut distinct_event_times: Vec<f64> = Vec::new();
        for &(t, e, _) in &records {
            if e > 0.0
                && (distinct_event_times.is_empty()
                    || *distinct_event_times.last().unwrap() < t)
            {
                distinct_event_times.push(t);
            }
        }

        let mut cum_hazard = Vec::with_capacity(distinct_event_times.len());
        let mut survival = Vec::with_capacity(distinct_event_times.len());
        let mut current_h = 0.0;
        let mut current_s = 1.0;
        let mut median_time = records.last().map(|r| r.0).unwrap_or(0.0);
        let mut found_median = false;

        for &tj in &distinct_event_times {
            let mut dj = 0.0;
            let mut nj = 0.0;
            for &(ti, ei, wi) in &records {
                if ti >= tj {
                    nj += wi;
                    if (ti - tj).abs() < 1e-9 && ei > 0.0 {
                        dj += wi;
                    }
                }
            }
            if nj > 0.0 {
                let hj = dj / nj;
                current_h += hj;
                current_s *= (1.0 - hj).clamp(0.0, 1.0);
            }
            cum_hazard.push(current_h);
            survival.push(current_s);
            if !found_median && current_s <= 0.5 {
                median_time = tj;
                found_median = true;
            }
        }

        return LeafModel {
            stats,
            linear: None,
            center: Some(median_time),
            family,
            softmax: None,
            survival: Some(SurvivalStats {
                times: distinct_event_times,
                cum_hazard,
                survival,
                median_time,
            }),
            heteroscedastic: None,
        };
    }

    if family == LeafFamily::Heteroscedastic {
        let (linear, scale_linear, scale) = if control.leaf == LeafKind::Linear {
            let chosen = chosen_cols_policy(control.leaf_cols, control.leaf_features.as_deref(), active, path);
            let (any_missing, dim) = compute_leaf_design(cols, rows, &chosen);
            let mut g = Gram::new(dim);
            let mut row = vec![0.0; dim];
            for &r in rows {
                design_row(&mut row, cols, r, &chosen, any_missing);
                g.add_row(&row, y[r], w[r]);
            }
            let loc_model = fit_linear(
                &g,
                &chosen,
                any_missing,
                &control.monotone,
                control.l1,
                control.l2,
            );

            let mut g_scale = Gram::new(dim);
            let mut sw = 0.0;
            let mut s_res2 = 0.0;
            for &r in rows {
                if w[r] <= 0.0 {
                    continue;
                }
                design_row(&mut row, cols, r, &chosen, any_missing);
                let pred = loc_model.eval_design(&row);
                let res2 = (y[r] - pred).powi(2);
                sw += w[r];
                s_res2 += w[r] * res2;
                let log_res2 = (res2 + 1e-8).ln();
                g_scale.add_row(&row, log_res2, w[r]);
            }
            let mut s_model = fit_linear(
                &g_scale,
                &chosen,
                any_missing,
                &vec![0; chosen.len()],
                control.l1,
                control.l2,
            );
            // Harvey (1976) bias correction for log chi-squared(1) residuals:
            // E[ln(e_i^2)] = ln(sigma_i^2) + E[ln(chi_1^2)] where E[ln(chi_1^2)] = -1.2703628.
            // Adding 1.2703628 to the intercept makes exp(0.5 * z'gamma) unbiased for sigma.
            s_model.intercept += 1.2703628;
            let raw_scale = if sw > 0.0 { (s_res2 / sw).sqrt().max(1e-6) } else { 1.0 };
            (Some(loc_model), Some(s_model), raw_scale)
        } else {
            let mut sw = 0.0;
            let mut s_res2 = 0.0;
            for &r in rows {
                if w[r] > 0.0 {
                    sw += w[r];
                    s_res2 += w[r] * (y[r] - stats.mean).powi(2);
                }
            }
            let raw_scale = if sw > 0.0 { (s_res2 / sw).sqrt().max(1e-6) } else { stats.std.max(1e-6) };
            (None, None, raw_scale)
        };

        let mean_val = stats.mean;
        return LeafModel {
            stats,
            linear,
            center: Some(mean_val),
            family,
            softmax: None,
            survival: None,
            heteroscedastic: Some(HeteroscedasticModel {
                scale,
                scale_linear,
            }),
        };
    }

    if family != LeafFamily::Gaussian
        && (control.leaf == LeafKind::Mean
            || ((family == LeafFamily::Binomial || family == LeafFamily::Probit) && (stats.mean == 0.0 || stats.mean == 1.0))
            || ((family == LeafFamily::Poisson || family == LeafFamily::Tweedie) && stats.mean == 0.0)
            || (family == LeafFamily::Tobit && stats.mean <= 0.0))
    {
        return LeafModel {
            center: Some(stats.mean),
            stats,
            linear: None,
            family,
            softmax: None,
            survival: None,
            heteroscedastic: None,
        };
    }
    let linear = if control.leaf == LeafKind::Linear {
        let chosen = chosen_cols_policy(control.leaf_cols, control.leaf_features.as_deref(), active, path);
        let (any_missing, dim) = compute_leaf_design(cols, rows, &chosen);
        if family != LeafFamily::Gaussian {
            let fitted = fit_glm(
                cols, y, w, rows, &chosen, any_missing,
                &control.monotone, control.l1, control.l2,
                family, stats.mean,
            );
            // The tree fit rejects nonfinite final models; scans reject
            // their loss. A failed GLM must never masquerade as a mean fit.
            let linear = fitted.unwrap_or_else(|| LinearModel {
                coef: vec![0.0; chosen.len()],
                na_coef: if any_missing { vec![0.0; chosen.len()] } else { Vec::new() },
                cols: chosen,
                intercept: f64::NAN,
            });
            return LeafModel {
                stats,
                linear: Some(linear),
                center: None,
                family,
                softmax: None,
                survival: None,
                heteroscedastic: None,
            };
        }
        let mut g = Gram::new(dim);
        let mut row = vec![0.0; dim];
        for &r in rows {
            design_row(&mut row, cols, r, &chosen, any_missing);
            g.add_row(&row, y[r], w[r]);
        }
        Some(fit_linear(
            &g,
            &chosen,
            any_missing,
            &control.monotone,
            control.l1,
            control.l2,
        ))
    } else {
        None
    };
    // Non-mean constant leaves: fit the center on the positive-weight
    // targets only (same population as the stats).
    let yws: Vec<(f64, f64)> = if matches!(
        control.leaf, LeafKind::Median | LeafKind::Quantile(_) | LeafKind::Huber
    ) {
        rows.iter().filter(|&&r| w[r] > 0.0)
            .map(|&r| (y[r], w[r])).collect()
    } else {
        Vec::new()
    };
    let center = match control.leaf {
        LeafKind::Median => Some(w_median(&yws)),
        LeafKind::Quantile(t) => Some(w_quantile(&yws, t)),
        LeafKind::Huber => Some(huber_location(&yws)),
        _ => None,
    };
    LeafModel {
        stats,
        linear,
        center,
        family,
        softmax: None,
        survival: None,
        heteroscedastic: None,
    }
}

impl<'a> Grower<'a> {
    /// Columns a leaf model uses for a given root→leaf path (restricted to
    /// the pre-selected active columns).
    pub(crate) fn chosen_cols(&self, path: &[usize]) -> Vec<usize> {
        chosen_cols_policy(self.params.leaf_cols, None, self.active, path)
    }

    /// Leaf design for a node: the leaf columns under the column policy,
    /// whether any of them is missing on these rows (NA-indicator columns),
    /// and the resulting design dimension (intercept included).
    fn leaf_design(&self, rows: &[usize], path: &[usize]) -> (Vec<usize>, bool, usize) {
        let chosen = self.chosen_cols(path);
        let (any_missing, dim) = compute_leaf_design(self.cols, rows, &chosen);
        (chosen, any_missing, dim)
    }

    /// Weighted residual sum of squares of a leaf fit on `rows`, with the
    /// leaf's parameter count (mean-style leaves: 1; linear: the design
    /// dimension). Returns NaN RSS when the fit cannot produce finite
    /// predictions (the gate then skips).
    pub(crate) fn node_rss(&self, rows: &[usize], path: &[usize]) -> (f64, usize) {
        let leaf = self.make_leaf(rows, path);
        let k = if self.params.leaf == LeafKind::Linear {
            self.leaf_design(rows, path).2
        } else {
            1
        };
        let mut xrow = vec![None; self.cols.len()];
        let mut rss = 0.0;
        for &r in rows {
            for (j, c) in self.cols.iter().enumerate() {
                xrow[j] = c[r];
            }
            let e = self.y[r] - leaf.predict(&xrow);
            if !e.is_finite() {
                return (f64::NAN, k);
            }
            rss += self.w[r] * e * e;
        }
        (rss, k)
    }

    /// Rows a leaf model fits on: the honesty estimation rows when there
    /// are enough (`min_rows`), else the node's structure rows.
    pub(crate) fn leaf_rows<'b>(&self, rows: &'b [usize], est: &'b [usize]) -> &'b [usize] {
        if self.params.honesty && est.len() >= self.params.min_rows {
            est
        } else {
            rows
        }
    }

    /// Fit the leaf model for a node's rows.
    pub(crate) fn make_leaf(&self, rows: &[usize], path: &[usize]) -> LeafModel {
        self.fit_leaf(rows, path, self.params.leaf_digest)
    }

    /// Candidate fitting shares the payload fitter but does not sort targets
    /// or allocate a target distribution. Never store this leaf in the tree.
    pub(crate) fn make_leaf_for_scan(&self, rows: &[usize], path: &[usize]) -> LeafModel {
        self.fit_leaf(rows, path, false)
    }

    /// Response-scale weighted deviance of an already fitted leaf.
    pub(crate) fn leaf_deviance(&self, rows: &[usize], leaf: &LeafModel) -> f64 {
        let mut total = 0.0;
        for &r in rows {
            if self.w[r] <= 0.0 {
                continue;
            }
            let loss = leaf.family.deviance(self.y[r], leaf.predict_columns(self.cols, r));
            if !loss.is_finite() {
                return loss;
            }
            total += self.w[r] * loss;
        }
        total
    }

    fn fit_leaf(&self, rows: &[usize], path: &[usize], digest: bool) -> LeafModel {
        let mut ctrl = LeafControl::from(self.params);
        ctrl.leaf_digest = digest;
        fit_leaf_model(self.cols, self.y, self.w, rows, path, self.active, &ctrl)
    }
}
