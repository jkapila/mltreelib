//! Metrics: complete reports computed in one call — the yardstick every
//! learner is measured with.
//!
//! The regression suite is the legacy featuregen `SummarizeRegression` set
//! (`mean_error`, `bias`, `mae`, `mad`, `mape`, `rmse`, `r2`, `max_error`,
//! `explained_variance`) with the standard MAE formula — the legacy
//! implementation divided by `n` twice. Extended with the DAR **decile error
//! profile** (error by target decile — the bathtub view) and a full
//! classification suite. House semantics: pairs where either side is missing
//! or NaN are dropped from the math and reported in `n_missing`, never hidden.

use std::collections::HashMap;

/// Error statistics for one target decile (DAR bathtub profile).
#[derive(Clone, Debug)]
pub struct DecileStat {
    pub decile: usize,
    pub n: usize,
    pub mean_error: f64,
    pub mae: f64,
    pub mape: f64,
}

/// Complete regression report.
#[derive(Clone, Debug)]
pub struct RegressionReport {
    pub n: usize,
    pub n_missing: usize,
    pub mean_error: f64,
    pub bias: f64,
    pub mae: f64,
    pub mad: f64,
    pub mape: f64,
    pub rmse: f64,
    pub r2: f64,
    pub max_error: f64,
    pub explained_variance: f64,
    pub deciles: Vec<DecileStat>,
}

fn variance(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

/// All regression metrics plus the decile error profile, computed at once.
pub fn regression_report(
    y_true: &[Option<f64>],
    y_pred: &[Option<f64>],
) -> Result<RegressionReport, String> {
    if y_true.len() != y_pred.len() {
        return Err("y_true and y_pred length mismatch".into());
    }
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    let mut n_missing = 0usize;
    for (t, p) in y_true.iter().zip(y_pred) {
        match (t, p) {
            (Some(t), Some(p)) if !t.is_nan() && !p.is_nan() => pairs.push((*t, *p)),
            _ => n_missing += 1,
        }
    }
    let n = pairs.len();
    if n == 0 {
        return Err("no observed pairs to score".into());
    }

    let errors: Vec<f64> = pairs.iter().map(|(t, p)| t - p).collect();
    let mean_error = errors.iter().sum::<f64>() / n as f64;
    let mean_true = pairs.iter().map(|(t, _)| t).sum::<f64>() / n as f64;
    let mean_pred = pairs.iter().map(|(_, p)| p).sum::<f64>() / n as f64;
    let bias = if mean_true != 0.0 { mean_pred / mean_true } else { mean_pred };
    let abs_err: Vec<f64> = errors.iter().map(|e| e.abs()).collect();
    let mae = abs_err.iter().sum::<f64>() / n as f64;
    let mad = errors.iter().map(|e| (e - mean_error).abs()).sum::<f64>() / n as f64;
    // MAPE over non-zero targets only (the legacy convention)
    let mape_pairs: Vec<(f64, f64)> = pairs.iter().copied().filter(|(t, _)| *t != 0.0).collect();
    let mape = if mape_pairs.is_empty() {
        f64::NAN
    } else {
        100.0
            * mape_pairs
                .iter()
                .map(|(t, p)| ((t - p) / t).abs())
                .sum::<f64>()
            / mape_pairs.len() as f64
    };
    let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / n as f64).sqrt();
    let ss_res: f64 = errors.iter().map(|e| e * e).sum();
    let true_var = variance(&pairs.iter().map(|(t, _)| *t).collect::<Vec<f64>>());
    let ss_tot = true_var * (n as f64 - 1.0);
    let r2 = if ss_tot != 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
    let max_error = abs_err.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let err_var = variance(&errors);
    let explained_variance = if true_var != 0.0 {
        1.0 - err_var / true_var
    } else {
        0.0
    };

    // DAR decile profile: sort by target, ten equal-frequency groups
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| pairs[a].0.partial_cmp(&pairs[b].0).unwrap_or(std::cmp::Ordering::Equal));
    let mut deciles = Vec::with_capacity(10);
    for d in 0..10 {
        let lo = d * n / 10;
        let hi = ((d + 1) * n / 10).max(lo + 1).min(n);
        if lo >= n {
            break;
        }
        let slice = &order[lo..hi];
        let m = slice.len() as f64;
        let me = slice.iter().map(|&i| errors[i]).sum::<f64>() / m;
        let mae_d = slice.iter().map(|&i| abs_err[i]).sum::<f64>() / m;
        let mape_slice: Vec<f64> = slice
            .iter()
            .filter(|&&i| pairs[i].0 != 0.0)
            .map(|&i| (errors[i] / pairs[i].0).abs())
            .collect();
        let mape_d = if mape_slice.is_empty() {
            f64::NAN
        } else {
            100.0 * mape_slice.iter().sum::<f64>() / mape_slice.len() as f64
        };
        deciles.push(DecileStat {
            decile: d + 1,
            n: slice.len(),
            mean_error: me,
            mae: mae_d,
            mape: mape_d,
        });
    }

    Ok(RegressionReport {
        n,
        n_missing,
        mean_error,
        bias,
        mae,
        mad,
        mape,
        rmse,
        r2,
        max_error,
        explained_variance,
        deciles,
    })
}

/// Per-class classification statistics.
#[derive(Clone, Debug)]
pub struct ClassStat {
    pub class: u32,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

/// Complete classification report. `logloss`, `brier` and `auc` are NaN when
/// `probs` is not supplied (brier/auc need binary targets with `probs` =
/// probability of class 1).
#[derive(Clone, Debug)]
pub struct ClassificationReport {
    pub n: usize,
    pub n_missing: usize,
    pub n_classes: usize,
    pub accuracy: f64,
    pub macro_precision: f64,
    pub macro_recall: f64,
    pub macro_f1: f64,
    pub per_class: Vec<ClassStat>,
    pub confusion: Vec<Vec<f64>>,
    pub logloss: f64,
    pub brier: f64,
    pub auc: f64,
}

/// All classification metrics computed at once.
pub fn classification_report(
    y_true: &[Option<f64>],
    y_pred: &[Option<f64>],
    probs: Option<&[Option<f64>]>,
) -> Result<ClassificationReport, String> {
    if y_true.len() != y_pred.len() {
        return Err("y_true and y_pred length mismatch".into());
    }
    if let Some(p) = probs {
        if p.len() != y_true.len() {
            return Err("probs length mismatch".into());
        }
    }
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut prob_pairs: Vec<f64> = Vec::new();
    let mut n_missing = 0usize;
    let mut n_classes = 0usize;
    for (i, (t, p)) in y_true.iter().zip(y_pred).enumerate() {
        match (t, p) {
            (Some(t), Some(p)) if !t.is_nan() && !p.is_nan() => {
                if *t < 0.0 || t.fract() != 0.0 || *p < 0.0 || p.fract() != 0.0 {
                    return Err("y_true and y_pred must be integer class labels".into());
                }
                let (t, p) = (*t as usize, *p as usize);
                n_classes = n_classes.max(t + 1).max(p + 1);
                pairs.push((t, p));
                if let Some(probs) = probs {
                    match probs[i] {
                        Some(pr) if !pr.is_nan() => prob_pairs.push(pr),
                        _ => prob_pairs.push(f64::NAN),
                    }
                }
            }
            _ => {
                n_missing += 1;
                if probs.is_some() {
                    prob_pairs.push(f64::NAN);
                }
            }
        }
    }
    let n = pairs.len();
    if n == 0 {
        return Err("no observed pairs to score".into());
    }
    if n_classes < 2 {
        return Err("need at least 2 classes".into());
    }

    let mut confusion = vec![vec![0.0f64; n_classes]; n_classes];
    for &(t, p) in &pairs {
        confusion[t][p] += 1.0;
    }
    let correct = pairs.iter().filter(|(t, p)| t == p).count() as f64;
    let accuracy = correct / n as f64;

    let mut per_class = Vec::with_capacity(n_classes);
    for c in 0..n_classes {
        let tp = confusion[c][c];
        let fp: f64 = (0..n_classes).map(|r| confusion[r][c]).sum::<f64>() - tp;
        let fn_: f64 = (0..n_classes).map(|k| confusion[c][k]).sum::<f64>() - tp;
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        per_class.push(ClassStat {
            class: c as u32,
            precision,
            recall,
            f1,
            support: (tp + fn_) as usize,
        });
    }
    let k = n_classes as f64;
    let macro_precision = per_class.iter().map(|c| c.precision).sum::<f64>() / k;
    let macro_recall = per_class.iter().map(|c| c.recall).sum::<f64>() / k;
    let macro_f1 = per_class.iter().map(|c| c.f1).sum::<f64>() / k;

    let mut logloss = f64::NAN;
    let mut brier = f64::NAN;
    let mut auc = f64::NAN;
    if let Some(probs) = probs {
        let rows: Vec<(usize, f64)> = pairs
            .iter()
            .zip(prob_pairs.iter())
            .filter(|(_, pr)| !pr.is_nan())
            .map(|(&(t, _), pr)| (t, *pr))
            .collect();
        if !rows.is_empty() {
            let eps = 1e-15;
            logloss = -rows
                .iter()
                .map(|(t, pr)| {
                    let p = pr.clamp(eps, 1.0 - eps);
                    if *t == 1 {
                        p.ln()
                    } else {
                        (1.0 - p).ln()
                    }
                })
                .sum::<f64>()
                / rows.len() as f64;
        }
        if n_classes == 2 && !rows.is_empty() {
            brier = rows
                .iter()
                .map(|(t, pr)| (pr - if *t == 1 { 1.0 } else { 0.0 }).powi(2))
                .sum::<f64>()
                / rows.len() as f64;
            // rank-based AUC (Mann-Whitney), ties get average ranks
            let mut sorted = rows.clone();
            sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut ranks = vec![0.0f64; sorted.len()];
            let mut i = 0usize;
            while i < sorted.len() {
                let mut j = i + 1;
                while j < sorted.len() && sorted[j].1 == sorted[i].1 {
                    j += 1;
                }
                let avg = (i + j + 1) as f64 / 2.0; // 1-based average rank
                for r in ranks[i..j].iter_mut() {
                    *r = avg;
                }
                i = j;
            }
            let n_pos: f64 = rows.iter().filter(|(t, _)| *t == 1).count() as f64;
            let n_neg = rows.len() as f64 - n_pos;
            if n_pos > 0.0 && n_neg > 0.0 {
                let rank_sum: f64 = sorted
                    .iter()
                    .zip(ranks.iter())
                    .filter(|((t, _), _)| *t == 1)
                    .map(|(_, r)| *r)
                    .sum();
                auc = (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg);
            }
        }
    }

    Ok(ClassificationReport {
        n,
        n_missing,
        n_classes,
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        per_class,
        confusion,
        logloss,
        brier,
        auc,
    })
}

/// Regression report per group (the legacy `summarize_key` idea), sorted by
/// key for determinism.
pub fn grouped_regression_report(
    keys: &[Option<String>],
    y_true: &[Option<f64>],
    y_pred: &[Option<f64>],
) -> Result<Vec<(String, RegressionReport)>, String> {
    if keys.len() != y_true.len() || keys.len() != y_pred.len() {
        return Err("keys, y_true and y_pred length mismatch".into());
    }
    let mut groups: HashMap<String, (Vec<Option<f64>>, Vec<Option<f64>>)> = HashMap::new();
    for ((k, t), p) in keys.iter().zip(y_true).zip(y_pred) {
        match k {
            Some(k) => {
                let e = groups.entry(k.clone()).or_default();
                e.0.push(*t);
                e.1.push(*p);
            }
            None => continue,
        }
    }
    let mut out: Vec<(String, RegressionReport)> = groups
        .into_iter()
        .map(|(k, (t, p))| regression_report(&t, &p).map(|r| (k, r)))
        .collect::<Result<_, _>>()?;
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}
