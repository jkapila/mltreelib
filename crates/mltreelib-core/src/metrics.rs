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

/// One bin of a calibration (reliability) curve.
#[derive(Clone, Copy, Debug)]
pub struct CalibBin {
    pub left: f64,
    pub right: f64,
    pub mean_score: f64,
    pub observed: f64,
    pub n: usize,
}

/// Stable descending-by-score index (ties keep input order).
fn rank_desc(y_true: &[f64], scores: &[f64]) -> Result<Vec<usize>, String> {
    if y_true.len() != scores.len() {
        return Err("y_true and scores must have the same length".to_string());
    }
    if scores.is_empty() {
        return Err("empty input".to_string());
    }
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(idx)
}

/// Equal-width calibration bins over [0, 1]: mean score vs observed rate per
/// bin. Scores clip into edge bins; empty bins are omitted. Positive class
/// = y > 0.5.
pub fn calibration_bins(
    y_true: &[f64],
    scores: &[f64],
    n_bins: usize,
) -> Result<Vec<CalibBin>, String> {
    if y_true.len() != scores.len() || y_true.is_empty() {
        return Err("y_true and scores must have the same non-zero length".to_string());
    }
    let nb = n_bins.max(1);
    let mut acc = vec![(0.0f64, 0.0f64, 0usize); nb];
    for (&t, &s) in y_true.iter().zip(scores.iter()) {
        let c = if s.is_finite() {
            s.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let b = ((c * nb as f64) as usize).min(nb - 1);
        acc[b].0 += c;
        acc[b].1 += if t > 0.5 { 1.0 } else { 0.0 };
        acc[b].2 += 1;
    }
    Ok(acc
        .iter()
        .enumerate()
        .filter(|(_, &(_, _, n))| n > 0)
        .map(|(i, &(ss, oo, n))| CalibBin {
            left: i as f64 / nb as f64,
            right: (i + 1) as f64 / nb as f64,
            mean_score: ss / n as f64,
            observed: oo / n as f64,
            n,
        })
        .collect())
}

/// ROC points (fpr, tpr) from a threshold sweep on scores descending, plus
/// AUC (trapezoid over the empirical curve). Positive class = y > 0.5.
pub fn roc_curve(y_true: &[f64], scores: &[f64]) -> Result<(Vec<(f64, f64)>, f64), String> {
    let idx = rank_desc(y_true, scores)?;
    let n_pos = y_true.iter().filter(|&&t| t > 0.5).count() as f64;
    let n_neg = y_true.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return Err("roc_curve needs both classes present".to_string());
    }
    let (mut tp, mut fp) = (0.0f64, 0.0f64);
    let mut pts = vec![(0.0, 0.0)];
    let mut i = 0;
    while i < idx.len() {
        let s = scores[idx[i]];
        while i < idx.len() && scores[idx[i]] == s {
            if y_true[idx[i]] > 0.5 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            i += 1;
        }
        pts.push((fp / n_neg, tp / n_pos));
    }
    let auc = pts
        .windows(2)
        .map(|w| (w[1].0 - w[0].0) * (w[1].1 + w[0].1) / 2.0)
        .sum();
    Ok((pts, auc))
}

/// Precision-recall points (recall, precision) + average precision (step-wise,
/// sklearn convention). Positive class = y > 0.5.
pub fn pr_curve(y_true: &[f64], scores: &[f64]) -> Result<(Vec<(f64, f64)>, f64), String> {
    let idx = rank_desc(y_true, scores)?;
    let n_pos = y_true.iter().filter(|&&t| t > 0.5).count() as f64;
    if n_pos == 0.0 {
        return Err("pr_curve needs at least one positive".to_string());
    }
    let (mut tp, mut fp) = (0.0f64, 0.0f64);
    let mut pts = Vec::new();
    let mut ap = 0.0;
    let mut prev_recall = 0.0;
    let mut i = 0;
    while i < idx.len() {
        let s = scores[idx[i]];
        while i < idx.len() && scores[idx[i]] == s {
            if y_true[idx[i]] > 0.5 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            i += 1;
        }
        let recall = tp / n_pos;
        let precision = tp / (tp + fp);
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
        pts.push((recall, precision));
    }
    Ok((pts, ap))
}

/// Cumulative gains points (population fraction, capture rate) on the same
/// score-descending sweep. Lift at a point = capture / fraction.
pub fn gain_curve(y_true: &[f64], scores: &[f64]) -> Result<Vec<(f64, f64)>, String> {
    let idx = rank_desc(y_true, scores)?;
    let n_pos = y_true.iter().filter(|&&t| t > 0.5).count() as f64;
    if n_pos == 0.0 {
        return Err("gain_curve needs at least one positive".to_string());
    }
    let mut tp = 0.0f64;
    let mut pts = vec![(0.0, 0.0)];
    let mut i = 0;
    while i < idx.len() {
        let s = scores[idx[i]];
        while i < idx.len() && scores[idx[i]] == s {
            if y_true[idx[i]] > 0.5 {
                tp += 1.0;
            }
            i += 1;
        }
        pts.push((i as f64 / idx.len() as f64, tp / n_pos));
    }
    Ok(pts)
}

/// KS sweep report: per-step (fpr, tpr, score) on the score-descending sweep,
/// the KS statistic (max |tpr - fpr|) and Score@KS (the score at that step).
#[derive(Clone, Debug)]
pub struct KsReport {
    pub points: Vec<(f64, f64, f64)>,
    pub ks: f64,
    pub score_at_ks: f64,
    pub tpr_at_ks: f64,
    pub fpr_at_ks: f64,
}

/// Kolmogorov-Smirnov separation of the score distributions (kds/PerfScoreCard).
/// Positive class = y > 0.5.
pub fn ks_statistic(y_true: &[f64], scores: &[f64]) -> Result<KsReport, String> {
    let idx = rank_desc(y_true, scores)?;
    let n_pos = y_true.iter().filter(|&&t| t > 0.5).count() as f64;
    let n_neg = y_true.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return Err("ks_statistic needs both classes present".to_string());
    }
    let (mut tp, mut fp) = (0.0f64, 0.0f64);
    let mut points = Vec::new();
    let mut best = (f64::NEG_INFINITY, 0.0f64, 0.0f64, 0.0f64); // d, score, tpr, fpr
    let mut i = 0;
    while i < idx.len() {
        let s = scores[idx[i]];
        while i < idx.len() && scores[idx[i]] == s {
            if y_true[idx[i]] > 0.5 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            i += 1;
        }
        let (fpr, tpr) = (fp / n_neg, tp / n_pos);
        points.push((fpr, tpr, s));
        let d = (tpr - fpr).abs();
        if d > best.0 {
            best = (d, s, tpr, fpr);
        }
    }
    Ok(KsReport {
        points,
        ks: best.0,
        score_at_ks: best.1,
        tpr_at_ks: best.2,
        fpr_at_ks: best.3,
    })
}

/// One row of the kds-style decile profile (decile 1 = highest scores).
#[derive(Clone, Copy, Debug)]
pub struct DecileRow {
    pub decile: usize,
    pub n: usize,
    pub avg_score: f64,
    pub events: f64,
    pub event_rate: f64,
    pub cum_capture: f64,
    pub cum_event_rate: f64,
    pub lift: f64,
}

/// Decile profile (kds `decile_table`, PerfScoreCard `deciles_analysis`,
/// modelplotpy response/cumulative response at decile granularity): equal-count
/// buckets by score descending. `lift` = event_rate / overall event rate.
/// Positive class = y > 0.5.
pub fn decile_table(
    y_true: &[f64],
    scores: &[f64],
    n_buckets: usize,
) -> Result<Vec<DecileRow>, String> {
    let idx = rank_desc(y_true, scores)?;
    let n_pos = y_true.iter().filter(|&&t| t > 0.5).count() as f64;
    let rate = n_pos / y_true.len() as f64;
    if n_pos == 0.0 || rate == 0.0 {
        return Err("decile_table needs at least one positive".to_string());
    }
    let nb = n_buckets.max(1);
    let chunk = (idx.len() + nb - 1) / nb;
    let mut rows = Vec::new();
    let (mut cum_events, mut cum_n) = (0.0f64, 0usize);
    for b in 0..nb {
        let start = b * chunk;
        if start >= idx.len() {
            break;
        }
        let end = (start + chunk).min(idx.len());
        let events: f64 = idx[start..end]
            .iter()
            .filter(|&&i| y_true[i] > 0.5)
            .count() as f64;
        let avg_score: f64 =
            idx[start..end].iter().map(|&i| scores[i]).sum::<f64>() / (end - start) as f64;
        cum_events += events;
        cum_n += end - start;
        rows.push(DecileRow {
            decile: b + 1,
            n: end - start,
            avg_score,
            events,
            event_rate: events / (end - start) as f64,
            cum_capture: cum_events / n_pos,
            cum_event_rate: cum_events / cum_n as f64,
            lift: (events / (end - start) as f64) / rate,
        });
    }
    Ok(rows)
}

/// Cumulative lift points (population fraction, lift): lift = capture /
/// population fraction. Lift at 0 is undefined and reported as 0.
pub fn lift_curve(y_true: &[f64], scores: &[f64]) -> Result<Vec<(f64, f64)>, String> {
    Ok(gain_curve(y_true, scores)?
        .into_iter()
        .map(|(pop, capture)| {
            (pop, if pop > 0.0 { capture / pop } else { 0.0 })
        })
        .collect())
}

/// One bin of a stability (PSI/CSI) comparison.
#[derive(Clone, Copy, Debug)]
pub struct StabilityBin {
    pub left: f64,
    pub right: f64,
    pub reference_share: f64,
    pub current_share: f64,
    pub contribution: f64,
}

/// Population Stability Index between a reference and a current distribution
/// (scorecard practice: equal-width bins over the reference range, outer bins
/// open-ended, zero shares floored at 1e-6). CSI is the same measure applied
/// to a feature column. Returns (psi, per-bin detail).
pub fn stability_index(
    reference: &[f64],
    current: &[f64],
    n_bins: usize,
) -> Result<(f64, Vec<StabilityBin>), String> {
    if reference.is_empty() || current.is_empty() {
        return Err("reference and current must be non-empty".to_string());
    }
    let nb = n_bins.max(1) as f64;
    let lo = reference.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = reference.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let width = if hi > lo { (hi - lo) / nb } else { 1.0 };
    let bin_of = |v: f64| -> usize {
        if width > 0.0 {
            (((v - lo) / width) as usize).min(nb as usize - 1)
        } else {
            0
        }
    };
    let mut rc = vec![0usize; nb as usize];
    let mut cc = vec![0usize; nb as usize];
    for &v in reference {
        rc[bin_of(v)] += 1;
    }
    for &v in current {
        cc[bin_of(v)] += 1;
    }
    let mut psi = 0.0;
    let mut bins = Vec::new();
    for b in 0..nb as usize {
        let rs = (rc[b] as f64 / reference.len() as f64).max(1e-6);
        let cs = (cc[b] as f64 / current.len() as f64).max(1e-6);
        let contrib = (cs - rs) * (cs / rs).ln();
        psi += contrib;
        bins.push(StabilityBin {
            left: lo + b as f64 * width,
            right: lo + (b + 1) as f64 * width,
            reference_share: rs,
            current_share: cs,
            contribution: contrib,
        });
    }
    Ok((psi, bins))
}

/// One bucket of a double-lift comparison (CAS monograph): buckets are equal
/// count by scores_a descending; `actual` is the bucket's target mean,
/// `pred_a`/`pred_b` the two models' mean predictions. The additive vs ratio
/// display (loss cost vs loss ratio) is a plotting concern over these rows.
#[derive(Clone, Copy, Debug)]
pub struct DoubleLiftRow {
    pub bucket: usize,
    pub n: usize,
    pub actual: f64,
    pub pred_a: f64,
    pub pred_b: f64,
}

/// Double-lift table: actual vs model A vs model B along A's ranking.
pub fn double_lift(
    y_true: &[f64],
    scores_a: &[f64],
    scores_b: &[f64],
    n_buckets: usize,
) -> Result<Vec<DoubleLiftRow>, String> {
    if y_true.len() != scores_a.len() || y_true.len() != scores_b.len() {
        return Err("y_true, scores_a, scores_b must have the same length".to_string());
    }
    if y_true.is_empty() {
        return Err("empty input".to_string());
    }
    let mut idx: Vec<usize> = (0..y_true.len()).collect();
    idx.sort_by(|&a, &b| {
        scores_a[b]
            .partial_cmp(&scores_a[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let nb = n_buckets.max(1);
    let chunk = (idx.len() + nb - 1) / nb;
    let mut rows = Vec::new();
    for b in 0..nb {
        let start = b * chunk;
        if start >= idx.len() {
            break;
        }
        let end = (start + chunk).min(idx.len());
        let n = end - start;
        let (mut ya, mut pa, mut pb) = (0.0f64, 0.0f64, 0.0f64);
        for &i in &idx[start..end] {
            ya += y_true[i];
            pa += scores_a[i];
            pb += scores_b[i];
        }
        rows.push(DoubleLiftRow {
            bucket: b + 1,
            n,
            actual: ya / n as f64,
            pred_a: pa / n as f64,
            pred_b: pb / n as f64,
        });
    }
    Ok(rows)
}

/// Cohen's kappa with `none`, `linear`, or `quadratic` weights over the
/// sorted distinct labels (PerfScoreCard's weighted kappas).
pub fn weighted_kappa(
    y_true: &[f64],
    y_pred: &[f64],
    weights: &str,
) -> Result<f64, String> {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return Err("y_true and y_pred must have the same non-zero length".to_string());
    }
    let power = match weights {
        "none" => 0,
        "linear" => 1,
        "quadratic" => 2,
        _ => return Err("weights must be \"none\", \"linear\", or \"quadratic\"".to_string()),
    };
    let mut labels: Vec<f64> = y_true.iter().chain(y_pred.iter()).cloned().collect();
    labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    labels.dedup();
    let k = labels.len();
    let mut obs = vec![vec![0.0f64; k]; k];
    let pos = |v: f64| labels.iter().position(|l| *l == v).unwrap();
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        obs[pos(t)][pos(p)] += 1.0;
    }
    let n = y_true.len() as f64;
    let row: Vec<f64> = obs.iter().map(|r| r.iter().sum()).collect();
    let col: Vec<f64> = (0..k).map(|j| obs.iter().map(|r| r[j]).sum()).collect();
    let (mut wo, mut we) = (0.0f64, 0.0f64);
    for i in 0..k {
        for j in 0..k {
            let w = if power == 0 {
                if i == j { 0.0 } else { 1.0 }
            } else {
                ((i as f64 - j as f64).abs() / (k as f64 - 1.0)).powi(power)
            };
            wo += w * obs[i][j];
            we += w * row[i] * col[j] / n;
        }
    }
    if we == 0.0 {
        return Err("kappa undefined: no expected disagreement".to_string());
    }
    Ok(1.0 - wo / we)
}

/// Log loss of probability predictions (sklearn convention; probabilities
/// clip to (1e-15, 1 - 1e-15)). Positive class = y > 0.5.
pub fn log_loss(y_true: &[f64], probs: &[f64]) -> Result<f64, String> {
    if y_true.len() != probs.len() || y_true.is_empty() {
        return Err("y_true and probs must have the same non-zero length".to_string());
    }
    let eps = 1e-15;
    let mut total = 0.0;
    for (&t, &p) in y_true.iter().zip(probs.iter()) {
        let p = p.clamp(eps, 1.0 - eps);
        let y = if t > 0.5 { 1.0 } else { 0.0 };
        total += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
    }
    Ok(total / y_true.len() as f64)
}

/// Brier score (mean squared error of probability predictions).
/// Positive class = y > 0.5.
pub fn brier_score(y_true: &[f64], probs: &[f64]) -> Result<f64, String> {
    if y_true.len() != probs.len() || y_true.is_empty() {
        return Err("y_true and probs must have the same non-zero length".to_string());
    }
    let sse: f64 = y_true
        .iter()
        .zip(probs.iter())
        .map(|(&t, &p)| {
            let y = if t > 0.5 { 1.0 } else { 0.0 };
            (y - p) * (y - p)
        })
        .sum();
    Ok(sse / y_true.len() as f64)
}

/// Hinge loss over signed decision margins (sklearn convention: labels map to
/// -1/+1; loss = mean max(0, 1 - signed * margin)).
pub fn hinge_loss(y_true: &[f64], decision: &[f64]) -> Result<f64, String> {
    if y_true.len() != decision.len() || y_true.is_empty() {
        return Err("y_true and decision must have the same non-zero length".to_string());
    }
    let total: f64 = y_true
        .iter()
        .zip(decision.iter())
        .map(|(&t, &d)| {
            let signed = if t > 0.5 { 1.0 } else { -1.0 };
            (1.0 - signed * d).max(0.0)
        })
        .sum();
    Ok(total / y_true.len() as f64)
}

/// Single-label classification scores (binary or multiclass): accuracy,
/// balanced accuracy, macro/micro/weighted precision-recall-F1 and Jaccard,
/// Matthews correlation (multiclass form), Hamming and zero-one loss. Labels
/// are the sorted distinct union of truth and predictions.
#[derive(Clone, Debug, Default)]
pub struct ClassificationScores {
    pub n: usize,
    pub n_classes: usize,
    pub accuracy: f64,
    pub balanced_accuracy: f64,
    pub precision_macro: f64,
    pub recall_macro: f64,
    pub f1_macro: f64,
    pub precision_micro: f64,
    pub recall_micro: f64,
    pub f1_micro: f64,
    pub precision_weighted: f64,
    pub recall_weighted: f64,
    pub f1_weighted: f64,
    pub mcc: f64,
    pub jaccard_macro: f64,
    pub jaccard_micro: f64,
    pub jaccard_weighted: f64,
    pub hamming_loss: f64,
    pub zero_one_loss: f64,
}

fn prf(tp: f64, fp: f64, fn_: f64) -> (f64, f64, f64) {
    let p = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let r = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    let f = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
    (p, r, f)
}

/// Full classification scorecard over the confusion matrix.
pub fn classification_scores(
    y_true: &[f64],
    y_pred: &[f64],
) -> Result<ClassificationScores, String> {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return Err("y_true and y_pred must have the same non-zero length".to_string());
    }
    let mut labels: Vec<f64> = y_true.iter().chain(y_pred.iter()).cloned().collect();
    labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    labels.dedup();
    let k = labels.len();
    let pos = |v: f64| labels.iter().position(|l| *l == v).unwrap();
    let mut c = vec![vec![0.0f64; k]; k];
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        c[pos(t)][pos(p)] += 1.0;
    }
    let n = y_true.len() as f64;
    let row: Vec<f64> = c.iter().map(|r| r.iter().sum()).collect();
    let col: Vec<f64> = (0..k).map(|j| c.iter().map(|r| r[j]).sum()).collect();
    let trace: f64 = (0..k).map(|i| c[i][i]).sum();

    let (mut pm, mut rm, mut fm, mut jm, mut bacc) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut pw, mut rw, mut fw, mut jw) = (0.0, 0.0, 0.0, 0.0);
    let mut tp_sum = 0.0;
    for i in 0..k {
        let tp = c[i][i];
        let fp = col[i] - tp;
        let fn_ = row[i] - tp;
        tp_sum += tp;
        let (p, r, f) = prf(tp, fp, fn_);
        let jac = if tp + fp + fn_ > 0.0 {
            tp / (tp + fp + fn_)
        } else {
            0.0
        };
        pm += p;
        rm += r;
        fm += f;
        jm += jac;
        bacc += r;
        let w = row[i] / n;
        pw += w * p;
        rw += w * r;
        fw += w * f;
        jw += w * jac;
    }
    let (pm, rm, fm, jm, bacc) = (
        pm / k as f64,
        rm / k as f64,
        fm / k as f64,
        jm / k as f64,
        bacc / k as f64,
    );
    let (p_micro, r_micro, f_micro) = prf(tp_sum, n - tp_sum, n - tp_sum);
    let j_micro = if n > 0.0 {
        // micro Jaccard over all decisions of the positive class per label
        let mut inter = 0.0;
        let mut uni = 0.0;
        for i in 0..k {
            inter += c[i][i];
            uni += row[i] + col[i] - c[i][i];
        }
        if uni > 0.0 {
            inter / uni
        } else {
            0.0
        }
    } else {
        0.0
    };
    // multiclass Matthews correlation (Gorodkin form)
    let pq: f64 = (0..k).map(|i| row[i] * col[i]).sum();
    let s2 = n * n;
    let p2: f64 = row.iter().map(|v| v * v).sum();
    let q2: f64 = col.iter().map(|v| v * v).sum();
    let denom = ((s2 - p2) * (s2 - q2)).sqrt();
    let mcc = if denom > 0.0 {
        (trace * n - pq) / denom
    } else {
        0.0
    };
    let errs = n - trace;
    Ok(ClassificationScores {
        n: y_true.len(),
        n_classes: k,
        accuracy: trace / n,
        balanced_accuracy: bacc,
        precision_macro: pm,
        recall_macro: rm,
        f1_macro: fm,
        precision_micro: p_micro,
        recall_micro: r_micro,
        f1_micro: f_micro,
        precision_weighted: pw,
        recall_weighted: rw,
        f1_weighted: fw,
        mcc,
        jaccard_macro: jm,
        jaccard_micro: j_micro,
        jaccard_weighted: jw,
        hamming_loss: errs / n,
        zero_one_loss: errs / n,
    })
}

/// Multilabel scores: rows are instances, columns are labels; entries > 0.5
/// are positive. The raw entries double as ranking scores for LRAP and
/// coverage (sklearn semantics). Rows with no true labels are skipped in the
/// samples averages and ranking metrics.
#[derive(Clone, Debug, Default)]
pub struct MultilabelScores {
    pub n: usize,
    pub n_labels: usize,
    pub subset_accuracy: f64,
    pub hamming_loss: f64,
    pub precision_micro: f64,
    pub recall_micro: f64,
    pub f1_micro: f64,
    pub precision_macro: f64,
    pub recall_macro: f64,
    pub f1_macro: f64,
    pub precision_samples: f64,
    pub recall_samples: f64,
    pub f1_samples: f64,
    pub label_ranking_ap: f64,
    pub coverage_error: f64,
}

/// Multilabel scorecard (subset accuracy, hamming, micro/macro/samples
/// precision-recall-F1, label-ranking average precision, coverage error).
pub fn multilabel_scores(
    y_true: &[Vec<f64>],
    y_scores: &[Vec<f64>],
) -> Result<MultilabelScores, String> {
    if y_true.len() != y_scores.len() || y_true.is_empty() {
        return Err("y_true and y_scores must have the same non-zero length".to_string());
    }
    let l = y_true[0].len();
    if l == 0 || y_scores.iter().any(|r| r.len() != l) || y_true.iter().any(|r| r.len() != l) {
        return Err("rows must be non-empty and all the same width".to_string());
    }
    let n = y_true.len() as f64;
    let mut exact = 0usize;
    let mut mismatches = 0usize;
    let mut tp = vec![0.0f64; l];
    let mut fp = vec![0.0f64; l];
    let mut fn_ = vec![0.0f64; l];
    let (mut sp, mut sr, mut sf, mut rank_rows) = (0.0, 0.0, 0.0, 0usize);
    let (mut lrap_sum, mut cov_sum) = (0.0, 0.0);
    for (trow, srow) in y_true.iter().zip(y_scores.iter()) {
        let mut exact_row = true;
        for j in 0..l {
            let t = trow[j] > 0.5;
            let p = srow[j] > 0.5;
            match (t, p) {
                (true, true) => tp[j] += 1.0,
                (false, true) => fp[j] += 1.0,
                (true, false) => fn_[j] += 1.0,
                (false, false) => {}
            }
            if t != p {
                mismatches += 1;
                exact_row = false;
            }
        }
        if exact_row {
            exact += 1;
        }
        // samples averages + ranking metrics (skip label-free rows)
        let relevant: Vec<usize> = (0..l).filter(|&j| trow[j] > 0.5).collect();
        if !relevant.is_empty() {
            rank_rows += 1;
            let (mut tp_r, mut fp_r, mut fn_r) = (0.0f64, 0.0f64, 0.0f64);
            for j in 0..l {
                let t = trow[j] > 0.5;
                let p = srow[j] > 0.5;
                match (t, p) {
                    (true, true) => tp_r += 1.0,
                    (false, true) => fp_r += 1.0,
                    (true, false) => fn_r += 1.0,
                    (false, false) => {}
                }
            }
            let (p, r, f) = prf(tp_r, fp_r, fn_r);
            sp += p;
            sr += r;
            sf += f;
            // LRAP over the raw scores
            let mut ap_row = 0.0;
            for &j in &relevant {
                let above: Vec<usize> = (0..l).filter(|&k| srow[k] >= srow[j]).collect();
                let rel_above = above.iter().filter(|&&k| trow[k] > 0.5).count() as f64;
                ap_row += rel_above / above.len() as f64;
            }
            lrap_sum += ap_row / relevant.len() as f64;
            // coverage: top-scored labels needed to cover every true label
            let cov = relevant
                .iter()
                .map(|&j| (0..l).filter(|&k| srow[k] >= srow[j]).count())
                .max()
                .unwrap_or(0) as f64;
            cov_sum += cov;
        }
    }
    let mut mp = 0.0;
    let mut mr = 0.0;
    let mut mf = 0.0;
    for j in 0..l {
        let (p, r, f) = prf(tp[j], fp[j], fn_[j]);
        mp += p;
        mr += r;
        mf += f;
    }
    let l = f64::from(l as u32);
    let (tp_m, fp_m, fn_m) = (
        tp.iter().sum::<f64>(),
        fp.iter().sum::<f64>(),
        fn_.iter().sum::<f64>(),
    );
    let (p_mi, r_mi, f_mi) = prf(tp_m, fp_m, fn_m);
    let rr = rank_rows as f64;
    Ok(MultilabelScores {
        n: y_true.len(),
        n_labels: l as usize,
        subset_accuracy: exact as f64 / n,
        hamming_loss: mismatches as f64 / (n * l),
        precision_micro: p_mi,
        recall_micro: r_mi,
        f1_micro: f_mi,
        precision_macro: mp / l,
        recall_macro: mr / l,
        f1_macro: mf / l,
        precision_samples: if rr > 0.0 { sp / rr } else { 0.0 },
        recall_samples: if rr > 0.0 { sr / rr } else { 0.0 },
        f1_samples: if rr > 0.0 { sf / rr } else { 0.0 },
        label_ranking_ap: if rr > 0.0 { lrap_sum / rr } else { 0.0 },
        coverage_error: if rr > 0.0 { cov_sum / rr } else { 0.0 },
    })
}

/// Pairwise distances between two row sets. Metrics: `euclidean`, `manhattan`,
/// `chebyshev`, `minkowski` (power `p`), `cosine`, `correlation`, `hamming`,
/// `jaccard` (binary presence). Zero-variance rows yield correlation 1.0;
/// zero vectors yield cosine 1.0; both-empty jaccard yields 0.0.
pub fn pairwise_distances(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    metric: &str,
    p: f64,
) -> Result<Vec<Vec<f64>>, String> {
    if a.is_empty() || b.is_empty() {
        return Err("both row sets must be non-empty".to_string());
    }
    let d = a[0].len();
    if d == 0
        || a.iter().any(|r| r.len() != d)
        || b.iter().any(|r| r.len() != d)
    {
        return Err("rows must be non-empty and all the same width".to_string());
    }
    let one = |u: &[f64], v: &[f64]| -> f64 {
        match metric {
            "euclidean" => u
                .iter()
                .zip(v)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt(),
            "manhattan" => u.iter().zip(v).map(|(x, y)| (x - y).abs()).sum(),
            "chebyshev" => u
                .iter()
                .zip(v)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max),
            "minkowski" => u
                .iter()
                .zip(v)
                .map(|(x, y)| (x - y).abs().powf(p))
                .sum::<f64>()
                .powf(1.0 / p),
            "cosine" => {
                let dot: f64 = u.iter().zip(v).map(|(x, y)| x * y).sum();
                let nu: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
                let nv: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if nu > 0.0 && nv > 0.0 {
                    1.0 - dot / (nu * nv)
                } else {
                    1.0
                }
            }
            "correlation" => {
                let d = u.len() as f64;
                let mu = u.iter().sum::<f64>() / d;
                let mv = v.iter().sum::<f64>() / d;
                let (mut cu, mut cv) = (0.0f64, 0.0f64);
                let (mut su, mut sv) = (0.0f64, 0.0f64);
                for (x, y) in u.iter().zip(v) {
                    cu += (x - mu) * (y - mv);
                    su += (x - mu) * (x - mu);
                    sv += (y - mv) * (y - mv);
                }
                let den = su.sqrt() * sv.sqrt();
                if den > 0.0 {
                    1.0 - cu / den
                } else {
                    1.0
                }
            }
            "hamming" => u
                .iter()
                .zip(v)
                .filter(|(x, y)| x != y)
                .count() as f64
                / d as f64,
            "jaccard" => {
                let inter = u
                    .iter()
                    .zip(v)
                    .filter(|(x, y)| **x != 0.0 && **y != 0.0)
                    .count() as f64;
                let uni = u
                    .iter()
                    .zip(v)
                    .filter(|(x, y)| **x != 0.0 || **y != 0.0)
                    .count() as f64;
                if uni > 0.0 {
                    1.0 - inter / uni
                } else {
                    0.0
                }
            }
            other => {
                let _ = other;
                f64::NAN // unreachable: metric validated below
            }
        }
    };
    if !matches!(
        metric,
        "euclidean"
            | "manhattan"
            | "chebyshev"
            | "minkowski"
            | "cosine"
            | "correlation"
            | "hamming"
            | "jaccard"
    ) {
        return Err(format!("unknown metric: {metric}"));
    }
    Ok(a.iter()
        .map(|u| b.iter().map(|v| one(u, v)).collect())
        .collect())
}
