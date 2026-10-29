//! Column pre-selection — the "winnowing" layer: cheap per-column screens
//! that shrink the split search *before* it starts. Simple, boring, and
//! effective (C5.0's winnowing; the preselect catalog in
//! `references/research/preselect-preprocessing-catalog.md`).

/// Winnowing-style relevance: |weighted correlation with the target| per
/// column, computed over rows where the column is observed. 0.0 for
/// constant columns (they carry no split information anyway).
pub fn relevance(cols: &[Vec<Option<f64>>], y: &[f64], w: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(cols.len());
    for col in cols {
        let (mut n, mut sw) = (0usize, 0.0f64);
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (r, v) in col.iter().enumerate() {
            if let Some(x) = v {
                n += 1;
                let (x, y, wt) = (*x, y[r], w[r]);
                sw += wt;
                sx += wt * x;
                sy += wt * y;
                sxx += wt * x * x;
                syy += wt * y * y;
                sxy += wt * x * y;
            }
        }
        if n < 2 || sw <= 0.0 {
            out.push(0.0);
            continue;
        }
        let cov = sxy - sx * sy / sw;
        let vx = sxx - sx * sx / sw;
        let vy = syy - sy * sy / sw;
        let denom = (vx * vy).sqrt();
        out.push(if denom > 0.0 { (cov / denom).abs() } else { 0.0 });
    }
    out
}

/// Winnowing: screen columns by univariate relevance against the target and
/// keep those at or above `min_relevance` (the cheap pre-split screening
/// LightGBM's docs call feature pre-filtering / winnowing). Row-major `x`.
/// Returns (kept ORIGINAL column indices, relevance score per column).
pub fn winnow(
    x: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    min_relevance: f64,
) -> (Vec<usize>, Vec<f64>) {
    let d = x.first().map(|r| r.len()).unwrap_or(0);
    let mut cols: Vec<Vec<Option<f64>>> = vec![Vec::with_capacity(x.len()); d];
    for row in x {
        for (j, v) in row.iter().enumerate() {
            cols[j].push(*v);
        }
    }
    let rel = relevance(&cols, y, w);
    let kept = rel
        .iter()
        .enumerate()
        .filter(|(_, &r)| r >= min_relevance)
        .map(|(j, _)| j)
        .collect();
    (kept, rel)
}

/// Keep columns with relevance >= `min_relevance` (and non-constant). Returns
/// original column indices, ascending.
pub fn select_columns(
    cols: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    min_relevance: f64,
) -> Vec<usize> {
    let rel = relevance(cols, y, w);
    rel.iter()
        .enumerate()
        .filter(|(_, &r)| r >= min_relevance)
        .map(|(j, _)| j)
        .collect()
}
