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

/// Preselection strategy family — which columns enter the split search.
/// Selection only (feature engineering like EFB bundling belongs to
/// [`crate::transform`]). Extensible like `gain` / `leaf`: `winnow`
/// (univariate relevance), `boruta` (shadow comparison), …
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Preselect {
    /// No screening (default).
    None,
    /// Winnowing: keep columns whose |weighted correlation with y| reaches
    /// `min_relevance` (C5.0 winnowing).
    Winnow,
    /// Boruta-style all-relevant: keep columns beating their own permuted
    /// shadow copies (relevance vs the max shadow relevance).
    Boruta,
    /// Boruta done properly: `n_shadows` permuted copies per column; keep
    /// columns beating the strongest shadow.
    ShadowMax,
    /// Drop zero / near-constant columns (weighted variance floor).
    Variance,
    /// Normalized mutual information (symmetric uncertainty) — catches
    /// non-monotone relevance that correlation misses.
    MutualInfo,
    /// Univariate surrogate trees: grow a small single-column tree per
    /// variable and keep those whose explained variation (R²) reaches
    /// `min_relevance`. The per-variable "how well does it explain
    /// behaviour" score.
    UniTree,
}

/// Boruta-style shadow comparison: a column is kept when its relevance
/// beats every permuted shadow copy of it (seeded permutation). Cheap
/// "all-relevant" flavor of Boruta (no RF importance — the winnow
/// relevance is the statistic).
pub fn boruta(
    cols: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    seed: u64,
    min_relevance: f64,
) -> Vec<usize> {
    let rel = relevance(cols, y, w);
    let n = cols.first().map(|c| c.len()).unwrap_or(0);
    let mut kept = Vec::new();
    for (j, col) in cols.iter().enumerate() {
        if rel[j] <= 0.0 || rel[j] < min_relevance {
            continue;
        }
        // one permuted shadow per column (seeded, deterministic)
        let mut shadow = col.clone();
        let mut rng = seed ^ (j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for i in (1..n).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let k = (rng as usize) % (i + 1);
            shadow.swap(i, k);
        }
        let srel = relevance(std::slice::from_ref(&shadow), y, w)[0];
        if rel[j] > srel {
            kept.push(j);
        }
    }
    kept
}

/// Exclusive feature bundling: greedily merge columns that are never
/// jointly nonzero into one column per bundle, each member shifted into a
/// disjoint value range (one scan column covers the bundle). Returns the
/// bundled column-major matrix and the map original column j ->
/// (bundle index, offset) so predictions can rebuild the bundles.
pub fn efb_bundle(
    cols: &[Vec<Option<f64>>],
) -> (Vec<Vec<Option<f64>>>, Vec<(usize, f64)>) {
    let n = cols.first().map(|c| c.len()).unwrap_or(0);
    let eps = 1e-12;
    // (member columns, per-member offset) per bundle
    let mut bundles: Vec<(Vec<usize>, Vec<f64>)> = Vec::new();
    let mut map: Vec<(usize, f64)> = vec![(0, 0.0); cols.len()];
    for (j, col) in cols.iter().enumerate() {
        let (lo, hi) = col
            .iter()
            .flatten()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        let (lo, hi) = if lo.is_finite() { (lo, hi) } else { (0.0, 0.0) };
        let mut placed = false;
        for (bi, (members, offsets)) in bundles.iter_mut().enumerate() {
            let exclusive = members.iter().all(|&m| {
                (0..n).all(|r| {
                    let a = cols[m][r].unwrap_or(0.0).abs();
                    let b = col[r].unwrap_or(0.0).abs();
                    a <= eps || b <= eps
                })
            });
            if exclusive {
                // shift past every member's shifted range
                let off = members
                    .iter()
                    .zip(offsets.iter())
                    .map(|(&m, &o)| {
                        let (a, b) = cols[m]
                            .iter()
                            .flatten()
                            .fold((f64::INFINITY, f64::NEG_INFINITY), |(x, y), &v| {
                                (x.min(v + o), y.max(v + o))
                            });
                        b.max(a)
                    })
                    .fold(hi, f64::max)
                    + 1.0;
                map[j] = (bi, off);
                members.push(j);
                offsets.push(off);
                placed = true;
                break;
            }
        }
        if !placed {
            map[j] = (bundles.len(), 0.0);
            bundles.push((vec![j], vec![0.0]));
        }
        let _ = lo;
    }
    // materialize bundled columns: sum of the (at most one) nonzero member
    let mut out: Vec<Vec<Option<f64>>> = vec![Vec::with_capacity(n); bundles.len()];
    for r in 0..n {
        for (bi, (members, offsets)) in bundles.iter().enumerate() {
            let mut acc = 0.0;
            let mut any = false;
            for (&m, &off) in members.iter().zip(offsets.iter()) {
                if let Some(v) = cols[m][r] {
                    any = true;
                    if v.abs() > eps {
                        acc = v + off;
                    }
                }
            }
            out[bi].push(if any { Some(acc) } else { None });
        }
    }
    (out, map)
}

/// Apply a stored bundle map to row-major input (predict-side rebundling).
pub fn efb_apply(
    x: &[Vec<Option<f64>>],
    map: &[(usize, f64)],
    n_bundles: usize,
) -> Vec<Vec<Option<f64>>> {
    let eps = 1e-12;
    x.iter()
        .map(|row| {
            let mut b: Vec<Option<f64>> = vec![None; n_bundles];
            let mut any: Vec<bool> = vec![false; n_bundles];
            for (j, v) in row.iter().enumerate() {
                let (bi, off) = map[j];
                if let Some(v) = v {
                    any[bi] = true;
                    if v.abs() > eps {
                        b[bi] = Some(v + off);
                    } else if b[bi].is_none() {
                        b[bi] = Some(0.0);
                    }
                }
            }
            for bi in 0..n_bundles {
                if !any[bi] {
                    b[bi] = None;
                }
            }
            b
        })
        .collect()
}

/// Zero / near-constant filter: keep columns whose weighted variance
/// exceeds `min_relevance` (as an absolute floor; 0.0 drops only exact
/// constants).
pub fn variance(
    cols: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    min_variance: f64,
) -> Vec<usize> {
    let _ = y;
    let mut kept = Vec::new();
    for (j, col) in cols.iter().enumerate() {
        let (mut sw, mut sx, mut sxx) = (0.0f64, 0.0f64, 0.0f64);
        for (r, v) in col.iter().enumerate() {
            if let Some(x) = v {
                sw += w[r];
                sx += w[r] * x;
                sxx += w[r] * x * x;
            }
        }
        if sw > 0.0 {
            let var = (sxx - sx * sx / sw) / sw;
            if var > min_variance {
                kept.push(j);
            }
        }
    }
    kept
}

/// Normalized mutual information (symmetric uncertainty): bin both
/// variables by quantiles and report MI / sqrt(Hx*Hy) in [0, 1] — a
/// threshold-compatible, non-monotone relevance.
pub fn mutual_info(
    cols: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    min_mi: f64,
) -> Vec<usize> {
    let n = y.len();
    let bins = 10.min(n / 20).max(2);
    // y quantile bins (weighted-free, by rank)
    let mut y_sorted: Vec<(f64, usize)> =
        y.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    y_sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut y_bin = vec![0usize; n];
    for (rank, &(_, i)) in y_sorted.iter().enumerate() {
        y_bin[i] = rank * bins / n.max(1);
    }
    let mut kept = Vec::new();
    for (j, col) in cols.iter().enumerate() {
        // x quantile bins over observed rows
        let mut x_sorted: Vec<(f64, usize)> = col
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|v| (v, i)))
            .collect();
        let nx = x_sorted.len();
        if nx < 2 * bins {
            continue;
        }
        x_sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut x_bin = vec![usize::MAX; n];
        for (rank, &(_, i)) in x_sorted.iter().enumerate() {
            x_bin[i] = rank * bins / nx.max(1);
        }
        // joint weighted counts
        let mut joint = vec![vec![0.0f64; bins]; bins];
        let (mut px, mut py) = (vec![0.0f64; bins], vec![0.0f64; bins]);
        let mut tot = 0.0f64;
        for i in 0..n {
            if x_bin[i] != usize::MAX {
                let wt = w[i];
                joint[x_bin[i]][y_bin[i]] += wt;
                px[x_bin[i]] += wt;
                py[y_bin[i]] += wt;
                tot += wt;
            }
        }
        if tot <= 0.0 {
            continue;
        }
        let mut mi = 0.0f64;
        for a in 0..bins {
            for b in 0..bins {
                let pxy = joint[a][b] / tot;
                if pxy > 0.0 {
                    mi += pxy * (pxy / ((px[a] / tot) * (py[b] / tot))).ln();
                }
            }
        }
        let hx = -px
            .iter()
            .map(|&p| if p > 0.0 { (p / tot) * (p / tot).ln() } else { 0.0 })
            .sum::<f64>();
        let hy = -py
            .iter()
            .map(|&p| if p > 0.0 { (p / tot) * (p / tot).ln() } else { 0.0 })
            .sum::<f64>();
        let norm = (hx * hy).sqrt();
        let su = if norm > 0.0 { mi / norm } else { 0.0 };
        if su >= min_mi {
            kept.push(j);
        }
    }
    kept
}

/// Boruta done properly: `n_shadows` permuted copies per column; the
/// column must beat the strongest shadow.
pub fn shadowmax(
    cols: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    seed: u64,
    n_shadows: usize,
    min_relevance: f64,
) -> Vec<usize> {
    let rel = relevance(cols, y, w);
    let n = cols.first().map(|c| c.len()).unwrap_or(0);
    let mut kept = Vec::new();
    for (j, col) in cols.iter().enumerate() {
        if rel[j] <= 0.0 || rel[j] < min_relevance {
            continue;
        }
        let mut best_shadow = 0.0f64;
        for s in 0..n_shadows.max(1) {
            let mut shadow = col.clone();
            let mut rng = seed
                ^ (j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (s as u64).wrapping_mul(0xC2B2_AE35_57F7_C877);
            for i in (1..n).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let k = (rng as usize) % (i + 1);
                shadow.swap(i, k);
            }
            let sr = relevance(std::slice::from_ref(&shadow), y, w)[0];
            best_shadow = best_shadow.max(sr);
        }
        if rel[j] > best_shadow {
            kept.push(j);
        }
    }
    kept
}

/// Univariate surrogate-tree relevance: grow a small single-column tree
/// per variable (the user's "surrogate tree by each variable") and score
/// the variation it explains, R² = 1 - SSE/SST. A column that explains
/// behaviour through any split shape — not just monotone correlation —
/// scores high. `min_r2` screens; returns (kept indices, R² per column).
pub fn uni_tree_r2(
    cols: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    min_r2: f64,
    seed: u64,
) -> (Vec<usize>, Vec<f64>) {
    use crate::leaf::LeafKind;
    use crate::model_tree::{ModelTree, ModelTreeParams};
    let n = y.len();
    let (mut sw, mut sy) = (0.0f64, 0.0f64);
    for i in 0..n {
        sw += w[i];
        sy += w[i] * y[i];
    }
    let mean = sy / sw.max(1e-300);
    let sst: f64 = (0..n).map(|i| w[i] * (y[i] - mean) * (y[i] - mean)).sum();
    let mut kept = Vec::new();
    let mut out = Vec::with_capacity(cols.len());
    for (j, col) in cols.iter().enumerate() {
        let x: Vec<Vec<Option<f64>>> = col.iter().map(|&v| vec![v]).collect();
        let yv: Vec<Option<f64>> = y.iter().copied().map(Some).collect();
        let params = ModelTreeParams {
            max_depth: 2,
            min_rows: 5,
            leaf: LeafKind::Mean,
            l2: 0.0,
            split_precision: None,
            seed,
            ..Default::default()
        };
        let r2 = match ModelTree::fit(&x, &yv, params) {
            Ok(tree) => {
                let pred = tree.predict(&x);
                let sse: f64 = (0..n)
                    .map(|i| w[i] * (y[i] - pred[i]) * (y[i] - pred[i]))
                    .sum();
                if sst > 0.0 { 1.0 - sse / sst } else { 0.0 }
            }
            Err(_) => 0.0,
        };
        out.push(r2);
        if r2 >= min_r2 {
            kept.push(j);
        }
    }
    (kept, out)
}
