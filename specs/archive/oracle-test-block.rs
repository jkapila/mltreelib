// ============================ scan oracle (issue #7) ============================
// Brute-force reference for the split scan: enumerates every candidate the
// scan may consider, scores each with an INDEPENDENT weighted-least-squares
// solver (naive Gaussian elimination on explicit design rows — deliberately
// sharing no code with the implementation), and asserts the fitted tree
// chose the same split and predicts the same values. This is the
// characterization net for the incremental-factorization speed work: the
// scan's linear algebra changes; its external behavior must not.

/// Precision grid for one column — mirrors the documented handler rule:
/// integer columns scan exact unique values (no grid), float columns use
/// `split_precision`.
fn oracle_grid(dtypes: &[ColType], split_precision: Option<f64>, j: usize) -> Option<f64> {
    match dtypes.get(j) {
        Some(ColType::Integer) => None,
        _ => split_precision,
    }
}

/// The documented router: grid mode rounds to the grid before the `<=`
/// compare; raw mode compares directly.
#[allow(dead_code)]
fn oracle_goes_left(threshold: f64, v: f64, grid: Option<f64>) -> bool {
    match grid {
        Some(p) if p > 0.0 => {
            let q = 1.0 / p;
            (v * q).round() / q <= threshold
        }
        _ => v <= threshold,
    }
}

/// Design row for the scan's candidate scoring: `[1, values (0 where
/// missing), NA indicators]` — indicator slots always present, matching
/// the scan's per-candidate design.
fn oracle_scan_design(x: &[Option<f64>], chosen: &[usize]) -> Vec<f64> {
    let nc = chosen.len();
    let mut d = vec![0.0; 1 + 2 * nc];
    d[0] = 1.0;
    for (k, &c) in chosen.iter().enumerate() {
        match x[c] {
            Some(v) => d[1 + k] = v,
            None => d[1 + nc + k] = 1.0,
        }
    }
    d
}

/// Independent weighted least squares over explicit design rows: solves
/// `(X'X + l2·I*)b = X'y` by Gaussian elimination with partial pivoting
/// (intercept unpenalized; singular directions keep coefficient 0), and
/// returns the scan's score convention `max(0, yty - b'X'y)`.
fn oracle_solve(rows: &[Vec<f64>], y: &[f64], w: &[f64], dim: usize, l2: f64) -> (Vec<f64>, f64) {
    let mut xtx = vec![0.0; dim * dim];
    let mut xty = vec![0.0; dim];
    let mut yty = 0.0;
    for (i, r) in rows.iter().enumerate() {
        let (yi, wi) = (y[i], w[i]);
        yty += wi * yi * yi;
        for a in 0..dim {
            if r[a] == 0.0 {
                continue;
            }
            xty[a] += wi * r[a] * yi;
            for b in 0..dim {
                xtx[a * dim + b] += wi * r[a] * r[b];
            }
        }
    }
    for i in 1..dim {
        xtx[i * dim + i] += l2;
    }
    let mut a = xtx;
    let mut b = xty.clone();
    for col in 0..dim {
        let mut piv = col;
        for r in col + 1..dim {
            if a[r * dim + col].abs() > a[piv * dim + col].abs() {
                piv = r;
            }
        }
        if a[piv * dim + col].abs() < 1e-12 {
            continue; // singular direction: coefficient stays 0
        }
        if piv != col {
            for c in 0..dim {
                a.swap(col * dim + c, piv * dim + c);
            }
            b.swap(col, piv);
        }
        let diag = a[col * dim + col];
        for r in col + 1..dim {
            let f = a[r * dim + col] / diag;
            if f != 0.0 {
                for c in col..dim {
                    a[r * dim + c] -= f * a[col * dim + c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    for col in (0..dim).rev() {
        let mut s = b[col];
        for c in col + 1..dim {
            s -= a[col * dim + c] * b[c];
        }
        let d = a[col * dim + col];
        b[col] = if d.abs() < 1e-12 { 0.0 } else { s / d };
    }
    let explained: f64 = xty.iter().zip(&b).map(|(t, s)| t * s).sum();
    (b, (yty - explained).max(0.0))
}

/// Independent elastic-net fit by weighted coordinate descent directly on
/// the design matrix (the implementation descends on the standardized Gram;
/// this descends on standardized rows — same strictly-convex objective for
/// l2 > 0, hence the same unique optimum). Coefficients come back
/// unstandardized, intercept free, same score convention as `oracle_solve`.
fn oracle_elastic(
    rows: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    dim: usize,
    l1: f64,
    l2: f64,
) -> (Vec<f64>, f64) {
    let n = rows.len();
    let sw: f64 = w.iter().sum::<f64>().max(1e-300);
    let mut mean = vec![0.0; dim];
    let mut scale = vec![1.0; dim];
    for j in 1..dim {
        mean[j] = rows.iter().zip(w).map(|(r, wi)| wi * r[j]).sum::<f64>() / sw;
        let var: f64 = rows
            .iter()
            .zip(w)
            .map(|(r, wi)| wi * (r[j] - mean[j]) * (r[j] - mean[j]))
            .sum();
        scale[j] = if var > 1e-12 { var.sqrt() } else { 1.0 };
    }
    let z: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| {
            let mut zr = vec![0.0; dim];
            zr[0] = 1.0;
            for j in 1..dim {
                zr[j] = (r[j] - mean[j]) / scale[j];
            }
            zr
        })
        .collect();
    let mut bstd = vec![0.0; dim];
    let mut yhat = vec![0.0; n];
    for _ in 0..500 {
        let mut delta = 0.0f64;
        for j in 1..dim {
            let mut partial = 0.0;
            let mut gjj = 0.0;
            for i in 0..n {
                let zi = z[i][j];
                gjj += w[i] * zi * zi;
                partial += w[i] * zi * (y[i] - yhat[i] + zi * bstd[j]);
            }
            if gjj <= 0.0 {
                continue;
            }
            let raw = partial / (gjj + l2);
            let shrunk = raw.signum() * (raw.abs() - l1 / (gjj + l2)).max(0.0);
            let d = shrunk - bstd[j];
            if d != 0.0 {
                for i in 0..n {
                    yhat[i] += z[i][j] * d;
                }
                delta = delta.max(d.abs());
            }
            bstd[j] = shrunk;
        }
        if delta < 1e-12 {
            break;
        }
    }
    let mut coef = vec![0.0; dim];
    let sum_y: f64 = y.iter().zip(w).map(|(yi, wi)| wi * yi).sum();
    coef[0] = sum_y / sw;
    for j in 1..dim {
        coef[j] = bstd[j] / scale[j];
        coef[0] -= mean[j] * coef[j];
    }
    let mut xty = vec![0.0; dim];
    let mut yty = 0.0;
    for (i, r) in rows.iter().enumerate() {
        yty += w[i] * y[i] * y[i];
        for j in 0..dim {
            xty[j] += w[i] * r[j] * y[i];
        }
    }
    let explained: f64 = xty.iter().zip(&coef).map(|(t, s)| t * s).sum();
    (coef, (yty - explained).max(0.0))
}

/// Weighted (w, wy, wy2) of one side — the categorical scan's y-slice
/// gain convention.
fn oracle_side_var(ys: &[(f64, f64)]) -> (f64, f64, f64) {
    let ws: f64 = ys.iter().map(|&(_, w)| w).sum();
    let wy: f64 = ys.iter().map(|&(y, w)| w * y).sum();
    let wy2: f64 = ys.iter().map(|&(y, w)| w * y * y).sum();
    (ws, wy, wy2)
}

/// Brute-force best split over all numeric columns, mirroring the scan's
/// candidate rules (group boundaries, `min_child` gates, both
/// `missing_left` placements, the `beats` strict-plus-epsilon comparison
/// and the documented deterministic tie-break: lowest column, lowest
/// threshold, missing_left first). Returns
/// (col, threshold, missing_left, gain, left_rows, right_rows).
#[allow(clippy::type_complexity)]
fn oracle_best_split(
    x: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    dtypes: &[ColType],
    split_precision: Option<f64>,
    min_rows: usize,
    minprob: f64,
    l1: f64,
    l2: f64,
) -> (usize, f64, bool, f64, Vec<usize>, Vec<usize>) {
    let n = y.len();
    let m = min_rows.max(1).max((minprob * n as f64).ceil() as usize);
    let chosen: Vec<usize> = (0..x[0].len()).collect();
    let dim = 1 + 2 * chosen.len();

    let beats = |gain: f64, best: f64| gain > best * (1.0 + 1e-12);

    // parent score: all node rows in the scan's design
    let all_rows: Vec<Vec<f64>> = x.iter().map(|r| oracle_scan_design(r, &chosen)).collect();
    let parent_sse = if l1 > 0.0 {
        oracle_elastic(&all_rows, y, w, dim, l1, l2).1
    } else {
        oracle_solve(&all_rows, y, w, dim, l2).1
    };

    let mut best: Option<(f64, usize, f64, bool, Vec<usize>, Vec<usize>)> = None;
    for j in 0..chosen.len() {
        let grid = oracle_grid(dtypes, split_precision, j);
        // sort non-missing rows by (rounded) value; missing rows aside
        let mut pairs: Vec<(f64, usize)> = x
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                r[j].map(|v| {
                    let vv = match grid {
                        Some(p) if p > 0.0 => {
                            let q = 1.0 / p;
                            (v * q).round() / q
                        }
                        _ => v,
                    };
                    (vv, i)
                })
            })
            .collect();
        if pairs.len() < 2 {
            continue;
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let miss: Vec<usize> = (0..n).filter(|&i| x[i][j].is_none()).collect();

        // group boundaries: equal (rounded) values stay together
        let mut gi = 0;
        while gi < pairs.len() {
            let v = pairs[gi].0;
            let mut gj = gi + 1;
            while gj < pairs.len() && pairs[gj].0 == v {
                gj += 1;
            }
            // candidate cut after group gi..gj (needs a group beyond)
            if gj < pairs.len() {
                let threshold = match grid {
                    Some(_) => v,
                    None => (v + pairs[gj].0) / 2.0,
                };
                let left_core: Vec<usize> = pairs[..gj].iter().map(|&(_, i)| i).collect();
                let right_core: Vec<usize> = pairs[gj..].iter().map(|&(_, i)| i).collect();
                if left_core.len() >= m && right_core.len() >= m {
                    for missing_left in [true, false] {
                        let mut left = left_core.clone();
                        let mut right = right_core.clone();
                        if missing_left {
                            left.extend_from_slice(&miss);
                        } else {
                            right.extend_from_slice(&miss);
                        }
                        if left.len() < m || right.len() < m {
                            continue;
                        }
                        let ls: Vec<Vec<f64>> =
                            left.iter().map(|&i| oracle_scan_design(x[i], &chosen)).collect();
                        let rs: Vec<Vec<f64>> =
                            right.iter().map(|&i| oracle_scan_design(x[i], &chosen)).collect();
                        let ly: Vec<f64> = left.iter().map(|&i| y[i]).collect();
                        let ry: Vec<f64> = right.iter().map(|&i| y[i]).collect();
                        let lw: Vec<f64> = left.iter().map(|&i| w[i]).collect();
                        let rw: Vec<f64> = right.iter().map(|&i| w[i]).collect();
                        let (sl, sr) = if l1 > 0.0 {
                            (
                                oracle_elastic(&ls, &ly, &lw, dim, l1, l2).1,
                                oracle_elastic(&rs, &ry, &rw, dim, l1, l2).1,
                            )
                        } else {
                            (
                                oracle_solve(&ls, &ly, &lw, dim, l2).1,
                                oracle_solve(&rs, &ry, &rw, dim, l2).1,
                            )
                        };
                        let gain = parent_sse - (sl + sr);
                        if gain <= 1e-9 {
                            continue;
                        }
                        let better = match &best {
                            None => true,
                            Some((g, bj, bt, ..)) => {
                                beats(gain, *g)
                                    || (!beats(*g, gain) && (j, threshold) < (*bj, *bt))
                            }
                        };
                        if better {
                            best = Some((gain, j, threshold, missing_left, left, right));
                        }
                    }
                }
            }
            gi = gj;
        }
    }
    let (gain, j, threshold, missing_left, left, right) =
        best.expect("oracle found no admissible split");
    (j, threshold, missing_left, gain, left, right)
}

/// Per-side leaf prediction, mirroring the leaf design rule (indicator
/// columns only when the side's rows carry missing values in chosen
/// columns).
fn oracle_leaf_predict(
    side_rows: &[usize],
    x: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    chosen: &[usize],
    l1: f64,
    l2: f64,
    probe: &[Option<f64>],
) -> f64 {
    let any_missing = side_rows
        .iter()
        .any(|&i| chosen.iter().any(|&c| x[i][c].is_none()));
    let nc = chosen.len();
    let dim = 1 + nc + if any_missing { nc } else { 0 };
    let design = |r: &[Option<f64>]| {
        let mut d = vec![0.0; dim];
        d[0] = 1.0;
        for (k, &c) in chosen.iter().enumerate() {
            match r[c] {
                Some(v) => d[1 + k] = v,
                None => {
                    if any_missing {
                        d[1 + nc + k] = 1.0;
                    }
                }
            }
        }
        d
    };
    let rows: Vec<Vec<f64>> = side_rows.iter().map(|&i| design(x[i])).collect();
    let sy: Vec<f64> = side_rows.iter().map(|&i| y[i]).collect();
    let sw: Vec<f64> = side_rows.iter().map(|&i| w[i]).collect();
    let (coef, _) = if l1 > 0.0 {
        oracle_elastic(&rows, &sy, &sw, dim, l1, l2)
    } else {
        oracle_solve(&rows, &sy, &sw, dim, l2)
    };
    let d = design(probe);
    coef.iter().zip(&d).map(|(c, v)| c * v).sum()
}

fn oracle_close(a: f64, b: f64, tol: f64, what: &str) {
    let scale = a.abs().max(b.abs()).max(1.0);
    assert!((a - b).abs() <= tol * scale, "{what}: {a} vs {b} (tol {tol})");
}

#[test]
fn oracle_agrees_on_random_signal() {
    // Two numeric columns, one carrying the signal; exact-midpoint cuts,
    // plain OLS leaves. Asserts split identity, gain, and predictions.
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();
    for i in 0..240 {
        let a = ((i * 37) % 240) as f64 / 240.0;
        let b = ((i * 91) % 240) as f64 / 240.0;
        let noise = ((i * 13) % 17) as f64 / 170.0;
        x.push(vec![Some(a), Some(b)]);
        y.push(2.0 + 6.0 * (a - 0.5).abs() + noise);
        w.push(1.0);
    }
    let dtypes = vec![ColType::Float, ColType::Float];
    let (col, thr, ml, gain, left, right) =
        oracle_best_split(&x, &y, &w, &dtypes, None, 5, 0.0, 0.0, 0.0);

    let p = ModelTreeParams {
        max_depth: 1,
        min_rows: 5,
        minprob: 0.0,
        l1: 0.0,
        l2: 0.0,
        split_precision: None,
        prune: PruneCost::None,
        ..Default::default()
    };
    let yopt: Vec<Option<f64>> = y.iter().map(|&v| Some(v)).collect();
    let tree = ModelTree::fit(&x, &yopt, p).unwrap();
    let splits = tree.splits();
    assert_eq!(splits.len(), 1);
    let s = &splits[0];
    assert_eq!(s.col, col, "split column");
    oracle_close(s.threshold, thr, 1e-12, "threshold");
    assert_eq!(s.missing_left, ml, "missing_left");
    oracle_close(s.gain, gain, 1e-9, "gain");

    let chosen: Vec<usize> = (0..2).collect();
    let preds = tree.predict(&x);
    for i in 0..x.len() {
        let side = if left.contains(&i) { &left } else { &right };
        assert!(side.contains(&i));
        let expect = oracle_leaf_predict(side, &x, &y, &w, &chosen, 0.0, 0.0, &x[i]);
        oracle_close(preds[i], expect, 1e-9, "prediction");
    }
}

#[test]
fn oracle_agrees_with_missing_weights_and_grid() {
    // NA-heavy + weighted rows + split_precision grid on one column and an
    // integer handler on the other. Exercises the rounded-cut semantics and
    // both missing_left placements.
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();
    for i in 0..300 {
        let a = ((i * 53) % 300) as f64 / 97.0;
        let b = ((i * 7) % 13) as f64;
        let noise = ((i * 29) % 23) as f64 / 230.0;
        let av = if i % 5 == 0 { None } else { Some(a) };
        x.push(vec![av, Some(b)]);
        y.push(1.0 + 4.0 * b / 13.0 + noise);
        w.push(0.5 + ((i * 11) % 7) as f64 / 7.0);
    }
    let dtypes = vec![ColType::Float, ColType::Integer];
    let (col, thr, ml, gain, left, right) =
        oracle_best_split(&x, &y, &w, &dtypes, Some(0.25), 8, 0.0, 0.0, 0.0);

    let p = ModelTreeParams {
        max_depth: 1,
        min_rows: 8,
        minprob: 0.0,
        l1: 0.0,
        l2: 0.0,
        split_precision: Some(0.25),
        dtypes: dtypes.clone(),
        prune: PruneCost::None,
        ..Default::default()
    };
    let yopt: Vec<Option<f64>> = y.iter().map(|&v| Some(v)).collect();
    let tree = ModelTree::fit(&x, &yopt, p).unwrap();
    let splits = tree.splits();
    assert_eq!(splits.len(), 1);
    let s = &splits[0];
    assert_eq!(s.col, col, "split column");
    oracle_close(s.threshold, thr, 1e-12, "threshold");
    assert_eq!(s.missing_left, ml, "missing_left");
    oracle_close(s.gain, gain, 1e-9, "gain");

    // prediction parity including missing rows
    let chosen: Vec<usize> = (0..2).collect();
    let preds = tree.predict(&x);
    for i in 0..x.len() {
        let side = if left.contains(&i) { &left } else { &right };
        let expect = oracle_leaf_predict(side, &x, &y, &w, &chosen, 0.0, 0.0, &x[i]);
        oracle_close(preds[i], expect, 1e-9, "prediction");
    }
}

#[test]
fn oracle_agrees_on_collinear_ridge() {
    // Collinear columns with ridge (l2 > 0): the solution is unique, so the
    // brute-force and implementation solves must agree closely.
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();
    for i in 0..180 {
        let a = ((i * 31) % 180) as f64 / 180.0;
        let c = 2.0 * a + 0.1;
        let noise = ((i * 7) % 11) as f64 / 220.0;
        x.push(vec![Some(a), Some(c)]);
        y.push(3.0 * a - 1.5 * c + noise);
        w.push(1.0);
    }
    let dtypes = vec![ColType::Float, ColType::Float];
    let (col, thr, ml, gain, ..) =
        oracle_best_split(&x, &y, &w, &dtypes, None, 5, 0.02, 0.0, 0.5);

    let p = ModelTreeParams {
        max_depth: 1,
        min_rows: 5,
        minprob: 0.02,
        l1: 0.0,
        l2: 0.5,
        split_precision: None,
        prune: PruneCost::None,
        ..Default::default()
    };
    let yopt: Vec<Option<f64>> = y.iter().map(|&v| Some(v)).collect();
    let tree = ModelTree::fit(&x, &yopt, p).unwrap();
    let splits = tree.splits();
    assert_eq!(splits.len(), 1);
    assert_eq!(splits[0].col, col, "split column");
    oracle_close(splits[0].threshold, thr, 1e-12, "threshold");
    assert_eq!(splits[0].missing_left, ml, "missing_left");
    oracle_close(splits[0].gain, gain, 1e-9, "gain");
}

#[test]
fn oracle_agrees_with_elastic_net() {
    // l1 > 0 path (coordinate descent, warm-started in the implementation):
    // same strictly-convex optimum as the brute-force descent, so the split
    // and the predictions must agree to solver tolerance.
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();
    for i in 0..200 {
        let a = ((i * 41) % 200) as f64 / 200.0;
        let b = ((i * 17) % 200) as f64 / 200.0;
        let noise = ((i * 3) % 19) as f64 / 380.0;
        x.push(vec![Some(a), Some(b)]);
        y.push(1.0 + 5.0 * (a - 0.3).abs() + 2.0 * b + noise);
        w.push(1.0);
    }
    let dtypes = vec![ColType::Float, ColType::Float];
    let (col, thr, ml, gain, left, right) =
        oracle_best_split(&x, &y, &w, &dtypes, None, 5, 0.0, 0.3, 0.5);

    let p = ModelTreeParams {
        max_depth: 1,
        min_rows: 5,
        minprob: 0.0,
        l1: 0.3,
        l2: 0.5,
        split_precision: None,
        prune: PruneCost::None,
        ..Default::default()
    };
    let yopt: Vec<Option<f64>> = y.iter().map(|&v| Some(v)).collect();
    let tree = ModelTree::fit(&x, &yopt, p).unwrap();
    let splits = tree.splits();
    assert_eq!(splits.len(), 1);
    assert_eq!(splits[0].col, col, "split column");
    oracle_close(splits[0].threshold, thr, 1e-12, "threshold");
    assert_eq!(splits[0].missing_left, ml, "missing_left");
    oracle_close(splits[0].gain, gain, 1e-7, "gain (elastic solver tolerance)");

    let chosen: Vec<usize> = (0..2).collect();
    let preds = tree.predict(&x);
    for i in 0..x.len() {
        let side = if left.contains(&i) { &left } else { &right };
        let expect = oracle_leaf_predict(side, &x, &y, &w, &chosen, 0.3, 0.5, &x[i]);
        oracle_close(preds[i], expect, 1e-6, "prediction (elastic solver tolerance)");
    }
}

#[test]
fn oracle_agrees_on_categorical_split() {
    // Two-level categorical column against a weak numeric column: the level
    // partition must win, and its y-slice gain must match the brute force
    // (mirror-invariant: leaves are fitted per side).
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();
    for i in 0..120 {
        let lvl = if i % 4 == 0 { 0.0 } else { 1.0 };
        let b = ((i * 23) % 120) as f64 / 120.0;
        let noise = ((i * 5) % 13) as f64 / 130.0;
        x.push(vec![Some(lvl), Some(b)]);
        y.push(if lvl == 0.0 { 1.0 } else { 7.0 } + noise);
        w.push(1.0);
    }
    let dtypes = vec![ColType::Categorical, ColType::Float];

    // brute force over the level partition (the only binary bipartition of
    // two levels), scored in the categorical scan's y-slice convention
    let mut left_ys: Vec<(f64, f64)> = Vec::new();
    let mut right_ys: Vec<(f64, f64)> = Vec::new();
    for i in 0..120 {
        if x[i][0] == Some(0.0) {
            left_ys.push((y[i], w[i]));
        } else {
            right_ys.push((y[i], w[i]));
        }
    }
    let (wl, wyl, _) = oracle_side_var(&left_ys);
    let (wr, wyr, _) = oracle_side_var(&right_ys);
    let mut all: Vec<(f64, f64)> = Vec::new();
    all.extend_from_slice(&left_ys);
    all.extend_from_slice(&right_ys);
    let (wt, wyt, _) = oracle_side_var(&all);
    let oracle_gain = wyl * wyl / wl + wyr * wyr / wr - wyt * wyt / wt;

    let p = ModelTreeParams {
        max_depth: 1,
        min_rows: 5,
        minprob: 0.0,
        l1: 0.0,
        l2: 0.0,
        split_precision: None,
        dtypes: dtypes.clone(),
        prune: PruneCost::None,
        ..Default::default()
    };
    let yopt: Vec<Option<f64>> = y.iter().map(|&v| Some(v)).collect();
    let tree = ModelTree::fit(&x, &yopt, p).unwrap();
    let splits = tree.splits();
    assert_eq!(splits.len(), 1);
    assert_eq!(splits[0].col, 0, "categorical column must win");
    assert!(splits[0].cat.is_some(), "categorical partition handle");
    oracle_close(splits[0].gain, oracle_gain, 1e-9, "categorical gain");
}
