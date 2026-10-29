//! Reproduction of R's `rpart` on the canonical mtcars regression task —
//! the "is the tree even built right?" sanity check.
//!
//! Ground truth: live rpart 4.1.23 via `examples/rpart_repro.R`, run with
//! growth-only controls (`cp = 0, minsplit = 4, minbucket = 2`) so the
//! comparison is pure CART growth — no pruning (that is a later weighing
//! criterion in our plan). With `LeafKind::Mean` our split criterion
//! (fitted-leaf SSE reduction) *is* rpart's deviance criterion, and both use
//! midpoint split points, so the structures must coincide exactly. Child
//! ordering may mirror (cosmetic); comparisons are membership-based.

use mltreelib_core::model_tree::{LeafKind, ModelTree, ModelTreeParams};

const MPG: [f64; 32] = [
    21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2,
    10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4,
    15.8, 19.7, 15.0, 21.4,
];
const WT: [f64; 32] = [
    2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44, 3.44, 4.07, 3.73, 3.78,
    5.25, 5.424, 5.345, 2.2, 1.615, 1.835, 2.465, 3.52, 3.435, 3.84, 3.845, 1.935, 2.14,
    1.513, 3.17, 2.77, 3.57, 2.78,
];
const HP: [f64; 32] = [
    110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0, 123.0, 180.0,
    180.0, 180.0, 205.0, 215.0, 230.0, 66.0, 52.0, 65.0, 97.0, 150.0, 150.0, 245.0,
    175.0, 66.0, 91.0, 113.0, 264.0, 175.0, 335.0, 109.0,
];
const QSEC: [f64; 32] = [
    16.46, 17.02, 18.61, 19.44, 17.02, 20.22, 15.84, 20.0, 22.9, 18.3, 18.9, 17.4, 17.6,
    18.0, 17.98, 17.82, 17.42, 19.47, 18.52, 19.9, 20.01, 16.87, 17.3, 15.41, 17.05,
    18.9, 16.7, 16.9, 14.5, 15.5, 14.6, 18.6,
];
/// rpart 4.1.23 predictions, row order as above.
const RPART_PRED: [f64; 32] = [
    21.0, 21.0, 23.3333333333, 21.4333333333, 19.2, 18.3666666667, 14.1, 23.3333333333,
    23.3333333333, 18.3666666667, 18.3666666667, 15.8, 16.25, 15.8, 10.4, 10.4, 14.1,
    33.15, 30.4, 33.15, 21.4333333333, 17.35, 16.25, 14.1, 17.35, 26.65, 26.65, 30.4,
    15.4, 19.2, 15.4, 21.4333333333,
];
/// The 13 (variable, midpoint) splits rpart found, arbitrary order.
const RPART_SPLITS: [(&str, f64); 13] = [
    ("wt", 2.26),
    ("hp", 136.5),
    ("hp", 192.5),
    ("hp", 222.5),
    ("qsec", 15.005),
    ("qsec", 17.175),
    ("wt", 3.755),
    ("wt", 3.48),
    ("wt", 3.3275),
    ("hp", 96.0),
    ("qsec", 17.81),
    ("qsec", 19.185),
    ("wt", 1.775),
];

/// Same-partition tie alternatives — splits that cut the rows into identical
/// groups (hence identical gains); which label survives is arithmetic noise
/// in both rpart and ours (rpart's own pick flips with summation order).
/// rpart's recorded label is canonical here:
///   hp@254.5 ~ qsec@15.005, hp@222.5 ~ qsec@17.62, wt@3.755 ~ qsec@17.5.
fn tie_canonical(name: &str, t: f64) -> (String, f64) {
    let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
    if name == "hp" && close(t, 254.5) {
        ("qsec".to_string(), 15.005)
    } else if name == "qsec" && close(t, 17.62) {
        ("hp".to_string(), 222.5)
    } else if name == "qsec" && close(t, 17.5) {
        ("wt".to_string(), 3.755)
    } else {
        (name.to_string(), t)
    }
}

fn mtcars() -> (Vec<Vec<Option<f64>>>, Vec<Option<f64>>) {
    let x = (0..32)
        .map(|i| vec![Some(WT[i]), Some(HP[i]), Some(QSEC[i])])
        .collect();
    let y = MPG.iter().map(|&v| Some(v)).collect();
    (x, y)
}

fn fit(tree_x: &[Vec<Option<f64>>], y: &[Option<f64>]) -> ModelTree {
    ModelTree::fit_named(
        vec!["wt".to_string(), "hp".to_string(), "qsec".to_string()],
        tree_x,
        y,
        cart_params(),
    )
    .unwrap()
}

fn cart_params() -> ModelTreeParams {
    // rpart.control(cp = 0, minsplit = 4, minbucket = 2):
    // min_rows = 2 = minbucket; our split attempt gate is 2 * min_rows = 4.
    ModelTreeParams {
        max_depth: 30,
        min_rows: 2,
        min_gain: 1e-9,
        leaf: LeafKind::Mean,
        split_precision: None,
        ..Default::default()
    }
}

#[test]
fn reproduces_rpart_mtcars_tree() {
    let (x, y) = mtcars();
    let tree = fit(&x, &y);

    // Structure: 13 splits. Split *labels* match rpart's up to one documented
    // same-partition tie: at one node hp@254.5 and qsec@15.005 cut the rows
    // into identical groups (hence identical gains); rpart happened to record
    // qsec, we record hp. The leaf checks below prove the partitions coincide.
    let mut got: Vec<(String, f64)> = tree
        .splits()
        .iter()
        .map(|s| tie_canonical(&tree.col_names[s.col], s.threshold))
        .collect();
    let mut want: Vec<(String, f64)> = RPART_SPLITS
        .iter()
        .map(|(v, t)| (v.to_string(), *t))
        .collect();
    let key = |a: &(String, f64)| (a.0.clone(), (a.1 * 1e6).round() as i64);
    got.sort_by_key(key);
    want.sort_by_key(key);
    assert_eq!(got.len(), want.len(), "split count: got {got:?}");
    for (g, w) in got.iter().zip(want.iter()) {
        assert_eq!(g.0, w.0, "split variable: {got:?} vs {want:?}");
        assert!((g.1 - w.1).abs() < 1e-9, "split point {} vs {}", g.1, w.1);
    }

    // Values: identical predictions row by row.
    let preds = tree.predict(&x);
    for (i, (g, w)) in preds.iter().zip(RPART_PRED.iter()).enumerate() {
        assert!((g - w).abs() < 1e-6, "row {i}: prediction {g} vs rpart {w}");
    }

    // Sum of rpart's leaf deviances is 16.4 — total residual SSE must match.
    let sse: f64 = preds
        .iter()
        .zip(MPG.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum();
    assert!((sse - 16.4).abs() < 1e-6, "residual SSE {sse} vs rpart 16.4");

    // Leaf signatures (n, deviance, mean) must be rpart's 14 leaves exactly —
    // this is the partition-level identity check. Group rows by prediction.
    let mut groups: std::collections::BTreeMap<i64, Vec<usize>> = Default::default();
    for (i, p) in preds.iter().enumerate() {
        groups
            .entry((p * 1e6).round() as i64)
            .or_default()
            .push(i);
    }
    let mut got_leaves: Vec<(usize, f64, f64)> = groups
        .values()
        .map(|rows| {
            let mean = rows.iter().map(|&i| MPG[i]).sum::<f64>() / rows.len() as f64;
            let dev = rows.iter().map(|&i| (MPG[i] - mean).powi(2)).sum::<f64>();
            (rows.len(), dev, mean)
        })
        .collect();
    let mut want_leaves: Vec<(usize, f64, f64)> = vec![
        (2, 0.0, 10.4),
        (3, 1.04, 14.1),
        (2, 0.32, 15.4),
        (2, 0.72, 15.8),
        (2, 2.205, 16.25),
        (2, 6.845, 17.35),
        (2, 0.5, 19.2),
        (3, 1.0866666666666667, 18.366666666666667),
        (2, 0.0, 21.0),
        (3, 0.006666666666666667, 21.433333333333334),
        (3, 1.7066666666666668, 23.333333333333332),
        (2, 0.845, 26.65),
        (2, 0.0, 30.4),
        (2, 1.125, 33.15),
    ];
    let lkey = |a: &(usize, f64, f64)| (a.0, (a.1 * 1e6).round() as i64);
    got_leaves.sort_by_key(lkey);
    want_leaves.sort_by_key(lkey);
    assert_eq!(got_leaves.len(), want_leaves.len(), "{got_leaves:?}");
    for (g, w) in got_leaves.iter().zip(want_leaves.iter()) {
        assert_eq!(g.0, w.0, "leaf n: {got_leaves:?}");
        assert!((g.1 - w.1).abs() < 1e-6, "leaf deviance {} vs {}", g.1, w.1);
        assert!((g.2 - w.2).abs() < 1e-6, "leaf mean {} vs {}", g.2, w.2);
    }
}

#[test]
fn rpart_na_variant_still_builds_and_predicts() {
    // Same data with wt knocked out for rows 2, 6, 15 (1-based). NA routing
    // mechanisms legitimately differ: rpart uses surrogate splits, we use
    // learned default directions + NA-indicator leaf terms — so the exact
    // structure may diverge from rpart's NA tree. The contract here: NAs
    // never lose a row (fit + predict), and the tree still finds structure.
    let (mut x, y) = mtcars();
    for r in [1usize, 5, 14] {
        x[r][0] = None;
    }
    let tree = fit(&x, &y);

    assert!(!tree.splits().is_empty(), "NA rows must not stump the tree");

    // Every row — NA rows included — gets a finite prediction.
    let preds = tree.predict(&x);
    assert_eq!(preds.len(), 32);
    assert!(preds.iter().all(|p| p.is_finite()));
}
