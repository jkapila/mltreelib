//! Model tree tests: honest-OLS leaves, structure recovery (house signature),
//! the two leaf-column policies, and NA-first behavior (learnable missing
//! direction + NA-indicator leaf terms).

use mltreelib_core::model_tree::{LeafCols, LeafKind, ModelTree, ModelTreeParams};

#[test]
fn stump_is_global_ols() {
    // y = 2 + 3a - b exactly; depth 0 => one leaf recovering the coefficients.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..200 {
        let a = i as f64 / 200.0;
        let b = ((i * 7) % 50) as f64 / 50.0;
        x.push(vec![Some(a), Some(b)]);
        y.push(Some(2.0 + 3.0 * a - b));
    }
    let p = ModelTreeParams {
        max_depth: 0,
        min_rows: 5,
        l1: 0.0,
        l2: 0.0,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    assert_eq!(tree.n_leaves(), 1);
    assert!(tree.splits().is_empty());
    let (leaf, n) = tree.leaves()[0].clone();
    let m = leaf.linear.as_ref().unwrap();
    assert_eq!(n, 200);
    assert_eq!(m.cols, vec![0, 1]);
    assert!(m.na_coef.is_empty(), "no NAs -> no indicator terms");
    assert!((m.intercept - 2.0).abs() < 1e-6, "{}", m.intercept);
    assert!((m.coef[0] - 3.0).abs() < 1e-6, "{:?}", m.coef);
    assert!((m.coef[1] + 1.0).abs() < 1e-6, "{:?}", m.coef);
    // Predictions are always produced.
    let preds = tree.predict(&[vec![Some(0.5), Some(0.25)]]);
    assert!((preds[0] - (2.0 + 1.5 - 0.25)).abs() < 1e-6);
}

#[test]
fn recovers_planted_split_and_leaf_models() {
    // Two linear regimes split on z at 0: y = 1 + 2x (z<0), y = 5 - 3x (z>=0).
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..200 {
        let xv = ((i * 13) % 100) as f64 / 100.0;
        let zv = (i as f64 / 200.0) - 0.5;
        let yv = if zv < 0.0 {
            1.0 + 2.0 * xv
        } else {
            5.0 - 3.0 * xv
        };
        x.push(vec![Some(xv), Some(zv)]);
        y.push(Some(yv));
    }
    let p = ModelTreeParams {
        max_depth: 3,
        min_rows: 20,
        l1: 0.0,
        l2: 0.0,
        split_precision: None,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();

    // Structure recovery: the split must be on z (column 1) near 0.
    let splits = tree.splits();
    assert_eq!(splits.len(), 1, "expected exactly one planted split");
    assert_eq!(splits[0].col, 1);
    assert!(splits[0].threshold.abs() < 0.05, "{}", splits[0].threshold);

    // Leaf models recover the planted coefficients (z-coef ~ 0 inside leaves).
    assert_eq!(tree.n_leaves(), 2);
    let mut leaves = tree.leaves();
    leaves.sort_by(|a, b| {
        a.0.linear
            .as_ref()
            .unwrap()
            .intercept
            .partial_cmp(&b.0.linear.as_ref().unwrap().intercept)
            .unwrap()
    });
    let lo = leaves[0].0.linear.as_ref().unwrap();
    let hi = leaves[1].0.linear.as_ref().unwrap();
    assert!((lo.intercept - 1.0).abs() < 1e-6, "{}", lo.intercept);
    assert!((lo.coef[0] - 2.0).abs() < 1e-6, "{:?}", lo.coef);
    assert!((hi.intercept - 5.0).abs() < 1e-6, "{}", hi.intercept);
    assert!((hi.coef[0] + 3.0).abs() < 1e-6, "{:?}", hi.coef);

    // Every row (NA-free here) gets a prediction.
    let preds = tree.predict(&x);
    assert_eq!(preds.len(), 200);
    assert!(preds.iter().all(|p| p.is_finite()));
}

#[test]
fn path_used_leaves_are_piecewise_linear() {
    // y is piecewise linear in z itself: y = 2z (z<0), y = 4z+1 (z>=0).
    // PathUsed leaves regress on the split column only -> the two segments.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..200 {
        let z = (i as f64 / 200.0) - 0.5;
        let yv = if z < 0.0 { 2.0 * z } else { 4.0 * z + 1.0 };
        x.push(vec![Some(z)]);
        y.push(Some(yv));
    }
    let p = ModelTreeParams {
        max_depth: 2,
        min_rows: 10,
        l1: 0.0,
        l2: 0.0,
        leaf_cols: LeafCols::PathUsed,
        split_precision: None,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    assert_eq!(tree.splits().len(), 1);
    let mut leaves = tree.leaves();
    leaves.sort_by(|a, b| {
        a.0.linear
            .as_ref()
            .unwrap()
            .intercept
            .partial_cmp(&b.0.linear.as_ref().unwrap().intercept)
            .unwrap()
    });
    let lo = leaves[0].0.linear.as_ref().unwrap();
    let hi = leaves[1].0.linear.as_ref().unwrap();
    assert_eq!(lo.cols, vec![0]); // the split column is the leaf regressor
    assert_eq!(hi.cols, vec![0]);
    assert!((lo.intercept).abs() < 1e-6, "{}", lo.intercept);
    assert!((lo.coef[0] - 2.0).abs() < 1e-6, "{:?}", lo.coef);
    assert!((hi.intercept - 1.0).abs() < 1e-6, "{}", hi.intercept);
    assert!((hi.coef[0] - 4.0).abs() < 1e-6, "{:?}", hi.coef);
}

#[test]
fn na_direction_is_learned_from_the_gain() {
    // y = z^2 on observed rows (z = 0..99) — quadratic, so splitting helps;
    // 30 missing-z rows have y = 0, sitting with the low-z regime. The learned
    // default direction must send them left (missing_left = true).
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..100 {
        let z = i as f64;
        x.push(vec![Some(z)]);
        y.push(Some(z * z));
    }
    for _ in 0..30 {
        x.push(vec![None]);
        y.push(Some(0.0));
    }
    let p = ModelTreeParams {
        max_depth: 2,
        min_rows: 10,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    assert!(!tree.splits().is_empty(), "quadratic signal must split");
    let sp = tree.splits()[0];
    assert_eq!(sp.col, 0);
    assert!(sp.missing_left, "missing rows belong with the low-z side");
    // All 130 observed-target rows got a prediction, NAs included.
    let preds = tree.predict(&x);
    assert_eq!(preds.len(), 130);
    assert!(preds.iter().all(|p| p.is_finite()));
}

#[test]
fn na_indicator_makes_missingness_predictive() {
    // y = 2z on observed rows (z in [-0.5, 0.5)), y = 5 exactly when z is
    // missing. A stump leaf must learn value slope 2 AND the NA-indicator
    // term -> missing rows predict ~5, observed rows ~2z.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..100 {
        let z = (i as f64 / 100.0) - 0.5;
        x.push(vec![Some(z)]);
        y.push(Some(2.0 * z));
    }
    for _ in 0..30 {
        x.push(vec![None]);
        y.push(Some(5.0));
    }
    x.push(vec![None]);
    y.push(None); // missing target -> the only dropped row
    let p = ModelTreeParams {
        max_depth: 0,
        min_rows: 5,
        l1: 0.0,
        l2: 0.0,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    let (leaf, n) = tree.leaves()[0].clone();
    let m = leaf.linear.as_ref().unwrap();
    assert_eq!(n, 130, "NA rows train; only missing-y rows drop");
    assert_eq!(m.na_coef.len(), 1, "missingness is a learned term");
    let preds = tree.predict(&[vec![None], vec![Some(0.25)]]);
    assert!((preds[0] - 5.0).abs() < 1e-6, "NA row predicts its regime: {}", preds[0]);
    assert!((preds[1] - 0.5).abs() < 1e-6, "observed row keeps the slope: {}", preds[1]);
}

#[test]
fn predict_missing_on_no_na_leaf_does_not_panic() {
    // A leaf fitted on fully-observed rows has no indicator terms; a missing
    // value at predict time must still produce a finite prediction.
    let x: Vec<Vec<Option<f64>>> = (0..50).map(|i| vec![Some(i as f64)]).collect();
    let y: Vec<Option<f64>> = (0..50).map(|i| Some(2.0 * i as f64)).collect();
    let p = ModelTreeParams {
        max_depth: 0,
        min_rows: 5,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    let preds = tree.predict(&[vec![None]]);
    assert!(preds[0].is_finite(), "missing must never panic or vanish");
}

#[test]
fn weights_ignore_zero_weight_rows() {
    // A zero-weight row is an absent row: the fit and predictions must match
    // the unweighted data exactly.
    let x: Vec<Vec<Option<f64>>> = (0..100).map(|i| vec![Some(i as f64 / 100.0)]).collect();
    let y: Vec<Option<f64>> = (0..100).map(|i| Some(3.0 * i as f64 / 100.0)).collect();
    let mut xw = x.clone();
    xw.push(vec![Some(0.5)]);
    let mut yw = y.clone();
    yw.push(Some(1e6));
    let mut w = vec![1.0; 100];
    w.push(0.0);
    let p = ModelTreeParams {
        max_depth: 0,
        min_rows: 5,
        l1: 0.0,
        l2: 0.0,
        ..Default::default()
    };
    let t1 = ModelTree::fit(&x, &y, p.clone()).unwrap();
    let t2 = ModelTree::fit_weighted(&xw, &yw, &w, p).unwrap();
    let probe = vec![vec![Some(0.25)], vec![Some(0.75)]];
    for (a, b) in t1.predict(&probe).iter().zip(t2.predict(&probe).iter()) {
        assert!((a - b).abs() < 1e-9, "zero-weight row leaked: {a} vs {b}");
    }
}

#[test]
fn weights_shift_the_fit() {
    // Down-weighting the high-x half must change the fitted slope.
    let x: Vec<Vec<Option<f64>>> = (0..100).map(|i| vec![Some(i as f64 / 100.0)]).collect();
    let y: Vec<Option<f64>> = (0..100).map(|i| Some(i as f64 / 100.0)).collect();
    let w: Vec<f64> = (0..100)
        .map(|i| if i < 50 { 1.0 } else { 0.01 })
        .collect();
    let p = ModelTreeParams {
        max_depth: 0,
        min_rows: 5,
        l1: 0.0,
        l2: 0.0,
        ..Default::default()
    };
    let t = ModelTree::fit_weighted(&x, &y, &w, p).unwrap();
    let m = t.leaves()[0].0.linear.as_ref().unwrap().clone();
    // Slope pulled toward the low-x region's relation.
    assert!(m.coef[0] < 1.5, "weights did not influence the fit: {:?}", m.coef);
}

#[test]
fn monotone_constraint_orders_predictions() {
    // y = x0 + noise-like extras; with mono +1 on x0, predictions must be
    // non-decreasing along x0 no matter the noise.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..200 {
        let x0 = (i % 20) as f64 / 20.0;
        let jitter = (((i * 37) % 11) as f64 - 5.0) / 5.0;
        x.push(vec![Some(x0), Some(jitter)]);
        y.push(Some(2.0 * x0 + jitter));
    }
    let p = ModelTreeParams {
        max_depth: 3,
        min_rows: 10,
        l1: 0.0,
        l2: 0.0,
        monotone: vec![1, 0],
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    let mut last = f64::NEG_INFINITY;
    for i in 0..21 {
        let x0 = i as f64 / 20.0;
        let pred = tree.predict(&[vec![Some(x0), Some(0.0)]])[0];
        assert!(pred >= last - 1e-9, "monotone +1 violated at x0={x0}: {pred} < {last}");
        last = pred;
    }
}

#[test]
fn monotone_constraint_orders_children() {
    // y increases in x0 but is noisy; monotone +1 on x0 must produce
    // non-decreasing predictions along x0.
    let x: Vec<Vec<Option<f64>>> = (0..200)
        .map(|i| {
            let t = (i % 20) as f64 / 20.0;
            vec![Some(t), Some(((i * 31) % 17) as f64 / 17.0)]
        })
        .collect();
    let y: Vec<Option<f64>> = (0..200)
        .map(|i| Some(2.0 * (i % 20) as f64 / 20.0))
        .collect();
    let mut p = ModelTreeParams {
        max_depth: 3,
        min_rows: 5,
        l1: 0.0,
        l2: 0.0,
        leaf: LeafKind::Mean,
        ..Default::default()
    };
    p.monotone = vec![1, 0];
    let tree = ModelTree::fit(&x, &y, p).unwrap();

    let grid: Vec<Vec<Option<f64>>> = (0..21)
        .map(|i| vec![Some(i as f64 / 20.0), Some(0.5)])
        .collect();
    let preds: Vec<f64> = tree.predict(&grid);
    for w in preds.windows(2) {
        assert!(w[1] >= w[0] - 1e-9, "monotone +1 violated: {w:?}");
    }
}

#[test]
fn monotone_constrains_linear_leaf_coefficients() {
    // Linear leaves carry the feature effect in their COEFFICIENTS, not in
    // splits - a decreasing signal with mono=+1 must produce a flat (not
    // negative-slope) response along x0. Regression for the projection fix:
    // before it, split-time ordering never fired because there were no
    // splits at all.
    let x: Vec<Vec<Option<f64>>> = (0..200)
        .map(|i| vec![Some((i % 20) as f64 / 20.0), Some(((i * 31) % 17) as f64 / 17.0)])
        .collect();
    let y: Vec<Option<f64>> = (0..200)
        .map(|i| Some(-5.0 * (i % 20) as f64 / 20.0))
        .collect();
    let mut p = ModelTreeParams {
        max_depth: 0, // single linear leaf: the whole effect lives in coefs
        min_rows: 5,
        l1: 0.0,
        l2: 0.0,
        leaf: LeafKind::Linear,
        ..Default::default()
    };
    p.monotone = vec![1, 0];
    let tree = ModelTree::fit(&x, &y, p).unwrap();

    let grid: Vec<Vec<Option<f64>>> = (0..21)
        .map(|i| vec![Some(i as f64 / 20.0), Some(0.5)])
        .collect();
    let preds: Vec<f64> = tree.predict(&grid);
    for w in preds.windows(2) {
        assert!(w[1] >= w[0] - 1e-9, "linear leaf leaked a decreasing x0 effect: {w:?}");
    }
}

#[test]
fn printer_matches_rpart_layout() {
    // Gold format: installed rpart's print (references/repos/rpart
    // print.rpart.R). Preorder rows, binary-heap numbering right-aligned to
    // the widest, two-space-per-depth indent, R format(signif(x, 7)) columns,
    // "*" on leaves.
    let x: Vec<Vec<Option<f64>>> = (0..4).map(|i| vec![Some(i as f64)]).collect();
    let y: Vec<Option<f64>> = vec![Some(0.0), Some(0.0), Some(1.0), Some(1.0)];
    let p = ModelTreeParams {
        max_depth: 1,
        min_rows: 1,
        l1: 0.0,
        l2: 0.0,
        split_precision: None,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    let expected = "n= 4 \n\nnode), split, n, deviance, yval\n      * denotes terminal node\n\n1) root 4 1 0.5  \n  2) x0< 1.5 2 0 0.0 *\n  3) x0>=1.5 2 0 1.0 *\n";
    assert_eq!(format!("{tree}"), expected);
}

#[test]
fn dot_export_names_nodes_and_edges() {
    let x: Vec<Vec<Option<f64>>> = (0..4).map(|i| vec![Some(i as f64)]).collect();
    let y: Vec<Option<f64>> = vec![Some(0.0), Some(0.0), Some(1.0), Some(1.0)];
    let p = ModelTreeParams {
        max_depth: 1,
        min_rows: 1,
        l1: 0.0,
        l2: 0.0,
        split_precision: None,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    let dot = tree.to_dot();
    assert!(dot.starts_with("digraph tree {"));
    assert!(dot.contains("n1 -> n2 [label=\"< 1.5\"]"), "{dot}");
    assert!(dot.contains("n1 -> n3 [label=\">= 1.5\"]"), "{dot}");
    assert!(dot.contains("label=\"leaf\\nn=2  yval=0.000\""), "{dot}");
}

#[test]
fn feature_importance_ranks_the_signal_column() {
    // x0 drives y; x1/x2 are noise. Gain-share importance must rank x0 first
    // with the clear majority of the total split gain.
    let x: Vec<Vec<Option<f64>>> = (0..400)
        .map(|i| {
            vec![
                Some((i % 40) as f64 / 40.0),
                Some(((i * 37) % 31) as f64 / 31.0),
                Some(((i * 17) % 29) as f64 / 29.0),
            ]
        })
        .collect();
    let y: Vec<Option<f64>> = (0..400).map(|i| Some(7.0 * (i % 40) as f64 / 40.0)).collect();
    let p = ModelTreeParams {
        max_depth: 3,
        min_rows: 10,
        l1: 0.0,
        l2: 0.0,
        leaf: LeafKind::Mean,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    let imp = tree.feature_importances();
    assert_eq!(imp[0].0, "x0", "{imp:?}");
    assert!(imp[0].1 > 0.8, "x0 should own most of the gain: {imp:?}");
}

#[test]
fn break_down_telescopes_on_mean_and_linear_leaves() {
    let x: Vec<Vec<Option<f64>>> = (0..200)
        .map(|i| {
            vec![
                Some((i % 20) as f64 / 20.0),
                Some(((i * 7) % 13) as f64 / 13.0),
                if i % 17 == 0 {
                    None
                } else {
                    Some(i as f64 % 3.0)
                },
            ]
        })
        .collect();
    let y: Vec<Option<f64>> = (0..200)
        .map(|i| Some(2.0 * (i % 20) as f64 / 20.0 + if i % 20 > 10 { 3.0 } else { 0.0 }))
        .collect();
    for leaf in [LeafKind::Mean, LeafKind::Linear] {
        let p = ModelTreeParams {
            max_depth: 3,
            min_rows: 10,
            l1: 0.0,
            l2: 0.0,
            leaf,
            ..Default::default()
        };
        let tree = ModelTree::fit(&x, &y, p).unwrap();
        for row in x.iter().step_by(7) {
            let bd = tree.break_down(row);
            let sum: f64 = bd.steps.iter().map(|(_, v)| v).sum();
            assert!(
                (bd.baseline + sum + bd.residual - bd.prediction).abs() < 1e-9,
                "{leaf:?} {bd:?}"
            );
        }
    }
}

#[test]
fn break_down_attributes_the_step_column() {
    // y jumps on x0; a high-x0 row must show the dominant contribution there.
    let x: Vec<Vec<Option<f64>>> = (0..200)
        .map(|i| vec![Some((i % 20) as f64 / 20.0), Some(((i * 11) % 13) as f64 / 13.0)])
        .collect();
    let y: Vec<Option<f64>> = (0..200)
        .map(|i| Some(if i % 20 > 10 { 8.0 } else { 0.0 }))
        .collect();
    let p = ModelTreeParams {
        max_depth: 2,
        min_rows: 10,
        l1: 0.0,
        l2: 0.0,
        leaf: LeafKind::Mean,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    let bd = tree.break_down(&[Some(1.0), Some(0.5)]);
    let x0: f64 = bd.steps.iter().filter(|(c, _)| *c == 0).map(|(_, v)| *v).sum();
    let x1: f64 = bd.steps.iter().filter(|(c, _)| *c == 1).map(|(_, v)| *v).sum();
    assert!((x0 - (bd.prediction - bd.baseline)).abs() < 1e-9, "x0 carries the step: {bd:?}");
    assert!(x1.abs() < 1.0, "x1 is noise: {bd:?}");
}

#[test]
fn interaction_groups_respected() {
    // x2 matters as a STEP deep in the tree (x0 > 0.75 region) - a linear
    // leaf cannot absorb it, so the unconstrained tree must mix groups along
    // some path (x0 group, then x2 group). The constraint: every root-to-leaf
    // path stays within one group.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..300 {
        let x0 = (i % 30) as f64 / 30.0;
        let x2 = ((i * 13) % 17) as f64 / 17.0;
        x.push(vec![Some(x0), Some(((i * 7) % 11) as f64 / 11.0), Some(x2)]);
        y.push(Some(
            5.0 * x0 + if x0 > 0.75 && x2 > 0.5 { 8.0 } else { 0.0 },
        ));
    }
    let base = ModelTreeParams {
        max_depth: 4,
        min_rows: 10,
        l1: 0.0,
        l2: 0.0,
        ..Default::default()
    };
    let tree_c = ModelTree::fit(
        &x,
        &y,
        ModelTreeParams {
            interaction_groups: vec![vec![0, 1], vec![2]],
            ..base.clone()
        },
    )
    .unwrap();
    let tree_u = ModelTree::fit(&x, &y, base).unwrap();

    let group_of = |c: usize| if c < 2 { 0 } else { 1 };
    for path in tree_c.split_paths() {
        let mut it = path.iter().map(|&c| group_of(c));
        if let Some(g) = it.next() {
            assert!(it.all(|x| x == g), "constrained path mixes groups: {path:?}");
        }
    }
    assert!(
        tree_u
            .split_paths()
            .iter()
            .any(|p| {
                let mut it = p.iter().map(|&c| group_of(c));
                match it.next() {
                    Some(g) => it.any(|x| x != g),
                    None => false,
                }
            }),
        "unconstrained tree never mixed groups - constraint not exercised"
    );
}

#[test]
fn preselect_drops_junk_columns() {
    // Column 0 = signal, column 1 = pure noise; pre-selection keeps only 0.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..200 {
        let signal = ((i * 29) % 20) as f64 / 20.0;
        let noise = ((i * 43) % 17) as f64 / 17.0;
        x.push(vec![Some(signal), Some(noise)]);
        y.push(Some(4.0 * signal));
    }
    let p = ModelTreeParams {
        max_depth: 3,
        min_rows: 10,
        l1: 0.0,
        l2: 0.0,
        min_relevance: Some(0.2),
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    assert_eq!(tree.active_cols, vec![0], "pre-selection kept the junk column");
    for s in tree.splits() {
        assert_eq!(s.col, 0, "junk column split the tree");
    }
}

#[test]
fn parallel_scan_is_deterministic() {
    // Wide enough to cross the parallel threshold; two fits must agree
    // exactly (tie-breaks are by scan order, never thread order).
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..300 {
        let mut row = Vec::new();
        for j in 0..120 {
            row.push(Some((((i * 31 + j * 17) % 97) as f64) / 97.0));
        }
        x.push(row);
        y.push(Some((((i * 13) % 23) as f64) / 23.0));
    }
    let p = ModelTreeParams {
        max_depth: 3,
        min_rows: 10,
        l1: 0.0,
        l2: 0.0,
        ..Default::default()
    };
    let t1 = ModelTree::fit(&x, &y, p.clone()).unwrap();
    let t2 = ModelTree::fit(&x, &y, p).unwrap();
    let s1 = t1.splits();
    let s2 = t2.splits();
    assert_eq!(s1.len(), s2.len());
    for (a, b) in s1.iter().zip(s2.iter()) {
        assert_eq!((a.col, a.missing_left), (b.col, b.missing_left));
        assert!((a.threshold - b.threshold).abs() == 0.0);
    }
}

#[test]
fn min_rows_blocks_tiny_leaves() {
    // Without enough rows per side the tree stays a single mean leaf.
    let x: Vec<Vec<Option<f64>>> = (0..10).map(|i| vec![Some(i as f64)]).collect();
    let y: Vec<Option<f64>> = (0..10).map(|i| Some(i as f64)).collect();
    let p = ModelTreeParams {
        max_depth: 4,
        min_rows: 20,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    assert_eq!(tree.n_leaves(), 1);
    assert!(tree.splits().is_empty());
    let preds = tree.predict(&x);
    assert!(preds.iter().all(|p| p.is_finite()), "mean leaf still predicts");
}

// ------------------------- split-search knobs (2026-10-29) --------------

use mltreelib_core::model_tree::{
    CatOrder, CatSearch, ColType, CustomGain, GainFn,
};

#[test]
fn categorical_split_partitions_levels() {
    // Level 0 sits at y≈0, levels 1 and 2 at y≈5: the true binary
    // partition is {0} | {1,2}. Codes are the levels; dtypes declare the
    // column categorical — no encoding anywhere.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (code, base) in [(0.0, 0.0), (1.0, 5.0), (2.0, 5.0)] {
        for i in 0..20 {
            x.push(vec![Some(code)]);
            y.push(Some(base + (i as f64 - 10.0) * 0.01));
        }
    }
    let p = ModelTreeParams {
        max_depth: 1,
        min_rows: 5,
        dtypes: vec![ColType::Categorical],
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    let splits = tree.splits();
    assert_eq!(splits.len(), 1);
    assert!(splits[0].cat.is_some(), "categorical split must be a partition");
    // The partition lives in the side table; membership decides routing.
    let left = tree.predict(&[vec![Some(0.0)], vec![Some(1.0)], vec![Some(2.0)]]);
    assert!(left[0] < 1.0 && left[1] > 4.0 && left[2] > 4.0, "{left:?}");
}

#[test]
fn categorical_single_vs_rest_stops_at_first_acceptable() {
    // Four levels, one clearly separate; the most frequent level is the
    // last one by code. SingleVsRest visits in frequency order and stops
    // at the first acceptable level.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (code, base, n) in [(0.0, 0.0, 5), (1.0, 0.0, 5), (2.0, 0.0, 5), (3.0, 9.0, 30)] {
        for _ in 0..n {
            x.push(vec![Some(code)]);
            y.push(Some(base));
        }
    }
    let p = ModelTreeParams {
        max_depth: 1,
        min_rows: 3,
        dtypes: vec![ColType::Categorical],
        cat_search: CatSearch::SingleVsRest,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    let pred = tree.predict(&[vec![Some(3.0)], vec![Some(0.0)]]);
    assert!(pred[0] > 5.0 && pred[1] < 4.0, "{pred:?}");
}

#[test]
fn custom_gain_matches_builtin_entropy() {
    struct EntropyGain;
    impl CustomGain for EntropyGain {
        fn gain(&self, left: &[f64], right: &[f64]) -> f64 {
            fn h(ys: &[f64]) -> f64 {
                let n = ys.len() as f64;
                let p1 = ys.iter().filter(|&&v| v == 1.0).count() as f64 / n;
                let p0 = 1.0 - p1;
                let e = |p: f64| if p > 0.0 { -p * p.log2() } else { 0.0 };
                e(p0) + e(p1)
            }
            let n = (left.len() + right.len()) as f64;
            let mut all: Vec<f64> = Vec::new();
            all.extend_from_slice(left);
            all.extend_from_slice(right);
            h(&all) - (left.len() as f64 / n) * h(left) - (right.len() as f64 / n) * h(right)
        }
    }
    let x: Vec<Vec<Option<f64>>> = (0..40).map(|i| vec![Some((i / 20) as f64)]).collect();
    let y: Vec<Option<f64>> = (0..40)
        .map(|i| Some(if i % 4 < 2 { 0.0 } else { 1.0 }))
        .collect();
    let mut p = ModelTreeParams {
        max_depth: 1,
        min_rows: 2,
        ..Default::default()
    };
    p.gain = GainFn::Entropy;
    let t_ent = ModelTree::fit(&x, &y, p.clone()).unwrap();
    p.gain = GainFn::Custom(std::sync::Arc::new(EntropyGain));
    let t_custom = ModelTree::fit(&x, &y, p).unwrap();
    let a = t_ent.splits();
    let b = t_custom.splits();
    assert_eq!(a.len(), b.len());
    if let (Some(sa), Some(sb)) = (a.first(), b.first()) {
        assert_eq!(sa.col, sb.col);
        assert_eq!(sa.threshold, sb.threshold);
    }
}

#[test]
fn split_precision_grid_rounds_thresholds() {
    // Values scattered inside tenths; the grid must produce a grid-valued
    // threshold and route raw values consistently at predict time.
    let x: Vec<Vec<Option<f64>>> = (0..40)
        .map(|i| {
            let v = if i < 20 { 1.24 + (i as f64) * 0.001 } else { 2.27 + (i as f64) * 0.001 };
            vec![Some(v)]
        })
        .collect();
    let y: Vec<Option<f64>> = (0..40).map(|i| Some(if i < 20 { 0.0 } else { 1.0 })).collect();
    let p = ModelTreeParams {
        max_depth: 1,
        min_rows: 2,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    let splits = tree.splits();
    assert_eq!(splits.len(), 1);
    let t = splits[0].threshold;
    assert!((t * 10.0 - (t * 10.0).round()).abs() < 1e-9, "threshold {t} not on 0.1 grid");
    let pred = tree.predict(&x);
    assert!(pred[..20].iter().all(|&v| v < 0.5));
    assert!(pred[20..].iter().all(|&v| v > 0.5));
}

#[test]
fn quantile_uses_leaf_digest_and_honors_lean_leaves() {
    let x: Vec<Vec<Option<f64>>> = (0..60).map(|i| vec![Some((i / 30) as f64)]).collect();
    let y: Vec<Option<f64>> = (0..60).map(|i| Some((i % 30) as f64)).collect();
    let p = ModelTreeParams {
        max_depth: 1,
        min_rows: 5,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p.clone()).unwrap();
    let med = tree.quantile(&[vec![Some(0.0)]], 0.5);
    assert!(med[0].is_finite(), "digest quantile should be answerable");
    let p_lean = ModelTreeParams {
        leaf_digest: false,
        ..p
    };
    let lean = ModelTree::fit(&x, &y, p_lean).unwrap();
    let gone = lean.quantile(&[vec![Some(0.0)]], 0.5);
    assert!(gone[0].is_nan(), "lean leaves must not answer quantiles");
}
