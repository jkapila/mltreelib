//! Model tree tests: honest-OLS leaves, structure recovery (house signature),
//! the two leaf-column policies, and NA-first behavior (learnable missing
//! direction + NA-indicator leaf terms).

use mltreelib_core::gate::SplitTest;
use mltreelib_core::leaf::{LeafCols, LeafKind};
use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};

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
        prune: PruneCost::None,
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
        prune: PruneCost::None,
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
        prune: PruneCost::None,
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

use mltreelib_core::criteria::{CustomGain, GainFn};
use mltreelib_core::prune::PruneCost;
use mltreelib_core::split::{CatSearch, ColType};

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
        fn gain(&self, left: &[(f64, f64)], right: &[(f64, f64)]) -> f64 {
            fn h(p: &[(f64, f64)]) -> f64 {
                let total: f64 = p.iter().map(|&(_, w)| w).sum();
                if total <= 0.0 {
                    return 0.0;
                }
                let p1: f64 = p.iter().filter(|&&(y, _)| y == 1.0).map(|&(_, w)| w).sum::<f64>() / total;
                let p0 = 1.0 - p1;
                let e = |q: f64| if q > 0.0 { -q * q.log2() } else { 0.0 };
                e(p0) + e(p1)
            }
            let wl: f64 = left.iter().map(|&(_, w)| w).sum();
            let wr: f64 = right.iter().map(|&(_, w)| w).sum();
            let w = (wl + wr).max(1e-300);
            let mut all: Vec<(f64, f64)> = Vec::new();
            all.extend_from_slice(left);
            all.extend_from_slice(right);
            h(&all) - (wl / w) * h(left) - (wr / w) * h(right)
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
        split_precision: Some(0.1),
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

// ------- gain family, weights, imbalance, column sample, pruning ---------

#[test]
fn gain_ratio_sdr_twoing_find_the_regime_split() {
    // 0/1 classes in two regimes; any criterion family must find x<=0.5.
    let x: Vec<Vec<Option<f64>>> = (0..60)
        .map(|i| vec![Some(if i < 30 { (i % 10) as f64 * 0.05 } else { 1.0 + (i % 10) as f64 * 0.05 })])
        .collect();
    let y: Vec<Option<f64>> = (0..60).map(|i| Some(if i < 30 { 0.0 } else { 1.0 })).collect();
    for gain in [
        GainFn::FriedmanMse,
        GainFn::AbsoluteError,
        GainFn::Huber,
        GainFn::GainRatio,
        GainFn::Sdr,
        GainFn::Twoing,
    ] {
        let p = ModelTreeParams {
            max_depth: 1,
            min_rows: 5,
            gain: gain.clone(),
            ..Default::default()
        };
        let tree = ModelTree::fit(&x, &y, p).unwrap();
        let splits = tree.splits();
        assert_eq!(splits.len(), 1, "{gain:?}");
        assert!(splits[0].threshold < 1.0, "{gain:?} split too high");
    }
}

#[test]
fn poisson_criterion_splits_counts() {
    // Count targets (rates): the Poisson deviance must separate the two
    // rate regimes.
    let x: Vec<Vec<Option<f64>>> = (0..60)
        .map(|i| vec![Some(if i < 30 { (i % 5) as f64 * 0.1 } else { 1.0 + (i % 5) as f64 * 0.1 })])
        .collect();
    let y: Vec<Option<f64>> = (0..60).map(|i| Some(if i < 30 { 1.0 } else { 8.0 })).collect();
    let p = ModelTreeParams {
        max_depth: 1,
        min_rows: 5,
        gain: GainFn::Poisson,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    assert_eq!(tree.splits().len(), 1);
    let pred = tree.predict(&[vec![Some(0.2)], vec![Some(1.2)]]);
    assert!(pred[0] < 2.0 && pred[1] > 5.0, "{pred:?}");
}

#[test]
fn weights_change_the_split() {
    // Equal rows on both sides of the truth at 0.5; heavy weights on the
    // right block make the weight-aware split land right of 1.0.
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();
    for i in 0..20 {
        x.push(vec![Some((i as f64) * 0.1)]);
        y.push(Some(if i < 10 { 0.0 } else { 1.0 }));
        w.push(if i >= 10 { 50.0 } else { 1.0 });
    }
    let p = ModelTreeParams {
        max_depth: 1,
        min_rows: 2,
        split_precision: None,
        ..Default::default()
    };
    let tree = ModelTree::fit_weighted(&x, &y, &w, p).unwrap();
    let pred = tree.predict(&[vec![Some(1.2)], vec![Some(0.2)]]);
    // The 50x block dominates: the split must isolate the light left rows.
    assert!(pred[0] > 0.5 && pred[1] < 0.5, "{pred:?}");
}

#[test]
fn imbalance_penalty_blocks_lopsided_splits() {
    // A single 5.0 outlier invites a 1-vs-rest split (gain ≈ 24); without
    // the penalty it forms, with a strong penalty it must not.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..40 {
        x.push(vec![Some((i as f64) * 0.1)]);
        y.push(Some(0.0));
    }
    x.push(vec![Some(9.0)]);
    y.push(Some(5.0));
    let base = ModelTreeParams {
        max_depth: 1,
        min_rows: 2,
        split_precision: None,
        ..Default::default()
    };
    let plain = ModelTree::fit(&x, &y, base.clone()).unwrap();
    assert_eq!(plain.splits().len(), 1, "unpenalized: the outlier split forms");
    let penalized = ModelTree::fit(
        &x,
        &y,
        ModelTreeParams {
            imbalance_penalty: 1000.0,
            ..base
        },
    )
    .unwrap();
    assert_eq!(
        penalized.splits().len(),
        0,
        "penalized split should not form"
    );
}

#[test]
fn col_sample_is_deterministic_per_seed() {
    // Four redundant informative columns; one column per node (0.25). Same
    // seed must reproduce exactly; across seeds the chosen root column
    // must not be constant.
    let x: Vec<Vec<Option<f64>>> = (0..120)
        .map(|i| {
            vec![
                Some((i / 10) as f64),
                Some((i / 10) as f64 + 1.0),
                Some((i / 10) as f64 - 1.0),
                Some((i / 10) as f64 * 1.001),
            ]
        })
        .collect();
    let y: Vec<Option<f64>> = (0..120).map(|i| Some((i / 10) as f64 * 2.0)).collect();
    let mk = |seed: u64| ModelTreeParams {
        max_depth: 2,
        min_rows: 3,
        col_sample: Some(0.25),
        seed,
        ..Default::default()
    };
    let a = ModelTree::fit(&x, &y, mk(7)).unwrap();
    let b = ModelTree::fit(&x, &y, mk(7)).unwrap();
    assert_eq!(a.predict(&x), b.predict(&x), "same seed must reproduce exactly");
    let root_cols: Vec<usize> = (1u64..=12)
        .map(|s| {
            let t = ModelTree::fit(&x, &y, mk(s)).unwrap();
            t.splits().first().map(|sp| sp.col).unwrap_or(99)
        })
        .collect();
    assert!(
        root_cols.iter().min() != root_cols.iter().max(),
        "seeds must explore different columns: {root_cols:?}"
    );
}

#[test]
fn bic_prune_shrinks_noisy_depth() {
    // Pure noise: cost pruning must collapse the tree to a single leaf.
    let x: Vec<Vec<Option<f64>>> = (0..80)
        .map(|i| vec![Some((i % 17) as f64), Some((i % 5) as f64)])
        .collect();
    let y: Vec<Option<f64>> = (0..80)
        .map(|i| Some(((i * 7919) % 101) as f64 / 101.0))
        .collect();
    let base = ModelTreeParams {
        max_depth: 4,
        min_rows: 2,
        prune: PruneCost::None,
        cp_alpha: 0.0,
        ..Default::default()
    };
    let unpruned = ModelTree::fit(&x, &y, base.clone()).unwrap();
    let pruned = ModelTree::fit(
        &x,
        &y,
        ModelTreeParams {
            prune: PruneCost::Bic,
            ..base
        },
    )
    .unwrap();
    assert!(
        pruned.splits().len() < unpruned.splits().len(),
        "BIC must shrink noisy structure ({} vs {})",
        pruned.splits().len(),
        unpruned.splits().len()
    );
}

// ------------------------------ simple leaf kinds ------------------------

#[test]
fn median_leaf_resists_outliers() {
    // A 100x outlier in one region drags the mean leaf up; the median leaf
    // must stay near the bulk.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..40 {
        x.push(vec![Some(if i < 20 { 0.0 } else { 1.0 })]);
        y.push(Some(if i < 20 { 5.0 } else if i < 39 { 6.0 } else { 100.0 }));
    }
    let base = ModelTreeParams {
        max_depth: 1,
        min_rows: 5,
        ..Default::default()
    };
    let mean_tree = ModelTree::fit(&x, &y, base.clone()).unwrap();
    let med_tree = ModelTree::fit(
        &x,
        &y,
        ModelTreeParams {
            leaf: LeafKind::Median,
            ..base
        },
    )
    .unwrap();
    let p_mean = mean_tree.predict(&[vec![Some(1.0)]])[0];
    let p_med = med_tree.predict(&[vec![Some(1.0)]])[0];
    assert!(p_med < 7.0, "median leaf: {p_med}");
    assert!(p_mean > 9.0, "mean leaf should be dragged: {p_mean}");
}

#[test]
fn quantile_leaf_honors_tau() {
    let yv: Vec<Option<f64>> = (0..100).map(|i| Some((i % 50) as f64)).collect();
    let x: Vec<Vec<Option<f64>>> = (0..100).map(|_| vec![Some(0.0)]).collect();
    let q = |tau: f64| {
        ModelTree::fit(
            &x,
            &yv,
            ModelTreeParams {
                max_depth: 0,
                min_rows: 1,
                leaf: LeafKind::Quantile(tau),
                ..Default::default()
            },
        )
        .unwrap()
        .predict(&[vec![Some(0.0)]])[0]
    };
    let (lo, mid, hi) = (q(0.1), q(0.5), q(0.9));
    assert!(lo < mid && mid < hi, "quantile leaf ordering: {lo} {mid} {hi}");
}

#[test]
fn huber_leaf_sits_between_mean_and_median() {
    let mut y: Vec<Option<f64>> = (0..39).map(|_| Some(5.0)).collect();
    y.push(Some(100.0));
    let x: Vec<Vec<Option<f64>>> = (0..40).map(|_| vec![Some(0.0)]).collect();
    let fit = |leaf: LeafKind| {
        ModelTree::fit(
            &x,
            &y,
            ModelTreeParams {
                max_depth: 0,
                min_rows: 1,
                leaf,
                ..Default::default()
            },
        )
        .unwrap()
        .predict(&[vec![Some(0.0)]])[0]
    };
    let (med, hub, mean) = (
        fit(LeafKind::Median),
        fit(LeafKind::Huber),
        fit(LeafKind::Mean),
    );
    assert!(med <= hub && hub < mean, "med={med} hub={hub} mean={mean}");
}

// ---- Chow split-significance gate (split_test="chow") ----

/// Deterministic pseudo-noise in [-0.5, 0.5) — splitmix64 avalanche, no
/// RNG dependency, decorrelated from the row index.
fn noise(i: usize) -> f64 {
    let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z = z ^ (z >> 31);
    (z % 1000) as f64 / 1000.0 - 0.5
}

#[test]
fn chow_gate_stops_noise_splits() {
    // Pure noise: no x carries signal. The ungated scan always finds a
    // split (best-of-many SSE reduction), but none of them is
    // statistically real — the gate must keep the node a leaf.
    let x: Vec<Vec<Option<f64>>> = (0..400)
        .map(|i| vec![Some((i % 20) as f64 / 20.0)])
        .collect();
    let y: Vec<Option<f64>> = (0..400).map(|i| Some(noise(i))).collect();

    let base = ModelTreeParams {
        max_depth: 3,
        min_rows: 10,
        leaf: LeafKind::Mean,
        l2: 0.0,
        prune: PruneCost::None,
        ..Default::default()
    };
    let ungated = ModelTree::fit(&x, &y, base.clone()).unwrap();
    assert!(ungated.n_leaves() > 1, "the ungated scan does split noise");

    // Plain Chow at a strict level stops it.
    let gated = ModelTree::fit(
        &x,
        &y,
        ModelTreeParams {
            split_test: SplitTest::Chow,
            split_alpha: 1e-6,
            ..base.clone()
        },
    )
    .unwrap();
    assert_eq!(gated.n_leaves(), 1, "chow gate must stop noise splits");

    // Selection-corrected Chow at the conventional level stops it too.
    let bf = ModelTree::fit(
        &x,
        &y,
        ModelTreeParams {
            split_test: SplitTest::Chow,
            split_alpha: 0.05,
            bonferroni: true,
            ..base
        },
    )
    .unwrap();
    assert_eq!(bf.n_leaves(), 1, "bonferroni chow must stop noise splits");
}

#[test]
fn chow_gate_allows_clear_regime() {
    // Two planted regimes with a real structural break: the gate must let
    // the split through and the tree must recover the break.
    let n = 200;
    let x: Vec<Vec<Option<f64>>> = (0..n).map(|i| vec![Some(i as f64 / n as f64)]).collect();
    let y: Vec<Option<f64>> = (0..n)
        .map(|i| {
            let x0 = i as f64 / n as f64;
            Some(if x0 >= 0.5 { 5.0 } else { 0.0 } + 0.02 * noise(i))
        })
        .collect();
    let tree = ModelTree::fit(
        &x,
        &y,
        ModelTreeParams {
            max_depth: 3,
            min_rows: 10,
            leaf: LeafKind::Mean,
            l2: 0.0,
            split_test: SplitTest::Chow,
            split_alpha: 0.05,
            bonferroni: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(tree.n_leaves() > 1, "a real break must pass the gate");
    assert!(
        tree.splits()
            .iter()
            .any(|s| s.col == 0 && (s.threshold - 0.5).abs() < 0.06),
        "the recovered break sits at 0.5: {:?}",
        tree.splits()
    );
    let p0 = tree.predict(&[vec![Some(0.1)]])[0];
    let p1 = tree.predict(&[vec![Some(0.9)]])[0];
    assert!(p0 < 1.0 && p1 > 4.0, "regime means recovered: {p0} {p1}");
}

#[test]
fn bonferroni_gates_at_least_as_much() {
    // Borderline signal: the selection-corrected gate can only stop more
    // often than the uncorrected one (alpha_eff <= alpha).
    let n = 400;
    let x: Vec<Vec<Option<f64>>> = (0..n).map(|i| vec![Some((i % 2) as f64)]).collect();
    let y: Vec<Option<f64>> = (0..n)
        .map(|i| Some(if i % 2 == 1 { 0.3 } else { 0.0 } + noise(i)))
        .collect();
    let fit = |bonf: bool| {
        ModelTree::fit(
            &x,
            &y,
            ModelTreeParams {
                max_depth: 3,
                min_rows: 10,
                leaf: LeafKind::Mean,
                l2: 0.0,
                split_test: SplitTest::Chow,
                split_alpha: 0.05,
                bonferroni: bonf,
                ..Default::default()
            },
        )
        .unwrap()
        .n_leaves()
    };
    let (plain, bf) = (fit(false), fit(true));
    assert!(bf <= plain, "bonferroni gates more: plain={plain} bf={bf}");
}

// ---- split_cols: the split search's column subset (knob 2 of #3) ----

#[test]
fn split_cols_restrict_split_search_only() {
    // Both columns carry signal; split_cols=[1] must split only on column 1
    // while the leaf model may still use every column.
    let x: Vec<Vec<Option<f64>>> = (0..300)
        .map(|i| {
            vec![
                Some((i % 10) as f64 / 10.0),
                Some(((i * 3) % 7) as f64 / 7.0),
            ]
        })
        .collect();
    let y: Vec<Option<f64>> = (0..300)
        .map(|i| Some(2.0 * x[i][1].unwrap() - 0.5 * x[i][0].unwrap() + 0.05 * noise(i)))
        .collect();
    let tree = ModelTree::fit(
        &x,
        &y,
        ModelTreeParams {
            max_depth: 3,
            min_rows: 10,
            l2: 0.0,
            split_cols: Some(vec![1]),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(tree.n_leaves() > 1);
    assert!(
        tree.splits().iter().all(|s| s.col == 1),
        "splits must be on col 1 only: {:?}",
        tree.splits()
    );
    let (leaf, _) = tree.leaves()[0].clone();
    let m = leaf.linear.as_ref().unwrap();
    assert_eq!(m.cols, vec![0, 1], "leaves still see all columns");
}

#[test]
fn split_cols_excludes_the_signal_column() {
    // Signal lives in column 1; split_cols=[0] bars it from the scan, so
    // every split must sit on column 0 (or none happen).
    let x: Vec<Vec<Option<f64>>> = (0..300)
        .map(|i| vec![Some(noise(i)), Some((i % 2) as f64)])
        .collect();
    let y: Vec<Option<f64>> = (0..300)
        .map(|i| Some(if i % 2 == 1 { 3.0 } else { 0.0 } + 0.05 * noise(i + 7)))
        .collect();
    let tree = ModelTree::fit(
        &x,
        &y,
        ModelTreeParams {
            max_depth: 3,
            min_rows: 10,
            l2: 0.0,
            split_cols: Some(vec![0]),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        tree.splits().iter().all(|s| s.col == 0),
        "no split may touch col 1: {:?}",
        tree.splits()
    );
}

#[test]
fn split_cols_rejects_out_of_range() {
    let x: Vec<Vec<Option<f64>>> = (0..20).map(|i| vec![Some(i as f64)]).collect();
    let y: Vec<Option<f64>> = (0..20).map(|i| Some(i as f64)).collect();
    let err = ModelTree::fit(
        &x,
        &y,
        ModelTreeParams {
            split_cols: Some(vec![0, 5]),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(err.contains("split_cols"), "{err}");
}

// ---- honesty: structure/estimation sample split (knob 3 of #3) ----

#[test]
fn honesty_fits_leaves_on_estimation_rows() {
    // 200 rows, honesty_fraction=0.2: 40 structure rows drive the growth
    // (node counts), the 160 estimation rows fit the leaves (leaf stats).
    let x: Vec<Vec<Option<f64>>> = (0..200)
        .map(|i| vec![Some((i % 2) as f64), Some((i % 5) as f64 / 5.0)])
        .collect();
    let y: Vec<Option<f64>> = (0..200)
        .map(|i| Some(2.0 * x[i][1].unwrap() + 0.05 * noise(i)))
        .collect();
    let fit = |honesty: bool, frac: f64| {
        ModelTree::fit(
            &x,
            &y,
            ModelTreeParams {
                max_depth: 2,
                min_rows: 5,
                l2: 0.0,
                honesty,
                honesty_fraction: frac,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let honest = fit(true, 0.2);
    let est_rows: usize = honest.leaves().iter().map(|(m, _)| m.stats.n).sum();
    assert_eq!(est_rows, 160, "leaves must fit the estimation rows only");
    // Node counts are structure-side (print/prune stay growth-consistent).
    let node_rows: usize = honest.leaves().iter().map(|(_, n)| n).sum();
    assert_eq!(node_rows, 40, "node counts are the structure rows");

    // Determinism: same seed, same tree.
    let again = fit(true, 0.2);
    assert_eq!(honest.predict(&x), again.predict(&x));

    // Default off: leaves fit all rows (the estimation set IS the node).
    let plain = fit(false, 0.2);
    let all_rows: usize = plain.leaves().iter().map(|(m, _)| m.stats.n).sum();
    assert_eq!(all_rows, 200);
}

#[test]
fn honesty_fraction_controls_the_split() {
    let x: Vec<Vec<Option<f64>>> = (0..200).map(|i| vec![Some((i % 4) as f64)]).collect();
    let y: Vec<Option<f64>> = (0..200).map(|i| Some((i % 4) as f64 + 0.05 * noise(i))).collect();
    let est = |frac: f64| {
        ModelTree::fit(
            &x,
            &y,
            ModelTreeParams {
                max_depth: 2,
                min_rows: 10,
                l2: 0.0,
                honesty: true,
                honesty_fraction: frac,
                ..Default::default()
            },
        )
        .unwrap()
        .leaves()
        .iter()
        .map(|(m, _)| m.stats.n)
        .sum::<usize>()
    };
    assert_eq!(est(0.2), 160, "20% structure -> 80% estimation");
    assert_eq!(est(0.8), 40, "80% structure -> 20% estimation");
}

// ---- NA surrogate splits (maxsurrogate; knob 4 of #3) ----

#[test]
fn surrogate_routes_missing_rows() {
    // x0 carries the break but goes missing on 1 of 20 rows; x1 is a
    // jittery mirror of x0 (always observed) — noisier than x0-with-holes,
    // so x0 always wins the split, while x1 stays a >0.5-agreement mimic.
    // With maxsurrogate=1 the missing-x0 rows route by the x1 surrogate.
    let x: Vec<Vec<Option<f64>>> = (0..300)
        .map(|i| {
            let v = i as f64 / 300.0;
            vec![
                if i % 20 == 0 { None } else { Some(v) },
                Some(v + 0.2 * noise(i)),
            ]
        })
        .collect();
    let y: Vec<Option<f64>> = (0..300)
        .map(|i| Some(if i as f64 / 300.0 >= 0.5 { 5.0 } else { 0.0 }))
        .collect();
    let fit = |x: &[Vec<Option<f64>>], maxsurrogate: usize| {
        ModelTree::fit(
            x,
            &y,
            ModelTreeParams {
                max_depth: 2,
                min_rows: 10,
                leaf: LeafKind::Mean,
                l2: 0.0,
                maxsurrogate,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let with_s = fit(&x, 1);
    assert!(
        with_s.splits().iter().any(|s| s.col == 0),
        "the primary split must be on x0: {:?}",
        with_s.splits()
    );
    let p_lo = with_s.predict(&[vec![None, Some(0.1)]])[0];
    let p_hi = with_s.predict(&[vec![None, Some(0.9)]])[0];
    assert!(p_lo < 1.0 && p_hi > 4.0, "surrogate must route both: {p_lo} {p_hi}");

    // The mirror case: x1 = 1 - x0 (flipped mimic) must also route.
    let xm: Vec<Vec<Option<f64>>> = (0..300)
        .map(|i| {
            let v = i as f64 / 300.0;
            vec![
                if i % 20 == 0 { None } else { Some(v) },
                Some(1.0 - v + 0.2 * noise(i)),
            ]
        })
        .collect();
    let m = fit(&xm, 1);
    let p_lo = m.predict(&[vec![None, Some(0.9)]])[0]; // x1=0.9 -> x0=0.1
    let p_hi = m.predict(&[vec![None, Some(0.1)]])[0]; // x1=0.1 -> x0=0.9
    assert!(p_lo < 1.0 && p_hi > 4.0, "flipped surrogate must route: {p_lo} {p_hi}");
}

#[test]
fn surrogate_default_off_keeps_missing_left() {
    // x1 carries no signal here, so the primary is x0 and the two probe
    // rows (x0 missing) both follow missing_left -> equal predictions.
    let x: Vec<Vec<Option<f64>>> = (0..300)
        .map(|i| {
            let v = i as f64 / 300.0;
            vec![if i % 20 == 0 { None } else { Some(v) }, Some(noise(i))]
        })
        .collect();
    let y: Vec<Option<f64>> = (0..300)
        .map(|i| Some(if i as f64 / 300.0 >= 0.5 { 5.0 } else { 0.0 }))
        .collect();
    let t = ModelTree::fit(
        &x,
        &y,
        ModelTreeParams {
            max_depth: 1,
            min_rows: 10,
            leaf: LeafKind::Mean,
            l2: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(t.splits().iter().any(|s| s.col == 0), "primary must be x0");
    let a = t.predict(&[vec![None, Some(0.1)]])[0];
    let b = t.predict(&[vec![None, Some(0.9)]])[0];
    assert_eq!(a, b, "no surrogates -> one missing direction for both");
}

// ---- pruning family: cost-complexity (prune="cp"; knob 5 of #3) ----

#[test]
fn cp_pruning_sweeps_leaf_counts() {
    // Multi-regime target: unpruned grows several leaves; a huge cp_alpha
    // collapses everything to one leaf; counts fall monotonically.
    let x: Vec<Vec<Option<f64>>> = (0..400)
        .map(|i| vec![Some((i % 8) as f64 / 8.0)])
        .collect();
    let y: Vec<Option<f64>> = (0..400)
        .map(|i| Some((i % 8) as f64 * (i % 8) as f64 * 0.2 + 0.05 * noise(i)))
        .collect();
    let leaves = |cp_alpha: f64| {
        ModelTree::fit(
            &x,
            &y,
            ModelTreeParams {
                max_depth: 4,
                min_rows: 10,
                leaf: LeafKind::Mean,
                l2: 0.0,
                prune: PruneCost::Cp,
                cp_alpha,
                ..Default::default()
            },
        )
        .unwrap()
        .n_leaves()
    };
    let (l0, l1, l2, l3) = (leaves(0.0), leaves(0.5), leaves(5.0), leaves(1e9));
    assert!(l0 >= l1 && l1 >= l2 && l2 >= l3, "monotone: {l0} {l1} {l2} {l3}");
    assert_eq!(l3, 1, "a huge cp_alpha collapses to one leaf");
    assert!(l0 > 1, "the unpruned tree grows: {l0}");
}

#[test]
fn cp_alpha_rejects_negative() {
    let x: Vec<Vec<Option<f64>>> = (0..20).map(|i| vec![Some(i as f64)]).collect();
    let y: Vec<Option<f64>> = (0..20).map(|i| Some(i as f64)).collect();
    let err = ModelTree::fit(
        &x,
        &y,
        ModelTreeParams {
            prune: PruneCost::Cp,
            cp_alpha: -1.0,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(err.contains("cp_alpha"), "{err}");
}

// ---- three data handlers (integer / float / levels) ----

#[test]
fn integer_handler_scans_exact_uniques() {
    // Planted break between 3 and 4; with split_precision=2.0 the float
    // handler would snap candidates to a 2.0 grid, but the integer handler
    // scans exact unique values (midpoint cut at 3.5).
    let x: Vec<Vec<Option<f64>>> = (0..100).map(|i| vec![Some((i % 10) as f64)]).collect();
    let y: Vec<Option<f64>> = (0..100)
        .map(|i| Some(if i % 10 <= 3 { 0.0 } else { 5.0 } + 0.02 * noise(i)))
        .collect();
    let fit = |dt: ColType| {
        ModelTree::fit(
            &x,
            &y,
            ModelTreeParams {
                max_depth: 1,
                min_rows: 5,
                leaf: LeafKind::Mean,
                l2: 0.0,
                split_precision: Some(2.0),
                dtypes: vec![dt],
                ..Default::default()
            },
        )
        .unwrap()
    };
    let int = fit(ColType::Integer);
    let thr = int.splits()[0].threshold;
    assert!((thr - 3.5).abs() < 1e-9, "integer handler cuts exactly: {thr}");
    let flt = fit(ColType::Float);
    let thr = flt.splits()[0].threshold;
    assert!(
        (thr / 2.0 - (thr / 2.0).round()).abs() < 1e-9,
        "float handler cuts on the precision grid: {thr}"
    );
}

// ---- knob batch close: leaf-cap / min trio / purity gate / preselect ----

#[test]
fn leaf_cap_bounds_leaves_exactly() {
    let x: Vec<Vec<Option<f64>>> = (0..400)
        .map(|i| vec![Some((i % 8) as f64 / 8.0)])
        .collect();
    let y: Vec<Option<f64>> = (0..400)
        .map(|i| Some((i % 8) as f64 * 0.7 + 0.05 * noise(i)))
        .collect();
    for cap in [1usize, 2, 3, 5] {
        let tree = ModelTree::fit(
            &x,
            &y,
            ModelTreeParams {
                max_depth: 6,
                min_rows: 5,
                leaf: LeafKind::Mean,
                l2: 0.0,
                max_leaf_nodes: Some(cap),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(tree.n_leaves() <= cap, "cap {cap}: {} leaves", tree.n_leaves());
    }
}

#[test]
fn min_trio_gates_split_eligibility() {
    let x: Vec<Vec<Option<f64>>> = (0..200)
        .map(|i| vec![Some((i % 4) as f64 / 4.0)])
        .collect();
    let y: Vec<Option<f64>> = (0..200)
        .map(|i| Some((i % 4) as f64 * 1.5 + 0.05 * noise(i)))
        .collect();
    let fit = |extra: ModelTreeParams| {
        ModelTree::fit(
            &x,
            &y,
            ModelTreeParams {
                max_depth: 3,
                min_rows: 5,
                leaf: LeafKind::Mean,
                l2: 0.0,
                ..extra
            },
        )
        .unwrap()
        .n_leaves()
    };
    let base = fit(ModelTreeParams::default());
    let gated_split = fit(ModelTreeParams {
        minsplit: Some(10_000),
        ..Default::default()
    });
    assert_eq!(gated_split, 1, "huge minsplit keeps the root a leaf");
    let gated_bucket = fit(ModelTreeParams {
        minbucket: Some(10_000),
        ..Default::default()
    });
    assert_eq!(gated_bucket, 1, "huge minbucket keeps the root a leaf");
    let balanced = fit(ModelTreeParams {
        minprob: 0.45,
        ..Default::default()
    });
    assert!(
        balanced <= base,
        "minprob cannot grow the tree: {balanced} vs {base}"
    );
}

#[test]
fn purity_gate_stops_noisy_pure_nodes() {
    // tiny noise everywhere: ungated splits (best-of-noise beats min_gain);
    // the purity gate sees std << threshold and keeps one leaf.
    let x: Vec<Vec<Option<f64>>> = (0..300).map(|i| vec![Some(noise(i + 3))]).collect();
    let y: Vec<Option<f64>> = (0..300).map(|i| Some(5.0 + 0.01 * noise(i))).collect();
    let fit = |gate: Option<f64>| {
        ModelTree::fit(
            &x,
            &y,
            ModelTreeParams {
                max_depth: 3,
                min_rows: 5,
                leaf: LeafKind::Mean,
                l2: 0.0,
                purity_gate: gate,
                ..Default::default()
            },
        )
        .unwrap()
        .n_leaves()
    };
    let ungated = fit(None);
    let gated = fit(Some(0.1));
    assert_eq!(gated, 1, "the gate sees no signal left");
    assert!(ungated >= 1);
}

#[test]
fn preselect_family_screens_noise_columns() {
    // one signal column among five noise columns; both strategies must
    // drop the noise columns (all splits land on the signal).
    let x: Vec<Vec<Option<f64>>> = (0..300)
        .map(|i| {
            let mut row: Vec<Option<f64>> = (0..5).map(|j| Some(noise(i * 7 + j))).collect();
            row.push(Some((i % 2) as f64));
            row
        })
        .collect();
    let y: Vec<Option<f64>> = (0..300)
        .map(|i| Some(if i % 2 == 1 { 4.0 } else { 0.0 } + 0.05 * noise(i)))
        .collect();
    for ps in [
        mltreelib_core::preselect::Preselect::Winnow,
        // Boruta keeps anything beating ONE shadow (a coin flip on pure
        // noise); the strict screener is shadowmax (beats all shadows).
        mltreelib_core::preselect::Preselect::ShadowMax,
    ] {
        let tree = ModelTree::fit(
            &x,
            &y,
            ModelTreeParams {
                max_depth: 2,
                min_rows: 5,
                leaf: LeafKind::Mean,
                l2: 0.0,
                preselect: ps,
                min_relevance: Some(0.15),
                seed: 42,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            tree.splits().iter().all(|s| s.col == 5),
            "{ps:?} must screen to the signal column: {:?}",
            tree.splits()
        );
    }
}

#[test]
fn unitree_catches_nonmonotone_signal_winnow_misses() {
    // y = (x - 0.5)^2: zero correlation (winnow blind), fully explained by
    // a 2-split surrogate tree (unitree keeps it).
    let x: Vec<Vec<Option<f64>>> = (0..300)
        .map(|i| vec![Some(i as f64 / 300.0)])
        .collect();
    let y: Vec<Option<f64>> = (0..300)
        .map(|i| {
            let v = i as f64 / 300.0;
            Some((v - 0.5) * (v - 0.5) * 4.0 + 0.01 * noise(i))
        })
        .collect();
    let (kept, r2) =
        mltreelib_core::preselect::uni_tree_r2(&x_cols(&x), &yv(&y), &wones(300), 0.1, 42);
    assert_eq!(kept, vec![0], "unitree sees the U-shape: r2={r2:?}");
    let (kept_w, rel) =
        mltreelib_core::preselect::winnow(&x, &yv(&y), &wones(300), 0.1);
    assert!(kept_w.is_empty(), "winnow is blind to it: rel={rel:?}");
}

#[test]
fn variance_drops_constant_columns() {
    let mut x: Vec<Vec<Option<f64>>> = Vec::new();
    for i in 0..50 {
        x.push(vec![Some(3.0), Some(i as f64)]);
    }
    let y: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let w = vec![1.0; 50];
    let kept = mltreelib_core::preselect::variance(&x_cols(&x), &y, &w, 0.0);
    assert_eq!(kept, vec![1], "the constant column is dropped");
}

// column-major helper for the preselect unit tests
fn x_cols(x: &[Vec<Option<f64>>]) -> Vec<Vec<Option<f64>>> {
    let d = x.first().map(|r| r.len()).unwrap_or(0);
    let mut cols: Vec<Vec<Option<f64>>> = vec![Vec::with_capacity(x.len()); d];
    for row in x {
        for (j, v) in row.iter().enumerate() {
            cols[j].push(*v);
        }
    }
    cols
}
fn yv(y: &[Option<f64>]) -> Vec<f64> {
    y.iter().map(|v| v.unwrap_or(0.0)).collect()
}
fn wones(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

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
                            left.iter().map(|&i| oracle_scan_design(&x[i], &chosen)).collect();
                        let rs: Vec<Vec<f64>> =
                            right.iter().map(|&i| oracle_scan_design(&x[i], &chosen)).collect();
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
    let rows: Vec<Vec<f64>> = side_rows.iter().map(|&i| design(&x[i])).collect();
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
    let tree = ModelTree::fit_weighted(&x, &yopt, &w, p).unwrap();
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

// Family tests use the public fitted-tree seam: coefficients are on the
// link scale; predictions are conditional response means.
#[test]
fn binomial_grouped_mle_recovers_log_odds_with_case_weights() {
    use mltreelib_core::family::LeafFamily;
    let x = vec![vec![Some(0.0)], vec![Some(0.0)], vec![Some(1.0)], vec![Some(1.0)]];
    let y = vec![Some(0.0), Some(1.0), Some(0.0), Some(1.0)];
    let tree = ModelTree::fit_weighted(
        &x, &y, &[3.0, 1.0, 1.0, 3.0],
        ModelTreeParams {
            max_depth: 0, min_rows: 1, l1: 0.0, l2: 0.0,
            leaf_family: LeafFamily::Binomial,
            ..Default::default()
        },
    ).unwrap();
    let leaves = tree.leaves();
    let linear = leaves[0].0.linear.as_ref().unwrap();
    assert!((linear.intercept + 3.0_f64.ln()).abs() < 1e-7, "{linear:?}");
    assert!((linear.coef[0] - 2.0 * 3.0_f64.ln()).abs() < 1e-7, "{linear:?}");
    let predicted = tree.predict(&[vec![Some(0.0)], vec![Some(1.0)]]);
    assert!((predicted[0] - 0.25).abs() < 1e-8, "{predicted:?}");
    assert!((predicted[1] - 0.75).abs() < 1e-8, "{predicted:?}");
}

#[test]
fn poisson_grouped_mle_recovers_log_rate_and_missing_indicator() {
    use mltreelib_core::family::LeafFamily;
    let x = vec![
        vec![Some(0.0)], vec![Some(0.0)],
        vec![Some(1.0)], vec![Some(1.0)],
        vec![None], vec![None],
    ];
    // Weighted group means: 1, 4, 2. Missingness has its own log-rate.
    let y = vec![Some(0.0), Some(2.0), Some(2.0), Some(5.0), Some(1.0), Some(3.0)];
    let tree = ModelTree::fit_weighted(
        &x, &y, &[1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
        ModelTreeParams {
            max_depth: 0, min_rows: 1, l1: 0.0, l2: 0.0,
            leaf_family: LeafFamily::Poisson,
            ..Default::default()
        },
    ).unwrap();
    let leaves = tree.leaves();
    let linear = leaves[0].0.linear.as_ref().unwrap();
    assert!(linear.intercept.abs() < 1e-7, "{linear:?}");
    assert!((linear.coef[0] - 4.0_f64.ln()).abs() < 1e-7, "{linear:?}");
    assert!((linear.na_coef[0] - 2.0_f64.ln()).abs() < 1e-7, "{linear:?}");
    let predicted = tree.predict(&[vec![Some(0.0)], vec![Some(1.0)], vec![None]]);
    for (actual, expected) in predicted.iter().zip([1.0, 4.0, 2.0]) {
        assert!((actual - expected).abs() < 1e-7, "{predicted:?}");
    }
}

#[test]
fn family_pure_targets_and_separation_produce_finite_response_means() {
    use mltreelib_core::family::LeafFamily;
    let x: Vec<_> = (0..12).map(|i| vec![Some((i % 2) as f64)]).collect();
    for (family, target) in [
        (LeafFamily::Binomial, 0.0),
        (LeafFamily::Binomial, 1.0),
        (LeafFamily::Poisson, 0.0),
    ] {
        let tree = ModelTree::fit(
            &x, &vec![Some(target); x.len()],
            ModelTreeParams {
                max_depth: 0, min_rows: 1, l1: 0.0, l2: 0.0,
                leaf_family: family, ..Default::default()
            },
        ).unwrap();
        assert_eq!(tree.predict(&[vec![Some(-1e6)], vec![None], vec![Some(1e6)]]), vec![target; 3]);
    }
    // Duplicate and constant predictors make this separated design singular.
    let x: Vec<_> = (0..20).map(|i| {
        let group = (i / 10) as f64;
        vec![Some(group), Some(group), Some(1.0)]
    }).collect();
    let y: Vec<_> = (0..20).map(|i| Some((i / 10) as f64)).collect();
    let tree = ModelTree::fit(
        &x, &y, ModelTreeParams {
            max_depth: 0, min_rows: 1, l1: 0.0, l2: 0.0,
            leaf_family: LeafFamily::Binomial, ..Default::default()
        },
    ).unwrap();
    let predictions = tree.predict(&x);
    assert!(predictions.iter().all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0));
    assert!(predictions[0] < 1e-6 && predictions[19] > 1.0 - 1e-6, "{predictions:?}");
    let leaves = tree.leaves();
    let model = leaves[0].0.linear.as_ref().unwrap();
    assert!(model.intercept.is_finite() && model.coef.iter().all(|b| b.is_finite()));
}

#[test]
fn family_constant_predictions_and_conditional_cdfs_stay_on_response_scale() {
    use mltreelib_core::family::LeafFamily;
    let x = vec![vec![Some(0.0)]; 4];
    for (family, y, expected_mean, cdf_zero, cdf_one) in [
        (LeafFamily::Binomial, vec![Some(0.0), Some(0.0), Some(0.0), Some(1.0)],
            0.25, 0.75, 1.0),
        (LeafFamily::Poisson, vec![Some(2.0), Some(4.0), Some(4.0), Some(6.0)],
            4.0, (-4.0_f64).exp(), 5.0 * (-4.0_f64).exp()),
    ] {
        let tree = ModelTree::fit(
            &x, &y, ModelTreeParams {
                max_depth: 0, min_rows: 1, leaf: LeafKind::Mean,
                leaf_family: family, ..Default::default()
            },
        ).unwrap();
        assert_eq!(tree.predict(&x), vec![expected_mean; 4]);
        assert_eq!(tree.cdf(&x, -0.1).unwrap(), vec![0.0; 4]);
        assert_eq!(tree.cdf(&x, f64::INFINITY).unwrap(), vec![1.0; 4]);
        for p in tree.cdf(&x, 0.9).unwrap() {
            assert!((p - cdf_zero).abs() < 1e-12, "{family:?}: {p}");
        }
        for p in tree.cdf(&x, 1.9).unwrap() {
            assert!((p - cdf_one).abs() < 1e-12, "{family:?}: {p}");
        }
    }
}

#[test]
fn family_linear_penalties_and_monotone_constraints_are_effective() {
    use mltreelib_core::family::LeafFamily;
    let x = vec![vec![Some(0.0)], vec![Some(0.0)], vec![Some(1.0)], vec![Some(1.0)]];
    for (family, y, w) in [
        (LeafFamily::Binomial, vec![Some(0.0), Some(1.0), Some(0.0), Some(1.0)],
            vec![3.0, 1.0, 1.0, 3.0]),
        (LeafFamily::Poisson, vec![Some(0.0), Some(2.0), Some(2.0), Some(6.0)],
            vec![1.0; 4]),
    ] {
        let fit = |l1, l2, mono| {
            ModelTree::fit_weighted(&x, &y, &w, ModelTreeParams {
                max_depth: 0, min_rows: 1, l1, l2, monotone: mono,
                leaf_family: family, ..Default::default()
            }).unwrap()
        };
        let unpenalized = fit(0.0, 0.0, vec![]);
        let ridge = fit(0.0, 10.0, vec![]);
        let lasso = fit(100.0, 0.0, vec![]);
        let increasing = fit(0.0, 0.0, vec![1]);
        let decreasing = fit(0.0, 0.0, vec![-1]);
        let coefficient = |tree: &ModelTree| tree.leaves()[0].0.linear.as_ref().unwrap().coef[0];
        assert!(coefficient(&ridge) > 0.0 && coefficient(&ridge) < coefficient(&unpenalized));
        assert!(coefficient(&lasso).abs() < 1e-10);
        assert!((coefficient(&increasing) - coefficient(&unpenalized)).abs() < 1e-6);
        assert!(coefficient(&decreasing).abs() < 1e-10);
    }
}

#[test]
fn family_leaf_column_policies_control_link_scale_design() {
    use mltreelib_core::family::LeafFamily;
    let x = vec![vec![Some(0.0)], vec![Some(0.0)], vec![Some(1.0)], vec![Some(1.0)]];
    let y = vec![Some(0.0), Some(1.0), Some(0.0), Some(1.0)];
    for policy in [LeafCols::All, LeafCols::PathUsed] {
        let tree = ModelTree::fit_weighted(&x, &y, &[3.0, 1.0, 1.0, 3.0], ModelTreeParams {
            max_depth: 0, min_rows: 1, l1: 0.0, l2: 0.0,
            leaf_family: LeafFamily::Binomial, leaf_cols: policy,
            ..Default::default()
        }).unwrap();
        let predicted = tree.predict(&[vec![Some(0.0)], vec![Some(1.0)]]);
        let expected = if policy == LeafCols::All { [0.25, 0.75] } else { [0.5, 0.5] };
        for (actual, expected) in predicted.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-7, "{policy:?}: {predicted:?}");
        }
    }
}

// Independent likelihood oracle for constant-response partitions. It does
// not call family likelihood helpers or re-fit the implementation under test.
fn oracle_family_constant_loss(binomial: bool, rows: &[usize], y: &[f64], w: &[f64]) -> f64 {
    let sw: f64 = rows.iter().map(|&r| w[r]).sum();
    let mean = rows.iter().map(|&r| w[r] * y[r]).sum::<f64>() / sw;
    rows.iter().map(|&r| {
        let loss = if binomial {
            if y[r] == 1.0 { -2.0 * mean.ln() } else { -2.0 * (1.0 - mean).ln() }
        } else if y[r] == 0.0 {
            2.0 * mean
        } else {
            2.0 * (y[r] * (y[r] / mean).ln() - y[r] + mean)
        };
        w[r] * loss
    }).sum()
}

#[test]
fn family_numeric_deviance_scan_matches_weighted_brute_force_with_missing_directions() {
    use mltreelib_core::family::LeafFamily;
    let x: Vec<_> = [Some(0.0), Some(1.0), Some(2.0), Some(3.0), None]
        .into_iter().flat_map(|value| [vec![value], vec![value]]).collect();
    let w = vec![4.0, 1.0, 3.0, 1.0, 2.0, 3.0, 1.0, 4.0, 1.0, 4.0];
    for (family, y) in [
        (LeafFamily::Binomial, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        (LeafFamily::Poisson, vec![0.0, 2.0, 1.0, 3.0, 3.0, 8.0, 4.0, 10.0, 6.0, 12.0]),
    ] {
        let binomial = family == LeafFamily::Binomial;
        let all: Vec<_> = (0..y.len()).collect();
        let parent = oracle_family_constant_loss(binomial, &all, &y, &w);
        let mut best = (f64::NEG_INFINITY, 0.0, false);
        for threshold in [0.5, 1.5, 2.5] {
            for missing_left in [true, false] {
                let (left, right): (Vec<_>, Vec<_>) = all.iter().copied()
                    .partition(|&r| x[r][0].map(|v| v < threshold).unwrap_or(missing_left));
                let gain = parent - oracle_family_constant_loss(binomial, &left, &y, &w)
                    - oracle_family_constant_loss(binomial, &right, &y, &w);
                if gain > best.0 { best = (gain, threshold, missing_left); }
            }
        }
        let tree = ModelTree::fit_weighted(
            &x, &y.iter().copied().map(Some).collect::<Vec<_>>(), &w,
            ModelTreeParams {
                max_depth: 1, min_rows: 1, minprob: 0.0, split_precision: None,
                leaf: LeafKind::Mean, leaf_family: family, gain: GainFn::Deviance,
                prune: PruneCost::None, ..Default::default()
            },
        ).unwrap();
        let splits = tree.splits();
        assert_eq!(splits.len(), 1, "{family:?}");
        assert_eq!(splits[0].threshold, best.1, "{family:?}");
        assert_eq!(splits[0].missing_left, best.2, "{family:?}");
        assert!((splits[0].gain - best.0).abs() < 1e-9, "{family:?}: {} vs {}", splits[0].gain, best.0);
    }
}

#[test]
fn family_categorical_linear_deviance_refits_link_models() {
    use mltreelib_core::family::LeafFamily;
    // Category separates two opposite x effects; global linear logit cannot
    // represent the interaction. Codes are not silently one-hot encoded.
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();
    for category in 0..2 {
        for value in 0..2 {
            for target in 0..2 {
                x.push(vec![Some(category as f64), Some(value as f64)]);
                y.push(Some(target as f64));
                w.push(if target == (category ^ value) { 3.0 } else { 1.0 });
            }
        }
    }
    let tree = ModelTree::fit_weighted(&x, &y, &w, ModelTreeParams {
        max_depth: 1, min_rows: 1, minprob: 0.0,
        l1: 0.0, l2: 0.0, split_cols: Some(vec![0]),
        dtypes: vec![ColType::Categorical, ColType::Float],
        leaf_family: LeafFamily::Binomial, gain: GainFn::Deviance,
        prune: PruneCost::None, ..Default::default()
    }).unwrap();
    let splits = tree.splits();
    assert_eq!(splits.len(), 1);
    assert!(splits[0].cat.is_some());
    // Four independent groups each contribute 4 Bernoulli observations:
    // deviance improvement = 16 * 2 * (ln2 + .25ln.25 + .75ln.75).
    let expected_gain = 32.0 * (2.0_f64.ln() + 0.25 * 0.25_f64.ln() + 0.75 * 0.75_f64.ln());
    assert!((splits[0].gain - expected_gain).abs() < 1e-8, "{} vs {expected_gain}", splits[0].gain);
    let predicted = tree.predict(&[
        vec![Some(0.0), Some(0.0)], vec![Some(0.0), Some(1.0)],
        vec![Some(1.0), Some(0.0)], vec![Some(1.0), Some(1.0)],
    ]);
    for (actual, expected) in predicted.iter().zip([0.25, 0.75, 0.75, 0.25]) {
        assert!((actual - expected).abs() < 1e-7, "{predicted:?}");
    }
}

#[test]
fn family_cp_pruning_uses_deviance_not_squared_error() {
    use mltreelib_core::family::LeafFamily;
    let x = vec![vec![Some(0.0)], vec![Some(0.0)], vec![Some(1.0)], vec![Some(1.0)]];
    let y = vec![Some(0.0), Some(1.0), Some(0.0), Some(1.0)];
    // Weighted groups have p=.25 and p=.75. SSE reduction is .5,
    // deviance reduction is 16*(ln2+.25ln.25+.75ln.75) = 2.09299...
    for (alpha, expected_leaves) in [(1.0, 2), (3.0, 1)] {
        let tree = ModelTree::fit_weighted(&x, &y, &[3.0, 1.0, 1.0, 3.0], ModelTreeParams {
            max_depth: 1, min_rows: 1, minprob: 0.0,
            leaf: LeafKind::Mean, leaf_family: LeafFamily::Binomial,
            gain: GainFn::Deviance, prune: PruneCost::Cp, cp_alpha: alpha,
            split_precision: None, ..Default::default()
        }).unwrap();
        assert_eq!(tree.n_leaves(), expected_leaves, "alpha={alpha}");
    }
}

#[test]
fn family_honesty_fits_estimation_distribution_and_predictions() {
    use mltreelib_core::family::LeafFamily;
    let x: Vec<_> = (0..200).map(|i| vec![Some((i % 2) as f64)]).collect();
    for family in [LeafFamily::Binomial, LeafFamily::Poisson] {
        let y: Vec<_> = (0..200).map(|i| Some(if family == LeafFamily::Binomial {
            (i % 5 == 0) as u8 as f64
        } else { (i % 7) as f64 })).collect();
        let params = ModelTreeParams {
            max_depth: 0, min_rows: 2, honesty: true, honesty_fraction: 0.25,
            leaf: LeafKind::Mean, leaf_family: family,
            ..Default::default()
        };
        let tree = ModelTree::fit(&x, &y, params.clone()).unwrap();
        let leaves = tree.leaves();
        assert_eq!(leaves[0].0.stats.n, 150);
        assert_eq!(leaves[0].1, 50);
        assert_eq!(tree.predict(&x), vec![leaves[0].0.stats.mean; 200]);
        assert_eq!(tree.predict(&x), ModelTree::fit(&x, &y, params).unwrap().predict(&x));
    }
}

#[test]
fn family_numerical_failure_is_reported_instead_of_a_successful_constant_fit() {
    use mltreelib_core::family::LeafFamily;
    // Squared design entries overflow. Returning a plausible global mean
    // here would conceal that the requested linear likelihood fit failed.
    let x = vec![vec![Some(-1e300)], vec![Some(1e300)]];
    for (family, y) in [
        (LeafFamily::Binomial, vec![Some(0.0), Some(1.0)]),
        (LeafFamily::Poisson, vec![Some(1.0), Some(4.0)]),
    ] {
        assert!(ModelTree::fit(&x, &y, ModelTreeParams {
            max_depth: 0, min_rows: 1, l1: 0.0, l2: 0.0,
            leaf_family: family, ..Default::default()
        }).is_err(), "{family:?} must report an unrepresentable fit");
    }
}

#[test]
fn poisson_numeric_linear_deviance_refits_and_selects_missing_direction() {
    use mltreelib_core::family::LeafFamily;
    // A two-way count table with an interaction: group 0 and missing share
    // rates (1,4), group 1 reverses them. A global additive log-rate model
    // has MLE rates (1.5,3.5) in every group; split models fit all cells.
    let x = vec![
        vec![Some(0.0), Some(0.0)], vec![Some(0.0), Some(1.0)],
        vec![Some(1.0), Some(0.0)], vec![Some(1.0), Some(1.0)],
        vec![None, Some(0.0)], vec![None, Some(1.0)],
    ];
    let y = vec![Some(1.0), Some(4.0), Some(4.0), Some(1.0), Some(1.0), Some(4.0)];
    let tree = ModelTree::fit_weighted(&x, &y, &[2.0, 2.0, 1.0, 1.0, 3.0, 3.0], ModelTreeParams {
        max_depth: 1, min_rows: 1, minprob: 0.0,
        l1: 0.0, l2: 0.0, split_cols: Some(vec![0]), split_precision: None,
        leaf_family: LeafFamily::Poisson, gain: GainFn::Deviance,
        prune: PruneCost::None, ..Default::default()
    }).unwrap();
    let splits = tree.splits();
    assert_eq!(splits.len(), 1);
    assert_eq!(splits[0].threshold, 0.5);
    assert!(splits[0].missing_left, "missing group must join its matching log-rate model");
    let expected_gain = 2.0 * (
        5.0 * ((1.0_f64 / 1.5).ln() + 4.0 * (4.0_f64 / 3.5).ln())
        + 4.0 * (4.0_f64 / 1.5).ln() + (1.0_f64 / 3.5).ln()
    );
    assert!((splits[0].gain - expected_gain).abs() < 1e-7, "{} vs {expected_gain}", splits[0].gain);
    let predictions = tree.predict(&x);
    for (actual, expected) in predictions.iter().zip([1.0, 4.0, 4.0, 1.0, 1.0, 4.0]) {
        assert!((actual - expected).abs() < 1e-6, "{predictions:?}");
    }
}
