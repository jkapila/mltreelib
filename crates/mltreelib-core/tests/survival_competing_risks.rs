use mltreelib_core::criteria::GainFn;
use mltreelib_core::family::LeafFamily;
use mltreelib_core::leaf::LeafKind;
use mltreelib_core::metrics::competing_risks_report;
use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};

#[test]
fn test_core_survival_logrank_and_martingale_split() {
    let mut x: Vec<Vec<Option<f64>>> = Vec::new();
    let mut durations: Vec<Option<f64>> = Vec::new();
    let mut events = Vec::new();

    // 40 samples in group 0 (longer survival) and 40 in group 1 (shorter survival)
    for i in 0..80 {
        if i < 40 {
            x.push(vec![Some(0.0), Some(i as f64)]);
            durations.push(Some(50.0 + (i as f64) * 0.5));
            events.push(if i % 3 != 0 { 1.0 } else { 0.0 });
        } else {
            x.push(vec![Some(1.0), Some(i as f64)]);
            durations.push(Some(10.0 + ((i - 40) as f64) * 0.2));
            events.push(if i % 4 != 0 { 1.0 } else { 0.0 });
        }
    }

    let mut params = ModelTreeParams::default();
    params.max_depth = 2;
    params.min_rows = 10;
    params.leaf = LeafKind::Mean;
    params.leaf_family = LeafFamily::Survival;
    params.gain = GainFn::LogRank;
    params.events = Some(events.clone());

    let tree = ModelTree::fit(&x, &durations, params).expect("fit logrank tree");
    let eval_times = vec![10.0, 20.0, 40.0, 60.0];

    let surv = tree.predict_survival(&x, &eval_times).expect("predict survival");
    assert_eq!(surv.len(), 80);

    // Group 0 should have higher survival probability at t=20 than Group 1
    assert!(surv[10][1] > surv[60][1]);

    // Test Martingale split gain
    let mut params_m = ModelTreeParams::default();
    params_m.max_depth = 2;
    params_m.min_rows = 10;
    params_m.leaf = LeafKind::Mean;
    params_m.leaf_family = LeafFamily::Survival;
    params_m.gain = GainFn::Martingale;
    params_m.events = Some(events);

    let tree_m = ModelTree::fit(&x, &durations, params_m).expect("fit martingale tree");
    let surv_m = tree_m.predict_survival(&x, &eval_times).expect("predict survival");
    assert_eq!(surv_m.len(), 80);
    assert!(surv_m[10][1] > surv_m[60][1]);
}

#[test]
fn test_core_competing_risks_gray_test_and_simplex() {
    let mut x: Vec<Vec<Option<f64>>> = Vec::new();
    let mut durations: Vec<Option<f64>> = Vec::new();
    let mut events = Vec::new();

    // 90 samples with competing risks (0 = censored, 1 = cause 1, 2 = cause 2)
    for i in 0..90 {
        let feat = (i as f64) / 10.0;
        x.push(vec![Some(feat)]);
        if i % 3 == 0 {
            events.push(0.0);
            durations.push(Some(20.0 + feat));
        } else if i % 3 == 1 {
            events.push(1.0);
            durations.push(Some(5.0 + feat * 0.5));
        } else {
            events.push(2.0);
            durations.push(Some(12.0 + feat * 0.8));
        }
    }

    let mut params = ModelTreeParams::default();
    params.max_depth = 2;
    params.min_rows = 10;
    params.leaf = LeafKind::Mean;
    params.leaf_family = LeafFamily::Survival;
    params.gain = GainFn::GrayTest { cause: 1 };
    params.events = Some(events.clone());

    let tree = ModelTree::fit(&x, &durations, params).expect("fit competing risks tree");
    let eval_times = vec![5.0, 10.0, 15.0, 20.0, 25.0];

    let surv = tree.predict_survival(&x, &eval_times).expect("predict survival");
    let cif1 = tree.predict_cif(&x, &eval_times, 1).expect("predict cif cause 1");
    let cif2 = tree.predict_cif(&x, &eval_times, 2).expect("predict cif cause 2");

    assert_eq!(cif1.len(), 90);
    assert_eq!(cif2.len(), 90);

    for i in 0..90 {
        for t_idx in 0..eval_times.len() {
            let s = surv[i][t_idx];
            let f1 = cif1[i][t_idx];
            let f2 = cif2[i][t_idx];

            assert!(f1 >= 0.0 && f1 <= 1.0);
            assert!(f2 >= 0.0 && f2 <= 1.0);
            assert!(s >= 0.0 && s <= 1.0);

            // Aalen-Johansen Simplex Identity: S(t) + F_1(t) + F_2(t) == 1.0
            let total = s + f1 + f2;
            assert!((total - 1.0).abs() < 1e-6, "Simplex violation: total={total} at t_idx={t_idx}");
        }
    }
}

#[test]
fn test_core_competing_risks_report_metrics() {
    let durations = vec![2.0, 4.0, 7.0, 10.0, 15.0, 20.0];
    let events = vec![1.0, 2.0, 1.0, 0.0, 2.0, 1.0];
    let eval_times = vec![5.0, 12.0, 18.0];

    // Cause 1 & 2 synthetic CIF curves
    let mut cif_curves = Vec::with_capacity(durations.len());
    for _ in 0..durations.len() {
        cif_curves.push(vec![
            vec![0.1, 0.25, 0.4], // Cause 1
            vec![0.05, 0.15, 0.3], // Cause 2
        ]);
    }

    let risk_scores = vec![
        vec![0.5, 0.2],
        vec![0.2, 0.6],
        vec![0.4, 0.3],
        vec![0.1, 0.1],
        vec![0.2, 0.7],
        vec![0.6, 0.2],
    ];

    let report = competing_risks_report(
        &durations,
        &events,
        Some(&cif_curves),
        Some(&risk_scores),
        Some(&eval_times),
    ).expect("competing risks report");

    assert_eq!(report.n_samples, 6);
    assert_eq!(report.causes, vec![1, 2]);
    assert_eq!(report.cause_counts, vec![3, 2]);
    assert_eq!(report.cause_c_indices.len(), 2);
    assert_eq!(report.cause_uno_c_indices.len(), 2);
    assert_eq!(report.cause_ibs.len(), 2);
    assert_eq!(report.brier_at_horizons.len(), 2);
}
