use mltreelib_core::encode::{
    BaseNEncoder, CatBoostEncoder, CountEncoder, EmbeddingEncoder, EmbeddingTask, HashingEncoder,
    JamesSteinEncoder, LeaveOneOutEncoder, MEstimatorEncoder, OneHotEncoder, QuantileEncoder,
    SumEncoder, TargetEncoder, TfidfEncoder, WoEEncoder,
};

fn s(v: &str) -> Option<String> {
    Some(v.to_string())
}

#[test]
fn one_hot_levels_unseen_and_missing() {
    let fit_vals = vec![s("a"), s("b"), s("a"), None];
    let vals = vec![s("a"), s("b"), s("a"), None, s("zz")];
    let mut e = OneHotEncoder::new();
    e.fit(&fit_vals);
    assert_eq!(e.levels().unwrap(), &["a".to_string(), "b".to_string()]);
    let cols = e.transform(&vals).unwrap();
    assert_eq!(cols.len(), 2);
    assert_eq!(cols[0].1, vec![1.0, 0.0, 1.0, 0.0, 0.0]);
    assert_eq!(cols[1].1, vec![0.0, 1.0, 0.0, 0.0, 0.0]); // unseen -> zeros
    assert!(OneHotEncoder::new().transform(&vals).is_err());
}

#[test]
fn hashing_is_deterministic_and_covers_unseen() {
    let e = HashingEncoder::new(8).unwrap();
    let a = e.transform(&[s("x"), s("y"), s("x"), s("never-seen")]);
    assert_eq!(a.len(), 8);
    // same level -> same bucket
    let xb: Vec<usize> = a.iter().enumerate().filter(|(_, c)| c.1[0] == 1.0).map(|(i, _)| i).collect();
    let xb2: Vec<usize> = a.iter().enumerate().filter(|(_, c)| c.1[2] == 1.0).map(|(i, _)| i).collect();
    assert_eq!(xb, xb2);
    // every non-missing row is in exactly one bucket
    for row in [0usize, 1, 3] {
        let total: f64 = a.iter().map(|c| c.1[row]).sum();
        assert_eq!(total, 1.0);
    }
    assert!(HashingEncoder::new(0).is_err());
}

#[test]
fn target_encoder_smooths_toward_global() {
    let vals = vec![s("a"), s("a"), s("b"), s("b")];
    let y = vec![Some(0.0), Some(0.0), Some(10.0), Some(10.0)];
    let mut e = TargetEncoder::new(2.0).unwrap();
    e.fit(&vals, &y).unwrap();
    let out = e.transform(&vals).unwrap();
    // global mean 5; level means 0 and 10; w = 2/(2+2) = .5 -> 2.5 and 7.5
    assert_eq!(out[0], Some(2.5));
    assert_eq!(out[2], Some(7.5));
    // unseen -> global prior; missing -> missing
    let out = e.transform(&[s("zz"), None]).unwrap();
    assert_eq!(out, vec![Some(5.0), None]);
    assert_eq!(e.prior().unwrap(), 5.0);
    assert!(TargetEncoder::new(-1.0).is_err());
}

#[test]
fn catboost_ordered_stats_are_row_dependent_and_leak_free() {
    let vals = vec![s("a"), s("a"), s("a"), s("a")];
    let y = vec![Some(1.0), Some(3.0), Some(5.0), Some(7.0)];
    let mut e = CatBoostEncoder::new(1.0, 1, 42).unwrap();
    let enc = e.fit_transform(&vals, &y).unwrap();

    // first occurrence of the level in the permutation sees only the prior
    // (global mean 4, prior 1 -> 4.0); later rows see running means
    assert_eq!(enc.iter().filter(|v| **v == Some(4.0)).count(), 1);
    let distinct: std::collections::HashSet<u64> = enc
        .iter()
        .map(|v| v.unwrap().to_bits())
        .collect();
    assert!(distinct.len() > 1, "same level must get row-dependent values");

    // determinism
    let mut e2 = CatBoostEncoder::new(1.0, 1, 42).unwrap();
    assert_eq!(e2.fit_transform(&vals, &y).unwrap(), enc);

    // transform on new data uses fitted level means: (16 + 1*4)/(4+1) = 4.0
    let out = e.transform(&[s("a"), s("zz"), None]).unwrap();
    assert_eq!(out[0], Some(4.0));
    assert_eq!(out[1], Some(4.0)); // unseen -> global
    assert_eq!(out[2], None); // missing -> missing

    assert!(CatBoostEncoder::new(0.0, 1, 1).is_err());
    assert!(CatBoostEncoder::new(1.0, 0, 1).is_err());
}

#[test]
fn tfidf_encoder_class_vectors() {
    // level a: 3x class 0 + 1x class 1; level b: mirror image
    let vals = vec![s("a"), s("a"), s("a"), s("a"), s("b"), s("b"), s("b"), s("b")];
    let y = vec![
        Some(0.0), Some(0.0), Some(0.0), Some(1.0),
        Some(0.0), Some(1.0), Some(1.0), Some(1.0),
    ];
    let mut e = TfidfEncoder::new(false);
    e.fit(&vals, &y, 2).unwrap();
    let out = e.transform(&vals).unwrap();
    // tf(a) = [0.75, 0.25]; class totals are equal so idf matches on both dims
    let idf = (9.0f64 / 5.0).ln() + 1.0; // (1+8)/(1+4) = 9/5
    assert!((out[0].as_ref().unwrap()[0] - 0.75 * idf).abs() < 1e-9);
    assert!((out[0].as_ref().unwrap()[1] - 0.25 * idf).abs() < 1e-9);
    // missing -> None; unseen -> global shares [0.5, 0.5] * idf
    let out = e.transform(&[None, s("zz")]).unwrap();
    assert_eq!(out[0], None);
    assert!((out[1].as_ref().unwrap()[0] - 0.5 * idf).abs() < 1e-9);
    // normalized variant yields unit vectors
    let mut n = TfidfEncoder::new(true);
    n.fit(&vals, &y, 2).unwrap();
    let v = n.transform(&[s("a")]).unwrap();
    let v = v[0].as_ref().unwrap();
    assert!((v.iter().map(|x| x * x).sum::<f64>().sqrt() - 1.0).abs() < 1e-9);
    assert!(TfidfEncoder::new(true).transform(&vals).is_err());
}

#[test]
fn woe_encoder_signs_and_fallbacks() {
    // level a = pure class 1, level b = pure class 0
    let vals = vec![s("a"), s("a"), s("b"), s("b")];
    let y = vec![Some(1.0), Some(1.0), Some(0.0), Some(0.0)];
    let mut e = WoEEncoder::new(1.0).unwrap();
    e.fit(&vals, &y, 2).unwrap();
    let out = e.transform(&vals).unwrap();
    let w = out[0].as_ref().unwrap();
    assert!(w[1] > 0.0 && w[0] < 0.0, "level a is over-represented in class 1");
    // unseen -> neutral zeros; missing -> None
    let out = e.transform(&[s("zz"), None]).unwrap();
    assert_eq!(out[0].as_ref().unwrap(), &vec![0.0, 0.0]);
    assert_eq!(out[1], None);
    assert!(WoEEncoder::new(0.0).is_err());
    assert!(WoEEncoder::new(1.0).unwrap().fit(&vals, &y, 1).is_err());
}

#[test]
fn embedding_classifier_learns_and_is_deterministic() {
    let vals = vec![s("a"), s("a"), s("b"), s("b")];
    let y = vec![Some(0.0), Some(0.0), Some(1.0), Some(1.0)];
    let mut e = EmbeddingEncoder::new(4, 0.5, 50, 7, EmbeddingTask::Classification).unwrap();
    e.fit(&vals, &y, Some(2)).unwrap();
    assert_eq!(
        e.predict(&vals).unwrap(),
        vec![Some(0.0), Some(0.0), Some(1.0), Some(1.0)]
    );
    let emb = e.transform(&[s("a"), s("zz"), None]).unwrap();
    assert!(emb[0].as_ref().unwrap().iter().any(|x| x.abs() > 0.0));
    assert_eq!(emb[1].as_ref().unwrap(), &vec![0.0; 4]); // unseen -> zeros
    assert_eq!(emb[2], None);
    let mut e2 = EmbeddingEncoder::new(4, 0.5, 50, 7, EmbeddingTask::Classification).unwrap();
    e2.fit(&vals, &y, Some(2)).unwrap();
    assert_eq!(e2.transform(&vals).unwrap(), e.transform(&vals).unwrap());
}

#[test]
fn embedding_regressor_activation_follows_target_sign() {
    // all-positive targets -> ReLU output: predictions stay non-negative
    let vals = vec![s("a"), s("b"), s("a"), s("b")];
    let y = vec![Some(5.0), Some(0.0), Some(5.0), Some(0.0)];
    let mut e = EmbeddingEncoder::new(3, 0.1, 300, 11, EmbeddingTask::Regression).unwrap();
    e.fit(&vals, &y, None).unwrap();
    let preds = e.predict(&vals).unwrap();
    assert!(preds.iter().all(|p| p.unwrap() >= 0.0), "relu output");

    // a negative target present -> linear output may go negative
    let y2 = vec![Some(-5.0), Some(0.0), Some(-5.0), Some(0.0)];
    let mut e2 = EmbeddingEncoder::new(3, 0.1, 300, 11, EmbeddingTask::Regression).unwrap();
    e2.fit(&vals, &y2, None).unwrap();
    let preds = e2.predict(&vals).unwrap();
    assert!(preds.iter().any(|p| p.unwrap() < 0.0), "linear output");
    assert!(EmbeddingEncoder::new(0, 0.1, 1, 1, EmbeddingTask::Regression).is_err());
}

#[test]
fn count_loo_mestimate_encoders() {
    let vals = vec![s("a"), s("a"), s("b"), None];
    let y = vec![Some(0.0), Some(10.0), Some(4.0), Some(99.0)];

    let mut c = CountEncoder::new(false);
    c.fit(&vals);
    assert_eq!(
        c.transform(&vals).unwrap(),
        vec![Some(2.0), Some(2.0), Some(1.0), None]
    );
    let mut cn = CountEncoder::new(true);
    cn.fit(&vals);
    assert_eq!(cn.transform(&vals).unwrap()[0], Some(2.0 / 3.0));
    assert_eq!(c.transform(&[s("zz")]).unwrap(), vec![Some(0.0)]);

    let mut loo = LeaveOneOutEncoder::new();
    let out = loo.fit_transform(&vals, &y).unwrap();
    assert_eq!(out[0], Some(10.0)); // level a minus own 0
    assert_eq!(out[1], Some(0.0)); // level a minus own 10
    assert!((out[2].unwrap() - 14.0 / 3.0).abs() < 1e-9); // single-obs -> global
    assert_eq!(out[3], None);
    assert_eq!(loo.transform(&[s("a")]).unwrap(), vec![Some(5.0)]);

    let mut m = MEstimatorEncoder::new(2.0).unwrap();
    m.fit(&vals, &y).unwrap();
    let expected = (10.0 + 2.0 * 14.0 / 3.0) / 4.0;
    assert!((m.transform(&[s("a")]).unwrap()[0].unwrap() - expected).abs() < 1e-9);
    assert!(MEstimatorEncoder::new(0.0).is_err());
}

#[test]
fn basen_and_sum_encoders() {
    let vals = vec![s("a"), s("b"), s("c"), s("d"), None, s("zz")];
    let mut b = BaseNEncoder::new(2).unwrap();
    b.fit(&vals[..4]);
    let cols = b.transform(&vals).unwrap();
    assert_eq!(cols.len(), 2); // 4 levels -> 2 binary digits
    assert_eq!((cols[0].1[3], cols[1].1[3]), (1.0, 1.0)); // level d = 3 -> 11
    assert_eq!(cols[0].1[4], 0.0); // missing -> zeros
    assert_eq!(cols[0].1[5], 0.0); // unseen -> zeros
    assert!(BaseNEncoder::new(1).is_err());

    let mut sm = SumEncoder::new();
    sm.fit(&vals[..3]);
    let out = sm.transform(&vals).unwrap();
    assert_eq!(out[0], Some(vec![1.0, 0.0]));
    assert_eq!(out[1], Some(vec![0.0, 1.0]));
    assert_eq!(out[2], Some(vec![-1.0, -1.0])); // last level
    assert_eq!(out[4], None); // missing
    assert_eq!(out[5], Some(vec![0.0, 0.0])); // unseen
}

#[test]
fn quantile_and_jamesstein_encoders() {
    let vals = vec![s("a"), s("a"), s("a"), s("b"), s("b"), s("b")];
    let y = vec![
        Some(0.0), Some(10.0), Some(20.0), Some(100.0), Some(100.0), Some(100.0),
    ];
    let mut q = QuantileEncoder::new(0.5, 0.0).unwrap();
    q.fit(&vals, &y).unwrap();
    let out = q.transform(&vals).unwrap();
    assert_eq!(out[0], Some(10.0)); // median of a
    assert_eq!(out[3], Some(100.0)); // median of b
    assert_eq!(q.transform(&[s("zz")]).unwrap()[0], Some(60.0)); // global median

    // shrinkage needs more than three levels; grand mean of this set is 55
    let vals4 = vec![
        s("a"), s("a"), s("a"), s("b"), s("b"), s("b"),
        s("c"), s("c"), s("c"), s("d"), s("d"), s("d"),
    ];
    let y4 = vec![
        Some(0.0), Some(10.0), Some(20.0),
        Some(100.0), Some(100.0), Some(100.0),
        Some(50.0), Some(50.0), Some(50.0),
        Some(60.0), Some(60.0), Some(60.0),
    ];
    let mut js = JamesSteinEncoder::new();
    js.fit(&vals4, &y4).unwrap();
    let out = js.transform(&vals4).unwrap();
    let a = out[0].unwrap();
    assert!(a > 10.0 && a < 55.0, "shrunk a = {a}");
    let bb = out[3].unwrap();
    assert!(bb > 55.0 && bb < 100.0, "shrunk b = {bb}");
    // fewer than three levels -> everything collapses to the grand mean
    let mut js2 = JamesSteinEncoder::new();
    js2.fit(&[s("a"), s("b")], &[Some(0.0), Some(10.0)]).unwrap();
    assert_eq!(
        js2.transform(&[s("a"), s("b")]).unwrap(),
        vec![Some(5.0), Some(5.0)]
    );
}

#[test]
fn catboost_leakage_is_lower_than_plain_target_mean() {
    // one observation per level: the plain fitted mean equals the row's own
    // target (fully leaked); the ordered statistic cannot see it first.
    let vals = vec![s("a"), s("b")];
    let y = vec![Some(100.0), Some(0.0)];
    let mut plain = TargetEncoder::new(0.0).unwrap();
    plain.fit(&vals, &y).unwrap();
    let plain_out = plain.transform(&vals).unwrap();
    assert_eq!(plain_out[0], Some(100.0)); // its own target = perfect "prediction"

    let mut ordered = CatBoostEncoder::new(1.0, 4, 7).unwrap();
    let ord = ordered.fit_transform(&vals, &y).unwrap();
    assert!(
        (ord[0].unwrap() - 100.0).abs() > 1.0,
        "ordered statistics must not see the row's own target first"
    );
}
