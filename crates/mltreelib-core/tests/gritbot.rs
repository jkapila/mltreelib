use mltreelib_core::gritbot::{ConditionOp, GritBot, GritCondition};
use mltreelib_core::split::ColType;

#[test]
fn gritbot_condition_matches_correctly() {
    let cond_le = GritCondition {
        col: 0,
        col_name: "feature0".into(),
        op: ConditionOp::Le(10.0),
        actual_val: 0.0,
    };
    assert!(cond_le.matches(&[5.0, 1.0]));
    assert!(cond_le.matches(&[10.0, 1.0]));
    assert!(!cond_le.matches(&[10.1, 1.0]));
    assert!(!cond_le.matches(&[f64::NAN, 1.0]));

    let cond_gt = GritCondition {
        col: 1,
        col_name: "feature1".into(),
        op: ConditionOp::Gt(5.0),
        actual_val: 0.0,
    };
    assert!(cond_gt.matches(&[0.0, 6.0]));
    assert!(!cond_gt.matches(&[0.0, 5.0]));
}

#[test]
fn gritbot_detects_contextual_anomaly_and_preserves_context() {
    // 60 cases with pregnant = 1.0, age ~ 28 (std ~ 2)
    // 60 cases with pregnant = 0.0, age ~ 65 (std ~ 3)
    // One anomaly: row 0 has pregnant = 1.0, but age = 75.0 (extreme in pregnant context, normal globally)
    let mut data = Vec::new();
    let col_names = vec!["pregnant".to_string(), "age".to_string()];
    let dtypes = vec![ColType::Integer, ColType::Float];

    // Case 0: Contextual anomaly! (pregnant=1, age=75)
    data.push(vec![1.0, 75.0]);

    // 50 pregnant cases with age 25..32
    for i in 0..50 {
        let age = 26.0 + ((i % 7) as f64);
        data.push(vec![1.0, age]);
    }

    // 50 non-pregnant cases with age 60..78
    for i in 0..50 {
        let age = 62.0 + ((i % 15) as f64);
        data.push(vec![0.0, age]);
    }

    let grit = GritBot::new()
        .with_min_items(20)
        .with_max_norm(3.0)
        .with_sig_threshold(0.05);

    let anomalies = grit.scan(&data, &col_names, &dtypes);

    // Verify case 0 is flagged as anomalous on age
    let case0_anom = anomalies.iter().find(|a| a.case_id == 0 && a.attribute_name == "age");
    assert!(case0_anom.is_some(), "Expected case 0 to be flagged as contextual anomaly on age");

    let anom = case0_anom.unwrap();
    assert_eq!(anom.actual_value, 75.0);
    assert!(anom.z_score >= 3.0);
    assert!(anom.significance <= 0.05);

    // Context should condition on pregnant
    assert!(!anom.conditions.is_empty(), "Contextual anomaly must have conditions");
    let cond = &anom.conditions[0];
    assert_eq!(cond.col_name, "pregnant");
    assert!(cond.matches(&data[0]));

    // Anomaly on age in non-pregnant cases: age 75 is normal in non-pregnant group, so non-pregnant cases near 75 are NOT anomalies
    let non_preg_false_positives: Vec<_> = anomalies
        .iter()
        .filter(|a| a.case_id > 50 && a.attribute_name == "age" && a.actual_value <= 78.0)
        .collect();
    assert_eq!(non_preg_false_positives.len(), 0, "Non-pregnant cases with age <= 78 must not be flagged");
}

#[test]
fn gritbot_detects_global_outlier_with_empty_conditions() {
    let mut data = Vec::new();
    let col_names = vec!["x1".to_string(), "val".to_string()];
    let dtypes = vec![ColType::Float, ColType::Float];

    // 50 normal rows with val ~ 10.0
    for i in 0..50 {
        data.push(vec![i as f64, 10.0 + ((i % 3) as f64) * 0.1]);
    }
    // Plant extreme global outlier: val = 500.0
    data.push(vec![25.0, 500.0]);
    let outlier_id = 50;

    let grit = GritBot::new()
        .with_min_items(20)
        .with_max_norm(3.0)
        .with_sig_threshold(0.05);

    let anomalies = grit.scan(&data, &col_names, &dtypes);

    let found = anomalies.iter().find(|a| a.case_id == outlier_id && a.attribute_name == "val");
    assert!(found.is_some(), "Expected global outlier at row 50 to be flagged");
    let anom = found.unwrap();
    assert_eq!(anom.actual_value, 500.0);
    assert!(anom.conditions.is_empty(), "Global anomaly should have empty conditions");
}

#[test]
fn gritbot_handles_missing_na_without_panics() {
    let mut data = Vec::new();
    let col_names = vec!["feat".to_string(), "target".to_string()];
    let dtypes = vec![ColType::Float, ColType::Float];

    for i in 0..60 {
        let feat = if i % 5 == 0 { f64::NAN } else { i as f64 };
        let target = if i % 7 == 0 { f64::NAN } else { 20.0 + (i as f64) * 0.5 };
        data.push(vec![feat, target]);
    }
    // Add 1 extreme outlier on non-NA target
    data.push(vec![30.0, 9999.0]);

    let grit = GritBot::new().with_min_items(15);
    let anomalies = grit.scan(&data, &col_names, &dtypes);
    assert!(!anomalies.is_empty());
}
