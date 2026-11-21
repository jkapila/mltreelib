use mltreelib_core::calibration::IsotonicCalibrator;

#[test]
fn pools_tied_scores_and_adjacent_violations() {
    let labels = [1.0, 0.0, 0.0, 1.0, 0.0];
    let scores = [0.2, 0.2, 0.4, 0.9, 0.1];
    let calibrator = IsotonicCalibrator::fit(&labels, &scores).unwrap();
    let predictions = calibrator
        .predict(&[0.0, 0.2, 0.3, 0.4, 0.65, 1.0])
        .unwrap();
    for (prediction, expected) in
        predictions
            .iter()
            .zip([0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    {
        assert!((prediction - expected).abs() < 1e-12);
    }
}

#[test]
fn rejects_invalid_labels_and_probabilities() {
    assert!(IsotonicCalibrator::fit(&[0.5], &[0.8]).is_err());
    assert!(IsotonicCalibrator::fit(&[0.0], &[]).is_err());
    let calibrator = IsotonicCalibrator::fit(&[0.0], &[0.8]).unwrap();
    assert!(calibrator.predict(&[f64::NAN]).is_err());
    assert!(calibrator.predict(&[1.1]).is_err());
}
