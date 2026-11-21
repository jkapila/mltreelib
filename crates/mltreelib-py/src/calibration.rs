use mltreelib_core::calibration::IsotonicCalibrator;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Fitted isotonic calibration for binary positive-class probabilities.
#[pyclass(name = "IsotonicCalibrator", module = "mltreelib.calibration")]
pub(crate) struct PyIsotonicCalibrator {
    inner: IsotonicCalibrator,
}

#[pymethods]
impl PyIsotonicCalibrator {
    /// Fit on validation or out-of-fold predictions and 0/1 labels.
    #[staticmethod]
    fn fit(y_true: Vec<f64>, probabilities: Vec<f64>) -> PyResult<Self> {
        let inner =
            IsotonicCalibrator::fit(&y_true, &probabilities).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Apply the fitted calibration to positive-class probabilities.
    fn predict(&self, probabilities: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .predict(&probabilities)
            .map_err(PyValueError::new_err)
    }
}
