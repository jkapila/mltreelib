use mltreelib_core::solver::{fit_dense, FittedGlm, GlmFamily, SolverOptions};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList};

#[pyclass(name = "FitResult", module = "mltreelib._core")]
struct PyFitResult(FittedGlm);

#[pymethods]
impl PyFitResult {
    #[getter]
    fn intercept(&self) -> f64 {
        self.0.intercept
    }

    #[getter]
    fn coef(&self) -> Vec<f64> {
        self.0.coef.clone()
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.0.convergence.iterations
    }

    #[getter]
    fn converged(&self) -> bool {
        self.0.convergence.converged()
    }

    #[getter]
    fn stop_reason(&self) -> &'static str {
        self.0.convergence.reason.as_str()
    }

    #[getter]
    fn objective(&self) -> f64 {
        self.0.objective
    }

    #[getter]
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(summary) = &self.0.summary else {
            return Ok(None);
        };
        let values = PyDict::new(py);
        values.set_item("n_obs", summary.n_obs)?;
        values.set_item("df_residual", summary.df_residual)?;
        values.set_item("r_squared", summary.r_squared)?;
        values.set_item("adj_r_squared", summary.adj_r_squared)?;
        values.set_item("dispersion", summary.dispersion)?;
        let coefficients: Vec<_> = summary
            .coefficients
            .iter()
            .map(|coef| {
                let row = PyDict::new(py);
                row.set_item("name", &coef.name)?;
                row.set_item("estimate", coef.estimate)?;
                row.set_item("std_err", coef.std_err)?;
                row.set_item("stat_value", coef.stat_value)?;
                row.set_item("p_value", coef.p_value)?;
                Ok::<_, PyErr>(row)
            })
            .collect::<PyResult<_>>()?;
        values.set_item("coefficients", PyList::new(py, coefficients)?)?;
        Ok(Some(values))
    }

    #[pyo3(signature = (X))]
    #[allow(non_snake_case)]
    fn predict(&self, X: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        self.0.predict(&X).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (X, y, *, sample_weight=None))]
    #[allow(non_snake_case)]
    fn mean_deviance(
        &self,
        X: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<f64> {
        self.0
            .mean_deviance(&X, &y, sample_weight.as_deref())
            .map_err(PyValueError::new_err)
    }
}

#[pyfunction]
#[pyo3(signature = (X, y, *, family="gaussian", solver="auto", l1=0.0, l2=0.0, sample_weight=None, monotone=None, max_iter=250, tol=1e-9, seed=42, tweedie_p=None, tobit_threshold=None))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn fit_glm(
    py: Python<'_>,
    X: Vec<Vec<f64>>,
    y: Vec<f64>,
    family: &str,
    solver: &str,
    l1: f64,
    l2: f64,
    sample_weight: Option<Vec<f64>>,
    monotone: Option<Vec<i8>>,
    max_iter: usize,
    tol: f64,
    seed: u64,
    tweedie_p: Option<f64>,
    tobit_threshold: Option<f64>,
) -> PyResult<PyFitResult> {
    if (tweedie_p.is_some() && family != "tweedie")
        || (tobit_threshold.is_some() && family != "tobit")
    {
        return Err(PyValueError::new_err(
            "Family-specific options must match family",
        ));
    }
    let family = match family {
        "gaussian" => GlmFamily::Gaussian,
        "binomial" | "logit" => GlmFamily::Binomial,
        "probit" => GlmFamily::Probit,
        "poisson" => GlmFamily::Poisson,
        "gamma" => GlmFamily::Gamma,
        "tweedie" => GlmFamily::Tweedie {
            p: tweedie_p.unwrap_or(1.5),
        },
        "tobit" => GlmFamily::Tobit {
            threshold: tobit_threshold.unwrap_or(0.0),
        },
        other => return Err(PyValueError::new_err(format!("Unknown family {other:?}"))),
    };
    use mltreelib_core::leaf::LeafSolver;
    let monotone = monotone.unwrap_or_default();
    let constrained = monotone.iter().any(|&s| s != 0);
    let solver = match solver {
        "direct" => None,
        "auto" if family == GlmFamily::Gaussian => {
            (l1 > 0.0 || constrained).then_some(LeafSolver::Ccd)
        }
        "auto" if l1 > 0.0 || constrained => Some(LeafSolver::Irls),
        "auto" => Some(LeafSolver::Lbfgs),
        name => Some(super::parse_leaf_solver(name)?),
    };
    let options = SolverOptions {
        family,
        l1,
        l2,
        monotone,
        max_iter,
        tol,
        verbose: 0,
    };
    py.detach(|| fit_dense(&X, &y, sample_weight.as_deref(), solver, &options, seed))
        .map(PyFitResult)
        .map_err(PyValueError::new_err)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFitResult>()?;
    m.add_function(wrap_pyfunction!(fit_glm, m)?)
}
