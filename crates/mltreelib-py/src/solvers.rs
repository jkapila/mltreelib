use std::sync::{Arc, Mutex};

use mltreelib_core::solver::generic::{
    self as generic, BasinHoppingOptions, Bounds, ConstraintEvaluator, ConstraintKind,
    LatinHypercubeOptions, MethodOptions, Objective, ObjectiveError, OptimizationRequest,
    OptimizationResult, OptimizerKind, RandomSearchOptions, SharedOptions, SmoothObjective,
};
use mltreelib_core::solver::{fit_dense, FittedGlm, GlmFamily, SolverOptions};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList};

/// Fitted standalone model from `mltreelib.solvers.fit`.
///
/// Holds the intercept, the coefficient vector, the convergence state, and the
/// statistical summary when the fitted family supports inference statistics.
#[pyclass(name = "FitResult", module = "mltreelib._core")]
struct PyFitResult(FittedGlm);

#[pymethods]
impl PyFitResult {
    /// Fitted intercept.
    #[getter]
    fn intercept(&self) -> f64 {
        self.0.intercept
    }

    /// Fitted feature coefficients, without the intercept.
    #[getter]
    fn coef(&self) -> Vec<f64> {
        self.0.coef.clone()
    }

    /// Iterations the solver performed.
    #[getter]
    fn iterations(&self) -> usize {
        self.0.convergence.iterations
    }

    /// Reports whether the solver reached its convergence test.
    #[getter]
    fn converged(&self) -> bool {
        self.0.convergence.converged()
    }

    /// Reason the solver stopped, for example "exact" or "tolerance".
    #[getter]
    fn stop_reason(&self) -> &'static str {
        self.0.convergence.reason.as_str()
    }

    /// Objective value at the fitted coefficients.
    #[getter]
    fn objective(&self) -> f64 {
        self.0.objective
    }

    /// R-style statistical summary, when the fitted family supports it.
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

    /// Predicts the response for one numeric design matrix.
    #[pyo3(signature = (X))]
    #[allow(non_snake_case)]
    fn predict(&self, X: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        self.0.predict(&X).map_err(PyValueError::new_err)
    }

    /// Mean deviance of the fit on one evaluation set.
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

/// Fits a weighted Gaussian or generalized linear model without growing a tree.
///
/// Pass one numeric design matrix and one response per row. Use nonnegative
/// case weights with a positive total weight. The solver picks the exact
/// Gaussian path when it fits the family and the requested penalties.
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
    m.add_class::<PyOptimizationResult>()?;
    m.add_function(wrap_pyfunction!(fit_glm, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)
}

/// Python objective callable wrapped for the generic seam.
struct PyObjective {
    callable: Py<PyAny>,
    dim: usize,
    bounds: Bounds,
    failure: Arc<Mutex<Option<String>>>,
}

impl PyObjective {
    /// Records one callback failure so the Python caller sees the original error.
    fn fail(&self, message: String) -> ObjectiveError {
        if let Ok(mut slot) = self.failure.lock() {
            *slot = Some(message.clone());
        }
        ObjectiveError::Callback(message)
    }

    /// Calls the Python objective with one point.
    fn call_value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        Python::attach(|py| {
            let list = PyList::new(py, x)
                .map_err(|err| self.fail(format!("objective callback failed: {err}")))?;
            let output = self
                .callable
                .call1(py, (list,))
                .map_err(|err| self.fail(format!("objective callback failed: {err}")))?;
            output.extract::<f64>(py).map_err(|_| {
                self.fail("objective must return one finite scalar".to_string())
            })
        })
    }
}

impl Objective for PyObjective {
    fn dim(&self) -> usize {
        self.dim
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        self.call_value(x)
    }
}

/// Objective with an exact gradient from a Python callable.
struct PyGradientObjective {
    base: PyObjective,
    gradient: Py<PyAny>,
}

impl Objective for PyGradientObjective {
    fn dim(&self) -> usize {
        self.base.dim()
    }

    fn bounds(&self) -> &Bounds {
        self.base.bounds()
    }

    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        self.base.call_value(x)
    }
}

impl SmoothObjective for PyGradientObjective {
    fn gradient(&self, x: &[f64]) -> Result<Vec<f64>, ObjectiveError> {
        Python::attach(|py| {
            let list = PyList::new(py, x)
                .map_err(|err| self.base.fail(format!("gradient callback failed: {err}")))?;
            let output = self
                .gradient
                .call1(py, (list,))
                .map_err(|err| self.base.fail(format!("gradient callback failed: {err}")))?;
            output.extract::<Vec<f64>>(py).map_err(|_| {
                self.base.fail("gradient must return one value per parameter".to_string())
            })
        })
    }
}

/// Python constraint callable wrapped for the generic seam.
struct PyConstraintEvaluator {
    callable: Py<PyAny>,
    failure: Arc<Mutex<Option<String>>>,
}

impl PyConstraintEvaluator {
    /// Records one callback failure so the Python caller sees the original error.
    fn fail(&self, message: String) -> String {
        if let Ok(mut slot) = self.failure.lock() {
            *slot = Some(message.clone());
        }
        message
    }
}

impl ConstraintEvaluator for PyConstraintEvaluator {
    fn values(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        Python::attach(|py| {
            let list = PyList::new(py, x)
                .map_err(|err| self.fail(format!("constraint callback failed: {err}")))?;
            let output = self
                .callable
                .call1(py, (list,))
                .map_err(|err| self.fail(format!("constraint callback failed: {err}")))?;
            output.extract::<Vec<f64>>(py).map_err(|_| {
                self.fail("constraint callback must return one value per constraint".to_string())
            })
        })
    }
}

/// Result of one generic optimization run from `mltreelib.solvers.optimize`.
///
/// Reports the best point, the best objective value, the iteration and
/// evaluation counts, the termination status, and optional convergence
/// history.
#[pyclass(name = "OptimizationResult", module = "mltreelib._core")]
struct PyOptimizationResult(OptimizationResult);

#[pymethods]
impl PyOptimizationResult {
    /// Best point found. Finite and inside the bounds.
    #[getter]
    fn best_x(&self) -> Vec<f64> {
        self.0.best_x.clone()
    }

    /// Objective value at the best point.
    #[getter]
    fn best_value(&self) -> f64 {
        self.0.best_value
    }

    /// Solver iterations performed.
    #[getter]
    fn iterations(&self) -> usize {
        self.0.iterations
    }

    /// User objective evaluations performed.
    #[getter]
    fn evaluations(&self) -> usize {
        self.0.evaluations
    }

    /// Termination status. Budget exhaustion never reports success.
    #[getter]
    fn status(&self) -> &'static str {
        self.0.status.as_str()
    }

    /// Reports whether the run reached an accepted optimum.
    #[getter]
    fn converged(&self) -> bool {
        self.0.status.is_success()
    }

    /// Failure detail, when the run did not succeed.
    #[getter]
    fn message(&self) -> Option<String> {
        self.0.message.clone()
    }

    /// Largest constraint violation at the best point.
    #[getter]
    fn max_constraint_violation(&self) -> Option<f64> {
        self.0.max_constraint_violation
    }

    /// Convergence history, when the method records one.
    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
        let Some(history) = &self.0.history else {
            return Ok(None);
        };
        let mut rows = Vec::with_capacity(history.len());
        for entry in history {
            let row = PyDict::new(py);
            row.set_item("iteration", entry.iteration)?;
            row.set_item("value", entry.value)?;
            row.set_item("best_value", entry.best_value)?;
            rows.push(row);
        }
        Ok(Some(rows))
    }
}

/// Minimizes a callable scalar objective inside bounds.
///
/// The objective receives one list of parameter values and returns one finite
/// scalar. The optional `gradient` callable receives the same list and returns
/// one value per parameter. Smooth methods use a central-difference fallback
/// when no gradient is supplied.
/// For COBYLA, `constraints` returns a fixed-length sequence of finite
/// inequality residuals. Nonnegative residuals are feasible.
/// For basin hopping, `local_method` selects the inner optimizer.
#[pyfunction]
#[pyo3(signature = (objective, bounds, *, method="random_search", start=None, max_evaluations=None, max_iterations=None, tol=1e-6, seed=0, gradient=None, samples=None, initial_population=None, constraints=None, constraint_kind="inequality", local_method=None))]
#[allow(clippy::too_many_arguments)]
fn optimize(
    py: Python<'_>,
    objective: Py<PyAny>,
    bounds: Vec<(f64, f64)>,
    method: &str,
    start: Option<Vec<f64>>,
    max_evaluations: Option<usize>,
    max_iterations: Option<usize>,
    tol: f64,
    seed: u64,
    gradient: Option<Py<PyAny>>,
    samples: Option<usize>,
    initial_population: Option<Vec<Vec<f64>>>,
    constraints: Option<Py<PyAny>>,
    constraint_kind: &str,
    local_method: Option<&str>,
) -> PyResult<PyOptimizationResult> {
    let kind = OptimizerKind::from_name(method)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown method {method:?}")))?;
    if local_method.is_some() && kind != OptimizerKind::BasinHopping {
        return Err(PyValueError::new_err("local_method applies only to basin_hopping"));
    }
    let dim = bounds.len();
    let mut lower = Vec::with_capacity(dim);
    let mut upper = Vec::with_capacity(dim);
    for (low, high) in &bounds {
        lower.push(*low);
        upper.push(*high);
    }
    let box_bounds = Bounds::new(lower, upper).map_err(PyValueError::new_err)?;

    let base = PyObjective {
        callable: objective.clone_ref(py),
        dim,
        bounds: box_bounds.clone(),
        failure: Arc::new(Mutex::new(None)),
    };
    // The seam supplies a central-difference gradient when no gradient source
    // is given, so the binding only forwards an exact gradient callable.
    let smooth = gradient.map(|gradient| PyGradientObjective {
        base: PyObjective {
            callable: objective.clone_ref(py),
            dim,
            bounds: box_bounds.clone(),
            failure: base.failure.clone(),
        },
        gradient: gradient.clone_ref(py),
    });

    let shared = SharedOptions {
        max_evaluations,
        max_iterations,
        tol,
        seed,
        initial_population,
        verbose: 0,
    };
    let objective_ref: &dyn Objective = match &smooth {
        Some(smooth) => smooth as &dyn Objective,
        None => &base as &dyn Objective,
    };
    let smooth_ref: Option<&dyn SmoothObjective> = smooth
        .as_ref()
        .map(|smooth| smooth as &dyn SmoothObjective);

    let mut request = OptimizationRequest::new(objective_ref, smooth_ref, kind, shared);
    let constraint_kind = match constraint_kind {
        "inequality" => ConstraintKind::Inequality,
        "equality" => ConstraintKind::Equality,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown constraint kind {other:?}"
            )));
        }
    };
    let constraint_evaluator = constraints.map(|callable| PyConstraintEvaluator {
        callable: callable.clone_ref(py),
        failure: base.failure.clone(),
    });
    if let Some(evaluator) = &constraint_evaluator {
        request = request.with_constraints(evaluator, constraint_kind);
    }
    match kind {
        OptimizerKind::RandomSearch => {
            request = request.with_options(MethodOptions::Random(RandomSearchOptions {
                samples: samples.unwrap_or(64),
            }));
        }
        OptimizerKind::LatinHypercube => {
            request = request.with_options(MethodOptions::LatinHypercube(LatinHypercubeOptions {
                samples: samples.unwrap_or(64),
            }));
        }
        OptimizerKind::BasinHopping => {
            if let Some(name) = local_method {
                let inner = OptimizerKind::from_name(name)
                    .ok_or_else(|| PyValueError::new_err(format!("Unknown local method {name:?}")))?;
                request = request.with_options(MethodOptions::BasinHopping(BasinHoppingOptions {
                    inner,
                    ..BasinHoppingOptions::default()
                }));
            }
        }
        _ => {}
    }
    if let Some(start) = start {
        request = request.with_start(start);
    }

    let result = generic::optimize(&request).map_err(PyValueError::new_err)?;
    if let Some(message) = base.failure.lock().ok().and_then(|slot| slot.clone()) {
        return Err(PyRuntimeError::new_err(message));
    }
    Ok(PyOptimizationResult(result))
}
