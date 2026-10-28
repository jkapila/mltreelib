//! Thin PyO3 shell around `mltreelib-core`. Exposed to Python as
//! `mltreelib._core`; the Python package in `python/mltreelib` re-exports it.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyModuleMethods};

use mltreelib_core::builder::{DatasetBuilder, NaPolicy};
use mltreelib_core::frame::{Column, Dataset};
use mltreelib_core::generator::{generate as generate_data, GeneratedData, GeneratorParams, TailBehavior, Task};
use mltreelib_core::index::IndexMapper;
use mltreelib_core::schema::NCategoryRule;
use mltreelib_core::timefeat::{calendar_parts, lag_by_days as lag_core, parse_ymd};
use mltreelib_core::transform::{
    summarize as summarize_stats, GroupedScaler, GroupedStrategy, GroupedTransform, SummaryKind,
};

fn generated_to_dict<'py>(py: Python<'py>, gd: &GeneratedData) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let cols = PyDict::new(py);
    for c in gd.dataset.columns() {
        match c {
            Column::Numeric { name, values } => {
                cols.set_item(name, values.clone())?;
            }
            Column::Categorical { name, codes, vocab } => {
                let strs: Vec<Option<String>> = codes
                    .iter()
                    .map(|code| code.map(|i| vocab[i as usize].clone()))
                    .collect();
                cols.set_item(name, strs)?;
            }
        }
    }
    out.set_item("columns", cols)?;
    out.set_item("target", gd.target.clone())?;
    match &gd.labels {
        Some(labels) => out.set_item("labels", labels.clone())?,
        None => out.set_item("labels", py.None())?,
    }

    let truth = PyDict::new(py);
    truth.set_item("numeric_betas", gd.truth.numeric_betas.clone())?;
    let cat_eff = PyDict::new(py);
    for (k, v) in &gd.truth.categorical_effects {
        cat_eff.set_item(k, v.clone())?;
    }
    truth.set_item("categorical_effects", cat_eff)?;
    truth.set_item("hierarchy_effects", gd.truth.hierarchy_effects.clone())?;
    let var = PyDict::new(py);
    let v = &gd.truth.variance;
    var.set_item("numeric", v.numeric)?;
    var.set_item("categorical", v.categorical)?;
    var.set_item("hierarchy", v.hierarchy)?;
    var.set_item("interaction", v.interaction)?;
    var.set_item("noise", v.noise)?;
    var.set_item("total", v.total)?;
    truth.set_item("variance", var)?;
    out.set_item("truth", truth)?;
    Ok(out)
}

/// Generate an adjustable benchmark dataset with known ground truth.
///
/// Knobs mirror `mltreelib_core::generator::GeneratorParams`; see its docs for
/// semantics. Deterministic for a given `seed`.
#[pyfunction]
#[pyo3(signature = (
    n_rows=10_000,
    n_numeric=5,
    cardinalities=None,
    hierarchy_levels=3,
    branching=None,
    group_effect_strength=0.5,
    hierarchy_effect_strength=0.3,
    interaction_strength=0.2,
    noise=0.1,
    task="regression",
    n_classes=2,
    tail_behavior="gaussian",
    missingness=0.0,
    seed=42,
))]
#[allow(clippy::too_many_arguments)]
fn generate<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_numeric: usize,
    cardinalities: Option<Vec<usize>>,
    hierarchy_levels: usize,
    branching: Option<Vec<usize>>,
    group_effect_strength: f64,
    hierarchy_effect_strength: f64,
    interaction_strength: f64,
    noise: f64,
    task: &str,
    n_classes: usize,
    tail_behavior: &str,
    missingness: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let defaults = GeneratorParams::default();
    let params = GeneratorParams {
        n_rows,
        n_numeric,
        cardinalities: cardinalities.unwrap_or(defaults.cardinalities),
        hierarchy_levels,
        branching: branching.unwrap_or(defaults.branching),
        group_effect_strength,
        hierarchy_effect_strength,
        interaction_strength,
        noise,
        task: Task::parse(task, n_classes).map_err(PyValueError::new_err)?,
        tail_behavior: TailBehavior::parse(tail_behavior).map_err(PyValueError::new_err)?,
        missingness,
        seed,
    };
    let gd = generate_data(&params).map_err(PyValueError::new_err)?;
    generated_to_dict(py, &gd)
}

/// Per-row group summary of `values` within groups defined by `keys`.
#[pyfunction]
#[pyo3(signature = (keys, values, kind="mean"))]
fn grouped_transform(
    keys: Vec<Option<String>>,
    values: Vec<Option<f64>>,
    kind: &str,
) -> PyResult<Vec<Option<f64>>> {
    let k = SummaryKind::parse(kind).map_err(PyValueError::new_err)?;
    let mut gt = GroupedTransform::new();
    gt.fit(&keys, &values, k).map_err(PyValueError::new_err)?;
    gt.transform(&keys).map_err(PyValueError::new_err)
}

/// Summary statistics of observed values (NAs skipped).
#[pyfunction]
fn summarize<'py>(py: Python<'py>, values: Vec<Option<f64>>) -> PyResult<Bound<'py, PyDict>> {
    let s = summarize_stats(&values);
    let d = PyDict::new(py);
    d.set_item("count", s.count)?;
    d.set_item("mean", s.mean)?;
    d.set_item("std", s.std)?;
    d.set_item("min", s.min)?;
    d.set_item("q1", s.q1)?;
    d.set_item("median", s.median)?;
    d.set_item("q3", s.q3)?;
    d.set_item("max", s.max)?;
    Ok(d)
}

/// Categorical string <-> stable code mapping (sklearn-style fit/transform).
#[pyclass(name = "IndexMapper")]
struct PyIndexMapper {
    inner: IndexMapper,
}

#[pymethods]
impl PyIndexMapper {
    #[new]
    fn new() -> Self {
        Self {
            inner: IndexMapper::new(),
        }
    }

    /// Learn the vocabulary of `name` (first-seen order). Re-fitting resets it.
    fn fit(&mut self, name: &str, values: Vec<Option<String>>) -> PyResult<()> {
        self.inner.fit(name, &values);
        Ok(())
    }

    /// Codes for `values`; unseen levels map to None (missing).
    fn transform(&self, name: &str, values: Vec<Option<String>>) -> PyResult<Vec<Option<u32>>> {
        self.inner
            .transform(name, &values)
            .map_err(PyValueError::new_err)
    }

    fn inverse_transform(
        &self,
        name: &str,
        codes: Vec<Option<u32>>,
    ) -> PyResult<Vec<Option<String>>> {
        self.inner
            .inverse_transform(name, &codes)
            .map_err(PyValueError::new_err)
    }

    fn get_index(&self, name: &str, value: &str) -> PyResult<Option<u32>> {
        self.inner
            .get_index(name, value)
            .map_err(PyValueError::new_err)
    }

    fn get_name(&self, name: &str, index: u32) -> PyResult<String> {
        self.inner
            .get_name(name, index)
            .map_err(PyValueError::new_err)
    }

    fn get_all_names(&self, name: &str) -> PyResult<Vec<String>> {
        self.inner
            .get_all_names(name)
            .map_err(PyValueError::new_err)
    }
}

fn dataset_to_dict<'py>(py: Python<'py>, ds: &Dataset) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let cols = PyDict::new(py);
    let dtypes = PyDict::new(py);
    for c in ds.columns() {
        match c {
            Column::Numeric { name, values } => {
                cols.set_item(name, values.clone())?;
                dtypes.set_item(name, "numeric")?;
            }
            Column::Categorical { name, codes, vocab } => {
                let strs: Vec<Option<String>> = codes
                    .iter()
                    .map(|code| code.map(|i| vocab[i as usize].clone()))
                    .collect();
                cols.set_item(name, strs)?;
                dtypes.set_item(name, "categorical")?;
            }
        }
    }
    out.set_item("columns", cols)?;
    out.set_item("dtypes", dtypes)?;
    out.set_item("n_rows", ds.n_rows())?;
    out.set_item("n_cols", ds.n_cols())?;
    Ok(out)
}

/// Ingest any data into the typed dataset: `n_category` category inference,
/// NA policy (allow/drop/impute), digit rounding, optional intercept column.
#[pyclass(name = "DatasetBuilder")]
struct PyDatasetBuilder {
    inner: DatasetBuilder,
}

#[pymethods]
impl PyDatasetBuilder {
    #[new]
    #[pyo3(signature = (n_category=None, na_policy="allow", digits=None, add_intercept=false))]
    fn new(
        n_category: Option<f64>,
        na_policy: &str,
        digits: Option<u32>,
        add_intercept: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: DatasetBuilder::new()
                .n_category(n_category.map(NCategoryRule::parse))
                .na_policy(NaPolicy::parse(na_policy).map_err(PyValueError::new_err)?)
                .digits(digits)
                .add_intercept(add_intercept),
        })
    }

    fn add_numeric(&mut self, name: &str, values: Vec<Option<f64>>) {
        let inner = std::mem::take(&mut self.inner);
        self.inner = inner.add_numeric(name, values);
    }

    fn add_categorical(&mut self, name: &str, values: Vec<Option<String>>) {
        let inner = std::mem::take(&mut self.inner);
        self.inner = inner.add_categorical(name, values);
    }

    /// Build the typed dataset; returns `columns`, `dtypes`, `n_rows`, `n_cols`.
    fn build(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let inner = std::mem::take(&mut self.inner);
        let ds = inner.build().map_err(PyValueError::new_err)?;
        Ok(dataset_to_dict(py, &ds)?.unbind())
    }
}

/// One-shot group scaling: fit on (keys, values) and transform the same rows.
#[pyfunction]
#[pyo3(signature = (keys, values, strategy="zscore"))]
fn grouped_scale(
    keys: Vec<Option<String>>,
    values: Vec<Option<f64>>,
    strategy: &str,
) -> PyResult<Vec<Option<f64>>> {
    let s = GroupedStrategy::parse(strategy).map_err(PyValueError::new_err)?;
    let mut gs = GroupedScaler::new();
    gs.fit(&keys, &values, s).map_err(PyValueError::new_err)?;
    gs.transform(&keys, &values).map_err(PyValueError::new_err)
}

/// Fit/transform group distributional scaling with exact inverse
/// (sklearn-shaped: fit once, then transform / inverse_transform).
#[pyclass(name = "GroupedScaler")]
struct PyGroupedScaler {
    inner: GroupedScaler,
}

#[pymethods]
impl PyGroupedScaler {
    #[new]
    fn new() -> Self {
        Self {
            inner: GroupedScaler::new(),
        }
    }

    /// Fit group statistics of `values` under the given strategy.
    fn fit(&mut self, keys: Vec<Option<String>>, values: Vec<Option<f64>>, strategy: &str) -> PyResult<()> {
        let s = GroupedStrategy::parse(strategy).map_err(PyValueError::new_err)?;
        self.inner
            .fit(&keys, &values, s)
            .map_err(PyValueError::new_err)?;
        Ok(())
    }

    fn transform(&self, keys: Vec<Option<String>>, values: Vec<Option<f64>>) -> PyResult<Vec<Option<f64>>> {
        self.inner
            .transform(&keys, &values)
            .map_err(PyValueError::new_err)
    }

    fn inverse_transform(
        &self,
        keys: Vec<Option<String>>,
        values: Vec<Option<f64>>,
    ) -> PyResult<Vec<Option<f64>>> {
        self.inner
            .inverse_transform(&keys, &values)
            .map_err(PyValueError::new_err)
    }
}

/// Calendar features for ISO dates (YYYY-MM-DD or timestamp): year, halfyear,
/// quarter, month, week (ISO), iso_year, weekday (0=Mon), day, day_of_year,
/// season, weekend. Missing dates give missing features.
#[pyfunction]
#[pyo3(signature = (dates, hemisphere="north"))]
fn calendar_features<'py>(
    py: Python<'py>,
    dates: Vec<Option<String>>,
    hemisphere: &str,
) -> PyResult<Py<PyDict>> {
    let southern = match hemisphere {
        "north" => false,
        "south" => true,
        other => return Err(PyValueError::new_err(format!("unknown hemisphere '{other}'"))),
    };
    const NAMES: [&str; 11] = [
        "year", "halfyear", "quarter", "month", "week", "iso_year", "weekday", "day",
        "day_of_year", "season", "weekend",
    ];
    let mut cols: Vec<Vec<Option<f64>>> = vec![Vec::with_capacity(dates.len()); NAMES.len()];
    for d in &dates {
        match d {
            None => {
                for c in cols.iter_mut() {
                    c.push(None);
                }
            }
            Some(s) => {
                let (y, m, dd) = parse_ymd(s)
                    .ok_or_else(|| PyValueError::new_err(format!("unparseable date '{s}'")))?;
                for (i, (_, v)) in calendar_parts(y, m, dd, southern).into_iter().enumerate() {
                    cols[i].push(Some(v));
                }
            }
        }
    }
    let out = PyDict::new(py);
    for (name, col) in NAMES.iter().zip(cols) {
        out.set_item(*name, col)?;
    }
    Ok(out.unbind())
}

/// Lag values by whole days within each key (day granularity).
#[pyfunction]
#[pyo3(signature = (keys, dates, values, lag_days))]
fn lag_by_days(
    keys: Vec<Option<String>>,
    dates: Vec<Option<String>>,
    values: Vec<Option<f64>>,
    lag_days: u32,
) -> PyResult<Vec<Option<f64>>> {
    lag_core(&keys, &dates, &values, lag_days).map_err(PyValueError::new_err)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatasetBuilder>()?;
    m.add_class::<PyGroupedScaler>()?;
    m.add_class::<PyIndexMapper>()?;
    m.add_function(wrap_pyfunction!(calendar_features, m)?)?;
    m.add_function(wrap_pyfunction!(generate, m)?)?;
    m.add_function(wrap_pyfunction!(grouped_scale, m)?)?;
    m.add_function(wrap_pyfunction!(grouped_transform, m)?)?;
    m.add_function(wrap_pyfunction!(lag_by_days, m)?)?;
    m.add_function(wrap_pyfunction!(summarize, m)?)?;
    Ok(())
}
