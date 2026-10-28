//! Thin PyO3 shell around `mltreelib-core`. Exposed to Python as
//! `mltreelib._core`; the Python package in `python/mltreelib` re-exports it.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyModuleMethods};

use mltreelib_core::builder::{DatasetBuilder, NaPolicy};
use mltreelib_core::encode::{
    BaseNEncoder, CatBoostEncoder, CountEncoder, EmbeddingEncoder, EmbeddingTask, HashingEncoder,
    JamesSteinEncoder, LeaveOneOutEncoder, MEstimatorEncoder, OneHotEncoder, QuantileEncoder,
    SumEncoder, TargetEncoder, TfidfEncoder, WoEEncoder,
};
use mltreelib_core::frame::{Column, Dataset};
use mltreelib_core::model_tree::{LeafCols, LeafKind, ModelTree, ModelTreeParams};
use mltreelib_core::metrics::{
    classification_report, grouped_regression_report, regression_report, ClassificationReport,
    RegressionReport,
};
use mltreelib_core::generator::{generate as generate_data, GeneratedData, GeneratorParams, TailBehavior, Task};
use mltreelib_core::index::IndexMapper;
use mltreelib_core::schema::NCategoryRule;
use mltreelib_core::timefeat::{
    calendar_parts, decomposed_transform as decomposed_core, lag_by_days as lag_core, parse_ymd,
    trended_transform as trend_core, windowed_transform as windowed_core,
};
use mltreelib_core::transform::{
    summarize as summarize_stats, GroupedScaler, GroupedStrategy, GroupedTransform, SummaryKind,
    VariableScaler,
};

/// Rows missing key/date/value are dropped from time-series estimation.
fn incomplete_rows(
    keys: &[Option<String>],
    dates: &[Option<String>],
    values: &[Option<f64>],
) -> usize {
    keys.iter()
        .zip(dates)
        .zip(values)
        .filter(|((k, d), v)| k.is_none() || d.is_none() || v.is_none() || v.is_some_and(|x| x.is_nan()))
        .count()
}

fn warn_incomplete(py: Python<'_>, fn_name: &str, n: usize) -> PyResult<()> {
    if n > 0 {
        let warnings = py.import("warnings")?;
        warnings.call_method1(
            "warn",
            (format!(
                "{fn_name}: {n} rows with missing key/date/value dropped from estimation; outputs left as missing"
            ),),
        )?;
    }
    Ok(())
}

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
/// The knobs control how much signal lives in numeric effects, categorical
/// levels, hierarchical groups and their interaction — so learners can be
/// scored on **structure recovery**, not just accuracy. Deterministic for a
/// given `seed`.
///
/// Args:
///     n_rows: number of rows to generate.
///     n_numeric: numeric feature columns; the first two carry linear signal.
///     cardinalities: level count per categorical column (each >= 2).
///     hierarchy_levels: depth of the categorical hierarchy (0 = none).
///     branching: child count per hierarchy level (last value repeats).
///     group_effect_strength: scale of categorical level effects.
///     hierarchy_effect_strength: scale of hierarchy node effects.
///     interaction_strength: scale of the numeric x categorical interaction.
///     noise: scale of the residual noise.
///     task: "regression" or "classification".
///     n_classes: class count for classification (equal-frequency cuts).
///     tail_behavior: "gaussian", "heavy" (Student-t residuals) or "bathtub"
///         (noise grows with the signal — the DAR motivation).
///     missingness: fraction of feature cells set to missing, in [0, 1).
///     seed: RNG seed.
///
/// Returns: dict with `columns` (name -> values, None = missing), `target`,
///     `labels` (classification only) and `truth` — the true effects plus
///     `truth["variance"]`, the realized variance share of each component.
///
/// Raises:
///     ValueError: invalid parameter combination.
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
///
/// Args:
///     keys: group label per row (None = missing).
///     values: numeric value per row (None/NaN = missing; excluded from the
///         group statistics but counted in each group's `n_missing`).
///     kind: "mean", "sum", "std", "count", "min", "max", "median",
///         "q1" or "q3".
///
/// Returns: the summary of each row's group; rows with a missing key/value or
///     an unseen group give None.
///
/// Raises:
///     ValueError: unknown kind or mismatched lengths.
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

/// Summary of observed values, with missing counts shown by default.
///
/// NAs and NaNs are excluded from the statistics but always reported in
/// `n_missing` — nothing is silently hidden.
///
/// Args:
///     values: numeric values (None/NaN = missing).
///
/// Returns: dict with `count` (observed), `n_missing`, `sum`, `mean`, `std`,
///     `min`, `q1`, `median`, `q3`, `max` (NaN for the statistics of an
///     all-missing input).
#[pyfunction]
fn summarize<'py>(py: Python<'py>, values: Vec<Option<f64>>) -> PyResult<Bound<'py, PyDict>> {
    let s = summarize_stats(&values);
    let d = PyDict::new(py);
    d.set_item("count", s.count)?;
    d.set_item("n_missing", s.n_missing)?;
    d.set_item("sum", s.sum)?;
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

    /// Map codes back to their string levels.
    ///
    /// Args:
    ///     name: fitted column name.
    ///     codes: codes per row (None = missing).
    ///
    /// Returns: string levels (None where the code is missing).
    ///
    /// Raises:
    ///     ValueError: column not fitted or code out of range.
    fn inverse_transform(&self, name: &str, codes: Vec<Option<u32>>) -> PyResult<Vec<Option<String>>> {
        self.inner
            .inverse_transform(name, &codes)
            .map_err(PyValueError::new_err)
    }

    /// Code for one level (None when the level is unseen).
    ///
    /// Raises:
    ///     ValueError: column not fitted.
    fn get_index(&self, name: &str, value: &str) -> PyResult<Option<u32>> {
        self.inner
            .get_index(name, value)
            .map_err(PyValueError::new_err)
    }

    /// Level for one code.
    ///
    /// Raises:
    ///     ValueError: column not fitted or code out of range.
    fn get_name(&self, name: &str, index: u32) -> PyResult<String> {
        self.inner
            .get_name(name, index)
            .map_err(PyValueError::new_err)
    }

    /// All fitted levels in code order.
    ///
    /// Raises:
    ///     ValueError: column not fitted.
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
    let na_counts = PyDict::new(py);
    for c in ds.columns() {
        na_counts.set_item(c.name(), c.na_count())?;
    }
    out.set_item("na_counts", na_counts)?;
    out.set_item("n_rows", ds.n_rows())?;
    out.set_item("n_cols", ds.n_cols())?;
    Ok(out)
}

/// Ingest any data into the typed dataset.
///
/// Owns the ingestion semantics: `n_category` category inference, explicit NA
/// policy, digit rounding and an optional intercept column. Missing is `None`
/// **or NaN** — both are counted in `na_counts`, and filling is opt-in only.
///
/// Args (constructor):
///     n_category: integral numeric columns with at most this many distinct
///         levels become categorical; values < 1 mean a fraction of rows
///         (None = no inference).
///     na_policy: "allow" (keep missing), "drop" (drop rows containing any
///         missing) or "impute" (numeric mean, categorical mode).
///     digits: round floats to this many decimal places (None = no rounding).
///     add_intercept: append a constant 1.0 column named `intercept`.
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

    /// Add a numeric column. NaN is treated as missing.
    ///
    /// Args:
    ///     name: column name.
    ///     values: one value per row (None/NaN = missing).
    fn add_numeric(&mut self, name: &str, values: Vec<Option<f64>>) {
        let inner = std::mem::take(&mut self.inner);
        self.inner = inner.add_numeric(name, values);
    }

    /// Add a categorical column from strings; the vocabulary is first-seen
    /// order. None = missing.
    ///
    /// Args:
    ///     name: column name.
    ///     values: one level per row (None = missing).
    fn add_categorical(&mut self, name: &str, values: Vec<Option<String>>) {
        let inner = std::mem::take(&mut self.inner);
        self.inner = inner.add_categorical(name, values);
    }

    /// Build the typed dataset with the configured policies applied.
    ///
    /// Returns: dict with `columns` (name -> values), `dtypes` ("numeric" or
    ///     "categorical"), `na_counts` (missing per column), `n_rows`,
    ///     `n_cols`.
    ///
    /// Raises:
    ///     ValueError: inconsistent column lengths, duplicate names, invalid
    ///         policy usage (e.g. "drop" removing every row).
    fn build(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let inner = std::mem::take(&mut self.inner);
        let ds = inner.build().map_err(PyValueError::new_err)?;
        Ok(dataset_to_dict(py, &ds)?.unbind())
    }
}

/// One-shot group scaling: fit on (keys, values) and transform the same rows.
///
/// Each row's value is scaled by its group's summary under `strategy`
/// (centered family like "zscore" or ratio family like "mean"; see
/// `GroupedScaler` for the full list and the exact inverse).
///
/// Args:
///     keys: group label per row (None = missing).
///     values: numeric value per row (None/NaN = missing).
///     strategy: one of "zscore", "min-max", "median-iqr", "mean-iqr",
///         "median-qd", "median-std", "median-range", "mean-range",
///         "min-qd", "min-iqr", "min", "max", "mean", "median", "iqr",
///         "qd", "std", "range".
///
/// Returns: scaled values; rows with a missing key/value, an unseen group or
///     a degenerate (zero) scale give None — missing in, missing out.
///
/// Raises:
///     ValueError: unknown strategy or mismatched lengths.
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
///
/// Strategies: centered family — "zscore", "min-max", "median-iqr",
/// "mean-iqr", "median-qd", "median-std", "median-range", "mean-range",
/// "min-qd", "min-iqr"; ratio family — "min", "max", "mean", "median",
/// "iqr", "qd", "std", "range". Missing in -> missing out; degenerate
/// (zero) scales give missing, never inf.
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
    ///
    /// Args:
    ///     keys: group label per row (None = missing).
    ///     values: numeric value per row (None/NaN = missing; counted in
    ///         each group's `n_missing`).
    ///     strategy: strategy name (see the class docstring).
    ///
    /// Raises:
    ///     ValueError: unknown strategy or mismatched lengths.
    fn fit(&mut self, keys: Vec<Option<String>>, values: Vec<Option<f64>>, strategy: &str) -> PyResult<()> {
        let s = GroupedStrategy::parse(strategy).map_err(PyValueError::new_err)?;
        self.inner
            .fit(&keys, &values, s)
            .map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Scale values against the fitted group statistics.
    ///
    /// Args:
    ///     keys: group label per row.
    ///     values: numeric value per row.
    ///
    /// Returns: scaled values; None for missing key/value, unseen groups and
    ///     degenerate scales.
    ///
    /// Raises:
    ///     ValueError: not fitted or mismatched lengths.
    fn transform(&self, keys: Vec<Option<String>>, values: Vec<Option<f64>>) -> PyResult<Vec<Option<f64>>> {
        self.inner
            .transform(&keys, &values)
            .map_err(PyValueError::new_err)
    }

    /// Exact inverse of `transform` under the fitted statistics.
    ///
    /// Args:
    ///     keys: group label per row.
    ///     values: previously transformed values.
    ///
    /// Returns: original-scale values (up to floating-point error).
    ///
    /// Raises:
    ///     ValueError: not fitted or mismatched lengths.
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

/// Calendar features for ISO dates (YYYY-MM-DD or timestamp prefix).
///
/// Produces year, halfyear, quarter, month, week (ISO), iso_year,
/// weekday (0 = Monday), day, day_of_year, season and weekend as numeric
/// parts, ready for tree learners.
///
/// Args:
///     dates: ISO date strings or timestamps (None = missing).
///     hemisphere: "north" or "south" (southern shifts the season by six
///         months).
///
/// Returns: dict of 11 feature lists aligned with `dates`; rows with a
///     missing date give None features.
///
/// Raises:
///     ValueError: unparseable date or unknown hemisphere.
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
///
/// Each row receives the value of its key at `date - lag_days`; duplicate
/// (key, date) rows keep the last occurrence. Incomplete rows (missing
/// key/date/value) are dropped from estimation and reported via a warning.
///
/// Args:
///     keys: series label per row (None = missing).
///     dates: ISO date strings per row (None = missing).
///     values: numeric values per row (None/NaN = missing).
///     lag_days: lag in whole days (0 returns the row's own value).
///
/// Returns: lagged values; None where no source row exists.
///
/// Raises:
///     ValueError: unparseable date or mismatched lengths.
#[pyfunction]
#[pyo3(signature = (keys, dates, values, lag_days))]
fn lag_by_days(
    py: Python<'_>,
    keys: Vec<Option<String>>,
    dates: Vec<Option<String>>,
    values: Vec<Option<f64>>,
    lag_days: u32,
) -> PyResult<Vec<Option<f64>>> {
    warn_incomplete(py, "lag_by_days", incomplete_rows(&keys, &dates, &values))?;
    lag_core(&keys, &dates, &values, lag_days).map_err(PyValueError::new_err)
}

/// Grouped summary trended at a time granularity.
///
/// Each row gets the summary of its (key x time-bucket) cell — e.g. with
/// granularity "year_month" and kind "mean", the row's key-and-month mean.
/// Incomplete rows (missing key/date/value) are dropped from estimation and
/// reported via a warning.
///
/// Args:
///     keys: series label per row (None = missing).
///     dates: ISO date strings per row (None = missing).
///     values: numeric values per row (None/NaN = missing).
///     granularity: bucket spec joined by "_" from {year, halfyear, season,
///         quarter, month, week, weekday, day} — e.g. "year_month",
///         "year_week", "month_weekday".
///     kind: summary kind ("mean", "sum", "std", "count", "min", "max",
///         "median", "q1", "q3").
///     hemisphere: "north" or "south" (affects the season component).
///
/// Returns: per-row cell summaries; incomplete rows give None.
///
/// Raises:
///     ValueError: unknown granularity/kind/hemisphere, unparseable date or
///         mismatched lengths.
#[pyfunction]
#[pyo3(signature = (keys, dates, values, granularity="year_month", kind="mean", hemisphere="north"))]
fn trended_transform(
    py: Python<'_>,
    keys: Vec<Option<String>>,
    dates: Vec<Option<String>>,
    values: Vec<Option<f64>>,
    granularity: &str,
    kind: &str,
    hemisphere: &str,
) -> PyResult<Vec<Option<f64>>> {
    let k = SummaryKind::parse(kind).map_err(PyValueError::new_err)?;
    let southern = match hemisphere {
        "north" => false,
        "south" => true,
        other => return Err(PyValueError::new_err(format!("unknown hemisphere '{other}'"))),
    };
    warn_incomplete(py, "trended_transform", incomplete_rows(&keys, &dates, &values))?;
    trend_core(&keys, &dates, &values, granularity, k, southern).map_err(PyValueError::new_err)
}

/// Per-column distributional scaling — each variable transformed
/// independently (legacy `VariableTransformation` semantics).
///
/// Same strategy names as `GroupedScaler`, applied per column, with exact
/// `inverse_transform`. None/NaN are missing and pass through as missing;
/// degenerate scales give missing, never inf.
#[pyclass(name = "VariableScaler")]
struct PyVariableScaler {
    inner: VariableScaler,
}

#[pymethods]
impl PyVariableScaler {
    #[new]
    fn new() -> Self {
        Self {
            inner: VariableScaler::new(),
        }
    }

    /// Fit all columns at once under the given strategy.
    ///
    /// Args:
    ///     columns: {name: values} for every column to scale (None/NaN =
    ///         missing; counted in each column's `n_missing`).
    ///     strategy: strategy name (see `GroupedScaler`).
    ///
    /// Raises:
    ///     ValueError: unknown strategy.
    fn fit(
        &mut self,
        columns: std::collections::HashMap<String, Vec<Option<f64>>>,
        strategy: &str,
    ) -> PyResult<()> {
        let s = GroupedStrategy::parse(strategy).map_err(PyValueError::new_err)?;
        let refs: Vec<(&str, &[Option<f64>])> = columns
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        self.inner.fit(&refs, s).map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Scale one fitted column's values.
    ///
    /// Args:
    ///     name: fitted column name.
    ///     values: numeric values (None/NaN = missing).
    ///
    /// Returns: scaled values; None for missing inputs and degenerate scales.
    ///
    /// Raises:
    ///     ValueError: not fitted or unknown column.
    fn transform(&self, name: &str, values: Vec<Option<f64>>) -> PyResult<Vec<Option<f64>>> {
        self.inner
            .transform(name, &values)
            .map_err(PyValueError::new_err)
    }

    /// Exact inverse of `transform` under the fitted statistics.
    ///
    /// Args:
    ///     name: fitted column name.
    ///     values: previously transformed values.
    ///
    /// Returns: original-scale values (up to floating-point error).
    ///
    /// Raises:
    ///     ValueError: not fitted or unknown column.
    fn inverse_transform(&self, name: &str, values: Vec<Option<f64>>) -> PyResult<Vec<Option<f64>>> {
        self.inner
            .inverse_transform(name, &values)
            .map_err(PyValueError::new_err)
    }
}

/// Additive decomposition, following the statsmodels `seasonal_decompose`
/// (additive) convention.
///
/// The trend is a centered moving average over each key's rows ordered by
/// date; the seasonal component is the mean of (series - trend) per
/// (key, season bucket), estimated from rows with a complete trend window.
/// Incomplete rows (missing key/date/value) are dropped from estimation and
/// reported via a warning.
///
/// Args:
///     keys: series label per row (None = missing).
///     dates: ISO date strings per row (None = missing).
///     values: numeric values per row (None/NaN = missing).
///     season_granularity: bucket spec for the seasonal component (e.g.
///         "month", "weekday" — same vocabulary as `trended_transform`).
///     trend_window: centered moving-average window in rows (>= 1).
///     hemisphere: "north" or "south" (affects the season component).
///
/// Returns: dict with `detrended` (series - trend), `deseasonalized`
///     (series - seasonal) and `residual` (series - trend - seasonal);
///     incomplete rows give None.
///
/// Raises:
///     ValueError: trend_window == 0, unknown granularity/hemisphere,
///         unparseable date or mismatched lengths.
#[pyfunction]
#[pyo3(signature = (keys, dates, values, season_granularity="month", trend_window=3, hemisphere="north"))]
fn decomposed_transform<'py>(
    py: Python<'py>,
    keys: Vec<Option<String>>,
    dates: Vec<Option<String>>,
    values: Vec<Option<f64>>,
    season_granularity: &str,
    trend_window: usize,
    hemisphere: &str,
) -> PyResult<Py<PyDict>> {
    let southern = match hemisphere {
        "north" => false,
        "south" => true,
        other => return Err(PyValueError::new_err(format!("unknown hemisphere '{other}'"))),
    };
    warn_incomplete(
        py,
        "decomposed_transform",
        incomplete_rows(&keys, &dates, &values),
    )?;
    let d = decomposed_core(&keys, &dates, &values, season_granularity, trend_window, southern)
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("detrended", d.detrended)?;
    out.set_item("deseasonalized", d.deseasonalized)?;
    out.set_item("residual", d.residual)?;
    Ok(out.unbind())
}

/// Trailing-window aggregation per key over day windows.
///
/// Each row summarizes the observed values of its key in a window of
/// `window_days` days; incomplete rows (missing key/date/value) are dropped
/// from estimation and reported via a warning.
///
/// Args:
///     keys: series label per row (None = missing).
///     dates: ISO date strings per row (None = missing).
///     values: numeric values per row (None/NaN = missing).
///     window_days: window length in days (>= 1); covering the `window_days`
///         days ending at the row's date, or strictly earlier days when
///         `include_current` is false.
///     kind: summary kind ("mean", "sum", "std", "count", "min", "max",
///         "median", "q1", "q3").
///     include_current: include the row's own date in the window.
///
/// Returns: per-row window summaries; None for incomplete rows and for
///     empty/all-missing windows.
///
/// Raises:
///     ValueError: window_days == 0, unknown kind, unparseable date or
///         mismatched lengths.
#[pyfunction]
#[pyo3(signature = (keys, dates, values, window_days=7, kind="mean", include_current=true))]
fn windowed_transform(
    py: Python<'_>,
    keys: Vec<Option<String>>,
    dates: Vec<Option<String>>,
    values: Vec<Option<f64>>,
    window_days: u32,
    kind: &str,
    include_current: bool,
) -> PyResult<Vec<Option<f64>>> {
    let k = SummaryKind::parse(kind).map_err(PyValueError::new_err)?;
    warn_incomplete(
        py,
        "windowed_transform",
        incomplete_rows(&keys, &dates, &values),
    )?;
    windowed_core(&keys, &dates, &values, window_days, k, include_current)
        .map_err(PyValueError::new_err)
}

/// One-hot expansion: one indicator column per fitted level (first-seen
/// order). Missing values and levels unseen at fit produce an all-zero row.
#[pyclass(name = "OneHotEncoder")]
struct PyOneHotEncoder {
    inner: OneHotEncoder,
}

#[pymethods]
impl PyOneHotEncoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: OneHotEncoder::new(),
        }
    }

    /// Learn the levels (first-seen order). Re-fitting resets them.
    fn fit(&mut self, values: Vec<Option<String>>) {
        self.inner.fit(&values);
    }

    /// Fitted levels, in column order.
    fn levels(&self) -> PyResult<Vec<String>> {
        self.inner
            .levels()
            .map(|v| v.to_vec())
            .map_err(PyValueError::new_err)
    }

    /// Returns {level: indicator list}; unseen/missing rows are all-zero.
    fn transform(&self, py: Python<'_>, values: Vec<Option<String>>) -> PyResult<Py<PyDict>> {
        let cols = self.inner.transform(&values).map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        for (name, col) in cols {
            d.set_item(name, col)?;
        }
        Ok(d.unbind())
    }
}

/// Hashed one-hot: levels are hashed into `n_buckets` indicator columns, so
/// unseen levels still land somewhere. Deterministic.
#[pyclass(name = "HashingEncoder")]
struct PyHashingEncoder {
    inner: HashingEncoder,
}

#[pymethods]
impl PyHashingEncoder {
    #[new]
    fn new(n_buckets: usize) -> PyResult<Self> {
        Ok(Self {
            inner: HashingEncoder::new(n_buckets).map_err(PyValueError::new_err)?,
        })
    }

    /// Returns {bucket name: indicator list}.
    fn transform(&self, py: Python<'_>, values: Vec<Option<String>>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        for (name, col) in self.inner.transform(&values) {
            d.set_item(name, col)?;
        }
        Ok(d.unbind())
    }
}

/// Smoothed target encoding (fitted-mean flavour).
///
/// `w * level_mean + (1 - w) * global_mean` with `w = n / (n + smoothing)`.
/// **Leaky on training data** (the row's own target is in its level mean) —
/// prefer `CatBoostEncoder.fit_transform` for training features. Unseen levels
/// fall back to the global mean; missing stays missing.
#[pyclass(name = "TargetEncoder")]
struct PyTargetEncoder {
    inner: TargetEncoder,
}

#[pymethods]
impl PyTargetEncoder {
    #[new]
    fn new(smoothing: f64) -> PyResult<Self> {
        Ok(Self {
            inner: TargetEncoder::new(smoothing).map_err(PyValueError::new_err)?,
        })
    }

    /// Fit level means on (values, y).
    fn fit(&mut self, values: Vec<Option<String>>, y: Vec<Option<f64>>) -> PyResult<()> {
        self.inner.fit(&values, &y).map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Encode values; unseen -> global mean, missing -> None.
    fn transform(&self, values: Vec<Option<String>>) -> PyResult<Vec<Option<f64>>> {
        self.inner.transform(&values).map_err(PyValueError::new_err)
    }

    /// The global mean used as the unseen-level fallback.
    fn prior(&self) -> PyResult<f64> {
        self.inner.prior().map_err(PyValueError::new_err)
    }
}

/// CatBoost-style ordered target statistics — the leakage-safe flavour.
///
/// `fit_transform` gives each row the running statistic of its level using
/// only rows earlier in a shuffled order: `(sum_y_prefix + prior * global) /
/// (n_prefix + 1)`, averaged over `permutations` seeded passes. The same level
/// gets different values in different rows — that row-dependence is the
/// anti-leakage mechanism. `transform` (new data) uses prior-smoothed fitted
/// level means.
#[pyclass(name = "CatBoostEncoder")]
struct PyCatBoostEncoder {
    inner: CatBoostEncoder,
}

#[pymethods]
impl PyCatBoostEncoder {
    #[new]
    fn new(prior: f64, permutations: usize, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: CatBoostEncoder::new(prior, permutations, seed)
                .map_err(PyValueError::new_err)?,
        })
    }

    /// Fit prior-smoothed level means (used by `transform` on new data).
    fn fit(&mut self, values: Vec<Option<String>>, y: Vec<Option<f64>>) -> PyResult<()> {
        self.inner.fit(&values, &y).map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Leakage-safe ordered encodings of the training rows themselves.
    fn fit_transform(
        &mut self,
        values: Vec<Option<String>>,
        y: Vec<Option<f64>>,
    ) -> PyResult<Vec<Option<f64>>> {
        self.inner
            .fit_transform(&values, &y)
            .map_err(PyValueError::new_err)
    }

    /// Encode new rows from the fitted level means; unseen -> global mean,
    /// missing -> None.
    fn transform(&self, values: Vec<Option<String>>) -> PyResult<Vec<Option<f64>>> {
        self.inner.transform(&values).map_err(PyValueError::new_err)
    }
}

/// Class-level **TF-IDF**: each category becomes a vector over the target
/// classes — `tf(v,j) = count(v,j)/count(v)` times
/// `idf(j) = ln(1 + n/(1 + count(j))) + 1`; `normalize` scales each level
/// vector to unit L2 norm. Task: **classification**. Unseen levels fall back
/// to the global class shares; missing stays missing.
#[pyclass(name = "TfidfEncoder")]
struct PyTfidfEncoder {
    inner: TfidfEncoder,
}

#[pymethods]
impl PyTfidfEncoder {
    #[new]
    fn new(normalize: bool) -> Self {
        Self {
            inner: TfidfEncoder::new(normalize),
        }
    }

    /// Fit class counts on (values, y, n_classes).
    fn fit(
        &mut self,
        values: Vec<Option<String>>,
        y: Vec<Option<f64>>,
        n_classes: usize,
    ) -> PyResult<()> {
        self.inner
            .fit(&values, &y, n_classes)
            .map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// One class-space vector per row (None = missing).
    fn transform(&self, values: Vec<Option<String>>) -> PyResult<Vec<Option<Vec<f64>>>> {
        self.inner.transform(&values).map_err(PyValueError::new_err)
    }
}

/// **Weight of Evidence** (credit-risk classic): one-vs-rest
/// `ln(distr_j(v) / distr_not_j(v))` per category, smoothed with additive
/// `prior`. Task: **classification**. Positive = over-representation of the
/// class in that level. Unseen levels give neutral zeros; missing stays
/// missing.
#[pyclass(name = "WoEEncoder")]
struct PyWoEEncoder {
    inner: WoEEncoder,
}

#[pymethods]
impl PyWoEEncoder {
    #[new]
    fn new(prior: f64) -> PyResult<Self> {
        Ok(Self {
            inner: WoEEncoder::new(prior).map_err(PyValueError::new_err)?,
        })
    }

    /// Fit class counts on (values, y, n_classes).
    fn fit(
        &mut self,
        values: Vec<Option<String>>,
        y: Vec<Option<f64>>,
        n_classes: usize,
    ) -> PyResult<()> {
        self.inner
            .fit(&values, &y, n_classes)
            .map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// One WoE vector per row (None = missing).
    fn transform(&self, values: Vec<Option<String>>) -> PyResult<Vec<Option<Vec<f64>>>> {
        self.inner.transform(&values).map_err(PyValueError::new_err)
    }
}

/// NN-style **learned embeddings** per category: a small embedding table and a
/// linear head trained by SGD, one table per categorical feature. Task:
/// **both** — softmax head for classification; for regression a ReLU output
/// when all fitted targets are non-negative, linear otherwise. `transform`
/// returns the embeddings, `predict` the trained head. Unseen levels give
/// zeros; missing stays missing.
#[pyclass(name = "EmbeddingEncoder")]
struct PyEmbeddingEncoder {
    inner: EmbeddingEncoder,
}

#[pymethods]
impl PyEmbeddingEncoder {
    #[new]
    fn new(dim: usize, lr: f64, epochs: usize, seed: u64, task: &str) -> PyResult<Self> {
        Ok(Self {
            inner: EmbeddingEncoder::new(
                dim,
                lr,
                epochs,
                seed,
                EmbeddingTask::parse(task).map_err(PyValueError::new_err)?,
            )
            .map_err(PyValueError::new_err)?,
        })
    }

    /// Train on (values, y); `n_classes` is inferred for classification when
    /// omitted, ignored for regression.
    #[pyo3(signature = (values, y, n_classes=None))]
    fn fit(
        &mut self,
        values: Vec<Option<String>>,
        y: Vec<Option<f64>>,
        n_classes: Option<usize>,
    ) -> PyResult<()> {
        self.inner
            .fit(&values, &y, n_classes)
            .map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Learned embedding per row (None = missing, zeros = unseen).
    fn transform(&self, values: Vec<Option<String>>) -> PyResult<Vec<Option<Vec<f64>>>> {
        self.inner.transform(&values).map_err(PyValueError::new_err)
    }

    /// Predictions of the trained head (class label or activated value).
    fn predict(&self, values: Vec<Option<String>>) -> PyResult<Vec<Option<f64>>> {
        self.inner.predict(&values).map_err(PyValueError::new_err)
    }
}

/// **Count / frequency** encoding: occurrence count per category (`normalize`
/// gives the share of rows). Task: **both** (unsupervised). Unseen levels give
/// 0.0; missing stays missing.
#[pyclass(name = "CountEncoder")]
struct PyCountEncoder {
    inner: CountEncoder,
}

#[pymethods]
impl PyCountEncoder {
    #[new]
    fn new(normalize: bool) -> Self {
        Self {
            inner: CountEncoder::new(normalize),
        }
    }

    /// Fit counts on `values`.
    fn fit(&mut self, values: Vec<Option<String>>) {
        self.inner.fit(&values);
    }

    /// Count (or share) per row; unseen -> 0.0, missing -> None.
    fn transform(&self, values: Vec<Option<String>>) -> PyResult<Vec<Option<f64>>> {
        self.inner.transform(&values).map_err(PyValueError::new_err)
    }
}

/// **Leave-one-out** encoding: `fit_transform` gives each row its level's mean
/// target **without that row** (leakage-aware); `transform` on new data uses
/// the full fitted level mean. Task: **both**. Unseen -> global mean;
/// missing -> None.
#[pyclass(name = "LeaveOneOutEncoder")]
struct PyLeaveOneOutEncoder {
    inner: LeaveOneOutEncoder,
}

#[pymethods]
impl PyLeaveOneOutEncoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: LeaveOneOutEncoder::new(),
        }
    }

    /// Fit level means on (values, y).
    fn fit(&mut self, values: Vec<Option<String>>, y: Vec<Option<f64>>) -> PyResult<()> {
        self.inner.fit(&values, &y).map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Row-level means excluding each row's own target.
    fn fit_transform(
        &mut self,
        values: Vec<Option<String>>,
        y: Vec<Option<f64>>,
    ) -> PyResult<Vec<Option<f64>>> {
        self.inner
            .fit_transform(&values, &y)
            .map_err(PyValueError::new_err)
    }

    /// Full fitted level means for new rows.
    fn transform(&self, values: Vec<Option<String>>) -> PyResult<Vec<Option<f64>>> {
        self.inner.transform(&values).map_err(PyValueError::new_err)
    }
}

/// **M-estimate** encoding: `(sum_y + m * global) / (n_level + m)` — target
/// smoothing with fixed strength `m`. Task: **both** (leaky on training data;
/// prefer `CatBoostEncoder` or `LeaveOneOutEncoder` there). Unseen -> global;
/// missing -> None.
#[pyclass(name = "MEstimatorEncoder")]
struct PyMEstimatorEncoder {
    inner: MEstimatorEncoder,
}

#[pymethods]
impl PyMEstimatorEncoder {
    #[new]
    fn new(m: f64) -> PyResult<Self> {
        Ok(Self {
            inner: MEstimatorEncoder::new(m).map_err(PyValueError::new_err)?,
        })
    }

    /// Fit smoothed level means on (values, y).
    fn fit(&mut self, values: Vec<Option<String>>, y: Vec<Option<f64>>) -> PyResult<()> {
        self.inner.fit(&values, &y).map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Smoothed mean per row; unseen -> global, missing -> None.
    fn transform(&self, values: Vec<Option<String>>) -> PyResult<Vec<Option<f64>>> {
        self.inner.transform(&values).map_err(PyValueError::new_err)
    }
}

/// **Base-N / binary** encoding: each level becomes its base-`base` digits as
/// indicator columns (`base = 2` is the classic Binary encoder). Task:
/// **both** (unsupervised). Unseen and missing rows are all-zero.
#[pyclass(name = "BaseNEncoder")]
struct PyBaseNEncoder {
    inner: BaseNEncoder,
}

#[pymethods]
impl PyBaseNEncoder {
    #[new]
    fn new(base: usize) -> PyResult<Self> {
        Ok(Self {
            inner: BaseNEncoder::new(base).map_err(PyValueError::new_err)?,
        })
    }

    /// Fit the level vocabulary on `values`.
    fn fit(&mut self, values: Vec<Option<String>>) {
        self.inner.fit(&values);
    }

    /// Returns {digit name: indicator list}.
    fn transform(&self, py: Python<'_>, values: Vec<Option<String>>) -> PyResult<Py<PyDict>> {
        let cols = self.inner.transform(&values).map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        for (name, col) in cols {
            d.set_item(name, col)?;
        }
        Ok(d.unbind())
    }
}

/// **Sum contrast** coding: `k-1` columns; level `i < k-1` gets +1 in column
/// `i`, the last level gets -1 across all columns. Task: **both**
/// (unsupervised). Unseen rows give zeros; missing stays missing.
#[pyclass(name = "SumEncoder")]
struct PySumEncoder {
    inner: SumEncoder,
}

#[pymethods]
impl PySumEncoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: SumEncoder::new(),
        }
    }

    /// Fit the level vocabulary on `values`.
    fn fit(&mut self, values: Vec<Option<String>>) {
        self.inner.fit(&values);
    }

    /// Contrast row per row (None = missing).
    fn transform(&self, values: Vec<Option<String>>) -> PyResult<Vec<Option<Vec<f64>>>> {
        self.inner.transform(&values).map_err(PyValueError::new_err)
    }
}

/// **Quantile** encoding: each level becomes its smoothed target quantile
/// (default median), `w * level_quantile + (1 - w) * global_quantile` with
/// `w = n / (n + smoothing)`. Task: **regression**. Unseen -> global
/// quantile; missing -> None.
#[pyclass(name = "QuantileEncoder")]
struct PyQuantileEncoder {
    inner: QuantileEncoder,
}

#[pymethods]
impl PyQuantileEncoder {
    #[new]
    fn new(q: f64, smoothing: f64) -> PyResult<Self> {
        Ok(Self {
            inner: QuantileEncoder::new(q, smoothing).map_err(PyValueError::new_err)?,
        })
    }

    /// Fit quantiles on (values, y).
    fn fit(&mut self, values: Vec<Option<String>>, y: Vec<Option<f64>>) -> PyResult<()> {
        self.inner.fit(&values, &y).map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Smoothed quantile per row; unseen -> global, missing -> None.
    fn transform(&self, values: Vec<Option<String>>) -> PyResult<Vec<Option<f64>>> {
        self.inner.transform(&values).map_err(PyValueError::new_err)
    }
}

/// **James-Stein** encoding: level means shrunk toward the grand mean with
/// `B = max(0, 1 - ((L - 3) * sigma2) / sum((mean_i - global)^2))`; fewer than
/// three levels collapse to the grand mean. Task: **both**. Unseen -> global;
/// missing -> None.
#[pyclass(name = "JamesSteinEncoder")]
struct PyJamesSteinEncoder {
    inner: JamesSteinEncoder,
}

#[pymethods]
impl PyJamesSteinEncoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: JamesSteinEncoder::new(),
        }
    }

    /// Fit shrunk means on (values, y).
    fn fit(&mut self, values: Vec<Option<String>>, y: Vec<Option<f64>>) -> PyResult<()> {
        self.inner.fit(&values, &y).map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Shrunk mean per row; unseen -> global, missing -> None.
    fn transform(&self, values: Vec<Option<String>>) -> PyResult<Vec<Option<f64>>> {
        self.inner.transform(&values).map_err(PyValueError::new_err)
    }
}

fn regression_to_dict<'py>(py: Python<'py>, r: &RegressionReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n", r.n)?;
    d.set_item("n_missing", r.n_missing)?;
    d.set_item("mean_error", r.mean_error)?;
    d.set_item("bias", r.bias)?;
    d.set_item("mae", r.mae)?;
    d.set_item("mad", r.mad)?;
    d.set_item("mape", r.mape)?;
    d.set_item("rmse", r.rmse)?;
    d.set_item("r2", r.r2)?;
    d.set_item("max_error", r.max_error)?;
    d.set_item("explained_variance", r.explained_variance)?;
    let decs = PyList::empty(py);
    for ds in &r.deciles {
        let dd = PyDict::new(py);
        dd.set_item("decile", ds.decile)?;
        dd.set_item("n", ds.n)?;
        dd.set_item("mean_error", ds.mean_error)?;
        dd.set_item("mae", ds.mae)?;
        dd.set_item("mape", ds.mape)?;
        decs.append(dd)?;
    }
    d.set_item("deciles", decs)?;
    Ok(d)
}

fn classification_to_dict<'py>(
    py: Python<'py>,
    c: &ClassificationReport,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n", c.n)?;
    d.set_item("n_missing", c.n_missing)?;
    d.set_item("n_classes", c.n_classes)?;
    d.set_item("accuracy", c.accuracy)?;
    d.set_item("macro_precision", c.macro_precision)?;
    d.set_item("macro_recall", c.macro_recall)?;
    d.set_item("macro_f1", c.macro_f1)?;
    let pcs = PyList::empty(py);
    for pc in &c.per_class {
        let pd = PyDict::new(py);
        pd.set_item("class", pc.class)?;
        pd.set_item("precision", pc.precision)?;
        pd.set_item("recall", pc.recall)?;
        pd.set_item("f1", pc.f1)?;
        pd.set_item("support", pc.support)?;
        pcs.append(pd)?;
    }
    d.set_item("per_class", pcs)?;
    d.set_item("confusion", c.confusion.clone())?;
    d.set_item("logloss", c.logloss)?;
    d.set_item("brier", c.brier)?;
    d.set_item("auc", c.auc)?;
    Ok(d)
}

/// One metrics class computing complete reports at once — the yardstick every
/// learner is measured with. Task: **both**. Pairs with a missing/NaN side are
/// dropped from the math and reported in `n_missing`, never hidden.
#[pyclass(name = "Metrics")]
struct PyMetrics;

#[pymethods]
impl PyMetrics {
    #[new]
    fn new() -> Self {
        PyMetrics
    }

    /// Complete regression report: `n`, `n_missing`, `mean_error` (actual -
    /// fitted), `bias` (mean_pred / mean_true), `mae`, `mad`, `mape` (non-zero
    /// targets), `rmse`, `r2`, `max_error`, `explained_variance` and
    /// `deciles` — the DAR per-target-decile error profile (the bathtub view).
    ///
    /// Args:
    ///     y_true: actual values (None/NaN = missing).
    ///     y_pred: fitted values (None/NaN = missing).
    ///
    /// Returns: the full report as a dict.
    ///
    /// Raises:
    ///     ValueError: mismatched lengths or no observed pairs.
    fn regression(
        &self,
        py: Python<'_>,
        y_true: Vec<Option<f64>>,
        y_pred: Vec<Option<f64>>,
    ) -> PyResult<Py<PyDict>> {
        let r = regression_report(&y_true, &y_pred).map_err(PyValueError::new_err)?;
        Ok(regression_to_dict(py, &r)?.unbind())
    }

    /// Complete classification report: accuracy, macro averages, per-class
    /// precision/recall/f1/support, confusion matrix — and with
    /// `probabilities` (P(class 1)) also logloss, brier and ROC-AUC.
    ///
    /// Args:
    ///     y_true: integer class labels (None/NaN = missing).
    ///     y_pred: predicted labels (None/NaN = missing).
    ///     probabilities: P(class 1) per row; enables the probabilistic metrics.
    ///
    /// Returns: the full report as a dict.
    ///
    /// Raises:
    ///     ValueError: mismatched lengths, non-integer labels or no observed pairs.
    #[pyo3(signature = (y_true, y_pred, probabilities=None))]
    fn classification(
        &self,
        py: Python<'_>,
        y_true: Vec<Option<f64>>,
        y_pred: Vec<Option<f64>>,
        probabilities: Option<Vec<Option<f64>>>,
    ) -> PyResult<Py<PyDict>> {
        let c = classification_report(&y_true, &y_pred, probabilities.as_deref())
            .map_err(PyValueError::new_err)?;
        Ok(classification_to_dict(py, &c)?.unbind())
    }

    /// Regression report per group (keys: dates, categories, ...) — the
    /// legacy per-key summary lineage. Rows with a missing key are ignored.
    ///
    /// Args:
    ///     keys: group label per row (None = ignored).
    ///     y_true: actual values (None/NaN = missing).
    ///     y_pred: fitted values (None/NaN = missing).
    ///
    /// Returns: {group: regression report}, sorted by group.
    ///
    /// Raises:
    ///     ValueError: mismatched lengths or a group with no observed pairs.
    fn regression_by_group(
        &self,
        py: Python<'_>,
        keys: Vec<Option<String>>,
        y_true: Vec<Option<f64>>,
        y_pred: Vec<Option<f64>>,
    ) -> PyResult<Py<PyDict>> {
        let groups = grouped_regression_report(&keys, &y_true, &y_pred)
            .map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        for (k, r) in groups {
            d.set_item(k, regression_to_dict(py, &r)?)?;
        }
        Ok(d.unbind())
    }
}

/// Model tree: greedy splits by fitted-leaf SSE reduction, OLS or mean
/// leaves, NA-aware (learned split directions + NA-indicator leaf terms).
/// Task: regression.
#[pyclass(name = "ModelTree")]
pub struct PyModelTree {
    params: ModelTreeParams,
    inner: Option<ModelTree>,
}

impl PyModelTree {
    fn tree(&self) -> PyResult<&ModelTree> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ModelTree is not fitted"))
    }
}

#[pymethods]
impl PyModelTree {
    /// Create an unfitted model tree. Task: regression.
    ///
    /// Args:
    ///     max_depth: maximum tree depth (0 = a single leaf).
    ///     min_rows: minimum rows per child side at a split.
    ///     min_gain: minimum fitted-leaf SSE reduction to split.
    ///     leaf_cols: "all" (every column) or "path" (only split-path columns).
    ///     leaf: "linear" (model tree) or "mean" (CART/rpart-style).
    ///
    /// Returns: the model tree; call `.fit()` next.
    #[new]
    #[pyo3(signature = (max_depth = 4, min_rows = 10, min_gain = 1e-9, leaf_cols = "all", leaf = "linear", l2 = 0.1, l1 = 0.1, monotone = vec![], interaction_groups = vec![], min_relevance = None))]
    fn new(
        max_depth: usize,
        min_rows: usize,
        min_gain: f64,
        leaf_cols: &str,
        leaf: &str,
        l2: f64,
        l1: f64,
        monotone: Vec<i8>,
        interaction_groups: Vec<Vec<usize>>,
        min_relevance: Option<f64>,
    ) -> PyResult<Self> {
        let policy = match leaf_cols {
            "all" => LeafCols::All,
            "path" => LeafCols::PathUsed,
            other => {
                return Err(PyValueError::new_err(format!(
                    "leaf_cols must be 'all' or 'path', got {other}"
                )))
            }
        };
        let kind = match leaf {
            "linear" => LeafKind::Linear,
            "mean" => LeafKind::Mean,
            other => {
                return Err(PyValueError::new_err(format!(
                    "leaf must be 'linear' or 'mean', got {other}"
                )))
            }
        };
        Ok(Self {
            params: ModelTreeParams {
                max_depth,
                min_rows,
                min_gain,
                leaf_cols: policy,
                leaf: kind,
                l2,
                l1,
                monotone,
                interaction_groups,
                min_relevance,
            },
            inner: None,
        })
    }

    /// Fit the tree on row-major data. Task: regression.
    ///
    /// Args:
    ///     x: rows of predictors; `None` entries are missing (NA rows train).
    ///     y: targets; `None` drops that row.
    ///     w: optional per-row sample weights (row weightage); zero-weight
    ///        rows are ignored.
    ///
    /// Returns: nothing; the fitted tree replaces the inner model.
    #[pyo3(signature = (x, y, w = None))]
    fn fit(
        &mut self,
        x: Vec<Vec<Option<f64>>>,
        y: Vec<Option<f64>>,
        w: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let tree = match w {
            Some(w) => ModelTree::fit_weighted(&x, &y, &w, self.params.clone()),
            None => ModelTree::fit(&x, &y, self.params.clone()),
        }
        .map_err(PyValueError::new_err)?;
        self.inner = Some(tree);
        Ok(())
    }

    /// Predict for row-major data; NAs never block a prediction. Task: regression.
    ///
    /// Args:
    ///     x: rows of predictors; `None` entries are missing.
    ///
    /// Returns: one predicted value per row.
    fn predict(&self, x: Vec<Vec<Option<f64>>>) -> PyResult<Vec<f64>> {
        Ok(self.tree()?.predict(&x))
    }

    /// Fitted leaf models with their distribution stats. Task: regression.
    ///
    /// Args: none.
    ///
    /// Returns: one dict per leaf with `cols`, `coef`, `na_coef`,
    /// `intercept`, `n`, `mean`, `std`, `min`, `max`, `quantiles`.
    fn leaves(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let t = self.tree()?;
        let mut out = Vec::new();
        for (leaf, n) in t.leaves() {
            let d = PyDict::new(py);
            let (cols, coef, na_coef, intercept) = match &leaf.linear {
                Some(m) => (
                    m.cols.iter().map(|&j| t.col_names[j].clone()).collect::<Vec<_>>(),
                    m.coef.clone(),
                    m.na_coef.clone(),
                    m.intercept,
                ),
                None => (Vec::new(), Vec::new(), Vec::new(), leaf.stats.mean),
            };
            d.set_item("cols", cols)?;
            d.set_item("coef", coef)?;
            d.set_item("na_coef", na_coef)?;
            d.set_item("intercept", intercept)?;
            d.set_item("n", n)?;
            d.set_item("mean", leaf.stats.mean)?;
            d.set_item("std", leaf.stats.std)?;
            d.set_item("min", leaf.stats.min)?;
            d.set_item("max", leaf.stats.max)?;
            let qs: Vec<Option<f64>> = [0.1, 0.25, 0.5, 0.75, 0.9]
                .iter()
                .map(|&q| leaf.stats.quantile(q))
                .collect();
            d.set_item("quantiles", qs)?;
            out.push(d.into_any().unbind());
        }
        Ok(out)
    }

    /// Per-level timing profile (group/sort, scan, leaf fit). Task: regression.
    ///
    /// Args: none.
    ///
    /// Returns: one dict per depth level with `depth`, `nodes`, `rows`,
    /// `cols`, `candidates`, `group_ns`, `scan_ns`, `leaf_ns`.
    fn profile(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let t = self.tree()?;
        let mut out = Vec::new();
        for p in t.profile() {
            let d = PyDict::new(py);
            d.set_item("depth", p.depth)?;
            d.set_item("nodes", p.nodes)?;
            d.set_item("rows", p.rows)?;
            d.set_item("cols", p.cols)?;
            d.set_item("candidates", p.candidates)?;
            d.set_item("group_ns", p.group_ns)?;
            d.set_item("scan_ns", p.scan_ns)?;
            d.set_item("leaf_ns", p.leaf_ns)?;
            out.push(d.into_any().unbind());
        }
        Ok(out)
    }

    /// Learned splits. Task: regression.
    ///
    /// Args: none.
    ///
    /// Returns: one dict per split with `col`, `threshold`, `missing_left`.
    fn splits(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let t = self.tree()?;
        let mut out = Vec::new();
        for s in t.splits() {
            let d = PyDict::new(py);
            d.set_item("col", t.col_names[s.col].clone())?;
            d.set_item("threshold", s.threshold)?;
            d.set_item("missing_left", s.missing_left)?;
            out.push(d.into_any().unbind());
        }
        Ok(out)
    }

    /// rpart-style text print of the tree.
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.tree()?))
    }

    /// Self-contained SVG diagram of the tree (boxes with node stats, edges
    /// labeled with split thresholds). Write the returned string to a .svg
    /// file or display it inline.
    fn to_svg(&self) -> PyResult<String> {
        Ok(self.tree()?.to_svg())
    }

    /// Split columns along each root-to-leaf path. Task: regression.
    ///
    /// Args: none.
    ///
    /// Returns: one list of column names per root-to-leaf path (the split
    /// columns on that path) — the introspection behind interaction-group
    /// checking.
    fn split_paths(&self) -> PyResult<Vec<Vec<String>>> {
        let t = self.tree()?;
        Ok(t
            .split_paths()
            .into_iter()
            .map(|p| p.into_iter().map(|c| t.col_names[c].clone()).collect())
            .collect())
    }

    /// Number of leaves. Task: regression.
    ///
    /// Args: none.
    ///
    /// Returns: leaf count.
    fn n_leaves(&self) -> PyResult<usize> {
        Ok(self.tree()?.n_leaves())
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBaseNEncoder>()?;
    m.add_class::<PyCatBoostEncoder>()?;
    m.add_class::<PyCountEncoder>()?;
    m.add_class::<PyDatasetBuilder>()?;
    m.add_class::<PyEmbeddingEncoder>()?;
    m.add_class::<PyGroupedScaler>()?;
    m.add_class::<PyHashingEncoder>()?;
    m.add_class::<PyIndexMapper>()?;
    m.add_class::<PyJamesSteinEncoder>()?;
    m.add_class::<PyLeaveOneOutEncoder>()?;
    m.add_class::<PyMEstimatorEncoder>()?;
    m.add_class::<PyMetrics>()?;
    m.add_class::<PyModelTree>()?;
    m.add_class::<PyOneHotEncoder>()?;
    m.add_class::<PyQuantileEncoder>()?;
    m.add_class::<PySumEncoder>()?;
    m.add_class::<PyTargetEncoder>()?;
    m.add_class::<PyTfidfEncoder>()?;
    m.add_class::<PyVariableScaler>()?;
    m.add_class::<PyWoEEncoder>()?;
    m.add_function(wrap_pyfunction!(calendar_features, m)?)?;
    m.add_function(wrap_pyfunction!(decomposed_transform, m)?)?;
    m.add_function(wrap_pyfunction!(generate, m)?)?;
    m.add_function(wrap_pyfunction!(grouped_scale, m)?)?;
    m.add_function(wrap_pyfunction!(grouped_transform, m)?)?;
    m.add_function(wrap_pyfunction!(lag_by_days, m)?)?;
    m.add_function(wrap_pyfunction!(summarize, m)?)?;
    m.add_function(wrap_pyfunction!(trended_transform, m)?)?;
    m.add_function(wrap_pyfunction!(windowed_transform, m)?)?;
    Ok(())
}
