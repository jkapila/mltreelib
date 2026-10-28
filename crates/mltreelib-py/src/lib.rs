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

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatasetBuilder>()?;
    m.add_class::<PyGroupedScaler>()?;
    m.add_class::<PyIndexMapper>()?;
    m.add_class::<PyVariableScaler>()?;
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
