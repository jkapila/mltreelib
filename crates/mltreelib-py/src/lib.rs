//! Thin PyO3 shell around `mltreelib-core`. Exposed to Python as
//! `mltreelib._core`; the Python package in `python/mltreelib` re-exports it.


use pyo3::exceptions::{PyAttributeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyModule, PyModuleMethods, PyTuple};
use pyo3::IntoPyObjectExt;

mod calibration;
mod component;
mod solvers;

use mltreelib_core::builder::{DatasetBuilder, NaPolicy};
use mltreelib_core::encode::{
    BaseNEncoder, CatBoostEncoder, CountEncoder, EmbeddingEncoder, EmbeddingTask, HashingEncoder,
    JamesSteinEncoder, LeaveOneOutEncoder, MEstimatorEncoder, OneHotEncoder, QuantileEncoder,
    SumEncoder, TargetEncoder, TfidfEncoder, WoEEncoder,
};
use mltreelib_core::frame::{Column, Dataset};
use mltreelib_core::criteria::{CustomGain, GainFn, LogRankWeight};
use mltreelib_core::family::LeafFamily;
use mltreelib_core::gate::SplitTest;
use mltreelib_core::leaf::{LeafCols, LeafKind, LeafSolver};
use mltreelib_core::model_tree::{ModelTree, ModelTreeParams, PurityMetric};
use mltreelib_core::prune::PruneCost;
use mltreelib_core::solver::EvolutionaryKind;
use mltreelib_core::split::{CatOrder, CatSearch, ColType, SplitEval};
use mltreelib_core::metrics::{
    classification_report, competing_risks_report as core_competing_risks_report,
    grouped_regression_report, regression_report, survival_report,
    time_series_report, uplift_report, ClassificationReport, CompetingRisksReport,
    RegressionReport, SurvivalReport, TimeSeriesReport, UpliftReport,
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

fn uplift_to_dict<'py>(py: Python<'py>, u: &UpliftReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n_total", u.n_total)?;
    d.set_item("n_treated", u.n_treated)?;
    d.set_item("n_control", u.n_control)?;
    d.set_item("ate", u.ate)?;
    d.set_item("qini_coef", u.qini_coef)?;
    d.set_item("qini_norm", u.qini_norm)?;
    d.set_item("auuc", u.auuc)?;
    d.set_item("top_decile_uplift", u.top_decile_uplift)?;
    d.set_item("top_quintile_uplift", u.top_quintile_uplift)?;
    d.set_item("qini_curve", u.qini_curve.clone())?;
    d.set_item("cumulative_gain", u.cumulative_gain.clone())?;
    Ok(d)
}

fn survival_to_dict<'py>(py: Python<'py>, s: &SurvivalReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n_samples", s.n_samples)?;
    d.set_item("n_events", s.n_events)?;
    d.set_item("harrell_c", s.harrell_c)?;
    d.set_item("uno_c", s.uno_c)?;
    d.set_item("ibs", s.ibs)?;
    d.set_item("brier_at_horizons", s.brier_at_horizons.clone())?;
    Ok(d)
}

fn competing_risks_to_dict<'py>(py: Python<'py>, r: &CompetingRisksReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n_samples", r.n_samples)?;
    d.set_item("causes", r.causes.clone())?;
    d.set_item("cause_counts", r.cause_counts.clone())?;
    let c_ind = PyDict::new(py);
    for &(c, v) in &r.cause_c_indices {
        c_ind.set_item(c, v)?;
    }
    d.set_item("cause_c_indices", c_ind)?;
    let uno_c = PyDict::new(py);
    for &(c, v) in &r.cause_uno_c_indices {
        uno_c.set_item(c, v)?;
    }
    d.set_item("cause_uno_c_indices", uno_c)?;
    let ibs = PyDict::new(py);
    for &(c, v) in &r.cause_ibs {
        ibs.set_item(c, v)?;
    }
    d.set_item("cause_ibs", ibs)?;
    let brier = PyDict::new(py);
    for (c, pairs) in &r.brier_at_horizons {
        let p_dict = PyDict::new(py);
        for &(t, bs) in pairs {
            p_dict.set_item(t.to_string(), bs)?;
        }
        brier.set_item(c, p_dict)?;
    }
    d.set_item("brier_at_horizons", brier)?;
    Ok(d)
}

fn d_calibration_to_dict<'py>(py: Python<'py>, r: &mltreelib_core::metrics::DCalibrationReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n_samples", r.n_samples)?;
    d.set_item("n_bins", r.n_bins)?;
    d.set_item("chi2_stat", r.chi2_stat)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("bin_counts", r.bin_counts.clone())?;
    d.set_item("expected_count", r.expected_count)?;
    d.set_item("is_calibrated", r.is_calibrated)?;
    Ok(d)
}

fn ici_to_dict<'py>(py: Python<'py>, r: &mltreelib_core::metrics::IciReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("horizon", r.horizon)?;
    d.set_item("ici", r.ici)?;
    d.set_item("e50", r.e50)?;
    d.set_item("e90", r.e90)?;
    d.set_item("emax", r.emax)?;
    Ok(d)
}

fn one_calibration_to_dict<'py>(py: Python<'py>, r: &mltreelib_core::metrics::OneCalibrationReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("horizon", r.horizon)?;
    d.set_item("intercept", r.intercept)?;
    d.set_item("slope", r.slope)?;
    Ok(d)
}

fn cr_calibration_to_dict<'py>(py: Python<'py>, r: &mltreelib_core::metrics::CrCalibrationReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("horizon", r.horizon)?;
    let ici_dict = PyDict::new(py);
    for (&c, &v) in &r.cause_ici {
        ici_dict.set_item(c, v)?;
    }
    d.set_item("cause_ici", ici_dict)?;
    let brier_dict = PyDict::new(py);
    for (&c, &v) in &r.cause_brier {
        brier_dict.set_item(c, v)?;
    }
    d.set_item("cause_brier", brier_dict)?;
    Ok(d)
}

fn extract_cif_map(obj: &Bound<'_, PyAny>) -> PyResult<std::collections::HashMap<usize, Vec<f64>>> {
    if let Ok(map) = obj.extract::<std::collections::HashMap<usize, Vec<f64>>>() {
        return Ok(map);
    }
    if let Ok(list_of_lists) = obj.extract::<Vec<Vec<f64>>>() {
        let mut map = std::collections::HashMap::new();
        for (i, row) in list_of_lists.into_iter().enumerate() {
            map.insert(i + 1, row);
        }
        return Ok(map);
    }
    Err(PyTypeError::new_err("predicted_cif must be a dict {cause: [probabilities]} or list of lists"))
}

fn run_competing_risks_report(
    durations: &[f64],
    events: &[f64],
    cif_curves_obj: Option<&Bound<'_, PyAny>>,
    risk_scores_obj: Option<&Bound<'_, PyAny>>,
    times: Option<&[f64]>,
) -> Result<CompetingRisksReport, String> {
    let n = durations.len();
    let mut causes: Vec<usize> = Vec::new();
    for &e in events {
        let c = e.round() as usize;
        if c > 0 && !causes.contains(&c) {
            causes.push(c);
        }
    }
    causes.sort_unstable();

    let parsed_risk: Option<Vec<Vec<f64>>> = if let Some(obj) = risk_scores_obj {
        if let Ok(dict) = obj.cast::<PyDict>() {
            let mut mat = vec![vec![0.0; causes.len()]; n];
            for (c_idx, &c) in causes.iter().enumerate() {
                let item = dict.get_item(c).or_else(|_| dict.get_item(c as i64)).or_else(|_| dict.get_item(c.to_string()));
                if let Ok(Some(val)) = item {
                    if let Ok(vec) = val.extract::<Vec<f64>>() {
                        if vec.len() == n {
                            for i in 0..n {
                                mat[i][c_idx] = vec[i];
                            }
                        }
                    }
                }
            }
            Some(mat)
        } else if let Ok(mat) = obj.extract::<Vec<Vec<f64>>>() {
            Some(mat)
        } else if let Ok(vec) = obj.extract::<Vec<f64>>() {
            Some(vec.into_iter().map(|v| vec![v]).collect())
        } else {
            None
        }
    } else {
        None
    };

    let parsed_cif: Option<Vec<Vec<Vec<f64>>>> = if let Some(obj) = cif_curves_obj {
        if let Ok(dict) = obj.cast::<PyDict>() {
            let mut cause_mats: Vec<Option<Vec<Vec<f64>>>> = Vec::with_capacity(causes.len());
            let mut t_len = 0;
            for &c in &causes {
                let item = dict.get_item(c).or_else(|_| dict.get_item(c as i64)).or_else(|_| dict.get_item(c.to_string()));
                if let Ok(Some(val)) = item {
                    if let Ok(m) = val.extract::<Vec<Vec<f64>>>() {
                        if m.len() == n {
                            if t_len == 0 && !m.is_empty() {
                                t_len = m[0].len();
                            }
                            cause_mats.push(Some(m));
                            continue;
                        }
                    }
                }
                cause_mats.push(None);
            }
            if t_len > 0 {
                let mut out = vec![vec![vec![0.0; t_len]; causes.len()]; n];
                for (c_idx, mat_opt) in cause_mats.iter().enumerate() {
                    if let Some(mat) = mat_opt {
                        for i in 0..n {
                            for t_idx in 0..t_len {
                                out[i][c_idx][t_idx] = mat[i][t_idx];
                            }
                        }
                    }
                }
                Some(out)
            } else {
                None
            }
        } else if let Ok(tensor) = obj.extract::<Vec<Vec<Vec<f64>>>>() {
            Some(tensor)
        } else if let Ok(mat) = obj.extract::<Vec<Vec<f64>>>() {
            Some(mat.into_iter().map(|row| vec![row]).collect())
        } else {
            None
        }
    } else {
        None
    };

    core_competing_risks_report(
        durations,
        events,
        parsed_cif.as_deref(),
        parsed_risk.as_deref(),
        times,
    )
}

fn time_series_to_dict<'py>(py: Python<'py>, t: &TimeSeriesReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n_periods", t.n_periods)?;
    d.set_item("mae", t.mae)?;
    d.set_item("rmse", t.rmse)?;
    d.set_item("mase", t.mase)?;
    d.set_item("smape", t.smape)?;
    d.set_item("wape", t.wape)?;
    d.set_item("mda", t.mda)?;
    d.set_item("crps", t.crps)?;
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

    /// Complete causal uplift and Qini curve report.
    ///
    /// Args:
    ///     y_true: observed outcome per row.
    ///     treatment: binary treatment indicator (1.0 = treated, 0.0 = control).
    ///     tau_pred: predicted treatment effect / uplift score per row.
    ///
    /// Returns: dict with ATE, Qini coefficient, normalized Qini, AUUC, decile uplifts, Qini curve.
    #[pyo3(signature = (y_true, treatment, tau_pred))]
    fn uplift(
        &self,
        py: Python<'_>,
        y_true: Vec<f64>,
        treatment: Vec<f64>,
        tau_pred: Vec<f64>,
    ) -> PyResult<Py<PyDict>> {
        let u = uplift_report(&y_true, &treatment, &tau_pred).map_err(PyValueError::new_err)?;
        Ok(uplift_to_dict(py, &u)?.unbind())
    }

    /// Complete survival analysis report (Harrell C-index, Uno IPCW C-index, Integrated Brier Score).
    ///
    /// Args:
    ///     durations: observed duration / time-to-event per row.
    ///     events: event indicator (1.0 = event observed, 0.0 = right-censored).
    ///     risk_scores: predicted risk score (higher score = shorter expected survival).
    ///     surv_curves: optional individual survival probability curves S(t|x) matrix (N, K).
    ///     times: optional evaluation horizon times grid.
    #[pyo3(signature = (durations, events, risk_scores, surv_curves=None, times=None))]
    fn survival(
        &self,
        py: Python<'_>,
        durations: Vec<f64>,
        events: Vec<f64>,
        risk_scores: Vec<f64>,
        surv_curves: Option<Vec<Vec<f64>>>,
        times: Option<Vec<f64>>,
    ) -> PyResult<Py<PyDict>> {
        let s = survival_report(
            &durations,
            &events,
            &risk_scores,
            surv_curves.as_deref(),
            times.as_deref(),
        )
        .map_err(PyValueError::new_err)?;
        Ok(survival_to_dict(py, &s)?.unbind())
    }

    /// Complete competing risks report for multi-cause survival endpoints:
    /// cause-specific Harrell's C, Uno's IPCW C, cause-specific IBS and
    /// time-dependent Brier scores over specified or sample decile evaluation horizons.
    ///
    /// Args:
    ///     durations: observed follow-up times.
    ///     events: event indicators (0 = censored, 1..J = mutually exclusive causes).
    ///     cif_curves: optional CIF curves mapping {cause: (N, T)} or 3D list (N, n_causes, T).
    ///     risk_scores: optional risk scores mapping {cause: (N,)} or 2D list (N, n_causes).
    ///     times: optional evaluation horizon times grid.
    #[pyo3(signature = (durations, events, cif_curves=None, risk_scores=None, times=None))]
    fn competing_risks(
        &self,
        py: Python<'_>,
        durations: Vec<f64>,
        events: Vec<f64>,
        cif_curves: Option<Bound<'_, PyAny>>,
        risk_scores: Option<Bound<'_, PyAny>>,
        times: Option<Vec<f64>>,
    ) -> PyResult<Py<PyDict>> {
        let r = run_competing_risks_report(
            &durations,
            &events,
            cif_curves.as_ref(),
            risk_scores.as_ref(),
            times.as_deref(),
        ).map_err(PyValueError::new_err)?;
        Ok(competing_risks_to_dict(py, &r)?.unbind())
    }

    /// D-Calibration test (Haider et al. 2020) for survival distribution uniformity.
    #[pyo3(signature = (durations, events, surv_at_durations, n_bins = 10))]
    fn d_calibration(
        &self,
        py: Python<'_>,
        durations: Vec<f64>,
        events: Vec<f64>,
        surv_at_durations: Vec<f64>,
        n_bins: usize,
    ) -> PyResult<Py<PyDict>> {
        let r = mltreelib_core::metrics::d_calibration(&durations, &events, &surv_at_durations, n_bins)
            .map_err(PyValueError::new_err)?;
        Ok(d_calibration_to_dict(py, &r)?.unbind())
    }

    /// Integrated Calibration Index (Austin et al. 2020) at fixed horizon.
    #[pyo3(signature = (durations, events, predicted_survival, horizon))]
    fn integrated_calibration_index(
        &self,
        py: Python<'_>,
        durations: Vec<f64>,
        events: Vec<f64>,
        predicted_survival: Vec<f64>,
        horizon: f64,
    ) -> PyResult<Py<PyDict>> {
        let r = mltreelib_core::metrics::integrated_calibration_index(&durations, &events, &predicted_survival, horizon)
            .map_err(PyValueError::new_err)?;
        Ok(ici_to_dict(py, &r)?.unbind())
    }

    /// Alias for integrated_calibration_index.
    #[pyo3(signature = (durations, events, predicted_survival, horizon))]
    fn ici(
        &self,
        py: Python<'_>,
        durations: Vec<f64>,
        events: Vec<f64>,
        predicted_survival: Vec<f64>,
        horizon: f64,
    ) -> PyResult<Py<PyDict>> {
        self.integrated_calibration_index(py, durations, events, predicted_survival, horizon)
    }

    /// 1-Calibration slope and intercept (Austin et al. 2020) at fixed horizon.
    #[pyo3(signature = (durations, events, predicted_survival, horizon))]
    fn one_calibration(
        &self,
        py: Python<'_>,
        durations: Vec<f64>,
        events: Vec<f64>,
        predicted_survival: Vec<f64>,
        horizon: f64,
    ) -> PyResult<Py<PyDict>> {
        let r = mltreelib_core::metrics::one_calibration(&durations, &events, &predicted_survival, horizon)
            .map_err(PyValueError::new_err)?;
        Ok(one_calibration_to_dict(py, &r)?.unbind())
    }

    /// Competing risks calibration report across causes (Alberge et al. 2026).
    #[pyo3(signature = (durations, events, predicted_cif, horizon))]
    fn competing_risks_calibration(
        &self,
        py: Python<'_>,
        durations: Vec<f64>,
        events: Vec<f64>,
        predicted_cif: Bound<'_, PyAny>,
        horizon: f64,
    ) -> PyResult<Py<PyDict>> {
        let cif_map = extract_cif_map(&predicted_cif)?;
        let r = mltreelib_core::metrics::competing_risks_calibration(&durations, &events, &cif_map, horizon)
            .map_err(PyValueError::new_err)?;
        Ok(cr_calibration_to_dict(py, &r)?.unbind())
    }

    /// Complete time series forecasting report (MAE, RMSE, MASE, sMAPE, WAPE, MDA, CRPS).
    ///
    /// Args:
    ///     y_true: actual time series targets.
    ///     y_pred: predicted values.
    ///     y_train: optional in-sample training series (used to scale MASE).
    ///     seasonal_period: optional seasonality m (e.g. 7 for daily, 12 for monthly).
    ///     std_pred: optional predicted standard deviation per period (used for Gaussian CRPS).
    #[pyo3(signature = (y_true, y_pred, y_train=None, seasonal_period=None, std_pred=None))]
    fn time_series(
        &self,
        py: Python<'_>,
        y_true: Vec<f64>,
        y_pred: Vec<f64>,
        y_train: Option<Vec<f64>>,
        seasonal_period: Option<usize>,
        std_pred: Option<Vec<f64>>,
    ) -> PyResult<Py<PyDict>> {
        let t = time_series_report(
            &y_true,
            &y_pred,
            y_train.as_deref(),
            seasonal_period,
            std_pred.as_deref(),
        )
        .map_err(PyValueError::new_err)?;
        Ok(time_series_to_dict(py, &t)?.unbind())
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

    /// Classification report per group key (sorted by key). Task:
    /// classification.
    ///
    /// Args:
    ///     keys: group label per row (rows with missing key/labels skip).
    ///     y_true: integer class labels.
    ///     y_pred: integer class labels.
    ///
    /// Returns: `{group: classification_report}`.
    fn classification_by_group(
        &self,
        py: Python<'_>,
        keys: Vec<Option<String>>,
        y_true: Vec<Option<f64>>,
        y_pred: Vec<Option<f64>>,
    ) -> PyResult<Py<PyDict>> {
        let groups = mltreelib_core::metrics::grouped_classification_report(
            &keys, &y_true, &y_pred,
        )
        .map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        for (k, r) in groups {
            d.set_item(k, classification_to_dict(py, &r)?)?;
        }
        Ok(d.unbind())
    }

    /// Calibration (reliability) bins: mean predicted score vs observed
    /// frequency per bin. Task: classification.
    ///
    /// Args:
    ///     y_true: targets (positive = > 0.5).
    ///     scores: predicted scores/probabilities.
    ///     n_bins: equal-width bins over [0, 1] (default 10).
    ///
    /// Returns: `{"bins": [{"left", "right", "mean_score", "observed", "n"}]}`
    /// (empty bins omitted; scores clip into edge bins).
    #[pyo3(signature = (y_true, scores, n_bins = 10))]
    fn calibration(
        &self,
        py: Python<'_>,
        y_true: Vec<f64>,
        scores: Vec<f64>,
        n_bins: usize,
    ) -> PyResult<Py<PyDict>> {
        let bins =
            mltreelib_core::metrics::calibration_bins(&y_true, &scores, n_bins)
                .map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        let out: Vec<Py<PyDict>> = bins
            .iter()
            .map(|b| {
                let bd = PyDict::new(py);
                let _ = bd.set_item("left", b.left);
                let _ = bd.set_item("right", b.right);
                let _ = bd.set_item("mean_score", b.mean_score);
                let _ = bd.set_item("observed", b.observed);
                let _ = bd.set_item("n", b.n);
                bd.unbind()
            })
            .collect();
        d.set_item("bins", out)?;
        Ok(d.unbind())
    }

    /// ROC curve + AUC. Task: classification.
    ///
    /// Args:
    ///     y_true: targets (positive = > 0.5).
    ///     scores: predicted scores/probabilities.
    ///
    /// Returns: `{"points": [[fpr, tpr], ...], "auc": float}`.
    fn roc_curve(
        &self,
        py: Python<'_>,
        y_true: Vec<f64>,
        scores: Vec<f64>,
    ) -> PyResult<Py<PyDict>> {
        let (pts, auc) = mltreelib_core::metrics::roc_curve(&y_true, &scores)
            .map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        d.set_item("points", pts)?;
        d.set_item("auc", auc)?;
        Ok(d.unbind())
    }

    /// Precision-recall curve + average precision. Task: classification.
    ///
    /// Args:
    ///     y_true: targets (positive = > 0.5).
    ///     scores: predicted scores/probabilities.
    ///
    /// Returns: `{"points": [[recall, precision], ...], "average_precision":
    /// float}`.
    fn pr_curve(
        &self,
        py: Python<'_>,
        y_true: Vec<f64>,
        scores: Vec<f64>,
    ) -> PyResult<Py<PyDict>> {
        let (pts, ap) = mltreelib_core::metrics::pr_curve(&y_true, &scores)
            .map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        d.set_item("points", pts)?;
        d.set_item("average_precision", ap)?;
        Ok(d.unbind())
    }

    /// Cumulative gains curve. Task: classification.
    ///
    /// Args:
    ///     y_true: targets (positive = > 0.5).
    ///     scores: predicted scores/probabilities.
    ///
    /// Returns: `{"points": [[population_fraction, capture_rate], ...]}`
    /// (lift at a point = capture / fraction).
    fn gain_curve(
        &self,
        py: Python<'_>,
        y_true: Vec<f64>,
        scores: Vec<f64>,
    ) -> PyResult<Py<PyDict>> {
        let pts = mltreelib_core::metrics::gain_curve(&y_true, &scores)
            .map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        d.set_item("points", pts)?;
        Ok(d.unbind())
    }

    /// Cumulative lift curve. Task: classification.
    ///
    /// Args:
    ///     y_true: targets (positive = > 0.5).
    ///     scores: predicted scores/probabilities.
    ///
    /// Returns: `{"points": [[population_fraction, lift], ...]}`
    /// (lift = capture / population fraction).
    fn lift_curve(
        &self,
        py: Python<'_>,
        y_true: Vec<f64>,
        scores: Vec<f64>,
    ) -> PyResult<Py<PyDict>> {
        let pts = mltreelib_core::metrics::lift_curve(&y_true, &scores)
            .map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        d.set_item("points", pts)?;
        Ok(d.unbind())
    }

    /// KS sweep + statistic + Score@KS. Task: classification.
    ///
    /// Args:
    ///     y_true: targets (positive = > 0.5).
    ///     scores: predicted scores/probabilities.
    ///
    /// Returns: `{"points": [[fpr, tpr, score], ...], "ks", "score_at_ks",
    /// "tpr_at_ks", "fpr_at_ks"}`.
    fn ks_statistic(
        &self,
        py: Python<'_>,
        y_true: Vec<f64>,
        scores: Vec<f64>,
    ) -> PyResult<Py<PyDict>> {
        let r = mltreelib_core::metrics::ks_statistic(&y_true, &scores)
            .map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        d.set_item("points", r.points)?;
        d.set_item("ks", r.ks)?;
        d.set_item("score_at_ks", r.score_at_ks)?;
        d.set_item("tpr_at_ks", r.tpr_at_ks)?;
        d.set_item("fpr_at_ks", r.fpr_at_ks)?;
        Ok(d.unbind())
    }

    /// Decile profile (kds/PerfScoreCard decile table; modelplotpy response at
    /// decile granularity). Task: classification.
    ///
    /// Args:
    ///     y_true: targets (positive = > 0.5).
    ///     scores: predicted scores/probabilities.
    ///     n_buckets: equal-count buckets by score descending (default 10).
    ///
    /// Returns: `[{"decile", "n", "avg_score", "events", "event_rate",
    /// "cum_capture", "cum_event_rate", "lift"}, ...]` (decile 1 = top).
    #[pyo3(signature = (y_true, scores, n_buckets = 10))]
    fn decile_table(
        &self,
        py: Python<'_>,
        y_true: Vec<f64>,
        scores: Vec<f64>,
        n_buckets: usize,
    ) -> PyResult<Py<PyAny>> {
        let rows = mltreelib_core::metrics::decile_table(&y_true, &scores, n_buckets)
            .map_err(PyValueError::new_err)?;
        let out: Vec<Py<PyDict>> = rows
            .iter()
            .map(|r| {
                let d = PyDict::new(py);
                let _ = d.set_item("decile", r.decile);
                let _ = d.set_item("n", r.n);
                let _ = d.set_item("avg_score", r.avg_score);
                let _ = d.set_item("events", r.events);
                let _ = d.set_item("event_rate", r.event_rate);
                let _ = d.set_item("cum_capture", r.cum_capture);
                let _ = d.set_item("cum_event_rate", r.cum_event_rate);
                let _ = d.set_item("lift", r.lift);
                d.unbind()
            })
            .collect();
        Ok(pyo3::types::PyList::new(py, out)?.into_any().unbind())
    }

    /// Stability index (PSI on scores, CSI on a feature column). Task:
    /// classification or regression monitoring.
    ///
    /// Args:
    ///     reference: reference-period values (bins derive from their range).
    ///     current: current-period values.
    ///     n_bins: equal-width bins (default 10).
    ///
    /// Returns: `{"psi": float, "bins": [{"left", "right", "reference_share",
    /// "current_share", "contribution"}, ...]}`.
    #[pyo3(signature = (reference, current, n_bins = 10))]
    fn stability_index(
        &self,
        py: Python<'_>,
        reference: Vec<f64>,
        current: Vec<f64>,
        n_bins: usize,
    ) -> PyResult<Py<PyDict>> {
        let (psi, bins) = mltreelib_core::metrics::stability_index(&reference, &current, n_bins)
            .map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        d.set_item("psi", psi)?;
        let out: Vec<Py<PyDict>> = bins
            .iter()
            .map(|b| {
                let bd = PyDict::new(py);
                let _ = bd.set_item("left", b.left);
                let _ = bd.set_item("right", b.right);
                let _ = bd.set_item("reference_share", b.reference_share);
                let _ = bd.set_item("current_share", b.current_share);
                let _ = bd.set_item("contribution", b.contribution);
                bd.unbind()
            })
            .collect();
        d.set_item("bins", out)?;
        Ok(d.unbind())
    }

    /// Double-lift table: actual vs model A vs model B along A's ranking (CAS
    /// monograph chart; additive vs ratio display is a plotting concern).
    /// Task: regression or classification.
    ///
    /// Args:
    ///     y_true: targets.
    ///     scores_a: model A predictions (defines the ranking).
    ///     scores_b: model B predictions.
    ///     n_buckets: equal-count buckets (default 10).
    ///
    /// Returns: `[{"bucket", "n", "actual", "pred_a", "pred_b"}, ...]`.
    #[pyo3(signature = (y_true, scores_a, scores_b, n_buckets = 10))]
    fn double_lift(
        &self,
        py: Python<'_>,
        y_true: Vec<f64>,
        scores_a: Vec<f64>,
        scores_b: Vec<f64>,
        n_buckets: usize,
    ) -> PyResult<Py<PyAny>> {
        let rows = mltreelib_core::metrics::double_lift(&y_true, &scores_a, &scores_b, n_buckets)
            .map_err(PyValueError::new_err)?;
        let out: Vec<Py<PyDict>> = rows
            .iter()
            .map(|r| {
                let d = PyDict::new(py);
                let _ = d.set_item("bucket", r.bucket);
                let _ = d.set_item("n", r.n);
                let _ = d.set_item("actual", r.actual);
                let _ = d.set_item("pred_a", r.pred_a);
                let _ = d.set_item("pred_b", r.pred_b);
                d.unbind()
            })
            .collect();
        Ok(pyo3::types::PyList::new(py, out)?.into_any().unbind())
    }

    /// Cohen's kappa with optional weights over the label scale. Task:
    /// classification.
    ///
    /// Args:
    ///     y_true: integer-like labels.
    ///     y_pred: integer-like labels.
    ///     weights: "none" | "linear" | "quadratic" (default "quadratic").
    ///
    /// Returns: kappa (1 = perfect agreement).
    #[pyo3(signature = (y_true, y_pred, weights = "quadratic"))]
    fn weighted_kappa(
        &self,
        y_true: Vec<f64>,
        y_pred: Vec<f64>,
        weights: &str,
    ) -> PyResult<f64> {
        mltreelib_core::metrics::weighted_kappa(&y_true, &y_pred, weights)
            .map_err(PyValueError::new_err)
    }

    /// Log loss of probability predictions. Task: classification.
    ///
    /// Args:
    ///     y_true: targets (positive = > 0.5).
    ///     probs: predicted probabilities.
    ///
    /// Returns: mean log loss.
    fn log_loss(&self, y_true: Vec<f64>, probs: Vec<f64>) -> PyResult<f64> {
        mltreelib_core::metrics::log_loss(&y_true, &probs).map_err(PyValueError::new_err)
    }

    /// Brier score of probability predictions. Task: classification.
    ///
    /// Args:
    ///     y_true: targets (positive = > 0.5).
    ///     probs: predicted probabilities.
    ///
    /// Returns: mean squared probability error.
    fn brier_score(&self, y_true: Vec<f64>, probs: Vec<f64>) -> PyResult<f64> {
        mltreelib_core::metrics::brier_score(&y_true, &probs).map_err(PyValueError::new_err)
    }

    /// Hinge loss over signed decision margins. Task: classification.
    ///
    /// Args:
    ///     y_true: targets (positive = > 0.5).
    ///     decision: decision-function margins (signed).
    ///
    /// Returns: mean hinge loss.
    fn hinge_loss(&self, y_true: Vec<f64>, decision: Vec<f64>) -> PyResult<f64> {
        mltreelib_core::metrics::hinge_loss(&y_true, &decision).map_err(PyValueError::new_err)
    }

    /// Single-label classification scorecard (binary or multiclass).
    /// Task: classification.
    ///
    /// Args:
    ///     y_true: integer-like labels.
    ///     y_pred: integer-like labels.
    ///
    /// Returns: `{"n", "n_classes", "accuracy", "balanced_accuracy",
    /// "precision_macro/micro/weighted", "recall_*", "f1_*", "mcc",
    /// "jaccard_macro/micro/weighted", "hamming_loss", "zero_one_loss"}`.
    fn classification_scores(
        &self,
        py: Python<'_>,
        y_true: Vec<f64>,
        y_pred: Vec<f64>,
    ) -> PyResult<Py<PyDict>> {
        let s = mltreelib_core::metrics::classification_scores(&y_true, &y_pred)
            .map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        d.set_item("n", s.n)?;
        d.set_item("n_classes", s.n_classes)?;
        d.set_item("accuracy", s.accuracy)?;
        d.set_item("balanced_accuracy", s.balanced_accuracy)?;
        d.set_item("precision_macro", s.precision_macro)?;
        d.set_item("recall_macro", s.recall_macro)?;
        d.set_item("f1_macro", s.f1_macro)?;
        d.set_item("precision_micro", s.precision_micro)?;
        d.set_item("recall_micro", s.recall_micro)?;
        d.set_item("f1_micro", s.f1_micro)?;
        d.set_item("precision_weighted", s.precision_weighted)?;
        d.set_item("recall_weighted", s.recall_weighted)?;
        d.set_item("f1_weighted", s.f1_weighted)?;
        d.set_item("mcc", s.mcc)?;
        d.set_item("jaccard_macro", s.jaccard_macro)?;
        d.set_item("jaccard_micro", s.jaccard_micro)?;
        d.set_item("jaccard_weighted", s.jaccard_weighted)?;
        d.set_item("hamming_loss", s.hamming_loss)?;
        d.set_item("zero_one_loss", s.zero_one_loss)?;
        Ok(d.unbind())
    }

    /// Multilabel scorecard: rows are instances, columns are labels, entries
    /// > 0.5 are positive (raw entries double as ranking scores).
    /// Task: multilabel classification.
    ///
    /// Args:
    ///     y_true: label matrix (rows x labels, 0/1).
    ///     y_scores: prediction/score matrix (same shape).
    ///
    /// Returns: `{"n", "n_labels", "subset_accuracy", "hamming_loss",
    /// "precision_micro/macro/samples", "recall_*", "f1_*",
    /// "label_ranking_ap", "coverage_error"}`.
    fn multilabel_scores(
        &self,
        py: Python<'_>,
        y_true: Vec<Vec<f64>>,
        y_scores: Vec<Vec<f64>>,
    ) -> PyResult<Py<PyDict>> {
        let s = mltreelib_core::metrics::multilabel_scores(&y_true, &y_scores)
            .map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        d.set_item("n", s.n)?;
        d.set_item("n_labels", s.n_labels)?;
        d.set_item("subset_accuracy", s.subset_accuracy)?;
        d.set_item("hamming_loss", s.hamming_loss)?;
        d.set_item("precision_micro", s.precision_micro)?;
        d.set_item("recall_micro", s.recall_micro)?;
        d.set_item("f1_micro", s.f1_micro)?;
        d.set_item("precision_macro", s.precision_macro)?;
        d.set_item("recall_macro", s.recall_macro)?;
        d.set_item("f1_macro", s.f1_macro)?;
        d.set_item("precision_samples", s.precision_samples)?;
        d.set_item("recall_samples", s.recall_samples)?;
        d.set_item("f1_samples", s.f1_samples)?;
        d.set_item("label_ranking_ap", s.label_ranking_ap)?;
        d.set_item("coverage_error", s.coverage_error)?;
        Ok(d.unbind())
    }

    /// Pairwise distances between two row sets. Task: any.
    ///
    /// Args:
    ///     a: first rows (rows x features).
    ///     b: second rows (same width).
    ///     metric: "euclidean" | "manhattan" | "chebyshev" | "minkowski" |
    ///         "cosine" | "correlation" | "hamming" | "jaccard".
    ///     p: Minkowski power (default 2.0).
    ///
    /// Returns: `len(a) x len(b)` distance matrix.
    #[pyo3(signature = (a, b, metric = "euclidean", p = 2.0))]
    fn pairwise_distances(
        &self,
        py: Python<'_>,
        a: Vec<Vec<f64>>,
        b: Vec<Vec<f64>>,
        metric: &str,
        p: f64,
    ) -> PyResult<Py<PyAny>> {
        let m = mltreelib_core::metrics::pairwise_distances(&a, &b, metric, p)
            .map_err(PyValueError::new_err)?;
        Ok(pyo3::types::PyList::new(py, m)?.into_any().unbind())
    }
}

/// Custom split criterion bridge: a Python callable over lazy y-slices.
/// Per-candidate callback cost is on the user (the generic slow path).
struct PyGain {
    fun: Py<PyAny>,
}

impl CustomGain for PyGain {
    fn gain(&self, left: &[(f64, f64)], right: &[(f64, f64)]) -> f64 {
        Python::attach(|py| {
            let l = match PyList::new(py, left.iter().map(|&(y, w)| (y, w))) {
                Ok(l) => l,
                Err(_) => return f64::NAN,
            };
            let r = match PyList::new(py, right.iter().map(|&(y, w)| (y, w))) {
                Ok(r) => r,
                Err(_) => return f64::NAN,
            };
            self.fun
                .call1(py, (l, r))
                .and_then(|v| v.extract::<f64>(py))
                .unwrap_or(f64::NAN)
        })
    }
}

fn parse_feature_indices(
    obj: &Bound<'_, PyAny>,
    col_names: &[String],
) -> PyResult<Vec<usize>> {
    let mut indices = Vec::new();
    if let Ok(seq) = obj.extract::<Vec<Bound<'_, PyAny>>>() {
        for item in seq {
            if let Ok(idx) = item.extract::<usize>() {
                if idx >= col_names.len() && !col_names.is_empty() {
                    return Err(PyValueError::new_err(format!(
                        "leaf feature index {idx} out of bounds for width {}",
                        col_names.len()
                    )));
                }
                indices.push(idx);
            } else if let Ok(name) = item.extract::<String>() {
                match col_names.iter().position(|c| c == &name) {
                    Some(idx) => indices.push(idx),
                    None => {
                        return Err(PyValueError::new_err(format!(
                            "leaf feature column '{name}' not found in data columns"
                        )))
                    }
                }
            } else {
                return Err(PyTypeError::new_err(
                    "leaf feature items must be integers or column name strings",
                ));
            }
        }
    } else {
        return Err(PyTypeError::new_err(
            "leaf_features / retrain_cols must be a sequence of column names or indices",
        ));
    }
    indices.sort_unstable();
    indices.dedup();
    Ok(indices)
}

fn parse_leaf_solver(name: &str) -> PyResult<LeafSolver> {
    match name {
        "irls" => Ok(LeafSolver::Irls),
        "lbfgs" => Ok(LeafSolver::Lbfgs),
        "ccd" => Ok(LeafSolver::Ccd),
        other => EvolutionaryKind::from_name(other)
            .map(EvolutionaryKind::leaf_solver)
            .ok_or_else(|| PyValueError::new_err(format!(
                "solver must be 'irls', 'lbfgs', 'ccd', 'cmaes', 'shade', 'nes', or 'ga', got {other:?}"
            ))),
    }
}

fn leaf_solver_name(solver: LeafSolver) -> &'static str {
    match solver {
        LeafSolver::Irls => "irls",
        LeafSolver::Lbfgs => "lbfgs",
        LeafSolver::Ccd => "ccd",
        other => EvolutionaryKind::from_leaf_solver(other)
            .expect("non-evolutionary solvers are handled above")
            .as_str(),
    }
}

fn select_alias<'a>(
    canonical: Option<&'a str>,
    legacy: Option<&'a str>,
    name: &str,
) -> PyResult<Option<&'a str>> {
    if let (Some(_), Some(_)) = (canonical, legacy) {
        return Err(PyValueError::new_err(format!(
            "Supply either {name} or leaf_{name}"
        )));
    }
    Ok(canonical.or(legacy))
}

fn inherit_leaf_inputs(
    mut control: mltreelib_core::leaf::LeafControl,
    params: &ModelTreeParams,
) -> mltreelib_core::leaf::LeafControl {
    control.events = control.events.or_else(|| params.events.clone());
    control.treatments = control.treatments.or_else(|| params.treatments.clone());
    if control.leaf_monotone.is_empty() {
        control.leaf_monotone.clone_from(&params.leaf_monotone);
    }
    control
}

fn extract_leaf_control<'py>(
    _py: Python<'py>,
    control: Option<Bound<'py, PyAny>>,
    tree_params: &ModelTreeParams,
    leaf: Option<&str>,
    leaf_family: Option<&str>,
    leaf_cols: Option<&str>,
    l1: Option<f64>,
    l2: Option<f64>,
    leaf_digest: Option<bool>,
    tau: Option<f64>,
    leaf_features_arg: Option<Bound<'py, PyAny>>,
    effective_col_names: &[String],
) -> PyResult<mltreelib_core::leaf::LeafControl> {
    let mut ctrl = match control {
        Some(ref obj) => {
            if let Ok(c) = obj.extract::<PyRef<PyLeafControl>>() {
                c.inner.clone()
            } else if let Ok(dict) = obj.extract::<Bound<'_, PyDict>>() {
                let mut c = mltreelib_core::leaf::LeafControl::from(tree_params);
                if let Some(item) = dict.get_item("leaf")? {
                    let s = item.extract::<String>()?;
                    c.leaf = match s.as_str() {
                        "linear" => LeafKind::Linear,
                        "mean" => LeafKind::Mean,
                        "median" => LeafKind::Median,
                        "quantile" => {
                            let t = match dict.get_item("tau")? {
                                Some(v) => v.extract::<f64>()?,
                                None => 0.5,
                            };
                            if !(0.0..=1.0).contains(&t) {
                                return Err(PyValueError::new_err("tau must be in [0, 1]"));
                            }
                            LeafKind::Quantile(t)
                        }
                        "huber" => LeafKind::Huber,
                        "mars" => LeafKind::Mars,
                        "knn" | "dwknn" | "uniform_knn" | "distance_knn" => LeafKind::Knn,
                        "prob" => LeafKind::Prob,
                        "odds" => LeafKind::Odds,
                        "log_odds" | "logit" => LeafKind::LogOdds,
                        other => {
                            return Err(PyValueError::new_err(format!(
                                "unrecognized leaf kind '{other}' in control dict"
                            )))
                        }
                    };
                }
                let fam_item = match dict.get_item("leaf_family")? {
                    Some(it) => Some(it),
                    None => dict.get_item("family")?,
                };
                if let Some(item) = fam_item {
                    let s = item.extract::<String>()?;
                    c.family = match s.as_str() {
                        "gaussian" => LeafFamily::Gaussian,
                        "binomial" => LeafFamily::Binomial,
                        "poisson" => LeafFamily::Poisson,
                        "probit" => LeafFamily::Probit,
                        "gamma" => LeafFamily::Gamma,
                        "tweedie" => LeafFamily::Tweedie,
                        "tobit" => LeafFamily::Tobit,
                        "softmax" => LeafFamily::Softmax,
                        "survival" => LeafFamily::Survival,
                        "heteroscedastic" => LeafFamily::Heteroscedastic,
                        "aft" => LeafFamily::Aft,
                        "cure" => LeafFamily::Cure,
                        "spline" => LeafFamily::Spline,
                        other => {
                            return Err(PyValueError::new_err(format!(
                                "unrecognized leaf family '{other}' in control dict"
                            )))
                        }
                    };
                }
                let solver = dict.get_item("solver")?;
                let legacy_solver = dict.get_item("leaf_solver")?;
                if solver.is_some() && legacy_solver.is_some() {
                    return Err(PyValueError::new_err("Supply either solver or leaf_solver"));
                }
                if let Some(item) = solver.or(legacy_solver) {
                    c.solver = parse_leaf_solver(item.extract::<&str>()?)?;
                }
                if let Some(item) = dict.get_item("leaf_cols")? {
                    let s = item.extract::<String>()?;
                    c.leaf_cols = match s.as_str() {
                        "all" => LeafCols::All,
                        "path_used" | "path" => LeafCols::PathUsed,
                        other => {
                            return Err(PyValueError::new_err(format!(
                                "unrecognized leaf_cols '{other}' in control dict"
                            )))
                        }
                    };
                }
                if let Some(item) = dict.get_item("l1")? {
                    c.l1 = item.extract::<f64>()?;
                }
                if let Some(item) = dict.get_item("l2")? {
                    c.l2 = item.extract::<f64>()?;
                }
                if let Some(item) = dict.get_item("leaf_digest")? {
                    c.leaf_digest = item.extract::<bool>()?;
                }
                if let Some(item) = dict.get_item("tau")? {
                    let t = item.extract::<f64>()?;
                    if !(0.0..=1.0).contains(&t) {
                        return Err(PyValueError::new_err("tau must be in [0, 1]"));
                    }
                    if matches!(c.leaf, LeafKind::Quantile(_)) {
                        c.leaf = LeafKind::Quantile(t);
                    }
                }
                let feat_item = match dict.get_item("leaf_features")? {
                    Some(it) => Some(it),
                    None => dict.get_item("retrain_cols")?,
                };
                if let Some(item) = feat_item {
                    c.leaf_features = Some(parse_feature_indices(&item, effective_col_names)?);
                }
                if let Some(item) = dict.get_item("knn_k")? {
                    c.knn_k = item.extract::<usize>()?;
                }
                if let Some(item) = dict.get_item("knn_k_validity")? {
                    c.knn_k_validity = item.extract::<usize>()?;
                }
                if let Some(item) = dict.get_item("knn_variant")? {
                    let s = item.extract::<String>()?;
                    c.knn_variant = mltreelib_core::knn::KnnVariant::from_str(&s).map_err(PyValueError::new_err)?;
                }
                if let Some(item) = dict.get_item("knn_metric")? {
                    let s = item.extract::<String>()?;
                    let p = dict.get_item("knn_p")?.and_then(|v| v.extract::<f64>().ok());
                    c.knn_metric = mltreelib_core::knn::DistanceMetric::from_str(&s, p).map_err(PyValueError::new_err)?;
                }
                if let Some(item) = dict.get_item("knn_gamma")? {
                    c.knn_gamma = item.extract::<Option<f64>>()?;
                }
                if let Some(item) = dict.get_item("knn_p")? {
                    c.knn_p = item.extract::<f64>()?;
                }
                if let Some(item) = dict.get_item("shrinkage")? {
                    c.shrinkage = item.extract::<f64>()?;
                }
                if let Some(item) = dict.get_item("max_classes")? {
                    c.max_classes = item.extract::<usize>()?;
                }
                if let Some(item) = dict.get_item("leaf_na_policy")? {
                    let s = item.extract::<String>()?;
                    c.leaf_na_policy = match s.as_str() {
                        "drop_fallback" | "drop" | "fallback" => mltreelib_core::leaf::LeafNaPolicy::DropFallback,
                        "impute_local" | "impute" | "local" => mltreelib_core::leaf::LeafNaPolicy::ImputeLocal,
                        "indicator" => mltreelib_core::leaf::LeafNaPolicy::Indicator,
                        other => return Err(PyValueError::new_err(format!("unrecognized leaf_na_policy '{other}' in control dict"))),
                    };
                }
                if let Some(item) = dict.get_item("aft_dist")? {
                    let s = item.extract::<String>()?;
                    c.aft_dist = match s.to_lowercase().as_str() {
                        "weibull" => mltreelib_core::node_stats::AftDist::Weibull,
                        "lognormal" | "log_normal" => mltreelib_core::node_stats::AftDist::LogNormal,
                        "loglogistic" | "log_logistic" => mltreelib_core::node_stats::AftDist::LogLogistic,
                        other => return Err(PyValueError::new_err(format!("unrecognized aft_dist '{other}' in control dict"))),
                    };
                }
                if let Some(item) = dict.get_item("cure_fraction")? {
                    c.cure_fraction = item.extract::<Option<f64>>()?;
                }
                if let Some(item) = dict.get_item("spline_knots")? {
                    c.spline_knots = item.extract::<usize>()?;
                }
                for (k, v) in dict.iter() {
                    let key = k.extract::<String>()?;
                    if !matches!(
                        key.as_str(),
                        "leaf"
                            | "family"
                            | "leaf_family"
                            | "leaf_cols"
                            | "l1"
                            | "l2"
                            | "leaf_digest"
                            | "tau"
                            | "leaf_features"
                            | "retrain_cols"
                            | "shrinkage"
                            | "max_classes"
                            | "leaf_na_policy"
                            | "aft_dist"
                            | "cure_fraction"
                            | "spline_knots"
                    ) {
                        if let Ok(num) = v.extract::<f64>() {
                            c.extra.insert(key, num);
                        }
                    }
                }
                c
            } else {
                return Err(PyTypeError::new_err(
                    "control must be a LeafControl instance or dict",
                ));
            }
        }
        None => mltreelib_core::leaf::LeafControl::from(tree_params),
    };

    if let Some(l) = leaf {
        ctrl.leaf = match l {
            "linear" => LeafKind::Linear,
            "mean" => LeafKind::Mean,
            "median" => LeafKind::Median,
            "quantile" => {
                let t_val = tau.unwrap_or(0.5);
                if !(0.0..=1.0).contains(&t_val) {
                    return Err(PyValueError::new_err("tau must be in [0, 1]"));
                }
                LeafKind::Quantile(t_val)
            }
            "huber" => LeafKind::Huber,
            "mars" => LeafKind::Mars,
            "knn" | "dwknn" | "uniform_knn" | "distance_knn" => LeafKind::Knn,
            "prob" => LeafKind::Prob,
            "odds" => LeafKind::Odds,
            "log_odds" | "logit" => LeafKind::LogOdds,
            other => return Err(PyValueError::new_err(format!("unrecognized leaf kind '{other}'"))),
        };
    }
    if let Some(fam) = leaf_family {
        ctrl.family = match fam {
            "gaussian" => LeafFamily::Gaussian,
            "binomial" => LeafFamily::Binomial,
            "poisson" => LeafFamily::Poisson,
            "probit" => LeafFamily::Probit,
            "gamma" => LeafFamily::Gamma,
            "tweedie" => LeafFamily::Tweedie,
            "tobit" => LeafFamily::Tobit,
            "softmax" => LeafFamily::Softmax,
            "survival" => LeafFamily::Survival,
            "heteroscedastic" => LeafFamily::Heteroscedastic,
            "aft" => LeafFamily::Aft,
            "cure" => LeafFamily::Cure,
            "spline" => LeafFamily::Spline,
            other => return Err(PyValueError::new_err(format!("unrecognized leaf family '{other}'"))),
        };
    }
    if let Some(lc) = leaf_cols {
        ctrl.leaf_cols = match lc {
            "all" => LeafCols::All,
            "path_used" | "path" => LeafCols::PathUsed,
            other => return Err(PyValueError::new_err(format!("unrecognized leaf_cols policy '{other}'"))),
        };
    }
    if let Some(v) = l1 {
        ctrl.l1 = v;
    }
    if let Some(v) = l2 {
        ctrl.l2 = v;
    }
    if let Some(v) = leaf_digest {
        ctrl.leaf_digest = v;
    }
    if let Some(t) = tau {
        if !(0.0..=1.0).contains(&t) {
            return Err(PyValueError::new_err("tau must be in [0, 1]"));
        }
        if matches!(ctrl.leaf, LeafKind::Quantile(_)) {
            ctrl.leaf = LeafKind::Quantile(t);
        }
    }
    if let Some(ref feat_obj) = leaf_features_arg {
        ctrl.leaf_features = Some(parse_feature_indices(feat_obj, effective_col_names)?);
    }
    if let Some(dict) = control.as_ref().and_then(|obj| obj.cast::<PyDict>().ok()) {
        if dict.contains("solver")? {
            ctrl.validate_solver().map_err(PyValueError::new_err)?;
        }
    }

    Ok(ctrl)
}

/// Parameter controls for fitting leaf models independently of tree topology.
#[pyclass(name = "LeafControl")]
#[derive(Clone, Debug)]
pub struct PyLeafControl {
    pub(crate) inner: mltreelib_core::leaf::LeafControl,
}

#[pymethods]
impl PyLeafControl {
    /// Controls for leaf model refitting.
    ///
    /// Args:
    ///     leaf: leaf kind ("linear", "mean", "median", "quantile", "huber").
    ///     leaf_family: leaf distribution ("gaussian", "binomial", "poisson").
    ///     leaf_cols: column policy ("all", "path_used").
    ///     l1: L1 lasso penalty.
    ///     l2: L2 ridge penalty.
    ///     leaf_digest: whether to build and store t-digests.
    ///     tau: target quantile level when leaf is "quantile" (in [0, 1]).
    ///     leaf_features: optional list of column indices to use for leaf models.
    #[new]
    #[pyo3(signature = (
        leaf = "linear",
        leaf_family = None,
        leaf_solver = None,
        leaf_cols = "all",
        l1 = 0.0,
        l2 = 1e-3,
        leaf_digest = true,
        tau = 0.5,
        leaf_features = None,
        leaf_monotone = None,
        monotone = None,
        mars_degree = 1,
        mars_terms = 10,
        mars_penalty = 2.0,
        tweedie_p = 1.5,
        tobit_threshold = 0.0,
        knn_k = 5,
        knn_k_validity = 5,
        knn_variant = "uniform",
        knn_metric = "euclidean",
        knn_gamma = None,
        knn_p = 2.0,
        events = None,
        treatments = None,
        n_classes = None,
        shrinkage = 0.0,
        max_classes = 256,
        leaf_na_policy = "drop_fallback",
        aft_dist = "weibull",
        cure_fraction = None,
        spline_knots = 3, *, family = None, solver = None
    ))]
    fn new(
        leaf: &str,
        leaf_family: Option<&str>,
        leaf_solver: Option<&str>,
        leaf_cols: &str,
        l1: f64,
        l2: f64,
        leaf_digest: bool,
        tau: f64,
        leaf_features: Option<Vec<usize>>,
        leaf_monotone: Option<Vec<i8>>,
        monotone: Option<Vec<i8>>,
        mars_degree: usize,
        mars_terms: usize,
        mars_penalty: f64,
        tweedie_p: f64,
        tobit_threshold: f64,
        knn_k: usize,
        knn_k_validity: usize,
        knn_variant: &str,
        knn_metric: &str,
        knn_gamma: Option<f64>,
        knn_p: f64,
        events: Option<Vec<f64>>,
        treatments: Option<Vec<f64>>,
        n_classes: Option<usize>,
        shrinkage: f64,
        max_classes: usize,
        leaf_na_policy: &str,
        aft_dist: &str,
        cure_fraction: Option<f64>,
        spline_knots: usize,
        family: Option<&str>,
        solver: Option<&str>,
    ) -> PyResult<Self> {
        let strict_solver = solver.is_some();
        let leaf_solver = select_alias(solver, leaf_solver, "solver")?;
        let leaf_family = select_alias(family, leaf_family, "family")?.unwrap_or("gaussian");
        let kind = match leaf {
            "linear" => LeafKind::Linear,
            "mean" => LeafKind::Mean,
            "median" => LeafKind::Median,
            "quantile" => {
                if !(0.0..=1.0).contains(&tau) {
                    return Err(PyValueError::new_err("tau must be in [0, 1]"));
                }
                LeafKind::Quantile(tau)
            }
            "huber" => LeafKind::Huber,
            "mars" => LeafKind::Mars,
            "knn" | "dwknn" | "uniform_knn" | "distance_knn" => LeafKind::Knn,
            "prob" => LeafKind::Prob,
            "odds" => LeafKind::Odds,
            "log_odds" | "logit" => LeafKind::LogOdds,
            other => return Err(PyValueError::new_err(format!("unrecognized leaf kind '{other}'"))),
        };
        let family = match leaf_family {
            "gaussian" => LeafFamily::Gaussian,
            "binomial" => LeafFamily::Binomial,
            "poisson" => LeafFamily::Poisson,
            "probit" => LeafFamily::Probit,
            "gamma" => LeafFamily::Gamma,
            "tweedie" => LeafFamily::Tweedie,
            "tobit" => LeafFamily::Tobit,
            "softmax" => LeafFamily::Softmax,
            "survival" => LeafFamily::Survival,
            "heteroscedastic" => LeafFamily::Heteroscedastic,
            "aft" => LeafFamily::Aft,
            "cure" => LeafFamily::Cure,
            "spline" => LeafFamily::Spline,
            other => return Err(PyValueError::new_err(format!("unrecognized leaf family '{other}'"))),
        };
        let solver = parse_leaf_solver(leaf_solver.unwrap_or("irls"))?;
        let leaf_na_policy_enum = match leaf_na_policy {
            "drop_fallback" | "drop" | "fallback" => mltreelib_core::leaf::LeafNaPolicy::DropFallback,
            "impute_local" | "impute" | "local" => mltreelib_core::leaf::LeafNaPolicy::ImputeLocal,
            "indicator" => mltreelib_core::leaf::LeafNaPolicy::Indicator,
            other => return Err(PyValueError::new_err(format!("unrecognized leaf_na_policy '{other}', expected 'drop_fallback', 'impute_local', or 'indicator'"))),
        };
        let policy = match leaf_cols {
            "all" => LeafCols::All,
            "path_used" | "path" => LeafCols::PathUsed,
            other => return Err(PyValueError::new_err(format!("unrecognized leaf_cols policy '{other}'"))),
        };
        let mono = leaf_monotone.or(monotone).unwrap_or_default();
        let knn_var = mltreelib_core::knn::KnnVariant::from_str(knn_variant).map_err(PyValueError::new_err)?;
        let knn_met = mltreelib_core::knn::DistanceMetric::from_str(knn_metric, Some(knn_p)).map_err(PyValueError::new_err)?;
        let aft_dist_enum = match aft_dist.to_lowercase().as_str() {
            "weibull" => mltreelib_core::node_stats::AftDist::Weibull,
            "lognormal" | "log_normal" => mltreelib_core::node_stats::AftDist::LogNormal,
            "loglogistic" | "log_logistic" => mltreelib_core::node_stats::AftDist::LogLogistic,
            other => return Err(PyValueError::new_err(format!("unrecognized aft_dist '{other}', expected 'weibull', 'lognormal', or 'loglogistic'"))),
        };
        let control = Self {
            inner: mltreelib_core::leaf::LeafControl {
                leaf: kind,
                solver,
                family,
                leaf_cols: policy,
                l1,
                l2,
                leaf_digest,
                leaf_monotone: mono.clone(),
                monotone: mono,
                leaf_features,
                extra: std::collections::HashMap::new(),
                events,
                treatments,
                n_classes,
                mars_degree,
                mars_terms,
                mars_penalty,
                tweedie_p,
                tobit_threshold,
                knn_k,
                knn_k_validity,
                knn_variant: knn_var,
                knn_metric: knn_met,
                knn_gamma,
                knn_p,
                verbose: 0,
                shrinkage,
                max_classes,
                leaf_na_policy: leaf_na_policy_enum,
                aft_dist: aft_dist_enum,
                cure_fraction,
                spline_knots,
            },
        };
        if strict_solver {
            control.inner.validate_solver().map_err(PyValueError::new_err)?;
        }
        Ok(control)
    }

    #[getter]
    fn get_leaf(&self) -> String {
        match self.inner.leaf {
            LeafKind::Linear => "linear".to_string(),
            LeafKind::Mean => "mean".to_string(),
            LeafKind::Median => "median".to_string(),
            LeafKind::Quantile(t) => format!("quantile({t})"),
            LeafKind::Huber => "huber".to_string(),
            LeafKind::Mars => "mars".to_string(),
            LeafKind::Knn => "knn".to_string(),
            LeafKind::Prob => "prob".to_string(),
            LeafKind::Odds => "odds".to_string(),
            LeafKind::LogOdds => "log_odds".to_string(),
        }
    }

    #[getter]
    fn get_leaf_family(&self) -> String {
        match self.inner.family {
            LeafFamily::Gaussian => "gaussian".to_string(),
            LeafFamily::Binomial => "binomial".to_string(),
            LeafFamily::Poisson => "poisson".to_string(),
            LeafFamily::Probit => "probit".to_string(),
            LeafFamily::Gamma => "gamma".to_string(),
            LeafFamily::Tweedie => "tweedie".to_string(),
            LeafFamily::Tobit => "tobit".to_string(),
            LeafFamily::Softmax => "softmax".to_string(),
            LeafFamily::Survival => "survival".to_string(),
            LeafFamily::Heteroscedastic => "heteroscedastic".to_string(),
            LeafFamily::Aft => "aft".to_string(),
            LeafFamily::Cure => "cure".to_string(),
            LeafFamily::Spline => "spline".to_string(),
        }
    }

    #[getter]
    fn get_leaf_solver(&self) -> String {
        leaf_solver_name(self.inner.solver).to_string()
    }

    #[getter]
    fn solver(&self) -> String {
        self.get_leaf_solver()
    }

    #[getter]
    fn family(&self) -> String {
        self.get_leaf_family()
    }

    #[getter]
    fn get_leaf_na_policy(&self) -> String {
        match self.inner.leaf_na_policy {
            mltreelib_core::leaf::LeafNaPolicy::DropFallback => "drop_fallback".to_string(),
            mltreelib_core::leaf::LeafNaPolicy::ImputeLocal => "impute_local".to_string(),
            mltreelib_core::leaf::LeafNaPolicy::Indicator => "indicator".to_string(),
        }
    }

    #[getter]
    fn get_leaf_cols(&self) -> String {
        match self.inner.leaf_cols {
            LeafCols::All => "all".to_string(),
            LeafCols::PathUsed => "path_used".to_string(),
        }
    }

    #[getter]
    fn get_l1(&self) -> f64 {
        self.inner.l1
    }

    #[getter]
    fn get_l2(&self) -> f64 {
        self.inner.l2
    }

    #[getter]
    fn get_leaf_digest(&self) -> bool {
        self.inner.leaf_digest
    }

    #[getter]
    fn get_leaf_features(&self) -> Option<Vec<usize>> {
        self.inner.leaf_features.clone()
    }

    #[getter]
    fn get_mars_degree(&self) -> usize {
        self.inner.mars_degree
    }

    #[getter]
    fn get_mars_terms(&self) -> usize {
        self.inner.mars_terms
    }

    #[getter]
    fn get_mars_penalty(&self) -> f64 {
        self.inner.mars_penalty
    }

    #[getter]
    fn get_tweedie_p(&self) -> f64 {
        self.inner.tweedie_p
    }

    #[getter]
    fn get_tobit_threshold(&self) -> f64 {
        self.inner.tobit_threshold
    }

    #[getter]
    fn get_leaf_monotone(&self) -> Vec<i8> {
        self.inner.leaf_monotone.clone()
    }

    #[getter]
    fn get_extra(&self) -> std::collections::HashMap<String, f64> {
        self.inner.extra.clone()
    }

    #[pyo3(signature = (
        leaf = None,
        leaf_family = None,
        leaf_cols = None,
        l1 = None,
        l2 = None,
        leaf_digest = None,
        tau = None,
        leaf_features = None,
        leaf_monotone = None,
        mars_degree = None,
        mars_terms = None,
        mars_penalty = None,
        tweedie_p = None,
        tobit_threshold = None
    ))]
    fn with_params(
        &self,
        leaf: Option<&str>,
        leaf_family: Option<&str>,
        leaf_cols: Option<&str>,
        l1: Option<f64>,
        l2: Option<f64>,
        leaf_digest: Option<bool>,
        tau: Option<f64>,
        leaf_features: Option<Vec<usize>>,
        leaf_monotone: Option<Vec<i8>>,
        mars_degree: Option<usize>,
        mars_terms: Option<usize>,
        mars_penalty: Option<f64>,
        tweedie_p: Option<f64>,
        tobit_threshold: Option<f64>,
    ) -> PyResult<Self> {
        let mut copy = self.clone();
        if let Some(l) = leaf {
            copy.inner.leaf = match l {
                "linear" => LeafKind::Linear,
                "mean" => LeafKind::Mean,
                "median" => LeafKind::Median,
                "quantile" => LeafKind::Quantile(tau.unwrap_or(0.5)),
                "huber" => LeafKind::Huber,
                "mars" => LeafKind::Mars,
                "prob" => LeafKind::Prob,
                "odds" => LeafKind::Odds,
                "log_odds" | "logit" => LeafKind::LogOdds,
                other => return Err(PyValueError::new_err(format!("unrecognized leaf kind '{other}'"))),
            };
        }
        if let Some(fam) = leaf_family {
            copy.inner.family = match fam {
                "gaussian" => LeafFamily::Gaussian,
                "binomial" => LeafFamily::Binomial,
                "poisson" => LeafFamily::Poisson,
                "probit" => LeafFamily::Probit,
                "gamma" => LeafFamily::Gamma,
                "tweedie" => LeafFamily::Tweedie,
                "tobit" => LeafFamily::Tobit,
                "softmax" => LeafFamily::Softmax,
                "survival" => LeafFamily::Survival,
                "heteroscedastic" => LeafFamily::Heteroscedastic,
                other => return Err(PyValueError::new_err(format!("unrecognized leaf family '{other}'"))),
            };
        }
        if let Some(lc) = leaf_cols {
            copy.inner.leaf_cols = match lc {
                "all" => LeafCols::All,
                "path_used" | "path" => LeafCols::PathUsed,
                other => return Err(PyValueError::new_err(format!("unrecognized leaf_cols '{other}'"))),
            };
        }
        if let Some(val) = l1 {
            copy.inner.l1 = val;
        }
        if let Some(val) = l2 {
            copy.inner.l2 = val;
        }
        if let Some(val) = leaf_digest {
            copy.inner.leaf_digest = val;
        }
        if let Some(t) = tau {
            if matches!(copy.inner.leaf, LeafKind::Quantile(_)) {
                copy.inner.leaf = LeafKind::Quantile(t);
            }
        }
        if leaf_features.is_some() {
            copy.inner.leaf_features = leaf_features;
        }
        if let Some(lm) = leaf_monotone {
            copy.inner.leaf_monotone = lm.clone();
            copy.inner.monotone = lm;
        }
        if let Some(d) = mars_degree {
            copy.inner.mars_degree = d;
        }
        if let Some(t) = mars_terms {
            copy.inner.mars_terms = t;
        }
        if let Some(p) = mars_penalty {
            copy.inner.mars_penalty = p;
        }
        if let Some(tp) = tweedie_p {
            copy.inner.tweedie_p = tp;
        }
        if let Some(tt) = tobit_threshold {
            copy.inner.tobit_threshold = tt;
        }
        Ok(copy)
    }

    fn __repr__(&self) -> String {
        format!(
            "LeafControl(leaf='{}', leaf_family='{}', leaf_cols='{}', l1={}, l2={}, leaf_digest={}, mars_degree={}, mars_terms={})",
            self.get_leaf(),
            self.get_leaf_family(),
            self.get_leaf_cols(),
            self.inner.l1,
            self.inner.l2,
            self.inner.leaf_digest,
            self.inner.mars_degree,
            self.inner.mars_terms
        )
    }
}

/// An individual IF-THEN rule extracted from a ModelTree.
#[pyclass(name = "Rule")]
#[derive(Clone)]
pub struct PyRule {
    inner: mltreelib_core::rules::Rule,
}

#[pymethods]
impl PyRule {
    #[getter]
    fn id(&self) -> usize {
        self.inner.id
    }

    #[getter]
    fn prediction(&self) -> f64 {
        self.inner.prediction
    }

    #[getter]
    fn cover(&self) -> usize {
        self.inner.cover
    }

    #[getter]
    fn errors(&self) -> usize {
        self.inner.errors
    }

    #[getter]
    fn confidence(&self) -> f64 {
        self.inner.confidence
    }

    #[getter]
    fn leaf_id(&self) -> usize {
        self.inner.leaf_id
    }

    #[getter]
    fn conditions(&self) -> Vec<String> {
        self.inner.conditions.iter().map(|c| c.to_string()).collect()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// An ordered set of IF-THEN rules with fallback default prediction.
#[pyclass(name = "RuleSet")]
#[derive(Clone)]
pub struct PyRuleSet {
    inner: mltreelib_core::rules::RuleSet,
}

#[pymethods]
impl PyRuleSet {
    #[getter]
    fn default_prediction(&self) -> f64 {
        self.inner.default_prediction
    }

    #[getter]
    fn default_confidence(&self) -> f64 {
        self.inner.default_confidence
    }

    #[getter]
    fn rules(&self) -> Vec<PyRule> {
        self.inner.rules.iter().cloned().map(|r| PyRule { inner: r }).collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    #[pyo3(signature = (x))]
    fn predict(&self, x: Vec<Vec<Option<f64>>>) -> Vec<f64> {
        self.inner.predict(&x)
    }
}

/// Model tree: greedy splits with Gaussian, binomial-logit or Poisson-log
/// leaves, NA-aware (learned split directions + NA-indicator leaf terms).
/// Task: regression, binary probability estimation, or count modeling.
#[pyclass(name = "ModelTree", module = "mltreelib._core")]
#[derive(Clone)]
pub struct PyModelTree {
    params: ModelTreeParams,
    inner: Option<ModelTree>,
    /// Per-column level vocabularies (categorical columns that carried
    /// string cells at fit): identity codes, first-seen order. `None` for
    /// columns whose values were already numeric level codes.
    vocabs: Vec<Option<std::collections::HashMap<String, f64>>>,
    interaction_group_names: Option<Vec<Vec<String>>>,
}

impl PyModelTree {
    fn tree(&self) -> PyResult<&ModelTree> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ModelTree is not fitted"))
    }

    fn from_fitted_tree(tree: ModelTree) -> Self {
        let vocabs = tree
            .category_levels
            .iter()
            .map(|levels| levels.as_ref().map(|levels| levels.clone().into_iter().collect()))
            .collect();
        Self {
            params: tree.params.clone(),
            inner: Some(tree),
            vocabs,
            interaction_group_names: None,
        }
    }
}

fn model_persistence_error(error: mltreelib_core::PersistenceError) -> PyErr {
    match error {
        mltreelib_core::PersistenceError::Io(error) => error.into(),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// Coerce raw cells to the core's rows — the three data handlers at the
/// door. Integer/float columns take int/float cells; categorical columns
/// take level values (a distinct value IS a level; string cells carry an
/// internal identity vocabulary, first-seen codes — identity, never
/// order). `None`/NaN = missing.
fn coerce_rows(
    rows: Vec<Vec<Option<Bound<'_, PyAny>>>>,
    dtypes: &[ColType],
    vocabs: &mut Vec<Option<std::collections::HashMap<String, f64>>>,
    building: bool,
) -> PyResult<Vec<Vec<Option<f64>>>> {
    let n_cols = rows.first().map(|r| r.len()).unwrap_or(0);
    if building {
        vocabs.clear();
        for j in 0..n_cols {
            let cat = dtypes.get(j) == Some(&ColType::Categorical);
            let has_str = cat
                && rows.iter().any(|r| {
                    r.get(j)
                        .and_then(|c| c.as_ref())
                        .map(|v| v.extract::<String>().is_ok())
                        .unwrap_or(false)
                });
            vocabs.push(if has_str {
                Some(std::collections::HashMap::new())
            } else {
                None
            });
        }
    }
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        if r.len() != n_cols {
            return Err(PyValueError::new_err("ragged x: rows differ in width"));
        }
        let mut orow = Vec::with_capacity(n_cols);
        for (j, c) in r.into_iter().enumerate() {
            let v = match c {
                None => None,
                Some(cell) => {
                    if let Ok(f) = cell.extract::<f64>() {
                        if f.is_nan() {
                            None
                        } else {
                            Some(f)
                        }
                    } else if let Ok(s) = cell.extract::<String>() {
                        match vocabs.get_mut(j).and_then(|v| v.as_mut()) {
                            Some(vocab) => {
                                if building {
                                    let next = vocab.len() as f64;
                                    Some(*vocab.entry(s).or_insert(next))
                                } else {
                                    // unseen level at predict: maps to missing (None)
                                    // per AGENTS.md contract ("Unseen categorical levels map to missing")
                                    vocab.get(&s).copied()
                                }
                            }
                            None => {
                                return Err(PyValueError::new_err(format!(
                                    "column {j}: string value in a numeric column"
                                )))
                            }
                        }
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "column {j}: cells must be numbers, strings or None"
                        )));
                    }
                }
            };
            orow.push(v);
        }
        out.push(orow);
    }
    Ok(out)
}

#[pymethods]
impl PyModelTree {
    /// Create an unfitted model tree for regression, binary probabilities,
    /// or counts. Data is raw (int / float / level values, None/NaN = missing)
    /// — the three data handlers, no preprocessing.
    ///
    /// Args (grouped by family):
    ///   GROWTH
    ///     max_depth: maximum tree depth (0 = a single leaf).
    ///     min_rows: minimum rows per child side at a split.
    ///     minsplit: minimum node rows to attempt a split (default 2*min_rows).
    ///     minbucket: minimum rows per leaf (default min_rows).
    ///     minprob: minimum weight fraction of the node per leaf (0 = off).
    ///     max_leaf_nodes: leaf cap — stop splitting at this many leaves
    ///        (None = unlimited, bounded by max_depth).
    ///     min_gain: minimum improvement in the selected split criterion.
    ///     purity_gate: a node whose weighted target variation is at most this
    ///        stays a leaf (None = off).
    ///     purity_gate_metric: "sd" (weighted std dev) or "cv" (coefficient of variation).
    ///     tail_quantile: extrema protection quantile (e.g. 0.05). Extrema nodes bypass purity gate.
    ///   SEARCH
    ///     gain: split criterion — "sse", "deviance", "friedman_mse",
    ///        "absolute_error", "huber", "poisson", "entropy", "gini",
    ///        "gain_ratio", "sdr", "twoing" or a callable
    ///        `gain_fn(left_y, right_y) -> float` (higher = better).
    ///        "deviance" scores fitted leaf-family deviance and is recommended
    ///        for binomial/Poisson; selecting a family does not change gain.
    ///     split_precision: float candidate grid for numeric columns;
    ///        None = exact unique-value scan.
    ///     split_cols: column indices eligible for the split search (None =
    ///        all). Splits only — leaves see all columns.
    ///     col_sample: per-node random column subset fraction (None = all).
    ///     seed: seed for the per-node column subset.
    ///     split_eval: "fast" (default), "histogram" (up to 256 bins for
    ///         classification gains), or "exhaustive".
    ///     presort: upfront numeric column sorting (default false).
    ///   LEAF
    ///     leaf: "linear", "mean", "median", "quantile" (with tau), "huber", or "mars".
    ///     leaf_family: "gaussian" (default), "binomial" (logit link), "poisson" (log link),
    ///        "probit", "gamma", "tweedie", "tobit", "softmax", "survival", or "heteroscedastic".
    ///     tau: quantile in (0, 1) for leaf="quantile".
    ///     l1: lasso penalty on leaf coefficients (with l2 = elastic net).
    ///     l2: ridge penalty on leaf coefficients (intercept free).
    ///     leaf_cols: "all" (every column) or "path" (split-path columns).
    ///     leaf_digest: keep each leaf's empirical target t-digest (quantile
    ///        answers any tau post-fit; false = lean leaves).
    ///     mars_degree: maximum degree of interaction for MARS leaves (default 1).
    ///     mars_terms: maximum number of terms for MARS leaves (default 10).
    ///     mars_penalty: GCV penalty per knot for MARS leaves (default 2.0).
    ///     tweedie_p: Tweedie power parameter (default 1.5, with 1 < p < 2).
    ///     tobit_threshold: censoring threshold for Tobit response (default 0.0).
    ///     events: optional event indicators for survival analysis.
    ///     n_classes: number of distinct classes for softmax leaf family.
    ///   CATEGORICAL
    ///     cat_order: "frequency_desc", "target_mean", or a level-code list.
    ///     cat_search: "cumulative", "single_vs_rest" or "both".
    ///   CONSTRAINTS
    ///     monotone: per-column direction -1/0/+1 (XGBoost-style).
    ///     interaction_groups: feature groups allowed to interact.
    ///   BALANCE
    ///     alpha: minimum child weight fraction of the node (grf).
    ///     imbalance_penalty: subtract this * (1/n_left + 1/n_right) from
    ///        the gain (grf-style).
    ///   PRUNING
    ///     prune: "none", "bic", "mdl" or "cp" (cost-complexity).
    ///     cp_alpha: per-leaf penalty with prune="cp".
    ///   SAMPLING (honesty)
    ///     honesty: structure/estimation sample split (grf-style).
    ///     honesty_fraction: share of rows assigned to structure.
    ///   NA
    ///     maxsurrogate: backup splits routing rows missing the split
    ///        column (rpart lineage; 0 = off).
    ///   GATES
    ///     split_test: "none" or "chow" (Gaussian-only significance gate).
    ///     split_alpha: significance level for split_test.
    ///     bonferroni: divide split_alpha by the node's candidate count.
    ///   SCREENING
    ///     preselect: "none", "winnow", "boruta", "shadowmax", "variance",
    ///        "mutual_info" or "unitree" (per-variable surrogate trees
    ///        scored by explained variation R²).
    ///     min_relevance: threshold for the screening strategies.
    ///   OTHER
    ///     verbose: training log — 0 silent, 1 per depth, 2 per node.
    ///
    /// Returns: the model tree; call `.fit()` next.
    #[new]
    #[pyo3(
        signature = (*args, **kwargs),
        text_signature = "(*args, growth_control=None, leaf_control=None, **kwargs)"
    )]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<Self>> {
        let factory = py.get_type::<Self>().getattr("_from_flat")?;
        if kwargs.is_none() {
            return Ok(factory.call1(args)?.cast_into::<Self>()?.unbind());
        }
        let options = py
            .import("mltreelib._controls")?
            .getattr("_normalize_model_options")?
            .call1((&factory, args, kwargs))?;
        Ok(factory
            .call(args, Some(options.cast::<PyDict>()?))?
            .cast_into::<Self>()?
            .unbind())
    }

    #[staticmethod]
    #[pyo3(signature = (
        max_depth = 6, min_rows = 15, minsplit = None, minbucket = None,
        minprob = 0.02, max_leaf_nodes = None, min_gain = 1e-9, purity_gate = None,
        purity_gate_metric = "sd", tail_quantile = None,
        gain = None, split_precision = Some(0.001), split_cols = None,
        col_sample = None, seed = 42,
        leaf = "linear", tau = 0.5, l1 = 0.0, l2 = 0.5,
        leaf_cols = "all", leaf_digest = true,
        cat_order = None, cat_search = "cumulative",
        monotone = vec![], tree_monotone = None, leaf_monotone = None,
        col_weights = None, interaction_groups = None,
        alpha = 0.0, imbalance_penalty = 0.0,
        prune = "cp", cp_alpha = 5.0,
        honesty = false, honesty_fraction = 0.5,
        maxsurrogate = 0,
        split_test = "none", split_alpha = 0.05, bonferroni = false,
        preselect = "none", min_relevance = None,
        verbose = 0, leaf_family = None, leaf_solver = None, split_eval = "fast",
        presort = false, events = None, treatments = None, train_groups = true, train_ids = false, n_classes = None,
        tweedie_p = 1.5, tobit_threshold = 0.0,
        mars_degree = 1, mars_terms = 10, mars_penalty = 2.0,
        knn_k = 5, knn_k_validity = 5, knn_variant = "uniform", knn_metric = "euclidean",
        knn_gamma = None, knn_p = 2.0,
        shrinkage = 0.0, max_classes = 256, leaf_na_policy = "drop_fallback", preset = None,
        control = None, leaf_control = None,
        mmd_kernel = None, mmd_bandwidth = None,
        fairness_penalty = None, protected_col = None, fairness_metric = None,
        sporf_d = 0, sporf_projections = 10,
        extratrees = false, ebm_fast = false,
        target_cause = 1,
        aft_dist = "weibull", cure_fraction = None, spline_knots = 3, *,
        family = None, solver = None
    ))]
    fn _from_flat(
        max_depth: usize,
        min_rows: usize,
        minsplit: Option<usize>,
        minbucket: Option<usize>,
        minprob: f64,
        max_leaf_nodes: Option<usize>,
        min_gain: f64,
        purity_gate: Option<f64>,
        purity_gate_metric: &str,
        tail_quantile: Option<f64>,
        gain: Option<Bound<'_, PyAny>>,
        split_precision: Option<f64>,
        split_cols: Option<Vec<usize>>,
        col_sample: Option<f64>,
        seed: u64,
        leaf: &str,
        tau: f64,
        l1: f64,
        l2: f64,
        leaf_cols: &str,
        leaf_digest: bool,
        cat_order: Option<Bound<'_, PyAny>>,
        cat_search: &str,
        monotone: Vec<i8>,
        tree_monotone: Option<Vec<i8>>,
        leaf_monotone: Option<Vec<i8>>,
        col_weights: Option<Vec<f64>>,
        interaction_groups: Option<Bound<'_, PyAny>>,
        alpha: f64,
        imbalance_penalty: f64,
        prune: &str,
        cp_alpha: f64,
        honesty: bool,
        honesty_fraction: f64,
        maxsurrogate: usize,
        split_test: &str,
        split_alpha: f64,
        bonferroni: bool,
        preselect: &str,
        min_relevance: Option<f64>,
        verbose: u8,
        leaf_family: Option<&str>,
        leaf_solver: Option<&str>,
        split_eval: &str,
        presort: bool,
        events: Option<Vec<f64>>,
        treatments: Option<Vec<f64>>,
        train_groups: bool,
        train_ids: bool,
        n_classes: Option<usize>,
        tweedie_p: f64,
        tobit_threshold: f64,
        mars_degree: usize,
        mars_terms: usize,
        mars_penalty: f64,
        knn_k: usize,
        knn_k_validity: usize,
        knn_variant: &str,
        knn_metric: &str,
        knn_gamma: Option<f64>,
        knn_p: f64,
        shrinkage: f64,
        max_classes: usize,
        leaf_na_policy: &str,
        preset: Option<&str>,
        control: Option<Bound<'_, PyAny>>,
        leaf_control: Option<Bound<'_, PyAny>>,
        mmd_kernel: Option<&str>,
        mmd_bandwidth: Option<f64>,
        fairness_penalty: Option<f64>,
        protected_col: Option<usize>,
        fairness_metric: Option<&str>,
        sporf_d: usize,
        sporf_projections: usize,
        extratrees: bool,
        ebm_fast: bool,
        target_cause: usize,
        aft_dist: &str,
        cure_fraction: Option<f64>,
        spline_knots: usize,
        family: Option<&str>,
        solver: Option<&str>,
    ) -> PyResult<Self> {
        let strict_solver = solver.is_some();
        let leaf_solver = select_alias(solver, leaf_solver, "solver")?;
        let leaf_family = select_alias(family, leaf_family, "family")?.unwrap_or("gaussian");
        let mut max_depth = max_depth;
        let mut min_rows = min_rows;
        let mut shrinkage = shrinkage;
        let mut prune = prune;
        let mut split_eval = split_eval;

        if let Some(p) = preset {
            match p.to_lowercase().as_str() {
                "accurate" => {
                    max_depth = 8;
                    min_rows = 10;
                    shrinkage = 10.0;
                    prune = "none";
                }
                "fast" => {
                    split_eval = "fast";
                    max_depth = 5;
                    min_rows = 20;
                    shrinkage = 0.0;
                }
                "interpretable" => {
                    max_depth = 3;
                    min_rows = 30;
                    shrinkage = 0.0;
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown preset: '{other}'. Expected 'accurate', 'fast', or 'interpretable'"
                    )));
                }
            }
        }
        let (tree_mono, leaf_mono) = {
            let tm = tree_monotone.unwrap_or_else(|| monotone.clone());
            let lm = leaf_monotone.unwrap_or_else(|| monotone.clone());
            (tm, lm)
        };
        let mut ig_indices = Vec::new();
        let mut ig_names = None;
        if let Some(ref ig_obj) = interaction_groups {
            if let Ok(idx_groups) = ig_obj.extract::<Vec<Vec<usize>>>() {
                ig_indices = idx_groups;
            } else if let Ok(name_groups) = ig_obj.extract::<Vec<Vec<String>>>() {
                ig_names = Some(name_groups);
            } else {
                return Err(PyValueError::new_err(
                    "interaction_groups must be a list of lists of column indices (ints) or column names (strings)"
                ));
            }
        }
        let policy = match leaf_cols {
            "all" => LeafCols::All,
            "path" | "path_used" => LeafCols::PathUsed,
            other => {
                return Err(PyValueError::new_err(format!(
                    "leaf_cols must be 'all' or 'path', got {other}"
                )))
            }
        };
        let kind = match leaf {
            "linear" => LeafKind::Linear,
            "mean" => LeafKind::Mean,
            "median" => LeafKind::Median,
            "quantile" => LeafKind::Quantile(tau),
            "huber" => LeafKind::Huber,
            "mars" => LeafKind::Mars,
            "knn" | "dwknn" | "uniform_knn" | "distance_knn" => LeafKind::Knn,
            "prob" => LeafKind::Prob,
            "odds" => LeafKind::Odds,
            "log_odds" | "logit" => LeafKind::LogOdds,
            other => {
                return Err(PyValueError::new_err(format!(
                    "leaf must be 'linear', 'mean', 'median', 'quantile', 'huber', 'mars', 'prob', 'odds', or 'log_odds', got {other}"
                )))
            }
        };
        let mut family = match leaf_family {
            "gaussian" => LeafFamily::Gaussian,
            "binomial" => LeafFamily::Binomial,
            "poisson" => LeafFamily::Poisson,
            "probit" => LeafFamily::Probit,
            "gamma" => LeafFamily::Gamma,
            "tweedie" => LeafFamily::Tweedie,
            "tobit" => LeafFamily::Tobit,
            "softmax" => LeafFamily::Softmax,
            "survival" => LeafFamily::Survival,
            "heteroscedastic" => LeafFamily::Heteroscedastic,
            "aft" => LeafFamily::Aft,
            "cure" => LeafFamily::Cure,
            "spline" => LeafFamily::Spline,
            other => {
                return Err(PyValueError::new_err(format!(
                    "leaf_family must be 'gaussian', 'binomial', 'poisson', 'probit', 'gamma', 'tweedie', 'tobit', 'softmax', 'survival', or 'heteroscedastic', got {other}"
                )))
            }
        };
        let leaf_solver_enum = match leaf_solver {
            Some(name) => parse_leaf_solver(name)?,
            None if family == LeafFamily::Gaussian => LeafSolver::Irls,
            None => LeafSolver::Lbfgs,
        };
        let leaf_na_policy_enum = match leaf_na_policy {
            "drop_fallback" | "drop" | "fallback" => mltreelib_core::leaf::LeafNaPolicy::DropFallback,
            "impute_local" | "impute" | "local" => mltreelib_core::leaf::LeafNaPolicy::ImputeLocal,
            "indicator" => mltreelib_core::leaf::LeafNaPolicy::Indicator,
            other => {
                return Err(PyValueError::new_err(format!(
                    "leaf_na_policy must be 'drop_fallback', 'impute_local', or 'indicator', got {other}"
                )))
            }
        };
        let eval_mode = match split_eval {
            "fast" => SplitEval::Fast,
            "histogram" => SplitEval::Histogram,
            "exhaustive" => SplitEval::Exhaustive,
            other => {
                return Err(PyValueError::new_err(format!(
                    "split_eval must be 'fast', 'histogram', or 'exhaustive', got {other}"
                )))
            }
        };
        let gain_fn = match &gain {
            None => GainFn::Sse,
            Some(g) => {
                if let Ok(name) = g.extract::<String>() {
                    match name.as_str() {
                        "sse" => GainFn::Sse,
                        "deviance" => GainFn::Deviance,
                        "friedman_mse" => GainFn::FriedmanMse,
                        "absolute_error" => GainFn::AbsoluteError,
                        "huber" => GainFn::Huber,
                        "poisson" => GainFn::Poisson,
                        "entropy" => GainFn::Entropy,
                        "gini" => GainFn::Gini,
                        "gain_ratio" => GainFn::GainRatio,
                        "sdr" => GainFn::Sdr,
                        "twoing" => GainFn::Twoing,
                        "causal" | "uplift" => GainFn::Causal,
                        "logrank" | "log_rank" => GainFn::LogRank,
                        "tarone_ware" | "taroneware" => GainFn::LogRankWeighted { scheme: LogRankWeight::TaroneWare },
                        "peto" | "peto_prentice" => GainFn::LogRankWeighted { scheme: LogRankWeight::PetoPrentice },
                        "gray" | "gray_test" | "logrank_cr" => GainFn::GrayTest { cause: target_cause },
                        "survival_brier" | "brier" => GainFn::SurvivalBrier,
                        "pseudo_value" | "pseudovalue" | "pv" => GainFn::PseudoValue,
                        "rmst" => GainFn::Rmst,
                        "aft_deviance" | "aftdeviance" => GainFn::AftDeviance,
                        "tweedie" => GainFn::Tweedie { p: tweedie_p },
                        "heteroscedastic" | "gaussian_nll" | "ngboost" => GainFn::Heteroscedastic,
                        "tobit" => GainFn::Tobit { threshold: tobit_threshold },
                        "cv" => GainFn::Cv,
                        "martingale" => GainFn::Martingale,
                        "mob" => GainFn::Mob,
                        "ctree" => GainFn::Ctree,
                        "energy_distance" | "energy" => GainFn::EnergyDistance,
                        "wasserstein" | "wasserstein2" => GainFn::Wasserstein2,
                        "hellinger" => GainFn::Hellinger,
                        "maxstat" | "max_stat" => GainFn::MaxStat,
                        "fair" => GainFn::FairConstrained {
                            base_gain: Box::new(GainFn::Sse),
                            lambda: fairness_penalty.unwrap_or(1.0),
                            protected_col: protected_col.unwrap_or(0),
                        },
                        "mmd" => {
                            let k = match mmd_kernel.unwrap_or("rbf").to_lowercase().as_str() {
                                "linear" => mltreelib_core::criteria::KernelKind::Linear,
                                "poly" | "polynomial" => mltreelib_core::criteria::KernelKind::Polynomial { degree: 2, c: 1.0 },
                                _ => mltreelib_core::criteria::KernelKind::Rbf,
                            };
                            GainFn::Mmd {
                                kernel: k,
                                bandwidth: mmd_bandwidth.unwrap_or(0.0),
                            }
                        }
                        other => {
                            return Err(PyValueError::new_err(format!(
                                "gain must be 'sse', 'deviance', 'friedman_mse', 'absolute_error', 'huber', 'poisson', 'entropy', 'gini', 'gain_ratio', 'sdr', 'twoing', 'causal', 'logrank', 'tarone_ware', 'peto', 'gray', 'survival_brier', 'pseudo_value', 'rmst', 'aft_deviance', 'tweedie', 'heteroscedastic', 'tobit', 'cv', 'martingale', 'mob', 'ctree', 'energy_distance', 'mmd', 'wasserstein', 'hellinger', 'maxstat', 'fair' or a callable, got {other}"
                            )))
                        }
                    }
                } else if g.is_callable() {
                    GainFn::Custom(std::sync::Arc::new(PyGain {
                        fun: g.clone().unbind(),
                    }))
                } else {
                    return Err(PyValueError::new_err(
                        "gain must be 'sse', 'entropy', 'gini' or a callable",
                    ));
                }
            }
        };
        if family == LeafFamily::Gaussian && matches!(gain_fn, GainFn::LogRank | GainFn::LogRankWeighted { .. } | GainFn::GrayTest { .. } | GainFn::SurvivalBrier | GainFn::PseudoValue | GainFn::Rmst | GainFn::AftDeviance) {
            family = LeafFamily::Survival;
        }
        let order = match &cat_order {
            None => CatOrder::FrequencyDesc,
            Some(co) => {
                if let Ok(name) = co.extract::<String>() {
                    match name.as_str() {
                        "frequency_desc" => CatOrder::FrequencyDesc,
                        "target_mean" => CatOrder::TargetMean,
                        other => {
                            return Err(PyValueError::new_err(format!(
                                "cat_order must be 'frequency_desc', 'target_mean' or a level list, got {other}"
                            )))
                        }
                    }
                } else {
                    let levels: Vec<f64> = co.extract().map_err(|_| {
                        PyValueError::new_err(
                            "cat_order must be a string or a list of level codes",
                        )
                    })?;
                    CatOrder::Explicit(levels)
                }
            }
        };
        let search = match cat_search {
            "cumulative" => CatSearch::Cumulative,
            "single_vs_rest" => CatSearch::SingleVsRest,
            "both" => CatSearch::Both,
            other => {
                return Err(PyValueError::new_err(format!(
                    "cat_search must be 'cumulative', 'single_vs_rest' or 'both', got {other}"
                )))
            }
        };
        let prune_cost = match prune {
            "none" => PruneCost::None,
            "bic" => PruneCost::Bic,
            "mdl" => PruneCost::Mdl,
            "cp" => PruneCost::Cp,
            other => {
                return Err(PyValueError::new_err(format!(
                    "prune must be 'none', 'bic', 'mdl' or 'cp', got {other}"
                )))
            }
        };
        let split_gate = match split_test {
            "none" => {
                if matches!(gain_fn, GainFn::Mob) {
                    SplitTest::Mob
                } else if matches!(gain_fn, GainFn::Ctree) {
                    SplitTest::Ctree
                } else {
                    SplitTest::None
                }
            }
            "chow" => SplitTest::Chow,
            "mob" => SplitTest::Mob,
            "ctree" => SplitTest::Ctree,
            other => {
                return Err(PyValueError::new_err(format!(
                    "split_test must be 'none', 'chow', 'mob' or 'ctree', got {other}"
                )))
            }
        };
        if !(0.0..=1.0).contains(&split_alpha) {
            return Err(PyValueError::new_err(format!(
                "split_alpha must be in [0, 1], got {split_alpha}"
            )));
        }
        let preselect = match preselect {
            "none" => mltreelib_core::preselect::Preselect::None,
            "winnow" => mltreelib_core::preselect::Preselect::Winnow,
            "boruta" => mltreelib_core::preselect::Preselect::Boruta,
            "shadowmax" => mltreelib_core::preselect::Preselect::ShadowMax,
            "variance" => mltreelib_core::preselect::Preselect::Variance,
            "mutual_info" => mltreelib_core::preselect::Preselect::MutualInfo,
            "unitree" => mltreelib_core::preselect::Preselect::UniTree,
            other => {
                return Err(PyValueError::new_err(format!(
                    "preselect must be 'none', 'winnow', 'boruta', 'shadowmax', \
                     'variance', 'mutual_info' or 'unitree', got {other}"
                )))
            }
        };
        let purity_metric = match purity_gate_metric {
            "sd" => mltreelib_core::model_tree::PurityMetric::Sd,
            "cv" => mltreelib_core::model_tree::PurityMetric::Cv,
            other => {
                return Err(PyValueError::new_err(format!(
                    "purity_gate_metric must be 'sd' or 'cv', got {other}"
                )))
            }
        };
        let knn_var = mltreelib_core::knn::KnnVariant::from_str(knn_variant).map_err(PyValueError::new_err)?;
        let knn_met = mltreelib_core::knn::DistanceMetric::from_str(knn_metric, Some(knn_p)).map_err(PyValueError::new_err)?;

        let mut params = ModelTreeParams {
            max_depth,
            min_rows,
            min_gain,
            leaf_cols: policy,
            leaf: kind,
            leaf_family: family,
            leaf_solver: leaf_solver_enum,
            l2,
            l1,
            tree_monotone: tree_mono,
            leaf_monotone: leaf_mono,
            monotone,
            interaction_groups: ig_indices,
            col_weights,
            min_relevance,
            dtypes: Vec::new(),
            gain: gain_fn,
            split_precision,
            cat_order: order,
            cat_search: search,
            leaf_digest,
            verbose,
            alpha,
            imbalance_penalty,
            col_sample,
            seed,
            prune: prune_cost,
            split_test: split_gate,
            split_alpha,
            bonferroni,
            split_cols,
            honesty,
            honesty_fraction,
            maxsurrogate,
            cp_alpha,
            max_leaf_nodes,
            minsplit,
            minbucket,
            minprob,
            preselect,
            purity_gate,
            purity_gate_metric: purity_metric,
            tail_quantile,
            tweedie_p,
            tobit_threshold,
            split_eval: eval_mode,
            presort,
            events,
            treatments,
            train_groups,
            train_ids,
            n_classes,
            mars_degree,
            mars_terms,
            mars_penalty,
            knn_k,
            knn_k_validity,
            knn_variant: knn_var,
            knn_metric: knn_met,
            knn_gamma,
            knn_p,
            shrinkage,
            max_classes,
            leaf_na_policy: leaf_na_policy_enum,
            target_vectors: None,
            fairness_penalty: fairness_penalty.unwrap_or(0.0),
            protected_col,
            fairness_metric: match fairness_metric.unwrap_or("demographic_parity").to_lowercase().as_str() {
                "equalized_odds" | "equalizedodds" | "eo" => mltreelib_core::model_tree::FairnessMetric::EqualizedOdds,
                "demographic_parity" | "demographicparity" | "dp" => mltreelib_core::model_tree::FairnessMetric::DemographicParity,
                other => return Err(PyValueError::new_err(format!("Unknown fairness_metric: '{other}'. Expected 'demographic_parity' or 'equalized_odds'"))),
            },
            sporf_d,
            sporf_projections,
            extratrees,
            ebm_fast,
            aft_dist: match aft_dist.to_lowercase().as_str() {
                "weibull" => mltreelib_core::node_stats::AftDist::Weibull,
                "lognormal" | "log_normal" => mltreelib_core::node_stats::AftDist::LogNormal,
                "loglogistic" | "log_logistic" => mltreelib_core::node_stats::AftDist::LogLogistic,
                other => return Err(PyValueError::new_err(format!("unrecognized aft_dist '{other}', expected 'weibull', 'lognormal', or 'loglogistic'"))),
            },
            cure_fraction,
            spline_knots,
        };
        if let Some(ctrl_obj) = leaf_control.or(control) {
            let ctrl = extract_leaf_control(
                ctrl_obj.py(),
                Some(ctrl_obj),
                &params,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                &[],
            )?;
            params.n_classes = params.n_classes.or(ctrl.n_classes);
            params.apply_leaf_control(inherit_leaf_inputs(ctrl, &params));
        }
        if strict_solver {
            mltreelib_core::leaf::LeafControl::from(&params)
                .validate_solver()
                .map_err(PyValueError::new_err)?;
        }
        Ok(Self {
            params,
            inner: None,
            vocabs: Vec::new(),
            interaction_group_names: ig_names,
        })
    }

    /// Fit the tree on row-major data using the selected leaf family.
    ///
    /// Args:
    ///     x: rows of predictors; `None` entries are missing (NA rows train).
    ///     y: targets; `None`/NaN drops that row for binomial/Poisson.
    ///        Gaussian NaN handling follows the historical Gaussian path.
    ///        Binomial targets are 0 or 1; Poisson targets are finite
    ///        nonnegative counts or rates (including fractional values).
    ///        Gaussian targets retain ordinary regression.
    ///     w: optional per-row sample weights (row weightage); zero-weight
    ///        rows are ignored.
    ///     dtypes: optional per-column types — "numeric" (default) or
    ///        "categorical" (values are level codes; the split search
    ///        partitions levels, no encoding).
    ///
    /// Returns: nothing; the fitted tree replaces the inner model.
    #[pyo3(signature = (x, y, w = None, dtypes = None, control = None, id_col = None, group_col = None, col_names = None, event = None, treatment = None, n_classes = None, sample_weight = None, row_weights = None, col_weights = None, feature_weights = None, train_groups = None, train_ids = None, fairness_penalty = None, protected_col = None, protected_attribute = None, fairness_metric = None, sporf_d = None, sporf_projections = None, extratrees = None, ebm_fast = None))]
    fn fit(
        &mut self,
        mut x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
        y: Bound<'_, PyAny>,
        w: Option<Vec<f64>>,
        mut dtypes: Option<Vec<String>>,
        control: Option<Bound<'_, PyAny>>,
        id_col: Option<Bound<'_, PyAny>>,
        group_col: Option<Bound<'_, PyAny>>,
        mut col_names: Option<Vec<String>>,
        event: Option<Vec<f64>>,
        treatment: Option<Vec<f64>>,
        n_classes: Option<usize>,
        sample_weight: Option<Vec<f64>>,
        row_weights: Option<Vec<f64>>,
        col_weights: Option<Vec<f64>>,
        feature_weights: Option<Vec<f64>>,
        train_groups: Option<bool>,
        train_ids: Option<bool>,
        fairness_penalty: Option<f64>,
        protected_col: Option<usize>,
        protected_attribute: Option<Bound<'_, PyAny>>,
        fairness_metric: Option<&str>,
        sporf_d: Option<usize>,
        sporf_projections: Option<usize>,
        extratrees: Option<bool>,
        ebm_fast: Option<bool>,
    ) -> PyResult<()> {
        if let Some(fm) = fairness_metric {
            self.params.fairness_metric = match fm.to_lowercase().as_str() {
                "equalized_odds" | "equalizedodds" | "eo" => mltreelib_core::model_tree::FairnessMetric::EqualizedOdds,
                "demographic_parity" | "demographicparity" | "dp" => mltreelib_core::model_tree::FairnessMetric::DemographicParity,
                other => return Err(PyValueError::new_err(format!("Unknown fairness_metric: '{other}'. Expected 'demographic_parity' or 'equalized_odds'"))),
            };
        }
        if let Some(et) = extratrees {
            self.params.extratrees = et;
        }
        if let Some(ef) = ebm_fast {
            self.params.ebm_fast = ef;
        }
        if let Some(sd) = sporf_d {
            self.params.sporf_d = sd;
        }
        if let Some(sp) = sporf_projections {
            self.params.sporf_projections = sp;
        }
        if let Some(tg) = train_groups {
            self.params.train_groups = tg;
        }
        if let Some(ti) = train_ids {
            self.params.train_ids = ti;
        }
        if let Some(fp) = fairness_penalty {
            self.params.fairness_penalty = fp;
        }
        if let Some(pc) = protected_col {
            self.params.protected_col = Some(pc);
        }
        if let Some(tr) = treatment {
            self.params.treatments = Some(tr);
            if matches!(self.params.gain, GainFn::Sse) {
                self.params.gain = GainFn::Causal;
            }
        }

        let mut drop_indices = std::collections::HashSet::new();
        let resolve_col = |spec: &Bound<'_, PyAny>, names: Option<&[String]>| -> Option<usize> {
            if let Ok(idx) = spec.extract::<usize>() {
                Some(idx)
            } else if let Ok(name) = spec.extract::<String>() {
                names.and_then(|ns| ns.iter().position(|n| n == &name))
            } else {
                None
            }
        };

        if self.params.protected_col.is_none() {
            if let Some(ref pa) = protected_attribute {
                if let Some(idx) = resolve_col(pa, col_names.as_deref()) {
                    self.params.protected_col = Some(idx);
                }
            }
        }

        if !self.params.train_ids {
            if let Some(ref id_spec) = id_col {
                if let Some(idx) = resolve_col(id_spec, col_names.as_deref()) {
                    drop_indices.insert(idx);
                }
            }
        }
        if !self.params.train_groups {
            if let Some(ref grp_spec) = group_col {
                if let Some(idx) = resolve_col(grp_spec, col_names.as_deref()) {
                    drop_indices.insert(idx);
                }
            }
        }

        if !drop_indices.is_empty() {
            let n_cols = x.first().map(|r| r.len()).unwrap_or(0);
            let keep_indices: Vec<usize> = (0..n_cols).filter(|i| !drop_indices.contains(i)).collect();
            x = x.into_iter()
                .map(|row| keep_indices.iter().map(|&i| row.get(i).cloned().unwrap_or(None)).collect())
                .collect();
            if let Some(names) = col_names.take() {
                col_names = Some(keep_indices.iter().filter_map(|&i| names.get(i).cloned()).collect());
            }
            if let Some(dts) = dtypes.take() {
                dtypes = Some(keep_indices.iter().filter_map(|&i| dts.get(i).cloned()).collect());
            }
        }

        let effective_w = w.or(sample_weight).or(row_weights);
        if let Some(cw) = col_weights.or(feature_weights) {
            self.params.col_weights = Some(cw);
        }
        if let (Some(names_groups), Some(cols)) = (&self.interaction_group_names, &col_names) {
            let mut resolved = Vec::new();
            for grp in names_groups {
                let mut idx_grp = Vec::new();
                for col_name in grp {
                    if let Some(pos) = cols.iter().position(|c| c == col_name) {
                        idx_grp.push(pos);
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "interaction group column '{col_name}' not found in col_names"
                        )));
                    }
                }
                resolved.push(idx_grp);
            }
            self.params.interaction_groups = resolved;
        }
        let (y_vec, aux_vec): (Vec<Option<f64>>, Option<Vec<f64>>) =
            if let Ok(tup) = y.extract::<(Vec<Option<f64>>, Vec<f64>)>() {
                (tup.0, Some(tup.1))
            } else if let Ok(pairs) = y.extract::<Vec<(f64, f64)>>() {
                let times = pairs.iter().map(|p| Some(p.0)).collect();
                let evs = pairs.iter().map(|p| p.1).collect();
                (times, Some(evs))
            } else if let Ok(multi) = y.extract::<Vec<Vec<f64>>>() {
                let y_first: Vec<Option<f64>> = multi.iter().map(|v| v.first().copied()).collect();
                self.params.target_vectors = Some(multi);
                (y_first, None)
            } else if let Ok(single) = y.extract::<Vec<Option<f64>>>() {
                (single, None)
            } else {
                return Err(PyTypeError::new_err(
                    "y must be a list of floats, a 2D list of vector targets, or a 2-tuple (values, aux)",
                ));
            };

        if let Some(aux) = aux_vec {
            if matches!(self.params.gain, GainFn::Causal) || self.params.treatments.is_some() {
                self.params.treatments = Some(aux);
                self.params.gain = GainFn::Causal;
            } else {
                self.params.events = Some(aux);
                if matches!(self.params.leaf_family, LeafFamily::Gaussian) {
                    self.params.leaf_family = LeafFamily::Survival;
                }
            }
        } else if let Some(evs) = event {
            self.params.events = Some(evs);
            if matches!(self.params.leaf_family, LeafFamily::Gaussian) {
                self.params.leaf_family = LeafFamily::Survival;
            }
        }
        if let Some(nc) = n_classes {
            self.params.n_classes = Some(nc);
        }
        if let Some(ctrl_obj) = control {
            let ctrl = extract_leaf_control(
                ctrl_obj.py(),
                Some(ctrl_obj),
                &self.params,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                &[],
            )?;
            self.params.n_classes = self.params.n_classes.or(ctrl.n_classes);
            self.params
                .apply_leaf_control(inherit_leaf_inputs(ctrl, &self.params));
        }
        if let Some(dts) = dtypes {
            self.params.dtypes = dts
                .iter()
                .map(|d| match d.as_str() {
                    "numeric" | "float" => Ok(ColType::Float),
                    "integer" | "int" => Ok(ColType::Integer),
                    "categorical" | "cat" | "level" => Ok(ColType::Categorical),
                    other => Err(PyValueError::new_err(format!(
                        "dtypes entries must be 'integer', 'float' or 'categorical', got {other}"
                    ))),
                })
                .collect::<PyResult<Vec<_>>>()?;
        }
        let x = coerce_rows(x, &self.params.dtypes, &mut self.vocabs, true)?;
        let d = x.first().map(|r| r.len()).unwrap_or(0);
        let names = col_names.unwrap_or_else(|| (0..d).map(|j| format!("x{j}")).collect());
        if names.len() != d {
            return Err(PyValueError::new_err(format!(
                "col_names length ({}) does not match column count ({})",
                names.len(),
                d
            )));
        }
        let weights = effective_w.unwrap_or_else(|| vec![1.0; x.len()]);
        let mut tree = ModelTree::fit_weighted_named(names, &x, &y_vec, &weights, self.params.clone())
            .map_err(PyValueError::new_err)?;
        tree.category_levels = self
            .vocabs
            .iter()
            .map(|levels| levels.as_ref().map(|levels| levels.clone().into_iter().collect()))
            .collect();

        self.inner = Some(tree);
        Ok(())
    }

    /// Retrain all leaf models on new or existing data without altering tree topology or splits.
    ///
    /// Args:
    ///     x: rows of predictors; `None` entries are missing.
    ///     y: targets.
    ///     control: optional LeafControl instance or dict.
    ///     w: optional sample weights.
    ///     mode: "internal" (updates self in place and returns self) or "external" (returns new ModelTree clone).
    ///     leaf_features: optional subset of column indices or names for leaf models.
    ///     retrain_cols: alias for leaf_features.
    ///     col_names: optional data column names to validate and remap splits.
    ///     leaf: optional leaf kind override ("linear", "mean", "median", "quantile", "huber").
    ///     leaf_family: optional leaf distribution override ("gaussian", "binomial", "poisson").
    ///     leaf_cols: optional leaf columns policy override ("all", "path_used").
    ///     l1: optional lasso penalty override.
    ///     l2: optional ridge penalty override.
    ///     leaf_digest: optional bool whether to keep leaf t-digests.
    ///     tau: optional quantile tau when leaf is "quantile" (default 0.5).
    ///
    /// Returns: updated ModelTree (self for internal mode, or new ModelTree for external mode).
    #[pyo3(signature = (
        x,
        y,
        control = None,
        w = None,
        mode = "internal",
        leaf_features = None,
        retrain_cols = None,
        col_names = None,
        leaf = None,
        leaf_family = None,
        leaf_cols = None,
        l1 = None,
        l2 = None,
        leaf_digest = None,
        tau = None,
        event = None,
        n_classes = None
    ))]
    fn retrain<'py>(
        slf: &Bound<'py, PyModelTree>,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
        y: Bound<'_, PyAny>,
        control: Option<Bound<'py, PyAny>>,
        w: Option<Vec<f64>>,
        mode: Option<&str>,
        leaf_features: Option<Bound<'py, PyAny>>,
        retrain_cols: Option<Bound<'py, PyAny>>,
        col_names: Option<Vec<String>>,
        leaf: Option<&str>,
        leaf_family: Option<&str>,
        leaf_cols: Option<&str>,
        l1: Option<f64>,
        l2: Option<f64>,
        leaf_digest: Option<bool>,
        tau: Option<f64>,
        event: Option<Vec<f64>>,
        n_classes: Option<usize>,
    ) -> PyResult<Py<PyModelTree>> {
        let is_external = match mode.unwrap_or("internal") {
            "internal" | "update" => false,
            "external" | "expose" | "new" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unrecognized mode '{other}': expected 'internal' or 'external'"
                )))
            }
        };

        let (y_vec, events_vec): (Vec<Option<f64>>, Option<Vec<f64>>) =
            if let Ok(tup) = y.extract::<(Vec<Option<f64>>, Vec<f64>)>() {
                (tup.0, Some(tup.1))
            } else if let Ok(pairs) = y.extract::<Vec<(f64, f64)>>() {
                let times = pairs.iter().map(|p| Some(p.0)).collect();
                let evs = pairs.iter().map(|p| p.1).collect();
                (times, Some(evs))
            } else if let Ok(single) = y.extract::<Vec<Option<f64>>>() {
                (single, event)
            } else {
                return Err(PyTypeError::new_err(
                    "y must be a list of floats, or a (times, events) tuple, or a list of (time, event) pairs",
                ));
            };

        let w_vec = match w {
            Some(wv) => wv,
            None => vec![1.0; y_vec.len()],
        };

        if is_external {
            let (mut new_tree, data_col_names, coerced_x, ctrl) = {
                let b = slf.borrow();
                let t = b.inner.as_ref().ok_or_else(|| PyValueError::new_err("ModelTree is not fitted"))?;
                let data_col_names = match col_names {
                    Some(names) => names,
                    None => t.col_names.clone(),
                };
                let mut ctrl = extract_leaf_control(
                    slf.py(),
                    control,
                    &b.params,
                    leaf,
                    leaf_family,
                    leaf_cols,
                    l1,
                    l2,
                    leaf_digest,
                    tau,
                    leaf_features.or(retrain_cols),
                    &data_col_names,
                )?;
                if events_vec.is_some() {
                    ctrl.events = events_vec.clone();
                }
                ctrl.n_classes = n_classes.or(b.params.n_classes).or(ctrl.n_classes);
                let mut vocabs = b.vocabs.clone();
                let coerced_x = coerce_rows(x, &b.params.dtypes, &mut vocabs, false)?;
                let mut new_tree = b.clone();
                new_tree.vocabs = vocabs;
                (new_tree, data_col_names, coerced_x, ctrl)
            };
            let t = new_tree.inner.as_mut().ok_or_else(|| PyValueError::new_err("ModelTree is not fitted"))?;
            t.retrain_weighted_named(&data_col_names, &coerced_x, &y_vec, &w_vec, ctrl)
                .map_err(PyValueError::new_err)?;
            new_tree.params = t.params.clone();
            Py::new(slf.py(), new_tree)
        } else {
            let (data_col_names, coerced_x, vocabs, ctrl) = {
                let b = slf.borrow();
                let t = b.inner.as_ref().ok_or_else(|| PyValueError::new_err("ModelTree is not fitted"))?;
                let data_col_names = match col_names {
                    Some(names) => names,
                    None => t.col_names.clone(),
                };
                let mut ctrl = extract_leaf_control(
                    slf.py(),
                    control,
                    &b.params,
                    leaf,
                    leaf_family,
                    leaf_cols,
                    l1,
                    l2,
                    leaf_digest,
                    tau,
                    leaf_features.or(retrain_cols),
                    &data_col_names,
                )?;
                if events_vec.is_some() {
                    ctrl.events = events_vec.clone();
                }
                ctrl.n_classes = n_classes.or(b.params.n_classes).or(ctrl.n_classes);
                let mut vocabs = b.vocabs.clone();
                let coerced_x = coerce_rows(x, &b.params.dtypes, &mut vocabs, false)?;
                (data_col_names, coerced_x, vocabs, ctrl)
            };
            {
                let mut b = slf.borrow_mut();
                let t = b.inner.as_mut().ok_or_else(|| PyValueError::new_err("ModelTree is not fitted"))?;
                t.retrain_weighted_named(&data_col_names, &coerced_x, &y_vec, &w_vec, ctrl)
                    .map_err(PyValueError::new_err)?;
                b.params = t.params.clone();
                b.vocabs = vocabs;
            }
            Ok(slf.clone().unbind())
        }
    }

    /// Empirical conditional target quantiles from the leaves' t-digests:
    /// any tau in [0, 1] after fitting. These are not quantiles of the fitted
    /// binomial/Poisson distribution; use cdf for that distribution's CDF.
    ///
    /// Args:
    ///     x: rows of predictors (NA never blocks).
    ///     tau: quantile level.
    ///
    /// Returns: one quantile per row (NaN when a leaf's digest was dropped).
    fn quantile(
        &mut self,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
        tau: f64,
    ) -> PyResult<Vec<f64>> {
        let x = coerce_rows(x, &self.params.dtypes, &mut self.vocabs, false)?;
        Ok(self.tree()?.quantile(&x, tau))
    }

    /// The training log (populated when `verbose > 0`): scan config, per
    /// depth/per node candidate counts, precision grid, timings.
    ///
    /// Args: none.
    ///
    /// Returns: log lines.
    fn training_log(&self) -> PyResult<Vec<String>> {
        Ok(self.tree()?.trace.clone())
    }

    /// Serialize the fitted model as versioned JSON.
    fn to_json(&self) -> PyResult<String> {
        self.tree()?.to_json().map_err(model_persistence_error)
    }

    /// Restore a fitted model from versioned JSON.
    #[staticmethod]
    fn from_json(data: &str) -> PyResult<Self> {
        ModelTree::from_json(data)
            .map(Self::from_fitted_tree)
            .map_err(model_persistence_error)
    }

    /// Serialize the fitted model as UTF-8 JSON bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let data = self.tree()?.to_bytes().map_err(model_persistence_error)?;
        Ok(PyBytes::new(py, &data))
    }

    /// Restore a fitted model from UTF-8 JSON bytes.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        ModelTree::from_bytes(data)
            .map(Self::from_fitted_tree)
            .map_err(model_persistence_error)
    }

    /// Save the fitted model to a UTF-8 JSON file.
    fn save(&self, path: std::path::PathBuf) -> PyResult<()> {
        self.tree()?.save(path).map_err(model_persistence_error)
    }

    /// Load a fitted model from a UTF-8 JSON file.
    #[staticmethod]
    fn load(path: std::path::PathBuf) -> PyResult<Self> {
        ModelTree::load(path)
            .map(Self::from_fitted_tree)
            .map_err(model_persistence_error)
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.to_bytes(py)
    }

    fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        *self = Self::from_bytes(state)?;
        Ok(())
    }

    /// Predict with flexible output kinds:
    ///   kind="response" (default): expected value / class / median survival time.
    ///   kind="proba": class probabilities ((N, C) matrix for Softmax, 1D float list for Binomial/Probit).
    ///   kind="dist": (mean, std) location-scale tuples for Heteroscedastic.
    ///   kind="survival": S(t) survival curve over times grid.
    ///   kind="hazard": Lambda(t) cumulative hazard curve over times grid.
    ///   kind="cif": F_c(t) cumulative incidence function for cause over times grid.
    ///   kind="risk": predicted risk score (higher score = shorter expected survival / higher risk).
    ///   kind="leaf_id": integer leaf ID per row.
    #[pyo3(signature = (
        x, kind = "response", times = None, neighbors = None, cause = None,
        *, coverage = None, alpha = None, threshold = 0.5
    ))]
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
        kind: Option<&str>,
        times: Option<Vec<f64>>,
        neighbors: Option<usize>,
        cause: Option<usize>,
        coverage: Option<f64>,
        alpha: Option<f64>,
        threshold: f64,
    ) -> PyResult<Py<PyAny>> {
        let x = coerce_rows(x, &self.params.dtypes, &mut self.vocabs, false)?;
        let t = self.tree()?;
        match kind.unwrap_or("response") {
            "response" => {
                if t.params.target_vectors.is_some() {
                    if let Some(vecs) = t.predict_vector(&x) {
                        return vecs.into_py_any(py);
                    }
                }
                let preds = if let Some(k) = neighbors {
                    t.predict_adjusted(&x, k)
                } else {
                    t.predict(&x)
                };
                preds.into_py_any(py)
            }
            "adjusted" => t.predict_adjusted(&x, neighbors.unwrap_or(5)).into_py_any(py),
            "interval" => {
                if coverage.is_some() && alpha.is_some() {
                    return Err(PyValueError::new_err("Supply either coverage or alpha"));
                }
                let coverage = coverage.or_else(|| alpha.map(|alpha| 1.0 - alpha)).unwrap_or(0.90);
                t.predict_interval(&x, coverage)
                    .map_err(PyValueError::new_err)?
                    .into_py_any(py)
            }
            "label" => t.predict_label(&x, threshold)
                .map_err(PyValueError::new_err)?
                .into_py_any(py),
            "vector" => {
                let vecs = t.predict_vector(&x).ok_or_else(|| PyValueError::new_err("ModelTree has no vector predictions (was not fitted on vector targets)"))?;
                vecs.into_py_any(py)
            }
            "proba" | "prob" => {
                match t.params.leaf_family {
                    LeafFamily::Softmax => {
                        let mat = t.predict_proba_matrix(&x).map_err(PyValueError::new_err)?;
                        mat.into_py_any(py)
                    }
                    LeafFamily::Binomial | LeafFamily::Probit => {
                        let vec = t.predict_proba(&x).map_err(PyValueError::new_err)?;
                        vec.into_py_any(py)
                    }
                    _ => {
                        match t.params.leaf {
                            LeafKind::Prob => {
                                let preds = t.predict(&x);
                                preds.into_py_any(py)
                            }
                            LeafKind::Odds => {
                                let odds = t.predict(&x);
                                let probs: Vec<f64> = odds.iter().map(|&o| o / (1.0 + o)).collect();
                                probs.into_py_any(py)
                            }
                            LeafKind::LogOdds => {
                                let log_odds = t.predict(&x);
                                let probs: Vec<f64> = log_odds.iter().map(|&lo| 1.0 / (1.0 + (-lo).exp())).collect();
                                probs.into_py_any(py)
                            }
                            _ => {
                                return Err(PyValueError::new_err(format!(
                                    "kind='proba' requires 'binomial', 'probit', or 'softmax' family, got {:?}",
                                    t.params.leaf_family
                                )));
                            }
                        }
                    }
                }
            }
            "odds" => {
                match t.params.leaf {
                    LeafKind::Odds => {
                        let preds = t.predict(&x);
                        preds.into_py_any(py)
                    }
                    LeafKind::LogOdds => {
                        let preds = t.predict(&x);
                        let odds: Vec<f64> = preds.iter().map(|&lo| lo.exp()).collect();
                        odds.into_py_any(py)
                    }
                    _ => {
                        let probs = if matches!(t.params.leaf_family, LeafFamily::Binomial | LeafFamily::Probit) {
                            t.predict_proba(&x).map_err(PyValueError::new_err)?
                        } else {
                            t.predict(&x).into_iter().map(|p| p.clamp(1e-6, 1.0 - 1e-6)).collect()
                        };
                        let odds: Vec<f64> = probs.iter().map(|&p| {
                            let p_sm = p.clamp(1e-6, 1.0 - 1e-6);
                            p_sm / (1.0 - p_sm)
                        }).collect();
                        odds.into_py_any(py)
                    }
                }
            }
            "log_odds" | "logit" => {
                match t.params.leaf {
                    LeafKind::LogOdds => {
                        let preds = t.predict(&x);
                        preds.into_py_any(py)
                    }
                    LeafKind::Odds => {
                        let preds = t.predict(&x);
                        let log_odds: Vec<f64> = preds.iter().map(|&o| o.max(1e-6).ln()).collect();
                        log_odds.into_py_any(py)
                    }
                    _ => {
                        let probs = if matches!(t.params.leaf_family, LeafFamily::Binomial | LeafFamily::Probit) {
                            t.predict_proba(&x).map_err(PyValueError::new_err)?
                        } else {
                            t.predict(&x).into_iter().map(|p| p.clamp(1e-6, 1.0 - 1e-6)).collect()
                        };
                        let log_odds: Vec<f64> = probs.iter().map(|&p| {
                            let p_sm = p.clamp(1e-6, 1.0 - 1e-6);
                            (p_sm / (1.0 - p_sm)).ln()
                        }).collect();
                        log_odds.into_py_any(py)
                    }
                }
            }
            "median" => {
                let leaf_ids = t.apply(&x);
                let entries = t.leaf_entries();
                let medians: Vec<f64> = leaf_ids.iter().map(|&lid| {
                    entries.get(lid).map(|e| e.0.stats.median).unwrap_or(0.0)
                }).collect();
                medians.into_py_any(py)
            }
            "stats" | "leaf_stats" => {
                let leaf_ids = t.apply(&x);
                let entries = t.leaf_entries();
                let mut list = Vec::with_capacity(x.len());
                for &lid in &leaf_ids {
                    let st = if let Some(e) = entries.get(lid) {
                        &e.0.stats
                    } else if let Some(first) = entries.first() {
                        &first.0.stats
                    } else {
                        return Err(PyValueError::new_err("ModelTree has no leaves"));
                    };
                    let d = PyDict::new(py);
                    d.set_item("mean", st.mean)?;
                    d.set_item("std", st.std)?;
                    d.set_item("var", st.var)?;
                    d.set_item("median", st.median)?;
                    d.set_item("mad", st.mad)?;
                    d.set_item("iqr", st.iqr)?;
                    d.set_item("min", st.min)?;
                    d.set_item("max", st.max)?;
                    d.set_item("skew", st.skew)?;
                    d.set_item("kurtosis", st.kurtosis)?;
                    d.set_item("prob", st.prob)?;
                    d.set_item("odds", st.odds)?;
                    d.set_item("log_odds", st.log_odds)?;
                    d.set_item("majority_class", st.majority_class)?;
                    d.set_item("entropy", st.entropy)?;
                    d.set_item("gini", st.gini)?;
                    d.set_item("zero_fraction", st.zero_fraction)?;
                    d.set_item("dispersion", st.dispersion)?;
                    list.push(d);
                }
                list.into_py_any(py)
            }
            "dist" | "params" => {
                let dist = t.predict_dist(&x).map_err(PyValueError::new_err)?;
                dist.into_py_any(py)
            }
            "survival" => {
                let t_grid = times.unwrap_or_else(|| vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]);
                let surv = t.predict_survival(&x, &t_grid).map_err(PyValueError::new_err)?;
                surv.into_py_any(py)
            }
            "hazard" => {
                let t_grid = times.unwrap_or_else(|| vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]);
                let haz = t.predict_hazard(&x, &t_grid).map_err(PyValueError::new_err)?;
                haz.into_py_any(py)
            }
            "cif" => {
                let t_grid = times.unwrap_or_else(|| vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]);
                let target_cause = cause.unwrap_or(1);
                let cif = t.predict_cif(&x, &t_grid, target_cause).map_err(PyValueError::new_err)?;
                cif.into_py_any(py)
            }
            "risk" => {
                let risk = t.predict_risk(&x).map_err(PyValueError::new_err)?;
                risk.into_py_any(py)
            }
            "uplift" | "cate" => {
                let uplift = t.predict_uplift(&x).map_err(PyValueError::new_err)?;
                uplift.into_py_any(py)
            }
            "leaf_id" => {
                let ids = t.apply(&x);
                ids.into_py_any(py)
            }
            other => {
                let mut kinds = vec!["response", "interval", "adjusted", "odds", "log_odds", "median", "stats", "leaf_id"];
                if t.params.target_vectors.is_some() {
                    kinds.push("vector");
                }
                if t.params.treatments.is_some() {
                    kinds.extend(["uplift", "cate"]);
                }
                match t.params.leaf_family {
                    LeafFamily::Binomial | LeafFamily::Probit => kinds.extend(["proba", "label"]),
                    LeafFamily::Softmax => kinds.push("proba"),
                    LeafFamily::Heteroscedastic => kinds.push("dist"),
                    LeafFamily::Survival | LeafFamily::Aft | LeafFamily::Cure | LeafFamily::Spline => {
                        kinds.extend(["survival", "hazard", "cif", "risk"]);
                    }
                    _ if matches!(t.params.leaf, LeafKind::Prob | LeafKind::Odds | LeafKind::LogOdds) => {
                        kinds.push("proba");
                    }
                    _ => {}
                }
                Err(PyValueError::new_err(format!(
                    "unrecognized prediction kind {other:?}; available kinds: {}",
                    kinds.join(", ")
                )))
            }
        }
    }

    /// Predict multi-dimensional vector targets for row-major `x` (when trained on vector targets).
    #[pyo3(signature = (x))]
    fn predict_vector<'py>(
        &mut self,
        py: Python<'py>,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
    ) -> PyResult<Py<PyAny>> {
        let x = coerce_rows(x, &self.params.dtypes, &mut self.vocabs, false)?;
        let t = self.tree()?;
        let vecs = t.predict_vector(&x).ok_or_else(|| PyValueError::new_err("ModelTree has no vector predictions (was not fitted on vector targets)"))?;
        vecs.into_py_any(py)
    }

    /// Predict class probabilities (convenience alias for `predict(x, kind="proba")`).
    fn predict_proba<'py>(
        &mut self,
        py: Python<'py>,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
    ) -> PyResult<Py<PyAny>> {
        self.predict(py, x, Some("proba"), None, None, None, None, None, 0.5)
    }

    /// Hard binary class labels (0 or 1), one label per input row.
    ///
    /// Args:
    ///     x: raw rows of predictors.
    ///     threshold: probability threshold (default 0.5).
    ///
    /// Returns: one integer label (0 or 1) per row.
    ///
    /// Raises: ValueError unless the fitted leaf family is binomial.
    #[pyo3(signature = (x, threshold = 0.5))]
    fn predict_label(
        &mut self,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
        threshold: f64,
    ) -> PyResult<Vec<usize>> {
        let x = coerce_rows(x, &self.params.dtypes, &mut self.vocabs, false)?;
        self.tree()?.predict_label(&x, threshold).map_err(PyValueError::new_err)
    }

    /// Predict with Cubist-style k-NN sample neighbor adjustment.
    #[pyo3(signature = (x, neighbors = 5))]
    fn predict_adjusted<'py>(
        &mut self,
        py: Python<'py>,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
        neighbors: usize,
    ) -> PyResult<Py<PyAny>> {
        let x = coerce_rows(x, &self.params.dtypes, &mut self.vocabs, false)?;
        let t = self.tree()?;
        let preds = t.predict_adjusted(&x, neighbors);
        preds.into_py_any(py)
    }

    /// Extract a C5.0-style rule set from this fitted tree.
    #[pyo3(signature = (x = None, y = None, prune = true, cf = 0.25))]
    fn extract_rules(
        &mut self,
        x: Option<Vec<Vec<Option<Bound<'_, PyAny>>>>>,
        y: Option<Vec<f64>>,
        prune: Option<bool>,
        cf: Option<f64>,
    ) -> PyResult<PyRuleSet> {
        let x_coerced = match x {
            Some(rows) => Some(coerce_rows(rows, &self.params.dtypes, &mut self.vocabs, false)?),
            None => None,
        };
        let t = self.tree()?;
        let ruleset = t.extract_rules(
            x_coerced.as_deref(),
            y.as_deref(),
            prune.unwrap_or(true),
            cf.unwrap_or(0.25),
        );
        Ok(PyRuleSet { inner: ruleset })
    }

    /// Format a leaf's regression model as an R summary(lm) / Python statsmodels table.
    fn leaf_summary(&self, leaf_id: usize) -> PyResult<String> {
        let t = self.tree()?;
        t.leaf_summary(leaf_id).map_err(PyValueError::new_err)
    }

    /// Print comprehensive summary of all leaf models across the tree.
    fn summary(&self) -> PyResult<String> {
        let t = self.tree()?;
        Ok(t.summary())
    }

    /// Empirical prediction interval `(lower, upper)` per row at the requested coverage.
    ///
    /// Args:
    ///     x: raw rows of predictors.
    ///     coverage: coverage probability in (0, 1) (default 0.90 for a 90% interval).
    ///
    /// Returns: list of `(lower, upper)` tuples.
    #[pyo3(signature = (x, coverage = 0.90))]
    fn predict_interval(
        &mut self,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
        coverage: f64,
    ) -> PyResult<Vec<(f64, f64)>> {
        let x = coerce_rows(x, &self.params.dtypes, &mut self.vocabs, false)?;
        self.tree()?.predict_interval(&x, coverage).map_err(PyValueError::new_err)
    }

    /// Fitted binomial/Poisson CDF P(Y <= value), conditional on each row.
    ///
    /// Args:
    ///     x: raw rows of predictors, with the same missing/level handling as predict.
    ///     value: response threshold; discrete CDFs use its integer floor.
    ///
    /// Returns: one fitted-distribution probability per row. Unlike quantile,
    /// this uses the fitted family, not the empirical target t-digest.
    ///
    /// Raises: ValueError for Gaussian leaves or an invalid threshold.
    fn cdf(
        &mut self,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
        value: f64,
    ) -> PyResult<Vec<f64>> {
        let x = coerce_rows(x, &self.params.dtypes, &mut self.vocabs, false)?;
        self.tree()?.cdf(&x, value).map_err(PyValueError::new_err)
    }

    /// Fitted leaf models with their empirical target distribution stats.
    ///
    /// Args: none.
    ///
    /// Returns: one dict per leaf with `cols`, `coef`, `na_coef`,
    /// `intercept`, `response_constant`, `n`, `mean`, `std`, `min`, `max`,
    /// `quantiles`.
    /// Linear `coef`, `na_coef`, and `intercept` use the family's link scale
    /// (identity, logit, or log); `mean` and other stats describe raw targets.
    /// Constant leaves have no link-scale coefficient vector: `intercept`
    /// is `None`, while `response_constant` is the response-scale constant.
    fn leaves(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let t = self.tree()?;
        let mut out = Vec::new();
        for (leaf, n, leaf_id, node_id) in t.leaf_entries() {
            let d = PyDict::new(py);
            d.set_item("id", leaf_id)?;
            d.set_item("node_id", node_id)?;
            let (cols, coef, na_coef, intercept) = match &leaf.linear {
                Some(m) => (
                    m.cols.iter().map(|&j| t.col_names[j].clone()).collect::<Vec<_>>(),
                    m.coef.clone(),
                    m.na_coef.clone(),
                    Some(m.intercept),
                ),
                None => (Vec::new(), Vec::new(), Vec::new(), None),
            };
            d.set_item("cols", cols)?;
            d.set_item("coef", coef)?;
            d.set_item("na_coef", na_coef)?;
            d.set_item("intercept", intercept)?;
            d.set_item("response_constant", leaf.center.unwrap_or(leaf.stats.mean))?;
            d.set_item("n", n)?;
            let st = &leaf.stats;
            d.set_item("mean", st.mean)?;
            d.set_item("std", st.std)?;
            d.set_item("var", st.var)?;
            d.set_item("median", st.median)?;
            d.set_item("mad", st.mad)?;
            d.set_item("iqr", st.iqr)?;
            d.set_item("min", st.min)?;
            d.set_item("max", st.max)?;
            d.set_item("skew", st.skew)?;
            d.set_item("kurtosis", st.kurtosis)?;
            d.set_item("prob", st.prob)?;
            d.set_item("odds", st.odds)?;
            d.set_item("log_odds", st.log_odds)?;
            d.set_item("majority_class", st.majority_class)?;
            d.set_item("entropy", st.entropy)?;
            d.set_item("gini", st.gini)?;
            d.set_item("zero_fraction", st.zero_fraction)?;
            d.set_item("dispersion", st.dispersion)?;
            let stats_dict = PyDict::new(py);
            stats_dict.set_item("mean", st.mean)?;
            stats_dict.set_item("std", st.std)?;
            stats_dict.set_item("var", st.var)?;
            stats_dict.set_item("median", st.median)?;
            stats_dict.set_item("mad", st.mad)?;
            stats_dict.set_item("iqr", st.iqr)?;
            stats_dict.set_item("min", st.min)?;
            stats_dict.set_item("max", st.max)?;
            stats_dict.set_item("skew", st.skew)?;
            stats_dict.set_item("kurtosis", st.kurtosis)?;
            stats_dict.set_item("prob", st.prob)?;
            stats_dict.set_item("odds", st.odds)?;
            stats_dict.set_item("log_odds", st.log_odds)?;
            stats_dict.set_item("majority_class", st.majority_class)?;
            stats_dict.set_item("entropy", st.entropy)?;
            stats_dict.set_item("gini", st.gini)?;
            stats_dict.set_item("zero_fraction", st.zero_fraction)?;
            stats_dict.set_item("dispersion", st.dispersion)?;
            d.set_item("stats", stats_dict)?;
            let qs: Vec<Option<f64>> = [0.1, 0.25, 0.5, 0.75, 0.9]
                .iter()
                .map(|&q| leaf.stats.quantile(q))
                .collect();
            d.set_item("quantiles", qs)?;
            if let Some(sm) = &leaf.softmax {
                d.set_item("softmax_probs", sm.probs.clone())?;
                d.set_item("n_classes", sm.n_classes)?;
            }
            if let Some(surv) = &leaf.survival {
                d.set_item("survival_times", surv.times.clone())?;
                d.set_item("survival_probs", surv.survival.clone())?;
                d.set_item("cum_hazard", surv.cum_hazard.clone())?;
                d.set_item("median_time", surv.median_time)?;
            }
            if let Some(het) = &leaf.heteroscedastic {
                d.set_item("scale", het.scale)?;
            }
            if let Some(m) = &leaf.mars {
                let m_cols: Vec<String> = leaf
                    .mars_cols
                    .as_ref()
                    .map(|cols| cols.iter().map(|&j| t.col_names[j].clone()).collect())
                    .unwrap_or_default();
                d.set_item("mars_cols", m_cols)?;
                d.set_item("mars_terms", m.n_terms())?;
                d.set_item("mars_rss", m.rss)?;
                d.set_item("mars_gcv", m.gcv)?;
            }
            out.push(d.into_any().unbind());
        }
        Ok(out)
    }

    /// Return the list of 0D stats dictionaries for all leaves.
    fn leaf_stats<'py>(&self, py: Python<'py>) -> PyResult<Vec<Py<PyDict>>> {
        let t = self.tree()?;
        let leaves = t.leaves();
        let mut out = Vec::with_capacity(leaves.len());
        for (leaf, _) in leaves {
            let st = &leaf.stats;
            let d = PyDict::new(py);
            d.set_item("mean", st.mean)?;
            d.set_item("std", st.std)?;
            d.set_item("var", st.var)?;
            d.set_item("median", st.median)?;
            d.set_item("mad", st.mad)?;
            d.set_item("iqr", st.iqr)?;
            d.set_item("min", st.min)?;
            d.set_item("max", st.max)?;
            d.set_item("skew", st.skew)?;
            d.set_item("kurtosis", st.kurtosis)?;
            d.set_item("prob", st.prob)?;
            d.set_item("odds", st.odds)?;
            d.set_item("log_odds", st.log_odds)?;
            d.set_item("majority_class", st.majority_class)?;
            d.set_item("entropy", st.entropy)?;
            d.set_item("gini", st.gini)?;
            d.set_item("zero_fraction", st.zero_fraction)?;
            d.set_item("dispersion", st.dispersion)?;
            out.push(d.unbind());
        }
        Ok(out)
    }

    /// Return the leaf ID (0..n_leaves-1) for each input row.
    ///
    /// Args:
    ///     x: raw rows of predictors.
    ///
    /// Returns: one integer leaf ID per row.
    #[pyo3(signature = (x))]
    fn apply(
        &mut self,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
    ) -> PyResult<Vec<usize>> {
        let x = coerce_rows(x, &self.params.dtypes, &mut self.vocabs, false)?;
        Ok(self.tree()?.apply(&x))
    }

    /// Return row indices or entity IDs grouped by leaf ID: {leaf_id: [indices_or_ids]}.
    ///
    /// Args:
    ///     x: raw rows of predictors.
    ///     id_col: optional column name, column index, or list of IDs.
    ///         If None (default), returns integer row indices [0, 1, ...].
    ///         If provided, returns the corresponding entity ID values.
    ///     col_names: optional data column names when x has different layout than training data.
    ///
    /// Returns: dict mapping integer leaf_id to a list of row indices or entity IDs.
    #[pyo3(signature = (x, id_col = None, col_names = None))]
    fn leaf_samples(
        &mut self,
        py: Python<'_>,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
        id_col: Option<Bound<'_, PyAny>>,
        col_names: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let t = self.inner.as_ref().ok_or_else(|| PyValueError::new_err("ModelTree is not fitted"))?;
        let n_rows = x.len();

        let ids: Option<Vec<Py<PyAny>>> = match id_col {
            None => None,
            Some(id_obj) => {
                if let Ok(name) = id_obj.extract::<String>() {
                    let col_idx = match &col_names {
                        Some(names) => names.iter().position(|c| c == &name),
                        None => t.col_names.iter().position(|c| c == &name),
                    };
                    let idx = col_idx.ok_or_else(|| {
                        PyValueError::new_err(format!("id_col '{name}' not found in columns"))
                    })?;
                    let mut extracted = Vec::with_capacity(n_rows);
                    for row in &x {
                        let id_val = row
                            .get(idx)
                            .and_then(|c| c.as_ref())
                            .map(|c| c.clone().unbind())
                            .unwrap_or_else(|| py.None());
                        extracted.push(id_val);
                    }
                    Some(extracted)
                } else if let Ok(idx) = id_obj.extract::<usize>() {
                    let mut extracted = Vec::with_capacity(n_rows);
                    for row in &x {
                        let id_val = row
                            .get(idx)
                            .and_then(|c| c.as_ref())
                            .map(|c| c.clone().unbind())
                            .unwrap_or_else(|| py.None());
                        extracted.push(id_val);
                    }
                    Some(extracted)
                } else if let Ok(seq) = id_obj.extract::<Vec<Py<PyAny>>>() {
                    if seq.len() != n_rows {
                        return Err(PyValueError::new_err(format!(
                            "id_col sequence length ({}) does not match number of rows ({})",
                            seq.len(),
                            n_rows
                        )));
                    }
                    Some(seq)
                } else {
                    return Err(PyTypeError::new_err(
                        "id_col must be a column name string, integer index, or list of IDs",
                    ));
                }
            }
        };

        let coerced_x = coerce_rows(x, &self.params.dtypes, &mut self.vocabs, false)?;
        let dict = PyDict::new(py);

        let sample_map = match col_names {
            Some(names) => {
                let mut col_map = vec![usize::MAX; t.col_names.len()];
                for (j, col_name) in t.col_names.iter().enumerate() {
                    if let Some(pos) = names.iter().position(|c| c == col_name) {
                        col_map[j] = pos;
                    } else if t.is_split_column(j) {
                        return Err(PyValueError::new_err(format!(
                            "required split column '{col_name}' missing from data columns"
                        )));
                    }
                }
                t.leaf_samples_mapped(&coerced_x, &col_map)
            }
            None => t.leaf_samples(&coerced_x),
        };

        for (leaf_id, indices) in sample_map {
            match &ids {
                Some(id_vec) => {
                    let leaf_ids: Vec<Py<PyAny>> = indices
                        .into_iter()
                        .map(|idx| id_vec[idx].clone_ref(py))
                        .collect();
                    dict.set_item(leaf_id, leaf_ids)?;
                }
                None => {
                    dict.set_item(leaf_id, indices)?;
                }
            }
        }

        Ok(dict.into_any().unbind())
    }

    /// Per-level timing profile (group/sort, scan, leaf fit).
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

    /// Learned splits.
    ///
    /// Args: none.
    ///
    /// Returns: one dict per split with `col`, `threshold`, `missing_left`,
    /// `gain` (goodness of split).
    fn splits(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let t = self.tree()?;
        let mut out = Vec::new();
        for s in t.splits() {
            let d = PyDict::new(py);
            d.set_item("col", t.col_names[s.col].clone())?;
            d.set_item("threshold", s.threshold)?;
            d.set_item("missing_left", s.missing_left)?;
            d.set_item("gain", s.gain)?;
            out.push(d.into_any().unbind());
        }
        Ok(out)
    }

    /// Root node split, if the tree split at the root.
    fn root_split(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let t = self.tree()?;
        if let Some(s) = t.root_split() {
            let d = PyDict::new(py);
            d.set_item("col", t.col_names[s.col].clone())?;
            d.set_item("threshold", s.threshold)?;
            d.set_item("missing_left", s.missing_left)?;
            d.set_item("gain", s.gain)?;
            if let Some(ob_idx) = s.oblique {
                if let Some(proj) = t.oblique_parts().get(ob_idx) {
                    let ob_dict = PyDict::new(py);
                    let names: Vec<String> = proj.cols.iter().map(|&c| t.col_names.get(c).cloned().unwrap_or_else(|| format!("col_{}", c))).collect();
                    ob_dict.set_item("cols", names.clone())?;
                    ob_dict.set_item("weights", proj.weights.clone())?;
                    let eq = names.iter().zip(proj.weights.iter())
                        .map(|(n, &w)| format!("{:+.*}*{}", 2, w, n))
                        .collect::<Vec<_>>()
                        .join(" ");
                    ob_dict.set_item("equation", format!("{} <= {:.4}", eq, s.threshold))?;
                    d.set_item("oblique", ob_dict)?;
                }
            }
            if let Some(prof) = t.profiles().iter().find(|p| p.depth == 0) {
                if let Some(gate) = &prof.gate {
                    let gd = PyDict::new(py);
                    gd.set_item("test_name", gate.test_name)?;
                    gd.set_item("statistic", gate.f)?;
                    gd.set_item("p_value", gate.p)?;
                    gd.set_item("alpha", gate.alpha)?;
                    gd.set_item("passed", gate.passed)?;
                    let mut vars = Vec::new();
                    for &(col, stat, adj_p) in &gate.var_p_values {
                        let vd = PyDict::new(py);
                        vd.set_item("col", col)?;
                        vd.set_item("name", t.col_names.get(col).cloned().unwrap_or_else(|| format!("col_{}", col)))?;
                        vd.set_item("statistic", stat)?;
                        vd.set_item("p_value", adj_p)?;
                        vars.push(vd.into_any().unbind());
                    }
                    gd.set_item("variables", vars)?;
                    d.set_item("gate", gd)?;
                }
            }
            Ok(Some(d.into_any().unbind()))
        } else {
            Ok(None)
        }
    }

    /// Statistical split diagnostics per node (MOB / Chow / CTree hypothesis tests).
    /// Returns a list of dicts with test name, statistic, p-value, alpha,
    /// passed flag, and per-variable test results.
    fn split_diagnostics(&self, py: Python<'_>) -> PyResult<Vec<Option<Py<PyAny>>>> {
        let t = self.tree()?;
        let mut out = Vec::new();
        for prof in t.profiles() {
            if let Some(gate) = &prof.gate {
                let d = PyDict::new(py);
                d.set_item("depth", prof.depth)?;
                d.set_item("rows", prof.rows)?;
                d.set_item("test_name", gate.test_name)?;
                d.set_item("statistic", gate.f)?;
                d.set_item("p_value", gate.p)?;
                d.set_item("alpha", gate.alpha)?;
                d.set_item("passed", gate.passed)?;

                let mut vars = Vec::new();
                for &(col, stat, adj_p) in &gate.var_p_values {
                    let vd = PyDict::new(py);
                    vd.set_item("col", col)?;
                    vd.set_item("name", t.col_names.get(col).cloned().unwrap_or_else(|| format!("col_{}", col)))?;
                    vd.set_item("statistic", stat)?;
                    vd.set_item("p_value", adj_p)?;
                    vars.push(vd.into_any().unbind());
                }
                d.set_item("variables", vars)?;
                out.push(Some(d.into_any().unbind()));
            } else {
                out.push(None);
            }
        }
        Ok(out)
    }

    /// Returns list of detected pairwise feature interactions from EBM FAST screening.
    fn fast_interactions(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let t = self.tree()?;
        let mut out = Vec::new();
        for inter in t.fast_interactions() {
            let d = PyDict::new(py);
            d.set_item("col1", inter.col1)?;
            d.set_item("col2", inter.col2)?;
            d.set_item("col1_name", &inter.col1_name)?;
            d.set_item("col2_name", &inter.col2_name)?;
            d.set_item("score", inter.score)?;
            d.set_item("cut1", inter.cut1)?;
            d.set_item("cut2", inter.cut2)?;
            out.push(d.into_any().unbind());
        }
        Ok(out)
    }

    /// Feature importance — supports `"gain"` (default, rpart split gain share),
    /// `"shapley"` (mean absolute TreeSHAP over rows), or `"permutation"` (score degradation).
    #[pyo3(signature = (method="gain", x=None, y=None, metric=None, n_repeats=None, seed=None))]
    fn feature_importance(
        &self,
        method: &str,
        x: Option<Vec<Vec<Option<f64>>>>,
        y: Option<Vec<Option<f64>>>,
        metric: Option<&str>,
        n_repeats: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Vec<(String, f64)>> {
        let t = self.tree()?;
        match method {
            "gain" => Ok(t.feature_importances()),
            "shapley" => {
                let x_data = x.ok_or_else(|| PyValueError::new_err("feature_importance(method='shapley') requires x"))?;
                Ok(t.shapley_importance(&x_data))
            }
            "permutation" => {
                let x_data = x.ok_or_else(|| PyValueError::new_err("feature_importance(method='permutation') requires x"))?;
                let y_data = y.ok_or_else(|| PyValueError::new_err("feature_importance(method='permutation') requires y"))?;
                t.permutation_importance(&x_data, &y_data, metric.unwrap_or("rmse"), n_repeats.unwrap_or(5), seed.unwrap_or(42))
                    .map_err(PyValueError::new_err)
            }
            other => Err(PyValueError::new_err(format!(
                "unknown importance method '{other}'; expected 'gain', 'shapley', or 'permutation'"
            ))),
        }
    }

    /// Permutation feature importance: per-column validation loss degradation
    /// when the column is shuffled across rows.
    #[pyo3(signature = (x, y, metric="rmse", n_repeats=5, seed=42))]
    fn permutation_importance(
        &self,
        x: Vec<Vec<Option<f64>>>,
        y: Vec<Option<f64>>,
        metric: &str,
        n_repeats: usize,
        seed: u64,
    ) -> PyResult<Vec<(String, f64)>> {
        self.tree()?
            .permutation_importance(&x, &y, metric, n_repeats, seed)
            .map_err(PyValueError::new_err)
    }

    /// Exact TreeSHAP feature attributions for a single row prediction:
    /// efficiency invariant: `base_value + \sum values == prediction`.
    fn shapley(&self, py: Python<'_>, row: Vec<Option<Bound<'_, PyAny>>>) -> PyResult<Py<PyDict>> {
        let t = self.tree()?;
        let mut vocabs = self.vocabs.clone();
        let coerced = coerce_rows(vec![row], &self.params.dtypes, &mut vocabs, false)?;
        let row_f64 = coerced.into_iter().next().unwrap_or_default();
        let sh = t.shapley(&row_f64);
        let d = PyDict::new(py);
        d.set_item("base_value", sh.base_value)?;
        d.set_item(
            "values",
            sh.values
                .into_iter()
                .map(|(c, v)| (t.col_names[c].clone(), v))
                .collect::<Vec<_>>(),
        )?;
        d.set_item("prediction", sh.prediction)?;
        Ok(d.unbind())
    }

    /// DALEX-style break-down of one response-scale prediction.
    ///
    /// Args:
    ///     row: one row of predictors (`None` = missing). Accepts mixed numeric and categorical strings.
    ///
    /// Returns: `{"baseline", "steps": [[column_name, contribution], ...],
    /// "residual", "prediction"}`; baseline + steps + residual == prediction.
    fn break_down(&self, py: Python<'_>, row: Vec<Option<Bound<'_, PyAny>>>) -> PyResult<Py<PyDict>> {
        let t = self.tree()?;
        let mut vocabs = self.vocabs.clone();
        let coerced = coerce_rows(vec![row], &self.params.dtypes, &mut vocabs, false)?;
        let row_f64 = coerced.into_iter().next().unwrap_or_default();
        let bd = t.break_down(&row_f64);
        let d = PyDict::new(py);
        d.set_item("baseline", bd.baseline)?;
        d.set_item(
            "steps",
            bd.steps
                .into_iter()
                .map(|(c, v)| (t.col_names[c].clone(), v))
                .collect::<Vec<_>>(),
        )?;
        d.set_item("residual", bd.residual)?;
        d.set_item("prediction", bd.prediction)?;
        Ok(d.unbind())
    }

    /// rpart-style text print of the tree.
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.tree()?))
    }

    /// Graphviz DOT for the tree (render with pygraphviz/graphviz — layout is
    /// graphviz's job):
    /// `pgv.AGraph(string=model.to_dot()).draw("tree.svg", prog="dot")`.
    fn to_dot(&self) -> PyResult<String> {
        Ok(self.tree()?.to_dot())
    }

    /// Split columns along each root-to-leaf path.
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

    /// Number of leaves.
    ///
    /// Args: none.
    ///
    /// Returns: leaf count.
    fn n_leaves(&self) -> PyResult<usize> {
        Ok(self.tree()?.n_leaves())
    }

    /// Return all hyperparameter values as a dict.
    fn config(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.get_params(py, true)
    }

    /// Return all hyperparameter values as a dict (scikit-learn compatible).
    #[pyo3(signature = (deep = true))]
    fn get_params(&self, py: Python<'_>, deep: bool) -> PyResult<Py<PyDict>> {
        let _ = deep;
        let d = PyDict::new(py);
        d.set_item("max_depth", self.params.max_depth)?;
        d.set_item("min_rows", self.params.min_rows)?;
        d.set_item("minsplit", self.params.minsplit)?;
        d.set_item("minbucket", self.params.minbucket)?;
        d.set_item("minprob", self.params.minprob)?;
        d.set_item("max_leaf_nodes", self.params.max_leaf_nodes)?;
        d.set_item("min_gain", self.params.min_gain)?;
        d.set_item("purity_gate", self.params.purity_gate)?;
        d.set_item(
            "purity_gate_metric",
            match self.params.purity_gate_metric {
                PurityMetric::Sd => "sd",
                PurityMetric::Cv => "cv",
            },
        )?;
        d.set_item("tail_quantile", self.params.tail_quantile)?;
        d.set_item("split_precision", self.params.split_precision)?;
        d.set_item("split_cols", self.params.split_cols.clone())?;
        d.set_item("col_sample", self.params.col_sample)?;
        d.set_item("seed", self.params.seed)?;
        d.set_item(
            "leaf",
            match &self.params.leaf {
                LeafKind::Linear => "linear",
                LeafKind::Mean => "mean",
                LeafKind::Median => "median",
                LeafKind::Quantile(_) => "quantile",
                LeafKind::Huber => "huber",
                LeafKind::Mars => "mars",
                LeafKind::Knn => "knn",
                LeafKind::Prob => "prob",
                LeafKind::Odds => "odds",
                LeafKind::LogOdds => "log_odds",
            },
        )?;
        if let LeafKind::Quantile(tau) = self.params.leaf {
            d.set_item("tau", tau)?;
        }
        d.set_item(
            "leaf_family",
            match self.params.leaf_family {
                LeafFamily::Gaussian => "gaussian",
                LeafFamily::Binomial => "binomial",
                LeafFamily::Poisson => "poisson",
                LeafFamily::Probit => "probit",
                LeafFamily::Gamma => "gamma",
                LeafFamily::Tweedie => "tweedie",
                LeafFamily::Tobit => "tobit",
                LeafFamily::Softmax => "softmax",
                LeafFamily::Survival => "survival",
                LeafFamily::Heteroscedastic => "heteroscedastic",
                LeafFamily::Aft => "aft",
                LeafFamily::Cure => "cure",
                LeafFamily::Spline => "spline",
            },
        )?;
        d.set_item("l1", self.params.l1)?;
        d.set_item("l2", self.params.l2)?;
        d.set_item("leaf_solver", leaf_solver_name(self.params.leaf_solver))?;
        d.set_item(
            "leaf_cols",
            match self.params.leaf_cols {
                LeafCols::All => "all",
                LeafCols::PathUsed => "path",
            },
        )?;
        d.set_item("leaf_digest", self.params.leaf_digest)?;
        d.set_item("tree_monotone", self.params.tree_monotone.clone())?;
        d.set_item("leaf_monotone", self.params.leaf_monotone.clone())?;
        d.set_item("monotone", self.params.monotone.clone())?;
        d.set_item("col_weights", self.params.col_weights.clone())?;
        d.set_item("interaction_groups", self.params.interaction_groups.clone())?;
        d.set_item("alpha", self.params.alpha)?;
        d.set_item("imbalance_penalty", self.params.imbalance_penalty)?;
        d.set_item(
            "prune",
            match self.params.prune {
                PruneCost::None => "none",
                PruneCost::Bic => "bic",
                PruneCost::Mdl => "mdl",
                PruneCost::Cp => "cp",
            },
        )?;
        d.set_item("cp_alpha", self.params.cp_alpha)?;
        d.set_item("honesty", self.params.honesty)?;
        d.set_item("honesty_fraction", self.params.honesty_fraction)?;
        d.set_item("maxsurrogate", self.params.maxsurrogate)?;
        d.set_item(
            "split_test",
            match self.params.split_test {
                SplitTest::None => "none",
                SplitTest::Chow => "chow",
                SplitTest::Mob => "mob",
                SplitTest::Ctree => "ctree",
            },
        )?;
        d.set_item("split_alpha", self.params.split_alpha)?;
        d.set_item("bonferroni", self.params.bonferroni)?;
        d.set_item("min_relevance", self.params.min_relevance)?;
        d.set_item("verbose", self.params.verbose)?;
        d.set_item(
            "split_eval",
            match self.params.split_eval {
                SplitEval::Fast => "fast",
                SplitEval::Histogram => "histogram",
                SplitEval::Exhaustive => "exhaustive",
            },
        )?;
        d.set_item("presort", self.params.presort)?;
        d.set_item("events", self.params.events.clone())?;
        d.set_item("n_classes", self.params.n_classes)?;
        d.set_item("tweedie_p", self.params.tweedie_p)?;
        d.set_item("tobit_threshold", self.params.tobit_threshold)?;
        d.set_item("mars_degree", self.params.mars_degree)?;
        d.set_item("mars_terms", self.params.mars_terms)?;
        d.set_item("mars_penalty", self.params.mars_penalty)?;
        d.set_item("knn_k", self.params.knn_k)?;
        d.set_item("knn_k_validity", self.params.knn_k_validity)?;
        d.set_item("knn_variant", self.params.knn_variant.as_str())?;
        d.set_item("knn_metric", self.params.knn_metric.as_str())?;
        d.set_item("knn_gamma", self.params.knn_gamma)?;
        d.set_item("knn_p", self.params.knn_p)?;
        d.set_item("shrinkage", self.params.shrinkage)?;
        d.set_item("max_classes", self.params.max_classes)?;
        Ok(d.into())
    }

    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        if name == "params" {
            let p = self.get_params(py, true)?;
            return Ok(p.into_any());
        }
        let params = self.get_params(py, true)?;
        let dict = params.bind(py);
        if let Some(val) = dict.get_item(name)? {
            return Ok(val.into_any().unbind());
        }
        Err(PyAttributeError::new_err(format!(
            "'ModelTree' object has no attribute '{name}'"
        )))
    }
}

/// Winnowing: screen columns by univariate relevance against the target and
/// keep the relevant ones (the cheap pre-split screening LightGBM's docs call
/// feature pre-filtering / winnowing).
///
/// Args:
///     x: rows of predictors; `None` entries are missing.
///     y: targets (rows with `None` targets or zero weight are dropped).
///     w: optional row weights.
///     min_relevance: keep columns with relevance >= this (default 0.05).
///
/// Returns: `{"kept": [column indices], "relevance": [score per column]}`.
#[pyfunction]
#[pyo3(signature = (x, y, w = None, min_relevance = 0.05))]
fn winnow(
    py: Python<'_>,
    x: Vec<Vec<Option<f64>>>,
    y: Vec<Option<f64>>,
    w: Option<Vec<f64>>,
    min_relevance: f64,
) -> PyResult<Py<PyDict>> {
    let wv: Vec<f64> = w.unwrap_or_else(|| vec![1.0; y.len()]);
    let keep: Vec<usize> = (0..y.len())
        .filter(|&i| i < x.len() && y[i].is_some() && wv.get(i).copied().unwrap_or(0.0) > 0.0)
        .collect();
    let rows: Vec<Vec<Option<f64>>> = keep.iter().map(|&i| x[i].clone()).collect();
    let yy: Vec<f64> = keep.iter().map(|&i| y[i].unwrap()).collect();
    let ww: Vec<f64> = keep.iter().map(|&i| wv[i]).collect();
    let (kept, rel) = mltreelib_core::preselect::winnow(&rows, &yy, &ww, min_relevance);
    let d = PyDict::new(py);
    d.set_item("kept", kept)?;
    d.set_item("relevance", rel)?;
    Ok(d.unbind())
}

#[pyclass(name = "GritBot")]
pub struct PyGritBot {
    inner: mltreelib_core::gritbot::GritBot,
}

#[pymethods]
impl PyGritBot {
    #[new]
    #[pyo3(signature = (min_items=None, max_norm=None, sig_threshold=None, max_depth=None, check_cols=None))]
    fn new(
        min_items: Option<usize>,
        max_norm: Option<f64>,
        sig_threshold: Option<f64>,
        max_depth: Option<usize>,
        check_cols: Option<Vec<usize>>,
    ) -> Self {
        let mut bot = mltreelib_core::gritbot::GritBot::new();
        if let Some(m) = min_items {
            bot.min_items = Some(m);
        }
        if let Some(z) = max_norm {
            bot.max_norm = z;
        }
        if let Some(alpha) = sig_threshold {
            bot.sig_threshold = alpha;
        }
        if let Some(d) = max_depth {
            bot.max_depth = d;
        }
        if let Some(cols) = check_cols {
            bot.check_cols = Some(cols);
        }
        PyGritBot { inner: bot }
    }

    #[pyo3(signature = (x, col_names=None, dtypes=None))]
    fn scan(
        &self,
        py: Python<'_>,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
        col_names: Option<Vec<String>>,
        dtypes: Option<Vec<String>>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let n_cols = x.first().map(|r| r.len()).unwrap_or(0);
        let names = match col_names {
            Some(n) => n,
            None => (0..n_cols).map(|i| format!("col_{}", i)).collect(),
        };
        let resolved_dtypes: Vec<ColType> = match dtypes {
            Some(dt_strings) => dt_strings
                .into_iter()
                .map(|s| match s.to_lowercase().as_str() {
                    "cat" | "categorical" => ColType::Categorical,
                    "int" | "integer" => ColType::Integer,
                    _ => ColType::Float,
                })
                .collect(),
            None => (0..n_cols)
                .map(|j| {
                    let is_str = x.iter().any(|r| {
                        r.get(j)
                            .and_then(|c| c.as_ref())
                            .map(|v| v.extract::<String>().is_ok())
                            .unwrap_or(false)
                    });
                    if is_str {
                        ColType::Categorical
                    } else {
                        ColType::Float
                    }
                })
                .collect(),
        };

        let mut vocabs = Vec::new();
        let coerced = coerce_rows(x, &resolved_dtypes, &mut vocabs, true)?;
        let data: Vec<Vec<f64>> = coerced
            .into_iter()
            .map(|row| row.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect())
            .collect();

        let anomalies = self.inner.scan(&data, &names, &resolved_dtypes);
        let mut out = Vec::with_capacity(anomalies.len());
        for a in anomalies {
            let d = PyDict::new(py);
            d.set_item("case_id", a.case_id)?;
            d.set_item("attribute", a.attribute)?;
            d.set_item("attribute_name", a.attribute_name)?;
            d.set_item("actual_value", a.actual_value)?;
            d.set_item("expected_value", a.expected_value)?;
            d.set_item("std_dev", a.std_dev)?;
            d.set_item("z_score", a.z_score)?;
            d.set_item("significance", a.significance)?;
            d.set_item("subgroup_size", a.subgroup_size)?;
            d.set_item("normal_limit", a.normal_limit)?;
            d.set_item("normal_fraction", a.normal_fraction)?;
            d.set_item("global_z", a.global_z)?;
            d.set_item("context_ratio", a.context_ratio)?;
            d.set_item("density_score", a.density_score)?;
            d.set_item("is_pure_contextual", a.is_pure_contextual)?;
            d.set_item("summary", a.summary)?;

            let cond_list = pyo3::types::PyList::empty(py);
            for c in a.conditions {
                let cd = PyDict::new(py);
                cd.set_item("col", c.col)?;
                cd.set_item("display", c.display_str())?;
                cd.set_item("col_name", c.col_name)?;
                cd.set_item("actual_val", c.actual_val)?;
                cond_list.append(cd)?;
            }
            d.set_item("conditions", cond_list)?;
            out.push(d.unbind());
        }
        Ok(out)
    }
}

#[pyclass(name = "DensityEstimationTree")]
pub struct PyDensityEstimationTree {
    inner: mltreelib_core::anomaly::det::DensityEstimationTree,
    vocabs: Vec<Option<std::collections::HashMap<String, f64>>>,
}

#[pymethods]
impl PyDensityEstimationTree {
    fn _snapshot(&self) -> PyResult<(String, Vec<Option<std::collections::HashMap<String, f64>>>)> {
        self.inner.to_json().map(|data| (data, self.vocabs.clone())).map_err(model_persistence_error)
    }
    #[staticmethod]
    fn _from_snapshot(data: &str, vocabs: Vec<Option<std::collections::HashMap<String, f64>>>) -> PyResult<Self> {
        let inner = mltreelib_core::anomaly::DensityEstimationTree::from_json(data).map_err(model_persistence_error)?;
        check_unsupervised_vocabs(&vocabs, &inner.dtypes)?;
        Ok(Self { inner, vocabs })
    }
    #[new]
    #[pyo3(signature = (max_depth=8, min_samples_split=10, min_leaf_samples=5, min_density_gain=0.0, alpha=0.0))]
    fn new(
        max_depth: usize,
        min_samples_split: usize,
        min_leaf_samples: usize,
        min_density_gain: f64,
        alpha: f64,
    ) -> Self {
        let mut det = mltreelib_core::anomaly::det::DensityEstimationTree::new();
        det.max_depth = max_depth;
        det.min_samples_split = min_samples_split;
        det.min_leaf_samples = min_leaf_samples;
        det.min_density_gain = min_density_gain;
        det.alpha = alpha;
        PyDensityEstimationTree {
            inner: det,
            vocabs: Vec::new(),
        }
    }

    #[pyo3(signature = (x, col_names=None, dtypes=None))]
    fn fit(
        &mut self,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
        col_names: Option<Vec<String>>,
        dtypes: Option<Vec<String>>,
    ) -> PyResult<()> {
        let n_cols = x.first().map(|r| r.len()).unwrap_or(0);
        let names = match col_names {
            Some(n) => n,
            None => (0..n_cols).map(|i| format!("col_{}", i)).collect(),
        };
        let resolved_dtypes: Vec<ColType> = match dtypes {
            Some(dt_strings) => dt_strings
                .into_iter()
                .map(|s| match s.to_lowercase().as_str() {
                    "cat" | "categorical" => ColType::Categorical,
                    "int" | "integer" => ColType::Integer,
                    _ => ColType::Float,
                })
                .collect(),
            None => (0..n_cols)
                .map(|j| {
                    let is_str = x.iter().any(|r| {
                        r.get(j)
                            .and_then(|c| c.as_ref())
                            .map(|v| v.extract::<String>().is_ok())
                            .unwrap_or(false)
                    });
                    if is_str { ColType::Categorical } else { ColType::Float }
                })
                .collect(),
        };

        let mut vocabs = Vec::new();
        let coerced = coerce_rows(x, &resolved_dtypes, &mut vocabs, true)?;
        self.vocabs = vocabs;
        let data: Vec<Vec<f64>> = coerced
            .into_iter()
            .map(|row| row.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect())
            .collect();

        self.inner.fit(&data, &names, &resolved_dtypes);
        Ok(())
    }

    fn predict_density(&self, x: Vec<Vec<Option<Bound<'_, PyAny>>>>) -> PyResult<Vec<f64>> {
        let mut vocabs = self.vocabs.clone();
        let coerced = coerce_rows(x, &self.inner.dtypes, &mut vocabs, false)?;
        let data: Vec<Vec<f64>> = coerced
            .into_iter()
            .map(|row| row.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect())
            .collect();
        Ok(data.iter().map(|r| self.inner.predict_density(r)).collect())
    }

    fn predict_anomaly_score(&self, x: Vec<Vec<Option<Bound<'_, PyAny>>>>) -> PyResult<Vec<f64>> {
        let mut vocabs = self.vocabs.clone();
        let coerced = coerce_rows(x, &self.inner.dtypes, &mut vocabs, false)?;
        let data: Vec<Vec<f64>> = coerced
            .into_iter()
            .map(|row| row.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect())
            .collect();
        Ok(data.iter().map(|r| self.inner.predict_anomaly_score(r)).collect())
    }
}

#[pyclass(name = "ExtendedIsolationTree")]
pub struct PyExtendedIsolationTree {
    inner: mltreelib_core::anomaly::eif_tree::ExtendedIsolationTree,
    vocabs: Vec<Option<std::collections::HashMap<String, f64>>>,
}

fn check_unsupervised_vocabs(
    vocabs: &[Option<std::collections::HashMap<String, f64>>],
    dtypes: &[ColType],
) -> PyResult<()> {
    if vocabs.len() != dtypes.len()
        || vocabs.iter().zip(dtypes).any(|(vocab, dtype)| {
            vocab.as_ref().is_some_and(|map| {
                *dtype != ColType::Categorical
                    || map.values().any(|code| {
                        !code.is_finite() || *code < 0.0 || *code >= map.len() as f64
                            || code.fract() != 0.0
                    })
            })
        })
    {
        return Err(PyValueError::new_err("invalid model category vocabulary"));
    }
    Ok(())
}

#[pymethods]
impl PyExtendedIsolationTree {
    #[new]
    #[pyo3(signature = (max_depth=None, extension_level=None, subsample_size=None, seed=42))]
    fn new(
        max_depth: Option<usize>,
        extension_level: Option<usize>,
        subsample_size: Option<usize>,
        seed: u64,
    ) -> Self {
        let mut eif = mltreelib_core::anomaly::eif_tree::ExtendedIsolationTree::new(seed);
        eif.max_depth = max_depth;
        eif.extension_level = extension_level;
        eif.subsample_size = subsample_size;
        PyExtendedIsolationTree {
            inner: eif,
            vocabs: Vec::new(),
        }
    }

    fn _snapshot(&self) -> PyResult<(String, Vec<Option<std::collections::HashMap<String, f64>>>)> {
        let data = self.inner.to_json().map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok((data, self.vocabs.clone()))
    }

    #[staticmethod]
    fn _from_snapshot(data: &str, vocabs: Vec<Option<std::collections::HashMap<String, f64>>>) -> PyResult<Self> {
        let inner = mltreelib_core::anomaly::ExtendedIsolationTree::from_json(data)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        check_unsupervised_vocabs(&vocabs, &inner.dtypes)?;
        Ok(Self { inner, vocabs })
    }

    #[pyo3(signature = (x, col_names=None, dtypes=None))]
    fn fit(
        &mut self,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
        col_names: Option<Vec<String>>,
        dtypes: Option<Vec<String>>,
    ) -> PyResult<()> {
        let n_cols = x.first().map(|r| r.len()).unwrap_or(0);
        let names = match col_names {
            Some(n) => n,
            None => (0..n_cols).map(|i| format!("col_{}", i)).collect(),
        };
        let resolved_dtypes: Vec<ColType> = match dtypes {
            Some(dt_strings) => dt_strings
                .into_iter()
                .map(|s| match s.to_lowercase().as_str() {
                    "cat" | "categorical" => ColType::Categorical,
                    "int" | "integer" => ColType::Integer,
                    _ => ColType::Float,
                })
                .collect(),
            None => (0..n_cols)
                .map(|j| {
                    let is_str = x.iter().any(|r| {
                        r.get(j)
                            .and_then(|c| c.as_ref())
                            .map(|v| v.extract::<String>().is_ok())
                            .unwrap_or(false)
                    });
                    if is_str { ColType::Categorical } else { ColType::Float }
                })
                .collect(),
        };

        let mut vocabs = Vec::new();
        let coerced = coerce_rows(x, &resolved_dtypes, &mut vocabs, true)?;
        self.vocabs = vocabs;
        let data: Vec<Vec<f64>> = coerced
            .into_iter()
            .map(|row| row.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect())
            .collect();

        self.inner.fit(&data, &names, &resolved_dtypes);
        Ok(())
    }

    fn path_length(&self, x: Vec<Vec<Option<Bound<'_, PyAny>>>>) -> PyResult<Vec<f64>> {
        if x.first().is_some_and(|row| row.len() != self.inner.dtypes.len()) {
            return Err(PyValueError::new_err("x width must match fitted isolation tree"));
        }
        let mut vocabs = self.vocabs.clone();
        let coerced = coerce_rows(x, &self.inner.dtypes, &mut vocabs, false)?;
        let data: Vec<Vec<f64>> = coerced
            .into_iter()
            .map(|row| row.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect())
            .collect();
        Ok(data.iter().map(|r| self.inner.path_length(r)).collect())
    }

    fn anomaly_score(&self, x: Vec<Vec<Option<Bound<'_, PyAny>>>>) -> PyResult<Vec<f64>> {
        if x.first().is_some_and(|row| row.len() != self.inner.dtypes.len()) {
            return Err(PyValueError::new_err("x width must match fitted isolation tree"));
        }
        let mut vocabs = self.vocabs.clone();
        let coerced = coerce_rows(x, &self.inner.dtypes, &mut vocabs, false)?;
        let data: Vec<Vec<f64>> = coerced
            .into_iter()
            .map(|row| row.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect())
            .collect();
        Ok(data.iter().map(|r| self.inner.anomaly_score(r)).collect())
    }
}

fn numeric_row(row: Vec<Option<f64>>) -> Vec<f64> {
    row.into_iter()
        .map(|cell| cell.filter(|value| value.is_finite()).unwrap_or(f64::NAN))
        .collect()
}

fn numeric_rows(data: Vec<Vec<Option<f64>>>) -> PyResult<Vec<Vec<f64>>> {
    if let Some(first) = data.first() {
        if first.is_empty() || data.iter().any(|row| row.len() != first.len()) {
            return Err(PyValueError::new_err(
                "numeric input must have nonempty rows of equal width",
            ));
        }
    }
    Ok(data.into_iter().map(numeric_row).collect())
}

fn check_numeric_width(data: &[Vec<Option<f64>>], width: usize) -> PyResult<()> {
    if data.iter().any(|row| row.len() != width) {
        return Err(PyValueError::new_err("x width must match fitted numeric tree"));
    }
    Ok(())
}

#[pyclass(name = "HalfSpaceTree")]
pub struct PyHalfSpaceTree {
    inner: mltreelib_core::anomaly::hs_tree::HalfSpaceTree,
}

#[pymethods]
impl PyHalfSpaceTree {
    #[getter]
    fn _snapshot_rows(&self) -> usize { self.inner.n_samples }
    fn _snapshot(&self) -> PyResult<(String, Vec<Option<std::collections::HashMap<String, f64>>>)> {
        self.inner.to_json().map(|data| (data, Vec::new())).map_err(model_persistence_error)
    }
    #[staticmethod]
    fn _from_snapshot(data: &str, _vocabs: Vec<Option<std::collections::HashMap<String, f64>>>) -> PyResult<Self> {
        Ok(Self { inner: mltreelib_core::anomaly::HalfSpaceTree::from_json(data).map_err(model_persistence_error)? })
    }
    #[new]
    #[pyo3(signature = (max_depth=6, window_size=250, initial_mass=0.0, seed=42))]
    fn new(max_depth: usize, window_size: usize, initial_mass: f64, seed: u64) -> Self {
        let mut hst = mltreelib_core::anomaly::hs_tree::HalfSpaceTree::new(max_depth, seed);
        hst.window_size = window_size;
        hst.initial_mass = initial_mass;
        PyHalfSpaceTree {
            inner: hst,
        }
    }

    fn fit(&mut self, data: Vec<Vec<Option<f64>>>) -> PyResult<()> {
        self.inner.fit(&numeric_rows(data)?);
        Ok(())
    }

    fn anomaly_score(&self, data: Vec<Vec<Option<f64>>>) -> PyResult<Vec<f64>> {
        check_numeric_width(&data, self.inner.bounds.len())?;
        Ok(numeric_rows(data)?
            .iter()
            .map(|r| self.inner.anomaly_score(r))
            .collect())
    }

    fn mass_score(&self, data: Vec<Vec<Option<f64>>>) -> PyResult<Vec<f64>> {
        check_numeric_width(&data, self.inner.bounds.len())?;
        Ok(numeric_rows(data)?
            .iter()
            .map(|r| self.inner.mass_score(r))
            .collect())
    }

    fn insert(&mut self, point: Vec<Option<f64>>) -> PyResult<()> {
        self.inner.insert(&numeric_row(point));
        Ok(())
    }
}

#[pyclass(name = "RandomCutTree")]
pub struct PyRandomCutTree {
    inner: mltreelib_core::anomaly::rc_tree::RandomCutTree,
}

#[pymethods]
impl PyRandomCutTree {
    #[getter]
    fn _snapshot_rows(&self) -> usize { self.inner.n_points }
    fn _snapshot(&self) -> PyResult<(String, Vec<Option<std::collections::HashMap<String, f64>>>)> {
        self.inner.to_json().map(|data| (data, Vec::new())).map_err(model_persistence_error)
    }
    #[staticmethod]
    fn _from_snapshot(data: &str, _vocabs: Vec<Option<std::collections::HashMap<String, f64>>>) -> PyResult<Self> {
        Ok(Self { inner: mltreelib_core::anomaly::RandomCutTree::from_json(data).map_err(model_persistence_error)? })
    }
    #[new]
    #[pyo3(signature = (sample_size=256, seed=42))]
    fn new(sample_size: usize, seed: u64) -> Self {
        let mut rct = mltreelib_core::anomaly::rc_tree::RandomCutTree::new(seed);
        rct.sample_size = sample_size;
        PyRandomCutTree {
            inner: rct,
        }
    }

    fn fit(&mut self, data: Vec<Vec<Option<f64>>>) -> PyResult<()> {
        self.inner.fit(&numeric_rows(data)?);
        Ok(())
    }

    fn codisp(&self, data: Vec<Vec<Option<f64>>>) -> PyResult<Vec<f64>> {
        let width = self.inner.nodes.first().map_or(0, |node| node.bbox.len());
        check_numeric_width(&data, width)?;
        Ok(numeric_rows(data)?
            .iter()
            .map(|r| self.inner.codisp(r))
            .collect())
    }
}

#[pyclass(name = "ActiveAnomalyTree")]
pub struct PyActiveAnomalyTree {
    inner: mltreelib_core::anomaly::aad_tree::ActiveAnomalyTree,
}

#[pymethods]
impl PyActiveAnomalyTree {
    #[getter]
    fn _snapshot_rows(&self) -> usize { self.inner.n_samples }
    fn _snapshot(&self) -> PyResult<(String, Vec<Option<std::collections::HashMap<String, f64>>>)> {
        self.inner.to_json().map(|data| (data, Vec::new())).map_err(model_persistence_error)
    }
    #[staticmethod]
    fn _from_snapshot(data: &str, _vocabs: Vec<Option<std::collections::HashMap<String, f64>>>) -> PyResult<Self> {
        Ok(Self { inner: mltreelib_core::anomaly::ActiveAnomalyTree::from_json(data).map_err(model_persistence_error)? })
    }
    #[new]
    #[pyo3(signature = (max_depth=8, eta=0.1, prior_weight=1.0, seed=42))]
    fn new(max_depth: usize, eta: f64, prior_weight: f64, seed: u64) -> Self {
        let mut aad = mltreelib_core::anomaly::aad_tree::ActiveAnomalyTree::new(max_depth, seed);
        aad.eta = eta;
        aad.prior_weight = prior_weight;
        PyActiveAnomalyTree {
            inner: aad,
        }
    }

    fn fit(&mut self, data: Vec<Vec<Option<f64>>>) -> PyResult<()> {
        self.inner.fit(&numeric_rows(data)?);
        Ok(())
    }

    fn anomaly_score(&self, data: Vec<Vec<Option<f64>>>) -> PyResult<Vec<f64>> {
        Ok(numeric_rows(data)?
            .iter()
            .map(|r| self.inner.score(r))
            .collect())
    }

    #[pyo3(signature = (point, is_anomaly, eta=0.5))]
    fn update_feedback(
        &mut self,
        point: Vec<Option<f64>>,
        is_anomaly: bool,
        eta: f64,
    ) -> PyResult<()> {
        if !eta.is_finite() || eta < 0.0 {
            return Err(PyValueError::new_err("eta must be finite and nonnegative"));
        }
        self.inner
            .update_feedback(&numeric_row(point), is_anomaly, eta);
        Ok(())
    }
}

#[pyclass(name = "ContextAnomalyTree")]
pub struct PyContextAnomalyTree {
    inner: mltreelib_core::anomaly::context_tree::ContextAnomalyTree,
}

#[pymethods]
impl PyContextAnomalyTree {
    #[getter]
    fn _snapshot_rows(&self) -> usize { self.inner.nodes.first().map_or(0, |n| n.total_count) }
    fn _snapshot(&self) -> PyResult<(String, Vec<Option<std::collections::HashMap<String, f64>>>)> {
        self.inner.to_json().map(|data| (data, Vec::new())).map_err(model_persistence_error)
    }
    #[staticmethod]
    fn _from_snapshot(data: &str, _vocabs: Vec<Option<std::collections::HashMap<String, f64>>>) -> PyResult<Self> {
        Ok(Self { inner: mltreelib_core::anomaly::ContextAnomalyTree::from_json(data).map_err(model_persistence_error)? })
    }
    #[new]
    #[pyo3(signature = (max_depth=5, alphabet_size=256, smoothing_discount=0.5))]
    fn new(max_depth: usize, alphabet_size: usize, smoothing_discount: f64) -> Self {
        let mut cat = mltreelib_core::anomaly::context_tree::ContextAnomalyTree::new(max_depth, alphabet_size);
        cat.smoothing_discount = smoothing_discount;
        PyContextAnomalyTree {
            inner: cat,
        }
    }

    fn fit(&mut self, sequences: Vec<Vec<Option<u64>>>) -> PyResult<()> {
        self.inner.fit_with_gaps(&sequences);
        Ok(())
    }

    fn insert(&mut self, context: Vec<u64>, symbol: u64) -> PyResult<()> {
        self.inner.insert(&context, symbol);
        Ok(())
    }

    fn surprise_score(&self, context: Vec<u64>, symbol: u64) -> PyResult<f64> {
        Ok(self.inner.surprise_score(&context, symbol))
    }

    fn conditional_prob(&self, context: Vec<u64>, symbol: u64) -> PyResult<f64> {
        Ok(self.inner.conditional_prob(&context, symbol))
    }
}

#[pyclass(name = "TrieKNN")]
pub struct PyTrieKnn {
    inner: mltreelib_core::trie_knn::TrieKnn,
    vocabs: Vec<Option<std::collections::HashMap<String, f64>>>,
    dtypes: Vec<ColType>,
}

#[pymethods]
impl PyTrieKnn {
    #[new]
    #[pyo3(signature = (task="classification", k=5, weights="distance", metric="euclidean", leaf_min_size=None))]
    fn new(
        task: &str,
        k: usize,
        weights: &str,
        metric: &str,
        leaf_min_size: Option<usize>,
    ) -> PyResult<Self> {
        let task_enum = if task.to_lowercase().starts_with("reg") {
            mltreelib_core::trie_knn::TrieKnnTask::Regression
        } else {
            mltreelib_core::trie_knn::TrieKnnTask::Classification
        };
        let weights_enum = if weights.to_lowercase().starts_with("dist") {
            mltreelib_core::trie_knn::KnnWeights::Distance
        } else {
            mltreelib_core::trie_knn::KnnWeights::Uniform
        };
        let metric_enum = match metric.to_lowercase().as_str() {
            "euclidean" | "l2" => mltreelib_core::trie_knn::KnnMetric::Euclidean,
            "manhattan" | "l1" => mltreelib_core::trie_knn::KnnMetric::Manhattan,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown metric '{}', expected 'euclidean' or 'manhattan'",
                    other
                )))
            }
        };

        let mut inner = mltreelib_core::trie_knn::TrieKnn::new(
            task_enum,
            k,
            weights_enum,
            Vec::new(),
            Vec::new(),
        );
        inner.metric = metric_enum;
        inner.leaf_min_size = leaf_min_size;

        Ok(PyTrieKnn {
            inner,
            vocabs: Vec::new(),
            dtypes: Vec::new(),
        })
    }

    #[pyo3(signature = (x, y, col_names=None, dtypes=None))]
    fn fit(
        &mut self,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
        y: Vec<f64>,
        col_names: Option<Vec<String>>,
        dtypes: Option<Vec<String>>,
    ) -> PyResult<()> {
        let n_cols = x.first().map(|r| r.len()).unwrap_or(0);
        let _names = match col_names {
            Some(n) => n,
            None => (0..n_cols).map(|i| format!("col_{}", i)).collect(),
        };
        let resolved_dtypes: Vec<ColType> = match dtypes {
            Some(dt_strings) => dt_strings
                .into_iter()
                .map(|s| match s.to_lowercase().as_str() {
                    "cat" | "categorical" => ColType::Categorical,
                    "int" | "integer" => ColType::Integer,
                    _ => ColType::Float,
                })
                .collect(),
            None => (0..n_cols)
                .map(|j| {
                    let is_str = x.iter().any(|r| {
                        r.get(j)
                            .and_then(|c| c.as_ref())
                            .map(|v| v.extract::<String>().is_ok())
                            .unwrap_or(false)
                    });
                    if is_str { ColType::Categorical } else { ColType::Float }
                })
                .collect(),
        };

        let mut vocabs = Vec::new();
        let coerced = coerce_rows(x, &resolved_dtypes, &mut vocabs, true)?;
        self.vocabs = vocabs;
        self.dtypes = resolved_dtypes.clone();

        let data: Vec<Vec<f64>> = coerced
            .into_iter()
            .map(|row| row.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect())
            .collect();

        let cat_cols: Vec<usize> = (0..n_cols)
            .filter(|&j| resolved_dtypes[j] == ColType::Categorical)
            .collect();
        let num_cols: Vec<usize> = (0..n_cols)
            .filter(|&j| resolved_dtypes[j] != ColType::Categorical)
            .collect();

        self.inner.cat_cols = cat_cols;
        self.inner.num_cols = num_cols;
        self.inner.fit(&data, &y);
        Ok(())
    }

    fn predict(&self, x: Vec<Vec<Option<Bound<'_, PyAny>>>>) -> PyResult<Vec<f64>> {
        let mut vocabs = self.vocabs.clone();
        let coerced = coerce_rows(x, &self.dtypes, &mut vocabs, false)?;
        let data: Vec<Vec<f64>> = coerced
            .into_iter()
            .map(|row| row.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect())
            .collect();
        Ok(data.iter().map(|r| self.inner.predict(r)).collect())
    }

    fn predict_proba(&self, x: Vec<Vec<Option<Bound<'_, PyAny>>>>) -> PyResult<Vec<Vec<f64>>> {
        let mut vocabs = self.vocabs.clone();
        let coerced = coerce_rows(x, &self.dtypes, &mut vocabs, false)?;
        let data: Vec<Vec<f64>> = coerced
            .into_iter()
            .map(|row| row.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect())
            .collect();
        Ok(data.iter().map(|r| self.inner.predict_proba(r)).collect())
    }

    #[getter]
    fn classes(&self) -> Vec<i64> {
        self.inner.classes.clone()
    }
}

#[pyclass(name = "PredictiveClusteringTree")]
pub struct PyPredictiveClusteringTree {
    inner: mltreelib_core::cluster::PredictiveClusteringTree,
    vocabs: Vec<Option<std::collections::HashMap<String, f64>>>,
    dtypes: Vec<ColType>,
}

#[pymethods]
impl PyPredictiveClusteringTree {
    #[new]
    #[pyo3(signature = (max_depth=6, min_samples_split=10, min_samples_leaf=5, min_variance_reduction=1e-4, target_weights=None))]
    fn new(
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_variance_reduction: f64,
        target_weights: Option<Vec<f64>>,
    ) -> Self {
        let mut pct = mltreelib_core::cluster::PredictiveClusteringTree::new(
            max_depth,
            min_samples_split,
            min_samples_leaf,
        );
        pct.min_variance_reduction = min_variance_reduction;
        pct.target_weights = target_weights;
        PyPredictiveClusteringTree {
            inner: pct,
            vocabs: Vec::new(),
            dtypes: Vec::new(),
        }
    }

    fn _snapshot(&self) -> PyResult<(String, Vec<Option<std::collections::HashMap<String, f64>>>)> {
        let data = self.inner.to_json().map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok((data, self.vocabs.clone()))
    }

    #[staticmethod]
    fn _from_snapshot(data: &str, vocabs: Vec<Option<std::collections::HashMap<String, f64>>>) -> PyResult<Self> {
        let inner = mltreelib_core::cluster::PredictiveClusteringTree::from_json(data)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        check_unsupervised_vocabs(&vocabs, &inner.dtypes)?;
        let dtypes = inner.dtypes.clone();
        Ok(Self { inner, dtypes, vocabs })
    }

    #[getter]
    fn _snapshot_rows(&self) -> usize {
        self.inner.nodes.first().map_or(0, |node| node.size)
    }

    #[pyo3(signature = (x, y=None, col_names=None, dtypes=None, target_names=None))]
    fn fit(
        &mut self,
        x: Vec<Vec<Option<Bound<'_, PyAny>>>>,
        y: Option<Vec<Vec<f64>>>,
        col_names: Option<Vec<String>>,
        dtypes: Option<Vec<String>>,
        target_names: Option<Vec<String>>,
    ) -> PyResult<()> {
        let n_cols = x.first().map(|r| r.len()).unwrap_or(0);
        let names = match col_names {
            Some(n) => n,
            None => (0..n_cols).map(|i| format!("col_{}", i)).collect(),
        };
        let resolved_dtypes: Vec<ColType> = match dtypes {
            Some(dt_strings) => dt_strings
                .into_iter()
                .map(|s| match s.to_lowercase().as_str() {
                    "cat" | "categorical" => ColType::Categorical,
                    "int" | "integer" => ColType::Integer,
                    _ => ColType::Float,
                })
                .collect(),
            None => (0..n_cols)
                .map(|j| {
                    let is_str = x.iter().any(|r| {
                        r.get(j)
                            .and_then(|c| c.as_ref())
                            .map(|v| v.extract::<String>().is_ok())
                            .unwrap_or(false)
                    });
                    if is_str { ColType::Categorical } else { ColType::Float }
                })
                .collect(),
        };

        let mut vocabs = Vec::new();
        let coerced = coerce_rows(x, &resolved_dtypes, &mut vocabs, true)?;
        self.vocabs = vocabs;
        self.dtypes = resolved_dtypes.clone();

        let data: Vec<Vec<f64>> = coerced
            .into_iter()
            .map(|row| row.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect())
            .collect();

        self.inner.fit(
            &data,
            y.as_deref(),
            Some(&names),
            Some(&resolved_dtypes),
            target_names.as_deref(),
        );
        Ok(())
    }

    fn predict_cluster(&self, x: Vec<Vec<Option<Bound<'_, PyAny>>>>) -> PyResult<Vec<usize>> {
        let mut vocabs = self.vocabs.clone();
        let coerced = coerce_rows(x, &self.dtypes, &mut vocabs, false)?;
        let data: Vec<Vec<f64>> = coerced
            .into_iter()
            .map(|row| row.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect())
            .collect();
        Ok(data.iter().map(|r| self.inner.predict_cluster(r)).collect())
    }

    fn predict_prototype(&self, x: Vec<Vec<Option<Bound<'_, PyAny>>>>) -> PyResult<Vec<Vec<f64>>> {
        let mut vocabs = self.vocabs.clone();
        let coerced = coerce_rows(x, &self.dtypes, &mut vocabs, false)?;
        let data: Vec<Vec<f64>> = coerced
            .into_iter()
            .map(|row| row.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect())
            .collect();
        Ok(data.iter().map(|r| self.inner.predict_prototype(r)).collect())
    }

    fn cluster_rules(&self) -> Vec<(usize, String, Vec<f64>, usize)> {
        self.inner.cluster_rules()
    }

    #[getter]
    fn n_clusters(&self) -> usize {
        self.inner.n_clusters
    }
}

#[pyclass(name = "CLTree")]
pub struct PyClTree {
    inner: mltreelib_core::cluster::ClTree,
}

#[pymethods]
impl PyClTree {
    #[getter]
    fn _snapshot_rows(&self) -> usize { self.inner.nodes.first().map_or(0, |n| n.n_data) }
    fn _snapshot(&self) -> PyResult<(String, Vec<Option<std::collections::HashMap<String, f64>>>)> {
        self.inner.to_json().map(|data| (data, Vec::new())).map_err(model_persistence_error)
    }
    #[staticmethod]
    fn _from_snapshot(data: &str, _vocabs: Vec<Option<std::collections::HashMap<String, f64>>>) -> PyResult<Self> {
        Ok(Self { inner: mltreelib_core::cluster::ClTree::from_json(data).map_err(model_persistence_error)? })
    }
    #[new]
    #[pyo3(signature = (max_depth=6, min_samples_leaf=5, min_density=0.6))]
    fn new(max_depth: usize, min_samples_leaf: usize, min_density: f64) -> Self {
        PyClTree {
            inner: mltreelib_core::cluster::ClTree::new(max_depth, min_samples_leaf, min_density),
        }
    }

    #[pyo3(signature = (data, col_names=None))]
    fn fit(&mut self, data: Vec<Vec<Option<f64>>>, col_names: Option<Vec<String>>) -> PyResult<()> {
        self.inner.fit(&numeric_rows(data)?, col_names.as_deref());
        Ok(())
    }

    fn predict_cluster(&self, data: Vec<Vec<Option<f64>>>) -> PyResult<Vec<Option<usize>>> {
        Ok(numeric_rows(data)?
            .iter()
            .map(|r| self.inner.predict_cluster(r))
            .collect())
    }

    fn cluster_bounding_boxes(&self) -> PyResult<Vec<(usize, Vec<(f64, f64)>, usize, f64)>> {
        Ok(self.inner.cluster_bounding_boxes())
    }

    #[getter]
    fn n_clusters(&self) -> usize {
        self.inner.n_clusters
    }
}

#[pyclass(name = "HDBSCAN")]
pub struct PyHdbscanTree {
    inner: mltreelib_core::cluster::HdbscanTree,
}

#[pymethods]
impl PyHdbscanTree {
    #[getter]
    fn _snapshot_rows(&self) -> usize { self.inner.train_data.len() }
    fn _snapshot(&self) -> PyResult<(String, Vec<Option<std::collections::HashMap<String, f64>>>)> {
        self.inner.to_json().map(|data| (data, Vec::new())).map_err(model_persistence_error)
    }
    #[staticmethod]
    fn _from_snapshot(data: &str, _vocabs: Vec<Option<std::collections::HashMap<String, f64>>>) -> PyResult<Self> {
        Ok(Self { inner: mltreelib_core::cluster::HdbscanTree::from_json(data).map_err(model_persistence_error)? })
    }
    #[new]
    #[pyo3(signature = (
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        max_cluster_size=None,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=false,
    ))]
    fn new(
        min_cluster_size: usize,
        min_samples: Option<usize>,
        cluster_selection_epsilon: f64,
        max_cluster_size: Option<usize>,
        metric: &str,
        cluster_selection_method: &str,
        allow_single_cluster: bool,
    ) -> PyResult<Self> {
        let actual_min_samples = min_samples.unwrap_or(min_cluster_size);
        let mut inner =
            mltreelib_core::cluster::HdbscanTree::new(min_cluster_size, actual_min_samples);
        inner.cluster_selection_epsilon = cluster_selection_epsilon;
        inner.max_cluster_size = max_cluster_size;
        inner.allow_single_cluster = allow_single_cluster;
        inner.cluster_selection_method = match cluster_selection_method.to_lowercase().as_str() {
            "leaf" => mltreelib_core::cluster::ClusterSelectionMethod::Leaf,
            "eom" | "excess_of_mass" => mltreelib_core::cluster::ClusterSelectionMethod::Eom,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown cluster_selection_method '{}', expected 'eom' or 'leaf'",
                    other
                )))
            }
        };
        inner.metric = match metric.to_lowercase().as_str() {
            "euclidean" | "l2" => mltreelib_core::cluster::HdbscanMetric::Euclidean,
            "manhattan" | "l1" | "cityblock" => mltreelib_core::cluster::HdbscanMetric::Manhattan,
            "chebyshev" | "linf" | "infinity" => mltreelib_core::cluster::HdbscanMetric::Chebyshev,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown metric '{}', expected 'euclidean', 'manhattan', or 'chebyshev'",
                    other
                )))
            }
        };
        Ok(PyHdbscanTree { inner })
    }

    fn fit(&mut self, data: Vec<Vec<Option<f64>>>) -> PyResult<()> {
        self.inner.fit(&numeric_rows(data)?);
        Ok(())
    }

    fn predict(&self, data: Vec<Vec<Option<f64>>>) -> PyResult<Vec<Option<usize>>> {
        let width = self.inner.train_data.first().map_or(0, |row| row.len());
        check_numeric_width(&data, width)?;
        Ok(numeric_rows(data)?
            .iter()
            .map(|r| self.inner.predict(r))
            .collect())
    }

    #[getter]
    fn labels(&self) -> Vec<Option<usize>> {
        self.inner.labels.clone()
    }

    #[getter]
    fn outlier_scores(&self) -> Vec<f64> {
        self.inner.outlier_scores.clone()
    }

    #[getter]
    fn n_clusters(&self) -> usize {
        self.inner.n_clusters
    }

    fn condensed_tree<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut list = Vec::new();
        for node in &self.inner.condensed_nodes {
            let dict = PyDict::new(py);
            dict.set_item("node_id", node.node_id)?;
            dict.set_item("parent_id", node.parent_id)?;
            dict.set_item("lambda_birth", node.lambda_birth)?;
            dict.set_item("lambda_death", node.lambda_death)?;
            dict.set_item("size", node.size)?;
            dict.set_item("stability", node.stability)?;
            dict.set_item("selected", node.selected)?;
            list.push(dict);
        }
        Ok(list)
    }
}

#[pyclass(name = "UnsupervisedRandomForest")]
pub struct PyUnsupervisedProximityForest {
    inner: mltreelib_core::cluster::UnsupervisedProximityForest,
}

#[pymethods]
impl PyUnsupervisedProximityForest {
    #[getter]
    fn _snapshot_rows(&self) -> usize { self.inner.cluster_labels.len() }
    fn _snapshot(&self) -> PyResult<(String, Vec<Option<std::collections::HashMap<String, f64>>>)> {
        self.inner.to_json().map(|data| (data, Vec::new())).map_err(model_persistence_error)
    }
    #[staticmethod]
    fn _from_snapshot(data: &str, _vocabs: Vec<Option<std::collections::HashMap<String, f64>>>) -> PyResult<Self> {
        Ok(Self { inner: mltreelib_core::cluster::UnsupervisedProximityForest::from_json(data).map_err(model_persistence_error)? })
    }
    #[new]
    #[pyo3(signature = (
        n_trees=50,
        max_depth=6,
        min_samples_leaf=2,
        max_features=None,
        linkage="average",
        seed=42,
    ))]
    fn new(
        n_trees: usize,
        max_depth: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        linkage: &str,
        seed: u64,
    ) -> PyResult<Self> {
        let mut inner = mltreelib_core::cluster::UnsupervisedProximityForest::new(
            n_trees,
            max_depth,
            min_samples_leaf,
            seed,
        );
        inner.max_features = max_features;
        inner.linkage = match linkage.to_lowercase().as_str() {
            "average" | "avg" => mltreelib_core::cluster::UrfLinkage::Average,
            "single" => mltreelib_core::cluster::UrfLinkage::Single,
            "complete" | "max" => mltreelib_core::cluster::UrfLinkage::Complete,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown linkage '{}', expected 'average', 'single', or 'complete'",
                    other
                )))
            }
        };
        Ok(PyUnsupervisedProximityForest { inner })
    }

    #[pyo3(signature = (data, n_clusters=2))]
    fn fit(&mut self, data: Vec<Vec<Option<f64>>>, n_clusters: usize) -> PyResult<()> {
        self.inner.fit(&numeric_rows(data)?, n_clusters);
        Ok(())
    }

    #[getter]
    fn proximity_matrix(&self) -> Vec<Vec<f64>> {
        self.inner.proximity_matrix.clone()
    }

    fn pairwise_distances(&self) -> Vec<Vec<f64>> {
        self.inner.pairwise_distances()
    }

    fn predict(&self, data: Vec<Vec<Option<f64>>>) -> PyResult<Vec<usize>> {
        Ok(self.inner.predict(&numeric_rows(data)?))
    }

    fn distances(&self, data: Vec<Vec<Option<f64>>>) -> PyResult<Vec<Vec<f64>>> {
        Ok(self.inner.distances(&numeric_rows(data)?))
    }

    #[getter]
    fn labels(&self) -> Vec<usize> {
        self.inner.cluster_labels.clone()
    }

    #[getter]
    fn outlier_scores(&self) -> Vec<f64> {
        self.inner.outlier_scores.clone()
    }

    #[getter]
    fn n_clusters(&self) -> usize {
        self.inner.n_clusters
    }

    fn apply(&self, data: Vec<Vec<Option<f64>>>) -> PyResult<Vec<Vec<usize>>> {
        let data = numeric_rows(data)?;
        let mut res = Vec::with_capacity(data.len());
        for row in &data {
            let row_leaves: Vec<usize> = self.inner.trees.iter().map(|t| t.apply(row)).collect();
            res.push(row_leaves);
        }
        Ok(res)
    }
}

/// Multivariate Adaptive Regression Splines (MARS / Friedman 1991, `earth` compatible).
#[pyclass(name = "MarsRegressor")]
pub struct PyMarsRegressor {
    inner: mltreelib_core::mars::MarsRegressor,
}

#[pymethods]
impl PyMarsRegressor {
    #[new]
    #[pyo3(signature = (
        degree = 1,
        max_terms = 21,
        penalty = 2.0,
        thresh = 0.001,
        minspan = 0,
        endspan = 0,
        linpreds = None,
        fast_k = 10,
        pmethod = "backward",
        glm = "gaussian",
        allowed = None,
        verbose = 0
    ))]
    fn new(
        degree: usize,
        max_terms: usize,
        penalty: f64,
        thresh: f64,
        minspan: usize,
        endspan: usize,
        linpreds: Option<Vec<usize>>,
        fast_k: usize,
        pmethod: &str,
        glm: &str,
        allowed: Option<Vec<Vec<bool>>>,
        verbose: u8,
    ) -> PyResult<Self> {
        let pm = mltreelib_core::mars::MarsPruningMethod::from_str(pmethod)
            .map_err(PyValueError::new_err)?;
        let gm = mltreelib_core::mars::MarsGlmFamily::from_str(glm)
            .map_err(PyValueError::new_err)?;

        let mut reg = mltreelib_core::mars::MarsRegressor::new(degree, max_terms, penalty);
        reg.params.thresh = thresh;
        reg.params.minspan = minspan;
        reg.params.endspan = endspan;
        reg.params.linpreds = linpreds.unwrap_or_default();
        reg.params.fast_k = fast_k;
        reg.params.pmethod = pm;
        reg.params.glm = gm;
        reg.params.allowed = allowed;
        reg.params.verbose = verbose;

        Ok(PyMarsRegressor { inner: reg })
    }

    #[pyo3(signature = (x, y, sample_weights = None))]
    fn fit(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weights: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.inner
            .fit_weighted(&x, &y, sample_weights.as_deref())
            .map_err(PyValueError::new_err)
    }

    fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        self.inner.predict(&x).map_err(PyValueError::new_err)
    }

    fn predict_proba(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        self.inner.predict_proba(&x).map_err(PyValueError::new_err)
    }

    #[getter]
    fn n_terms(&self) -> PyResult<usize> {
        let m = self
            .inner
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MarsRegressor is not fitted"))?;
        Ok(m.n_terms())
    }

    #[getter]
    fn rss(&self) -> PyResult<f64> {
        let m = self
            .inner
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MarsRegressor is not fitted"))?;
        Ok(m.rss)
    }

    #[getter]
    fn gcv(&self) -> PyResult<f64> {
        let m = self
            .inner
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MarsRegressor is not fitted"))?;
        Ok(m.gcv)
    }

    #[pyo3(signature = (col_names = None))]
    fn equation(&self, col_names: Option<Vec<String>>) -> PyResult<String> {
        let m = self
            .inner
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MarsRegressor is not fitted"))?;
        Ok(m.equation(col_names.as_deref()))
    }
}

/// Screen all feature pairs for bivariate interactions using EBM FAST quad-stumps (Lou et al. 2013).
#[pyfunction]
#[pyo3(signature = (x, y, w = None, col_names = None, top_k = 10, min_samples = 5))]
fn fast_interaction_screen(
    py: Python<'_>,
    x: Vec<Vec<Option<f64>>>,
    y: Vec<f64>,
    w: Option<Vec<f64>>,
    col_names: Option<Vec<String>>,
    top_k: usize,
    min_samples: usize,
) -> PyResult<Vec<Py<PyAny>>> {
    let n = y.len();
    if x.len() != n {
        return Err(PyValueError::new_err("x and y length mismatch"));
    }
    let p = x.first().map(|r| r.len()).unwrap_or(0);
    let names = col_names.unwrap_or_else(|| (0..p).map(|j| format!("x{j}")).collect());
    let weights = w.unwrap_or_else(|| vec![1.0; n]);

    let mut cols: Vec<Vec<Option<f64>>> = vec![Vec::with_capacity(n); p];
    for row in x {
        for (j, val) in row.into_iter().enumerate() {
            if j < p {
                cols[j].push(val);
            }
        }
    }

    let results = mltreelib_core::fast_interaction_screen(&cols, &y, &weights, &names, top_k, min_samples);
    let mut out = Vec::new();
    for inter in results {
        let d = PyDict::new(py);
        d.set_item("col1", inter.col1)?;
        d.set_item("col2", inter.col2)?;
        d.set_item("col1_name", inter.col1_name)?;
        d.set_item("col2_name", inter.col2_name)?;
        d.set_item("score", inter.score)?;
        d.set_item("cut1", inter.cut1)?;
        d.set_item("cut2", inter.cut2)?;
        out.push(d.into_any().unbind());
    }
    Ok(out)
}

/// Complete competing risks evaluation report (cause C-indices, Uno IPCW C-indices, cause-specific IBS and time-dependent Brier scores).
#[pyfunction]
#[pyo3(signature = (durations, events, cif_curves = None, risk_scores = None, times = None))]
fn competing_risks_report<'py>(
    py: Python<'py>,
    durations: Vec<f64>,
    events: Vec<f64>,
    cif_curves: Option<Bound<'_, PyAny>>,
    risk_scores: Option<Bound<'_, PyAny>>,
    times: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let r = run_competing_risks_report(
        &durations,
        &events,
        cif_curves.as_ref(),
        risk_scores.as_ref(),
        times.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    competing_risks_to_dict(py, &r)
}

/// D-Calibration test (Haider et al. 2020) for survival distribution uniformity.
#[pyfunction]
#[pyo3(signature = (durations, events, surv_at_durations, n_bins = 10))]
fn d_calibration<'py>(
    py: Python<'py>,
    durations: Vec<f64>,
    events: Vec<f64>,
    surv_at_durations: Vec<f64>,
    n_bins: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = mltreelib_core::metrics::d_calibration(&durations, &events, &surv_at_durations, n_bins)
        .map_err(PyValueError::new_err)?;
    d_calibration_to_dict(py, &r)
}

/// Integrated Calibration Index (Austin et al. 2020) at fixed horizon.
#[pyfunction]
#[pyo3(signature = (durations, events, predicted_survival, horizon))]
fn integrated_calibration_index<'py>(
    py: Python<'py>,
    durations: Vec<f64>,
    events: Vec<f64>,
    predicted_survival: Vec<f64>,
    horizon: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let r = mltreelib_core::metrics::integrated_calibration_index(&durations, &events, &predicted_survival, horizon)
        .map_err(PyValueError::new_err)?;
    ici_to_dict(py, &r)
}

/// 1-Calibration slope and intercept (Austin et al. 2020) at fixed horizon.
#[pyfunction]
#[pyo3(signature = (durations, events, predicted_survival, horizon))]
fn one_calibration<'py>(
    py: Python<'py>,
    durations: Vec<f64>,
    events: Vec<f64>,
    predicted_survival: Vec<f64>,
    horizon: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let r = mltreelib_core::metrics::one_calibration(&durations, &events, &predicted_survival, horizon)
        .map_err(PyValueError::new_err)?;
    one_calibration_to_dict(py, &r)
}

/// Competing risks calibration report across causes (Alberge et al. 2026).
#[pyfunction]
#[pyo3(signature = (durations, events, predicted_cif, horizon))]
fn competing_risks_calibration<'py>(
    py: Python<'py>,
    durations: Vec<f64>,
    events: Vec<f64>,
    predicted_cif: Bound<'_, PyAny>,
    horizon: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let cif_map = extract_cif_map(&predicted_cif)?;
    let r = mltreelib_core::metrics::competing_risks_calibration(&durations, &events, &cif_map, horizon)
        .map_err(PyValueError::new_err)?;
    cr_calibration_to_dict(py, &r)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    solvers::register(m)?;
    m.add_class::<PyMarsRegressor>()?;
    m.add_class::<PyHdbscanTree>()?;
    m.add_class::<PyUnsupervisedProximityForest>()?;
    m.add_class::<PyClTree>()?;
    m.add_class::<PyPredictiveClusteringTree>()?;
    m.add_class::<PyTrieKnn>()?;
    m.add_class::<PyDensityEstimationTree>()?;
    m.add_class::<PyExtendedIsolationTree>()?;
    m.add_class::<PyHalfSpaceTree>()?;
    m.add_class::<PyRandomCutTree>()?;
    m.add_class::<PyActiveAnomalyTree>()?;
    m.add_class::<PyContextAnomalyTree>()?;
    m.add_class::<PyGritBot>()?;
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
    m.add_class::<calibration::PyIsotonicCalibrator>()?;
    m.add_class::<PyLeafControl>()?;
    m.add_class::<PyModelTree>()?;
    m.add_class::<component::PyComponentModel>()?;
    m.add_class::<component::PyModelReport>()?;
    m.add_class::<PyRule>()?;
    m.add_class::<PyRuleSet>()?;
    m.add_class::<PyOneHotEncoder>()?;
    m.add_class::<PyQuantileEncoder>()?;
    m.add_class::<PySumEncoder>()?;
    m.add_class::<PyTargetEncoder>()?;
    m.add_class::<PyTfidfEncoder>()?;
    m.add_class::<PyVariableScaler>()?;
    m.add_class::<PyWoEEncoder>()?;
    m.add_function(wrap_pyfunction!(calendar_features, m)?)?;
    m.add_function(wrap_pyfunction!(winnow, m)?)?;
    m.add_function(wrap_pyfunction!(decomposed_transform, m)?)?;
    m.add_function(wrap_pyfunction!(generate, m)?)?;
    m.add_function(wrap_pyfunction!(grouped_scale, m)?)?;
    m.add_function(wrap_pyfunction!(grouped_transform, m)?)?;
    m.add_function(wrap_pyfunction!(lag_by_days, m)?)?;
    m.add_function(wrap_pyfunction!(summarize, m)?)?;
    m.add_function(wrap_pyfunction!(trended_transform, m)?)?;
    m.add_function(wrap_pyfunction!(windowed_transform, m)?)?;
    m.add_function(wrap_pyfunction!(fast_interaction_screen, m)?)?;
    m.add_function(wrap_pyfunction!(competing_risks_report, m)?)?;
    m.add_function(wrap_pyfunction!(d_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(integrated_calibration_index, m)?)?;
    m.add_function(wrap_pyfunction!(one_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(competing_risks_calibration, m)?)?;
    Ok(())
}
