//! Summary statistics and grouped variable transformations (legacy
//! `summary_func` / `GroupedVariableTransformation` semantics).

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SummaryKind {
    Mean,
    Sum,
    Std,
    Count,
    Min,
    Max,
    Median,
    Q1,
    Q3,
}

impl SummaryKind {
    pub fn parse(s: &str) -> Result<Self, String> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "mean" => SummaryKind::Mean,
            "sum" => SummaryKind::Sum,
            "std" => SummaryKind::Std,
            "count" => SummaryKind::Count,
            "min" => SummaryKind::Min,
            "max" => SummaryKind::Max,
            "median" => SummaryKind::Median,
            "q1" => SummaryKind::Q1,
            "q3" => SummaryKind::Q3,
            other => return Err(format!("unknown summary kind '{other}'")),
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Summary {
    pub count: usize,
    /// Missing values (NA or NaN) seen in the input — shown by default.
    pub n_missing: usize,
    pub sum: f64,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
}

impl Summary {
    fn empty() -> Self {
        let nan = f64::NAN;
        Summary {
            count: 0,
            n_missing: 0,
            sum: 0.0,
            mean: nan,
            std: nan,
            min: nan,
            q1: nan,
            median: nan,
            q3: nan,
            max: nan,
        }
    }

    pub fn value(&self, kind: SummaryKind) -> f64 {
        match kind {
            SummaryKind::Mean => self.mean,
            SummaryKind::Sum => self.sum,
            SummaryKind::Std => self.std,
            SummaryKind::Count => self.count as f64,
            SummaryKind::Min => self.min,
            SummaryKind::Max => self.max,
            SummaryKind::Median => self.median,
            SummaryKind::Q1 => self.q1,
            SummaryKind::Q3 => self.q3,
        }
    }
}

fn summarize_owned(mut xs: Vec<f64>) -> Summary {
    if xs.is_empty() {
        return Summary::empty();
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    let sum = xs.iter().sum::<f64>();
    let mean = sum / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    // linear-interpolated quantiles over the sorted sample
    let q = |p: f64| -> f64 {
        let pos = p * (n - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            xs[lo]
        } else {
            xs[lo] + (pos - lo as f64) * (xs[hi] - xs[lo])
        }
    };
    Summary {
        count: n,
        n_missing: 0,
        sum,
        mean,
        std: var.sqrt(),
        min: xs[0],
        q1: q(0.25),
        median: q(0.5),
        q3: q(0.75),
        max: xs[n - 1],
    }
}

/// Summary of observed values with missing counts **shown by default**: NAs
/// and NaNs are excluded from the statistics but always reported in
/// `n_missing` — nothing is silently hidden.
pub fn summarize(values: &[Option<f64>]) -> Summary {
    let mut xs = Vec::with_capacity(values.len());
    let mut n_missing = 0usize;
    for v in values {
        match v {
            Some(v) if !v.is_nan() => xs.push(*v),
            _ => n_missing += 1,
        }
    }
    let mut s = summarize_owned(xs);
    s.n_missing = n_missing;
    s
}

/// Fit/transform group-level summaries: the value a row gets is the summary of
/// its group (keys are e.g. a categorical or a hierarchy level).
#[derive(Clone, Debug, Default)]
pub struct GroupedTransform {
    kind: Option<SummaryKind>,
    stats: HashMap<String, Summary>,
}

impl GroupedTransform {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(
        &mut self,
        keys: &[Option<String>],
        values: &[Option<f64>],
        kind: SummaryKind,
    ) -> Result<&mut Self, String> {
        if keys.len() != values.len() {
            return Err("keys and values length mismatch".into());
        }
        let mut groups: HashMap<String, Vec<Option<f64>>> = HashMap::new();
        for (k, v) in keys.iter().zip(values) {
            if let Some(k) = k {
                groups.entry(k.clone()).or_default().push(*v);
            }
        }
        self.stats = groups
            .into_iter()
            .map(|(k, vals)| (k, summarize(&vals)))
            .collect();
        self.kind = Some(kind);
        Ok(self)
    }

    /// Per-row group summary; rows with a missing key or an unseen group give `None`.
    pub fn transform(&self, keys: &[Option<String>]) -> Result<Vec<Option<f64>>, String> {
        let kind = self
            .kind
            .ok_or_else(|| "GroupedTransform is not fitted".to_string())?;
        Ok(keys
            .iter()
            .map(|k| {
                k.as_ref()
                    .and_then(|g| self.stats.get(g).map(|s| s.value(kind)))
            })
            .collect())
    }

    pub fn groups(&self) -> &HashMap<String, Summary> {
        &self.stats
    }
}

/// Distributional group-normalization strategies, ported from featuregen's
/// `GroupedVariableTransformation`. Two families: centered/scaled
/// `(x - center) / scale` and ratio `x / divisor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupedStrategy {
    Zscore,
    MinMax,
    MedianIqr,
    MeanIqr,
    MedianQd,
    MedianStd,
    MedianRange,
    MeanRange,
    MinQd,
    MinIqr,
    MinRatio,
    MaxRatio,
    MeanRatio,
    MedianRatio,
    IqrRatio,
    QdRatio,
    StdRatio,
    RangeRatio,
}

impl GroupedStrategy {
    /// Parse featuregen strategy names: "zscore", "min-max", "median-iqr",
    /// "mean-iqr", "median-qd", "median-std", "median-range", "mean-range",
    /// "min-qd", "min-iqr", and the ratio names "min", "max", "mean",
    /// "median", "iqr", "qd", "std", "range".
    pub fn parse(s: &str) -> Result<Self, String> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "zscore" => GroupedStrategy::Zscore,
            "min-max" => GroupedStrategy::MinMax,
            "median-iqr" => GroupedStrategy::MedianIqr,
            "mean-iqr" => GroupedStrategy::MeanIqr,
            "median-qd" => GroupedStrategy::MedianQd,
            "median-std" => GroupedStrategy::MedianStd,
            "median-range" => GroupedStrategy::MedianRange,
            "mean-range" => GroupedStrategy::MeanRange,
            "min-qd" => GroupedStrategy::MinQd,
            "min-iqr" => GroupedStrategy::MinIqr,
            "min" => GroupedStrategy::MinRatio,
            "max" => GroupedStrategy::MaxRatio,
            "mean" => GroupedStrategy::MeanRatio,
            "median" => GroupedStrategy::MedianRatio,
            "iqr" => GroupedStrategy::IqrRatio,
            "qd" => GroupedStrategy::QdRatio,
            "std" => GroupedStrategy::StdRatio,
            "range" => GroupedStrategy::RangeRatio,
            other => return Err(format!("unknown grouped strategy '{other}'")),
        })
    }

    /// (center, scale, is_ratio) for a group summary.
    fn params(self, s: &Summary) -> (f64, f64, bool) {
        let iqr = s.q3 - s.q1;
        let qd = iqr / 2.0;
        let range = s.max - s.min;
        match self {
            GroupedStrategy::Zscore => (s.mean, s.std, false),
            GroupedStrategy::MinMax => (s.min, range, false),
            GroupedStrategy::MedianIqr => (s.median, iqr, false),
            GroupedStrategy::MeanIqr => (s.mean, iqr, false),
            GroupedStrategy::MedianQd => (s.median, qd, false),
            GroupedStrategy::MedianStd => (s.median, s.std, false),
            GroupedStrategy::MedianRange => (s.median, range, false),
            GroupedStrategy::MeanRange => (s.mean, range, false),
            GroupedStrategy::MinQd => (s.min, qd, false),
            GroupedStrategy::MinIqr => (s.min, iqr, false),
            GroupedStrategy::MinRatio => (0.0, s.min, true),
            GroupedStrategy::MaxRatio => (0.0, s.max, true),
            GroupedStrategy::MeanRatio => (0.0, s.mean, true),
            GroupedStrategy::MedianRatio => (0.0, s.median, true),
            GroupedStrategy::IqrRatio => (0.0, iqr, true),
            GroupedStrategy::QdRatio => (0.0, qd, true),
            GroupedStrategy::StdRatio => (0.0, s.std, true),
            GroupedStrategy::RangeRatio => (0.0, range, true),
        }
    }
}

/// Scales one value under a strategy against a fitted summary.
pub(crate) fn scale_with(
    strategy: GroupedStrategy,
    s: &Summary,
    v: f64,
    inverse: bool,
) -> Option<f64> {
    let (center, scale, is_ratio) = strategy.params(s);
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    Some(if is_ratio {
        if inverse {
            v * scale
        } else {
            v / scale
        }
    } else if inverse {
        v * scale + center
    } else {
        (v - center) / scale
    })
}

/// Per-column scaling with the same strategies (legacy `VariableTransformation`:
/// every variable transformed independently). Unknown column -> error;
/// degenerate scale -> missing, never inf.
#[derive(Clone, Debug, Default)]
pub struct VariableScaler {
    strategy: Option<GroupedStrategy>,
    stats: HashMap<String, Summary>,
}

impl VariableScaler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(
        &mut self,
        columns: &[(&str, &[Option<f64>])],
        strategy: GroupedStrategy,
    ) -> Result<&mut Self, String> {
        self.stats = columns
            .iter()
            .map(|(name, vals)| (name.to_string(), summarize(vals)))
            .collect();
        self.strategy = Some(strategy);
        Ok(self)
    }

    pub fn transform(&self, name: &str, values: &[Option<f64>]) -> Result<Vec<Option<f64>>, String> {
        self.apply(name, values, false)
    }

    pub fn inverse_transform(
        &self,
        name: &str,
        values: &[Option<f64>],
    ) -> Result<Vec<Option<f64>>, String> {
        self.apply(name, values, true)
    }

    fn apply(
        &self,
        name: &str,
        values: &[Option<f64>],
        inverse: bool,
    ) -> Result<Vec<Option<f64>>, String> {
        let strategy = self
            .strategy
            .ok_or_else(|| "VariableScaler is not fitted".to_string())?;
        let s = self
            .stats
            .get(name)
            .ok_or_else(|| format!("column '{name}' is not fitted"))?;
        Ok(values
            .iter()
            .map(|v| match v {
                Some(v) if !v.is_nan() => scale_with(strategy, s, *v, inverse),
                _ => None,
            })
            .collect())
    }

    pub fn columns(&self) -> Vec<String> {
        let mut v: Vec<String> = self.stats.keys().cloned().collect();
        v.sort();
        v
    }
}

/// Fit/transform group-level distributional scaling with exact inverse.
/// Rows with a missing key/value or an unseen group give `None`; a degenerate
/// scale (zero) gives `None` instead of inf.
#[derive(Clone, Debug, Default)]
pub struct GroupedScaler {
    strategy: Option<GroupedStrategy>,
    stats: HashMap<String, Summary>,
}

impl GroupedScaler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(
        &mut self,
        keys: &[Option<String>],
        values: &[Option<f64>],
        strategy: GroupedStrategy,
    ) -> Result<&mut Self, String> {
        if keys.len() != values.len() {
            return Err("keys and values length mismatch".into());
        }
        let mut groups: HashMap<String, Vec<Option<f64>>> = HashMap::new();
        for (k, v) in keys.iter().zip(values) {
            if let Some(k) = k {
                groups.entry(k.clone()).or_default().push(*v);
            }
        }
        self.stats = groups
            .into_iter()
            .map(|(k, vals)| (k, summarize(&vals)))
            .collect();
        self.strategy = Some(strategy);
        Ok(self)
    }

    pub fn transform(
        &self,
        keys: &[Option<String>],
        values: &[Option<f64>],
    ) -> Result<Vec<Option<f64>>, String> {
        self.apply(keys, values, false)
    }

    pub fn inverse_transform(
        &self,
        keys: &[Option<String>],
        values: &[Option<f64>],
    ) -> Result<Vec<Option<f64>>, String> {
        self.apply(keys, values, true)
    }

    fn apply(
        &self,
        keys: &[Option<String>],
        values: &[Option<f64>],
        inverse: bool,
    ) -> Result<Vec<Option<f64>>, String> {
        let strategy = self
            .strategy
            .ok_or_else(|| "GroupedScaler is not fitted".to_string())?;
        if keys.len() != values.len() {
            return Err("keys and values length mismatch".into());
        }
        Ok(keys
            .iter()
            .zip(values)
            .map(|(k, v)| {
                let k = k.as_ref()?;
                let v = v.as_ref()?;
                if v.is_nan() {
                    return None;
                }
                let s = self.stats.get(k)?;
                scale_with(strategy, s, *v, inverse)
            })
            .collect())
    }

    pub fn groups(&self) -> &HashMap<String, Summary> {
        &self.stats
    }
}
