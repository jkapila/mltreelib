//! Summary statistics and grouped variable transformations (legacy
//! `summary_func` / `GroupedVariableTransformation` semantics).

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SummaryKind {
    Mean,
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
            mean: nan,
            std: nan,
            min: nan,
            q1: nan,
            median: nan,
            q3: nan,
            max: nan,
        }
    }

    fn value(&self, kind: SummaryKind) -> f64 {
        match kind {
            SummaryKind::Mean => self.mean,
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
    let mean = xs.iter().sum::<f64>() / n as f64;
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
        mean,
        std: var.sqrt(),
        min: xs[0],
        q1: q(0.25),
        median: q(0.5),
        q3: q(0.75),
        max: xs[n - 1],
    }
}

/// Summary of observed values; NAs are skipped.
pub fn summarize(values: &[Option<f64>]) -> Summary {
    summarize_owned(values.iter().flatten().copied().collect())
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
        let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
        for (k, v) in keys.iter().zip(values) {
            if let (Some(k), Some(v)) = (k, v) {
                groups.entry(k.clone()).or_default().push(*v);
            }
        }
        self.stats = groups
            .into_iter()
            .map(|(k, vals)| (k, summarize_owned(vals)))
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
