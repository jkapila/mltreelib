//! `DatasetBuilder`: ingest any data into the typed core with the legacy
//! `Data` adapter semantics — category inference (`n_category`), NA policy,
//! digit rounding, and an optional intercept column for model-based trees.

use crate::frame::{Column, Dataset};
use crate::schema::{apply_n_category, NCategoryRule};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NaPolicy {
    /// Keep NAs as-is (default); models decide how to split them.
    Allow,
    /// Drop every row containing at least one NA.
    DropRows,
    /// Impute numeric NAs with the column mean, categorical NAs with the mode.
    Impute,
}

impl NaPolicy {
    pub fn parse(s: &str) -> Result<Self, String> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "allow" => NaPolicy::Allow,
            "drop" => NaPolicy::DropRows,
            "impute" => NaPolicy::Impute,
            other => return Err(format!("unknown NA policy '{other}'")),
        })
    }
}

#[derive(Clone, Debug)]
pub struct DatasetBuilder {
    n_category: Option<NCategoryRule>,
    na_policy: NaPolicy,
    digits: Option<u32>,
    add_intercept: bool,
    columns: Vec<Column>,
}

impl Default for DatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DatasetBuilder {
    pub fn new() -> Self {
        Self {
            n_category: None,
            na_policy: NaPolicy::Allow,
            digits: None,
            add_intercept: false,
            columns: Vec::new(),
        }
    }

    /// Category inference rule for integral numeric columns.
    pub fn n_category(mut self, rule: Option<NCategoryRule>) -> Self {
        self.n_category = rule;
        self
    }

    pub fn na_policy(mut self, policy: NaPolicy) -> Self {
        self.na_policy = policy;
        self
    }

    /// Round floats to `digits` decimal places.
    pub fn digits(mut self, digits: Option<u32>) -> Self {
        self.digits = digits;
        self
    }

    pub fn add_intercept(mut self, yes: bool) -> Self {
        self.add_intercept = yes;
        self
    }

    /// Numeric column ingest; NaN is treated as missing.
    pub fn add_numeric(mut self, name: impl Into<String>, values: Vec<Option<f64>>) -> Self {
        self.columns
            .push(Column::numeric(name, crate::frame::normalize_numeric(values)));
        self
    }

    pub fn add_categorical(mut self, name: impl Into<String>, values: Vec<Option<String>>) -> Self {
        self.columns.push(Column::categorical(name, values));
        self
    }

    pub fn build(self) -> Result<Dataset, String> {
        // digit rounding
        let columns: Vec<Column> = self
            .columns
            .into_iter()
            .map(|c| match (c, self.digits) {
                (Column::Numeric { name, values }, Some(d)) => {
                    let f = 10f64.powi(d as i32);
                    Column::numeric(
                        name,
                        values
                            .into_iter()
                            .map(|v| v.map(|x| (x * f).round() / f))
                            .collect(),
                    )
                }
                (other, _) => other,
            })
            .collect();

        let mut ds = Dataset::new(columns)?;
        ds = apply_n_category(&ds, self.n_category);
        ds = match self.na_policy {
            NaPolicy::Allow => ds,
            NaPolicy::DropRows => drop_na_rows(&ds)?,
            NaPolicy::Impute => impute(&ds),
        };

        if self.add_intercept {
            let mut cols = ds.columns().to_vec();
            cols.push(Column::numeric("intercept", vec![Some(1.0); ds.n_rows()]));
            ds = Dataset::new(cols)?;
        }
        Ok(ds)
    }
}

fn drop_na_rows(ds: &Dataset) -> Result<Dataset, String> {
    let n = ds.n_rows();
    let mut keep = vec![true; n];
    for c in ds.columns() {
        for row in 0..n {
            if c.is_na(row) {
                keep[row] = false;
            }
        }
    }
    if !keep.iter().any(|&k| k) {
        return Err("NA policy 'drop' removed every row".into());
    }
    let columns = ds.columns().iter().map(|c| filter_rows(c, &keep)).collect();
    Dataset::new(columns)
}

fn filter_rows(col: &Column, keep: &[bool]) -> Column {
    match col {
        Column::Numeric { name, values } => Column::numeric(
            name.clone(),
            values
                .iter()
                .zip(keep)
                .filter(|(_, &k)| k)
                .map(|(v, _)| *v)
                .collect(),
        ),
        Column::Categorical { name, codes, vocab } => Column::Categorical {
            name: name.clone(),
            codes: codes
                .iter()
                .zip(keep)
                .filter(|(_, &k)| k)
                .map(|(c, _)| *c)
                .collect(),
            vocab: vocab.clone(),
        },
    }
}

fn impute(ds: &Dataset) -> Dataset {
    let columns = ds
        .columns()
        .iter()
        .map(|c| match c {
            Column::Numeric { name, values } => {
                let observed = values.iter().flatten().copied().collect::<Vec<f64>>();
                if observed.is_empty() {
                    return c.clone();
                }
                let mean = observed.iter().sum::<f64>() / observed.len() as f64;
                Column::numeric(name.clone(), values.iter().map(|v| v.or(Some(mean))).collect())
            }
            Column::Categorical { name, codes, vocab } => {
                let mut counts = vec![0usize; vocab.len()];
                for code in codes.iter().flatten() {
                    counts[*code as usize] += 1;
                }
                // mode: most frequent observed level; ties resolve to lowest code
                let mode = counts
                    .iter()
                    .enumerate()
                    .max_by_key(|(i, &n)| (n, std::cmp::Reverse(*i)))
                    .map(|(i, _)| i as u32);
                let mode = match mode {
                    Some(m) if counts[m as usize] > 0 => Some(m),
                    _ => return c.clone(),
                };
                Column::Categorical {
                    name: name.clone(),
                    codes: codes.iter().map(|code| code.or(mode)).collect(),
                    vocab: vocab.clone(),
                }
            }
        })
        .collect();
    Dataset::new(columns).expect("impute preserves column shapes")
}
