//! Schema inference: the `n_category` rule ported from the legacy `Data`
//! adapter (levels counted absolutely, or as a fraction of rows when < 1).

use crate::frame::{Column, Dataset};

/// How many distinct levels an integral numeric column may have before it is
/// treated as a categorical.
#[derive(Clone, Copy, Debug)]
pub enum NCategoryRule {
    /// Absolute number of levels.
    Absolute(usize),
    /// Fraction of rows, e.g. `0.05` means 5% of the row count.
    Fraction(f64),
}

impl NCategoryRule {
    pub fn threshold(&self, n_rows: usize) -> usize {
        match self {
            NCategoryRule::Absolute(n) => *n,
            NCategoryRule::Fraction(f) => (f.max(0.0) * n_rows as f64).ceil() as usize,
        }
    }

    /// Legacy `Data(n_category=...)` semantics: int = levels, float < 1 = row fraction.
    pub fn parse(value: f64) -> Self {
        if (0.0..1.0).contains(&value) {
            NCategoryRule::Fraction(value)
        } else {
            NCategoryRule::Absolute(value as usize)
        }
    }
}

fn is_integral(values: &[Option<f64>]) -> bool {
    values.iter().flatten().all(|v| v.fract() == 0.0 && v.is_finite())
}

fn distinct(values: &[Option<f64>]) -> usize {
    let mut seen = std::collections::HashSet::new();
    for v in values.iter().flatten() {
        seen.insert(v.to_bits());
    }
    seen.len()
}

/// Convert integral numeric columns with few enough levels into categoricals.
pub fn apply_n_category(dataset: &Dataset, rule: Option<NCategoryRule>) -> Dataset {
    let columns = dataset
        .columns()
        .iter()
        .map(|c| match c {
            Column::Numeric { name, values } => {
                let convert = match rule {
                    Some(r) => {
                        is_integral(values)
                            && distinct(values) <= r.threshold(dataset.n_rows())
                            && distinct(values) > 0
                    }
                    None => false,
                };
                if convert {
                    Column::categorical(
                        name.clone(),
                        values.iter().map(|v| v.map(|x| x.to_string())).collect(),
                    )
                } else {
                    c.clone()
                }
            }
            other => other.clone(),
        })
        .collect();
    Dataset::new(columns).expect("apply_n_category preserves column shapes")
}
