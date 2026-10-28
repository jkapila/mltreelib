//! In-memory columnar dataset: the core of the data layer.
//!
//! Values are row-major with `None` as the NA slot. Categorical columns keep
//! integer codes plus a vocabulary so nothing is lost round-tripping to
//! Python/R.

use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub enum Column {
    Numeric {
        name: String,
        values: Vec<Option<f64>>,
    },
    Categorical {
        name: String,
        codes: Vec<Option<u32>>,
        vocab: Vec<String>,
    },
}

impl Column {
    pub fn numeric(name: impl Into<String>, values: Vec<Option<f64>>) -> Self {
        Column::Numeric {
            name: name.into(),
            values,
        }
    }

    /// Build a categorical column from string values; the vocabulary is
    /// first-seen order, so generation is deterministic.
    pub fn categorical(name: impl Into<String>, values: Vec<Option<String>>) -> Self {
        let mut vocab: Vec<String> = Vec::new();
        let mut lookup: HashMap<String, u32> = HashMap::new();
        let mut codes = Vec::with_capacity(values.len());
        for v in values {
            match v {
                None => codes.push(None),
                Some(s) => {
                    let next = vocab.len() as u32;
                    let code = *lookup.entry(s.clone()).or_insert_with(|| {
                        vocab.push(s);
                        next
                    });
                    codes.push(Some(code));
                }
            }
        }
        Column::Categorical {
            name: name.into(),
            codes,
            vocab,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Column::Numeric { name, .. } | Column::Categorical { name, .. } => name,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Column::Numeric { values, .. } => values.len(),
            Column::Categorical { codes, .. } => codes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn na_count(&self) -> usize {
        match self {
            Column::Numeric { values, .. } => values.iter().filter(|v| v.is_none()).count(),
            Column::Categorical { codes, .. } => codes.iter().filter(|c| c.is_none()).count(),
        }
    }

    pub fn is_na(&self, row: usize) -> bool {
        match self {
            Column::Numeric { values, .. } => values.get(row).is_none_or(|v| v.is_none()),
            Column::Categorical { codes, .. } => codes.get(row).is_none_or(|c| c.is_none()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Dataset {
    columns: Vec<Column>,
    n_rows: usize,
}

impl Dataset {
    pub fn new(columns: Vec<Column>) -> Result<Self, String> {
        if columns.is_empty() {
            return Err("dataset needs at least one column".into());
        }
        let n_rows = columns[0].len();
        let mut seen = HashSet::new();
        for c in &columns {
            if c.len() != n_rows {
                return Err(format!(
                    "column '{}' has {} rows, expected {}",
                    c.name(),
                    c.len(),
                    n_rows
                ));
            }
            if !seen.insert(c.name().to_string()) {
                return Err(format!("duplicate column name '{}'", c.name()));
            }
        }
        Ok(Self { columns, n_rows })
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    pub fn column(&self, name: &str) -> Result<&Column, String> {
        self.columns
            .iter()
            .find(|c| c.name() == name)
            .ok_or_else(|| format!("no column named '{name}'"))
    }

    pub fn numeric(&self, name: &str) -> Result<&[Option<f64>], String> {
        match self.column(name)? {
            Column::Numeric { values, .. } => Ok(values),
            Column::Categorical { .. } => Err(format!("column '{name}' is categorical")),
        }
    }

    pub fn categorical(&self, name: &str) -> Result<(&[Option<u32>], &[String]), String> {
        match self.column(name)? {
            Column::Categorical { codes, vocab, .. } => Ok((codes, vocab)),
            Column::Numeric { .. } => Err(format!("column '{name}' is numeric")),
        }
    }
}
