//! Categorical index mapping — port of the legacy `IndexMapper` semantics:
//! multi-column string labeler with stable codes and exact inverse.
//!
//! Unseen values at `transform` time map to `None` (treated as missing), so
//! scoring never panics on new levels; `get_index` reports absence explicitly.

use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
struct CatMap {
    code2idx: HashMap<String, u32>,
    idx2code: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct IndexMapper {
    maps: HashMap<String, CatMap>,
}

impl IndexMapper {
    pub fn new() -> Self {
        Self::default()
    }

    /// Learn the vocabulary of `name` (first-seen order). Re-fitting resets it.
    pub fn fit(&mut self, name: &str, values: &[Option<String>]) -> &mut Self {
        let mut m = CatMap::default();
        for v in values.iter().flatten() {
            if !m.code2idx.contains_key(v) {
                let idx = m.idx2code.len() as u32;
                m.idx2code.push(v.clone());
                m.code2idx.insert(v.clone(), idx);
            }
        }
        self.maps.insert(name.to_string(), m);
        self
    }

    pub fn transform(&self, name: &str, values: &[Option<String>]) -> Result<Vec<Option<u32>>, String> {
        let m = self
            .maps
            .get(name)
            .ok_or_else(|| format!("'{name}' is not fitted"))?;
        Ok(values
            .iter()
            .map(|v| v.as_ref().and_then(|s| m.code2idx.get(s).copied()))
            .collect())
    }

    pub fn inverse_transform(
        &self,
        name: &str,
        codes: &[Option<u32>],
    ) -> Result<Vec<Option<String>>, String> {
        let m = self
            .maps
            .get(name)
            .ok_or_else(|| format!("'{name}' is not fitted"))?;
        codes
            .iter()
            .map(|c| match c {
                None => Ok(None),
                Some(i) => m
                    .idx2code
                    .get(*i as usize)
                    .cloned()
                    .map(Some)
                    .ok_or_else(|| format!("code {i} out of range for '{name}'")),
            })
            .collect()
    }

    pub fn get_index(&self, name: &str, value: &str) -> Result<Option<u32>, String> {
        let m = self
            .maps
            .get(name)
            .ok_or_else(|| format!("'{name}' is not fitted"))?;
        Ok(m.code2idx.get(value).copied())
    }

    pub fn get_name(&self, name: &str, index: u32) -> Result<String, String> {
        let m = self
            .maps
            .get(name)
            .ok_or_else(|| format!("'{name}' is not fitted"))?;
        m.idx2code
            .get(index as usize)
            .cloned()
            .ok_or_else(|| format!("code {index} out of range for '{name}'"))
    }

    pub fn get_all_names(&self, name: &str) -> Result<Vec<String>, String> {
        let m = self
            .maps
            .get(name)
            .ok_or_else(|| format!("'{name}' is not fitted"))?;
        Ok(m.idx2code.clone())
    }

    pub fn fitted_columns(&self) -> Vec<String> {
        let mut v: Vec<String> = self.maps.keys().cloned().collect();
        v.sort();
        v
    }
}
