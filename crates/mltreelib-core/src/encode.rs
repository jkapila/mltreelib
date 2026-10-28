//! Categorical encoders (the roadmap's "later by design" item, moved up for
//! testing and comparisons).
//!
//! House semantics throughout: fit/transform, missing in -> missing out,
//! unseen levels -> a documented fallback (never a panic). Supervised encoders
//! come in two flavours: a plain fitted-mean version (fast, **leaky** — use on
//! training data only at your own risk) and the leakage-safe ordered version
//! following CatBoost's ordered target statistics (prefix statistics of a
//! shuffled order, per row).
//!
//! Reference: Prokhorenkova et al., "CatBoost: unbiased boosting with
//! categorical features", NeurIPS 2018; CatBoost docs, "Transforming
//! categorical features to numerical features".

use std::collections::HashMap;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::nn::{EmbeddingNet, Head};

fn level_index(values: &[Option<String>]) -> (Vec<String>, HashMap<String, u32>) {
    let mut vocab: Vec<String> = Vec::new();
    let mut lookup: HashMap<String, u32> = HashMap::new();
    for v in values.iter().flatten() {
        if !lookup.contains_key(v) {
            lookup.insert(v.clone(), vocab.len() as u32);
            vocab.push(v.clone());
        }
    }
    (vocab, lookup)
}

/// One-hot expansion: one indicator column per fitted level (first-seen
/// order). Missing values and levels unseen at fit time produce an all-zero
/// row, matching the category_encoders convention.
#[derive(Clone, Debug, Default)]
pub struct OneHotEncoder {
    vocab: Option<Vec<String>>,
    lookup: HashMap<String, u32>,
}

impl OneHotEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(&mut self, values: &[Option<String>]) -> &mut Self {
        let (vocab, lookup) = level_index(values);
        self.vocab = Some(vocab);
        self.lookup = lookup;
        self
    }

    pub fn levels(&self) -> Result<&[String], String> {
        self.vocab
            .as_deref()
            .ok_or_else(|| "OneHotEncoder is not fitted".to_string())
    }

    /// Returns one (level name, indicator) column per fitted level.
    pub fn transform(&self, values: &[Option<String>]) -> Result<Vec<(String, Vec<f64>)>, String> {
        let vocab = self.levels()?;
        let mut cols: Vec<(String, Vec<f64>)> = vocab
            .iter()
            .map(|l| (l.clone(), vec![0.0; values.len()]))
            .collect();
        for (row, v) in values.iter().enumerate() {
            if let Some(idx) = v.as_ref().and_then(|s| self.lookup.get(s)) {
                cols[*idx as usize].1[row] = 1.0;
            }
        }
        Ok(cols)
    }
}

/// Hashed one-hot: levels are hashed into `n_buckets` indicator columns, so
/// unseen levels at transform time still land somewhere. Deterministic
/// (FNV-1a over the level string).
#[derive(Clone, Debug)]
pub struct HashingEncoder {
    n_buckets: usize,
}

impl HashingEncoder {
    pub fn new(n_buckets: usize) -> Result<Self, String> {
        if n_buckets == 0 {
            return Err("n_buckets must be >= 1".into());
        }
        Ok(Self { n_buckets })
    }

    fn bucket(&self, level: &str) -> usize {
        // FNV-1a 64-bit
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in level.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        (h % self.n_buckets as u64) as usize
    }

    /// Returns `n_buckets` (bucket name, indicator) columns.
    pub fn transform(&self, values: &[Option<String>]) -> Vec<(String, Vec<f64>)> {
        let mut cols: Vec<(String, Vec<f64>)> = (0..self.n_buckets)
            .map(|i| (format!("hash_{i}"), vec![0.0; values.len()]))
            .collect();
        for (row, v) in values.iter().enumerate() {
            if let Some(s) = v {
                cols[self.bucket(s)].1[row] = 1.0;
            }
        }
        cols
    }
}

fn level_sums(
    values: &[Option<String>],
    y: &[Option<f64>],
) -> Result<(f64, usize, HashMap<String, (f64, usize)>), String> {
    if values.len() != y.len() {
        return Err("values and y length mismatch".into());
    }
    let mut total = 0.0;
    let mut n = 0usize;
    let mut per: HashMap<String, (f64, usize)> = HashMap::new();
    for (v, t) in values.iter().zip(y) {
        if let (Some(v), Some(t)) = (v, t) {
            if t.is_nan() {
                continue;
            }
            total += t;
            n += 1;
            let e = per.entry(v.clone()).or_insert((0.0, 0));
            e.0 += t;
            e.1 += 1;
        }
    }
    Ok((total, n, per))
}

/// Smoothed target encoding (the plain, fitted-mean flavour).
///
/// Each level maps to `w * level_mean + (1 - w) * global_mean` with
/// `w = n_level / (n_level + smoothing)`. **Leaky on training data** (the row's
/// own target is in its level mean) — prefer [`CatBoostEncoder`]'s ordered
/// statistics for training features. Unseen levels fall back to the global
/// mean; missing values stay missing.
#[derive(Clone, Debug, Default)]
pub struct TargetEncoder {
    prior: Option<f64>,
    smoothing: f64,
    encodings: HashMap<String, f64>,
}

impl TargetEncoder {
    pub fn new(smoothing: f64) -> Result<Self, String> {
        if !(smoothing >= 0.0) {
            return Err("smoothing must be >= 0".into());
        }
        Ok(Self {
            prior: None,
            smoothing,
            encodings: HashMap::new(),
        })
    }

    pub fn fit(&mut self, values: &[Option<String>], y: &[Option<f64>]) -> Result<&mut Self, String> {
        let (total, n, per) = level_sums(values, y)?;
        if n == 0 {
            return Err("no observed targets to fit".into());
        }
        let global = total / n as f64;
        self.prior = Some(global);
        self.encodings = per
            .into_iter()
            .map(|(k, (s, c))| {
                let w = c as f64 / (c as f64 + self.smoothing);
                (k, w * (s / c as f64) + (1.0 - w) * global)
            })
            .collect();
        Ok(self)
    }

    pub fn transform(&self, values: &[Option<String>]) -> Result<Vec<Option<f64>>, String> {
        let global = self
            .prior
            .ok_or_else(|| "TargetEncoder is not fitted".to_string())?;
        Ok(values
            .iter()
            .map(|v| v.as_ref().map(|s| *self.encodings.get(s).unwrap_or(&global)))
            .collect())
    }

    /// The global mean used as the unseen-level fallback.
    pub fn prior(&self) -> Result<f64, String> {
        self.prior
            .ok_or_else(|| "TargetEncoder is not fitted".to_string())
    }
}

/// CatBoost-style **ordered target statistics** — the leakage-safe flavour.
///
/// `fit_transform` permutes the rows (`permutations` seeded passes) and gives
/// each row the running statistic of its level using only rows earlier in the
/// permutation: `(sum_y_prefix + prior * global_mean) / (n_prefix + 1)` — the
/// regression form of CatBoost's `(countInClass + prior) / (totalCount + 1)`.
/// Averaged over permutations. The same level therefore gets **different
/// values in different rows**: that is the anti-leakage mechanism, not a bug.
///
/// `transform` (new data) uses the prior-smoothed full level means fitted on
/// the training rows — unseen levels fall back to the global mean, missing
/// values stay missing.
#[derive(Clone, Debug)]
pub struct CatBoostEncoder {
    prior: f64,
    permutations: usize,
    seed: u64,
    global: Option<f64>,
    encodings: HashMap<String, f64>,
}

impl CatBoostEncoder {
    pub fn new(prior: f64, permutations: usize, seed: u64) -> Result<Self, String> {
        if !(prior > 0.0) {
            return Err("prior must be > 0".into());
        }
        if permutations == 0 {
            return Err("permutations must be >= 1".into());
        }
        Ok(Self {
            prior,
            permutations,
            seed,
            global: None,
            encodings: HashMap::new(),
        })
    }

    /// Store the prior-smoothed full means (used by `transform` on new data).
    pub fn fit(&mut self, values: &[Option<String>], y: &[Option<f64>]) -> Result<&mut Self, String> {
        let (total, n, per) = level_sums(values, y)?;
        if n == 0 {
            return Err("no observed targets to fit".into());
        }
        let global = total / n as f64;
        self.global = Some(global);
        self.encodings = per
            .into_iter()
            .map(|(k, (s, c))| (k, (s + self.prior * global) / (c as f64 + 1.0)))
            .collect();
        Ok(self)
    }

    /// Ordered statistics on the training rows themselves (leakage-safe).
    pub fn fit_transform(
        &mut self,
        values: &[Option<String>],
        y: &[Option<f64>],
    ) -> Result<Vec<Option<f64>>, String> {
        self.fit(values, y)?;
        if values.len() != y.len() {
            return Err("values and y length mismatch".into());
        }
        let global = self.global.unwrap();
        let n = values.len();
        let mut out = vec![0.0; n];
        let mut seen = vec![0usize; n];
        for p in 0..self.permutations {
            let mut rng = ChaCha8Rng::seed_from_u64(self.seed ^ (p as u64).wrapping_mul(0x9E37_79B9));
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.gen_range(0..=i);
                order.swap(i, j);
            }
            let mut sums: HashMap<String, (f64, usize)> = HashMap::new();
            for &row in &order {
                let (v, t) = (&values[row], y[row]);
                if let (Some(v), Some(t)) = (v, t) {
                    if t.is_nan() {
                        continue;
                    }
                    let e = sums.entry(v.clone()).or_insert((0.0, 0));
                    let enc = (e.0 + self.prior * global) / (e.1 as f64 + 1.0);
                    out[row] += enc;
                    seen[row] += 1;
                    e.0 += t;
                    e.1 += 1;
                }
            }
        }
        Ok((0..n)
            .map(|i| {
                if seen[i] == 0 {
                    values[i].as_ref().map(|_| global)
                } else {
                    values[i].as_ref().map(|_| out[i] / seen[i] as f64)
                }
            })
            .collect())
    }

    pub fn transform(&self, values: &[Option<String>]) -> Result<Vec<Option<f64>>, String> {
        let global = self
            .global
            .ok_or_else(|| "CatBoostEncoder is not fitted".to_string())?;
        Ok(values
            .iter()
            .map(|v| v.as_ref().map(|s| *self.encodings.get(s).unwrap_or(&global)))
            .collect())
    }
}

/// Per-level class counts shared by the classification encoders.
#[derive(Clone, Debug)]
struct ClassStats {
    n_classes: usize,
    class_totals: Vec<f64>,
    level_counts: HashMap<String, Vec<f64>>,
    level_total: HashMap<String, f64>,
    n_levels: usize,
    n_obs: f64,
}

fn class_stats(
    values: &[Option<String>],
    y: &[Option<f64>],
    n_classes: usize,
) -> Result<ClassStats, String> {
    if values.len() != y.len() {
        return Err("values and y length mismatch".into());
    }
    if n_classes < 2 {
        return Err("n_classes must be >= 2".into());
    }
    let mut class_totals = vec![0.0; n_classes];
    let mut level_counts: HashMap<String, Vec<f64>> = HashMap::new();
    let mut level_total: HashMap<String, f64> = HashMap::new();
    let mut n_obs = 0.0;
    for (v, t) in values.iter().zip(y) {
        if let (Some(v), Some(t)) = (v, t) {
            if t.is_nan() {
                continue;
            }
            if *t < 0.0 || *t >= n_classes as f64 || t.fract() != 0.0 {
                return Err(format!(
                    "y must be integer class labels in [0, {n_classes}); got {t}"
                ));
            }
            let j = *t as usize;
            class_totals[j] += 1.0;
            let c = level_counts.entry(v.clone()).or_insert_with(|| vec![0.0; n_classes]);
            c[j] += 1.0;
            *level_total.entry(v.clone()).or_insert(0.0) += 1.0;
            n_obs += 1.0;
        }
    }
    if n_obs == 0.0 {
        return Err("no observed targets to fit".into());
    }
    Ok(ClassStats {
        n_classes,
        class_totals,
        level_counts,
        level_total,
        n_levels: 0, // filled by callers that need it
        n_obs,
    })
}

/// Class-level **TF-IDF** encoding: each category becomes a vector over the
/// target classes. Task: **classification**.
///
/// For level `v` and class `j`:
/// `tf(v,j) = count(v,j) / count(v)` and `idf(j) = ln(1 + n / (1 + count(j))) + 1`,
/// so rare classes up-weight the vector; `normalize` (default) scales each
/// level vector to unit L2 norm. Unseen levels fall back to the global class
/// shares (same idf); missing values stay missing.
#[derive(Clone, Debug)]
pub struct TfidfEncoder {
    n_classes: usize,
    normalize: bool,
    stats: Option<ClassStats>,
}

impl TfidfEncoder {
    pub fn new(normalize: bool) -> Self {
        Self {
            n_classes: 0,
            normalize,
            stats: None,
        }
    }

    pub fn fit(
        &mut self,
        values: &[Option<String>],
        y: &[Option<f64>],
        n_classes: usize,
    ) -> Result<&mut Self, String> {
        self.n_classes = n_classes;
        self.stats = Some(class_stats(values, y, n_classes)?);
        Ok(self)
    }

    fn idf(&self, s: &ClassStats, j: usize) -> f64 {
        ((1.0 + s.n_obs) / (1.0 + s.class_totals[j])).ln() + 1.0
    }

    fn weight(&self, counts: &[f64], total: f64) -> Vec<f64> {
        let s = self.stats.as_ref().unwrap();
        let mut w: Vec<f64> = (0..self.n_classes)
            .map(|j| (counts[j] / total) * self.idf(s, j))
            .collect();
        if self.normalize {
            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in w.iter_mut() {
                    *x /= norm;
                }
            }
        }
        w
    }

    pub fn transform(&self, values: &[Option<String>]) -> Result<Vec<Option<Vec<f64>>>, String> {
        let s = self
            .stats
            .as_ref()
            .ok_or_else(|| "TfidfEncoder is not fitted".to_string())?;
        let global: Vec<f64> = s.class_totals.iter().map(|c| c / s.n_obs).collect();
        Ok(values
            .iter()
            .map(|v| {
                v.as_ref().map(|level| match s.level_counts.get(level) {
                    Some(c) => self.weight(c, s.level_total[level]),
                    None => {
                        let mut w: Vec<f64> = (0..self.n_classes)
                            .map(|j| global[j] * self.idf(s, j))
                            .collect();
                        if self.normalize {
                            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
                            if norm > 0.0 {
                                for x in w.iter_mut() {
                                    *x /= norm;
                                }
                            }
                        }
                        w
                    }
                })
            })
            .collect())
    }

    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
}

/// **Weight of Evidence** encoding (credit-risk classic): each category
/// becomes one-vs-rest WoE values, `ln(distr_j(v) / distr_not_j(v))`, smoothed
/// with additive `prior` to keep logs finite. Task: **classification**.
///
/// `distr_j(v) = (count(v,j) + prior) / (count(j) + prior * L)` over the
/// `L` fitted levels; `distr_not_j` uses the complement class. Positive =
/// over-representation of the class in that level. Unseen levels fall back to
/// neutral zeros; missing stays missing.
#[derive(Clone, Debug)]
pub struct WoEEncoder {
    prior: f64,
    stats: Option<ClassStats>,
}

impl WoEEncoder {
    pub fn new(prior: f64) -> Result<Self, String> {
        if !(prior > 0.0) {
            return Err("prior must be > 0".into());
        }
        Ok(Self {
            prior,
            stats: None,
        })
    }

    pub fn fit(
        &mut self,
        values: &[Option<String>],
        y: &[Option<f64>],
        n_classes: usize,
    ) -> Result<&mut Self, String> {
        let mut s = class_stats(values, y, n_classes)?;
        s.n_levels = s.level_counts.len().max(1);
        self.stats = Some(s);
        Ok(self)
    }

    fn woe(&self, counts: &[f64], total: f64) -> Vec<f64> {
        let s = self.stats.as_ref().unwrap();
        let l = s.n_levels as f64;
        (0..s.n_classes)
            .map(|j| {
                let other = total - counts[j];
                let other_total = s.n_obs - s.class_totals[j];
                let d_j = (counts[j] + self.prior) / (s.class_totals[j] + self.prior * l);
                let d_o = (other + self.prior) / (other_total + self.prior * l);
                (d_j / d_o).ln()
            })
            .collect()
    }

    pub fn transform(&self, values: &[Option<String>]) -> Result<Vec<Option<Vec<f64>>>, String> {
        let s = self
            .stats
            .as_ref()
            .ok_or_else(|| "WoEEncoder is not fitted".to_string())?;
        Ok(values
            .iter()
            .map(|v| {
                v.as_ref().map(|level| match s.level_counts.get(level) {
                    Some(c) => self.woe(c, s.level_total[level]),
                    None => vec![0.0; s.n_classes],
                })
            })
            .collect())
    }
}

/// Task head of [`EmbeddingEncoder`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EmbeddingTask {
    /// Softmax head, cross-entropy on integer class labels.
    Classification,
    /// Linear head, squared loss on continuous targets.
    Regression,
}

impl EmbeddingTask {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "classification" => Ok(EmbeddingTask::Classification),
            "regression" => Ok(EmbeddingTask::Regression),
            other => Err(format!("unknown embedding task '{other}'")),
        }
    }

    fn head(self) -> Head {
        match self {
            EmbeddingTask::Classification => Head::Classification,
            EmbeddingTask::Regression => Head::Regression,
        }
    }
}

/// NN-style **learned embeddings** per category: a small embedding table and a
/// linear head trained by SGD, one table per categorical feature — the
/// entity-embedding trick that later blurred into tabular foundation models.
/// Task: **both**.
///
/// The head's output activation follows the targets: for regression, **ReLU**
/// when all fitted targets are non-negative, **linear** otherwise; for
/// classification it is a softmax over the classes. `fit` trains only the
/// levels it sees (`epochs` passes over the rows in order, learning rate `lr`,
/// seeded init); `transform` returns each row's learned embedding. Unseen
/// levels give a zero vector; missing stays missing.
#[derive(Clone, Debug)]
pub struct EmbeddingEncoder {
    dim: usize,
    lr: f64,
    epochs: usize,
    seed: u64,
    task: EmbeddingTask,
    lookup: HashMap<String, usize>,
    net: Option<EmbeddingNet>,
}

impl EmbeddingEncoder {
    pub fn new(dim: usize, lr: f64, epochs: usize, seed: u64, task: EmbeddingTask) -> Result<Self, String> {
        if dim == 0 {
            return Err("dim must be >= 1".into());
        }
        if !(lr > 0.0) {
            return Err("lr must be > 0".into());
        }
        if epochs == 0 {
            return Err("epochs must be >= 1".into());
        }
        Ok(Self {
            dim,
            lr,
            epochs,
            seed,
            task,
            lookup: HashMap::new(),
            net: None,
        })
    }

    /// Train the embedding table on (values, y). For classification
    /// `n_classes` may be None to infer `max(y) + 1`; it is ignored for
    /// regression. Delegates the SGD training to [`crate::nn::EmbeddingNet`].
    pub fn fit(
        &mut self,
        values: &[Option<String>],
        y: &[Option<f64>],
        n_classes: Option<usize>,
    ) -> Result<&mut Self, String> {
        if values.len() != y.len() {
            return Err("values and y length mismatch".into());
        }
        let (vocab, lookup) = level_index(values);
        if vocab.is_empty() {
            return Err("no levels to fit".into());
        }
        let lookup: HashMap<String, usize> = lookup
            .into_iter()
            .map(|(k, v)| (k, v as usize))
            .collect();
        let out_dim = match self.task {
            EmbeddingTask::Classification => {
                let k = match n_classes {
                    Some(k) => k,
                    None => {
                        let mut mx = -1.0f64;
                        for t in y.iter().flatten().filter(|t| !t.is_nan()) {
                            mx = mx.max(*t);
                        }
                        (mx + 1.0) as usize
                    }
                };
                if k < 2 {
                    return Err("classification needs >= 2 classes".into());
                }
                k
            }
            EmbeddingTask::Regression => 1,
        };
        // rows without a level cannot train; mask their targets
        let y_masked: Vec<Option<f64>> = values
            .iter()
            .zip(y)
            .map(|(v, t)| if v.is_some() { *t } else { None })
            .collect();
        let ids: Vec<usize> = values
            .iter()
            .map(|v| v.as_ref().and_then(|s| lookup.get(s).copied()).unwrap_or(0))
            .collect();
        let mut net = EmbeddingNet::new(
            self.dim,
            out_dim,
            self.task.head(),
            self.lr,
            self.epochs,
            self.seed,
        )?;
        net.train(&ids, &y_masked, vocab.len())?;
        self.lookup = lookup;
        self.net = Some(net);
        Ok(self)
    }

    /// Learned embedding per row; unseen levels give a zero vector, missing
    /// gives None.
    pub fn transform(&self, values: &[Option<String>]) -> Result<Vec<Option<Vec<f64>>>, String> {
        let net = self
            .net
            .as_ref()
            .ok_or_else(|| "EmbeddingEncoder is not fitted".to_string())?;
        Ok(values
            .iter()
            .map(|v| {
                v.as_ref().map(|level| {
                    let id = self.lookup.get(level).copied().unwrap_or(usize::MAX);
                    net.embedding(id)
                })
            })
            .collect())
    }

    /// Predictions from the trained head: the class label (classification) or
    /// the activated value (regression — ReLU when all fitted targets were
    /// non-negative, linear otherwise). Unseen levels predict from the bias
    /// alone; missing gives None.
    pub fn predict(&self, values: &[Option<String>]) -> Result<Vec<Option<f64>>, String> {
        let net = self
            .net
            .as_ref()
            .ok_or_else(|| "EmbeddingEncoder is not fitted".to_string())?;
        Ok(values
            .iter()
            .map(|v| {
                v.as_ref().map(|level| {
                    let id = self.lookup.get(level).copied().unwrap_or(usize::MAX);
                    let out = net.forward(id);
                    match self.task {
                        EmbeddingTask::Classification => out
                            .iter()
                            .enumerate()
                            .fold(0usize, |best, (j, z)| if *z > out[best] { j } else { best })
                            as f64,
                        EmbeddingTask::Regression => out[0],
                    }
                })
            })
            .collect())
    }
}

/// **Count / frequency** encoding: occurrence count per category (`normalize`
/// gives the share of rows). Task: **both** (unsupervised). Unseen levels give
/// 0.0; missing stays missing.
#[derive(Clone, Debug, Default)]
pub struct CountEncoder {
    normalize: bool,
    counts: HashMap<String, f64>,
    n_obs: f64,
    fitted: bool,
}

impl CountEncoder {
    pub fn new(normalize: bool) -> Self {
        Self {
            normalize,
            ..Default::default()
        }
    }

    pub fn fit(&mut self, values: &[Option<String>]) -> &mut Self {
        self.counts.clear();
        self.n_obs = 0.0;
        for v in values.iter().flatten() {
            *self.counts.entry(v.clone()).or_insert(0.0) += 1.0;
            self.n_obs += 1.0;
        }
        self.fitted = true;
        self
    }

    pub fn transform(&self, values: &[Option<String>]) -> Result<Vec<Option<f64>>, String> {
        if !self.fitted {
            return Err("CountEncoder is not fitted".to_string());
        }
        Ok(values
            .iter()
            .map(|v| {
                v.as_ref().map(|s| {
                    let c = *self.counts.get(s).unwrap_or(&0.0);
                    if self.normalize && self.n_obs > 0.0 {
                        c / self.n_obs
                    } else {
                        c
                    }
                })
            })
            .collect())
    }
}

/// **Leave-one-out** encoding: `fit_transform` gives each row its level's mean
/// target computed **without that row** (the leakage-aware classic);
/// `transform` on new data uses the full fitted level mean. Task: **both**.
/// Unseen levels give the global mean; missing stays missing.
#[derive(Clone, Debug, Default)]
pub struct LeaveOneOutEncoder {
    global: Option<f64>,
    means: HashMap<String, f64>,
    sums: HashMap<String, f64>,
    counts: HashMap<String, f64>,
}

impl LeaveOneOutEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(&mut self, values: &[Option<String>], y: &[Option<f64>]) -> Result<&mut Self, String> {
        let (total, n, per) = level_sums(values, y)?;
        if n == 0 {
            return Err("no observed targets to fit".into());
        }
        self.global = Some(total / n as f64);
        self.means = per
            .iter()
            .map(|(k, (s, c))| (k.clone(), s / *c as f64))
            .collect();
        self.sums = per.iter().map(|(k, (s, _))| (k.clone(), *s)).collect();
        self.counts = per.iter().map(|(k, (_, c))| (k.clone(), *c as f64)).collect();
        Ok(self)
    }

    /// Row-level means excluding each row's own target (leakage-aware).
    pub fn fit_transform(
        &mut self,
        values: &[Option<String>],
        y: &[Option<f64>],
    ) -> Result<Vec<Option<f64>>, String> {
        self.fit(values, y)?;
        let global = self.global.unwrap();
        Ok(values
            .iter()
            .zip(y)
            .map(|(v, t)| {
                v.as_ref().map(|level| {
                    let c = self.counts.get(level).copied().unwrap_or(0.0);
                    if c > 1.0 {
                        (self.sums[level] - t.unwrap_or(0.0)) / (c - 1.0)
                    } else {
                        global
                    }
                })
            })
            .collect())
    }

    pub fn transform(&self, values: &[Option<String>]) -> Result<Vec<Option<f64>>, String> {
        let global = self
            .global
            .ok_or_else(|| "LeaveOneOutEncoder is not fitted".to_string())?;
        Ok(values
            .iter()
            .map(|v| v.as_ref().map(|s| *self.means.get(s).unwrap_or(&global)))
            .collect())
    }
}

/// **M-estimate** encoding: `(sum_y + m * global) / (n_level + m)` — target
/// smoothing with fixed strength `m`. Task: **both** (leaky on training data
/// like [`TargetEncoder`]; prefer [`CatBoostEncoder`]` or
/// [`LeaveOneOutEncoder`] there). Unseen -> global; missing -> None.
#[derive(Clone, Debug)]
pub struct MEstimatorEncoder {
    m: f64,
    global: Option<f64>,
    enc: HashMap<String, f64>,
}

impl MEstimatorEncoder {
    pub fn new(m: f64) -> Result<Self, String> {
        if !(m > 0.0) {
            return Err("m must be > 0".into());
        }
        Ok(Self {
            m,
            global: None,
            enc: HashMap::new(),
        })
    }

    pub fn fit(&mut self, values: &[Option<String>], y: &[Option<f64>]) -> Result<&mut Self, String> {
        let (total, n, per) = level_sums(values, y)?;
        if n == 0 {
            return Err("no observed targets to fit".into());
        }
        let global = total / n as f64;
        self.global = Some(global);
        self.enc = per
            .into_iter()
            .map(|(k, (s, c))| (k, (s + self.m * global) / (c as f64 + self.m)))
            .collect();
        Ok(self)
    }

    pub fn transform(&self, values: &[Option<String>]) -> Result<Vec<Option<f64>>, String> {
        let global = self
            .global
            .ok_or_else(|| "MEstimatorEncoder is not fitted".to_string())?;
        Ok(values
            .iter()
            .map(|v| v.as_ref().map(|s| *self.enc.get(s).unwrap_or(&global)))
            .collect())
    }
}

/// **Base-N / binary** encoding: each level becomes its base-`base` digits as
/// indicator columns (`base = 2` is the classic Binary encoder). Task:
/// **both** (unsupervised). Unseen and missing rows are all-zero.
#[derive(Clone, Debug)]
pub struct BaseNEncoder {
    base: usize,
    vocab: Vec<String>,
    lookup: HashMap<String, u32>,
    fitted: bool,
}

impl BaseNEncoder {
    pub fn new(base: usize) -> Result<Self, String> {
        if base < 2 {
            return Err("base must be >= 2".into());
        }
        Ok(Self {
            base,
            vocab: Vec::new(),
            lookup: HashMap::new(),
            fitted: false,
        })
    }

    pub fn fit(&mut self, values: &[Option<String>]) -> &mut Self {
        let (vocab, lookup) = level_index(values);
        self.vocab = vocab;
        self.lookup = lookup;
        self.fitted = true;
        self
    }

    /// Indicator columns, one per base-`base` digit (names `d0`, `d1`, ...).
    pub fn transform(&self, values: &[Option<String>]) -> Result<Vec<(String, Vec<f64>)>, String> {
        if !self.fitted {
            return Err("BaseNEncoder is not fitted".to_string());
        }
        let max_idx = self.vocab.len().saturating_sub(1);
        let mut digits = 1;
        let mut cap = self.base;
        while max_idx >= cap {
            cap *= self.base;
            digits += 1;
        }
        let mut cols: Vec<(String, Vec<f64>)> = (0..digits)
            .map(|d| (format!("d{d}"), vec![0.0; values.len()]))
            .collect();
        for (row, v) in values.iter().enumerate() {
            if let Some(idx) = v.as_ref().and_then(|s| self.lookup.get(s)) {
                let mut n = *idx as usize;
                for d in 0..digits {
                    cols[d].1[row] = (n % self.base) as f64;
                    n /= self.base;
                }
            }
        }
        Ok(cols)
    }
}

/// **Sum contrast** coding: `k-1` columns; level `i < k-1` gets +1 in column
/// `i` and the last level gets -1 across all columns. Task: **both**
/// (unsupervised). Unseen rows give a zero vector; missing stays missing.
#[derive(Clone, Debug, Default)]
pub struct SumEncoder {
    vocab: Vec<String>,
    lookup: HashMap<String, u32>,
    fitted: bool,
}

impl SumEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(&mut self, values: &[Option<String>]) -> &mut Self {
        let (vocab, lookup) = level_index(values);
        self.vocab = vocab;
        self.lookup = lookup;
        self.fitted = true;
        self
    }

    pub fn transform(&self, values: &[Option<String>]) -> Result<Vec<Option<Vec<f64>>>, String> {
        if !self.fitted {
            return Err("SumEncoder is not fitted".to_string());
        }
        let k = self.vocab.len();
        let cols = k.saturating_sub(1).max(1);
        Ok(values
            .iter()
            .map(|v| {
                v.as_ref().map(|s| match self.lookup.get(s) {
                    Some(idx) => {
                        let mut row = vec![0.0; cols];
                        let idx = *idx as usize;
                        if idx < cols {
                            row[idx] = 1.0;
                        } else {
                            for r in row.iter_mut() {
                                *r = -1.0;
                            }
                        }
                        row
                    }
                    None => vec![0.0; cols],
                })
            })
            .collect())
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

/// **Quantile** encoding: each level becomes its smoothed target quantile
/// (default median), `w * level_quantile + (1 - w) * global_quantile` with
/// `w = n_level / (n_level + smoothing)`. Task: **regression**. Unseen ->
/// global quantile; missing -> None.
#[derive(Clone, Debug)]
pub struct QuantileEncoder {
    q: f64,
    smoothing: f64,
    global: Option<f64>,
    enc: HashMap<String, f64>,
}

impl QuantileEncoder {
    pub fn new(q: f64, smoothing: f64) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&q) {
            return Err("q must be in [0, 1]".into());
        }
        if !(smoothing >= 0.0) {
            return Err("smoothing must be >= 0".into());
        }
        Ok(Self {
            q,
            smoothing,
            global: None,
            enc: HashMap::new(),
        })
    }

    pub fn fit(&mut self, values: &[Option<String>], y: &[Option<f64>]) -> Result<&mut Self, String> {
        if values.len() != y.len() {
            return Err("values and y length mismatch".into());
        }
        let mut all: Vec<f64> = Vec::new();
        let mut per: HashMap<String, Vec<f64>> = HashMap::new();
        for (v, t) in values.iter().zip(y) {
            if let (Some(v), Some(t)) = (v, t) {
                if t.is_nan() {
                    continue;
                }
                all.push(*t);
                per.entry(v.clone()).or_default().push(*t);
            }
        }
        if all.is_empty() {
            return Err("no observed targets to fit".into());
        }
        all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let g = quantile(&all, self.q);
        self.global = Some(g);
        self.enc = per
            .into_iter()
            .map(|(k, mut xs)| {
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let w = xs.len() as f64 / (xs.len() as f64 + self.smoothing);
                (k, w * quantile(&xs, self.q) + (1.0 - w) * g)
            })
            .collect();
        Ok(self)
    }

    pub fn transform(&self, values: &[Option<String>]) -> Result<Vec<Option<f64>>, String> {
        let global = self
            .global
            .ok_or_else(|| "QuantileEncoder is not fitted".to_string())?;
        Ok(values
            .iter()
            .map(|v| v.as_ref().map(|s| *self.enc.get(s).unwrap_or(&global)))
            .collect())
    }
}

/// **James-Stein** encoding: level means shrunk toward the grand mean,
/// `shrunk_i = global + B * (mean_i - global)` with
/// `B = max(0, 1 - ((L - 3) * sigma2) / sum_i (mean_i - global)^2)` and
/// `sigma2` the pooled within-level variance; with fewer than three levels
/// `B = 0` and everything collapses to the grand mean. Task: **both**.
/// Unseen -> global; missing -> None.
#[derive(Clone, Debug, Default)]
pub struct JamesSteinEncoder {
    global: Option<f64>,
    enc: HashMap<String, f64>,
}

impl JamesSteinEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(&mut self, values: &[Option<String>], y: &[Option<f64>]) -> Result<&mut Self, String> {
        if values.len() != y.len() {
            return Err("values and y length mismatch".into());
        }
        let mut per: HashMap<String, Vec<f64>> = HashMap::new();
        let mut all: Vec<f64> = Vec::new();
        for (v, t) in values.iter().zip(y) {
            if let (Some(v), Some(t)) = (v, t) {
                if t.is_nan() {
                    continue;
                }
                all.push(*t);
                per.entry(v.clone()).or_default().push(*t);
            }
        }
        if all.is_empty() {
            return Err("no observed targets to fit".into());
        }
        let n = all.len() as f64;
        let global = all.iter().sum::<f64>() / n;

        let mut within_ss = 0.0;
        let mut between = 0.0;
        let mut means: HashMap<String, (f64, usize)> = HashMap::new();
        for (k, xs) in &per {
            let c = xs.len();
            let m = xs.iter().sum::<f64>() / c as f64;
            within_ss += xs.iter().map(|x| (x - m).powi(2)).sum::<f64>();
            between += (m - global).powi(2);
            means.insert(k.clone(), (m, c));
        }
        let levels = per.len();
        let sigma2 = if n > levels as f64 {
            within_ss / (n - levels as f64)
        } else {
            0.0
        };
        let b = if levels >= 3 && between > 0.0 {
            (1.0 - ((levels as f64 - 3.0) * sigma2) / between).max(0.0)
        } else {
            0.0
        };
        self.global = Some(global);
        self.enc = means
            .into_iter()
            .map(|(k, (m, _))| (k, global + b * (m - global)))
            .collect();
        Ok(self)
    }

    pub fn transform(&self, values: &[Option<String>]) -> Result<Vec<Option<f64>>, String> {
        let global = self
            .global
            .ok_or_else(|| "JamesSteinEncoder is not fitted".to_string())?;
        Ok(values
            .iter()
            .map(|v| v.as_ref().map(|s| *self.enc.get(s).unwrap_or(&global)))
            .collect())
    }
}
