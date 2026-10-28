//! Adjustable benchmark-data generator with known ground truth.
//!
//! The knobs encode the research hypothesis: how much of the signal lives in
//! categorical / grouped / hierarchical structure (which flat learners lose)
//! versus smooth numeric effects and noise. The generated `GroundTruth`
//! reports the *realized* variance share of every component, so any method can
//! be scored on structure recovery, not just accuracy.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, StudentT};

use crate::frame::{Column, Dataset};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Task {
    Regression,
    Classification { n_classes: usize },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TailBehavior {
    /// Homoscedastic Gaussian noise.
    Gaussian,
    /// Unit-variance Student-t(3): rare extreme residuals (DAR's bathtub edges).
    Heavy,
    /// Noise grows with |signal|: the target tails become hardest to fit.
    Bathtub,
}

impl Task {
    pub fn parse(s: &str, n_classes: usize) -> Result<Self, String> {
        match s {
            "regression" => Ok(Task::Regression),
            "classification" => Ok(Task::Classification { n_classes }),
            other => Err(format!("unknown task '{other}'")),
        }
    }
}

impl TailBehavior {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "gaussian" => Ok(TailBehavior::Gaussian),
            "heavy" => Ok(TailBehavior::Heavy),
            "bathtub" => Ok(TailBehavior::Bathtub),
            other => Err(format!("unknown tail behavior '{other}'")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GeneratorParams {
    pub n_rows: usize,
    pub n_numeric: usize,
    /// Cardinality of each categorical column.
    pub cardinalities: Vec<usize>,
    /// Number of hierarchy levels (e.g. region -> city -> store).
    pub hierarchy_levels: usize,
    /// Branching factor per hierarchy level (last value repeats if short).
    pub branching: Vec<usize>,
    pub group_effect_strength: f64,
    pub hierarchy_effect_strength: f64,
    pub interaction_strength: f64,
    pub noise: f64,
    pub task: Task,
    pub tail_behavior: TailBehavior,
    pub missingness: f64,
    pub seed: u64,
}

impl Default for GeneratorParams {
    fn default() -> Self {
        Self {
            n_rows: 10_000,
            n_numeric: 5,
            cardinalities: vec![3, 8, 50, 200],
            hierarchy_levels: 3,
            branching: vec![4, 10, 25],
            group_effect_strength: 0.5,
            hierarchy_effect_strength: 0.3,
            interaction_strength: 0.2,
            noise: 0.1,
            task: Task::Regression,
            tail_behavior: TailBehavior::Gaussian,
            missingness: 0.0,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VarianceBreakdown {
    pub numeric: f64,
    pub categorical: f64,
    pub hierarchy: f64,
    pub interaction: f64,
    pub noise: f64,
    pub total: f64,
}

#[derive(Clone, Debug)]
pub struct GroundTruth {
    pub numeric_betas: Vec<f64>,
    /// (column name, effect per level) for each categorical column.
    pub categorical_effects: Vec<(String, Vec<f64>)>,
    /// Effect per node, per hierarchy level.
    pub hierarchy_effects: Vec<Vec<f64>>,
    /// Realized variance share of each scaled component (var component / var target).
    pub variance: VarianceBreakdown,
}

#[derive(Clone, Debug)]
pub struct GeneratedData {
    pub dataset: Dataset,
    pub target: Vec<f64>,
    pub labels: Option<Vec<u32>>,
    pub truth: GroundTruth,
}

fn validate(p: &GeneratorParams) -> Result<(), String> {
    if p.n_rows == 0 {
        return Err("n_rows must be > 0".into());
    }
    if p.cardinalities.iter().any(|&k| k < 2) {
        return Err("all cardinalities must be >= 2".into());
    }
    if p.branching.iter().any(|&b| b < 2) {
        return Err("all branching factors must be >= 2".into());
    }
    if p.hierarchy_levels > 0 && p.branching.is_empty() {
        return Err("branching must be non-empty when hierarchy_levels > 0".into());
    }
    for (name, v) in [
        ("group_effect_strength", p.group_effect_strength),
        ("hierarchy_effect_strength", p.hierarchy_effect_strength),
        ("interaction_strength", p.interaction_strength),
        ("noise", p.noise),
    ] {
        if !(v >= 0.0) {
            return Err(format!("{name} must be >= 0"));
        }
    }
    if !(0.0..1.0).contains(&p.missingness) {
        return Err("missingness must be in [0, 1)".into());
    }
    if let Task::Classification { n_classes } = p.task {
        if n_classes < 2 {
            return Err("n_classes must be >= 2".into());
        }
    }
    Ok(())
}

fn variance(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
}

fn scatter_missing<T>(values: &mut [Option<T>], rate: f64, rng: &mut ChaCha8Rng) {
    if rate <= 0.0 {
        return;
    }
    for v in values.iter_mut() {
        if rng.gen::<f64>() < rate {
            *v = None;
        }
    }
}

/// Generate a dataset, its target, and the ground-truth manifest.
/// Deterministic for a given `seed`.
pub fn generate(p: &GeneratorParams) -> Result<GeneratedData, String> {
    validate(p)?;
    let mut rng = ChaCha8Rng::seed_from_u64(p.seed);
    let normal = Normal::new(0.0, 1.0).map_err(|e| e.to_string())?;
    let student = StudentT::new(3.0).map_err(|e| e.to_string())?;

    // ---- hierarchy: node counts per level, row paths, node effects --------
    let levels = p.hierarchy_levels;
    let mut branch: Vec<usize> = p.branching.clone();
    if branch.len() < levels {
        let fill = *branch.last().unwrap_or(&2);
        branch.resize(levels, fill);
    }
    branch.truncate(levels);

    let mut node_counts = Vec::with_capacity(levels);
    let mut acc = 1usize;
    for &b in &branch {
        acc = acc.saturating_mul(b);
        node_counts.push(acc);
    }
    let hier_effects: Vec<Vec<f64>> = node_counts
        .iter()
        .map(|&n| (0..n).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    let mut hier_codes: Vec<Vec<u32>> = vec![vec![0u32; p.n_rows]; levels];
    let mut hier_comps = vec![0.0f64; p.n_rows];
    for row in 0..p.n_rows {
        let mut parent = 0usize;
        for (l, &b) in branch.iter().enumerate() {
            let child = rng.gen_range(0..b);
            let node = parent * b + child;
            hier_codes[l][row] = node as u32;
            hier_comps[row] += hier_effects[l][node];
            parent = node;
        }
    }

    // ---- categoricals: skewed level draws + per-level effects -------------
    let mut cat_values: Vec<Vec<Option<u32>>> = Vec::new();
    let mut cat_effects: Vec<(String, Vec<f64>)> = Vec::new();
    let mut cat_comps = vec![0.0f64; p.n_rows];
    for (i, &k) in p.cardinalities.iter().enumerate() {
        let effects: Vec<f64> = (0..k).map(|_| normal.sample(&mut rng)).collect();
        // zipf-like level popularity so contingency tables end up sparse
        let total: f64 = (1..=k).map(|j| 1.0 / j as f64).sum();
        let cum: Vec<f64> = (1..=k)
            .scan(0.0, |acc, j| {
                *acc += (1.0 / j as f64) / total;
                Some(*acc)
            })
            .collect();
        let mut codes = Vec::with_capacity(p.n_rows);
        for row in 0..p.n_rows {
            let u: f64 = rng.gen();
            let lvl = cum.partition_point(|&c| c < u).min(k - 1);
            codes.push(Some(lvl as u32));
            cat_comps[row] += effects[lvl];
        }
        cat_values.push(codes);
        cat_effects.push((format!("cat_{i}"), effects));
    }

    // ---- numerics: N(0,1) features, first two carry linear signal --------
    let mut num_values: Vec<Vec<Option<f64>>> = Vec::new();
    let mut num_comps = vec![0.0f64; p.n_rows];
    let mut numeric_betas = Vec::with_capacity(p.n_numeric);
    for j in 0..p.n_numeric {
        let beta = if j < 2 { 1.0 } else { 0.0 };
        numeric_betas.push(beta);
        let mut vals = Vec::with_capacity(p.n_rows);
        for row in 0..p.n_rows {
            let x: f64 = normal.sample(&mut rng);
            vals.push(Some(x));
            num_comps[row] += beta * x;
        }
        num_values.push(vals);
    }

    // ---- interaction: first numeric x first categorical level ------------
    let inter_comps = match (num_values.first(), cat_effects.first(), cat_values.first()) {
        (Some(x0), Some((_, eff0)), Some(lvl0)) => {
            let k_sqrt = (eff0.len() as f64).sqrt();
            (0..p.n_rows)
                .map(|r| match (x0[r], lvl0[r]) {
                    (Some(x), Some(l)) => x * eff0[l as usize] / k_sqrt,
                    _ => 0.0,
                })
                .collect()
        }
        _ => vec![0.0; p.n_rows],
    };

    // ---- assemble target --------------------------------------------------
    let scale_cat = 1.0 / (p.cardinalities.len().max(1) as f64).sqrt();
    let scale_hier = 1.0 / (levels.max(1) as f64).sqrt();

    let cat_scaled: Vec<f64> = cat_comps.iter().map(|c| c * scale_cat).collect();
    let hier_scaled: Vec<f64> = hier_comps.iter().map(|c| c * scale_hier).collect();

    let mut signal = vec![0.0f64; p.n_rows];
    let mut target = vec![0.0f64; p.n_rows];
    let mut noise_comps = vec![0.0f64; p.n_rows];
    for row in 0..p.n_rows {
        signal[row] = num_comps[row]
            + p.group_effect_strength * cat_scaled[row]
            + p.hierarchy_effect_strength * hier_scaled[row]
            + p.interaction_strength * inter_comps[row];
        let sigma = match p.tail_behavior {
            TailBehavior::Bathtub => 1.0 + 2.0 * signal[row].abs(),
            _ => 1.0,
        };
        let eps: f64 = match p.tail_behavior {
            TailBehavior::Gaussian | TailBehavior::Bathtub => normal.sample(&mut rng),
            TailBehavior::Heavy => {
                let t: f64 = student.sample(&mut rng);
                t / 3.0f64.sqrt()
            }
        };
        noise_comps[row] = p.noise * eps * sigma;
        target[row] = signal[row] + noise_comps[row];
    }

    // ---- labels for classification (equal-frequency cuts) -----------------
    let labels = match p.task {
        Task::Regression => None,
        Task::Classification { n_classes } => {
            let mut sorted = target.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Some(
                target
                    .iter()
                    .map(|t| {
                        let rank = sorted.partition_point(|x| x < t);
                        let cls = rank * n_classes / p.n_rows;
                        (cls as u32).min(n_classes as u32 - 1)
                    })
                    .collect(),
            )
        }
    };

    // ---- missingness obscures features, never the truth ------------------
    let mut rng2 = ChaCha8Rng::seed_from_u64(p.seed ^ 0x9E37_79B9_7F4A_7C15);
    for v in num_values.iter_mut() {
        scatter_missing(v, p.missingness, &mut rng2);
    }
    for v in cat_values.iter_mut() {
        scatter_missing(v, p.missingness, &mut rng2);
    }

    // ---- columns ---------------------------------------------------------
    let mut columns = Vec::new();
    for (i, vals) in num_values.into_iter().enumerate() {
        columns.push(Column::numeric(format!("num_{i}"), vals));
    }
    for (i, codes) in cat_values.into_iter().enumerate() {
        columns.push(Column::Categorical {
            name: format!("cat_{i}"),
            codes,
            vocab: (0..p.cardinalities[i]).map(|l| format!("c{l}")).collect(),
        });
    }
    for (l, codes) in hier_codes.into_iter().enumerate() {
        columns.push(Column::Categorical {
            name: format!("hier_{l}"),
            codes: codes.into_iter().map(Some).collect(),
            vocab: (0..node_counts[l]).map(|n| format!("N{l}_{n}")).collect(),
        });
    }
    let dataset = Dataset::new(columns)?;

    // ---- realized variance shares ----------------------------------------
    let total = variance(&target);
    let share = |xs: &[f64]| if total > 0.0 { variance(xs) / total } else { 0.0 };
    let variance_breakdown = VarianceBreakdown {
        numeric: share(&num_comps),
        categorical: share(&cat_scaled),
        hierarchy: share(&hier_scaled),
        interaction: share(&inter_comps),
        noise: share(&noise_comps),
        total,
    };

    Ok(GeneratedData {
        dataset,
        target,
        labels,
        truth: GroundTruth {
            numeric_betas,
            categorical_effects: cat_effects,
            hierarchy_effects: hier_effects,
            variance: variance_breakdown,
        },
    })
}
