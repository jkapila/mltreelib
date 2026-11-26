//! Non-parametric KNN leaf models for ModelTree.
//!
//! Supports 5 KNN variants:
//! 1. Uniform KNN (Cover & Hart, 1967)
//! 2. Distance-Weighted KNN (Dudani, 1976: inverse distance weighting)
//! 3. Exponential / Kernel KNN (exp(-gamma * distance))
//! 4. Dual-Weighted KNN (Gou et al., 2012: distance * compactness)
//! 5. Double-Weighted KNN: DW-KNN (Pathak, Karthik, Madan, Kapila, 2025; arXiv:2512.08956:
//!    distance consistency * neighbor reliability / validity v_i)
//!
//! Includes 10 distance metrics:
//! Euclidean, Manhattan, Chebyshev, Minkowski, Cosine, Correlation,
//! Canberra, BrayCurtis, Gower, Mahalanobis (diagonal).

use crate::family::LeafFamily;

/// The weighting variant for KNN inference and neighbor contribution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KnnVariant {
    /// Equal weight for all k neighbors.
    #[default]
    Uniform,
    /// Inverse distance weighting: 1 / (d + eps)^p.
    Distance,
    /// Exponential decay: exp(-gamma * d).
    Exponential,
    /// Dual-weighted: distance weight * neighbor compactness.
    Dual,
    /// Double-Weighted KNN: distance consistency * neighbor reliability / validity v_i.
    Double,
}

impl KnnVariant {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().replace(['-', '_'], "").as_str() {
            "uniform" => Ok(Self::Uniform),
            "distance" | "inversedistance" | "inv" | "dudani" => Ok(Self::Distance),
            "exponential" | "exp" | "kernel" => Ok(Self::Exponential),
            "dual" | "dualweighted" => Ok(Self::Dual),
            "double" | "dwknn" | "doubleweighted" => Ok(Self::Double),
            other => Err(format!(
                "unknown knn_variant '{}'; valid: uniform, distance, exponential, dual, double",
                other
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Distance => "distance",
            Self::Exponential => "exponential",
            Self::Dual => "dual",
            Self::Double => "double",
        }
    }
}

/// Distance metric used to calculate pairwise proximity between samples.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[derive(Default)]
pub enum DistanceMetric {
    /// Standard L2 norm: sqrt(sum (x_j - z_j)^2).
    #[default]
    Euclidean,
    /// L1 taxicab norm: sum |x_j - z_j|.
    Manhattan,
    /// L_infinity norm: max |x_j - z_j|.
    Chebyshev,
    /// General Minkowski norm with power p: (sum |x_j - z_j|^p)^(1/p).
    Minkowski(#[serde(with = "crate::persistence::float")] f64),
    /// Cosine distance: 1 - (x . z) / (|x| |z|).
    Cosine,
    /// Pearson correlation distance: 1 - r(x, z).
    Correlation,
    /// Canberra distance: sum |x_j - z_j| / (|x_j| + |z_j|).
    Canberra,
    /// Bray-Curtis dissimilarity: sum |x_j - z_j| / sum |x_j + z_j|.
    BrayCurtis,
    /// Range-normalized Gower metric with missing-data skipping.
    Gower,
    /// Diagonal Mahalanobis: sum (x_j - z_j)^2 / var_j.
    Mahalanobis,
}


impl DistanceMetric {
    pub fn from_str(s: &str, p: Option<f64>) -> Result<Self, String> {
        match s.to_lowercase().replace(['-', '_'], "").as_str() {
            "euclidean" | "l2" => Ok(Self::Euclidean),
            "manhattan" | "l1" | "cityblock" | "taxicab" => Ok(Self::Manhattan),
            "chebyshev" | "linf" | "maximum" => Ok(Self::Chebyshev),
            "minkowski" | "lp" => Ok(Self::Minkowski(p.unwrap_or(2.0))),
            "cosine" => Ok(Self::Cosine),
            "correlation" | "pearson" => Ok(Self::Correlation),
            "canberra" => Ok(Self::Canberra),
            "braycurtis" => Ok(Self::BrayCurtis),
            "gower" => Ok(Self::Gower),
            "mahalanobis" | "varianceweighted" => Ok(Self::Mahalanobis),
            other => Err(format!(
                "unknown distance metric '{}'; valid: euclidean, manhattan, chebyshev, minkowski, cosine, correlation, canberra, braycurtis, gower, mahalanobis",
                other
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Euclidean => "euclidean",
            Self::Manhattan => "manhattan",
            Self::Chebyshev => "chebyshev",
            Self::Minkowski(_) => "minkowski",
            Self::Cosine => "cosine",
            Self::Correlation => "correlation",
            Self::Canberra => "canberra",
            Self::BrayCurtis => "braycurtis",
            Self::Gower => "gower",
            Self::Mahalanobis => "mahalanobis",
        }
    }
}

/// Computes distance between query row x and stored row z across active features.
pub fn compute_distance(
    x: &[Option<f64>],
    z: &[Option<f64>],
    metric: DistanceMetric,
    scales: Option<&[f64]>,
) -> f64 {
    let p = x.len().min(z.len());
    if p == 0 {
        return 0.0;
    }

    match metric {
        DistanceMetric::Euclidean => {
            let mut sum_sq = 0.0;
            let mut count = 0usize;
            for j in 0..p {
                if let (Some(xj), Some(zj)) = (x[j], z[j]) {
                    let diff = xj - zj;
                    sum_sq += diff * diff;
                    count += 1;
                }
            }
            if count == 0 {
                0.0
            } else {
                (sum_sq * (p as f64 / count as f64)).sqrt()
            }
        }
        DistanceMetric::Manhattan => {
            let mut sum_abs = 0.0;
            let mut count = 0usize;
            for j in 0..p {
                if let (Some(xj), Some(zj)) = (x[j], z[j]) {
                    sum_abs += (xj - zj).abs();
                    count += 1;
                }
            }
            if count == 0 {
                0.0
            } else {
                sum_abs * (p as f64 / count as f64)
            }
        }
        DistanceMetric::Chebyshev => {
            let mut max_abs = 0.0;
            for j in 0..p {
                if let (Some(xj), Some(zj)) = (x[j], z[j]) {
                    let d = (xj - zj).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
            }
            max_abs
        }
        DistanceMetric::Minkowski(power) => {
            let power = if power <= 0.0 { 2.0 } else { power };
            let mut sum_p = 0.0;
            let mut count = 0usize;
            for j in 0..p {
                if let (Some(xj), Some(zj)) = (x[j], z[j]) {
                    sum_p += (xj - zj).abs().powf(power);
                    count += 1;
                }
            }
            if count == 0 {
                0.0
            } else {
                (sum_p * (p as f64 / count as f64)).powf(1.0 / power)
            }
        }
        DistanceMetric::Cosine => {
            let mut dot = 0.0;
            let mut norm_x = 0.0;
            let mut norm_z = 0.0;
            for j in 0..p {
                if let (Some(xj), Some(zj)) = (x[j], z[j]) {
                    dot += xj * zj;
                    norm_x += xj * xj;
                    norm_z += zj * zj;
                }
            }
            let denom = (norm_x * norm_z).sqrt();
            if denom <= 1e-12 {
                1.0
            } else {
                (1.0 - dot / denom).clamp(0.0, 2.0)
            }
        }
        DistanceMetric::Correlation => {
            let mut count = 0usize;
            let mut sx = 0.0;
            let mut sz = 0.0;
            for j in 0..p {
                if let (Some(xj), Some(zj)) = (x[j], z[j]) {
                    sx += xj;
                    sz += zj;
                    count += 1;
                }
            }
            if count < 2 {
                return 0.0;
            }
            let mx = sx / count as f64;
            let mz = sz / count as f64;
            let mut num = 0.0;
            let mut vx = 0.0;
            let mut vz = 0.0;
            for j in 0..p {
                if let (Some(xj), Some(zj)) = (x[j], z[j]) {
                    let dx = xj - mx;
                    let dz = zj - mz;
                    num += dx * dz;
                    vx += dx * dx;
                    vz += dz * dz;
                }
            }
            let denom = (vx * vz).sqrt();
            if denom <= 1e-12 {
                1.0
            } else {
                (1.0 - num / denom).clamp(0.0, 2.0)
            }
        }
        DistanceMetric::Canberra => {
            let mut sum_can = 0.0;
            let mut count = 0usize;
            for j in 0..p {
                if let (Some(xj), Some(zj)) = (x[j], z[j]) {
                    let num = (xj - zj).abs();
                    let den = xj.abs() + zj.abs();
                    if den > 1e-12 {
                        sum_can += num / den;
                    }
                    count += 1;
                }
            }
            if count == 0 {
                0.0
            } else {
                sum_can * (p as f64 / count as f64)
            }
        }
        DistanceMetric::BrayCurtis => {
            let mut num = 0.0;
            let mut den = 0.0;
            for j in 0..p {
                if let (Some(xj), Some(zj)) = (x[j], z[j]) {
                    num += (xj - zj).abs();
                    den += (xj + zj).abs();
                }
            }
            if den <= 1e-12 {
                0.0
            } else {
                num / den
            }
        }
        DistanceMetric::Gower => {
            let mut sum_gower = 0.0;
            let mut valid_w = 0.0;
            for j in 0..p {
                if let (Some(xj), Some(zj)) = (x[j], z[j]) {
                    let scale = scales.and_then(|sc| sc.get(j)).copied().unwrap_or(1.0).max(1e-9);
                    let diff = (xj - zj).abs() / scale;
                    sum_gower += diff.min(1.0);
                    valid_w += 1.0;
                }
            }
            if valid_w == 0.0 {
                0.0
            } else {
                sum_gower / valid_w
            }
        }
        DistanceMetric::Mahalanobis => {
            let mut sum_sq = 0.0;
            let mut count = 0usize;
            for j in 0..p {
                if let (Some(xj), Some(zj)) = (x[j], z[j]) {
                    let var = scales.and_then(|sc| sc.get(j)).copied().unwrap_or(1.0).max(1e-9);
                    let diff = xj - zj;
                    sum_sq += (diff * diff) / var;
                    count += 1;
                }
            }
            if count == 0 {
                0.0
            } else {
                (sum_sq * (p as f64 / count as f64)).sqrt()
            }
        }
    }
}

/// Leaf-resident KNN model instance containing partitioned samples and precomputed weights.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LeafKnnModel {
    /// Active predictor column indices in the original frame.
    pub chosen_cols: Vec<usize>,
    /// Sample feature vectors in this leaf: [N_L, P_chosen].
    #[serde(with = "crate::persistence::float")]
    pub samples: Vec<Vec<Option<f64>>>,
    /// Target values: continuous for regression, integer class for classification.
    #[serde(with = "crate::persistence::float")]
    pub targets: Vec<f64>,
    /// Observation weights.
    #[serde(with = "crate::persistence::float")]
    pub weights: Vec<f64>,
    /// Precomputed neighbor reliability / validity scores v_i for DW-KNN (in [0, 1]).
    #[serde(with = "crate::persistence::float")]
    pub validity_scores: Vec<f64>,
    /// Precomputed compactness scores for Dual-Weighted KNN.
    #[serde(with = "crate::persistence::float")]
    pub compactness_scores: Vec<f64>,
    /// Per-feature scales (range for Gower, variance for Mahalanobis).
    #[serde(with = "crate::persistence::float")]
    pub feature_scales: Vec<f64>,
    /// Number of nearest neighbors to query at test time.
    pub k: usize,
    /// Neighborhood size for precomputing validity / compactness during fit.
    pub k_validity: usize,
    /// Weighting variant.
    pub variant: KnnVariant,
    /// Proximity metric.
    pub metric: DistanceMetric,
    /// Exponential decay parameter gamma. None => auto-scale by inverse median distance.
    #[serde(with = "crate::persistence::float")]
    pub gamma: Option<f64>,
    /// Power parameter p for inverse distance / Minkowski.
    #[serde(with = "crate::persistence::float")]
    pub p: f64,
    /// Fallback mean / mode when leaf sample count is tiny.
    #[serde(with = "crate::persistence::float")]
    pub fallback_mean: f64,
    /// Fallback empirical class probabilities.
    #[serde(with = "crate::persistence::float")]
    pub fallback_probs: Vec<f64>,
    /// Number of distinct classes for classification tasks.
    pub n_classes: usize,
}

impl LeafKnnModel {
    /// Fits a leaf KNN model on the specified rows and chosen features.
    pub fn fit(
        cols: &[Vec<Option<f64>>],
        y: &[f64],
        w: &[f64],
        rows: &[usize],
        chosen_cols: Vec<usize>,
        k: usize,
        k_validity: usize,
        variant: KnnVariant,
        metric: DistanceMetric,
        gamma: Option<f64>,
        p: f64,
        family: LeafFamily,
        n_classes: usize,
    ) -> Self {
        let n_samples = rows.len();
        let n_feats = chosen_cols.len();

        let mut samples = Vec::with_capacity(n_samples);
        let mut targets = Vec::with_capacity(n_samples);
        let mut weights = Vec::with_capacity(n_samples);

        for &r in rows {
            let mut row = Vec::with_capacity(n_feats);
            for &c in &chosen_cols {
                row.push(cols[c][r]);
            }
            samples.push(row);
            targets.push(y[r]);
            weights.push(w[r]);
        }

        // Compute feature scales (range for Gower, variance for Mahalanobis)
        let mut feature_scales = vec![1.0; n_feats];
        for (j, &c) in chosen_cols.iter().enumerate() {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            let mut sum_val = 0.0;
            let mut count = 0usize;
            for &r in rows {
                if let Some(v) = cols[c][r] {
                    if v < min_val { min_val = v; }
                    if v > max_val { max_val = v; }
                    sum_val += v;
                    count += 1;
                }
            }
            if count > 1 && min_val.is_finite() && max_val.is_finite() {
                let range = (max_val - min_val).max(1e-6);
                if matches!(metric, DistanceMetric::Mahalanobis) {
                    let mean = sum_val / count as f64;
                    let mut sum_sq = 0.0;
                    for &r in rows {
                        if let Some(v) = cols[c][r] {
                            let diff = v - mean;
                            sum_sq += diff * diff;
                        }
                    }
                    let var = (sum_sq / (count - 1) as f64).max(1e-6);
                    feature_scales[j] = var;
                } else {
                    feature_scales[j] = range;
                }
            }
        }

        // Compute fallback statistics
        let mut sw = 0.0;
        let mut swy = 0.0;
        let mut class_counts = vec![0.0; n_classes.max(2)];
        for i in 0..n_samples {
            let wt = weights[i];
            if wt > 0.0 {
                sw += wt;
                swy += wt * targets[i];
                let c = targets[i] as usize;
                if c < class_counts.len() {
                    class_counts[c] += wt;
                }
            }
        }
        let fallback_mean = if sw > 0.0 { swy / sw } else { 0.0 };
        let mut fallback_probs = vec![0.0; class_counts.len()];
        if sw > 0.0 {
            for (c, cnt) in class_counts.iter().enumerate() {
                fallback_probs[c] = cnt / sw;
            }
        } else {
            let n = fallback_probs.len() as f64;
            fallback_probs.fill(1.0 / n);
        }

        // Precompute validity scores (DW-KNN) and compactness scores (Dual-Weighted KNN)
        let mut validity_scores = vec![1.0; n_samples];
        let mut compactness_scores = vec![1.0; n_samples];

        if (variant == KnnVariant::Double || variant == KnnVariant::Dual) && n_samples > 2 {
            let kv = k_validity.max(1).min(n_samples - 1);

            for i in 0..n_samples {
                let xi = &samples[i];
                let yi = targets[i];

                // Distances to all other samples in this leaf
                let mut dists: Vec<(usize, f64)> = Vec::with_capacity(n_samples - 1);
                for j in 0..n_samples {
                    if i == j { continue; }
                    let d = compute_distance(xi, &samples[j], metric, Some(&feature_scales));
                    dists.push((j, d));
                }
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let top_kv = &dists[..kv];

                // 1. Neighbor Compactness: mean distance to top_kv neighbors
                let mean_dist = top_kv.iter().map(|&(_, d)| d).sum::<f64>() / kv as f64;
                compactness_scores[i] = 1.0 / (mean_dist + 1e-4);

                // 2. Neighbor Reliability / Validity (DW-KNN)
                if matches!(family, LeafFamily::Binomial | LeafFamily::Probit | LeafFamily::Softmax) {
                    // Classification: fraction of neighbors sharing the same class label
                    let same_class_count = top_kv.iter().filter(|&&(j, _)| targets[j] == yi).count();
                    let v = (same_class_count as f64) / (kv as f64);
                    // Floor reliability at 0.05 to avoid complete vanishing
                    validity_scores[i] = v.max(0.05);
                } else {
                    // Regression: consistency with local mean target
                    let local_mean = top_kv.iter().map(|&(j, _)| targets[j]).sum::<f64>() / kv as f64;
                    let local_res = (yi - local_mean).abs();
                    let mut local_var = 0.0;
                    for &(j, _) in top_kv {
                        local_var += (targets[j] - local_mean).powi(2);
                    }
                    let local_std = (local_var / kv as f64).sqrt().max(1e-4);
                    let v = (-local_res / local_std).exp();
                    validity_scores[i] = v.clamp(0.05, 1.0);
                }
            }
        }

        Self {
            chosen_cols,
            samples,
            targets,
            weights,
            validity_scores,
            compactness_scores,
            feature_scales,
            k: k.max(1),
            k_validity: k_validity.max(1),
            variant,
            metric,
            gamma,
            p: if p <= 0.0 { 2.0 } else { p },
            fallback_mean,
            fallback_probs,
            n_classes: n_classes.max(2),
        }
    }

    /// Predicts continuous response for query row across chosen features.
    pub fn predict(&self, query: &[Option<f64>], family: LeafFamily) -> f64 {
        let n_samples = self.samples.len();
        if n_samples == 0 {
            return self.fallback_mean;
        }

        // Extract query features matching chosen columns
        let q_feats: Vec<Option<f64>> = self.chosen_cols.iter().map(|&c| query.get(c).copied().flatten()).collect();

        // Compute distances to all leaf training samples
        let mut dists: Vec<(usize, f64)> = (0..n_samples)
            .map(|i| (i, compute_distance(&q_feats, &self.samples[i], self.metric, Some(&self.feature_scales))))
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let eff_k = self.k.min(n_samples);
        let top_k = &dists[..eff_k];

        // Resolve gamma for exponential decay
        let gamma = self.gamma.unwrap_or_else(|| {
            let med_d = top_k[eff_k / 2].1;
            1.0 / (med_d + 1e-4)
        });

        // Compute weights W_j per chosen variant
        let mut sum_weights = 0.0;
        let mut weighted_target = 0.0;

        for &(idx, d) in top_k {
            let obs_w = self.weights[idx].max(1e-6);
            let w_base = match self.variant {
                KnnVariant::Uniform => 1.0,
                KnnVariant::Distance => 1.0 / (d + 1e-4).powf(self.p),
                KnnVariant::Exponential => (-gamma * d).exp(),
                KnnVariant::Dual => (1.0 / (d + 1e-4).powf(self.p)) * self.compactness_scores[idx],
                KnnVariant::Double => (-gamma * d).exp() * self.validity_scores[idx],
            };
            let w_eff = w_base * obs_w;
            sum_weights += w_eff;
            weighted_target += w_eff * self.targets[idx];
        }

        if sum_weights <= 1e-12 {
            self.fallback_mean
        } else {
            let pred = weighted_target / sum_weights;
            match family {
                LeafFamily::Binomial | LeafFamily::Probit => pred.clamp(1e-6, 1.0 - 1e-6),
                _ => pred,
            }
        }
    }

    /// Predicts class probability distribution for classification tasks.
    pub fn predict_proba(&self, query: &[Option<f64>]) -> Vec<f64> {
        let n_samples = self.samples.len();
        if n_samples == 0 {
            return self.fallback_probs.clone();
        }

        let q_feats: Vec<Option<f64>> = self.chosen_cols.iter().map(|&c| query.get(c).copied().flatten()).collect();

        let mut dists: Vec<(usize, f64)> = (0..n_samples)
            .map(|i| (i, compute_distance(&q_feats, &self.samples[i], self.metric, Some(&self.feature_scales))))
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let eff_k = self.k.min(n_samples);
        let top_k = &dists[..eff_k];

        let gamma = self.gamma.unwrap_or_else(|| {
            let med_d = top_k[eff_k / 2].1;
            1.0 / (med_d + 1e-4)
        });

        let mut class_votes = vec![0.0; self.n_classes];
        let mut sum_weights = 0.0;

        for &(idx, d) in top_k {
            let obs_w = self.weights[idx].max(1e-6);
            let w_base = match self.variant {
                KnnVariant::Uniform => 1.0,
                KnnVariant::Distance => 1.0 / (d + 1e-4).powf(self.p),
                KnnVariant::Exponential => (-gamma * d).exp(),
                KnnVariant::Dual => (1.0 / (d + 1e-4).powf(self.p)) * self.compactness_scores[idx],
                KnnVariant::Double => (-gamma * d).exp() * self.validity_scores[idx],
            };
            let w_eff = w_base * obs_w;
            sum_weights += w_eff;

            let c = self.targets[idx] as usize;
            if c < class_votes.len() {
                class_votes[c] += w_eff;
            }
        }

        if sum_weights <= 1e-12 {
            self.fallback_probs.clone()
        } else {
            for v in &mut class_votes {
                *v /= sum_weights;
            }
            class_votes
        }
    }
}
