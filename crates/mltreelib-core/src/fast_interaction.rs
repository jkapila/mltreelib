//! EBM FAST Interaction Screening (Lou, Caruana, Gehrke 2013).
//!
//! Fast screening of all P*(P-1)/2 feature pairs for bivariate interactions
//! on target residuals using 2D quad-stump partitions.

/// Result of an interaction test between two features.
#[derive(Clone, Debug, PartialEq)]
pub struct FastInteraction {
    pub col1: usize,
    pub col2: usize,
    pub col1_name: String,
    pub col2_name: String,
    pub score: f64,
    pub cut1: f64,
    pub cut2: f64,
}

/// Find best 1D stump cutpoint for a single numeric column on y (minimizing SSE).
fn find_1d_stump_cut(col: &[Option<f64>], y: &[f64], w: &[f64]) -> Option<(f64, f64)> {
    let mut pairs: Vec<(f64, f64, f64)> = col
        .iter()
        .zip(y.iter())
        .zip(w.iter())
        .filter_map(|((&c_opt, &yi), &wi)| c_opt.map(|x| (x, yi, wi)))
        .collect();

    if pairs.len() < 4 {
        return None;
    }

    pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let min_v = pairs.first().unwrap().0;
    let max_v = pairs.last().unwrap().0;
    if (max_v - min_v).abs() < 1e-12 {
        return None;
    }

    let mut tot_w = 0.0;
    let mut tot_wy = 0.0;
    let mut tot_wy2 = 0.0;
    for &(_, yi, wi) in &pairs {
        tot_w += wi;
        tot_wy += wi * yi;
        tot_wy2 += wi * yi * yi;
    }

    let parent_sse = tot_wy2 - (tot_wy * tot_wy / tot_w.max(1e-12));

    let mut l_w = 0.0;
    let mut l_wy = 0.0;
    let mut l_wy2 = 0.0;

    let mut best_gain = 0.0;
    let mut best_cut = (min_v + max_v) / 2.0;

    for i in 0..(pairs.len() - 1) {
        let (x, yi, wi) = pairs[i];
        l_w += wi;
        l_wy += wi * yi;
        l_wy2 += wi * yi * yi;

        if pairs[i + 1].0 == x {
            continue;
        }

        let r_w = tot_w - l_w;
        let r_wy = tot_wy - l_wy;
        let r_wy2 = tot_wy2 - l_wy2;

        if l_w <= 0.0 || r_w <= 0.0 {
            continue;
        }

        let sse_l = l_wy2 - (l_wy * l_wy / l_w);
        let sse_r = r_wy2 - (r_wy * r_wy / r_w);
        let gain = parent_sse - (sse_l + sse_r);

        if gain > best_gain {
            best_gain = gain;
            best_cut = (x + pairs[i + 1].0) / 2.0;
        }
    }

    Some((best_cut, best_gain))
}

/// Evaluates FAST bivariate interaction score for feature pair (j, k) using a 2D quad-stump.
pub fn evaluate_pair_interaction(
    col1: &[Option<f64>],
    col2: &[Option<f64>],
    y: &[f64],
    w: &[f64],
    cut1: f64,
    cut2: f64,
    min_samples_leaf: usize,
) -> f64 {
    let mut q_w = [[0.0; 2]; 2];
    let mut q_wy = [[0.0; 2]; 2];
    let mut q_count = [[0usize; 2]; 2];

    let mut tot_w = 0.0;
    let mut tot_wy = 0.0;
    let mut tot_wy2 = 0.0;

    for i in 0..y.len() {
        if let (Some(x1), Some(x2)) = (col1[i], col2[i]) {
            let wi = w[i];
            let yi = y[i];
            let b1 = if x1 <= cut1 { 0 } else { 1 };
            let b2 = if x2 <= cut2 { 0 } else { 1 };

            q_w[b1][b2] += wi;
            q_wy[b1][b2] += wi * yi;
            q_count[b1][b2] += 1;

            tot_w += wi;
            tot_wy += wi * yi;
            tot_wy2 += wi * yi * yi;
        }
    }

    if tot_w <= 0.0 {
        return 0.0;
    }

    // Require sufficient rows in at least 3 quadrants to establish interaction
    let non_empty_quads = q_count.iter().flatten().filter(|&&c| c >= min_samples_leaf).count();
    if non_empty_quads < 3 {
        return 0.0;
    }

    let parent_mean = tot_wy / tot_w;
    let parent_sse = tot_wy2 - (tot_wy * tot_wy / tot_w);
    if parent_sse <= 1e-12 {
        return 0.0;
    }

    // 1D marginal means:
    let w1_0 = q_w[0][0] + q_w[0][1];
    let w1_1 = q_w[1][0] + q_w[1][1];
    let mean1_0 = if w1_0 > 0.0 { (q_wy[0][0] + q_wy[0][1]) / w1_0 } else { parent_mean };
    let mean1_1 = if w1_1 > 0.0 { (q_wy[1][0] + q_wy[1][1]) / w1_1 } else { parent_mean };

    let w2_0 = q_w[0][0] + q_w[1][0];
    let w2_1 = q_w[0][1] + q_w[1][1];
    let mean2_0 = if w2_0 > 0.0 { (q_wy[0][0] + q_wy[1][0]) / w2_0 } else { parent_mean };
    let mean2_1 = if w2_1 > 0.0 { (q_wy[0][1] + q_wy[1][1]) / w2_1 } else { parent_mean };

    let mut inter_gain = 0.0;
    for a in 0..2 {
        for b in 0..2 {
            let qw = q_w[a][b];
            if qw > 0.0 {
                let actual_mean = q_wy[a][b] / qw;
                let m1 = if a == 0 { mean1_0 } else { mean1_1 };
                let m2 = if b == 0 { mean2_0 } else { mean2_1 };
                let additive_mean = parent_mean + (m1 - parent_mean) + (m2 - parent_mean);
                let diff = actual_mean - additive_mean;
                inter_gain += qw * diff * diff;
            }
        }
    }

    (inter_gain / parent_sse).max(0.0)
}

/// Run FAST interaction screening across all pairs of numeric columns.
pub fn fast_interaction_screen(
    cols: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    col_names: &[String],
    top_k: usize,
    min_samples_leaf: usize,
) -> Vec<FastInteraction> {
    let p = cols.len();
    if p < 2 {
        return Vec::new();
    }

    // Step 1: Precompute best 1D stump cutpoint for each column
    let mut cuts = Vec::with_capacity(p);
    for col in cols {
        cuts.push(find_1d_stump_cut(col, y, w));
    }

    // Step 2: Screen all P*(P-1)/2 pairs
    let mut interactions = Vec::new();
    for j in 0..p {
        let (cut1, _) = match cuts[j] {
            Some(c) => c,
            None => continue,
        };
        for k in (j + 1)..p {
            let (cut2, _) = match cuts[k] {
                Some(c) => c,
                None => continue,
            };

            let score = evaluate_pair_interaction(
                &cols[j],
                &cols[k],
                y,
                w,
                cut1,
                cut2,
                min_samples_leaf,
            );

            if score > 1e-6 {
                interactions.push(FastInteraction {
                    col1: j,
                    col2: k,
                    col1_name: col_names.get(j).cloned().unwrap_or_else(|| format!("x{j}")),
                    col2_name: col_names.get(k).cloned().unwrap_or_else(|| format!("x{k}")),
                    score,
                    cut1,
                    cut2,
                });
            }
        }
    }

    // Sort descending by interaction score
    interactions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    interactions.truncate(top_k);
    interactions
}
