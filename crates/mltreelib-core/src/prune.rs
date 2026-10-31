//! Post-growth pruning (the `prune` knob): encoding cost (MDL/BIC) of a
//! subtree versus its parent kept as one leaf.

use crate::model_tree::Grower;
/// Optional post-growth pruning cost (bottom-up: a subtree is replaced by
/// its parent leaf when the leaf's cost is lower).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PruneCost {
    /// No cost pruning (min_gain only).
    None,
    /// Bayesian information criterion: `n·ln(SSE/n) + ln(n)` per leaf for
    /// Gaussian, `family deviance + k·ln(n)` (k = fitted parameter count)
    /// for distributional families.
    Bic,
    /// Mehta/Rissanen/Agrawal MDL node cost over y-as-class-labels.
    Mdl,
    /// Cost-complexity (rpart `cp` / sklearn `ccp_alpha` semantics):
    /// node deviance + `cp_alpha` per leaf, greedy bottom-up (the
    /// whole-tree weakest-link path arrives with `cp_xval`).
    Cp,
}

impl PruneCost {
    /// Cost of keeping one split in the subtree's total (`ln(n)` for the
    /// encoding criteria; free for cost-complexity, whose per-leaf penalty
    /// is already in the leaf costs).
    pub(crate) fn split_bit(self, n: f64) -> f64 {
        match self {
            PruneCost::Bic | PruneCost::Mdl => n.ln().max(0.0),
            _ => 0.0,
        }
    }
}
impl<'a> Grower<'a> {
    /// Weighted SSE of `rows` around their weighted mean — the node
    /// deviance (rpart's convention, independent of leaf kind).
    fn mean_sse(&self, rows: &[usize]) -> f64 {
        let sw: f64 = rows.iter().map(|&r| self.w[r]).sum();
        let mean = rows.iter().map(|&r| self.w[r] * self.y[r]).sum::<f64>() / sw.max(1e-300);
        rows.iter()
            .map(|&r| self.w[r] * (self.y[r] - mean) * (self.y[r] - mean))
            .sum()
    }

    /// Encoding cost of treating `rows` as one leaf (prune criterion).
    pub(crate) fn leaf_cost(&self, rows: &[usize], path: &[usize]) -> f64 {
        let n = rows.len() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        if self.params.leaf_family != crate::family::LeafFamily::Gaussian
            && matches!(self.params.prune, PruneCost::Cp | PruneCost::Bic)
        {
            let leaf = self.make_leaf_for_scan(rows, path);
            let deviance = self.leaf_deviance(rows, &leaf);
            if deviance.is_nan() {
                return f64::NAN;
            }
            return match self.params.prune {
                PruneCost::Cp => deviance + self.params.cp_alpha,
                PruneCost::Bic => {
                    let k = leaf.linear.as_ref()
                        .map(|m| 1 + m.coef.len() + m.na_coef.len()).unwrap_or(1);
                    deviance + k as f64 * n.ln()
                }
                _ => unreachable!(),
            };
        }
        match self.params.prune {
            PruneCost::None => 0.0,
            PruneCost::Cp => self.mean_sse(rows) + self.params.cp_alpha,
            PruneCost::Bic => {
                let sse = self.mean_sse(rows);
                n * (sse / n).max(1e-300).ln() + n.ln()
            }
            PruneCost::Mdl => {
                // Mehta/Rissanen/Agrawal node cost over y-as-class-labels
                // (weighted class masses).
                let mut counts: std::collections::HashMap<u64, f64> =
                    std::collections::HashMap::new();
                for &r in rows {
                    *counts.entry(self.y[r].to_bits()).or_insert(0.0) += self.w[r];
                }
                let k = counts.len() as f64;
                let total: f64 = counts.values().sum();
                let data: f64 = counts
                    .values()
                    .map(|&c| if c > 0.0 { c * (total / c).ln() } else { 0.0 })
                    .sum();
                data + (k - 1.0) / 2.0 * (total / 2.0).ln() + k / 2.0 * (k / 2.0).ln().max(0.0)
            }
        }
    }

}
