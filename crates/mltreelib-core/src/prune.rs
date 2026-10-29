//! Post-growth pruning (the `prune` knob): encoding cost (MDL/BIC) of a
//! subtree versus its parent kept as one leaf.

use crate::model_tree::Grower;
/// Optional post-growth pruning cost (bottom-up: a subtree is replaced by
/// its parent leaf when the leaf's encoding cost is lower).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PruneCost {
    /// No cost pruning (min_gain only).
    None,
    /// Bayesian information criterion: `n·ln(SSE/n) + ln(n)` per leaf,
    /// `ln(n)` per split — universal (regression and class targets).
    Bic,
    /// Mehta/Rissanen/Agrawal MDL node cost over y-as-class-labels.
    Mdl,
}
impl<'a> Grower<'a> {
    /// Encoding cost of treating `rows` as one leaf (prune criterion).
    pub(crate) fn leaf_cost(&self, rows: &[usize]) -> f64 {
        let n = rows.len() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        match self.params.prune {
            PruneCost::None => 0.0,
            PruneCost::Bic => {
                let sw: f64 = rows.iter().map(|&r| self.w[r]).sum();
                let mean =
                    rows.iter().map(|&r| self.w[r] * self.y[r]).sum::<f64>() / sw.max(1e-300);
                let sse: f64 = rows
                    .iter()
                    .map(|&r| self.w[r] * (self.y[r] - mean) * (self.y[r] - mean))
                    .sum();
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
