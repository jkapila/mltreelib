//! Leaves: what a leaf model is ([`LeafKind`]), which columns it may use
//! ([`LeafCols`]), and fitting / evaluating it on a node's rows.

use crate::model_tree::Grower;
use crate::node_stats::{design_row, fit_linear, Gram, LeafModel, LeafStats};
use crate::stats::{huber_location, w_median, w_quantile};
/// Which columns a leaf's linear model may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafCols {
    /// Every usable predictor column.
    All,
    /// Only the columns used in splits along the root→leaf path.
    PathUsed,
}

/// What the leaf model is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LeafKind {
    /// Linear model over the chosen columns (the model tree).
    Linear,
    /// Plain mean — the CART/rpart leaf. The split criterion then *is*
    /// rpart's deviance criterion.
    Mean,
    /// Weighted median — L1-optimal constant, outlier-robust.
    Median,
    /// Weighted quantile at tau (0.5 = median; any tau in (0, 1)).
    Quantile(f64),
    /// Huber M-location (quadratic near the center, linear in the tails).
    Huber,
}

impl<'a> Grower<'a> {
    /// Columns a leaf model uses for a given root→leaf path (restricted to
    /// the pre-selected active columns).
    pub(crate) fn chosen_cols(&self, path: &[usize]) -> Vec<usize> {
        match self.params.leaf_cols {
            LeafCols::All => self.active.to_vec(),
            LeafCols::PathUsed => {
                let mut p: Vec<usize> = path
                    .iter()
                    .copied()
                    .filter(|c| self.active.contains(c))
                    .collect();
                p.sort_unstable();
                p.dedup();
                p
            }
        }
    }

    /// Leaf design for a node: the leaf columns under the column policy,
    /// whether any of them is missing on these rows (NA-indicator columns),
    /// and the resulting design dimension (intercept included).
    fn leaf_design(&self, rows: &[usize], path: &[usize]) -> (Vec<usize>, bool, usize) {
        let chosen = self.chosen_cols(path);
        let any_missing = rows
            .iter()
            .any(|&r| chosen.iter().any(|&c| self.cols[c][r].is_none()));
        let dim = 1 + chosen.len() + if any_missing { chosen.len() } else { 0 };
        (chosen, any_missing, dim)
    }

    /// Weighted residual sum of squares of a leaf fit on `rows`, with the
    /// leaf's parameter count (mean-style leaves: 1; linear: the design
    /// dimension). Returns NaN RSS when the fit cannot produce finite
    /// predictions (the gate then skips).
    pub(crate) fn node_rss(&self, rows: &[usize], path: &[usize]) -> (f64, usize) {
        let leaf = self.make_leaf(rows, path);
        let k = if self.params.leaf == LeafKind::Linear {
            self.leaf_design(rows, path).2
        } else {
            1
        };
        let mut xrow = vec![None; self.cols.len()];
        let mut rss = 0.0;
        for &r in rows {
            for (j, c) in self.cols.iter().enumerate() {
                xrow[j] = c[r];
            }
            let e = self.y[r] - leaf.predict(&xrow);
            if !e.is_finite() {
                return (f64::NAN, k);
            }
            rss += self.w[r] * e * e;
        }
        (rss, k)
    }

    /// Rows a leaf model fits on: the honesty estimation rows when there
    /// are enough (`min_rows`), else the node's structure rows.
    pub(crate) fn leaf_rows<'b>(&self, rows: &'b [usize], est: &'b [usize]) -> &'b [usize] {
        if self.params.honesty && est.len() >= self.params.min_rows {
            est
        } else {
            rows
        }
    }

    /// Fit the leaf model for a node's rows.
    pub(crate) fn make_leaf(&self, rows: &[usize], path: &[usize]) -> LeafModel {
        let yv: Vec<f64> = rows.iter().map(|&r| self.y[r]).collect();
        let wv: Vec<f64> = rows.iter().map(|&r| self.w[r]).collect();
        let mut stats = LeafStats::from_targets(&yv, &wv);
        if !self.params.leaf_digest {
            // Lean leaves: drop the t-digest (`quantile` then returns NaN).
            stats.digest = tdigest::TDigest::new_with_size(1);
        }
        let linear = if self.params.leaf == LeafKind::Linear {
            let (chosen, any_missing, dim) = self.leaf_design(rows, path);
            let mut g = Gram::new(dim);
            let mut row = vec![0.0; dim];
            for &r in rows {
                design_row(&mut row, self.cols, r, &chosen, any_missing);
                g.add_row(&row, self.y[r], self.w[r]);
            }
            Some(fit_linear(
                &g,
                &chosen,
                any_missing,
                &self.params.monotone,
                self.params.l1,
                self.params.l2,
            ))
        } else {
            None
        };
        // Non-mean constant leaves: fit the center on the positive-weight
        // targets only (same population as the stats).
        let yws: Vec<(f64, f64)> = yv
            .iter()
            .zip(wv.iter())
            .map(|(&y, &w)| (y, w))
            .filter(|&(_, w)| w > 0.0)
            .collect();
        let center = match self.params.leaf {
            LeafKind::Median => Some(w_median(&yws)),
            LeafKind::Quantile(t) => Some(w_quantile(&yws, t)),
            LeafKind::Huber => Some(huber_location(&yws)),
            _ => None,
        };
        LeafModel { stats, linear, center }
    }
}
