//! Leaves: what a leaf model is ([`LeafKind`]), which columns it may use
//! ([`LeafCols`]), and fitting / evaluating it on a node's rows.

use crate::family::LeafFamily;
use crate::model_tree::Grower;
use crate::node_stats::{design_row, fit_glm, fit_linear, Gram, LeafModel, LeafStats, LinearModel};
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
        self.fit_leaf(rows, path, self.params.leaf_digest)
    }

    /// Candidate fitting shares the payload fitter but does not sort targets
    /// or allocate a target distribution. Never store this leaf in the tree.
    pub(crate) fn make_leaf_for_scan(&self, rows: &[usize], path: &[usize]) -> LeafModel {
        self.fit_leaf(rows, path, false)
    }

    /// Response-scale weighted deviance of an already fitted leaf.
    pub(crate) fn leaf_deviance(&self, rows: &[usize], leaf: &LeafModel) -> f64 {
        let mut total = 0.0;
        for &r in rows {
            if self.w[r] <= 0.0 {
                continue;
            }
            let loss = leaf.family.deviance(self.y[r], leaf.predict_columns(self.cols, r));
            if !loss.is_finite() {
                return loss;
            }
            total += self.w[r] * loss;
        }
        total
    }

    fn fit_leaf(&self, rows: &[usize], path: &[usize], digest: bool) -> LeafModel {
        let family = self.params.leaf_family;
        let stats = if digest || family == LeafFamily::Gaussian {
            let yv: Vec<f64> = rows.iter().map(|&r| self.y[r]).collect();
            let wv: Vec<f64> = rows.iter().map(|&r| self.w[r]).collect();
            let mut stats = LeafStats::from_targets(&yv, &wv);
            if !digest {
                // Lean Gaussian leaves retain historical stats semantics;
                // only the digest is dropped.
                stats.digest = tdigest::TDigest::new_with_size(1);
            }
            stats
        } else {
            LeafStats::from_rows(self.y, self.w, rows)
        };
        let family = self.params.leaf_family;
        if family != LeafFamily::Gaussian
            && (self.params.leaf == LeafKind::Mean
                || (family == LeafFamily::Binomial && (stats.mean == 0.0 || stats.mean == 1.0))
                || (family == LeafFamily::Poisson && stats.mean == 0.0))
        {
            return LeafModel { center: Some(stats.mean), stats, linear: None, family };
        }
        let linear = if self.params.leaf == LeafKind::Linear {
            let (chosen, any_missing, dim) = self.leaf_design(rows, path);
            if family != LeafFamily::Gaussian {
                let fitted = fit_glm(
                    self.cols, self.y, self.w, rows, &chosen, any_missing,
                    &self.params.monotone, self.params.l1, self.params.l2,
                    family, stats.mean,
                );
                // The tree fit rejects nonfinite final models; scans reject
                // their loss. A failed GLM must never masquerade as a mean fit.
                let linear = fitted.unwrap_or_else(|| LinearModel {
                    coef: vec![0.0; chosen.len()],
                    na_coef: if any_missing { vec![0.0; chosen.len()] } else { Vec::new() },
                    cols: chosen,
                    intercept: f64::NAN,
                });
                return LeafModel { stats, linear: Some(linear), center: None, family };
            }
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
        let yws: Vec<(f64, f64)> = if matches!(
            self.params.leaf, LeafKind::Median | LeafKind::Quantile(_) | LeafKind::Huber
        ) {
            rows.iter().filter(|&&r| self.w[r] > 0.0)
                .map(|&r| (self.y[r], self.w[r])).collect()
        } else {
            Vec::new()
        };
        let center = match self.params.leaf {
            LeafKind::Median => Some(w_median(&yws)),
            LeafKind::Quantile(t) => Some(w_quantile(&yws, t)),
            LeafKind::Huber => Some(huber_location(&yws)),
            _ => None,
        };
        LeafModel { stats, linear, center, family }
    }
}
