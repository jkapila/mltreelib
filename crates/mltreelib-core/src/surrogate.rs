//! NA surrogate splits (the `maxsurrogate` knob): when a prediction row is
//! missing the node's split column, a *surrogate* split — a backup split on
//! another column that best mimics the primary split's row partition —
//! routes it. rpart lineage (`usesurrogate`/`maxsurrogate`); the per-split
//! `missing_left` remains the last resort when no surrogate is observed.
//!
//! Surrogate quality = weighted agreement with the primary partition over
//! rows where the surrogate column is observed; only mimics better than
//! chance (agreement > 0.5) are kept, best first, up to `maxsurrogate` per
//! node. A surrogate whose best agreement comes from *mirrored* routing
//! stores `flip: true`. v1 keeps surrogates numeric-threshold splits.

use std::collections::HashSet;

use crate::model_tree::{Grower, Split};

/// One backup split: threshold on column `col`, routed like a primary
/// split (mirrored when `flip`).
#[derive(Clone, Copy, Debug)]
pub struct Surrogate {
    pub col: usize,
    pub threshold: f64,
    /// The mimic agrees by mirroring the primary (left/right swapped).
    pub flip: bool,
    /// Weighted agreement with the primary partition (observed rows only).
    pub agreement: f64,
}

impl<'a> Grower<'a> {
    /// Search the surrogate splits of a chosen primary split: for every
    /// other active column, the threshold whose routing best agrees with
    /// the primary partition. Kept when agreement > 0.5, best first, at
    /// most `params.maxsurrogate`.
    pub(crate) fn find_surrogates(
        &self,
        rows: &[usize],
        primary: &Split,
        primary_left: &[usize],
    ) -> Vec<Surrogate> {
        if self.params.maxsurrogate == 0 {
            return Vec::new();
        }
        let left: HashSet<usize> = primary_left.iter().copied().collect();
        let mut out: Vec<Surrogate> = Vec::new();
        for j in 0..self.cols.len() {
            if j == primary.col || !self.active.contains(&j) {
                continue;
            }
            if let Some(s) = self.best_agreement(rows, j, &left) {
                if s.agreement > 0.5 {
                    out.push(s);
                }
            }
        }
        out.sort_by(|a, b| {
            b.agreement
                .total_cmp(&a.agreement)
                .then(a.col.cmp(&b.col))
        });
        out.truncate(self.params.maxsurrogate);
        out
    }

    /// The threshold on column `j` whose routing (or its mirror) best
    /// agrees with the primary partition (weighted, observed values only).
    fn best_agreement(
        &self,
        rows: &[usize],
        j: usize,
        left: &HashSet<usize>,
    ) -> Option<Surrogate> {
        let mut pairs: Vec<(f64, f64, bool)> = rows
            .iter()
            .filter_map(|&r| self.cols[j][r].map(|v| (v, self.w[r], left.contains(&r))))
            .collect();
        if pairs.len() < 2 {
            return None;
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let total_w: f64 = pairs.iter().map(|(_, w, _)| w).sum();
        if total_w <= 0.0 {
            return None;
        }
        // Sweep thresholds at value changes: routing left = value <= v.
        let mut best: Option<Surrogate> = None;
        let mut best_agree = 0.5;
        let mut lw = 0.0;
        let mut lw_left = 0.0;
        let rw_left: f64 = pairs
            .iter()
            .filter(|(_, _, l)| *l)
            .map(|(_, w, _)| w)
            .sum();
        for k in 0..pairs.len() {
            let (v, w, is_left) = pairs[k];
            lw += w;
            if is_left {
                lw_left += w;
            }
            if k + 1 == pairs.len() || pairs[k + 1].0 != v {
                // threshold = v (equal values stay on one side)
                let rw_left_now = rw_left - lw_left;
                let straight = lw_left + ((total_w - lw) - rw_left_now);
                let mirrored = (lw - lw_left) + rw_left_now;
                let agree = straight.max(mirrored) / total_w;
                if agree > best_agree {
                    best_agree = agree;
                    best = Some(Surrogate {
                        col: j,
                        threshold: v,
                        flip: mirrored > straight,
                        agreement: agree,
                    });
                }
            }
        }
        best
    }
}
