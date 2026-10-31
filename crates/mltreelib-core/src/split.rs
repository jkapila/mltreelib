//! The split search: per-column candidate scans (mean-leaf rpart-style
//! running sums, linear-leaf LightGBM-style grouped Gram accumulators,
//! categorical level partitions) and the admissibility knobs they honor
//! (monotone, interaction groups, balance). Loop-shape references live on
//! [`crate::model_tree`]'s module doc.

use std::time::Instant;

use crate::criteria::{beats, gain_of, GainFn};
use crate::leaf::{LeafCols, LeafKind};
use crate::model_tree::{Grower, NodeProfile, Split};
use crate::node_stats::{design_row, fit_linear, solve_normal_sum, Gram, SideTrack, SolveScratch};

/// Below this much work (rows x columns) the scan stays single-threaded —
/// thread overhead would lose.
const PAR_MIN_WORK: usize = 20_000;
/// Total rows across level buckets (categorical scan bookkeeping).
fn levels_total(levels: &[(f64, Vec<(f64, f64)>)]) -> usize {
    levels.iter().map(|(_, rows)| rows.len()).sum()
}
/// Column-independent parent statistics for one node's scan — valid
/// under the `all` leaf-column policy, where the design class does not
/// depend on which column is being scanned.
struct ParentPre {
    parent_sse: f64,
    mean_row: Vec<f64>,
}

/// Per-column data handler — exactly three kinds, the split loop's whole
/// world. Values are `f64` at the core boundary; for categorical columns a
/// distinct value IS a level (identity, never order). Missing (`None`/NA)
/// is a first-class state: a surrogate split routes it when present, else
/// `missing_left` — or it can be carried as a level of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColType {
    /// Integer handler: scan unique values small → big; the cut is exact
    /// (`v < a`) — no precision grid.
    Integer,
    /// Float handler: scan unique values small → big on the
    /// `split_precision` grid (the precision parameter).
    Float,
    /// Categorical/level handler: scan levels largest frequency → smallest
    /// (`cat_order`); the cut is set membership (`v in {a,b,c}`).
    Categorical,
}

/// Precision grid of column `j`: the integer handler scans exact unique
/// values (no grid); the float handler uses the `split_precision`
/// parameter.
pub(crate) fn grid_of(
    dtypes: &[ColType],
    split_precision: Option<f64>,
    j: usize,
) -> Option<f64> {
    match dtypes.get(j) {
        Some(ColType::Integer) => None,
        _ => split_precision,
    }
}

/// Order in which a categorical column's levels are visited when building
/// the candidate partitions.
#[derive(Clone, Debug, PartialEq)]
pub enum CatOrder {
    /// Most frequent level first (counted at the node).
    FrequencyDesc,
    /// Highest target mean first (CART-style ordering).
    TargetMean,
    /// Explicit level order (codes); levels not listed are appended after,
    /// by descending frequency.
    Explicit(Vec<f64>),
}

/// Which candidate partitions a categorical column contributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatSearch {
    /// Cumulative sweep along the order: {top-1}|rest, {top-2}|rest, ...
    /// `k-1` candidates. Default.
    Cumulative,
    /// Each level alone vs the rest, visited in order; stops early at
    /// `gain >= min_gain` on the first acceptable level.
    SingleVsRest,
    /// The union of both candidate sets; best gain wins.
    Both,
}

/// Work of one column scan, merged into the node profile.
#[derive(Clone, Copy, Debug, Default)]
struct ScanStats {
    group_ns: u64,
    scan_ns: u64,
    candidates: usize,
}

impl<'a> Grower<'a> {
    /// Interaction constraint (LightGBM-style): a path may only use columns
    /// from one interaction group; ungrouped columns never interact.
    fn interact_ok(&self, path: &[usize], j: usize) -> bool {
        let g = &self.params.interaction_groups;
        if g.is_empty() {
            return true;
        }
        let group_of = |c: usize| g.iter().position(|grp| grp.contains(&c));
        match group_of(j) {
            None => path.iter().all(|&p| group_of(p).is_none()),
            Some(gj) => path.iter().all(|&p| group_of(p) == Some(gj)),
        }
    }

    /// Monotone check: a split on constrained column j must order the
    /// children's (weighted) target means in the constraint's direction.
    fn mono_ok(&self, j: usize, mean_l: f64, mean_r: f64) -> bool {
        if self.params.monotone.is_empty() {
            return true;
        }
        let m = self.params.monotone[j];
        m == 0 || (mean_l - mean_r) * (m as f64) <= 0.0
    }

    /// Parent fit score (the gain's origin) and the node's mean design row
    /// (the monotone gate's evaluation point) for one scan design.
    /// Computed once per node by `best_split` when the design is
    /// column-independent (`leaf_cols = all`), once per column otherwise.
    fn parent_stats(&self, rows: &[usize], chosen: &[usize], dim: usize) -> (f64, Vec<f64>) {
        let mut pgram = Gram::new(dim);
        let mut prow = vec![0.0; dim];
        for &r in rows {
            design_row(&mut prow, self.cols, r, chosen, true);
            pgram.add_row(&prow, self.y[r], self.w[r]);
        }
        let mut scratch = SolveScratch::new(dim);
        let parent_sse = solve_normal_sum(
            &pgram,
            None,
            dim,
            self.params.l1,
            self.params.l2,
            &mut scratch,
            None,
        );
        // Monotone columns with linear leaves need one more gate per
        // candidate: the children's fitted outputs must be ordered AT the
        // threshold (their intercepts meet there - child means alone don't
        // order linear responses). The eval point holds other columns at
        // the node's mean design row.
        let mut mean_row = vec![0.0; dim];
        for &r in rows {
            design_row(&mut prow, self.cols, r, chosen, true);
            for (m, v) in mean_row.iter_mut().zip(&prow) {
                *m += *v;
            }
        }
        let inv_n = 1.0 / rows.len().max(1) as f64;
        for v in mean_row.iter_mut() {
            *v *= inv_n;
        }
        (parent_sse, mean_row)
    }

    /// Best split across columns. Columns scan in parallel above a work
    /// threshold; the winner is chosen deterministically (gain, then
    /// scan order: lowest column index, lowest threshold, missing_left
    /// first) — never by thread completion order.
    pub(crate) fn best_split(
        &mut self,
        rows: &[usize],
        path: &[usize],
        prof: &mut NodeProfile,
    ) -> Option<(Split, Vec<usize>, Vec<usize>)> {
        if rows.len() < self.min_split() {
            return None;
        }

        let mut child_path = path.to_vec();
        let mut candidates: Vec<(usize, Option<(f64, Split, Option<Vec<f64>>)>, ScanStats)> =
            Vec::new();
        let work = rows.len() * self.cols.len();
        let mut cols: Vec<usize> = self
            .active
            .iter()
            .copied()
            .filter(|&j| self.interact_ok(path, j))
            .filter(|&j| {
                self.params
                    .split_cols
                    .as_ref()
                    .map(|sc| sc.contains(&j))
                    .unwrap_or(true)
            })
            .collect();
        // Per-node random column subset (random-forest spirit): a seeded
        // xorshift draw, deterministic across runs and threads.
        if let Some(frac) = self.params.col_sample {
            let k = ((cols.len() as f64) * frac).round().max(1.0) as usize;
            if k < cols.len() {
                let mut rng = self.params.seed
                    ^ (self.rng_counter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                for i in 0..k {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let j = i + (rng as usize) % (cols.len() - i);
                    cols.swap(i, j);
                }
                cols.truncate(k);
                cols.sort_unstable(); // deterministic scan order
            }
        }
        self.rng_counter += 1;

        // Parent score + mean design row are column-independent under the
        // `all` leaf-column policy: hoisted from per-column to per-node.
        let parent_pre: Option<ParentPre> = match self.params.leaf_cols {
            LeafCols::All
                if self.params.leaf_family == crate::family::LeafFamily::Gaussian
                    && self.params.leaf == LeafKind::Linear =>
            {
                let chosen = self.chosen_cols(path);
                let dim = 1 + 2 * chosen.len();
                let (parent_sse, mean_row) = self.parent_stats(rows, &chosen, dim);
                Some(ParentPre { parent_sse, mean_row })
            }
            LeafCols::All
                if self.params.leaf_family != crate::family::LeafFamily::Gaussian
                    && self.params.leaf == LeafKind::Linear
                    && matches!(self.params.gain, GainFn::Sse | GainFn::Deviance) =>
            {
                let leaf = self.make_leaf_for_scan(rows, path);
                Some(ParentPre {
                    parent_sse: self.fitted_family_score(rows, &leaf),
                    mean_row: Vec::new(),
                })
            }
            _ => None,
        };

        if work >= PAR_MIN_WORK {
            use rayon::prelude::*;
            let mut out: Vec<(usize, Option<(f64, Split, Option<Vec<f64>>)>, ScanStats)> = cols
                .par_iter()
                .map(|&j| {
                    let mut cp = path.to_vec();
                    cp.push(j);
                    let (found, st) = self.scan_column(rows, j, &cp, parent_pre.as_ref());
                    (j, found, st)
                })
                .collect();
            out.sort_by_key(|(j, _, _)| *j); // deterministic order
            candidates = out;
        } else {
            for &j in &cols {
                child_path.push(j);
                let (found, st) = self.scan_column(rows, j, &child_path, parent_pre.as_ref());
                child_path.pop();
                candidates.push((j, found, st));
            }
        }

        let mut best: Option<(f64, Split, Option<Vec<f64>>, usize, f64)> = None;
        for (j, found, st) in candidates {
            prof.cols += 1;
            prof.group_ns += st.group_ns;
            prof.scan_ns += st.scan_ns;
            prof.candidates += st.candidates;
            if let Some((gain, sp, part)) = found {
                let better = match &best {
                    None => true,
                    Some((g, _, _, bj, bt)) => {
                        beats(gain, *g)
                            || (!beats(*g, gain) && (j, sp.threshold) < (*bj, *bt))
                    }
                };
                if better {
                    best = Some((gain, sp, part, j, sp.threshold));
                }
            }
        }

        best.map(|(_, mut s, part, ..)| {
            if let Some(p) = part {
                s.cat = Some(self.cat_parts.len());
                self.cat_parts.push(p);
            }
            let (lv, rv) = self.partition_rows(rows, &s);
            (s, lv, rv)
        })
    }

    /// Route `rows` through a split (missing values follow
    /// `missing_left`). Shared by the structure partition and the honesty
    /// estimation partition.
    pub(crate) fn partition_rows(&self, rows: &[usize], sp: &Split) -> (Vec<usize>, Vec<usize>) {
        let (mut lv, mut rv) = (Vec::new(), Vec::new());
        for &r in rows {
            let go_left = match self.cols[sp.col][r] {
                Some(v) => self.goes_left(sp, v),
                None => sp.missing_left,
            };
            if go_left {
                lv.push(r);
            } else {
                rv.push(r);
            }
        }
        (lv, rv)
    }

    /// Dispatch one column's scan to the dtype/leaf-kind-specific loop.
    /// The hit carries an optional categorical partition (registered by
    /// `best_split` into `cat_parts`).
    fn scan_column(
        &self,
        rows: &[usize],
        j: usize,
        path: &[usize],
        pre: Option<&ParentPre>,
    ) -> (Option<(f64, Split, Option<Vec<f64>>)>, ScanStats) {
        if self.params.leaf_family != crate::family::LeafFamily::Gaussian
            && self.params.leaf == LeafKind::Linear
        {
            return self.scan_column_family(rows, j, path, pre);
        }
        if self.col_type(j) == ColType::Categorical {
            return self.scan_column_categorical(rows, j);
        }
        let (hit, st) = match self.params.leaf {
            LeafKind::Linear => self.scan_column_linear(rows, j, path, pre),
            // Constant leaves (mean/median/quantile/huber) share the stats
            // scan; the `gain` knob picks the quality function.
            _ => self.scan_column_mean(rows, j),
        };
        (hit.map(|(g, s)| (g, s, None)), st)
    }

    /// Column type for the split search: `params.dtypes` when set, else
    /// float.
    fn col_type(&self, j: usize) -> ColType {
        self.params.dtypes.get(j).copied().unwrap_or(ColType::Float)
    }

    /// Precision grid for a column's scan: integer handlers scan exact
    /// unique values (no grid); float handlers use `split_precision`.
    fn grid_for(&self, j: usize) -> Option<f64> {
        grid_of(&self.params.dtypes, self.params.split_precision, j)
    }

    /// Minimum node rows to attempt a split (ctree `minsplit`); defaults to
    /// the historical `2 * min_rows` gate.
    fn min_split(&self) -> usize {
        self.params.minsplit.unwrap_or(2 * self.params.min_rows)
    }

    /// Minimum rows per child: the `minbucket` floor and the `minprob`
    /// share of the node (ctree trio).
    fn min_child(&self, node_rows: usize) -> usize {
        let floor = self.params.minbucket.unwrap_or(self.params.min_rows).max(1);
        let share = (self.params.minprob * node_rows as f64).ceil() as usize;
        floor.max(share)
    }

    /// Categorical scan — binary level partitions per `cat_order` /
    /// `cat_search`. Levels are distinct non-missing values; missing rows
    /// are scored in both directions. Monotone constraints are ignored
    /// here (they shape numeric trends). Gain ranking for `LeafKind::Linear`
    /// columns also uses `params.gain` on the y-slices (partition quality),
    /// independent of the leaf fit.
    fn scan_column_categorical(
        &self,
        rows: &[usize],
        j: usize,
    ) -> (Option<(f64, Split, Option<Vec<f64>>)>, ScanStats) {
        let mut st = ScanStats::default();
        let t0 = Instant::now();
        let mut missing_yws: Vec<(f64, f64)> = Vec::new();
        let mut by_level: std::collections::HashMap<u64, Vec<(f64, f64)>> =
            std::collections::HashMap::new();
        for &r in rows {
            let (y, w) = (self.y[r], self.w[r]);
            match self.cols[j][r] {
                Some(v) => {
                    by_level.entry(v.to_bits()).or_default().push((y, w));
                }
                None => missing_yws.push((y, w)),
            }
        }
        let mut levels: Vec<(f64, Vec<(f64, f64)>)> = by_level
            .into_iter()
            .map(|(b, ys)| (f64::from_bits(b), ys))
            .collect();
        if levels.len() < 2 {
            st.group_ns = t0.elapsed().as_nanos() as u64;
            return (None, st);
        }
        // Level order (deterministic: ties by level code ascending).
        match &self.params.cat_order {
            CatOrder::FrequencyDesc => {
                levels.sort_by(|a, b| {
                    b.1.len()
                        .cmp(&a.1.len())
                        .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                });
            }
            CatOrder::TargetMean => {
                let wmean = |p: &[(f64, f64)]| {
                    let sw: f64 = p.iter().map(|&(_, w)| w).sum();
                    if sw <= 0.0 {
                        0.0
                    } else {
                        p.iter().map(|&(y, w)| w * y).sum::<f64>() / sw
                    }
                };
                levels.sort_by(|a, b| {
                    wmean(&b.1)
                        .partial_cmp(&wmean(&a.1))
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                });
            }
            CatOrder::Explicit(order) => {
                let rank = |v: f64| {
                    order
                        .iter()
                        .position(|o| *o == v)
                        .unwrap_or(usize::MAX)
                };
                levels.sort_by(|a, b| {
                    rank(a.0)
                        .cmp(&rank(b.0))
                        .then(b.1.len().cmp(&a.1.len()))
                        .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                });
            }
        }
        st.group_ns += t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();

        let k = levels.len();
        let m = self.min_child(rows.len()) as f64;
        // Candidate partitions as level-index sets (left side).
        let mut candidates: Vec<Vec<usize>> = Vec::new();
        match self.params.cat_search {
            CatSearch::Cumulative => {
                for i in 1..k {
                    candidates.push((0..i).collect());
                }
            }
            CatSearch::SingleVsRest => {
                for i in 0..k {
                    candidates.push(vec![i]);
                }
            }
            CatSearch::Both => {
                for i in 1..k {
                    candidates.push((0..i).collect());
                }
                for i in 0..k {
                    candidates.push(vec![i]);
                }
            }
        }

        let wsum = |p: &[(f64, f64)]| p.iter().map(|&(_, w)| w).sum::<f64>();
        let total_w: f64 = wsum(&missing_yws) + levels.iter().map(|(_, r)| wsum(r)).sum::<f64>();
        let mut best: Option<(f64, Vec<f64>, bool)> = None; // (gain, left levels, missing_left)
        for left_idx in candidates.iter() {
            let in_left: std::collections::HashSet<usize> = left_idx.iter().copied().collect();
            let mut ln = 0usize;
            let mut left_yws: Vec<(f64, f64)> = Vec::new();
            let mut right_yws: Vec<(f64, f64)> = Vec::new();
            for (li, (_, ys)) in levels.iter().enumerate() {
                if in_left.contains(&li) {
                    ln += ys.len();
                    left_yws.extend_from_slice(ys);
                } else {
                    right_yws.extend_from_slice(ys);
                }
            }
            let rn = levels_total(&levels) - ln;
            if (ln as f64) < m || (rn as f64) < m {
                continue;
            }
            st.candidates += 1;
            for missing_left in [true, false] {
                let (aln, arn) = if missing_left {
                    (ln + missing_yws.len(), rn)
                } else {
                    (ln, rn + missing_yws.len())
                };
                if (aln as f64) < m || (arn as f64) < m {
                    continue;
                }
                let mut ls = left_yws.clone();
                let mut rs = right_yws.clone();
                if missing_left {
                    ls.extend_from_slice(&missing_yws);
                } else {
                    rs.extend_from_slice(&missing_yws);
                }
                // grf-style alpha: each child carries at least alpha of the
                // node's total weight.
                if self.params.alpha > 0.0
                    && (wsum(&ls) < self.params.alpha * total_w
                        || wsum(&rs) < self.params.alpha * total_w)
                {
                    continue;
                }
                let gain = self.partition_gain(&ls, &rs);
                let gain = if self.params.imbalance_penalty > 0.0 {
                    gain - self.params.imbalance_penalty * (1.0 / aln as f64 + 1.0 / arn as f64)
                } else {
                    gain
                };
                if gain <= self.params.min_gain {
                    continue;
                }
                if best.as_ref().map(|(g, ..)| beats(gain, *g)).unwrap_or(true) {
                    let left_levels: Vec<f64> =
                        left_idx.iter().map(|&li| levels[li].0).collect();
                    best = Some((gain, left_levels, missing_left));
                }
                // Single-vs-rest loops in `cat_order` and stops at the
                // first acceptable level (gain > min_gain, checked above).
                if matches!(self.params.cat_search, CatSearch::SingleVsRest) {
                    break;
                }
            }
        }

        st.scan_ns += t1.elapsed().as_nanos() as u64;
        (
            best.map(|(gain, left_levels, missing_left)| {
                (
                    gain,
                    Split {
                        col: j,
                        threshold: 0.0,
                        missing_left,
                        gain,
                        cat: None,
                        sur: None,
                    },
                    Some(left_levels),
                )
            }),
            st,
        )
    }

    /// Mean-leaf scan — rpart's `anova.c` shape: weighted running sums per
    /// side, O(1) per value, split only where the value changes.
    fn scan_column_mean(&self, rows: &[usize], j: usize) -> (Option<(f64, Split)>, ScanStats) {
        let mut st = ScanStats::default();
        let m = self.min_child(rows.len()) as f64;
        let t0 = Instant::now();
        let mut pairs: Vec<(f64, f64, f64)> = rows
            .iter()
            .filter_map(|&r| self.cols[j][r].map(|v| (v, self.y[r], self.w[r])))
            .collect();
        if pairs.len() < 2 {
            return (None, st);
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN in predictors"));
        // split_precision grid: values round to the grid before the scan;
        // candidates are grid boundaries (midpoints when unset).
        if let Some(p) = self.grid_for(j) {
            if p > 0.0 {
                let q = 1.0 / p;
                for t in pairs.iter_mut() {
                    t.0 = (t.0 * q).round() / q;
                }
            }
        }
        let yws_sorted: Vec<(f64, f64)> = pairs.iter().map(|&(_, y, w)| (y, w)).collect();

        // Missing rows: one extra group, scored in both directions.
        let mut mn = 0usize;
        let mut missing_yws: Vec<(f64, f64)> = Vec::new();
        let (mut msw, mut mswy, mut mswy2) = (0.0, 0.0, 0.0);
        for &r in rows {
            if self.cols[j][r].is_none() {
                mn += 1;
                let (y, w) = (self.y[r], self.w[r]);
                missing_yws.push((y, w));
                msw += w;
                mswy += w * y;
                mswy2 += w * y * y;
            }
        }
        let mut tn = 0usize;
        let (mut tsw, mut tswy, mut tswy2) = (0.0, 0.0, 0.0);
        for &(_, y, w) in &pairs {
            tn += 1;
            tsw += w;
            tswy += w * y;
            tswy2 += w * y * y;
        }

        fn sse(sw: f64, swy: f64, swy2: f64) -> f64 {
            if sw <= 0.0 {
                0.0
            } else {
                swy2 - swy * swy / sw
            }
        }
        let parent_sse = sse(tsw, tswy, tswy2) + sse(msw, mswy, mswy2);
        st.group_ns += t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();

        let mut ln = 0usize;
        let (mut lsw, mut lswy, mut lswy2) = (0.0, 0.0, 0.0);
        let mut best: Option<(f64, Split)> = None;
        for k in 0..pairs.len() {
            let (v, y, w) = pairs[k];
            ln += 1;
            lsw += w;
            lswy += w * y;
            lswy2 += w * y * y;
            let is_boundary = k + 1 < pairs.len() && pairs[k + 1].0 != v;
            if !is_boundary {
                continue;
            }
            // Grid mode: thresholds are grid values (compare rounds to the
            // same grid, see `goes_left`); exact mode keeps midpoints.
            let threshold = match self.grid_for(j) {
                Some(_) => v,
                None => (v + pairs[k + 1].0) / 2.0,
            };
            let rn = tn - ln;
            let (rsw, rswy, rswy2) = (tsw - lsw, tswy - lswy, tswy2 - lswy2);
            st.candidates += 1;
            for missing_left in [true, false] {
                let (aln, alsw, alswy, alswy2) = if missing_left {
                    (ln + mn, lsw + msw, lswy + mswy, lswy2 + mswy2)
                } else {
                    (ln, lsw, lswy, lswy2)
                };
                let (arn, arsw, arswy, arswy2) = if missing_left {
                    (rn, rsw, rswy, rswy2)
                } else {
                    (rn + mn, rsw + msw, rswy + mswy, rswy2 + mswy2)
                };
                if (aln as f64) < m || (arn as f64) < m {
                    continue;
                }
                // grf-style alpha: each child carries at least alpha of the
                // node's total weight.
                if self.params.alpha > 0.0 && (alsw < self.params.alpha * (tsw + msw)
                    || arsw < self.params.alpha * (tsw + msw))
                {
                    continue;
                }
                if !self.mono_ok(j, alswy / alsw.max(1e-300), arswy / arsw.max(1e-300)) {
                    continue;
                }
                let gain = match self.params.gain {
                    GainFn::Deviance if self.params.leaf_family != crate::family::LeafFamily::Gaussian => {
                        self.constant_family_cost(tsw + msw, tswy + mswy)
                            - self.constant_family_cost(alsw, alswy)
                            - self.constant_family_cost(arsw, arswy)
                    }
                    GainFn::Sse | GainFn::Deviance => {
                        parent_sse - (sse(alsw, alswy, alswy2) + sse(arsw, arswy, arswy2))
                    }
                    ref gfn => {
                        // Lazy (y, w) slices over the value-sorted buffer;
                        // missing rows appended to the side they route to
                        // (allocated only when the column has missing).
                        if missing_yws.is_empty() {
                            gain_of(gfn, &yws_sorted[..k + 1], &yws_sorted[k + 1..])
                        } else {
                            let mut ls: Vec<(f64, f64)> = yws_sorted[..k + 1].to_vec();
                            let mut rs: Vec<(f64, f64)> = yws_sorted[k + 1..].to_vec();
                            if missing_left {
                                ls.extend_from_slice(&missing_yws);
                            } else {
                                rs.extend_from_slice(&missing_yws);
                            }
                            gain_of(gfn, &ls, &rs)
                        }
                    }
                };
                let gain = if self.params.imbalance_penalty > 0.0 {
                    gain - self.params.imbalance_penalty * (1.0 / aln as f64 + 1.0 / arn as f64)
                } else {
                    gain
                };
                if gain <= self.params.min_gain {
                    continue;
                }
                if best.as_ref().map(|(g, ..)| beats(gain, *g)).unwrap_or(true) {
                    best = Some((
                        gain,
                        Split {
                            col: j,
                            threshold,
                            missing_left,
                            gain,
                            cat: None,
                            sur: None,
                        },
                    ));
                }
            }
        }
        st.scan_ns += t1.elapsed().as_nanos() as u64;
        (best, st)
    }

    /// Linear-leaf scan — LightGBM's histogram shape: one weighted Gram per
    /// distinct value, running left/right aggregates, per-candidate solve on
    /// preallocated scratch (no allocation in the loop).
    fn scan_column_linear(
        &self,
        rows: &[usize],
        j: usize,
        path: &[usize],
        pre: Option<&ParentPre>,
    ) -> (Option<(f64, Split)>, ScanStats) {
        let mut st = ScanStats::default();
        let m = self.min_child(rows.len());
        let t0 = Instant::now();

        let chosen = self.chosen_cols(path);
        let dim = 1 + 2 * chosen.len();
        let mut row_buf = vec![0.0; dim];

        let mut pairs: Vec<(f64, usize)> = rows
            .iter()
            .filter_map(|&r| self.cols[j][r].map(|v| (v, r)))
            .collect();
        if pairs.len() < 2 {
            return (None, st);
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN in predictors"));
        // split_precision grid (see scan_column_mean).
        if let Some(p) = self.grid_for(j) {
            if p > 0.0 {
                let q = 1.0 / p;
                for t in pairs.iter_mut() {
                    t.0 = (t.0 * q).round() / q;
                }
            }
        }

        // Groups carry their row range into `pairs` too: the Grams move
        // across the cut in whole groups (exact historical arithmetic)
        // while the maintained factors follow the same rows one rank-1
        // step at a time.
        let mut groups: Vec<(f64, Gram, usize, usize)> = Vec::new();
        for (idx, &(v, r)) in pairs.iter().enumerate() {
            design_row(&mut row_buf, self.cols, r, &chosen, true);
            match groups.last_mut() {
                Some((gv, g, _, e)) if *gv == v => {
                    g.add_row(&row_buf, self.y[r], self.w[r]);
                    *e = idx + 1;
                }
                _ => {
                    let mut g = Gram::new(dim);
                    g.add_row(&row_buf, self.y[r], self.w[r]);
                    groups.push((v, g, idx, idx + 1));
                }
            }
        }
        let mut miss = Gram::new(dim);
        for &r in rows {
            if self.cols[j][r].is_none() {
                design_row(&mut row_buf, self.cols, r, &chosen, true);
                miss.add_row(&row_buf, self.y[r], self.w[r]);
            }
        }

        // Parent score + the node's mean design row (the monotone gate's
        // evaluation point): column-independent under the `all` leaf
        // policy and hoisted to one call per node by `best_split`.
        let (parent_sse, mut mean_row) = match pre {
            Some(p) => (p.parent_sse, p.mean_row.clone()),
            None => self.parent_stats(rows, &chosen, dim),
        };
        let mut scratch = SolveScratch::new(dim);
        let jslot = 1 + chosen.iter().position(|&c| c == j).unwrap();
        let mut gbuf0 = Gram::new(dim);
        let mut gbuf1 = Gram::new(dim);
        st.group_ns += t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();

        let mut total = Gram::new(dim);
        for (_, g, ..) in &groups {
            total.add(g);
        }
        let total_ws = total.ws;
        let has_miss = miss.n > 0;
        let miss_extra = if has_miss { Some(miss.clone()) } else { None };
        let mut left = SideTrack::new(Gram::new(dim), miss_extra.clone(), self.params.l2);
        let mut right = SideTrack::new(total, miss_extra, self.params.l2);
        let mut best: Option<(f64, Split)> = None;

        for gi in 0..groups.len() {
            let (v, g, start, end) = &groups[gi];
            // A group wider than ~dim/3 rows is cheaper to re-factor than
            // to fold row by row (rank-1 cost is linear in the group, the
            // refactor is not): drop the maintained factors there and let
            // solves take the exact path until a finer group re-materializes
            // them. Smaller groups fold in one rank-1 step per row.
            let fine = (*end - *start) * 3 <= dim;
            if fine {
                left.ensure();
                right.ensure();
            } else {
                left.drop_factors();
                right.drop_factors();
            }
            left.gram.add(g);
            right.gram.sub(g);
            if fine {
                for &(_, r) in &pairs[*start..*end] {
                    design_row(&mut row_buf, self.cols, r, &chosen, true);
                    let w = self.w[r];
                    left.move_row(&row_buf, w, 1);
                    right.move_row(&row_buf, w, -1);
                }
            }
            if gi + 1 >= groups.len() || left.gram.n < m || right.gram.n < m {
                continue;
            }
            let threshold = match self.grid_for(j) {
                Some(_) => *v,
                None => (v + groups[gi + 1].0) / 2.0,
            };
            st.candidates += 1;
            // With no missing rows both routing variants are identical and
            // the first-found tie-break keeps `missing_left = true`, so the
            // false variant is exactly redundant.
            let variants: &[bool] = if has_miss { &[true, false] } else { &[true] };
            for &missing_left in variants {
                let left_n = left.gram.n + if missing_left { miss.n } else { 0 };
                let right_n = right.gram.n + if missing_left { 0 } else { miss.n };
                if left_n < m || right_n < m {
                    continue;
                }
                let (wl, wr) = if missing_left {
                    (left.gram.ws + miss.ws, right.gram.ws)
                } else {
                    (left.gram.ws, right.gram.ws + miss.ws)
                };
                // grf-style alpha: each child carries at least alpha of the
                // node's total weight.
                if self.params.alpha > 0.0
                    && (wl < self.params.alpha * total_ws || wr < self.params.alpha * total_ws)
                {
                    continue;
                }
                let (ml, mr) = if missing_left {
                    (
                        (left.gram.wy + miss.wy) / wl.max(1e-300),
                        right.gram.wy / wr.max(1e-300),
                    )
                } else {
                    (
                        left.gram.wy / wl.max(1e-300),
                        (right.gram.wy + miss.wy) / wr.max(1e-300),
                    )
                };
                if !self.mono_ok(j, ml, mr) {
                    continue;
                }
                if self.params.monotone.get(j).copied().unwrap_or(0) != 0 {
                    gbuf0.clone_from(&left.gram);
                    gbuf1.clone_from(&right.gram);
                    if missing_left {
                        gbuf0.add(&miss);
                    } else {
                        gbuf1.add(&miss);
                    }
                    let ml_mod = fit_linear(
                        &gbuf0,
                        &chosen,
                        true,
                        &self.params.monotone,
                        self.params.l1,
                        self.params.l2,
                    );
                    let mr_mod = fit_linear(
                        &gbuf1,
                        &chosen,
                        true,
                        &self.params.monotone,
                        self.params.l1,
                        self.params.l2,
                    );
                    mean_row[jslot] = threshold;
                    let fl = ml_mod.eval_design(&mean_row);
                    let fr = mr_mod.eval_design(&mean_row);
                    if (fl - fr) * self.params.monotone[j] as f64 > 1e-9 {
                        continue;
                    }
                }
                let (sl, sr) = if missing_left {
                    (
                        left.sse(true, self.params.l1, &mut scratch),
                        right.sse(false, self.params.l1, &mut scratch),
                    )
                } else {
                    (
                        left.sse(false, self.params.l1, &mut scratch),
                        right.sse(true, self.params.l1, &mut scratch),
                    )
                };
                let gain = parent_sse - (sl + sr);
                let gain = if self.params.imbalance_penalty > 0.0 {
                    gain - self.params.imbalance_penalty
                        * (1.0 / left_n as f64 + 1.0 / right_n as f64)
                } else {
                    gain
                };
                if gain <= self.params.min_gain {
                    continue;
                }
                if best.as_ref().map(|(g, ..)| beats(gain, *g)).unwrap_or(true) {
                    best = Some((
                        gain,
                        Split {
                            col: j,
                            threshold,
                            missing_left,
                            gain,
                            cat: None,
                            sur: None,
                        },
                    ));
                }
            }
        }
        st.scan_ns += t1.elapsed().as_nanos() as u64;
        (best, st)
    }

    fn fitted_family_score(&self, rows: &[usize], leaf: &crate::node_stats::LeafModel) -> f64 {
        if matches!(self.params.gain, GainFn::Deviance) {
            self.leaf_deviance(rows, leaf)
        } else {
            rows.iter().map(|&r| {
                self.w[r] * (self.y[r] - leaf.predict_columns(self.cols, r)).powi(2)
            }).sum()
        }
    }

    /// Constant-family likelihood cost without the saturated-target term,
    /// which cancels between parent and children. O(1) per numeric cut.
    fn constant_family_cost(&self, sw: f64, sy: f64) -> f64 {
        use crate::family::LeafFamily;
        if sw <= 0.0 { return 0.0; }
        let xlog = |mass: f64, mean: f64| {
            if mass <= 0.0 { 0.0 } else { mass * mean.ln() }
        };
        let mean = sy / sw;
        match self.params.leaf_family {
            LeafFamily::Binomial => -2.0 * (
                xlog(sy, mean) + xlog((sw - sy).max(0.0), 1.0 - mean)
            ),
            LeafFamily::Poisson => 2.0 * (sy - xlog(sy, mean)),
            LeafFamily::Gaussian => unreachable!("Gaussian uses its SSE scan"),
        }
    }

    fn partition_gain(&self, left: &[(f64, f64)], right: &[(f64, f64)]) -> f64 {
        if matches!(self.params.gain, GainFn::Deviance)
            && self.params.leaf_family != crate::family::LeafFamily::Gaussian
        {
            let sums = |rs: &[(f64, f64)]| rs.iter().fold(
                (0.0, 0.0), |(sw, sy), &(y, w)| (sw + w, sy + w * y));
            let (wl, yl) = sums(left);
            let (wr, yr) = sums(right);
            self.constant_family_cost(wl + wr, yl + yr)
                - self.constant_family_cost(wl, yl) - self.constant_family_cost(wr, yr)
        } else {
            gain_of(&self.params.gain, left, right)
        }
    }
    /// Distributional candidates refit the two child GLMs. Unlike Gaussian
    /// least squares, their Hessians depend on the current fitted means:
    /// one moving Gram cannot represent every IRLS iteration.
    fn scan_column_family(
        &self,
        rows: &[usize],
        j: usize,
        path: &[usize],
        pre: Option<&ParentPre>,
    ) -> (Option<(f64, Split, Option<Vec<f64>>)>, ScanStats) {
        let mut st = ScanStats::default();
        let t0 = Instant::now();
        let categorical = self.col_type(j) == ColType::Categorical;
        let grid = self.grid_for(j).filter(|p| *p > 0.0);
        let mut observed: Vec<(f64, usize)> = rows.iter().filter_map(|&r| {
            self.cols[j][r].map(|v| {
                let v = if categorical { v } else {
                    grid.map(|p| { let q = 1.0 / p; (v * q).round() / q }).unwrap_or(v)
                };
                (v, r)
            })
        }).collect();
        observed.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
        for (v, r) in observed {
            match groups.last_mut() {
                Some((last, rs)) if *last == v => rs.push(r),
                _ => groups.push((v, vec![r])),
            }
        }
        if groups.len() < 2 {
            return (None, st);
        }
        if categorical {
            let target_mean = |rs: &[usize]| {
                let sw: f64 = rs.iter().map(|&r| self.w[r]).sum();
                rs.iter().map(|&r| self.w[r] * self.y[r]).sum::<f64>() / sw
            };
            match &self.params.cat_order {
                CatOrder::FrequencyDesc => groups.sort_by(|a, b| {
                    b.1.len().cmp(&a.1.len()).then(a.0.total_cmp(&b.0))
                }),
                CatOrder::TargetMean => groups.sort_by(|a, b| {
                    target_mean(&b.1).total_cmp(&target_mean(&a.1))
                        .then(a.0.total_cmp(&b.0))
                }),
                CatOrder::Explicit(order) => groups.sort_by(|a, b| {
                    let rank = |v| order.iter().position(|&o| o == v).unwrap_or(usize::MAX);
                    rank(a.0).cmp(&rank(b.0))
                        .then(b.1.len().cmp(&a.1.len())).then(a.0.total_cmp(&b.0))
                }),
            }
        }
        let missing: Vec<usize> = rows.iter().copied()
            .filter(|&r| self.cols[j][r].is_none()).collect();
        let m = self.min_child(rows.len());
        let total_w: f64 = rows.iter().map(|&r| self.w[r]).sum();
        let fitted_score = matches!(self.params.gain, GainFn::Deviance | GainFn::Sse);
        let score = |rs: &[usize], leaf: &crate::node_stats::LeafModel| {
            self.fitted_family_score(rs, leaf)
        };
        let parent_score = if !fitted_score { 0.0 } else if let Some(p) = pre {
            p.parent_sse
        } else {
            score(rows, &self.make_leaf_for_scan(rows, path))
        };
        let mut left = Vec::with_capacity(rows.len());
        let mut right = Vec::with_capacity(rows.len());
        let mut ls = Vec::with_capacity(rows.len());
        let mut rs = Vec::with_capacity(rows.len());
        let mut boundary: Vec<Option<f64>> = self.cols.iter().map(|col| {
            let mut sw = 0.0;
            let mut sum = 0.0;
            for &r in rows {
                if let Some(v) = col[r] { sw += self.w[r]; sum += self.w[r] * v; }
            }
            (sw > 0.0).then(|| sum / sw)
        }).collect();
        st.group_ns = t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();
        let mut best: Option<(f64, Split, Option<Vec<f64>>)> = None;
        let k = groups.len();
        let cumulative = !categorical || !matches!(self.params.cat_search, CatSearch::SingleVsRest);
        let singles = categorical && !matches!(self.params.cat_search, CatSearch::Cumulative);
        let n_cumulative = if cumulative { k - 1 } else { 0 };
        for ci in 0..n_cumulative + if singles { k } else { 0 } {
            let single = ci >= n_cumulative;
            let cut = if single { ci - n_cumulative } else { ci };
            let in_left = |gi: usize| if single { gi == cut } else { gi <= cut };
            let threshold = if categorical { 0.0 } else {
                if grid.is_some() { groups[cut].0 }
                else { groups[cut].0 / 2.0 + groups[cut + 1].0 / 2.0 }
            };
            st.candidates += 1;
            let variants: &[bool] = if missing.is_empty() { &[true] } else { &[true, false] };
            for &missing_left in variants {
                left.clear();
                right.clear();
                for (gi, (_, gr)) in groups.iter().enumerate() {
                    if in_left(gi) { left.extend_from_slice(gr); }
                    else { right.extend_from_slice(gr); }
                }
                if missing_left { left.extend_from_slice(&missing); }
                else { right.extend_from_slice(&missing); }
                if left.len() < m || right.len() < m { continue; }
                let sw = |r: &[usize]| r.iter().map(|&i| self.w[i]).sum::<f64>();
                let (wl, wr) = (sw(&left), sw(&right));
                if wl < self.params.alpha * total_w || wr < self.params.alpha * total_w {
                    continue;
                }
                let mean = |r: &[usize], w: f64| {
                    r.iter().map(|&i| self.w[i] * self.y[i]).sum::<f64>() / w
                };
                if !categorical && !self.mono_ok(j, mean(&left, wl), mean(&right, wr)) {
                    continue;
                }
                let constrained = !categorical && self.params.monotone.get(j)
                    .copied().unwrap_or(0) != 0;
                let fits = (fitted_score || constrained).then(|| (
                    self.make_leaf_for_scan(&left, path),
                    self.make_leaf_for_scan(&right, path),
                ));
                if constrained {
                    boundary[j] = Some(threshold);
                    let (l, r) = fits.as_ref().unwrap();
                    if (l.predict(&boundary) - r.predict(&boundary))
                        * self.params.monotone[j] as f64 > 1e-9 { continue; }
                }
                let gain = if fitted_score {
                    let (l, r) = fits.as_ref().unwrap();
                    parent_score - score(&left, l) - score(&right, r)
                } else {
                    ls.clear();
                    rs.clear();
                    ls.extend(left.iter().map(|&i| (self.y[i], self.w[i])));
                    rs.extend(right.iter().map(|&i| (self.y[i], self.w[i])));
                    gain_of(&self.params.gain, &ls, &rs)
                };
                let gain = gain - self.params.imbalance_penalty
                    * (1.0 / left.len() as f64 + 1.0 / right.len() as f64);
                if !gain.is_finite() || gain <= self.params.min_gain { continue; }
                if best.as_ref().map(|(g, ..)| beats(gain, *g)).unwrap_or(true) {
                    let part = categorical.then(|| groups.iter().enumerate()
                        .filter(|(gi, _)| in_left(*gi)).map(|(_, g)| g.0).collect());
                    best = Some((gain, Split { col: j, threshold, missing_left,
                        gain, cat: None, sur: None }, part));
                }
            }
            if categorical && matches!(self.params.cat_search, CatSearch::SingleVsRest)
                && best.is_some() { break; }
        }
        st.scan_ns = t1.elapsed().as_nanos() as u64;
        (best, st)
    }

}
