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

/// Oblique hyperplane linear combination projection for SPORF (Tomita et al. 2020).
/// Defines Z = sum_{k=0}^{d-1} weights[k] * X_{cols[k]}.
#[derive(Clone, Debug, PartialEq)]
pub struct ObliqueProjection {
    pub cols: Vec<usize>,
    pub weights: Vec<f64>,
}

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

/// Split evaluation strategy: fast scalar variance reduction (M5/Cubist style, default)
/// vs historical exhaustive candidate-level multivariate regression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitEval {
    /// Fast scalar variance/deviance reduction (O(1) per threshold, M5/Cubist oracle).
    Fast,
    /// Exhaustive candidate-level linear Gram regression (historical research mode).
    Exhaustive,
}

/// Work of one column scan, merged into the node profile.
#[derive(Clone, Copy, Debug, Default)]
struct ScanStats {
    group_ns: u64,
    scan_ns: u64,
    candidates: usize,
}

/// Reusable per-thread scratch buffers to eliminate heap allocation during column scans.
#[derive(Clone, Debug, Default)]
pub(crate) struct SplitScratch {
    pub pairs: Vec<(f64, f64, f64, f64)>,
    pub pairs_yw: Vec<(f64, f64)>,
    pub missing_yws: Vec<(f64, f64)>,
    pub level_moments: Vec<(f64, usize, f64, f64, f64, f64, f64, f64, f64, f64)>, // (level, n, sw, swy, swy2, sw_prot, s1_y1, s0_y1, s1_y0, s0_y0)
    pub pairs_causal: Vec<(f64, f64, f64, f64)>,        // (v, y, w, treat)
    pub level_causal: Vec<(f64, usize, f64, f64, f64, f64)>, // (level, n, sw_t, swy_t, sw_c, swy_c)
    pub h_parent: Vec<f64>,
    pub h_left: Vec<f64>,
    pub h_miss: Vec<f64>,
    pub h_cand_left: Vec<f64>,
    pub h_cand_right: Vec<f64>,
}

impl<'a> Grower<'a> {
    /// Interaction constraint (LightGBM-style): a path may only use columns
    /// from one interaction group; ungrouped columns never interact.
    pub(crate) fn interact_ok(&self, path: &[usize], j: usize) -> bool {
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
        let m = if !self.params.tree_monotone.is_empty() {
            self.params.tree_monotone.get(j).copied().unwrap_or(0)
        } else if !self.params.monotone.is_empty() {
            self.params.monotone.get(j).copied().unwrap_or(0)
        } else {
            0
        };
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

        let (mut node_sw, mut node_swy, mut node_swy2) = (0.0, 0.0, 0.0);
        for &r in rows {
            let (y, w) = (self.y[r], self.w[r]);
            node_sw += w;
            node_swy += w * y;
            node_swy2 += w * y * y;
        }
        let node_moments = (node_sw, node_swy, node_swy2);

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
                match &self.params.col_weights {
                    Some(cw) => {
                        let mut weighted_cols: Vec<(f64, usize)> = cols
                            .iter()
                            .map(|&j| {
                                let w = cw.get(j).copied().unwrap_or(1.0).max(1e-6);
                                rng ^= rng << 13;
                                rng ^= rng >> 7;
                                rng ^= rng << 17;
                                let u = ((rng >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
                                let key = u.powf(1.0 / w);
                                (key, j)
                            })
                            .collect();
                        weighted_cols.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                        cols = weighted_cols.into_iter().take(k).map(|(_, j)| j).collect();
                        cols.sort_unstable();
                    }
                    None => {
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
            }
        }
        self.rng_counter += 1;

        // Parent score + mean design row are column-independent under the
        // `all` leaf-column policy: hoisted from per-column to per-node (exhaustive mode only).
        let parent_pre: Option<ParentPre> = match self.params.split_eval {
            SplitEval::Exhaustive => match self.params.leaf_cols {
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
            },
            SplitEval::Fast => None,
        };

        let mut scratch = SplitScratch::default();
        if work >= PAR_MIN_WORK {
            use rayon::prelude::*;
            let mut out: Vec<(usize, Option<(f64, Split, Option<Vec<f64>>)>, ScanStats)> = cols
                .par_iter()
                .map_init(
                    SplitScratch::default,
                    |sc, &j| {
                        let cp = if self.params.split_eval == SplitEval::Exhaustive {
                            let mut p = path.to_vec();
                            p.push(j);
                            p
                        } else {
                            Vec::new()
                        };
                        let (found, st) = self.scan_column(
                            rows,
                            j,
                            &cp,
                            parent_pre.as_ref(),
                            sc,
                            node_moments,
                        );
                        (j, found, st)
                    },
                )
                .collect();
            out.sort_by_key(|(j, _, _)| *j); // deterministic order
            candidates = out;
        } else {
            for &j in &cols {
                let cp = if self.params.split_eval == SplitEval::Exhaustive {
                    child_path.push(j);
                    child_path.clone()
                } else {
                    Vec::new()
                };
                let (found, st) = self.scan_column(
                    rows,
                    j,
                    &cp,
                    parent_pre.as_ref(),
                    &mut scratch,
                    node_moments,
                );
                if self.params.split_eval == SplitEval::Exhaustive {
                    child_path.pop();
                }
                candidates.push((j, found, st));
            }
        }

        let mut best: Option<(f64, Split, Option<Vec<f64>>, usize, f64)> = None;
        for (j, found, st) in candidates {
            prof.cols += 1;
            prof.group_ns += st.group_ns;
            prof.scan_ns += st.scan_ns;
            prof.candidates += st.candidates;
            if let Some((mut gain, sp, part)) = found {
                if let Some(cw) = &self.params.col_weights {
                    if let Some(&w) = cw.get(j) {
                        gain *= w;
                    }
                }
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

    /// SPORF: Sparse Projection Oblique Randomer Forest split search (Tomita et al. 2020).
    /// Generates random linear combinations of `sporf_d` features: Z = sum_{k=0}^{d-1} w_k * X_{cols[k]}
    /// with random signs w_k in {-1, +1}, evaluates 1D cutpoint scans on Z, and returns the best
    /// oblique hyperplane partition if its gain exceeds the threshold.
    pub(crate) fn sporf_split(
        &mut self,
        rows: &[usize],
        path: &[usize],
        prof: &mut NodeProfile,
    ) -> Option<(Split, Vec<usize>, Vec<usize>)> {
        if rows.len() < self.min_split() {
            return None;
        }
        let d = self.params.sporf_d;
        let num_cols: Vec<usize> = self.active
            .iter()
            .copied()
            .filter(|&j| self.interact_ok(path, j))
            .filter(|&j| matches!(self.col_type(j), ColType::Float | ColType::Integer))
            .collect();

        if num_cols.len() < d {
            return None;
        }

        let num_projections = self.params.sporf_projections.max(1);
        let min_leaf = self.min_child(rows.len());
        let mut best_gain = self.params.min_gain;
        let mut best_split: Option<Split> = None;
        let mut best_proj: Option<ObliqueProjection> = None;

        for _ in 0..num_projections {
            prof.candidates += 1;
            // Deterministic pseudo-random selection of d columns and signs
            let mut sampled_cols = Vec::with_capacity(d);
            let mut sampled_weights = Vec::with_capacity(d);
            let mut available = num_cols.clone();
            for _ in 0..d {
                self.rng_counter = self.rng_counter.wrapping_add(1);
                let mut rng = self.params.seed
                    .wrapping_add(self.rng_counter.wrapping_mul(6364136223846793005))
                    ^ 0xA076_1D64_78BD_642F;
                rng ^= rng >> 33;
                rng = rng.wrapping_mul(0xff51afd7ed558ccd);
                rng ^= rng >> 33;
                let idx = (rng as usize) % available.len();
                let col = available.swap_remove(idx);
                let sign = if (rng >> 7) & 1 == 0 { 1.0 } else { -1.0 };
                sampled_cols.push(col);
                sampled_weights.push(sign);
            }

            // Project rows: z_r = sum_{k=0}^{d-1} w_k * X_{cols[k], r}
            let mut vals: Vec<(f64, usize)> = Vec::with_capacity(rows.len());
            for (local_idx, &r) in rows.iter().enumerate() {
                let mut z = 0.0;
                let mut ok = true;
                for (&c, &w) in sampled_cols.iter().zip(sampled_weights.iter()) {
                    match self.cols[c][r] {
                        Some(v) => z += w * v,
                        None => { ok = false; break; }
                    }
                }
                if ok {
                    vals.push((z, local_idx));
                }
            }

            if vals.len() < 2 * min_leaf {
                continue;
            }

            vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut w_l = 0.0;
            let mut wy_l = 0.0;
            let mut wy2_l = 0.0;
            let (mut total_w, mut total_wy, mut total_wy2) = (0.0, 0.0, 0.0);
            for &(_, local_idx) in &vals {
                let r = rows[local_idx];
                let (y, wt) = (self.y[r], self.w[r]);
                total_w += wt;
                total_wy += wt * y;
                total_wy2 += wt * y * y;
            }

            let parent_sse = total_wy2 - (total_wy * total_wy) / total_w.max(1e-12);

            for m in 0..(vals.len() - 1) {
                let (z_curr, local_idx) = vals[m];
                let r = rows[local_idx];
                let (y, wt) = (self.y[r], self.w[r]);
                w_l += wt;
                wy_l += wt * y;
                wy2_l += wt * y * y;

                let z_next = vals[m + 1].0;
                let n_l = m + 1;
                let n_r = vals.len() - n_l;

                if z_curr < z_next && n_l >= min_leaf && n_r >= min_leaf {
                    let w_r = total_w - w_l;
                    let wy_r = total_wy - wy_l;
                    let wy2_r = total_wy2 - wy2_l;

                    let sse_l = wy2_l - (wy_l * wy_l) / w_l;
                    let sse_r = wy2_r - (wy_r * wy_r) / w_r;
                    let mut gain = parent_sse - sse_l - sse_r;

                    if self.params.imbalance_penalty > 0.0 {
                        gain -= self.params.imbalance_penalty * (1.0 / n_l as f64 + 1.0 / n_r as f64);
                    }

                    if gain > best_gain {
                        best_gain = gain;
                        let threshold = (z_curr + z_next) / 2.0;
                        best_split = Some(Split {
                            col: sampled_cols[0],
                            threshold,
                            missing_left: true,
                            gain,
                            cat: None,
                            sur: None,
                            oblique: None,
                        });
                        best_proj = Some(ObliqueProjection {
                            cols: sampled_cols.clone(),
                            weights: sampled_weights.clone(),
                        });
                    }
                }
            }
        }

        if let (Some(mut sp), Some(proj)) = (best_split, best_proj) {
            let ob_idx = self.oblique_parts.len();
            self.oblique_parts.push(proj);
            sp.oblique = Some(ob_idx);
            let (lv, rv) = self.partition_rows(rows, &sp);
            Some((sp, lv, rv))
        } else {
            None
        }
    }

    /// Route `rows` through a split (missing values follow
    /// `missing_left`). Shared by the structure partition and the honesty
    /// estimation partition.
    pub(crate) fn partition_rows(&self, rows: &[usize], sp: &Split) -> (Vec<usize>, Vec<usize>) {
        let (mut lv, mut rv) = (Vec::new(), Vec::new());
        for &r in rows {
            let go_left = if let Some(ob_idx) = sp.oblique {
                let proj = &self.oblique_parts[ob_idx];
                let mut sum = 0.0;
                let mut has_na = false;
                for (&c, &w) in proj.cols.iter().zip(proj.weights.iter()) {
                    match self.cols[c][r] {
                        Some(v) => sum += w * v,
                        None => { has_na = true; break; }
                    }
                }
                if has_na {
                    sp.missing_left
                } else {
                    sum <= sp.threshold
                }
            } else {
                match self.cols[sp.col][r] {
                    Some(v) => self.goes_left(sp, v),
                    None => sp.missing_left,
                }
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
        scratch: &mut SplitScratch,
        node_moments: (f64, f64, f64),
    ) -> (Option<(f64, Split, Option<Vec<f64>>)>, ScanStats) {
        if matches!(self.params.gain, GainFn::Causal) {
            return if self.col_type(j) == ColType::Categorical {
                self.scan_column_categorical_causal(rows, j, scratch)
            } else {
                let (hit, st) = self.scan_column_causal(rows, j, scratch);
                (hit.map(|(g, s)| (g, s, None)), st)
            };
        }
        let (hit, st) = match self.params.split_eval {
            SplitEval::Exhaustive => {
                if self.params.leaf_family != crate::family::LeafFamily::Gaussian
                    && self.params.leaf == LeafKind::Linear
                {
                    return self.scan_column_family(rows, j, path, pre);
                }
                if self.col_type(j) == ColType::Categorical {
                    return self.scan_column_categorical(rows, j, scratch);
                }
                match self.params.leaf {
                    LeafKind::Linear => self.scan_column_linear(rows, j, path, pre),
                    _ => self.scan_column_mean(rows, j, scratch, node_moments),
                }
            }
            SplitEval::Fast => {
                if self.col_type(j) == ColType::Categorical {
                    return self.scan_column_categorical(rows, j, scratch);
                }
                self.scan_column_mean(rows, j, scratch, node_moments)
            }
        };
        (hit.map(|(g, s)| (g, s, None)), st)
    }

    /// Column type for the split search: `params.dtypes` when set, else
    /// float.
    pub(crate) fn col_type(&self, j: usize) -> ColType {
        self.params.dtypes.get(j).copied().unwrap_or(ColType::Float)
    }

    /// Precision grid for a column's scan: integer handlers scan exact
    /// unique values (no grid); float handlers use `split_precision`.
    pub(crate) fn grid_for(&self, j: usize) -> Option<f64> {
        grid_of(&self.params.dtypes, self.params.split_precision, j)
    }

    /// Minimum node rows to attempt a split (ctree `minsplit`); defaults to
    /// the historical `2 * min_rows` gate.
    pub(crate) fn min_split(&self) -> usize {
        self.params.minsplit.unwrap_or(2 * self.params.min_rows)
    }

    /// Minimum rows per child: the `minbucket` floor and the `minprob`
    /// share of the node (ctree trio).
    pub(crate) fn min_child(&self, node_rows: usize) -> usize {
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
        scratch: &mut SplitScratch,
    ) -> (Option<(f64, Split, Option<Vec<f64>>)>, ScanStats) {
        let mut st = ScanStats::default();
        let t0 = Instant::now();

        let is_standard_gain = matches!(self.params.gain, GainFn::Sse | GainFn::Deviance);
        if is_standard_gain {
            let prot_col_opt = self.params.protected_col;
            let mut by_level: std::collections::HashMap<u64, (usize, f64, f64, f64, f64, f64, f64, f64, f64)> =
                std::collections::HashMap::new();
            let (mut mn, mut msw, mut mswy, mut mswy2, mut msw_prot) = (0usize, 0.0, 0.0, 0.0, 0.0);
            let (mut msw_s1_y1, mut msw_s0_y1, mut msw_s1_y0, mut msw_s0_y0) = (0.0, 0.0, 0.0, 0.0);
            for &r in rows {
                let (y, w) = (self.y[r], self.w[r]);
                let p_wt = if let Some(pc) = prot_col_opt {
                    if self.cols[pc][r].unwrap_or(0.0) >= 0.5 { w } else { 0.0 }
                } else {
                    0.0
                };
                let is_prot = p_wt > 0.0;
                let is_y1 = y >= 0.5;
                let s1_y1 = if is_prot && is_y1 { w } else { 0.0 };
                let s0_y1 = if !is_prot && is_y1 { w } else { 0.0 };
                let s1_y0 = if is_prot && !is_y1 { w } else { 0.0 };
                let s0_y0 = if !is_prot && !is_y1 { w } else { 0.0 };

                match self.cols[j][r] {
                    Some(v) => {
                        let e = by_level.entry(v.to_bits()).or_insert((0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
                        e.0 += 1;
                        e.1 += w;
                        e.2 += w * y;
                        e.3 += w * y * y;
                        e.4 += p_wt;
                        e.5 += s1_y1;
                        e.6 += s0_y1;
                        e.7 += s1_y0;
                        e.8 += s0_y0;
                    }
                    None => {
                        mn += 1;
                        msw += w;
                        mswy += w * y;
                        mswy2 += w * y * y;
                        msw_prot += p_wt;
                        msw_s1_y1 += s1_y1;
                        msw_s0_y1 += s0_y1;
                        msw_s1_y0 += s1_y0;
                        msw_s0_y0 += s0_y0;
                    }
                }
            }
            if by_level.len() < 2 {
                st.group_ns = t0.elapsed().as_nanos() as u64;
                return (None, st);
            }
            scratch.level_moments.clear();
            for (b, (n, sw, swy, swy2, sw_prot, s1_y1, s0_y1, s1_y0, s0_y0)) in by_level {
                scratch.level_moments.push((f64::from_bits(b), n, sw, swy, swy2, sw_prot, s1_y1, s0_y1, s1_y0, s0_y0));
            }

            // Level order (deterministic: ties by level code ascending).
            match &self.params.cat_order {
                CatOrder::FrequencyDesc => {
                    scratch.level_moments.sort_by(|a, b| {
                        b.1.cmp(&a.1)
                            .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    });
                }
                CatOrder::TargetMean => {
                    let wmean = |sw: f64, swy: f64| if sw <= 0.0 { 0.0 } else { swy / sw };
                    scratch.level_moments.sort_by(|a, b| {
                        wmean(b.2, b.3)
                            .partial_cmp(&wmean(a.2, a.3))
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    });
                }
                CatOrder::Explicit(order) => {
                    let rank = |v: f64| {
                        order.iter().position(|o| *o == v).unwrap_or(usize::MAX)
                    };
                    scratch.level_moments.sort_by(|a, b| {
                        rank(a.0)
                            .cmp(&rank(b.0))
                            .then(b.1.cmp(&a.1))
                            .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    });
                }
            }
            st.group_ns += t0.elapsed().as_nanos() as u64;
            let t1 = Instant::now();

            let k = scratch.level_moments.len();
            let m = self.min_child(rows.len()) as f64;
            let (mut tn, mut tsw, mut tswy, mut tswy2, mut tsw_prot) = (0usize, 0.0, 0.0, 0.0, 0.0);
            let (mut tsw_s1_y1, mut tsw_s0_y1, mut tsw_s1_y0, mut tsw_s0_y0) = (0.0, 0.0, 0.0, 0.0);
            for &(_, n, sw, swy, swy2, sw_prot, s1_y1, s0_y1, s1_y0, s0_y0) in scratch.level_moments.iter() {
                tn += n;
                tsw += sw;
                tswy += swy;
                tswy2 += swy2;
                tsw_prot += sw_prot;
                tsw_s1_y1 += s1_y1;
                tsw_s0_y1 += s0_y1;
                tsw_s1_y0 += s1_y0;
                tsw_s0_y0 += s0_y0;
            }
            let total_w = tsw + msw;

            fn sse(sw: f64, swy: f64, swy2: f64) -> f64 {
                if sw <= 0.0 { 0.0 } else { swy2 - swy * swy / sw }
            }
            let parent_sse = sse(tsw, tswy, tswy2) + sse(msw, mswy, mswy2);

            let mut best: Option<(f64, Vec<usize>, bool)> = None;

            let mut eval_split = |left_indices: &[usize], ln: usize, lsw: f64, lswy: f64, lswy2: f64, lsw_prot: f64, ls1_y1: f64, ls0_y1: f64, ls1_y0: f64, ls0_y0: f64, best: &mut Option<(f64, Vec<usize>, bool)>| -> bool {
                let rn = tn - ln;
                let (rsw, rswy, rswy2) = (tsw - lsw, tswy - lswy, tswy2 - lswy2);
                if (ln as f64) < m || (rn as f64) < m {
                    return false;
                }
                st.candidates += 1;
                let mut candidate_hit = false;
                for missing_left in [true, false] {
                    let (aln, alsw, alswy, alswy2, alsw_prot) = if missing_left {
                        (ln + mn, lsw + msw, lswy + mswy, lswy2 + mswy2, lsw_prot + msw_prot)
                    } else {
                        (ln, lsw, lswy, lswy2, lsw_prot)
                    };
                    let (arn, arsw, arswy, arswy2, arsw_prot) = if missing_left {
                        (rn, rsw, rswy, rswy2, tsw_prot - lsw_prot)
                    } else {
                        (rn + mn, rsw + msw, rswy + mswy, rswy2 + mswy2, tsw_prot + msw_prot - lsw_prot)
                    };
                    if (aln as f64) < m || (arn as f64) < m {
                        continue;
                    }
                    if self.params.alpha > 0.0
                        && (alsw < self.params.alpha * total_w || arsw < self.params.alpha * total_w)
                    {
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
                        _ => unreachable!(),
                    };
                    let gain = if self.params.imbalance_penalty > 0.0 {
                        gain - self.params.imbalance_penalty * (1.0 / aln as f64 + 1.0 / arn as f64)
                    } else {
                        gain
                    };
                    let gain = if self.params.fairness_penalty > 0.0 && self.params.protected_col.is_some() {
                        let disparity = match self.params.fairness_metric {
                            crate::model_tree::FairnessMetric::DemographicParity => {
                                let p_l = if alsw > 0.0 { alsw_prot / alsw } else { 0.0 };
                                let p_r = if arsw > 0.0 { arsw_prot / arsw } else { 0.0 };
                                (p_l - p_r).abs()
                            }
                            crate::model_tree::FairnessMetric::EqualizedOdds => {
                                let (alsw_s1_y1, alsw_s0_y1, alsw_s1_y0, alsw_s0_y0) = if missing_left {
                                    (ls1_y1 + msw_s1_y1, ls0_y1 + msw_s0_y1, ls1_y0 + msw_s1_y0, ls0_y0 + msw_s0_y0)
                                } else {
                                    (ls1_y1, ls0_y1, ls1_y0, ls0_y0)
                                };
                                let tot_s1_y1 = tsw_s1_y1 + msw_s1_y1;
                                let tot_s0_y1 = tsw_s0_y1 + msw_s0_y1;
                                let tot_s1_y0 = tsw_s1_y0 + msw_s1_y0;
                                let tot_s0_y0 = tsw_s0_y0 + msw_s0_y0;

                                let diff_tpr = if tot_s1_y1 > 0.0 && tot_s0_y1 > 0.0 {
                                    let tpr_1 = alsw_s1_y1 / tot_s1_y1;
                                    let tpr_0 = alsw_s0_y1 / tot_s0_y1;
                                    Some((tpr_1 - tpr_0).abs())
                                } else {
                                    None
                                };

                                let diff_fpr = if tot_s1_y0 > 0.0 && tot_s0_y0 > 0.0 {
                                    let fpr_1 = alsw_s1_y0 / tot_s1_y0;
                                    let fpr_0 = alsw_s0_y0 / tot_s0_y0;
                                    Some((fpr_1 - fpr_0).abs())
                                } else {
                                    None
                                };

                                match (diff_tpr, diff_fpr) {
                                    (Some(dt), Some(df)) => 0.5 * (dt + df),
                                    (Some(dt), None) => dt,
                                    (None, Some(df)) => df,
                                    (None, None) => 0.0,
                                }
                            }
                        };
                        (gain - self.params.fairness_penalty * parent_sse * disparity).max(0.0)
                    } else {
                        gain
                    };
                    if gain <= self.params.min_gain {
                        continue;
                    }
                    candidate_hit = true;
                    if best.as_ref().map(|(g, ..)| beats(gain, *g)).unwrap_or(true) {
                        *best = Some((gain, left_indices.to_vec(), missing_left));
                    }
                }
                candidate_hit
            };

            match self.params.cat_search {
                CatSearch::Cumulative => {
                    let (mut ln, mut lsw, mut lswy, mut lswy2, mut lsw_prot) = (0usize, 0.0, 0.0, 0.0, 0.0);
                    let (mut ls1_y1, mut ls0_y1, mut ls1_y0, mut ls0_y0) = (0.0, 0.0, 0.0, 0.0);
                    let mut left_indices = Vec::with_capacity(k);
                    for i in 1..k {
                        let (_, cn, csw, cswy, cswy2, csw_prot, cs1_y1, cs0_y1, cs1_y0, cs0_y0) = scratch.level_moments[i - 1];
                        ln += cn;
                        lsw += csw;
                        lswy += cswy;
                        lswy2 += cswy2;
                        lsw_prot += csw_prot;
                        ls1_y1 += cs1_y1;
                        ls0_y1 += cs0_y1;
                        ls1_y0 += cs1_y0;
                        ls0_y0 += cs0_y0;
                        left_indices.push(i - 1);
                        eval_split(&left_indices, ln, lsw, lswy, lswy2, lsw_prot, ls1_y1, ls0_y1, ls1_y0, ls0_y0, &mut best);
                    }
                }
                CatSearch::SingleVsRest => {
                    for i in 0..k {
                        let (_, ln, lsw, lswy, lswy2, lsw_prot, s1_y1, s0_y1, s1_y0, s0_y0) = scratch.level_moments[i];
                        let hit = eval_split(&[i], ln, lsw, lswy, lswy2, lsw_prot, s1_y1, s0_y1, s1_y0, s0_y0, &mut best);
                        if hit {
                            break;
                        }
                    }
                }
                CatSearch::Both => {
                    let (mut ln, mut lsw, mut lswy, mut lswy2, mut lsw_prot) = (0usize, 0.0, 0.0, 0.0, 0.0);
                    let (mut ls1_y1, mut ls0_y1, mut ls1_y0, mut ls0_y0) = (0.0, 0.0, 0.0, 0.0);
                    let mut left_indices = Vec::with_capacity(k);
                    for i in 1..k {
                        let (_, cn, csw, cswy, cswy2, csw_prot, cs1_y1, cs0_y1, cs1_y0, cs0_y0) = scratch.level_moments[i - 1];
                        ln += cn;
                        lsw += csw;
                        lswy += cswy;
                        lswy2 += cswy2;
                        lsw_prot += csw_prot;
                        ls1_y1 += cs1_y1;
                        ls0_y1 += cs0_y1;
                        ls1_y0 += cs1_y0;
                        ls0_y0 += cs0_y0;
                        left_indices.push(i - 1);
                        eval_split(&left_indices, ln, lsw, lswy, lswy2, lsw_prot, ls1_y1, ls0_y1, ls1_y0, ls0_y0, &mut best);
                    }
                    for i in 0..k {
                        let (_, ln, lsw, lswy, lswy2, lsw_prot, s1_y1, s0_y1, s1_y0, s0_y0) = scratch.level_moments[i];
                        eval_split(&[i], ln, lsw, lswy, lswy2, lsw_prot, s1_y1, s0_y1, s1_y0, s0_y0, &mut best);
                    }
                }
            }

            st.scan_ns += t1.elapsed().as_nanos() as u64;
            return (
                best.map(|(gain, left_indices, missing_left)| {
                    let left_levels: Vec<f64> = left_indices.iter().map(|&li| scratch.level_moments[li].0).collect();
                    (
                        gain,
                        Split {
                            col: j,
                            threshold: 0.0,
                            missing_left,
                            gain,
                            cat: None,
                            sur: None,
                            oblique: None,
                        },
                        Some(left_levels),
                    )
                }),
                st,
            );
        }

        let is_multiclass_gain = matches!(
            self.params.gain,
            GainFn::Gini | GainFn::Entropy | GainFn::GainRatio | GainFn::Twoing
        );
        let mut max_c = self.params.n_classes.unwrap_or(0);
        let mut all_int = is_multiclass_gain;
        if is_multiclass_gain {
            for &r in rows {
                let y = self.y[r];
                if y >= 0.0 && y < 256.0 && y.fract() == 0.0 {
                    max_c = max_c.max((y as usize) + 1);
                } else {
                    all_int = false;
                    break;
                }
            }
        }
        let n_classes_opt = if all_int && max_c >= 2 { Some(max_c) } else { None };

        if let Some(nc) = n_classes_opt {
            let mut by_level: std::collections::HashMap<u64, (usize, f64, f64, Vec<f64>)> =
                std::collections::HashMap::new();
            let mut miss_h = vec![0.0; nc];
            let (mut mn, mut msw) = (0usize, 0.0);
            for &r in rows {
                let (y, w) = (self.y[r], self.w[r]);
                let c = y as usize;
                match self.cols[j][r] {
                    Some(v) => {
                        let e = by_level.entry(v.to_bits()).or_insert_with(|| (0, 0.0, 0.0, vec![0.0; nc]));
                        e.0 += 1;
                        e.1 += w;
                        e.2 += w * y;
                        e.3[c] += w;
                    }
                    None => {
                        mn += 1;
                        msw += w;
                        miss_h[c] += w;
                    }
                }
            }
            if by_level.len() < 2 {
                st.group_ns = t0.elapsed().as_nanos() as u64;
                return (None, st);
            }
            let mut levels: Vec<(f64, usize, f64, f64, Vec<f64>)> = by_level
                .into_iter()
                .map(|(b, (n, sw, swy, h))| (f64::from_bits(b), n, sw, swy, h))
                .collect();

            match &self.params.cat_order {
                CatOrder::FrequencyDesc => {
                    levels.sort_by(|a, b| {
                        b.1.cmp(&a.1)
                            .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    });
                }
                CatOrder::TargetMean => {
                    let wmean = |sw: f64, swy: f64| if sw <= 0.0 { 0.0 } else { swy / sw };
                    levels.sort_by(|a, b| {
                        wmean(b.2, b.3)
                            .partial_cmp(&wmean(a.2, a.3))
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    });
                }
                CatOrder::Explicit(order) => {
                    let rank = |v: f64| order.iter().position(|o| *o == v).unwrap_or(usize::MAX);
                    levels.sort_by(|a, b| {
                        rank(a.0)
                            .cmp(&rank(b.0))
                            .then(b.1.cmp(&a.1))
                            .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    });
                }
            }

            st.group_ns += t0.elapsed().as_nanos() as u64;
            let t1 = Instant::now();
            let k = levels.len();
            let m = self.min_child(rows.len()) as f64;
            let mut candidates: Vec<Vec<usize>> = Vec::new();
            match self.params.cat_search {
                CatSearch::Cumulative => {
                    for i in 1..k { candidates.push((0..i).collect()); }
                }
                CatSearch::SingleVsRest => {
                    for i in 0..k { candidates.push(vec![i]); }
                }
                CatSearch::Both => {
                    for i in 1..k { candidates.push((0..i).collect()); }
                    for i in 0..k { candidates.push(vec![i]); }
                }
            }

            let mut h_parent = vec![0.0; nc];
            for c in 0..nc { h_parent[c] += miss_h[c]; }
            for (_, _, _, _, h) in &levels {
                for c in 0..nc { h_parent[c] += h[c]; }
            }
            let total_w: f64 = h_parent.iter().sum();
            let total_n = rows.len();

            let mut best: Option<(f64, Vec<f64>, bool)> = None;
            for left_idx in candidates.iter() {
                let in_left: std::collections::HashSet<usize> = left_idx.iter().copied().collect();
                let mut ln = 0usize;
                let mut lsw = 0.0;
                let mut h_left_pure = vec![0.0; nc];
                for (li, (_, n, sw, _, h)) in levels.iter().enumerate() {
                    if in_left.contains(&li) {
                        ln += n;
                        lsw += sw;
                        for c in 0..nc { h_left_pure[c] += h[c]; }
                    }
                }
                let rn = total_n - mn - ln;
                let rsw = total_w - msw - lsw;
                if (ln as f64) < m || (rn as f64) < m {
                    continue;
                }
                st.candidates += 1;
                for missing_left in [true, false] {
                    let (aln, alsw) = if missing_left {
                        (ln + mn, lsw + msw)
                    } else {
                        (ln, lsw)
                    };
                    let (arn, arsw) = if missing_left {
                        (rn, rsw)
                    } else {
                        (rn + mn, rsw + msw)
                    };
                    if (aln as f64) < m || (arn as f64) < m {
                        continue;
                    }
                    if self.params.alpha > 0.0 && (alsw < self.params.alpha * total_w || arsw < self.params.alpha * total_w) {
                        continue;
                    }
                    let mut h_l = h_left_pure.clone();
                    if missing_left {
                        for c in 0..nc { h_l[c] += miss_h[c]; }
                    }
                    let mut h_r = vec![0.0; nc];
                    for c in 0..nc { h_r[c] = h_parent[c] - h_l[c]; }

                    let gain = crate::stats::histogram_multiclass_gain(
                        &self.params.gain,
                        &h_parent,
                        total_w,
                        &h_l,
                        alsw,
                        &h_r,
                        arsw,
                    );
                    let gain = if self.params.imbalance_penalty > 0.0 {
                        gain - self.params.imbalance_penalty * (1.0 / aln as f64 + 1.0 / arn as f64)
                    } else {
                        gain
                    };
                    if gain <= self.params.min_gain {
                        continue;
                    }
                    if best.as_ref().map(|(g, ..)| beats(gain, *g)).unwrap_or(true) {
                        let left_levels: Vec<f64> = left_idx.iter().map(|&li| levels[li].0).collect();
                        best = Some((gain, left_levels, missing_left));
                    }
                    if matches!(self.params.cat_search, CatSearch::SingleVsRest) {
                        break;
                    }
                }
            }

            st.scan_ns += t1.elapsed().as_nanos() as u64;
            return (
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
                            oblique: None,
                        },
                        Some(left_levels),
                    )
                }),
                st,
            );
        }

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
        let mut best: Option<(f64, Vec<f64>, bool)> = None;
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
                        oblique: None,
                    },
                    Some(left_levels),
                )
            }),
            st,
        )
    }

    /// Mean-leaf scan — rpart's `anova.c` shape: weighted running sums per
    /// side, O(1) per value, split only where the value changes.
    fn scan_column_mean(
        &self,
        rows: &[usize],
        j: usize,
        scratch: &mut SplitScratch,
        node_moments: (f64, f64, f64),
    ) -> (Option<(f64, Split)>, ScanStats) {
        let mut st = ScanStats::default();
        let m = self.min_child(rows.len()) as f64;
        let t0 = Instant::now();

        scratch.pairs.clear();
        let grid = self.grid_for(j);
        let q = grid.filter(|&p| p > 0.0).map(|p| 1.0 / p);
        let prot_col_opt = self.params.protected_col;

        let mut tn = 0usize;
        let (mut tsw, mut tswy, mut tswy2, mut tsw_prot) = (0.0, 0.0, 0.0, 0.0);
        let (mut tsw_s1_y1, mut tsw_s0_y1, mut tsw_s1_y0, mut tsw_s0_y0) = (0.0, 0.0, 0.0, 0.0);

        for &r in rows {
            if let Some(v_raw) = self.cols[j][r] {
                let v = match q {
                    Some(inv) => (v_raw * inv).round() / inv,
                    None => v_raw,
                };
                let (y, w) = (self.y[r], self.w[r]);
                let p = if let Some(pc) = prot_col_opt {
                    if self.cols[pc][r].unwrap_or(0.0) >= 0.5 { 1.0 } else { 0.0 }
                } else {
                    0.0
                };
                let is_prot = p >= 0.5;
                let is_y1 = y >= 0.5;
                if is_prot && is_y1 {
                    tsw_s1_y1 += w;
                } else if !is_prot && is_y1 {
                    tsw_s0_y1 += w;
                } else if is_prot && !is_y1 {
                    tsw_s1_y0 += w;
                } else {
                    tsw_s0_y0 += w;
                }
                scratch.pairs.push((v, y, w, p));
                tn += 1;
                tsw += w;
                tswy += w * y;
                tswy2 += w * y * y;
                tsw_prot += w * p;
            }
        }

        let n_pairs = scratch.pairs.len();
        if n_pairs < 2 {
            return (None, st);
        }

        scratch.pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN in predictors"));

        let (node_sw, node_swy, node_swy2) = node_moments;
        let mn = rows.len().saturating_sub(tn);
        let (msw, mswy, mswy2) = if mn == 0 {
            (0.0, 0.0, 0.0)
        } else {
            (
                (node_sw - tsw).max(0.0),
                node_swy - tswy,
                (node_swy2 - tswy2).max(0.0),
            )
        };
        let mut msw_prot = 0.0;
        let (mut msw_s1_y1, mut msw_s0_y1, mut msw_s1_y0, mut msw_s0_y0) = (0.0, 0.0, 0.0, 0.0);
        if let Some(pc) = prot_col_opt {
            if mn > 0 {
                for &r in rows {
                    if self.cols[j][r].is_none() {
                        let is_prot = self.cols[pc][r].unwrap_or(0.0) >= 0.5;
                        let is_y1 = self.y[r] >= 0.5;
                        let w = self.w[r];
                        if is_prot {
                            msw_prot += w;
                        }
                        if is_prot && is_y1 {
                            msw_s1_y1 += w;
                        } else if !is_prot && is_y1 {
                            msw_s0_y1 += w;
                        } else if is_prot && !is_y1 {
                            msw_s1_y0 += w;
                        } else {
                            msw_s0_y0 += w;
                        }
                    }
                }
            }
        }

        let is_multiclass_gain = matches!(
            self.params.gain,
            GainFn::Gini | GainFn::Entropy | GainFn::GainRatio | GainFn::Twoing
        );
        let n_classes_opt = if is_multiclass_gain {
            let mut max_c = self.params.n_classes.unwrap_or(0);
            let mut all_int = true;
            for &(_, y, _, _) in &scratch.pairs {
                if y >= 0.0 && y < 256.0 && y.fract() == 0.0 {
                    max_c = max_c.max((y as usize) + 1);
                } else {
                    all_int = false;
                    break;
                }
            }
            if all_int && max_c >= 2 {
                Some(max_c)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(nc) = n_classes_opt {
            scratch.h_parent.clear();
            scratch.h_parent.resize(nc, 0.0);
            for &(_, y, w, _) in &scratch.pairs {
                scratch.h_parent[y as usize] += w;
            }
            scratch.h_miss.clear();
            scratch.h_miss.resize(nc, 0.0);
            if mn > 0 {
                for &r in rows {
                    if self.cols[j][r].is_none() {
                        let y = self.y[r];
                        if y >= 0.0 && (y as usize) < nc && y.fract() == 0.0 {
                            scratch.h_miss[y as usize] += self.w[r];
                            scratch.h_parent[y as usize] += self.w[r];
                        }
                    }
                }
            }
            scratch.h_left.clear();
            scratch.h_left.resize(nc, 0.0);
            scratch.h_cand_left.clear();
            scratch.h_cand_left.resize(nc, 0.0);
            scratch.h_cand_right.clear();
            scratch.h_cand_right.resize(nc, 0.0);
        } else {
            let is_custom_gain = !matches!(self.params.gain, GainFn::Sse | GainFn::Deviance);
            if is_custom_gain {
                scratch.pairs_yw.clear();
                scratch.pairs_yw.extend(scratch.pairs.iter().map(|&(_, y, w, _)| (y, w)));
                scratch.missing_yws.clear();
                if mn > 0 {
                    for &r in rows {
                        if self.cols[j][r].is_none() {
                            scratch.missing_yws.push((self.y[r], self.w[r]));
                        }
                    }
                }
            }
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

        let min_v = scratch.pairs.first().unwrap().0;
        let max_v = scratch.pairs.last().unwrap().0;
        if self.params.extratrees && (max_v - min_v).abs() < 1e-12 {
            st.scan_ns += t1.elapsed().as_nanos() as u64;
            return (None, st);
        }
        let extratrees_cut = if self.params.extratrees {
            let mut state = self.params.seed
                .wrapping_add((rows.len() as u64).wrapping_mul(0x517cc1b727220a95))
                .wrapping_add((j as u64).wrapping_mul(0x9e3779b97f4a7c15))
                .wrapping_add((rows.first().copied().unwrap_or(0) as u64).wrapping_mul(0x85ebca6b));
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let rand_u = (state.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / ((1u64 << 53) as f64);
            Some(min_v + (0.05 + 0.90 * rand_u) * (max_v - min_v))
        } else {
            None
        };

        let mut ln = 0usize;
        let (mut lsw, mut lswy, mut lswy2, mut lsw_prot) = (0.0, 0.0, 0.0, 0.0);
        let (mut lsw_s1_y1, mut lsw_s0_y1, mut lsw_s1_y0, mut lsw_s0_y0) = (0.0, 0.0, 0.0, 0.0);
        let mut best: Option<(f64, Split)> = None;
        let total_w = tsw + msw;

        for k in 0..n_pairs {
            let (v, y, w, p) = scratch.pairs[k];
            ln += 1;
            lsw += w;
            lswy += w * y;
            lswy2 += w * y * y;
            lsw_prot += w * p;
            let is_prot = p >= 0.5;
            let is_y1 = y >= 0.5;
            if is_prot && is_y1 {
                lsw_s1_y1 += w;
            } else if !is_prot && is_y1 {
                lsw_s0_y1 += w;
            } else if is_prot && !is_y1 {
                lsw_s1_y0 += w;
            } else {
                lsw_s0_y0 += w;
            }
            if n_classes_opt.is_some() {
                scratch.h_left[y as usize] += w;
            }
            let is_boundary = if let Some(et_cut) = extratrees_cut {
                scratch.pairs[k].0 <= et_cut && (k + 1 == n_pairs || scratch.pairs[k + 1].0 > et_cut)
            } else {
                k + 1 < n_pairs && scratch.pairs[k + 1].0 != v
            };
            if !is_boundary {
                continue;
            }
            let threshold = if let Some(et_cut) = extratrees_cut {
                et_cut
            } else {
                match grid {
                    Some(_) => v,
                    None => (v + scratch.pairs[k + 1].0) / 2.0,
                }
            };
            let rn = tn - ln;
            let (rsw, rswy, rswy2) = (tsw - lsw, tswy - lswy, tswy2 - lswy2);
            st.candidates += 1;
            for missing_left in [true, false] {
                let (aln, alsw, alswy, alswy2, alsw_prot) = if missing_left {
                    (ln + mn, lsw + msw, lswy + mswy, lswy2 + mswy2, lsw_prot + msw_prot)
                } else {
                    (ln, lsw, lswy, lswy2, lsw_prot)
                };
                let (arn, arsw, arswy, arswy2, arsw_prot) = if missing_left {
                    (rn, rsw, rswy, rswy2, tsw_prot - lsw_prot)
                } else {
                    (rn + mn, rsw + msw, rswy + mswy, rswy2 + mswy2, tsw_prot + msw_prot - lsw_prot)
                };
                if (aln as f64) < m || (arn as f64) < m {
                    continue;
                }
                if self.params.alpha > 0.0 && (alsw < self.params.alpha * total_w
                    || arsw < self.params.alpha * total_w)
                {
                    continue;
                }
                if !self.mono_ok(j, alswy / alsw.max(1e-300), arswy / arsw.max(1e-300)) {
                    continue;
                }
                let gain = if let Some(nc) = n_classes_opt {
                    if missing_left {
                        for c in 0..nc {
                            let left_c = scratch.h_left[c] + scratch.h_miss[c];
                            scratch.h_cand_left[c] = left_c;
                            scratch.h_cand_right[c] = scratch.h_parent[c] - left_c;
                        }
                    } else {
                        for c in 0..nc {
                            let left_c = scratch.h_left[c];
                            scratch.h_cand_left[c] = left_c;
                            scratch.h_cand_right[c] = scratch.h_parent[c] - left_c;
                        }
                    }
                    crate::stats::histogram_multiclass_gain(
                        &self.params.gain,
                        &scratch.h_parent,
                        total_w,
                        &scratch.h_cand_left,
                        alsw,
                        &scratch.h_cand_right,
                        arsw,
                    )
                } else {
                    match self.params.gain {
                        GainFn::Deviance if self.params.leaf_family != crate::family::LeafFamily::Gaussian => {
                            self.constant_family_cost(tsw + msw, tswy + mswy)
                                - self.constant_family_cost(alsw, alswy)
                                - self.constant_family_cost(arsw, arswy)
                        }
                        GainFn::Sse | GainFn::Deviance => {
                            parent_sse - (sse(alsw, alswy, alswy2) + sse(arsw, arswy, arswy2))
                        }
                        ref gfn => {
                            let yws_sorted = &scratch.pairs_yw;
                            if scratch.missing_yws.is_empty() {
                                gain_of(gfn, &yws_sorted[..k + 1], &yws_sorted[k + 1..])
                            } else {
                                let mut ls: Vec<(f64, f64)> = yws_sorted[..k + 1].to_vec();
                                let mut rs: Vec<(f64, f64)> = yws_sorted[k + 1..].to_vec();
                                if missing_left {
                                    ls.extend_from_slice(&scratch.missing_yws);
                                } else {
                                    rs.extend_from_slice(&scratch.missing_yws);
                                }
                                gain_of(gfn, &ls, &rs)
                            }
                        }
                    }
                };
                let gain = if self.params.imbalance_penalty > 0.0 {
                    gain - self.params.imbalance_penalty * (1.0 / aln as f64 + 1.0 / arn as f64)
                } else {
                    gain
                };
                let gain = if self.params.fairness_penalty > 0.0 && self.params.protected_col.is_some() {
                    let disparity = match self.params.fairness_metric {
                        crate::model_tree::FairnessMetric::DemographicParity => {
                            let p_l = if alsw > 0.0 { alsw_prot / alsw } else { 0.0 };
                            let p_r = if arsw > 0.0 { arsw_prot / arsw } else { 0.0 };
                            (p_l - p_r).abs()
                        }
                        crate::model_tree::FairnessMetric::EqualizedOdds => {
                            let (alsw_s1_y1, alsw_s0_y1, alsw_s1_y0, alsw_s0_y0) = if missing_left {
                                (lsw_s1_y1 + msw_s1_y1, lsw_s0_y1 + msw_s0_y1, lsw_s1_y0 + msw_s1_y0, lsw_s0_y0 + msw_s0_y0)
                            } else {
                                (lsw_s1_y1, lsw_s0_y1, lsw_s1_y0, lsw_s0_y0)
                            };
                            let tot_s1_y1 = tsw_s1_y1 + msw_s1_y1;
                            let tot_s0_y1 = tsw_s0_y1 + msw_s0_y1;
                            let tot_s1_y0 = tsw_s1_y0 + msw_s1_y0;
                            let tot_s0_y0 = tsw_s0_y0 + msw_s0_y0;

                            let diff_tpr = if tot_s1_y1 > 0.0 && tot_s0_y1 > 0.0 {
                                let tpr_1 = alsw_s1_y1 / tot_s1_y1;
                                let tpr_0 = alsw_s0_y1 / tot_s0_y1;
                                Some((tpr_1 - tpr_0).abs())
                            } else {
                                None
                            };

                            let diff_fpr = if tot_s1_y0 > 0.0 && tot_s0_y0 > 0.0 {
                                let fpr_1 = alsw_s1_y0 / tot_s1_y0;
                                let fpr_0 = alsw_s0_y0 / tot_s0_y0;
                                Some((fpr_1 - fpr_0).abs())
                            } else {
                                None
                            };

                            match (diff_tpr, diff_fpr) {
                                (Some(dt), Some(df)) => 0.5 * (dt + df),
                                (Some(dt), None) => dt,
                                (None, Some(df)) => df,
                                (None, None) => 0.0,
                            }
                        }
                    };
                    let scale = parent_sse.max(1e-12);
                    (gain - self.params.fairness_penalty * scale * disparity).max(0.0)
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
                            oblique: None,
                        },
                    ));
                }
            }
            if extratrees_cut.is_some() {
                break;
            }
        }

        st.scan_ns += t1.elapsed().as_nanos() as u64;
        (best, st)
    }

    /// Fast O(1) causal split scan maximizing treatment effect divergence between partitions.
    fn scan_column_causal(
        &self,
        rows: &[usize],
        j: usize,
        scratch: &mut SplitScratch,
    ) -> (Option<(f64, Split)>, ScanStats) {
        let mut st = ScanStats::default();
        let Some(treat) = self.treatments else {
            return (None, st);
        };
        let m = self.min_child(rows.len()) as f64;
        let t0 = Instant::now();

        scratch.pairs_causal.clear();
        let grid = self.grid_for(j);
        let q = grid.filter(|&p| p > 0.0).map(|p| 1.0 / p);

        let (mut tn_t, mut tn_c) = (0usize, 0usize);
        let (mut tsw_t, mut tswy_t) = (0.0, 0.0);
        let (mut tsw_c, mut tswy_c) = (0.0, 0.0);

        for &r in rows {
            if let Some(v_raw) = self.cols[j][r] {
                let v = match q {
                    Some(inv) => (v_raw * inv).round() / inv,
                    None => v_raw,
                };
                let (y, w) = (self.y[r], self.w[r]);
                let tr = treat.get(r).copied().unwrap_or(0.0);
                scratch.pairs_causal.push((v, y, w, tr));
                if tr > 0.0 {
                    tn_t += 1;
                    tsw_t += w;
                    tswy_t += w * y;
                } else {
                    tn_c += 1;
                    tsw_c += w;
                    tswy_c += w * y;
                }
            }
        }

        let n_pairs = scratch.pairs_causal.len();
        if n_pairs < 2 || tsw_t <= 0.0 || tsw_c <= 0.0 {
            return (None, st);
        }

        scratch.pairs_causal.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN in predictors"));

        let (mut mn_t, mut mn_c) = (0usize, 0usize);
        let (mut msw_t, mut mswy_t) = (0.0, 0.0);
        let (mut msw_c, mut mswy_c) = (0.0, 0.0);
        if rows.len() > n_pairs {
            for &r in rows {
                if self.cols[j][r].is_none() {
                    let (y, w) = (self.y[r], self.w[r]);
                    let tr = treat.get(r).copied().unwrap_or(0.0);
                    if tr > 0.0 {
                        mn_t += 1;
                        msw_t += w;
                        mswy_t += w * y;
                    } else {
                        mn_c += 1;
                        msw_c += w;
                        mswy_c += w * y;
                    }
                }
            }
        }

        st.group_ns += t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();

        let (mut ln_t, mut ln_c) = (0usize, 0usize);
        let (mut lsw_t, mut lswy_t) = (0.0, 0.0);
        let (mut lsw_c, mut lswy_c) = (0.0, 0.0);
        let mut best: Option<(f64, Split)> = None;
        let total_w = tsw_t + tsw_c + msw_t + msw_c;

        for k in 0..n_pairs {
            let (v, y, w, tr) = scratch.pairs_causal[k];
            if tr > 0.0 {
                ln_t += 1;
                lsw_t += w;
                lswy_t += w * y;
            } else {
                ln_c += 1;
                lsw_c += w;
                lswy_c += w * y;
            }
            let is_boundary = k + 1 < n_pairs && scratch.pairs_causal[k + 1].0 != v;
            if !is_boundary {
                continue;
            }
            let threshold = match grid {
                Some(_) => v,
                None => (v + scratch.pairs_causal[k + 1].0) / 2.0,
            };
            let (rn_t, rn_c) = (tn_t - ln_t, tn_c - ln_c);
            let (rsw_t, rswy_t) = (tsw_t - lsw_t, tswy_t - lswy_t);
            let (rsw_c, rswy_c) = (tsw_c - lsw_c, tswy_c - lswy_c);
            st.candidates += 1;

            for missing_left in [true, false] {
                let (aln_t, aln_c, alsw_t, alswy_t, alsw_c, alswy_c) = if missing_left {
                    (ln_t + mn_t, ln_c + mn_c, lsw_t + msw_t, lswy_t + mswy_t, lsw_c + msw_c, lswy_c + mswy_c)
                } else {
                    (ln_t, ln_c, lsw_t, lswy_t, lsw_c, lswy_c)
                };
                let (arn_t, arn_c, arsw_t, arswy_t, arsw_c, arswy_c) = if missing_left {
                    (rn_t, rn_c, rsw_t, rswy_t, rsw_c, rswy_c)
                } else {
                    (rn_t + mn_t, rn_c + mn_c, rsw_t + msw_t, rswy_t + mswy_t, rsw_c + msw_c, rswy_c + mswy_c)
                };
                let aln = aln_t + aln_c;
                let arn = arn_t + arn_c;
                if (aln as f64) < m || (arn as f64) < m {
                    continue;
                }
                let alsw = alsw_t + alsw_c;
                let arsw = arsw_t + arsw_c;
                if self.params.alpha > 0.0 && (alsw < self.params.alpha * total_w || arsw < self.params.alpha * total_w) {
                    continue;
                }
                // Both child sides must have positive treated and control masses
                if alsw_t <= 0.0 || alsw_c <= 0.0 || arsw_t <= 0.0 || arsw_c <= 0.0 {
                    continue;
                }
                let tau_l = (alswy_t / alsw_t) - (alswy_c / alsw_c);
                let tau_r = (arswy_t / arsw_t) - (arswy_c / arsw_c);
                let mut gain = (alsw * arsw / (alsw + arsw)) * (tau_l - tau_r).powi(2);
                if self.params.imbalance_penalty > 0.0 {
                    gain -= self.params.imbalance_penalty * (1.0 / aln as f64 + 1.0 / arn as f64);
                }
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
                            oblique: None,
                        },
                    ));
                }
            }
        }

        st.scan_ns += t1.elapsed().as_nanos() as u64;
        (best, st)
    }

    /// Fast O(1) categorical causal split scan maximizing treatment effect divergence.
    fn scan_column_categorical_causal(
        &self,
        rows: &[usize],
        j: usize,
        scratch: &mut SplitScratch,
    ) -> (Option<(f64, Split, Option<Vec<f64>>)>, ScanStats) {
        let mut st = ScanStats::default();
        let Some(treat) = self.treatments else {
            return (None, st);
        };
        let t0 = Instant::now();

        let mut by_level: std::collections::HashMap<u64, (usize, f64, f64, f64, f64)> =
            std::collections::HashMap::new();
        let (mut mn_t, mut mn_c) = (0usize, 0usize);
        let (mut msw_t, mut mswy_t) = (0.0, 0.0);
        let (mut msw_c, mut mswy_c) = (0.0, 0.0);

        for &r in rows {
            let (y, w) = (self.y[r], self.w[r]);
            let tr = treat.get(r).copied().unwrap_or(0.0);
            match self.cols[j][r] {
                Some(v) => {
                    let e = by_level.entry(v.to_bits()).or_insert((0, 0.0, 0.0, 0.0, 0.0));
                    e.0 += 1;
                    if tr > 0.0 {
                        e.1 += w;
                        e.2 += w * y;
                    } else {
                        e.3 += w;
                        e.4 += w * y;
                    }
                }
                None => {
                    if tr > 0.0 {
                        mn_t += 1;
                        msw_t += w;
                        mswy_t += w * y;
                    } else {
                        mn_c += 1;
                        msw_c += w;
                        mswy_c += w * y;
                    }
                }
            }
        }
        if by_level.len() < 2 {
            st.group_ns = t0.elapsed().as_nanos() as u64;
            return (None, st);
        }

        scratch.level_causal.clear();
        for (b, (n, sw_t, swy_t, sw_c, swy_c)) in by_level {
            scratch.level_causal.push((f64::from_bits(b), n, sw_t, swy_t, sw_c, swy_c));
        }

        scratch.level_causal.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        });

        st.group_ns += t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();

        let k = scratch.level_causal.len();
        let m = self.min_child(rows.len()) as f64;
        let (mut tsw_t, mut tswy_t) = (0.0, 0.0);
        let (mut tsw_c, mut tswy_c) = (0.0, 0.0);
        for &(_, _, sw_t, swy_t, sw_c, swy_c) in scratch.level_causal.iter() {
            tsw_t += sw_t;
            tswy_t += swy_t;
            tsw_c += sw_c;
            tswy_c += swy_c;
        }
        let total_w = tsw_t + tsw_c + msw_t + msw_c;

        let mut best: Option<(f64, Vec<usize>, bool)> = None;

        let mut eval_split = |left_indices: &[usize], best: &mut Option<(f64, Vec<usize>, bool)>| {
            let mut lsw_t = 0.0;
            let mut lswy_t = 0.0;
            let mut lsw_c = 0.0;
            let mut lswy_c = 0.0;
            let mut ln = 0usize;
            for &idx in left_indices {
                let (_, n, sw_t, swy_t, sw_c, swy_c) = scratch.level_causal[idx];
                ln += n;
                lsw_t += sw_t;
                lswy_t += swy_t;
                lsw_c += sw_c;
                lswy_c += swy_c;
            }
            let rn = rows.len().saturating_sub(ln + mn_t + mn_c);
            if (ln as f64) < m || (rn as f64) < m {
                return;
            }
            let rsw_t = tsw_t - lsw_t;
            let rswy_t = tswy_t - lswy_t;
            let rsw_c = tsw_c - lsw_c;
            let rswy_c = tswy_c - lswy_c;
            st.candidates += 1;

            for missing_left in [true, false] {
                let (alsw_t, alswy_t, alsw_c, alswy_c) = if missing_left {
                    (lsw_t + msw_t, lswy_t + mswy_t, lsw_c + msw_c, lswy_c + mswy_c)
                } else {
                    (lsw_t, lswy_t, lsw_c, lswy_c)
                };
                let (arsw_t, arswy_t, arsw_c, arswy_c) = if missing_left {
                    (rsw_t, rswy_t, rsw_c, rswy_c)
                } else {
                    (rsw_t + msw_t, rswy_t + mswy_t, rsw_c + msw_c, rswy_c + mswy_c)
                };
                let aln = if missing_left { ln + mn_t + mn_c } else { ln };
                let arn = if missing_left { rn } else { rn + mn_t + mn_c };
                if (aln as f64) < m || (arn as f64) < m {
                    continue;
                }
                let alsw = alsw_t + alsw_c;
                let arsw = arsw_t + arsw_c;
                if self.params.alpha > 0.0 && (alsw < self.params.alpha * total_w || arsw < self.params.alpha * total_w) {
                    continue;
                }
                if alsw_t <= 0.0 || alsw_c <= 0.0 || arsw_t <= 0.0 || arsw_c <= 0.0 {
                    continue;
                }
                let tau_l = (alswy_t / alsw_t) - (alswy_c / alsw_c);
                let tau_r = (arswy_t / arsw_t) - (arswy_c / arsw_c);
                let mut gain = (alsw * arsw / (alsw + arsw)) * (tau_l - tau_r).powi(2);
                if self.params.imbalance_penalty > 0.0 {
                    gain -= self.params.imbalance_penalty * (1.0 / aln as f64 + 1.0 / arn as f64);
                }
                if gain <= self.params.min_gain {
                    continue;
                }
                if best.as_ref().map(|(g, ..)| beats(gain, *g)).unwrap_or(true) {
                    *best = Some((gain, left_indices.to_vec(), missing_left));
                }
            }
        };

        match self.params.cat_search {
            CatSearch::Cumulative => {
                for i in 1..k {
                    let left_idx: Vec<usize> = (0..i).collect();
                    eval_split(&left_idx, &mut best);
                }
            }
            CatSearch::SingleVsRest => {
                for i in 0..k {
                    eval_split(&[i], &mut best);
                }
            }
            CatSearch::Both => {
                for i in 1..k {
                    let left_idx: Vec<usize> = (0..i).collect();
                    eval_split(&left_idx, &mut best);
                }
                for i in 0..k {
                    eval_split(&[i], &mut best);
                }
            }
        }

        st.scan_ns += t1.elapsed().as_nanos() as u64;
        let Some((gain, left_indices, missing_left)) = best else {
            return (None, st);
        };
        let left_levels: Vec<f64> = left_indices.iter().map(|&idx| scratch.level_causal[idx].0).collect();
        let split = Split {
            col: j,
            threshold: 0.0,
            missing_left,
            gain,
            cat: None,
            sur: None,
            oblique: None,
        };
        (Some((gain, split, Some(left_levels))), st)
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
                            oblique: None,
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
            LeafFamily::Binomial | LeafFamily::Probit => -2.0 * (
                xlog(sy, mean) + xlog((sw - sy).max(0.0), 1.0 - mean)
            ),
            LeafFamily::Poisson => 2.0 * (sy - xlog(sy, mean)),
            LeafFamily::Gamma => -2.0 * sw * mean.max(1e-12).ln(),
            LeafFamily::Tweedie => {
                let p = self.params.tweedie_p;
                let m = mean.max(1e-12);
                -2.0 * (sy * m.powf(1.0 - p) / (1.0 - p) - sw * m.powf(2.0 - p) / (2.0 - p))
            }
            LeafFamily::Tobit => 0.0,
            LeafFamily::Gaussian | LeafFamily::Softmax | LeafFamily::Survival | LeafFamily::Heteroscedastic => {
                unreachable!("Gaussian, Softmax, Survival, and Heteroscedastic use SSE/custom scans")
            }
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
                        gain, cat: None, sur: None, oblique: None }, part));
                }
            }
            if categorical && matches!(self.params.cat_search, CatSearch::SingleVsRest)
                && best.is_some() { break; }
        }
        st.scan_ns = t1.elapsed().as_nanos() as u64;
        (best, st)
    }

}
