//! Abstract Design Matrix representation for zero-copy regression.
//!
//! Provides two backends sharing the same mathematical API:
//! - `ColumnSliceView`: zero-copy view over tree row subsets (`rows: &[usize]`),
//!   individual column vectors, sample weights, and optional missing-value indicators.
//! - `DenseView`: 2D slice view for full-table regression and MARS basis functions.

use crate::node_stats::Gram;

/// Trait for design matrix access across standalone and tree-leaf regressions.
pub trait DesignMatrix {
    /// Number of observations $N$.
    fn n_rows(&self) -> usize;

    /// Number of parameters $P$ (including intercept).
    fn n_features(&self) -> usize;

    /// Evaluates linear prediction $\eta_i = x_i^T \beta$ for observation `i`.
    fn eval_row(&self, row_idx: usize, coef: &[f64]) -> f64;

    /// Target response $y_i$ for observation `i`.
    fn get_y(&self, row_idx: usize) -> f64;

    /// Sample weight $w_i$ for observation `i`.
    fn get_weight(&self, row_idx: usize) -> f64;

    /// Accumulates design row $x_i$ into a preallocated slice `[1.0, x_1, ..., x_p, na_1, ...]`.
    fn copy_design_row(&self, row_idx: usize, out: &mut [f64]);

    /// Computes full Gram matrix ($X^T W X$, $X^T W y$, $y^T W y$).
    fn compute_gram(&self, custom_weights: Option<&[f64]>) -> Gram;

    /// Inner product of column $j$ with a residual vector: $\sum_i w_i X_{ij} r_i$.
    fn col_dot_residuals(&self, j: usize, residuals: &[f64], custom_weights: Option<&[f64]>) -> f64;

    /// Weighted squared norm of column $j$: $\sum_i w_i X_{ij}^2$.
    fn col_norm_sq(&self, j: usize, custom_weights: Option<&[f64]>) -> f64;
}

/// Zero-copy design matrix view over columnar tree-node data.
#[derive(Clone)]
pub struct ColumnSliceView<'a> {
    pub cols: &'a [Vec<Option<f64>>],
    pub y: &'a [f64],
    pub w: &'a [f64],
    pub rows: &'a [usize],
    pub chosen: &'a [usize],
    pub with_indicators: bool,
}

impl<'a> ColumnSliceView<'a> {
    pub fn new(
        cols: &'a [Vec<Option<f64>>],
        y: &'a [f64],
        w: &'a [f64],
        rows: &'a [usize],
        chosen: &'a [usize],
        with_indicators: bool,
    ) -> Self {
        Self {
            cols,
            y,
            w,
            rows,
            chosen,
            with_indicators,
        }
    }
}

impl<'a> DesignMatrix for ColumnSliceView<'a> {
    #[inline]
    fn n_rows(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    fn n_features(&self) -> usize {
        1 + self.chosen.len() + if self.with_indicators { self.chosen.len() } else { 0 }
    }

    #[inline]
    fn get_y(&self, row_idx: usize) -> f64 {
        let r = self.rows[row_idx];
        self.y[r]
    }

    #[inline]
    fn get_weight(&self, row_idx: usize) -> f64 {
        let r = self.rows[row_idx];
        self.w[r]
    }

    #[inline]
    fn eval_row(&self, row_idx: usize, coef: &[f64]) -> f64 {
        let r = self.rows[row_idx];
        let nc = self.chosen.len();
        let mut eta = coef[0];
        for (k, &c) in self.chosen.iter().enumerate() {
            match self.cols[c][r] {
                Some(v) => eta += coef[1 + k] * v,
                None => {
                    if self.with_indicators {
                        eta += coef[1 + nc + k];
                    }
                }
            }
        }
        eta
    }

    fn copy_design_row(&self, row_idx: usize, out: &mut [f64]) {
        let r = self.rows[row_idx];
        let nc = self.chosen.len();
        out[0] = 1.0;
        for (k, &c) in self.chosen.iter().enumerate() {
            match self.cols[c][r] {
                Some(v) => {
                    out[1 + k] = v;
                    if self.with_indicators {
                        out[1 + nc + k] = 0.0;
                    }
                }
                None => {
                    out[1 + k] = 0.0;
                    if self.with_indicators {
                        out[1 + nc + k] = 1.0;
                    }
                }
            }
        }
    }

    fn compute_gram(&self, custom_weights: Option<&[f64]>) -> Gram {
        let dim = self.n_features();
        let mut gram = Gram::new(dim);
        let mut row = vec![0.0; dim];
        for (i, &r) in self.rows.iter().enumerate() {
            let wi = match custom_weights {
                Some(cw) => cw[i],
                None => self.w[r],
            };
            if wi <= 0.0 {
                continue;
            }
            self.copy_design_row(i, &mut row);
            gram.add_row(&row, self.y[r], wi);
        }
        gram
    }

    fn col_dot_residuals(&self, j: usize, residuals: &[f64], custom_weights: Option<&[f64]>) -> f64 {
        let nc = self.chosen.len();
        let mut dot = 0.0;
        for (i, &r) in self.rows.iter().enumerate() {
            let wi = match custom_weights {
                Some(cw) => cw[i],
                None => self.w[r],
            };
            if wi <= 0.0 {
                continue;
            }
            let xj = if j == 0 {
                1.0
            } else if j <= nc {
                let col = self.chosen[j - 1];
                self.cols[col][r].unwrap_or(0.0)
            } else {
                let col = self.chosen[j - 1 - nc];
                if self.cols[col][r].is_none() { 1.0 } else { 0.0 }
            };
            dot += wi * xj * residuals[i];
        }
        dot
    }

    fn col_norm_sq(&self, j: usize, custom_weights: Option<&[f64]>) -> f64 {
        let nc = self.chosen.len();
        let mut sum_sq = 0.0;
        for (i, &r) in self.rows.iter().enumerate() {
            let wi = match custom_weights {
                Some(cw) => cw[i],
                None => self.w[r],
            };
            if wi <= 0.0 {
                continue;
            }
            let xj = if j == 0 {
                1.0
            } else if j <= nc {
                let col = self.chosen[j - 1];
                self.cols[col][r].unwrap_or(0.0)
            } else {
                let col = self.chosen[j - 1 - nc];
                if self.cols[col][r].is_none() { 1.0 } else { 0.0 }
            };
            sum_sq += wi * xj * xj;
        }
        sum_sq
    }
}

/// Dense 2D design matrix view for standalone modeling and MARS bases.
#[derive(Clone)]
pub struct DenseView<'a> {
    pub x: &'a [Vec<f64>],
    pub y: &'a [f64],
    pub w: Option<&'a [f64]>,
    pub with_intercept: bool,
}

impl<'a> DenseView<'a> {
    pub fn new(x: &'a [Vec<f64>], y: &'a [f64], w: Option<&'a [f64]>, with_intercept: bool) -> Self {
        Self {
            x,
            y,
            w,
            with_intercept,
        }
    }
}

impl<'a> DesignMatrix for DenseView<'a> {
    #[inline]
    fn n_rows(&self) -> usize {
        self.x.len()
    }

    #[inline]
    fn n_features(&self) -> usize {
        let p = self.x.first().map(|r| r.len()).unwrap_or(0);
        if self.with_intercept { p + 1 } else { p }
    }

    #[inline]
    fn get_y(&self, row_idx: usize) -> f64 {
        self.y[row_idx]
    }

    #[inline]
    fn get_weight(&self, row_idx: usize) -> f64 {
        self.w.map_or(1.0, |ws| ws[row_idx])
    }

    #[inline]
    fn eval_row(&self, row_idx: usize, coef: &[f64]) -> f64 {
        let row = &self.x[row_idx];
        if self.with_intercept {
            let mut eta = coef[0];
            for (j, &val) in row.iter().enumerate() {
                eta += coef[j + 1] * val;
            }
            eta
        } else {
            let mut eta = 0.0;
            for (j, &val) in row.iter().enumerate() {
                eta += coef[j] * val;
            }
            eta
        }
    }

    fn copy_design_row(&self, row_idx: usize, out: &mut [f64]) {
        let row = &self.x[row_idx];
        if self.with_intercept {
            out[0] = 1.0;
            out[1..=row.len()].copy_from_slice(row);
        } else {
            out[..row.len()].copy_from_slice(row);
        }
    }

    fn compute_gram(&self, custom_weights: Option<&[f64]>) -> Gram {
        let dim = self.n_features();
        let mut gram = Gram::new(dim);
        let mut row = vec![0.0; dim];
        for i in 0..self.x.len() {
            let wi = match custom_weights {
                Some(cw) => cw[i],
                None => self.get_weight(i),
            };
            if wi <= 0.0 {
                continue;
            }
            self.copy_design_row(i, &mut row);
            gram.add_row(&row, self.y[i], wi);
        }
        gram
    }

    fn col_dot_residuals(&self, j: usize, residuals: &[f64], custom_weights: Option<&[f64]>) -> f64 {
        let mut dot = 0.0;
        for i in 0..self.x.len() {
            let wi = match custom_weights {
                Some(cw) => cw[i],
                None => self.get_weight(i),
            };
            if wi <= 0.0 {
                continue;
            }
            let val = if self.with_intercept {
                if j == 0 { 1.0 } else { self.x[i][j - 1] }
            } else {
                self.x[i][j]
            };
            dot += wi * val * residuals[i];
        }
        dot
    }

    fn col_norm_sq(&self, j: usize, custom_weights: Option<&[f64]>) -> f64 {
        let mut sum_sq = 0.0;
        for i in 0..self.x.len() {
            let wi = match custom_weights {
                Some(cw) => cw[i],
                None => self.get_weight(i),
            };
            if wi <= 0.0 {
                continue;
            }
            let val = if self.with_intercept {
                if j == 0 { 1.0 } else { self.x[i][j - 1] }
            } else {
                self.x[i][j]
            };
            sum_sq += wi * val * val;
        }
        sum_sq
    }
}
