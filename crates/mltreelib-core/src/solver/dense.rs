//! Checked standalone Gaussian and generalized linear model fits.

use crate::family::LeafFamily;
use crate::leaf::LeafSolver;
use crate::node_stats::fit_glm_design;

use super::{
    solve_iterative, solve_normal, Convergence, DenseView, DesignMatrix, GlmFamily, GlmSummary,
    SolverOptions, StopReason,
};

/// A fitted GLM on a dense numeric design with an unpenalized intercept.
#[derive(Clone, Debug)]
pub struct FittedGlm {
    pub intercept: f64,
    pub coef: Vec<f64>,
    pub family: GlmFamily,
    pub convergence: Convergence,
    /// Half of the total weighted training deviance plus the solver's penalty.
    pub objective: f64,
    pub summary: Option<GlmSummary>,
}

impl FittedGlm {
    /// Predict on the response scale, or the latent mean for censored Tobit.
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        check_features(x, self.coef.len())?;
        x.iter()
            .map(|row| {
                let eta = row
                    .iter()
                    .zip(&self.coef)
                    .fold(self.intercept, |sum, (x, b)| sum + x * b);
                if !eta.is_finite() {
                    return Err("Linear predictor is not finite".into());
                }
                Ok(self.family.mean_for_loss(eta))
            })
            .collect()
    }

    /// Weighted mean unit deviance on an independent dataset.
    pub fn mean_deviance(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        weights: Option<&[f64]>,
    ) -> Result<f64, String> {
        let predictions = self.predict(x)?;
        let mass = check_response(y, weights, x.len(), self.family)?;
        let total = y
            .iter()
            .zip(predictions)
            .enumerate()
            .filter_map(|(i, (&y, mean))| {
                let w = weights.map_or(1.0, |weights| weights[i]);
                (w > 0.0).then(|| w * self.family.deviance(y, mean))
            })
            .sum::<f64>();
        if !total.is_finite() {
            return Err("Evaluation deviance is not finite".into());
        }
        Ok(total / mass)
    }
}

fn check_features(x: &[Vec<f64>], cols: usize) -> Result<(), String> {
    if x.iter()
        .any(|row| row.len() != cols || row.iter().any(|v| !v.is_finite()))
    {
        return Err(format!("X must have {cols} finite numeric values per row"));
    }
    Ok(())
}

fn check_response(
    y: &[f64],
    weights: Option<&[f64]>,
    n: usize,
    family: GlmFamily,
) -> Result<f64, String> {
    if y.len() != n || weights.is_some_and(|w| w.len() != n) {
        return Err("X, y, and sample_weight must have the same row count".into());
    }
    if y.iter().any(|&v| {
        !v.is_finite()
            || match family {
                GlmFamily::Binomial | GlmFamily::Probit => !(0.0..=1.0).contains(&v),
                GlmFamily::Poisson | GlmFamily::Tweedie { .. } => v < 0.0,
                GlmFamily::Gamma => v <= 0.0,
                GlmFamily::Gaussian | GlmFamily::Tobit { .. } => false,
            }
    }) {
        return Err(format!("y contains values outside the {family:?} domain"));
    }
    let mass = match weights {
        Some(w) if w.iter().all(|&v| v.is_finite() && v >= 0.0) => w.iter().sum(),
        Some(_) => return Err("sample_weight must be finite and non-negative".into()),
        None => n as f64,
    };
    if !mass.is_finite() || mass <= 0.0 {
        return Err("Total sample weight must be positive and finite".into());
    }
    Ok(mass)
}

/// Fit a weighted dense design using the same kernels as linear tree leaves.
/// Passing `None` as solver selects a direct Gaussian solve.
pub fn fit_dense(
    x: &[Vec<f64>],
    y: &[f64],
    weights: Option<&[f64]>,
    solver: Option<LeafSolver>,
    options: &SolverOptions,
    seed: u64,
) -> Result<FittedGlm, String> {
    let cols = x.first().ok_or("X must contain at least one row")?.len();
    check_features(x, cols)?;
    let family = options.family;
    let mass = check_response(y, weights, x.len(), family)?;
    let SolverOptions {
        l1,
        l2,
        tol,
        max_iter,
        ref monotone,
        ..
    } = *options;
    if [l1, l2].iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err("l1 and l2 must be non-negative and finite".into());
    }
    if !tol.is_finite() || tol <= 0.0 || max_iter == 0 {
        return Err("tol must be positive and finite and max_iter must be positive".into());
    }
    if !monotone.is_empty()
        && (monotone.len() != cols || monotone.iter().any(|s| !(-1..=1).contains(s)))
    {
        return Err("monotone must contain one of -1, 0, or 1 per feature".into());
    }
    match family {
        GlmFamily::Tweedie { p } if !(p > 1.0 && p < 2.0) => {
            return Err("tweedie_p must be strictly between 1 and 2".into());
        }
        GlmFamily::Tobit { threshold } if !threshold.is_finite() => {
            return Err("tobit_threshold must be finite".into());
        }
        _ => {}
    }
    let constrained = monotone.iter().any(|&s| s != 0);
    let direct =
        solver.is_none() || (solver == Some(LeafSolver::Irls) && family == GlmFamily::Gaussian);
    if direct && (family != GlmFamily::Gaussian || l1 > 0.0 || constrained) {
        return Err("The direct Gaussian solver supports Gaussian with L2 only; use ccd for constrained or L1 Gaussian fits".into());
    }
    if solver == Some(LeafSolver::Ccd) && family != GlmFamily::Gaussian {
        return Err("ccd supports only Gaussian fits".into());
    }
    if solver == Some(LeafSolver::Lbfgs) && (l1 > 0.0 || constrained) {
        return Err("lbfgs supports L2 only".into());
    }

    let design = DenseView::new(x, y, weights, true);
    let (beta, convergence, summary, scales) = if direct {
        let (beta, summary) = solve_normal(&design, l2, None)?;
        (
            beta,
            Convergence {
                iterations: 0,
                reason: StopReason::Exact,
            },
            summary,
            Vec::new(),
        )
    } else if solver == Some(LeafSolver::Irls) {
        let mut extra = std::collections::HashMap::new();
        let leaf_family = match family {
            GlmFamily::Gaussian => unreachable!(),
            GlmFamily::Binomial => LeafFamily::Binomial,
            GlmFamily::Probit => LeafFamily::Probit,
            GlmFamily::Poisson => LeafFamily::Poisson,
            GlmFamily::Gamma => LeafFamily::Gamma,
            GlmFamily::Tweedie { p } => {
                extra.insert("tweedie_p".into(), p);
                LeafFamily::Tweedie
            }
            GlmFamily::Tobit { threshold } => {
                extra.insert("tobit_threshold".into(), threshold);
                LeafFamily::Tobit
            }
        };
        let mut mean = (0..y.len())
            .map(|i| design.get_weight(i) * y[i])
            .sum::<f64>()
            / mass;
        if matches!(family, GlmFamily::Binomial | GlmFamily::Probit) {
            mean = mean.clamp(1e-12, 1.0 - 1e-12);
        }
        let chosen: Vec<_> = (0..cols).collect();
        let (model, report) = fit_glm_design(
            &design,
            &chosen,
            monotone,
            leaf_family,
            mean,
            Some(&extra),
            options,
        )?;
        let beta = std::iter::once(model.intercept).chain(model.coef).collect();
        let scales = if l1 > 0.0 {
            let gram = design.compute_gram(None);
            (1..=cols)
                .map(|j| {
                    let var = gram.xtx[j * (1 + cols) + j] - gram.xtx[j].powi(2) / gram.ws;
                    if var > 1e-12 {
                        var.sqrt()
                    } else {
                        1.0
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        (beta, report, None, scales)
    } else {
        let solver = solver.unwrap();
        let (beta, report) = solve_iterative(&design, solver, options, seed, None)?;
        let scales = if l1 > 0.0 && solver == LeafSolver::Ccd {
            (1..=cols)
                .map(|j| {
                    let scale = (design.col_norm_sq(j, None) / y.len() as f64).sqrt();
                    if scale > 1e-12 {
                        scale
                    } else {
                        1.0
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        (beta, report, None, scales)
    };

    let mut objective = 0.0;
    for i in 0..design.n_rows() {
        let w = design.get_weight(i);
        if w > 0.0 {
            let mean = family.mean_for_loss(design.eval_row(i, &beta));
            objective += 0.5 * w * family.deviance(y[i], mean);
        }
    }
    for (j, &b) in beta[1..].iter().enumerate() {
        let b = scales.get(j).copied().unwrap_or(1.0) * b;
        objective += l1 * b.abs() + 0.5 * l2 * b * b;
    }
    if !objective.is_finite() || beta.iter().any(|b| !b.is_finite()) {
        return Err("Fit produced a non-finite objective or coefficient".into());
    }
    Ok(FittedGlm {
        intercept: beta[0],
        coef: beta[1..].to_vec(),
        family,
        convergence,
        objective,
        summary,
    })
}
