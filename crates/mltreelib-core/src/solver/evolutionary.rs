//! Evolutionary and Derivative-Free Optimization Solvers.
//!
//! Provides top state-of-the-art global and continuous derivative-free optimization algorithms:
//! 1. `CmaEs`: Covariance Matrix Adaptation Evolution Strategy (Hansen & Ostermeier 2001).
//!    The gold standard for continuous non-convex and ill-conditioned loss landscapes.
//! 2. `Shade`: Success-History based Adaptive Differential Evolution with current-to-pbest/1
//!    and external archive (Tanabe & Fukunaga 2013, IEEE CEC multi-year competition winner).
//! 3. `NaturalGradient`: Separable Natural Evolution Strategies (sNES / NGL, Wierstra et al. 2014).
//!    Natural gradient updates over coordinate center and diagonal scales; $O(P)$ time per generation.
//! 4. `GeneticAlgorithm`: Real-Coded Genetic Algorithm (RCGA) with tournament selection,
//!    Simulated Binary Crossover (SBX), polynomial mutation, and strict elitism (Deb & Agrawal 1995).

use crate::solver::design::DesignMatrix;
use crate::solver::family::GlmFamily;

/// Deterministic 64-bit XorShift PRNG for reproducible evolutionary runs across platforms.
#[derive(Clone, Debug)]
pub struct Rng64 {
    state: u64,
}

impl Rng64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x853c_49e6_748f_ea9b
            } else {
                seed ^ 0x517c_c1b7_2722_0a95
            },
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Box-Muller transform for standard normal variable $\mathcal{N}(0, 1)$.
    #[inline]
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Standard Cauchy random variable $X \sim \text{Cauchy}(x_0, \gamma)$.
    #[inline]
    pub fn next_cauchy(&mut self, x0: f64, gamma: f64) -> f64 {
        let u = self.next_f64();
        x0 + gamma * (std::f64::consts::PI * (u - 0.5)).tan()
    }
}

/// Derivative-free optimization algorithm variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvolutionaryKind {
    /// Covariance Matrix Adaptation Evolution Strategy (Hansen & Ostermeier 2001).
    CmaEs,
    /// Success-History Based Adaptive Differential Evolution (Tanabe & Fukunaga 2013).
    Shade,
    /// Separable Natural Evolution Strategies / Natural Gradient Learning (Wierstra et al. 2014).
    NaturalGradient,
    /// Real-Coded Genetic Algorithm with SBX and polynomial mutation (Deb & Agrawal 1995).
    GeneticAlgorithm,
}

/// Options controlling evolutionary optimization.
#[derive(Clone, Debug)]
pub struct EvolutionaryOptions {
    pub kind: EvolutionaryKind,
    pub family: GlmFamily,
    pub l1: f64,
    pub l2: f64,
    pub monotone: Vec<i8>,
    pub population_size: Option<usize>,
    pub max_generations: usize,
    pub tol: f64,
    pub seed: u64,
    pub verbose: u8,
}

impl Default for EvolutionaryOptions {
    fn default() -> Self {
        Self {
            kind: EvolutionaryKind::CmaEs,
            family: GlmFamily::Gaussian,
            l1: 0.0,
            l2: 1e-4,
            monotone: Vec::new(),
            population_size: None,
            max_generations: 100,
            tol: 1e-6,
            seed: 42,
            verbose: 0,
        }
    }
}

/// Evaluates objective value (deviance + penalties) and applies monotone projection.
#[inline]
pub fn eval_fitness<D: DesignMatrix>(
    design: &D,
    family: &GlmFamily,
    beta: &mut [f64],
    l1: f64,
    l2: f64,
    monotone: &[i8],
) -> f64 {
    let p = beta.len();
    for j in 1..p {
        let mono = monotone.get(j.saturating_sub(1)).copied().unwrap_or(0);
        match mono {
            1 => {
                if beta[j] < 0.0 {
                    beta[j] = 0.0;
                }
            }
            -1 => {
                if beta[j] > 0.0 {
                    beta[j] = 0.0;
                }
            }
            _ => {}
        }
    }

    let n = design.n_rows();
    let mut total_loss = 0.0;
    for i in 0..n {
        let wi = design.get_weight(i);
        if wi <= 0.0 {
            continue;
        }
        let yi = design.get_y(i);
        let eta = design.eval_row(i, beta);
        let mu = family.mean_for_loss(eta);
        let dev = family.deviance(yi, mu);
        total_loss += 0.5 * wi * dev;
    }

    if l1 > 0.0 {
        for j in 1..p {
            total_loss += l1 * beta[j].abs();
        }
    }
    if l2 > 0.0 {
        for j in 1..p {
            total_loss += 0.5 * l2 * beta[j] * beta[j];
        }
    }

    if total_loss.is_finite() {
        total_loss
    } else {
        1e20
    }
}

/// Unified entry point for evolutionary optimization over a `DesignMatrix`.
pub fn solve_evolutionary<D: DesignMatrix>(
    design: &D,
    options: &EvolutionaryOptions,
    warm_start: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    solve_evolutionary_report(design, options, warm_start).map(|(beta, _)| beta)
}

pub(super) fn solve_evolutionary_report<D: DesignMatrix>(
    design: &D,
    options: &EvolutionaryOptions,
    warm_start: Option<&[f64]>,
) -> Result<(Vec<f64>, super::Convergence), String> {
    let p = design.n_features();
    if p == 0 {
        return Err("Zero features in design matrix".to_string());
    }

    match options.kind {
        EvolutionaryKind::CmaEs => solve_cmaes_report(design, options, warm_start),
        EvolutionaryKind::Shade => solve_shade_report(design, options, warm_start),
        EvolutionaryKind::NaturalGradient => solve_snes_report(design, options, warm_start),
        EvolutionaryKind::GeneticAlgorithm => solve_rcga_report(design, options, warm_start),
    }
}

/// Covariance Matrix Adaptation Evolution Strategy (Hansen & Ostermeier 2001).
pub fn solve_cmaes<D: DesignMatrix>(
    design: &D,
    options: &EvolutionaryOptions,
    warm_start: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    solve_cmaes_report(design, options, warm_start).map(|(beta, _)| beta)
}

pub(super) fn solve_cmaes_report<D: DesignMatrix>(
    design: &D,
    options: &EvolutionaryOptions,
    warm_start: Option<&[f64]>,
) -> Result<(Vec<f64>, super::Convergence), String> {
    let n = p_dim(design);
    let mut rng = Rng64::new(options.seed);

    let mut m = vec![0.0; n];
    if let Some(warm) = warm_start {
        let copy_len = n.min(warm.len());
        m[..copy_len].copy_from_slice(&warm[..copy_len]);
    } else {
        m[0] = initial_intercept(design, &options.family);
    }

    let lambda = options
        .population_size
        .unwrap_or_else(|| 4 + (3.0 * (n as f64).ln()).floor() as usize);
    let mu = (lambda / 2).max(1);

    let mut raw_weights = Vec::with_capacity(mu);
    for i in 1..=mu {
        let w = ((mu as f64) + 0.5).ln() - (i as f64).ln();
        raw_weights.push(w.max(0.0));
    }
    let sum_w: f64 = raw_weights.iter().sum();
    let weights: Vec<f64> = raw_weights.iter().map(|w| w / sum_w).collect();
    let mueff: f64 = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

    let cc = (4.0 + mueff / (n as f64)) / ((n as f64) + 4.0 + 2.0 * mueff / (n as f64));
    let cs = (mueff + 2.0) / ((n as f64) + mueff + 5.0);
    let c1 = 2.0 / (((n as f64) + 1.3).powi(2) + mueff);
    let cmu = (1.0 - c1).min(
        2.0 * (mueff - 2.0 + 1.0 / mueff) / (((n as f64) + 2.0).powi(2) + mueff),
    );
    let damps = 1.0
        + 2.0 * (0.0f64).max(((mueff - 1.0) / ((n as f64) + 1.0)).sqrt() - 1.0)
        + cs;
    let chi_n = (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * (n as f64)) + 1.0 / (21.0 * (n as f64).powi(2)));

    let mut sigma = 0.5f64;
    let mut pc = vec![0.0; n];
    let mut ps = vec![0.0; n];
    let mut cov = faer::Mat::<f64>::zeros(n, n);
    for i in 0..n {
        cov[(i, i)] = 1.0;
    }

    let mut best_solution = m.clone();
    let mut best_fitness = eval_fitness(
        design,
        &options.family,
        &mut best_solution,
        options.l1,
        options.l2,
        &options.monotone,
    );

    let mut candidates: Vec<(Vec<f64>, Vec<f64>, f64)> = (0..lambda)
        .map(|_| (vec![0.0; n], vec![0.0; n], 0.0))
        .collect();

    for gen in 0..options.max_generations {
        let l_mat = cholesky_factor(&cov, n);

        for (x_k, y_k, fit_k) in candidates.iter_mut() {
            let mut z = vec![0.0; n];
            for j in 0..n {
                z[j] = rng.next_gaussian();
            }
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..=i {
                    sum += l_mat[(i, j)] * z[j];
                }
                y_k[i] = sum;
                x_k[i] = m[i] + sigma * sum;
            }
            *fit_k = eval_fitness(
                design,
                &options.family,
                x_k,
                options.l1,
                options.l2,
                &options.monotone,
            );
        }

        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        if candidates[0].2 < best_fitness {
            best_fitness = candidates[0].2;
            best_solution = candidates[0].0.clone();
        }

        let mut m_new = vec![0.0; n];
        let mut y_w = vec![0.0; n];
        for (i, &w_i) in weights.iter().enumerate() {
            for j in 0..n {
                m_new[j] += w_i * candidates[i].0[j];
                y_w[j] += w_i * candidates[i].1[j];
            }
        }

        let ps_factor = (cs * (2.0 - cs) * mueff).sqrt();
        for j in 0..n {
            ps[j] = (1.0 - cs) * ps[j] + ps_factor * y_w[j];
        }
        let ps_norm = ps.iter().map(|v| v * v).sum::<f64>().sqrt();

        let hsig = if ps_norm / (1.0 - (1.0 - cs).powi(2 * (gen as i32 + 1))).sqrt()
            < (1.4 + 2.0 / ((n as f64) + 1.0)) * chi_n
        {
            1.0
        } else {
            0.0
        };

        let pc_factor = (cc * (2.0 - cc) * mueff).sqrt();
        for j in 0..n {
            pc[j] = (1.0 - cc) * pc[j] + hsig * pc_factor * y_w[j];
        }

        let d_hsig = (1.0 - hsig) * cc * (2.0 - cc);
        for i in 0..n {
            for j in 0..n {
                let mut rank_mu = 0.0;
                for (k, &w_k) in weights.iter().enumerate() {
                    rank_mu += w_k * candidates[k].1[i] * candidates[k].1[j];
                }
                cov[(i, j)] = (1.0 - c1 - cmu + c1 * d_hsig) * cov[(i, j)]
                    + c1 * pc[i] * pc[j]
                    + cmu * rank_mu;
            }
        }

        sigma *= ((cs / damps) * (ps_norm / chi_n - 1.0)).exp();
        m = m_new;

        if options.verbose >= 3 && (gen % 10 == 0 || gen == options.max_generations - 1) {
            println!(
                "      [CMA-ES gen {}] best_fit={:.6} sigma={:.4e}",
                gen, best_fitness, sigma
            );
        }

        if sigma < options.tol
            || (candidates.last().unwrap().2 - candidates[0].2).abs() < options.tol
        {
            return Ok((best_solution, super::Convergence::tolerance(gen + 1)));
        }
    }

    Ok((
        best_solution,
        super::Convergence::limit(options.max_generations),
    ))
}

/// Success-History Based Adaptive Differential Evolution (SHADE).
///
/// Implements `current-to-pbest/1` mutation with external archive, success memories $M_F, M_{CR}$,
/// and Lehmer mean parameter adaptation (Tanabe & Fukunaga 2013).
pub fn solve_shade<D: DesignMatrix>(
    design: &D,
    options: &EvolutionaryOptions,
    warm_start: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    solve_shade_report(design, options, warm_start).map(|(beta, _)| beta)
}

pub(super) fn solve_shade_report<D: DesignMatrix>(
    design: &D,
    options: &EvolutionaryOptions,
    warm_start: Option<&[f64]>,
) -> Result<(Vec<f64>, super::Convergence), String> {
    let n = p_dim(design);
    let mut rng = Rng64::new(options.seed);

    let pop_size = options.population_size.unwrap_or(25).max(6);
    let mut pop: Vec<Vec<f64>> = Vec::with_capacity(pop_size);

    let mut ind0 = vec![0.0; n];
    if let Some(warm) = warm_start {
        let copy_len = n.min(warm.len());
        ind0[..copy_len].copy_from_slice(&warm[..copy_len]);
    } else {
        ind0[0] = initial_intercept(design, &options.family);
    }
    pop.push(ind0);

    for _ in 1..pop_size {
        let mut ind = vec![0.0; n];
        ind[0] = pop[0][0] + rng.next_gaussian() * 0.5;
        for j in 1..n {
            ind[j] = rng.next_gaussian() * 0.5;
        }
        pop.push(ind);
    }

    let mut fitness: Vec<f64> = vec![0.0; pop_size];
    for i in 0..pop_size {
        fitness[i] = eval_fitness(
            design,
            &options.family,
            &mut pop[i],
            options.l1,
            options.l2,
            &options.monotone,
        );
    }

    let h_mem = 6;
    let mut m_cr = vec![0.5f64; h_mem];
    let mut m_f = vec![0.5f64; h_mem];
    let mut k_mem = 0;

    let mut archive: Vec<Vec<f64>> = Vec::with_capacity(pop_size);
    let p_best_rate = 0.15f64;
    let mut best_solution = pop[0].clone();
    let mut best_fitness = fitness[0];

    for i in 0..pop_size {
        if fitness[i] < best_fitness {
            best_fitness = fitness[i];
            best_solution = pop[i].clone();
        }
    }

    for gen in 0..options.max_generations {
        let mut s_cr = Vec::new();
        let mut s_f = Vec::new();
        let mut delta_f = Vec::new();

        let mut pbest_order: Vec<usize> = (0..pop_size).collect();
        pbest_order.sort_by(|&a, &b| fitness[a].partial_cmp(&fitness[b]).unwrap_or(std::cmp::Ordering::Equal));

        let p_best_count = ((pop_size as f64) * p_best_rate).ceil().max(1.0) as usize;

        for i in 0..pop_size {
            let r_mem = (rng.next_u64() as usize) % h_mem;
            let cr_i = (rng.next_gaussian() * 0.1 + m_cr[r_mem]).clamp(0.0, 1.0);

            let mut f_i = rng.next_cauchy(m_f[r_mem], 0.1);
            let mut tries = 0;
            while f_i <= 0.0 && tries < 10 {
                f_i = rng.next_cauchy(m_f[r_mem], 0.1);
                tries += 1;
            }
            let f_i = f_i.clamp(0.01, 1.0);

            let pbest_idx = pbest_order[(rng.next_u64() as usize) % p_best_count];
            let x_pbest = &pop[pbest_idx];

            let mut r1 = (rng.next_u64() as usize) % pop_size;
            while r1 == i {
                r1 = (rng.next_u64() as usize) % pop_size;
            }
            let x_r1 = &pop[r1];

            let total_union = pop_size + archive.len();
            let mut r2 = (rng.next_u64() as usize) % total_union;
            while r2 == i || r2 == r1 {
                r2 = (rng.next_u64() as usize) % total_union;
            }
            let x_r2 = if r2 < pop_size {
                &pop[r2]
            } else {
                &archive[r2 - pop_size]
            };

            let mut trial = vec![0.0; n];
            let j_rand = (rng.next_u64() as usize) % n;
            for j in 0..n {
                if rng.next_f64() < cr_i || j == j_rand {
                    let val = pop[i][j] + f_i * (x_pbest[j] - pop[i][j]) + f_i * (x_r1[j] - x_r2[j]);
                    trial[j] = val;
                } else {
                    trial[j] = pop[i][j];
                }
            }

            let trial_fit = eval_fitness(
                design,
                &options.family,
                &mut trial,
                options.l1,
                options.l2,
                &options.monotone,
            );

            if trial_fit <= fitness[i] {
                let diff = fitness[i] - trial_fit;
                s_cr.push(cr_i);
                s_f.push(f_i);
                delta_f.push(diff);

                if archive.len() < pop_size {
                    archive.push(pop[i].clone());
                } else {
                    let replace_idx = (rng.next_u64() as usize) % pop_size;
                    archive[replace_idx] = pop[i].clone();
                }

                pop[i] = trial;
                fitness[i] = trial_fit;

                if trial_fit < best_fitness {
                    best_fitness = trial_fit;
                    best_solution = pop[i].clone();
                }
            }
        }

        if !s_cr.is_empty() {
            let sum_df: f64 = delta_f.iter().sum();
            if sum_df > 1e-12 {
                let weights: Vec<f64> = delta_f.iter().map(|&df| df / sum_df).collect();
                let mean_cr = weights.iter().zip(&s_cr).map(|(w, cr)| w * cr).sum::<f64>();
                m_cr[k_mem] = mean_cr.clamp(0.0, 1.0);

                let sum_w_f2: f64 = weights.iter().zip(&s_f).map(|(w, f)| w * f * f).sum();
                let sum_w_f: f64 = weights.iter().zip(&s_f).map(|(w, f)| w * f).sum();
                if sum_w_f > 1e-12 {
                    m_f[k_mem] = (sum_w_f2 / sum_w_f).clamp(0.01, 1.0);
                }

                k_mem = (k_mem + 1) % h_mem;
            }
        }

        if options.verbose >= 3 && (gen % 10 == 0 || gen == options.max_generations - 1) {
            println!(
                "      [SHADE gen {}] best_fit={:.6} archive_sz={}",
                gen, best_fitness, archive.len()
            );
        }
    }

    Ok((
        best_solution,
        super::Convergence::limit(options.max_generations),
    ))
}

/// Separable Natural Evolution Strategies (sNES / Natural Gradient Learning).
pub fn solve_snes<D: DesignMatrix>(
    design: &D,
    options: &EvolutionaryOptions,
    warm_start: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    solve_snes_report(design, options, warm_start).map(|(beta, _)| beta)
}

pub(super) fn solve_snes_report<D: DesignMatrix>(
    design: &D,
    options: &EvolutionaryOptions,
    warm_start: Option<&[f64]>,
) -> Result<(Vec<f64>, super::Convergence), String> {
    let n = p_dim(design);
    let mut rng = Rng64::new(options.seed);

    let mut m = vec![0.0; n];
    if let Some(warm) = warm_start {
        let copy_len = n.min(warm.len());
        m[..copy_len].copy_from_slice(&warm[..copy_len]);
    } else {
        m[0] = initial_intercept(design, &options.family);
    }

    let mut sigma = vec![0.5f64; n];
    let lambda = options
        .population_size
        .unwrap_or_else(|| 4 + (3.0 * (n as f64).ln()).floor() as usize);

    let mu = lambda / 2;
    let mut raw_u = vec![0.0; lambda];
    for i in 0..mu {
        raw_u[i] = ((mu as f64) + 0.5).ln() - ((i + 1) as f64).ln();
    }
    let u_sum: f64 = raw_u.iter().sum();
    let u: Vec<f64> = raw_u.iter().map(|v| v / u_sum.max(1e-12)).collect();

    let eta_m = 1.0;
    let eta_s = (3.0 + (n as f64).ln()) / (5.0 * (n as f64).sqrt());

    let mut best_solution = m.clone();
    let mut best_fitness = eval_fitness(
        design,
        &options.family,
        &mut best_solution,
        options.l1,
        options.l2,
        &options.monotone,
    );

    let mut candidates: Vec<(Vec<f64>, Vec<f64>, f64)> = (0..lambda)
        .map(|_| (vec![0.0; n], vec![0.0; n], 0.0))
        .collect();

    for gen in 0..options.max_generations {
        for (x_k, z_k, fit_k) in candidates.iter_mut() {
            for j in 0..n {
                z_k[j] = rng.next_gaussian();
                x_k[j] = m[j] + sigma[j] * z_k[j];
            }
            *fit_k = eval_fitness(
                design,
                &options.family,
                x_k,
                options.l1,
                options.l2,
                &options.monotone,
            );
        }

        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        if candidates[0].2 < best_fitness {
            best_fitness = candidates[0].2;
            best_solution = candidates[0].0.clone();
        }

        let mut grad_m = vec![0.0; n];
        let mut grad_s = vec![0.0; n];
        for (i, &u_i) in u.iter().enumerate() {
            if u_i <= 0.0 {
                continue;
            }
            let (_, z_i, _) = &candidates[i];
            for j in 0..n {
                grad_m[j] += u_i * z_i[j];
                grad_s[j] += u_i * (z_i[j] * z_i[j] - 1.0);
            }
        }

        let mut max_step = 0.0f64;
        for j in 0..n {
            let delta_m = eta_m * sigma[j] * grad_m[j];
            m[j] += delta_m;
            sigma[j] *= (0.5 * eta_s * grad_s[j]).exp().clamp(0.1, 10.0);
            max_step = max_step.max(delta_m.abs());
        }

        if options.verbose >= 3 && (gen % 10 == 0 || gen == options.max_generations - 1) {
            println!(
                "      [sNES gen {}] best_fit={:.6} max_step={:.4e}",
                gen, best_fitness, max_step
            );
        }

        if max_step < options.tol {
            return Ok((best_solution, super::Convergence::tolerance(gen + 1)));
        }
    }

    Ok((
        best_solution,
        super::Convergence::limit(options.max_generations),
    ))
}

/// Real-Coded Genetic Algorithm (RCGA) with SBX crossover and polynomial mutation.
pub fn solve_rcga<D: DesignMatrix>(
    design: &D,
    options: &EvolutionaryOptions,
    warm_start: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    solve_rcga_report(design, options, warm_start).map(|(beta, _)| beta)
}

pub(super) fn solve_rcga_report<D: DesignMatrix>(
    design: &D,
    options: &EvolutionaryOptions,
    warm_start: Option<&[f64]>,
) -> Result<(Vec<f64>, super::Convergence), String> {
    let n = p_dim(design);
    let mut rng = Rng64::new(options.seed);

    let pop_size = options.population_size.unwrap_or(40).max(10);
    let mut population: Vec<Vec<f64>> = Vec::with_capacity(pop_size);

    let mut ind0 = vec![0.0; n];
    if let Some(warm) = warm_start {
        let copy_len = n.min(warm.len());
        ind0[..copy_len].copy_from_slice(&warm[..copy_len]);
    } else {
        ind0[0] = initial_intercept(design, &options.family);
    }
    population.push(ind0);

    for _ in 1..pop_size {
        let mut ind = vec![0.0; n];
        ind[0] = population[0][0] + rng.next_gaussian() * 0.5;
        for j in 1..n {
            ind[j] = rng.next_gaussian() * 0.5;
        }
        population.push(ind);
    }

    let mut fitness: Vec<f64> = vec![0.0; pop_size];
    for i in 0..pop_size {
        fitness[i] = eval_fitness(
            design,
            &options.family,
            &mut population[i],
            options.l1,
            options.l2,
            &options.monotone,
        );
    }

    let eta_c = 20.0;
    let eta_m = 20.0;
    let mut best_solution = population[0].clone();
    let mut best_fitness = fitness[0];

    for gen in 0..options.max_generations {
        let mut next_pop: Vec<Vec<f64>> = Vec::with_capacity(pop_size);

        let mut order: Vec<usize> = (0..pop_size).collect();
        order.sort_by(|&a, &b| fitness[a].partial_cmp(&fitness[b]).unwrap_or(std::cmp::Ordering::Equal));
        next_pop.push(population[order[0]].clone());
        next_pop.push(population[order[1]].clone());

        if fitness[order[0]] < best_fitness {
            best_fitness = fitness[order[0]];
            best_solution = population[order[0]].clone();
        }

        while next_pop.len() < pop_size {
            let p1 = tournament_select(&fitness, 3, &mut rng);
            let p2 = tournament_select(&fitness, 3, &mut rng);

            let mut c1 = population[p1].clone();
            let mut c2 = population[p2].clone();

            if rng.next_f64() < 0.9 {
                for j in 0..n {
                    if rng.next_f64() <= 0.5 {
                        let y1 = c1[j].min(c2[j]);
                        let y2 = c1[j].max(c2[j]);
                        if (y2 - y1).abs() > 1e-12 {
                            let u = rng.next_f64();
                            let beta = if u <= 0.5 {
                                (2.0 * u).powf(1.0 / (eta_c + 1.0))
                            } else {
                                (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta_c + 1.0))
                            };
                            c1[j] = 0.5 * ((y1 + y2) - beta * (y2 - y1));
                            c2[j] = 0.5 * ((y1 + y2) + beta * (y2 - y1));
                        }
                    }
                }
            }

            let mut_rate = 1.0 / (n as f64);
            for ind in [&mut c1, &mut c2] {
                for j in 0..n {
                    if rng.next_f64() < mut_rate {
                        let u = rng.next_f64();
                        let delta = if u < 0.5 {
                            (2.0 * u).powf(1.0 / (eta_m + 1.0)) - 1.0
                        } else {
                            1.0 - (2.0 * (1.0 - u)).powf(1.0 / (eta_m + 1.0))
                        };
                        ind[j] += delta * 0.5;
                    }
                }
            }

            next_pop.push(c1);
            if next_pop.len() < pop_size {
                next_pop.push(c2);
            }
        }

        population = next_pop;
        for i in 0..pop_size {
            fitness[i] = eval_fitness(
                design,
                &options.family,
                &mut population[i],
                options.l1,
                options.l2,
                &options.monotone,
            );
        }

        if options.verbose >= 3 && (gen % 20 == 0 || gen == options.max_generations - 1) {
            println!(
                "      [RCGA gen {}] best_fit={:.6}",
                gen, best_fitness
            );
        }
    }

    Ok((
        best_solution,
        super::Convergence::limit(options.max_generations),
    ))
}

fn tournament_select(fitness: &[f64], k: usize, rng: &mut Rng64) -> usize {
    let mut best_idx = (rng.next_u64() as usize) % fitness.len();
    let mut best_fit = fitness[best_idx];
    for _ in 1..k {
        let idx = (rng.next_u64() as usize) % fitness.len();
        if fitness[idx] < best_fit {
            best_fit = fitness[idx];
            best_idx = idx;
        }
    }
    best_idx
}

#[inline]
fn p_dim<D: DesignMatrix>(design: &D) -> usize {
    design.n_features()
}

fn initial_intercept<D: DesignMatrix>(design: &D, family: &GlmFamily) -> f64 {
    let n = design.n_rows();
    let mut sum_y = 0.0;
    let mut sum_w = 0.0;
    for i in 0..n {
        let w = design.get_weight(i);
        sum_y += w * design.get_y(i);
        sum_w += w;
    }
    let y_bar = if sum_w > 0.0 { sum_y / sum_w } else { 0.0 };
    family.initial_intercept(y_bar)
}

fn cholesky_factor(cov: &faer::Mat<f64>, n: usize) -> faer::Mat<f64> {
    let mut l = faer::Mat::<f64>::zeros(n, n);
    let mut jitter = 0.0f64;
    for attempt in 0..5 {
        let mut success = true;
        for j in 0..n {
            let mut sum_sq = 0.0;
            for k in 0..j {
                sum_sq += l[(j, k)] * l[(j, k)];
            }
            let diag_val = cov[(j, j)] + jitter - sum_sq;
            if diag_val <= 1e-14 {
                success = false;
                break;
            }
            let l_jj = diag_val.sqrt();
            l[(j, j)] = l_jj;
            for i in (j + 1)..n {
                let mut sum_cross = 0.0;
                for k in 0..j {
                    sum_cross += l[(i, k)] * l[(j, k)];
                }
                l[(i, j)] = (cov[(i, j)] - sum_cross) / l_jj;
            }
        }
        if success {
            return l;
        }
        jitter = if attempt == 0 { 1e-6 } else { jitter * 10.0 };
    }
    let mut eye = faer::Mat::<f64>::zeros(n, n);
    for i in 0..n {
        eye[(i, i)] = 1.0;
    }
    eye
}
