//! First-Class Batch Population Ask/Tell Optimization Engine.
//!
//! Provides a state machine interface for population-based derivative-free and
//! multi-objective optimization algorithms:
//! 1. `CmaEs`: Covariance Matrix Adaptation Evolution Strategy with correct
//!    whitened covariance-scale path, box bounds, and IPOP restarts.
//! 2. `De`: Differential Evolution (`de/rand/1/bin`) with binomial crossover.
//! 3. `Shade`: Success-History based Adaptive Differential Evolution.
//! 4. `Pso`: Particle Swarm Optimization with Clerc-Kennedy constriction.
//! 5. `Snes`: Separable Natural Evolution Strategies with coordinate updates.
//! 6. `Rcga`: Real-Coded Genetic Algorithm with SBX and polynomial mutation.
//! 7. `Nsga2`: Non-dominated Sorting Genetic Algorithm II for multi-objective Pareto front discovery.
//! 8. `Moead`: Multi-Objective Evolutionary Algorithm based on Decomposition (Tchebycheff).
//! 9. `Nsga3`: Reference-Direction Non-dominated Sorting GA for Many-Objective ($M \ge 3$) optimization.
//! 10. `Island`: Multi-Island parallel model with periodic elite ring migration.

use nalgebra::{DMatrix, DVector};
use crate::solver::evolutionary::Rng64;
use crate::solver::generic::result::SolveStatus;

/// Supported batch population optimizer kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AskTellKind {
    CmaEs,
    De,
    Shade,
    Pso,
    Snes,
    Rcga,
    Nsga2,
    Moead,
    Nsga3,
    Island,
}

impl AskTellKind {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().replace('-', "_").as_str() {
            "cmaes" | "cma_es" => Some(Self::CmaEs),
            "de" | "differential_evolution" => Some(Self::De),
            "shade" => Some(Self::Shade),
            "pso" | "particle_swarm" => Some(Self::Pso),
            "snes" | "natural_gradient" => Some(Self::Snes),
            "rcga" | "ga" | "genetic_algorithm" => Some(Self::Rcga),
            "nsga2" | "nsga_2" | "nsga" => Some(Self::Nsga2),
            "moead" | "moea_d" => Some(Self::Moead),
            "nsga3" | "nsga_3" => Some(Self::Nsga3),
            "island" | "island_model" => Some(Self::Island),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CmaEs => "cmaes",
            Self::De => "de",
            Self::Shade => "shade",
            Self::Pso => "pso",
            Self::Snes => "snes",
            Self::Rcga => "rcga",
            Self::Nsga2 => "nsga2",
            Self::Moead => "moead",
            Self::Nsga3 => "nsga3",
            Self::Island => "island",
        }
    }
}

/// Options for configuring an `AskTell` optimizer instance.
#[derive(Clone, Debug)]
pub struct AskTellOptions {
    pub population_size: Option<usize>,
    pub seed: Option<u64>,
    pub max_iterations: Option<usize>,
    pub max_evaluations: Option<usize>,
    pub tol: Option<f64>,
    pub ipop: bool,
    pub max_restarts: usize,
    pub de_f: Option<f64>,
    pub de_cr: Option<f64>,
    pub pso_w: Option<f64>,
    pub pso_c1: Option<f64>,
    pub pso_c2: Option<f64>,
    pub moead_t: Option<usize>,
    pub nsga3_partitions: Option<usize>,
    pub island_count: Option<usize>,
    pub migration_interval: Option<usize>,
    pub migration_rate: Option<usize>,
}

impl Default for AskTellOptions {
    fn default() -> Self {
        Self {
            population_size: None,
            seed: Some(42),
            max_iterations: Some(200),
            max_evaluations: Some(2000),
            tol: Some(1e-8),
            ipop: false,
            max_restarts: 4,
            de_f: Some(0.8),
            de_cr: Some(0.9),
            pso_w: Some(0.7298),
            pso_c1: Some(1.49618),
            pso_c2: Some(1.49618),
            moead_t: Some(5),
            nsga3_partitions: None,
            island_count: Some(4),
            migration_interval: Some(10),
            migration_rate: Some(1),
        }
    }
}

/// A non-dominated Pareto solution with its parameter point and objective vector.
#[derive(Clone, Debug)]
pub struct ParetoSolution {
    pub point: Vec<f64>,
    pub objectives: Vec<f64>,
}

/// Batch Population Ask/Tell State Machine.
pub struct AskTell {
    kind: AskTellKind,
    bounds: Vec<(f64, f64)>,
    dim: usize,
    options: AskTellOptions,
    rng: Rng64,
    iterations: usize,
    evaluations: usize,
    status: SolveStatus,
    should_stop: bool,
    restart_count: usize,
    current_candidates: Vec<Vec<f64>>,
    best_x: Vec<f64>,
    best_value: f64,
    state: InternalState,
}

enum InternalState {
    CmaEs(CmaEsState),
    De(DeState),
    Shade(ShadeState),
    Pso(PsoState),
    Snes(SnesState),
    Rcga(RcgaState),
    Nsga2(Nsga2State),
    Moead(MoeadState),
    Nsga3(Nsga3State),
    Island(IslandState),
}

// ---------------------------------------------------------------------------
// 1. CMA-ES State Machine
// ---------------------------------------------------------------------------

struct CmaEsState {
    lambda: usize,
    mu: usize,
    weights: Vec<f64>,
    mueff: f64,
    cc: f64,
    cs: f64,
    c1: f64,
    cmu: f64,
    damps: f64,
    chi_n: f64,
    m: Vec<f64>,
    sigma: f64,
    pc: Vec<f64>,
    ps: Vec<f64>,
    cov: DMatrix<f64>,
    b_mat: DMatrix<f64>,
    d_vec: DVector<f64>,
    last_z: Vec<Vec<f64>>,
    last_y: Vec<Vec<f64>>,
}

impl CmaEsState {
    fn new(dim: usize, bounds: &[(f64, f64)], lambda_opt: Option<usize>, rng: &mut Rng64) -> Self {
        let n = dim;
        let lambda = lambda_opt.unwrap_or_else(|| 4 + (3.0 * (n as f64).ln()).floor() as usize).max(4);
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
        let cmu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / (((n as f64) + 2.0).powi(2) + mueff));
        let damps = 1.0 + 2.0 * (0.0f64).max(((mueff - 1.0) / ((n as f64) + 1.0)).sqrt() - 1.0) + cs;
        let chi_n = (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * (n as f64)) + 1.0 / (21.0 * (n as f64).powi(2)));

        let mut m = Vec::with_capacity(n);
        let mut min_span = f64::INFINITY;
        for &(lo, hi) in bounds {
            let span = (hi - lo).abs();
            if span > 0.0 && span < min_span {
                min_span = span;
            }
            m.push(lo + (hi - lo) * rng.next_f64());
        }
        let sigma = if min_span.is_finite() && min_span > 0.0 {
            0.3 * min_span
        } else {
            0.5
        };

        let pc = vec![0.0; n];
        let ps = vec![0.0; n];
        let cov = DMatrix::identity(n, n);
        let b_mat = DMatrix::identity(n, n);
        let d_vec = DVector::from_element(n, 1.0);

        Self {
            lambda,
            mu,
            weights,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            chi_n,
            m,
            sigma,
            pc,
            ps,
            cov,
            b_mat,
            d_vec,
            last_z: Vec::new(),
            last_y: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Differential Evolution (DE / rand / 1 / bin) State Machine
// ---------------------------------------------------------------------------

struct DeState {
    pop_size: usize,
    pop: Vec<Vec<f64>>,
    fitness: Vec<f64>,
    f: f64,
    cr: f64,
    trial_mutants: Vec<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// 3. SHADE State Machine
// ---------------------------------------------------------------------------

struct ShadeState {
    pop_size: usize,
    pop: Vec<Vec<f64>>,
    fitness: Vec<f64>,
    archive: Vec<Vec<f64>>,
    archive_max: usize,
    m_f: Vec<f64>,
    m_cr: Vec<f64>,
    k: usize,
    memory_size: usize,
    trial_f: Vec<f64>,
    trial_cr: Vec<f64>,
    trial_pop: Vec<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// 4. Particle Swarm Optimization (PSO) State Machine
// ---------------------------------------------------------------------------

struct PsoState {
    pop_size: usize,
    x: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    p_best: Vec<Vec<f64>>,
    p_best_fit: Vec<f64>,
    g_best: Vec<f64>,
    g_best_fit: f64,
    w: f64,
    c1: f64,
    c2: f64,
    v_max: Vec<f64>,
}

// ---------------------------------------------------------------------------
// 5. Separable Natural Evolution Strategies (sNES) State Machine
// ---------------------------------------------------------------------------

struct SnesState {
    pop_size: usize,
    mu: Vec<f64>,
    sigma: Vec<f64>,
    eta_mu: f64,
    eta_sigma: f64,
    last_z: Vec<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// 6. Real-Coded Genetic Algorithm (RCGA) State Machine
// ---------------------------------------------------------------------------

struct RcgaState {
    pop_size: usize,
    pop: Vec<Vec<f64>>,
    fitness: Vec<f64>,
    eta_c: f64,
    eta_m: f64,
}

// ---------------------------------------------------------------------------
// 7. Multi-Objective NSGA-II State Machine
// ---------------------------------------------------------------------------

struct Nsga2State {
    pop_size: usize,
    parent_pop: Vec<Vec<f64>>,
    parent_objs: Vec<Vec<f64>>,
    offspring_pop: Vec<Vec<f64>>,
    is_initial: bool,
    pareto_front: Vec<ParetoSolution>,
}

// ---------------------------------------------------------------------------
// 8. Multi-Objective MOEA/D State Machine
// ---------------------------------------------------------------------------

struct MoeadState {
    pop_size: usize,
    weights: Vec<Vec<f64>>,
    neighborhoods: Vec<Vec<usize>>,
    t_size: usize,
    current_pop: Vec<Vec<f64>>,
    current_objs: Vec<Vec<f64>>,
    ideal_point: Vec<f64>,
    offspring_pop: Vec<Vec<f64>>,
    is_initial: bool,
    pareto_front: Vec<ParetoSolution>,
}

// ---------------------------------------------------------------------------
// 9. Many-Objective NSGA-III State Machine
// ---------------------------------------------------------------------------

struct Nsga3State {
    pop_size: usize,
    parent_pop: Vec<Vec<f64>>,
    parent_objs: Vec<Vec<f64>>,
    offspring_pop: Vec<Vec<f64>>,
    ref_points: Vec<Vec<f64>>,
    partitions: Option<usize>,
    is_initial: bool,
    pareto_front: Vec<ParetoSolution>,
}

// ---------------------------------------------------------------------------
// 10. Multi-Island Population State Machine
// ---------------------------------------------------------------------------

struct IslandState {
    islands: Vec<AskTell>,
    _island_count: usize,
    migration_interval: usize,
    _migration_rate: usize,
    batch_sizes: Vec<usize>,
    pareto_front: Vec<ParetoSolution>,
}

// ---------------------------------------------------------------------------
// AskTell Main Implementation
// ---------------------------------------------------------------------------

impl AskTell {
    /// Create a new batch population optimizer instance.
    pub fn new(
        kind: AskTellKind,
        bounds: Vec<(f64, f64)>,
        options: AskTellOptions,
    ) -> Result<Self, String> {
        if bounds.is_empty() {
            return Err("bounds cannot be empty".into());
        }
        for (i, &(lo, hi)) in bounds.iter().enumerate() {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(format!("dimension {} bound [{}, {}] is not finite", i, lo, hi));
            }
            if lo >= hi {
                return Err(format!("dimension {} lower bound {} >= upper bound {}", i, lo, hi));
            }
        }
        let dim = bounds.len();
        let mut rng = Rng64::new(options.seed.unwrap_or(42));

        let state = match kind {
            AskTellKind::CmaEs => {
                InternalState::CmaEs(CmaEsState::new(dim, &bounds, options.population_size, &mut rng))
            }
            AskTellKind::De => {
                let pop_size = options.population_size.unwrap_or((dim * 10).clamp(20, 100));
                let mut pop = Vec::with_capacity(pop_size);
                for _ in 0..pop_size {
                    let cand: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * rng.next_f64()).collect();
                    pop.push(cand);
                }
                InternalState::De(DeState {
                    pop_size,
                    pop,
                    fitness: vec![f64::INFINITY; pop_size],
                    f: options.de_f.unwrap_or(0.8),
                    cr: options.de_cr.unwrap_or(0.9),
                    trial_mutants: Vec::new(),
                })
            }
            AskTellKind::Shade => {
                let pop_size = options.population_size.unwrap_or((dim * 10).clamp(20, 100));
                let mut pop = Vec::with_capacity(pop_size);
                for _ in 0..pop_size {
                    let cand: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * rng.next_f64()).collect();
                    pop.push(cand);
                }
                let memory_size = 6;
                InternalState::Shade(ShadeState {
                    pop_size,
                    pop,
                    fitness: vec![f64::INFINITY; pop_size],
                    archive: Vec::new(),
                    archive_max: pop_size,
                    m_f: vec![0.5; memory_size],
                    m_cr: vec![0.5; memory_size],
                    k: 0,
                    memory_size,
                    trial_f: Vec::new(),
                    trial_cr: Vec::new(),
                    trial_pop: Vec::new(),
                })
            }
            AskTellKind::Pso => {
                let pop_size = options.population_size.unwrap_or((dim * 10).clamp(20, 80));
                let mut x = Vec::with_capacity(pop_size);
                let mut v = Vec::with_capacity(pop_size);
                let mut v_max = Vec::with_capacity(dim);
                for &(lo, hi) in &bounds {
                    v_max.push(0.2 * (hi - lo));
                }
                for _ in 0..pop_size {
                    let pos: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * rng.next_f64()).collect();
                    let vel: Vec<f64> = (0..dim).map(|d| (rng.next_f64() * 2.0 - 1.0) * v_max[d]).collect();
                    x.push(pos);
                    v.push(vel);
                }
                InternalState::Pso(PsoState {
                    pop_size,
                    x: x.clone(),
                    v,
                    p_best: x,
                    p_best_fit: vec![f64::INFINITY; pop_size],
                    g_best: vec![0.0; dim],
                    g_best_fit: f64::INFINITY,
                    w: options.pso_w.unwrap_or(0.7298),
                    c1: options.pso_c1.unwrap_or(1.49618),
                    c2: options.pso_c2.unwrap_or(1.49618),
                    v_max,
                })
            }
            AskTellKind::Snes => {
                let pop_size = options.population_size.unwrap_or(4 + (3.0 * (dim as f64).ln()).floor() as usize);
                let mu: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * rng.next_f64()).collect();
                let sigma: Vec<f64> = bounds.iter().map(|&(lo, hi)| 0.2 * (hi - lo)).collect();
                let eta_mu = 1.0;
                let eta_sigma = (3.0 + (dim as f64).ln()) / (5.0 * (dim as f64).sqrt());
                InternalState::Snes(SnesState {
                    pop_size,
                    mu,
                    sigma,
                    eta_mu,
                    eta_sigma,
                    last_z: Vec::new(),
                })
            }
            AskTellKind::Rcga => {
                let pop_size = options.population_size.unwrap_or((dim * 10).clamp(20, 100));
                let mut pop = Vec::with_capacity(pop_size);
                for _ in 0..pop_size {
                    let cand: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * rng.next_f64()).collect();
                    pop.push(cand);
                }
                InternalState::Rcga(RcgaState {
                    pop_size,
                    pop,
                    fitness: vec![f64::INFINITY; pop_size],
                    eta_c: 20.0,
                    eta_m: 20.0,
                })
            }
            AskTellKind::Nsga2 => {
                let pop_size = options.population_size.unwrap_or(40).max(10);
                let mut pop = Vec::with_capacity(pop_size);
                for _ in 0..pop_size {
                    let cand: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * rng.next_f64()).collect();
                    pop.push(cand);
                }
                InternalState::Nsga2(Nsga2State {
                    pop_size,
                    parent_pop: pop,
                    parent_objs: Vec::new(),
                    offspring_pop: Vec::new(),
                    is_initial: true,
                    pareto_front: Vec::new(),
                })
            }
            AskTellKind::Moead => {
                let pop_size = options.population_size.unwrap_or(40).max(10);
                let mut pop = Vec::with_capacity(pop_size);
                for _ in 0..pop_size {
                    let cand: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * rng.next_f64()).collect();
                    pop.push(cand);
                }
                let t_size = options.moead_t.unwrap_or(pop_size / 5).clamp(2, pop_size);
                InternalState::Moead(MoeadState {
                    pop_size,
                    weights: Vec::new(),
                    neighborhoods: Vec::new(),
                    t_size,
                    current_pop: pop,
                    current_objs: Vec::new(),
                    ideal_point: Vec::new(),
                    offspring_pop: Vec::new(),
                    is_initial: true,
                    pareto_front: Vec::new(),
                })
            }
            AskTellKind::Nsga3 => {
                let pop_size = options.population_size.unwrap_or(40).max(10);
                let mut pop = Vec::with_capacity(pop_size);
                for _ in 0..pop_size {
                    let cand: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * rng.next_f64()).collect();
                    pop.push(cand);
                }
                InternalState::Nsga3(Nsga3State {
                    pop_size,
                    parent_pop: pop,
                    parent_objs: Vec::new(),
                    offspring_pop: Vec::new(),
                    ref_points: Vec::new(),
                    partitions: options.nsga3_partitions,
                    is_initial: true,
                    pareto_front: Vec::new(),
                })
            }
            AskTellKind::Island => {
                let k = options.island_count.unwrap_or(4).clamp(2, 16);
                let total_pop = options.population_size.unwrap_or(40);
                let sub_pop = (total_pop / k).max(6);
                let sub_kinds = [
                    AskTellKind::CmaEs,
                    AskTellKind::De,
                    AskTellKind::Rcga,
                    AskTellKind::Pso,
                ];
                let mut islands = Vec::with_capacity(k);
                for i in 0..k {
                    let mut sub_opts = options.clone();
                    sub_opts.population_size = Some(sub_pop);
                    sub_opts.seed = Some(options.seed.unwrap_or(42).wrapping_add((i as u64) * 1000 + 7));
                    let sub_kind = sub_kinds[i % sub_kinds.len()];
                    let island = AskTell::new(sub_kind, bounds.clone(), sub_opts)?;
                    islands.push(island);
                }
                InternalState::Island(IslandState {
                    islands,
                    _island_count: k,
                    migration_interval: options.migration_interval.unwrap_or(10).max(1),
                    _migration_rate: options.migration_rate.unwrap_or(1).max(1),
                    batch_sizes: Vec::new(),
                    pareto_front: Vec::new(),
                })
            }
        };

        Ok(Self {
            kind,
            bounds,
            dim,
            options,
            rng,
            iterations: 0,
            evaluations: 0,
            status: SolveStatus::Converged,
            should_stop: false,
            restart_count: 0,
            current_candidates: Vec::new(),
            best_x: Vec::new(),
            best_value: f64::INFINITY,
            state,
        })
    }

    pub fn kind(&self) -> AskTellKind {
        self.kind
    }

    /// Ask for the next batch of candidate parameter vectors.
    pub fn ask(&mut self) -> Result<Vec<Vec<f64>>, String> {
        if self.should_stop {
            return Ok(Vec::new());
        }

        let candidates = match &mut self.state {
            InternalState::CmaEs(st) => {
                let mut cands = Vec::with_capacity(st.lambda);
                let mut zs = Vec::with_capacity(st.lambda);
                let mut ys = Vec::with_capacity(st.lambda);

                for _ in 0..st.lambda {
                    let mut z = DVector::zeros(self.dim);
                    for j in 0..self.dim {
                        z[j] = self.rng.next_gaussian();
                    }
                    // y = B * (D .* z)
                    let mut d_z = DVector::zeros(self.dim);
                    for j in 0..self.dim {
                        d_z[j] = st.d_vec[j] * z[j];
                    }
                    let y = &st.b_mat * d_z;
                    let mut x = Vec::with_capacity(self.dim);
                    for j in 0..self.dim {
                        let val = (st.m[j] + st.sigma * y[j]).clamp(self.bounds[j].0, self.bounds[j].1);
                        x.push(val);
                    }
                    zs.push(z.as_slice().to_vec());
                    ys.push(y.as_slice().to_vec());
                    cands.push(x);
                }
                st.last_z = zs;
                st.last_y = ys;
                cands
            }
            InternalState::De(st) => {
                if st.fitness.iter().any(|&f| f.is_infinite()) {
                    // Initial evaluation of base population
                    st.pop.clone()
                } else {
                    let mut trials = Vec::with_capacity(st.pop_size);
                    for i in 0..st.pop_size {
                        let (r1, r2, r3) = pick_distinct_3(st.pop_size, i, &mut self.rng);
                        let j_rand = (self.rng.next_u64() as usize) % self.dim;
                        let mut trial = st.pop[i].clone();
                        for j in 0..self.dim {
                            if self.rng.next_f64() < st.cr || j == j_rand {
                                let v = st.pop[r1][j] + st.f * (st.pop[r2][j] - st.pop[r3][j]);
                                trial[j] = v.clamp(self.bounds[j].0, self.bounds[j].1);
                            }
                        }
                        trials.push(trial);
                    }
                    st.trial_mutants = trials.clone();
                    trials
                }
            }
            InternalState::Shade(st) => {
                if st.fitness.iter().any(|&f| f.is_infinite()) {
                    st.pop.clone()
                } else {
                    let mut trials = Vec::with_capacity(st.pop_size);
                    let mut trial_f = Vec::with_capacity(st.pop_size);
                    let mut trial_cr = Vec::with_capacity(st.pop_size);

                    // Sort population indices by fitness
                    let mut rank_idx: Vec<usize> = (0..st.pop_size).collect();
                    rank_idx.sort_by(|&a, &b| st.fitness[a].partial_cmp(&st.fitness[b]).unwrap());

                    for i in 0..st.pop_size {
                        let r = (self.rng.next_u64() as usize) % st.memory_size;
                        let mut cr = self.rng.next_gaussian() * 0.1 + st.m_cr[r];
                        cr = cr.clamp(0.0, 1.0);

                        let mut f = self.rng.next_cauchy(st.m_f[r], 0.1);
                        while f <= 0.0 {
                            f = self.rng.next_cauchy(st.m_f[r], 0.1);
                        }
                        f = f.min(1.0);

                        // pbest selection (top p% with p >= 2)
                        let p_count = ((st.pop_size as f64) * 0.11).ceil() as usize;
                        let pbest_idx = rank_idx[(self.rng.next_u64() as usize) % p_count.max(2).min(st.pop_size)];

                        // r1 from pop != i
                        let mut r1 = (self.rng.next_u64() as usize) % st.pop_size;
                        while r1 == i {
                            r1 = (self.rng.next_u64() as usize) % st.pop_size;
                        }

                        // r2 from pop U archive != i, r1
                        let total_pool = st.pop_size + st.archive.len();
                        let mut r2 = (self.rng.next_u64() as usize) % total_pool;
                        while r2 == i || r2 == r1 {
                            r2 = (self.rng.next_u64() as usize) % total_pool;
                        }
                        let x_r2 = if r2 < st.pop_size {
                            &st.pop[r2]
                        } else {
                            &st.archive[r2 - st.pop_size]
                        };

                        let j_rand = (self.rng.next_u64() as usize) % self.dim;
                        let mut trial = st.pop[i].clone();
                        for j in 0..self.dim {
                            if self.rng.next_f64() < cr || j == j_rand {
                                let v = st.pop[i][j]
                                    + f * (st.pop[pbest_idx][j] - st.pop[i][j])
                                    + f * (st.pop[r1][j] - x_r2[j]);
                                trial[j] = v.clamp(self.bounds[j].0, self.bounds[j].1);
                            }
                        }
                        trials.push(trial);
                        trial_f.push(f);
                        trial_cr.push(cr);
                    }
                    st.trial_f = trial_f;
                    st.trial_cr = trial_cr;
                    st.trial_pop = trials.clone();
                    trials
                }
            }
            InternalState::Pso(st) => {
                st.x.clone()
            }
            InternalState::Snes(st) => {
                let mut cands = Vec::with_capacity(st.pop_size);
                let mut zs = Vec::with_capacity(st.pop_size);
                for _ in 0..st.pop_size {
                    let mut z = Vec::with_capacity(self.dim);
                    let mut x = Vec::with_capacity(self.dim);
                    for j in 0..self.dim {
                        let g = self.rng.next_gaussian();
                        let val = (st.mu[j] + st.sigma[j] * g).clamp(self.bounds[j].0, self.bounds[j].1);
                        z.push(g);
                        x.push(val);
                    }
                    zs.push(z);
                    cands.push(x);
                }
                st.last_z = zs;
                cands
            }
            InternalState::Rcga(st) => {
                if st.fitness.iter().any(|&f| f.is_infinite()) {
                    st.pop.clone()
                } else {
                    let mut offspring = Vec::with_capacity(st.pop_size);
                    while offspring.len() < st.pop_size {
                        let p1 = binary_tournament(&st.pop, &st.fitness, &mut self.rng);
                        let p2 = binary_tournament(&st.pop, &st.fitness, &mut self.rng);
                        let (mut c1, mut c2) = sbx_crossover(&p1, &p2, &self.bounds, st.eta_c, &mut self.rng);
                        poly_mutation(&mut c1, &self.bounds, st.eta_m, &mut self.rng);
                        poly_mutation(&mut c2, &self.bounds, st.eta_m, &mut self.rng);
                        offspring.push(c1);
                        if offspring.len() < st.pop_size {
                            offspring.push(c2);
                        }
                    }
                    offspring
                }
            }
            InternalState::Nsga2(st) => {
                if st.is_initial {
                    st.parent_pop.clone()
                } else {
                    let mut offspring = Vec::with_capacity(st.pop_size);
                    while offspring.len() < st.pop_size {
                        let p1 = nsga2_tournament(&st.parent_pop, &st.parent_objs, &mut self.rng);
                        let p2 = nsga2_tournament(&st.parent_pop, &st.parent_objs, &mut self.rng);
                        let (mut c1, mut c2) = sbx_crossover(&p1, &p2, &self.bounds, 20.0, &mut self.rng);
                        poly_mutation(&mut c1, &self.bounds, 20.0, &mut self.rng);
                        poly_mutation(&mut c2, &self.bounds, 20.0, &mut self.rng);
                        offspring.push(c1);
                        if offspring.len() < st.pop_size {
                            offspring.push(c2);
                        }
                    }
                    st.offspring_pop = offspring.clone();
                    offspring
                }
            }
            InternalState::Moead(st) => {
                if st.is_initial {
                    st.current_pop.clone()
                } else {
                    let mut offspring = Vec::with_capacity(st.pop_size);
                    for i in 0..st.pop_size {
                        let nbrs = &st.neighborhoods[i];
                        let idx1 = (self.rng.next_u64() as usize) % nbrs.len();
                        let mut idx2 = (self.rng.next_u64() as usize) % nbrs.len();
                        while idx2 == idx1 && nbrs.len() > 1 {
                            idx2 = (self.rng.next_u64() as usize) % nbrs.len();
                        }
                        let p1 = &st.current_pop[nbrs[idx1]];
                        let p2 = &st.current_pop[nbrs[idx2]];

                        let (mut c1, _) = sbx_crossover(p1, p2, &self.bounds, 20.0, &mut self.rng);
                        poly_mutation(&mut c1, &self.bounds, 20.0, &mut self.rng);
                        offspring.push(c1);
                    }
                    st.offspring_pop = offspring.clone();
                    offspring
                }
            }
            InternalState::Nsga3(st) => {
                if st.is_initial {
                    st.parent_pop.clone()
                } else {
                    let mut offspring = Vec::with_capacity(st.pop_size);
                    while offspring.len() < st.pop_size {
                        let p1 = nsga2_tournament(&st.parent_pop, &st.parent_objs, &mut self.rng);
                        let p2 = nsga2_tournament(&st.parent_pop, &st.parent_objs, &mut self.rng);
                        let (mut c1, mut c2) = sbx_crossover(&p1, &p2, &self.bounds, 20.0, &mut self.rng);
                        poly_mutation(&mut c1, &self.bounds, 20.0, &mut self.rng);
                        poly_mutation(&mut c2, &self.bounds, 20.0, &mut self.rng);
                        offspring.push(c1);
                        if offspring.len() < st.pop_size {
                            offspring.push(c2);
                        }
                    }
                    st.offspring_pop = offspring.clone();
                    offspring
                }
            }
            InternalState::Island(st) => {
                let mut all_cands = Vec::new();
                st.batch_sizes.clear();
                for isl in &mut st.islands {
                    match isl.ask() {
                        Ok(b) => {
                            st.batch_sizes.push(b.len());
                            all_cands.extend(b);
                        }
                        Err(e) => return Err(e),
                    }
                }
                all_cands
            }
        };

        self.current_candidates = candidates.clone();
        Ok(candidates)
    }

    /// Provide 1D scalar objective scores for the candidates returned by `ask()`.
    pub fn tell_scalar(&mut self, fitnesses: &[f64]) -> Result<(), String> {
        if self.current_candidates.is_empty() {
            return Err("call ask() before tell()".into());
        }
        if fitnesses.len() != self.current_candidates.len() {
            return Err(format!(
                "expected {} fitnesses but received {}",
                self.current_candidates.len(),
                fitnesses.len()
            ));
        }
        for &fit in fitnesses {
            if !fit.is_finite() {
                self.status = SolveStatus::InvalidObjective;
                self.should_stop = true;
                return Err("objective value must be finite".into());
            }
        }

        self.evaluations += fitnesses.len();
        self.iterations += 1;

        // Track global best
        for (cand, &fit) in self.current_candidates.iter().zip(fitnesses.iter()) {
            if fit < self.best_value {
                self.best_value = fit;
                self.best_x = cand.clone();
            }
        }

        match &mut self.state {
            InternalState::CmaEs(st) => {
                let mut idxs: Vec<usize> = (0..st.lambda).collect();
                idxs.sort_by(|&a, &b| fitnesses[a].partial_cmp(&fitnesses[b]).unwrap());

                // Update mean
                let old_m = st.m.clone();
                let mut new_m = vec![0.0; self.dim];
                let mut z_w = vec![0.0; self.dim];
                let mut y_w = vec![0.0; self.dim];

                for (w_idx, &cand_idx) in idxs[..st.mu].iter().enumerate() {
                    let w = st.weights[w_idx];
                    for j in 0..self.dim {
                        z_w[j] += w * st.last_z[cand_idx][j];
                        y_w[j] += w * st.last_y[cand_idx][j];
                    }
                }
                for j in 0..self.dim {
                    new_m[j] = old_m[j] + st.sigma * y_w[j];
                }

                // Whitened path update: B * z_w
                let z_w_vec = DVector::from_row_slice(&z_w);
                let b_zw = &st.b_mat * z_w_vec;
                for j in 0..self.dim {
                    st.ps[j] = (1.0 - st.cs) * st.ps[j] + (st.cs * (2.0 - st.cs) * st.mueff).sqrt() * b_zw[j];
                }
                let ps_norm = st.ps.iter().map(|&x| x * x).sum::<f64>().sqrt();
                let h_sig = if ps_norm / (1.0 - (1.0 - st.cs).powi(2 * (self.iterations as i32 + 1))).sqrt()
                    < (1.4 + 2.0 / (self.dim as f64 + 1.0)) * st.chi_n
                {
                    1.0
                } else {
                    0.0
                };

                for j in 0..self.dim {
                    st.pc[j] = (1.0 - st.cc) * st.pc[j] + h_sig * (st.cc * (2.0 - st.cc) * st.mueff).sqrt() * y_w[j];
                }

                // Covariance adaptation
                let pc_vec = DVector::from_row_slice(&st.pc);
                let mut c_update = &pc_vec * pc_vec.transpose() * st.c1;
                for (w_idx, &cand_idx) in idxs[..st.mu].iter().enumerate() {
                    let y_k = DVector::from_row_slice(&st.last_y[cand_idx]);
                    c_update += (&y_k * y_k.transpose()) * (st.weights[w_idx] * st.cmu);
                }
                let old_weight = 1.0 - st.c1 - st.cmu + (1.0 - h_sig) * st.c1 * st.cc * (2.0 - st.cc);
                st.cov = &st.cov * old_weight + c_update;

                // Step-size adaptation
                st.sigma *= ((st.cs / st.damps) * (ps_norm / st.chi_n - 1.0)).exp();
                st.m = new_m;

                // Eigendecomposition C = B * D^2 * B^T
                let eigen = st.cov.clone().symmetric_eigen();
                st.b_mat = eigen.eigenvectors;
                let mut d_vec = DVector::zeros(self.dim);
                let mut min_ev = f64::INFINITY;
                let mut max_ev = 0.0f64;
                for j in 0..self.dim {
                    let ev = eigen.eigenvalues[j].max(1e-18);
                    d_vec[j] = ev.sqrt();
                    min_ev = min_ev.min(ev);
                    max_ev = max_ev.max(ev);
                }
                st.d_vec = d_vec;

                // Stagnation / tolerance check
                let tol = self.options.tol.unwrap_or(1e-8);
                let cond = max_ev / min_ev.max(1e-18);
                let spread = (fitnesses[idxs[st.lambda - 1]] - fitnesses[idxs[0]]).abs();

                if st.sigma < 1e-14 || spread < tol || cond > 1e14 {
                    if self.options.ipop && self.restart_count < self.options.max_restarts {
                        self.restart_count += 1;
                        let next_lambda = st.lambda * 2;
                        *st = CmaEsState::new(self.dim, &self.bounds, Some(next_lambda), &mut self.rng);
                    } else {
                        self.should_stop = true;
                        self.status = if spread < tol { SolveStatus::Tolerance } else { SolveStatus::Converged };
                    }
                }
            }
            InternalState::De(st) => {
                if st.fitness.iter().any(|&f| f.is_infinite()) {
                    st.fitness = fitnesses.to_vec();
                } else {
                    for i in 0..st.pop_size {
                        if fitnesses[i] <= st.fitness[i] {
                            st.pop[i] = st.trial_mutants[i].clone();
                            st.fitness[i] = fitnesses[i];
                        }
                    }
                }
            }
            InternalState::Shade(st) => {
                if st.fitness.iter().any(|&f| f.is_infinite()) {
                    st.fitness = fitnesses.to_vec();
                } else {
                    let mut s_f = Vec::new();
                    let mut s_cr = Vec::new();
                    let mut delta_f = Vec::new();

                    for i in 0..st.pop_size {
                        if fitnesses[i] < st.fitness[i] {
                            delta_f.push((st.fitness[i] - fitnesses[i]).abs());
                            s_f.push(st.trial_f[i]);
                            s_cr.push(st.trial_cr[i]);

                            // Archive old point
                            if st.archive.len() < st.archive_max {
                                st.archive.push(st.pop[i].clone());
                            } else if !st.archive.is_empty() {
                                let rep = (self.rng.next_u64() as usize) % st.archive.len();
                                st.archive[rep] = st.pop[i].clone();
                            }

                            st.pop[i] = st.trial_pop[i].clone();
                            st.fitness[i] = fitnesses[i];
                        }
                    }

                    // Update memory M_F and M_CR
                    if !s_f.is_empty() {
                        let sum_delta: f64 = delta_f.iter().sum();
                        let weights: Vec<f64> = delta_f.iter().map(|&d| d / sum_delta.max(1e-12)).collect();

                        // Lehmer mean for F
                        let numer: f64 = weights.iter().zip(&s_f).map(|(&w, &f)| w * f * f).sum();
                        let denom: f64 = weights.iter().zip(&s_f).map(|(&w, &f)| w * f).sum();
                        if denom > 0.0 {
                            st.m_f[st.k] = numer / denom;
                        }

                        // Weighted mean for CR
                        let w_cr: f64 = weights.iter().zip(&s_cr).map(|(&w, &cr)| w * cr).sum();
                        st.m_cr[st.k] = w_cr;
                        st.k = (st.k + 1) % st.memory_size;
                    }
                }
            }
            InternalState::Pso(st) => {
                for i in 0..st.pop_size {
                    let fit = fitnesses[i];
                    if fit < st.p_best_fit[i] {
                        st.p_best_fit[i] = fit;
                        st.p_best[i] = self.current_candidates[i].clone();
                    }
                    if fit < st.g_best_fit {
                        st.g_best_fit = fit;
                        st.g_best = self.current_candidates[i].clone();
                    }
                }
                // Update velocity and position
                for i in 0..st.pop_size {
                    for j in 0..self.dim {
                        let r1 = self.rng.next_f64();
                        let r2 = self.rng.next_f64();
                        st.v[i][j] = st.w * st.v[i][j]
                            + st.c1 * r1 * (st.p_best[i][j] - st.x[i][j])
                            + st.c2 * r2 * (st.g_best[j] - st.x[i][j]);
                        st.v[i][j] = st.v[i][j].clamp(-st.v_max[j], st.v_max[j]);
                        st.x[i][j] = (st.x[i][j] + st.v[i][j]).clamp(self.bounds[j].0, self.bounds[j].1);
                    }
                }
            }
            InternalState::Snes(st) => {
                let mut idxs: Vec<usize> = (0..st.pop_size).collect();
                idxs.sort_by(|&a, &b| fitnesses[a].partial_cmp(&fitnesses[b]).unwrap());

                // Rank utilities
                let mut utilities = vec![0.0; st.pop_size];
                let half = st.pop_size as f64 / 2.0;
                let mut sum_u = 0.0;
                for rank in 0..st.pop_size {
                    let u = ((half + 1.0).ln() - ((rank + 1) as f64).ln()).max(0.0);
                    utilities[idxs[rank]] = u;
                    sum_u += u;
                }
                for u in utilities.iter_mut() {
                    *u = (*u / sum_u.max(1e-12)) - (1.0 / st.pop_size as f64);
                }

                // Natural gradient updates
                for j in 0..self.dim {
                    let mut grad_mu = 0.0;
                    let mut grad_sigma = 0.0;
                    for i in 0..st.pop_size {
                        let z = st.last_z[i][j];
                        grad_mu += utilities[i] * z;
                        grad_sigma += utilities[i] * (z * z - 1.0);
                    }
                    st.mu[j] += st.eta_mu * st.sigma[j] * grad_mu;
                    st.sigma[j] *= (0.5 * st.eta_sigma * grad_sigma).exp();
                    st.mu[j] = st.mu[j].clamp(self.bounds[j].0, self.bounds[j].1);
                }
            }
            InternalState::Rcga(st) => {
                if st.fitness.iter().any(|&f| f.is_infinite()) {
                    st.fitness = fitnesses.to_vec();
                } else {
                    // Elitist replacement: pool current parents + offspring
                    let mut combined_pop = st.pop.clone();
                    combined_pop.extend_from_slice(&self.current_candidates);
                    let mut combined_fit = st.fitness.clone();
                    combined_fit.extend_from_slice(fitnesses);

                    let mut idxs: Vec<usize> = (0..combined_pop.len()).collect();
                    idxs.sort_by(|&a, &b| combined_fit[a].partial_cmp(&combined_fit[b]).unwrap());

                    for (i, &idx) in idxs[..st.pop_size].iter().enumerate() {
                        st.pop[i] = combined_pop[idx].clone();
                        st.fitness[i] = combined_fit[idx];
                    }
                }
            }
            InternalState::Nsga2(_) | InternalState::Moead(_) | InternalState::Nsga3(_) => {
                // If user called scalar tell on multi-objective solver, wrap scalar in 1-element vector
                let wrapped: Vec<Vec<f64>> = fitnesses.iter().map(|&f| vec![f]).collect();
                return self.tell_multi(&wrapped);
            }
            InternalState::Island(st) => {
                let mut offset = 0;
                for (i, isl) in st.islands.iter_mut().enumerate() {
                    let sz = st.batch_sizes[i];
                    if offset + sz <= fitnesses.len() {
                        isl.tell_scalar(&fitnesses[offset..offset + sz])?;
                        offset += sz;
                    }
                }
                for isl in &st.islands {
                    if let Some((x, v)) = isl.best_solution() {
                        if v < self.best_value {
                            self.best_value = v;
                            self.best_x = x;
                        }
                    }
                }
                if self.iterations > 0 && self.iterations % st.migration_interval == 0 {
                    let best_per_island: Vec<Option<Vec<f64>>> = st
                        .islands
                        .iter()
                        .map(|isl| isl.best_solution().map(|(x, _)| x))
                        .collect();
                    let k = st.islands.len();
                    for i in 0..k {
                        let target = (i + 1) % k;
                        if let Some(ref best_cand) = best_per_island[i] {
                            st.islands[target].inject_candidate(best_cand);
                        }
                    }
                }
            }
        }

        self.check_termination();
        self.current_candidates.clear();
        Ok(())
    }

    /// Provide vector multi-objective scores for candidates returned by `ask()`.
    pub fn tell_multi(&mut self, objectives: &[Vec<f64>]) -> Result<(), String> {
        if self.current_candidates.is_empty() {
            return Err("call ask() before tell()".into());
        }
        if objectives.len() != self.current_candidates.len() {
            return Err(format!(
                "expected {} objective vectors but received {}",
                self.current_candidates.len(),
                objectives.len()
            ));
        }

        self.evaluations += objectives.len();
        self.iterations += 1;

        match &mut self.state {
            InternalState::Nsga2(st) => {
                let m_dim = objectives[0].len();
                if st.is_initial {
                    st.parent_objs = objectives.to_vec();
                    st.is_initial = false;
                } else {
                    // Combined pool R = P U Q of size 2N
                    let mut r_pop = st.parent_pop.clone();
                    r_pop.extend_from_slice(&st.offspring_pop);
                    let mut r_objs = st.parent_objs.clone();
                    r_objs.extend_from_slice(objectives);

                    // Fast non-dominated sort
                    let fronts = non_dominated_sort(&r_objs);

                    // Update pareto front solutions
                    st.pareto_front = fronts[0]
                        .iter()
                        .map(|&idx| ParetoSolution {
                            point: r_pop[idx].clone(),
                            objectives: r_objs[idx].clone(),
                        })
                        .collect();

                    // Truncate to N for next generation
                    let mut next_pop = Vec::with_capacity(st.pop_size);
                    let mut next_objs = Vec::with_capacity(st.pop_size);

                    for front in fronts {
                        if next_pop.len() + front.len() <= st.pop_size {
                            for &idx in &front {
                                next_pop.push(r_pop[idx].clone());
                                next_objs.push(r_objs[idx].clone());
                            }
                        } else {
                            // Crowding distance sorting for partial front
                            let remaining = st.pop_size - next_pop.len();
                            let crowded = sort_by_crowding_distance(&front, &r_objs, m_dim);
                            for &idx in &crowded[..remaining] {
                                next_pop.push(r_pop[idx].clone());
                                next_objs.push(r_objs[idx].clone());
                            }
                            break;
                        }
                    }

                    st.parent_pop = next_pop;
                    st.parent_objs = next_objs;
                }
            }
            InternalState::Moead(st) => {
                let m_dim = objectives[0].len();
                if st.is_initial {
                    st.current_objs = objectives.to_vec();
                    st.weights = generate_simplex_weights(st.pop_size, m_dim);
                    st.neighborhoods = compute_neighborhoods(&st.weights, st.t_size);
                    st.ideal_point = vec![f64::INFINITY; m_dim];
                    for obj in objectives {
                        for m in 0..m_dim {
                            if obj[m] < st.ideal_point[m] {
                                st.ideal_point[m] = obj[m];
                            }
                        }
                    }
                    let fronts = non_dominated_sort(&st.current_objs);
                    st.pareto_front = fronts[0]
                        .iter()
                        .map(|&idx| ParetoSolution {
                            point: st.current_pop[idx].clone(),
                            objectives: st.current_objs[idx].clone(),
                        })
                        .collect();
                    st.is_initial = false;
                } else {
                    for obj in objectives {
                        for m in 0..m_dim {
                            if obj[m] < st.ideal_point[m] {
                                st.ideal_point[m] = obj[m];
                            }
                        }
                    }
                    for k in 0..st.pop_size {
                        let y = &st.offspring_pop[k];
                        let f_y = &objectives[k];
                        let nbrs = st.neighborhoods[k].clone();
                        for &j in &nbrs {
                            let lambda_j = &st.weights[j];
                            let g_curr = tchebycheff(&st.current_objs[j], lambda_j, &st.ideal_point);
                            let g_off = tchebycheff(f_y, lambda_j, &st.ideal_point);
                            if g_off <= g_curr {
                                st.current_pop[j] = y.clone();
                                st.current_objs[j] = f_y.clone();
                            }
                        }
                    }
                    let fronts = non_dominated_sort(&st.current_objs);
                    st.pareto_front = fronts[0]
                        .iter()
                        .map(|&idx| ParetoSolution {
                            point: st.current_pop[idx].clone(),
                            objectives: st.current_objs[idx].clone(),
                        })
                        .collect();
                }
            }
            InternalState::Nsga3(st) => {
                let m_dim = objectives[0].len();
                if st.is_initial {
                    st.parent_objs = objectives.to_vec();
                    st.ref_points = if let Some(p) = st.partitions {
                        generate_das_dennis(m_dim, p)
                    } else if m_dim == 2 {
                        generate_das_dennis(m_dim, st.pop_size.saturating_sub(1).max(1))
                    } else {
                        generate_das_dennis(m_dim, 12)
                    };
                    let fronts = non_dominated_sort(&st.parent_objs);
                    st.pareto_front = fronts[0]
                        .iter()
                        .map(|&idx| ParetoSolution {
                            point: st.parent_pop[idx].clone(),
                            objectives: st.parent_objs[idx].clone(),
                        })
                        .collect();
                    st.is_initial = false;
                } else {
                    let mut r_pop = st.parent_pop.clone();
                    r_pop.extend_from_slice(&st.offspring_pop);
                    let mut r_objs = st.parent_objs.clone();
                    r_objs.extend_from_slice(objectives);

                    let fronts = non_dominated_sort(&r_objs);
                    st.pareto_front = fronts[0]
                        .iter()
                        .map(|&idx| ParetoSolution {
                            point: r_pop[idx].clone(),
                            objectives: r_objs[idx].clone(),
                        })
                        .collect();

                    let (next_pop, next_objs) = nsga3_select(
                        st.pop_size,
                        &fronts,
                        &r_pop,
                        &r_objs,
                        &st.ref_points,
                        m_dim,
                    );
                    st.parent_pop = next_pop;
                    st.parent_objs = next_objs;
                }
            }
            InternalState::Island(st) => {
                let mut offset = 0;
                for (i, isl) in st.islands.iter_mut().enumerate() {
                    let sz = st.batch_sizes[i];
                    if offset + sz <= objectives.len() {
                        isl.tell_multi(&objectives[offset..offset + sz])?;
                        offset += sz;
                    }
                }
                let mut all_solutions = Vec::new();
                for isl in &st.islands {
                    all_solutions.extend(isl.pareto_front());
                }
                if !all_solutions.is_empty() {
                    let all_objs: Vec<Vec<f64>> = all_solutions.iter().map(|s| s.objectives.clone()).collect();
                    let fronts = non_dominated_sort(&all_objs);
                    st.pareto_front = fronts[0].iter().map(|&idx| all_solutions[idx].clone()).collect();
                }
                if self.iterations > 0 && self.iterations % st.migration_interval == 0 {
                    let best_per_island: Vec<Option<Vec<f64>>> = st
                        .islands
                        .iter()
                        .map(|isl| {
                            isl.pareto_front().first().map(|s| s.point.clone())
                                .or_else(|| isl.best_solution().map(|(x, _)| x))
                        })
                        .collect();
                    let k = st.islands.len();
                    for i in 0..k {
                        let target = (i + 1) % k;
                        if let Some(ref best_cand) = best_per_island[i] {
                            st.islands[target].inject_candidate(best_cand);
                        }
                    }
                }
            }
            _ => {
                // If multi-objective passed to single-objective solver, reduce using L2 norm
                let scalar_fits: Vec<f64> = objectives
                    .iter()
                    .map(|vec| vec.iter().map(|&x| x * x).sum::<f64>().sqrt())
                    .collect();
                return self.tell_scalar(&scalar_fits);
            }
        }

        self.check_termination();
        self.current_candidates.clear();
        Ok(())
    }

    fn check_termination(&mut self) {
        if let Some(max_iter) = self.options.max_iterations {
            if self.iterations >= max_iter {
                self.status = SolveStatus::IterationLimit;
                self.should_stop = true;
            }
        }
        if let Some(max_eval) = self.options.max_evaluations {
            if self.evaluations >= max_eval {
                self.status = SolveStatus::EvaluationLimit;
                self.should_stop = true;
            }
        }
    }

    /// Best evaluated candidate and objective value (for single-objective runs).
    pub fn best_solution(&self) -> Option<(Vec<f64>, f64)> {
        match &self.state {
            InternalState::Island(st) => {
                let mut best: Option<(Vec<f64>, f64)> = None;
                for isl in &st.islands {
                    if let Some((x, v)) = isl.best_solution() {
                        match &best {
                            None => best = Some((x, v)),
                            Some((_, cur_v)) => {
                                if v < *cur_v {
                                    best = Some((x, v));
                                }
                            }
                        }
                    }
                }
                best
            }
            _ => {
                if self.best_value.is_finite() {
                    Some((self.best_x.clone(), self.best_value))
                } else {
                    None
                }
            }
        }
    }

    /// Extracted Pareto front solutions (for multi-objective runs).
    pub fn pareto_front(&self) -> Vec<ParetoSolution> {
        match &self.state {
            InternalState::Nsga2(st) => st.pareto_front.clone(),
            InternalState::Moead(st) => st.pareto_front.clone(),
            InternalState::Nsga3(st) => st.pareto_front.clone(),
            InternalState::Island(st) => {
                let mut all_solutions = Vec::new();
                for isl in &st.islands {
                    all_solutions.extend(isl.pareto_front());
                }
                if all_solutions.is_empty() {
                    Vec::new()
                } else {
                    let all_objs: Vec<Vec<f64>> = all_solutions.iter().map(|s| s.objectives.clone()).collect();
                    let fronts = non_dominated_sort(&all_objs);
                    fronts[0].iter().map(|&idx| all_solutions[idx].clone()).collect()
                }
            }
            _ => {
                if let Some((x, v)) = self.best_solution() {
                    vec![ParetoSolution {
                        point: x,
                        objectives: vec![v],
                    }]
                } else {
                    Vec::new()
                }
            }
        }
    }

    /// Inject an elite candidate into the population (used for island migration).
    pub fn inject_candidate(&mut self, cand: &[f64]) {
        match &mut self.state {
            InternalState::De(st) => {
                if let Some(worst_idx) = (0..st.pop_size).max_by(|&a, &b| st.fitness[a].partial_cmp(&st.fitness[b]).unwrap()) {
                    st.pop[worst_idx] = cand.to_vec();
                    st.fitness[worst_idx] = f64::INFINITY;
                }
            }
            InternalState::Rcga(st) => {
                if let Some(worst_idx) = (0..st.pop_size).max_by(|&a, &b| st.fitness[a].partial_cmp(&st.fitness[b]).unwrap()) {
                    st.pop[worst_idx] = cand.to_vec();
                    st.fitness[worst_idx] = f64::INFINITY;
                }
            }
            InternalState::Pso(st) => {
                if let Some(worst_idx) = (0..st.pop_size).max_by(|&a, &b| st.p_best_fit[a].partial_cmp(&st.p_best_fit[b]).unwrap()) {
                    st.x[worst_idx] = cand.to_vec();
                    st.p_best[worst_idx] = cand.to_vec();
                    st.p_best_fit[worst_idx] = f64::INFINITY;
                }
            }
            InternalState::CmaEs(st) => {
                for j in 0..self.dim {
                    st.m[j] = 0.8 * st.m[j] + 0.2 * cand[j];
                }
            }
            InternalState::Moead(st) => {
                let n = st.current_pop.len();
                if n > 0 {
                    let replace_idx = (self.rng.next_u64() as usize) % n;
                    st.current_pop[replace_idx] = cand.to_vec();
                }
            }
            InternalState::Nsga2(st) => {
                let n = st.parent_pop.len();
                if n > 0 {
                    let replace_idx = (self.rng.next_u64() as usize) % n;
                    st.parent_pop[replace_idx] = cand.to_vec();
                }
            }
            InternalState::Nsga3(st) => {
                let n = st.parent_pop.len();
                if n > 0 {
                    let replace_idx = (self.rng.next_u64() as usize) % n;
                    st.parent_pop[replace_idx] = cand.to_vec();
                }
            }
            InternalState::Island(st) => {
                if !st.islands.is_empty() {
                    st.islands[0].inject_candidate(cand);
                }
            }
            _ => {}
        }
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    pub fn status(&self) -> SolveStatus {
        self.status
    }

    pub fn should_stop(&self) -> bool {
        self.should_stop
    }

    pub fn restart_count(&self) -> usize {
        self.restart_count
    }
}

// ---------------------------------------------------------------------------
// Helper algorithms: SBX Crossover, Polynomial Mutation, Non-Dominated Sort
// ---------------------------------------------------------------------------

fn pick_distinct_3(n: usize, exclude: usize, rng: &mut Rng64) -> (usize, usize, usize) {
    let mut r1 = (rng.next_u64() as usize) % n;
    while r1 == exclude {
        r1 = (rng.next_u64() as usize) % n;
    }
    let mut r2 = (rng.next_u64() as usize) % n;
    while r2 == exclude || r2 == r1 {
        r2 = (rng.next_u64() as usize) % n;
    }
    let mut r3 = (rng.next_u64() as usize) % n;
    while r3 == exclude || r3 == r1 || r3 == r2 {
        r3 = (rng.next_u64() as usize) % n;
    }
    (r1, r2, r3)
}

fn binary_tournament(pop: &[Vec<f64>], fit: &[f64], rng: &mut Rng64) -> Vec<f64> {
    let n = pop.len();
    let i1 = (rng.next_u64() as usize) % n;
    let i2 = (rng.next_u64() as usize) % n;
    if fit[i1] <= fit[i2] {
        pop[i1].clone()
    } else {
        pop[i2].clone()
    }
}

fn sbx_crossover(
    p1: &[f64],
    p2: &[f64],
    bounds: &[(f64, f64)],
    eta: f64,
    rng: &mut Rng64,
) -> (Vec<f64>, Vec<f64>) {
    let dim = p1.len();
    let mut c1 = Vec::with_capacity(dim);
    let mut c2 = Vec::with_capacity(dim);

    for j in 0..dim {
        if rng.next_f64() <= 0.5 && (p1[j] - p2[j]).abs() > 1e-14 {
            let y1 = p1[j].min(p2[j]);
            let y2 = p1[j].max(p2[j]);
            let (lo, hi) = bounds[j];
            let rand_u = rng.next_f64();

            let beta1 = 1.0 + (2.0 * (y1 - lo) / (y2 - y1));
            let alpha1 = 2.0 - beta1.powf(-(eta + 1.0));
            let betaq1 = if rand_u <= 1.0 / alpha1 {
                (rand_u * alpha1).powf(1.0 / (eta + 1.0))
            } else {
                (1.0 / (2.0 - rand_u * alpha1)).powf(1.0 / (eta + 1.0))
            };
            let child1 = 0.5 * ((y1 + y2) - betaq1 * (y2 - y1));

            let beta2 = 1.0 + (2.0 * (hi - y2) / (y2 - y1));
            let alpha2 = 2.0 - beta2.powf(-(eta + 1.0));
            let betaq2 = if rand_u <= 1.0 / alpha2 {
                (rand_u * alpha2).powf(1.0 / (eta + 1.0))
            } else {
                (1.0 / (2.0 - rand_u * alpha2)).powf(1.0 / (eta + 1.0))
            };
            let child2 = 0.5 * ((y1 + y2) + betaq2 * (y2 - y1));

            c1.push(child1.clamp(lo, hi));
            c2.push(child2.clamp(lo, hi));
        } else {
            c1.push(p1[j]);
            c2.push(p2[j]);
        }
    }
    (c1, c2)
}

fn poly_mutation(cand: &mut [f64], bounds: &[(f64, f64)], eta: f64, rng: &mut Rng64) {
    let prob = 1.0 / cand.len() as f64;
    for j in 0..cand.len() {
        if rng.next_f64() <= prob {
            let (lo, hi) = bounds[j];
            let delta1 = (cand[j] - lo) / (hi - lo);
            let delta2 = (hi - cand[j]) / (hi - lo);
            let rand_u = rng.next_f64();
            let mut deltaq = if rand_u <= 0.5 {
                let xy = 1.0 - delta1;
                let val = 2.0 * rand_u + (1.0 - 2.0 * rand_u) * xy.powf(eta + 1.0);
                val.powf(1.0 / (eta + 1.0)) - 1.0
            } else {
                let xy = 1.0 - delta2;
                let val = 2.0 * (1.0 - rand_u) + 2.0 * (rand_u - 0.5) * xy.powf(eta + 1.0);
                1.0 - val.powf(1.0 / (eta + 1.0))
            };
            deltaq = deltaq.clamp(-1.0, 1.0);
            cand[j] = (cand[j] + deltaq * (hi - lo)).clamp(lo, hi);
        }
    }
}

fn nsga2_tournament(pop: &[Vec<f64>], _objs: &[Vec<f64>], rng: &mut Rng64) -> Vec<f64> {
    let n = pop.len();
    let i1 = (rng.next_u64() as usize) % n;
    let i2 = (rng.next_u64() as usize) % n;
    // Fast binary tournament
    pop[i1.min(i2)].clone()
}

fn non_dominated_sort(objs: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = objs.len();
    let mut domination_counts = vec![0usize; n];
    let mut dominated_sets: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut first_front = Vec::new();

    for p in 0..n {
        for q in 0..n {
            if p == q {
                continue;
            }
            if dominates(&objs[p], &objs[q]) {
                dominated_sets[p].push(q);
            } else if dominates(&objs[q], &objs[p]) {
                domination_counts[p] += 1;
            }
        }
        if domination_counts[p] == 0 {
            first_front.push(p);
        }
    }

    let mut current_front = first_front;
    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &p in &current_front {
            for &q in &dominated_sets[p] {
                domination_counts[q] -= 1;
                if domination_counts[q] == 0 {
                    next_front.push(q);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    if fronts.is_empty() {
        fronts.push((0..n).collect());
    }
    fronts
}

fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut better_in_any = false;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x > y {
            return false;
        }
        if x < y {
            better_in_any = true;
        }
    }
    better_in_any
}

fn sort_by_crowding_distance(front: &[usize], objs: &[Vec<f64>], m_dim: usize) -> Vec<usize> {
    if front.len() <= 2 {
        return front.to_vec();
    }
    let mut dists = vec![0.0f64; front.len()];

    for m in 0..m_dim {
        let mut sorted_indices: Vec<usize> = (0..front.len()).collect();
        sorted_indices.sort_by(|&a, &b| objs[front[a]][m].partial_cmp(&objs[front[b]][m]).unwrap());

        dists[sorted_indices[0]] = f64::INFINITY;
        dists[sorted_indices[front.len() - 1]] = f64::INFINITY;

        let obj_range = objs[front[sorted_indices[front.len() - 1]]][m] - objs[front[sorted_indices[0]]][m];
        if obj_range > 1e-14 {
            for i in 1..(front.len() - 1) {
                let diff = objs[front[sorted_indices[i + 1]]][m] - objs[front[sorted_indices[i - 1]]][m];
                dists[sorted_indices[i]] += diff / obj_range;
            }
        }
    }

    let mut result_indices: Vec<usize> = (0..front.len()).collect();
    result_indices.sort_by(|&a, &b| dists[b].partial_cmp(&dists[a]).unwrap());
    result_indices.iter().map(|&i| front[i]).collect()
}

// ---------------------------------------------------------------------------
// MOEA/D & NSGA-III Specialized Helpers
// ---------------------------------------------------------------------------

fn tchebycheff(f: &[f64], lambda: &[f64], z_star: &[f64]) -> f64 {
    let mut max_val = 0.0f64;
    for m in 0..f.len() {
        let diff = (f[m] - z_star[m]).abs();
        let val = lambda[m].max(1e-6) * diff;
        if val > max_val {
            max_val = val;
        }
    }
    max_val
}

fn compute_neighborhoods(weights: &[Vec<f64>], t: usize) -> Vec<Vec<usize>> {
    let n = weights.len();
    let mut nbrs = Vec::with_capacity(n);
    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .map(|j| {
                let d = weights[i]
                    .iter()
                    .zip(&weights[j])
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>();
                (j, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        nbrs.push(dists.iter().take(t.min(n)).map(|&(idx, _)| idx).collect());
    }
    nbrs
}

fn generate_das_dennis(m: usize, p: usize) -> Vec<Vec<f64>> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![vec![1.0]];
    }
    let mut points = Vec::new();
    let mut current = vec![0usize; m];
    das_dennis_recurse(0, p, &mut current, &mut points);
    points
        .into_iter()
        .map(|pt| pt.into_iter().map(|x| x as f64 / p.max(1) as f64).collect())
        .collect()
}

fn das_dennis_recurse(
    dim: usize,
    remaining: usize,
    current: &mut [usize],
    result: &mut Vec<Vec<usize>>,
) {
    if dim == current.len() - 1 {
        current[dim] = remaining;
        result.push(current.to_vec());
        return;
    }
    for val in 0..=remaining {
        current[dim] = val;
        das_dennis_recurse(dim + 1, remaining - val, current, result);
    }
}

fn generate_simplex_weights(n: usize, m: usize) -> Vec<Vec<f64>> {
    if m <= 1 {
        return vec![vec![1.0]; n];
    }
    if m == 2 {
        let mut weights = Vec::with_capacity(n);
        for i in 0..n {
            let w1 = i as f64 / ((n - 1).max(1) as f64);
            weights.push(vec![w1, 1.0 - w1]);
        }
        weights
    } else {
        let p = ((n as f64).powf(1.0 / (m as f64 - 1.0)).round() as usize).clamp(2, 30);
        let mut pts = generate_das_dennis(m, p);
        if pts.len() < n {
            let remaining = n - pts.len();
            for i in 0..remaining {
                let w1 = i as f64 / remaining as f64;
                let mut pt = vec![(1.0 - w1) / (m - 1) as f64; m];
                pt[0] = w1;
                pts.push(pt);
            }
        } else {
            pts.truncate(n);
        }
        pts
    }
}

fn perpendicular_distance(point: &[f64], ref_line: &[f64]) -> f64 {
    let norm_sq: f64 = ref_line.iter().map(|&x| x * x).sum();
    if norm_sq < 1e-14 {
        return point.iter().map(|&x| x * x).sum::<f64>().sqrt();
    }
    let dot: f64 = point.iter().zip(ref_line.iter()).map(|(&a, &b)| a * b).sum();
    let scale = dot / norm_sq;
    let mut dist_sq = 0.0;
    for (&p, &r) in point.iter().zip(ref_line.iter()) {
        let diff = p - scale * r;
        dist_sq += diff * diff;
    }
    dist_sq.sqrt()
}

fn nsga3_select(
    pop_size: usize,
    fronts: &[Vec<usize>],
    r_pop: &[Vec<f64>],
    r_objs: &[Vec<f64>],
    ref_points: &[Vec<f64>],
    m_dim: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut selected_indices = Vec::new();
    let mut l_front_idx = 0;

    for (f_idx, front) in fronts.iter().enumerate() {
        if selected_indices.len() + front.len() <= pop_size {
            selected_indices.extend(front.iter().copied());
        } else {
            l_front_idx = f_idx;
            break;
        }
    }

    if selected_indices.len() == pop_size {
        let next_pop = selected_indices.iter().map(|&i| r_pop[i].clone()).collect();
        let next_objs = selected_indices.iter().map(|&i| r_objs[i].clone()).collect();
        return (next_pop, next_objs);
    }

    let remaining_k = pop_size - selected_indices.len();
    let fl = &fronts[l_front_idx];

    let mut pool = selected_indices.clone();
    pool.extend_from_slice(fl);

    let mut z_min = vec![f64::INFINITY; m_dim];
    for &idx in &pool {
        for m in 0..m_dim {
            if r_objs[idx][m] < z_min[m] {
                z_min[m] = r_objs[idx][m];
            }
        }
    }

    let mut intercepts = vec![1.0; m_dim];
    for m in 0..m_dim {
        let mut max_diff = 1e-6f64;
        for &idx in &pool {
            let diff = (r_objs[idx][m] - z_min[m]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        intercepts[m] = max_diff;
    }

    let mut assoc_ref = vec![0usize; pool.len()];
    let mut assoc_dist = vec![0.0f64; pool.len()];

    for (p_i, &idx) in pool.iter().enumerate() {
        let norm_pt: Vec<f64> = (0..m_dim)
            .map(|m| (r_objs[idx][m] - z_min[m]) / intercepts[m])
            .collect();
        let mut min_d = f64::INFINITY;
        let mut best_r = 0;
        for (r_i, r_line) in ref_points.iter().enumerate() {
            let d = perpendicular_distance(&norm_pt, r_line);
            if d < min_d {
                min_d = d;
                best_r = r_i;
            }
        }
        assoc_ref[p_i] = best_r;
        assoc_dist[p_i] = min_d;
    }

    let mut rho = vec![0usize; ref_points.len()];
    for p_i in 0..selected_indices.len() {
        if assoc_ref[p_i] < rho.len() {
            rho[assoc_ref[p_i]] += 1;
        }
    }

    let mut fl_available: Vec<(usize, usize, f64)> = fl
        .iter()
        .enumerate()
        .map(|(fl_i, &orig_idx)| {
            let p_idx = selected_indices.len() + fl_i;
            (orig_idx, assoc_ref[p_idx], assoc_dist[p_idx])
        })
        .collect();

    let mut added_from_fl = Vec::new();
    while added_from_fl.len() < remaining_k && !fl_available.is_empty() {
        let mut min_rho = usize::MAX;
        let mut chosen_ref = None;

        for r in 0..ref_points.len() {
            if fl_available.iter().any(|&(_, r_idx, _)| r_idx == r) && rho[r] < min_rho {
                min_rho = rho[r];
                chosen_ref = Some(r);
            }
        }

        let Some(target_ref) = chosen_ref else { break; };

        let cand_pos = if rho[target_ref] == 0 {
            fl_available
                .iter()
                .enumerate()
                .filter(|(_, &(_, r_idx, _))| r_idx == target_ref)
                .min_by(|a, b| a.1.2.partial_cmp(&b.1.2).unwrap())
                .map(|(pos, _)| pos)
                .unwrap()
        } else {
            fl_available
                .iter()
                .position(|&(_, r_idx, _)| r_idx == target_ref)
                .unwrap()
        };

        let (chosen_orig_idx, _, _) = fl_available.remove(cand_pos);
        added_from_fl.push(chosen_orig_idx);
        rho[target_ref] += 1;
    }

    selected_indices.extend(added_from_fl);
    let next_pop = selected_indices.iter().map(|&i| r_pop[i].clone()).collect();
    let next_objs = selected_indices.iter().map(|&i| r_objs[i].clone()).collect();
    (next_pop, next_objs)
}
