//! Request, method, and option contracts for the generic optimization seam.

use super::objective::{Objective, SmoothObjective};

/// Default seed used by every stochastic method when no seed is supplied.
///
/// The fixed seed keeps omitted seeds reproducible across runs.
pub const DEFAULT_SEED: u64 = 0;

/// One optimizer method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerKind {
    /// Uniform random search inside bounds.
    RandomSearch,
    /// Latin hypercube sampling inside bounds.
    LatinHypercube,
    /// Limited-memory BFGS with box bounds.
    LbfgsB,
    /// Full-memory BFGS without box bounds.
    Bfgs,
    /// Nonlinear conjugate gradient with Polak-Ribiere plus restart.
    NonlinearConjugateGradient,
    /// Nelder-Mead simplex search.
    NelderMead,
    /// Powell direction-set search.
    Powell,
    /// Bounded simulated annealing.
    SimulatedAnnealing,
    /// COBYLA nonlinear constraint solver.
    Cobyla,
    /// Basin hopping composition.
    BasinHopping,
}

impl OptimizerKind {
    /// Canonical method name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RandomSearch => "random_search",
            Self::LatinHypercube => "latin_hypercube",
            Self::LbfgsB => "lbfgs_b",
            Self::Bfgs => "bfgs",
            Self::NonlinearConjugateGradient => "nonlinear_cg",
            Self::NelderMead => "nelder_mead",
            Self::Powell => "powell",
            Self::SimulatedAnnealing => "simulated_annealing",
            Self::Cobyla => "cobyla",
            Self::BasinHopping => "basin_hopping",
        }
    }

    /// Parses a canonical method name.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "random_search" => Self::RandomSearch,
            "latin_hypercube" => Self::LatinHypercube,
            "lbfgs_b" => Self::LbfgsB,
            "bfgs" => Self::Bfgs,
            "nonlinear_cg" => Self::NonlinearConjugateGradient,
            "nelder_mead" => Self::NelderMead,
            "powell" => Self::Powell,
            "simulated_annealing" => Self::SimulatedAnnealing,
            "cobyla" => Self::Cobyla,
            "basin_hopping" => Self::BasinHopping,
            _ => return None,
        })
    }

    /// Reports whether the method consumes gradients.
    pub fn uses_gradient(self) -> bool {
        matches!(
            self,
            Self::LbfgsB | Self::Bfgs | Self::NonlinearConjugateGradient
        )
    }
}

/// Options shared by every method.
#[derive(Clone, Debug)]
pub struct SharedOptions {
    /// Maximum user objective evaluations.
    pub max_evaluations: Option<usize>,
    /// Maximum solver iterations.
    pub max_iterations: Option<usize>,
    /// Convergence tolerance.
    pub tol: f64,
    /// Seed for stochastic methods.
    pub seed: u64,
    /// Initial population for population methods.
    pub initial_population: Option<Vec<Vec<f64>>>,
    /// Verbosity level.
    pub verbose: u8,
}

impl Default for SharedOptions {
    fn default() -> Self {
        Self {
            max_evaluations: None,
            max_iterations: None,
            tol: 1e-6,
            seed: DEFAULT_SEED,
            initial_population: None,
            verbose: 0,
        }
    }
}

/// Options for random search.
#[derive(Clone, Debug)]
pub struct RandomSearchOptions {
    /// Number of sampled points.
    pub samples: usize,
}

impl Default for RandomSearchOptions {
    fn default() -> Self {
        Self { samples: 64 }
    }
}

/// Options for Latin hypercube sampling.
#[derive(Clone, Debug)]
pub struct LatinHypercubeOptions {
    /// Number of sampled points.
    pub samples: usize,
}

impl Default for LatinHypercubeOptions {
    fn default() -> Self {
        Self { samples: 64 }
    }
}

/// Options for L-BFGS-B.
#[derive(Clone, Debug)]
pub struct LbfgsBOptions {
    /// Number of limited-memory correction pairs.
    pub m: usize,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Function value tolerance.
    pub ftol: f64,
    /// Projected gradient tolerance.
    pub gtol: f64,
}

impl Default for LbfgsBOptions {
    fn default() -> Self {
        Self {
            m: 6,
            max_iter: 500,
            ftol: 1e-9,
            gtol: 1e-5,
        }
    }
}

/// Options for full-memory BFGS.
#[derive(Clone, Debug)]
pub struct BfgsOptions {
    /// Maximum iterations.
    pub max_iter: usize,
    /// Gradient tolerance.
    pub gtol: f64,
}

impl Default for BfgsOptions {
    fn default() -> Self {
        Self {
            max_iter: 500,
            gtol: 1e-6,
        }
    }
}

/// Options for nonlinear conjugate gradient.
#[derive(Clone, Debug)]
pub struct NonlinearCgOptions {
    /// Maximum iterations.
    pub max_iter: usize,
    /// Gradient tolerance.
    pub gtol: f64,
    /// Restart period. `None` restarts on a negative Polak-Ribiere coefficient.
    pub restart_period: Option<usize>,
}

impl Default for NonlinearCgOptions {
    fn default() -> Self {
        Self {
            max_iter: 500,
            gtol: 1e-6,
            restart_period: None,
        }
    }
}

/// Options for Nelder-Mead.
#[derive(Clone, Debug)]
pub struct NelderMeadOptions {
    /// Maximum iterations.
    pub max_iter: usize,
    /// Point convergence tolerance.
    pub xatol: f64,
    /// Function value convergence tolerance.
    pub fatol: f64,
    /// Optional explicit initial simplex.
    pub initial_simplex: Option<Vec<Vec<f64>>>,
}

impl Default for NelderMeadOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            xatol: 1e-8,
            fatol: 1e-8,
            initial_simplex: None,
        }
    }
}

/// Options for Powell.
#[derive(Clone, Debug)]
pub struct PowellOptions {
    /// Maximum iterations.
    pub max_iter: usize,
    /// Line search parameter tolerance.
    pub xtol: f64,
    /// Function value tolerance.
    pub ftol: f64,
    /// Optional cap on the direction set size.
    pub max_directions: Option<usize>,
}

impl Default for PowellOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            xtol: 1e-8,
            ftol: 1e-8,
            max_directions: None,
        }
    }
}

/// Boundary policy for proposals that leave the box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryPolicy {
    /// Draw a new proposal inside the box.
    Resample,
    /// Reflect the proposal back into the box.
    Reflect,
}

/// Options for bounded simulated annealing.
#[derive(Clone, Debug)]
pub struct AnnealingOptions {
    /// Initial temperature. `None` derives it from the observed objective spread.
    pub initial_temperature: Option<f64>,
    /// Geometric cooling factor.
    pub alpha: f64,
    /// Minimum temperature.
    pub min_temperature: f64,
    /// Proposal scale as a fraction of the bound width.
    pub proposal_scale: f64,
    /// Boundary policy.
    pub boundary: BoundaryPolicy,
    /// Stagnation limit before the run stops.
    pub max_stagnation: usize,
}

impl Default for AnnealingOptions {
    fn default() -> Self {
        Self {
            initial_temperature: None,
            alpha: 0.95,
            min_temperature: 1e-8,
            proposal_scale: 0.1,
            boundary: BoundaryPolicy::Resample,
            max_stagnation: 100,
        }
    }
}

/// Options for COBYLA.
#[derive(Clone, Debug)]
pub struct CobylaOptions {
    /// Initial trust region radius.
    pub rhobeg: f64,
    /// Feasibility tolerance.
    pub feasibility_tol: f64,
}

impl Default for CobylaOptions {
    fn default() -> Self {
        Self {
            rhobeg: 1.0,
            feasibility_tol: 1e-8,
        }
    }
}

/// Options for basin hopping.
#[derive(Clone, Debug)]
pub struct BasinHoppingOptions {
    /// Inner local solver.
    pub inner: OptimizerKind,
    /// Perturbation size as a fraction of the bound width.
    pub step_size: f64,
    /// Metropolis temperature.
    pub temperature: f64,
    /// Target acceptance rate for adaptive step size. `None` keeps a fixed step.
    pub target_acceptance: Option<f64>,
    /// Monotonic mode. The run accepts only improvements.
    pub monotonic: bool,
    /// Number of perturbation and local solve rounds.
    pub iterations: usize,
}

impl Default for BasinHoppingOptions {
    fn default() -> Self {
        Self {
            inner: OptimizerKind::LbfgsB,
            step_size: 0.1,
            temperature: 1.0,
            target_acceptance: None,
            monotonic: false,
            iterations: 100,
        }
    }
}

/// Method-specific options.
#[derive(Clone, Debug)]
pub enum MethodOptions {
    /// Random search options.
    Random(RandomSearchOptions),
    /// Latin hypercube options.
    LatinHypercube(LatinHypercubeOptions),
    /// L-BFGS-B options.
    LbfgsB(LbfgsBOptions),
    /// BFGS options.
    Bfgs(BfgsOptions),
    /// Nonlinear conjugate gradient options.
    NonlinearCg(NonlinearCgOptions),
    /// Nelder-Mead options.
    NelderMead(NelderMeadOptions),
    /// Powell options.
    Powell(PowellOptions),
    /// Simulated annealing options.
    Annealing(AnnealingOptions),
    /// COBYLA options.
    Cobyla(CobylaOptions),
    /// Basin hopping options.
    BasinHopping(BasinHoppingOptions),
}

impl MethodOptions {
    /// Method that owns these options.
    pub fn kind(&self) -> OptimizerKind {
        match self {
            Self::Random(_) => OptimizerKind::RandomSearch,
            Self::LatinHypercube(_) => OptimizerKind::LatinHypercube,
            Self::LbfgsB(_) => OptimizerKind::LbfgsB,
            Self::Bfgs(_) => OptimizerKind::Bfgs,
            Self::NonlinearCg(_) => OptimizerKind::NonlinearConjugateGradient,
            Self::NelderMead(_) => OptimizerKind::NelderMead,
            Self::Powell(_) => OptimizerKind::Powell,
            Self::Annealing(_) => OptimizerKind::SimulatedAnnealing,
            Self::Cobyla(_) => OptimizerKind::Cobyla,
            Self::BasinHopping(_) => OptimizerKind::BasinHopping,
        }
    }
}

impl Default for MethodOptions {
    fn default() -> Self {
        Self::Random(RandomSearchOptions::default())
    }
}

/// One optimization request.
pub struct OptimizationRequest<'a> {
    /// Objective to minimize.
    pub objective: &'a dyn Objective,
    /// Gradient source for smooth methods. `None` makes the seam use a
    /// central-difference gradient over the counted objective.
    pub smooth: Option<&'a dyn SmoothObjective>,
    /// Starting point. `None` lets the method draw one.
    pub start: Option<Vec<f64>>,
    /// Selected method.
    pub method: OptimizerKind,
    /// Method-specific options.
    pub options: MethodOptions,
    /// Shared options.
    pub shared: SharedOptions,
}

impl<'a> OptimizationRequest<'a> {
    /// Builds a request with default method options for the selected method.
    pub fn new(
        objective: &'a dyn Objective,
        smooth: Option<&'a dyn SmoothObjective>,
        method: OptimizerKind,
        shared: SharedOptions,
    ) -> Self {
        let options = match method {
            OptimizerKind::RandomSearch => MethodOptions::Random(RandomSearchOptions::default()),
            OptimizerKind::LatinHypercube => {
                MethodOptions::LatinHypercube(LatinHypercubeOptions::default())
            }
            OptimizerKind::LbfgsB => MethodOptions::LbfgsB(LbfgsBOptions::default()),
            OptimizerKind::Bfgs => MethodOptions::Bfgs(BfgsOptions::default()),
            OptimizerKind::NonlinearConjugateGradient => {
                MethodOptions::NonlinearCg(NonlinearCgOptions::default())
            }
            OptimizerKind::NelderMead => MethodOptions::NelderMead(NelderMeadOptions::default()),
            OptimizerKind::Powell => MethodOptions::Powell(PowellOptions::default()),
            OptimizerKind::SimulatedAnnealing => {
                MethodOptions::Annealing(AnnealingOptions::default())
            }
            OptimizerKind::Cobyla => MethodOptions::Cobyla(CobylaOptions::default()),
            OptimizerKind::BasinHopping => {
                MethodOptions::BasinHopping(BasinHoppingOptions::default())
            }
        };
        Self {
            objective,
            smooth,
            start: None,
            method,
            options,
            shared,
        }
    }

    /// Sets the starting point.
    pub fn with_start(mut self, start: Vec<f64>) -> Self {
        self.start = Some(start);
        self
    }

    /// Sets the method-specific options.
    pub fn with_options(mut self, options: MethodOptions) -> Self {
        self.method = options.kind();
        self.options = options;
        self
    }

    /// Checks the request before any objective evaluation.
    ///
    /// Returns one error for a method and option mismatch, one for a starting
    /// point with the wrong length, and one for a starting point outside the
    /// bounds. Smooth methods accept a missing gradient source because the seam
    /// falls back to central differences.
    pub fn validate(&self) -> Result<(), String> {
        if self.options.kind() != self.method {
            return Err(format!(
                "method {} does not match options for method {}",
                self.method.as_str(),
                self.options.kind().as_str()
            ));
        }
        if let Some(start) = &self.start {
            if start.len() != self.objective.dim() {
                return Err(format!(
                    "starting point has {} values, objective has {} dimensions",
                    start.len(),
                    self.objective.dim()
                ));
            }
            if !self.objective.bounds().contains(start) {
                return Err("starting point lies outside the bounds".to_string());
            }
        }
        Ok(())
    }
}
