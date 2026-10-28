//! Reusable small NN trainer: an embedding table plus a linear head, trained
//! by SGD. Kept separate from any encoder so later learners can reuse it.
//!
//! The head's output activation follows the targets: with a Regression head,
//! **ReLU** when every observed target is non-negative, **linear** otherwise;
//! Classification uses a softmax over the classes. Training is deterministic
//! for a given seed and processes rows in input order.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Task head of the network.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Head {
    /// Softmax over classes, cross-entropy on integer class labels.
    Classification,
    /// Single output, squared loss on continuous targets.
    Regression,
}

/// Embedding table + linear head trained by SGD.
#[derive(Clone, Debug)]
pub struct EmbeddingNet {
    dim: usize,
    out_dim: usize,
    head: Head,
    lr: f64,
    epochs: usize,
    seed: u64,
    emb: Vec<Vec<f64>>,
    w: Vec<f64>,
    b: Vec<f64>,
    relu_out: bool,
}

impl EmbeddingNet {
    pub fn new(
        dim: usize,
        out_dim: usize,
        head: Head,
        lr: f64,
        epochs: usize,
        seed: u64,
    ) -> Result<Self, String> {
        if dim == 0 {
            return Err("dim must be >= 1".into());
        }
        if out_dim == 0 {
            return Err("out_dim must be >= 1".into());
        }
        if !(lr > 0.0) {
            return Err("lr must be > 0".into());
        }
        if epochs == 0 {
            return Err("epochs must be >= 1".into());
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 0.01).map_err(|e| e.to_string())?;
        let w = (0..out_dim * dim).map(|_| normal.sample(&mut rng)).collect();
        Ok(Self {
            dim,
            out_dim,
            head,
            lr,
            epochs,
            seed,
            emb: Vec::new(),
            w,
            b: vec![0.0; out_dim],
            relu_out: false,
        })
    }

    /// Train on rows given as (level id, target) pairs; the table is sized to
    /// `n_levels`. Missing targets are skipped. Deterministic per seed.
    pub fn train(
        &mut self,
        level_ids: &[usize],
        y: &[Option<f64>],
        n_levels: usize,
    ) -> Result<(), String> {
        if level_ids.len() != y.len() {
            return Err("level_ids and y length mismatch".into());
        }
        for &id in level_ids {
            if id >= n_levels {
                return Err(format!("level id {id} out of range (n_levels {n_levels})"));
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let normal = Normal::new(0.0, 0.01).map_err(|e| e.to_string())?;
        self.emb = (0..n_levels)
            .map(|_| (0..self.dim).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        match self.head {
            Head::Classification => {
                for t in y.iter().flatten() {
                    if t.is_nan() {
                        continue;
                    }
                    if *t < 0.0 || *t >= self.out_dim as f64 || t.fract() != 0.0 {
                        return Err(format!(
                            "y must be integer class labels in [0, {}); got {t}",
                            self.out_dim
                        ));
                    }
                }
            }
            Head::Regression => {
                self.relu_out = y.iter().flatten().filter(|t| !t.is_nan()).all(|t| *t >= 0.0);
            }
        }

        for _ in 0..self.epochs {
            for (row, t) in level_ids.iter().zip(y) {
                let t = match t {
                    Some(t) if !t.is_nan() => *t,
                    _ => continue,
                };
                let e = &mut self.emb[*row];
                match self.head {
                    Head::Classification => {
                        let mut z = self.b.clone();
                        for j in 0..self.out_dim {
                            for d in 0..self.dim {
                                z[j] += self.w[j * self.dim + d] * e[d];
                            }
                        }
                        let zmax = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let mut p: Vec<f64> = z.iter().map(|zj| (zj - zmax).exp()).collect();
                        let sum = p.iter().sum::<f64>();
                        for pj in p.iter_mut() {
                            *pj /= sum;
                        }
                        p[t as usize] -= 1.0;
                        let e_old = e.clone();
                        for j in 0..self.out_dim {
                            self.b[j] -= self.lr * p[j];
                            for d in 0..self.dim {
                                self.w[j * self.dim + d] -= self.lr * p[j] * e_old[d];
                                e[d] -= self.lr * p[j] * self.w[j * self.dim + d];
                            }
                        }
                    }
                    Head::Regression => {
                        let mut z = self.b[0];
                        for d in 0..self.dim {
                            z += self.w[d] * e[d];
                        }
                        let pred = if self.relu_out { z.max(0.0) } else { z };
                        let gate = if self.relu_out && z <= 0.0 { 0.0 } else { 1.0 };
                        let g = (pred - t) * gate;
                        let e_old = e.clone();
                        self.b[0] -= self.lr * g;
                        for d in 0..self.dim {
                            self.w[d] -= self.lr * g * e_old[d];
                            e[d] -= self.lr * g * self.w[d];
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Learned embedding of one level (zeros for out-of-range ids).
    pub fn embedding(&self, level_id: usize) -> Vec<f64> {
        self.emb
            .get(level_id)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.dim])
    }

    /// Head output for one level: class logits (classification) or the
    /// activated value wrapped in a 1-vector (regression).
    pub fn forward(&self, level_id: usize) -> Vec<f64> {
        let e = self.emb.get(level_id).map(|e| e.as_slice()).unwrap_or(&[]);
        let mut z = self.b.clone();
        for j in 0..self.out_dim {
            for d in 0..e.len() {
                z[j] += self.w[j * self.dim + d] * e[d];
            }
        }
        if self.head == Head::Regression && self.relu_out {
            z[0] = z[0].max(0.0);
        }
        z
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// ReLU output was selected for a Regression head (all targets >= 0).
    pub fn relu_out(&self) -> bool {
        self.relu_out
    }
}
