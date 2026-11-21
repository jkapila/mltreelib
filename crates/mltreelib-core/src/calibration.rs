//! Post-hoc calibration for binary event probabilities.

/// A monotone probability mapping fitted by isotonic regression.
///
/// Fit on validation or out-of-fold probabilities. Use independent observations
/// to evaluate the calibrated probabilities.
#[derive(Debug, Clone)]
pub struct IsotonicCalibrator {
    knots: Vec<(f64, f64)>,
}

#[derive(Clone, Copy)]
struct Block {
    first: f64,
    last: f64,
    positives: usize,
    count: usize,
}

impl Block {
    fn rate(self) -> f64 {
        self.positives as f64 / self.count as f64
    }
}

fn check_probability(value: f64) -> Result<(), &'static str> {
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err("probabilities must be finite and in [0, 1]")
    }
}

impl IsotonicCalibrator {
    /// Fit the positive-class probability mapping using binary labels (0 or 1).
    pub fn fit(y_true: &[f64], probabilities: &[f64]) -> Result<Self, &'static str> {
        if y_true.is_empty() || y_true.len() != probabilities.len() {
            return Err("y_true and probabilities must have the same non-zero length");
        }
        let mut observations = Vec::with_capacity(y_true.len());
        for (&label, &probability) in y_true.iter().zip(probabilities) {
            check_probability(probability)?;
            if label != 0.0 && label != 1.0 {
                return Err("y_true must contain binary labels (0 or 1)");
            }
            observations.push((probability, label as usize));
        }
        observations.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        let mut blocks = Vec::<Block>::new();
        let mut observations = observations.into_iter().peekable();
        while let Some((probability, label)) = observations.next() {
            let mut block = Block {
                first: probability,
                last: probability,
                positives: label,
                count: 1,
            };
            while let Some((_, label)) = observations.next_if(|&(p, _)| p == probability) {
                block.positives += label;
                block.count += 1;
            }
            // Pool adjacent violations of the fitted monotone event rates.
            while blocks
                .last()
                .is_some_and(|previous| previous.rate() >= block.rate())
            {
                let previous = blocks.pop().unwrap();
                block.first = previous.first;
                block.positives += previous.positives;
                block.count += previous.count;
            }
            blocks.push(block);
        }

        let mut knots = Vec::new();
        for block in blocks {
            knots.push((block.first, block.rate()));
            if block.last > block.first {
                knots.push((block.last, block.rate()));
            }
        }
        Ok(Self { knots })
    }

    /// Interpolate between fitted scores and clip to the boundary rates.
    pub fn predict(&self, probabilities: &[f64]) -> Result<Vec<f64>, &'static str> {
        probabilities
            .iter()
            .map(|&p| {
                check_probability(p)?;
                let right = self.knots.partition_point(|&(x, _)| x < p);
                if right == 0 {
                    return Ok(self.knots[0].1);
                }
                let (start, lower_rate) = self.knots[right - 1];
                let Some(&(end, upper_rate)) = self.knots.get(right) else {
                    return Ok(lower_rate);
                };
                Ok(lower_rate + (upper_rate - lower_rate) * ((p - start) / (end - start)))
            })
            .collect()
    }
}
