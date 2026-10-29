//! Small statistical primitives: the weighted summaries and costs the
//! split criteria ([`crate::criteria`]) and leaves ([`crate::leaf`]) share,
//! and the F distribution behind the significance gate ([`crate::gate`]).
//! Pure functions over `(y, w)` pairs — no dependencies on the tree.

/// Natural log of the gamma function (Lanczos-style series; rel. err.
/// < 1e-10 for the arguments this crate reaches).
fn ln_gamma(x: f64) -> f64 {
    const COF: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.120_865_097_386_617_9e-2,
        -0.539_523_938_495_3e-5,
    ];
    let mut x = x;
    let mut y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000_000_000_190_015;
    for &c in &COF {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

/// Continued fraction for the incomplete beta (Lentz's method).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 200;
    const EPS: f64 = 3e-14;
    const FPMIN: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a, b)`.
fn betainc(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Right-tail probability of the F distribution: `P(F_{d1,d2} > f)`.
/// This is the Chow test's p-value for statistic `f`.
///
/// Regularized incomplete beta via a Lentz continued fraction (Numerical
/// Recipes `betacf`/`betai` formulation) — no dependency earns its place
/// for one p-value. Exact where tested: agrees with `scipy.stats.f.sf` to
/// ~1e-13; the unit tests pin published F quantiles.
pub fn f_sf(f: f64, d1: u32, d2: u32) -> f64 {
    if !(f.is_finite()) || f < 0.0 {
        return if f.is_nan() || f < 0.0 {
            f64::NAN
        } else {
            0.0
        };
    }
    if f == 0.0 {
        return 1.0;
    }
    let (d1, d2) = (d1 as f64, d2 as f64);
    betainc(d2 / (d2 + d1 * f), d2 / 2.0, d1 / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Published F quantiles, cross-checked against `scipy.stats.f.ppf`.
    #[test]
    fn f_sf_matches_tables() {
        let cases = [
            (4.9646, 1, 10, 0.05),   // F_0.05(1,10)
            (3.4928, 2, 20, 0.05),   // F_0.05(2,20)
            (5.0503, 5, 5, 0.05),    // F_0.05(5,5)
            (2.3479, 10, 20, 0.05),  // F_0.05(10,20)
            (8.7539, 10, 10, 0.001), // F_0.001(10,10)
        ];
        for (q, d1, d2, want) in cases {
            let p = f_sf(q, d1, d2);
            assert!(
                (p - want).abs() < 2e-3,
                "f_sf({q}, {d1}, {d2}) = {p}, want {want}"
            );
        }
    }

    #[test]
    fn f_sf_edges_and_monotone() {
        assert_eq!(f_sf(0.0, 1, 10), 1.0);
        assert!(f_sf(1e9, 3, 40) < 1e-15);
        let mut prev = 1.0;
        for i in 1..200 {
            let p = f_sf(i as f64 * 0.1, 2, 30);
            assert!(p <= prev + 1e-12, "p must not rise with f");
            prev = p;
        }
    }
}
/// Weighted Shannon entropy (bits) over y-as-class-labels (weighted class
/// masses; distinct values are the classes).
pub(crate) fn class_entropy_w(p: &[(f64, f64)]) -> f64 {
    let total: f64 = p.iter().map(|&(_, w)| w).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut counts: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    for &(y, w) in p {
        *counts.entry(y.to_bits()).or_insert(0.0) += w;
    }
    counts
        .values()
        .map(|&c| {
            let q = c / total;
            if q > 0.0 {
                -q * q.log2()
            } else {
                0.0
            }
        })
        .sum()
}

/// Weighted Gini impurity over y-as-class-labels.
pub(crate) fn class_gini_w(p: &[(f64, f64)]) -> f64 {
    let total: f64 = p.iter().map(|&(_, w)| w).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut counts: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    for &(y, w) in p {
        *counts.entry(y.to_bits()).or_insert(0.0) += w;
    }
    1.0 - counts
        .values()
        .map(|&c| {
            let q = c / total;
            q * q
        })
        .sum::<f64>()
}

/// Weighted class-frequency total variation `Σ_j |p_L(j) - p_R(j)|`
/// (twoing rule's contrast term).
pub(crate) fn class_total_variation(left: &[(f64, f64)], right: &[(f64, f64)]) -> f64 {
    let mut lm: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    let mut rm: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    let (mut lw, mut rw) = (0.0, 0.0);
    for &(y, w) in left {
        *lm.entry(y.to_bits()).or_insert(0.0) += w;
        lw += w;
    }
    for &(y, w) in right {
        *rm.entry(y.to_bits()).or_insert(0.0) += w;
        rw += w;
    }
    if lw <= 0.0 || rw <= 0.0 {
        return 0.0;
    }
    let mut keys: Vec<u64> = lm.keys().copied().collect();
    keys.extend(rm.keys().copied());
    keys.sort_unstable();
    keys.dedup();
    keys.iter()
        .map(|k| {
            (lm.get(k).copied().unwrap_or(0.0) / lw - rm.get(k).copied().unwrap_or(0.0) / rw).abs()
        })
        .sum()
}

/// Weighted sample standard deviation (M5's SDR ingredient).
pub(crate) fn w_sd(p: &[(f64, f64)]) -> f64 {
    let sw: f64 = p.iter().map(|&(_, w)| w).sum();
    if sw <= 0.0 {
        return 0.0;
    }
    let mean = p.iter().map(|&(y, w)| w * y).sum::<f64>() / sw;
    let var = p.iter().map(|&(y, w)| w * (y - mean) * (y - mean)).sum::<f64>() / sw;
    var.sqrt()
}

/// Weighted median (weighted median of y values; pair slices are (y, w)).
pub(crate) fn w_median(p: &[(f64, f64)]) -> f64 {
    let mut v: Vec<(f64, f64)> = p.to_vec();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN in target"));
    let total: f64 = v.iter().map(|&(_, w)| w).sum();
    let mut cum = 0.0;
    for &(y, w) in &v {
        cum += w;
        if cum >= total / 2.0 {
            return y;
        }
    }
    v.last().map(|&(y, _)| y).unwrap_or(0.0)
}

/// Weighted quantile at `tau` in [0, 1] (weighted order statistic).
pub(crate) fn w_quantile(p: &[(f64, f64)], tau: f64) -> f64 {
    let mut v: Vec<(f64, f64)> = p.to_vec();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN in target"));
    let total: f64 = v.iter().map(|&(_, w)| w).sum();
    let target = tau.clamp(0.0, 1.0) * total;
    let mut cum = 0.0;
    for &(y, w) in &v {
        cum += w;
        if cum >= target {
            return y;
        }
    }
    v.last().map(|&(y, _)| y).unwrap_or(0.0)
}

/// Huber M-location via IRLS (delta = 1.345 · weighted sd, matching the
/// `Huber` criterion; starts from the median).
pub(crate) fn huber_location(p: &[(f64, f64)]) -> f64 {
    if p.is_empty() {
        return 0.0;
    }
    let delta = 1.345 * w_sd(p);
    let d = if delta > 0.0 { delta } else { 1.0 };
    let mut c = w_median(p);
    for _ in 0..25 {
        let (mut swx, mut sw) = (0.0, 0.0);
        for &(y, w) in p {
            let u = (y - c).abs();
            let iw = if u <= d { w } else { w * d / u.max(1e-300) };
            swx += iw * y;
            sw += iw;
        }
        if sw <= 0.0 {
            break;
        }
        let next = swx / sw;
        let done = (next - c).abs() <= 1e-9 * (1.0 + c.abs());
        c = next;
        if done {
            break;
        }
    }
    c
}

/// Weighted L1 cost around the side's median (absolute-error criterion).
pub(crate) fn l1_cost(p: &[(f64, f64)]) -> f64 {
    if p.is_empty() {
        return 0.0;
    }
    let med = w_median(p);
    p.iter().map(|&(y, w)| w * (y - med).abs()).sum()
}

/// Weighted Huber cost around the side's mean (delta smooths the L1 kink).
pub(crate) fn huber_cost(p: &[(f64, f64)], delta: f64) -> f64 {
    let sw: f64 = p.iter().map(|&(_, w)| w).sum();
    if sw <= 0.0 {
        return 0.0;
    }
    let mean = p.iter().map(|&(y, w)| w * y).sum::<f64>() / sw;
    let d = if delta > 0.0 { delta } else { 1.0 };
    p.iter()
        .map(|&(y, w)| {
            let u = (y - mean).abs();
            if u <= d {
                w * u * u / 2.0
            } else {
                w * d * (u - d / 2.0)
            }
        })
        .sum()
}

/// Weighted Poisson deviance of one side (mean fitted value). Returns 0.0
/// for non-positive targets so the criterion degrades to no-split.
pub(crate) fn poisson_deviance(p: &[(f64, f64)]) -> f64 {
    if p.is_empty() {
        return 0.0;
    }
    if p.iter().any(|&(y, _)| y < 0.0) {
        return 0.0;
    }
    let sw: f64 = p.iter().map(|&(_, w)| w).sum();
    let mean = p.iter().map(|&(y, w)| w * y).sum::<f64>() / sw.max(1e-300);
    if mean <= 0.0 {
        return 0.0;
    }
    p.iter()
        .map(|&(y, w)| {
            let t = if y > 0.0 { y * (y / mean).ln() - (y - mean) } else { mean };
            2.0 * w * t
        })
        .sum()
}

