//! Calendar and lag features — the featuregen `time_features` lineage, on
//! dependency-free civil-date math (Hinnant's days_from_civil).
//!
//! Dates are ISO `YYYY-MM-DD`. Feature values are numeric encodings so they
//! flow straight into tree learners; names/casing of featuregen's string
//! versions (month names, "week_N") are presentation and stay in Python.

use crate::transform::{summarize, GroupedTransform, SummaryKind};

/// Season encoding: 0 spring, 1 summer, 2 fall, 3 winter (northern calendar
/// windows from featuregen: spring 3/21–6/20, summer 6/21–9/22, fall 9/23–12/22).
///
/// Note: featuregen's southern-hemisphere shift `(s + 2) % 3` collapses winter
/// into fall; we use the correct six-month shift `(s + 2) % 4`.
pub fn season(month: u32, day: u32, southern: bool) -> u8 {
    let md = month * 100 + day;
    let s = if (320..621).contains(&md) {
        0
    } else if (620..923).contains(&md) {
        1
    } else if (922..1223).contains(&md) {
        2
    } else {
        3
    };
    if southern {
        ((s + 2) % 4) as u8
    } else {
        s
    }
}

/// Parse ISO `YYYY-MM-DD` (also accepts a `T`-separated timestamp prefix).
pub fn parse_ymd(s: &str) -> Option<(i32, u32, u32)> {
    let s = s.split('T').next().unwrap_or(s);
    let mut it = s.split('-');
    let y: i32 = it.next()?.parse().ok()?;
    let m: u32 = it.next()?.parse().ok()?;
    let d: u32 = it.next()?.parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    Some((y, m, d))
}

/// Days since 1970-01-01 (proleptic Gregorian).
pub fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y } as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Inverse of `days_from_civil`.
pub fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    ((if m <= 2 { y + 1 } else { y }) as i32, m, d)
}

/// Day of week for days-since-epoch: 0 = Monday ... 6 = Sunday.
pub fn weekday(days: i64) -> u8 {
    // 1970-01-01 (days=0) was a Thursday
    (((days % 7) + 10) % 7) as u8
}

/// Day of year, 1-based.
pub fn day_of_year(y: i32, m: u32, d: u32) -> u32 {
    (days_from_civil(y, m, d) - days_from_civil(y, 1, 1) + 1) as u32
}

/// ISO week number and its ISO week-year (may differ from the calendar year).
pub fn iso_week(y: i32, m: u32, d: u32) -> (i32, u32) {
    let days = days_from_civil(y, m, d);
    // Thursday of the current ISO week determines the week-year
    let dow = weekday(days);
    let thursday = days + (3 - dow as i64);
    let (wy, _, _) = civil_from_days(thursday);
    let jan1 = days_from_civil(wy, 1, 1);
    let week = ((thursday - jan1) / 7 + 1) as u32;
    (wy, week)
}

/// Every calendar feature for one date, as named numeric parts.
pub fn calendar_parts(y: i32, m: u32, d: u32, southern: bool) -> Vec<(&'static str, f64)> {
    let days = days_from_civil(y, m, d);
    let dow = weekday(days);
    let (iso_y, iso_w) = iso_week(y, m, d);
    vec![
        ("year", y as f64),
        ("halfyear", if m > 6 { 1.0 } else { 0.0 }),
        ("quarter", ((m - 1) / 3 + 1) as f64),
        ("month", m as f64),
        ("week", iso_w as f64),
        ("iso_year", iso_y as f64),
        ("weekday", dow as f64),
        ("day", d as f64),
        ("day_of_year", day_of_year(y, m, d) as f64),
        ("season", season(m, d, southern) as f64),
        ("weekend", if dow >= 5 { 1.0 } else { 0.0 }),
    ]
}

/// Lag values by whole days within each key (featuregen `LagFeatures`, day
/// granularity): row i gets the value of its key at `date - lag_days`.
/// Duplicated (key, date) rows keep the last occurrence; nothing found -> None.
pub fn lag_by_days(
    keys: &[Option<String>],
    dates: &[Option<String>],
    values: &[Option<f64>],
    lag_days: u32,
) -> Result<Vec<Option<f64>>, String> {
    if keys.len() != dates.len() || keys.len() != values.len() {
        return Err("keys, dates and values length mismatch".into());
    }
    let mut index: std::collections::HashMap<(String, i64), Option<f64>> =
        std::collections::HashMap::new();
    for ((k, dt), v) in keys.iter().zip(dates).zip(values) {
        if let (Some(k), Some(dt)) = (k, dt) {
            let (y, m, d) = parse_ymd(dt).ok_or_else(|| format!("unparseable date '{dt}'"))?;
            let v = match v {
                Some(x) if x.is_nan() => None,
                other => *other,
            };
            index.insert((k.clone(), days_from_civil(y, m, d)), v);
        }
    }
    let lag = lag_days as i64;
    let mut out = Vec::with_capacity(keys.len());
    for ((k, dt), v) in keys.iter().zip(dates).zip(values) {
        out.push(match (k, dt, v) {
            (Some(k), Some(dt), Some(v)) if !v.is_nan() => {
                let (y, m, d) = parse_ymd(dt).ok_or_else(|| format!("unparseable date '{dt}'"))?;
                index
                    .get(&(k.clone(), days_from_civil(y, m, d) - lag))
                    .copied()
                    .flatten()
            }
            _ => None,
        });
    }
    Ok(out)
}

/// Time bucket label for a date under a granularity spec. Components are
/// joined with `-` and come from {year, halfyear, season, quarter, month,
/// week (ISO), weekday, day}; composites like "year_month" or "month_weekday"
/// work (legacy `VariableTransformation` TODO: "trend at level of granularity
/// of time (year, year-month, year-week, ...) given by user").
pub fn time_bucket(
    y: i32,
    m: u32,
    d: u32,
    granularity: &str,
    southern: bool,
) -> Result<String, String> {
    let days = days_from_civil(y, m, d);
    let dow = weekday(days);
    let (_, iso_w) = iso_week(y, m, d);
    let mut parts = Vec::new();
    for comp in granularity.split('_') {
        parts.push(match comp {
            "year" => format!("{y:04}"),
            "halfyear" => format!("H{}", if m > 6 { 2 } else { 1 }),
            "season" => format!("S{}", season(m, d, southern)),
            "quarter" => format!("Q{}", (m - 1) / 3 + 1),
            "month" => format!("{m:02}"),
            "week" => format!("W{iso_w:02}"),
            "weekday" => format!("D{dow}"),
            "day" => format!("{d:02}"),
            other => return Err(format!("unknown time granularity component '{other}'")),
        });
    }
    Ok(parts.join("-"))
}

/// Grouped summary trended at a time granularity: each row gets the summary of
/// its (key x time-bucket) cell. Rows with a missing key/date/value or unseen
/// cell give `None`.
pub fn trended_transform(
    keys: &[Option<String>],
    dates: &[Option<String>],
    values: &[Option<f64>],
    granularity: &str,
    kind: SummaryKind,
    southern: bool,
) -> Result<Vec<Option<f64>>, String> {
    if keys.len() != dates.len() || keys.len() != values.len() {
        return Err("keys, dates and values length mismatch".into());
    }
    let mut composite = Vec::with_capacity(keys.len());
    for ((k, dt), v) in keys.iter().zip(dates).zip(values) {
        composite.push(match (k, dt, v) {
            (Some(k), Some(dt), Some(v)) if !v.is_nan() => {
                let (y, m, d) = parse_ymd(dt).ok_or_else(|| format!("unparseable date '{dt}'"))?;
                Some(format!(
                    "{k}|{}",
                    time_bucket(y, m, d, granularity, southern)?
                ))
            }
            _ => None,
        });
    }
    let mut gt = GroupedTransform::new();
    gt.fit(&composite, values, kind)?;
    gt.transform(&composite)
}

/// Additive decomposition output: series − trend, series − seasonal,
/// series − trend − seasonal (legacy TODO: "Decomposed (series -trend
/// -seasonality) based variable transformation").
#[derive(Clone, Debug)]
pub struct Decomposed {
    pub detrended: Vec<Option<f64>>,
    pub deseasonalized: Vec<Option<f64>>,
    pub residual: Vec<Option<f64>>,
}

/// Classical additive decomposition per key series, aligned with the
/// statsmodels `seasonal_decompose` (additive) convention: the trend is a
/// centered moving average over the key's rows ordered by date
/// (`trend_window` rows, edges truncated); the seasonal component is the mean
/// of (series − trend) per (key, season bucket), estimated only from rows with
/// a complete trend window (its nanmean-at-edges semantics); residual =
/// series − trend − seasonal. `season_granularity` is a `time_bucket` spec
/// such as "month" or "weekday". Incomplete rows (missing key/date/value) are
/// dropped from estimation and produce missing outputs.
#[allow(clippy::too_many_arguments)]
pub fn decomposed_transform(
    keys: &[Option<String>],
    dates: &[Option<String>],
    values: &[Option<f64>],
    season_granularity: &str,
    trend_window: usize,
    southern: bool,
) -> Result<Decomposed, String> {
    if keys.len() != dates.len() || keys.len() != values.len() {
        return Err("keys, dates and values length mismatch".into());
    }
    if trend_window == 0 {
        return Err("trend_window must be >= 1".into());
    }
    // (key, days, season bucket, value) per row; complete rows only
    let mut rows: Vec<Option<(String, i64, String, Option<f64>)>> = Vec::with_capacity(keys.len());
    for ((k, dt), v) in keys.iter().zip(dates).zip(values) {
        rows.push(match (k, dt) {
            (Some(k), Some(dt)) => {
                let (y, m, d) = parse_ymd(dt).ok_or_else(|| format!("unparseable date '{dt}'"))?;
                let bucket = time_bucket(y, m, d, season_granularity, southern)?;
                Some((k.clone(), days_from_civil(y, m, d), bucket, *v))
            }
            _ => None,
        });
    }

    use std::collections::HashMap as Map;
    let mut by_key: Map<String, Vec<usize>> = Map::new();
    for (i, r) in rows.iter().enumerate() {
        if let Some((k, _, _, _)) = r {
            by_key.entry(k.clone()).or_default().push(i);
        }
    }

    // trend = centered moving average; detrended feeds the seasonal component
    let mut detrended = vec![None; rows.len()];
    let mut per_bucket: Map<(String, String), Vec<f64>> = Map::new();
    for (_, mut idxs) in by_key {
        idxs.sort_by_key(|&i| (rows[i].as_ref().unwrap().1, i));
        let n = idxs.len();
        let half = trend_window / 2;
        for (pos, &i) in idxs.iter().enumerate() {
            let val = match rows[i].as_ref().unwrap().3 {
                Some(v) if !v.is_nan() => v,
                _ => continue,
            };
            let lo = pos.saturating_sub(half);
            let hi = (pos + half + 1).min(n);
            let win: Vec<f64> = idxs[lo..hi]
                .iter()
                .filter_map(|&j| rows[j].as_ref().unwrap().3)
                .filter(|v| !v.is_nan())
                .collect();
            if win.is_empty() {
                continue;
            }
            let trend = win.iter().sum::<f64>() / win.len() as f64;
            let det = val - trend;
            detrended[i] = Some(det);
            // the seasonal component is estimated only from rows whose trend
            // window is complete; edge estimates would bias it
            if pos >= half && pos + half < n {
                let r = rows[i].as_ref().unwrap();
                per_bucket
                    .entry((r.0.clone(), r.2.clone()))
                    .or_default()
                    .push(det);
            }
        }
    }

    let seasonal: Map<(String, String), f64> = per_bucket
        .iter()
        .map(|(k, vs)| (k.clone(), vs.iter().sum::<f64>() / vs.len() as f64))
        .collect();

    let mut deseasonalized = vec![None; rows.len()];
    let mut residual = vec![None; rows.len()];
    for (i, r) in rows.iter().enumerate() {
        if let Some((k, _, bucket, Some(v))) = r {
            if let Some(s) = seasonal.get(&(k.clone(), bucket.clone())) {
                deseasonalized[i] = Some(v - s);
                if let Some(det) = detrended[i] {
                    residual[i] = Some(det - s);
                }
            }
        }
    }
    Ok(Decomposed {
        detrended,
        deseasonalized,
        residual,
    })
}

/// Trailing-window aggregation per key: each row summarizes the observed
/// values of its key in the day window ending at its own date. With
/// `include_current` the window covers (date − window_days, date]; otherwise
/// strictly earlier days (date − window_days, date). Empty/all-missing windows
/// give `None`.
pub fn windowed_transform(
    keys: &[Option<String>],
    dates: &[Option<String>],
    values: &[Option<f64>],
    window_days: u32,
    kind: SummaryKind,
    include_current: bool,
) -> Result<Vec<Option<f64>>, String> {
    if keys.len() != dates.len() || keys.len() != values.len() {
        return Err("keys, dates and values length mismatch".into());
    }
    if window_days == 0 {
        return Err("window_days must be >= 1".into());
    }
    use std::collections::HashMap as Map;
    let mut parsed: Vec<Option<(String, i64, Option<f64>)>> = Vec::with_capacity(keys.len());
    for ((k, dt), v) in keys.iter().zip(dates).zip(values) {
        parsed.push(match (k, dt, v) {
            (Some(k), Some(dt), Some(v)) if !v.is_nan() => {
                let (y, m, d) = parse_ymd(dt).ok_or_else(|| format!("unparseable date '{dt}'"))?;
                Some((k.clone(), days_from_civil(y, m, d), Some(*v)))
            }
            _ => None,
        });
    }
    let mut by_key: Map<String, Vec<(i64, Option<f64>)>> = Map::new();
    for p in parsed.iter().flatten() {
        by_key.entry(p.0.clone()).or_default().push((p.1, p.2));
    }
    for series in by_key.values_mut() {
        series.sort_by_key(|(d, _)| *d);
    }

    let w = window_days as i64;
    let mut out = Vec::with_capacity(parsed.len());
    for p in &parsed {
        out.push(match p {
            None => None,
            Some((k, day, _)) => {
                let series = &by_key[k];
                let (lo, hi) = if include_current {
                    (day - w + 1, day + 1)
                } else {
                    (day - w, *day)
                };
                let win: Vec<Option<f64>> = series
                    .iter()
                    .filter(|(d, _)| *d >= lo && *d < hi)
                    .map(|(_, v)| *v)
                    .collect();
                let s = summarize(&win);
                if s.count == 0 {
                    None
                } else {
                    Some(s.value(kind))
                }
            }
        });
    }
    Ok(out)
}
