//! Calendar and lag features — the featuregen `time_features` lineage, on
//! dependency-free civil-date math (Hinnant's days_from_civil).
//!
//! Dates are ISO `YYYY-MM-DD`. Feature values are numeric encodings so they
//! flow straight into tree learners; names/casing of featuregen's string
//! versions (month names, "week_N") are presentation and stay in Python.

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
            index.insert((k.clone(), days_from_civil(y, m, d)), *v);
        }
    }
    let lag = lag_days as i64;
    let mut out = Vec::with_capacity(keys.len());
    for (k, dt) in keys.iter().zip(dates) {
        out.push(match (k, dt) {
            (Some(k), Some(dt)) => {
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
