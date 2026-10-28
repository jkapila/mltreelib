use mltreelib_core::timefeat::{
    calendar_parts, civil_from_days, day_of_year, days_from_civil, iso_week, lag_by_days,
    parse_ymd, season, weekday,
};

#[test]
fn civil_date_math_roundtrips_and_known_days() {
    assert_eq!(days_from_civil(1970, 1, 1), 0);
    assert_eq!(days_from_civil(2000, 1, 1), 10957);
    assert_eq!(civil_from_days(0), (1970, 1, 1));
    assert_eq!(civil_from_days(10957), (2000, 1, 1));
    // leap day
    assert_eq!(days_from_civil(2024, 2, 29) - days_from_civil(2024, 1, 1), 59);
    for days in [-1000_i64, 0, 1, 365, 20000, 20245] {
        assert_eq!(days_from_civil(civil_from_days(days).0, civil_from_days(days).1, civil_from_days(days).2), days);
    }
}

#[test]
fn weekday_matches_known_dates() {
    assert_eq!(weekday(0), 3); // 1970-01-01 Thursday
    let (y, m, d) = (2000, 1, 1); // Saturday
    assert_eq!(weekday(days_from_civil(y, m, d)), 5);
    let (y, m, d) = (2024, 2, 29); // Thursday
    assert_eq!(weekday(days_from_civil(y, m, d)), 3);
}

#[test]
fn iso_week_handles_year_boundaries() {
    assert_eq!(iso_week(2020, 1, 1), (2020, 1)); // Wed -> week 1 of 2020
    assert_eq!(iso_week(2021, 1, 1), (2020, 53)); // Fri -> week 53 of 2020
    assert_eq!(iso_week(2026, 1, 1), (2026, 1)); // Thu -> week 1 of 2026
    assert_eq!(iso_week(1970, 1, 1), (1970, 1)); // Thu
}

#[test]
fn season_windows_and_hemisphere() {
    assert_eq!(season(3, 21, false), 0); // spring starts
    assert_eq!(season(6, 20, false), 0);
    assert_eq!(season(6, 21, false), 1); // summer starts
    assert_eq!(season(9, 23, false), 2); // fall starts
    assert_eq!(season(12, 25, false), 3); // winter
    assert_eq!(season(1, 15, false), 3);
    // southern: six-month shift
    assert_eq!(season(1, 15, true), 1);
    assert_eq!(season(7, 15, true), 3);
}

#[test]
fn calendar_parts_are_consistent() {
    let parts = calendar_parts(2026, 10, 28, false);
    let get = |name: &str| {
        parts
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| *v)
            .unwrap()
    };
    assert_eq!(get("year"), 2026.0);
    assert_eq!(get("month"), 10.0);
    assert_eq!(get("quarter"), 4.0);
    assert_eq!(get("halfyear"), 1.0);
    assert_eq!(get("day"), 28.0);
    assert_eq!(get("day_of_year"), 301.0);
    assert_eq!(day_of_year(2026, 12, 31), 365);
    assert_eq!(day_of_year(2024, 12, 31), 366); // leap year
    // 2026-10-28 is a Wednesday
    assert_eq!(get("weekday"), 2.0);
    assert_eq!(get("weekend"), 0.0);
}

#[test]
fn parse_ymd_accepts_date_and_timestamp() {
    assert_eq!(parse_ymd("2026-10-28"), Some((2026, 10, 28)));
    assert_eq!(parse_ymd("2026-10-28T14:30:00"), Some((2026, 10, 28)));
    assert_eq!(parse_ymd("not-a-date"), None);
}

#[test]
fn lag_by_days_within_key() {
    let keys = vec![
        Some("a".into()),
        Some("a".into()),
        Some("a".into()),
        Some("b".into()),
        None,
    ];
    let dates = vec![
        Some("2026-01-01".into()),
        Some("2026-01-02".into()),
        Some("2026-01-05".into()),
        Some("2026-01-02".into()),
        Some("2026-01-02".into()),
    ];
    let vals = vec![Some(10.0), Some(20.0), Some(30.0), Some(99.0), Some(5.0)];

    // lag 1 day: row 2 gets value of "a" at 2026-01-01; row 3 has no 01-04 -> None
    let out = lag_by_days(&keys, &dates, &vals, 1).unwrap();
    assert_eq!(out[0], None); // no 2025-12-31
    assert_eq!(out[1], Some(10.0));
    assert_eq!(out[2], None);
    assert_eq!(out[3], None); // "b" has no previous day
    assert_eq!(out[4], None); // missing key

    // lag 0 returns the row's own value (lookup semantics)
    let out = lag_by_days(&keys, &dates, &vals, 0).unwrap();
    assert_eq!(out, vec![Some(10.0), Some(20.0), Some(30.0), Some(99.0), None]);

    // a 4-day lag reaches across the gap: row 3 (01-05) gets 01-01 value
    let out = lag_by_days(&keys, &dates, &vals, 4).unwrap();
    assert_eq!(out[2], Some(10.0));
}
