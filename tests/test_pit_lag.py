"""Tests for pit_lag.py: one or more per market rule, then filing dates and times, and audit().

Run them with pytest, or without it:
    python tests/test_pit_lag.py
Where a public holiday decides the answer, the test expects the holiday to be skipped
when the holidays package is installed and not otherwise.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import traceback
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pit_lag  # noqa: E402

HAVE_HOLIDAYS = importlib.util.find_spec("holidays") is not None


def day(period_end, market, report_type, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pit_lag.available_date(period_end, market, report_type, **kwargs)


def check(got, expected: str) -> None:
    assert got == pd.Timestamp(expected), f"got {got}, expected {expected}"


def raises(function, *args, match: str = "", **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert match in str(error), f"{match!r} not in {str(error)!r}"
    else:
        raise AssertionError(f"expected a ValueError from {args} {kwargs}")


# ---------------------------------------------------------------------------
# The rules table
# ---------------------------------------------------------------------------

def test_rules_table():
    rules = pit_lag.RULES
    assert list(rules.columns) == ["market", "report_type", "filer_class", "effective", "days", "months",
                                   "extension_days", "source"]
    assert pd.api.types.is_datetime64_any_dtype(rules["effective"])
    assert ((rules["days"] > 0) ^ (rules["months"] > 0)).all()
    for market in ("US", "KR", "HK", "CN", "TH"):
        assert "annual" in set(rules.loc[rules["market"] == market, "report_type"])


# ---------------------------------------------------------------------------
# US: by filer status, the 12b-25 extension and the 2006 rule
# ---------------------------------------------------------------------------

def test_us_10k_by_filer_status():
    # FY2024: 60 days is Sat 1 Mar 2025 (due Mon 3 Mar); 75 is Sun 16 Mar (Mon 17); 90 is Mon 31 Mar.
    for status, expected in (("large accelerated", "2025-03-04"), ("accelerated", "2025-03-18"),
                             ("non-accelerated", "2025-04-01")):
        check(day("2024-12-31", "US", "10-K", filer_class=status, extension=False), expected)
    for code, expected in (("1-LAF", "2025-03-04"), ("2-ACC", "2025-03-18"), ("3-SRA", "2025-03-18"),
                           ("4-NON", "2025-04-01"), ("5-SML", "2025-04-01")):
        check(day("2024-12-31", "US", "10-K", filer_class=code, extension=False), expected)


def test_us_10q_by_filer_status():
    # Q1 2025: 40 days is Sat 10 May (due Mon 12 May); 45 days is Thu 15 May.
    for status, expected in (("large accelerated", "2025-05-13"), ("accelerated", "2025-05-13"),
                             ("non-accelerated", "2025-05-16")):
        check(day("2025-03-31", "US", "10-Q", filer_class=status, extension=False), expected)


def test_us_12b25_extension():
    check(day("2024-12-31", "US", "annual", filer_class="large accelerated"), "2025-03-19")  # Mon 3 Mar + 15
    check(day("2025-03-31", "US", "quarterly", filer_class="non-accelerated"), "2025-05-21")  # Thu 15 May + 5


def test_us_unknown_filer_status_takes_the_latest_deadline():
    check(day("2024-12-31", "US", "10-K", extension=False), "2025-04-01")
    check(day("2025-03-31", "US", "10-Q", extension=False), "2025-05-16")


def test_us_deadlines_before_and_after_2006():
    check(day("2005-12-31", "US", "10-K", filer_class="large accelerated", extension=False), "2006-04-03")  # 90
    check(day("2006-12-31", "US", "10-K", filer_class="large accelerated", extension=False), "2007-03-02")  # 60
    check(day("2006-09-30", "US", "10-Q", filer_class="accelerated", extension=False), "2006-11-15")  # 45
    check(day("2007-03-31", "US", "10-Q", filer_class="accelerated", extension=False), "2007-05-11")  # 40


def test_us_deadline_on_a_federal_holiday():
    # 40 days after 25 May 2025 is Fri 4 Jul, so the deadline is Mon 7 Jul. Comes from pandas, always.
    check(day("2025-05-25", "US", "10-Q", filer_class="accelerated", extension=False), "2025-07-08")


# ---------------------------------------------------------------------------
# Korea
# ---------------------------------------------------------------------------

def test_kr_quarterly_45_days():
    check(day("2025-03-31", "KR", "quarterly"), "2025-05-16")  # Thu 15 May


def test_kr_half_year_45_days():
    # Thu 14 Aug 2025; Fri 15 Aug is Liberation Day.
    check(day("2025-06-30", "KR", "half-year"), "2025-08-18" if HAVE_HOLIDAYS else "2025-08-15")


def test_kr_annual_90_days():
    check(day("2024-12-31", "KR", "annual"), "2025-04-01")  # Mon 31 Mar
    check(day("2018-12-31", "KR", "annual"), "2019-04-02")  # Sun 31 Mar, so Mon 1 Apr (KIND's deadline day)
    check(day("2023-12-31", "KR", "annual"), "2024-04-02")  # leap year: Sat 30 Mar, so Mon 1 Apr


def test_kr_deadline_on_a_holiday():
    # 45 days after Q1 2024 is Wed 15 May 2024, Buddha's Birthday, so the deadline is Thu 16 May.
    check(day("2024-03-31", "KR", "quarterly"), "2024-05-17" if HAVE_HOLIDAYS else "2024-05-16")


# ---------------------------------------------------------------------------
# Hong Kong, with its 2010 change
# ---------------------------------------------------------------------------

def test_hk_annual_results_4_then_3_months():
    check(day("2009-12-31", "HK", "annual"), "2010-05-03")  # 4 months: Fri 30 Apr 2010
    check(day("2010-12-31", "HK", "annual"), "2011-04-01")  # 3 months: Thu 31 Mar 2011


def test_hk_interim_results_3_then_2_months():
    check(day("2009-04-30", "HK", "interim"), "2009-08-03")  # 3 months: Fri 31 Jul 2009
    check(day("2010-06-30", "HK", "half-year"), "2010-09-01")  # 2 months: Tue 31 Aug 2010, not 30 Aug


def test_hk_quarterly_has_no_deadline():
    raises(pit_lag.available_date, "2025-03-31", "HK", "quarterly", match="no deadline")
    check(day("2025-03-31", "HK", "quarterly", filed="2025-05-12"), "2025-05-13")


# ---------------------------------------------------------------------------
# China
# ---------------------------------------------------------------------------

def test_cn_annual_4_months():
    # Wed 30 Apr 2025; the market is shut 1-5 May for Labour Day.
    check(day("2024-12-31", "CN", "annual"), "2025-05-06" if HAVE_HOLIDAYS else "2025-05-01")


def test_cn_half_year_2_months():
    check(day("2025-06-30", "CN", "half-year"), "2025-09-02")  # Sun 31 Aug, so Mon 1 Sep


def test_cn_quarterly_1_month():
    check(day("2025-09-30", "CN", "quarterly"), "2025-11-03")  # Fri 31 Oct


# ---------------------------------------------------------------------------
# Thailand
# ---------------------------------------------------------------------------

def test_th_quarterly_and_half_year_45_days():
    check(day("2025-03-31", "TH", "quarterly"), "2025-05-16")  # Thu 15 May
    check(day("2025-06-30", "TH", "half-year"), "2025-08-15")  # Thu 14 Aug


def test_th_annual_3_or_2_months():
    check(day("2024-12-31", "TH", "annual"), "2025-04-01")  # 3 months: Mon 31 Mar
    check(day("2024-12-31", "TH", "annual", filer_class="no Q4 statements"), "2025-03-03")  # Fri 28 Feb
    check(day("2024-12-31", "TH", "annual", filer_class="with Q4 statements"), "2025-04-01")
    # Thai quarterly rules ignore the annual-report class rather than rejecting it.
    check(day("2025-03-31", "TH", "quarterly", filer_class="no Q4 statements"), "2025-05-16")


# ---------------------------------------------------------------------------
# Filing dates and times
# ---------------------------------------------------------------------------

def test_filing_date_beats_the_deadline():
    check(day("2025-03-31", "KR", "quarterly", filed="2025-04-29"), "2025-04-30")
    check(day("2024-12-31", "US", "10-K", filed="2025-02-14"), "2025-02-18")  # Mon 17 Feb is Presidents' Day
    check(day("2025-03-31", "KR", "quarterly", filed="2025-05-30"), "2025-06-02")  # a late filer


def test_filing_time_at_the_close_and_at_the_open():
    check(day("2024-12-31", "US", "10-K", accepted="2025-02-12 15:59"), "2025-02-12")
    check(day("2024-12-31", "US", "10-K", accepted="2025-02-12 16:05"), "2025-02-13")
    check(day("2024-12-31", "US", "10-K", accepted="2025-02-12 08:15", at="open"), "2025-02-12")
    check(day("2024-12-31", "US", "10-K", accepted="2025-02-12 10:00", at="open"), "2025-02-13")
    check(day("2024-12-31", "US", "10-K", accepted="2025-02-14 17:00"), "2025-02-18")
    check(day("2024-12-31", "US", "10-K", accepted="2025-02-15 10:00"), "2025-02-18")  # a Saturday
    check(day("2024-12-31", "US", "10-K", accepted="2025-02-12 00:00"), "2025-02-13")  # midnight: date only
    check(day("2024-12-31", "US", "10-K", filed="2025-02-13", accepted="2025-02-12 09:00"), "2025-02-12")
    raises(pit_lag.available_date, "2024-12-31", "US", "10-K", accepted="2025-02-12 09:00", at="noon",
           match="at must be")


def test_filing_time_zones_and_the_korean_close():
    def utc(text):
        return pd.Timestamp(text, tz="UTC")
    check(day("2025-03-31", "KR", "quarterly", accepted=utc("2025-05-15 06:00")), "2025-05-15")  # 15:00 Seoul
    check(day("2025-03-31", "KR", "quarterly", accepted=utc("2025-05-15 06:45")), "2025-05-16")  # 15:45 Seoul
    check(day("2015-03-31", "KR", "quarterly", accepted="2015-05-14 15:10"), "2015-05-15")  # closed 15:00 then


# ---------------------------------------------------------------------------
# Many rows at once, and bad input
# ---------------------------------------------------------------------------

def test_series_in_series_out():
    ends = pd.Series(pd.to_datetime(["2025-03-31", "2025-03-31", "2024-12-31", None]), index=list("abcd"))
    got = day(ends, ["KR", "US", "TH", "KR"], ["quarterly", "10-Q", "annual", "annual"],
              filer_class=[None, "1-LAF", None, None], filed=[None, "2025-04-30", None, None], extension=False)
    assert got.name == "available_date"
    assert list(got.index) == list("abcd")
    assert list(got.iloc[:3].dt.strftime("%Y-%m-%d")) == ["2025-05-16", "2025-05-01", "2025-04-01"]
    assert pd.isna(got.iloc[3])


def test_bad_input_raises():
    raises(pit_lag.available_date, "2025-03-31", "JP", "quarterly", match="unknown market")
    raises(pit_lag.available_date, "2025-03-31", "US", "10-Q", filer_class="mega", match="unknown US filer class")
    raises(pit_lag.available_date, "2025-03-31", "KR", "quarterly", filer_class="KOSPI", match="no filer classes")
    raises(pit_lag.available_date, ["2025-03-31"] * 2, "KR", ["quarterly"] * 3, match="different lengths")


def test_extra_holidays():
    check(day("2025-03-31", "TH", "quarterly", extra_holidays={"TH": ["2025-05-16"]}), "2025-05-19")


def test_without_the_holidays_package():
    original = pit_lag._holiday_package
    pit_lag._holiday_package = lambda: None
    pit_lag._base_holidays.cache_clear()
    pit_lag._WARNED.clear()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check(pit_lag.available_date("2025-06-30", "KR", "half-year"), "2025-08-15")
            check(pit_lag.available_date("2024-12-31", "US", "10-K", filed="2025-02-14"), "2025-02-18")
        messages = [str(warning.message) for warning in caught]
        assert any("holidays package" in message and "KR" in message for message in messages), messages
        assert not any("US public holidays" in message for message in messages), messages
    finally:
        pit_lag._holiday_package = original
        pit_lag._base_holidays.cache_clear()
        pit_lag._WARNED.clear()


# ---------------------------------------------------------------------------
# audit()
# ---------------------------------------------------------------------------

def test_audit_flags_rows_used_too_early():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2025-05-15", "2025-05-16", "2025-05-14", "2025-05-15", "2025-06-02"]),
        "period_end": pd.to_datetime(["2025-03-31"] * 4 + [None]),
        "filed": pd.to_datetime([None, None, "2025-05-14", "2025-05-14", None]),
    })
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pit_lag.audit(panel, market="KR", report_type="quarterly")
    assert result["too_early"].tolist() == [True, False, True, False, False]
    assert result["days_early"].tolist() == [1, 0, 1, 0, 0]
    assert result["available_date"].iloc[0] == pd.Timestamp("2025-05-16")
    text = printed.getvalue()
    assert "2 of 5 rows (40.0%)" in text, text
    assert "filing dates for 2 and legal deadlines for 2" in text, text
    assert "1 rows had no period end" in text, text
    assert list(panel.columns) == ["date", "period_end", "filed"]  # the input is left alone


def test_audit_reads_columns_and_needs_a_market():
    panel = pd.DataFrame({"used_on": pd.to_datetime(["2025-03-04", "2025-03-03"]),
                          "period_end": pd.to_datetime(["2024-12-31"] * 2),
                          "market": ["US", "US"], "report_type": ["10-K", "10-K"], "afs": ["1-LAF", "1-LAF"]})
    result = pit_lag.audit(panel, used="used_on", filer_class="afs", extension=False, quiet=True)
    assert result["too_early"].tolist() == [False, True]
    raises(pit_lag.audit, panel.drop(columns="market"), used="used_on", quiet=True, match="market=")
    raises(pit_lag.audit, panel, used="used_on", filed="filing_date", quiet=True, match="no column 'filing_date'")


# ---------------------------------------------------------------------------

def main() -> int:
    tests = [(name, test) for name, test in globals().items() if name.startswith("test_") and callable(test)]
    failed = 0
    for name, test in tests:
        try:
            test()
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    status = "installed" if HAVE_HOLIDAYS else "not installed"
    print(f"\n{len(tests) - failed} passed, {failed} failed (holidays package {status})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
