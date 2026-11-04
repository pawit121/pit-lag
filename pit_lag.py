# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Paul Uth
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""pit_lag: the first day a company report can be used in a backtest.

Backtests that join fundamentals at the period end, or at a fixed lag, use some
reports before they were public. pit_lag dates each report by when it was public:

* the filing time, if you have it (accepted);
* otherwise the filing date (filed), plus one trading day, because many reports
  arrive after the close;
* otherwise the legal deadline for its market, report type and filer class, plus
  one trading day.

    >>> import pit_lag
    >>> pit_lag.available_date("2025-03-31", "KR", "quarterly")
    Timestamp('2025-05-16 00:00:00')
    >>> checked = pit_lag.audit(panel, used="date", market="KR", report_type="quarterly")

The result is the first trading day on which the report was public by the close
(at="close", the default) or before the open (at="open"). It works on any data
with period ends; no data is bundled. It needs pandas and numpy. Install the
holidays package as well (pip install holidays): without it only weekends and US
federal holidays are skipped, so a deadline that falls on a Korean, Hong Kong,
Chinese or Thai holiday comes out a day early.

Deadlines (RULES below; each row applies to period ends on or after `effective`):

US  10-K: 60, 75 or 90 days, and 10-Q: 40, 40 or 45 days, for large accelerated,
    accelerated and non-accelerated filers, for fiscal years ending on or after
    15 Dec 2006 (SEC Release 33-8644). Earlier periods get 90 and 45 days: exact
    before 2003, conservative during the 2003-06 phase-in. A Form 12b-25
    extension adds 15 days to a 10-K and 5 to a 10-Q; it is counted unless
    extension=False.
KR  Annual 90 days; half-year and quarterly 45 days (Capital Markets Act, Arts.
    159 and 160). Not modelled: the 60 days allowed in a company's first two
    years of consolidated reporting.
HK  Main Board. Annual results 3 months (4 for years ending before 31 Dec 2010);
    interim results 2 months (3 for periods ending before 30 Jun 2010) (Listing
    Rules 13.49). Quarterly reports are voluntary, so they have no deadline.
CN  Annual 4 months; half-year 2 months; quarterly (Q1 and Q3) 1 month (CSRC
    disclosure measures, 2021, Art. 13, and the exchange listing rules).
TH  Quarterly and half-year 45 days; annual 3 months, or 2 months for companies
    that file no Q4 statements (filer_class="no Q4 statements").

A deadline on a weekend or holiday moves to the next business day. Without a
filer class, the latest deadline for the market and report type applies, so an
unknown class never makes a report usable too early. No deadline catches late
filers, which are common among small US companies: use filing dates where you
have them.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

__version__ = "0.1.0"
__all__ = ["RULES", "SESSIONS", "available_date", "audit"]

_US = "SEC Forms 10-K and 10-Q; Release 33-8644; Rule 12b-25"
_US_OLD = "SEC deadlines before Release 33-8644 (conservative for 2003-06)"
_KR = "Capital Markets Act Arts. 159-160"
_HK = "HKEX Main Board Listing Rules 13.49(1) and 13.49(6)"
_CN = "CSRC disclosure measures (2021) Art. 13; exchange listing rules"
_TH = "SET rules on submitting financial statements"

RULES = pd.DataFrame.from_records(
    [
        # market, report_type, filer_class ("" = everyone), effective, days, months, extension_days, source
        ("US", "annual", "large accelerated", None, 90, 0, 15, _US_OLD),
        ("US", "annual", "large accelerated", "2006-12-15", 60, 0, 15, _US),
        ("US", "annual", "accelerated", None, 90, 0, 15, _US_OLD),
        ("US", "annual", "accelerated", "2006-12-15", 75, 0, 15, _US),
        ("US", "annual", "non-accelerated", None, 90, 0, 15, _US),
        ("US", "quarterly", "large accelerated", None, 45, 0, 5, _US_OLD),
        ("US", "quarterly", "large accelerated", "2006-12-15", 40, 0, 5, _US),
        ("US", "quarterly", "accelerated", None, 45, 0, 5, _US_OLD),
        ("US", "quarterly", "accelerated", "2006-12-15", 40, 0, 5, _US),
        ("US", "quarterly", "non-accelerated", None, 45, 0, 5, _US),
        ("KR", "annual", "", None, 90, 0, 0, _KR),
        ("KR", "half-year", "", None, 45, 0, 0, _KR),
        ("KR", "quarterly", "", None, 45, 0, 0, _KR),
        ("HK", "annual", "", None, 0, 4, 0, _HK),
        ("HK", "annual", "", "2010-12-31", 0, 3, 0, _HK),
        ("HK", "half-year", "", None, 0, 3, 0, _HK),
        ("HK", "half-year", "", "2010-06-30", 0, 2, 0, _HK),
        ("CN", "annual", "", None, 0, 4, 0, _CN),
        ("CN", "half-year", "", None, 0, 2, 0, _CN),
        ("CN", "quarterly", "", None, 0, 1, 0, _CN),
        ("TH", "annual", "with Q4 statements", None, 0, 3, 0, _TH),
        ("TH", "annual", "no Q4 statements", None, 0, 2, 0, _TH),
        ("TH", "half-year", "", None, 45, 0, 0, _TH),
        ("TH", "quarterly", "", None, 45, 0, 0, _TH),
    ],
    columns=["market", "report_type", "filer_class", "effective", "days", "months", "extension_days", "source"],
)
RULES["effective"] = pd.to_datetime(RULES["effective"])

SESSIONS = pd.DataFrame.from_records(
    [
        # market, time zone, effective, regular session open, close (local time)
        ("US", "America/New_York", None, "09:30", "16:00"),
        ("KR", "Asia/Seoul", None, "09:00", "15:00"),
        ("KR", "Asia/Seoul", "2016-08-01", "09:00", "15:30"),
        ("HK", "Asia/Hong_Kong", None, "09:30", "16:00"),
        ("CN", "Asia/Shanghai", None, "09:30", "15:00"),
        ("TH", "Asia/Bangkok", None, "10:00", "16:30"),
    ],
    columns=["market", "timezone", "effective", "open", "close"],
)
SESSIONS["effective"] = pd.to_datetime(SESSIONS["effective"])

# Exchange calendars in the holidays package, added to each country's public holidays.
_EXCHANGES = {"US": "XNYS", "KR": "XKRX", "HK": "XHKG", "CN": "XSHG"}

_MARKET_NAMES = {
    "us": "US", "usa": "US", "united states": "US",
    "kr": "KR", "kor": "KR", "korea": "KR", "south korea": "KR",
    "hk": "HK", "hkg": "HK", "hong kong": "HK",
    "cn": "CN", "chn": "CN", "china": "CN",
    "th": "TH", "tha": "TH", "thailand": "TH",
}
_REPORT_NAMES = {
    "10-k": "annual", "10-k/a": "annual", "10-kt": "annual", "annual report": "annual",
    "half year": "half-year", "half-yearly": "half-year", "semi-annual": "half-year",
    "semiannual": "half-year", "interim": "half-year",
    "quarter": "quarterly", "10-q": "quarterly", "10-q/a": "quarterly",
}
_CLASS_NAMES = {  # the SEC Financial Statement Data Sets' afs codes, and short forms
    "1-laf": "large accelerated", "laf": "large accelerated", "large accelerated filer": "large accelerated",
    "2-acc": "accelerated", "acc": "accelerated", "accelerated filer": "accelerated",
    "3-sra": "accelerated", "sra": "accelerated",
    "4-non": "non-accelerated", "non": "non-accelerated", "non-accelerated filer": "non-accelerated",
    "5-sml": "non-accelerated", "sml": "non-accelerated", "smaller reporting company": "non-accelerated",
    "no q4": "no q4 statements", "without q4": "no q4 statements", "without q4 statements": "no q4 statements",
    "with q4": "with q4 statements",
}

_NAT = np.datetime64("NaT", "D")
_DAWN = np.datetime64("0001-01-01", "D")  # stands in for "no start date" in RULES and SESSIONS
_WARNED: set[str] = set()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def available_date(period_end, market, report_type, filer_class=None, filed=None, accepted=None, *,
                   at: str = "close", extension: bool = True, extra_holidays: Mapping | None = None):
    """The first trading day on which a report was public by the close (or before the open).

    Every argument takes one value or a list-like (Series, array or list); list-likes
    must have the same length, and single values apply to every row.

    period_end      the last day of the period the report covers.
    market          "US", "KR", "HK", "CN" or "TH".
    report_type     "annual", "half-year" or "quarterly"; "10-K", "10-Q" and "interim" work too.
    filer_class     US filer status: "large accelerated", "accelerated" or "non-accelerated", or
                    the SEC data sets' afs codes such as "1-LAF". For Thai annual reports,
                    "no Q4 statements". Leave it out when unknown: the latest deadline applies.
    filed           the filing date. The report becomes usable the next trading day.
    accepted        the filing time. Usable the same day if it came before the close (or the
                    open, with at="open") of a trading day, else the next trading day. Naive
                    times are read as exchange time; a time of exactly midnight counts as a
                    date with no time.
    at              "close" for a backtest that trades on the close with what was public by
                    then; "open" for one that needs the report before the session starts.
    extension       count the US Form 12b-25 extension when falling back on the deadline.
    extra_holidays  {market: dates} to skip on top of the built-in calendars.

    Returns a Timestamp for single values, otherwise a Series named available_date (with the
    index of the first Series argument). Rows with neither a period end nor a filing date get NaT.
    """
    frame, single = _columns(period_end=period_end, market=market, report_type=report_type,
                             filer_class=filer_class, filed=filed, accepted=accepted)
    days = _available(_normalise(frame), at, extension, extra_holidays)
    if single:
        return pd.Timestamp(days[0])
    return pd.Series(days.astype("datetime64[ns]"), index=frame.index, name="available_date")


def audit(df: pd.DataFrame, used: str = "date", *, period_end: str = "period_end", market: str = "market",
          report_type: str = "report_type", filer_class: str = "filer_class", filed: str = "filed",
          accepted: str = "accepted", at: str = "close", extension: bool = True,
          extra_holidays: Mapping | None = None, quiet: bool = False) -> pd.DataFrame:
    """Flag the rows of a backtest that used a report before it was public.

    df has one row per use of a report: the date it was used (column `used`) and the
    report's period end, market and report type, plus its filer class, filing date or
    filing time where known. Pass column names; market, report_type and filer_class
    also take a single value for every row, as in
        audit(panel, used="date", market="KR", report_type="quarterly")

    Returns a copy of df with three more columns: available_date, too_early (used
    before available_date) and days_early (calendar days, 0 when not early).
    Prints a summary unless quiet=True.
    """
    def column(name, default, required=True, constant=False):
        if isinstance(name, str) and name in df.columns:
            return df[name]
        if name is None or (name == default and not required):
            return None
        if constant and name != default:
            return name
        hint = f"; pass {default}= a column name or a single value" if constant else ""
        raise ValueError(f"df has no column {name!r}{hint}")

    frame, _ = _columns(
        period_end=column(period_end, "period_end"),
        market=column(market, "market", constant=True),
        report_type=column(report_type, "report_type", constant=True),
        filer_class=column(filer_class, "filer_class", required=False, constant=True),
        filed=column(filed, "filed", required=False),
        accepted=column(accepted, "accepted", required=False),
        used=column(used, "date"),
    )
    frame = _normalise(frame)
    available = _available(frame, at, extension, extra_holidays)
    used_days = _days(_dates(frame["used"]))
    too_early = used_days < available  # False wherever either date is missing
    days_early = np.where(too_early, (available - used_days).astype("timedelta64[D]").astype(np.int64), 0)

    out = df.copy()
    out["available_date"] = available.astype("datetime64[ns]")
    out["too_early"] = too_early
    out["days_early"] = days_early
    if not quiet:
        print(_summary(frame, available, too_early, days_early))
    return out


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _columns(**values) -> tuple[pd.DataFrame, bool]:
    """Line the arguments up as columns; single values are repeated on every row."""
    lists = {name: value for name, value in values.items() if pd.api.types.is_list_like(value)}
    if not lists:
        return pd.DataFrame({name: [value] for name, value in values.items()}), True
    lengths = {len(value) for value in lists.values()}
    if len(lengths) > 1:
        raise ValueError(f"list-like arguments have different lengths: {sorted(lengths)}")
    size = lengths.pop()
    index = next((value.index for value in lists.values() if isinstance(value, pd.Series)), pd.RangeIndex(size))
    data = {}
    for name, value in values.items():
        if isinstance(value, (pd.Series, pd.Index)):
            data[name] = pd.Series(value.array, index=index)
        elif name in lists:
            data[name] = pd.Series(list(value), index=index, dtype=object if size == 0 else None)
        else:
            data[name] = pd.Series([value] * size, index=index, dtype=object if value is None else None)
    return pd.DataFrame(data, index=index), False


def _clean(values: pd.Series, names: Mapping[str, str]) -> pd.Series:
    """Strip, lower-case and resolve aliases; missing and blank values become ""."""
    def one(value):
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return ""
        key = str(value).strip().lower()
        return names.get(key, key)
    return values.astype(object).map(one)


def _dates(values: pd.Series) -> pd.Series:
    """Dates at midnight with no time zone (local dates for zoned values)."""
    dates = pd.to_datetime(values)
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.normalize()


def _days(values: pd.Series) -> np.ndarray:
    return np.asarray(values.to_numpy(), dtype="datetime64[D]")


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["period_end"] = _dates(frame["period_end"])
    out["market"] = _clean(frame["market"], _MARKET_NAMES).str.upper()
    out["report_type"] = _clean(frame["report_type"], _REPORT_NAMES)
    out["filer_class"] = _clean(frame["filer_class"], _CLASS_NAMES)
    out["filed"] = _dates(frame["filed"])
    out["accepted"] = pd.to_datetime(frame["accepted"])
    if "used" in frame:
        out["used"] = frame["used"]
    markets = set(out["market"])
    if "" in markets:
        raise ValueError("market is missing for some rows")
    unknown = markets - set(SESSIONS["market"])
    if unknown:
        raise ValueError(f"unknown market {sorted(unknown)}; pit_lag knows {sorted(set(SESSIONS['market']))}")
    return out


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------

def _holiday_package():
    try:
        import holidays
    except ImportError:
        return None
    return holidays


@functools.lru_cache(maxsize=64)
def _base_holidays(market: str, first_year: int, last_year: int) -> tuple:
    days = set()
    if market == "US":
        days.update(USFederalHolidayCalendar().holidays(f"{first_year}-01-01", f"{last_year}-12-31").date)
    package = _holiday_package()
    if package is None:
        if market != "US" and market not in _WARNED:
            _WARNED.add(market)
            warnings.warn(f"pit_lag: the holidays package isn't installed, so {market} public holidays count "
                          "as business days and a deadline on one comes out a day early. "
                          "Run: pip install holidays", stacklevel=2)
        return tuple(sorted(days))
    years = range(first_year, last_year + 1)
    days.update(package.country_holidays(market, years=years).keys())
    if market in _EXCHANGES:
        days.update(package.financial_holidays(_EXCHANGES[market], years=years).keys())
    return tuple(sorted(days))


def _calendar(market: str, frame: pd.DataFrame, extra_holidays: Mapping | None) -> np.ndarray:
    """Holidays for the years the rows can touch, as datetime64[D]."""
    dates = pd.concat([frame["period_end"], frame["filed"], _dates(frame["accepted"])]).dropna()
    first, last = (2000, 2000) if dates.empty else (dates.min().year, dates.max().year)
    days = set(_base_holidays(market, first - 1, last + 2))
    if extra_holidays and market in extra_holidays:
        days.update(pd.to_datetime(list(extra_holidays[market])).date)
    return np.array(sorted(days), dtype="datetime64[D]")


def _on_or_after(days: np.ndarray, holidays: np.ndarray) -> np.ndarray:
    return np.busday_offset(days, 0, roll="forward", holidays=holidays)


def _after(days: np.ndarray, holidays: np.ndarray) -> np.ndarray:
    return np.busday_offset(days, 1, roll="backward", holidays=holidays)


def _pick(starts: pd.Series, days: np.ndarray) -> np.ndarray:
    """For each day, the position of the last row whose start is on or before it (-1 if none)."""
    starts = np.asarray(starts.to_numpy(), dtype="datetime64[D]")
    starts = np.where(np.isnat(starts), _DAWN, starts)
    return np.searchsorted(starts, days, side="right") - 1


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def _available(frame: pd.DataFrame, at: str, extension: bool, extra_holidays: Mapping | None) -> np.ndarray:
    if at not in ("close", "open"):
        raise ValueError('at must be "close" or "open"')
    result = np.full(len(frame), _NAT)
    for market, rows in frame.groupby("market", sort=False).indices.items():
        result[rows] = _available_in(frame.iloc[rows], market, at, extension, extra_holidays)
    return result


def _available_in(frame: pd.DataFrame, market: str, at: str, extension: bool,
                  extra_holidays: Mapping | None) -> np.ndarray:
    holidays = _calendar(market, frame, extra_holidays)
    result = np.full(len(frame), _NAT)

    # 1. Filing time: the same day if it came before the cut-off on a trading day.
    accepted = frame["accepted"]
    sessions = SESSIONS[SESSIONS["market"] == market].sort_values("effective", na_position="first")
    if accepted.dt.tz is not None:
        accepted = accepted.dt.tz_convert(sessions["timezone"].iloc[0]).dt.tz_localize(None)
    has_time = accepted.notna().to_numpy()
    if has_time.any():
        minutes = (accepted.dt.hour * 60 + accepted.dt.minute + accepted.dt.second / 60).to_numpy(
            dtype=float, na_value=np.nan)
        day = _days(accepted.dt.normalize())
        cutoffs = np.array([int(text[:2]) * 60 + int(text[3:]) for text in sessions[at]])
        cutoff = cutoffs[np.clip(_pick(sessions["effective"], day), 0, None)]
        same_day = (minutes > 0) & (minutes < cutoff)
        result[has_time] = np.where(same_day, _on_or_after(day, holidays), _after(day, holidays))[has_time]

    # 2. Filing date with no time: the next trading day.
    filed = _days(frame["filed"])
    has_date = ~has_time & ~np.isnat(filed)
    result[has_date] = _after(filed[has_date], holidays)

    # 3. Neither: the next trading day after the legal deadline.
    rest = ~has_time & ~has_date
    if rest.any():
        result[rest] = _after(_deadlines(frame[rest], market, extension, holidays), holidays)
    return result


def _deadlines(frame: pd.DataFrame, market: str, extension: bool, holidays: np.ndarray) -> np.ndarray:
    rules = RULES[RULES["market"] == market]
    market_classes = set(rules["filer_class"].str.lower()) - {""}
    ends = _days(frame["period_end"])
    result = np.full(len(frame), _NAT)
    for (report, cls), rows in frame.groupby(["report_type", "filer_class"], sort=False).indices.items():
        for_report = rules[rules["report_type"] == report]
        if for_report.empty:
            if not report:
                raise ValueError("report_type is missing for some rows without a filing date")
            have = ", ".join(sorted(set(rules["report_type"])))
            raise ValueError(f"no deadline for {market} {report} reports (rules cover: {have}); "
                             "pass filing dates instead")
        if cls and cls not in market_classes:
            if not market_classes:
                raise ValueError(f"{market} rules have no filer classes; leave filer_class empty")
            known = sorted(set(rules["filer_class"]) - {""})
            raise ValueError(f"unknown {market} filer class {cls!r}; use one of {known} or leave it empty")
        classes = for_report["filer_class"].str.lower()
        if cls in set(classes) - {""}:
            for_report = for_report[classes.isin([cls, ""])]
        # Otherwise the class is unknown, or this report type doesn't distinguish it: every
        # class's deadline is a candidate and the latest one wins.
        latest = None
        for _, timeline in for_report.groupby("filer_class", sort=False):
            due = _due_dates(ends[rows], timeline.sort_values("effective", na_position="first"),
                             extension, holidays)
            latest = due if latest is None else np.maximum(latest, due)
        result[rows] = latest
    return result


def _due_dates(ends: np.ndarray, timeline: pd.DataFrame, extension: bool, holidays: np.ndarray) -> np.ndarray:
    """Each period's deadline under the rule in force at its period end."""
    pick = _pick(timeline["effective"], ends)
    if ((pick < 0) & ~np.isnat(ends)).any():
        first = timeline["effective"].iloc[0].date()
        raise ValueError(f"no {timeline['market'].iloc[0]} {timeline['report_type'].iloc[0]} rule "
                         f"covers period ends before {first}")
    due = np.full(len(ends), _NAT)
    for position, rule in enumerate(timeline.itertuples(index=False)):
        chosen = pick == position
        if not chosen.any():
            continue
        start = ends[chosen]
        if rule.months:
            # Month-end periods end on a month end ("by 31 August" for 30 June); others keep the day.
            index = pd.DatetimeIndex(start)
            months = int(rule.months)
            later = np.where(index.is_month_end, index + pd.offsets.MonthEnd(months),
                             index + pd.DateOffset(months=months))
            start = np.asarray(later, dtype="datetime64[D]")
        deadline = _on_or_after(start + np.timedelta64(int(rule.days), "D"), holidays)
        if extension and rule.extension_days:
            deadline = _on_or_after(deadline + np.timedelta64(int(rule.extension_days), "D"), holidays)
        due[chosen] = deadline
    return due


# ---------------------------------------------------------------------------
# Audit summary
# ---------------------------------------------------------------------------

def _summary(frame: pd.DataFrame, available: np.ndarray, too_early: np.ndarray, days_early: np.ndarray) -> str:
    rows = len(frame)
    if rows == 0:
        return "pit_lag audit: no rows"
    early = int(too_early.sum())
    lines = [f"pit_lag audit: {early:,} of {rows:,} rows ({early / rows:.1%}) were used before the report "
             "was public."]
    if early:
        median = float(np.median(days_early[too_early]))
        most = int(days_early[too_early].max())
        lines.append(f"Those rows were a median {median:g} day{'s' if median != 1 else ''} early, "
                     f"at most {most:,}.")
    has_time = frame["accepted"].notna().to_numpy()
    has_date = ~has_time & frame["filed"].notna().to_numpy()
    undated = np.isnat(available)
    from_deadline = ~has_time & ~has_date & ~undated
    lines.append(f"Available dates came from filing times for {has_time.sum():,} rows, filing dates for "
                 f"{has_date.sum():,} and legal deadlines for {from_deadline.sum():,}.")
    if undated.any():
        lines.append(f"{undated.sum():,} rows had no period end or filing date and could not be checked.")
    table = (pd.DataFrame({"market": frame["market"].to_numpy(), "report_type": frame["report_type"].to_numpy(),
                           "early": too_early})
             .groupby(["market", "report_type"]).agg(rows=("early", "size"), too_early=("early", "sum")))
    table["share"] = (100 * table["too_early"] / table["rows"]).map("{:.1f}%".format)
    for name in ("rows", "too_early"):
        table[name] = table[name].map("{:,}".format)
    lines.append("")
    lines.append(table.reset_index().to_string(index=False))
    return "\n".join(lines)
