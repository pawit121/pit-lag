#!/usr/bin/env python3
"""Korean filing lags and look-ahead leaks, from the English OpenDART API.

The Korean half of the #4 article: how long KOSPI and KOSDAQ companies take to file
their periodic reports, and how often common backtest conventions use one before it
was public. The tables match us_filing_lags.py so the two markets line up.

Setup, once (Windows PowerShell):
    pip install pandas requests holidays
    DART_API_KEY must hold your 40-character key (a Windows user environment variable).

Run:
    python kr_filing_lags.py --check    one 3-month window: coverage, report titles and a
                                        cross-check against the Korean API. Run this first;
                                        it also saves its report to results/kr_check.txt.
    python kr_filing_lags.py            full pull, filings from 2015-01-01 to 2026-09-30

Data: https://engopendart.fss.or.kr/engapi/list.json (Search disclosures). The script asks
for periodic reports only (pblntf_ty=A), for KOSPI and KOSDAQ separately. It uses 3-month
windows, which is the API's limit without a company code, and 100 results per page, with
every version including corrections (last_reprt_at=N). The full pull is about 2,000 calls,
and OpenDART allows about 20,000 a day. Each window is cached in data/raw, so a run that
stops (call limit, network) picks up where it left off.

Outputs, in results/:
    kr_lag_summary.csv           lag by report type and market, share filed late
    kr_lag_by_year.csv           median lag by period year
    kr_lag_cdf.csv               share filed by day N after period end (data for Chart 1)
    kr_leak_table.csv            share of reports each convention uses early (Finding 2)
    kr_leak_table_by_market.csv  the same, split into KOSPI and KOSDAQ
    kr_extras.csv                deadline-day filing, late filing, later corrections
    kr_unmatched_titles.csv      report titles the script couldn't classify
plus data/kr_filings.csv.gz, one row per report, for the notebook.

Definitions to carry into the article's method section:
  * Universe: annual (business), half-year and quarterly reports of KOSPI and KOSDAQ
    companies for periods ending 2015-01-01 to 2025-12-31; the earliest filing per company,
    report type and period. Corrections only flag the original as later corrected.
  * Report type and period come from the report title, e.g. "Quarterly Report (2025.03)";
    the period end is the last day of that month. Check the titles with --check.
  * OpenDART gives a filing date but no time, so a report counts as usable from the next
    business day (conservative). The leak tables also show the share if reports were
    usable on the filing date itself.
  * Legal deadlines: 45 days after period end for quarterly and half-year reports, and 90
    days for annual reports (Capital Markets Act Arts. 160 and 159), moved to the next
    business day. Companies in their first two years of consolidated reporting get 60 days
    for quarterly and half-year reports; that exception isn't modelled.
  * Business days are weekdays minus Korean public holidays (from the holidays package).
    That's close to the KRX calendar, which also closes on 31 December and Labour Day.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ENG_LIST_URL = "https://engopendart.fss.or.kr/engapi/list.json"
KOR_LIST_URL = "https://opendart.fss.or.kr/api/list.json"
HERE = Path(__file__).resolve().parent

MARKETS = {"Y": "KOSPI", "K": "KOSDAQ"}
REPORT_TYPES = ("annual", "half-year", "quarterly")
DEADLINE_DAYS = {"annual": 90, "half-year": 45, "quarterly": 45}
US_LARGE_ACCELERATED = {"annual": 60, "half-year": 40, "quarterly": 40}
FIXED_LAGS = (0, 30, 45, 60, 90)
PAGE_SIZE = 100
CHECK_WINDOW = (date(2025, 4, 1), date(2025, 6, 30))

# Checked in order, first match wins: "semi-annual" has to be tested before "annual".
TITLE_PATTERNS = [
    ("half-year", re.compile(r"semi[-\s]?annual|half[-\s]?year|반기보고서", re.IGNORECASE)),
    ("quarterly", re.compile(r"quarter|분기보고서", re.IGNORECASE)),
    ("annual", re.compile(r"business report|annual report|사업보고서", re.IGNORECASE)),
]
PERIOD_IN_BRACKETS = re.compile(r"\((\d{4})\.(\d{1,2})\)")
PERIOD_ANYWHERE = re.compile(r"(\d{4})\.(\d{1,2})")
CORRECTION_PREFIX = re.compile(r"^\s*(?:\[[^\]]*\]\s*)+")

STOP_STATUSES = {
    "010": "this key isn't registered",
    "011": "this key is suspended",
    "012": "requests from this IP address aren't allowed",
    "020": "the daily call limit is used up; run again tomorrow, finished windows are cached",
    "021": "too many companies in one request",
    "100": "a request parameter was rejected",
    "101": "access was refused",
    "800": "OpenDART is down for maintenance; try again later",
    "901": "the key expired with your account's data retention period; ask opendart@fss.or.kr",
}


# ---------------------------------------------------------------------------
# Calling the API
# ---------------------------------------------------------------------------

class DartError(Exception):
    """OpenDART refused the request in a way that retrying won't fix."""


class DartClient:
    def __init__(self, key: str, pause: float):
        self.key = key
        self.pause = pause
        self.calls = 0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "filing-lags research script"

    def redact(self, text) -> str:
        return str(text).replace(self.key, "<key>")

    def list(self, url: str, params: dict, tries: int = 4) -> dict:
        """One page of disclosure search results. Status 013 (no data) comes back as an empty page."""
        last_error = None
        for attempt in range(tries):
            time.sleep(self.pause)
            try:
                response = self.session.get(url, params={"crtfc_key": self.key, **params}, timeout=60)
                self.calls += 1
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = self.redact(exc)
            else:
                status = str(payload.get("status", ""))
                if status in ("000", "013"):
                    return payload
                if status in STOP_STATUSES:
                    raise DartError(f"OpenDART status {status}: {STOP_STATUSES[status]} "
                                    f"({self.redact(payload.get('message', ''))})")
                last_error = f"OpenDART status {status}: {self.redact(payload.get('message', ''))}"
            wait = 10 * self.pause * 2 ** attempt
            print(f"  {last_error}; retrying in {wait:.0f}s")
            time.sleep(wait)
        raise DartError(f"Giving up after {tries} tries: {last_error}")

    def periodic_reports(self, url: str, market: str, start: date, end: date) -> list[dict]:
        """Every periodic-report filing for one market and window, all pages."""
        rows, page, total_pages = [], 1, 1
        while page <= total_pages:
            payload = self.list(url, {
                "bgn_de": f"{start:%Y%m%d}", "end_de": f"{end:%Y%m%d}", "pblntf_ty": "A",
                "corp_cls": market, "last_reprt_at": "N", "sort": "date", "sort_mth": "asc",
                "page_no": page, "page_count": PAGE_SIZE,
            })
            if payload.get("status") == "013":
                break
            rows += payload.get("list") or []
            total_pages = int(payload.get("total_page") or 1)
            page += 1
        return rows

    def total_count(self, url: str, market: str, start: date, end: date) -> int:
        payload = self.list(url, {
            "bgn_de": f"{start:%Y%m%d}", "end_de": f"{end:%Y%m%d}", "pblntf_ty": "A",
            "corp_cls": market, "last_reprt_at": "N", "page_no": 1, "page_count": 1,
        })
        return 0 if payload.get("status") == "013" else int(payload.get("total_count") or 0)


def fetch_window(client: DartClient, url: str, market: str, start: date, end: date,
                 cache_dir: Path) -> tuple[list[dict], bool]:
    """Rows for one market and window, from the cache when it's there."""
    path = cache_dir / f"kr_{market}_{start:%Y%m%d}_{end:%Y%m%d}.json.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle), True
    rows = client.periodic_reports(url, market, start, end)
    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with gzip.open(partial, "wt", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False)
    partial.replace(path)
    return rows, False


def quarter_windows(first: date, last: date) -> list[tuple[date, date]]:
    """Calendar-quarter windows from first to last, the last one cut short at `last`."""
    windows, start = [], first
    while start <= last:
        quarter_end_month = ((start.month - 1) // 3 + 1) * 3
        end = (pd.Timestamp(start.year, quarter_end_month, 1) + pd.offsets.MonthEnd(0)).date()
        end = min(end, last)
        windows.append((start, end))
        start = end + timedelta(days=1)
    return windows


# ---------------------------------------------------------------------------
# Titles and dates
# ---------------------------------------------------------------------------

def report_type_of(title: str) -> str | None:
    return next((name for name, pattern in TITLE_PATTERNS if pattern.search(title)), None)


def title_pattern(title: str) -> str:
    """A title with its correction prefix removed and its period replaced by YYYY.MM."""
    return PERIOD_ANYWHERE.sub("YYYY.MM", CORRECTION_PREFIX.sub("", title).strip())


def parse_reports(rows: list[dict]) -> tuple[pd.DataFrame, Counter]:
    records, unmatched = [], Counter()
    for item in rows:
        title = str(item.get("report_nm", "")).strip()
        base = CORRECTION_PREFIX.sub("", title).strip()
        kind = report_type_of(base)
        found = PERIOD_IN_BRACKETS.search(base) or PERIOD_ANYWHERE.search(base)
        if kind is None or found is None or not 1 <= int(found.group(2)) <= 12:
            unmatched[title_pattern(title)] += 1
            continue
        records.append({
            "rcept_no": str(item.get("rcept_no", "")),
            "corp_code": str(item.get("corp_code", "")),
            "corp_name": str(item.get("corp_name", "")),
            "stock_code": str(item.get("stock_code", "")),
            "market": MARKETS.get(str(item.get("corp_cls", ""))),
            "report_type": kind,
            "period": pd.Timestamp(int(found.group(1)), int(found.group(2)), 1) + pd.offsets.MonthEnd(0),
            "filed": pd.to_datetime(str(item.get("rcept_dt", "")), format="%Y%m%d", errors="coerce"),
            "title": title,
            "is_correction": base != title,
            "rm": str(item.get("rm", "") or ""),
        })
    columns = ["rcept_no", "corp_code", "corp_name", "stock_code", "market", "report_type",
               "period", "filed", "title", "is_correction", "rm"]
    reports = pd.DataFrame(records, columns=columns)
    reports = reports[reports["market"].notna() & reports["filed"].notna()]
    return reports.drop_duplicates("rcept_no"), unmatched


def korean_holidays(first_year: int, last_year: int) -> np.ndarray:
    try:
        import holidays as holiday_lib
    except ImportError:
        print("Note: the holidays package isn't installed, so only weekends count as non-business "
              "days. Run: pip install holidays")
        return np.array([], dtype="datetime64[D]")
    years = range(first_year, last_year + 1)
    try:
        calendar = holiday_lib.country_holidays("KR", years=years)
    except AttributeError:  # older versions of the package
        calendar = holiday_lib.KR(years=years)
    return np.array(sorted(calendar.keys()), dtype="datetime64[D]")


def to_days(values) -> np.ndarray:
    return np.asarray(values).astype("datetime64[D]")


def roll_forward(days: np.ndarray, holidays: np.ndarray) -> np.ndarray:
    """The first business day on or after each date."""
    return np.busday_offset(days, 0, roll="forward", holidays=holidays)


def next_business_day(days: np.ndarray, holidays: np.ndarray) -> np.ndarray:
    """The first business day strictly after each date."""
    return np.busday_offset(days, 1, roll="backward", holidays=holidays)


# ---------------------------------------------------------------------------
# Building the filing table
# ---------------------------------------------------------------------------

def build_filings(reports: pd.DataFrame, period_from: pd.Timestamp, period_to: pd.Timestamp,
                  holidays: np.ndarray) -> tuple[pd.DataFrame, dict]:
    key = ["corp_code", "report_type", "period"]
    counts = {"periodic-report filings with a recognised title": len(reports)}
    reports = reports[reports["period"].between(period_from, period_to)]
    counts["in the period window"] = len(reports)

    reports = reports.sort_values(["filed", "is_correction", "rcept_no"])
    filings = reports.drop_duplicates(key, keep="first").copy()
    counts["after keeping the earliest per company, report type and period"] = len(filings)
    correction_only = filings["is_correction"]
    counts["dropped: only a correction is on file (original missing)"] = int(correction_only.sum())
    filings = filings[~correction_only].copy()

    last_correction = (reports[reports["is_correction"]].groupby(key)["filed"].max()
                       .rename("last_correction_filed"))
    filings = filings.join(last_correction, on=key)
    filings["corrected_later"] = (filings["last_correction_filed"].ge(filings["filed"]).fillna(False).astype(bool)
                                  | filings["rm"].str.contains("U", regex=False))

    filings["lag_days"] = (filings["filed"] - filings["period"]).dt.days
    filed_before_period = filings["lag_days"] < 0
    counts["dropped: filed before the period ended (bad data)"] = int(filed_before_period.sum())
    filings = filings[~filed_before_period].copy()

    filings["deadline_days"] = filings["report_type"].map(DEADLINE_DAYS).astype(int)
    period_days = to_days(filings["period"])
    filed_days = to_days(filings["filed"])
    deadline = roll_forward(period_days + filings["deadline_days"].to_numpy().astype("timedelta64[D]"), holidays)
    filings["deadline"] = deadline
    filings["late"] = filed_days > deadline
    filings["on_deadline_day"] = filed_days == deadline
    filings["first_usable"] = next_business_day(filed_days, holidays)
    filings["first_usable_same_day"] = roll_forward(filed_days, holidays)
    filings["period_year"] = filings["period"].dt.year
    return filings, counts


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _groups(filings: pd.DataFrame):
    """Yield (report type, market, rows) for each market and for both together."""
    for kind in REPORT_TYPES:
        by_kind = filings[filings["report_type"] == kind]
        for market in list(MARKETS.values()) + ["all"]:
            rows = by_kind if market == "all" else by_kind[by_kind["market"] == market]
            if len(rows):
                yield kind, market, rows


def lag_summary(filings: pd.DataFrame) -> pd.DataFrame:
    records = []
    for kind, market, rows in _groups(filings):
        lags = rows["lag_days"]
        records.append({
            "report_type": kind,
            "market": market,
            "filings": len(rows),
            "deadline_days": DEADLINE_DAYS[kind],
            "median_lag": float(lags.median()),
            "p10_lag": float(lags.quantile(0.10)),
            "p90_lag": float(lags.quantile(0.90)),
            "on_deadline_day_pct": round(100 * float(rows["on_deadline_day"].mean()), 2),
            "late_pct": round(100 * float(rows["late"].mean()), 2),
            "over_365_days_pct": round(100 * float((lags > 365).mean()), 2),
        })
    return pd.DataFrame(records)


def lag_by_year(filings: pd.DataFrame) -> pd.DataFrame:
    table = (filings.groupby(["period_year", "report_type", "market"])["lag_days"].median()
             .unstack(["report_type", "market"]))
    ordered = [(kind, market) for kind in REPORT_TYPES for market in MARKETS.values()
               if (kind, market) in table.columns]
    table = table[ordered]
    table.columns = [f"{kind} | {market}" for kind, market in ordered]
    return table.reset_index()


def lag_cdf(filings: pd.DataFrame, max_days: int) -> pd.DataFrame:
    days = np.arange(max_days + 1)
    curves = {"day": days}
    for kind, market, rows in _groups(filings):
        lags = np.sort(rows["lag_days"].to_numpy())
        curves[f"{kind} | {market}"] = np.round(np.searchsorted(lags, days, side="right") / len(lags), 4)
    return pd.DataFrame(curves)


def conventions(filings: pd.DataFrame, holidays: np.ndarray):
    """Yield (label, date each report's numbers get attached) for each convention."""
    period_days = to_days(filings["period"])

    def attach(day_counts) -> np.ndarray:
        offsets = np.asarray(day_counts, dtype=np.int64).astype("timedelta64[D]")
        return roll_forward(period_days + offsets, holidays)

    for lag in FIXED_LAGS:
        yield ("period end" if lag == 0 else f"period end + {lag} days"), attach(np.full(len(filings), lag))
    yield ("US large-accelerated deadlines (quarterly and half-year 40, annual 60)",
           attach(filings["report_type"].map(US_LARGE_ACCELERATED).to_numpy()))
    yield ("Korean legal deadline (quarterly and half-year 45, annual 90)",
           attach(filings["deadline_days"].to_numpy()))
    yield "actual filing, next business day (baseline)", to_days(filings["first_usable"])


def leak_table(filings: pd.DataFrame, holidays: np.ndarray, by: list[str]) -> pd.DataFrame:
    first_usable = to_days(filings["first_usable"])
    same_day = to_days(filings["first_usable_same_day"])
    keys = filings[by].reset_index(drop=True)
    records = []
    for order, (label, attached) in enumerate(conventions(filings, holidays)):
        days_early = (first_usable - attached).astype("timedelta64[D]").astype(np.int64)
        days_early_same_day = (same_day - attached).astype("timedelta64[D]").astype(np.int64)
        frame = keys.assign(days_early=days_early, days_early_same_day=days_early_same_day)
        for key, rows in frame.groupby(by, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            early = rows["days_early"]
            leaked, unused = early > 0, early < 0
            records.append({
                **dict(zip(by, key)),
                "order": order,
                "convention": label,
                "filings": len(early),
                "leaked_pct": round(100 * float(leaked.mean()), 2),
                "leaked_pct_if_usable_same_day": round(100 * float((rows["days_early_same_day"] > 0).mean()), 2),
                "median_days_early": float(early[leaked].median()) if leaked.any() else 0.0,
                "unused_pct": round(100 * float(unused.mean()), 2),
                "median_days_unused": float((-early[unused]).median()) if unused.any() else 0.0,
            })
    table = pd.DataFrame(records).sort_values(by + ["order"]).drop(columns="order")
    return table.reset_index(drop=True)


def extras(filings: pd.DataFrame, reports: pd.DataFrame, unmatched: Counter) -> pd.DataFrame:
    records = []
    for kind in REPORT_TYPES:
        rows = filings[filings["report_type"] == kind]
        if not len(rows):
            continue
        records += [
            {"report_type": kind, "measure": "filed on the deadline day (%)",
             "value": round(100 * float(rows["on_deadline_day"].mean()), 2)},
            {"report_type": kind, "measure": "filed after the deadline (%)",
             "value": round(100 * float(rows["late"].mean()), 2)},
            {"report_type": kind, "measure": "corrected later: a correction filing or rm flag U (%)",
             "value": round(100 * float(rows["corrected_later"].mean()), 2)},
            {"report_type": kind, "measure": "correction filings (count)",
             "value": int(((reports["report_type"] == kind) & reports["is_correction"]).sum())},
        ]
    records.append({"report_type": "all", "measure": "filings with an unrecognised title (count)",
                    "value": int(sum(unmatched.values()))})
    return pd.DataFrame(records)


def print_summary(counts: dict, summary: pd.DataFrame, leaks: pd.DataFrame) -> None:
    print("\nReports")
    for label, value in counts.items():
        print(f"  {label}: {value:,}")
    print("\nMedian days from period end to filing (share filed on the deadline day, share late)")
    for row in summary.itertuples():
        print(f"  {row.report_type:<10} {row.market:<7} {row.median_lag:>5.0f}   "
              f"({row.on_deadline_day_pct:.1f}% on the deadline day, {row.late_pct:.1f}% late, "
              f"{row.filings:,} reports)")
    print("\nShare of reports a backtest uses before they were public (if usable the same day)")
    for row in leaks.itertuples():
        print(f"  {row.report_type:<10} {row.convention:<72} {row.leaked_pct:>6.1f}%  "
              f"({row.leaked_pct_if_usable_same_day:.1f}%)")


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def run_check(client: DartClient, eng_url: str, kor_url: str, cache_dir: Path, out_dir: Path) -> None:
    lines = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    start, end = CHECK_WINDOW
    say(f"Coverage check: periodic reports filed {start} to {end}")
    say()
    rows = []
    for market, name in MARKETS.items():
        got, cached = fetch_window(client, eng_url, market, start, end, cache_dir)
        say(f"  {name:<7} {len(got):>6,} filings {'(cached)' if cached else ''}")
        rows += got

    titles = Counter(title_pattern(str(item.get("report_nm", ""))) for item in rows)
    say()
    say("Report titles (correction prefixes removed), most common first:")
    for title, count in titles.most_common(15):
        say(f"  {count:>6,}  {title:<50} -> {report_type_of(title) or 'NOT RECOGNISED'}")

    reports, unmatched = parse_reports(rows)
    quarterly = reports[(reports["report_type"] == "quarterly") & (reports["period"] == "2025-03-31")
                        & ~reports["is_correction"]]
    say()
    say("Companies with a quarterly report for 2025.03 (compare with the number of listed companies):")
    for name in MARKETS.values():
        say(f"  {name:<7} {quarterly.loc[quarterly['market'] == name, 'corp_code'].nunique():>6,}")

    say()
    say("Cross-check against the Korean API with the same query:")
    try:
        for market, name in MARKETS.items():
            english = client.total_count(eng_url, market, start, end)
            korean = client.total_count(kor_url, market, start, end)
            share = f"{100 * english / korean:.1f}%" if korean else "n/a"
            say(f"  {name:<7} English {english:>6,}   Korean {korean:>6,}   English/Korean {share}")
    except DartError as exc:
        say(f"  Not possible: {exc}.")
        say("  Compare the company counts above with the number of listed companies on KRX instead.")
    say()
    say(f"{client.calls} API calls. If every main title is recognised and the counts look complete, "
        "run the full pull: python kr_filing_lags.py")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kr_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Korean filing lags and look-ahead leaks (English OpenDART).")
    parser.add_argument("--check", action="store_true", help="coverage and title check on one window, then stop")
    parser.add_argument("--from-date", default="2015-01-01", help="first filing date to pull (default 2015-01-01)")
    parser.add_argument("--to-date", default="2026-09-30", help="last filing date to pull (default 2026-09-30)")
    parser.add_argument("--period-from", default="2015-01-01", help="earliest period end (default 2015-01-01)")
    parser.add_argument("--period-to", default="2025-12-31", help="latest period end (default 2025-12-31)")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--cdf-days", type=int, default=365, help="last day in the CDF table (default 365)")
    parser.add_argument("--pause", type=float, default=0.3, help=argparse.SUPPRESS)
    parser.add_argument("--eng-url", default=ENG_LIST_URL, help=argparse.SUPPRESS)
    parser.add_argument("--kor-url", default=KOR_LIST_URL, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # Korean titles on consoles that can't show them

    key = os.environ.get("DART_API_KEY", "").strip()
    if len(key) != 40:
        sys.exit("Set DART_API_KEY to your 40-character OpenDART key first (Windows: Edit environment "
                 "variables for your account), then open a new terminal.")
    client = DartClient(key, args.pause)
    cache_dir = args.data_dir / "raw"

    try:
        if args.check:
            run_check(client, args.eng_url, args.kor_url, cache_dir, args.out_dir)
            return
        rows = []
        first, last = date.fromisoformat(args.from_date), date.fromisoformat(args.to_date)
        for start, end in quarter_windows(first, last):
            for market, name in MARKETS.items():
                got, cached = fetch_window(client, args.eng_url, market, start, end, cache_dir)
                print(f"{start} to {end}  {name:<7} {len(got):>6,} filings{'  (cached)' if cached else ''}")
                rows += got
    except DartError as exc:
        sys.exit(f"\nStopped: {exc}")
    print(f"{client.calls} API calls this run")

    reports, unmatched = parse_reports(rows)
    period_from, period_to = pd.Timestamp(args.period_from), pd.Timestamp(args.period_to)
    holidays = korean_holidays(period_from.year - 1, period_to.year + 3)
    filings, counts = build_filings(reports, period_from, period_to, holidays)
    if filings.empty:
        sys.exit("No reports in that window. Check the dates, and run --check to see the titles.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = lag_summary(filings)
    leaks = leak_table(filings, holidays, ["report_type"])
    tables = {
        "kr_lag_summary.csv": summary,
        "kr_lag_by_year.csv": lag_by_year(filings),
        "kr_lag_cdf.csv": lag_cdf(filings, args.cdf_days),
        "kr_leak_table.csv": leaks,
        "kr_leak_table_by_market.csv": leak_table(filings, holidays, ["report_type", "market"]),
        "kr_extras.csv": extras(filings, reports, unmatched),
        "kr_unmatched_titles.csv": pd.DataFrame(unmatched.most_common(), columns=["title", "filings"]),
    }
    for name, table in tables.items():
        table.to_csv(args.out_dir / name, index=False)

    columns = ["corp_code", "stock_code", "corp_name", "market", "report_type", "period", "filed",
               "first_usable", "first_usable_same_day", "lag_days", "deadline_days", "deadline", "late",
               "on_deadline_day", "corrected_later", "rcept_no", "title"]
    filings_path = args.data_dir / "kr_filings.csv.gz"
    filings[columns].to_csv(filings_path, index=False, compression="gzip")

    print_summary(counts, summary, leaks)
    if unmatched:
        print(f"\n{sum(unmatched.values()):,} filings had titles the script didn't recognise; "
              "see kr_unmatched_titles.csv")
    print(f"\nWrote {len(tables)} tables to {args.out_dir}")
    print(f"and one row per report to {filings_path}")


if __name__ == "__main__":
    main()
