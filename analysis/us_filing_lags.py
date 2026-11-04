#!/usr/bin/env python3
"""US filing lags and look-ahead leaks, from the SEC Financial Statement Data Sets.

Starter script for the #4 article: how long US companies take to file 10-Ks and
10-Qs, and how often common backtest conventions use a filing before it was public.

Setup, once (Windows PowerShell):
    pip install pandas requests
    setx SEC_USER_AGENT "Your Name your.email@example.com"
    Then open a new terminal. The SEC asks every script to identify itself this way.

Run:
    python us_filing_lags.py --start 2025q1 --end 2025q2    quick test on two quarters
    python us_filing_lags.py                                full run, 2015q1 to 2026q3

Data: sub.txt from each quarterly zip at
https://www.sec.gov/files/dera/data/financial-statement-data-sets/<yyyy>q<n>.zip
Field definitions: https://www.sec.gov/files/financial-statement-data-sets.pdf
The script first tries to pull only sub.txt out of each zip with HTTP range
requests, a few MB per quarter. If the server won't serve byte ranges, it downloads
the whole zip (60-120 MB each, about 4.5 GB for the full run), extracts sub.txt and
deletes the zip unless you pass --keep-zips. Extracted files are cached in data/raw,
so reruns skip the downloads.

Outputs, in results/:
    us_lag_summary.csv          lag by form and filer status, and the share filed late
    us_lag_by_year.csv          median lag by period year
    us_lag_cdf.csv              share filed by day N after period end (data for Chart 1)
    us_leak_table.csv           share of filings each convention uses early (Finding 2)
    us_leak_table_by_filer.csv  the same, split by filer status
    us_extras.csv               after-close acceptance and later amendments
plus data/us_filings.csv.gz, one row per filing, for the notebook.

Definitions to carry into the article's method section:
  * Universe: original 10-K and 10-Q filings whose period ends between 2015-01-01
    and 2025-12-31, keeping the earliest per company, form and period. Amendments
    (10-K/A, 10-Q/A) are only used to flag originals that were later amended.
  * period: the SEC rounds the balance-sheet date to the nearest month-end, so
    52/53-week fiscal years are off by a few days.
  * First usable date: the acceptance date if EDGAR accepted the filing before 16:00
    New York time on a business day, otherwise the next business day. Business days
    are weekdays minus US federal holidays, close to but not the same as the NYSE
    calendar.
  * Legal deadlines (unchanged 2015-2025), moved to the next business day:
        10-K: 60 days large accelerated, 75 accelerated, 90 non-accelerated
        10-Q: 40 days large accelerated and accelerated, 45 non-accelerated
    Form 12b-25 (NT) adds 15 days for a 10-K and 5 for a 10-Q. Filers with no status
    in the data get the non-accelerated deadlines.
  * A convention attaches a filing's numbers on period end + k days, moved to the
    next business day. It leaks the filing if that date is before the first usable
    date, and leaves it unused for a while if it is after.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar

URL_TEMPLATE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
HERE = Path(__file__).resolve().parent

ORIGINAL_FORMS = ("10-K", "10-Q")
AMENDMENT_FORMS = ("10-K/A", "10-Q/A")

FILER_GROUPS = {
    "1-LAF": "large accelerated",
    "2-ACC": "accelerated",
    "3-SRA": "accelerated",  # accelerated filer that is also a smaller reporting company
    "4-NON": "non-accelerated",
    "5-SML": "non-accelerated",  # smaller reporting company that is not accelerated
}
NO_STATUS = "no status"
GROUP_ORDER = ["large accelerated", "accelerated", "non-accelerated", NO_STATUS]

DEADLINE_DAYS = {
    "10-K": {"large accelerated": 60, "accelerated": 75, "non-accelerated": 90, NO_STATUS: 90},
    "10-Q": {"large accelerated": 40, "accelerated": 40, "non-accelerated": 45, NO_STATUS: 45},
}
NT_EXTENSION_DAYS = {"10-K": 15, "10-Q": 5}
US_LARGE_ACCELERATED = {"10-K": 60, "10-Q": 40}
FIXED_LAGS = (0, 30, 45, 60, 90)

CLOSE_MINUTES = 16 * 60  # 16:00 New York time
EDGAR_CUTOFF_MINUTES = 17 * 60 + 30  # accepted after 17:30 is dated the next business day
REQUEST_PAUSE = 0.15  # seconds between requests; the SEC allows up to 10 per second

SUB_COLUMNS = ["adsh", "cik", "name", "afs", "form", "period", "fy", "fp", "filed", "accepted"]


# ---------------------------------------------------------------------------
# Downloading sub.txt
# ---------------------------------------------------------------------------

class NoRangeSupport(Exception):
    """The server answered a byte-range request with something other than 206."""


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "identity"})
    return session


def http_get(session: requests.Session, url: str, headers: dict | None = None,
             stream: bool = False, tries: int = 4) -> requests.Response:
    """GET with a polite pause, retrying rate limits and server errors."""
    last_error = None
    for attempt in range(tries):
        time.sleep(REQUEST_PAUSE)
        try:
            response = session.get(url, headers=headers or {}, stream=stream, timeout=120)
        except requests.RequestException as exc:
            last_error = exc
        else:
            if response.status_code not in (403, 429, 500, 502, 503, 504):
                return response
            last_error = f"HTTP {response.status_code}"
            response.close()
            if response.status_code == 403 and attempt == 0:
                print("  403 from the server. If this repeats, check SEC_USER_AGENT is 'Name email'.")
        wait = 10 * 2 ** attempt
        print(f"  {last_error}; retrying in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"Giving up on {url}: {last_error}")


class HttpRangeFile(io.RawIOBase):
    """A seekable, read-only remote file that fetches bytes with HTTP range requests."""

    def __init__(self, session: requests.Session, url: str, size: int):
        super().__init__()
        self.session = session
        self.url = url
        self.size = size
        self.position = 0
        self.request_count = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.size:
            return 0
        end = min(self.position + len(buffer), self.size) - 1
        response = http_get(self.session, self.url, headers={"Range": f"bytes={self.position}-{end}"})
        if response.status_code != 206:
            response.close()
            raise NoRangeSupport(f"HTTP {response.status_code} to a range request")
        data = response.content[: len(buffer)]
        buffer[: len(data)] = data
        self.position += len(data)
        self.request_count += 1
        return len(data)


def read_member_by_range(session: requests.Session, url: str, member: str) -> tuple[bytes, int]:
    """Read one file out of a remote zip without downloading the rest of it."""
    probe = http_get(session, url, headers={"Range": "bytes=0-0"}, stream=True)
    try:
        if probe.status_code == 404:
            raise FileNotFoundError(url)
        if probe.status_code != 206 or "Content-Range" not in probe.headers:
            raise NoRangeSupport(f"HTTP {probe.status_code} to a range request")
        size = int(probe.headers["Content-Range"].rsplit("/", 1)[1])
    finally:
        probe.close()
    remote = HttpRangeFile(session, url, size)
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=256 * 1024)) as archive:
        data = archive.read(member)
    return data, remote.request_count + 1


def read_member_by_download(session: requests.Session, url: str, member: str,
                            zip_path: Path, keep_zip: bool) -> bytes:
    """Download the whole zip (unless it is already on disk) and read one file from it."""
    if not zip_path.exists():
        response = http_get(session, url, stream=True)
        if response.status_code == 404:
            response.close()
            raise FileNotFoundError(url)
        response.raise_for_status()
        partial = zip_path.with_suffix(".part")
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
        partial.replace(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        data = archive.read(member)
    if not keep_zip:
        zip_path.unlink()
    return data


def fetch_sub_files(session: requests.Session, quarters: list[str], raw_dir: Path,
                    url_template: str, keep_zips: bool) -> list[Path]:
    """Return local copies of sub.txt for each quarter, fetching the missing ones."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    use_range = not keep_zips
    for quarter in quarters:
        target = raw_dir / f"{quarter}_sub.txt"
        if target.exists():
            print(f"{quarter}: cached")
            paths.append(target)
            continue
        url = url_template.format(quarter=quarter)
        data, how = None, ""
        try:
            if use_range:
                try:
                    data, calls = read_member_by_range(session, url, "sub.txt")
                    how = f"{calls} range requests"
                except (NoRangeSupport, zipfile.BadZipFile) as exc:
                    print(f"{quarter}: range requests didn't work ({exc}); downloading whole zips instead")
                    use_range = False
            if data is None:
                data = read_member_by_download(session, url, "sub.txt", raw_dir / f"{quarter}.zip", keep_zips)
                how = "full download"
        except FileNotFoundError:
            print(f"{quarter}: not published yet (404), skipped")
            continue
        partial = target.with_suffix(".part")
        partial.write_bytes(data)
        partial.replace(target)
        print(f"{quarter}: sub.txt {len(data) / 1e6:.1f} MB via {how}")
        paths.append(target)
    return paths


def quarter_range(start: str, end: str) -> list[str]:
    def parse(text: str) -> tuple[int, int]:
        year, quarter = text.lower().strip().split("q")
        if quarter not in {"1", "2", "3", "4"}:
            raise ValueError(f"not a quarter: {text}")
        return int(year), int(quarter)

    (year, quarter), last = parse(start), parse(end)
    quarters = []
    while (year, quarter) <= last:
        quarters.append(f"{year}q{quarter}")
        year, quarter = (year, quarter + 1) if quarter < 4 else (year + 1, 1)
    return quarters


def load_sub_files(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(
            path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
            encoding="utf-8", encoding_errors="replace",
            usecols=lambda column: column in SUB_COLUMNS,
        )
        frame = frame.reindex(columns=SUB_COLUMNS).fillna("")
        frame["source_quarter"] = path.name.split("_")[0]
        frames.append(frame)
    if not frames:
        sys.exit("No sub.txt files to read.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def business_holidays(first_year: int, last_year: int) -> np.ndarray:
    holidays = USFederalHolidayCalendar().holidays(start=f"{first_year}-01-01", end=f"{last_year}-12-31")
    return np.asarray(holidays).astype("datetime64[D]")


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

def build_filings(sub: pd.DataFrame, period_from: pd.Timestamp, period_to: pd.Timestamp,
                  holidays: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    period = pd.to_datetime(sub["period"].str.strip(), format="%Y%m%d", errors="coerce")
    base = pd.DataFrame({
        "adsh": sub["adsh"].str.strip(),
        "cik": sub["cik"].str.strip(),
        "name": sub["name"].str.strip(),
        "form": sub["form"].str.strip().str.upper(),
        "afs": sub["afs"].str.strip(),
        "fy": sub["fy"].str.strip(),
        "fp": sub["fp"].str.strip(),
        "period": period,
        "filed": pd.to_datetime(sub["filed"].str.strip(), format="%Y%m%d", errors="coerce"),
        "accepted": pd.to_datetime(sub["accepted"].str.strip().str.slice(0, 19),
                                   format="%Y-%m-%d %H:%M:%S", errors="coerce"),
        "source_quarter": sub["source_quarter"],
        "in_window": period.between(period_from, period_to),
    })
    base = base.drop_duplicates("adsh")
    counts = {"submissions read": len(base)}

    amendments = base[base["form"].isin(AMENDMENT_FORMS) & base["in_window"]]
    filings = base[base["form"].isin(ORIGINAL_FORMS) & base["in_window"] & base["filed"].notna()]
    counts["original 10-K and 10-Q in the period window"] = len(filings)
    filings = (filings.sort_values(["filed", "accepted"])
               .drop_duplicates(["cik", "form", "period"], keep="first")
               .copy())
    counts["after keeping the earliest per company, form and period"] = len(filings)

    # Later amendments: a 10-K/A or 10-Q/A for the same company and period, filed on or after
    # the original. The SEC's prevrpt flag isn't used: in the 2025 data it marks 1 original
    # in 11,021 although 96 of them have an amendment in the same two quarters.
    last_amendment = (amendments.assign(form=amendments["form"].str.replace("/A", "", regex=False))
                      .groupby(["cik", "form", "period"])["filed"].max()
                      .rename("last_amendment_filed"))
    filings = filings.join(last_amendment, on=["cik", "form", "period"])
    filings["amended_later"] = filings["last_amendment_filed"].ge(filings["filed"]).fillna(False).astype(bool)

    filings["lag_days"] = (filings["filed"] - filings["period"]).dt.days
    filed_before_period = filings["lag_days"] < 0
    counts["dropped: filed before the period ended (bad data)"] = int(filed_before_period.sum())
    filings = filings[~filed_before_period].copy()

    # Legal deadline for each filing's status, then the 12b-25 extension.
    filings["filer_group"] = filings["afs"].map(FILER_GROUPS).fillna(NO_STATUS)
    filings["deadline_days"] = [DEADLINE_DAYS[form][group]
                                for form, group in zip(filings["form"], filings["filer_group"])]
    period_days = to_days(filings["period"])
    filed_days = to_days(filings["filed"])
    deadline = roll_forward(period_days + filings["deadline_days"].to_numpy().astype("timedelta64[D]"), holidays)
    extension = filings["form"].map(NT_EXTENSION_DAYS).to_numpy().astype("timedelta64[D]")
    deadline_nt = roll_forward(deadline + extension, holidays)
    filings["deadline"] = deadline
    filings["late"] = filed_days > deadline
    filings["late_after_nt"] = filed_days > deadline_nt

    # First day a daily-close backtest could have used the filing.
    accepted = filings["accepted"]
    has_time = accepted.notna().to_numpy()
    minutes = (accepted.dt.hour * 60 + accepted.dt.minute).fillna(-1).to_numpy()
    accept_day = to_days(accepted.fillna(filings["filed"]))
    usable_same_day = has_time & (minutes < CLOSE_MINUTES) & np.is_busday(accept_day, holidays=holidays)
    filings["first_usable"] = np.where(usable_same_day, accept_day, next_business_day(accept_day, holidays))
    filings["after_close"] = has_time & (minutes >= CLOSE_MINUTES)
    filings["after_edgar_cutoff"] = has_time & (minutes >= EDGAR_CUTOFF_MINUTES)
    filings["period_year"] = filings["period"].dt.year
    return filings, amendments, counts


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _groups(filings: pd.DataFrame):
    """Yield (form, filer group, rows) for each status and for all filers together."""
    for form, by_form in filings.groupby("form"):
        for group in GROUP_ORDER + ["all"]:
            rows = by_form if group == "all" else by_form[by_form["filer_group"] == group]
            if len(rows):
                yield form, group, rows


def lag_summary(filings: pd.DataFrame) -> pd.DataFrame:
    records = []
    for form, group, rows in _groups(filings):
        lags = rows["lag_days"]
        deadlines = rows["deadline_days"].unique()
        records.append({
            "form": form,
            "filer_group": group,
            "filings": len(rows),
            "deadline_days": int(deadlines[0]) if len(deadlines) == 1 else None,
            "median_lag": float(lags.median()),
            "p10_lag": float(lags.quantile(0.10)),
            "p90_lag": float(lags.quantile(0.90)),
            "late_pct": round(100 * float(rows["late"].mean()), 2),
            "late_after_nt_pct": round(100 * float(rows["late_after_nt"].mean()), 2),
            "over_365_days_pct": round(100 * float((lags > 365).mean()), 2),
        })
    return pd.DataFrame(records)


def lag_by_year(filings: pd.DataFrame) -> pd.DataFrame:
    table = (filings.groupby(["period_year", "form", "filer_group"])["lag_days"].median()
             .unstack(["form", "filer_group"]))
    ordered = [(form, group) for form in ORIGINAL_FORMS for group in GROUP_ORDER if (form, group) in table.columns]
    table = table[ordered]
    table.columns = [f"{form} | {group}" for form, group in ordered]
    return table.reset_index()


def lag_cdf(filings: pd.DataFrame, max_days: int) -> pd.DataFrame:
    days = np.arange(max_days + 1)
    curves = {"day": days}
    for form, group, rows in _groups(filings):
        lags = np.sort(rows["lag_days"].to_numpy())
        curves[f"{form} | {group}"] = np.round(np.searchsorted(lags, days, side="right") / len(lags), 4)
    return pd.DataFrame(curves)


def conventions(filings: pd.DataFrame, holidays: np.ndarray):
    """Yield (label, date each filing's numbers get attached) for each convention."""
    period_days = to_days(filings["period"])

    def attach(day_counts) -> np.ndarray:
        offsets = np.asarray(day_counts, dtype=np.int64).astype("timedelta64[D]")
        return roll_forward(period_days + offsets, holidays)

    for lag in FIXED_LAGS:
        yield ("period end" if lag == 0 else f"period end + {lag} days"), attach(np.full(len(filings), lag))
    yield ("US large-accelerated deadlines for all (10-Q 40, 10-K 60)",
           attach(filings["form"].map(US_LARGE_ACCELERATED).to_numpy()))
    yield "legal deadline for the filer's status", attach(filings["deadline_days"].to_numpy())
    yield "actual filing, first usable day (baseline)", to_days(filings["first_usable"])


def leak_table(filings: pd.DataFrame, holidays: np.ndarray, by: list[str]) -> pd.DataFrame:
    first_usable = to_days(filings["first_usable"])
    keys = filings[by].reset_index(drop=True)
    records = []
    for order, (label, attached) in enumerate(conventions(filings, holidays)):
        days_early = (first_usable - attached).astype("timedelta64[D]").astype(np.int64)
        for key, rows in keys.assign(days_early=days_early).groupby(by, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            early = rows["days_early"]
            leaked, unused = early > 0, early < 0
            records.append({
                **dict(zip(by, key)),
                "order": order,
                "convention": label,
                "filings": len(early),
                "leaked_pct": round(100 * float(leaked.mean()), 2),
                "median_days_early": float(early[leaked].median()) if leaked.any() else 0.0,
                "unused_pct": round(100 * float(unused.mean()), 2),
                "median_days_unused": float((-early[unused]).median()) if unused.any() else 0.0,
            })
    table = pd.DataFrame(records).sort_values(by + ["order"]).drop(columns="order")
    return table.reset_index(drop=True)


def extras(filings: pd.DataFrame, amendments: pd.DataFrame) -> pd.DataFrame:
    records = []
    for form, rows in filings.groupby("form"):
        records += [
            {"form": form, "measure": "accepted at or after 16:00 New York time (%)",
             "value": round(100 * float(rows["after_close"].mean()), 2)},
            {"form": form, "measure": "accepted after 17:30, so EDGAR dates it the next business day (%)",
             "value": round(100 * float(rows["after_edgar_cutoff"].mean()), 2)},
            {"form": form, "measure": f"later amended: a {form}/A for the same company and period (%)",
             "value": round(100 * float(rows["amended_later"].mean()), 2)},
            {"form": form, "measure": f"{form}/A amendments for periods in the window (count)",
             "value": int((amendments["form"] == f"{form}/A").sum())},
        ]
    return pd.DataFrame(records)


def print_summary(counts: dict, summary: pd.DataFrame, leaks: pd.DataFrame) -> None:
    print("\nFilings")
    for label, value in counts.items():
        print(f"  {label}: {value:,}")
    print("\nMedian days from period end to filing (share filed after the legal deadline)")
    for row in summary.itertuples():
        print(f"  {row.form}  {row.filer_group:<18} {row.median_lag:>5.0f}   "
              f"({row.late_pct:.1f}% late, {row.filings:,} filings)")
    print("\nShare of filings a backtest uses before they were public")
    for row in leaks.itertuples():
        days = row.median_days_early
        detail = f"median {days:.0f} day{'' if days == 1 else 's'} early" if row.leaked_pct > 0 else "none early"
        print(f"  {row.form}  {row.convention:<58} {row.leaked_pct:>6.1f}%   {detail}")


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="US filing lags and look-ahead leaks (SEC data sets).")
    parser.add_argument("--start", default="2015q1", help="first quarterly data set (default 2015q1)")
    parser.add_argument("--end", default="2026q3", help="last quarterly data set (default 2026q3)")
    parser.add_argument("--period-from", default="2015-01-01", help="earliest period end (default 2015-01-01)")
    parser.add_argument("--period-to", default="2025-12-31", help="latest period end (default 2025-12-31)")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--keep-zips", action="store_true",
                        help="download and keep the whole zips instead of using range requests")
    parser.add_argument("--cdf-days", type=int, default=365, help="last day in the CDF table (default 365)")
    parser.add_argument("--url-template", default=URL_TEMPLATE, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in user_agent:
        sys.exit('Set SEC_USER_AGENT first. In PowerShell: setx SEC_USER_AGENT "Your Name you@example.com", '
                 "then open a new terminal.")

    session = make_session(user_agent)
    paths = fetch_sub_files(session, quarter_range(args.start, args.end), args.data_dir / "raw",
                            args.url_template, args.keep_zips)
    sub = load_sub_files(paths)

    period_from, period_to = pd.Timestamp(args.period_from), pd.Timestamp(args.period_to)
    holidays = business_holidays(period_from.year - 1, period_to.year + 3)
    filings, amendments, counts = build_filings(sub, period_from, period_to, holidays)
    if filings.empty:
        sys.exit("No 10-K or 10-Q filings in that window. Check --start/--end and the period dates.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = lag_summary(filings)
    leaks = leak_table(filings, holidays, ["form"])
    tables = {
        "us_lag_summary.csv": summary,
        "us_lag_by_year.csv": lag_by_year(filings),
        "us_lag_cdf.csv": lag_cdf(filings, args.cdf_days),
        "us_leak_table.csv": leaks,
        "us_leak_table_by_filer.csv": leak_table(filings, holidays, ["form", "filer_group"]),
        "us_extras.csv": extras(filings, amendments),
    }
    for name, table in tables.items():
        table.to_csv(args.out_dir / name, index=False)

    columns = ["cik", "name", "form", "filer_group", "fy", "fp", "period", "filed", "accepted",
               "first_usable", "lag_days", "deadline_days", "deadline", "late", "late_after_nt",
               "after_close", "amended_later", "adsh", "source_quarter"]
    filings_path = args.data_dir / "us_filings.csv.gz"
    filings[columns].to_csv(filings_path, index=False, compression="gzip")

    print_summary(counts, summary, leaks)
    print(f"\nWrote {len(tables)} tables to {args.out_dir}")
    print(f"and one row per filing to {filings_path}")


if __name__ == "__main__":
    main()
