"""End-to-end tests for us_filing_lags.py against synthetic SEC-style zips on a local server.

Usage, from the filing-lags folder: python tests/run_tests.py us_filing_lags.py
"""
import gzip
import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

SCRIPT = Path(sys.argv[1]).resolve()
COLUMNS = ("adsh cik name sic countryba stprba cityba zipba bas1 bas2 baph countryma stprma cityma zipma "
           "mas1 mas2 countryinc stprinc ein former changed afs wksi fye form period fy fp filed accepted "
           "prevrpt detail instance nciks aciks").split()


def row(adsh, cik, name, afs, form, period, filed, accepted, fy="", fp="", prevrpt="0"):
    values = dict.fromkeys(COLUMNS, "")
    values.update(adsh=adsh, cik=cik, name=name, afs=afs, form=form, period=period, filed=filed,
                  accepted=accepted, fy=fy, fp=fp, prevrpt=prevrpt, sic="1000", countryba="US",
                  nciks="1", instance="x.htm", detail="0", wksi="0", fye="1231")
    return "\t".join(values[c] for c in COLUMNS)


QUARTERS = {
    "2025q2": [
        row("0000000100-25-000001", "100", "Alpha Corp", "1-LAF", "10-Q", "20250331", "20250501", "2025-05-01 16:05:00.0", "2025", "Q1"),
        row("0000000100-25-000004", "100", "Alpha Corp", "1-LAF", "10-Q", "20250331", "20250510", "2025-05-10 10:00:00.0", "2025", "Q1"),
        row("0000000500-15-000005", "500", "Echo Co", "4-NON", "10-K", "20141231", "20150330", "2015-03-30 12:00:00.0", "2014", "FY"),
        row("0000000700-25-000007", "700", "Golf plc", "", "20-F", "20241231", "20250430", "2025-04-30 08:00:00.0", "2024", "FY"),
        row("0000000800-25-000008", "800", "Hotel Inc", "", "10-Q", "20250630", "20250814", "2025-08-14 17:45:00.0", "2025", "Q2"),
    ],
    "2025q3": [
        row("0000000200-25-000002", "200", "Beta Inc", "4-NON", "10-K", "20241231", "20250415", "2025-04-15 09:00:00.0", "2024", "FY"),
        row("0000000200-25-000003", "200", "Beta Inc", "4-NON", "10-K/A", "20241231", "20250601", "2025-06-01 10:00:00.0", "2024", "FY"),
        row("0000000600-25-000006", "600", "Foxtrot Ltd", "2-ACC", "10-Q", "20250630", "20250601", "2025-06-01 10:00:00.0", "2025", "Q2"),
        row("0000000300-25-000009", "300", "Gamma LLC", "2-ACC", "10-Q", "20250531", "20250704", "2025-07-04 10:00:00.0", "2025", "Q2"),
        row("0000000900-25-000010", "900", 'Kilo "K" Corp', "3-SRA", "10-K", "20250630", "20250912", "2025-09-12 15:30:00.0", "2025", "FY"),
    ],
}


def build_zips(root: Path) -> dict:
    sizes = {}
    for quarter, rows in QUARTERS.items():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("sub.txt", "\t".join(COLUMNS) + "\n" + "\n".join(rows) + "\n")
            archive.writestr("num.txt", os.urandom(4_000_000))  # big and incompressible, like the real num.txt
            archive.writestr("pre.txt", "adsh\treport\n")
            archive.writestr("tag.txt", "tag\tversion\n")
            archive.writestr("readme.htm", "<html></html>")
        (root / f"{quarter}.zip").write_bytes(buffer.getvalue())
        sizes[quarter] = len(buffer.getvalue())
    return sizes


def make_handler(root: Path, ranges: bool, log: list):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = root / self.path.lstrip("/")
            if not path.is_file():
                self.send_error(404)
                log.append((self.path, None, 0))
                return
            data = path.read_bytes()
            requested = self.headers.get("Range")
            if requested and ranges:
                start, end = requested.split("=", 1)[1].split("-")
                if start == "":
                    start, end = len(data) - int(end), len(data) - 1
                else:
                    start, end = int(start), (int(end) if end else len(data) - 1)
                end = min(end, len(data) - 1)
                body = data[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            else:
                body = data
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/zip")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            log.append((self.path, requested, len(body)))

        def log_message(self, *args):
            pass

    return Handler


def serve(root: Path, ranges: bool):
    log = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root, ranges, log))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, log


def run(port: int, workdir: Path, *extra) -> str:
    env = dict(os.environ, SEC_USER_AGENT="Test Runner test@example.org",
               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    command = [sys.executable, str(SCRIPT), "--start", "2025q2", "--end", "2025q4",
               "--data-dir", str(workdir / "data"), "--out-dir", str(workdir / "results"),
               "--url-template", f"http://127.0.0.1:{port}/{{quarter}}.zip", *extra]
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=300)
    if result.returncode != 0:
        print(result.stdout, result.stderr)
        raise SystemExit(f"script failed with exit code {result.returncode}")
    return result.stdout


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


def validate(results: Path, data: Path):
    filings = pd.read_csv(data / "us_filings.csv.gz", dtype={"cik": str})
    check(sorted(filings["cik"]) == ["100", "200", "300", "800", "900"], "kept the 5 expected filings")
    by_cik = filings.set_index("cik")
    expected = {
        "100": dict(lag_days=31, first_usable="2025-05-02", deadline="2025-05-12", late=False, filer_group="large accelerated"),
        "200": dict(lag_days=105, first_usable="2025-04-15", deadline="2025-03-31", late=True, late_after_nt=False, filer_group="non-accelerated"),
        "300": dict(lag_days=34, first_usable="2025-07-07", deadline="2025-07-10", late=False, filer_group="accelerated"),
        "800": dict(lag_days=45, first_usable="2025-08-15", deadline="2025-08-14", late=False, filer_group="no status"),
        "900": dict(lag_days=74, first_usable="2025-09-12", deadline="2025-09-15", late=False, filer_group="accelerated"),
    }
    for cik, fields in expected.items():
        for field, value in fields.items():
            actual = by_cik.loc[cik, field]
            if field in ("first_usable", "deadline"):
                actual = str(actual)[:10]
            check(actual == value, f"cik {cik} {field} = {value} (got {actual})")
    check(by_cik.loc["900", "name"] == 'Kilo "K" Corp', "names with quotes survive")

    leaks = pd.read_csv(results / "us_leak_table.csv").set_index(["form", "convention"])
    expect_leaks = {
        ("10-K", "period end"): (100.0, 89.5),
        ("10-K", "period end + 90 days"): (50.0, 15.0),
        ("10-K", "US large-accelerated deadlines for all (10-Q 40, 10-K 60)"): (100.0, 28.5),
        ("10-K", "legal deadline for the filer's status"): (50.0, 15.0),
        ("10-Q", "period end"): (100.0, 35.0),
        ("10-Q", "period end + 30 days"): (100.0, 7.0),
        ("10-Q", "period end + 45 days"): (33.33, 1.0),
        ("10-Q", "period end + 60 days"): (0.0, 0.0),
        ("10-Q", "US large-accelerated deadlines for all (10-Q 40, 10-K 60)"): (33.33, 4.0),
        ("10-Q", "legal deadline for the filer's status"): (33.33, 1.0),
        ("10-Q", "actual filing, first usable day (baseline)"): (0.0, 0.0),
    }
    for key, (pct, median) in expect_leaks.items():
        got = leaks.loc[key]
        check(abs(got["leaked_pct"] - pct) < 0.01 and abs(got["median_days_early"] - median) < 0.01,
              f"{key[0]} '{key[1]}' leaks {pct}% (median {median}) -> got {got['leaked_pct']}% ({got['median_days_early']})")
    check(abs(leaks.loc[("10-Q", "period end + 45 days"), "median_days_unused"] - 10.5) < 0.01, "unused median for 10-Q +45 is 10.5")

    summary = pd.read_csv(results / "us_lag_summary.csv").set_index(["form", "filer_group"])
    check(summary.loc[("10-K", "all"), "median_lag"] == 89.5, "10-K median lag 89.5")
    check(summary.loc[("10-K", "non-accelerated"), "late_pct"] == 100.0, "late non-accelerated 10-K flagged")
    check(summary.loc[("10-K", "non-accelerated"), "late_after_nt_pct"] == 0.0, "NT extension clears it")
    check(summary.loc[("10-Q", "all"), "median_lag"] == 34.0, "10-Q median lag 34")

    extras = pd.read_csv(results / "us_extras.csv")
    value = lambda form, start: extras[(extras["form"] == form) & extras["measure"].str.startswith(start)]["value"].iloc[0]
    check(abs(value("10-Q", "accepted at or after 16:00") - 66.67) < 0.01, "after-close share for 10-Q")
    check(abs(value("10-Q", "accepted after 17:30") - 33.33) < 0.01, "after-cutoff share for 10-Q")
    check(value("10-K", "later amended") == 50.0, "10-K/A matched to its original")
    check(value("10-Q", "later amended") == 0.0, "no 10-Q flagged as amended")
    check(bool(by_cik.loc["200", "amended_later"]) and not bool(by_cik.loc["900", "amended_later"]), "amended flag per filing")
    check(value("10-K", "10-K/A amendments") == 1, "10-K/A counted")

    cdf = pd.read_csv(results / "us_lag_cdf.csv").set_index("day")
    check(abs(cdf.loc[34, "10-Q | all"] - 0.6667) < 0.001 and cdf.loc[45, "10-Q | all"] == 1.0, "CDF values")
    by_year = pd.read_csv(results / "us_lag_by_year.csv").set_index("period_year")
    check(by_year.loc[2024, "10-K | non-accelerated"] == 105, "lag by year")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        site = tmp / "site"
        site.mkdir()
        sizes = build_zips(site)

        print("1. Server with range support")
        server, log = serve(site, ranges=True)
        out = run(server.server_address[1], tmp / "run1")
        print(out)
        served = sum(n for path, _, n in log if path.endswith(("q2.zip", "q3.zip")))
        check("range requests" in out and "2025q4: not published yet (404), skipped" in out, "range path used, missing quarter skipped")
        check(served < 0.1 * sum(sizes.values()), f"downloaded {served:,} of {sum(sizes.values()):,} bytes")
        validate(tmp / "run1" / "results", tmp / "run1" / "data")

        print("\n2. Rerun uses the cache")
        log.clear()
        out = run(server.server_address[1], tmp / "run1")
        check(out.count(": cached") == 2, "both quarters cached")
        server.shutdown()

        print("\n3. Server without range support")
        server, log = serve(site, ranges=False)
        out = run(server.server_address[1], tmp / "run2")
        check("downloading whole zips instead" in out and "full download" in out, "fell back to full download")
        check(not list((tmp / "run2" / "data" / "raw").glob("*.zip")), "zips deleted after extracting")
        validate(tmp / "run2" / "results", tmp / "run2" / "data")

        print("\n4. --keep-zips keeps them")
        run(server.server_address[1], tmp / "run3", "--keep-zips")
        check(len(list((tmp / "run3" / "data" / "raw").glob("*.zip"))) == 2, "two zips kept")
        server.shutdown()

        print("\n5. Missing SEC_USER_AGENT stops with a clear message")
        env = {k: v for k, v in os.environ.items() if k != "SEC_USER_AGENT"}
        result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, env=env)
        check(result.returncode != 0 and "SEC_USER_AGENT" in result.stderr, "refuses to run without a user agent")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
