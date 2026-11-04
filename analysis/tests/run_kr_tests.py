"""End-to-end tests for kr_filing_lags.py against a mock OpenDART on a local server.

Usage, from the filing-lags folder: python tests/run_kr_tests.py kr_filing_lags.py
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

SCRIPT = Path(sys.argv[1]).resolve()
VALID_KEY = "a1" * 20
WRONG_KEY = "b2" * 20


def filing(cls, code, name, title, filed, seq, rm=""):
    return {"corp_cls": cls, "corp_code": code, "corp_name": name, "stock_code": code[-6:],
            "report_nm": title, "rcept_no": f"{filed}{seq:06d}", "flr_nm": name, "rcept_dt": filed, "rm": rm}


FILINGS = [
    filing("Y", "00000001", "Alpha", "Quarterly Report (2025.03)", "20250515", 1),
    filing("Y", "00000001", "Alpha", "[Revision of statements]Quarterly Report (2025.03)", "20250601", 2),
    filing("Y", "00000002", "Beta", "Business Report (2024.12)", "20250331", 3),
    filing("Y", "00000003", "Gamma", "Semi-Annual Report (2025.06)", "20250814", 4),
    filing("K", "00000004", "Delta", "Quarterly Report (2025.03)", "20250520", 5),
    filing("K", "00000005", "Echo", "분기보고서 (2025.03)", "20250512", 6),
    filing("K", "00000006", "Foxtrot", "Settlement Documents (2025.03)", "20250513", 7),
    filing("Y", "00000007", "Golf", "Business Report (2014.12)", "20150331", 8),
    filing("Y", "00000002", "Beta", "Business Report (2024.12)", "20250402", 9),
    filing("N", "00000009", "Hotel", "Quarterly Report (2025.03)", "20250515", 10),
    filing("K", "00000008", "India", "Quarterly Report (2025.03)", "20250515", 11, rm="U"),
] + [filing("K", f"1{i:07d}", f"Filler {i}", "Quarterly Report (2024.03)", "20240514", 100 + i) for i in range(250)]


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, payload, code=200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            state["calls"] += 1
            if state["fail_first"] > 0:
                state["fail_first"] -= 1
                self.send_error(500)
                return
            if query.get("crtfc_key") != VALID_KEY or (url.path == "/api/list.json" and not state["korean_ok"]):
                return self.reply({"status": "010", "message": "등록되지 않은 키입니다."})
            if state["limit_after"] is not None and state["calls"] > state["limit_after"]:
                return self.reply({"status": "020", "message": "요청 제한을 초과하였습니다."})
            start, end = query["bgn_de"], query["end_de"]
            span = (date.fromisoformat(f"{end[:4]}-{end[4:6]}-{end[6:]}") -
                    date.fromisoformat(f"{start[:4]}-{start[4:6]}-{start[6:]}")).days
            if span > 92:
                return self.reply({"status": "100", "message": "필드의 부적절한 값입니다."})
            rows = sorted((f for f in FILINGS if start <= f["rcept_dt"] <= end
                           and f["corp_cls"] == query.get("corp_cls", f["corp_cls"])),
                          key=lambda f: (f["rcept_dt"], f["rcept_no"]))
            if not rows:
                return self.reply({"status": "013", "message": "조회된 데이타가 없습니다."})
            size = min(int(query.get("page_count", 10)), 100)
            page = int(query.get("page_no", 1))
            self.reply({"status": "000", "message": "정상", "page_no": page, "page_count": size,
                        "total_count": len(rows), "total_page": math.ceil(len(rows) / size),
                        "list": rows[(page - 1) * size: page * size]})

        def log_message(self, *args):
            pass

    return Handler


def serve(**options):
    state = {"calls": 0, "fail_first": 0, "korean_ok": False, "limit_after": None, **options}
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def run(server, workdir: Path, *extra, key=VALID_KEY, expect_ok=True):
    port = server.server_address[1]
    env = dict(os.environ, DART_API_KEY=key, PYTHONIOENCODING="utf-8",
               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    command = [sys.executable, str(SCRIPT), "--from-date", "2015-01-01", "--to-date", "2025-09-30",
               "--data-dir", str(workdir / "data"), "--out-dir", str(workdir / "results"),
               "--pause", "0.01", "--eng-url", f"http://127.0.0.1:{port}/engapi/list.json",
               "--kor-url", f"http://127.0.0.1:{port}/api/list.json", *extra]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
    output = result.stdout + result.stderr
    if expect_ok and result.returncode != 0:
        print(output)
        raise SystemExit(f"script failed with exit code {result.returncode}")
    check(key not in output, "the API key never appears in the output")
    return result.returncode, output


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


def validate(results: Path, data: Path):
    filings = pd.read_csv(data / "kr_filings.csv.gz", dtype={"corp_code": str})
    check(len(filings) == 256, f"256 reports kept (got {len(filings)})")
    named = filings[~filings["corp_name"].str.startswith("Filler")].set_index("corp_name")
    check(sorted(named.index) == ["Alpha", "Beta", "Delta", "Echo", "Gamma", "India"], "the expected companies")
    expected = {
        "Alpha": dict(report_type="quarterly", lag_days=45, deadline="2025-05-15", late=False, on_deadline_day=True,
                      first_usable="2025-05-16", first_usable_same_day="2025-05-15", corrected_later=True),
        "Beta": dict(report_type="annual", lag_days=90, deadline="2025-03-31", on_deadline_day=True,
                     first_usable="2025-04-01", corrected_later=False, filed="2025-03-31"),
        "Gamma": dict(report_type="half-year", lag_days=45, deadline="2025-08-14", first_usable="2025-08-18",
                      first_usable_same_day="2025-08-14"),
        "Delta": dict(lag_days=50, late=True, first_usable="2025-05-21", market="KOSDAQ"),
        "Echo": dict(report_type="quarterly", lag_days=42, first_usable="2025-05-13"),
        "India": dict(corrected_later=True, late=False),
    }
    for name, fields in expected.items():
        for field, value in fields.items():
            actual = named.loc[name, field]
            if field in ("deadline", "first_usable", "first_usable_same_day", "filed"):
                actual = str(actual)[:10]
            check(actual == value, f"{name} {field} = {value} (got {actual})")
    filler = filings[filings["corp_name"].str.startswith("Filler")].iloc[0]
    check(str(filler["deadline"])[:10] == "2024-05-16" and str(filler["first_usable"])[:10] == "2024-05-16",
          "Buddha's Birthday 2024 moves the deadline and the first usable day")

    leaks = pd.read_csv(results / "kr_leak_table.csv").set_index(["report_type", "convention"])
    expect = {
        ("annual", "period end + 60 days"): (100.0, 100.0, 28.0),
        ("annual", "Korean legal deadline (quarterly and half-year 45, annual 90)"): (100.0, 0.0, 1.0),
        ("annual", "US large-accelerated deadlines (quarterly and half-year 40, annual 60)"): (100.0, 100.0, 28.0),
        ("half-year", "period end + 45 days"): (100.0, 0.0, 4.0),
        ("half-year", "US large-accelerated deadlines (quarterly and half-year 40, annual 60)"): (100.0, 100.0, 7.0),
        ("half-year", "actual filing, next business day (baseline)"): (0.0, 0.0, 0.0),
        ("quarterly", "Korean legal deadline (quarterly and half-year 45, annual 90)"): (1.18, 0.39, 1.0),
    }
    for key, (pct, same_day, median) in expect.items():
        row = leaks.loc[key]
        check(abs(row["leaked_pct"] - pct) < 0.01 and abs(row["leaked_pct_if_usable_same_day"] - same_day) < 0.01
              and abs(row["median_days_early"] - median) < 0.01,
              f"{key[0]} '{key[1]}': {pct}% ({same_day}% same day), median {median} -> "
              f"{row['leaked_pct']}% ({row['leaked_pct_if_usable_same_day']}%), {row['median_days_early']}")

    summary = pd.read_csv(results / "kr_lag_summary.csv").set_index(["report_type", "market"])
    check(summary.loc[("annual", "KOSPI"), "median_lag"] == 90 and summary.loc[("annual", "KOSPI"), "on_deadline_day_pct"] == 100,
          "annual summary")
    check(summary.loc[("quarterly", "KOSDAQ"), "median_lag"] == 44, "quarterly KOSDAQ median 44")
    extras = pd.read_csv(results / "kr_extras.csv")
    value = lambda kind, start: extras[(extras["report_type"] == kind) & extras["measure"].str.startswith(start)]["value"].iloc[0]
    check(abs(value("quarterly", "corrected later") - 0.79) < 0.01, "corrections and rm flag U counted")
    check(value("quarterly", "correction filings") == 1, "one correction filing")
    check(value("all", "filings with an unrecognised title") == 1, "one unrecognised title")
    unmatched = pd.read_csv(results / "kr_unmatched_titles.csv")
    check(unmatched["title"].tolist() == ["Settlement Documents (YYYY.MM)"], "unmatched title listed")
    cdf = pd.read_csv(results / "kr_lag_cdf.csv").set_index("day")
    check(cdf.loc[89, "annual | all"] == 0 and cdf.loc[90, "annual | all"] == 1, "CDF")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        print("1. --check, Korean API rejects the key")
        server, state = serve()
        code, out = run(server, tmp / "check", "--check")
        print(out)
        check(re.search(r"KOSPI\s+3 filings", out) and re.search(r"KOSDAQ\s+4 filings", out), "window counts")
        check(re.search(r"KOSPI\s+1\n", out) and re.search(r"KOSDAQ\s+3\n", out), "companies with a 2025.03 quarterly report")
        check("Settlement Documents (YYYY.MM)" in out and "NOT RECOGNISED" in out, "unrecognised title flagged")
        check("분기보고서 (YYYY.MM)" in out and "-> quarterly" in out, "Korean title recognised")
        check("Not possible: OpenDART status 010" in out, "Korean API refusal reported")
        saved = (tmp / "check" / "results" / "kr_check.txt").read_text(encoding="utf-8")
        check("Cross-check against the Korean API" in saved and "NOT RECOGNISED" in saved, "check report saved to kr_check.txt")
        server.shutdown()

        print("\n2. --check, Korean API accepts the key")
        server, state = serve(korean_ok=True)
        code, out = run(server, tmp / "check", "--check")
        check("(cached)" in out, "check window reused from the cache")
        check(re.search(r"KOSPI\s+English\s+3\s+Korean\s+3\s+English/Korean 100\.0%", out)
              and re.search(r"KOSDAQ\s+English\s+4\s+Korean\s+4", out), "cross-check counts")
        server.shutdown()

        print("\n3. Full pull, interrupted by the daily limit, then resumed")
        server, state = serve(limit_after=40)
        code, out = run(server, tmp / "full", expect_ok=False)
        check(code != 0 and "daily call limit" in out, "stops cleanly on status 020")
        cached = len(list((tmp / "full" / "data" / "raw").glob("*.json.gz")))
        check(0 < cached < 86, f"{cached} windows cached before the stop")
        server.shutdown()
        server, state = serve(fail_first=2)
        code, out = run(server, tmp / "full")
        check("500" in out and "retrying" in out, "retries server errors")
        check(state["calls"] < 100, f"resumed from the cache ({state['calls']} calls)")
        validate(tmp / "full" / "results", tmp / "full" / "data")
        server.shutdown()

        print("\n4. A wrong key stops with a clear message")
        server, state = serve()
        code, out = run(server, tmp / "wrong", key=WRONG_KEY, expect_ok=False)
        check(code != 0 and "this key isn't registered" in out, "status 010 explained")
        server.shutdown()

        print("\n5. A missing key stops before any call")
        env = {k: v for k, v in os.environ.items() if k != "DART_API_KEY"}
        result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, env=env)
        check(result.returncode != 0 and "DART_API_KEY" in result.stderr, "asks for DART_API_KEY")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
