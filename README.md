# pit-lag

When could a backtest first have used a company's quarterly or annual report? `pit_lag` works that out for the US, Korea, Hong Kong, China and Thailand: the filing time or date when you have it, the legal deadline when you don't, then the next trading day.

It's one file and bundles no data, so it works on any vendor export that has period ends.

## Why

Most backtests join fundamentals at the period end, or at the period end plus a fixed lag. I checked that against the filing records of 356,323 US and Korean reports with period ends in 2015–2025 (see [analysis](analysis/)):

- Lags of 40 days for quarterly and 60 for annual reports, the deadlines for large US companies, use 99% of Korean reports too early.
- Korea's own deadline, used on the deadline day, takes about half of its quarterly reports a day early. Most are filed after the close.
- Filing date when known, deadline when not, then the next trading day: 5.2% of US filings and 2.3% of Korean reports still come in early, all of them late filers.

Write-up: [One lag doesn't fit all markets: look-ahead in US and Korean fundamentals](https://pit-lag.pages.dev/)

## Install

Copy `pit_lag.py` into your project. It needs pandas and numpy. Install `holidays` too, so deadlines skip public holidays (without it, only weekends and US federal holidays are skipped, and it warns):

    pip install pandas numpy holidays

## Use

```python
import pit_lag

pit_lag.available_date("2025-03-31", "KR", "quarterly")
# Timestamp('2025-05-16 00:00:00')

# Filing dates where you have them, deadlines where you don't
fund["available"] = pit_lag.available_date(
    fund["period_end"], fund["market"], fund["report_type"],
    filer_class=fund["filer_class"], filed=fund["filed"])

# Check a backtest you already have
checked = pit_lag.audit(panel, used="date", market="KR", report_type="quarterly")
```

- `filer_class`: US filer status ("large accelerated", "accelerated", "non-accelerated") or the SEC data sets' afs codes such as `1-LAF`. Leave it out when unknown and the latest deadline applies.
- `accepted`: the filing time. Before the close of a trading day, the report counts from that day. Use `at="open"` if your backtest needs it before the session starts.
- `extension=True` (the default) adds the US Form 12b-25 extension when falling back on the deadline.
- `audit()` returns your table with `available_date`, `too_early` and `days_early`, and prints a summary.
- `pit_lag.RULES` is the dated rules table. Add rows for other markets.

## Rules

| Market | Annual | Half-year | Quarterly |
|---|---|---|---|
| US | 10-K: 60 / 75 / 90 days by filer status, +15 with 12b-25 | – | 10-Q: 40 / 40 / 45 days, +5 with 12b-25 |
| Korea | 90 days | 45 days | 45 days |
| Hong Kong (Main Board) | 3 months (4 for years ending before 31 Dec 2010) | 2 months (3 before 30 Jun 2010) | no deadline |
| China | 4 months | 2 months | 1 month |
| Thailand | 3 months (2 without Q4 statements) | 45 days | 45 days |

A deadline on a weekend or holiday moves to the next business day. Sources and caveats are in the docstring of `pit_lag.py`.

## Tests

    python tests/test_pit_lag.py

Works with or without pytest.

## The analysis

`analysis/` holds the scripts behind the numbers above, and their output.

- `us_filing_lags.py`: reads the SEC Financial Statement Data Sets. Set `SEC_USER_AGENT` to your name and email first, as the SEC asks.
- `kr_filing_lags.py`: reads OpenDART's disclosure list. It needs a free API key in `DART_API_KEY`. Run it with `--check` first.
- `make_charts.py`: draws the two charts and `leak_comparison.csv` from `results/`.
- `results/`: the tables and charts. `kr_kind_timing_sample.csv` counts filing times on KRX's KIND site for five deadline days.

The raw filing lists aren't included; the scripts rebuild them in `analysis/data/`. The analysis also needs `requests` and `matplotlib` (`pip install -r analysis/requirements.txt`). Offline tests, from `analysis/` (the Korean one needs `holidays`):

    python tests/run_tests.py us_filing_lags.py
    python tests/run_kr_tests.py kr_filing_lags.py

## Licence

MIT. See [LICENSE](LICENSE).
