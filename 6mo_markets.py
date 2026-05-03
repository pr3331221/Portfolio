# 6-Month Stock Portfolio Optimizer

A factor-based stock selection system — optionally enhanced with a walk-forward LightGBM model — that selects and allocates capital across 50 stocks to maximize 6-month alpha over SPY. Designed for investors who want a data-driven, systematic approach to portfolio construction backed by survivorship-corrected factor signals.

---

## What It Does

You give it a list of stocks and $10,000. It gives you back a fully allocated portfolio — ticker by ticker, dollar by dollar — ranked and weighted by empirically-validated factor signals that were tested against 10,701 stock outcomes including recovered dead stocks.

It is **not** a trading bot. It makes one decision: where to put money today for the next 6 months. Rebalance semi-annually by re-running.

---

## How It Works (Overview)

```
tickers.csv
    │
    ▼
[1] Filter universe (remove warrants, OTC, micro-caps under $1B,
    CEFs/shells, extreme leverage, <3Y history, illiquid stocks)
    │
    ▼
[2] Fetch data (price history + fundamentals via yfinance)
    │
    ▼
[3] Score stocks using 9 survivorship-corrected factor signals
    (70% validated price factors + 30% fundamental regularizer),
    optionally blended with LightGBM walk-forward ML predictions
    │
    ▼
[4] Allocate capital (rank by composite score, apply data-integrity
    penalties, weight positions, enforce 15% per-stock and 40%
    per-sector caps)
    │
    ▼
[4b] Compute momentum baseline (top-50 by 6mo return, equal
     weighted — the benchmark to measure factor value-add against)
    │
    ▼
[5] Export final_portfolio.xlsx (4 sheets: Portfolio, All Rankings,
    How It Works, Baseline Comparison)
    │
    ▼
[5b] Monte Carlo simulation (10k correlated paths → VaR, CVaR,
     probability of loss, probability of beating SPY)
    │
    ▼
[6] Market health assessment (VIX, yield curve, breadth — context only)
```

---

## Files

| File | Purpose |
|---|---|
| `main.py` | Main pipeline — run this to generate your portfolio |
| `train_model.py` | Walk-forward LightGBM training (optional ML layer) |
| `backtest.py` | Historical replay of the ML top-N strategy using walk-forward OOS predictions |
| `build_delisted_universe.py` | Builds dead stock data from EDGAR indexes for survivorship correction |
| `edgar_fundamentals.py` | SEC EDGAR point-in-time fundamentals provider |
| `data/tickers.csv` | Your stock universe (download from NASDAQ screener) |
| `data/raw_ohlc_cache.pkl` | Auto-generated price history cache (created during fetch) |
| `data/edgar_cik_map.json` | Ticker → CIK mapping for EDGAR lookups |
| `data/edgar_facts/` | Cached EDGAR company facts JSONs (lazy-loaded per CIK) |
| `data/edgar_quarterly_index/` | Cached EDGAR full-index parses per year/quarter |
| `data/edgar_submissions_cache/` | Cached EDGAR submissions API responses |
| `data/delisted_universe.pkl` | Recovered dead stock data (price series + dead CIK metadata) |
| `data/trained_model.pkl` | Trained LightGBM model (created by `train_model.py`) |
| `data/oos_predictions.csv` | Walk-forward out-of-sample predictions (produced by `train_model.py`, consumed by `backtest.py`) |
| `data/backtest_semiannual.csv` | Semiannual non-overlapping rebalance backtest results |
| `data/backtest_monthly.csv` | Monthly rebalance backtest results |
| `data/backtest_deciles.csv` | Per-decile forward-return spread diagnostics |
| `final_portfolio.xlsx` | Output — your portfolio in a 4-sheet Excel workbook |
| `final_portfolio_portfolio.csv` | CSV mirror of the Portfolio sheet |
| `final_portfolio_all_rankings.csv` | CSV mirror of the All Rankings sheet |
| `final_portfolio_baseline.csv` | CSV mirror of the Baseline Comparison sheet |
| `raw_fetched_data.csv` | Output — all fetched data for every stock scored |
| `portfolio_run.log` | Full run log with model diagnostics and explanations |

---

## Setup

### Requirements

```bash
pip install yfinance pandas numpy lightgbm xlsxwriter tqdm scipy
```

Python 3.10 or higher recommended.

### Get Your Ticker List

1. Go to https://www.nasdaq.com/market-activity/stocks/screener
2. Set filters if desired (or leave blank for the full universe)
3. Click **Download CSV**
4. Save it to `data/tickers.csv`

The system automatically filters out warrants, rights, units, OTC stocks, ADRs, and anything under $1B market cap.

---

## Running the System

### Step 1 — Build delisted universe (optional, run once)

```bash
python build_delisted_universe.py
```

Scans EDGAR full-index files to find all historical filers (28,192 unique CIKs from 2005–2025), identifies the 12,963 that stopped filing (dead), recovers price data for 109 via yfinance, and saves everything to `data/delisted_universe.pkl`. This data is used by `train_model.py` to include dead stocks in training — the key to survivorship-corrected factor signals.

Runtime: 30–60 minutes first run. ~5 seconds after caching.

### Step 2 — Train the ML model (optional, improves scoring)

```bash
python train_model.py          # full training (2012–2025, ~15-30 min)
python train_model.py --quick  # fast mode (2020–2025, for testing)
```

Trains a walk-forward LightGBM model on monthly cross-sectional data from the OHLC cache + EDGAR point-in-time fundamentals + recovered dead stocks. Folds use an expanding training window with a **252-day purge** (full 1-year horizon protection) plus a **21-day embargo** to decorrelate adjacent test months. The training target is within-month percentile rank (`Forward_Rank`) rather than raw 6-month return — this removes common-factor regime noise and typically lifts monthly IC several-fold. Features are cross-sectionally z-scored per `Score_Date` (macro features excluded so regime conditioning stays intact). Raw `Vol_1Y` / `Vol_3Y` are deliberately dropped from the feature set to prevent the model from collapsing into a low-volatility factor proxy.

Saves to `data/trained_model.pkl` **and** writes `data/oos_predictions.csv` (every walk-forward test prediction, used by `backtest.py`). If the model file exists when `main.py` runs and the model clears all three gates — pooled **OOS IC > 0.03**, **monthly IC mean > 0.02**, and **sign-correct hit rate > 55%** — its predictions are blended with the factor scores. Otherwise, factor-only scoring is used.

**You can skip this step** — the factor scoring system works standalone and is the primary scoring method.

### Step 3 — Generate your portfolio

```bash
python main.py
```

Runtime: 30–60 minutes depending on universe size.

If an OHLC cache exists but no trained model is found (or the cache is newer), `main.py` will auto-train the ML model in quick mode before scoring.

In addition to `final_portfolio.xlsx`, `main.py` writes CSV mirrors of the portfolio, full rankings, and baseline sheets (`final_portfolio_portfolio.csv`, `final_portfolio_all_rankings.csv`, `final_portfolio_baseline.csv`) for programmatic access, and `portfolio_run.log` for full diagnostics.

### Step 4 — (Optional) Backtest the ML signal

```bash
python backtest.py
```

Replays "pick top-N by predicted score, equal-weight, hold 6 months" over all walk-forward OOS months in `data/oos_predictions.csv`. Produces three diagnostics in `data/`:

- `backtest_semiannual.csv` — non-overlapping Jan/Jul rebalance returns vs SPY, equal-weight universe, and bottom-N
- `backtest_monthly.csv` — monthly rebalance path (overlapping 6-month holds)
- `backtest_deciles.csv` — per-decile forward-return spread (tests signal monotonicity)

Out-of-sample by construction — every prediction comes from a walk-forward fold that never saw its own test month.

---

## Understanding the Output

### Excel Workbook — 4 Sheets

**Sheet 1: Portfolio** — the 50 allocated stocks. Columns flow left-to-right showing the ranking math chain:

| Column | What It Means |
|---|---|
| `Allocation $` | How much of your $10,000 goes into this stock |
| `Predicted 6mo Return` | Factor model's predicted 6-month return (or factor+ML blend if model is active) |
| `Model Agreement` | How consistently the validated price factors and fundamental factors agree on this stock's ranking (1.0 = full agreement) |
| `= Composite Score` | Predicted Return × Agreement — the actual ranking number used for selection |
| `Weight Adj %` | Net adjustment to allocation weight, if any (e.g. −12% for missing FCF data). "None" means no adjustment |
| `Why Adjusted` | Plain-English explanation of any adjustment applied |
| `Composite / Quality / Low Vol Pct Rank` | Each factor sub-score's percentile ranking in the full universe |
| `DCF Implied Return` | Fair value gap — positive means the stock appears undervalued vs projected cash flows |
| `Piotroski Score` | 9-point fundamental quality checklist (0–1, higher = better) |
| `Earnings Growth` | YoY earnings growth rate (positive = growing, negative = contracting) |
| `Return Consistency` | How smoothly returns were distributed over time (1.0 = steady compounder, low = spike-driven) |
| `Confidence Tier` | HIGH / MEDIUM / LOW / SPECULATIVE based on data completeness |
| `Green / Red Flags` | Key positive and warning signals for this stock |

**Sheet 2: All Rankings** — every stock in the screened universe, ranked by Composite Score. Rows highlighted in gold appear in the final portfolio.

**Sheet 3: How Scores Work** — plain-English explanation of the factor scoring formula.

**Sheet 4: Baseline Comparison** — the momentum baseline portfolio (top-50 by 6mo return, equal weighted) shown side by side with the factor portfolio.

---

## How Scoring Works

### Primary: 9-Factor Survivorship-Corrected Model

Factor weights were derived from Spearman rank correlations measured over **10,701 samples** (7,413 live stocks + 1,388 recovered dead stocks + 950 Shumway synthetic dead stocks) across **29 annual test dates from 1996–2025**.

#### The Critical Survivorship Bias Finding

The most important discovery: **the apparent "volatility premium" was 100% survivorship bias**. Before including dead stocks, high volatility appeared to predict higher returns (rho = +0.070). After correction, the signal flipped: **low volatility predicts higher returns** (rho = −0.197). High-vol stocks that went bankrupt (returning −100%) were simply invisible in the live-only dataset.

This single correction changed the entire factor model.

#### Validated Price Factors (70% of composite weight)

| Factor | Weight | Rho | Direction | What It Captures |
|---|---|---|---|---|
| Low Volatility | 15% | −0.197 | Lower vol → higher returns | **Corrected!** Dead stocks removed the false vol premium |
| Shallow Drawdown | 15% | +0.181 | Less negative drawdowns → higher returns | Risk control signal |
| 6-Month Momentum | 15% | +0.180 | 6mo winners continue outperforming | Strong monotonic quintile spread |
| 12-2mo Momentum | 10% | +0.179 | Jegadeesh-Titman momentum is alive | **Corrected!** Was rho=+0.008 without dead stocks |
| Above 200SMA | 10% | +0.149 | Stocks above 200SMA continue higher | Trend following signal |
| Return Consistency | 5% | +0.115 | Consistent returners outperform | **Corrected!** Flipped from −0.082 |

#### Fundamental Factors (30% — unvalidated regularizer)

These serve as a reality check against pure price chasing:

| Factor | Weight | Components |
|---|---|---|
| Quality | 15% | Piotroski score (30%), Profit margin (30%), Earnings quality (20%), ROE (20%) |
| Value | 10% | Book-to-market, low PE, FCF margin, DCF implied return (equal weighted) |
| Growth | 5% | Revenue growth, earnings growth (equal weighted) |

#### Composite Score

```
Composite = 0.15×LowVol + 0.15×ShallowDD + 0.15×Mom6mo + 0.10×Mom12_2
          + 0.10×Above200SMA + 0.05×Consistency
          + 0.15×Quality + 0.10×Value + 0.05×Growth
```

Factor agreement (spread between validated and fundamental signals) modulates the final score: `Composite_Score = Predicted_Return × Agreement`.

### Optional: LightGBM ML Layer

If `train_model.py` has been run and the model cleared all three OOS gates (pooled IC > 0.03, monthly IC mean > 0.02, hit rate > 55%), the scoring blends ML predictions with factor scores:

```
Final = (1 − α) × Factor Score + α × ML Score
α     = clip(3 × OOS_IC, 0, 0.6)
```

Factors always get ≥40% weight. If no model exists or any gate fails, factor-only scoring is used.

The ML model is a single LightGBM trained via walk-forward expanding windows with a **252-day purge** and **21-day embargo**, predicting within-month percentile rank. Inference replays the exact feature neutralization used in training (cross-sectional z-scoring, with sector-relative z-scoring if the saved model was trained that way). It captures non-linear interactions and regime-dependent effects that fixed-weight factors miss.

---

## The Allocation Adjustment System

### Design Principle

Factor weights already account for volatility, momentum, leverage, and other stock characteristics. The adjustment system is deliberately narrow — correcting only for **missing or fabricated input data** that the factor scorer cannot detect.

### Adjustments Applied (data quality only)

| Situation | Adjustment | Reason |
|---|---|---|
| Earnings quality unavailable | −8% | One cash flow feature missing |
| FCF unavailable | −12% | DCF used assumed −15% FCF margin |
| Both FCF + earnings quality missing | −20% | Two cash flow features unavailable |
| Revenue unavailable | −90% | All margin/DCF features are fabricated — effectively disqualified |

---

## Universe Filtering

Hard pre-scoring filters applied before any ranking:

| Filter | Threshold | Rationale |
|---|---|---|
| Non-operating companies | CEFs, shells, SPACs, ETFs excluded | Misleading financials (NAV changes as "profit") |
| Impossible fundamentals | Profit margin > 100% or EBITDA > Revenue | Data corruption indicator |
| Leverage cap | D/E > 5x (or >10x for REITs/Utilities) | Equity is a thin residual, extreme return swings |
| Minimum history | 3 years of price data | Momentum and volatility features require it |
| Market cap floor | $1B | Micro-caps can't be traded at model prices |
| Liquidity | ADV ≥ 100K shares | Avoid illiquid stocks with wide spreads |
| Sector classification | Must have sector | Required for sector cap enforcement |

Additional filters in allocation: quality gate (at least one positive signal among 1Y return, 3Y return, profit margin, or ROE) and $500K daily dollar volume minimum.

---

## Position Sizing

Three allocation methods via `ALLOCATION_METHOD`:

- **`"composite"` (default):** proportional to adjusted Composite Score
- **`"equal"`:** equal dollar per stock ($200 each for $10K portfolio)
- **`"hrp"` (Hierarchical Risk Parity):** uses actual daily return correlations from the OHLC cache to cluster correlated stocks, then allocates inversely to cluster variance (Lopez de Prado 2016). Maximises diversification without unstable covariance inversion.

All modes apply 15% per-stock cap and 40% per-sector cap. No industry cap — if the scoring says buy 11 gold miners and the scoring is empirically validated, it lets it.

---

## Momentum Baseline

Every run computes a **momentum baseline**: top-50 by 6-month return, equal weighted, same sector caps. This appears in Sheet 4.

The factor-only stocks (those the multi-factor model picks that pure momentum would skip) are the most important ones to monitor. If they outperform the baseline-only stocks after 6 months, the factor model is adding genuine value beyond raw momentum.

---

## Market Health Assessment

Five signals scored −2 to +2 each:

| Signal | What It Measures |
|---|---|
| SPY Trend | Price vs 200SMA, drawdown from ATH |
| Market Breadth | % of universe above 200SMA / positive 1Y/6mo returns |
| VIX | Fear gauge (<15 calm, >30 stress, >40 crisis) |
| Yield Curve | 10Y−3mo spread (Fed's preferred recession indicator) |
| Valuation | Median PE vs historical norms |

This is context for *how* to enter (lump sum vs phased), not whether to invest. A failure here never blocks portfolio output.

---

## Monte Carlo Simulation

10,000-path simulation of the portfolio's 6-month outcome:

1. Each stock's return sampled from a log-normal distribution (factor prediction as mean, historical volatility as std)
2. Returns correlated via Cholesky decomposition of **actual daily return correlations** from the OHLC cache (most recent 252 trading days, Ledoit-Wolf shrinkage)
3. Bayesian shrinkage: predictions are shrunk toward SPY when model confidence is low

**Reports:** median return, 5th/95th percentile, VaR(5%), CVaR(5%), probability of loss, probability of beating SPY.

Purely informational — does not affect portfolio selection.

---

## Configuration

```python
TOTAL_CAPITAL      = 10_000        # Total dollars to allocate
PREDICTION_YEARS   = 0.5           # 6-month prediction horizon
MIN_HISTORY_DAYS   = 126           # Minimum price history (~6 months)
MIN_MARKET_CAP     = 500_000_000   # Minimum market cap ($500M) for initial ticker load
MIN_ALLOCATION     = 25            # Ignore positions below $25
MAX_ALLOC_FRAC     = 0.15          # Max 15% per stock
TOP_N              = 50            # Portfolio size
ALLOCATION_METHOD  = "composite"   # "composite" | "equal" | "hrp"
```

Note: `filter_universe()` applies a stricter $1B market cap floor and 3-year history requirement during pre-scoring filtering.

---

## Data Sources

| Data | Source | Notes |
|---|---|---|
| Price history | yfinance | Full daily history, 30+ years for majors |
| Fundamentals | yfinance + SEC EDGAR | EDGAR provides point-in-time historical fundamentals (2009+) |
| EDGAR filings | SEC XBRL API | Official 10-K/10-Q data, free, no API key |
| VIX | yfinance (`^VIX`) | Cached for macro regime features |
| 10Y Treasury | yfinance (`^TNX`) | Historical yield for DCF + macro features |
| 3-month T-bill | yfinance (`^IRX`) | Historical yield spread |
| Delisted stocks | EDGAR full-index + yfinance | 109 recovered price series + 12,963 dead CIK metadata |

---

## Survivorship Bias Correction

Unlike most retail-grade stock screeners, this system **actively corrects for survivorship bias**:

1. **`build_delisted_universe.py`** scans EDGAR full-index files (2005–2025) to find all 28,192 historical filers, identifies 12,963 that stopped filing, recovers price data for 109 via yfinance, and saves dead stock metadata.

2. **Factor validation** (which produced the weights used in scoring) included 1,388 recovered dead stocks and 950 Shumway (1997) synthetic dead stocks (−30% return with distressed profiles) alongside 7,413 live stocks across 29 test dates.

3. **The ML training pipeline** (`train_model.py`) includes recovered dead stocks in its training cross-sections, so the model learns from real failure outcomes.

**Key correction:** The Vol_1Y factor flipped from rho=+0.070 (high vol appears good) to rho=−0.197 (high vol is bad) after including dead stocks. Three other factors (12-2mo momentum, 200SMA trend, return consistency) also reversed or significantly strengthened. Without this correction, the system would systematically overweight exactly the stocks most likely to blow up.

The ticker list still contains only stocks that exist today, so the live scoring universe has survivors-only. The correction happens in factor weight derivation and ML training, not in the scoring universe itself.

---

## Known Limitations

**Residual survivorship bias in live scoring.** Factor weights and ML training include dead stocks, but the scoring universe is still the current NASDAQ screener. Companies that will delist in the next 6 months are scored as if they'll survive. The shorter 6-month horizon limits this exposure compared to longer horizons.

**Fundamental data quality.** yfinance fundamentals are sometimes stale or incorrect. EDGAR provides point-in-time historical filing data for training windows (2009+), but live scoring still relies on yfinance for current fundamentals.

**Unvalidated fundamental factors.** The 30% fundamental weight (quality, value, growth) is included as a regularizer but could not be survivorship-corrected without point-in-time fundamental data for dead stocks. These weights are based on academic priors, not empirical validation on this dataset.

**The model has never predicted from today's environment.** It has learned from diverse historical regimes, but 2026's specific conditions are unseen. This is unavoidable — no model can train on the future.

**Return prediction is inherently noisy.** The system's edge is in **ranking** — the top 50 should outperform a random 50 on average — not in precise return magnitude predictions. Validated composite Spearman rho is +0.188 (modest but real).

---

## Technical Notes

### EDGAR Point-in-Time Integration

The ML training pipeline uses SEC EDGAR XBRL data (2009+) for point-in-time fundamentals — the actual revenue, FCF, EBITDA, and Piotroski scores that were filed before each training window date. EDGAR facts are lazy-loaded per CIK on demand (7.5GB total would exceed RAM if loaded at once).

### Correlation Matrices

Both the Monte Carlo simulation and HRP allocation use **actual pairwise daily return correlations** computed from the OHLC cache (most recent 252 trading days), with Ledoit-Wolf-style shrinkage (0.2 × identity + 0.8 × sample) toward identity for regularization.

### Auto-Training

`main.py` automatically triggers `train_model.py --quick` if no trained model exists or the OHLC cache is newer than the model file. This ensures the ML layer stays current without manual intervention.

"""
build_delisted_universe.py
==========================
Build a comprehensive historical universe that includes dead/delisted stocks.

This script:
  1. Scans EDGAR full-index files to find ALL companies that filed 10-K/10-Q
  2. Uses EDGAR submissions API to recover tickers for dead CIKs
  3. Attempts to fetch price data for dead tickers via yfinance
  4. Saves everything to data/delisted_universe.pkl for use by validate_factors.py

The result is a mapping of:
  - Dead ticker → price series (for stocks yfinance still has data for)
  - Dead CIK → metadata (name, SIC, last filing, ticker if known)
  - Per-year universe counts (how many companies were alive at each date)

These allow validate_factors.py to:
  - Include recovered dead stocks in factor correlations (with real prices)
  - Apply Shumway (1997) delist-return correction for unrecoverable stocks
  - Report honest, survivorship-corrected factor signals

Runtime: ~30-60 min first run (EDGAR API calls + yfinance fetches), ~5s after caching.
"""

import os
import sys
import json
import time
import pickle
import logging
import urllib.request
import urllib.error
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

USER_AGENT = "portfolio-ml-research prod1221@gmail.com"
DATA_DIR = "data"
OHLC_CACHE = os.path.join(DATA_DIR, "raw_ohlc_cache.pkl")
EDGAR_INDEX_DIR = os.path.join(DATA_DIR, "edgar_quarterly_index")
SUBMISSIONS_CACHE_DIR = os.path.join(DATA_DIR, "edgar_submissions_cache")
OUTPUT_FILE = os.path.join(DATA_DIR, "delisted_universe.pkl")

# SEC rate limit: 10 req/sec
class _RateLimiter:
    def __init__(self, rps: float):
        self._min_interval = 1.0 / rps
        self._lock = threading.Lock()
        self._last = 0.0
    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

_limiter = _RateLimiter(8)  # stay under 10/sec limit


# ─── Step 1: Load EDGAR full-index data ─────────────────────────────────

def download_edgar_quarterly_index(year: int, qtr: int) -> list:
    """
    Download and parse the EDGAR full-index company.idx for a given
    year/quarter, filter to 10-K / 10-Q filings, and return a list of
    entries: [{"cik": str, "name": str, "date": "YYYY-MM-DD", "form": str}].

    Caches the parsed result to data/edgar_quarterly_index/{year}_Q{qtr}.json.
    """
    os.makedirs(EDGAR_INDEX_DIR, exist_ok=True)
    cache_file = os.path.join(EDGAR_INDEX_DIR, f"{year}_Q{qtr}.json")

    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/company.idx"
    _limiter.acquire()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("latin-1", errors="replace")
    except Exception as e:
        log.warning(f"  {year}/Q{qtr}: download failed ({e})")
        return []

    # company.idx is fixed-width. Header ends with a line of dashes.
    lines = raw.splitlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("---"):
            data_start = i + 1
            break
    if data_start == 0:
        log.warning(f"  {year}/Q{qtr}: unrecognized index format")
        return []

    # Column widths per SEC spec (company.idx):
    #   Company Name: 62, Form Type: 12, CIK: 12, Date Filed: 12, Filename: rest
    entries = []
    wanted_forms = {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KSB", "10-KSB/A"}
    for line in lines[data_start:]:
        if len(line) < 86:
            continue
        name = line[0:62].strip()
        form = line[62:74].strip()
        cik = line[74:86].strip()
        date = line[86:98].strip()
        if form not in wanted_forms or not cik or not date:
            continue
        entries.append({"cik": cik, "name": name, "form": form, "date": date})

    with open(cache_file, "w") as f:
        json.dump(entries, f)
    log.info(f"  {year}/Q{qtr}: downloaded and parsed {len(entries)} 10-K/10-Q entries")
    return entries


def load_all_edgar_indexes(years=range(2005, 2026), auto_download=True):
    """
    Load cached EDGAR quarterly indexes. Returns:
      cik_info: dict {cik_str: {name, years_active: set, last_date: str}}

    If auto_download is True, missing cache files are fetched from sec.gov.
    """
    cik_info = defaultdict(lambda: {"name": "", "years": set(), "last_date": ""})

    total_downloaded = 0
    for year in years:
        for qtr in [1, 3]:  # Q1 + Q3 for semi-annual snapshots
            cache_file = os.path.join(EDGAR_INDEX_DIR, f"{year}_Q{qtr}.json")
            if not os.path.exists(cache_file):
                if auto_download:
                    entries = download_edgar_quarterly_index(year, qtr)
                    if not entries:
                        continue
                    total_downloaded += 1
                else:
                    log.debug(f"No cached index for {year}/Q{qtr}, skipping")
                    continue
            else:
                with open(cache_file, "r") as f:
                    entries = json.load(f)

            for e in entries:
                info = cik_info[e["cik"]]
                info["name"] = e["name"]
                info["years"].add(year)
                if e["date"] > info["last_date"]:
                    info["last_date"] = e["date"]

    if total_downloaded:
        log.info(f"Downloaded {total_downloaded} new quarterly index files from EDGAR")
    
    # Convert sets to sorted lists for JSON serialization
    result = {}
    for cik, info in cik_info.items():
        result[cik] = {
            "name": info["name"],
            "years": sorted(info["years"]),
            "last_date": info["last_date"],
        }
    
    return result


# ─── Step 2: Identify dead companies ────────────────────────────────────

def _norm_cik(c) -> str:
    """Normalize any CIK representation to a canonical unpadded string."""
    try:
        return str(int(str(c).strip()))
    except (ValueError, TypeError):
        return str(c).strip().lstrip("0") or "0"


def find_dead_ciks(cik_info, ohlc_tickers, cik_ticker_map):
    """
    Find CIKs that:
      - Filed 10-K/10-Q at some point
      - Are NOT in our current OHLC cache
      - Stopped filing (last active year < 2024)
    
    Returns: dict {cik: {name, last_date, years, ticker (if known)}}
    """
    # Build reverse map: ticker → canonical CIK.
    # edgar_cik_map.json pads CIKs ("0001045810") while the EDGAR full-index
    # stores them unpadded ("1045810"). Normalize both sides to a canonical
    # form before any set/dict lookup, otherwise nothing ever matches.
    ticker_to_cik = {}
    cik_to_ticker = {}
    for cik, info in cik_ticker_map.items():
        t = info.get("ticker", "")
        if not t:
            continue
        ncik = _norm_cik(cik)
        ticker_to_cik[t.upper()] = ncik
        cik_to_ticker.setdefault(ncik, t.upper())

    # CIKs in our live OHLC cache
    cache_ciks = set()
    for t in ohlc_tickers:
        ncik = ticker_to_cik.get(t.upper())
        if ncik is not None:
            cache_ciks.add(ncik)

    dead = {}
    for cik, info in cik_info.items():
        ncik = _norm_cik(cik)
        # Skip if in our cache
        if ncik in cache_ciks:
            continue

        # Must have stopped filing
        years = info["years"]
        if not years:
            continue
        last_year = max(years)
        if last_year >= 2024:  # still active
            continue

        # Must have filed for at least 2 years (filters out one-off filers)
        if len(years) < 2:
            continue

        # Recover ticker if we have it mapped
        ticker = cik_to_ticker.get(ncik)

        dead[ncik] = {
            "name": info["name"],
            "last_date": info["last_date"],
            "last_year": last_year,
            "years": years,
            "ticker": ticker,
        }

    return dead, cache_ciks


# ─── Step 3: Recover tickers via EDGAR submissions API ──────────────────

def fetch_submission(cik: str) -> dict:
    """
    Fetch a company's submission metadata from EDGAR.
    Returns: {tickers: list, name: str, sic: str, exchanges: list}
    """
    _limiter.acquire()
    
    padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{padded}.json"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return {
            "tickers": data.get("tickers", []),
            "name": data.get("name", ""),
            "sic": data.get("sic", ""),
            "exchanges": data.get("exchanges", []),
            "sicDescription": data.get("sicDescription", ""),
        }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        log.debug(f"  CIK {cik}: HTTP {e.code}")
        return None
    except Exception as e:
        log.debug(f"  CIK {cik}: {e}")
        return None


def recover_tickers_from_submissions(dead_ciks: dict, max_lookups=5000):
    """
    For dead CIKs without a ticker, query EDGAR submissions API.
    Returns updated dead_ciks with recovered tickers + SIC codes.
    """
    cache_file = os.path.join(SUBMISSIONS_CACHE_DIR, "submissions_cache.json")
    os.makedirs(SUBMISSIONS_CACHE_DIR, exist_ok=True)
    
    # Load existing cache
    submissions_cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            submissions_cache = json.load(f)
    
    # Find CIKs we need to look up
    need_lookup = []
    for cik, info in dead_ciks.items():
        if info["ticker"] is not None:
            continue  # already have ticker
        if cik in submissions_cache:
            # Apply cached result
            cached = submissions_cache[cik]
            if cached and cached.get("tickers"):
                info["ticker"] = cached["tickers"][0]
                info["sic"] = cached.get("sic", "")
                info["exchanges"] = cached.get("exchanges", [])
            continue
        need_lookup.append(cik)
    
    if not need_lookup:
        log.info("All dead CIKs already have ticker info (from cache)")
        return dead_ciks
    
    # Limit lookups
    need_lookup = need_lookup[:max_lookups]
    log.info(f"Looking up {len(need_lookup)} dead CIKs via EDGAR submissions API...")
    
    recovered = 0
    checkpoint_every = 500
    
    for i, cik in enumerate(need_lookup):
        result = fetch_submission(cik)
        submissions_cache[cik] = result
        
        if result and result.get("tickers"):
            dead_ciks[cik]["ticker"] = result["tickers"][0]
            dead_ciks[cik]["sic"] = result.get("sic", "")
            dead_ciks[cik]["exchanges"] = result.get("exchanges", [])
            recovered += 1
        
        if (i + 1) % 100 == 0:
            log.info(f"  Checked {i+1}/{len(need_lookup)}, recovered {recovered} tickers so far")
        
        # Checkpoint cache
        if (i + 1) % checkpoint_every == 0:
            with open(cache_file, "w") as f:
                json.dump(submissions_cache, f)
    
    # Final save
    with open(cache_file, "w") as f:
        json.dump(submissions_cache, f)
    
    log.info(f"  Recovered {recovered} tickers from {len(need_lookup)} submissions lookups")
    return dead_ciks


# ─── Step 4: Fetch price data for recovered dead tickers ────────────────

def _strip_tz(series):
    if hasattr(series.index, 'tz') and series.index.tz is not None:
        try:
            series.index = series.index.tz_convert(None)
        except TypeError:
            series.index = series.index.tz_localize(None)
    return series


def _fetch_stooq_close(ticker: str) -> "pd.Series | None":
    """
    Try to fetch a daily close series for a (possibly delisted) US ticker
    from stooq.com. Stooq keeps post-delist history for many names Yahoo
    has pruned. Returns a pd.Series of close prices (tz-naive) or None.
    """
    # Stooq uses lowercase ticker + ".us" for US stocks; dots become dashes.
    sym = ticker.lower().replace(".", "-").replace("$", "").strip()
    if not sym:
        return None
    url = f"https://stooq.com/q/d/l/?s={sym}.us&i=d"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "portfolio-ml-research/1.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    # Stooq returns the literal body "No data" on failure
    if not raw or "No data" in raw[:200]:
        return None
    lines = raw.strip().splitlines()
    if len(lines) < 50:  # need >= ~50 trading days for anything useful
        return None
    header = lines[0].split(",")
    try:
        date_idx = header.index("Date")
        close_idx = header.index("Close")
    except ValueError:
        return None
    dates, closes = [], []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) <= max(date_idx, close_idx):
            continue
        try:
            closes.append(float(parts[close_idx]))
            dates.append(parts[date_idx])
        except ValueError:
            continue
    if len(closes) < 50:
        return None
    idx = pd.to_datetime(dates, errors="coerce")
    s = pd.Series(closes, index=idx).dropna()
    if s.empty:
        return None
    return _strip_tz(s).sort_index()


def fetch_dead_stock_prices(dead_ciks: dict, existing_cache: dict, max_fetch=2000):
    """
    Try to fetch price data for dead tickers via yfinance.
    Returns: dict {ticker: pd.Series of close prices}
    """
    import yfinance as yf
    
    prices_cache_file = os.path.join(DATA_DIR, "delisted_prices.pkl")
    
    # Load existing recovered prices
    recovered_prices = {}
    if os.path.exists(prices_cache_file):
        with open(prices_cache_file, "rb") as f:
            recovered_prices = pickle.load(f)
        log.info(f"Loaded {len(recovered_prices)} previously recovered price series")

    # Also seed from the legacy recovered_delisted_ohlc.pkl if present
    legacy_file = os.path.join(DATA_DIR, "recovered_delisted_ohlc.pkl")
    if os.path.exists(legacy_file):
        try:
            with open(legacy_file, "rb") as f:
                legacy = pickle.load(f)
            seeded = 0
            for t, s in legacy.items():
                if t not in recovered_prices:
                    recovered_prices[t] = s
                    seeded += 1
            if seeded:
                log.info(f"Seeded {seeded} price series from legacy recovered_delisted_ohlc.pkl")
        except Exception as e:
            log.debug(f"  Could not seed from legacy file: {e}")
    
    # Collect dead tickers to try fetching
    tickers_to_try = set()
    for cik, info in dead_ciks.items():
        ticker = info.get("ticker")
        if not ticker:
            continue
        if ticker in existing_cache:  # already in our live cache
            continue
        if ticker in recovered_prices:  # already recovered previously
            continue
        tickers_to_try.add(ticker)
    
    tickers_to_try = sorted(tickers_to_try)[:max_fetch]
    
    if not tickers_to_try:
        log.info("No new dead tickers to fetch")
        return recovered_prices
    
    log.info(f"Attempting to fetch prices for {len(tickers_to_try)} dead tickers...")

    fetched = 0
    failed = 0
    stooq_hits = 0

    for i, ticker in enumerate(tickers_to_try):
        got = None
        # Try Stooq first for dead stocks — it retains more delisted history
        # than Yahoo. Rate-limit via our global SEC-style limiter (also used
        # for Stooq here as a courtesy, though Stooq has no strict cap).
        _limiter.acquire()
        try:
            got = _fetch_stooq_close(ticker)
            if got is not None and len(got) >= 100:
                recovered_prices[ticker] = got
                fetched += 1
                stooq_hits += 1
                got = "ok"
        except Exception:
            got = None

        # Fallback to yfinance if Stooq had nothing
        if got != "ok":
            try:
                df = yf.download(ticker, period="max", progress=False, auto_adjust=True, timeout=10)
                if df is not None and len(df) >= 100:
                    if isinstance(df.columns, pd.MultiIndex):
                        close = df["Close"].iloc[:, 0].dropna()
                    else:
                        close = df["Close"].dropna()
                    close = _strip_tz(close)
                    recovered_prices[ticker] = close
                    fetched += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

        if (i + 1) % 100 == 0:
            log.info(f"  Checked {i+1}/{len(tickers_to_try)}: "
                     f"{fetched} recovered ({stooq_hits} via Stooq), {failed} failed")
            with open(prices_cache_file, "wb") as f:
                pickle.dump(recovered_prices, f)

        time.sleep(0.05)  # gentle rate limit

    with open(prices_cache_file, "wb") as f:
        pickle.dump(recovered_prices, f)

    log.info(f"  Total recovered: {len(recovered_prices)} dead stock price series")
    log.info(f"  This run: {fetched} new ({stooq_hits} via Stooq), {failed} failed")

    return recovered_prices


# ─── Step 5: Build per-year universe statistics ─────────────────────────

def build_universe_stats(cik_info, cache_ciks, dead_ciks, recovered_tickers):
    """
    For each year, count:
      - Total companies filing 10-K/10-Q (the real universe)
      - How many are in our OHLC cache
      - How many are dead but recovered (we have prices)
      - How many are dead and unrecoverable (need Shumway correction)
    """
    stats = {}
    
    for year in range(2005, 2026):
        # Who was filing this year?
        year_ciks = set()
        for cik, info in cik_info.items():
            if year in info["years"]:
                year_ciks.add(cik)
        
        in_cache = year_ciks & cache_ciks
        dead_this_year = set()
        recovered_this_year = set()
        unrecoverable_this_year = set()
        
        for cik in year_ciks - cache_ciks:
            if cik in dead_ciks:
                dead_this_year.add(cik)
                ticker = dead_ciks[cik].get("ticker")
                if ticker and ticker in recovered_tickers:
                    recovered_this_year.add(cik)
                else:
                    unrecoverable_this_year.add(cik)
        
        stats[year] = {
            "total_filers": len(year_ciks),
            "in_cache": len(in_cache),
            "dead_recovered": len(recovered_this_year),
            "dead_unrecoverable": len(unrecoverable_this_year),
            "dead_total": len(dead_this_year),
        }
    
    return stats


# ─── Main ───────────────────────────────────────────────────────────────

def build(force=False):
    """Build the comprehensive delisted universe dataset."""
    
    # Check if we already have a fully built output
    if not force and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "rb") as f:
            existing = pickle.load(f)
        n_prices = len(existing.get("recovered_prices", {}))
        n_dead = len(existing.get("dead_ciks", {}))
        n_stats = len(existing.get("universe_stats", {}))
        log.info(f"Existing delisted_universe.pkl: {n_prices} price series, {n_dead} dead CIKs, {n_stats} year stats")
        
        # If it looks complete, just return it
        if n_dead > 100 and n_stats >= 20:
            log.info("Using existing build. Pass force=True or delete data/delisted_universe.pkl to rebuild.")
            return existing
    
    # Load OHLC cache
    log.info("Loading OHLC cache...")
    with open(OHLC_CACHE, "rb") as f:
        ohlc_cache = pickle.load(f)
    ohlc_tickers = {k for k in ohlc_cache if not k.startswith("__")}
    log.info(f"  {len(ohlc_tickers)} stock tickers in live cache")
    
    # Load CIK→ticker map
    cik_map_file = os.path.join(DATA_DIR, "edgar_cik_ticker_exchange.json")
    if os.path.exists(cik_map_file):
        with open(cik_map_file, "r") as f:
            cik_ticker_map = json.load(f)
    else:
        # Fall back to the simple CIK map
        cik_map_file2 = os.path.join(DATA_DIR, "edgar_cik_map.json")
        if os.path.exists(cik_map_file2):
            with open(cik_map_file2, "r") as f:
                raw = json.load(f)
            cik_ticker_map = {v: {"ticker": k, "name": "", "exchange": ""} for k, v in raw.items()}
        else:
            log.error("No CIK→ticker map found. Run survivorship_bias.py first.")
            return None
    log.info(f"  {len(cik_ticker_map)} CIK→ticker mappings")
    
    # Step 1: Load all EDGAR index data
    print("\n" + "=" * 80)
    print("STEP 1: Loading EDGAR full-index data (2005-2025)")
    print("=" * 80)
    cik_info = load_all_edgar_indexes(range(2005, 2026))
    print(f"  {len(cik_info)} unique CIKs filed 10-K/10-Q across 2005-2025")

    # Safety guard: refuse to proceed (and overwrite any existing pkl) if the
    # index load returned essentially nothing. This prevents silently wiping a
    # previously-built universe when network or parse errors occur.
    if len(cik_info) < 1000:
        log.error(
            f"EDGAR index load returned only {len(cik_info)} CIKs — expected ≥10,000. "
            "Aborting before any existing delisted_universe.pkl is overwritten."
        )
        if os.path.exists(OUTPUT_FILE):
            log.error(f"Existing {OUTPUT_FILE} is preserved.")
        return None
    
    # Step 2: Identify dead companies
    print("\n" + "=" * 80)
    print("STEP 2: Identifying dead/delisted companies")
    print("=" * 80)
    dead_ciks, cache_ciks = find_dead_ciks(cik_info, ohlc_tickers, cik_ticker_map)
    
    n_with_ticker = sum(1 for v in dead_ciks.values() if v.get("ticker"))
    n_without = sum(1 for v in dead_ciks.values() if not v.get("ticker"))
    print(f"  {len(dead_ciks)} dead companies found")
    print(f"    With ticker: {n_with_ticker}")
    print(f"    Without ticker: {n_without}")
    
    # Step 3: Try to recover tickers from EDGAR submissions API.
    # Note: Per past testing, EDGAR often returns empty ticker lists for
    # fully-delisted companies. But for recently-dead and some long-dead
    # filers it still returns data, so it's worth trying — a few hundred
    # recovered tickers is better than zero. Cached after first lookup.
    print("\n" + "=" * 80)
    print("STEP 3: Recovering tickers via EDGAR submissions API")
    print("=" * 80)
    dead_ciks = recover_tickers_from_submissions(dead_ciks, max_lookups=5000)
    n_with_ticker = sum(1 for v in dead_ciks.values() if v.get("ticker"))
    n_without = sum(1 for v in dead_ciks.values() if not v.get("ticker"))
    print(f"  After submissions lookup: {n_with_ticker} with ticker, {n_without} without")
    
    # Step 4: Fetch price data for dead tickers
    print("\n" + "=" * 80)
    print("STEP 4: Fetching prices for dead tickers via yfinance")
    print("=" * 80)
    recovered_prices = fetch_dead_stock_prices(dead_ciks, ohlc_cache, max_fetch=2000)
    print(f"  {len(recovered_prices)} dead stocks with recovered price data")
    
    # Step 5: Build per-year universe statistics
    print("\n" + "=" * 80)
    print("STEP 5: Building per-year universe statistics")
    print("=" * 80)
    
    universe_stats = build_universe_stats(
        cik_info, cache_ciks, dead_ciks, set(recovered_prices.keys())
    )
    
    print(f"\n  {'Year':>6s}  {'Total':>7s}  {'In Cache':>9s}  {'Recovered':>10s}  {'Unrecov.':>9s}  {'Missing%':>9s}")
    print("  " + "-" * 55)
    
    total_missing_pct = []
    for year in sorted(universe_stats.keys()):
        s = universe_stats[year]
        total = s["in_cache"] + s["dead_recovered"] + s["dead_unrecoverable"]
        miss_pct = s["dead_unrecoverable"] / total * 100 if total > 0 else 0
        total_missing_pct.append(miss_pct)
        print(f"  {year:>6d}  {s['total_filers']:>7d}  {s['in_cache']:>9d}  {s['dead_recovered']:>10d}  "
              f"{s['dead_unrecoverable']:>9d}  {miss_pct:>8.1f}%")
    
    avg_missing = np.mean(total_missing_pct)
    
    # ─── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("BUILD COMPLETE")
    print("=" * 80)
    print(f"""
  Total dead companies identified:  {len(dead_ciks):,d}
  Dead with recovered tickers:      {n_with_ticker:,d}
  Dead with recovered PRICES:       {len(recovered_prices):,d}
  
  Average unrecoverable per year:   {avg_missing:.1f}%
  
  Academic correction will be applied for unrecoverable stocks:
    - Shumway (1997): avg delisting return = -30% over 6mo
    - Applied proportionally to the unrecoverable fraction at each test date
""")
    
    # Save output
    output = {
        "recovered_prices": recovered_prices,  # ticker → pd.Series
        "dead_ciks": dead_ciks,                # cik → {name, ticker, last_date, years, sic, ...}
        "universe_stats": universe_stats,       # year → {total_filers, in_cache, dead_recovered, dead_unrecoverable}
        "cache_ciks": list(cache_ciks),         # CIKs in our live OHLC cache
        "cik_info": cik_info,                   # cik → {name, years, last_date}
    }
    
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(output, f)
    log.info(f"Saved to {OUTPUT_FILE}")
    
    return output


if __name__ == "__main__":
    build(force=True)
"""
edgar_fundamentals.py
=====================
Point-in-time fundamental data from SEC EDGAR XBRL API.

For each training window date, returns fundamental values that were
actually known as of that date -- using the most recent 10-K or 10-Q
filing submitted to the SEC before the window date.

This eliminates look-ahead bias in training: a window dated 2010-06-01
uses 2009-2010 revenue, income, and cash flows -- not 2026 values.

Coverage:
  - 2009 onward: XBRL was required by the SEC for large accelerated filers
    starting with fiscal years ending on or after June 15, 2009.
  - Pre-2009 windows: returns NaN for all fundamental features. Those
    windows still train on price/momentum/macro features which are
    correctly point-in-time from the OHLC cache.

Data source: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
  - Free, no API key required
  - Full filing history per company in one JSON
  - Each entry has: val, filed (submission date), end (period end), form

Rate limit: max 10 requests/second to EDGAR. Use User-Agent header.
Cache: save companyfacts JSONs to data/edgar_facts/{cik}.json
       so we never hit EDGAR more than once per company.
"""

import os
import json
import time
import logging
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from typing import Optional

log = logging.getLogger(__name__)

EDGAR_BASE       = "https://data.sec.gov"
CIK_MAP_URL      = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL        = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_CACHE_DIR  = "data/edgar_facts"
CIK_MAP_FILE     = "data/edgar_cik_map.json"
USER_AGENT       = "portfolio-ml-research prod1221@gmail.com"
FETCH_PAUSE      = 0.15   # ~6.5 req/sec, safely under 10 req/sec limit
MAX_RETRIES      = 3
RETRY_BACKOFF    = [2.0, 5.0, 15.0]  # seconds to wait before each retry
BULK_WORKERS     = 8      # parallel download threads for bulk_download_edgar_facts
BULK_RPS         = 9      # max requests/sec across all threads (SEC limit is 10)


class _RateLimiter:
    """Thread-safe token-bucket rate limiter for SEC EDGAR's 10 req/sec limit."""
    def __init__(self, rps: float):
        self._min_interval = 1.0 / rps
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# CIK MAP  (ticker → CIK string, zero-padded to 10 digits)
# ─────────────────────────────────────────────────────────────────────────────

def download_cik_map(save_path: str = CIK_MAP_FILE) -> dict:
    """
    Download SEC's master ticker-to-CIK mapping.
    Returns dict: {ticker_upper: cik_string_10digits}
    Saves to disk for reuse.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    log.info("Downloading CIK map from SEC...")
    req = urllib.request.Request(
        CIK_MAP_URL,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = json.loads(r.read())

    # raw is {idx: {cik_str, ticker, title}}
    cik_map = {}
    for entry in raw.values():
        ticker = str(entry.get("ticker", "")).upper().strip()
        cik    = str(entry.get("cik_str", "")).zfill(10)
        if ticker and cik:
            cik_map[ticker] = cik

    with open(save_path, "w") as f:
        json.dump(cik_map, f)

    log.info(f"CIK map: {len(cik_map)} tickers saved to {save_path}")
    return cik_map


def load_cik_map(cache_path: str = CIK_MAP_FILE) -> dict:
    """Load cached CIK map, downloading if not present."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return download_cik_map(cache_path)


# ─────────────────────────────────────────────────────────────────────────────
# COMPANYFACTS DOWNLOAD + CACHE
# ─────────────────────────────────────────────────────────────────────────────

def download_companyfacts(cik: str, cache_dir: str = EDGAR_CACHE_DIR) -> Optional[dict]:
    """
    Download companyfacts JSON for one CIK from EDGAR.
    Saves to cache_dir/{cik}.json. Returns the facts dict or None on error.
    Retries with exponential backoff on transient failures (403, 429, 5xx).
    """
    os.makedirs(cache_dir, exist_ok=True)
    url = FACTS_URL.format(cik=cik)
    for attempt in range(MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            path = os.path.join(cache_dir, f"{cik}.json")
            with open(path, "w") as f:
                json.dump(data, f)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                log.debug(f"EDGAR CIK={cik}: 404 not found (no XBRL filings)")
                return None  # no point retrying
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                log.debug(f"EDGAR CIK={cik}: HTTP {e.code}, retry {attempt+1}/{MAX_RETRIES} in {wait}s")
                time.sleep(wait)
            else:
                log.warning(f"EDGAR CIK={cik}: HTTP {e.code} after {MAX_RETRIES} retries")
                return None
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                log.debug(f"EDGAR CIK={cik}: {e}, retry {attempt+1}/{MAX_RETRIES} in {wait}s")
                time.sleep(wait)
            else:
                log.warning(f"EDGAR CIK={cik}: {e} after {MAX_RETRIES} retries")
                return None
    return None


def load_companyfacts(cik: str, cache_dir: str = EDGAR_CACHE_DIR) -> Optional[dict]:
    """Load companyfacts from disk cache. Returns None if not cached."""
    path = os.path.join(cache_dir, f"{cik}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_or_download_companyfacts(
    cik: str,
    cache_dir: str = EDGAR_CACHE_DIR,
    pause: float = FETCH_PAUSE
) -> Optional[dict]:
    """Load from disk, or download if not cached."""
    facts = load_companyfacts(cik, cache_dir)
    if facts is not None:
        return facts
    time.sleep(pause)
    return download_companyfacts(cik, cache_dir)


# ─────────────────────────────────────────────────────────────────────────────
# POINT-IN-TIME VALUE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(facts: dict) -> dict:
    """
    Strip the outer 'facts' wrapper that SEC EDGAR wraps companyfacts in.

    The EDGAR companyfacts API returns:
        {"cik": 320193, "entityName": "Apple Inc.", "facts": {"us-gaap": {...}}}

    All internal helpers expect the inner dict {"us-gaap": {...}}.
    Calling facts.get("facts", facts) is backward-compatible: if the outer
    wrapper is absent (e.g. in unit tests) it returns facts unchanged.
    """
    return facts.get("facts", facts)


def _get_concept_entries(facts: dict, concept: str) -> list:
    """
    Return all XBRL entries for a us-gaap concept.
    Each entry is a dict with keys: val, filed, end, form, accn, fp, fy, etc.
    Handles both the raw EDGAR companyfacts JSON (has outer "facts" key) and
    already-unwrapped dicts (used in tests and legacy cached files).
    """
    return (
        _unwrap(facts).get("us-gaap", {})
                      .get(concept, {})
                      .get("units", {})
                      .get("USD", [])
    )


def _get_concept_entries_shares(facts: dict, concept: str) -> list:
    """Same but for share-denominated concepts."""
    return (
        _unwrap(facts).get("us-gaap", {})
                      .get(concept, {})
                      .get("units", {})
                      .get("shares", [])
    )


def get_annual_as_of(
    facts: dict,
    concepts: list,
    as_of_date: pd.Timestamp,
    use_shares: bool = False
) -> float:
    """
    Get the most recent annual (10-K) value for any of the given concepts,
    filed on or before as_of_date.

    Tries each concept in order; returns the first non-NaN result.
    Returns np.nan if no annual filing exists before as_of_date.
    """
    as_of_str = as_of_date.strftime("%Y-%m-%d")

    for concept in concepts:
        if use_shares:
            entries = _get_concept_entries_shares(facts, concept)
        else:
            entries = _get_concept_entries(facts, concept)

        # Filter: annual form, filed before as_of_date, has a value
        annual = [
            x for x in entries
            if x.get("form") in ("10-K", "10-K/A")
            and x.get("filed", "9999") <= as_of_str
            and "val" in x
            and x["val"] is not None
        ]
        if not annual:
            continue

        # Sort by filed date descending, take most recent
        annual.sort(key=lambda x: x["filed"], reverse=True)
        return float(annual[0]["val"])

    return np.nan


def get_two_annual_as_of(
    facts: dict,
    concepts: list,
    as_of_date: pd.Timestamp
) -> tuple:
    """
    Get the two most recent annual values for a concept filed before as_of_date.
    Returns (current, prior) — both may be np.nan.
    Used for growth rate computation.
    """
    as_of_str = as_of_date.strftime("%Y-%m-%d")

    for concept in concepts:
        entries = _get_concept_entries(facts, concept)
        annual = [
            x for x in entries
            if x.get("form") in ("10-K", "10-K/A")
            and x.get("filed", "9999") <= as_of_str
            and "val" in x
            and x["val"] is not None
        ]
        if len(annual) < 2:
            continue
        annual.sort(key=lambda x: x["filed"], reverse=True)
        return float(annual[0]["val"]), float(annual[1]["val"])

    return np.nan, np.nan


def get_three_annual_as_of(
    facts: dict,
    concepts: list,
    as_of_date: pd.Timestamp
) -> tuple:
    """
    Get the three most recent annual values. Returns (curr, prior, prior2).
    Used for growth acceleration computation.
    """
    as_of_str = as_of_date.strftime("%Y-%m-%d")

    for concept in concepts:
        entries = _get_concept_entries(facts, concept)
        annual = [
            x for x in entries
            if x.get("form") in ("10-K", "10-K/A")
            and x.get("filed", "9999") <= as_of_str
            and "val" in x
            and x["val"] is not None
        ]
        if len(annual) < 3:
            continue
        annual.sort(key=lambda x: x["filed"], reverse=True)
        return float(annual[0]["val"]), float(annual[1]["val"]), float(annual[2]["val"])

    return np.nan, np.nan, np.nan


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT NAME FALLBACK LISTS
# (different companies use different XBRL concept names for the same item)
# ─────────────────────────────────────────────────────────────────────────────

REVENUE_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
    "RevenuesNetOfInterestExpense",
]

NET_INCOME_CONCEPTS = [
    "NetIncomeLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
    "ProfitLoss",
]

OPERATING_CF_CONCEPTS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]

CAPEX_CONCEPTS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsForCapitalImprovements",
    "CapitalExpendituresIncurredButNotYetPaid",
    "PaymentsToAcquireProductiveAssets",
]

GROSS_PROFIT_CONCEPTS = [
    "GrossProfit",
]

OPERATING_INCOME_CONCEPTS = [
    "OperatingIncomeLoss",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
]

DA_CONCEPTS = [
    "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization",
    "Depreciation",
    "AmortizationOfIntangibleAssets",
]

ASSETS_CONCEPTS = [
    "Assets",
]

LIABILITIES_CONCEPTS = [
    "Liabilities",
]

EQUITY_CONCEPTS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "LiabilitiesAndStockholdersEquity",
]

LONG_TERM_DEBT_CONCEPTS = [
    "LongTermDebtNoncurrent",
    "LongTermDebt",
    "LongTermNotesPayable",
    "LongTermDebtAndCapitalLeaseObligations",
]

SHARES_CONCEPTS = [
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
]

EPS_CONCEPTS = [
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
]

RD_CONCEPTS = [
    "ResearchAndDevelopmentExpense",
    "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
]

CURRENT_ASSETS_CONCEPTS = [
    "AssetsCurrent",
]

CURRENT_LIABILITIES_CONCEPTS = [
    "LiabilitiesCurrent",
]

CASH_CONCEPTS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsAndShortTermInvestments",
    "Cash",
]

SHORT_TERM_DEBT_CONCEPTS = [
    "ShortTermBorrowings",
    "DebtCurrent",
    "LongTermDebtCurrent",
]


# ─────────────────────────────────────────────────────────────────────────────
# PIOTROSKI F-SCORE (point-in-time from EDGAR)
# ─────────────────────────────────────────────────────────────────────────────

def compute_piotroski(
    facts: dict,
    as_of_date: pd.Timestamp,
    revenue_curr: float,
    net_income_curr: float,
    op_cf_curr: float,
    assets_curr: float,
    ltd_curr: float,
    gross_profit_curr: float,
) -> float:
    """
    Compute Piotroski F-Score (0–9, scaled to 0–1) from EDGAR data.
    Uses the two most recent annual filings before as_of_date.

    F1: ROA > 0                    (net_income / assets > 0)
    F2: CFO > 0                    (operating cash flow > 0)
    F3: delta ROA > 0              (ROA improved year over year)
    F4: Accruals < 0               (CFO/assets > ROA — cash earnings quality)
    F5: delta Leverage <= 0        (long-term debt / assets did not increase)
    F6: delta Current Ratio >= 0   (liquidity improved or held)
    F7: No equity dilution         (shares did not increase)
    F8: delta Gross Margin >= 0    (gross margin improved)
    F9: delta Asset Turnover >= 0  (revenue / assets improved)
    """
    score     = 0
    available = 0  # track how many criteria had sufficient data

    def _safe(x): return x if (not np.isnan(x) and x is not None) else np.nan

    # Fetch all needed values upfront
    rev_curr, rev_prior = get_two_annual_as_of(facts, REVENUE_CONCEPTS,       as_of_date)
    ni_curr,  ni_prior  = get_two_annual_as_of(facts, NET_INCOME_CONCEPTS,    as_of_date)
    cf_curr,  cf_prior  = get_two_annual_as_of(facts, OPERATING_CF_CONCEPTS,  as_of_date)
    ass_curr, ass_prior = get_two_annual_as_of(facts, ASSETS_CONCEPTS,        as_of_date)
    ltd_c,    ltd_p     = get_two_annual_as_of(facts, LONG_TERM_DEBT_CONCEPTS, as_of_date)
    gp_curr,  gp_prior  = get_two_annual_as_of(facts, GROSS_PROFIT_CONCEPTS,  as_of_date)
    ca_curr,  ca_prior  = get_two_annual_as_of(facts, CURRENT_ASSETS_CONCEPTS, as_of_date)
    cl_curr,  cl_prior  = get_two_annual_as_of(facts, CURRENT_LIABILITIES_CONCEPTS, as_of_date)

    # Fallback to caller-supplied values if EDGAR lookup returned NaN
    if np.isnan(ni_curr):  ni_curr  = _safe(net_income_curr)
    if np.isnan(cf_curr):  cf_curr  = _safe(op_cf_curr)
    if np.isnan(ass_curr): ass_curr = _safe(assets_curr)

    # F1: ROA > 0
    if not np.isnan(ni_curr) and not np.isnan(ass_curr) and ass_curr > 0:
        available += 1
        score += int(ni_curr / ass_curr > 0)

    # F2: CFO > 0
    if not np.isnan(cf_curr):
        available += 1
        score += int(cf_curr > 0)

    # F3: ROA improved YoY
    if (not np.isnan(ni_curr) and not np.isnan(ni_prior) and
            not np.isnan(ass_curr) and not np.isnan(ass_prior) and
            ass_curr > 0 and ass_prior > 0):
        available += 1
        score += int(ni_curr / ass_curr > ni_prior / ass_prior)

    # F4: Accruals -- CFO/Assets > ROA (cash quality signal)
    if (not np.isnan(cf_curr) and not np.isnan(ni_curr) and
            not np.isnan(ass_curr) and ass_curr > 0):
        available += 1
        score += int(cf_curr / ass_curr > ni_curr / ass_curr)

    # F5: Leverage (LTD/Assets) did not increase YoY
    if (not np.isnan(ltd_c) and not np.isnan(ltd_p) and
            not np.isnan(ass_curr) and not np.isnan(ass_prior) and
            ass_curr > 0 and ass_prior > 0):
        available += 1
        score += int(ltd_c / ass_curr <= ltd_p / ass_prior)

    # F6: Current ratio improved YoY
    if (not np.isnan(ca_curr) and not np.isnan(cl_curr) and
            not np.isnan(ca_prior) and not np.isnan(cl_prior) and
            cl_curr > 0 and cl_prior > 0):
        available += 1
        score += int(ca_curr / cl_curr >= ca_prior / cl_prior)

    # F7: No share dilution (shares did not increase materially YoY)
    sh_p_list = [
        x for x in _get_concept_entries_shares(facts, "CommonStockSharesOutstanding")
        if x.get("form") in ("10-K", "10-K/A")
        and x.get("filed", "9999") <= as_of_date.strftime("%Y-%m-%d")
        and "val" in x
    ]
    sh_p_list.sort(key=lambda x: x["filed"], reverse=True)
    sh_c = float(sh_p_list[0]["val"]) if len(sh_p_list) >= 1 else np.nan
    sh_p = float(sh_p_list[1]["val"]) if len(sh_p_list) >= 2 else np.nan

    if not np.isnan(sh_c) and not np.isnan(sh_p) and sh_p > 0:
        available += 1
        score += int(sh_c <= sh_p * 1.02)  # allow 2% rounding tolerance

    # F8: Gross margin improved YoY
    if (not np.isnan(gp_curr) and not np.isnan(gp_prior) and
            not np.isnan(rev_curr) and not np.isnan(rev_prior) and
            rev_curr > 0 and rev_prior > 0):
        available += 1
        score += int(gp_curr / rev_curr >= gp_prior / rev_prior)

    # F9: Asset turnover improved YoY
    if (not np.isnan(rev_curr) and not np.isnan(rev_prior) and
            not np.isnan(ass_curr) and not np.isnan(ass_prior) and
            ass_curr > 0 and ass_prior > 0):
        available += 1
        score += int(rev_curr / ass_curr >= rev_prior / ass_prior)

    # Normalise by available criteria (consistent with fetch_single).
    # Require at least 5 criteria to compute a valid score; fewer → NaN.
    if available < 5:
        return np.nan
    return round(score / available, 4)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT: get_pit_fundamentals
# ─────────────────────────────────────────────────────────────────────────────

def get_pit_fundamentals(
    facts: dict,
    as_of_date: pd.Timestamp,
    price: float,
    beta: float = 1.0,
    risk_free_rate: float = np.nan,
) -> dict:
    """
    Compute point-in-time fundamental features for a training window.

    Parameters
    ----------
    facts           : companyfacts dict from EDGAR (may be None if not available)
    as_of_date      : the window end date -- only filings before this date are used
    price           : stock price at as_of_date (from OHLC cache -- already point-in-time)
    beta            : beta computed from price history (already point-in-time)
    risk_free_rate  : historical 10Y Treasury yield at window date (decimal, e.g. 0.018).
                      If NaN, falls back to 4.5% (today's approximate rate). Using the
                      actual historical rate is critical for DCF accuracy across training
                      windows -- a 2012 window should use ~1.8%, not 4.5%.

    Returns
    -------
    dict of feature_name -> float
    All values are point-in-time. NaN means data was not available in EDGAR
    before as_of_date (pre-2009 or company didn't file that concept).

    The caller should update() the training row with these values,
    overwriting the stale yfinance snapshot.
    """
    nan = np.nan

    # Return all-NaN if no facts available (pre-2009 or company not in EDGAR)
    if facts is None:
        return _nan_fundamentals()

    # XBRL coverage begins 2009. Before that, EDGAR has no structured data.
    if as_of_date < pd.Timestamp("2009-01-01"):
        return _nan_fundamentals()

    try:
        # ── Annual income statement ──────────────────────────────────────────
        revenue      = get_annual_as_of(facts, REVENUE_CONCEPTS,      as_of_date)
        net_income   = get_annual_as_of(facts, NET_INCOME_CONCEPTS,   as_of_date)
        gross_profit = get_annual_as_of(facts, GROSS_PROFIT_CONCEPTS, as_of_date)
        op_income    = get_annual_as_of(facts, OPERATING_INCOME_CONCEPTS, as_of_date)
        rd_expense   = get_annual_as_of(facts, RD_CONCEPTS,           as_of_date)
        da           = get_annual_as_of(facts, DA_CONCEPTS,           as_of_date)

        # ── Annual cash flow statement ───────────────────────────────────────
        op_cf        = get_annual_as_of(facts, OPERATING_CF_CONCEPTS, as_of_date)
        capex        = get_annual_as_of(facts, CAPEX_CONCEPTS,        as_of_date)
        # capex is always negative in EDGAR (payment outflow); take abs
        if not np.isnan(capex):
            capex = abs(capex)

        # ── Balance sheet (most recent available filing, Q or K) ─────────────
        total_assets = get_annual_as_of(facts, ASSETS_CONCEPTS,           as_of_date)
        total_liab   = get_annual_as_of(facts, LIABILITIES_CONCEPTS,      as_of_date)
        equity       = get_annual_as_of(facts, EQUITY_CONCEPTS,           as_of_date)
        long_term_debt = get_annual_as_of(facts, LONG_TERM_DEBT_CONCEPTS, as_of_date)
        shares       = get_annual_as_of(facts, SHARES_CONCEPTS, as_of_date, use_shares=True)

        # EPS (diluted preferred, basic fallback)
        eps = np.nan
        for concept in EPS_CONCEPTS:
            entries = _get_concept_entries(facts, concept)
            # EPS uses USD/shares units
            eps_entries = (
                _unwrap(facts).get("us-gaap", {})
                              .get(concept, {})
                              .get("units", {})
                              .get("USD/shares", [])
            )
            annual_eps = [
                x for x in eps_entries
                if x.get("form") in ("10-K", "10-K/A")
                and x.get("filed", "9999") <= as_of_date.strftime("%Y-%m-%d")
                and "val" in x
            ]
            if annual_eps:
                annual_eps.sort(key=lambda x: x["filed"], reverse=True)
                eps = float(annual_eps[0]["val"])
                break

        # ── Earnings Growth Signal (YoY EPS change) ─────────────────────────
        # Previously always NaN in EDGAR PIT. Now computed from two consecutive
        # annual EPS values so the model can learn from it during training.
        earnings_growth_signal = nan
        for concept in EPS_CONCEPTS:
            eps_entries = (
                _unwrap(facts).get("us-gaap", {})
                              .get(concept, {})
                              .get("units", {})
                              .get("USD/shares", [])
            )
            annual_eps_list = [
                x for x in eps_entries
                if x.get("form") in ("10-K", "10-K/A")
                and x.get("filed", "9999") <= as_of_date.strftime("%Y-%m-%d")
                and "val" in x and x["val"] is not None
            ]
            if len(annual_eps_list) >= 2:
                annual_eps_list.sort(key=lambda x: x["filed"], reverse=True)
                eps_curr  = float(annual_eps_list[0]["val"])
                eps_prior = float(annual_eps_list[1]["val"])
                if not np.isnan(eps_curr) and not np.isnan(eps_prior) and eps_prior != 0:
                    earnings_growth_signal = float(
                        np.clip((eps_curr - eps_prior) / abs(eps_prior), -2.0, 10.0)
                    )
                break

        # ── Revenue growth (two most recent annual filings) ──────────────────
        rev_curr, rev_prior = get_two_annual_as_of(facts, REVENUE_CONCEPTS, as_of_date)
        rev_growth = nan
        if not np.isnan(rev_curr) and not np.isnan(rev_prior) and rev_prior != 0:
            rev_growth = float((rev_curr - rev_prior) / abs(rev_prior))
            rev_growth = float(np.clip(rev_growth, -0.9, 10.0))

        # ── Revenue growth acceleration (three annual filings) ───────────────
        rev_curr2, rev_prior2, rev_prior3 = get_three_annual_as_of(
            facts, REVENUE_CONCEPTS, as_of_date
        )
        rev_growth_accel = nan
        if not any(np.isnan(x) for x in [rev_curr2, rev_prior2, rev_prior3]):
            if rev_prior2 != 0 and rev_prior3 != 0:
                g1 = (rev_curr2  - rev_prior2) / abs(rev_prior2)
                g0 = (rev_prior2 - rev_prior3) / abs(rev_prior3)
                rev_growth_accel = float(np.clip(g1 - g0, -5.0, 5.0))

        # ── Gross margin trend ───────────────────────────────────────────────
        gp_curr, gp_prior = get_two_annual_as_of(facts, GROSS_PROFIT_CONCEPTS, as_of_date)
        rc_curr, rc_prior = (rev_curr, rev_prior)
        gross_margin_trend = nan
        if (not np.isnan(gp_curr) and not np.isnan(gp_prior) and
                not np.isnan(rc_curr) and not np.isnan(rc_prior) and
                rc_curr > 0 and rc_prior > 0):
            gm_new = gp_curr / rc_curr
            gm_old = gp_prior / rc_prior
            gross_margin_trend = float(gm_new - gm_old)

        # ── Derived metrics ──────────────────────────────────────────────────
        fcf          = nan
        if not np.isnan(op_cf) and not np.isnan(capex):
            fcf = float(op_cf - capex)

        fcf_margin   = nan
        if not np.isnan(fcf) and not np.isnan(revenue) and revenue > 0:
            fcf_margin = float(np.clip(fcf / revenue, -1.0, 0.60))

        profit_margin = nan
        if not np.isnan(net_income) and not np.isnan(revenue) and revenue > 0:
            profit_margin = float(np.clip(net_income / revenue, -1.0, 0.60))

        ebitda = nan
        if not np.isnan(op_income):
            if not np.isnan(da):
                ebitda = float(op_income + da)
            else:
                # Rough proxy: operating income + 3% of revenue for D&A
                if not np.isnan(revenue):
                    ebitda = float(op_income + 0.03 * revenue)

        earnings_quality = nan
        if not np.isnan(op_cf) and not np.isnan(net_income) and net_income != 0:
            earnings_quality = float(np.clip(op_cf / net_income, -5.0, 10.0))

        roe = nan
        if not np.isnan(net_income) and not np.isnan(equity) and equity > 0:
            roe = float(np.clip(net_income / equity, -2.0, 5.0))

        debt_equity = nan
        if not np.isnan(long_term_debt) and not np.isnan(equity) and equity > 0:
            # Multiply by 100 to match yfinance percentage format.
            # yfinance debtToEquity returns e.g. 102.63 for a 1.03 ratio.
            # Without this, training sees D/E=1.03 while scoring sees D/E=102.63
            # -- a 100x feature distribution shift that corrupts the model.
            debt_equity = float(np.clip(long_term_debt / equity * 100, -10, 500))

        # Market cap at window date: shares from EDGAR × price from OHLC
        market_cap = nan
        if not np.isnan(shares) and shares > 0 and not np.isnan(price) and price > 0:
            market_cap = float(shares * price)

        # PE: price / trailing EPS
        pe = nan
        if not np.isnan(eps) and eps > 0 and not np.isnan(price) and price > 0:
            pe = float(np.clip(price / eps, 0.0, 500.0))

        # ── DCF (recomputed from historical fundamentals) ────────────────────
        dcf_implied_return = nan
        try:
            if (not np.isnan(revenue) and revenue > 0 and
                    not np.isnan(market_cap) and market_cap > 0 and
                    not np.isnan(rev_growth)):

                fcf_m = fcf_margin if not np.isnan(fcf_margin) else -0.15
                beta_v = beta if not np.isnan(beta) else 1.0
                # Use historical risk-free rate if available. This is critical:
                # a 2012 training window with 10Y Treasury at 1.8% must NOT
                # discount at today's 4.5%. Using today's rate makes every 2012
                # stock look overvalued in DCF, corrupting what the model learns.
                rfr = risk_free_rate if not np.isnan(risk_free_rate) else 0.045
                rfr = max(0.005, min(rfr, 0.12))  # clip to plausible range
                wacc = rfr + max(0.5, min(beta_v, 2.5)) * 0.055

                g0 = max(-0.3, min(rev_growth, 1.5))
                terminal_growth = 0.05
                projected_revenue = revenue
                projected_fcf = 0.0
                # DCF margin target: realistic reversion, not hardcoded 8%
                dcf_tgt = min(0.08, fcf_m + 0.15)
                dcf_tgt = max(dcf_tgt, -0.05)

                for yr in range(1, 7):
                    frac = yr / 6.0
                    g_yr = g0 * (1 - frac) + terminal_growth * frac
                    projected_revenue *= (1 + g_yr)
                    fcf_m_proj = fcf_m + (dcf_tgt - fcf_m) * frac
                    fcf_m_proj = max(-0.3, min(fcf_m_proj, 0.5))
                    projected_fcf += projected_revenue * fcf_m_proj / ((1 + wacc) ** yr)

                t_mult = float(np.clip(12.0 + g0 * 53.3, 12.0, 28.0))
                # Use actual year-6 projected margin, not hardcoded 8%
                yr6_fcf_margin = dcf_tgt
                yr6_fcf      = projected_revenue * max(yr6_fcf_margin, 0.0)
                terminal_val = (yr6_fcf * t_mult) / ((1 + wacc) ** 6)
                intrinsic    = projected_fcf + terminal_val

                # Net debt: (long-term debt + short-term debt) - cash
                # Proper enterprise-to-equity bridge avoids undervaluing
                # cash-rich companies (e.g. AAPL) in training data.
                cash = get_annual_as_of(facts, CASH_CONCEPTS, as_of_date)
                std  = get_annual_as_of(facts, SHORT_TERM_DEBT_CONCEPTS, as_of_date)
                total_debt = 0.0
                if not np.isnan(long_term_debt):
                    total_debt += long_term_debt
                if not np.isnan(std):
                    total_debt += std
                cash_val = cash if not np.isnan(cash) else 0.0
                net_debt = total_debt - cash_val

                if total_debt > 0 or cash_val > 0:
                    equity_val = max(intrinsic - net_debt, intrinsic * 0.1)
                else:
                    equity_val = intrinsic

                if equity_val > 0 and market_cap > 0:
                    dcf_implied_return = float(np.clip(equity_val / market_cap - 1, -0.99, 50.0))
        except Exception:
            pass

        # ── Piotroski F-Score ────────────────────────────────────────────────
        piotroski = compute_piotroski(
            facts, as_of_date,
            revenue_curr=revenue,
            net_income_curr=net_income,
            op_cf_curr=op_cf,
            assets_curr=total_assets,
            ltd_curr=long_term_debt,
            gross_profit_curr=gross_profit,
        )

        # ── Accruals Ratio (Sloan 1996) ──────────────────────────────────────
        accruals_ratio = nan
        if not np.isnan(net_income) and not np.isnan(op_cf) and not np.isnan(total_assets) and total_assets > 0:
            accruals_ratio = float(np.clip((net_income - op_cf) / total_assets, -1.0, 1.0))

        # ── Book-to-Market (Fama & French 1992) ─────────────────────────────
        book_to_market = nan
        if not np.isnan(equity) and not np.isnan(market_cap) and market_cap > 0:
            book_to_market = float(np.clip(equity / market_cap, -1.0, 10.0))

        # ── Asset Growth (Cooper et al. 2008) ───────────────────────────────
        ass_c_ag, ass_p_ag = get_two_annual_as_of(facts, ASSETS_CONCEPTS, as_of_date)
        asset_growth = nan
        if not np.isnan(ass_c_ag) and not np.isnan(ass_p_ag) and ass_p_ag > 0:
            asset_growth = float(np.clip((ass_c_ag - ass_p_ag) / ass_p_ag, -0.5, 5.0))

        # ── Net Issuance (Pontiff & Woodgate 2008) ──────────────────────────
        net_issuance = nan
        sh_entries_ni = _get_concept_entries_shares(facts, "CommonStockSharesOutstanding")
        sh_annual_ni = [
            x for x in sh_entries_ni
            if x.get("form") in ("10-K", "10-K/A")
            and x.get("filed", "9999") <= as_of_date.strftime("%Y-%m-%d")
            and "val" in x and x["val"] is not None
        ]
        sh_annual_ni.sort(key=lambda x: x["filed"], reverse=True)
        if len(sh_annual_ni) >= 2:
            _sh_c = float(sh_annual_ni[0]["val"])
            _sh_p = float(sh_annual_ni[1]["val"])
            if _sh_p > 0:
                net_issuance = float(np.clip((_sh_c - _sh_p) / _sh_p, -0.3, 1.0))

        # ── R&D Intensity (Chan et al. 2001) ─────────────────────────────────
        rd = get_annual_as_of(facts, RD_CONCEPTS, as_of_date)
        rd_intensity = nan
        if not np.isnan(rd) and not np.isnan(revenue) and revenue > 0:
            rd_intensity = float(np.clip(abs(rd) / revenue, 0.0, 1.0))

        # ── Log-compressed versions ──────────────────────────────────────────
        def log_val(x):
            if np.isnan(x): return nan
            return float(np.sign(x) * np.log1p(abs(x)))

        return {
            # Raw fundamentals
            "Revenue":           revenue,
            "EBITDA":            ebitda,
            "FCF":               fcf,
            "Profit_Margin":     profit_margin,
            "PE":                pe,
            "Debt_Equity":       debt_equity,
            "ROE":               roe,
            "Market_Cap":        market_cap,
            # Log-compressed (prevent outlier dominance)
            "Log_MarketCap":     log_val(market_cap),
            "Log_Revenue":       log_val(revenue),
            "Log_EBITDA":        log_val(ebitda),
            "Log_FCF":           log_val(fcf),
            # Quality metrics
            "FCF_Margin":              fcf_margin,
            "Earnings_Quality":        earnings_quality,
            "Revenue_Growth_Rate":     rev_growth,
            "Revenue_Growth_Accel":    rev_growth_accel,
            "Gross_Margin_Trend":      gross_margin_trend,
            "Piotroski_Score":         piotroski,
            # DCF
            "DCF_Implied_Return":      dcf_implied_return,
            # Fundamental features that cannot come from EDGAR
            # (analyst estimates, proxy data, market microstructure)
            # These stay as NaN for historical windows; the model handles
            # NaN via median imputation. The model learns these signals
            # only from current-score rows where yfinance provides them.
            "Earnings_Growth_Signal":  earnings_growth_signal,
            "Forward_PE":              nan,
            "PEG_Ratio":               nan,
            "Insider_Pct":             nan,
            "Inst_Pct":                nan,
            "Short_Interest_Ratio":    nan,
            # Cross-sectional predictors (computed from EDGAR)
            "Accruals_Ratio":          accruals_ratio,
            "Book_to_Market":          book_to_market,
            "Asset_Growth":            asset_growth,
            "Net_Issuance":            net_issuance,
            "RD_Intensity":            rd_intensity,
        }

    except Exception as e:
        log.debug(f"get_pit_fundamentals error at {as_of_date}: {e}")
        return _nan_fundamentals()


def _nan_fundamentals() -> dict:
    """Return dict of all fundamental features set to NaN."""
    keys = [
        "Revenue", "EBITDA", "FCF", "Profit_Margin", "PE", "Debt_Equity",
        "ROE", "Market_Cap", "Log_MarketCap", "Log_Revenue", "Log_EBITDA",
        "Log_FCF", "FCF_Margin", "Earnings_Quality", "Revenue_Growth_Rate",
        "Revenue_Growth_Accel", "Gross_Margin_Trend", "Piotroski_Score",
        "DCF_Implied_Return", "Earnings_Growth_Signal", "Forward_PE", "PEG_Ratio",
        "Insider_Pct", "Inst_Pct", "Short_Interest_Ratio",
        "Accruals_Ratio", "Book_to_Market", "Asset_Growth", "Net_Issuance",
        "RD_Intensity",
    ]
    return {k: np.nan for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# BULK DOWNLOAD HELPER  (used by cache_ohlc.py)
# ─────────────────────────────────────────────────────────────────────────────

def bulk_download_edgar_facts(
    tickers: list,
    cik_map: dict,
    cache_dir: str = EDGAR_CACHE_DIR,
    pause: float = FETCH_PAUSE,
) -> dict:
    """
    Download and cache companyfacts JSONs for a list of tickers.
    Skips tickers already cached. Returns {ticker: cik} for successful downloads.

    Uses threaded parallel downloads (BULK_WORKERS threads) with a shared
    rate limiter to stay under SEC EDGAR's 10 req/sec limit.
    """
    os.makedirs(cache_dir, exist_ok=True)
    results = {}
    missing_cik = []
    already_cached = 0

    # Separate already-cached from to-download
    to_download = []  # list of (ticker, cik) tuples
    for ticker in tickers:
        cik = cik_map.get(ticker.upper())
        if not cik:
            missing_cik.append(ticker)
            continue

        cache_path = os.path.join(cache_dir, f"{cik}.json")
        if os.path.exists(cache_path):
            results[ticker] = cik
            already_cached += 1
            continue

        to_download.append((ticker, cik))

    log.info(
        f"EDGAR: {already_cached} already cached, "
        f"{len(to_download)} to download, "
        f"{len(missing_cik)} tickers not in CIK map"
    )

    if not to_download:
        return results

    limiter = _RateLimiter(BULK_RPS)
    downloaded = 0
    failed = 0
    lock = threading.Lock()

    def _fetch_one(ticker_cik):
        nonlocal downloaded, failed
        ticker, cik = ticker_cik
        limiter.acquire()
        facts = download_companyfacts(cik, cache_dir)
        with lock:
            if facts is not None:
                results[ticker] = cik
                downloaded += 1
            else:
                failed += 1
            done = downloaded + failed
            if done % 100 == 0 and done > 0:
                log.info(
                    f"EDGAR progress: {downloaded} downloaded, "
                    f"{already_cached} cached, {failed} failed "
                    f"({done}/{len(to_download)})"
                )

    log.info(f"Starting {BULK_WORKERS}-thread parallel download for {len(to_download)} tickers...")
    with ThreadPoolExecutor(max_workers=BULK_WORKERS) as pool:
        futures = [pool.submit(_fetch_one, item) for item in to_download]
        for f in as_completed(futures):
            # Propagate any unexpected exceptions
            exc = f.exception()
            if exc:
                log.debug(f"EDGAR thread error: {exc}")

    log.info(
        f"EDGAR bulk download complete: {downloaded} new, "
        f"{already_cached} already cached, {failed} failed, "
        f"{len(missing_cik)} tickers not in CIK map"
    )
    if missing_cik[:10]:
        log.debug(f"  No CIK found for: {missing_cik[:10]}{'...' if len(missing_cik) > 10 else ''}")

    return results

    """
train_model.py — Walk-Forward LightGBM for 6-Month Stock Return Prediction
============================================================================
Builds a gradient-boosted tree model from historical cross-sectional data:
  - Price features from OHLC cache (3009 stocks, 1980–2026)
  - Point-in-time fundamentals from EDGAR XBRL facts (2812 companies, filed dates)
  - Macro regime features from VIX/TNX/IRX cache
  - Survivorship-corrected: dead stocks from delisted universe included

Walk-forward validation:
  - Expanding training window: all data up to month T
  - 6-month purge gap (prevents label leakage)
  - Test on month T+7 cross-section
  - Records out-of-sample predictions for every test window

Output:
  data/trained_model.pkl — dict containing:
    model:              trained LightGBM Booster
    feature_names:      list of feature column names
    oos_r2:             out-of-sample R² across all test windows
    oos_ic:             out-of-sample rank IC (Spearman)
    training_months:    number of months used for training
    feature_importance: dict of feature→importance
    trained_date:       ISO timestamp

Usage:
    python train_model.py          # full training (~15-30 min)
    python train_model.py --quick  # fast mode (fewer months, for testing)
"""

import os
import sys
import json
import pickle
import logging
import argparse
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
OHLC_PATH       = "data/raw_ohlc_cache.pkl"
CIK_MAP_PATH    = "data/edgar_cik_map.json"
EDGAR_DIR       = "data/edgar_facts"
DELISTED_PATH   = "data/delisted_universe.pkl"
DELISTED_OHLC   = "data/recovered_delisted_ohlc.pkl"
SECTOR_MAP_PATH = "raw_fetched_data.csv"   # current-universe sector mapping
OUTPUT_PATH     = "data/trained_model.pkl"

FORWARD_DAYS    = 126          # 6 months of trading days
PURGE_DAYS      = 252          # 1Y purge (6mo label horizon + 6mo safety)
EMBARGO_DAYS    = 21           # post-test embargo to decorrelate folds
MIN_HISTORY     = 252          # 1 year of history required for features
MIN_STOCKS      = 100          # minimum cross-section size to be useful
SHUMWAY_RETURN  = -0.30        # synthetic return for dead stocks (Shumway 1997)
USE_RANK_TARGET = True         # predict within-month rank, not raw return
NEUTRALIZE_FEATURES = True     # cross-sectional standardize each month

# LightGBM hyperparameters (tuned for cross-sectional return prediction)
LGB_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "num_leaves": 63,          # moderate complexity
    "learning_rate": 0.05,
    "feature_fraction": 0.7,   # column subsampling (decorrelates trees)
    "bagging_fraction": 0.8,   # row subsampling
    "bagging_freq": 5,
    "min_child_samples": 50,   # regularization: no small leaves
    "lambda_l1": 0.1,          # L1 regularization
    "lambda_l2": 1.0,          # L2 regularization
    "max_depth": 7,            # prevent overfitting
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}
NUM_BOOST_ROUNDS = 500
EARLY_STOPPING   = 50


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────

def load_ohlc_cache():
    """Load OHLC price cache → {ticker: pd.Series of closes, __VIX__: ..., etc.}"""
    with open(OHLC_PATH, "rb") as f:
        cache = pickle.load(f)
    stocks = {k: v for k, v in cache.items() if not k.startswith("__")}
    macro = {k: v for k, v in cache.items() if k.startswith("__")}
    log.info(f"OHLC cache: {len(stocks)} stocks, macro keys: {list(macro.keys())}")
    return stocks, macro


def load_cik_map():
    """Load ticker→CIK mapping."""
    with open(CIK_MAP_PATH) as f:
        return json.load(f)


def load_edgar_facts():
    """
    Return a lazy-loading proxy for EDGAR facts.
    Facts are loaded per-CIK on demand (7.5GB total is too large for RAM).
    Returns an object supporting dict-like access: facts[cik_str] → data.
    """
    class LazyEdgarFacts:
        def __init__(self, edgar_dir):
            self._dir = edgar_dir
            self._cache = {}
            self._available = set()
            if os.path.isdir(edgar_dir):
                for f in os.listdir(edgar_dir):
                    if f.endswith(".json"):
                        self._available.add(f.replace(".json", ""))
            log.info(f"EDGAR facts index: {len(self._available)} companies (lazy-loaded)")

        def __contains__(self, cik):
            return cik in self._available

        def __getitem__(self, cik):
            if cik in self._cache:
                return self._cache[cik]
            path = os.path.join(self._dir, f"{cik}.json")
            if not os.path.exists(path):
                raise KeyError(cik)
            with open(path) as f:
                data = json.load(f)
            # Cache up to 200 most recently accessed (LRU-ish)
            if len(self._cache) > 200:
                # Drop oldest half
                keys = list(self._cache.keys())
                for k in keys[:100]:
                    del self._cache[k]
            self._cache[cik] = data
            return data

        def get(self, cik, default=None):
            try:
                return self[cik]
            except KeyError:
                return default

    if not os.path.isdir(EDGAR_DIR):
        log.warning(f"EDGAR directory not found: {EDGAR_DIR}")
        return {}

    return LazyEdgarFacts(EDGAR_DIR)


def load_delisted():
    """Load recovered OHLC for dead stocks.

    Prefers the full delisted_universe.pkl produced by
    build_delisted_universe.py (contains a 'recovered_prices' dict), and
    falls back to the legacy recovered_delisted_ohlc.pkl if only that
    exists.
    """
    dead_ohlc = {}

    # Primary source: comprehensive universe build
    if os.path.exists(DELISTED_PATH):
        try:
            with open(DELISTED_PATH, "rb") as f:
                universe = pickle.load(f)
            if isinstance(universe, dict) and "recovered_prices" in universe:
                dead_ohlc = dict(universe["recovered_prices"])
        except Exception as e:
            log.warning(f"Could not read {DELISTED_PATH}: {e}")

    # Legacy fallback / merge — never overwrite what we already have
    if os.path.exists(DELISTED_OHLC):
        try:
            with open(DELISTED_OHLC, "rb") as f:
                legacy = pickle.load(f)
            for t, s in legacy.items():
                dead_ohlc.setdefault(t, s)
        except Exception as e:
            log.warning(f"Could not read {DELISTED_OHLC}: {e}")

    log.info(f"Delisted OHLC: {len(dead_ohlc)} dead stocks recovered")
    return dead_ohlc


def load_sector_map():
    """Load ticker→sector map from the current live-universe fetch.

    Used for sector-relative feature z-scoring during training. Dead
    stocks and anything missing a mapping fall back to ``"Unknown"`` —
    they still get z-scored (just within the Unknown bucket), which
    behaves like the old cross-sectional scheme for that subset.
    """
    sector_map: dict[str, str] = {}
    if os.path.exists(SECTOR_MAP_PATH):
        try:
            sdf = pd.read_csv(SECTOR_MAP_PATH, usecols=["Ticker", "Sector"])
            sdf = sdf.dropna(subset=["Ticker"])
            sdf["Sector"] = sdf["Sector"].fillna("Unknown").astype(str)
            sector_map = dict(zip(sdf["Ticker"].str.upper(), sdf["Sector"]))
            log.info(f"Sector map loaded: {len(sector_map)} tickers, "
                     f"{sdf['Sector'].nunique()} distinct sectors")
        except Exception as e:
            log.warning(f"Could not load sector map from {SECTOR_MAP_PATH}: {e}")
    else:
        log.warning(f"Sector map file not found ({SECTOR_MAP_PATH}); "
                    f"training will fall back to date-only z-scoring.")
    return sector_map


# ─────────────────────────────────────────────────────────────────────
# EDGAR POINT-IN-TIME FUNDAMENTAL EXTRACTION
# ─────────────────────────────────────────────────────────────────────

def _get_pit_value(facts_data, concept, unit_key, as_of_date, form_types=("10-K", "10-Q")):
    """
    Get the most recent point-in-time value for an EDGAR concept
    that was FILED on or before as_of_date.

    Returns (value, period_end_date) or (NaN, None).
    """
    us_gaap = facts_data.get("facts", {}).get("us-gaap", {})
    concept_data = us_gaap.get(concept, {})
    units = concept_data.get("units", {})
    records = units.get(unit_key, [])

    if not records:
        return np.nan, None

    as_of_str = as_of_date.strftime("%Y-%m-%d")

    # Filter to records filed before our scoring date, from 10-K or 10-Q
    valid = [
        r for r in records
        if r.get("filed", "9999") <= as_of_str
        and r.get("form") in form_types
        and "val" in r
    ]

    if not valid:
        return np.nan, None

    # Most recently filed
    valid.sort(key=lambda r: r.get("filed", ""), reverse=True)
    best = valid[0]
    return float(best["val"]), best.get("end")


def _get_pit_growth(facts_data, concept, unit_key, as_of_date):
    """
    Compute YoY growth for an EDGAR concept using point-in-time data.
    Finds the two most recent annual (10-K) filings before as_of_date.
    """
    us_gaap = facts_data.get("facts", {}).get("us-gaap", {})
    concept_data = us_gaap.get(concept, {})
    units = concept_data.get("units", {})
    records = units.get(unit_key, [])

    if not records:
        return np.nan

    as_of_str = as_of_date.strftime("%Y-%m-%d")

    annual = sorted(
        [r for r in records
         if r.get("filed", "9999") <= as_of_str
         and r.get("form") == "10-K"
         and "val" in r],
        key=lambda r: r.get("end", ""),
        reverse=True,
    )

    if len(annual) < 2:
        return np.nan

    new_val = float(annual[0]["val"])
    old_val = float(annual[1]["val"])

    if old_val == 0 or np.isnan(new_val) or np.isnan(old_val):
        return np.nan

    return float(np.clip((new_val - old_val) / abs(old_val), -2.0, 10.0))


def extract_edgar_fundamentals(facts_data, as_of_date):
    """
    Extract point-in-time fundamental features from EDGAR facts
    for a given scoring date.

    Returns dict of feature_name→value (NaN if unavailable).
    """
    result = {}

    # EPS growth
    result["EDGAR_EPS_Growth"] = _get_pit_growth(
        facts_data, "EarningsPerShareDiluted", "USD/shares", as_of_date
    )

    # Revenue growth
    result["EDGAR_Revenue_Growth"] = _get_pit_growth(
        facts_data, "Revenues", "USD", as_of_date
    )
    # Try alternative revenue concept
    if np.isnan(result["EDGAR_Revenue_Growth"]):
        result["EDGAR_Revenue_Growth"] = _get_pit_growth(
            facts_data, "RevenueFromContractWithCustomerExcludingAssessedTax", "USD", as_of_date
        )

    # Net income (for ROE, profit margin)
    ni, _ = _get_pit_value(facts_data, "NetIncomeLoss", "USD", as_of_date)
    rev, _ = _get_pit_value(facts_data, "Revenues", "USD", as_of_date)
    if np.isnan(rev):
        rev, _ = _get_pit_value(
            facts_data, "RevenueFromContractWithCustomerExcludingAssessedTax", "USD", as_of_date
        )
    equity, _ = _get_pit_value(facts_data, "StockholdersEquity", "USD", as_of_date)
    assets, _ = _get_pit_value(facts_data, "Assets", "USD", as_of_date)
    ltd, _ = _get_pit_value(facts_data, "LongTermDebt", "USD", as_of_date)
    opcf, _ = _get_pit_value(facts_data, "NetCashProvidedByOperatingActivities", "USD", as_of_date)

    # ROE
    if not np.isnan(ni) and not np.isnan(equity) and equity != 0:
        result["EDGAR_ROE"] = float(np.clip(ni / equity, -5.0, 5.0))
    else:
        result["EDGAR_ROE"] = np.nan

    # Profit margin
    if not np.isnan(ni) and not np.isnan(rev) and rev != 0:
        result["EDGAR_Profit_Margin"] = float(np.clip(ni / rev, -5.0, 5.0))
    else:
        result["EDGAR_Profit_Margin"] = np.nan

    # Debt to equity
    if not np.isnan(ltd) and not np.isnan(equity) and equity > 0:
        result["EDGAR_Debt_Equity"] = float(np.clip(ltd / equity, 0, 50.0))
    else:
        result["EDGAR_Debt_Equity"] = np.nan

    # ROA
    if not np.isnan(ni) and not np.isnan(assets) and assets > 0:
        result["EDGAR_ROA"] = float(np.clip(ni / assets, -2.0, 2.0))
    else:
        result["EDGAR_ROA"] = np.nan

    # Earnings quality: operating CF / net income
    if not np.isnan(opcf) and not np.isnan(ni) and ni != 0:
        result["EDGAR_Earnings_Quality"] = float(np.clip(opcf / ni, -5.0, 10.0))
    else:
        result["EDGAR_Earnings_Quality"] = np.nan

    # Accruals ratio: (NI - OpCF) / Assets
    if not np.isnan(ni) and not np.isnan(opcf) and not np.isnan(assets) and assets > 0:
        result["EDGAR_Accruals"] = float(np.clip((ni - opcf) / assets, -1.0, 1.0))
    else:
        result["EDGAR_Accruals"] = np.nan

    return result


# ─────────────────────────────────────────────────────────────────────
# PRICE FEATURE COMPUTATION (VECTORIZED)
# ─────────────────────────────────────────────────────────────────────

def compute_all_features_vectorized(close: pd.Series) -> pd.DataFrame:
    """
    Compute ALL price-based features for a stock's entire history at once
    using vectorized rolling operations. Returns a DataFrame indexed by
    date with one row per trading day.

    This is ~100x faster than calling compute_price_features() per-date
    because pandas rolling operations are C-optimized.
    """
    # Safety: ensure we have a Series
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    n = len(close)
    if n < MIN_HISTORY:
        return pd.DataFrame()

    daily_ret = close.pct_change()
    feat = pd.DataFrame(index=close.index)

    # Returns: close[t] / close[t-days] - 1
    for days, name in [(21, "Return_1mo"), (63, "Return_3mo"),
                        (126, "Return_6mo"), (252, "Return_1Y"),
                        (756, "Return_3Y")]:
        feat[name] = close / close.shift(days) - 1

    # Jegadeesh-Titman 12-2mo momentum: close[-21] / close[-252] - 1
    feat["Return_2_12mo"] = close.shift(21) / close.shift(252) - 1

    # Volatility (annualized)
    feat["Vol_1Y"] = daily_ret.rolling(252).std() * np.sqrt(252)
    feat["Vol_3Y"] = daily_ret.rolling(756).std() * np.sqrt(252)

    # Max drawdown 1Y (rolling)
    rolling_max_1y = close.rolling(252).max()
    feat["Max_Drawdown_1Y"] = close / rolling_max_1y - 1

    # Sharpe ratio 1Y (rolling)
    _rf = 0.0002  # approximate daily risk-free
    excess = daily_ret - _rf
    roll_mean = excess.rolling(252).mean()
    roll_std = excess.rolling(252).std()
    feat["Sharpe_1Y"] = (roll_mean / roll_std) * np.sqrt(252)

    # Price vs SMAs
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    feat["Price_vs_50SMA"] = close / sma50 - 1
    feat["Price_vs_200SMA"] = close / sma200 - 1

    # Price vs 52-week high/low
    hi52 = close.rolling(252).max()
    lo52 = close.rolling(252).min()
    feat["Price_vs_52WHigh"] = close / hi52 - 1
    feat["Price_vs_52WLow"] = close / lo52 - 1

    # Return consistency: std of 4 non-overlapping 126-day returns
    r_seg1 = close / close.shift(126) - 1  # most recent 6mo
    r_seg2 = close.shift(126) / close.shift(252) - 1
    r_seg3 = close.shift(252) / close.shift(378) - 1
    r_seg4 = close.shift(378) / close.shift(504) - 1
    seg_df = pd.concat([r_seg1, r_seg2, r_seg3, r_seg4], axis=1)
    seg_std = seg_df.std(axis=1)
    feat["Return_Consistency"] = 1.0 / (1.0 + seg_std)

    # Return decay (acceleration)
    r1y = feat["Return_1Y"]
    r3y = feat["Return_3Y"]
    r3y_ann = (1 + r3y).pow(1/3) - 1
    feat["Return_Decay"] = r1y - r3y_ann

    # Forward 6-month return (LABEL) — computed from future data
    feat["Forward_6mo_Return"] = close.shift(-FORWARD_DAYS) / close - 1

    return feat


# ─────────────────────────────────────────────────────────────────────
# DATASET CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────

def build_training_dataset(
    stocks: dict,
    macro: dict,
    edgar_facts: dict,
    cik_map: dict,
    dead_ohlc: dict,
    sector_map: dict | None = None,
    start_year: int = 2012,
    end_year: int = 2025,
):
    """
    Build the full training dataset using VECTORIZED feature pre-computation.

    Phase 1: Pre-compute all price features for every stock's full history
             at once using rolling windows (~30s for 3000 stocks).
    Phase 2: For each monthly scoring date, sample the pre-computed row
             for each stock (~instant per month).
    Phase 3: Add EDGAR PIT fundamentals and macro features.

    Returns pd.DataFrame with ~500K rows.
    """

    # Build reverse CIK map: ticker → CIK string (zero-padded)
    ticker_to_cik = {}
    for ticker, cik in cik_map.items():
        cik_padded = str(cik).zfill(10)
        ticker_to_cik[ticker.upper()] = cik_padded

    # Pre-process: convert all stock close series to timezone-naive DatetimeIndex
    # Some cache entries may be DataFrames — squeeze to Series
    def _to_series(obj):
        if isinstance(obj, pd.DataFrame):
            if obj.shape[1] == 1:
                return obj.iloc[:, 0]
            elif 'Close' in obj.columns:
                return obj['Close']
            elif 'Adj Close' in obj.columns:
                return obj['Adj Close']
            else:
                return obj.iloc[:, 0]
        return obj

    processed_stocks = {}
    for tk, series in stocks.items():
        try:
            s = _to_series(series)
            idx = pd.to_datetime(s.index)
            if hasattr(idx, "tz") and idx.tz is not None:
                idx = idx.tz_localize(None)
            s = s.copy()
            s.index = idx
            s = s.sort_index()
            processed_stocks[tk] = s
        except Exception:
            continue

    # Process dead stocks the same way
    processed_dead = {}
    for tk, series in dead_ohlc.items():
        try:
            s = _to_series(series)
            idx = pd.to_datetime(s.index)
            if hasattr(idx, "tz") and idx.tz is not None:
                idx = idx.tz_localize(None)
            s = s.copy()
            s.index = idx
            s = s.sort_index()
            processed_dead[tk] = s
        except Exception:
            continue

    # Macro series (VIX, TNX, IRX) — timezone-naive
    macro_processed = {}
    for key, series in macro.items():
        try:
            idx = pd.to_datetime(series.index)
            if hasattr(idx, "tz") and idx.tz is not None:
                idx = idx.tz_localize(None)
            s = series.copy()
            s.index = idx
            macro_processed[key] = s.sort_index()
        except Exception:
            continue

    # SPY for market regime
    spy_close = processed_stocks.get("SPY")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Vectorized feature pre-computation for ALL stocks
    # ══════════════════════════════════════════════════════════════════
    log.info("Phase 1: Pre-computing price features for all stocks...")
    t_phase1 = datetime.now()

    stock_features = {}  # ticker → DataFrame of features indexed by date
    all_stocks_combined = {**processed_stocks, **processed_dead}

    for idx_i, (tk, close) in enumerate(all_stocks_combined.items()):
        if tk.startswith("__"):  # skip macro keys
            continue
        feat_df = compute_all_features_vectorized(close)
        if len(feat_df) > 0:
            stock_features[tk] = feat_df
        if (idx_i + 1) % 500 == 0:
            log.info(f"  ...computed features for {idx_i + 1}/{len(all_stocks_combined)} stocks")

    elapsed1 = (datetime.now() - t_phase1).total_seconds()
    log.info(f"Phase 1 complete: {len(stock_features)} stocks in {elapsed1:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1.5: Batch-extract EDGAR fundamentals for all tickers/dates
    # Load each JSON file exactly ONCE, extract features for all scoring dates
    # ══════════════════════════════════════════════════════════════════
    scoring_dates = pd.date_range(
        start=f"{start_year}-01-31",
        end=f"{end_year}-12-31",
        freq="ME",
    )
    scoring_dts = [sd.to_pydatetime() for sd in scoring_dates]

    log.info("Phase 1.5: Batch-extracting EDGAR fundamentals...")
    t_phase15 = datetime.now()

    # edgar_cache[ticker][score_date_str] → dict of EDGAR features
    edgar_cache = {}
    n_loaded = 0
    for tk in stock_features:
        cik = ticker_to_cik.get(tk.upper())
        if not cik or cik not in edgar_facts:
            continue
        # Load JSON once (triggers LazyEdgarFacts.__getitem__)
        try:
            facts = edgar_facts[cik]
        except Exception:
            continue
        n_loaded += 1
        tk_cache = {}
        for sd in scoring_dts:
            tk_cache[sd] = extract_edgar_fundamentals(facts, sd)
        edgar_cache[tk] = tk_cache
        if n_loaded % 200 == 0:
            log.info(f"  ...extracted EDGAR for {n_loaded} tickers")

    elapsed15 = (datetime.now() - t_phase15).total_seconds()
    log.info(f"Phase 1.5 complete: {n_loaded} tickers with EDGAR data in {elapsed15:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Sample features at monthly scoring dates
    # ══════════════════════════════════════════════════════════════════
    log.info(f"Phase 2: Sampling {len(scoring_dates)} monthly cross-sections, "
             f"{start_year}-{end_year}")

    # Determine which tickers are dead vs live
    dead_tickers = set(processed_dead.keys())

    all_rows = []

    for i, score_date in enumerate(scoring_dates):
        t0 = datetime.now()
        score_dt = score_date.to_pydatetime()
        month_rows = []

        # ── Macro features for this date ──────────────────────────────
        macro_feat = {}
        for key, label in [("__VIX__", "VIX_Level"),
                           ("__TNX__", "TNX_Level"),
                           ("__IRX__", "IRX_Level")]:
            ms = macro_processed.get(key)
            if ms is not None:
                before = ms[ms.index <= score_date]
                if len(before) > 0:
                    macro_feat[label] = float(before.iloc[-1])
                else:
                    macro_feat[label] = np.nan
            else:
                macro_feat[label] = np.nan

        # Yield spread
        tnx = macro_feat.get("TNX_Level", np.nan)
        irx = macro_feat.get("IRX_Level", np.nan)
        if not np.isnan(tnx) and not np.isnan(irx):
            macro_feat["Yield_Spread"] = tnx - irx
        else:
            macro_feat["Yield_Spread"] = np.nan

        # SPY regime
        if spy_close is not None:
            spy_before = spy_close[spy_close.index <= score_date]
            if len(spy_before) >= 252:
                spy_r1y = float(spy_before.iloc[-1] / spy_before.iloc[-252] - 1)
                spy_v1y = float(spy_before.pct_change().iloc[-252:].std() * np.sqrt(252))
                spy_ath = float(spy_before.iloc[-252:].max())
                spy_200 = float(spy_before.rolling(200).mean().iloc[-1])
                macro_feat["SPY_Return_1Y"] = spy_r1y
                macro_feat["SPY_Vol_1Y"] = spy_v1y
                macro_feat["SPY_ATH_DD"] = float(spy_before.iloc[-1] / spy_ath - 1) if spy_ath > 0 else np.nan
                macro_feat["SPY_200SMA_Pct"] = float(spy_before.iloc[-1] / spy_200 - 1) if spy_200 > 0 else np.nan

        # ── Sample pre-computed features for each stock ───────────────
        for tk, feat_df in stock_features.items():
            if tk == "SPY":
                continue

            is_dead = tk in dead_tickers

            # Use searchsorted for O(log n) date lookup instead of O(n) boolean mask
            feat_idx = feat_df.index
            pos = feat_idx.searchsorted(score_date, side="right") - 1
            if pos < 0:
                continue

            nearest_date = feat_idx[pos]
            # Must be within 5 trading days of scoring date
            if (score_date - nearest_date).days > 7:
                continue

            # Check minimum history via position in the feature index
            # (features require MIN_HISTORY days to compute, so if pos >= 0
            # the stock has enough history — the vectorized function already
            # returns empty DataFrame if < MIN_HISTORY)

            row = feat_df.iloc[pos]
            feat = row.to_dict()

            # Forward return
            fwd_return = feat.pop("Forward_6mo_Return", np.nan)

            if is_dead:
                # Dead stock: check if still alive at scoring date
                close_series = all_stocks_combined[tk]
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
                last_date = close_series.index.max()
                if last_date < score_date - timedelta(days=180):
                    continue

                # Use actual forward return if available, else Shumway
                if np.isnan(fwd_return):
                    after = close_series[close_series.index > score_date]
                    if len(after) > 0:
                        before_price = float(close_series[close_series.index <= score_date].iloc[-1])
                        partial_ret = float(after.iloc[-1]) / before_price - 1
                        fwd_return = min(partial_ret, SHUMWAY_RETURN)
                    else:
                        fwd_return = SHUMWAY_RETURN
            else:
                # Live stock: must have forward return to be useful
                if np.isnan(fwd_return):
                    continue
                fwd_return = np.clip(fwd_return, -1.0, 10.0)

            feat["Forward_6mo_Return"] = fwd_return
            feat["Ticker"] = tk
            feat["Score_Date"] = score_date
            feat["Is_Dead"] = 1 if is_dead else 0
            feat["Sector"] = (sector_map.get(tk.upper(), "Unknown")
                               if sector_map else "Unknown")

            # EDGAR fundamentals (from pre-computed cache)
            if tk in edgar_cache:
                feat.update(edgar_cache[tk].get(score_dt, {}))

            # Macro features
            feat.update(macro_feat)

            month_rows.append(feat)

        if len(month_rows) >= MIN_STOCKS:
            all_rows.extend(month_rows)

        elapsed = (datetime.now() - t0).total_seconds()
        n_dead = sum(1 for r in month_rows if r.get("Is_Dead", 0) == 1)
        log.info(f"  [{i+1}/{len(scoring_dates)}] {score_date.strftime('%Y-%m')}: "
                 f"{len(month_rows)} stocks ({n_dead} dead), "
                 f"{elapsed:.1f}s, cumulative {len(all_rows)} rows")

    df = pd.DataFrame(all_rows)
    log.info(f"Dataset built: {len(df)} rows, {len(df.columns)} columns")
    log.info(f"  Dead stock fraction: {df['Is_Dead'].mean():.1%}")
    log.info(f"  Forward return stats: mean={df['Forward_6mo_Return'].mean():.3f}, "
             f"median={df['Forward_6mo_Return'].median():.3f}, "
             f"std={df['Forward_6mo_Return'].std():.3f}")

    # ── Cross-sectional rank target ──────────────────────────────────
    # Raw 6-month return is dominated by regime/market noise at the
    # horizon scale we care about. Within-month rank (0..1) removes that
    # common-factor component and lets the model focus on the stuff it
    # can actually learn: which stocks beat their peers in each window.
    df["Forward_Rank"] = (
        df.groupby("Score_Date")["Forward_6mo_Return"]
          .rank(pct=True, method="average")
    )

    # ── Cross-sectional feature neutralization ───────────────────────
    # For each Score_Date, z-score every numeric feature across stocks.
    # This removes month-specific scale/level effects (e.g. vol spikes
    # during 2020-03, yield regime shifts) and lets the model learn
    # relative positioning rather than absolute levels. Macro features
    # (VIX_Level, TNX_Level, etc.) are *intentionally* excluded so the
    # model can still condition on regime.
    #
    # NOTE: sector-relative z-scoring was tested and REDUCED OOS IC
    # (0.053 → 0.038). Within-sector variance is smaller and noisier,
    # and the best-performing sectors (banks, energy) lost their
    # legitimate cross-sector premium. Reverted to date-only.
    if NEUTRALIZE_FEATURES:
        macro_keep = {"VIX_Level", "TNX_Level", "IRX_Level", "Yield_Spread"}
        meta = {"Ticker", "Score_Date", "Sector", "Forward_6mo_Return",
                "Forward_Rank", "Is_Dead"}
        feat_cols = [c for c in df.columns
                     if c not in meta and c not in macro_keep
                     and pd.api.types.is_numeric_dtype(df[c])]
        log.info(f"Cross-sectionally neutralizing {len(feat_cols)} features (date-only)...")

        def _zscore(g):
            sd = float(g.std(ddof=0))
            if sd == 0 or np.isnan(sd):
                return g - g.mean()
            return (g - g.mean()) / sd

        df[feat_cols] = (
            df.groupby("Score_Date")[feat_cols]
              .transform(_zscore)
        )
        # Sanitize: replace inf with NaN, then fill NaN with 0 (neutral)
        df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


# ─────────────────────────────────────────────────────────────────────
# WALK-FORWARD TRAINING
# ─────────────────────────────────────────────────────────────────────

def get_feature_columns(df):
    """Return list of feature columns (everything except metadata and label).

    Raw volatility features (Vol_1Y, Vol_3Y) are *excluded*: they dominate
    feature importance and turn the model into a low-volatility factor
    proxy, which performs poorly in growth-led bull markets (2023–2025).
    Risk-adjusted features (Max_Drawdown_1Y, Sharpe_1Y) stay, since they
    combine vol with return and are not pure-risk sorts.
    """
    exclude = {"Ticker", "Score_Date", "Sector",
               "Forward_6mo_Return", "Forward_Rank", "Is_Dead",
               "Vol_1Y", "Vol_3Y"}
    return [c for c in df.columns if c not in exclude]


def walk_forward_train(df: pd.DataFrame, min_train_months: int = 36):
    """
    Walk-forward expanding-window training with LightGBM.

    For each test month T (starting after min_train_months):
      - Train on data with Score_Date <= T - PURGE_DAYS (no label overlap)
      - Also embargo EMBARGO_DAYS on either side of the test window
      - Test on data with Score_Date == T
      - Record out-of-sample predictions

    Target: Forward_Rank (within-month percentile rank) if USE_RANK_TARGET
    is enabled, otherwise raw Forward_6mo_Return. Rank target is strongly
    preferred — it removes common-factor regime noise and typically lifts
    monthly IC by 3–5x on survivors-heavy universes.

    Returns:
      model:       final trained LightGBM model (on all data)
      oos_preds:   DataFrame with Ticker, Score_Date, y_true, y_pred
      metrics:     dict of performance metrics
    """
    target_col = "Forward_Rank" if (USE_RANK_TARGET and "Forward_Rank" in df.columns) \
                 else "Forward_6mo_Return"
    log.info(f"Training target: {target_col}  "
             f"(purge={PURGE_DAYS}d, embargo={EMBARGO_DAYS}d)")

    feature_cols = get_feature_columns(df)
    log.info(f"Features ({len(feature_cols)}): {feature_cols}")

    # Sort by date
    df = df.sort_values("Score_Date").reset_index(drop=True)
    all_dates = sorted(df["Score_Date"].unique())

    if len(all_dates) < min_train_months + 7:
        raise ValueError(f"Not enough data: {len(all_dates)} months, "
                         f"need at least {min_train_months + 7}")

    # Walk-forward: test each month after the initial training window
    test_dates = all_dates[min_train_months:]
    oos_records = []
    fold_metrics = []

    log.info(f"Walk-forward: {len(test_dates)} test months, "
             f"starting from {test_dates[0]}")

    for i, test_date in enumerate(test_dates):
        # Training data: strict purge on each side.
        # Upper bound: any Score_Date within PURGE_DAYS *before* the test
        # month has a forward-return window that overlaps the test-period
        # features. Exclude.
        # Lower bound (embargo): optionally drop the last EMBARGO_DAYS of
        # pre-purge training to further decorrelate folds.
        purge_cutoff = test_date - pd.Timedelta(days=PURGE_DAYS)
        train_mask = df["Score_Date"] <= purge_cutoff
        if EMBARGO_DAYS > 0:
            embargo_cutoff = purge_cutoff - pd.Timedelta(days=EMBARGO_DAYS)
            # keep train rows older than embargo_cutoff OR in the purged
            # region (we've already excluded the latter above)
            train_mask = df["Score_Date"] <= embargo_cutoff
        test_mask = df["Score_Date"] == test_date

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) < 1000 or len(test_df) < MIN_STOCKS:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        y_test_raw = test_df["Forward_6mo_Return"]

        # Use last 20% of training data as validation for early stopping
        val_cutoff = train_df["Score_Date"].quantile(0.8)
        val_mask = train_df["Score_Date"] > val_cutoff
        if val_mask.sum() < 100:
            # Not enough validation data — use last 6 months
            val_cutoff = train_df["Score_Date"].max() - pd.Timedelta(days=180)
            val_mask = train_df["Score_Date"] > val_cutoff

        X_val = X_train[val_mask]
        y_val = y_train[val_mask]
        X_train_fold = X_train[~val_mask]
        y_train_fold = y_train[~val_mask]

        # Train LightGBM
        dtrain = lgb.Dataset(X_train_fold, label=y_train_fold)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [lgb.early_stopping(EARLY_STOPPING, verbose=False)]
        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUNDS,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        # Predict
        y_pred_lgb = model.predict(X_test)

        # NOTE: ElasticNet co-model was tested at 50/50 rank blend but dropped
        # OOS IC from 0.053 → 0.035 and hit rate from 75% → 70%. EN is too weak
        # to help here. Using LightGBM-only predictions; ElasticNet fit below
        # is kept for diagnostics only (not used for OOS IC).
        y_pred = y_pred_lgb

        # Record OOS predictions — store raw forward return as y_true so
        # IC is measured against the *actual* portfolio objective, not
        # the proxy rank target we trained on.
        for j, (idx, row) in enumerate(test_df.iterrows()):
            oos_records.append({
                "Ticker": row["Ticker"],
                "Score_Date": test_date,
                "y_true": y_test_raw.iloc[j],
                "y_pred": y_pred[j],
            })

        # Month-level metrics: IC measured against raw return
        rho, _ = stats.spearmanr(y_test_raw, y_pred)
        ss_res = np.sum((y_test_raw.values - y_pred) ** 2)
        ss_tot = np.sum((y_test_raw.values - y_test_raw.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        fold_metrics.append({
            "date": test_date,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "r2": r2,
            "ic": rho,
            "n_trees": model.num_trees(),
        })

        if (i + 1) % 12 == 0:
            recent = fold_metrics[-12:]
            avg_ic = np.mean([m["ic"] for m in recent])
            avg_r2 = np.mean([m["r2"] for m in recent])
            log.info(f"  Month {i+1}/{len(test_dates)} ({test_date.strftime('%Y-%m')}): "
                     f"last-12mo avg IC={avg_ic:.3f}, R²={avg_r2:.4f}")

    # ── Aggregate OOS metrics ─────────────────────────────────────────
    oos_df = pd.DataFrame(oos_records)
    overall_rho, _ = stats.spearmanr(oos_df["y_true"], oos_df["y_pred"])
    ss_res = np.sum((oos_df["y_true"].values - oos_df["y_pred"].values) ** 2)
    ss_tot = np.sum((oos_df["y_true"].values - oos_df["y_true"].mean()) ** 2)
    overall_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Monthly IC series
    monthly_ic = oos_df.groupby("Score_Date").apply(
        lambda g: stats.spearmanr(g["y_true"], g["y_pred"])[0]
    )

    log.info("=" * 60)
    log.info("WALK-FORWARD OUT-OF-SAMPLE RESULTS")
    log.info(f"  Total OOS predictions: {len(oos_df)}")
    log.info(f"  Overall Spearman IC:   {overall_rho:.4f}")
    log.info(f"  Overall R²:            {overall_r2:.4f}")
    log.info(f"  Monthly IC mean:       {monthly_ic.mean():.4f}")
    log.info(f"  Monthly IC std:        {monthly_ic.std():.4f}")
    log.info(f"  Hit rate (IC > 0):     {(monthly_ic > 0).mean():.1%}")

    # ── Survivorship-correction coverage report ──────────────────────
    # By year, how many dead-stock rows actually made it into training.
    # If pre-2015 windows have ~0 dead rows, the survivorship correction
    # is effectively inactive for those test folds.
    if "Is_Dead" in df.columns:
        dead_by_year = (
            df.assign(_year=df["Score_Date"].dt.year)
              .groupby("_year")
              .agg(total=("Ticker", "size"), dead=("Is_Dead", "sum"))
        )
        dead_by_year["dead_pct"] = dead_by_year["dead"] / dead_by_year["total"]
        log.info("Dead-stock coverage by year (survivorship-correction health):")
        low_cov_years = []
        for yr, r in dead_by_year.iterrows():
            flag = ""
            if r["dead_pct"] < 0.02:
                flag = "  ⚠ LOW"
                low_cov_years.append(int(yr))
            log.info(f"  {int(yr)}: {int(r['total']):>5} rows, "
                     f"{int(r['dead']):>3} dead ({r['dead_pct']:.1%}){flag}")
        if low_cov_years:
            log.warning(
                f"Years {low_cov_years} have <2% dead stocks — survivorship "
                "correction is weak for those windows. Expanding the delisted "
                "universe (build_delisted_universe.py) would improve it."
            )
    log.info("=" * 60)

    # ── Train final model on ALL data ─────────────────────────────────
    log.info("Training final model on all available data...")
    X_all = df[feature_cols]
    y_all = df[target_col]

    # 80/20 split for early stopping (by time)
    val_cutoff = df["Score_Date"].quantile(0.85)
    val_mask = df["Score_Date"] > val_cutoff

    dtrain = lgb.Dataset(X_all[~val_mask], label=y_all[~val_mask])
    dval = lgb.Dataset(X_all[val_mask], label=y_all[val_mask], reference=dtrain)

    callbacks = [lgb.early_stopping(EARLY_STOPPING, verbose=False)]
    final_model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    # ElasticNet ensemble disabled (hurt OOS IC 0.053 → 0.035 in testing).
    final_en = None

    # Feature importance
    importance = dict(zip(
        feature_cols,
        final_model.feature_importance(importance_type="gain"),
    ))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    log.info("Top 15 features by importance:")
    for feat_name, imp in sorted_imp[:15]:
        log.info(f"  {feat_name:<25} {imp:.0f}")

    metrics = {
        "oos_r2": overall_r2,
        "oos_ic": overall_rho,
        "monthly_ic_mean": float(monthly_ic.mean()),
        "monthly_ic_std": float(monthly_ic.std()),
        "ic_hit_rate": float((monthly_ic > 0).mean()),
        "n_oos_predictions": len(oos_df),
        "n_test_months": len(test_dates),
        "fold_metrics": fold_metrics,
    }

    return final_model, final_en, feature_cols, oos_df, metrics, importance


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train walk-forward LightGBM model")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer years, for testing")
    parser.add_argument("--start-year", type=int, default=2012,
                        help="First year of training data (default: 2012)")
    parser.add_argument("--end-year", type=int, default=2025,
                        help="Last year of training data (default: 2025)")
    args = parser.parse_args()

    if args.quick:
        args.start_year = max(args.start_year, 2020)
        log.info("Quick mode: using 2020-2025 only")

    # ── Load data ─────────────────────────────────────────────────────
    log.info("Loading data sources...")
    stocks, macro = load_ohlc_cache()
    cik_map = load_cik_map()
    edgar_facts = load_edgar_facts()
    dead_ohlc = load_delisted()
    sector_map = load_sector_map()

    # ── Build training dataset ────────────────────────────────────────
    df = build_training_dataset(
        stocks, macro, edgar_facts, cik_map, dead_ohlc,
        sector_map=sector_map,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    if len(df) < 5000:
        log.error(f"Dataset too small ({len(df)} rows). Need at least 5000.")
        sys.exit(1)

    # ── Walk-forward training ─────────────────────────────────────────
    min_months = 24 if args.quick else 36
    model, en_model, feature_cols, oos_df, metrics, importance = walk_forward_train(
        df, min_train_months=min_months
    )

    # ── Save ──────────────────────────────────────────────────────────
    # Identify which features were cross-sectionally neutralized during
    # training so inference can replay the exact same transform.
    macro_keep = {"VIX_Level", "TNX_Level", "IRX_Level", "Yield_Spread"}
    neutralized_cols = (
        [c for c in feature_cols if c not in macro_keep]
        if NEUTRALIZE_FEATURES else []
    )
    output = {
        "model": model,
        "elasticnet_model": en_model,
        "feature_names": feature_cols,
        "neutralized_features": neutralized_cols,
        "macro_features": sorted(macro_keep & set(feature_cols)),
        "target": "Forward_Rank" if USE_RANK_TARGET else "Forward_6mo_Return",
        "sector_neutralized": False,
        "ensemble": en_model is not None,
        "purge_days": PURGE_DAYS,
        "embargo_days": EMBARGO_DAYS,
        "oos_r2": metrics["oos_r2"],
        "oos_ic": metrics["oos_ic"],
        "monthly_ic_mean": metrics["monthly_ic_mean"],
        "ic_hit_rate": metrics["ic_hit_rate"],
        "training_months": metrics["n_test_months"],
        "feature_importance": importance,
        "metrics": metrics,
        "trained_date": datetime.now().isoformat(),
        "start_year": args.start_year,
        "end_year": args.end_year,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(f"Model saved to {OUTPUT_PATH}")
    log.info(f"  OOS Spearman IC: {metrics['oos_ic']:.4f}")
    log.info(f"  OOS R²:          {metrics['oos_r2']:.4f}")
    log.info(f"  IC hit rate:     {metrics['ic_hit_rate']:.1%}")

    # ── Save OOS predictions for analysis ─────────────────────────────
    oos_path = "data/oos_predictions.csv"
    oos_df.to_csv(oos_path, index=False)
    log.info(f"OOS predictions saved to {oos_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()

        """
main.py — Stock Selection & Capital Allocation
===============================================
Selects the top 50 stocks predicted to maximize 6-month alpha over SPY.

Pipeline
--------
1. Load tickers from data/tickers.csv
2. Fetch price history + fundamentals via yfinance (24h cache)
3. Filter universe: exclude non-operating companies, extreme leverage,
   short history (<3Y), small caps (<$1B), illiquid stocks
4. Score stocks using:
   a) 9 empirically-validated factor signals (survivorship-corrected)
   b) LightGBM walk-forward ML model (if trained via train_model.py)
   c) Ensemble blend: α×ML + (1-α)×factor, where α adapts to OOS performance
5. Allocate capital with data-quality adjustments, sector caps, position limits
6. Export final_portfolio.xlsx (4 sheets) + CSVs + Monte Carlo analysis

Factor weights are derived from Spearman rank correlations measured over
10,701 samples (7,413 live + 1,388 recovered dead + 950 synthetic dead)
across 29 annual test dates from 1996-2025. Key finding: the apparent
'volatility premium' (high vol → high returns) was 100% survivorship bias.
After correction, LOW volatility predicts higher returns (rho = -0.197).

ML layer (optional): LightGBM trained via train_model.py on historical
cross-sections from the OHLC cache + EDGAR point-in-time fundamentals.
Walk-forward validated with 6-month purge gap. Captures non-linear
interactions and regime-dependent effects that fixed-weight factors miss.
"""

import os
import time
import json
import logging
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

warnings.filterwarnings("ignore")
import sys as _sys

_stream_handler = logging.StreamHandler(_sys.stdout)
_stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
# Force UTF-8 on Windows consoles so Unicode chars don't crash
if hasattr(_stream_handler.stream, "reconfigure"):
    try:
        _stream_handler.stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        _stream_handler,
        logging.FileHandler("portfolio_run.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
TOTAL_CAPITAL       = 10_000      # USD
MIN_HISTORY_DAYS    = 126         # ~6 months
MIN_MARKET_CAP      = 500_000_000 # $500M
MIN_ALLOCATION      = 25          # discard allocations below $25
MAX_ALLOC_FRAC      = 0.15        # no single stock > 15%
TOP_N               = 50          # final portfolio size
FETCH_PAUSE         = 0.25        # seconds between yfinance calls; period="max" is heavy,
                                  # 0.08 triggered Yahoo rate-limiting after ~270 tickers
RANDOM_STATE        = 42

# -- Prediction horizon --
# Academic factor evidence is strongest at 3-12 month horizons.
# A 6-month horizon captures momentum and mean-reversion signals
# with better signal-to-noise than 1Y, and allows 2x more backtest periods.
PREDICTION_YEARS    = 0.5

# -- Position sizing method --
# EQUAL_WEIGHT = True:  every stock in the top-N gets the same dollar amount.
#   With noisy return predictions, allocating proportionally assumes
#   a precision the model may not have. Equal weighting is more robust when
#   ranking is reliable but magnitude estimates are noisy.
# EQUAL_WEIGHT = False: weight by adjusted composite score (original behaviour).
#   Captures more upside from top picks IF magnitude differences are meaningful.
# Both methods still apply the sector cap and per-stock MAX_ALLOC_FRAC limits.
EQUAL_WEIGHT        = False       # set True to test equal-weight allocation

# -- Allocation method --
# "composite": weight by adjusted composite score (default)
# "equal":     equal weight all positions
# "hrp":       Hierarchical Risk Parity (covariance-aware, diversification-maximising)
ALLOCATION_METHOD   = "composite"  # "composite" | "equal" | "hrp"

UNINVESTABLE = {
    "BRK-A","BRK/A","BRK.A",  # ~$700k/share
}

# Prediction horizon in trading days (6 months)
PREDICTION_WINDOW  = int(PREDICTION_YEARS * 252)

# ---------------------------------------------
# STEP 0 - AUTO-DOWNLOAD TICKERS
# ---------------------------------------------
def download_tickers(file_path: str) -> bool:
    """Download the full stock screener CSV from NASDAQ's public API.
    Returns True on success, False on failure."""
    url = ("https://api.nasdaq.com/api/screener/stocks"
           "?tableonly=true&limit=25000&offset=0&download=true")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    log.info("Downloading latest ticker universe from NASDAQ screener API...")
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rows = data["data"]["rows"]
        if not rows:
            log.error("NASDAQ API returned no rows.")
            return False
        # Map API field names to the CSV column names load_tickers() expects
        col_map = {
            "symbol": "Symbol", "name": "Name", "lastsale": "Last Sale",
            "netchange": "Net Change", "pctchange": "% Change",
            "marketCap": "Market Cap", "country": "Country",
            "ipoyear": "IPO Year", "volume": "Volume",
            "sector": "Sector", "industry": "Industry",
        }
        df = pd.DataFrame(rows)
        df = df.rename(columns=col_map)
        # Keep only the expected columns (drop 'url' etc.)
        keep = [c for c in col_map.values() if c in df.columns]
        df = df[keep]
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        log.info(f"Downloaded {len(df)} tickers → {file_path}")
        return True
    except (URLError, HTTPError, KeyError, json.JSONDecodeError) as e:
        log.error(f"Failed to download tickers from NASDAQ: {e}")
        return False


# How many days before the ticker file is considered stale and re-downloaded
TICKER_STALE_DAYS = 7


# ---------------------------------------------
# STEP 1 - LOAD TICKERS
# ---------------------------------------------
def parse_market_cap(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace("$", "").replace(",", "").strip().upper()
    try:
        if s.endswith("T"):
            return float(s[:-1]) * 1e12
        if s.endswith("B"):
            return float(s[:-1]) * 1e9
        if s.endswith("M"):
            return float(s[:-1]) * 1e6
        if s.endswith("K"):
            return float(s[:-1]) * 1e3
        return float(s)
    except ValueError:
        return np.nan


def load_tickers(file_path: str) -> list[str]:
    df = pd.read_csv(file_path)

    # Normalize symbol column (handle various column names)
    sym_col = next((c for c in df.columns if c.lower() in ("symbol","ticker","sym")), None)
    if sym_col is None:
        raise ValueError(f"No 'Symbol' column found in {file_path}. Columns: {df.columns.tolist()}")
    df["Symbol"] = df[sym_col].astype(str).str.replace("/", "-").str.replace(r"\s+", "", regex=True).str.strip().str.upper()

    # Filter by market cap if column exists
    cap_col = next((c for c in df.columns if "market" in c.lower() and "cap" in c.lower()), None)
    if cap_col:
        df["_mc"] = df[cap_col].apply(parse_market_cap)
        before = len(df)
        df = df[df["_mc"] >= MIN_MARKET_CAP]
        log.info(f"Market cap filter: {before} -> {len(df)} tickers")

    # Drop obviously non-investable patterns in the symbol itself
    df = df[~df["Symbol"].str.contains(r"OTC|\.PK|ADR|\^", regex=True, na=False)]

    # Drop non-stock instruments by suffix.
    # Real warrants/rights/units always have a suffix appended to a base
    # ticker, making total symbol length 5+ chars. This avoids false
    # positives on real stocks like UUUU, CAR, SNOW, FLOW, PRNU etc.
    #
    #   ends in W / WS / WW (5+ chars) -> warrant  (e.g. NAMSW, BWMCWS)
    #   ends in R            (6+ chars) -> rights   (e.g. ACMCR -- but CAR is fine)
    #   ends in U            (6+ chars) -> units    (e.g. DNAAU -- but UUUU is fine)
    before_suffix = len(df)
    warrant_pat = r"^[A-Z]{4,}(WS|WW|W)$"   # 5+ total chars ending W/WS/WW
    rights_pat  = r"^[A-Z]{4,}R$"            # 5+ total chars ending R (base 4+)
    units_pat   = r"^[A-Z]{4,}U$"            # 5+ total chars ending U (base 4+)
    combined    = f"({warrant_pat})|({rights_pat})|({units_pat})"
    df = df[~df["Symbol"].str.match(combined)]
    dropped = before_suffix - len(df)
    log.info(f"Non-stock suffix filter (warrants/rights/units): removed {dropped}, kept {len(df)}")

    tickers = sorted(set(df["Symbol"].tolist()) - UNINVESTABLE)
    log.info(f"Final ticker list: {len(tickers)}")
    return tickers


# ---------------------------------------------
# STEP 2 - FETCH DATA
# ---------------------------------------------

# Module-level cache for current risk-free rate (fetched once, used by all tickers)
_CACHED_RF_DAILY = None

def _get_current_rf_daily() -> float:
    """Return current daily risk-free rate from IRX, cached after first call."""
    global _CACHED_RF_DAILY
    if _CACHED_RF_DAILY is not None:
        return _CACHED_RF_DAILY
    try:
        irx_val = float(yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1])
        if not np.isnan(irx_val) and irx_val > 0:
            _CACHED_RF_DAILY = irx_val / 100.0 / 252
            return _CACHED_RF_DAILY
    except Exception:
        pass
    _CACHED_RF_DAILY = 0.0002  # fallback ~5% annual / 252
    return _CACHED_RF_DAILY


def safe_get(info: dict, *keys, default=np.nan):
    """Try multiple key names, return first non-None numeric value, else default."""
    for k in keys:
        v = info.get(k)
        if v is None:
            continue
        try:
            f = float(v)
            if not np.isnan(f):
                return f
        except (TypeError, ValueError):
            continue
    return default


def compute_return(close: pd.Series, days: int) -> float:
    if len(close) >= days and close.iloc[-days] > 0:
        return float(close.iloc[-1] / close.iloc[-days] - 1)
    return np.nan


def compute_vol(close: pd.Series, days: int) -> float:
    if len(close) >= days:
        return float(close.pct_change().iloc[-days:].std() * np.sqrt(252))
    return np.nan


def load_edgar_cik_map() -> dict:
    """
    Load SEC EDGAR's company-to-CIK mapping (free, official source).
    Returns {TICKER: cik_int} or empty dict on failure.
    Requires internet access; fails gracefully so yfinance is always the fallback.
    """
    import urllib.request, json
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        req = urllib.request.Request(
            url, headers={"User-Agent": "portfolio-model research@example.com"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = json.loads(resp.read())
        cik_map = {v["ticker"].upper(): v["cik_str"] for v in raw.values()}
        log.info(f"EDGAR CIK map loaded: {len(cik_map)} companies")
        return cik_map
    except Exception as e:
        log.warning(f"EDGAR CIK map unavailable ({e}) -- will use yfinance earnings only")
        return {}


def fetch_edgar_eps(ticker: str, cik_map: dict) -> float:
    """
    Fetch historical annual EPS from SEC EDGAR XBRL API and return the
    YoY growth rate -- a higher-quality earnings growth signal than
    yfinance's forward vs trailing EPS comparison.

    Data source: data.sec.gov (free, no API key needed, official SEC data).
    Falls back to NaN on any network or parsing error.
    """
    import urllib.request, json
    cik = cik_map.get(ticker.upper())
    if not cik:
        return np.nan
    try:
        url = (
            f"https://data.sec.gov/api/xbrl/companyfacts/"
            f"CIK{str(cik).zfill(10)}.json"
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": "portfolio-model research@example.com"}
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())

        us_gaap = data.get("facts", {}).get("us-gaap", {})
        # Try EPS first, fall back to net income (for companies reporting in total $)
        for key in ["EarningsPerShareDiluted", "EarningsPerShareBasic"]:
            if key not in us_gaap:
                continue
            units = us_gaap[key].get("units", {})
            eps_data = units.get("USD/shares", [])
            if not eps_data:
                continue
            # Annual 10-K filings only, sorted newest first
            annual = sorted(
                [x for x in eps_data if x.get("form") == "10-K" and "val" in x],
                key=lambda x: x.get("end", ""),
                reverse=True,
            )
            if len(annual) >= 2:
                eps_new = float(annual[0]["val"])
                eps_old = float(annual[1]["val"])
                if eps_old != 0 and not np.isnan(eps_new) and not np.isnan(eps_old):
                    growth = (eps_new - eps_old) / abs(eps_old)
                    return float(np.clip(growth, -2.0, 10.0))
    except Exception:
        pass
    return np.nan


def _fetch_with_retry(ticker: str, max_retries: int = 3):
    """
    Fetch history and info for a ticker with retries + exponential backoff.
    Returns (ticker_obj, hist_df, info_dict) or raises on persistent failure.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="max")
            if hist is None or hist.empty or "Close" not in hist.columns:
                raise ValueError(f"Empty history returned for {ticker}")
            close = hist["Close"].dropna()
            if len(close) == 0:
                raise ValueError(f"No Close data for {ticker}")
            info = {}
            try:
                info = t.info or {}
            except Exception:
                pass
            return t, hist, info
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                time.sleep(wait)
    raise last_err


def fetch_single(ticker: str):
    """
    Fetch price history and fundamentals for a single ticker.
    Returns a feature dict or None if the ticker should be skipped.
    """
    try:
        t, hist, info = _fetch_with_retry(ticker)

        close  = hist["Close"].dropna()
        n      = len(close)

        if n < MIN_HISTORY_DAYS:
            return None, None

        price  = float(close.iloc[-1])

        # -- Fundamentals --
        mc         = safe_get(info, "marketCap")
        revenue    = safe_get(info, "totalRevenue")
        ebitda     = safe_get(info, "ebitda")
        fcf        = safe_get(info, "freeCashflow")
        profit_m   = safe_get(info, "profitMargins")
        pe         = safe_get(info, "trailingPE", "forwardPE")
        # yfinance returns debtToEquity as a percent (e.g. 102.63 = 1.03x);
        # normalize to the ratio form used by downstream logic.
        de         = safe_get(info, "debtToEquity")
        if de is not None and not (isinstance(de, float) and np.isnan(de)):
            de = de / 100.0
        roe        = safe_get(info, "returnOnEquity")
        beta       = safe_get(info, "beta")
        sector     = info.get("sector",   "Unknown")
        industry   = info.get("industry", "Unknown")

        # -- Technicals --
        sma50  = float(close.rolling(50).mean().iloc[-1])  if n >= 50  else np.nan
        sma200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else np.nan
        hi52   = float(close.iloc[-min(252,n):].max())
        lo52   = float(close.iloc[-min(252,n):].min())

        def pct_vs(base):
            return float(price / base - 1) if (base and base > 0) else np.nan

        # -- Log-scaled fundamentals (handle negatives) --
        def log_val(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            return float(np.sign(v) * np.log1p(abs(v)))

        # -- Returns & vol --
        r1mo  = compute_return(close, 21)
        r3mo  = compute_return(close, 63)
        r6mo  = compute_return(close, 126)
        r1y   = compute_return(close, 252)
        r3y   = compute_return(close, 756)
        r5y   = compute_return(close, 1260)
        r_target = compute_return(close, PREDICTION_WINDOW)   # TARGET for training

        # Jegadeesh-Titman momentum: 12mo return excluding most recent month.
        # The most recent month exhibits reversal, not momentum. Skipping it
        # captures pure intermediate momentum (strongest documented alpha factor).
        r2_12mo = np.nan
        if n >= 252 and n >= 21 and close.iloc[-252] > 0:
            r2_12mo = float(close.iloc[-21] / close.iloc[-252] - 1)

        v1y   = compute_vol(close, 252)
        v3y   = compute_vol(close, 756)

        # -- Daily-resolution features --
        # These extract shape information from the full daily price series
        # that gets lost when you only look at start/end prices.
        daily_returns = close.pct_change().dropna()

        # Current risk-free rate for Sharpe ratio (cached at module level to
        # avoid 800+ redundant API calls). Consistent with rolling-window
        # training which uses PIT IRX from the macro cache.
        _snapshot_rf_daily = _get_current_rf_daily()

        def max_drawdown(series: pd.Series, days: int) -> float:
            """Max peak-to-trough decline over the window."""
            if len(series) < days:
                return np.nan
            s = series.iloc[-days:]
            rolling_max = s.expanding().max()
            drawdowns = s / rolling_max - 1
            return float(drawdowns.min())

        def sharpe_ratio(ret_series: pd.Series, days: int, rf_daily: float = 0.0002) -> float:
            """Annualised Sharpe ratio over the window. rf_daily ~= 5% annual / 252."""
            if len(ret_series) < days:
                return np.nan
            r = ret_series.iloc[-days:]
            excess = r - rf_daily
            if excess.std() == 0:
                return np.nan
            return float((excess.mean() / excess.std()) * np.sqrt(252))

        def return_skewness(close_series: pd.Series, days: int) -> float:
            """
            Skewness of monthly returns over the window.
            Computed by grouping daily prices into calendar months and
            computing each month's start-to-end return. Does NOT use
            resample() so it works regardless of index type.
            Positive skew = more large up-months (good for long-term growth).
            Negative skew = fat left tail / crash-prone (bad).
            """
            if len(close_series) < days:
                return np.nan
            s = close_series.iloc[-days:].copy()
            # Ensure DatetimeIndex for grouping -- convert timezone-aware to naive
            try:
                idx = pd.to_datetime(s.index)
                if hasattr(idx, "tz") and idx.tz is not None:
                    idx = idx.tz_localize(None)
                s.index = idx
            except Exception:
                return np.nan
            # Group by year-month and compute each month's total return
            monthly_returns = (
                s.groupby([s.index.year, s.index.month])
                 .apply(lambda g: float(g.iloc[-1] / g.iloc[0] - 1) if len(g) >= 10 else np.nan)
                 .dropna()
            )
            if len(monthly_returns) < 6:
                return np.nan
            return float(monthly_returns.skew())

        # Compute daily-resolution features
        mdd_1y   = max_drawdown(close, 252)
        mdd_3y   = max_drawdown(close, 756)
        sharpe1y = sharpe_ratio(daily_returns, 252, _snapshot_rf_daily)
        sharpe3y = sharpe_ratio(daily_returns, 756, _snapshot_rf_daily)
        # Skewness: uses close prices directly with robust month-grouping
        skew_1y  = return_skewness(close, 252)
        skew_3y  = return_skewness(close, 756)

        # -- Return consistency (spike detection) --
        # Measures whether gains were spread evenly or concentrated in one period.
        # Uses 4 non-overlapping 126-day (6-month) backward segments going back 504 days.
        # Each segment: close[end] / close[start] - 1, where segments do NOT overlap.
        # A stock with steady gains has LOW std across segments; a spike stock has HIGH.
        # We invert so higher = more consistent (better).
        #
        # IMPORTANT: segments must be computed as backward slices, NOT via compute_return()
        # on truncated series. compute_return() always reads from the END of whatever
        # series you pass it, so all three calls would return the same last-126-day value.
        sub_returns = []
        lookback_pts = [504, 378, 252, 126, 0]   # segment boundaries in days-ago
        for i in range(len(lookback_pts) - 1):
            back_end   = lookback_pts[i]          # older boundary (days ago)
            back_start = lookback_pts[i + 1]      # newer boundary (days ago)
            if n >= back_end:
                s_end = -back_start if back_start > 0 else None
                sl = close.iloc[-back_end:s_end]
                if len(sl) >= 2 and sl.iloc[0] > 0:
                    r = float(sl.iloc[-1] / sl.iloc[0] - 1)
                    sub_returns.append(r)
        if len(sub_returns) >= 2:
            consistency_vol   = float(np.std(sub_returns))
            return_consistency = float(1.0 / (1.0 + consistency_vol))
        else:
            return_consistency = np.nan

        # -- NEW: Return decay (is recent momentum fading?) --
        # Positive = recent momentum stronger than long-term (accelerating)
        # Negative = recent momentum weaker than long-term (decelerating / spike fading)
        if not np.isnan(r1y) and not np.isnan(r3y) and r3y != 0:
            # Annualized 3Y return vs actual 1Y return
            r3y_annualized = (1 + r3y) ** (1/3) - 1
            return_decay = float(r1y - r3y_annualized)
        else:
            return_decay = np.nan

        # -- NEW: Revenue growth rate (yoy from financials if available) --
        # Cache financials once -- used three times below (rev_growth, accel, gross_margin_trend)
        _financials = None
        try:
            _financials = t.financials
        except Exception:
            pass

        rev_growth = np.nan
        try:
            financials = _financials
            if financials is not None and not financials.empty:
                rev_row = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else None
                if rev_row is not None and len(rev_row) >= 2:
                    r_new = rev_row.iloc[0]
                    r_old = rev_row.iloc[1]
                    if r_old and r_old != 0 and not np.isnan(r_old):
                        rev_growth = float(np.clip((r_new - r_old) / abs(r_old), -0.9, 10.0))
        except Exception:
            pass

        # -- FCF margin and earnings quality --
        fcf_margin = np.nan
        earnings_quality = np.nan
        if revenue and not np.isnan(revenue) and revenue > 0:
            if fcf and not np.isnan(fcf):
                # Clip to [-1.0, 0.60] to match EDGAR training data bounds.
                # Unclipped values (e.g. ZNB at 205%) indicate garbage yfinance
                # data and corrupt the DCF calculation downstream.
                fcf_margin = float(np.clip(fcf / revenue, -1.0, 0.60))
        # Earnings quality = operating cash flow / net income.
        # Measures how much of reported net income is backed by real cash flow.
        # Ratio > 1.0 = cash earnings exceed accrual earnings (high quality).
        # Ratio < 0.5 = large gap between cash and accrual earnings (low quality).
        # This definition matches the EDGAR PIT module so training and scoring
        # use the same feature meaning. (Old definition was EBITDA/Revenue,
        # which is just the EBITDA margin -- a different and mislabeled concept.)
        op_cf_val = safe_get(info, "operatingCashflow")
        net_income_val = safe_get(info, "netIncomeToCommon", "netIncome")
        if not np.isnan(op_cf_val) and not np.isnan(net_income_val) and net_income_val != 0:
            earnings_quality = float(np.clip(op_cf_val / net_income_val, -5.0, 10.0))

        # -- Revenue growth acceleration (2nd derivative of growth) --
        # Tells the model if growth is speeding up or slowing down.
        rev_growth_accel = np.nan
        try:
            financials = _financials
            if financials is not None and not financials.empty:
                rev_row = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else None
                if rev_row is not None and len(rev_row) >= 3:
                    r0 = float(rev_row.iloc[0])
                    r1 = float(rev_row.iloc[1])
                    r2 = float(rev_row.iloc[2])
                    if r2 != 0 and r1 != 0 and not any(np.isnan(x) for x in [r0,r1,r2]):
                        g1 = (r0 - r1) / abs(r1)  # most recent YoY growth
                        g2 = (r1 - r2) / abs(r2)  # prior year growth
                        rev_growth_accel = float(np.clip(g1 - g2, -5.0, 5.0))  # positive = accelerating
        except Exception:
            pass

        # -- Short interest ratio --
        # shortRatio     = days-to-cover (e.g. 5.0 days), normalise to 0-1 via /30
        # shortPercentOfFloat = fraction of float shorted (e.g. 0.05 = 5%), already 0-1
        # Fetch separately so we know which scale we have and normalise correctly.
        short_ratio = np.nan
        _sr   = safe_get(info, "shortRatio")              # days-to-cover
        _spof = safe_get(info, "shortPercentOfFloat")     # decimal fraction
        if not np.isnan(_sr):
            short_ratio = min(_sr / 30.0, 1.0)           # cap at 1.0 (30+ day cover)
        elif not np.isnan(_spof):
            short_ratio = float(np.clip(_spof, 0.0, 1.0))  # already 0-1

        # - PIOTROSKI F-SCORE (0-9, normalized to 0-1) -
        # Developed by Joseph Piotroski (2000). Each criterion scores 1 or 0.
        # IMPORTANT: F4, F5, F6, F8, F9 are YoY CHANGE criteria, not absolute thresholds.
        # A company with leverage that DECREASED gets F4=1 even if leverage is still high.
        # We use yfinance t.financials (multi-year income statement) for YoY comparisons.
        piotroski = 0
        piotroski_available = 0

        # Profitability signals (F1-F4)
        roa          = safe_get(info, "returnOnAssets")
        op_cf        = safe_get(info, "operatingCashflow")
        total_assets = safe_get(info, "totalAssets")
        roa_prior    = np.nan
        # F3 (delta ROA) needs prior-year ROA — derive from t.financials if available
        try:
            fin = _financials
            if fin is not None and not fin.empty and total_assets and not np.isnan(total_assets):
                ni_row = None
                for k in ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operations"]:
                    if k in fin.index: ni_row = fin.loc[k]; break
                if ni_row is not None and len(ni_row) >= 2 and total_assets > 0:
                    # prior-year ROA approximation: prior net income / current assets
                    # (best we can do without prior-year total assets from yfinance)
                    prior_ni = float(ni_row.iloc[1])
                    if not np.isnan(prior_ni):
                        roa_prior = prior_ni / total_assets
        except Exception:
            pass

        # F1: ROA > 0
        if not np.isnan(roa):
            piotroski += int(roa > 0)
            piotroski_available += 1

        # F2: Operating cash flow > 0
        if not np.isnan(op_cf):
            piotroski += int(op_cf > 0)
            piotroski_available += 1

        # F3: ROA improved YoY (delta ROA > 0)
        if not np.isnan(roa) and not np.isnan(roa_prior):
            piotroski += int(roa > roa_prior)
            piotroski_available += 1

        # F4: Accruals -- CFO/Assets > ROA (cash earnings exceed accrual earnings)
        if not np.isnan(op_cf) and not np.isnan(total_assets) and total_assets > 0 and not np.isnan(roa):
            piotroski += int((op_cf / total_assets) > roa)
            piotroski_available += 1

        # F5: Leverage DECREASED YoY (not just "is low" -- YoY change is what matters)
        try:
            bs = t.balance_sheet  # multi-year balance sheet
            if bs is not None and not bs.empty:
                ltd_row = None
                for k in ["Long Term Debt", "Long-Term Debt", "LongTermDebt"]:
                    if k in bs.index:
                        ltd_row = bs.loc[k]
                        break
                ta_row = None
                for k in ["Total Assets", "TotalAssets"]:
                    if k in bs.index:
                        ta_row = bs.loc[k]
                        break
                if (ltd_row is not None and ta_row is not None and
                        len(ltd_row) >= 2 and len(ta_row) >= 2 and
                        ta_row.iloc[0] > 0 and ta_row.iloc[1] > 0):
                    lev_new = float(ltd_row.iloc[0]) / float(ta_row.iloc[0])
                    lev_old = float(ltd_row.iloc[1]) / float(ta_row.iloc[1])
                    piotroski += int(lev_new <= lev_old)  # F5: leverage did not increase
                    piotroski_available += 1
        except Exception:
            pass

        # F6: Current ratio IMPROVED YoY
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                ca_row = None
                cl_row = None
                for k in ["Current Assets", "Total Current Assets"]:
                    if k in bs.index: ca_row = bs.loc[k]; break
                for k in ["Current Liabilities", "Total Current Liabilities"]:
                    if k in bs.index: cl_row = bs.loc[k]; break
                if (ca_row is not None and cl_row is not None and
                        len(ca_row) >= 2 and len(cl_row) >= 2 and
                        cl_row.iloc[0] > 0 and cl_row.iloc[1] > 0):
                    cr_new = float(ca_row.iloc[0]) / float(cl_row.iloc[0])
                    cr_old = float(ca_row.iloc[1]) / float(cl_row.iloc[1])
                    piotroski += int(cr_new >= cr_old)  # F6: liquidity improved
                    piotroski_available += 1
        except Exception:
            pass

        # F7: No share dilution (shares did not increase materially)
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                sh_row = None
                for k in ["Share Issued", "Common Stock", "Shares Outstanding"]:
                    if k in bs.index: sh_row = bs.loc[k]; break
                if sh_row is not None and len(sh_row) >= 2:
                    sh_new = float(sh_row.iloc[0])
                    sh_old = float(sh_row.iloc[1])
                    if sh_old > 0:
                        piotroski += int(sh_new <= sh_old * 1.02)  # allow 2% tolerance
                        piotroski_available += 1
        except Exception:
            pass

        # F8: Gross margin IMPROVED YoY (using financials for multi-year data)
        try:
            fin = _financials
            if fin is not None and not fin.empty:
                gp_row, rv_row = None, None
                for k in ["Gross Profit", "GrossProfit"]:
                    if k in fin.index: gp_row = fin.loc[k]; break
                for k in ["Total Revenue", "Revenue"]:
                    if k in fin.index: rv_row = fin.loc[k]; break
                if (gp_row is not None and rv_row is not None and
                        len(gp_row) >= 2 and len(rv_row) >= 2 and
                        rv_row.iloc[0] != 0 and rv_row.iloc[1] != 0):
                    gm_new = float(gp_row.iloc[0]) / float(rv_row.iloc[0])
                    gm_old = float(gp_row.iloc[1]) / float(rv_row.iloc[1])
                    piotroski += int(gm_new >= gm_old)  # F8: gross margin improved
                    piotroski_available += 1
        except Exception:
            pass

        # F9: Asset turnover IMPROVED YoY (revenue/assets ratio increased)
        try:
            fin = _financials
            bs  = t.balance_sheet
            if (fin is not None and not fin.empty and
                    bs is not None and not bs.empty):
                rv_row, ta_row = None, None
                for k in ["Total Revenue", "Revenue"]:
                    if k in fin.index: rv_row = fin.loc[k]; break
                for k in ["Total Assets", "TotalAssets"]:
                    if k in bs.index: ta_row = bs.loc[k]; break
                if (rv_row is not None and ta_row is not None and
                        len(rv_row) >= 2 and len(ta_row) >= 2 and
                        ta_row.iloc[0] > 0 and ta_row.iloc[1] > 0):
                    at_new = float(rv_row.iloc[0]) / float(ta_row.iloc[0])
                    at_old = float(rv_row.iloc[1]) / float(ta_row.iloc[1])
                    piotroski += int(at_new >= at_old)  # F9: efficiency improved
                    piotroski_available += 1
        except Exception:
            pass

        # Normalize: require at least 5 criteria computable for a valid score
        piotroski_score = float(piotroski / piotroski_available) if piotroski_available >= 5 else np.nan

        # - DCF-IMPLIED RETURN -
        # Projects free cash flow 6 years forward using current growth rate
        # and margins, applies a terminal EV/FCF multiple, discounts back
        # at WACC, and computes implied return vs current market cap.
        # This is genuinely forward-looking -- not backward-looking momentum.
        dcf_implied_return = np.nan
        try:
            if (not np.isnan(revenue) and revenue > 0 and
                not np.isnan(mc) and mc > 0 and
                not np.isnan(rev_growth)):

                # For pre-profit companies (negative or NaN FCF margin),
                # use a conservative starting point. Do NOT assume they become
                # profitable — that is the key error that overvalues loss-makers.
                if np.isnan(fcf_margin) or fcf_margin < -0.5:
                    fcf_margin_for_dcf = -0.15  # loss-making but not terminal
                else:
                    fcf_margin_for_dcf = fcf_margin

                # WACC: risk-free 4.5% + beta * 5.5% equity risk premium
                beta_val = beta if not np.isnan(beta) else 1.0
                wacc = 0.045 + max(0.5, min(beta_val, 2.5)) * 0.055

                # Revenue growth mean-reverts toward 5% over 6 years
                # (high-growth companies slow down; this is empirically well-documented)
                terminal_growth = 0.05
                g0 = max(-0.3, min(rev_growth, 1.5))  # cap starting growth
                projected_revenue = revenue
                projected_fcf = 0.0

                # DCF margin target: revert toward a realistic target, not
                # a hardcoded 8%. Companies at -200% margin do NOT reach 8% in a few years.
                # Target = min(8%, current_margin + 15ppt). This caps the maximum
                # margin improvement at 15 percentage points over 6 years.
                dcf_target_margin = min(0.08, fcf_margin_for_dcf + 0.15)
                # If still deeply negative after cap, floor at -5% (not profitable)
                dcf_target_margin = max(dcf_target_margin, -0.05)

                for yr in range(1, 7):
                    frac = yr / 6.0
                    g_yr = g0 * (1 - frac) + terminal_growth * frac
                    projected_revenue *= (1 + g_yr)
                    # Margins revert toward the realistic target, not a fixed 8%
                    fcf_m_proj = fcf_margin_for_dcf + (dcf_target_margin - fcf_margin_for_dcf) * frac
                    fcf_m_proj = max(-0.3, min(fcf_m_proj, 0.5))
                    yr_fcf = projected_revenue * fcf_m_proj
                    projected_fcf += yr_fcf / ((1 + wacc) ** yr)

                # Terminal value
                # Continuous terminal multiple: linear interpolation from 12x (zero growth)
                # to 28x (30%+ growth), avoiding valuation cliffs from discrete tiers.
                terminal_multiple = float(np.clip(12.0 + g0 * 53.3, 12.0, 28.0))

                # Use the actual year-6 projected FCF margin (after realistic reversion),
                # not a hardcoded 8%. If the company is still unprofitable at year 6,
                # terminal value should reflect that (minimal or zero).
                yr6_fcf_margin = dcf_target_margin
                yr6_fcf = projected_revenue * max(yr6_fcf_margin, 0.0)
                terminal_val = (yr6_fcf * terminal_multiple) / ((1 + wacc) ** 6)

                intrinsic_value = projected_fcf + terminal_val

                # Net debt adjustment
                net_debt = safe_get(info, "netDebt", "totalDebt")
                if not np.isnan(net_debt):
                    equity_value = max(intrinsic_value - net_debt, intrinsic_value * 0.1)
                else:
                    equity_value = intrinsic_value

                if equity_value > 0 and mc > 0:
                    dcf_implied_return = float(equity_value / mc - 1)
                    dcf_implied_return = max(-0.99, min(dcf_implied_return, 50.0))
        except Exception:
            pass

        # - EARNINGS GROWTH SIGNAL -
        # YoY earnings growth rate from yfinance (earningsGrowth) or
        # forward-vs-trailing EPS spread.  This is backward-looking, NOT
        # actual analyst revision momentum (which yfinance cannot provide).
        earnings_growth_signal = np.nan
        try:
            fwd_eps    = safe_get(info, "forwardEps")
            trailing_e = safe_get(info, "trailingEps")
            earn_growth = safe_get(info, "earningsGrowth", "earningsQuarterlyGrowth")

            if not np.isnan(earn_growth):
                earnings_growth_signal = float(np.clip(earn_growth, -2.0, 10.0))
            elif not np.isnan(fwd_eps) and not np.isnan(trailing_e) and trailing_e != 0:
                earnings_growth_signal = float((fwd_eps - trailing_e) / abs(trailing_e))
                earnings_growth_signal = float(np.clip(earnings_growth_signal, -2.0, 10.0))
        except Exception:
            pass

        # - SECTOR-RELATIVE PE -
        # Raw PE is misleading across sectors. A PE of 25 is cheap for tech,
        # expensive for utilities. We store raw PE; sector medians are
        # computed across the full dataset in fetch_all after all stocks load.
        forward_pe = safe_get(info, "forwardPE")
        peg_ratio  = safe_get(info, "trailingPegRatio", "pegRatio")

        # - GROSS MARGIN TREND -
        # Expanding gross margins = pricing power. Compressing = competition.
        gross_margin_trend = np.nan
        try:
            financials = _financials
            if financials is not None and not financials.empty:
                if "Gross Profit" in financials.index and "Total Revenue" in financials.index:
                    gp  = financials.loc["Gross Profit"]
                    rev = financials.loc["Total Revenue"]
                    if len(gp) >= 2 and len(rev) >= 2:
                        gm_new = float(gp.iloc[0] / rev.iloc[0]) if rev.iloc[0] != 0 else np.nan
                        gm_old = float(gp.iloc[1] / rev.iloc[1]) if rev.iloc[1] != 0 else np.nan
                        if not np.isnan(gm_new) and not np.isnan(gm_old):
                            gross_margin_trend = gm_new - gm_old
        except Exception:
            pass

        # - INSIDER OWNERSHIP -
        # High insider ownership aligns management with shareholders.
        # When insiders own a large stake, they can't easily exit -- strong signal.
        insider_pct = safe_get(info, "heldPercentInsiders")
        inst_pct    = safe_get(info, "heldPercentInstitutions")

        # - ACCRUALS RATIO (Sloan 1996) -
        # (Net Income - Operating Cash Flow) / Total Assets
        # High accruals = earnings driven by accounting, not cash.
        # One of the most robust anomalies in finance: high accruals predict
        # poor future returns. Complementary to Earnings_Quality (which uses
        # the raw ratio of CF to NI without normalising by assets).
        accruals_ratio = np.nan
        try:
            _ni_val  = safe_get(info, "netIncomeToCommon", "netIncome")
            _opcf    = safe_get(info, "operatingCashflow")
            if not np.isnan(_ni_val) and not np.isnan(_opcf):
                bs = t.balance_sheet
                if bs is not None and not bs.empty:
                    for k in ["Total Assets", "TotalAssets"]:
                        if k in bs.index:
                            _ta = float(bs.loc[k].iloc[0])
                            if _ta > 0:
                                accruals_ratio = float(np.clip((_ni_val - _opcf) / _ta, -1.0, 1.0))
                            break
        except Exception:
            pass

        # - BOOK-TO-MARKET (Fama & French 1992) -
        # The canonical value factor. Low B/M = growth, high B/M = value.
        # High B/M stocks consistently outperform over long horizons.
        # Captures a different valuation dimension than PE.
        book_to_market = np.nan
        try:
            _bv     = safe_get(info, "bookValue")          # per share
            _shares = safe_get(info, "sharesOutstanding")
            if not np.isnan(_bv) and not np.isnan(_shares) and not np.isnan(mc) and mc > 0:
                book_to_market = float(np.clip(_bv * _shares / mc, -1.0, 10.0))
        except Exception:
            pass

        # - ASSET GROWTH (Cooper, Gulen, Schill 2008) -
        # YoY total asset growth. Firms that aggressively grow assets
        # (via acquisitions, capex, debt) tend to earn lower future returns.
        asset_growth = np.nan
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                for k in ["Total Assets", "TotalAssets"]:
                    if k in bs.index:
                        ta_row = bs.loc[k]
                        if len(ta_row) >= 2:
                            ta_new = float(ta_row.iloc[0])
                            ta_old = float(ta_row.iloc[1])
                            if ta_old > 0:
                                asset_growth = float(np.clip((ta_new - ta_old) / ta_old, -0.5, 5.0))
                        break
        except Exception:
            pass

        # - NET ISSUANCE (Pontiff & Woodgate 2008) -
        # YoY change in shares outstanding. Companies that issue equity
        # tend to underperform; companies buying back shares outperform.
        net_issuance = np.nan
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                for k in ["Share Issued", "Common Stock", "Shares Outstanding"]:
                    if k in bs.index:
                        sh_row = bs.loc[k]
                        if len(sh_row) >= 2:
                            sh_new = float(sh_row.iloc[0])
                            sh_old = float(sh_row.iloc[1])
                            if sh_old > 0:
                                net_issuance = float(np.clip((sh_new - sh_old) / sh_old, -0.3, 1.0))
                        break
        except Exception:
            pass

        # - R&D INTENSITY (Chan, Lakonishok, Sougiannis 2001) -
        # R&D expense / Revenue. R&D is expensed rather than capitalised,
        # which systematically understates the assets and true value of
        # R&D-intensive firms. Strong predictor for tech and biotech.
        rd_intensity = np.nan
        try:
            fin = _financials
            if fin is not None and not fin.empty and not np.isnan(revenue) and revenue > 0:
                for k in ["Research Development", "Research And Development",
                           "ResearchDevelopment", "Research & Development"]:
                    if k in fin.index:
                        _rd_val = float(fin.loc[k].iloc[0])
                        if not np.isnan(_rd_val):
                            rd_intensity = float(np.clip(abs(_rd_val) / revenue, 0.0, 1.0))
                        break
        except Exception:
            pass

        # - AVERAGE DAILY VOLUME (used for liquidity filter in allocate) -
        avg_daily_volume = safe_get(info, "averageDailyVolume10Day", "averageVolume")

        return {
            "Ticker":           ticker,
            "Price":            price,
            "Sector":           sector,
            "Industry":         industry,
            "Market_Cap":       mc,
            "Revenue":          revenue,
            "EBITDA":           ebitda,
            "FCF":              fcf,
            "Profit_Margin":    profit_m,
            "PE":               pe,
            "Debt_Equity":      de,
            "ROE":              roe,
            "Beta":             beta,
            # Log features
            "Log_MarketCap":    log_val(mc),
            "Log_Revenue":      log_val(revenue),
            "Log_EBITDA":       log_val(ebitda),
            "Log_FCF":          log_val(fcf),
            # Technicals
            "50D_SMA":          sma50,
            "200D_SMA":         sma200,
            "52W_High":         hi52,
            "52W_Low":          lo52,
            "Price_vs_50SMA":   pct_vs(sma50),
            "Price_vs_200SMA":  pct_vs(sma200),
            "Price_vs_52WHigh": pct_vs(hi52),
            "Price_vs_52WLow":  pct_vs(lo52),
            # Returns (raw)
            "Return_1mo":       r1mo,
            "Return_3mo":       r3mo,
            "Return_6mo":       r6mo,
            "Return_1Y":        r1y,
            "Return_3Y":        r3y,
            "Return_5Y":        r5y,
            "Return_2_12mo":    r2_12mo,
            "Return_Target":        r_target,   # NaN if insufficient history
            # Volatility
            "Vol_1Y":           v1y,
            "Vol_3Y":           v3y,
            # Flags
            "Has_3Y_History":   int(n >= 756),
            "Has_5Y_History":   int(n >= 1260),
            "Has_Full_History":   int(n >= PREDICTION_WINDOW),
            "Days_History":     n,
            # Quality & consistency features
            "Return_Consistency":     return_consistency,
            "Return_Decay":           return_decay,
            "Revenue_Growth_Rate":    rev_growth,
            "Revenue_Growth_Accel":   rev_growth_accel,
            "FCF_Margin":             fcf_margin,
            "Earnings_Quality":       earnings_quality,
            "Short_Interest_Ratio":   short_ratio,
            # Daily-resolution features (extracted from full price series)
            "Max_Drawdown_1Y":        mdd_1y,
            "Max_Drawdown_3Y":        mdd_3y,
            "Sharpe_1Y":              sharpe1y,
            "Sharpe_3Y":              sharpe3y,
            "Return_Skewness_1Y":     skew_1y,
            "Return_Skewness_3Y":     skew_3y,
            # SPY-relative and regime features added in fetch_all
            "Return_1Y_vs_SPY":       np.nan,
            "Return_3Y_vs_SPY":       np.nan,
            "SPY_Return_1Y":          np.nan,
            "SPY_Vol_1Y":             np.nan,
            "SPY_ATH_Drawdown":       np.nan,   # market stress at scoring date
            "SPY_200SMA_Pct":         np.nan,   # bull/bear regime at scoring date
            "VIX_Level":              np.nan,   # filled in fetch_all (same value all stocks)
            "Yield_Spread":           np.nan,   # filled in fetch_all (same value all stocks)
            # Hedge-fund grade forward-looking signals
            "Piotroski_Score":        piotroski_score,    # 0-1 fundamental quality
            "DCF_Implied_Return":     dcf_implied_return, # forward-looking fair value gap
            "Earnings_Growth_Signal": earnings_growth_signal,  # YoY earnings growth
            "Forward_PE":             forward_pe,         # forward earnings multiple
            "PEG_Ratio":              peg_ratio,          # PE relative to growth
            "Gross_Margin_Trend":     gross_margin_trend, # expanding vs compressing margins
            "Insider_Pct":            insider_pct,        # management skin in game
            "Inst_Pct":               inst_pct,           # institutional conviction
            # Academically-proven cross-sectional return predictors
            "Accruals_Ratio":         accruals_ratio,     # Sloan 1996 -- high accruals = poor returns
            "Book_to_Market":         book_to_market,     # Fama-French 1992 -- canonical value factor
            "Asset_Growth":           asset_growth,       # Cooper et al. 2008 -- negative predictor
            "Net_Issuance":           net_issuance,       # Pontiff-Woodgate 2008 -- dilution signal
            "RD_Intensity":           rd_intensity,       # Chan et al. 2001 -- undervalued R&D firms
            # Liquidity (not a model feature -- used for position sizing filter)
            "Avg_Daily_Volume":       avg_daily_volume,
            # Sector-relative PE computed in fetch_all after full dataset loads
            "PE_vs_Sector":           np.nan,
            "PE_vs_Sector_Fwd":       np.nan,
        }, close

    except Exception as e:
        log.debug(f"  {ticker} failed: {e}")
        return None, None


def fetch_all(tickers: list[str]) -> tuple[pd.DataFrame, dict]:
    rows = []
    ohlc_series = {}   # {ticker: pd.Series of close prices} — saved to OHLC cache
    failed = 0
    consecutive_failures = 0
    for tk in tqdm(tickers, desc="Fetching"):
        row, close_s = fetch_single(tk)
        if row:
            rows.append(row)
            if close_s is not None and len(close_s) >= 504:
                ohlc_series[tk] = close_s
            consecutive_failures = 0
        else:
            failed += 1
            consecutive_failures += 1
            # If many consecutive failures, Yahoo may be rate-limiting.
            # Back off progressively to let the API recover.
            if consecutive_failures >= 10:
                backoff = min(consecutive_failures * 0.5, 30)
                log.warning(f"  {consecutive_failures} consecutive fetch failures "
                            f"(last: {tk}) — backing off {backoff:.0f}s")
                time.sleep(backoff)
        time.sleep(FETCH_PAUSE)

    log.info(f"Fetched {len(rows)} tickers, {failed} failed/skipped")

    # Guard: if >80% of tickers failed, something systemic is wrong
    # (rate limiting, network issue, yfinance API change). Warn loudly.
    if len(tickers) > 50 and failed / len(tickers) > 0.80:
        log.error(
            f"CRITICAL: {failed}/{len(tickers)} tickers ({failed/len(tickers):.0%}) failed to fetch! "
            f"Only {len(rows)} succeeded. This likely indicates a yfinance API issue, "
            f"rate limiting, or network problem. The portfolio will be based on a "
            f"tiny, non-representative subset. Consider re-running after checking "
            f"your network and yfinance version (pip install --upgrade yfinance)."
        )
    df = pd.DataFrame(rows)

    # Drop tickers where yfinance returned no market cap at all.
    # These are almost always warrants, SPACs, or data-dead shells
    # that slipped through the CSV filter (blank market cap in source file).
    before_mc = len(df)
    df = df[df["Market_Cap"].notna() & (df["Market_Cap"] > 0)]
    log.info(f"Dropped {before_mc - len(df)} tickers with no market cap data after fetch")

    # Final safety net: re-apply the suffix filter on whatever tickers
    # survived fetching, in case any slipped through with internal spaces
    # or other encoding quirks in the source CSV.
    import re as _re
    _w = r"^[A-Z]{4,}(WS|WW|W)$"
    _r = r"^[A-Z]{4,}R$"
    _u = r"^[A-Z]{4,}U$"
    _pat = f"({_w})|({_r})|({_u})"
    before_recheck = len(df)
    df = df[~df["Ticker"].str.match(_pat)]
    if before_recheck - len(df) > 0:
        log.info(f"Post-fetch suffix recheck: removed {before_recheck - len(df)} non-stock instruments")

    # Force ALL columns to numeric - yfinance sometimes returns strings
    for col in df.columns:
        if col not in ("Ticker", "Sector", "Industry"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clip extreme values after numeric conversion
    clip_rules = {
        "PE":           (-50,  100),   # tightened from (-100, 500) to reduce PE feature dominance
        "Debt_Equity":  (-10,  500),
        "ROE":          (-5,   5),
        "Beta":         (-5,   10),
        "Profit_Margin":(-10,  10),
        "Return_1mo":   (-1,   10),
        "Return_3mo":   (-1,   20),
        "Return_6mo":   (-1,   50),
        "Return_1Y":    (-1,   50),
        "Return_3Y":    (-1,   200),
        "Return_5Y":    (-1,   500),
        "Return_Target":    (-1,   10),      # 1Y horizon: cap at 1000%
    }
    for col, (lo, hi) in clip_rules.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # -- Compute SPY-relative features for all tickers --
    # Fetch SPY once and compute its returns/vol to use as market context.
    # Return_1Y_vs_SPY = stock 1Y return minus SPY 1Y return (pure alpha).
    # SPY_Return_1Y / SPY_Vol_1Y give the model market regime context.
    log.info("Fetching SPY benchmark data...")
    try:
        _, _spy_hist_df, _ = _fetch_with_retry("SPY")
        spy_hist = _spy_hist_df["Close"].dropna()
        spy_r1y   = compute_return(spy_hist, 252)
        spy_r3y   = compute_return(spy_hist, 756)
        spy_v1y   = compute_vol(spy_hist, 252)
        if not np.isnan(spy_r1y):
            df["Return_1Y_vs_SPY"] = df["Return_1Y"] - spy_r1y
        if not np.isnan(spy_r3y):
            df["Return_3Y_vs_SPY"] = df["Return_3Y"] - spy_r3y
        df["SPY_Return_1Y"] = spy_r1y
        df["SPY_Vol_1Y"]    = spy_v1y
        log.info(f"SPY 1Y={spy_r1y:.1%}  3Y={spy_r3y:.1%}  Vol={spy_v1y:.1%}")

        # -- NEW: SPY macro regime features --
        # These give the model context about WHERE we are in the market cycle.
        # The same stock features mean different things in a deep bear vs a raging bull.
        spy_price     = float(spy_hist.iloc[-1])
        spy_ath_1y    = float(spy_hist.iloc[-min(252, len(spy_hist)):].max())
        spy_sma200_v  = float(spy_hist.rolling(200).mean().iloc[-1]) if len(spy_hist) >= 200 else np.nan
        spy_ath_dd    = float(spy_price / spy_ath_1y - 1)          if spy_ath_1y > 0          else np.nan
        spy_200pct    = float(spy_price / spy_sma200_v - 1)        if not np.isnan(spy_sma200_v) and spy_sma200_v > 0 else np.nan
        df["SPY_ATH_Drawdown"] = spy_ath_dd
        df["SPY_200SMA_Pct"]   = spy_200pct
        log.info(f"SPY regime: ATH_Drawdown={spy_ath_dd:.1%}  200SMA_Pct={spy_200pct:.1%}")
    except Exception as e:
        log.warning(f"SPY fetch failed ({e}) - relative strength features will be NaN")

    # -- Current macro regime: VIX fear gauge and yield curve spread --
    # VIX: < 15 = calm, 15-20 = normal, 20-30 = elevated, > 30 = stress/crisis.
    # Yield spread (10Y - 3mo): positive = normal curve, negative = inverted (recession warning).
    # These are the SAME value for every stock at scoring time (market-wide signals),
    # but differ ACROSS training windows (model learns how they affect outcomes).
    def _fetch_index_close(symbol, period="5d"):
        """Fetch index close with retry."""
        for _attempt in range(3):
            try:
                h = yf.Ticker(symbol).history(period=period)
                if h is not None and not h.empty and "Close" in h.columns:
                    c = h["Close"].dropna()
                    if len(c) > 0:
                        return float(c.iloc[-1])
            except Exception:
                pass
            time.sleep((2 ** _attempt) * 0.5)
        raise ValueError(f"Failed to fetch {symbol} after 3 retries")

    try:
        vix_now = _fetch_index_close("^VIX")
        df["VIX_Level"] = vix_now
        log.info(f"Current VIX: {vix_now:.1f}")
    except Exception as e:
        log.warning(f"VIX fetch failed ({e}) -- VIX_Level will be NaN")
        df["VIX_Level"] = np.nan

    try:
        tnx_now    = _fetch_index_close("^TNX")
        irx_now    = _fetch_index_close("^IRX")
        spread_now = tnx_now - irx_now
        df["Yield_Spread"] = spread_now
        log.info(f"Yield spread (10Y-3mo): {spread_now:+.2f}%  (10Y={tnx_now:.2f}%  3mo={irx_now:.2f}%)")
    except Exception as e:
        log.warning(f"Yield curve fetch failed ({e}) -- Yield_Spread will be NaN")
        df["Yield_Spread"] = np.nan

    # - Sector-relative valuation -
    # A PE of 30 means nothing without sector context.
    # PE_vs_Sector < 0 = cheaper than sector peers (potential value)
    # PE_vs_Sector > 0 = more expensive than sector peers
    log.info("Computing sector-relative valuations...")
    for col, out_col in [("PE", "PE_vs_Sector"), ("Forward_PE", "PE_vs_Sector_Fwd")]:
        if col not in df.columns:
            continue
        sector_medians = (
            df[df[col].notna() & (df[col] > 0) & (df[col] < 500)]
            .groupby("Sector")[col]
            .median()
        )
        def pe_vs_sector(row, col=col, medians=sector_medians):
            v = row.get(col, np.nan)
            s = row.get("Sector", "Unknown")
            med = medians.get(s, np.nan)
            if np.isnan(v) or np.isnan(med) or med == 0:
                return np.nan
            return float((v - med) / med)   # % premium/discount vs sector
        df[out_col] = df.apply(pe_vs_sector, axis=1)
    log.info("Sector-relative PE computed.")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # -- EDGAR earnings supplement --
    # For stocks where yfinance returned no Earnings_Growth_Signal, try SEC EDGAR.
    # EDGAR provides historical annual EPS from official 10-K filings -- more
    # reliable than yfinance's forward vs trailing EPS approximation.
    # This runs as a batch after the main fetch; failures are silent per-stock.
    if "Earnings_Growth_Signal" in df.columns:
        missing_er = df["Earnings_Growth_Signal"].isna()
        n_missing  = missing_er.sum()
        if n_missing > 0:
            log.info(f"Supplementing {n_missing} missing Earnings_Growth_Signal values from SEC EDGAR...")
            edgar_cik_map = load_edgar_cik_map()
            if edgar_cik_map:
                filled = 0
                for idx, row in df[missing_er].iterrows():
                    er = fetch_edgar_eps(row["Ticker"], edgar_cik_map)
                    if not np.isnan(er):
                        df.at[idx, "Earnings_Growth_Signal"] = er
                        filled += 1
                    time.sleep(0.05)   # respectful rate-limit for SEC servers
                log.info(f"EDGAR filled {filled}/{n_missing} missing Earnings_Growth_Signal values")
            else:
                log.info("EDGAR CIK map empty -- skipping earnings supplement")

    return df, ohlc_series


# ---------------------------------------------
# STEP 4 - SCORE ALL STOCKS + EXPLAIN
# ---------------------------------------------
def explain_stock(row: pd.Series) -> tuple[str, str, str]:
    """
    Return (green_flags, red_flags, confidence_tier) for a single stock row.
    Human-readable strings explaining why the model scored it as it did.
    """
    green, red = [], []

    # -- Momentum signals --
    r1y = row.get("Return_1Y", np.nan)
    r3y = row.get("Return_3Y", np.nan)
    consistency = row.get("Return_Consistency", np.nan)
    decay = row.get("Return_Decay", np.nan)

    if not np.isnan(r1y):
        if r1y > 0.3:
            green.append(f"Strong 1Y momentum (+{r1y:.0%})")
        elif r1y < -0.2:
            red.append(f"Weak 1Y momentum ({r1y:.0%})")

    if not np.isnan(r3y):
        if r3y > 0.5:
            green.append(f"Strong 3Y growth (+{r3y:.0%})")
        elif r3y < 0:
            red.append(f"Negative 3Y return ({r3y:.0%})")

    if not np.isnan(consistency):
        if consistency > 0.6:
            green.append("Consistent returns (not spike-driven)")
        elif consistency < 0.35:
            red.append("Returns appear spike-driven (low consistency)")

    if not np.isnan(decay):
        if decay > 0.1:
            green.append("Momentum accelerating recently")
        elif decay < -0.2:
            red.append("Momentum fading vs long-term trend")

    # -- Fundamental signals --
    roe = row.get("ROE", np.nan)
    pm  = row.get("Profit_Margin", np.nan)
    de  = row.get("Debt_Equity", np.nan)
    fcf_m = row.get("FCF_Margin", np.nan)
    rev_g = row.get("Revenue_Growth_Rate", np.nan)
    eq  = row.get("Earnings_Quality", np.nan)

    if not np.isnan(roe):
        if roe > 0.15:
            green.append(f"High ROE ({roe:.0%})")
        elif roe < 0:
            red.append(f"Negative ROE ({roe:.0%})")

    if not np.isnan(pm):
        if pm > 0.15:
            green.append(f"Strong profit margin ({pm:.0%})")
        elif pm < 0:
            red.append(f"Unprofitable (margin {pm:.0%})")

    if not np.isnan(de):
        if de < 0.5:
            green.append("Low debt load")
        elif de > 2.0:
            red.append(f"High debt/equity ({de:.1f}x)")

    if not np.isnan(fcf_m):
        if fcf_m > 0.1:
            green.append(f"Strong free cash flow margin ({fcf_m:.0%})")
        elif fcf_m < 0:
            red.append("Negative free cash flow")

    if not np.isnan(rev_g):
        if rev_g > 0.15:
            green.append(f"Revenue growing fast (+{rev_g:.0%} YoY)")
        elif rev_g < -0.05:
            red.append(f"Revenue declining ({rev_g:.0%} YoY)")

    if not np.isnan(eq):
        if eq > 1.0:
            green.append(f"High earnings quality (cash flow {eq:.1f}x net income)")
        elif eq < 0:
            red.append("Negative earnings quality (cash flow diverges from net income)")

    # -- Confidence tier based on data completeness --
    has_3y = row.get("Has_3Y_History", 0)
    has_5y = row.get("Has_5Y_History", 0)
    fund_count = sum(1 for v in [roe, pm, de, fcf_m, rev_g] if not np.isnan(v))

    if has_5y and fund_count >= 4:
        tier = "HIGH - strong history + full fundamentals"
    elif has_3y and fund_count >= 3:
        tier = "MEDIUM - good history, most fundamentals available"
    elif fund_count >= 2:
        tier = "LOW - limited history or sparse fundamentals"
    else:
        tier = "SPECULATIVE - minimal data, treat as high risk"

    green_str = "; ".join(green) if green else "No strong positive signals"
    red_str   = "; ".join(red)   if red   else "No major red flags"
    return green_str, red_str, tier


# ---------------------------------------------
def factor_score_stocks(df: pd.DataFrame, spy_expected_return: float = 0.08) -> pd.DataFrame:
    """
    Empirically-validated multi-factor scoring.

    Factor weights from SURVIVORSHIP-CORRECTED Spearman rank correlations
    between each factor and REAL 6-month forward returns.
    Validation: 10,701 samples (7,413 live + 1,388 recovered dead + 950 synthetic dead)
    across 29 annual test dates from 1996-2025. Dead stocks injected at
    Shumway (1997) -30% return with distressed factor profiles.

    CORRECTED price-based signals (rho with actual 6mo returns):
      Vol_1Y            rho=-0.197***  LOWER vol → higher returns (corrected!)
      Max_Drawdown_1Y   rho=+0.181***  Shallower drawdowns → higher returns
      Return_6mo        rho=+0.180***  6-month momentum continues (strong)
      Return_2_12mo     rho=+0.179***  12-2mo momentum is ALIVE (corrected!)
      Return_1Y         rho=+0.158***  1-year momentum continues
      Return_Decay      rho=+0.154***  Accelerating returns → continues
      Price_vs_200SMA   rho=+0.149***  Above 200SMA → continues higher
      Return_3mo        rho=+0.148***  3-month momentum continues
      Sharpe_1Y         rho=+0.130***  High risk-adjusted returns continue
      Return_Consistency rho=+0.115*** Consistent returners outperform (corrected!)
      Return_1mo        rho=+0.061***  Weak but positive continuation

    KEY FINDING: Vol_1Y FLIPPED from +0.070 to -0.197 after survivorship
    correction. The apparent 'volatility premium' was ENTIRELY survivorship
    bias — high-vol stocks that went bankrupt (returning -100%) were invisible.
    """
    df = df.copy()

    n = len(df)
    if n == 0:
        return df

    def _pct_rank(series, ascending=True):
        """Percentile rank 0-1. NaN gets 0.5 (neutral)."""
        ranked = series.rank(pct=True, ascending=ascending, na_option='keep')
        return ranked.fillna(0.5)

    # ══════════════════════════════════════════════════════════════════
    # EMPIRICALLY VALIDATED PRICE FACTORS (70% total weight)
    # Weights from survivorship-corrected validation (10,701 samples)
    # ══════════════════════════════════════════════════════════════════

    # ── Factor 1: Low Volatility (15% weight) ─────────────────────────
    # CORRECTED: Lower 1Y vol → higher 6mo returns. rho = -0.197 (STRONGEST).
    # The old +0.070 signal was 100% survivorship bias. High-vol stocks
    # that went bankrupt (returning -100%) were invisible in our cache.
    # Quintile: Q1(low vol)=+5.5%, Q3(mid)=+9.0%, Q5(high vol)=-30.0%
    low_vol = _pct_rank(df.get("Vol_1Y", pd.Series(np.nan, index=df.index)), ascending=False)

    # ── Factor 2: Max Drawdown (15% weight) ───────────────────────────
    # Less negative drawdowns → higher returns. rho = +0.181.
    # Quintile: Q5(shallow DD)=best, Q1(deep DD)=worst.
    shallow_dd = _pct_rank(df.get("Max_Drawdown_1Y", pd.Series(np.nan, index=df.index)), ascending=True)

    # ── Factor 3: 6-month momentum (15% weight) ──────────────────────
    # 6-month winners continue to outperform. rho = +0.180.
    # Quintile: Q1=-21.1%, Q3=+4.0%, Q5=+8.4% — strong monotonic.
    mom_6mo = _pct_rank(df.get("Return_6mo", pd.Series(np.nan, index=df.index)), ascending=True)

    # ── Factor 4: 12-2 month momentum (10% weight) ───────────────────
    # CORRECTED: Jegadeesh-Titman momentum IS alive! rho = +0.179.
    # Previously showed rho=+0.008 without dead stocks.
    mom_12_2 = _pct_rank(df.get("Return_2_12mo", pd.Series(np.nan, index=df.index)), ascending=True)

    # ── Factor 5: Above 200SMA (10% weight) ──────────────────────────
    # CORRECTED: Stocks ABOVE 200SMA continue higher. rho = +0.149.
    # Previously the 50SMA signal was -0.080 (below=better). After
    # correction, both SMA signals flip to positive (trend-following).
    above_sma200 = _pct_rank(df.get("Price_vs_200SMA", pd.Series(np.nan, index=df.index)), ascending=True)

    # ── Factor 6: Return Consistency (5% weight) ─────────────────────
    # CORRECTED: flip from -0.082 to +0.115. Consistent returners
    # outperform after dead stocks removed. Makes intuitive sense.
    consistency = _pct_rank(df.get("Return_Consistency", pd.Series(np.nan, index=df.index)), ascending=True)

    # ══════════════════════════════════════════════════════════════════
    # FUNDAMENTAL FACTORS (30% total weight — unvalidated regularizer)
    # ══════════════════════════════════════════════════════════════════
    # We can't validate these without point-in-time fundamental data,
    # but they serve as a reality check against pure price chasing.

    # ── Factor 6: Quality (15% weight) ─────────────────────────────────
    piotroski   = _pct_rank(df.get("Piotroski_Score", pd.Series(np.nan, index=df.index)))
    prof_margin = _pct_rank(df.get("Profit_Margin", pd.Series(np.nan, index=df.index)))
    earn_qual   = _pct_rank(df.get("Earnings_Quality", pd.Series(np.nan, index=df.index)))
    roe_rank    = _pct_rank(df.get("ROE", pd.Series(np.nan, index=df.index)))
    quality_score = 0.30 * piotroski + 0.30 * prof_margin + 0.20 * earn_qual + 0.20 * roe_rank

    # ── Factor 7: Value (10% weight) ───────────────────────────────────
    btm        = _pct_rank(df.get("Book_to_Market", pd.Series(np.nan, index=df.index)))
    low_pe     = _pct_rank(df.get("PE", pd.Series(np.nan, index=df.index)), ascending=False)
    fcf_margin = _pct_rank(df.get("FCF_Margin", pd.Series(np.nan, index=df.index)))
    dcf_ret    = _pct_rank(df.get("DCF_Implied_Return", pd.Series(np.nan, index=df.index)))
    value_score = 0.25 * btm + 0.25 * low_pe + 0.25 * fcf_margin + 0.25 * dcf_ret

    # ── Factor 8: Growth (5% weight) ──────────────────────────────────
    rev_growth  = _pct_rank(df.get("Revenue_Growth_Rate", pd.Series(np.nan, index=df.index)))
    earn_growth = _pct_rank(df.get("Earnings_Growth_Signal", pd.Series(np.nan, index=df.index)))
    growth_score = 0.50 * rev_growth + 0.50 * earn_growth

    # ══════════════════════════════════════════════════════════════════
    # COMPOSITE (weights from empirical validation)
    # ══════════════════════════════════════════════════════════════════
    composite = (0.15 * low_vol +          # low vol premium (corrected!)
                 0.15 * shallow_dd +      # shallow drawdowns
                 0.15 * mom_6mo +         # 6-month momentum
                 0.10 * mom_12_2 +        # 12-2mo momentum (corrected!)
                 0.10 * above_sma200 +    # trend following (corrected!)
                 0.05 * consistency +     # return consistency (corrected!)
                 0.15 * quality_score +   # fundamentals (unvalidated)
                 0.10 * value_score +     # value (unvalidated)
                 0.05 * growth_score)     # growth (unvalidated)

    # Convert composite rank (0-1) to a pseudo-predicted-return.
    # Wide range for meaningful differentiation in position sizing.
    factor_predicted = -0.20 + 0.70 * composite

    # ══════════════════════════════════════════════════════════════════
    # ML ENSEMBLE: Blend LightGBM walk-forward model with factor scores
    # ══════════════════════════════════════════════════════════════════
    # If a trained model exists (from train_model.py) and demonstrated
    # positive out-of-sample performance, blend its predictions with the
    # factor scores. The blend weight (alpha) is proportional to OOS IC.
    # If no model exists or OOS IC <= 0, use factor-only (safe fallback).
    ml_predicted = None
    ml_alpha = 0.0
    try:
        import pickle as _pkl
        _model_path = "data/trained_model.pkl"
        if os.path.exists(_model_path):
            with open(_model_path, "rb") as _f:
                _model_data = _pkl.load(_f)

            _model = _model_data["model"]
            _feat_names = _model_data["feature_names"]
            _oos_ic = _model_data.get("oos_ic", 0)
            _ic_hit = _model_data.get("ic_hit_rate", 0)
            _monthly_ic = _model_data.get("monthly_ic_mean", 0)

            # Gate: require BOTH pooled IC and month-by-month IC to be
            # meaningfully positive, and sign-correctness > 55%.
            # Pooled IC alone can be inflated by a few good months masking noise.
            if _oos_ic > 0.03 and _monthly_ic > 0.02 and _ic_hit > 0.55:
                # Build feature matrix matching training features
                _X = pd.DataFrame(index=df.index)
                for _fn in _feat_names:
                    if _fn in df.columns:
                        _X[_fn] = df[_fn].values
                    else:
                        _X[_fn] = np.nan  # missing features get NaN (LightGBM handles natively)

                # Cross-sectional neutralization must match training.
                # Training used sector-relative z-scoring (Score_Date ×
                # Sector bucket, with date-only fallback for <5-stock
                # buckets). Replay the same transform here so features
                # arrive at the model on the same scale.
                _neut = _model_data.get("neutralized_features", [])
                _sector_neutralized = _model_data.get("sector_neutralized", False)
                if _neut:
                    _cols = [c for c in _neut if c in _X.columns]
                    if _cols:
                        if _sector_neutralized and "Sector" in df.columns:
                            _sec = df["Sector"].fillna("Unknown").values
                            _tmp = _X[_cols].copy()
                            _tmp["__sector__"] = _sec
                            _bucket_size = _tmp.groupby("__sector__")[_cols[0]].transform("size")
                            _use_sector = _bucket_size >= 5

                            def _zscore_col(g):
                                _sd = float(g.std(ddof=0))
                                if _sd == 0 or np.isnan(_sd):
                                    return g - g.mean()
                                return (g - g.mean()) / _sd

                            # Sector-relative for big buckets
                            if _use_sector.any():
                                _X.loc[_use_sector.values, _cols] = (
                                    _tmp.loc[_use_sector.values]
                                        .groupby("__sector__")[_cols]
                                        .transform(_zscore_col)
                                        .values
                                )
                            # Date-only (here: global) fallback for small buckets
                            if (~_use_sector).any():
                                _mask = (~_use_sector).values
                                _mu = _X.loc[_mask, _cols].mean()
                                _sd = _X.loc[_mask, _cols].std(ddof=0).replace(0, np.nan)
                                _X.loc[_mask, _cols] = ((_X.loc[_mask, _cols] - _mu) / _sd).values
                            _X[_cols] = _X[_cols].fillna(0.0)
                            log.info(f"ML inference: sector-relative z-scored {len(_cols)} "
                                     f"features across {len(_X)} stocks "
                                     f"({int(_use_sector.sum())} in sector bucket, "
                                     f"{int((~_use_sector).sum())} global fallback)")
                        else:
                            _mu = _X[_cols].mean()
                            _sd = _X[_cols].std(ddof=0).replace(0, np.nan)
                            _X[_cols] = ((_X[_cols] - _mu) / _sd).fillna(0.0)
                            log.info(f"ML inference: z-scored {len(_cols)} features "
                                     f"across {len(_X)} stocks")

                _raw_lgb = _model.predict(_X)
                # NOTE: ElasticNet ensemble was tested and dropped OOS IC from 0.053 → 0.035
                # (EN is too weak to help at 50/50 blend). Using LightGBM-only predictions.
                ml_predicted = pd.Series(_raw_lgb, index=df.index)

                # Alpha: IC-proportional blend, capped at 0.6 (factor model always gets >=40%)
                ml_alpha = min(0.6, max(0.0, _oos_ic * 3.0))

                log.info(f"ML ensemble active: OOS IC={_oos_ic:.3f}, "
                         f"monthly IC mean={_monthly_ic:.3f}, "
                         f"hit rate={_ic_hit:.1%}, blend α={ml_alpha:.2f}")
            else:
                log.info(f"ML model found but weak OOS: IC={_oos_ic:.3f}, "
                         f"monthly IC={_monthly_ic:.3f}, hit={_ic_hit:.1%} — using factor-only")
    except Exception as _e:
        log.info(f"No ML model loaded ({_e}) — using factor-only scoring")

    # Final predicted return: blend factor + ML
    if ml_predicted is not None and ml_alpha > 0:
        # Normalize ML predictions to same scale as factor predictions
        _ml_rank = ml_predicted.rank(pct=True)
        _ml_scaled = -0.20 + 0.70 * _ml_rank
        df["Predicted_Return"] = (1 - ml_alpha) * factor_predicted + ml_alpha * _ml_scaled
        df["ML_Predicted_Return"] = _ml_scaled
        df["Factor_Predicted_Return"] = factor_predicted
        log.info(f"Ensemble blend: {ml_alpha:.0%} ML + {1-ml_alpha:.0%} factor")
    else:
        df["Predicted_Return"] = factor_predicted
        df["ML_Predicted_Return"] = np.nan
        df["Factor_Predicted_Return"] = factor_predicted

    # Store factor sub-scores for diagnostics
    df["Score_Composite"]  = composite       # primary composite
    df["Score_Quality"]    = quality_score   # quality-only view
    df["Score_LowVol"]     = low_vol         # low-vol view

    # Factor agreement: how aligned are the validated vs unvalidated signals?
    factor_ranks = np.column_stack([
        low_vol.values, shallow_dd.values,
        mom_6mo.values, quality_score.values
    ])
    pct_spread = factor_ranks.max(axis=1) - factor_ranks.min(axis=1)
    df["Model_Agreement"] = (1.0 - pct_spread).clip(0, 1)

    df["Forward_vs_Momentum_Gap"] = quality_score.values - low_vol.values

    log.info(f"Factor scoring: min={df['Predicted_Return'].min():.2f}  "
             f"max={df['Predicted_Return'].max():.2f}  "
             f"mean={df['Predicted_Return'].mean():.2f}")
    log.info(f"Factor agreement: mean={df['Model_Agreement'].mean():.2f}")

    # ── Composite ranking score ────────────────────────────────────────
    agree_clip = df["Model_Agreement"].clip(0.3, 1.0)
    df["Composite_Score"] = np.where(
        df["Predicted_Return"] >= 0,
        df["Predicted_Return"] * agree_clip,
        df["Predicted_Return"]
    )

    df.sort_values("Composite_Score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # -- Fundamental score --
    def _fundamental_score(row):
        score = 50.0
        roe   = row.get("ROE", np.nan)
        pm    = row.get("Profit_Margin", np.nan)
        de    = row.get("Debt_Equity", np.nan)
        fcf_m = row.get("FCF_Margin", np.nan)
        rev_g = row.get("Revenue_Growth_Rate", np.nan)
        eq    = row.get("Earnings_Quality", np.nan)
        if not np.isnan(roe):   score += np.clip(roe * 100, -30, 30)
        if not np.isnan(pm):    score += np.clip(pm  * 100, -20, 20)
        if not np.isnan(de):    score -= np.clip(de  * 5,     0, 25)
        if not np.isnan(fcf_m): score += np.clip(fcf_m * 80, -20, 20)
        if not np.isnan(rev_g): score += np.clip(rev_g * 60, -15, 20)
        if not np.isnan(eq):    score += np.clip(eq   * 50, -15, 15)
        return float(np.clip(score, 0, 100))

    df["Fundamental_Score"] = df.apply(_fundamental_score, axis=1)

    # -- Explain each stock --
    explanations = df.apply(explain_stock, axis=1, result_type="expand")
    explanations.columns = ["Green_Flags", "Red_Flags", "Confidence_Tier"]
    df = pd.concat([df, explanations], axis=1)

    log.info("Factor-based scoring complete.")
    return df


# ---------------------------------------------
# STEP 5 - ALLOCATE CAPITAL
# ---------------------------------------------
def compute_allocation_multiplier(row: pd.Series) -> tuple[float, str]:
    """
    Post-scoring position size adjustment for DATA QUALITY issues only.

    The factor scorer already accounts for volatility, momentum, leverage,
    and other stock characteristics. The only legitimate adjustment here is
    for missing or fabricated input data that the scorer cannot detect:

    1. Missing revenue: all margin/DCF features are fabricated → near-disqualify.
    2. Missing FCF/earnings quality: partial data gaps → mild discount.

    Everything else (leverage, history, momentum decay) is already captured
    by the factor scoring weights derived from survivorship-corrected validation.
    """
    multiplier = 1.0
    reasons    = []

    # -- Data quality: missing core financial data --
    fcf_m   = row.get("FCF_Margin",       np.nan)
    earn_q  = row.get("Earnings_Quality", np.nan)
    revenue = row.get("Revenue",          np.nan)

    revenue_missing = pd.isna(revenue) or revenue == 0
    fcf_missing   = pd.isna(fcf_m)
    eq_missing    = pd.isna(earn_q)

    if revenue_missing:
        multiplier *= 0.10
        reasons.append(
            "−90% NO REVENUE DATA: All margin and DCF features are fabricated — effectively disqualified."
        )
    elif fcf_missing and eq_missing:
        multiplier *= 0.80
        reasons.append(
            "−20% MISSING CASH FLOW DATA: Both FCF margin and earnings quality unavailable."
        )
    elif fcf_missing:
        multiplier *= 0.88
        reasons.append(
            "−12% MISSING FCF DATA: Free cash flow unavailable, DCF used assumed −15% FCF margin."
        )
    elif eq_missing:
        multiplier *= 0.92
        reasons.append(
            "−8% MISSING EARNINGS QUALITY: op_cf/net_income unavailable."
        )

    reason_str = "; ".join(reasons) if reasons else "No adjustment"
    return round(multiplier, 3), reason_str


def _compute_hrp_weights(investable: pd.DataFrame) -> np.ndarray:
    """
    Hierarchical Risk Parity (HRP) allocation (Lopez de Prado, 2016).

    Uses the correlation structure of stock features (momentum + vol) to
    cluster similar stocks together, then allocates inversely proportional
    to cluster variance. This maximises diversification without requiring
    a traditional (unstable) covariance matrix inversion.

    Falls back to equal weight if insufficient data for correlation.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    n = len(investable)
    if n <= 2:
        return np.ones(n) / n

    # Build correlation matrix from actual daily return co-movement (OHLC cache).
    # Cross-sectional feature similarity (previous approach) measures whether
    # two stocks have similar momentum/vol profiles, NOT whether their returns
    # actually co-move. HRP clustering needs true temporal correlation.
    tickers = investable["Ticker"].tolist() if "Ticker" in investable.columns else []
    if len(tickers) == n:
        corr = _ohlc_correlation_matrix(tickers, n)
    else:
        # Fallback: use feature-based proxy if Ticker column unavailable
        feat_cols = ["Return_6mo", "Return_1Y", "Return_3Y", "Return_5Y",
                     "Vol_1Y", "Vol_3Y", "Sharpe_1Y"]
        avail = [c for c in feat_cols if c in investable.columns]
        if len(avail) < 2:
            return np.ones(n) / n
        X = investable[avail].fillna(0).values
        corr = np.corrcoef(X)
        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)

    # Distance matrix: d = sqrt(0.5 * (1 - corr))
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, 1)

    # Hierarchical clustering
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method='single')
    sort_idx = leaves_list(link)

    # Quasi-diagonal variance allocation
    # Use per-stock vol as individual variance proxy
    vols = investable["Vol_1Y"].fillna(0.30).values
    var_vec = (vols * np.sqrt(PREDICTION_YEARS)) ** 2  # variance proxy scaled to horizon

    # Recursive bisection
    weights = np.ones(n)

    def _hrp_bisect(items):
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left, right = items[:mid], items[mid:]
        var_left  = sum(var_vec[i] for i in left) / max(len(left), 1)
        var_right = sum(var_vec[i] for i in right) / max(len(right), 1)
        total_var = var_left + var_right
        if total_var == 0:
            alpha = 0.5
        else:
            alpha = 1.0 - var_left / total_var  # allocate more to lower-variance cluster
        for i in left:
            weights[i] *= alpha
        for i in right:
            weights[i] *= (1.0 - alpha)
        _hrp_bisect(left)
        _hrp_bisect(right)

    _hrp_bisect(list(sort_idx))

    # Normalise
    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        weights = np.ones(n) / n

    return weights


def allocate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -- Pre-screen: compute multipliers on a wider candidate pool first --
    # This ensures the TOP_N selected are the top stocks AFTER penalties,
    # not the top stocks before penalties (which lets penalized stocks sneak in).
    # We evaluate the top 150 candidates, apply multipliers, re-rank, then take top TOP_N.
    candidates = df[(df["Predicted_Return"] > 0) & (df["Composite_Score"] > 0)].head(150).copy()

    # -- Quality gate: require at least ONE positive fundamental or momentum signal --
    # Without this, the model can fill the portfolio with stocks that are negative
    # on every observable metric (negative returns, negative margins, negative ROE)
    # and rely entirely on the factor score's PE-driven prediction. Require at least one of:
    #   positive 1Y return, positive 3Y return, positive profit margin, or positive ROE.
    _r1y = candidates.get("Return_1Y", pd.Series(np.nan, index=candidates.index))
    _r3y = candidates.get("Return_3Y", pd.Series(np.nan, index=candidates.index))
    _pm_gate  = candidates.get("Profit_Margin", pd.Series(np.nan, index=candidates.index))
    _roe_gate = candidates.get("ROE", pd.Series(np.nan, index=candidates.index))
    has_any_positive = (
        (_r1y > 0) | (_r3y > 0) | (_pm_gate > 0) | (_roe_gate > 0)
    )
    # Allow NaN-heavy stocks through (they might just have missing data)
    _all_negative = ~has_any_positive & (
        _r1y.notna() | _r3y.notna() | _pm_gate.notna() | _roe_gate.notna()
    )
    n_quality_filtered = _all_negative.sum()
    if n_quality_filtered > 0:
        filtered_tickers = candidates.loc[_all_negative, "Ticker"].tolist()
        log.warning(
            f"Quality gate: removing {n_quality_filtered} stocks with no positive signal "
            f"(negative on all of: 1Y return, 3Y return, profit margin, ROE): "
            f"{filtered_tickers[:15]}{'...' if n_quality_filtered > 15 else ''}"
        )
        candidates = candidates[~_all_negative].copy()

    # -- Data integrity filter: exclude stocks with impossible fundamental values --
    # yfinance returns corrupted data for certain asset types (crypto miners,
    # SPACs, royalty trusts) where EBITDA > Revenue or Profit_Margin > 1.0.
    # These are mathematical impossibilities for standard businesses and indicate
    # the model is scoring fabricated inputs, not real fundamentals.
    # Guard against missing columns (possible if all stocks had NaN for these fields)
    _pm   = candidates["Profit_Margin"] if "Profit_Margin" in candidates.columns else pd.Series(0.0, index=candidates.index)
    _ebit = candidates["EBITDA"]        if "EBITDA"        in candidates.columns else pd.Series(np.nan, index=candidates.index)
    _rev  = candidates["Revenue"]       if "Revenue"       in candidates.columns else pd.Series(np.nan, index=candidates.index)
    data_corrupt = (
        (_pm > 1.0) |
        (
            _ebit.notna() & _rev.notna() &
            (_rev > 0) & (_ebit > _rev)
        )
    )
    n_corrupt = data_corrupt.sum()
    if n_corrupt > 0:
        corrupt_tickers = candidates.loc[data_corrupt, "Ticker"].tolist()
        log.warning(f"Data integrity filter: removing {n_corrupt} stocks with impossible fundamentals: {corrupt_tickers}")
        candidates = candidates[~data_corrupt].copy()

    # -- Liquidity filter: exclude stocks with insufficient daily dollar volume --
    # A hedge fund would never invest in stocks that can't be traded at scale.
    # Stocks with ADV * Price < $500K daily dollar volume risk slippage,
    # wide spreads, and inability to exit positions at model prices.
    MIN_DAILY_DOLLAR_VOL = 500_000
    if "Avg_Daily_Volume" in candidates.columns and "Price" in candidates.columns:
        _adv = candidates["Avg_Daily_Volume"].fillna(0)
        _px  = candidates["Price"].fillna(0)
        dollar_vol = _adv * _px
        illiquid = dollar_vol < MIN_DAILY_DOLLAR_VOL
        n_illiquid = illiquid.sum()
        if n_illiquid > 0:
            illiquid_tickers = candidates.loc[illiquid, "Ticker"].tolist()
            log.warning(
                f"Liquidity filter: removing {n_illiquid} stocks with "
                f"<${MIN_DAILY_DOLLAR_VOL/1000:.0f}K daily dollar volume: "
                f"{illiquid_tickers[:10]}{'...' if n_illiquid > 10 else ''}"
            )
            candidates = candidates[~illiquid].copy()

    if candidates.empty:
        log.error("No stocks with positive predicted returns.")
        df["Allocation_USD"] = 0.0
        df["Allocation_Multiplier"] = np.nan
        df["Allocation_Reason"] = ""
        return df

    # Compute multipliers on the full candidate pool
    mult_results = candidates.apply(compute_allocation_multiplier, axis=1, result_type="expand")
    mult_results.columns = ["Allocation_Multiplier", "Allocation_Reason"]
    candidates["Allocation_Multiplier"] = mult_results["Allocation_Multiplier"]
    candidates["Allocation_Reason"]     = mult_results["Allocation_Reason"]

    # Re-rank by penalty-adjusted composite score
    candidates["_adj_composite"] = candidates["Composite_Score"] * candidates["Allocation_Multiplier"]
    candidates.sort_values("_adj_composite", ascending=False, inplace=True)

    # Now take the true top TOP_N after penalties.
    # No artificial industry cap — if the scoring says buy 11 gold miners
    # and the scoring is empirically validated, let it. Diversification
    # constraints were hurting returns per the user's explicit request.
    investable = candidates.head(TOP_N).copy()
    log.info(f"Post-penalty ranking: top {TOP_N} selected from {len(candidates)} candidates")

    # Log any stocks that dropped out due to penalties
    pre_tickers  = set(df[(df["Predicted_Return"] > 0)].head(TOP_N)["Ticker"])
    post_tickers = set(investable["Ticker"])
    dropped = pre_tickers - post_tickers
    added   = post_tickers - pre_tickers
    if dropped:
        log.info(f"  Dropped after penalty re-ranking: {sorted(dropped)}")
    if added:
        log.info(f"  Added after penalty re-ranking:   {sorted(added)}")

    # Log every stock that got penalised or boosted
    adjusted = investable[investable["Allocation_Multiplier"] != 1.0][
        ["Ticker", "Predicted_Return", "Allocation_Multiplier", "Allocation_Reason"]
    ].sort_values("Allocation_Multiplier")
    if not adjusted.empty:
        log.info(f"Allocation adjustments ({len(adjusted)} stocks):")
        for _, r in adjusted.iterrows():
            direction = "PENALISED" if r.Allocation_Multiplier < 1.0 else "BOOSTED"
            log.info(f"  {direction} {r.Ticker:<8} x{r.Allocation_Multiplier:.2f}  | {r.Allocation_Reason}")

    # -- Position sizing: equal weight, composite-score, or HRP --
    use_method = ALLOCATION_METHOD if not EQUAL_WEIGHT else "equal"
    if use_method == "equal":
        n_inv = len(investable)
        investable["_adj_weight"] = 1.0 / n_inv if n_inv > 0 else 0.0
        log.info(f"Position sizing: EQUAL WEIGHT ({n_inv} stocks x ${TOTAL_CAPITAL/max(n_inv,1):.0f} each)")
    elif use_method == "hrp":
        hrp_w = _compute_hrp_weights(investable)
        investable["_adj_weight"] = hrp_w
        log.info("Position sizing: HIERARCHICAL RISK PARITY (covariance-aware)")
    else:
        # Proportional to adjusted composite score.
        # Allocates more capital to higher-ranked stocks.
        investable["_adj_weight"] = (
            investable["Composite_Score"] * investable["Allocation_Multiplier"]
        ).clip(lower=0)
        log.info("Position sizing: COMPOSITE-SCORE PROPORTIONAL")

    total_adj = investable["_adj_weight"].sum()
    if total_adj == 0:
        log.error("All adjusted weights are zero after penalties. Check data quality.")
        df["Allocation_USD"] = 0.0
        return df

    # -- Cap at MAX_ALLOC_FRAC with iterative redistribution --
    weights = investable["_adj_weight"].values.copy()
    weights = weights / weights.sum()

    for _ in range(100):
        capped        = np.minimum(weights, MAX_ALLOC_FRAC)
        overflow      = weights.sum() - capped.sum()
        if overflow < 1e-9:
            break
        uncapped_mask = capped < MAX_ALLOC_FRAC
        if uncapped_mask.sum() == 0:
            break
        capped[uncapped_mask] += overflow * (
            capped[uncapped_mask] / capped[uncapped_mask].sum()
        )
        weights = capped.copy()

    investable["_weight"]       = weights
    investable["Allocation_USD"] = (investable["_weight"] * TOTAL_CAPITAL).round(2)

    # -- Drop below minimum, renormalise to exactly TOTAL_CAPITAL --
    investable.loc[investable["Allocation_USD"] < MIN_ALLOCATION, "Allocation_USD"] = 0.0
    active = investable["Allocation_USD"] > 0

    if active.sum() > 0:
        investable.loc[active, "Allocation_USD"] = (
            investable.loc[active, "Allocation_USD"]
            / investable.loc[active, "Allocation_USD"].sum()
            * TOTAL_CAPITAL
        ).round(2)
        residual = TOTAL_CAPITAL - investable["Allocation_USD"].sum()
        if residual != 0:
            top_idx = investable.loc[active, "Allocation_USD"].idxmax()
            investable.loc[top_idx, "Allocation_USD"] = round(
                investable.loc[top_idx, "Allocation_USD"] + residual, 2
            )

    investable.drop(columns=["_adj_weight", "_weight", "_adj_composite"], inplace=True)

    # -- Sector concentration cap --
    # Hard cap at 25% per sector for crash protection and diversification.
    # Tightened from 40% to 25% to reduce single-sector factor tilt (e.g.
    # banks / energy), which the backtest showed was dragging Sharpe in
    # growth-led regimes.
    MAX_SECTOR_FRAC = 0.25
    if "Sector" in investable.columns:
        sector_weights = investable.groupby("Sector")["Allocation_USD"].sum() / TOTAL_CAPITAL
        overweight_sectors = sector_weights[sector_weights > MAX_SECTOR_FRAC].index.tolist()
        for sector in overweight_sectors:
            sector_mask  = investable["Sector"] == sector
            other_mask   = ~sector_mask & (investable["Allocation_USD"] > 0)
            sector_total = investable.loc[sector_mask, "Allocation_USD"].sum()
            target_total = MAX_SECTOR_FRAC * TOTAL_CAPITAL
            scale_factor = target_total / sector_total
            freed        = sector_total - target_total
            investable.loc[sector_mask, "Allocation_USD"] *= scale_factor
            # Redistribute freed capital proportionally to other sectors
            if other_mask.sum() > 0 and freed > 0:
                other_total = investable.loc[other_mask, "Allocation_USD"].sum()
                investable.loc[other_mask, "Allocation_USD"] *= (1 + freed / other_total)
            log.info(
                f"Sector cap ({MAX_SECTOR_FRAC:.0%}): {sector} reduced from "
                f"{sector_total/TOTAL_CAPITAL:.1%} → {MAX_SECTOR_FRAC:.0%}, "
                f"${freed:.0f} redistributed to other sectors"
            )
        # Renormalise to exactly TOTAL_CAPITAL after sector capping
        active_mask = investable["Allocation_USD"] > 0
        if active_mask.sum() > 0:
            investable.loc[active_mask, "Allocation_USD"] = (
                investable.loc[active_mask, "Allocation_USD"]
                / investable.loc[active_mask, "Allocation_USD"].sum()
                * TOTAL_CAPITAL
            ).round(2)
            residual = TOTAL_CAPITAL - investable["Allocation_USD"].sum()
            if residual != 0:
                top_i = investable.loc[active_mask, "Allocation_USD"].idxmax()
                investable.loc[top_i, "Allocation_USD"] = round(
                    investable.loc[top_i, "Allocation_USD"] + residual, 2
                )

    # -- Merge back --
    df = df.merge(
        investable[["Ticker", "Allocation_USD", "Allocation_Multiplier", "Allocation_Reason"]],
        on="Ticker", how="left"
    )
    df["Allocation_USD"]        = df["Allocation_USD"].fillna(0.0)
    df["Allocation_Multiplier"] = df["Allocation_Multiplier"].fillna(np.nan)
    df["Allocation_Reason"]     = df["Allocation_Reason"].fillna("")

    n_pos = (df["Allocation_USD"] > 0).sum()
    n_vetoed = (investable["Allocation_Multiplier"] <= 0.2).sum()
    log.info(
        f"Portfolio: {n_pos} positions, total=${df['Allocation_USD'].sum():,.2f}, "
        f"{n_vetoed} stocks heavily penalised (<=0.2x)"
    )
    return df

# ---------------------------------------------
# STEP 6 - EXPORT CSV
# ---------------------------------------------

# ---------------------------------------------
# MOMENTUM BASELINE
# ---------------------------------------------
def build_momentum_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a simple momentum baseline portfolio for comparison:
      - Top-N stocks by 6mo return (pure price momentum, no factor scoring)
      - Equal weighted
      - Same sector cap as the factor portfolio
      - Same data integrity filter (no corrupt fundamentals)

    Purpose: if the factor portfolio does not outperform this over time,
    the multi-factor approach is not adding value beyond raw momentum.
    """
    # Data integrity filter (same as allocate())
    _pm2   = df["Profit_Margin"] if "Profit_Margin" in df.columns else pd.Series(0.0, index=df.index)
    _ebit2 = df["EBITDA"]        if "EBITDA"        in df.columns else pd.Series(np.nan, index=df.index)
    _rev2  = df["Revenue"]       if "Revenue"       in df.columns else pd.Series(np.nan, index=df.index)
    data_corrupt = (
        (_pm2 > 1.0) |
        (
            _ebit2.notna() & _rev2.notna() &
            (_rev2 > 0) & (_ebit2 > _rev2)
        )
    )
    clean = df[~data_corrupt & df["Return_6mo"].notna()].copy()

    # Rank by 6mo return -- pure momentum matching 6-month horizon
    clean = clean.sort_values("Return_6mo", ascending=False).head(TOP_N * 3).copy()

    # Apply sector cap: no sector > 40% of capital
    MAX_SECTOR_FRAC = 0.40
    selected = []
    sector_counts: dict = {}
    max_per_sector = max(1, int(TOP_N * MAX_SECTOR_FRAC))
    for _, row in clean.iterrows():
        sector = row.get("Sector", "Unknown")
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected.append(row)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) >= TOP_N:
            break

    baseline = pd.DataFrame(selected).copy()
    n = len(baseline)
    if n == 0:
        return baseline

    # Equal weight
    baseline["Baseline_Allocation_USD"] = round(TOTAL_CAPITAL / n, 2)
    # Fix rounding residual
    residual = TOTAL_CAPITAL - baseline["Baseline_Allocation_USD"].sum()
    if residual != 0:
        baseline.iloc[0, baseline.columns.get_loc("Baseline_Allocation_USD")] += residual

    baseline["Baseline_Rank"] = range(1, n + 1)

    # Overlap with factor portfolio
    factor_tickers  = set(df[df["Allocation_USD"] > 0]["Ticker"].tolist())
    base_tickers = set(baseline["Ticker"].tolist())
    overlap = factor_tickers & base_tickers

    log.info("=" * 60)
    log.info("MOMENTUM BASELINE (top-%d by 6mo return, equal weighted)", TOP_N)
    log.info("  Baseline tickers: %d", n)
    log.info("  Factor portfolio tickers: %d", len(factor_tickers))
    log.info("  Overlap (in both): %d/%d  -> %s", len(overlap), TOP_N, sorted(overlap))
    factor_only   = sorted(factor_tickers - base_tickers)
    base_only = sorted(base_tickers - factor_tickers)
    log.info("  Factor only (factors add these vs momentum): %s", factor_only)
    log.info("  Baseline only (momentum adds, factors skip): %s", base_only)
    log.info("  Interpretation: factor-only stocks reflect quality/value/low-vol")
    log.info("  signals diverging from pure momentum.")
    log.info("=" * 60)

    return baseline

def export(df: pd.DataFrame, path: str = "final_portfolio.xlsx", baseline: "pd.DataFrame | None" = None):
    """
    Write a 4-sheet Excel workbook:

    Sheet 1 – PORTFOLIO (50 allocated stocks)
      Columns flow left-to-right showing the ranking math chain:
      Rank | Identity | $Allocation | Predicted Return → ×Agreement → =Composite
           | ×Multiplier | Penalty Reason | Factor Sub-Scores | Fundamentals | Flags

    Sheet 2 – ALL RANKINGS (full universe, ranked by Composite Score)
      Condensed version of Sheet 1 for every stock screened.

    Sheet 3 – HOW SCORES WORK
      Plain-English explanation of the factor scoring formula.

    Sheet 4 – BASELINE COMPARISON
      Top-N stocks by pure 6mo momentum (equal weighted) alongside the
      factor portfolio. Used to evaluate whether multi-factor scoring adds value.
    """
    import xlsxwriter  # type: ignore

    if path.endswith(".csv"):
        path = path.replace(".csv", ".xlsx")

    wb = xlsxwriter.Workbook(path, {"nan_inf_to_errors": True})

    # ── colour palette ──────────────────────────────────────────────
    C = {
        "header_dark":  "#1F3864",   # dark navy  – section headers
        "header_mid":   "#2F5496",   # mid blue   – sub-headers
        "header_light": "#D6E4F7",   # pale blue  – column headers
        "row_alt":      "#F0F5FB",   # very pale blue – alternating rows
        "white":        "#FFFFFF",
        "green_bg":     "#E2EFDA",   # pale green – positive values
        "red_bg":       "#FFDEDE",   # pale red   – penalties / negatives
        "yellow_bg":    "#FFFACD",   # pale yellow – caution values
        "gold":         "#C9A227",   # gold text  – allocation $
        "dark_text":    "#1A1A2E",
        "grey_text":    "#5A5A72",
    }

    # ── shared formats ───────────────────────────────────────────────
    def fmt(**kw):
        base = {"font_name": "Calibri", "font_size": 10,
                "valign": "vcenter", "text_wrap": False}
        base.update(kw)
        return wb.add_format(base)

    F = {
        # headers
        "sec_hdr":    fmt(bold=True, font_size=11, font_color=C["white"],
                         bg_color=C["header_dark"], align="center", border=1),
        "col_hdr":    fmt(bold=True, font_color=C["dark_text"],
                         bg_color=C["header_light"], align="center",
                         border=1, text_wrap=True),
        # body – plain
        "body":       fmt(bg_color=C["white"],   border=1, align="center"),
        "body_alt":   fmt(bg_color=C["row_alt"], border=1, align="center"),
        "body_left":  fmt(bg_color=C["white"],   border=1, align="left"),
        "body_left_alt": fmt(bg_color=C["row_alt"], border=1, align="left"),
        # numbers
        "num":        fmt(bg_color=C["white"],   border=1, align="center",
                         num_format="0.00"),
        "num_alt":    fmt(bg_color=C["row_alt"], border=1, align="center",
                         num_format="0.00"),
        "mult":       fmt(bg_color=C["white"],   border=1, align="center",
                         num_format="0.00x"),
        "mult_alt":   fmt(bg_color=C["row_alt"], border=1, align="center",
                         num_format="0.00x"),
        "pct":        fmt(bg_color=C["white"],   border=1, align="center",
                         num_format="0%"),
        "pct_alt":    fmt(bg_color=C["row_alt"], border=1, align="center",
                         num_format="0%"),
        "usd":        fmt(bold=True, font_color=C["gold"],
                         bg_color=C["white"],   border=1, align="center",
                         num_format="$#,##0.00"),
        "usd_alt":    fmt(bold=True, font_color=C["gold"],
                         bg_color=C["row_alt"], border=1, align="center",
                         num_format="$#,##0.00"),
        "rank":       fmt(bold=True, font_size=11, bg_color=C["header_light"],
                         border=1, align="center"),
        # conditional overlays
        "green":      fmt(bg_color=C["green_bg"],  border=1, align="center",
                         num_format="0.00"),
        "red":        fmt(bg_color=C["red_bg"],    border=1, align="center",
                         num_format="0.00"),
        "yellow":     fmt(bg_color=C["yellow_bg"], border=1, align="center",
                         num_format="0.00"),
        "green_pct":  fmt(bg_color=C["green_bg"],  border=1, align="center",
                         num_format="0%"),
        "red_pct":    fmt(bg_color=C["red_bg"],    border=1, align="center",
                         num_format="0%"),
        "pen_red":    fmt(bold=True, font_color="#CC0000",
                         bg_color=C["red_bg"],    border=1, align="center",
                         num_format="0.00x"),
        "pen_none":   fmt(bg_color=C["green_bg"],  border=1, align="center",
                         num_format="0.00x"),
        "reason_left":fmt(bg_color=C["red_bg"],    border=1, align="left",
                         font_size=9, text_wrap=True),
        "reason_ok":  fmt(bg_color=C["green_bg"],  border=1, align="center",
                         font_size=9),
        "explain":    fmt(font_size=10, align="left", text_wrap=True,
                         valign="top"),
        "explain_hdr":fmt(bold=True, font_size=11, font_color=C["white"],
                         bg_color=C["header_mid"], align="left",
                         text_wrap=True, valign="vcenter"),
    }

    def g(r, field, default=None):
        """Safe field accessor."""
        v = r.get(field, default) if isinstance(r, dict) else getattr(r, field, default)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return v

    # ── pre-compute percentile scores for each model (full universe) ─
    df = df.copy()
    df["_pct_full"]  = df["Score_Composite"].rank(pct=True)
    df["_pct_fwd"]   = df["Score_Quality"].rank(pct=True)
    df["_pct_ridge"] = df["Score_LowVol"].rank(pct=True)

    # ── sort ─────────────────────────────────────────────────────────
    invested = df[df["Allocation_USD"] > 0].copy()
    invested.sort_values("Allocation_USD", ascending=False, inplace=True)
    invested.reset_index(drop=True, inplace=True)

    universe = df.copy()
    universe.sort_values("Composite_Score", ascending=False, inplace=True)
    universe.reset_index(drop=True, inplace=True)

    # ════════════════════════════════════════════════════════════════
    # SHEET 1 — PORTFOLIO
    # ════════════════════════════════════════════════════════════════
    ws1 = wb.add_worksheet("Portfolio")
    ws1.set_zoom(90)
    ws1.freeze_panes(3, 3)  # freeze rank+ticker+sector
    ws1.set_row(0, 22)
    ws1.set_row(1, 36)
    ws1.set_row(2, 36)

    # Column layout: (header, width, section)
    #   section codes: ID  ALLOC  RANK_MATH  MODELS  FUND  FLAGS
    COLS = [
        # Identity
        ("Rank",             5,   "ID"),
        ("Ticker",           8,   "ID"),
        ("Sector",           16,  "ID"),
        ("Price",            9,   "ID"),
        # Allocation
        ("Allocation $",     13,  "ALLOC"),
        # ── Ranking math chain ───────────────────────────────────────
        (f"Predicted\n{PREDICTION_YEARS}Y Return",  12, "RANK"),
        ("Model\nAgreement",      11, "RANK"),
        ("= Composite\nScore",    12, "RANK"),
        ("Weight\nAdj %",          9, "RANK"),
        ("Why Adjusted\n(plain English)", 52, "RANK"),
        # ── Factor sub-scores ───────────────────────────────────────
        ("Composite\n(raw)",        10, "MODELS"),
        ("Quality\n(raw)",          10, "MODELS"),
        ("Low Vol\n(raw)",          10, "MODELS"),
        ("Composite\nPct Rank",     10, "MODELS"),
        ("Quality\nPct Rank",       10, "MODELS"),
        ("Low Vol\nPct Rank",       10, "MODELS"),
        # ── Fundamentals ────────────────────────────────────────────
        ("DCF\nImplied Return",    12, "FUND"),
        ("Piotroski\nScore",       10, "FUND"),
        ("Earnings\nGrowth",       10, "FUND"),
        ("Return\nConsistency",    11, "FUND"),
        # ── Flags ───────────────────────────────────────────────────
        ("Confidence\nTier",       16, "FLAGS"),
        ("Green Flags",            28, "FLAGS"),
        ("Red Flags",              28, "FLAGS"),
    ]

    # Section header spans
    SECTIONS = [
        ("ID",     "Stock Identity",           C["header_dark"]),
        ("ALLOC",  "Capital Allocated",        "#7B2D8B"),   # purple
        ("RANK",   "← Ranking Math Chain  (read left → right: Predicted × Agreement = Composite × Multiplier = final weight)", C["header_mid"]),
        ("MODELS", "Factor Sub-Scores",  "#1A6B3C"),
        ("FUND",   "Fundamentals",             "#7A4100"),
        ("FLAGS",  "Analysis & Flags",         "#5A1A1A"),
    ]

    # Row 0: section merge headers
    col_idx = 0
    for sec_code, sec_label, sec_color in SECTIONS:
        span = sum(1 for _, _, s in COLS if s == sec_code)
        sec_fmt = wb.add_format({
            "bold": True, "font_name": "Calibri", "font_size": 10,
            "font_color": "#FFFFFF", "bg_color": sec_color,
            "align": "center", "valign": "vcenter", "border": 1,
        })
        ws1.merge_range(0, col_idx, 0, col_idx + span - 1, sec_label, sec_fmt)
        col_idx += span

    # Row 1: formula explainer for the ranking math section
    math_start = next(i for i, (_, _, s) in enumerate(COLS) if s == "RANK")
    math_span  = sum(1 for _, _, s in COLS if s == "RANK")
    explain_fmt = wb.add_format({
        "italic": True, "font_name": "Calibri", "font_size": 9,
        "font_color": "#1F3864", "bg_color": "#EBF3FB",
        "align": "center", "valign": "vcenter", "border": 1,
    })
    ws1.merge_range(1, math_start, 1, math_start + math_span - 1,
                    "Predicted Return  ×  Model Agreement  =  Composite Score   ×   Adj Multiplier  =  Allocation Weight",
                    explain_fmt)
    # blank out other row-1 cells in non-math sections
    blank_fmt = wb.add_format({"bg_color": "#FAFAFA", "border": 1})
    for i, (_, _, s) in enumerate(COLS):
        if s != "RANK":
            ws1.write(1, i, "", blank_fmt)

    # Row 2: column headers
    for i, (hdr, width, _) in enumerate(COLS):
        ws1.write(2, i, hdr, F["col_hdr"])
        ws1.set_column(i, i, width)

    # Data rows start at row 3
    DATA_ROW_START = 3

    for ri, row in invested.iterrows():
        r         = ri          # 0-based index
        excel_row = DATA_ROW_START + r
        alt       = (r % 2 == 1)
        b         = lambda f: F[f + ("_alt" if alt else "")]   # body fmt picker

        pred      = g(row, "Predicted_Return", 0.0)
        agree     = g(row, "Model_Agreement",      1.0)
        composite = g(row, "Composite_Score",      0.0)
        mult      = g(row, "Allocation_Multiplier", 1.0) or 1.0
        reason    = str(g(row, "Allocation_Reason", "") or "")
        alloc     = g(row, "Allocation_USD",        0.0)

        # Penalty % — shown as e.g. "−35%" or "None" for intuitive readability
        if abs(mult - 1.0) < 0.01:
            penalty_str = "None"
        else:
            pct_change = int(round((mult - 1.0) * 100))
            penalty_str = f"{pct_change:+d}%"   # e.g. "−35%" or "+15%"

        # Choose colour for composite (highlight the key ranking number)
        comp_fmt = (F["green"] if composite >= 5 else
                    F["yellow"] if composite >= 2 else
                    F["body"])
        if alt:
            comp_fmt = (F["green"] if composite >= 5 else
                        F["yellow"] if composite >= 2 else
                        F["body_alt"])

        # Penalty format — colour the adjustment cell
        mult_fmt  = F["pen_red"]  if mult < 0.95 else F["pen_none"]

        # Reason text colour
        reason_fmt = F["reason_left"] if mult < 0.95 else F["reason_ok"]

        ws1.write(excel_row, 0,  r + 1,                            F["rank"])
        ws1.write(excel_row, 1,  g(row, "Ticker", ""),             b("body"))
        ws1.write(excel_row, 2,  g(row, "Sector", ""),             b("body_left"))
        ws1.write(excel_row, 3,  g(row, "Price",  0.0),            b("num"))
        ws1.write(excel_row, 4,  alloc,                            F["usd"] if not alt else F["usd_alt"])
        ws1.write(excel_row, 5,  pred,                             b("mult"))
        ws1.write(excel_row, 6,  agree,                            b("pct"))
        ws1.write(excel_row, 7,  composite,                        comp_fmt)
        ws1.write(excel_row, 8,  penalty_str,                      mult_fmt)
        ws1.write(excel_row, 9,  reason or "No adjustment", reason_fmt)
        ws1.write(excel_row, 10, g(row, "Score_Composite",   0.0), b("num"))
        ws1.write(excel_row, 11, g(row, "Score_Quality",      0.0), b("num"))
        ws1.write(excel_row, 12, g(row, "Score_LowVol",       0.0), b("num"))
        ws1.write(excel_row, 13, g(row, "_pct_full",  0.0),        b("pct"))
        ws1.write(excel_row, 14, g(row, "_pct_fwd",   0.0),        b("pct"))
        ws1.write(excel_row, 15, g(row, "_pct_ridge", 0.0),        b("pct"))

        # DCF — red if deeply negative, green if positive
        dcf = g(row, "DCF_Implied_Return", None)
        if dcf is not None:
            dcf_fmt = (F["green"] if dcf > 0.05 else
                       F["red"]   if dcf < -0.50 else
                       b("num"))
            ws1.write(excel_row, 16, dcf, dcf_fmt)
        else:
            ws1.write(excel_row, 16, "N/A", b("body"))

        piotroski = g(row, "Piotroski_Score", None)
        if piotroski is not None:
            pio_fmt = (F["green"] if piotroski >= 0.75 else
                       F["yellow"] if piotroski >= 0.50 else
                       F["red"])
            ws1.write(excel_row, 17, piotroski, pio_fmt)
        else:
            ws1.write(excel_row, 17, "N/A", b("body"))

        ws1.write(excel_row, 18, g(row, "Earnings_Growth_Signal",  None) or "N/A", b("num"))
        ws1.write(excel_row, 19, g(row, "Return_Consistency", None) or "N/A", b("num"))
        ws1.write(excel_row, 20, g(row, "Confidence_Tier",    ""),             b("body_left"))
        ws1.write(excel_row, 21, g(row, "Green_Flags",        ""),             b("body_left"))
        ws1.write(excel_row, 22, g(row, "Red_Flags",          ""),             b("body_left"))
        ws1.set_row(excel_row, 18)

    # Totals row
    total_row = DATA_ROW_START + len(invested)
    total_fmt = wb.add_format({
        "bold": True, "font_name": "Calibri", "font_size": 10,
        "bg_color": C["header_dark"], "font_color": "#FFFFFF",
        "border": 2, "align": "center", "num_format": "$#,##0.00",
        "valign": "vcenter",
    })
    total_label_fmt = wb.add_format({
        "bold": True, "font_name": "Calibri", "font_size": 10,
        "bg_color": C["header_dark"], "font_color": "#FFFFFF",
        "border": 2, "align": "right", "valign": "vcenter",
    })
    ws1.merge_range(total_row, 0, total_row, 3, "TOTAL PORTFOLIO", total_label_fmt)
    ws1.write(total_row, 4, invested["Allocation_USD"].sum(), total_fmt)
    for c_idx in range(5, len(COLS)):
        ws1.write(total_row, c_idx, "", wb.add_format({"bg_color": C["header_dark"], "border": 2}))
    ws1.set_row(total_row, 20)

    # ════════════════════════════════════════════════════════════════
    # SHEET 2 — ALL RANKINGS
    # ════════════════════════════════════════════════════════════════
    ws2 = wb.add_worksheet("All Rankings")
    ws2.set_zoom(85)
    ws2.freeze_panes(1, 2)

    COLS2 = [
        ("Rank",                5),
        ("Ticker",              8),
        ("Sector",              15),
        (f"Predicted\n{PREDICTION_YEARS}Y Return",12),
        ("Model\nAgreement",    11),
        ("= Composite\nScore",  12),
        ("Adj\nMultiplier",     10),
        ("Allocation $",        12),
        ("Composite\nPct Rank", 10),
        ("Quality\nPct Rank",   10),
        ("Low Vol\nPct Rank",   10),
        ("DCF\nReturn",         10),
        ("Piotroski",            9),
        ("Confidence Tier",     15),
    ]
    for i, (hdr, width) in enumerate(COLS2):
        ws2.write(0, i, hdr, F["col_hdr"])
        ws2.set_column(i, i, width)
    ws2.set_row(0, 36)

    in_portfolio = set(invested["Ticker"].tolist())
    for ri, row in universe.iterrows():
        excel_row = ri + 1
        alt = (ri % 2 == 1)
        b   = lambda f: F[f + ("_alt" if alt else "")]
        alloc = g(row, "Allocation_USD", 0.0) or 0.0
        pred  = g(row, "Predicted_Return", 0.0)
        comp  = g(row, "Composite_Score", 0.0)
        mult  = g(row, "Allocation_Multiplier", 1.0) or 1.0

        # Highlight rows that are in the portfolio
        if g(row, "Ticker", "") in in_portfolio:
            row_bg = "#FFFDE7"  # pale gold = in portfolio
            rb = wb.add_format({"bg_color": row_bg, "border": 1,
                                 "align": "center", "font_name": "Calibri",
                                 "font_size": 10, "valign": "vcenter"})
            rb_left = wb.add_format({"bg_color": row_bg, "border": 1,
                                      "align": "left", "font_name": "Calibri",
                                      "font_size": 10, "valign": "vcenter"})
        else:
            rb = b("body"); rb_left = b("body_left")

        ws2.write(excel_row, 0,  ri + 1,                            b("body"))
        ws2.write(excel_row, 1,  g(row, "Ticker", ""),              rb_left)
        ws2.write(excel_row, 2,  g(row, "Sector", ""),              rb_left)
        ws2.write(excel_row, 3,  pred,                              wb.add_format({"bg_color":"#FFFDE7" if g(row,"Ticker","") in in_portfolio else (C["row_alt"] if alt else C["white"]),"border":1,"align":"center","font_name":"Calibri","font_size":10,"num_format":"0.00x"}))
        ws2.write(excel_row, 4,  g(row,"Model_Agreement",1.0),     rb)
        ws2.write(excel_row, 5,  comp,                              rb)
        ws2.write(excel_row, 6,  mult,                              rb)
        ws2.write(excel_row, 7,  alloc if alloc > 0 else "",        rb)
        ws2.write(excel_row, 8,  g(row,"_pct_full",0.0),            rb)
        ws2.write(excel_row, 9,  g(row,"_pct_fwd",0.0),             rb)
        ws2.write(excel_row, 10, g(row,"_pct_ridge",0.0),           rb)
        dcf = g(row, "DCF_Implied_Return", None)
        ws2.write(excel_row, 11, dcf if dcf is not None else "N/A", rb)
        ws2.write(excel_row, 12, g(row,"Piotroski_Score",None) or "N/A", rb)
        ws2.write(excel_row, 13, g(row,"Confidence_Tier",""),       rb_left)
        ws2.set_row(excel_row, 16)

    # ════════════════════════════════════════════════════════════════
    # SHEET 3 — HOW SCORES WORK
    # ════════════════════════════════════════════════════════════════
    ws3 = wb.add_worksheet("How Scores Work")
    ws3.set_column(0, 0, 28)
    ws3.set_column(1, 1, 90)

    content = [
        ("RANKING FORMULA", None),
        ("", "Every stock in the screened universe receives a Composite Score that determines both "
             "its rank and how much capital it receives. The scoring uses 9 empirically-validated "
             "factors with weights derived from survivorship-corrected validation."),
        ("Step 1 — Factor Score",
         f"Each stock is scored on 9 factors predicting {PREDICTION_YEARS}Y total return. "
         "Factor weights come from Spearman rank correlations against actual 6-month outcomes "
         "measured over 10,701 samples (including dead/delisted stocks) from 1996-2025:\n\n"
         "  Price factors (70% total weight):\n"
         "  • Low Volatility (15%) — lower 1Y vol → higher returns (rho = -0.197)\n"
         "  • Shallow Max Drawdown (15%) — less negative 1Y drawdown → higher returns (rho = +0.181)\n"
         "  • 6-Month Momentum (15%) — recent winners continue (rho = +0.180)\n"
         "  • 12-2mo Momentum (10%) — classic Jegadeesh-Titman signal (rho = +0.179)\n"
         "  • Above 200-day SMA (10%) — trend following (rho = +0.149)\n"
         "  • Return Consistency (5%) — steady compounders outperform (rho = +0.115)\n\n"
         "  Fundamental factors (30% total weight):\n"
         "  • Quality (15%) — Piotroski, profit margin, earnings quality, ROE\n"
         "  • Value (10%) — book-to-market, low PE, FCF margin, DCF implied return\n"
         "  • Growth (5%) — revenue growth, earnings growth"),
        ("Step 2 — × Factor Agreement",
         "Each stock is scored independently on multiple factor sub-groups (composite, quality, "
         "low-vol). If the sub-scores agree on a stock's rank, Agreement ≈ 1.0 and the "
         "Composite Score equals the Predicted Return. If they disagree, the Composite is "
         "discounted proportionally."),
        ("Step 3 — Weight Adj %",
         "After ranking, data quality corrections adjust each stock's allocation weight. "
         "'None' means the stock had complete data. Discounts are applied only for missing "
         "financial inputs:\n\n"
         "  • Revenue missing: −90% (all financial ratios are estimated — near-disqualification)\n"
         "  • Both FCF and earnings quality missing: −20%\n"
         "  • FCF only missing: −12%\n"
         "  • Earnings quality only missing: −8%\n\n"
         "NO penalties are applied for leverage, short history, or momentum patterns — these "
         "are already captured by the factor scoring weights."),
        ("= Composite Score",
         "Factor Score × Agreement = Composite Score. This is the primary ranking number. "
         "Higher Composite Score = higher allocation."),
        ("", ""),
        ("FACTOR SUB-SCORES (detail)", None),
        ("Composite (raw)",
         "The weighted blend of all 9 factors. This IS the primary score."),
        ("Quality (raw)",
         "Sub-score from fundamental quality factors only (Piotroski, profit margin, "
         "earnings quality, ROE). Useful for seeing how much fundamentals contributed."),
        ("Low Vol (raw)",
         "Sub-score from the low-volatility factor only. Stocks with lower 1Y volatility "
         "score higher here. KEY FINDING: the old 'volatility premium' was 100% survivorship "
         "bias — high-vol stocks that went bankrupt were invisible in historical data."),
        ("", ""),
        ("SURVIVORSHIP CORRECTION", None),
        ("Why this matters",
         "Standard backtests only include stocks that survived to today, making the stock market "
         "look better than it actually was. This factor scoring was validated against a universe "
         "that includes 109 recovered dead stock price histories + 950 synthetic dead stocks "
         "(using Shumway 1997 average delist return of -30%). Key signals that REVERSED after "
         "correction: Volatility (from +0.070 to -0.197), Return Consistency (from -0.082 to "
         "+0.115), 12-2mo Momentum (from +0.008 to +0.179)."),
        ("", ""),
        ("SECTOR CAP", None),
        ("40% maximum per sector",
         "No single sector can exceed 40% of total capital. Technology naturally dominates growth "
         "portfolios, but a 2000-style sector crash could wipe 40%+ of a concentrated portfolio. "
         "Capital freed by the cap is redistributed proportionally to underweight sectors."),
        ("", ""),
        ("ALL RANKINGS SHEET", None),
        ("Gold-highlighted rows",
         "Rows highlighted in gold on the All Rankings sheet are stocks that made it into "
         "the final 50-stock portfolio. You can use this sheet to see where any stock "
         "ranks in the full universe of 3,000+ screened companies."),
    ]

    HEADING_COLOR = C["header_mid"]
    hdr3_fmt = wb.add_format({
        "bold": True, "font_name": "Calibri", "font_size": 11,
        "font_color": "#FFFFFF", "bg_color": HEADING_COLOR,
        "border": 1, "valign": "vcenter", "align": "left",
        "text_wrap": False,
    })
    label_fmt = wb.add_format({
        "bold": True, "font_name": "Calibri", "font_size": 10,
        "bg_color": C["header_light"], "border": 1,
        "valign": "top", "align": "left", "text_wrap": True,
    })
    body3_fmt = wb.add_format({
        "font_name": "Calibri", "font_size": 10,
        "bg_color": C["white"], "border": 1,
        "valign": "top", "align": "left", "text_wrap": True,
    })
    body3_alt = wb.add_format({
        "font_name": "Calibri", "font_size": 10,
        "bg_color": C["row_alt"], "border": 1,
        "valign": "top", "align": "left", "text_wrap": True,
    })

    excel_r = 0
    alt_toggle = False
    for label, body in content:
        if body is None:  # section heading
            ws3.merge_range(excel_r, 0, excel_r, 1, label, hdr3_fmt)
            ws3.set_row(excel_r, 20)
            excel_r += 1
            alt_toggle = False
        elif label == "" and body == "":  # spacer
            ws3.set_row(excel_r, 8)
            excel_r += 1
        else:
            bf = body3_alt if alt_toggle else body3_fmt
            ws3.write(excel_r, 0, label, label_fmt)
            ws3.write(excel_r, 1, body,  bf)
            # Estimate row height from text length
            lines = max(body.count("\n") + 1, len(body) // 95 + 1)
            ws3.set_row(excel_r, max(18, lines * 14))
            excel_r += 1
            alt_toggle = not alt_toggle


    # ════════════════════════════════════════════════════════════════
    # SHEET 4 — BASELINE COMPARISON
    # ════════════════════════════════════════════════════════════════
    ws4 = wb.add_worksheet("Baseline Comparison")
    ws4.set_zoom(90)
    ws4.freeze_panes(1, 2)

    # Header
    hdr4_fmt = wb.add_format({
        "bold": True, "font_name": "Calibri", "font_size": 10,
        "font_color": "#FFFFFF", "bg_color": "#1A3C5E",
        "align": "center", "valign": "vcenter", "border": 1, "text_wrap": True,
    })
    intro_fmt = wb.add_format({
        "italic": True, "font_name": "Calibri", "font_size": 10,
        "font_color": "#1F3864", "bg_color": "#EBF3FB",
        "align": "left", "valign": "vcenter", "border": 1, "text_wrap": True,
    })
    ws4.merge_range(0, 0, 0, 8,
        "MOMENTUM BASELINE vs FACTOR PORTFOLIO  —  "
        "Top-50 by 6mo return (equal weighted) vs multi-factor ranked portfolio. "
        "If the factor portfolio does not outperform this baseline over time, "
        "the multi-factor approach is not adding value beyond simple momentum.",
        intro_fmt)
    ws4.set_row(0, 36)

    COLS4 = [
        ("Rank",           6),
        ("Ticker",         8),
        ("Sector",         16),
        ("3Y Return",      11),
        ("Baseline $",     12),
        ("In Factor Portfolio?", 10),
        ("Factor Rank",        9),
        ("Factor $",           10),
        ("Factor Predicted Return", 13),
    ]
    for i, (hdr, width) in enumerate(COLS4):
        ws4.write(1, i, hdr, hdr4_fmt)
        ws4.set_column(i, i, width)
    ws4.set_row(1, 30)

    ml_alloc = {r["Ticker"]: r for _, r in invested.iterrows()}
    ml_rank_map = {r["Ticker"]: int(ri+1) for ri, (_, r) in enumerate(invested.iterrows())}
    green4  = wb.add_format({"font_name":"Calibri","font_size":10,"bg_color":"#C6EFCE","border":1,"align":"center"})
    red4    = wb.add_format({"font_name":"Calibri","font_size":10,"bg_color":"#FFC7CE","border":1,"align":"center"})
    num4    = wb.add_format({"font_name":"Calibri","font_size":10,"border":1,"num_format":"0.00x","align":"center"})
    usd4    = wb.add_format({"font_name":"Calibri","font_size":10,"border":1,"num_format":"$#,##0.00","align":"center"})
    body4   = wb.add_format({"font_name":"Calibri","font_size":10,"border":1})
    rank4   = wb.add_format({"font_name":"Calibri","font_size":10,"border":1,"align":"center","bold":True})

    if baseline is not None and not baseline.empty:
        for ri, (_, row) in enumerate(baseline.iterrows()):
            er = ri + 2
            ticker  = row.get("Ticker","")
            in_ml   = ticker in ml_alloc
            in_ml_fmt = green4 if in_ml else red4
            in_ml_str = "YES" if in_ml else "NO"
            ml_r    = ml_rank_map.get(ticker, "")
            ml_usd  = ml_alloc[ticker]["Allocation_USD"] if in_ml else ""
            ml_pred = ml_alloc[ticker]["Predicted_Return"] if in_ml else ""

            ws4.write(er, 0, row.get("Baseline_Rank", ri+1), rank4)
            ws4.write(er, 1, ticker,                          body4)
            ws4.write(er, 2, row.get("Sector",""),            body4)
            ws4.write(er, 3, row.get("Return_3Y", ""),        num4)
            ws4.write(er, 4, row.get("Baseline_Allocation_USD",""), usd4)
            ws4.write(er, 5, in_ml_str,                       in_ml_fmt)
            ws4.write(er, 6, ml_r if ml_r else "—",           body4)
            ws4.write(er, 7, ml_usd if ml_usd != "" else "—", usd4 if ml_usd != "" else body4)
            ws4.write(er, 8, ml_pred if ml_pred != "" else "—", num4 if ml_pred != "" else body4)
            ws4.set_row(er, 16)

        # Summary stats
        summary_row = len(baseline) + 3
        n_overlap  = sum(1 for t in baseline["Ticker"] if t in ml_alloc)
        ws4.merge_range(summary_row, 0, summary_row, 8,
            f"OVERLAP: {n_overlap}/{len(baseline)} baseline stocks also appear in the factor portfolio.  "
            f"Factor-only stocks ({len(ml_alloc)-n_overlap}): where factor scoring diverges from "
            f"pure momentum — these reflect quality/value/low-vol signals.",
            intro_fmt)
        ws4.set_row(summary_row, 30)
    else:
        ws4.write(2, 0, "Baseline not computed (no data available)", body4)


    wb.close()
    log.info(f"Saved Excel workbook -> {path}  ({len(invested)} portfolio stocks, {len(universe)} total ranked)")

    # ── CSV export (mirrors key Excel sheets for easy programmatic access) ──
    csv_base = path.replace(".xlsx", "")
    portfolio_csv = csv_base + "_portfolio.csv"
    allranks_csv  = csv_base + "_all_rankings.csv"
    invested.to_csv(portfolio_csv, index=False)
    universe.to_csv(allranks_csv, index=False)
    log.info(f"Saved CSV files -> {portfolio_csv}, {allranks_csv}")
    if baseline is not None:
        baseline_csv = csv_base + "_baseline.csv"
        baseline.to_csv(baseline_csv, index=False)
        log.info(f"Saved CSV file  -> {baseline_csv}")

    # Console summary (unchanged)
    top5_lines = []
    for _, r in invested.head(5).iterrows():
        tier  = g(r, "Confidence_Tier", "N/A")
        green = g(r, "Green_Flags", "")
        red   = g(r, "Red_Flags",   "")
        top5_lines.append(
            f"    {r.Ticker:<8} ${r.Allocation_USD:>8.2f}  pred={r.Predicted_Return:.1%}  [{tier}]"
        )
        if green: top5_lines.append(f"      + {green}")
        if red:   top5_lines.append(f"      ! {red}")

    spec = invested[invested["Confidence_Tier"].str.startswith("SPEC", na=False)] \
           if "Confidence_Tier" in invested.columns else pd.DataFrame()

    log.info(
        f"\n{'='*60}\n"
        f"  Portfolio Summary\n"
        f"  Positions       : {len(invested)}\n"
        f"  Total allocated : ${invested['Allocation_USD'].sum():,.2f}\n"
        f"  Avg predicted return: {invested['Predicted_Return'].mean():.1%}\n"
        + (f"  WARNING: {len(spec)} SPECULATIVE-tier positions included\n" if len(spec) > 0 else "")
        + f"  Top 5:\n" + "\n".join(top5_lines)
        + f"\n{'='*60}"
    )
    if "Confidence_Tier" in invested.columns:
        tier_counts = invested["Confidence_Tier"].str.split(" - ").str[0].value_counts()
        log.info(f"  Confidence breakdown: {tier_counts.to_dict()}")




# ---------------------------------------------
# MARKET HEALTH & ENTRY TIMING ASSESSMENT
# ---------------------------------------------
def assess_market_health(df: pd.DataFrame, spy_close: pd.Series) -> dict:
    """
    Computes a Market Entry Risk score using five independent signals:

      1. SPY trend           -- price vs 200SMA, drawdown from ATH
      2. Market breadth      -- % of universe stocks above 200SMA / positive 1Y return
      3. VIX (fear gauge)    -- fetched live; >30 = stress, >40 = crisis
      4. Yield curve         -- 10Y minus 3-month spread; inversion = recession warning
      5. Valuation           -- median market PE vs historical norms

    Each signal contributes -2 to +2 points. Total score:
      >= +6  : GOOD TIME TO ENTER  (multiple tailwinds)
       +2..+4: NEUTRAL / SLIGHT TAILWIND
      -1..+1 : CAUTION -- mixed signals
      <= -2  : HIGH RISK -- consider waiting or phasing in slowly

    IMPORTANT: For this investment horizon, timing matters less than
    stock selection. Even entering at a market peak, a diversified portfolio recovers
    in >90% of historical cases. This signal is a CONTEXT flag, not a
    directive to delay a long-term investment.

    Recession timing note: yield curve inversion precedes recession by an
    empirically observed 6-24 month window (median ~14 months). We flag
    the inversion status but do not attempt to pinpoint the exact month.
    """
    result = {
        "signals": {},
        "score": 0,
        "verdict": "",
        "recession_risk": "",
        "entry_advice": "",
    }

    score = 0

    # ── 1. SPY TREND ──────────────────────────────────────────────────────────
    spy_signal = 0
    spy_notes  = []
    if len(spy_close) >= 200:
        spy_price  = float(spy_close.iloc[-1])
        spy_sma200 = float(spy_close.rolling(200).mean().iloc[-1])
        spy_ath    = float(spy_close.max())
        spy_dd     = spy_price / spy_ath - 1  # drawdown from all-time high

        if spy_price > spy_sma200:
            spy_signal += 1
            spy_notes.append(f"SPY above 200SMA ({spy_price/spy_sma200-1:+.1%})")
        else:
            spy_signal -= 1
            spy_notes.append(f"SPY below 200SMA ({spy_price/spy_sma200-1:.1%}) -- bearish trend")

        if spy_dd > -0.05:
            spy_signal += 1
            spy_notes.append(f"Near all-time high (drawdown {spy_dd:.1%})")
        elif spy_dd < -0.20:
            spy_signal -= 1
            spy_notes.append(f"In bear market territory (drawdown {spy_dd:.1%})")
        else:
            spy_notes.append(f"Moderate drawdown from ATH ({spy_dd:.1%})")

        spy_signal = max(-2, min(spy_signal, 2))
    else:
        spy_notes.append("Insufficient SPY history")

    result["signals"]["SPY_Trend"] = {"score": spy_signal, "notes": spy_notes}
    score += spy_signal

    # ── 2. MARKET BREADTH ────────────────────────────────────────────────────
    breadth_signal = 0
    breadth_notes  = []

    pct_up_1y  = (df["Return_1Y"]  > 0).sum() / df["Return_1Y"].notna().sum()  if df["Return_1Y"].notna().sum() > 0  else np.nan
    pct_up_6mo = (df["Return_6mo"] > 0).sum() / df["Return_6mo"].notna().sum() if df["Return_6mo"].notna().sum() > 0 else np.nan
    pct_up_200 = np.nan
    if "Price_vs_200SMA" in df.columns and df["Price_vs_200SMA"].notna().sum() > 100:
        pct_up_200 = (df["Price_vs_200SMA"] > 0).sum() / df["Price_vs_200SMA"].notna().sum()

    if not np.isnan(pct_up_1y):
        if pct_up_1y > 0.65:
            breadth_signal += 1
            breadth_notes.append(f"{pct_up_1y:.0%} of stocks positive 1Y (strong breadth)")
        elif pct_up_1y < 0.45:
            breadth_signal -= 1
            breadth_notes.append(f"Only {pct_up_1y:.0%} of stocks positive 1Y (weak breadth)")
        else:
            breadth_notes.append(f"{pct_up_1y:.0%} of stocks positive 1Y (neutral)")

    if not np.isnan(pct_up_200):
        if pct_up_200 > 0.60:
            breadth_signal += 1
            breadth_notes.append(f"{pct_up_200:.0%} of stocks above 200SMA")
        elif pct_up_200 < 0.40:
            breadth_signal -= 1
            breadth_notes.append(f"Only {pct_up_200:.0%} of stocks above 200SMA (deteriorating)")
    elif not np.isnan(pct_up_6mo):
        if pct_up_6mo > 0.55:
            breadth_signal += 1
            breadth_notes.append(f"{pct_up_6mo:.0%} of stocks positive 6mo")
        elif pct_up_6mo < 0.40:
            breadth_signal -= 1
            breadth_notes.append(f"Only {pct_up_6mo:.0%} of stocks positive 6mo")

    breadth_signal = max(-2, min(breadth_signal, 2))
    result["signals"]["Market_Breadth"] = {"score": breadth_signal, "notes": breadth_notes}
    score += breadth_signal

    # ── 3. VIX (FEAR GAUGE) ──────────────────────────────────────────────────
    vix_signal = 0
    vix_notes  = []
    vix_val    = np.nan
    try:
        vix_hist = yf.Ticker("^VIX").history(period="1mo")
        if not vix_hist.empty:
            vix_val = float(vix_hist["Close"].iloc[-1])

            if vix_val < 15:
                vix_signal = 2
                vix_notes.append(f"VIX={vix_val:.1f} -- very low fear (complacent, but bullish short-term)")
            elif vix_val < 20:
                vix_signal = 1
                vix_notes.append(f"VIX={vix_val:.1f} -- low fear, normal market conditions")
            elif vix_val < 30:
                vix_signal = 0
                vix_notes.append(f"VIX={vix_val:.1f} -- elevated uncertainty")
            elif vix_val < 40:
                vix_signal = -1
                vix_notes.append(f"VIX={vix_val:.1f} -- high fear / stress event underway")
            else:
                vix_signal = -2
                vix_notes.append(f"VIX={vix_val:.1f} -- crisis-level fear (historically a BUY signal for long-term, but painful short-term)")
    except Exception as e:
        vix_notes.append(f"VIX unavailable ({e})")

    result["signals"]["VIX"] = {"score": vix_signal, "notes": vix_notes, "value": vix_val}
    score += vix_signal

    # ── 4. YIELD CURVE (3-month vs 10Y) ────────────────────────────────────────
    # Uses ^IRX (13-week T-bill) as the short end, not the 2Y Treasury.
    # The 10Y minus 3mo spread is the Fed's preferred recession indicator and
    # has a strong historical inversion-to-recession record. The classic 2Y vs 10Y
    # spread is also well-known but the 3mo vs 10Y is more reliable empirically.
    yc_signal  = 0
    yc_notes   = []
    spread     = np.nan
    y10        = np.nan
    y3mo       = np.nan
    try:
        h10 = yf.Ticker("^TNX").history(period="3mo")  # 10Y yield (%)
        h2  = yf.Ticker("^IRX").history(period="3mo")  # 13-week T-bill (3-month end of curve)
        # ^IRX = 13-week T-bill yield. We use 10Y minus 3mo which is the Fed's
        # preferred leading indicator for recessions -- more reliable than 2Y vs 10Y.
        if not h10.empty:
            y10 = float(h10["Close"].iloc[-1])
        if not h2.empty:
            y3mo = float(h2["Close"].iloc[-1])

        if not np.isnan(y10) and not np.isnan(y3mo):
            spread = y10 - y3mo  # positive = normal curve, negative = inverted

            # Check inversion duration over the 3mo window
            months_inverted = 0.0
            if not h10.empty and not h2.empty:
                merged = pd.DataFrame({"y10": h10["Close"], "y3mo": h2["Close"]}).dropna()
                months_inverted = (merged["y10"] < merged["y3mo"]).sum() / 21

            if spread > 1.5:
                yc_signal = 2
                yc_notes.append(f"Yield curve steep (+{spread:.2f}%): expansion signal, low recession risk")
            elif spread > 0.25:
                yc_signal = 1
                yc_notes.append(f"Yield curve positive (+{spread:.2f}%): normal, moderate growth signal")
            elif spread > -0.25:
                yc_signal = 0
                yc_notes.append(f"Yield curve near-flat ({spread:+.2f}%): watch for inversion")
            elif spread > -1.0:
                yc_signal = -1
                yc_notes.append(f"Yield curve mildly inverted ({spread:.2f}%): inverted {months_inverted:.1f}mo of last 3mo -- recession risk elevated (~12-18mo lag)")
            else:
                yc_signal = -2
                yc_notes.append(f"Yield curve deeply inverted ({spread:.2f}%): inverted {months_inverted:.1f}mo of last 3mo -- historically precedes recession (median 14mo)")
        else:
            yc_notes.append("Yield data unavailable")
    except Exception as e:
        yc_notes.append(f"Yield curve unavailable ({e})")

    result["signals"]["Yield_Curve"] = {
        "score": yc_signal, "notes": yc_notes,
        "spread_pct": round(spread, 3) if not np.isnan(spread) else None,
        "10Y_yield":  round(y10, 3) if not np.isnan(y10) else None,
        "3mo_rate":   round(y3mo, 3) if not np.isnan(y3mo) else None,
    }
    score += yc_signal

    # ── 5. MARKET VALUATION (median PE across universe) ──────────────────────
    val_signal = 0
    val_notes  = []
    if "PE" in df.columns:
        valid_pe = df["PE"][(df["PE"] > 0) & (df["PE"] < 200)]
        if len(valid_pe) > 100:
            med_pe = float(valid_pe.median())
            # Historical median market PE ~15-18; elevated >22; stretched >28
            if med_pe < 15:
                val_signal = 2
                val_notes.append(f"Median PE={med_pe:.1f} -- historically cheap")
            elif med_pe < 20:
                val_signal = 1
                val_notes.append(f"Median PE={med_pe:.1f} -- fair value range")
            elif med_pe < 25:
                val_signal = 0
                val_notes.append(f"Median PE={med_pe:.1f} -- slightly elevated")
            elif med_pe < 35:
                val_signal = -1
                val_notes.append(f"Median PE={med_pe:.1f} -- expensive by historical standards")
            else:
                val_signal = -2
                val_notes.append(f"Median PE={med_pe:.1f} -- stretched valuation (tech-bubble range)")

    result["signals"]["Valuation"] = {"score": val_signal, "notes": val_notes}
    score += val_signal

    # ── COMPOSITE VERDICT ─────────────────────────────────────────────────────
    result["score"] = score

    if score >= 6:
        verdict = "STRONG ENTRY OPPORTUNITY"
        entry   = "Multiple indicators align favourably. Historical base rate for positive returns from this setup is high."
        rec_risk = "LOW -- no recession indicators flashing"
    elif score >= 3:
        verdict = "REASONABLE TIME TO ENTER"
        entry   = "More tailwinds than headwinds. A phased entry (e.g. 3 tranches over 6 months) reduces timing risk."
        rec_risk = "MODERATE -- some caution signals present"
    elif score >= 0:
        verdict = "NEUTRAL / MIXED SIGNALS"
        entry   = "No clear direction. Consider phasing in over 6-12 months (dollar-cost averaging) rather than a single lump sum."
        rec_risk = "MODERATE -- watch yield curve and VIX"
    elif score >= -3:
        verdict = "ELEVATED RISK -- CONSIDER PHASING IN SLOWLY"
        entry   = "Multiple warning signals. A 12-month phased entry or waiting for VIX to normalize may improve outcomes. For this horizon, waiting up to 6 months rarely changes final outcome materially."
        rec_risk = "ELEVATED -- recession indicators present; typical lag is 6-18 months"
    else:
        verdict = "HIGH RISK ENTRY -- STRONG CAUTION"
        entry   = "Significant macro headwinds. Historically, lump-sum entry under these conditions leads to 2-3 year drawdowns before recovery. Dollar-cost average over 12-18 months if possible."
        rec_risk = "HIGH -- multiple recession signals; consider recession likely within 12-24 months"

    result["verdict"]      = verdict
    result["entry_advice"] = entry
    result["recession_risk"] = rec_risk

    return result


def log_market_health(health: dict):
    """Write the market health assessment to the log in a readable format."""
    score = health["score"]
    bar   = ("█" * max(0, score + 5)).ljust(10) if score >= 0 else ("░" * max(0, -score)).ljust(10)

    log.info("\n" + "="*60)
    log.info("  MARKET HEALTH ASSESSMENT")
    log.info("="*60)
    log.info(f"  Overall Score : {score:+d} / 10  [{bar}]")
    log.info(f"  Verdict       : {health['verdict']}")
    log.info(f"  Recession Risk: {health['recession_risk']}")
    log.info(f"  Entry Advice  : {health['entry_advice']}")
    log.info("-"*60)
    log.info("  Signal Breakdown:")
    for sig_name, sig_data in health["signals"].items():
        sig_score = sig_data.get("score", 0)
        indicator = "▲" if sig_score > 0 else ("▼" if sig_score < 0 else "─")
        log.info(f"    {indicator} {sig_name:<20} {sig_score:+d}pt")
        for note in sig_data.get("notes", []):
            log.info(f"        {note}")
    log.info("="*60)
    log.info("  NOTE: This assessment is context, not a directive.")
    log.info("  For a long-term horizon, time-in-market beats timing-the-market")
    log.info("  in the vast majority of historical rolling windows. Use this to inform")
    log.info("  HOW you enter (lump-sum vs phased), not WHETHER you invest.")
    log.info("="*60)


def _ohlc_correlation_matrix(tickers: list, n: int) -> np.ndarray:
    """
    Compute pairwise daily-return correlation from the OHLC price cache.

    Uses the most recent 252 trading days of each stock's daily returns,
    aligned to common dates, then applies Ledoit-Wolf-style shrinkage
    toward the identity matrix to ensure positive-definiteness and
    reduce estimation noise (the n/T ratio for 50 stocks / 252 days
    makes the raw sample matrix noisy).

    Falls back to identity if the cache is unavailable or data is sparse.
    """
    import pickle as _pkl
    _ohlc_path = "data/raw_ohlc_cache.pkl"
    C = np.eye(n)
    try:
        if not os.path.exists(_ohlc_path):
            return C
        with open(_ohlc_path, "rb") as _f:
            _ohlc = _pkl.load(_f)
        # Extract ~1Y of daily returns for each ticker
        _ret_series = {}
        for tk in tickers:
            cs = _ohlc.get(tk)
            if cs is not None and len(cs) >= 252:
                _ret_series[tk] = cs.iloc[-252:].pct_change().dropna()
        if len(_ret_series) < 2:
            return C
        # Align to common trading dates
        _ret_df = pd.DataFrame(_ret_series).dropna(how="all")
        if len(_ret_df) < 100:
            return C
        _pair_corr = _ret_df.corr()
        # Map into n×n matrix preserving ticker ordering
        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                if ti in _pair_corr.index and tj in _pair_corr.index:
                    val = _pair_corr.loc[ti, tj]
                    if not np.isnan(val):
                        C[i, j] = val
        # Shrink toward identity to ensure PSD and reduce estimation noise
        # (Jorion 1986 / Ledoit-Wolf style; 0.2 = moderate shrinkage)
        C = 0.2 * np.eye(n) + 0.8 * C
    except Exception:
        C = np.eye(n)
    return C


# ---------------------------------------------
# MONTE CARLO PORTFOLIO SIMULATION
# ---------------------------------------------
def monte_carlo_portfolio(df: pd.DataFrame, n_sims: int = 10_000,
                          spy_expected_return: float = 0.80):
    """
    Monte Carlo simulation of portfolio outcomes over the prediction horizon.

    For each simulation:
      - Sample each stock's return from a distribution parameterised by
        the factor prediction (mean) and historical volatility (std)
      - Correlate returns using a Cholesky decomposition of the historical
        correlation matrix (stocks don't move independently)
      - Compute portfolio return under the current allocation weights

    Reports: median return, 5th/95th percentile, VaR(5%), CVaR(5%),
    probability of loss, probability of beating SPY.
    """
    invested = df[df["Allocation_USD"] > 0].copy()
    if invested.empty:
        log.warning("Monte Carlo: no invested positions.")
        return

    n = len(invested)
    tickers = invested["Ticker"].values
    weights = invested["Allocation_USD"].values / invested["Allocation_USD"].sum()
    pred_returns = invested["Predicted_Return"].values.copy()

    # -- Bayesian shrinkage: when model R2 is low, shrink predictions toward
    # the market return. With R2=0, predictions are pure noise and should
    # collapse entirely to SPY expected return. With R2=1 (hypothetically
    # perfect), predictions are used as-is. This prevents the Monte Carlo
    # from treating unreliable predictions as ground truth.
    best_r2 = max(0.0, min(float(df.attrs.get("best_r2", 0.0)), 1.0))
    shrinkage = max(best_r2, 0.0)  # 0 = full shrinkage, 1 = no shrinkage
    pred_returns = spy_expected_return + shrinkage * (pred_returns - spy_expected_return)
    if shrinkage < 0.5:
        log.info(f"  Monte Carlo shrinkage: {shrinkage:.1%} model confidence -> "
                 f"predictions shrunk {1-shrinkage:.0%} toward SPY return")

    # Use historical 3Y vol annualised, scaled to prediction horizon, as std dev
    # for each stock's return distribution.
    vols = invested["Vol_3Y"].fillna(invested["Vol_1Y"]).fillna(0.30).values
    # Scale annual vol to prediction horizon
    sds = vols * np.sqrt(PREDICTION_YEARS)
    # Floor at 0.10 to avoid degenerate distributions
    sds = np.clip(sds, 0.10, 5.0)

    # Build correlation matrix from actual daily return co-movement (OHLC cache).
    # This replaces the previous cross-sectional feature proxy: knowing two stocks
    # have similar P/E and momentum says nothing about whether their daily returns
    # actually co-move. True temporal correlation is required for realistic
    # portfolio VaR / CVaR estimation.
    corr_matrix = _ohlc_correlation_matrix(invested["Ticker"].tolist(), n)

    # Cholesky decomposition for correlated sampling
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # Fall back to diagonal (independent) if Cholesky fails
        L = np.eye(n)

    rng = np.random.RandomState(RANDOM_STATE)

    # Simulate using LOG-NORMAL distribution (standard quant approach).
    # Stock returns are multiplicative: a stock can't lose more than 100%.
    # Log-normal naturally produces right-skewed returns and enforces r > -1.
    #   R = exp(mu + sigma * z) - 1
    # where mu, sigma are the log-return parameters derived from
    # the predicted return and historical vol.
    #
    # Convert from arithmetic (pred_return, vol) to log-return space:
    #   E[R] = exp(mu + sigma^2/2) - 1  =>  mu = log(1 + E[R]) - sigma^2/2
    log_sds = np.sqrt(np.log(1 + (sds / (1 + np.clip(pred_returns, -0.99, None)))**2))
    log_mus = np.log(np.clip(1 + pred_returns, 0.01, None)) - 0.5 * log_sds**2

    portfolio_returns = np.empty(n_sims)
    for i in range(n_sims):
        z = rng.randn(n)
        correlated_z = L @ z
        # Log-normal: exp(mu + sigma * z) - 1  (always > -1 by construction)
        sim_returns = np.exp(log_mus + log_sds * correlated_z) - 1.0
        # Clip extreme upside only (downside naturally bounded by -1)
        sim_returns = np.clip(sim_returns, -1.0, 50.0)
        portfolio_returns[i] = np.dot(weights, sim_returns)

    # Statistics
    median_ret  = float(np.median(portfolio_returns))
    p5          = float(np.percentile(portfolio_returns, 5))
    p25         = float(np.percentile(portfolio_returns, 25))
    p75         = float(np.percentile(portfolio_returns, 75))
    p95         = float(np.percentile(portfolio_returns, 95))
    var_5       = float(np.percentile(portfolio_returns, 5))   # VaR at 5%
    cvar_5      = float(portfolio_returns[portfolio_returns <= var_5].mean())  # CVaR
    prob_loss   = float((portfolio_returns < 0).mean())
    prob_beat_spy = float((portfolio_returns > spy_expected_return).mean())

    log.info("=" * 60)
    log.info(f"MONTE CARLO SIMULATION ({n_sims} paths, {PREDICTION_YEARS}Y horizon)")
    log.info(f"  Stocks: {n}, correlated sampling: yes (OHLC daily returns)")
    log.info(f"  Median portfolio return: {median_ret:.1%}")
    log.info(f"  5th / 25th / 75th / 95th percentile: "
             f"{p5:.1%} / {p25:.1%} / {p75:.1%} / {p95:.1%}")
    log.info(f"  VaR(5%%): {var_5:.1%}  CVaR(5%%): {cvar_5:.1%}")
    log.info(f"  Probability of loss ({PREDICTION_YEARS}Y): {prob_loss:.1%}")
    log.info(f"  Probability of beating SPY ({spy_expected_return:.0%} over {PREDICTION_YEARS}Y): {prob_beat_spy:.1%}")
    log.info("=" * 60)

    return {
        "n_sims": n_sims,
        "median_return": median_ret,
        "p5": p5, "p25": p25, "p75": p75, "p95": p95,
        "var_5": var_5, "cvar_5": cvar_5,
        "prob_loss": prob_loss,
        "prob_beat_spy": prob_beat_spy,
    }


# ---------------------------------------------
# UNIVERSE FILTER  (Root Fix #1)
# ---------------------------------------------
def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hard pre-scoring filters that remove non-investable stocks BEFORE any
    ranking or scoring happens.  These are not penalties — they are absolute
    exclusions.  A stock that fails here can never enter the portfolio.

    Rationale for each filter:
      1. Non-operating companies (CEFs, ETFs, shells, SPACs) report
         misleading financials (NAV changes as "profit", no real revenue).
      2. Extreme leverage (D/E > 5x, or >10x for REITs/Utilities) means
         the equity is a thin residual — small asset moves cause huge
         return swings. Gold miners at 30x D/E are not "quality" stocks.
      3. Short history (< 3 years) means momentum and volatility features
         are unreliable or entirely NaN.
      4. Micro-caps and illiquid stocks can't be traded at model prices.
      5. Stocks with impossible fundamentals (Profit_Margin > 100%) are
         data errors or non-standard entities.
    """
    n_start = len(df)

    # ── 1. Exclude non-operating companies by industry ─────────────────
    # These are investment vehicles, not operating businesses.
    EXCLUDED_INDUSTRIES = {
        "Shell Companies", "Blank Checks",
        "Closed-End Fund - Debt", "Closed-End Fund - Equity",
        "Closed-End Fund - Foreign", "Closed-End Fund - Diversified",
        "Exchange Traded Fund",
    }
    if "Industry" in df.columns:
        # Exact match
        ind_mask = df["Industry"].isin(EXCLUDED_INDUSTRIES)
        # Also catch CEFs by proxy: "Asset Management" with profit margin > 100%
        # (real asset managers like BLK have normal margins; CEFs report NAV changes)
        _pm = df.get("Profit_Margin", pd.Series(np.nan, index=df.index))
        cef_proxy = (df["Industry"] == "Asset Management") & (_pm > 1.0)
        exclude_ind = ind_mask | cef_proxy
        n_ind = exclude_ind.sum()
        if n_ind > 0:
            log.info(f"Universe filter: removed {n_ind} non-operating companies "
                     f"(CEFs/shells/ETFs): {df.loc[exclude_ind, 'Ticker'].tolist()[:20]}")
            df = df[~exclude_ind].copy()

    # ── 2. Impossible fundamentals ─────────────────────────────────────
    _pm = df.get("Profit_Margin", pd.Series(np.nan, index=df.index))
    _ebit = df.get("EBITDA", pd.Series(np.nan, index=df.index))
    _rev = df.get("Revenue", pd.Series(np.nan, index=df.index))
    bad_data = (_pm > 1.0) | (_ebit.notna() & _rev.notna() & (_rev > 0) & (_ebit > _rev))
    n_bad = bad_data.sum()
    if n_bad > 0:
        log.info(f"Universe filter: removed {n_bad} stocks with impossible financials "
                 f"(profit margin >100% or EBITDA>Revenue): {df.loc[bad_data, 'Ticker'].tolist()[:20]}")
        df = df[~bad_data].copy()

    # ── 3. Leverage cap ────────────────────────────────────────────────
    # D/E > 5x for normal companies, > 10x for REITs/Utilities (naturally leveraged).
    # Stocks with missing D/E pass through (we don't exclude for missing data).
    if "Debt_Equity" in df.columns:
        _de = df["Debt_Equity"]
        _sector = df.get("Sector", pd.Series("Unknown", index=df.index))
        is_leveraged_sector = _sector.isin({"Real Estate", "Utilities"})
        over_leveraged = (
            (_de.notna()) &
            (
                (~is_leveraged_sector & (_de > 5.0)) |
                (is_leveraged_sector & (_de > 10.0))
            )
        )
        n_lev = over_leveraged.sum()
        if n_lev > 0:
            log.info(f"Universe filter: removed {n_lev} over-leveraged stocks "
                     f"(D/E > 5x, or >10x for REITs/Utilities): "
                     f"{df.loc[over_leveraged, 'Ticker'].tolist()[:20]}{'...' if n_lev > 20 else ''}")
            df = df[~over_leveraged].copy()

    # ── 4. Minimum history: 3 years ───────────────────────────────────
    if "Has_3Y_History" in df.columns:
        short = df["Has_3Y_History"] != 1
        n_short = short.sum()
        if n_short > 0:
            log.info(f"Universe filter: removed {n_short} stocks with < 3 years of price history")
            df = df[~short].copy()
    elif "Days_History" in df.columns:
        short = df["Days_History"] < 756
        n_short = short.sum()
        if n_short > 0:
            log.info(f"Universe filter: removed {n_short} stocks with < 3 years of price history")
            df = df[~short].copy()

    # ── 5. Market cap floor: $1B ──────────────────────────────────────
    if "Market_Cap" in df.columns:
        small = df["Market_Cap"].fillna(0) < 1_000_000_000
        n_small = small.sum()
        if n_small > 0:
            log.info(f"Universe filter: removed {n_small} stocks with market cap < $1B")
            df = df[~small].copy()

    # ── 6. Liquidity: avg daily volume ≥ 100K shares ─────────────────
    if "Avg_Daily_Volume" in df.columns:
        illiquid = df["Avg_Daily_Volume"].fillna(0) < 100_000
        n_illiq = illiquid.sum()
        if n_illiq > 0:
            log.info(f"Universe filter: removed {n_illiq} illiquid stocks (ADV < 100K)")
            df = df[~illiquid].copy()

    # ── 7. Must have sector classification ────────────────────────────
    if "Sector" in df.columns:
        no_sector = df["Sector"].isna() | (df["Sector"] == "Unknown") | (df["Sector"] == "")
        n_nosec = no_sector.sum()
        if n_nosec > 0:
            log.info(f"Universe filter: removed {n_nosec} stocks with no sector classification")
            df = df[~no_sector].copy()

    df.reset_index(drop=True, inplace=True)
    log.info(f"Universe filter complete: {n_start} → {len(df)} stocks "
             f"({n_start - len(df)} removed)")
    return df


# ---------------------------------------------
# MAIN
# ---------------------------------------------
def main():
    # -- 1. Load tickers (auto-download if missing or stale) --
    ticker_file = "data/tickers.csv"
    need_download = False
    if not os.path.exists(ticker_file):
        log.info("No tickers.csv found — will download from NASDAQ.")
        need_download = True
    else:
        age_days = (time.time() - os.path.getmtime(ticker_file)) / 86400
        if age_days > TICKER_STALE_DAYS:
            log.info(f"tickers.csv is {age_days:.0f} days old (>{TICKER_STALE_DAYS}d) — refreshing.")
            need_download = True
    if need_download:
        if not download_tickers(ticker_file):
            if not os.path.exists(ticker_file):
                raise FileNotFoundError(
                    f"Could not download tickers and no existing file at '{ticker_file}'.")
            # Download failed but a stale file exists — warn proportionally to staleness
            stale_days = (time.time() - os.path.getmtime(ticker_file)) / 86400
            if stale_days > 30:
                log.error(
                    f"⚠  TICKER DOWNLOAD FAILED and existing tickers.csv is {stale_days:.0f} "
                    f"days old. Scoring may include DELISTED stocks and MISS new IPOs. "
                    f"Investigate the download failure before trusting portfolio output."
                )
            else:
                log.warning(
                    f"Ticker download failed — falling back to existing tickers.csv "
                    f"({stale_days:.0f} days old)."
                )
    tickers = load_tickers(ticker_file)

    # -- 2. Fetch data (or reuse recent cache) --
    cache_csv  = "raw_fetched_data.csv"
    ohlc_cache = "data/raw_ohlc_cache.pkl"
    reuse_cache = False

    # Invalidate CSV cache if tickers.csv is newer (user uploaded new tickers)
    if os.path.exists(cache_csv) and os.path.exists(ticker_file):
        if os.path.getmtime(ticker_file) > os.path.getmtime(cache_csv):
            log.info("tickers.csv is newer than cached fetch data — forcing re-fetch.")
        else:
            cache_age_hours = (time.time() - os.path.getmtime(cache_csv)) / 3600
            if cache_age_hours < 24:
                log.info(f"Reusing cached fetch data ({cache_csv}, {cache_age_hours:.1f}h old). "
                         "Delete it to force a fresh fetch.")
                df = pd.read_csv(cache_csv)
                reuse_cache = True
            else:
                log.info(f"Cached fetch data is {cache_age_hours:.0f}h old (>24h) — re-fetching.")

    if not reuse_cache:
        df, ohlc_dict = fetch_all(tickers)
        if df.empty:
            log.error("No data fetched. Check internet connection and ticker file.")
            return
        df.to_csv(cache_csv, index=False)
        log.info(f"Raw data saved to {cache_csv}")

        # -- Save OHLC cache for train_model.py and Monte Carlo correlation --
        import pickle as _pkl_ohlc
        # Load existing cache to preserve macro series and stocks not in current fetch
        existing_ohlc = {}
        if os.path.exists(ohlc_cache):
            try:
                with open(ohlc_cache, "rb") as _f:
                    existing_ohlc = _pkl_ohlc.load(_f)
            except Exception:
                pass
        # Update with freshly fetched close series
        existing_ohlc.update(ohlc_dict)
        # Fetch macro series if missing or stale (>7 days)
        macro_symbols = [
            ("^VIX", "__VIX__",  "VIX"),
            ("^TNX", "__TNX__",  "10Y Treasury"),
            ("^IRX", "__IRX__",  "3mo T-bill"),
        ]
        for symbol, key, desc in macro_symbols:
            need_macro = key not in existing_ohlc
            if not need_macro and hasattr(existing_ohlc[key], 'index') and len(existing_ohlc[key]) > 0:
                try:
                    last_date = pd.Timestamp(existing_ohlc[key].index[-1])
                    if (pd.Timestamp.now() - last_date).days > 7:
                        need_macro = True
                except Exception:
                    need_macro = True
            if need_macro:
                try:
                    h = yf.Ticker(symbol).history(period="max")["Close"].dropna()
                    if len(h) > 252:
                        existing_ohlc[key] = h
                        log.info(f"  Cached macro series: {desc} ({len(h)} days)")
                    time.sleep(0.08)
                except Exception as e:
                    log.warning(f"  Failed to cache {desc}: {e}")
        with open(ohlc_cache, "wb") as _f:
            _pkl_ohlc.dump(existing_ohlc, _f)
        n_stocks = sum(1 for k in existing_ohlc if not k.startswith("__"))
        log.info(f"OHLC cache updated: {n_stocks} stocks → {ohlc_cache}")

    # -- 2b. Auto-train ML model if missing or stale --
    model_path = "data/trained_model.pkl"
    need_training = False
    if not os.path.exists(model_path):
        need_training = True
        log.info("No trained model found — will auto-train.")
    elif os.path.exists(ohlc_cache) and os.path.getmtime(ohlc_cache) > os.path.getmtime(model_path):
        need_training = True
        log.info("OHLC cache is newer than trained model — will re-train.")
    if need_training and os.path.exists(ohlc_cache):
        log.info("Auto-training ML model (this may take a few minutes)...")
        try:
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "train_model.py"],
                capture_output=True, text=True, timeout=1800
            )
            if result.returncode == 0:
                log.info("ML model training completed successfully.")
            else:
                log.warning(f"ML model training failed (exit code {result.returncode}). "
                            f"Continuing with factor-only scoring.\n{result.stderr[-500:] if result.stderr else ''}")
        except Exception as e:
            log.warning(f"ML model training failed ({e}). Continuing with factor-only scoring.")

    # -- 3. Filter universe (hard exclusions BEFORE scoring) --
    df = filter_universe(df)

    # -- 4. Score stocks using multi-factor ranking --
    spy_expected_return = 0.08  # conservative default
    try:
        _, _spy_hist_df, _ = _fetch_with_retry("SPY")
        _spy_close = _spy_hist_df["Close"].dropna()
        _spy_ret = compute_return(_spy_close, PREDICTION_WINDOW)
        if not np.isnan(_spy_ret):
            spy_expected_return = _spy_ret
            log.info(f"SPY {PREDICTION_YEARS}Y return: {spy_expected_return:.1%}")
    except Exception as e:
        log.warning(f"SPY return calculation failed ({e}), using default {spy_expected_return:.0%}")

    df = factor_score_stocks(df, spy_expected_return)

    # -- 5. Allocate --
    df = allocate(df)

    # -- 5b. Momentum baseline --
    baseline = build_momentum_baseline(df)

    # -- 6. Export --
    export(df, path="final_portfolio.xlsx", baseline=baseline)

    # -- 6b. Monte Carlo simulation --
    try:
        df.attrs["best_r2"] = 0.5  # factor model confidence
        monte_carlo_portfolio(df, spy_expected_return=spy_expected_return)
    except Exception as e:
        log.warning(f"Monte Carlo simulation failed ({e}) -- portfolio output is unaffected.")

    # -- 7. Market health assessment --
    try:
        spy_close_health = yf.Ticker("SPY").history(period="max")["Close"].dropna()
        health = assess_market_health(df, spy_close_health)
        log_market_health(health)
    except Exception as e:
        log.warning(f"Market health assessment failed ({e}) -- portfolio output is unaffected.")

    log.info("Done.")


if __name__ == "__main__":
    main()

"""
backtest.py — Historical replay of the ML top-N strategy.

Reads data/oos_predictions.csv (produced by train_model.py) and simulates
"pick top N by predicted score each rebalance, equal-weight, hold 6 months"
across 2015–2025. Out-of-sample by construction (predictions come from
walk-forward folds that never saw the future).

Compares against:
  - SPY (6-month returns on same dates)
  - Equal-weight universe (average forward return that month)
  - Bottom-N (short-side of the signal)
  - Decile spread (top decile vs bottom decile)

Usage:  python backtest.py
"""
from __future__ import annotations
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backtest")

# ── Config ────────────────────────────────────────────────────────────
OOS_PATH     = "data/oos_predictions.csv"
OHLC_PATH    = "data/raw_ohlc_cache.pkl"
TOP_N        = 50           # match main.py TOP_N
FORWARD_DAYS = 126          # 6 trading months, matches train_model.FORWARD_DAYS
SEMI_MONTHS  = (1, 7)       # Jan & Jul semiannual rebalance


# ── Helpers ───────────────────────────────────────────────────────────
def load_oos() -> pd.DataFrame:
    df = pd.read_csv(OOS_PATH, parse_dates=["Score_Date"])
    df = df.dropna(subset=["y_true", "y_pred"])
    log.info(f"Loaded {len(df):,} OOS predictions across "
             f"{df['Score_Date'].nunique()} months, "
             f"{df['Ticker'].nunique()} tickers")
    return df


def load_spy() -> pd.Series | None:
    """Return SPY daily closes. Tries OHLC cache first, then yfinance."""
    if Path(OHLC_PATH).exists():
        with open(OHLC_PATH, "rb") as f:
            cache = pickle.load(f)
        spy = cache.get("SPY")
        if spy is not None and len(spy) > 252:
            return spy.sort_index()
    try:
        import yfinance as yf
        log.info("SPY not in OHLC cache — fetching from yfinance...")
        data = yf.Ticker("SPY").history(period="20y", auto_adjust=True)
        if data.empty:
            return None
        spy = data["Close"]
        spy.index = pd.to_datetime(spy.index).tz_localize(None)
        return spy.sort_index()
    except Exception as e:
        log.warning(f"yfinance SPY fetch failed: {e}")
        return None


def winsorize(x: pd.Series, pct: float = 0.01) -> pd.Series:
    lo, hi = x.quantile(pct), x.quantile(1 - pct)
    return x.clip(lo, hi)


def spy_forward_return(spy: pd.Series, score_date: pd.Timestamp, fwd_days: int) -> float:
    """SPY return from score_date (closest prior trading day) to +fwd_days."""
    past = spy[spy.index <= score_date]
    future = spy[spy.index > score_date]
    if past.empty or len(future) < fwd_days:
        return np.nan
    entry = past.iloc[-1]
    exit_ = future.iloc[min(fwd_days - 1, len(future) - 1)]
    return float(exit_ / entry - 1)


# ── Backtest 1: semiannual non-overlapping rebalance ──────────────────
def backtest_semiannual(oos: pd.DataFrame, spy: pd.Series | None) -> pd.DataFrame:
    rows = []
    dates = sorted(oos["Score_Date"].unique())
    dates = [d for d in dates if pd.Timestamp(d).month in SEMI_MONTHS]

    for d in dates:
        month = oos[oos["Score_Date"] == d].copy()
        if len(month) < TOP_N * 2:
            continue

        month = month.sort_values("y_pred", ascending=False)
        top = month.head(TOP_N)["y_true"].mean()
        bot = month.tail(TOP_N)["y_true"].mean()
        uni = month["y_true"].mean()

        # Decile spread
        month["decile"] = pd.qcut(month["y_pred"], 10, labels=False,
                                  duplicates="drop")
        dec_ret = month.groupby("decile")["y_true"].mean()

        spy_r = spy_forward_return(spy, pd.Timestamp(d), FORWARD_DAYS) \
                if spy is not None else np.nan

        rows.append({
            "Date":      pd.Timestamp(d),
            "Top_N":     top,
            "Bot_N":     bot,
            "Universe":  uni,
            "SPY":       spy_r,
            "D1":        dec_ret.get(0, np.nan),
            "D10":       dec_ret.get(9, np.nan),
            "Spread":    dec_ret.get(9, np.nan) - dec_ret.get(0, np.nan),
            "N_stocks":  len(month),
        })
    return pd.DataFrame(rows)


def compound_equity_curve(returns: pd.Series) -> pd.Series:
    """Given period returns, build cumulative equity (starts at 1.0)."""
    return (1 + returns.fillna(0)).cumprod()


def summary_stats(returns: pd.Series, periods_per_year: float = 2) -> dict:
    r = returns.dropna()
    if r.empty:
        return {}
    total = (1 + r).prod() - 1
    n_years = len(r) / periods_per_year
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = (r.mean() * periods_per_year) / ann_vol if ann_vol > 0 else np.nan
    eq = compound_equity_curve(r)
    dd = (eq / eq.cummax() - 1).min()
    hit = (r > 0).mean()
    return {
        "Total return":   f"{total:+.1%}",
        "CAGR":           f"{cagr:+.1%}",
        "Ann. vol":       f"{ann_vol:.1%}",
        "Sharpe":         f"{sharpe:.2f}",
        "Max drawdown":   f"{dd:.1%}",
        "Hit rate":       f"{hit:.1%}",
        "# Periods":      len(r),
    }


# ── Backtest 2: monthly overlapping (averaged across 6 sleeves) ───────
def backtest_monthly_overlapping(oos: pd.DataFrame) -> pd.DataFrame:
    """Each month picks top-N; realized 6mo return = mean y_true. This is
    overlapping so returns are autocorrelated — use for IC/spread study,
    not for compounding."""
    rows = []
    for d, month in oos.groupby("Score_Date"):
        if len(month) < TOP_N * 2:
            continue
        m = month.sort_values("y_pred", ascending=False)
        rows.append({
            "Date":     pd.Timestamp(d),
            "Top_N":    m.head(TOP_N)["y_true"].mean(),
            "Bot_N":    m.tail(TOP_N)["y_true"].mean(),
            "Universe": m["y_true"].mean(),
            "N":        len(m),
        })
    return pd.DataFrame(rows).sort_values("Date")


# ── Backtest 3: decile monotonicity ───────────────────────────────────
def decile_table(oos: pd.DataFrame) -> pd.DataFrame:
    """Pooled top-to-bottom decile realized 6mo returns.
    Shows both raw mean and winsorized mean (penny-stock bankruptcies
    can produce 100x rebounds that distort raw means)."""
    df = oos.copy()
    df["decile"] = df.groupby("Score_Date")["y_pred"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    # Winsorize y_true within each month at 1%/99% before aggregation
    df["y_true_w"] = df.groupby("Score_Date")["y_true"].transform(
        lambda x: winsorize(x, 0.01)
    )
    g = df.groupby("decile").agg(
        mean_raw=("y_true", "mean"),
        mean_wins=("y_true_w", "mean"),
        median=("y_true", "median"),
        count=("y_true", "count"),
    )
    g.index = [f"D{int(i)+1}" for i in g.index]
    return g


# ── Yearly table ──────────────────────────────────────────────────────
def yearly_returns(semi: pd.DataFrame) -> pd.DataFrame:
    s = semi.copy()
    s["Year"] = s["Date"].dt.year
    y = s.groupby("Year").apply(
        lambda g: pd.Series({
            "Top_N":    (1 + g["Top_N"]).prod() - 1,
            "SPY":      (1 + g["SPY"]).prod() - 1,
            "Excess":   (1 + g["Top_N"]).prod() - (1 + g["SPY"]).prod(),
        })
    )
    return y


# ── Main ──────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("HISTORICAL BACKTEST — ML top-N strategy, walk-forward OOS")
    log.info("=" * 70)

    oos = load_oos()
    spy = load_spy()
    if spy is not None:
        log.info(f"SPY series loaded: {spy.index.min().date()} → "
                 f"{spy.index.max().date()}")
    else:
        log.warning("SPY not found in OHLC cache — benchmark disabled")

    # ── Decile monotonicity ───────────────────────────────────────────
    log.info("\n" + "─" * 70)
    log.info("DECILE TABLE (pooled, all OOS months)")
    log.info("─" * 70)
    dt = decile_table(oos)
    log.info(f"  {'Decile':<6} {'mean_raw':>10} {'mean_wins':>10} "
             f"{'median':>10} {'count':>8}")
    for idx, row in dt.iterrows():
        log.info(f"  {idx:<6} {row['mean_raw']:>+10.2%} "
                 f"{row['mean_wins']:>+10.2%} "
                 f"{row['median']:>+10.2%} "
                 f"{int(row['count']):>8,}")
    spread_w = dt["mean_wins"].iloc[-1] - dt["mean_wins"].iloc[0]
    spread_m = dt["median"].iloc[-1]    - dt["median"].iloc[0]
    log.info(f"  D10 − D1 (winsorized mean): {spread_w:+.2%}")
    log.info(f"  D10 − D1 (median):          {spread_m:+.2%}")
    # Monotonicity on winsorized means
    means = dt["mean_wins"].values
    up_steps = sum(means[i+1] > means[i] for i in range(len(means) - 1))
    log.info(f"  Monotonic steps (wins): {up_steps}/{len(means)-1} "
             f"({'clean' if up_steps >= 7 else 'noisy'})")

    # ── Semiannual compounded backtest ────────────────────────────────
    log.info("\n" + "─" * 70)
    log.info(f"SEMIANNUAL REBALANCE (top {TOP_N}, equal-weight, Jan & Jul)")
    log.info("─" * 70)
    semi = backtest_semiannual(oos, spy)
    if semi.empty:
        log.error("No semiannual periods found.")
        return
    log.info(f"Periods: {len(semi)}  "
             f"({semi['Date'].min().date()} → {semi['Date'].max().date()})")

    eq_top = compound_equity_curve(semi["Top_N"])
    eq_spy = compound_equity_curve(semi["SPY"])
    eq_uni = compound_equity_curve(semi["Universe"])
    eq_bot = compound_equity_curve(semi["Bot_N"])

    log.info("")
    log.info(f"  {'Metric':<18} {'Top-N':>10} {'Universe':>10} "
             f"{'Bot-N':>10} {'SPY':>10}")
    log.info("  " + "─" * 62)
    for metric in ["Total return", "CAGR", "Ann. vol", "Sharpe",
                   "Max drawdown", "Hit rate", "# Periods"]:
        st = summary_stats(semi["Top_N"])
        su = summary_stats(semi["Universe"])
        sb = summary_stats(semi["Bot_N"])
        ss = summary_stats(semi["SPY"])
        log.info(f"  {metric:<18} {st.get(metric, ''):>10} "
                 f"{su.get(metric, ''):>10} "
                 f"{sb.get(metric, ''):>10} "
                 f"{ss.get(metric, ''):>10}")

    # ── Year-by-year ──────────────────────────────────────────────────
    log.info("\n" + "─" * 70)
    log.info("YEAR-BY-YEAR (compounded semiannual returns)")
    log.info("─" * 70)
    yr = yearly_returns(semi)
    log.info(f"  {'Year':<6} {'Top_N':>10} {'SPY':>10} {'Excess':>10}")
    for year, row in yr.iterrows():
        top_s = f"{row['Top_N']:+.1%}" if pd.notna(row['Top_N']) else "  n/a"
        spy_s = f"{row['SPY']:+.1%}"   if pd.notna(row['SPY'])   else "  n/a"
        exc_s = f"{row['Excess']:+.1%}" if pd.notna(row['Excess']) else "  n/a"
        log.info(f"  {int(year):<6} {top_s:>10} {spy_s:>10} {exc_s:>10}")

    # ── Monthly overlapping, for IC/turnover context ──────────────────
    log.info("\n" + "─" * 70)
    log.info("MONTHLY OVERLAPPING (stats only — do not compound)")
    log.info("─" * 70)
    mon = backtest_monthly_overlapping(oos)
    log.info(f"  Periods: {len(mon)}")
    log.info(f"  Top-N avg 6mo:     {mon['Top_N'].mean():+.2%}   "
             f"std {mon['Top_N'].std():.2%}")
    log.info(f"  Bot-N avg 6mo:     {mon['Bot_N'].mean():+.2%}   "
             f"std {mon['Bot_N'].std():.2%}")
    log.info(f"  Universe avg 6mo:  {mon['Universe'].mean():+.2%}")
    log.info(f"  Top − Universe:    {(mon['Top_N'] - mon['Universe']).mean():+.2%}")
    log.info(f"  Top − Bot:         {(mon['Top_N'] - mon['Bot_N']).mean():+.2%}")
    log.info(f"  Top-beats-Uni rate:{(mon['Top_N'] > mon['Universe']).mean():.1%}")

    # ── Save outputs ──────────────────────────────────────────────────
    out = Path("data")
    semi.to_csv(out / "backtest_semiannual.csv", index=False)
    mon.to_csv(out / "backtest_monthly.csv", index=False)
    dt.to_csv(out / "backtest_deciles.csv")
    log.info("")
    log.info("Saved: data/backtest_semiannual.csv, "
             "data/backtest_monthly.csv, data/backtest_deciles.csv")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
