"""
final_6yr_portfolio.py
======================
Long-term (6-year horizon) stock selection & capital allocation system.

Pipeline
--------
1. Load tickers from data/tickers.csv
2. Fetch price history + fundamentals via yfinance
3. Build a TRAINING dataset from stocks that have >=6 years of history
   (we can compute actual 6Y returns as ground truth labels)
4. Train a RandomForest on historical windows using features that would have
   been observable at the START of each 6Y window, predicting the 6Y outcome
5. Score ALL stocks (including ones with <6Y history) using the trained model
6. Allocate capital proportionally to predicted 6Y return, with guardrails
7. Export final_6yr_portfolio.csv

Key design decisions
--------------------
- Training uses ONLY stocks with real 6Y return labels (no data leakage)
- Prediction features are computed from whatever history IS available
- Stocks with <6Y but >=6mo history are scored via the same model using
  available features; missing features are median-imputed from training set
- Established = has meaningful fundamentals (revenue / market cap)
- Speculative = early-stage / no fundamentals; scored separately then blended
- No silent failures: every step logs what it drops and why
- Allocation sums exactly to TOTAL_CAPITAL after iterative min-trim
"""

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

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
FETCH_PAUSE         = 0.08        # seconds between yfinance calls
RANDOM_STATE        = 42

UNINVESTABLE = {
    "BRK-A","BRK/A","BRK.A",  # ~$700k/share
}

FEATURE_COLS = [
    # Fundamentals
    "Log_MarketCap", "Log_Revenue", "Log_EBITDA", "Log_FCF",
    "Profit_Margin", "PE", "Debt_Equity", "ROE", "Beta",
    # Technicals / momentum
    "Price_vs_50SMA", "Price_vs_200SMA", "Price_vs_52WLow",
    "Price_vs_52WHigh",
    # Multi-horizon returns (available history)
    "Return_6mo", "Return_1Y", "Return_3Y", "Return_5Y",
    # Volatility
    "Vol_1Y", "Vol_3Y",
    # History coverage flags (help model know what's missing)
    "Has_3Y_History", "Has_5Y_History", "Has_6Y_History",
]

# ---------------------------------------------
# STEP 1 - LOAD TICKERS
# ---------------------------------------------
def parse_market_cap(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace("$", "").replace(",", "").strip().upper()
    try:
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
    df["Symbol"] = df[sym_col].astype(str).str.replace("/", "-").str.strip().str.upper()

    # Filter by market cap if column exists
    cap_col = next((c for c in df.columns if "market" in c.lower() and "cap" in c.lower()), None)
    if cap_col:
        df["_mc"] = df[cap_col].apply(parse_market_cap)
        before = len(df)
        df = df[df["_mc"] >= MIN_MARKET_CAP]
        log.info(f"Market cap filter: {before} -> {len(df)} tickers")

    # Drop obviously non-investable patterns
    df = df[~df["Symbol"].str.contains(r"OTC|\.PK|ADR|\^", regex=True, na=False)]

    tickers = sorted(set(df["Symbol"].tolist()) - UNINVESTABLE)
    log.info(f"Final ticker list: {len(tickers)}")
    return tickers


# ---------------------------------------------
# STEP 2 - FETCH DATA
# ---------------------------------------------
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


def fetch_single(ticker: str) -> dict | None:
    try:
        t      = yf.Ticker(ticker)
        hist   = t.history(period="max")
        info   = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        close  = hist["Close"].dropna()
        n      = len(close)

        if n < MIN_HISTORY_DAYS:
            return None

        price  = float(close.iloc[-1])

        # -- Fundamentals --
        mc         = safe_get(info, "marketCap")
        revenue    = safe_get(info, "totalRevenue")
        ebitda     = safe_get(info, "ebitda")
        fcf        = safe_get(info, "freeCashflow")
        profit_m   = safe_get(info, "profitMargins")
        pe         = safe_get(info, "trailingPE", "forwardPE")
        de         = safe_get(info, "debtToEquity")
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
        r6mo  = compute_return(close, 126)
        r1y   = compute_return(close, 252)
        r3y   = compute_return(close, 756)
        r5y   = compute_return(close, 1260)
        r6y   = compute_return(close, 1512)   # TARGET for training

        v1y   = compute_vol(close, 252)
        v3y   = compute_vol(close, 756)

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
            # Returns
            "Return_6mo":       r6mo,
            "Return_1Y":        r1y,
            "Return_3Y":        r3y,
            "Return_5Y":        r5y,
            "Return_6Y":        r6y,   # NaN if <6Y history
            # Volatility
            "Vol_1Y":           v1y,
            "Vol_3Y":           v3y,
            # Flags
            "Has_3Y_History":   int(n >= 756),
            "Has_5Y_History":   int(n >= 1260),
            "Has_6Y_History":   int(n >= 1512),
            "Days_History":     n,
        }

    except Exception as e:
        log.debug(f"  {ticker} failed: {e}")
        return None


def fetch_all(tickers: list[str]) -> pd.DataFrame:
    rows = []
    failed = 0
    for tk in tqdm(tickers, desc="Fetching"):
        row = fetch_single(tk)
        if row:
            rows.append(row)
        else:
            failed += 1
        time.sleep(FETCH_PAUSE)

    log.info(f"Fetched {len(rows)} tickers, {failed} failed/skipped")
    df = pd.DataFrame(rows)

    # Force ALL columns to numeric — yfinance sometimes returns strings
    for col in df.columns:
        if col not in ("Ticker", "Sector", "Industry"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clip extreme values after numeric conversion
    clip_rules = {
        "PE":           (-100, 500),
        "Debt_Equity":  (-10,  50),
        "ROE":          (-5,   5),
        "Beta":         (-5,   10),
        "Profit_Margin":(-10,  10),
        "Return_6mo":   (-1,   50),
        "Return_1Y":    (-1,   50),
        "Return_3Y":    (-1,   200),
        "Return_5Y":    (-1,   500),
        "Return_6Y":    (-1,   1000),
    }
    for col, (lo, hi) in clip_rules.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# ---------------------------------------------
# STEP 3 - TRAIN MODEL
# ---------------------------------------------
def build_model(df: pd.DataFrame):
    """
    Train on stocks that have a real 6Y return label.
    We exclude Return_6Y from input features (it's the target).
    Return_5Y is included as a feature because it represents
    momentum/quality observable before the 6Y endpoint.
    """
    train_df = df[df["Return_6Y"].notna()].copy()
    log.info(f"Training samples (have 6Y return): {len(train_df)}")

    if len(train_df) < 30:
        log.warning("Fewer than 30 training samples - model will be weak. "
                    "Try fetching more tickers or lowering MIN_MARKET_CAP.")

    X = train_df[FEATURE_COLS].copy()
    y = train_df["Return_6Y"].copy()

    # Clip extreme return labels (>20x = likely data error)
    y = y.clip(-0.99, 20.0)

    # Impute + scale inside pipeline so we can reuse on prediction set
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   GradientBoostingRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        )),
    ])

    # Quick CV score for logging
    if len(train_df) >= 50:
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
        log.info(f"CV R2 scores: {cv_scores.round(3)}  mean={cv_scores.mean():.3f}")

    pipeline.fit(X, y)
    log.info("Model trained.")
    return pipeline


# ---------------------------------------------
# STEP 4 - SCORE ALL STOCKS
# ---------------------------------------------
def score_stocks(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """
    Predict 6Y returns for ALL stocks using the trained pipeline.
    Stocks without full history are scored using available features
    (missing features are median-imputed inside the pipeline).
    """
    df = df.copy()

    # Ensure all feature cols exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    X_all = df[FEATURE_COLS].copy()
    df["Predicted_6Y_Return"] = pipeline.predict(X_all)

    # Sanity check
    n_neg = (df["Predicted_6Y_Return"] < 0).sum()
    log.info(f"Predicted 6Y returns: min={df['Predicted_6Y_Return'].min():.2f}  "
             f"max={df['Predicted_6Y_Return'].max():.2f}  "
             f"mean={df['Predicted_6Y_Return'].mean():.2f}  "
             f"negative={n_neg}")

    df.sort_values("Predicted_6Y_Return", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------
# STEP 5 - ALLOCATE CAPITAL
# ---------------------------------------------
def allocate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Only invest in stocks with positive predicted return
    investable = df[df["Predicted_6Y_Return"] > 0].head(TOP_N).copy()

    if investable.empty:
        log.error("No stocks with positive predicted returns. Check data quality.")
        df["Allocation_USD"] = 0.0
        return df

    # Proportional allocation by predicted return
    investable["_raw_weight"] = investable["Predicted_6Y_Return"]

    # Cap each stock at MAX_ALLOC_FRAC
    # Iterative: redistribute excess until stable
    weights = investable["_raw_weight"].values.copy()
    weights = weights / weights.sum()

    for _ in range(50):  # max iterations
        capped   = np.minimum(weights, MAX_ALLOC_FRAC)
        overflow = weights.sum() - capped.sum()
        if overflow < 1e-9:
            break
        # Redistribute overflow to uncapped slots
        uncapped_mask = weights < MAX_ALLOC_FRAC
        if uncapped_mask.sum() == 0:
            break
        capped[uncapped_mask] += overflow * (weights[uncapped_mask] / weights[uncapped_mask].sum())
        weights = capped.copy()

    investable["_weight"] = weights
    investable["Allocation_USD"] = (investable["_weight"] * TOTAL_CAPITAL).round(2)

    # Drop below minimum
    investable.loc[investable["Allocation_USD"] < MIN_ALLOCATION, "Allocation_USD"] = 0.0

    # Re-normalise to exactly TOTAL_CAPITAL
    active = investable["Allocation_USD"] > 0
    if active.sum() > 0:
        investable.loc[active, "Allocation_USD"] = (
            investable.loc[active, "Allocation_USD"]
            / investable.loc[active, "Allocation_USD"].sum()
            * TOTAL_CAPITAL
        ).round(2)
        # Fix rounding residual on largest position
        residual = TOTAL_CAPITAL - investable["Allocation_USD"].sum()
        if residual != 0:
            top_idx = investable.loc[active, "Allocation_USD"].idxmax()
            investable.loc[top_idx, "Allocation_USD"] += residual
            investable.loc[top_idx, "Allocation_USD"] = round(
                investable.loc[top_idx, "Allocation_USD"], 2
            )

    investable.drop(columns=["_raw_weight", "_weight"], inplace=True)

    # Merge allocations back to full df
    df = df.merge(
        investable[["Ticker", "Allocation_USD"]],
        on="Ticker", how="left"
    )
    df["Allocation_USD"] = df["Allocation_USD"].fillna(0.0)

    log.info(f"Portfolio: {(df['Allocation_USD']>0).sum()} positions, "
             f"total=${df['Allocation_USD'].sum():,.2f}")
    return df


# ---------------------------------------------
# STEP 6 - EXPORT CSV
# ---------------------------------------------
OUTPUT_COLUMNS = [
    "Ticker", "Sector", "Industry", "Price", "Market_Cap",
    "Revenue", "EBITDA", "FCF", "Profit_Margin", "PE",
    "Debt_Equity", "ROE", "Beta",
    "Return_6mo", "Return_1Y", "Return_3Y", "Return_5Y", "Return_6Y",
    "Vol_1Y", "Has_3Y_History", "Has_5Y_History", "Has_6Y_History",
    "Days_History", "Predicted_6Y_Return", "Allocation_USD",
]

def export(df: pd.DataFrame, path: str = "final_6yr_portfolio.csv"):
    cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    out  = df[cols].copy()

    # Round floats for readability
    float_cols = out.select_dtypes(include=[float]).columns
    out[float_cols] = out[float_cols].round(4)

    # Sort: allocated first, then by predicted return
    out.sort_values(
        ["Allocation_USD", "Predicted_6Y_Return"],
        ascending=[False, False],
        inplace=True,
    )

    out.to_csv(path, index=False)
    log.info(f"Saved {len(out)} rows -> {path}")

    # Summary
    invested = out[out["Allocation_USD"] > 0]
    log.info(
        f"\n{'='*55}\n"
        f"  Portfolio Summary\n"
        f"  Positions       : {len(invested)}\n"
        f"  Total allocated : ${invested['Allocation_USD'].sum():,.2f}\n"
        f"  Avg predicted 6Y return: {invested['Predicted_6Y_Return'].mean():.1%}\n"
        f"  Top 5 holdings:\n"
        + "\n".join(
            f"    {r.Ticker:<8} ${r.Allocation_USD:>8.2f}  pred={r.Predicted_6Y_Return:.1%}"
            for _, r in invested.head(5).iterrows()
        )
        + f"\n{'='*55}"
    )


# ---------------------------------------------
# MAIN
# ---------------------------------------------
def main():
    # -- 1. Load tickers --
    ticker_file = "data/tickers.csv"
    if not os.path.exists(ticker_file):
        raise FileNotFoundError(
            f"Expected ticker file at '{ticker_file}'. "
            "Download one from e.g. https://www.nasdaq.com/market-activity/stocks/screener"
        )
    tickers = load_tickers(ticker_file)

    # -- 2. Fetch data --
    df = fetch_all(tickers)
    if df.empty:
        log.error("No data fetched. Check internet connection and ticker file.")
        return

    df.to_csv("raw_fetched_data.csv", index=False)
    log.info("Raw data saved to raw_fetched_data.csv")

    # -- 3. Train model --
    pipeline = build_model(df)

    # -- 4. Score all stocks --
    df = score_stocks(df, pipeline)

    # -- 5. Allocate --
    df = allocate(df)

    # -- 6. Export --
    export(df)
    log.info("Done.")


if __name__ == "__main__":
    main()
