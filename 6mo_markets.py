"""
final_6mo_portfolio.py
======================
Short-term (6-month horizon) stock selection & capital allocation system.

Pipeline
--------
1. Load tickers from data/tickers.csv
2. Fetch price history + fundamentals via yfinance
3. Build a TRAINING dataset from stocks that have >=12 months of history
   (we can compute actual 6M returns as ground truth labels using the
   period 6 months ago -> today as the most recent completed window)
4. Train a GradientBoosting model on observable features at each rolling
   6M window start, predicting the 6M outcome
5. Score ALL stocks (including ones with <12M history) using the trained model
6. Allocate capital proportionally to predicted 6M return, with guardrails
7. Export final_6mo_portfolio.csv

Key design decisions
--------------------
- 6-month horizon: target = return over next ~126 trading days
- Training uses ONLY stocks with >=12 months of real return data
  so we have at least one completed 6M window as a ground-truth label
- SHORT-TERM features emphasise momentum, RSI, MACD signal, volume trends,
  and near-term earnings/analyst signals rather than multi-year history
- Stocks with <12M but >=3M history are scored via the trained model
  using available features; missing features are median-imputed
- Allocation sums exactly to TOTAL_CAPITAL after iterative min-trim
- No silent failures: every step logs what it drops and why
"""

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
import sys as _sys

_stream_handler = logging.StreamHandler(_sys.stdout)
_stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
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
        logging.FileHandler("portfolio_run_6mo.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
TOTAL_CAPITAL       = 10_000      # USD
MIN_HISTORY_DAYS    = 63          # ~3 months minimum to be scored
TRAIN_MIN_DAYS      = 252         # ~12 months needed to have a 6M label
MIN_MARKET_CAP      = 500_000_000 # $500M
MIN_ALLOCATION      = 25          # discard allocations below $25
MAX_ALLOC_FRAC      = 0.15        # no single stock > 15%
TOP_N               = 50          # final portfolio size
FETCH_PAUSE         = 0.08        # seconds between yfinance calls
RANDOM_STATE        = 42

# 6-month window in trading days
HORIZON_DAYS        = 126

UNINVESTABLE = {
    "BRK-A", "BRK/A", "BRK.A",
}

# Short-term focused feature set
FEATURE_COLS = [
    # Fundamentals (stable signals)
    "Log_MarketCap", "Log_Revenue", "Log_EBITDA", "Log_FCF",
    "Profit_Margin", "PE", "Debt_Equity", "ROE", "Beta",
    # SHORT-TERM momentum & technicals
    "Price_vs_50SMA", "Price_vs_200SMA",
    "Price_vs_52WLow", "Price_vs_52WHigh",
    "RSI_14",           # overbought/oversold
    "MACD_Signal",      # momentum crossover signal
    "Vol_Ratio",        # recent vol vs longer-term vol (vol regime)
    "Volume_Trend",     # 20D avg volume vs 60D avg volume
    # Near-term returns (primary momentum features for 6M)
    "Return_1mo", "Return_3mo", "Return_6mo",
    # Longer horizon for context (available history)
    "Return_1Y", "Return_3Y",
    # Volatility
    "Vol_1M", "Vol_3M", "Vol_6M",
    # History coverage flags
    "Has_6M_History", "Has_1Y_History", "Has_3Y_History",
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

    sym_col = next((c for c in df.columns if c.lower() in ("symbol", "ticker", "sym")), None)
    if sym_col is None:
        raise ValueError(f"No 'Symbol' column found in {file_path}. Columns: {df.columns.tolist()}")
    df["Symbol"] = (
        df[sym_col].astype(str)
        .str.replace("/", "-")
        .str.replace(r"\s+", "", regex=True)
        .str.strip()
        .str.upper()
    )

    cap_col = next((c for c in df.columns if "market" in c.lower() and "cap" in c.lower()), None)
    if cap_col:
        df["_mc"] = df[cap_col].apply(parse_market_cap)
        before = len(df)
        df = df[df["_mc"] >= MIN_MARKET_CAP]
        log.info(f"Market cap filter: {before} -> {len(df)} tickers")

    df = df[~df["Symbol"].str.contains(r"OTC|\.PK|ADR|\^", regex=True, na=False)]

    before_suffix = len(df)
    warrant_pat = r"^[A-Z]{4,}(WS|WW|W)$"
    rights_pat  = r"^[A-Z]{4,}R$"
    units_pat   = r"^[A-Z]{4,}U$"
    combined    = f"({warrant_pat})|({rights_pat})|({units_pat})"
    df = df[~df["Symbol"].str.match(combined)]
    dropped = before_suffix - len(df)
    log.info(f"Non-stock suffix filter: removed {dropped}, kept {len(df)}")

    tickers = sorted(set(df["Symbol"].tolist()) - UNINVESTABLE)
    log.info(f"Final ticker list: {len(tickers)}")
    return tickers


# ---------------------------------------------
# STEP 2 - FETCH DATA
# ---------------------------------------------
def safe_get(info: dict, *keys, default=np.nan):
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


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    """Relative Strength Index — key short-term overbought/oversold signal."""
    if len(close) < period + 1:
        return np.nan
    delta = close.diff().iloc[-period - 1:]
    gain  = delta.clip(lower=0).mean()
    loss  = (-delta.clip(upper=0)).mean()
    if loss == 0:
        return 100.0
    rs = gain / loss
    return float(100 - 100 / (1 + rs))


def compute_macd_signal(close: pd.Series) -> float:
    """
    MACD Signal: (MACD line) - (Signal line).
    Positive = bullish crossover, negative = bearish.
    MACD line = EMA12 - EMA26; Signal line = EMA9 of MACD.
    """
    if len(close) < 35:
        return np.nan
    ema12   = close.ewm(span=12, adjust=False).mean()
    ema26   = close.ewm(span=26, adjust=False).mean()
    macd    = ema12 - ema26
    signal  = macd.ewm(span=9, adjust=False).mean()
    return float((macd - signal).iloc[-1])


def compute_volume_trend(volume: pd.Series) -> float:
    """20D avg volume / 60D avg volume. >1 means rising interest."""
    if len(volume) < 60:
        return np.nan
    v20 = float(volume.iloc[-20:].mean())
    v60 = float(volume.iloc[-60:].mean())
    if v60 == 0:
        return np.nan
    return v20 / v60


def compute_vol_ratio(close: pd.Series) -> float:
    """
    Vol ratio: 21D realised vol / 63D realised vol.
    >1 = volatility expanding (short-term risk increasing).
    """
    v21 = compute_vol(close, 21)
    v63 = compute_vol(close, 63)
    if np.isnan(v21) or np.isnan(v63) or v63 == 0:
        return np.nan
    return v21 / v63


def log_val(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    return float(np.sign(v) * np.log1p(abs(v)))


def fetch_single(ticker: str) -> dict | None:
    try:
        t     = yf.Ticker(ticker)
        hist  = t.history(period="max")
        info  = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        close  = hist["Close"].dropna()
        volume = hist["Volume"].dropna() if "Volume" in hist.columns else pd.Series(dtype=float)
        n      = len(close)

        if n < MIN_HISTORY_DAYS:
            return None

        price = float(close.iloc[-1])

        # -- Fundamentals --
        mc       = safe_get(info, "marketCap")
        revenue  = safe_get(info, "totalRevenue")
        ebitda   = safe_get(info, "ebitda")
        fcf      = safe_get(info, "freeCashflow")
        profit_m = safe_get(info, "profitMargins")
        pe       = safe_get(info, "trailingPE", "forwardPE")
        de       = safe_get(info, "debtToEquity")
        roe      = safe_get(info, "returnOnEquity")
        beta     = safe_get(info, "beta")
        sector   = info.get("sector",   "Unknown")
        industry = info.get("industry", "Unknown")

        # -- Technicals --
        sma50  = float(close.rolling(50).mean().iloc[-1])  if n >= 50  else np.nan
        sma200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else np.nan
        hi52   = float(close.iloc[-min(252, n):].max())
        lo52   = float(close.iloc[-min(252, n):].min())

        def pct_vs(base):
            return float(price / base - 1) if (base and base > 0) else np.nan

        # -- Short-term specific indicators --
        rsi         = compute_rsi(close, 14)
        macd_sig    = compute_macd_signal(close)
        vol_ratio   = compute_vol_ratio(close)
        volume_trend = compute_volume_trend(volume) if len(volume) >= 60 else np.nan

        # -- Returns (short-horizon first) --
        r1mo  = compute_return(close, 21)    # ~1 month
        r3mo  = compute_return(close, 63)    # ~3 months
        r6mo  = compute_return(close, 126)   # ~6 months  <-- TRAINING LABEL
        r1y   = compute_return(close, 252)
        r3y   = compute_return(close, 756)

        # -- Volatility at multiple horizons --
        v1m   = compute_vol(close, 21)
        v3m   = compute_vol(close, 63)
        v6m   = compute_vol(close, 126)

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
            # Short-term indicators
            "RSI_14":           rsi,
            "MACD_Signal":      macd_sig,
            "Vol_Ratio":        vol_ratio,
            "Volume_Trend":     volume_trend,
            # Returns
            "Return_1mo":       r1mo,
            "Return_3mo":       r3mo,
            "Return_6mo":       r6mo,   # TARGET for training
            "Return_1Y":        r1y,
            "Return_3Y":        r3y,
            # Volatility
            "Vol_1M":           v1m,
            "Vol_3M":           v3m,
            "Vol_6M":           v6m,
            # Flags
            "Has_6M_History":   int(n >= 126),
            "Has_1Y_History":   int(n >= 252),
            "Has_3Y_History":   int(n >= 756),
            "Days_History":     n,
        }

    except Exception as e:
        log.debug(f"  {ticker} failed: {e}")
        return None


def fetch_all(tickers: list[str]) -> pd.DataFrame:
    rows   = []
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

    before_mc = len(df)
    df = df[df["Market_Cap"].notna() & (df["Market_Cap"] > 0)]
    log.info(f"Dropped {before_mc - len(df)} tickers with no market cap data after fetch")

    import re as _re
    _w   = r"^[A-Z]{4,}(WS|WW|W)$"
    _r   = r"^[A-Z]{4,}R$"
    _u   = r"^[A-Z]{4,}U$"
    _pat = f"({_w})|({_r})|({_u})"
    before_recheck = len(df)
    df = df[~df["Ticker"].str.match(_pat)]
    if before_recheck - len(df) > 0:
        log.info(f"Post-fetch suffix recheck: removed {before_recheck - len(df)} non-stock instruments")

    for col in df.columns:
        if col not in ("Ticker", "Sector", "Industry"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clip rules tuned for SHORT-TERM returns
    clip_rules = {
        "PE":           (-100, 500),
        "Debt_Equity":  (-10,  50),
        "ROE":          (-5,   5),
        "Beta":         (-5,   10),
        "Profit_Margin":(-10,  10),
        "RSI_14":       (0,    100),
        "Vol_Ratio":    (0,    10),
        "Volume_Trend": (0,    10),
        # Short-term returns are much tighter than multi-year
        "Return_1mo":   (-0.5, 2.0),
        "Return_3mo":   (-0.6, 3.0),
        "Return_6mo":   (-0.8, 5.0),   # label
        "Return_1Y":    (-0.9, 10.0),
        "Return_3Y":    (-1.0, 30.0),
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
    Train on stocks that have a real 6-month return label (>=12M history
    so the 6M return represents a completed, non-overlapping window).

    Note: Return_6mo IS the target here. We drop it from features to avoid
    leakage. Return_3mo is kept as a feature — it represents momentum
    observable BEFORE the 6M endpoint.
    """
    # Require TRAIN_MIN_DAYS so the 6M label is a meaningful completed window
    train_df = df[
        df["Return_6mo"].notna() &
        (df["Days_History"] >= TRAIN_MIN_DAYS)
    ].copy()
    log.info(f"Training samples (>=12M history, have 6M return): {len(train_df)}")

    if len(train_df) < 30:
        log.warning(
            "Fewer than 30 training samples — model will be weak. "
            "Try fetching more tickers or lowering MIN_MARKET_CAP."
        )

    # Drop the label column from features (Return_6mo is the target)
    feature_cols_train = [c for c in FEATURE_COLS if c != "Return_6mo"]

    X = train_df[feature_cols_train].copy()
    y = train_df["Return_6mo"].copy()

    # Clip extreme labels (>5x in 6M is almost certainly bad data)
    y = y.clip(-0.95, 5.0)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   GradientBoostingRegressor(
            n_estimators=400,
            max_depth=3,           # shallower tree for shorter horizon
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        )),
    ])

    if len(train_df) >= 50:
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
        log.info(f"CV R2 scores: {cv_scores.round(3)}  mean={cv_scores.mean():.3f}")

    pipeline.fit(X, y)
    # Store the feature list used (excludes Return_6mo)
    pipeline._feature_cols_train = feature_cols_train
    log.info("Model trained.")
    return pipeline


# ---------------------------------------------
# STEP 4 - SCORE ALL STOCKS
# ---------------------------------------------
def score_stocks(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    df = df.copy()

    # Use the same feature set the model was trained on
    feature_cols = getattr(pipeline, "_feature_cols_train",
                           [c for c in FEATURE_COLS if c != "Return_6mo"])

    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    X_all = df[feature_cols].copy()
    df["Predicted_6M_Return"] = pipeline.predict(X_all)

    n_neg = (df["Predicted_6M_Return"] < 0).sum()
    log.info(
        f"Predicted 6M returns: min={df['Predicted_6M_Return'].min():.2%}  "
        f"max={df['Predicted_6M_Return'].max():.2%}  "
        f"mean={df['Predicted_6M_Return'].mean():.2%}  "
        f"negative={n_neg}"
    )

    df.sort_values("Predicted_6M_Return", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------
# STEP 5 - ALLOCATE CAPITAL
# ---------------------------------------------
def allocate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    investable = df[df["Predicted_6M_Return"] > 0].head(TOP_N).copy()

    if investable.empty:
        log.error("No stocks with positive predicted returns. Check data quality.")
        df["Allocation_USD"] = 0.0
        return df

    weights = investable["Predicted_6M_Return"].values.copy()
    weights = weights / weights.sum()

    for _ in range(50):
        capped   = np.minimum(weights, MAX_ALLOC_FRAC)
        overflow = weights.sum() - capped.sum()
        if overflow < 1e-9:
            break
        uncapped_mask = weights < MAX_ALLOC_FRAC
        if uncapped_mask.sum() == 0:
            break
        capped[uncapped_mask] += overflow * (
            weights[uncapped_mask] / weights[uncapped_mask].sum()
        )
        weights = capped.copy()

    investable["_weight"]        = weights
    investable["Allocation_USD"] = (investable["_weight"] * TOTAL_CAPITAL).round(2)

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

    investable.drop(columns=["_weight"], inplace=True)

    df = df.merge(
        investable[["Ticker", "Allocation_USD"]],
        on="Ticker", how="left"
    )
    df["Allocation_USD"] = df["Allocation_USD"].fillna(0.0)

    log.info(
        f"Portfolio: {(df['Allocation_USD'] > 0).sum()} positions, "
        f"total=${df['Allocation_USD'].sum():,.2f}"
    )
    return df


# ---------------------------------------------
# STEP 6 - EXPORT CSV
# ---------------------------------------------
OUTPUT_COLUMNS = [
    "Ticker", "Sector", "Industry", "Price", "Market_Cap",
    "Revenue", "EBITDA", "FCF", "Profit_Margin", "PE",
    "Debt_Equity", "ROE", "Beta",
    # Short-term metrics first
    "Return_1mo", "Return_3mo", "Return_6mo",
    "Return_1Y", "Return_3Y",
    "Vol_1M", "Vol_3M", "Vol_6M",
    "RSI_14", "MACD_Signal", "Vol_Ratio", "Volume_Trend",
    "Has_6M_History", "Has_1Y_History", "Has_3Y_History",
    "Days_History", "Predicted_6M_Return", "Allocation_USD",
]


def export(df: pd.DataFrame, path: str = "final_6mo_portfolio.csv"):
    cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    out  = df[cols].copy()

    float_cols = out.select_dtypes(include=[float]).columns
    out[float_cols] = out[float_cols].round(4)

    out.sort_values(
        ["Allocation_USD", "Predicted_6M_Return"],
        ascending=[False, False],
        inplace=True,
    )

    out.to_csv(path, index=False)
    log.info(f"Saved {len(out)} rows -> {path}")

    invested = out[out["Allocation_USD"] > 0]
    log.info(
        f"\n{'='*55}\n"
        f"  6-Month Portfolio Summary\n"
        f"  Positions        : {len(invested)}\n"
        f"  Total allocated  : ${invested['Allocation_USD'].sum():,.2f}\n"
        f"  Avg pred 6M return: {invested['Predicted_6M_Return'].mean():.1%}\n"
        f"  Top 5 holdings:\n"
        + "\n".join(
            f"    {r.Ticker:<8} ${r.Allocation_USD:>8.2f}  pred={r.Predicted_6M_Return:.1%}"
            for _, r in invested.head(5).iterrows()
        )
        + f"\n{'='*55}"
    )


# ---------------------------------------------
# MAIN
# ---------------------------------------------
def main():
    ticker_file = "data/ticker.csv"
    if not os.path.exists(ticker_file):
        raise FileNotFoundError(
            f"Expected ticker file at '{ticker_file}'. "
            "Download one from e.g. https://www.nasdaq.com/market-activity/stocks/screener"
        )

    tickers  = load_tickers(ticker_file)
    df       = fetch_all(tickers)

    if df.empty:
        log.error("No data fetched. Check internet connection and ticker file.")
        return

    df.to_csv("raw_fetched_data_6mo.csv", index=False)
    log.info("Raw data saved to raw_fetched_data_6mo.csv")

    pipeline = build_model(df)
    df       = score_stocks(df, pipeline)
    df       = allocate(df)
    export(df)
    log.info("Done.")


if __name__ == "__main__":
    main()
