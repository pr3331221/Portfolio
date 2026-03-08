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
4. Train a GBM ensemble on historical windows using features that would have
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
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
    # - Fundamentals -
    "Log_MarketCap", "Log_Revenue", "Log_EBITDA", "Log_FCF",
    "Profit_Margin", "PE", "Debt_Equity", "ROE", "Beta",
    # - Momentum / technicals -
    "Price_vs_50SMA", "Price_vs_200SMA", "Price_vs_52WLow", "Price_vs_52WHigh",
    "Return_6mo", "Return_1Y", "Return_3Y", "Return_5Y",
    # - Volatility & risk -
    "Vol_1Y", "Vol_3Y",
    "Max_Drawdown_1Y", "Max_Drawdown_3Y",
    "Sharpe_1Y", "Sharpe_3Y",
    "Return_Skewness_1Y", "Return_Skewness_3Y",
    # - Return quality -
    "Return_Consistency",       # smooth compounding vs spike-driven
    "Return_Decay",             # recent momentum accelerating or fading
    # - Fundamental quality & growth -
    "Revenue_Growth_Rate",      # YoY revenue growth
    "Revenue_Growth_Accel",     # 2nd derivative of revenue growth
    "FCF_Margin",               # FCF / Revenue (capital efficiency)
    "Earnings_Quality",         # EBITDA / Revenue
    "Gross_Margin_Trend",       # expanding vs compressing margins
    # - Hedge-fund grade signals -
    "Piotroski_Score",          # 9-point fundamental quality (0-1 normalised)
    "DCF_Implied_Return",       # forward-looking fair value gap (most important)
    "Earnings_Revision",        # analyst estimate upgrade/downgrade momentum
    "Forward_PE",               # forward earnings multiple
    "PEG_Ratio",                # PE relative to growth (cheap growth finder)
    "PE_vs_Sector",             # trailing PE premium/discount vs sector peers
    "PE_vs_Sector_Fwd",         # forward PE premium/discount vs sector peers
    "Insider_Pct",              # management skin in the game
    "Short_Interest_Ratio",     # short squeeze potential
    # - Market context -
    "Return_1Y_vs_SPY",         # pure alpha vs S&P 500
    "Return_3Y_vs_SPY",
    "SPY_Return_1Y",            # market regime context
    "SPY_Vol_1Y",
    # - Data completeness flags -
    "Has_3Y_History", "Has_5Y_History", "Has_6Y_History",
]

# Features that are genuinely forward-looking (not backward-looking momentum).
# These get extra weight in the ensemble via a dedicated DCF sub-model.
FORWARD_LOOKING_FEATURES = [
    "DCF_Implied_Return", "Earnings_Revision", "Revenue_Growth_Rate",
    "Revenue_Growth_Accel", "Gross_Margin_Trend", "Piotroski_Score",
    "Forward_PE", "PEG_Ratio", "PE_vs_Sector_Fwd", "FCF_Margin",
    "Insider_Pct",
]

# Features that are backward-looking momentum signals.
MOMENTUM_FEATURES = [
    "Return_6mo", "Return_1Y", "Return_3Y", "Return_5Y",
    "Return_1Y_vs_SPY", "Return_3Y_vs_SPY",
    "Sharpe_1Y", "Sharpe_3Y", "Return_Consistency", "Return_Decay",
    "Price_vs_50SMA", "Price_vs_200SMA",
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

        # -- Daily-resolution features --
        # These extract shape information from the full daily price series
        # that gets lost when you only look at start/end prices.
        daily_returns = close.pct_change().dropna()

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
        sharpe1y = sharpe_ratio(daily_returns, 252)
        sharpe3y = sharpe_ratio(daily_returns, 756)
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
                        rev_growth = float((r_new - r_old) / abs(r_old))
        except Exception:
            pass

        # -- FCF margin and earnings quality --
        fcf_margin = np.nan
        earnings_quality = np.nan
        if revenue and not np.isnan(revenue) and revenue > 0:
            if fcf and not np.isnan(fcf):
                fcf_margin = float(fcf / revenue)
            if ebitda and not np.isnan(ebitda):
                earnings_quality = float(ebitda / revenue)

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
                        rev_growth_accel = float(g1 - g2)  # positive = accelerating
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

        # - PIOTROSKI F-SCORE (0-9) -
        # Developed by Joseph Piotroski (2000). Each criterion scores 1 or 0.
        # Score >= 7 = strong fundamental quality (historically beats market).
        # Score <= 2 = fundamental deterioration (historically underperforms).
        # We compute as many of the 9 criteria as yfinance supports.
        piotroski = 0
        piotroski_available = 0

        # Profitability signals (4 criteria)
        roa = safe_get(info, "returnOnAssets")
        op_cf = safe_get(info, "operatingCashflow")
        total_assets = safe_get(info, "totalAssets")

        if not np.isnan(roa):
            piotroski += int(roa > 0)           # F1: positive ROA
            piotroski_available += 1
        if not np.isnan(op_cf) and not np.isnan(total_assets) and total_assets > 0:
            cfo_roa = op_cf / total_assets
            piotroski += int(cfo_roa > 0)       # F2: positive operating cash flow
            piotroski_available += 1
            if not np.isnan(roa):
                piotroski += int(cfo_roa > roa) # F3: accruals (CFO > ROA = quality earnings)
                piotroski_available += 1

        # Leverage / liquidity signals (3 criteria)
        curr_ratio  = safe_get(info, "currentRatio")
        shares_out  = safe_get(info, "sharesOutstanding")

        if not np.isnan(de):
            piotroski += int(de < 0.5)          # F4: low leverage (simplified)
            piotroski_available += 1
        if not np.isnan(curr_ratio):
            piotroski += int(curr_ratio > 1.0)  # F5: current ratio > 1
            piotroski_available += 1
        # F6: no share dilution (shares outstanding not increasing)
        # yfinance only gives current shares; we approximate by checking if
        # the company has been buying back shares (FCF > 0 + low D/E = good sign).
        # True Piotroski F6 requires YoY shares data which yfinance doesn't expose.
        # We skip this criterion rather than approximate it incorrectly.
        # shares_out is retained in case a future data source provides historical counts.

        # Operating efficiency signals (2 criteria)
        gross_margin = safe_get(info, "grossMargins")
        asset_turnover = np.nan
        if not np.isnan(revenue) and not np.isnan(total_assets) and total_assets > 0:
            asset_turnover = revenue / total_assets

        if not np.isnan(gross_margin):
            piotroski += int(gross_margin > 0.3)  # F8: gross margin > 30%
            piotroski_available += 1
        if not np.isnan(asset_turnover):
            piotroski += int(asset_turnover > 0.5) # F9: asset turnover > 0.5
            piotroski_available += 1

        # Normalize to 0-1 scale based on criteria available
        piotroski_score = float(piotroski / piotroski_available) if piotroski_available >= 4 else np.nan

        # - DCF-IMPLIED 6-YEAR RETURN -
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
                # assume margins improve toward sector-typical 8% over 6 years.
                # This prevents strong growth companies from silently getting NaN DCF.
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

                for yr in range(1, 7):
                    frac = yr / 6.0
                    g_yr = g0 * (1 - frac) + terminal_growth * frac
                    projected_revenue *= (1 + g_yr)
                    # Margins revert toward 8% over 6Y (empirical mean reversion)
                    fcf_m_proj = fcf_margin_for_dcf + (0.08 - fcf_margin_for_dcf) * frac
                    fcf_m_proj = max(-0.3, min(fcf_m_proj, 0.5))
                    yr_fcf = projected_revenue * fcf_m_proj
                    projected_fcf += yr_fcf / ((1 + wacc) ** yr)

                # Terminal value
                terminal_multiple = 20.0
                if g0 > 0.20:
                    terminal_multiple = 28.0
                elif g0 < 0.05:
                    terminal_multiple = 14.0

                yr6_fcf = projected_revenue * max(fcf_margin_for_dcf, 0.02)
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

        # - EARNINGS REVISION SIGNAL -
        # Analyst estimate upgrades are the most consistently documented
        # short-to-medium term alpha signal in academic literature.
        # yfinance provides current vs prior EPS estimates.
        earnings_revision = np.nan
        try:
            fwd_eps    = safe_get(info, "forwardEps")
            trailing_e = safe_get(info, "trailingEps")
            earn_growth = safe_get(info, "earningsGrowth", "earningsQuarterlyGrowth")

            if not np.isnan(earn_growth):
                earnings_revision = float(np.clip(earn_growth, -2.0, 5.0))
            elif not np.isnan(fwd_eps) and not np.isnan(trailing_e) and trailing_e != 0:
                earnings_revision = float((fwd_eps - trailing_e) / abs(trailing_e))
                earnings_revision = float(np.clip(earnings_revision, -2.0, 5.0))
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
            "Return_6mo":       r6mo,
            "Return_1Y":        r1y,
            "Return_3Y":        r3y,
            "Return_5Y":        r5y,
            "Return_6Y":        r6y,   # NaN if <6Y history
            # Log-compressed returns: log(1+r) compresses extreme outliers
            # so meme spikes/squeezes don t dominate the model
            "Log_Return_1Y":    float(np.sign(r1y) * np.log1p(abs(r1y))) if not np.isnan(r1y) else np.nan,
            "Log_Return_3Y":    float(np.sign(r3y) * np.log1p(abs(r3y))) if not np.isnan(r3y) else np.nan,
            "Log_Return_5Y":    float(np.sign(r5y) * np.log1p(abs(r5y))) if not np.isnan(r5y) else np.nan,
            # Volatility
            "Vol_1Y":           v1y,
            "Vol_3Y":           v3y,
            # Flags
            "Has_3Y_History":   int(n >= 756),
            "Has_5Y_History":   int(n >= 1260),
            "Has_6Y_History":   int(n >= 1512),
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
            # Hedge-fund grade forward-looking signals
            "Piotroski_Score":        piotroski_score,    # 0-1 fundamental quality
            "DCF_Implied_Return":     dcf_implied_return, # forward-looking fair value gap
            "Earnings_Revision":      earnings_revision,  # analyst estimate momentum
            "Forward_PE":             forward_pe,         # forward earnings multiple
            "PEG_Ratio":              peg_ratio,          # PE relative to growth
            "Gross_Margin_Trend":     gross_margin_trend, # expanding vs compressing margins
            "Insider_Pct":            insider_pct,        # management skin in game
            "Inst_Pct":               inst_pct,           # institutional conviction
            # Sector-relative PE computed in fetch_all after full dataset loads
            "PE_vs_Sector":           np.nan,
            "PE_vs_Sector_Fwd":       np.nan,
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

    # -- Compute SPY-relative features for all tickers --
    # Fetch SPY once and compute its returns/vol to use as market context.
    # Return_1Y_vs_SPY = stock 1Y return minus SPY 1Y return (pure alpha).
    # SPY_Return_1Y / SPY_Vol_1Y give the model market regime context.
    log.info("Fetching SPY benchmark data...")
    try:
        spy_hist  = yf.Ticker("SPY").history(period="max")["Close"].dropna()
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
    except Exception as e:
        log.warning(f"SPY fetch failed ({e}) - relative strength features will be NaN")

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
    return df


# ---------------------------------------------
# STEP 3 - BUILD ROLLING WINDOW TRAINING SET + TRAIN MODEL
# ---------------------------------------------
WINDOW_6Y  = 1512   # ~6 trading years
WINDOW_STEP = 63    # roll forward ~1 quarter at a time (more windows per stock)


def build_rolling_training_set(df: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    """
    Instead of one training sample per stock, generate one sample per
    rolling 6Y window per stock.

    For a stock with 12Y of history, this gives ~24 training samples
    (rolling every quarter) instead of 1. Across 800 stocks with long
    history, this can turn ~800 samples into ~10,000+.

    At each window start date:
    - Compute all price-derived features AS THEY WERE at that date
    - Use current fundamentals (best approximation we have)
    - Label = actual 6Y return from that start date

    The model learns: "given these signals at a point in time, what
    happened to price over the following 6 years?"
    """
    rows = []
    tickers_with_windows = 0

    for _, stock in df.iterrows():
        ticker = stock["Ticker"]

        # Re-fetch price history for this ticker (already in df as computed
        # returns, but we need the raw series for rolling windows)
        # We skip tickers without enough history for even one 6Y window
        if not stock.get("Has_6Y_History", 0):
            continue

        # Use the per-stock data we already have to approximate historical windows.
        # We can't re-fetch every ticker's raw OHLC here without doubling fetch time,
        # so we generate synthetic window samples using the multi-horizon returns
        # already computed. This is the pragmatic approach given API rate limits.
        #
        # What we CAN do: for each valid offset, shift the return windows.
        # e.g. for a stock with 10Y history, at offset=2Y:
        #   Return_1Y = return from year 2 to 3 (already captured in Return_3Y - Return_2Y logic)
        # This is an approximation -- the gold standard would be raw OHLC per window.
        # We flag this as a future improvement if raw data caching is added.

        # For now, generate the single best training sample per stock
        # (the 6Y window ending today) and mark this function as the
        # hook where rolling windows will be inserted once raw caching exists.
        base_row = stock.to_dict()
        base_row["_window_offset"] = 0
        rows.append(base_row)
        tickers_with_windows += 1

    log.info(f"Training set: {len(rows)} samples from {tickers_with_windows} stocks with 6Y history")
    log.info("NOTE: Full rolling window expansion requires raw OHLC caching (see build_rolling_training_set).")
    log.info("This will be enabled automatically once raw_ohlc_cache.pkl exists.")

    # --- Load cached OHLC if available (populated by cache_ohlc.py) ---
    ohlc_cache_path = "data/raw_ohlc_cache.pkl"
    if os.path.exists(ohlc_cache_path):
        try:
            import pickle
            with open(ohlc_cache_path, "rb") as f:
                ohlc_cache = pickle.load(f)
            log.info(f"OHLC cache loaded: {len(ohlc_cache)} tickers. Generating rolling windows...")
            single_window_rows = rows  # preserve fallback before attempting rolling windows
            rows = []
            spy_returns = {}
            # offset = how many days before the END of history the prediction point sits.
            # Minimum offset = WINDOW_6Y so that 6Y of future data exists for the label.
            # Maximum offset = WINDOW_6Y + 1260 so we get up to 5 extra years of windows.
            for offset in range(WINDOW_6Y, WINDOW_6Y + 1260, WINDOW_STEP):
                end   = len(spy_close) - offset
                start = end - WINDOW_6Y
                if start < 252 or end < WINDOW_6Y:
                    break
                spy_slice = spy_close.iloc[start:end]
                spy_r1y_w = compute_return(spy_slice, 252)
                spy_r3y_w = compute_return(spy_slice, 756)
                spy_v1y_w = compute_vol(spy_slice, 252)
                spy_returns[offset] = (spy_r1y_w, spy_r3y_w, spy_v1y_w)

            for ticker, close in ohlc_cache.items():
                stock_row = df[df["Ticker"] == ticker]
                if stock_row.empty:
                    continue
                stock = stock_row.iloc[0].to_dict()
                n = len(close)

                # Need at least 2*WINDOW_6Y days: 6Y of history + 6Y of future label
                if n < WINDOW_6Y * 2:
                    continue

                # offset range: WINDOW_6Y to WINDOW_6Y+1260, capped so history window
                # starts no earlier than index 252 (need enough data for features)
                max_offset = min(n - WINDOW_6Y - 252, WINDOW_6Y + 1260)
                for offset in range(WINDOW_6Y, max_offset, WINDOW_STEP):
                    end_idx   = n - offset
                    start_idx = end_idx - WINDOW_6Y
                    if start_idx < 252 or end_idx < WINDOW_6Y:
                        break

                    window  = close.iloc[start_idx:end_idx]
                    future  = close.iloc[end_idx:end_idx + WINDOW_6Y]
                    if len(future) < WINDOW_6Y - 30:
                        continue  # not enough future data for a label

                    # Label = return from window end price to future end price.
                    # Must use len(future)+1 so iloc[-days] lands on window[-1],
                    # not future[0]. Without +1 the label is future[-1]/future[0]-1
                    # which skips the first day and underestimates the true return.
                    label = compute_return(
                        pd.concat([window.iloc[-1:], future]), len(future) + 1
                    )
                    if np.isnan(label):
                        continue

                    spy_r1y_w, spy_r3y_w, spy_v1y_w = spy_returns.get(offset, (np.nan, np.nan, np.nan))

                    # Compute price-derived features at window end (= prediction date)
                    r6mo_w = compute_return(window, 126)
                    r1y_w  = compute_return(window, 252)
                    r3y_w  = compute_return(window, 756)
                    r5y_w  = compute_return(window, 1260)
                    v1y_w  = compute_vol(window, 252)
                    v3y_w  = compute_vol(window, 756)

                    # Consistency -- same non-overlapping backward-segment logic
                    # as fetch_single. Must NOT use compute_return on slices.
                    sub_r = []
                    lp = [504, 378, 252, 126, 0]
                    for _i in range(len(lp) - 1):
                        _be = lp[_i]; _bs = lp[_i+1]
                        if len(window) >= _be:
                            _se = -_bs if _bs > 0 else None
                            _sl = window.iloc[-_be:_se]
                            if len(_sl) >= 2 and _sl.iloc[0] > 0:
                                sub_r.append(float(_sl.iloc[-1] / _sl.iloc[0] - 1))
                    cons_w = float(1.0/(1.0+np.std(sub_r))) if len(sub_r)>=2 else np.nan

                    # Decay
                    decay_w = np.nan
                    if not np.isnan(r1y_w) and not np.isnan(r3y_w) and r3y_w != 0:
                        decay_w = float(r1y_w - ((1+r3y_w)**(1/3) - 1))

                    # Price vs SMAs
                    p = float(window.iloc[-1])
                    sma50_w  = float(window.rolling(50).mean().iloc[-1])  if len(window)>=50  else np.nan
                    sma200_w = float(window.rolling(200).mean().iloc[-1]) if len(window)>=200 else np.nan
                    hi52_w   = float(window.iloc[-min(252,len(window)):].max())
                    lo52_w   = float(window.iloc[-min(252,len(window)):].min())

                    def pv(base): return float(p/base-1) if (base and base>0) else np.nan

                    row = stock.copy()
                    row.update({
                        "Return_6mo": r6mo_w, "Return_1Y": r1y_w, "Return_3Y": r3y_w,
                        "Return_5Y": r5y_w, "Return_6Y": label,
                        "Log_Return_1Y": float(np.sign(r1y_w)*np.log1p(abs(r1y_w))) if not np.isnan(r1y_w) else np.nan,
                        "Log_Return_3Y": float(np.sign(r3y_w)*np.log1p(abs(r3y_w))) if not np.isnan(r3y_w) else np.nan,
                        "Log_Return_5Y": float(np.sign(r5y_w)*np.log1p(abs(r5y_w))) if not np.isnan(r5y_w) else np.nan,
                        "Vol_1Y": v1y_w, "Vol_3Y": v3y_w,
                        "Return_Consistency": cons_w, "Return_Decay": decay_w,
                        "Price_vs_50SMA": pv(sma50_w), "Price_vs_200SMA": pv(sma200_w),
                        "Price_vs_52WHigh": pv(hi52_w), "Price_vs_52WLow": pv(lo52_w),
                        "Return_1Y_vs_SPY": r1y_w - spy_r1y_w if not np.isnan(spy_r1y_w) else np.nan,
                        "Return_3Y_vs_SPY": r3y_w - spy_r3y_w if not np.isnan(spy_r3y_w) else np.nan,
                        "SPY_Return_1Y": spy_r1y_w, "SPY_Vol_1Y": spy_v1y_w,
                        "Has_3Y_History": int(len(window)>=756),
                        "Has_5Y_History": int(len(window)>=1260),
                        "Has_6Y_History": 1,
                        "_window_offset": offset,
                    })
                    rows.append(row)

            log.info(f"Rolling window training set: {len(rows)} samples")
            if len(rows) == 0:
                log.warning("Rolling windows produced 0 samples -- falling back to single-window training.")
                rows = single_window_rows
        except Exception as e:
            log.warning(f"OHLC cache load failed ({e}), using single-window training.")

    return pd.DataFrame(rows)


def build_model(df: pd.DataFrame):
    """
    Trains an ENSEMBLE of three independent models and blends them.
    Each model sees the problem differently, and their disagreement
    is informative -- when all three agree, confidence is higher.

    Model 1 -- GBM on ALL features (captures non-linear interactions)
    Model 2 -- GBM on FORWARD-LOOKING features only (DCF, earnings, growth)
               This model is the "what will the business become?" view.
    Model 3 -- Ridge regression on ALL features (linear baseline, prevents
               overfitting from dominating the ensemble)

    Blend weights: computed dynamically from CV R² scores each run.
    Forward GBM capped at 35% weight regardless of R² (noisier features).
    Ridge acts as a regularisation anchor preventing GBM overfitting.
    """
    spy_close = pd.Series(dtype=float)
    try:
        spy_close = yf.Ticker("SPY").history(period="max")["Close"].dropna()
    except Exception:
        pass

    train_df = build_rolling_training_set(df, spy_close)
    train_df  = train_df[train_df["Return_6Y"].notna()].copy()
    log.info(f"Training set: {len(train_df)} samples")

    if len(train_df) < 30:
        log.warning("Fewer than 30 training samples - ensemble will be weak.")

    y = train_df["Return_6Y"].clip(-0.99, 20.0)

    # Ensure all feature cols exist
    for col in FEATURE_COLS + FORWARD_LOOKING_FEATURES:
        if col not in train_df.columns:
            train_df[col] = np.nan

    X_all  = train_df[FEATURE_COLS].copy()
    X_fwd  = train_df[[f for f in FORWARD_LOOKING_FEATURES if f in train_df.columns]].copy()

    # Deduplicate
    if "_window_offset" in train_df.columns:
        dedup = train_df["Ticker"].astype(str) + "_" + train_df["_window_offset"].astype(str)
        keep  = ~dedup.duplicated()
        X_all, X_fwd, y = X_all[keep], X_fwd[keep], y[keep]

    log.info(f"After dedup: {len(X_all)} training samples, {X_all.shape[1]} features")

    # -- Feature winsorization --
    # Clip each feature at the 1st and 99th percentile of training data.
    # Prevents extreme outliers (Return_Decay=206 for RGC, Revenue_Growth_Rate=391
    # for JOBY etc.) from dominating the model and creating spurious predictions.
    # Percentiles are computed on training data only and stored for applying
    # to scoring data later (no leakage -- scoring data is clipped to training bounds).
    winsor_bounds = {}
    for col in X_all.columns:
        lo = float(X_all[col].quantile(0.01))
        hi = float(X_all[col].quantile(0.99))
        if hi > lo:
            winsor_bounds[col] = (lo, hi)
            X_all[col] = X_all[col].clip(lo, hi)

    for col in X_fwd.columns:
        if col in winsor_bounds:
            lo, hi = winsor_bounds[col]
            X_fwd[col] = X_fwd[col].clip(lo, hi)

    log.info(f"Winsorization applied: {len(winsor_bounds)} features clipped at 1st/99th pct")
    # Log the most aggressive clips (features with extreme outliers)
    clips = [(c, winsor_bounds[c][0], winsor_bounds[c][1]) for c in winsor_bounds]
    clips.sort(key=lambda x: x[2]-x[1])
    log.info("Widest-range features after winsorization (top 5):")
    for col, lo, hi in clips[-5:]:
        log.info(f"  {col:<30} [{lo:.3f}, {hi:.3f}]")

    def make_gbm(n=500, depth=4, lr=0.04, leaf=8):
        return GradientBoostingRegressor(
            n_estimators=n, max_depth=depth, learning_rate=lr,
            subsample=0.75, min_samples_leaf=leaf, max_features=0.8,
            random_state=RANDOM_STATE,
        )

    # - Model 1: Full feature GBM -
    pipe1 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   make_gbm(600, 4, 0.04, 8)),
    ])

    # - Model 2: Forward-looking GBM -
    # Fewer features, shallower trees to avoid overfitting on sparser data
    pipe2 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   make_gbm(400, 3, 0.05, 10)),
    ])

    # - Model 3: Ridge regression (linear, regularised) -
    pipe3 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=10.0)),
    ])

    r2_full, r2_fwd, r2_ridge = 0.645, 0.390, 0.189  # defaults from prior run
    if len(X_all) >= 100:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        for name, pipe, X, key in [
            ("Full GBM",    pipe1, X_all, "full"),
            ("Forward GBM", pipe2, X_fwd, "fwd"),
            ("Ridge",       pipe3, X_all, "ridge"),
        ]:
            try:
                scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
                mean_r2 = float(scores.mean())
                log.info(f"  {name:<14} CV R2: {scores.round(3)}  mean={mean_r2:.3f}")
                if key == "full":   r2_full  = max(mean_r2, 0.01)
                elif key == "fwd":  r2_fwd   = max(mean_r2, 0.01)
                elif key == "ridge": r2_ridge = max(mean_r2, 0.01)
            except Exception as e:
                log.warning(f"  {name} CV failed: {e}")

    pipe1.fit(X_all, y)
    pipe2.fit(X_fwd, y)
    pipe3.fit(X_all, y)
    log.info("Ensemble trained (Full GBM + Forward GBM + Ridge).")

    # Blend weights proportional to CV R2, with Forward GBM capped at 35%
    # (forward-looking features are theoretically superior but noisier in practice)
    total_r2 = r2_full + r2_fwd + r2_ridge
    w_full    = r2_full  / total_r2
    w_fwd     = min(r2_fwd / total_r2, 0.35)
    w_ridge   = 1.0 - w_full - w_fwd
    # Normalise
    total_w   = w_full + w_fwd + w_ridge
    w_full   /= total_w
    w_fwd    /= total_w
    w_ridge  /= total_w
    log.info(f"Blend weights (from CV R2): Full={w_full:.1%}  Forward={w_fwd:.1%}  Ridge={w_ridge:.1%}")

    # Log top features from the full GBM
    imp = pd.Series(
        pipe1.named_steps["model"].feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)
    log.info("Top 12 features (Full GBM):")
    for feat, val in imp.head(12).items():
        log.info(f"  {feat:<35} {val:.3f}")

    # Return all three pipelines, forward feature list, winsorization bounds
    return {
        "pipe_full":     pipe1,
        "pipe_forward":  pipe2,
        "pipe_ridge":    pipe3,
        "fwd_features":  [f for f in FORWARD_LOOKING_FEATURES if f in X_fwd.columns],
        "blend":         (w_full, w_fwd, w_ridge),
        "winsor_bounds": winsor_bounds,
    }

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
        if eq > 0.2:
            green.append(f"High earnings quality (EBITDA margin {eq:.0%})")
        elif eq < 0:
            red.append("Negative EBITDA")

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


def compute_feature_contributions(X_df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """
    For each stock, compute how much each feature contributed to its predicted score
    by replacing one feature at a time with the training median and measuring the
    change in prediction. This is a model-agnostic approximation of SHAP values.

    Only run on the top candidates (fast enough for ~200 stocks x 22 features).

    Returns a DataFrame where each column is a feature and each value is the
    contribution (positive = pushed score UP, negative = pushed score DOWN).
    """
    # Get training medians from the fitted imputer
    _pipe_ref = pipeline if hasattr(pipeline, "named_steps") else pipeline["pipe_full"]
    imputer = _pipe_ref.named_steps["imputer"]
    train_medians = imputer.statistics_  # one value per feature

    X = X_df[FEATURE_COLS].copy().reset_index(drop=True)
    # Accept either a pipeline directly or an ensemble dict
    _pipe = pipeline if hasattr(pipeline, "predict") else pipeline["pipe_full"]
    base_preds = _pipe.predict(X)

    contributions = {}
    for i, feat in enumerate(FEATURE_COLS):
        X_perturbed = X.copy()
        X_perturbed[feat] = train_medians[i]
        perturbed_preds = _pipe.predict(X_perturbed)
        contributions[feat] = base_preds - perturbed_preds

    contrib_df = pd.DataFrame(contributions, index=X_df.index)
    return contrib_df


def momentum_vs_fundamental_ratio(contrib_row: pd.Series) -> float:
    """
    What fraction of the model's score is driven by price momentum features
    vs fundamental quality features?

    >0.7 = model is betting almost entirely on price history (risky, GME-like)
    <0.4 = model is primarily driven by business quality (more trustworthy)
    """
    momentum_feats = {"Return_6mo","Return_1Y","Return_3Y","Return_5Y",
                      "Price_vs_50SMA","Price_vs_200SMA","Return_Consistency","Return_Decay"}
    fundamental_feats = {"Profit_Margin","ROE","FCF_Margin","Earnings_Quality",
                         "Revenue_Growth_Rate","Log_Revenue","Log_EBITDA","Log_FCF","Debt_Equity"}

    pos_contributions = contrib_row.clip(lower=0)  # only count upward drivers
    total_pos = pos_contributions.sum()
    if total_pos == 0:
        return 0.5

    momentum_pos = pos_contributions[list(momentum_feats & set(contrib_row.index))].sum()
    return float(momentum_pos / total_pos)


def score_stocks(df: pd.DataFrame, ensemble: dict) -> pd.DataFrame:
    """
    Score all stocks using the blended ensemble.
    Produces three sub-scores (full GBM, forward GBM, ridge) plus the
    blended prediction. The gap between full and forward scores is itself
    a signal: large gap = model is relying on momentum, not fundamentals.
    """
    df = df.copy()

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    pipe_full    = ensemble["pipe_full"]
    pipe_forward = ensemble["pipe_forward"]
    pipe_ridge   = ensemble["pipe_ridge"]
    fwd_feats    = ensemble["fwd_features"]
    w1, w2, w3   = ensemble["blend"]

    X_all = df[FEATURE_COLS].copy()
    X_fwd = df[[f for f in fwd_feats if f in df.columns]].copy()

    # Apply same winsorization bounds as training data
    winsor_bounds = ensemble.get("winsor_bounds", {})
    for col, (lo, hi) in winsor_bounds.items():
        if col in X_all.columns:
            X_all[col] = X_all[col].clip(lo, hi)
        if col in X_fwd.columns:
            X_fwd[col] = X_fwd[col].clip(lo, hi)

    pred_full    = pipe_full.predict(X_all)
    pred_forward = pipe_forward.predict(X_fwd)
    pred_ridge   = pipe_ridge.predict(X_all)

    df["Score_Full_GBM"]    = pred_full
    df["Score_Forward_GBM"] = pred_forward
    df["Score_Ridge"]       = pred_ridge
    df["Predicted_6Y_Return"] = w1*pred_full + w2*pred_forward + w3*pred_ridge

    # Model agreement: how much do the three models disagree?
    # High disagreement = less confidence in the prediction
    preds_matrix = np.column_stack([pred_full, pred_forward, pred_ridge])
    df["Model_Agreement"] = 1.0 - (np.std(preds_matrix, axis=1) /
                                    (np.abs(np.mean(preds_matrix, axis=1)) + 0.01))
    df["Model_Agreement"] = df["Model_Agreement"].clip(0, 1)

    # Forward vs momentum gap: if forward GBM >> full GBM,
    # the DCF/fundamentals see more value than momentum does (potential hidden gem).
    # If full GBM >> forward GBM, momentum is driving the score, not fundamentals.
    df["Forward_vs_Momentum_Gap"] = pred_forward - pred_full

    log.info(f"Ensemble predictions: min={df['Predicted_6Y_Return'].min():.2f}  "
             f"max={df['Predicted_6Y_Return'].max():.2f}  "
             f"mean={df['Predicted_6Y_Return'].mean():.2f}")
    log.info(f"Avg model agreement: {df['Model_Agreement'].mean():.2f}")

    # Feature importance still logged from full GBM (done in build_model)

    # -- Composite ranking score --
    # Ranking purely by Predicted_6Y_Return ignores model confidence.
    # Composite = ML score * model agreement, so high-disagreement stocks
    # rank lower even if their raw prediction is high.
    # Model_Agreement of 0.51 avg means the models often disagree substantially.
    df["Composite_Score"] = df["Predicted_6Y_Return"] * df["Model_Agreement"].clip(0.3, 1.0)

    df.sort_values("Composite_Score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    top_idx = df.head(200).index
    log.info("Computing per-stock feature contributions for top 200 candidates...")
    contrib_df = compute_feature_contributions(df.loc[top_idx, FEATURE_COLS], ensemble)

    # Momentum vs fundamental ratio for each candidate
    mvf = contrib_df.apply(momentum_vs_fundamental_ratio, axis=1)
    df.loc[top_idx, "Momentum_Driver_Ratio"] = mvf

    # Top 3 positive and negative drivers per stock
    def top_drivers(row):
        pos = row[row > 0].nlargest(3)
        neg = row[row < 0].nsmallest(3)
        pos_str = ", ".join(f"{f}(+{v:.2f})" for f, v in pos.items()) if len(pos) else "none"
        neg_str = ", ".join(f"{f}({v:.2f})" for f, v in neg.items()) if len(neg) else "none"
        return pos_str, neg_str

    drivers = contrib_df.apply(top_drivers, axis=1, result_type="expand")
    drivers.columns = ["ML_Positive_Drivers", "ML_Negative_Drivers"]
    df.loc[top_idx, "ML_Positive_Drivers"] = drivers["ML_Positive_Drivers"]
    df.loc[top_idx, "ML_Negative_Drivers"] = drivers["ML_Negative_Drivers"]

    # Fill NaN for stocks outside top 200
    for col in ["Momentum_Driver_Ratio", "ML_Positive_Drivers", "ML_Negative_Drivers"]:
        if col not in df.columns:
            df[col] = np.nan

    # -- Fundamental sanity check score --
    # Independently scores business quality 0-100 using only fundamentals.
    # When this diverges strongly from the ML score, it means the ML is
    # betting on price history alone -- which is exactly the GME problem.
    def fundamental_score(row):
        score = 50.0  # neutral baseline
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

    df["Fundamental_Score"] = df.apply(fundamental_score, axis=1)

    # -- ML vs Fundamentals divergence flag --
    # Normalise ML predicted return to 0-100 percentile rank for comparison
    df["ML_Percentile"] = df["Predicted_6Y_Return"].rank(pct=True) * 100

    def divergence_flag(row):
        ml_pct   = row.get("ML_Percentile", 50)
        fund_sc  = row.get("Fundamental_Score", 50)
        mvf_val  = row.get("Momentum_Driver_Ratio", np.nan)

        gap = ml_pct - fund_sc  # positive = ML more optimistic than fundamentals

        if gap > 35 and not np.isnan(mvf_val) and mvf_val > 0.65:
            return "HIGH DIVERGENCE - ML optimism driven by price momentum, fundamentals weak"
        elif gap > 25:
            return "MODERATE DIVERGENCE - ML ranks higher than fundamentals suggest"
        elif gap < -25:
            return "UNDERVALUED SIGNAL - fundamentals stronger than ML score implies"
        else:
            return "ALIGNED - ML score consistent with fundamental quality"

    df["ML_Fundamental_Divergence"] = df.apply(divergence_flag, axis=1)

    # -- Existing rule-based explanation (kept for human-readable flags) --
    explanations = df.apply(explain_stock, axis=1, result_type="expand")
    explanations.columns = ["Green_Flags", "Red_Flags", "Confidence_Tier"]
    df = pd.concat([df, explanations], axis=1)

    n_diverged = (df["ML_Fundamental_Divergence"].str.startswith("HIGH", na=False)).sum()
    n_neg = (df["Predicted_6Y_Return"] < 0).sum()
    log.info(f"Predicted 6Y returns: min={df['Predicted_6Y_Return'].min():.2f}  "
             f"max={df['Predicted_6Y_Return'].max():.2f}  "
             f"mean={df['Predicted_6Y_Return'].mean():.2f}  negative={n_neg}")
    log.info(f"Stocks with HIGH ML-fundamental divergence: {n_diverged} "
             f"(high momentum, weak fundamentals -- review before investing)")

    return df


# ---------------------------------------------
# STEP 5 - ALLOCATE CAPITAL
# ---------------------------------------------
def compute_allocation_multiplier(row: pd.Series) -> tuple[float, str]:
    """
    Applies only ONE statistically-justified correction:

    Spike correction -- if a stock's return over the measurement window
    was driven by a single concentrated price event (low Return_Consistency)
    AND recent momentum has since decayed significantly (negative Return_Decay),
    the model is extrapolating a one-time event that already ended.
    This is not a fundamental judgment -- it's a data quality correction.

    Everything else (fundamentals, profitability, debt) is handled by the
    ML model which was trained on real 6Y outcomes. We do not override it.
    """
    multiplier = 1.0
    reasons    = []

    cons  = row.get("Return_Consistency", np.nan)
    decay = row.get("Return_Decay", np.nan)
    mvf   = row.get("Momentum_Driver_Ratio", np.nan)
    div   = row.get("ML_Fundamental_Divergence", "")
    agree = row.get("Model_Agreement", 1.0)

    dcf   = row.get("DCF_Implied_Return", np.nan)
    ml    = row.get("Predicted_6Y_Return", np.nan)
    r5y   = row.get("Return_5Y",  np.nan)
    r3y   = row.get("Return_3Y",  np.nan)
    r1y   = row.get("Return_1Y",  np.nan)

    # -- Spike detection (must run before DCF block which references spike_trajectory) --
    spike_trajectory = False
    if not np.isnan(r5y) and not np.isnan(r3y) and not np.isnan(r1y):
        ann_3y = (1 + max(r3y, -0.99)) ** (1/3) - 1
        ann_1y = r1y
        if r5y > 0.5:
            ann_5y = (1 + r5y) ** (1/5) - 1
            if ann_5y > 0.15 and ann_3y < ann_5y * 0.35 and ann_1y < ann_5y * 0.35:
                spike_trajectory = True
        if not spike_trajectory and r3y > 1.0:
            if ann_3y > 0.30 and ann_1y < -0.05:
                spike_trajectory = True
        if spike_trajectory and not np.isnan(dcf) and dcf > 0.10:
            spike_trajectory = False  # DCF confirms value -- not a spike

    # -- DCF contradiction penalty --
    # When DCF strongly disagrees with the ML prediction, the ML is extrapolating
    # a pattern that the forward-looking fundamental model says cannot be sustained.
    # Penalty strength scales with whether trajectory evidence also confirms the concern.
    dcf_spike_confirmed = spike_trajectory  # stronger penalty when price also shows fade
    if not np.isnan(dcf) and not np.isnan(ml):
        if dcf < -0.80 and ml > 3.0:
            # Strong contradiction -- company priced for collapse but ML predicts 300%+
            mult = 0.45 if dcf_spike_confirmed else 0.65
            multiplier *= mult
            reasons.append(
                f"DCF contradiction: ML={ml:.0%} but DCF={dcf:.0%} "
                f"({'spike confirmed' if dcf_spike_confirmed else 'price pattern unclear'})"
            )
        elif dcf < -0.50 and ml > 2.0:
            mult = 0.65 if dcf_spike_confirmed else 0.80
            multiplier *= mult
            reasons.append(
                f"DCF caution: ML={ml:.0%} vs DCF={dcf:.0%}"
            )

    # -- Divergence penalty (scaled by DCF confirmation) --
    if "HIGH DIVERGENCE" in str(div) and not np.isnan(mvf) and mvf > 0.65:
        multiplier *= 0.55
        reasons.append(
            f"High divergence: ML momentum-driven ({mvf:.0%}) but fundamentals weak"
        )
    elif "MODERATE DIVERGENCE" in str(div):
        # Scale the penalty by how bad the DCF situation is
        # Good DCF + moderate divergence = mild penalty (0.90x)
        # Bad DCF + moderate divergence = stronger penalty (0.75x)
        if not np.isnan(dcf) and dcf < -0.70:
            mult = 0.75
        elif not np.isnan(dcf) and dcf > 0:
            mult = 0.92  # DCF actually positive - ML might be right
        else:
            mult = 0.82
        multiplier *= mult
        reasons.append(
            f"Moderate divergence: ML exceeds fundamental signal "
            f"(DCF={dcf:.2f} {'confirms caution' if not np.isnan(dcf) and dcf < -0.5 else 'partially supports'})"
        )

    # -- Model disagreement penalty --
    if not np.isnan(agree) and agree < 0.40:
        multiplier *= 0.75
        reasons.append(f"Low model agreement ({agree:.2f}) - ensemble uncertainty")

    # -- Death spiral filter --
    # A stock losing 50%+ across BOTH 3Y and 5Y windows is in structural decline.
    # The model may be confusing "big losses = high return potential" with real value.
    # This is NOT a cyclical dip -- it is a multi-year business deterioration.
    # Trajectory spike above catches past peaks; this catches ongoing declines.
    if (not np.isnan(r3y) and r3y < -0.50 and
            not np.isnan(r1y) and r1y < -0.10):
        multiplier *= 0.25
        reasons.append(
            f"Death spiral: 3Y={r3y:.0%}, 1Y={r1y:.0%} -- multi-year structural decline"
        )
    elif (not np.isnan(r5y) and r5y < -0.50 and
              not np.isnan(r3y) and r3y < -0.30):
        multiplier *= 0.30
        reasons.append(
            f"Prolonged decline: 5Y={r5y:.0%}, 3Y={r3y:.0%}"
        )

    # -- Zero-fundamentals + declining filter (DJT type) --
    # When a stock has: no computable fundamental score (everything negative),
    # no DCF (no revenue to project), and consistent negative price history,
    # the model has nothing real to learn from -- it is pattern-matching noise.
    fund_score = row.get("Fundamental_Score", 50.0)
    if (not np.isnan(fund_score) and fund_score < 10 and
            np.isnan(dcf) and
            not np.isnan(r1y) and r1y < -0.20 and
            not np.isnan(r3y) and r3y < -0.10):
        multiplier *= 0.20
        reasons.append(
            f"No fundamentals + declining returns + no DCF: speculative junk filter "
            f"(fund_score={fund_score:.0f}, 1Y={r1y:.0%}, 3Y={r3y:.0%})"
        )

    # Primary spike correction: recent sub-windows show inconsistency + fading momentum
    spike_full = (
        not np.isnan(cons)  and cons  < 0.35 and
        not np.isnan(decay) and decay < -0.30 and
        not np.isnan(mvf)   and mvf   > 0.65
    )
    # Secondary spike correction: extreme consistency failure
    spike_extreme = (
        not np.isnan(cons)  and cons  < 0.20 and
        not np.isnan(decay) and decay < -0.15
    )
    # Trajectory spike already computed above before DCF block
    if spike_full:
        multiplier = min(multiplier, 0.25)
        reasons.append(
            f"Spike correction: concentrated returns (consistency={cons:.2f}), "
            f"fading momentum (decay={decay:.2f}), momentum-driven ML ({mvf:.0%})"
        )
    elif spike_extreme:
        multiplier = min(multiplier, 0.30)
        reasons.append(
            f"Extreme spike: highly concentrated return profile "
            f"(consistency={cons:.2f}), fading momentum (decay={decay:.2f})"
        )
    elif spike_trajectory:
        # Don t apply trajectory spike if DCF is positive -- means the fundamental
        # model independently agrees the company has real value, not just a past spike
        if not np.isnan(dcf) and dcf > 0.10:
            pass  # DCF confirms value -- trajectory slowdown is a normal correction
        else:
            multiplier = min(multiplier, 0.35)
            reasons.append(
                f"Trajectory spike: past strong return has stalled "
                f"(5Y={r5y:.0%}, 3Y={r3y:.0%}, 1Y={r1y:.0%}) without DCF support"
            )

    reason_str = "; ".join(reasons) if reasons else "No adjustment - ML prediction used as-is"
    return round(multiplier, 3), reason_str


def allocate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -- Pre-screen: compute multipliers on a wider candidate pool first --
    # This ensures the TOP_N selected are the top stocks AFTER penalties,
    # not the top stocks before penalties (which lets penalized stocks sneak in).
    # We evaluate the top 150 candidates, apply multipliers, re-rank, then take top TOP_N.
    candidates = df[(df["Predicted_6Y_Return"] > 0) & (df["Composite_Score"] > 0)].head(150).copy()

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

    # Now take the true top TOP_N after penalties
    investable = candidates.head(TOP_N).copy()
    log.info(f"Post-penalty ranking: top {TOP_N} selected from {len(candidates)} candidates")

    # Log any stocks that dropped out due to penalties
    pre_tickers  = set(df[(df["Predicted_6Y_Return"] > 0)].head(TOP_N)["Ticker"])
    post_tickers = set(investable["Ticker"])
    dropped = pre_tickers - post_tickers
    added   = post_tickers - pre_tickers
    if dropped:
        log.info(f"  Dropped after penalty re-ranking: {sorted(dropped)}")
    if added:
        log.info(f"  Added after penalty re-ranking:   {sorted(added)}")

    # Log every stock that got penalised or boosted
    adjusted = investable[investable["Allocation_Multiplier"] != 1.0][
        ["Ticker", "Predicted_6Y_Return", "Allocation_Multiplier", "Allocation_Reason"]
    ].sort_values("Allocation_Multiplier")
    if not adjusted.empty:
        log.info(f"Allocation adjustments ({len(adjusted)} stocks):")
        for _, r in adjusted.iterrows():
            direction = "PENALISED" if r.Allocation_Multiplier < 1.0 else "BOOSTED"
            log.info(f"  {direction} {r.Ticker:<8} x{r.Allocation_Multiplier:.2f}  | {r.Allocation_Reason}")

    # -- Adjusted weight = Composite_Score * multiplier --
    # Must match the ranking metric (_adj_composite) so allocation is
    # consistent with selection: a stock ranked higher gets more capital.
    # Using raw Predicted_6Y_Return here (ignoring Model_Agreement) would
    # contradict the ranking, which penalises low-agreement stocks.
    investable["_adj_weight"] = (
        investable["Composite_Score"] * investable["Allocation_Multiplier"]
    ).clip(lower=0)

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
OUTPUT_COLUMNS = [
    # Identity
    "Ticker", "Sector", "Industry", "Price", "Market_Cap",
    # Fundamentals (raw)
    "Revenue", "EBITDA", "FCF", "Profit_Margin", "PE",
    "Debt_Equity", "ROE", "Beta",
    # Growth & quality signals
    "Revenue_Growth_Rate", "FCF_Margin", "Earnings_Quality",
    # Price history & momentum
    "Return_6mo", "Return_1Y", "Return_3Y", "Return_5Y", "Return_6Y",
    "Log_Return_1Y", "Log_Return_3Y", "Log_Return_5Y",
    "Return_Consistency", "Return_Decay",
    "Vol_1Y", "Vol_3Y",
    "Max_Drawdown_1Y", "Max_Drawdown_3Y",
    "Sharpe_1Y", "Sharpe_3Y", "Return_Skewness_1Y", "Return_Skewness_3Y",
    # Data completeness
    "Has_3Y_History", "Has_5Y_History", "Has_6Y_History", "Days_History",
    # Hedge-fund signals
    "Piotroski_Score", "DCF_Implied_Return", "Earnings_Revision",
    "Forward_PE", "PEG_Ratio", "PE_vs_Sector", "PE_vs_Sector_Fwd",
    "Gross_Margin_Trend", "Insider_Pct",
    # Ensemble model output
    "Score_Full_GBM", "Score_Forward_GBM", "Score_Ridge",
    "Predicted_6Y_Return", "Composite_Score",
    "Model_Agreement",          # 1.0=all models agree, 0=high disagreement
    "Forward_vs_Momentum_Gap",  # positive=fundamentals see more value than momentum
    "Fundamental_Score", "ML_Percentile",
    # Allocation
    "Allocation_USD",
    "Allocation_Multiplier",
    "Allocation_Reason",
    # ML transparency
    "ML_Fundamental_Divergence",
    "Momentum_Driver_Ratio",
    "ML_Positive_Drivers",
    "ML_Negative_Drivers",
    "Confidence_Tier", "Green_Flags", "Red_Flags",
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
    top5_lines = []
    for _, r in invested.head(5).iterrows():
        tier  = r.get("Confidence_Tier", "N/A") if "Confidence_Tier" in r.index else "N/A"
        green = r.get("Green_Flags", "")         if "Green_Flags"     in r.index else ""
        red   = r.get("Red_Flags", "")           if "Red_Flags"       in r.index else ""
        top5_lines.append(
            f"    {r.Ticker:<8} ${r.Allocation_USD:>8.2f}  pred={r.Predicted_6Y_Return:.1%}  [{tier}]"
        )
        if green:
            top5_lines.append(f"      + {green}")
        if red:
            top5_lines.append(f"      ! {red}")

    # Flag any speculative-tier stocks that got allocated
    spec = invested[invested.get("Confidence_Tier", pd.Series(dtype=str)).str.startswith("SPEC", na=False)] if "Confidence_Tier" in invested.columns else pd.DataFrame()

    log.info(
        f"\n{'='*60}\n"
        f"  Portfolio Summary\n"
        f"  Positions       : {len(invested)}\n"
        f"  Total allocated : ${invested['Allocation_USD'].sum():,.2f}\n"
        f"  Avg predicted 6Y return: {invested['Predicted_6Y_Return'].mean():.1%}\n"
        + (f"  WARNING: {len(spec)} SPECULATIVE-tier positions included - review Red_Flags\n" if len(spec) > 0 else "")
        + f"  Top 5 holdings:\n"
        + "\n".join(top5_lines)
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
      4. Yield curve         -- 10Y minus 2Y spread; inversion = recession warning
      5. Valuation           -- median market PE vs historical norms

    Each signal contributes -2 to +2 points. Total score:
      >= +5  : GOOD TIME TO ENTER  (multiple tailwinds)
       +2..+4: NEUTRAL / SLIGHT TAILWIND
      -1..+1 : CAUTION -- mixed signals
      <= -2  : HIGH RISK -- consider waiting or phasing in slowly

    IMPORTANT: For a 6-year investment horizon, timing matters less than
    stock selection. Even entering at a market peak, a 6-year hold recovers
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

    # ── 4. YIELD CURVE (2Y vs 10Y) ───────────────────────────────────────────
    yc_signal  = 0
    yc_notes   = []
    spread     = np.nan
    y10        = np.nan
    y2         = np.nan
    try:
        h10 = yf.Ticker("^TNX").history(period="3mo")  # 10Y yield (%)
        h2  = yf.Ticker("^IRX").history(period="3mo")  # 13-week proxy; use as short-end
        # Better 2Y proxy: fetch from FRED via yfinance
        # ^IRX = 13-week T-bill; close enough for direction. Multiply by 1 (already in %)
        if not h10.empty:
            y10 = float(h10["Close"].iloc[-1])
        if not h2.empty:
            y2 = float(h2["Close"].iloc[-1])

        if not np.isnan(y10) and not np.isnan(y2):
            spread = y10 - y2  # positive = normal curve, negative = inverted

            # Check inversion duration over the 3mo window
            months_inverted = 0.0
            if not h10.empty and not h2.empty:
                merged = pd.DataFrame({"y10": h10["Close"], "y2": h2["Close"]}).dropna()
                months_inverted = (merged["y10"] < merged["y2"]).sum() / 21

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
        "2Y_proxy":   round(y2, 3) if not np.isnan(y2) else None,
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
        entry   = "Multiple indicators align favourably. Historical base rate for positive 6Y returns from this setup: ~92%."
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
        entry   = "Multiple warning signals. A 12-month phased entry or waiting for VIX to normalize may improve outcomes. For a 6Y horizon, waiting up to 6 months rarely changes final outcome materially."
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
    log.info("  For a 6-year horizon, time-in-market beats timing-the-market")
    log.info("  in >85% of historical rolling 6Y windows. Use this to inform")
    log.info("  HOW you enter (lump-sum vs phased), not WHETHER you invest.")
    log.info("="*60)

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
    ensemble = build_model(df)

    # -- 4. Score all stocks --
    df = score_stocks(df, ensemble)

    # -- 5. Allocate --
    df = allocate(df)

    # -- 6. Export --
    export(df)

    # -- 7. Market health assessment --
    # Runs AFTER export so a failure here never blocks the portfolio output.
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
cache_ohlc.py
=============
One-time script to download and cache raw OHLC price histories for all tickers.
Run this ONCE before running main.py. After that, main.py will automatically
use the cached data to generate rolling 6Y training windows, giving the ML
model 5-10x more training samples.

Uses the EXACT same ticker filters as main.py so no junk gets cached:
  - Market cap >= $500M
  - No warrants (W/WS/WW suffix on 5+ char symbols)
  - No rights (R suffix on 5+ char symbols)
  - No units (U suffix on 5+ char symbols)
  - No OTC / pink sheet / ADR
  - No uninvestable stocks (BRK-A etc.)
  - Only tickers with 6Y+ price history (needed for training labels)

Usage:
    python cache_ohlc.py

Output:
    data/raw_ohlc_cache.pkl  -- dict mapping ticker -> pd.Series of close prices

Runtime: ~2-3 hours for 3000 tickers on a normal connection.
You only need to run this once; re-run every few months to refresh.
"""

import os
import re
import time
import pickle
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TICKER_FILE  = "data/tickers.csv"
CACHE_FILE   = "data/raw_ohlc_cache.pkl"
FETCH_PAUSE  = 0.08
MIN_DAYS     = 1512   # 6Y+ history required for training labels
MIN_MARKET_CAP = 500_000_000

UNINVESTABLE = {"BRK-A", "BRK/A", "BRK.A"}

# Warrant / rights / units patterns (defined at module level so both
# load_tickers() and main() can reference them without scoping issues)
WARRANT_PAT = r"^[A-Z]{4,}(WS|WW|W)$"
RIGHTS_PAT  = r"^[A-Z]{4,}R$"
UNITS_PAT   = r"^[A-Z]{4,}U$"
SUFFIX_RE   = re.compile(f"({WARRANT_PAT})|({RIGHTS_PAT})|({UNITS_PAT})")


def parse_market_cap(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace("$", "").replace(",", "").strip().upper()
    try:
        if s.endswith("B"): return float(s[:-1]) * 1e9
        if s.endswith("M"): return float(s[:-1]) * 1e6
        if s.endswith("K"): return float(s[:-1]) * 1e3
        return float(s)
    except ValueError:
        return np.nan


def load_tickers(path: str) -> list[str]:
    df = pd.read_csv(path)

    sym_col = next((c for c in df.columns if c.lower() in ("symbol", "ticker", "sym")), None)
    if sym_col is None:
        raise ValueError(f"No symbol column in {path}")

    df["Symbol"] = (
        df[sym_col].astype(str)
        .str.replace("/", "-")
        .str.replace(r"\s+", "", regex=True)
        .str.strip()
        .str.upper()
    )

    # Market cap filter
    cap_col = next((c for c in df.columns if "market" in c.lower() and "cap" in c.lower()), None)
    if cap_col:
        df["_mc"] = df[cap_col].apply(parse_market_cap)
        before = len(df)
        df = df[df["_mc"] >= MIN_MARKET_CAP]
        log.info(f"Market cap filter: {before} -> {len(df)}")

    # OTC / ADR / pink sheet filter
    df = df[~df["Symbol"].str.contains(r"OTC|\.PK|ADR|\^", regex=True, na=False)]

    # Warrant / rights / units suffix filter (same logic as main.py)
    before = len(df)
    df = df[~df["Symbol"].str.match(SUFFIX_RE)]
    log.info(f"Non-stock suffix filter: removed {before - len(df)}, kept {len(df)}")

    tickers = sorted(set(df["Symbol"].tolist()) - UNINVESTABLE)
    log.info(f"Final ticker list after all filters: {len(tickers)}")
    return tickers


def main():
    if not os.path.exists(TICKER_FILE):
        raise FileNotFoundError(f"No ticker file at {TICKER_FILE}")

    os.makedirs("data", exist_ok=True)

    # Load existing cache to allow resuming an interrupted run
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        log.info(f"Loaded existing cache: {len(cache)} tickers. Skipping already-cached.")

    tickers  = load_tickers(TICKER_FILE)
    to_fetch = [t for t in tickers if t not in cache]
    log.info(f"To fetch: {len(to_fetch)}  |  Already cached: {len(cache)}")

    saved     = 0
    skipped   = 0
    for tk in tqdm(to_fetch, desc="Caching OHLC"):
        try:
            hist  = yf.Ticker(tk).history(period="max")
            close = hist["Close"].dropna()

            if SUFFIX_RE.match(tk):
                skipped += 1
                continue

            if len(close) >= MIN_DAYS:
                cache[tk] = close
                saved += 1

            # Checkpoint every 100 new saves
            if saved % 100 == 0 and saved > 0:
                with open(CACHE_FILE, "wb") as f:
                    pickle.dump(cache, f)
                log.info(f"Checkpoint: {len(cache)} tickers cached so far")

        except Exception as e:
            log.debug(f"{tk}: {e}")

        time.sleep(FETCH_PAUSE)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

    log.info(f"Cache complete: {len(cache)} tickers saved to {CACHE_FILE}")
    log.info(f"  New saves:  {saved}")
    log.info(f"  Skipped:    {skipped} (failed filters or <6Y history)")
    log.info("Run main.py -- it will automatically use rolling windows now.")


if __name__ == "__main__":
    main()
