"""
data_loader.py — 3-tier data loading with graceful degradation.

Tier 1: In-memory cache during a single Streamlit session (session_state).
Tier 2: On-disk CSV cache (chatbot/data/cache.csv, TTL=1h).
Tier 3: Live Yahoo Finance pull via yfinance.
Tier 4: Fall back to Part 1 simulated prices (chatbot/data/fallback_prices.csv)
        so the demo always runs, even offline or when every Yahoo symbol fails.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CACHE_FILE,
    CACHE_TTL_SECONDS,
    FALLBACK_FILE,
    FUND_CODES,
    FUND_MAP,
    HISTORY_MONTHS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_fund_prices(force_refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Return (prices_df, source_label).

    ``prices_df`` columns are ``Fund_01``..``Fund_10`` (index = Month 0..60 or
    a DatetimeIndex depending on the source).  ``source_label`` is one of:

    * ``"cache"``    - loaded from disk cache (< 1h old)
    * ``"yfinance"`` - freshly pulled from Yahoo Finance
    * ``"fallback"`` - synthetic Part 1 prices (offline demo mode)
    """
    # Tier 2: disk cache
    if not force_refresh and _cache_fresh():
        try:
            df = _read_cache()
            logger.info("Loaded prices from on-disk cache.")
            return df, "cache"
        except Exception as exc:
            logger.warning("Cache read failed, refreshing: %s", exc)

    # Tier 3: yfinance
    try:
        df = _pull_from_yfinance()
        if df is not None and df.shape[1] == len(FUND_CODES) and df.shape[0] >= 12:
            _write_cache(df)
            logger.info("Loaded prices from yfinance (%d rows).", len(df))
            return df, "yfinance"
    except Exception as exc:
        logger.warning("yfinance pull failed: %s", exc)

    # Tier 4: fallback
    df = _read_fallback()
    logger.info("Loaded prices from fallback (Part 1 simulated data).")
    return df, "fallback"


# ---------------------------------------------------------------------------
# Tier 2 - disk cache helpers
# ---------------------------------------------------------------------------

def _cache_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    age = time.time() - CACHE_FILE.stat().st_mtime
    return age < CACHE_TTL_SECONDS


def _read_cache() -> pd.DataFrame:
    df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    # Make sure column order matches FUND_CODES
    df = df[[c for c in FUND_CODES if c in df.columns]]
    if df.shape[1] != len(FUND_CODES):
        raise ValueError("cache.csv is missing some fund columns")
    return df


def _write_cache(df: pd.DataFrame) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_FILE)


# ---------------------------------------------------------------------------
# Tier 3 - yfinance pull
# ---------------------------------------------------------------------------

def _pull_from_yfinance() -> pd.DataFrame | None:
    """Pull ~HISTORY_MONTHS months of monthly adjusted closes for all funds."""
    try:
        import yfinance as yf
    except ImportError:
        logger.info("yfinance not installed, skipping live pull.")
        return None

    tickers = [meta["yahoo"] for meta in FUND_MAP.values()]
    # Ask for a slightly longer window so that monthly resampling has enough data.
    period = f"{max(HISTORY_MONTHS + 6, 72)}mo"

    try:
        raw = yf.download(
            tickers=" ".join(tickers),
            period=period,
            interval="1mo",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.warning("yf.download raised: %s", exc)
        return None

    # Response structure varies across yfinance versions: MultiIndex for
    # multi-ticker, flat for single-ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        # Prefer 'Close' (auto_adjust=True already applied).
        if "Close" in raw.columns.levels[0]:
            prices = raw["Close"]
        else:
            return None
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")
    if prices.empty:
        return None

    # Restrict to the last HISTORY_MONTHS rows and rename to Fund_XX.
    prices = prices.tail(HISTORY_MONTHS + 1)
    ticker_to_code = {meta["yahoo"]: code for code, meta in FUND_MAP.items()}
    prices = prices.rename(columns=ticker_to_code)

    # Ensure every fund column exists; fill missing ones from fallback.
    missing = [c for c in FUND_CODES if c not in prices.columns]
    if missing:
        logger.info("yfinance missing %s, filling from fallback.", missing)
        fallback = _read_fallback()
        # Align lengths: take tail of fallback matching prices length.
        tail = fallback.tail(len(prices)).reset_index(drop=True)
        tail.index = prices.index
        for c in missing:
            prices[c] = tail[c].values

    # Re-order columns to canonical order and drop any rows with NA left.
    prices = prices[FUND_CODES].ffill().bfill()
    return prices


# ---------------------------------------------------------------------------
# Tier 4 - fallback (Part 1 simulated data)
# ---------------------------------------------------------------------------

def _read_fallback() -> pd.DataFrame:
    df = pd.read_csv(FALLBACK_FILE, index_col="Month")
    return df[FUND_CODES]


# ---------------------------------------------------------------------------
# Helpers exposed to the rest of the app
# ---------------------------------------------------------------------------

def fund_display_name(code: str) -> str:
    return FUND_MAP.get(code, {}).get("name", code)


def fund_ticker(code: str) -> str:
    return FUND_MAP.get(code, {}).get("yahoo", "")


__all__ = [
    "load_fund_prices",
    "fund_display_name",
    "fund_ticker",
]
