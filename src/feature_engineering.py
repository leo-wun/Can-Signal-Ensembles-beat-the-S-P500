"""
Feature engineering for next-day return prediction.

Pipeline:
  1. Momentum features  — cumulative log returns over multiple lookback windows
  2. Reversal feature   — previous-day return (short-term mean reversion)
  3. Volatility features — rolling realized std of daily returns
  4. Market cap         — log market cap (size), cross-sectionally z-scored per date
  5. Futures features   — lagged log returns for each futures instrument
  6. build_features()   — combines all of the above into a single panel
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Stock-level features ──────────────────────────────────────────────────────

def add_momentum(df: pd.DataFrame, windows=None) -> pd.DataFrame:
    """
    Cumulative log return over each window, shifted by 2 days (skip t-1).

    Column names: mom_5d, mom_21d, …
    """
    windows = windows or config.MOMENTUM_WINDOWS
    df = df.copy().sort_values(["PERMNO", "date"])
    log_ret = np.log1p(df["ret"])
    # Attach as a named series so groupby can reference it
    df["_log_ret"] = log_ret

    for w in windows:
        df[f"mom_{w}d"] = df.groupby("PERMNO")["_log_ret"].transform(
            lambda x: x.shift(2).rolling(w, min_periods=w // 2).sum()
        )
    df.drop(columns=["_log_ret"], inplace=True)
    return df


def add_reversal(df: pd.DataFrame) -> pd.DataFrame:
    """Previous-day log return (short-term reversal signal)."""
    df = df.copy().sort_values(["PERMNO", "date"])
    df["_log_ret"] = np.log1p(df["ret"])
    df["reversal_1d"] = df.groupby("PERMNO")["_log_ret"].transform(lambda x: x.shift(1))
    df.drop(columns=["_log_ret"], inplace=True)
    return df


def add_volatility(df: pd.DataFrame, windows=None) -> pd.DataFrame:
    """
    Rolling realized volatility (annualised std of daily log returns).

    Uses returns up to and including t-1 (no look-ahead).
    Column names: vol_21d, vol_63d, vol_252d
    """
    windows = windows or config.VOLATILITY_WINDOWS
    df = df.copy().sort_values(["PERMNO", "date"])
    df["_log_ret"] = np.log1p(df["ret"])

    for w in windows:
        df[f"vol_{w}d"] = df.groupby("PERMNO")["_log_ret"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=w // 2).std() * np.sqrt(252)
        )
    df.drop(columns=["_log_ret"], inplace=True)
    return df


def add_mktcap_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score of log market cap within each date.

    Requires `log_mktcap` produced by preprocessing.clean_crsp (needs DlyPrc
    and ShrOut in the raw CRSP file — download these from WRDS if missing).
    """
    if "log_mktcap" not in df.columns:
        return df
    df = df.copy()
    df["size"] = df.groupby("date")["log_mktcap"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    return df


# ── Futures macro features ────────────────────────────────────────────────────

def add_futures_features(stock_df: pd.DataFrame, futures_df: pd.DataFrame,
                         lags=None) -> pd.DataFrame:
    """
    Merge lagged futures log returns into the stock panel as macro features.

    `futures_df` must have a `date` column and one column per instrument
    (log-returns, from preprocessing.clean_futures).

    For each lag l, appends columns named {instrument}__lag{l}.
    """
    lags = lags or config.FUTURES_LAGS
    fut = futures_df.copy().sort_values("date").set_index("date")

    lagged_frames = []
    for lag in lags:
        shifted = fut.shift(lag)
        shifted.columns = [f"{c}__lag{lag}" for c in fut.columns]
        lagged_frames.append(shifted)

    fut_lagged = pd.concat(lagged_frames, axis=1).reset_index()
    return pd.merge(stock_df, fut_lagged, on="date", how="left")


# ── Target ────────────────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Next-day return as the prediction target (NaN for the last row per stock)."""
    df = df.copy().sort_values(["PERMNO", "date"])
    df["target"] = df.groupby("PERMNO")["ret"].shift(-1)
    return df


# ── Cross-sectional normalisation ─────────────────────────────────────────────

def crosssectional_rank(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Map each feature to a cross-sectional rank score in [-0.5, 0.5] per date.

    Rank normalisation makes the neural net input stable across time and
    comparable across stocks of very different sizes.
    """
    df = df.copy()
    for col in feature_cols:
        df[col] = df.groupby("date")[col].transform(
            lambda x: x.rank(pct=True, na_option="keep") - 0.5
        )
    return df


# ── Master builder ────────────────────────────────────────────────────────────

def build_features(crsp: pd.DataFrame, futures: pd.DataFrame,
                   rank_normalize: bool = True) -> pd.DataFrame:
    """
    Assemble the full feature matrix for the neural net.

    Args:
        crsp:           Cleaned CRSP DataFrame (output of preprocessing.clean_crsp).
        futures:        Cleaned futures returns DataFrame (preprocessing.clean_futures).
        rank_normalize: Apply cross-sectional rank normalization to stock-level features.

    Returns:
        Panel DataFrame: one row per (PERMNO, date) with all features + `target`.
        Rows with NaN in features are kept — handle masking in the model pipeline.
    """
    df = crsp.copy()
    df = add_momentum(df)
    df = add_reversal(df)
    df = add_volatility(df)
    df = add_mktcap_zscore(df)
    df = add_target(df)
    df = add_futures_features(df, futures)

    if rank_normalize:
        mom_cols  = [f"mom_{w}d"  for w in config.MOMENTUM_WINDOWS  if f"mom_{w}d"  in df.columns]
        vol_cols  = [f"vol_{w}d"  for w in config.VOLATILITY_WINDOWS if f"vol_{w}d" in df.columns]
        size_col  = ["size"] if "size" in df.columns else []
        to_rank   = mom_cols + vol_cols + ["reversal_1d"] + size_col
        df = crosssectional_rank(df, [c for c in to_rank if c in df.columns])

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all feature column names (excludes identifiers, raw return, and target)."""
    non_feature = {
        "PERMNO", "HdrCUSIP", "CUSIP", "Ticker", "TradingSymbol",
        "PERMCO", "SICCD", "NAICS", "date", "ret", "mkt_ret",
        "DlyPrc", "ShrOut", "mktcap", "log_mktcap", "target",
    }
    return [c for c in df.columns if c not in non_feature]
