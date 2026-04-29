import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def clean_crsp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean CRSP daily returns.

    Expected columns: PERMNO, date, ret, mkt_ret, and optionally DlyPrc / ShrOut.
    - Drop rows where return is missing
    - Restrict to project date window
    - Cap extreme returns at ±300% (data errors / delistings)
    - Compute log market cap if price and shares are present
    """
    df = df.copy()
    df = df.dropna(subset=["ret"])
    df = df[(df["date"] >= config.START_DATE) & (df["date"] <= config.END_DATE)]
    df = df[df["ret"].abs() <= 3.0]

    # Market cap: price × shares outstanding (shrout is in thousands)
    if "DlyPrc" in df.columns and "ShrOut" in df.columns:
        mktcap = df["DlyPrc"].abs() * df["ShrOut"] * 1_000
        df["mktcap"] = mktcap.clip(lower=1)          # avoid log(0)
        df["log_mktcap"] = np.log(df["mktcap"])

    df = df.sort_values(["PERMNO", "date"]).reset_index(drop=True)
    return df


def clean_futures(df: pd.DataFrame, min_history_years: float = 5.0) -> pd.DataFrame:
    """
    Convert futures prices to daily log returns.

    - Compute daily log return for each instrument
    - Drop instruments with fewer than `min_history_years` * 252 observations
    - Restrict to project date window
    """
    df = df.copy().sort_values("date")
    df = df[(df["date"] >= config.START_DATE) & (df["date"] <= config.END_DATE)]

    price_cols = [c for c in df.columns if c != "date"]
    min_obs = int(min_history_years * 252)

    ret_df = df[["date"]].copy()
    kept = []
    for col in price_cols:
        series = df[col]
        if series.notna().sum() >= min_obs:
            log_ret = np.log(series / series.shift(1))
            ret_df[col] = log_ret
            kept.append(col)

    print(f"Futures: kept {len(kept)}/{len(price_cols)} instruments "
          f"(≥{min_history_years}y of data in {config.START_DATE}–{config.END_DATE})")
    return ret_df.reset_index(drop=True)
