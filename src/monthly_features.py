"""
Monthly technical characteristics for cross-sectional return prediction.

Timing convention: each row is indexed at month t. Features use returns up to
and including month t; the target is the return of month t+1. Momentum windows
skip the most recent month (the short-term-reversal month), following
Jegadeesh and Titman (1993).
"""

import numpy as np
import pandas as pd

# technical characteristics produced by build_monthly_panel
FEATURE_COLS = ["reversal_1m", "mom_12_1", "mom_6_1", "ltr_36_13", "vol_12m"]


def build_monthly_panel(df: pd.DataFrame, ret_col: str = "ret",
                        date_col: str = "date", id_col: str = "PERMNO") -> pd.DataFrame:
    """Add technical characteristics and the next-month target to a monthly
    returns panel.

    `df` must contain `id_col`, `date_col` and `ret_col`. Returns the same
    DataFrame, sorted by (id, date), with the extra columns:
      reversal_1m, mom_12_1, mom_6_1, ltr_36_13, vol_12m, target.
    """
    df = df.sort_values([id_col, date_col]).copy()
    df["_logr"] = np.log1p(df[ret_col].clip(lower=-0.99))
    g = df.groupby(id_col, sort=False)

    # next-month return -- the prediction target
    df["target"] = g[ret_col].shift(-1)

    # short-term reversal: the most recent month's return
    df["reversal_1m"] = df[ret_col]

    # momentum: cumulative log return over the window, skipping month t
    df["mom_12_1"] = g["_logr"].transform(
        lambda s: s.shift(1).rolling(11, min_periods=6).sum())
    df["mom_6_1"] = g["_logr"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=3).sum())

    # long-term reversal: cumulative log return over months t-36..t-13
    df["ltr_36_13"] = g["_logr"].transform(
        lambda s: s.shift(13).rolling(24, min_periods=12).sum())

    # volatility: standard deviation of monthly returns over the trailing year
    df["vol_12m"] = g[ret_col].transform(
        lambda s: s.rolling(12, min_periods=6).std())

    return df.drop(columns="_logr")
