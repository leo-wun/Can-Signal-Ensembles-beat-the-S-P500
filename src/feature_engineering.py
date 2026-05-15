"""
Feature engineering for next-day return prediction.

Pipeline (all signals shifted by 1 day — only data up to t-1 used on day t):
  1. Short-term reversal  — log return at t-1
  2. Momentum             — cumulative log return over multiple lookback windows
  3. Volatility           — rolling annualised std of daily log returns
  4. Vol-weighted momentum — momentum / realized vol (Sharpe-like)
  5. Market return        — log market return at t-1 (macro feature)
"""

import pandas as pd
import polars as pl
import polars.selectors as cs
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def polars_features(
        df,
        value_col: str = 'ret',
        date_col: str = 'date',
        target_col: str = 'ret',
        crosssectional_rank: bool = True
) -> pl.DataFrame:
    """
    Build the full feature matrix from a clean CRSP returns panel.

    Input ->
    df:                 CRSP panel (pandas or polars). Returns must already be clipped.
    value_col:          Column name for daily returns.
    date_col:           Column name for dates.
    target_col:         Column used as the prediction target (= current-day return).
    crosssectional_rank: If True, rank-normalize stock-level features to [-0.5, 0.5]
                         per date. Macro features (log_reversal_mkt_1d) are excluded.

    Output ->
    Polars DataFrame with original columns plus all feature and target columns.
    """

    shift_days = 1

    # Convert to Polars if necessary
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    # ── Momentum features ─────────────────────────────────────────────────────
    momentum_exprs = [
        (
            pl.col(value_col)
            .log1p()
            .rolling_sum(window_size=window)
            .exp()
            - 1
        )
        .shift(shift_days)
        .over("PERMNO")
        .alias(f"mom_{window}d")
        for window in config.MOMENTUM_WINDOWS
    ]

    # ── Volatility features ───────────────────────────────────────────────────
    vol_exprs = [
        (
            pl.col(value_col)
            .shift(1)
            .rolling_std(window_size=window)
        )
        .shift(shift_days)
        .over("PERMNO")
        .alias(f"vol_{window}d")
        for window in config.VOLATILITY_WINDOWS
    ]

    # ── Vol-weighted momentum ─────────────────────────────────────────────────
    vol_w_mom_exprs = [
        (
            (
                (pl.col(value_col).log1p().rolling_sum(window_size=w).exp() - 1)
                / pl.col(value_col).shift(1).rolling_std(window_size=w)
            )
            .shift(shift_days)
            .over("PERMNO")
            .alias(f"vol_w_mom_{w}d_std_{w}d")
        )
        for w in config.MOMENTUM_WINDOWS
    ]

    df = df.sort(['PERMNO', date_col]).with_columns([

        # Short-term reversal
        pl.col(value_col)
            .log1p()
            .shift(1)
            .over('PERMNO')
            .alias('log_reversal_1d'),

        # Lagged market return (macro — same for all stocks on a given date)
        pl.col('mkt_ret')
            .log1p()
            .shift(1)
            .over('PERMNO')
            .alias('log_reversal_mkt_1d'),

        # Target (current-day return; features use data up to t-1)
        pl.col(target_col).alias('target'),

        *momentum_exprs,
        *vol_exprs,
        *vol_w_mom_exprs,
    ])

    # ── Cross-sectional rank normalization ────────────────────────────────────
    # Applied per date on stock-level features only.
    # log_reversal_mkt_1d is excluded — it is identical across stocks on each date.
    if crosssectional_rank:
        cs_cols = (
            [f"mom_{w}d" for w in config.MOMENTUM_WINDOWS]
            + [f"vol_{w}d" for w in config.VOLATILITY_WINDOWS]
            + [f"vol_w_mom_{w}d_std_{w}d" for w in config.MOMENTUM_WINDOWS]
            + ["log_reversal_1d"]
        )

        rank_exprs = [
            (
                pl.col(col)
                .rank(method="average")
                .over(date_col)
                / pl.col(col).count().over(date_col)
                - 0.5
            )
            .alias(col)
            for col in cs_cols
        ]

        df = df.with_columns(rank_exprs)

    return df



def rank_ic_table(df, feature_cols, target='target', date_col='date'):
    records = []
    for col in feature_cols:
        sub = df[[date_col, col, target]].dropna()
        ic_series = (
            sub.groupby(date_col)
            .apply(lambda g: spearmanr(g[col], g[target])[0] if len(g) > 10 else np.nan,
                   include_groups=False)
            .dropna()
        )
        mean_ic = ic_series.mean()
        t_stat  = mean_ic / (ic_series.std() / np.sqrt(len(ic_series)))
        records.append({'feature': col, 'mean_IC': mean_ic, 't_stat': t_stat,
                        'IC_std': ic_series.std(), 'IC>0_pct': (ic_series > 0).mean()})
    return pd.DataFrame(records).set_index('feature').sort_values('t_stat', key=abs, ascending=False)
