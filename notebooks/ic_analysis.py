"""
Cross-sectional rank IC of the stock-level signals.

IC (information coefficient) = per-date Spearman correlation between a signal
at date t and the realised return at date t (the 'target' column). It measures
whether a signal ranks stocks in the right order on a given day. We report the
time-series mean IC, its t-stat, the IC information ratio, the hit rate, and a
per-year breakdown to gauge stability before committing to the stacking design.

Note: the naive t-stat assumes i.i.d. daily IC. Daily IC is mildly
autocorrelated, so it overstates significance somewhat, read t-stats as a
first cut, not a final verdict.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Stock-level signals. log_reversal_mkt_1d is excluded on purpose: it is a macro
# series (identical across stocks on a given date), so a cross-sectional IC is
# undefined for it. CZ factors and VIX are time-series signals — also out of scope.
FEATURE_COLS = [
    "log_reversal_1d",
    "mom_5d", "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "vol_21d", "vol_63d", "vol_252d",
    "vol_w_mom_5d_std_5d", "vol_w_mom_21d_std_21d", "vol_w_mom_63d_std_63d",
    "vol_w_mom_126d_std_126d", "vol_w_mom_252d_std_252d",
]
TARGET = "target"
DATE = "date"


def daily_rank_ic(
    df: pd.DataFrame
) -> pd.DataFrame:
    
    """
    
    Daily cross-sectional rank IC. Returns a DataFrame indexed by date with
    one column per signal.
    
    """
    
    ranked = df.groupby(DATE)[[TARGET] + FEATURE_COLS].rank().astype("float32")
    ranked[DATE] = df[DATE].values
    ic = (
        ranked.groupby(DATE)[[TARGET] + FEATURE_COLS]
        .apply(lambda g: g[FEATURE_COLS].corrwith(g[TARGET]))
    )
    return ic


def summarise(
    ic: pd.DataFrame
) -> pd.DataFrame:
    
    """
    
    Time-series summary of daily IC, one row per signal.
    
    """
    
    n = ic.count()
    mean = ic.mean()
    std = ic.std()
    t_stat = mean / (std / np.sqrt(n))
    icir = mean / std
    hit = (np.sign(ic) == np.sign(mean)).sum() / n
    
    return pd.DataFrame({
        "mean_IC": mean,
        "IC_std": std,
        "t_stat": t_stat,
        "ICIR_ann": icir * np.sqrt(252),
        "hit_rate": hit,
        "n_days": n,
    }).sort_values("t_stat", key=lambda s: s.abs(), ascending=False)


def main(
) -> None:
    
    # Get data
    df = pd.read_parquet(config.FEATURES_PATH_CLEAN, columns=[DATE, TARGET] + FEATURE_COLS)
    df[DATE] = pd.to_datetime(df[DATE])
    print(f"Loaded {len(df):,} rows | {df[DATE].nunique():,} dates | "
          f"{df[DATE].min().date()} -> {df[DATE].max().date()}\n")

    # Get daily IC
    ic = daily_rank_ic(df)

    # Display data
    print(" Full-sample rank IC (sorted by |t_stat|) ")
    print(summarise(ic).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n Mean IC by year ")
    by_year = ic.groupby(ic.index.year).mean()
    print(by_year.to_string(float_format=lambda x: f"{x:+.4f}"))

    # Create and save graph
    fig, ax = plt.subplots(figsize=(11, 6))
    ic.cumsum().plot(ax=ax, linewidth=1)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_title("Cumulative daily rank IC by signal "
                 "(steady slope = stable edge)")
    ax.set_ylabel("cumulative IC")
    ax.legend(fontsize=7, ncol=2)
    out = config.PLOTS_DIR / "signal_cumulative_ic.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved cumulative-IC plot -> {out}")


if __name__ == "__main__":
    main()
