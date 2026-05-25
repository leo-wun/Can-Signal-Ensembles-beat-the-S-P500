"""
Cross-sectional correlation between the stock-level signals.

For each date we compute the Pearson correlation matrix of the signal values
across stocks, then average over all dates. The signals are already
cross-sectionally rank-normalised, so this is effectively an average rank
correlation. It answers: how much *distinct* information do the 14 signals
carry? Two highly correlated signals are redundant — the meta-model would see
fewer effective inputs than it appears, and the ensemble gains nothing from
the pair.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-script import (ic_analysis)
import config
from ic_analysis import FEATURE_COLS, DATE

MIN_STOCKS = 100  # skip thin early dates (few stocks have long-window history)

# Display order grouped by construction family.
ORDER = [
    "log_reversal_1d",
    "mom_5d", "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "vol_w_mom_5d_std_5d", "vol_w_mom_21d_std_21d", "vol_w_mom_63d_std_63d",
    "vol_w_mom_126d_std_126d", "vol_w_mom_252d_std_252d",
    "vol_21d", "vol_63d", "vol_252d",
]


def avg_cross_sectional_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Average over dates of the per-date cross-sectional correlation matrix."""
    sizes = df.groupby(DATE).size()
    keep = sizes[sizes >= MIN_STOCKS].index
    df = df[df[DATE].isin(keep)]
    corr_by_date = df.groupby(DATE)[FEATURE_COLS].corr()
    avg = corr_by_date.groupby(level=1).mean()
    return avg.loc[ORDER, ORDER]


def main() -> None:
    df = pd.read_parquet(config.FEATURES_PATH_CLEAN, columns=[DATE] + FEATURE_COLS)
    df[DATE] = pd.to_datetime(df[DATE])
    print(f"Loaded {len(df):,} rows | {df[DATE].nunique():,} dates")

    corr = avg_cross_sectional_corr(df)
    print(f"\n=== Average cross-sectional correlation (dates with >= {MIN_STOCKS} stocks) ===")
    print(corr.to_string(float_format=lambda x: f"{x:+.2f}"))

    off = corr.where(~np.eye(len(corr), dtype=bool))
    print("\n=== Max |correlation| with any other signal (redundancy flag) ===")
    print(off.abs().max().sort_values(ascending=False).to_string(float_format=lambda x: f"{x:.2f}"))

    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(corr, annot=True, fmt="+.2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Average cross-sectional correlation between signals")
    out = config.PLOTS_DIR / "signal_correlation_matrix.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved heatmap -> {out}")


if __name__ == "__main__":
    main()
