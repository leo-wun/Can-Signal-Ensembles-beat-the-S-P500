"""
01 — EDA and Preprocessing pipeline  (converted from 01-02_EDA_preprocessing.ipynb)

Steps performed:
  1. Load raw CRSP daily CSV → cleaned parquet  (CRSP_PATH_CLEAN)
  2. Load raw CZ monthly CSV → cleaned forward-filled daily parquet, date as
     index (CZ_PATH_CLEAN) — same output as 02_features.py
  3. Load raw VIX parquet    → normalised daily parquet      (VIX_PATH_CLEAN)
  4. Build cross-sectional rank-normalised daily features    (FEATURES_PATH_CLEAN)
  5. Generate EDA plots saved to PLOTS_DIR

Run from the project root:
    python notebooks/01_preprocess_eda.py

Prerequisites:
    data/raw/daily_crsp_raw.parquet   (created by load_crsp_polars() in src/data_loading.py)
    data/raw/Chen_Zimmerman_monthly_raw.parquet  (created by load_cz_monthly())
    data/raw/vix_raw.parquet          (created by load_VIX() in src/data_loading.py)
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — figures are saved, not displayed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import spearmanr

# Ensure UTF-8 console output on Windows (src/utils.py prints use → arrows)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.utils import shrink_polars, generate_date_split
from src.preprocessing import clean_crsp, clean_cz_monthly
from src.preprocessing import clean_vix as _clean_vix   # path-based version
from src.data_loading import load_crsp_polars, load_cz_monthly
from src.feature_engineering import polars_features

os.makedirs(str(config.PLOTS_DIR), exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.float_format", "{:.4f}".format)


# ── Helpers ───────────────────────────────────────────────────────────────────

def rank_ic_table(feat: pd.DataFrame, cols: list[str], target: str = "target") -> pd.DataFrame:
    """Mean Spearman rank IC, t-stat and IC>0% for each signal column."""
    rows = []
    for col in cols:
        sub = feat[["date", col, target]].dropna()
        ic_ts = (
            sub.groupby("date")
            .apply(
                lambda g: spearmanr(g[col], g[target])[0] if len(g) > 10 else np.nan,
                include_groups=False,
            )
            .dropna()
        )
        n = len(ic_ts)
        mean_ic = ic_ts.mean()
        std_ic = ic_ts.std()
        t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 and n > 1 else 0.0
        rows.append(
            {
                "feature": col,
                "mean_IC": mean_ic,
                "t_stat": t_stat,
                "IC_std": std_ic,
                "IC>0_pct": (ic_ts > 0).mean(),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


# ── Step 1-4: Pipeline ────────────────────────────────────────────────────────

def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full preprocessing and feature-engineering pipeline.
    Returns (crsp, feat, feat_raw) as pandas DataFrames.

    Automatically creates missing raw parquets from the provided CSVs when needed.
    VIX processing is skipped gracefully if vix_raw.parquet does not exist
    (it requires a prior WRDS pull via src/data_loading.load_VIX).
    """
    # ── 1a. Bootstrap CRSP raw parquet from CSV if needed ────────────────────
    if not config.CRSP_PATH_RAW.exists():
        print("  CRSP raw parquet not found — loading from CSV (this may take a moment)...")
        load_crsp_polars()               # reads daily_crsp.csv → CRSP_PATH_RAW

    print("=" * 60)
    print("Step 1: Clean CRSP daily returns")
    print("=" * 60)
    crsp = clean_crsp()                  # reads CRSP_PATH_RAW, saves CRSP_PATH_CLEAN
    crsp["date"] = pd.to_datetime(crsp["date"])
    print(f"  Shape: {crsp.shape}  |  Stocks: {crsp['PERMNO'].nunique():,}  "
          f"|  Dates: {crsp['date'].nunique():,}")

    # ── 2a. Bootstrap CZ raw parquet from CSV if needed ──────────────────────
    if not config.CZ_PATH_RAW.exists():
        print("  CZ raw parquet not found — loading from CSV...")
        load_cz_monthly()                # reads Chen_Zimmerman_monthly.csv → CZ_PATH_RAW

    print("\n" + "=" * 60)
    print("Step 2: Clean Chen-Zimmermann monthly factors")
    print("=" * 60)
    cz = clean_cz_monthly()              # reads CZ_PATH_RAW, saves CZ_PATH_CLEAN (date as index)
    print(f"  Shape: {cz.shape}  |  Factors: {cz.shape[1]}  |  date as index: "
          f"{cz.index.name == 'date'}")

    # ── 3a. VIX: skip gracefully if raw parquet is absent ────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Clean VIX")
    print("=" * 60)
    if not config.VIX_PATH_RAW.exists():
        print("  vix_raw.parquet not found — skipping VIX processing.")
        print("  To enable: run src/data_loading.load_VIX() (requires WRDS access).")
    else:
        _clean_vix()                     # reads VIX_PATH_RAW, saves VIX_PATH_CLEAN

    print("\n" + "=" * 60)
    print("Step 4: Build daily technical features (rank-normalised)")
    print("=" * 60)
    feat_pl = polars_features(
        crsp, value_col="ret", date_col="date", target_col="ret", crosssectional_rank=True
    )
    feat_pl = shrink_polars(feat_pl)
    feat = feat_pl.to_pandas()
    feat.to_parquet(config.FEATURES_PATH_CLEAN, compression="zstd", index=False)
    print(f"  Saved → {config.FEATURES_PATH_CLEAN}")
    print(f"  Shape: {feat.shape}  |  Stocks: {feat['PERMNO'].nunique():,}  "
          f"|  Dates: {feat['date'].nunique():,}")

    # Also build raw (non-rank-normalised) features for EDA plots
    feat_raw_pl = polars_features(
        crsp, value_col="ret", date_col="date", target_col="ret", crosssectional_rank=False
    )
    feat_raw = feat_raw_pl.to_pandas()

    return crsp, feat, feat_raw


# ── Step 5: EDA Plots ─────────────────────────────────────────────────────────

def plot_return_distribution(crsp: pd.DataFrame) -> None:
    print("\nPlot: return distribution")
    ret = crsp["ret"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(ret.clip(-0.20, 0.20), bins=200, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_title("Daily return distribution (clipped ±20 % for display)")
    axes[0].set_xlabel("Return")

    pcts = [0.1, 0.5, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
    q = np.percentile(ret.dropna(), pcts)
    axes[1].barh(
        range(len(pcts)), q,
        color=["tomato" if v < 0 else "steelblue" for v in q],
    )
    axes[1].set_yticks(range(len(pcts)))
    axes[1].set_yticklabels([f"p{p}" for p in pcts])
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title("Return percentiles")

    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "crsp_return_distribution.png", bbox_inches="tight")
    plt.close()


def plot_universe_size(crsp: pd.DataFrame) -> None:
    print("Plot: universe size over time")
    monthly = crsp.set_index("date").resample("ME")["PERMNO"].nunique()
    obs_per_stock = crsp.groupby("PERMNO").size()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].fill_between(monthly.index, monthly.values, alpha=0.35, color="steelblue")
    axes[0].plot(monthly.index, monthly.values, color="steelblue", lw=1)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_title("Active stocks per month")
    axes[0].set_ylabel("# stocks")
    print(f"  Min: {monthly.min():,}  Median: {int(monthly.median()):,}  Max: {monthly.max():,}")

    axes[1].hist(obs_per_stock, bins=100, color="steelblue", alpha=0.8)
    axes[1].axvline(252, color="red", lw=1.5, ls="--", label="252 days (1 yr)")
    axes[1].axvline(int(obs_per_stock.median()), color="black", lw=1.5, ls="--",
                    label=f"Median = {int(obs_per_stock.median())}")
    axes[1].set_title("Trading days per PERMNO")
    axes[1].set_xlabel("# days")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "crsp_universe_size.png", bbox_inches="tight")
    plt.close()


def plot_cumulative_returns(crsp: pd.DataFrame) -> None:
    print("Plot: cumulative returns")
    daily = crsp.groupby("date").agg(
        ew_ret=("ret", "mean"),
        mkt_ret=("mkt_ret", "first"),
    ).reset_index()
    daily["cum_ew"]  = np.log1p(daily["ew_ret"]).cumsum()
    daily["cum_mkt"] = np.log1p(daily["mkt_ret"]).cumsum()

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(daily["date"], daily["cum_mkt"], label="Market (sprtrn)", color="black", lw=1)
    ax.plot(daily["date"], daily["cum_ew"],  label="Equal-weighted avg", color="steelblue",
            lw=1, alpha=0.8)
    for yr, label in [("2000-03", "Dot-com"), ("2008-09", "GFC"), ("2020-02", "COVID")]:
        ax.axvline(pd.Timestamp(yr), color="red", lw=0.8, ls="--", alpha=0.5)
        ax.text(pd.Timestamp(yr), ax.get_ylim()[0], label,
                fontsize=7, color="red", rotation=90, va="bottom")
    ax.legend()
    ax.set_title("Cumulative log return")
    ax.set_ylabel("Cum. log ret")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "crsp_cumulative_returns.png", bbox_inches="tight")
    plt.close()


def plot_feature_distributions(feat_raw: pd.DataFrame) -> None:
    print("Plot: feature distributions")
    groups = {
        "Short-term reversal":   ["log_reversal_1d"],
        "Momentum":              [f"mom_{w}d" for w in config.MOMENTUM_WINDOWS],
        "Volatility":            [f"vol_{w}d" for w in config.VOLATILITY_WINDOWS],
        "Vol-weighted momentum": [f"vol_w_mom_{w}d_std_{w}d" for w in config.MOMENTUM_WINDOWS],
    }
    sample = feat_raw.dropna().sample(min(200_000, len(feat_raw)), random_state=42)

    for group_name, cols in groups.items():
        cols = [c for c in cols if c in sample.columns]
        if not cols:
            continue
        n = len(cols)
        fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            vals = sample[col].dropna()
            lo, hi = vals.quantile(0.01), vals.quantile(0.99)
            ax.hist(vals.clip(lo, hi), bins=80, color="steelblue", edgecolor="none", alpha=0.8)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_title(col, fontsize=9)
            ax.set_xlabel("Raw value")
        fig.suptitle(
            f"{group_name} — raw distributions (clipped to [p1,p99] for display)", fontsize=11
        )
        plt.tight_layout()
        plt.savefig(
            config.PLOTS_DIR / f"{group_name.lower().replace(' ', '_')}_raw_distributions.png",
            bbox_inches="tight",
        )
        plt.close()


def plot_feature_correlations(feat_raw: pd.DataFrame) -> None:
    print("Plot: feature Spearman correlation matrix")
    raw_for_corr = [
        c for c in feat_raw.columns
        if c not in {"PERMNO", "date", "ret", "mkt_ret", "target", "log_reversal_mkt_1d"}
    ]
    sample_c = feat_raw[raw_for_corr].dropna().sample(min(100_000, len(feat_raw)), random_state=42)
    corr = sample_c.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.3, cbar_kws={"shrink": 0.6}, ax=ax,
        annot=True, fmt=".2f", annot_kws={"size": 6},
    )
    ax.set_title("Spearman cross-correlation — raw features")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "feature_spearman_correlation.png", bbox_inches="tight")
    plt.close()


def plot_ic_analysis(feat: pd.DataFrame, cs_cols: list[str]) -> None:
    print("Plot: rank IC analysis")
    ic = rank_ic_table(feat, cs_cols)
    print(ic.to_string(float_format=lambda x: f"{x:.4f}"))

    top5 = ic["t_stat"].abs().nlargest(5).index.tolist()
    colors = ["tomato" if v < 0 else "steelblue" for v in ic["mean_IC"]]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].bar(range(len(ic)), ic["mean_IC"].values, color=colors, alpha=0.8)
    axes[0].set_xticks(range(len(ic)))
    axes[0].set_xticklabels(ic.index, rotation=45, ha="right", fontsize=8)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Mean Rank IC per feature")
    axes[0].set_ylabel("Mean IC")

    axes[1].bar(range(len(ic)), ic["t_stat"].values, color=colors, alpha=0.8)
    axes[1].set_xticks(range(len(ic)))
    axes[1].set_xticklabels(ic.index, rotation=45, ha="right", fontsize=8)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axhline( 2, color="gray", ls="--", lw=0.8, label="|t|=2")
    axes[1].axhline(-2, color="gray", ls="--", lw=0.8)
    axes[1].set_title("t-statistic of mean IC  (|t| > 2 ≈ significant)")
    axes[1].set_ylabel("t-stat")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "feature_rank_ic.png", bbox_inches="tight")
    plt.close()

    # Rolling IC for top-5 features
    fig, ax = plt.subplots(figsize=(13, 4))
    for col in top5:
        sub = feat[["date", col, "target"]].dropna()
        ic_ts = (
            sub.groupby("date")
            .apply(
                lambda g: spearmanr(g[col], g["target"])[0] if len(g) > 10 else np.nan,
                include_groups=False,
            )
            .dropna()
            .rolling(63, min_periods=30)
            .mean()
        )
        ax.plot(ic_ts.index, ic_ts.values, label=col, lw=1.2)
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(ncol=3, fontsize=8)
    ax.set_title("Rolling 63-day mean Rank IC — top 5 features by |t-stat|")
    ax.set_ylabel("Mean IC (63d rolling)")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "feature_rank_ic_rolling.png", bbox_inches="tight")
    plt.close()

    # Cumulative IC
    fig, ax = plt.subplots(figsize=(13, 4))
    for col in top5:
        sub = feat[["date", col, "target"]].dropna()
        ic_ts = (
            sub.groupby("date")
            .apply(
                lambda g: spearmanr(g[col], g["target"])[0] if len(g) > 10 else np.nan,
                include_groups=False,
            )
            .dropna()
            .cumsum()
        )
        ax.plot(ic_ts.index, ic_ts.values, label=col, lw=1.2)
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(ncol=3, fontsize=8)
    ax.set_title("Cumulative Rank IC — top 5 features")
    ax.set_ylabel("Cumulative IC")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "signal_cumulative_ic.png", bbox_inches="tight")
    plt.close()


def plot_vix_analysis(feat: pd.DataFrame) -> None:
    vix_path = config.VIX_PATH_CLEAN
    if not vix_path.exists():
        print("VIX processed file not found — skipping VIX plots")
        return
    print("Plot: VIX analysis")
    vix = pd.read_parquet(vix_path)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    vix_pct = vix["vix"] * 100
    axes[0].plot(vix.index, vix_pct, color="steelblue", lw=0.8)
    axes[0].axhline(20, color="orange", lw=1, ls="--", label="VIX = 20")
    axes[0].axhline(30, color="red",    lw=1, ls="--", label="VIX = 30")
    for start, end in [("2000-03", "2002-10"), ("2008-09", "2009-06"), ("2020-02", "2020-05")]:
        axes[0].axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.12, color="red")
    axes[0].set_title("CBOE VIX level (normalised back to %)")
    axes[0].set_ylabel("VIX (%)")
    axes[0].legend(fontsize=8)

    vix_daily = vix["vix"].reset_index()
    vix_daily.columns = ["date", "vix_norm"]
    vix_daily["date"] = vix_daily["date"].astype(feat["date"].dtype)
    feat_m = feat[["date", "target"]].merge(vix_daily, on="date", how="left").dropna()
    feat_m["vix_q"] = pd.qcut(feat_m["vix_norm"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    cs_std = (
        feat_m.groupby(["date", "vix_q"], observed=True)["target"]
        .std()
        .groupby("vix_q", observed=True)
        .mean()
    )
    cs_std.plot(kind="bar", ax=axes[1], color="steelblue", alpha=0.8, edgecolor="none", width=0.6)
    axes[1].set_title("Mean cross-sectional return std by VIX quartile")
    axes[1].set_ylabel("Avg cross-sect. std of next-day return")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "vix_and_return_dispersion.png", bbox_inches="tight")
    plt.close()


def plot_cz_analysis(crsp: pd.DataFrame) -> None:
    cz_path = config.CZ_PATH_CLEAN
    if not cz_path.exists():
        print("CZ processed file not found — skipping CZ plots")
        return
    print("Plot: Chen-Zimmermann factor analysis")
    cz = pd.read_parquet(cz_path)
    if "date" not in cz.columns:          # date may be stored as a named index
        cz = cz.reset_index()
    cz_cols = [c for c in cz.columns if c != "date"]
    cz["date"] = pd.to_datetime(cz["date"])

    sample_factors = cz_cols[:min(12, len(cz_cols))]
    n = len(sample_factors)
    ncols, nrows = 4, (n + 3) // 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else list(axes)
    for ax, col in zip(axes, sample_factors):
        vals = cz[col].dropna()
        lo, hi = vals.quantile(0.01), vals.quantile(0.99)
        ax.hist(vals.clip(lo, hi), bins=40, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(col, fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("CZ factor return distributions (monthly, clipped to [p1,p99])", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "cz_factor_distributions.png", bbox_inches="tight")
    plt.close()

    daily_ew = crsp.groupby("date")["ret"].mean().reset_index()
    daily_ew.columns = ["date", "ew_ret"]
    daily_ew["date"] = daily_ew["date"].astype(cz["date"].dtype)
    ts = daily_ew.merge(cz, on="date", how="inner")
    ts_corr = pd.Series(
        {col: ts["ew_ret"].corr(ts[col]) for col in cz_cols}
    ).sort_values(key=abs, ascending=False)
    colors_ts = ["tomato" if v < 0 else "steelblue" for v in ts_corr.values]
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.bar(range(len(ts_corr)), ts_corr.values, color=colors_ts, alpha=0.8)
    ax.set_xticks(range(len(ts_corr)))
    ax.set_xticklabels(ts_corr.index, rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Pearson correlation of CZ factors with equal-weighted daily return")
    ax.set_ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "cz_factor_correlations.png", bbox_inches="tight")
    plt.close()


def plot_train_val_test_split(feat: pd.DataFrame) -> None:
    print("Plot: train / val / test split")
    splits = generate_date_split(feat, df_to_join=[], date_col="date",
                                 train_split=0.70, val_split=0.15)
    fig, ax = plt.subplots(figsize=(13, 1.8))
    split_colors = {"train": "steelblue", "val": "darkorange", "test": "seagreen"}
    left = 0
    for name, (start, end) in splits.items():
        width = (end - start).days
        ax.barh(0, width, left=left, color=split_colors[name], alpha=0.8, height=0.5)
        ax.text(left + width / 2, 0, f"{name}\n{start.year}–{end.year}",
                ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        left += width
    ax.set_yticks([])
    ax.set_xlabel("Calendar days")
    ax.set_title("Train / Val / Test — strict chronological split (70 / 15 / 15 by trading days)")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "date_splits.png", bbox_inches="tight")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # ---- Pipeline ----
    crsp, feat, feat_raw = run_pipeline()

    # ---- EDA plots ----
    feat_cols = [c for c in feat.columns
                 if c not in {"PERMNO", "date", "ret", "mkt_ret", "target"}]
    cs_cols   = [c for c in feat_cols if c != "log_reversal_mkt_1d"]

    print("\n" + "=" * 60)
    print("Step 5: EDA plots  →  " + str(config.PLOTS_DIR))
    print("=" * 60)

    plot_return_distribution(crsp)
    plot_universe_size(crsp)
    plot_cumulative_returns(crsp)
    plot_feature_distributions(feat_raw)
    plot_feature_correlations(feat_raw)
    plot_ic_analysis(feat, cs_cols)
    plot_vix_analysis(feat)
    plot_cz_analysis(crsp)
    plot_train_val_test_split(feat)

    print("\nDone. All outputs saved to:")
    print(f"  Processed data  →  {config.DATA_PROCESSED}")
    print(f"  Plots           →  {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
