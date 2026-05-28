"""

Regime comparison: rule-based strategy performance on 2000-2018 vs 2019-2024.

The strategies here are rule-based (no fitted model), so evaluating them on the
earlier 2000-2018 window is a clean comparison ; it characterises *when* the
factor strategies work and shows the regime-dependence behind the negative
2019-2024 result. Fitted models (XGBoost / neural net) are excluded: they are
trained on data up to 2015, so a 2000-2018 evaluation would be in-sample.

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.backtest import (CostModel, long_short_weights, long_only_weights,
                          held_weights, run_backtest, performance_metrics)

# Set Global Variable
WINDOWS = {
    "2000-2018": ("2000-01-01", "2018-12-31"),
    "2019-2024": ("2019-01-01", "2024-12-31"),
}

_LS = lambda s: long_short_weights(s, quantile=0.10)
_LO = lambda s: long_only_weights(s, quantile=0.20)
_EW = lambda s: long_only_weights(s, quantile=1.0)

# name -> (signal column, weight_fn, rebalance step in days, universe N or None)
STRATEGIES = {
    "Reversal L/S weekly, full":      ("reversal", _LS, 5,  None),
    "Reversal L/S weekly, top1000":   ("reversal", _LS, 5,  1000),
    "Low-vol long-only, top1000":     ("lowvol",   _LO, 21, 1000),
    "Momentum long-only, top1000":    ("momentum", _LO, 21, 1000),
    "Universe equal-weight, top1000": ("lowvol",   _EW, 21, 1000),
}


def run_window(
    feat: pd.DataFrame, 
    liq: pd.DataFrame, start: str,
    end: str
) -> pd.DataFrame:
    
    # Ranking on capitalization
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    df = feat[(feat["date"] >= s) & (feat["date"] <= e)].copy()
    lq = liq[(liq["date"] >= s) & (liq["date"] <= e)].copy()
    lq["cap_rank"] = lq.groupby("date")["cap_lag"].rank(ascending=False, method="first")
    df = df.merge(lq[["date", "PERMNO", "cap_rank"]], on=["date", "PERMNO"], how="left")
    df["cap_rank"] = df["cap_rank"].astype("float64")
    df = df.set_index(["date", "PERMNO"]).sort_index()

    # Initialize cost model
    returns = df["ret"]
    dates = returns.index.get_level_values("date").unique().sort_values()
    cost = CostModel(0.001, 0.01)

    # Backtest loop
    rows = {}
    for name, (col, wfn, k, N) in STRATEGIES.items():
        sig = df[col] if N is None else df.loc[df["cap_rank"] <= N, col]
        w = held_weights(sig, dates[::k], weight_fn=wfn)
        bt = run_backtest(w, returns, cost)
        rows[name] = performance_metrics(bt["net_return"])

    # Benchmark data (SP500)
    gspc = pd.read_parquet(config.GSPC_PATH_CLEAN)
    gspc["date"] = pd.to_datetime(gspc["date"])
    spx = gspc.set_index("date")["sprtrn"].reindex(dates).fillna(0.0)
    rows["S&P500 (buy & hold)"] = performance_metrics(spx)
    
    
    return pd.DataFrame(rows).T


def main(
) -> None:
    
    # Get data
    feat = pd.read_parquet(config.FEATURES_PATH_CLEAN,
        columns=["date", "PERMNO", "ret", "log_reversal_1d", "vol_252d", "mom_252d"])
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat.drop_duplicates(subset=["date", "PERMNO"], keep="last")
    feat["reversal"] = -feat["log_reversal_1d"].astype("float64")
    feat["lowvol"] = -feat["vol_252d"].astype("float64")
    feat["momentum"] = feat["mom_252d"].astype("float64")
    feat = feat.dropna(subset=["reversal", "lowvol", "momentum", "ret"])

    # Get liquidity
    liq = pd.read_parquet(config.CRSP_LIQUIDITY_RAW, columns=["date", "PERMNO", "dlycap"])
    liq["date"] = pd.to_datetime(liq["date"])
    liq["dlycap"] = liq["dlycap"].astype("float64")
    liq = liq.sort_values(["PERMNO", "date"])
    liq["cap_lag"] = liq.groupby("PERMNO")["dlycap"].shift(1) # no look-ahead

    # Get and display results
    results = {}
    for wname, (start, end) in WINDOWS.items():
        m = run_window(feat, liq, start, end)
        results[wname] = m
        print(f"\n {wname} (net of costs) ")
        print(m[["ann_return", "ann_vol", "sharpe", "max_drawdown"]].to_string(
            formatters={"ann_return": "{:+.2%}".format, "ann_vol": "{:.2%}".format,
                        "sharpe": "{:+.2f}".format, "max_drawdown": "{:.1%}".format}))

    sharpe = pd.DataFrame({w: results[w]["sharpe"] for w in WINDOWS})
    print("\n Sharpe comparison across regimes ")
    print(sharpe.to_string(float_format=lambda x: f"{x:+.2f}"))

    # Create and save graph
    ax = sharpe.plot.bar(figsize=(11, 5), rot=15)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Net Sharpe by strategy and regime")
    ax.set_ylabel("net Sharpe")
    plt.tight_layout()
    out = config.PLOTS_DIR / "regime_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()




"""

Commentary on "regime_comparison.png"


First of all, we observe that the Reversal L/S weekly (top 1000) achieves negative returns in both 
cases, so we won't analyze it further. The Reversal strategy is again very effective on the full 
universe (which in reality is driven by mid and small caps), but part of this edge could disappear 
when we apply more realistic trading fees. Additionally, a clear performance degradation can be observed 
through time. The low volatility strategy is relatively consistent but does decrease between the two periods.

Momentum, universe equal-weight, and the S&P 500 are all increasing impressively between the periods. 
We can hypothesize that the market transitioned to a strong bullish regime, thus heavily rewarding 
long-only strategies. This is consistent with the explosion of the tech sector, with AI heavily boosting 
the broader market.


Sharpe comparison across regimes 
                                2000-2018  2019-2024
Reversal L/S weekly, full           +2.31      +1.00
Reversal L/S weekly, top1000        -0.57      -0.13
Low-vol long-only, top1000          +0.76      +0.54
Momentum long-only, top1000         +0.33      +0.67
Universe equal-weight, top1000      +0.41      +0.62
S&P500 (buy & hold)                 +0.19      +0.76


"""
