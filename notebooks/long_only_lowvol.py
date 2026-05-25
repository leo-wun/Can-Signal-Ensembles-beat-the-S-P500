"""
Long-only low-volatility strategy.

Reversal did not survive the liquidity filter. This pivots to the
low-volatility anomaly in its natural long-only form: stay 100% invested but
hold the lowest-volatility stocks. The route to a market-beating Sharpe is
lower portfolio volatility for a roughly market-like return (Baker-Bradley-
Wurgler 2011), NOT pure long-short alpha.

Universe: the top-N stocks by lagged market cap (tradable). Volatility rank =
vol_252d (already lagged). Monthly rebalancing (slow signal -> low turnover),
10 bps cost, no borrow cost (long-only). Compared against an equal-weight
portfolio of the same universe (the like-for-like benchmark) and the S&P500
(the stated benchmark — note sprtrn is a price return, so the EW portfolios
get a small dividend edge).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.backtest import (CostModel, long_only_weights, held_weights,
                          run_backtest, performance_metrics)

VAL_END = pd.Timestamp(config.VAL_END)
UNIVERSE_N = 1000          # tradable universe = top-N by lagged market cap
QUANTILE = 0.20            # quintile portfolios
REBAL = 21                 # monthly rebalancing


def main() -> None:
    df = pd.read_parquet(config.FEATURES_PATH_CLEAN,
                         columns=["date", "PERMNO", "ret", "vol_252d"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "PERMNO"], keep="last")
    df["vol_252d"] = df["vol_252d"].astype("float64")
    df = df.dropna(subset=["vol_252d", "ret"])
    df = df[df["date"] > VAL_END]

    liq = pd.read_parquet(config.CRSP_LIQUIDITY_RAW, columns=["date", "PERMNO", "dlycap"])
    liq["date"] = pd.to_datetime(liq["date"])
    liq["dlycap"] = liq["dlycap"].astype("float64")
    liq = liq.sort_values(["PERMNO", "date"])
    liq["cap_lag"] = liq.groupby("PERMNO")["dlycap"].shift(1)        # no look-ahead
    liq = liq[liq["date"] > VAL_END]
    liq["cap_rank"] = liq.groupby("date")["cap_lag"].rank(ascending=False, method="first")

    df = df.merge(liq[["date", "PERMNO", "cap_rank"]], on=["date", "PERMNO"], how="left")
    df["cap_rank"] = df["cap_rank"].astype("float64")
    df = df.set_index(["date", "PERMNO"]).sort_index()

    returns = df["ret"]                                  # all test stocks
    uni = df[df["cap_rank"] <= UNIVERSE_N]                # tradable universe (signals)
    dates = returns.index.get_level_values("date").unique().sort_values()
    rebal = dates[::REBAL]
    cost = CostModel(trading_cost=0.001, borrow_cost_annual=0.0)     # long-only

    def lo(q):
        return lambda s: long_only_weights(s, quantile=q)

    strategies = {
        "Low-vol quintile":       (-uni["vol_252d"], lo(QUANTILE)),
        "High-vol quintile":      (uni["vol_252d"],  lo(QUANTILE)),
        "Universe equal-weight":  (uni["vol_252d"],  lo(1.0)),
    }

    gspc = pd.read_parquet(config.GSPC_PATH_CLEAN)
    gspc["date"] = pd.to_datetime(gspc["date"])
    spx = gspc.set_index("date")["sprtrn"].reindex(dates).fillna(0.0)

    rows, bts = {}, {}
    for name, (sig, wfn) in strategies.items():
        w = held_weights(sig, rebal, weight_fn=wfn)
        bt = run_backtest(w, returns, cost)
        bts[name] = bt
        rows[name] = performance_metrics(bt["net_return"])
    rows["S&P500 (buy & hold)"] = performance_metrics(spx)

    metrics = pd.DataFrame(rows).T[
        ["total_return", "ann_return", "ann_vol", "sharpe", "max_drawdown"]]
    print(f"test {dates.min().date()} -> {dates.max().date()} | "
          f"universe = top {UNIVERSE_N} by mkt cap | monthly rebalance | 10 bps cost\n")
    print(metrics.to_string(formatters={
        "total_return": "{:+.1%}".format, "ann_return": "{:+.2%}".format,
        "ann_vol": "{:.2%}".format, "sharpe": "{:+.2f}".format,
        "max_drawdown": "{:.1%}".format}))

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, bt in bts.items():
        ax.plot(bt.index, bt["equity"], label=name, lw=1.2)
    ax.plot(spx.index, (1.0 + spx).cumprod(), "k--", lw=1.2, label="S&P500")
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.set_title("Long-only low-volatility — equity curves (test, net of costs)")
    ax.set_ylabel("equity (start = 1)")
    ax.legend()
    out = config.PLOTS_DIR / "long_only_lowvol.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
