"""
Liquidity-filtered re-run of the reversal strategy.

The turnover experiment found weekly-rebalanced reversal reaches net positive Sharpe
but on the full CRSP universe, where the gross edge is largely the
uncapturable micro-cap bid-ask bounce. This script restricts the universe to
the N largest stocks by *lagged* market cap (no look-ahead) and re-runs reversal
at daily and weekly frequency, sweeping N, to measure how much of that net positive Sharpe
survives on a genuinely tradable universe, where the flat 10 bps cost is also
realistic (it was optimistic for micro-caps).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.backtest import CostModel, held_weights, run_backtest, performance_metrics

VAL_END = pd.Timestamp(config.VAL_END) # Retrieve validation end date
UNIVERSES = {"top 500": 500, "top 1000": 1000, "top 2000": 2000, "full": None} # Choose top N
FREQS = {"daily": 1, "weekly": 5} # Choose frequencies


def main(
    )-> None:
    
    # signals + returns (reversal signal = -log_reversal_1d)
    df = pd.read_parquet(config.FEATURES_PATH_CLEAN,
                         columns=["date", "PERMNO", "ret", "log_reversal_1d"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "PERMNO"], keep="last")
    df["reversal"] = -df["log_reversal_1d"].astype("float64")
    df = df.dropna(subset=["reversal", "ret"])
    df = df[df["date"] > VAL_END]

    # liquidity: lagged market-cap rank per date (rank 1 = largest)
    liq = pd.read_parquet(config.CRSP_LIQUIDITY_RAW, columns=["date", "PERMNO", "dlycap"])
    liq["date"] = pd.to_datetime(liq["date"])
    liq["dlycap"] = liq["dlycap"].astype("float64")
    liq = liq.sort_values(["PERMNO", "date"])
    liq["cap_lag"] = liq.groupby("PERMNO")["dlycap"].shift(1)        # no look-ahead
    liq = liq[liq["date"] > VAL_END]
    liq["cap_rank"] = liq.groupby("date")["cap_lag"].rank(ascending=False, method="first")

    # Merging together
    df = df.merge(liq[["date", "PERMNO", "cap_rank"]], on=["date", "PERMNO"], how="left")
    df["cap_rank"] = df["cap_rank"].astype("float64")
    df = df.set_index(["date", "PERMNO"]).sort_index()

    # Getting returns and initializing a cost model
    returns = df["ret"]
    dates = returns.index.get_level_values("date").unique().sort_values()
    cost = CostModel(trading_cost=0.001, borrow_cost_annual=0.01)

    # Get SP500 data as benchmark
    gspc = pd.read_parquet(config.GSPC_PATH_CLEAN)
    gspc["date"] = pd.to_datetime(gspc["date"])
    spx = gspc.set_index("date")["sprtrn"].reindex(dates).fillna(0.0)
    spx_sharpe = performance_metrics(spx)["sharpe"]

    # Get the results for each combination of Universe and Frequency
    net, gross, anncost, sizes, bts = {}, {}, {}, {}, {}
    for uni, N in UNIVERSES.items():
        sig = df["reversal"] if N is None else df.loc[df["cap_rank"] <= N, "reversal"]
        sizes[uni] = sig.groupby(level="date").size().mean()
        for freq, k in FREQS.items():
            w = held_weights(sig, dates[::k], quantile=0.10)
            bt = run_backtest(w, returns, cost)
            bts[(uni, freq)] = bt
            net[(uni, freq)] = performance_metrics(bt["net_return"])["sharpe"]
            gross[(uni, freq)] = performance_metrics(bt["gross_return"])["sharpe"]
            anncost[(uni, freq)] = bt["cost"].mean() * 252


    def grid(
        d: dict
        ) -> pd.DataFrame:
        
        return pd.DataFrame([[d[(u, f)] for f in FREQS] for u in UNIVERSES],
                            index=list(UNIVERSES), columns=list(FREQS))

    net_g, gross_g, cost_g = grid(net), grid(gross), grid(anncost)
    
    # Display data
    print(f"test {dates.min().date()} -> {dates.max().date()} | "
          f"S&P500 Sharpe {spx_sharpe:+.2f} | cost = 10 bps flat\n")
    print("avg universe size (stocks/day): "
          + ", ".join(f"{u} = {sizes[u]:.0f}" for u in UNIVERSES))
    print("\n NET Sharpe ")
    print(net_g.to_string(float_format=lambda x: f"{x:+.2f}"))
    print("\n GROSS Sharpe ")
    print(gross_g.to_string(float_format=lambda x: f"{x:+.2f}"))
    print("\n Annualised transaction cost ")
    print(cost_g.to_string(float_format=lambda x: f"{x:.1%}"))

    # Create and Save graph
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(net_g, annot=True, fmt="+.2f", cmap="RdBu_r", center=0, ax=axes[0],
                cbar_kws={"label": "net Sharpe"})
    axes[0].set_title(f"Net Sharpe — reversal (S&P500 = {spx_sharpe:+.2f})")
    for uni in UNIVERSES:
        bt = bts[(uni, "weekly")]
        axes[1].plot(bt.index, bt["equity"], label=f"weekly, {uni}", lw=1.1)
    axes[1].plot(spx.index, (1.0 + spx).cumprod(), "k--", lw=1.1, label="S&P500")
    axes[1].axhline(1.0, color="grey", lw=0.8)
    axes[1].set_title("Weekly reversal — equity curve by universe")
    axes[1].set_ylabel("equity (start = 1)")
    axes[1].legend(fontsize=8)
    out = config.PLOTS_DIR / "liquidity_filter.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
