"""
Turnover / rebalancing-frequency experiment.

v1 showed the gross edge is real but ~81%/yr transaction cost — from near-full
daily rebalancing of the fast reversal signal — turns net Sharpe negative.
This experiment maps the tradable frontier: it sweeps the rebalancing frequency
(daily / weekly / monthly) against the signal mix (full ensemble / no reversal
/ reversal only) and reports net Sharpe, gross Sharpe and the annualised cost.

The combiner is a simple equal-weight average of the orientation-corrected
signals — turnover is a property of the signal inputs, not of the combiner, so
the cost story is identical to XGBoost's. XGBoost can be layered back onto the
winning cell afterwards. Same test window, cost model and decile construction
as v1, so the (full, daily) cell reproduces v1's equal-weight result.
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

SIGNALS = {
    "reversal": ("log_reversal_1d", -1.0),
    "lowvol":   ("vol_63d", -1.0),
    "momentum": ("mom_252d", +1.0),
}
MIXES = {
    "full": ["reversal", "lowvol", "momentum"],
    "no_reversal": ["lowvol", "momentum"],
    "reversal_only": ["reversal"],
}
FREQS = {"daily": 1, "weekly": 5, "monthly": 21}
VAL_END = pd.Timestamp(config.VAL_END)


def main() -> None:
    raw = [c for c, _ in SIGNALS.values()]
    df = pd.read_parquet(config.FEATURES_PATH_CLEAN,
                         columns=["date", "PERMNO", "ret"] + raw)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "PERMNO"], keep="last")
    for name, (col, sign) in SIGNALS.items():
        df[name] = sign * df[col].astype("float64")
    df = df.dropna(subset=list(SIGNALS) + ["ret"])
    df = df[df["date"] > VAL_END]                       # test window, as in v1
    print(f"test window: {df['date'].min().date()} -> {df['date'].max().date()} "
          f"({df['date'].nunique()} days)")

    df = df.set_index(["date", "PERMNO"]).sort_index()
    returns = df["ret"]
    dates = returns.index.get_level_values("date").unique().sort_values()
    cost = CostModel(trading_cost=0.001, borrow_cost_annual=0.01)

    gspc = pd.read_parquet(config.GSPC_PATH_CLEAN)
    gspc["date"] = pd.to_datetime(gspc["date"])
    spx = gspc.set_index("date")["sprtrn"].reindex(dates).fillna(0.0)
    spx_sharpe = performance_metrics(spx)["sharpe"]

    net, gross, anncost = {}, {}, {}
    for mix_name, cols in MIXES.items():
        signal = df[cols].mean(axis=1)
        for freq_name, k in FREQS.items():
            w = held_weights(signal, dates[::k], quantile=0.10)
            bt = run_backtest(w, returns, cost)
            net[(mix_name, freq_name)] = performance_metrics(bt["net_return"])["sharpe"]
            gross[(mix_name, freq_name)] = performance_metrics(bt["gross_return"])["sharpe"]
            anncost[(mix_name, freq_name)] = bt["cost"].mean() * 252

    def grid(d: dict) -> pd.DataFrame:
        return pd.DataFrame([[d[(m, f)] for f in FREQS] for m in MIXES],
                            index=list(MIXES), columns=list(FREQS))

    net_g, gross_g, cost_g = grid(net), grid(gross), grid(anncost)
    print(f"\nS&P500 reference Sharpe: {spx_sharpe:+.2f}\n")
    print("=== NET Sharpe (net of costs) ===")
    print(net_g.to_string(float_format=lambda x: f"{x:+.2f}"))
    print("\n=== GROSS Sharpe ===")
    print(gross_g.to_string(float_format=lambda x: f"{x:+.2f}"))
    print("\n=== Annualised transaction cost ===")
    print(cost_g.to_string(float_format=lambda x: f"{x:.1%}"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(net_g, annot=True, fmt="+.2f", cmap="RdBu_r", center=0,
                ax=ax, cbar_kws={"label": "net Sharpe"})
    ax.set_title(f"Net Sharpe — signal mix x rebalance frequency "
                 f"(S&P500 = {spx_sharpe:+.2f})")
    out = config.PLOTS_DIR / "turnover_experiment.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved heatmap -> {out}")


if __name__ == "__main__":
    main()
