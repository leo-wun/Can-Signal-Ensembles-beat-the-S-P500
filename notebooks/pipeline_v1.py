"""

End-to-end v1 pipeline: cross-sectional signal ensemble -> XGBoost meta-model
-> long-short portfolio -> backtest vs S&P500.

Base signals (validated earlier): short-term reversal, low-volatility,
long-horizon momentum. They are oriented so a higher value means a higher
expected return, then fed to an XGBoost meta-model whose output is the master
signal. The meta-model target is the cross-sectional percentile rank of the
next-day return (robust to return outliers, consistent with the rank-IC work).

Benchmarked against (a) an equal-weight combination of the same three signals
and (b) the S&P500. An optional VIX-based exposure overlay is also reported.

Caveat: v1 trades the full CRSP universe with no liquidity filter — micro-cap
reversal is overstated and not realistically tradable. Read the results as a
pipeline check, not a deployable strategy.

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.backtest import CostModel, long_short_weights, run_backtest, performance_metrics

# Set Global Variable
# (raw column, orientation) — orientation makes "higher value = more bullish"
SIGNALS = {
    "reversal": ("log_reversal_1d", -1.0),
    "lowvol":   ("vol_63d",         -1.0),
    "momentum": ("mom_252d",        +1.0),
}
TRAIN_END = pd.Timestamp(config.TRAIN_END)   # 2015-12-31
VAL_END = pd.Timestamp(config.VAL_END)       # 2018-12-31


def main(
) -> None:
    
    # Get data
    feat = list(SIGNALS.keys())
    raw_cols = [c for c, _ in SIGNALS.values()]
    df = pd.read_parquet(config.FEATURES_PATH_CLEAN,
                         columns=["date", "PERMNO", "ret", "target"] + raw_cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "PERMNO"], keep="last")
    for name, (col, sign) in SIGNALS.items():
        df[name] = sign * df[col].astype("float64")
    df = df.dropna(subset=feat + ["target", "ret"])

    # Training, validation and test split
    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test = df[df["date"] > VAL_END].copy()
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")
    print(f"test window: {test['date'].min().date()} -> {test['date'].max().date()}")

    # Cross-sectional percentile rank of next-day return
    y_train = train.groupby("date")["target"].rank(pct=True)
    y_val = val.groupby("date")["target"].rank(pct=True)

    # Initialize, train and predict XGboost regressor
    model = XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        early_stopping_rounds=40, eval_metric="rmse", random_state=0,
    )
    model.fit(train[feat], y_train, eval_set=[(val[feat], y_val)], verbose=False)
    print(f"XGBoost best iteration: {model.best_iteration}")
    print("feature importance: "
          + ", ".join(f"{f}={imp:.2f}"
                      for f, imp in zip(feat, model.feature_importances_)))

    test["signal_xgb"] = model.predict(test[feat])
    test["signal_ew"] = test[feat].mean(axis=1)

    # Get test index and initialize cost model
    test_idx = test.set_index(["date", "PERMNO"]).sort_index()
    returns = test_idx["ret"]
    cost = CostModel(trading_cost=0.001, borrow_cost_annual=0.01)

    # Backtest loop
    backtests = {}
    for label, col in [("XGBoost ensemble", "signal_xgb"),
                       ("Equal-weight combo", "signal_ew")]:
        w = long_short_weights(test_idx[col], quantile=0.10)
        backtests[label] = run_backtest(w, returns, cost)

    # VIX exposure overlay on the XGBoost strategy (scale down when VIX is high)
    vix = pd.read_parquet(config.VIX_PATH_CLEAN)["vix"].astype("float64")
    bt_dates = backtests["XGBoost ensemble"].index
    vix_aligned = vix.reindex(bt_dates).ffill()
    exposure = (vix_aligned.median() / vix_aligned).clip(0.3, 2.0)
    w_xgb = long_short_weights(test_idx["signal_xgb"], quantile=0.10)
    backtests["XGBoost + VIX overlay"] = run_backtest(w_xgb, returns, cost, exposure)

    # S&P500 benchmark over the same dates
    gspc = pd.read_parquet(config.GSPC_PATH_CLEAN)
    gspc["date"] = pd.to_datetime(gspc["date"])
    spx = gspc.set_index("date")["sprtrn"].reindex(bt_dates).fillna(0.0)

    # Get results
    rows = {}
    for label, bt in backtests.items():
        m = performance_metrics(bt["net_return"])
        m["sharpe_gross"] = performance_metrics(bt["gross_return"])["sharpe"]
        m["ann_cost"] = bt["cost"].mean() * 252
        rows[label] = m
    rows["S&P500 (buy & hold)"] = performance_metrics(spx)
    
    #Display results
    metrics = pd.DataFrame(rows).T[
        ["total_return", "ann_return", "ann_vol", "sharpe",
         "sharpe_gross", "ann_cost", "max_drawdown"]]
    print("\n Performance on test set (net of costs) ")
    print(metrics.to_string(formatters={
        "total_return": "{:+.1%}".format, "ann_return": "{:+.2%}".format,
        "ann_vol": "{:.2%}".format, "sharpe": "{:+.2f}".format,
        "sharpe_gross": "{:+.2f}".format, "ann_cost": "{:.2%}".format,
        "max_drawdown": "{:.1%}".format}))

    # Create and save graph
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, bt in backtests.items():
        ax.plot(bt.index, bt["equity"], label=label, lw=1.2)
    ax.plot(spx.index, (1.0 + spx).cumprod(), label="S&P500 (buy & hold)",
            color="black", lw=1.2, ls="--")
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.set_title("v1 pipeline — equity curves (test set, net of costs)")
    ax.set_ylabel("equity (start = 1)")
    ax.legend()
    out = config.PLOTS_DIR / "pipeline_v1_equity.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved equity-curve plot -> {out}")


if __name__ == "__main__":
    main()


"""

Commentary on "pipeline_v1_equity.png"

Depict a clear failure. No prediction power. 

The features used maybe are of no use or maybe it is the more noisy nature of daily returns.
We observe that no model perform especially better than other so the source of problem could be the way 
we handled data.


train 13,746,339 | val 1,543,912 | test 2,566,885
test window: 2019-01-02 -> 2024-12-31
XGBoost best iteration: 43
feature importance: reversal=0.78, lowvol=0.16, momentum=0.06

 Performance on test set (net of costs) 
                      total_return ann_return ann_vol sharpe sharpe_gross ann_cost max_drawdown
XGBoost ensemble            -81.5%    -24.55%  19.96%  -1.23        +3.44   80.54%       -82.9%
Equal-weight combo          -94.0%    -37.43%  24.60%  -1.52        +0.71   62.91%       -93.8%
XGBoost + VIX overlay       -83.1%    -25.65%  16.77%  -1.53        +4.07   81.79%       -84.2%
S&P500 (buy & hold)        +134.6%    +15.29%  20.13%  +0.76          NaN      NaN       -33.9%

"""
