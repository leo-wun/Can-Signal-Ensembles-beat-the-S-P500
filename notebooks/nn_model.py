"""
Neural-net meta-model — comparison with XGBoost and an equal-weight baseline.

For the ML-methodology section: an MLP and XGBoost are trained on the 14
stock-level technical features to predict the cross-sectional rank of the
next-day return, then turned into a weekly decile long-short portfolio.
Compared with the equal-weight blend of the three validated signals and the
S&P500. Train <=2015, validate 2016-2018, test 2019-2024.

Each model is backtested on BOTH the full universe and the tradable top-1000
(by lagged market cap): the gap shows the full-universe edge is the micro-cap
bid-ask bounce, not tradable alpha. The lesson — with signals that lack a
tradable net edge, model choice (NN vs gradient boosting vs linear blend) does
not change the conclusion: the bottleneck is signal, not model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.backtest import CostModel, held_weights, run_backtest, performance_metrics
from src.models.mlp_model import train_mlp

FEATURES = ["log_reversal_1d", "mom_5d", "mom_21d", "mom_63d", "mom_126d", "mom_252d",
            "vol_21d", "vol_63d", "vol_252d", "vol_w_mom_5d_std_5d",
            "vol_w_mom_21d_std_21d", "vol_w_mom_63d_std_63d",
            "vol_w_mom_126d_std_126d", "vol_w_mom_252d_std_252d"]
SIGNALS_3 = {"log_reversal_1d": -1.0, "vol_63d": -1.0, "mom_252d": +1.0}  # EW baseline
TRAIN_END = pd.Timestamp(config.TRAIN_END)
VAL_END = pd.Timestamp(config.VAL_END)
TRAIN_SAMPLE = 3_000_000
UNIVERSES = {"full": None, "top 1000": 1000}


def main() -> None:
    df = pd.read_parquet(config.FEATURES_PATH_CLEAN,
                         columns=["date", "PERMNO", "ret", "target"] + FEATURES)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "PERMNO"], keep="last")
    df = df.dropna(subset=FEATURES + ["target", "ret"])

    train = df[df["date"] <= TRAIN_END].copy()
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
    test = df[df["date"] > VAL_END].copy()
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")

    # target = per-date percentile rank of the next-day return
    train["yrank"] = train.groupby("date")["target"].rank(pct=True)
    val["yrank"] = val.groupby("date")["target"].rank(pct=True)
    train_s = train.sample(n=min(TRAIN_SAMPLE, len(train)), random_state=0)

    scaler = StandardScaler().fit(train_s[FEATURES])
    Xtr = scaler.transform(train_s[FEATURES]).astype("float32")
    Xva = scaler.transform(val[FEATURES]).astype("float32")
    Xte = scaler.transform(test[FEATURES]).astype("float32")
    ytr = train_s["yrank"].to_numpy("float32")
    yva = val["yrank"].to_numpy("float32")

    # XGBoost
    xgb = XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                       early_stopping_rounds=40, eval_metric="rmse", random_state=0)
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    test["pred_xgb"] = xgb.predict(Xte)

    # neural net
    mlp, val_mse = train_mlp(Xtr, ytr, Xva, yva, n_features=len(FEATURES))
    mlp.eval()
    with torch.no_grad():
        dev = next(mlp.parameters()).device
        test["pred_mlp"] = mlp(torch.tensor(Xte, device=dev)).cpu().numpy()
    print(f"MLP best validation MSE: {val_mse:.5f}")

    # equal-weight baseline (3 validated signals, orientation-corrected)
    test["pred_ew"] = sum(s * test[c].astype("float64")
                          for c, s in SIGNALS_3.items()) / len(SIGNALS_3)

    # liquidity universe: lagged market-cap rank
    liq = pd.read_parquet(config.CRSP_LIQUIDITY_RAW, columns=["date", "PERMNO", "dlycap"])
    liq["date"] = pd.to_datetime(liq["date"])
    liq["dlycap"] = liq["dlycap"].astype("float64")
    liq = liq.sort_values(["PERMNO", "date"])
    liq["cap_lag"] = liq.groupby("PERMNO")["dlycap"].shift(1)
    liq = liq[liq["date"] > VAL_END]
    liq["cap_rank"] = liq.groupby("date")["cap_lag"].rank(ascending=False, method="first")
    test = test.merge(liq[["date", "PERMNO", "cap_rank"]], on=["date", "PERMNO"], how="left")
    test["cap_rank"] = test["cap_rank"].astype("float64")

    test_idx = test.set_index(["date", "PERMNO"]).sort_index()
    returns = test_idx["ret"]
    dates = returns.index.get_level_values("date").unique().sort_values()
    rebal = dates[::5]
    cost = CostModel(0.001, 0.01)

    def mean_ic(frame: pd.DataFrame, col: str) -> float:
        return frame.groupby("date").apply(
            lambda g: spearmanr(g[col], g["target"])[0], include_groups=False).mean()

    models = [("Neural net (MLP)", "pred_mlp"), ("XGBoost", "pred_xgb"),
              ("Equal-weight (3 sig.)", "pred_ew")]
    rows = {}
    for uni, N in UNIVERSES.items():
        frame = test if N is None else test[test["cap_rank"] <= N]
        idx = test_idx if N is None else test_idx[test_idx["cap_rank"] <= N]
        for mname, col in models:
            w = held_weights(idx[col], rebal, quantile=0.10)
            bt = run_backtest(w, returns, cost)
            m = performance_metrics(bt["net_return"])
            m["gross_sharpe"] = performance_metrics(bt["gross_return"])["sharpe"]
            m["test_IC"] = mean_ic(frame, col)
            rows[f"{uni:9s} | {mname}"] = m

    gspc = pd.read_parquet(config.GSPC_PATH_CLEAN)
    gspc["date"] = pd.to_datetime(gspc["date"])
    spx = gspc.set_index("date")["sprtrn"].reindex(dates).fillna(0.0)
    spx_sharpe = performance_metrics(spx)["sharpe"]

    out = pd.DataFrame(rows).T[["test_IC", "gross_sharpe", "sharpe",
                                "ann_return", "ann_vol", "max_drawdown"]]
    print(f"\n=== Meta-model comparison (weekly L/S decile, test 2019-2024) ===")
    print(f"S&P500 reference Sharpe: {spx_sharpe:+.2f}\n")
    print(out.to_string(formatters={
        "test_IC": "{:+.4f}".format, "gross_sharpe": "{:+.2f}".format,
        "sharpe": "{:+.2f}".format, "ann_return": "{:+.2%}".format,
        "ann_vol": "{:.2%}".format, "max_drawdown": "{:.1%}".format}))


if __name__ == "__main__":
    main()
