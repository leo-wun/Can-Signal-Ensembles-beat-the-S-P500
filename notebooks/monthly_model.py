"""
Step 3 of the monthly study: cross-sectional return prediction.

Trains a linear baseline (Lasso), gradient-boosted trees (XGBoost) and a
feed-forward neural network (MLP) to predict the cross-sectional rank of the
next-month return from 5 technical + 11 fundamental characteristics. Each
model is evaluated by its test-set predictive IC and by a monthly-rebalanced
decile long-short portfolio, net of 10 bps trading cost and 1%/yr borrow,
benchmarked against the S&P 500.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.backtest import CostModel, held_weights, run_backtest, performance_metrics
from src.models.mlp_model import train_mlp
from src.monthly_features import FEATURE_COLS as TECH_COLS
from src.compustat_features import FUNDAMENTAL_COLS

FEATURES = TECH_COLS + FUNDAMENTAL_COLS
START = pd.Timestamp("1980-01-01")
TRAIN_END = pd.Timestamp("2010-12-31")
VAL_END = pd.Timestamp("2014-12-31")
LASSO_CV_SAMPLE = 200_000


def main() -> None:
    df = pd.read_parquet(config.MONTHLY_DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= START]
    df = df.dropna(subset=TECH_COLS + ["target"])

    # cross-sectional median imputation per date for the fundamentals
    for c in FUNDAMENTAL_COLS:
        df[c] = df[c].fillna(df.groupby("date")[c].transform("median"))
        df[c] = df[c].fillna(0.0)

    # rank-normalise features cross-sectionally per date -> [-0.5, +0.5]
    for c in FEATURES:
        df[c] = df.groupby("date")[c].rank(pct=True) - 0.5
    # target -> cross-sectional percentile rank per date (robust ML target)
    df["yrank"] = df.groupby("date")["target"].rank(pct=True)

    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test = df[df["date"] > VAL_END].copy()
    print(f"modeling sample: {len(df):,} rows | "
          f"train {len(train):,} | val {len(val):,} | test {len(test):,}")
    print(f"test window: {test['date'].min().date()} -> {test['date'].max().date()}")

    Xtr = train[FEATURES].to_numpy("float32")
    Xva = val[FEATURES].to_numpy("float32")
    Xte = test[FEATURES].to_numpy("float32")
    ytr = train["yrank"].to_numpy("float32")
    yva = val["yrank"].to_numpy("float32")

    # Lasso
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xtr), size=min(LASSO_CV_SAMPLE, len(Xtr)), replace=False)
    lcv = LassoCV(cv=5, n_jobs=-1, max_iter=20000, random_state=0).fit(Xtr[idx], ytr[idx])
    test["pred_lasso"] = lcv.predict(Xte)
    print(f"\nLasso: alpha={lcv.alpha_:.2e} | "
          f"non-zero coefs={(lcv.coef_ != 0).sum()}/{len(FEATURES)}")
    coef = pd.Series(lcv.coef_, index=FEATURES).sort_values(key=abs, ascending=False)
    print("Lasso top coefficients:")
    print(coef.head(8).to_string(float_format=lambda x: f"{x:+.4f}"))

    # XGBoost
    xgb = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                       early_stopping_rounds=40, eval_metric="rmse", random_state=0)
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    test["pred_xgb"] = xgb.predict(Xte)
    print(f"\nXGBoost: best iteration={xgb.best_iteration}")
    fi = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("XGBoost top feature importances:")
    print(fi.head(8).to_string(float_format=lambda x: f"{x:.3f}"))

    # MLP (deep-learning model)
    sc = StandardScaler().fit(Xtr)
    mlp, val_mse = train_mlp(sc.transform(Xtr).astype("float32"), ytr,
                             sc.transform(Xva).astype("float32"), yva,
                             n_features=len(FEATURES))
    mlp.eval()
    with torch.no_grad():
        dev = next(mlp.parameters()).device
        test["pred_mlp"] = mlp(torch.tensor(sc.transform(Xte).astype("float32"),
                                            device=dev)).cpu().numpy()
    print(f"\nMLP: best validation MSE={val_mse:.5f}")

    # Equal-weight baseline (train-data sign-oriented)
    corrs = train[FEATURES].corrwith(train["yrank"])
    signs = np.sign(corrs.values).astype("float32")
    test["pred_ew"] = (test[FEATURES].to_numpy("float32") * signs).mean(axis=1)
    print("\nFeature/target correlations on the training set (used to orient EW):")
    print(corrs.sort_values(ascending=False).to_string(float_format=lambda x: f"{x:+.3f}"))

    # Evaluation
    def mean_ic(col: str) -> float:
        return test.groupby("date").apply(
            lambda g: spearmanr(g[col], g["target"])[0],
            include_groups=False).mean()

    test_idx = test.set_index(["date", "PERMNO"]).sort_index()
    returns = test_idx["target"]              # next-month realised return
    dates = returns.index.get_level_values("date").unique().sort_values()
    cost = CostModel(trading_cost=0.001, borrow_cost_annual=0.01, periods_per_year=12)

    # S&P 500 benchmark aligned to the strategy's holding period
    sp = (test.drop_duplicates("date").set_index("date")["sprtrn"]
              .sort_index().shift(-1).reindex(dates).fillna(0.0))
    sp_m = performance_metrics(sp, periods_per_year=12)

    rows, bts = {}, {}
    for name, col in [("Lasso", "pred_lasso"), ("XGBoost", "pred_xgb"),
                      ("MLP", "pred_mlp"), ("Equal-weight", "pred_ew")]:
        w = held_weights(test_idx[col], dates, quantile=0.10)
        bt = run_backtest(w, returns, cost)
        m = performance_metrics(bt["net_return"], periods_per_year=12)
        m["sharpe_gross"] = performance_metrics(bt["gross_return"], periods_per_year=12)["sharpe"]
        m["test_IC"] = mean_ic(col)
        rows[name] = m
        bts[name] = bt
    rows["S&P500"] = sp_m

    out = pd.DataFrame(rows).T[["test_IC", "sharpe_gross", "sharpe",
                                "ann_return", "ann_vol", "max_drawdown"]]
    print(f"\n=== Monthly L/S decile (test {test['date'].min().date()} -> "
          f"{test['date'].max().date()}, net 10 bps + 1% borrow) ===")
    print(f"S&P500 Sharpe = {sp_m['sharpe']:+.2f}\n")
    print(out.to_string(formatters={
        "test_IC": "{:+.4f}".format, "sharpe_gross": "{:+.2f}".format,
        "sharpe": "{:+.2f}".format, "ann_return": "{:+.2%}".format,
        "ann_vol": "{:.2%}".format, "max_drawdown": "{:.1%}".format}))

    pred_df = test[["date", "PERMNO", "target", "sprtrn",
                    "pred_lasso", "pred_xgb", "pred_mlp", "pred_ew"]]
    pred_df.to_parquet(config.MONTHLY_PREDICTIONS_PATH, index=False)
    print(f"\nsaved predictions -> {config.MONTHLY_PREDICTIONS_PATH}")

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, bt in bts.items():
        ax.plot(bt.index, bt["equity"], lw=1.3, label=name)
    sp_eq = (1.0 + sp).cumprod()
    ax.plot(sp_eq.index, sp_eq.values, "k--", lw=1.3, label="S&P500")
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.set_title("Monthly cross-sectional ML — equity curves (test, net of costs)")
    ax.set_ylabel("equity (start = 1)")
    ax.legend()
    out_plot = config.PLOTS_DIR / "monthly_model_equity.png"
    fig.savefig(out_plot, dpi=120, bbox_inches="tight")
    print(f"saved equity curves -> {out_plot}")


if __name__ == "__main__":
    main()
