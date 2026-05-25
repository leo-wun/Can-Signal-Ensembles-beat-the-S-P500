"""
Step 3b of the monthly study: universe-size sweep, top and bottom.

Re-runs the cross-sectional ML pipeline (Lasso, XGBoost, MLP, equal-weight)
across seven universe definitions: the full sample, the top-N stocks by lagged
log TTM revenue, and the bottom-N stocks. The symmetric sweep tests whether
the predictive edge is concentrated in larger or smaller firms -- the academic
literature suggests most factor anomalies are stronger in small / mid caps,
which the daily study could not separate from a micro-cap bid-ask-bounce
artifact (bounce vanishes at the monthly horizon since positions are held a
full month).

Each universe is filtered per month, features are re-rank-normalised within
the trading universe, and the four models are retrained from scratch.
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
UNIVERSES = [
    ("full",        None,     None),
    ("top 2000",    "top",    2000),
    ("top 1000",    "top",    1000),
    ("top 500",     "top",    500),
    ("bottom 2000", "bottom", 2000),
    ("bottom 1000", "bottom", 1000),
    ("bottom 500",  "bottom", 500),
]
LASSO_CV_SAMPLE = 200_000


def prepare(df_raw: pd.DataFrame, side, N) -> pd.DataFrame:
    """Filter the panel to the top-N (largest) or bottom-N (smallest) stocks
    per month by lagged log TTM revenue, then impute remaining NaN fundamentals
    cross-sectionally and rank-normalise all features per date within the
    resulting trading universe."""
    df = df_raw.copy()
    if N is not None:
        df = df.dropna(subset=["size_proxy"])
        ascending = (side == "bottom")              # rank 1 = smallest
        rank = df.groupby("date")["size_proxy"].rank(method="first", ascending=ascending)
        df = df[rank <= N]
    for c in FUNDAMENTAL_COLS:
        df[c] = df[c].fillna(df.groupby("date")[c].transform("median"))
        df[c] = df[c].fillna(0.0)
    for c in FEATURES:
        df[c] = df.groupby("date")[c].rank(pct=True) - 0.5
    df["yrank"] = df.groupby("date")["target"].rank(pct=True)
    return df


def run_one_universe(name: str, side, N, df_raw: pd.DataFrame):
    df = prepare(df_raw, side, N)
    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test = df[df["date"] > VAL_END].copy()
    print(f"\n--- universe '{name}' (side={side}, N={N}) | "
          f"train {len(train):,} | val {len(val):,} | test {len(test):,}")

    Xtr = train[FEATURES].to_numpy("float32"); ytr = train["yrank"].to_numpy("float32")
    Xva = val[FEATURES].to_numpy("float32");   yva = val["yrank"].to_numpy("float32")
    Xte = test[FEATURES].to_numpy("float32")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xtr), size=min(LASSO_CV_SAMPLE, len(Xtr)), replace=False)
    lcv = LassoCV(cv=5, n_jobs=-1, max_iter=20000, random_state=0).fit(Xtr[idx], ytr[idx])
    test["pred_lasso"] = lcv.predict(Xte)

    xgb = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                       early_stopping_rounds=40, eval_metric="rmse", random_state=0)
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    test["pred_xgb"] = xgb.predict(Xte)

    sc = StandardScaler().fit(Xtr)
    mlp, _ = train_mlp(sc.transform(Xtr).astype("float32"), ytr,
                       sc.transform(Xva).astype("float32"), yva,
                       n_features=len(FEATURES))
    mlp.eval()
    with torch.no_grad():
        dev = next(mlp.parameters()).device
        test["pred_mlp"] = mlp(torch.tensor(sc.transform(Xte).astype("float32"),
                                            device=dev)).cpu().numpy()

    corrs = train[FEATURES].corrwith(train["yrank"])
    signs = np.sign(corrs.values).astype("float32")
    test["pred_ew"] = (test[FEATURES].to_numpy("float32") * signs).mean(axis=1)

    test_idx = test.set_index(["date", "PERMNO"]).sort_index()
    returns = test_idx["target"]
    dates = returns.index.get_level_values("date").unique().sort_values()
    cost = CostModel(trading_cost=0.001, borrow_cost_annual=0.01, periods_per_year=12)

    def mean_ic(col):
        return test.groupby("date").apply(
            lambda g: spearmanr(g[col], g["target"])[0],
            include_groups=False).mean()

    rows, bts = {}, {}
    for mname, col in [("Lasso", "pred_lasso"), ("XGBoost", "pred_xgb"),
                       ("MLP", "pred_mlp"), ("EW", "pred_ew")]:
        w = held_weights(test_idx[col], dates, quantile=0.10)
        bt = run_backtest(w, returns, cost)
        m = performance_metrics(bt["net_return"], periods_per_year=12)
        m["test_IC"] = mean_ic(col)
        rows[mname] = m
        bts[mname] = bt
    return rows, bts


def main() -> None:
    df_raw = pd.read_parquet(config.MONTHLY_DATASET_PATH)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw[df_raw["date"] >= START]
    df_raw = df_raw.dropna(subset=TECH_COLS + ["target"])
    print(f"raw sample (>= {START.date()}): {len(df_raw):,} rows | "
          f"size_proxy present in {df_raw['size_proxy'].notna().mean():.1%} of rows")

    all_rows, all_bts = {}, {}
    for name, side, N in UNIVERSES:
        rows, bts = run_one_universe(name, side, N, df_raw)
        for mname, m in rows.items():
            all_rows[(name, mname)] = m
        all_bts[name] = bts

    sp_df = df_raw[df_raw["date"] > VAL_END].drop_duplicates("date").set_index("date")
    sp = sp_df["sprtrn"].sort_index().shift(-1).dropna()
    sp_m = performance_metrics(sp, periods_per_year=12)

    table = pd.DataFrame(all_rows).T
    table.index.names = ["universe", "model"]
    table = table[["test_IC", "sharpe", "ann_return", "ann_vol", "max_drawdown"]]
    print(f"\n=== Monthly L/S decile, net of 10 bps + 1% borrow "
          f"(test {sp.index.min().date()} -> {sp.index.max().date()}) ===")
    print(f"S&P500 reference Sharpe = {sp_m['sharpe']:+.2f}\n")
    print(table.to_string(formatters={
        "test_IC": "{:+.4f}".format, "sharpe": "{:+.2f}".format,
        "ann_return": "{:+.2%}".format, "ann_vol": "{:.2%}".format,
        "max_drawdown": "{:.1%}".format}))

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_set = [("full", "k"), ("top 500", "tab:red"), ("bottom 500", "tab:blue")]
    for name, color in plot_set:
        bt = all_bts[name]["XGBoost"]
        ax.plot(bt.index, bt["equity"], color=color, lw=1.5, label=f"XGBoost, {name}")
    sp_eq = (1.0 + sp).cumprod()
    ax.plot(sp_eq.index, sp_eq.values, "--", color="grey", lw=1.5, label="S&P500")
    ax.axhline(1.0, color="grey", lw=0.5)
    ax.set_title("XGBoost monthly L/S — full vs top 500 vs bottom 500 "
                 "(test, net of costs)")
    ax.set_ylabel("equity (start = 1)")
    ax.legend()
    out = config.PLOTS_DIR / "monthly_universe_sweep.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
