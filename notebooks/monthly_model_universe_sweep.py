"""
Step 3b of the monthly study: universe-size sweep, top and bottom.

Re-runs the cross-sectional ML pipeline (Lasso, XGBoost, MLP, equal-weight)
across seven universe definitions: the full sample, the top-N stocks by lagged
log TTM revenue, and the bottom-N stocks. The symmetric sweep tests whether
the predictive edge is concentrated in larger or smaller firms ; the academic
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
from src.utils import prepare


# Set Global Variable
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



def run_one_universe(
    name: str,
    side,
    N, 
    df_raw: pd.DataFrame
):
    
    # Prepare data and split train, validation and test dataset
    df = prepare(df_raw, side, N, FUNDAMENTAL_COLS, FEATURES)
    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test = df[df["date"] > VAL_END].copy()
    print(f"\n universe '{name}' (side={side}, N={N}) | "
          f"train {len(train):,} | val {len(val):,} | test {len(test):,}")

    # Convert to numpy with datatype
    Xtr = train[FEATURES].to_numpy("float32"); ytr = train["yrank"].to_numpy("float32")
    Xva = val[FEATURES].to_numpy("float32");   yva = val["yrank"].to_numpy("float32")
    Xte = test[FEATURES].to_numpy("float32")

    # Set seed
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xtr), size=min(LASSO_CV_SAMPLE, len(Xtr)), replace=False)
    
    # Initialize, train and predict LassoCV
    lcv = LassoCV(cv=5, n_jobs=1, max_iter=20000, random_state=0).fit(Xtr[idx], ytr[idx])
    test["pred_lasso"] = lcv.predict(Xte)

    # Initialize, train and predict XGboost regressor
    # n_jobs=1 for the same reason as above; XGBoost manages its own threads.
    xgb = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=1,
                       early_stopping_rounds=40, eval_metric="rmse", random_state=0)
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    test["pred_xgb"] = xgb.predict(Xte)

    # Initialize, train and predict MLP with scaling
    sc = StandardScaler().fit(Xtr)
    mlp, _ = train_mlp(sc.transform(Xtr).astype("float32"), ytr,
                       sc.transform(Xva).astype("float32"), yva,
                       n_features=len(FEATURES))
    mlp.eval()
    with torch.no_grad():
        dev = next(mlp.parameters()).device
        test["pred_mlp"] = mlp(torch.tensor(sc.transform(Xte).astype("float32"),
                                            device=dev)).cpu().numpy()

    # Generate the signal equal-weighted strategy
    corrs = train[FEATURES].corrwith(train["yrank"])
    signs = np.sign(corrs.values).astype("float32")
    test["pred_ew"] = (test[FEATURES].to_numpy("float32") * signs).mean(axis=1)

    # Get returns on test dataset
    test_idx = test.set_index(["date", "PERMNO"]).sort_index()
    returns = test_idx["target"]
    dates = returns.index.get_level_values("date").unique().sort_values()
    
    # Initialize the cost model 
    cost = CostModel(trading_cost=0.001, borrow_cost_annual=0.01, periods_per_year=12)

    # Evaluation
    def mean_ic(
        col
    ):
        
        return test.groupby("date").apply(
            lambda g: spearmanr(g[col], g["target"])[0],
            include_groups=False).mean()

    rows, bts = {}, {}
    
    # Backtest loop on models
    for mname, col in [("Lasso", "pred_lasso"), ("XGBoost", "pred_xgb"),
                       ("MLP", "pred_mlp"), ("EW", "pred_ew")]:
        w = held_weights(test_idx[col], dates, quantile=0.10)
        bt = run_backtest(w, returns, cost)
        m = performance_metrics(bt["net_return"], periods_per_year=12)
        m["test_IC"] = mean_ic(col)
        rows[mname] = m
        bts[mname] = bt
        
        
    return rows, bts


def main(
) -> None:
    
    
    # Get data
    df_raw = pd.read_parquet(config.MONTHLY_DATASET_PATH)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw[df_raw["date"] >= START]
    df_raw = df_raw.dropna(subset=TECH_COLS + ["target"])
    print(f"raw sample (>= {START.date()}): {len(df_raw):,} rows | "
          f"size_proxy present in {df_raw['size_proxy'].notna().mean():.1%} of rows")

    # Run code on UNIVERSES Global Variable
    all_rows, all_bts = {}, {}
    for name, side, N in UNIVERSES:
        rows, bts = run_one_universe(name, side, N, df_raw)
        for mname, m in rows.items():
            all_rows[(name, mname)] = m
        all_bts[name] = bts

    # Get benchmark data (SP500)
    sp_df = df_raw[df_raw["date"] > VAL_END].drop_duplicates("date").set_index("date")
    sp = sp_df["sprtrn"].sort_index().shift(-1).dropna()
    sp_m = performance_metrics(sp, periods_per_year=12)

    # Generate the results table
    table = pd.DataFrame(all_rows).T
    table.index.names = ["universe", "model"]
    table = table[["test_IC", "sharpe", "ann_return", "ann_vol", "max_drawdown"]]
    
    # Display results
    print(f"\n Monthly L/S decile, net of 10 bps + 1% borrow "
          f"(test {sp.index.min().date()} -> {sp.index.max().date()}) ")
    print(f"S&P500 reference Sharpe = {sp_m['sharpe']:+.2f}\n")
    print(table.to_string(formatters={
        "test_IC": "{:+.4f}".format, "sharpe": "{:+.2f}".format,
        "ann_return": "{:+.2%}".format, "ann_vol": "{:.2%}".format,
        "max_drawdown": "{:.1%}".format}))

    # Create and save graph
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
    
    
"""

Commentary on "monthly_universe_sweep.png"

XGboost on the bottom 500 of the universe display very good results. 
Clearly above other filtration on the universe, the performance is way above the SP500. 
However, a note on the fact that the trading costs used for those results are 10bps which correspond to
cost usually found for large cap. However, small cap are notorious for having larger trading costs as 
they are less traded. Thus what we will do next is study the breakeven point for costs of a bottom 500 strategy
to see if the performance remains.  



raw sample (>= 1980-01-01): 3,223,602 rows | size_proxy present in 46.1% of rows

 universe 'full' (side=None, N=None) | train 2,150,706 | val 281,008 | test 791,888

 universe 'top 2000' (side=top, N=2000) | train 392,135 | val 96,000 | test 238,000

 universe 'top 1000' (side=top, N=1000) | train 215,135 | val 48,000 | test 119,000

 universe 'top 500' (side=top, N=500) | train 125,413 | val 24,000 | test 59,500

 universe 'bottom 2000' (side=bottom, N=2000) | train 392,135 | val 96,000 | test 238,000

 universe 'bottom 1000' (side=bottom, N=1000) | train 215,135 | val 48,000 | test 119,000

 universe 'bottom 500' (side=bottom, N=500) | train 125,413 | val 24,000 | test 59,500

 Monthly L/S decile, net of 10 bps + 1% borrow (test 2015-01-30 -> 2024-10-31) 
S&P500 reference Sharpe = +0.77

                    test_IC sharpe ann_return ann_vol max_drawdown
universe    model                                                 
full        Lasso   +0.0849  +0.20     +5.18%  25.31%       -71.5%
            XGBoost +0.0948  +0.41    +10.40%  25.08%       -65.6%
            MLP     +0.0892  +0.38     +9.02%  23.95%       -67.0%
            EW      +0.0792  +0.17     +3.84%  22.80%       -68.5%
top 2000    Lasso   +0.0341  -0.11     -3.03%  28.48%       -69.0%
            XGBoost +0.0328  -0.05     -1.30%  28.52%       -63.4%
            MLP     +0.0002  -0.09     -1.03%  11.29%       -24.4%
            EW      +0.0303  -0.24     -6.10%  25.85%       -73.1%
top 1000    Lasso   +0.0224  -0.17     -4.28%  25.00%       -61.0%
            XGBoost +0.0184  -0.18     -4.24%  23.02%       -60.0%
            MLP     -0.0065  -0.40     -3.80%   9.38%       -34.4%
            EW      +0.0257  -0.26     -5.98%  22.91%       -66.4%
top 500     Lasso   +0.0180  -0.19     -4.13%  22.18%       -53.0%
            XGBoost +0.0249  -0.16     -3.15%  19.97%       -50.5%
            MLP     -0.0100  -0.54     -4.68%   8.66%       -39.2%
            EW      +0.0201  -0.34     -7.07%  20.88%       -65.2%
bottom 2000 Lasso   +0.1171  +0.50    +14.06%  28.28%       -76.1%
            XGBoost +0.1193  +0.45    +13.52%  29.96%       -78.8%
            MLP     +0.0536  +0.43     +6.63%  15.49%       -48.2%
            EW      +0.1073  +0.30     +8.70%  29.31%       -74.8%
bottom 1000 Lasso   +0.1354  +0.66    +21.58%  32.46%       -81.6%
            XGBoost +0.1372  +0.91    +29.87%  32.68%       -76.4%
            MLP     +0.0596  +0.24     +5.21%  21.55%       -56.4%
            EW      +0.1186  +0.33    +10.17%  30.71%       -79.0%
bottom 500  Lasso   +0.1347  +0.74    +27.64%  37.32%       -80.3%
            XGBoost +0.1299  +1.18    +40.86%  34.77%       -76.8%
            MLP     +0.0614  +0.11     +2.77%  26.20%       -65.1%
            EW      +0.0966  +0.09     +3.32%  36.33%       -86.7% 

"""

