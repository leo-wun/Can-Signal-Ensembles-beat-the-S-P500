"""
Step 3c of the monthly study: cost robustness.

The bottom-500 XGBoost strategy delivered positive net Sharpe at 10 bps trading
cost. This script sweeps the round-trip trading cost over
{0, 10, 25, 50, 100, 200} bps on the most relevant universes (full,
bottom 1000, bottom 500) and the three ML models (Lasso, XGBoost, MLP) to
identify the breakeven cost at which the strategy ties or loses to the
S&P 500's Sharpe ratio.

Literature on small-cap trading costs: Frazzini, Israel & Moskowitz (2012)
report 40-50 bps for institutional small-cap orders; Hasbrouck (2009) shows
costs scaling inversely with size; Novy-Marx & Velikov (2016) place effective
costs for small-cap factor strategies at 50-100 bps. 10 bps was the project
baseline (large-cap institutional). The borrow cost is held fixed at 1%/yr.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
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


# Set Global Variable
FEATURES = TECH_COLS + FUNDAMENTAL_COLS
START = pd.Timestamp("1980-01-01")
TRAIN_END = pd.Timestamp("2010-12-31")
VAL_END = pd.Timestamp("2014-12-31")
UNIVERSES = [
    ("full",        None,     None),
    ("bottom 1000", "bottom", 1000),
    ("bottom 500",  "bottom", 500),
]
COSTS_BPS = [0, 10, 25, 50, 100, 200]
LASSO_CV_SAMPLE = 200_000



# Model training and prediction
def train_predict(
    df, 
    name
):
    
    # Build train, validation and test set
    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test = df[df["date"] > VAL_END].copy()
    
    # Display sizes
    print(f"  {name}: train {len(train):,} | val {len(val):,} | test {len(test):,}")

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
        
    return test


def main(
) -> None:
    
    # Get data
    df_raw = pd.read_parquet(config.MONTHLY_DATASET_PATH)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw[df_raw["date"] >= START]
    df_raw = df_raw.dropna(subset=TECH_COLS + ["target"])

    # Get benchmark data (SP500)
    sp_df = df_raw[df_raw["date"] > VAL_END].drop_duplicates("date").set_index("date")
    sp = sp_df["sprtrn"].sort_index().shift(-1).dropna()
    sp_sharpe = performance_metrics(sp, periods_per_year=12)["sharpe"]
    print(f"S&P500 reference Sharpe = {sp_sharpe:+.2f}\n")

    # Loop on Universes and model to get results 
    sharpe_records, annret_records = {}, {}
    
    # Training and prediction
    for name, side, N in UNIVERSES:
        print(f"\n--- universe '{name}' ---")
        df = prepare(df_raw, side, N, FUNDAMENTAL_COLS, FEATURES)
        test = train_predict(df, name)
        test_idx = test.set_index(["date", "PERMNO"]).sort_index()
        returns = test_idx["target"]
        dates = returns.index.get_level_values("date").unique().sort_values()

        # Backtest with cost model initialization
        for model_name, pred_col in [("Lasso", "pred_lasso"),
                                     ("XGBoost", "pred_xgb"),
                                     ("MLP", "pred_mlp")]:
            w = held_weights(test_idx[pred_col], dates, quantile=0.10)
            for bps in COSTS_BPS:
                cost = CostModel(trading_cost=bps / 10000.0,
                                 borrow_cost_annual=0.01, periods_per_year=12)
                bt = run_backtest(w, returns, cost)
                m = performance_metrics(bt["net_return"], periods_per_year=12)
                sharpe_records[(name, model_name, bps)] = m["sharpe"]
                annret_records[(name, model_name, bps)] = m["ann_return"]

    # Generate table of results
    idx = pd.MultiIndex.from_tuples(
        [(u, m) for u, _, _ in UNIVERSES for m in ["Lasso", "XGBoost", "MLP"]],
        names=["universe", "model"])
    sharpe_table = pd.DataFrame(
        {f"{bps} bps": [sharpe_records[(u, m, bps)] for u, m in idx]
         for bps in COSTS_BPS}, index=idx)
    annret_table = pd.DataFrame(
        {f"{bps} bps": [annret_records[(u, m, bps)] for u, m in idx]
         for bps in COSTS_BPS}, index=idx)

    # Display results
    print(f"\n NET SHARPE by trading cost (S&P500 ref = {sp_sharpe:+.2f}) ")
    print(sharpe_table.to_string(float_format=lambda x: f"{x:+.2f}"))
    print(f"\n ANN RETURN by trading cost ")
    print(annret_table.to_string(float_format=lambda x: f"{x:+.1%}"))

    # Breakeven cost vs S&P for XGBoost
    print(f"\n breakeven cost vs S&P (Sharpe={sp_sharpe:+.2f}), XGBoost ")
    bps_arr = np.array(COSTS_BPS, dtype=float)
    for u, _, _ in UNIVERSES:
        sharpes = np.array([sharpe_records[(u, "XGBoost", bps)] for bps in COSTS_BPS])
        cross = np.where(sharpes < sp_sharpe)[0]
        if len(cross) == 0:
            print(f"  {u}: still above S&P at 200 bps (Sharpe {sharpes[-1]:+.2f})")
        elif cross[0] == 0:
            print(f"  {u}: never beats S&P (Sharpe at 0 bps = {sharpes[0]:+.2f})")
        else:
            i = cross[0]
            x = bps_arr[i - 1] + (sp_sharpe - sharpes[i - 1]) \
                / (sharpes[i] - sharpes[i - 1]) * (bps_arr[i] - bps_arr[i - 1])
            print(f"  {u}: breakeven ~{x:.0f} bps")


    # Create and save graph
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"full": "k", "bottom 1000": "tab:purple", "bottom 500": "tab:blue"}
    for u, _, _ in UNIVERSES:
        sharpes = [sharpe_records[(u, "XGBoost", bps)] for bps in COSTS_BPS]
        ax.plot(COSTS_BPS, sharpes, "o-", color=colors[u], lw=1.6, label=f"XGBoost, {u}")
    ax.axhline(sp_sharpe, color="grey", lw=1.2, ls="--",
               label=f"S&P500 = {sp_sharpe:+.2f}")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("trading cost (bps per round-trip turnover)")
    ax.set_ylabel("net Sharpe (annualised)")
    ax.set_title("Cost robustness — XGBoost monthly L/S Sharpe vs trading cost")
    ax.legend()
    ax.grid(alpha=0.3)
    out = config.PLOTS_DIR / "monthly_cost_robustness.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
    
    
"""

Commentary on "monthly_cost_robustness.png"

Breakeven point for the XGboost on the bottom 500 is reached at around 50bps which is in the range 
50-100 bps estimated from litterature for Small-Cap. 
It is important to note that the code does not include other source of cost like 
(1) Bid-ask spread
(2) Price impact

Which are notably higher on small-cap.
This mean that for this strategy to beat the SP500, all costs should hit below 50bps. 

"""

