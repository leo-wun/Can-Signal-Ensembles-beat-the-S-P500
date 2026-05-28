"""

Validation of the regime layer (VIX).

These signals are time series (one value per date), not cross-sectional
stock-pickers. The question is whether they have *market-timing* power:
do they predict the future market return, or at least the future market
volatility? We run predictive regressions (OLS with Newey-West HAC t-stats,
since the regressors are persistent and the multi-day targets overlap) and
report in-sample vs out-of-sample R2 ; the OOS R2 is the honest verdict.

Predictors are already lagged by 1 day in the cleaning step, so pairing them
with same-date returns is a genuine predictive regression (no look-ahead).

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# Set Global Variable
START = "2000-01-01" # project window
SPLIT = pd.Timestamp("2015-12-31") # train <= SPLIT


def ols_hac(
    y: np.ndarray,
    X: np.ndarray, 
    lags: int
):
    
    """
    
    OLS with HAC covariance. X must already include an intercept.
    
    """
    
    
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    u = X * resid[:, None]
    S = u.T @ u
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = u[l:].T @ u[:-l]
        S = S + w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    r2 = 1.0 - (resid @ resid) / ((y - y.mean()) ** 2).sum()
    
    return beta, se, beta / se, r2


def oos_r2(
    d: pd.DataFrame, 
    target: str, 
    predictors: list[str]
) -> float:
    
    """
    
    Out-of-sample R2 vs the prevailing (training) mean benchmark.
    
    """
    
    tr = d.index <= SPLIT
    Xtr = np.column_stack([np.ones(tr.sum())] + [d.loc[tr, p].values for p in predictors])
    Xte = np.column_stack([np.ones((~tr).sum())] + [d.loc[~tr, p].values for p in predictors])
    ytr, yte = d.loc[tr, target].values, d.loc[~tr, target].values
    beta = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    pred = Xte @ beta
    
    return 1.0 - ((yte - pred) ** 2).sum() / ((yte - ytr.mean()) ** 2).sum()


def report(
    df: pd.DataFrame, 
    target: str, 
    predictors: list[str], 
    lags: int, label: str
):
    
    """
    
    Report of HAC regression results and OOS R2.
    
    """
    
    d = df[[target] + predictors].dropna().astype("float64")
    y = d[target].values
    X = np.column_stack([np.ones(len(d))] + [d[p].values for p in predictors])
    beta, se, t, r2 = ols_hac(y, X, lags)
    
    print(f"\n  [{label}]  n={len(d)}  HAC lags={lags}")
    names = ["const"] + predictors
    
    for nm, b, tt in zip(names, beta, t):
        print(f"    {nm:<14s} beta={b:+.5f}   HAC_t={tt:+.2f}")
    print(f"in-sample R2 = {r2:+.5f},    OOS R2 = {oos_r2(d, target, predictors):+.5f}")


def main(
) -> None:
    
    # Load market data
    gspc = pd.read_parquet(config.GSPC_PATH_CLEAN)
    gspc["date"] = pd.to_datetime(gspc["date"])
    gspc = gspc.set_index("date").sort_index()

    # Load vix and Chen-Zimmerman data
    vix = pd.read_parquet(config.VIX_PATH_CLEAN)
    cz = pd.read_parquet(config.CZ_PATH_CLEAN)
    cz_cols = cz.columns.tolist()

    # Forward market targets (log returns)
    logret = np.log1p(gspc["sprtrn"])
    tgt = pd.DataFrame(index=gspc.index)
    tgt["fwd_1d"] = logret                                            # next day
    tgt["fwd_21d"] = logret.rolling(21).sum().shift(-20)              # next ~month
    tgt["rvol_21d"] = logret.rolling(21).std().shift(-20) * np.sqrt(252)

    # VIX predictors
    vfeat = pd.DataFrame(index=vix.index)
    vfeat["vix"] = vix["vix"]
    vfeat["vix_gap63"] = vix["vix"] / vix["vix_ma_63"] - 1
    vfeat["vix_gap252"] = vix["vix"] / vix["vix_ma_252"] - 1
    vfeat["dvix"] = vix["vix"].diff()

    # Merge 
    df_vix = tgt.join(vfeat, how="inner").loc[START:].dropna(subset=["vix"])
    df_cz = tgt.join(cz, how="inner").loc[START:]

    # Display data information
    print(" Regime layer validation ")
    print(f"VIX sample : {df_vix.index.min().date()} -> {df_vix.index.max().date()} "
          f"({len(df_vix)} trading days)")
    print(f"CZ  sample : {df_cz.index.min().date()} -> {df_cz.index.max().date()}")

    # Part 1: VIX -> market RETURN
    print("\n Part 1: VIX -> market RETURN (OLS, Newey-West HAC) ")
    report(df_vix, "fwd_1d", ["vix", "vix_gap63", "vix_gap252", "dvix"], 10,
           "next-day return ~ VIX")
    report(df_vix, "fwd_21d", ["vix", "vix_gap63", "vix_gap252", "dvix"], 42,
           "next-21d return ~ VIX")

    # Part 2: VIX -> market VOLATILITY (robust use case)
    print("\n Part 2: VIX -> market VOLATILITY (sanity check / robust use) ")
    report(df_vix, "rvol_21d", ["vix"], 42, "next-21d realised vol ~ VIX level")

    # Part 3: VIX regime table
    print("\n Part 3: VIX regime table (terciles of VIX level) ")
    reg = df_vix.dropna(subset=["fwd_21d", "rvol_21d"]).copy()
    reg["regime"] = pd.qcut(reg["vix"], 3, labels=["Low VIX", "Mid VIX", "High VIX"])
    g = reg.groupby("regime", observed=True)
    tbl = pd.DataFrame({
        "n_days": g.size(),
        "pct": g.size() / len(reg),
        "ann_return": g["fwd_21d"].mean() * (252 / 21),
        "avg_fwd_vol": g["rvol_21d"].mean(),
    })
    tbl["return/vol"] = tbl["ann_return"] / tbl["avg_fwd_vol"]
    print(tbl.to_string(formatters={
        "pct": "{:.1%}".format, "ann_return": "{:+.2%}".format,
        "avg_fwd_vol": "{:.2%}".format, "return/vol": "{:+.2f}".format,
    }))


if __name__ == "__main__":
    main()



"""

Commentary 

Part 2 shows that the VIX is a very strong predictor of future realized volatility (21 days).
Part 1 shows that the VIX seems to have no prediction power
on the market returns (at least not linear).
Part 4 shows an interesting results : best sharpe ratio of the market is reached on Low Vix regime. 

Without going to deep in the analysis, those results shows us that using the VIX as an estimator for 
the volatility/indicator of volatility is a strong tool that we could add to our analysis.


VIX sample : 2000-01-03 -> 2024-12-31 (6289 trading days)
CZ  sample : 2000-01-03 -> 2020-11-30

 Part 1: VIX -> market RETURN (OLS, Newey-West HAC) 

  [next-day return ~ VIX]  n=6289  HAC lags=10
    const          beta=-0.00015   HAC_t=-0.24
    vix            beta=+0.00187   HAC_t=+0.56
    vix_gap63      beta=-0.00126   HAC_t=-0.68
    vix_gap252     beta=+0.00074   HAC_t=+0.75
    dvix           beta=+0.06513   HAC_t=+2.07
in-sample R2 = +0.00927,    OOS R2 = +0.01254

  [next-21d return ~ VIX]  n=6269  HAC lags=42
    const          beta=+0.00229   HAC_t=+0.25
    vix            beta=+0.01255   HAC_t=+0.25
    vix_gap63      beta=-0.00219   HAC_t=-0.10
    vix_gap252     beta=+0.00866   HAC_t=+0.55
    dvix           beta=+0.00946   HAC_t=+0.17
in-sample R2 = +0.00467,    OOS R2 = -0.00690

 Part 2: VIX -> market VOLATILITY (sanity check / robust use) 

  [next-21d realised vol ~ VIX level]  n=6269  HAC lags=42
    const          beta=-0.01808   HAC_t=-1.54
    vix            beta=+0.91508   HAC_t=+13.06
in-sample R2 = +0.53287,    OOS R2 = +0.41417

 Part 3: VIX regime table (terciles of VIX level) 
          n_days   pct ann_return avg_fwd_vol return/vol
regime                                                  
Low VIX     2090 33.3%     +5.89%      10.48%      +0.56
Mid VIX     2089 33.3%     +3.44%      14.29%      +0.24
High VIX    2090 33.3%     +8.01%      24.40%      +0.33

"""