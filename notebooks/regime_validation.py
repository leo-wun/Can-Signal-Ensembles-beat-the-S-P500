"""
Validation of the regime layer (VIX + Chen-Zimmerman factors).

These signals are time series (one value per date), not cross-sectional
stock-pickers. The question is whether they have *market-timing* power:
do they predict the future market return, or at least the future market
volatility? We run predictive regressions (OLS with Newey-West HAC t-stats,
since the regressors are persistent and the multi-day targets overlap) and
report in-sample vs out-of-sample R2 — the OOS R2 is the honest verdict.

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

START = "2000-01-01" # project window
SPLIT = pd.Timestamp("2015-12-31") # train <= SPLIT


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int):
    """OLS with HAC covariance. X must already include an intercept."""
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


def oos_r2(d: pd.DataFrame, target: str, predictors: list[str]) -> float:
    """Out-of-sample R2 vs the prevailing (training) mean benchmark."""
    tr = d.index <= SPLIT
    Xtr = np.column_stack([np.ones(tr.sum())] + [d.loc[tr, p].values for p in predictors])
    Xte = np.column_stack([np.ones((~tr).sum())] + [d.loc[~tr, p].values for p in predictors])
    ytr, yte = d.loc[tr, target].values, d.loc[~tr, target].values
    beta = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    pred = Xte @ beta
    return 1.0 - ((yte - pred) ** 2).sum() / ((yte - ytr.mean()) ** 2).sum()


def report(df: pd.DataFrame, target: str, predictors: list[str], lags: int, label: str):
    """"Report of HAC regression results and OOS R2."""
    d = df[[target] + predictors].dropna().astype("float64")
    y = d[target].values
    X = np.column_stack([np.ones(len(d))] + [d[p].values for p in predictors])
    beta, se, t, r2 = ols_hac(y, X, lags)
    print(f"\n  [{label}]  n={len(d)}  HAC lags={lags}")
    names = ["const"] + predictors
    for nm, b, tt in zip(names, beta, t):
        print(f"    {nm:<14s} beta={b:+.5f}   HAC_t={tt:+.2f}")
    print(f"in-sample R2 = {r2:+.5f},    OOS R2 = {oos_r2(d, target, predictors):+.5f}")


def main() -> None:
    gspc = pd.read_parquet(config.GSPC_PATH_CLEAN)
    gspc["date"] = pd.to_datetime(gspc["date"])
    gspc = gspc.set_index("date").sort_index()

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

    df_vix = tgt.join(vfeat, how="inner").loc[START:].dropna(subset=["vix"])
    df_cz = tgt.join(cz, how="inner").loc[START:]

    print("=== Regime layer validation ===")
    print(f"VIX sample : {df_vix.index.min().date()} -> {df_vix.index.max().date()} "
          f"({len(df_vix)} trading days)")
    print(f"CZ  sample : {df_cz.index.min().date()} -> {df_cz.index.max().date()}")

    # Part 1: VIX -> market RETURN
    print("\n--- Part 1: VIX -> market RETURN (OLS, Newey-West HAC) ---")
    report(df_vix, "fwd_1d", ["vix", "vix_gap63", "vix_gap252", "dvix"], 10,
           "next-day return ~ VIX")
    report(df_vix, "fwd_21d", ["vix", "vix_gap63", "vix_gap252", "dvix"], 42,
           "next-21d return ~ VIX")

    # Part 2: VIX -> market VOLATILITY (robust use case)
    print("\n--- Part 2: VIX -> market VOLATILITY (sanity check / robust use) ---")
    report(df_vix, "rvol_21d", ["vix"], 42, "next-21d realised vol ~ VIX level")

    # Part 3: CZ factors -> market return (LassoCV, monthly obs)
    print("\n--- Part 3: CZ factors -> next-21d market RETURN (LassoCV, monthly) ---")
    monthly = df_cz.groupby(df_cz.index.to_period("M")).tail(1)
    d = monthly[["fwd_21d"] + cz_cols].dropna().astype("float64")
    tr = d.index <= SPLIT
    sc = StandardScaler().fit(d.loc[tr, cz_cols])
    Xtr, Xte = sc.transform(d.loc[tr, cz_cols]), sc.transform(d.loc[~tr, cz_cols])
    ytr, yte = d.loc[tr, "fwd_21d"].values, d.loc[~tr, "fwd_21d"].values
    m = LassoCV(cv=5, n_jobs=1, max_iter=50000, random_state=0).fit(Xtr, ytr)
    r2_is = m.score(Xtr, ytr)
    pred = m.predict(Xte)
    r2_oos = 1.0 - ((yte - pred) ** 2).sum() / ((yte - ytr.mean()) ** 2).sum()
    nz = pd.Series(m.coef_, index=cz_cols)
    nz = nz[nz != 0].sort_values(key=abs, ascending=False)
    print(f"  monthly obs: {tr.sum()} train / {(~tr).sum()} test   alpha={m.alpha_:.2e}")
    print(f"  in-sample R2 = {r2_is:+.4f}    OOS R2 = {r2_oos:+.4f}")
    print(f"  non-zero factors = {len(nz)}/{len(cz_cols)}"
          + (f"  | top: {', '.join(nz.index[:6])}" if len(nz) else ""))

    # Part 4: VIX regime table
    print("\n--- Part 4: VIX regime table (terciles of VIX level) ---")
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
