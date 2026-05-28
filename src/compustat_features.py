"""
Fundamental characteristics from the quarterly Compustat dataset.

The provided Compustat file contains income-statement / cash-flow flow items
only (no balance sheet, no price). All flow columns carry the `y` suffix =
fiscal year-to-date. They are de-cumulated to single quarters and re-aggregated
into trailing-twelve-month (TTM) figures, from which margin, growth, accrual
and intensity characteristics are computed.
"""

import numpy as np
import pandas as pd

# (Compustat YTD column -> short name) for the flow items we use
_FLOW = {
    "revty": "rev", "cogsy": "cogs", "niy": "ni", "oiadpy": "oiadp",
    "oibdpy": "oibdp", "oancfy": "ocf", "capxy": "capx", "xrdy": "xrd",
    "xsgay": "xsga", "dvy": "dv",
}

FUNDAMENTAL_COLS = [
    "op_margin", "gross_margin", "ebitda_margin", "net_margin", "accruals",
    "sales_growth", "earnings_chg", "rd_intensity", "capex_intensity",
    "sga_intensity", "div_intensity",
]

_LOAD_COLS = (["gvkey", "datadate", "fqtr", "indfmt", "consol", "popsrc",
               "datafmt", "cusip"] + list(_FLOW))


def load_compustat(path) -> pd.DataFrame:
    """Load the standard (industrial / consolidated / domestic / standardised)
    Compustat view, one row per (gvkey, fiscal quarter)."""
    df = pd.read_csv(path, usecols=_LOAD_COLS, dtype={"cusip": str},
                     low_memory=False)
    df = df[(df.indfmt == "INDL") & (df.datafmt == "STD")
            & (df.consol == "C") & (df.popsrc == "D")]
    df["datadate"] = pd.to_datetime(df["datadate"])
    for c in _FLOW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["gvkey", "datadate", "cusip"])
    df = (df.sort_values(["gvkey", "datadate"])
            .drop_duplicates(["gvkey", "datadate"], keep="last")
            .reset_index(drop=True))
    return df


def build_fundamental_features(cs: pd.DataFrame) -> pd.DataFrame:
    """De-cumulate YTD flows to single quarters, build TTM aggregates, and
    derive fundamental characteristics. Returns one row per (gvkey, datadate)
    with cusip8 and the columns in FUNDAMENTAL_COLS."""
    cs = cs.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    cs.loc[cs["xrdy"].isna(), "xrdy"] = 0.0 # missing R&D treated as 0
    gv = cs["gvkey"]
    g = cs.groupby("gvkey", sort=False)
    is_q1 = cs["fqtr"].eq(1)

    def ttm_of(col: str) -> pd.Series:
        q = cs[col] - g[col].shift(1) # YTD -> single quarter
        q = q.where(~is_q1, cs[col]) # Q1 YTD already quarterly
        return q.groupby(gv, sort=False).transform(
            lambda s: s.rolling(4, min_periods=4).sum())

    ttm = {short: ttm_of(ytd) for ytd, short in _FLOW.items()}
    rev = ttm["rev"].where(ttm["rev"] > 0) # avoid / 0 or log(0)

    f = pd.DataFrame({
        "gvkey": gv,
        "datadate": cs["datadate"],
        "cusip8": cs["cusip"].astype(str).str[:8],
    })
    f["size_proxy"]      = np.log(ttm["rev"].where(ttm["rev"] > 0)) # log TTM revenue
    f["op_margin"]       = ttm["oiadp"] / rev
    f["gross_margin"]    = (rev - ttm["cogs"]) / rev
    f["ebitda_margin"]   = ttm["oibdp"] / rev
    f["net_margin"]      = ttm["ni"] / rev
    f["accruals"]        = (ttm["ni"] - ttm["ocf"]) / rev
    f["rd_intensity"]    = ttm["xrd"] / rev
    f["capex_intensity"] = ttm["capx"] / rev
    f["sga_intensity"]   = ttm["xsga"] / rev
    f["div_intensity"]   = ttm["dv"] / rev

    # year-on-year change (4 quarters back, within firm)
    rev4 = ttm["rev"].groupby(gv, sort=False).transform(lambda s: s.shift(4))
    ni4 = ttm["ni"].groupby(gv, sort=False).transform(lambda s: s.shift(4))
    rev4 = rev4.where(rev4 > 0)
    f["sales_growth"] = ttm["rev"] / rev4 - 1.0
    f["earnings_chg"] = (ttm["ni"] - ni4) / rev4
    return f
