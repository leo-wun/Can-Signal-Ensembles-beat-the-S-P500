"""
Step 2 of the monthly study: merge Compustat fundamentals onto the panel.

Builds trailing-twelve-month fundamental characteristics from the quarterly
Compustat dataset and merges them onto the monthly panel using the
CRSP/Compustat Merged (CCM) linking table (gvkey <-> PERMNO with validity
ranges), as recommended by the project guidelines. A 4-month reporting lag is
applied so each month receives only data that was publicly available at the
time. Output: data/processed/monthly_dataset.parquet.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.monthly_features import FEATURE_COLS
from src.compustat_features import (load_compustat, build_fundamental_features,
                                    FUNDAMENTAL_COLS)

REPORTING_LAG_MONTHS = 4
FAR_FUTURE = pd.Timestamp("2099-12-31")


def main() -> None:
    panel = pd.read_parquet(config.MONTHLY_PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["PERMNO"] = panel["PERMNO"].astype("int64")
    print(f"panel: {len(panel):,} rows")

    cs = load_compustat(config.COMPUSTAT_PATH)
    cs["gvkey"] = pd.to_numeric(cs["gvkey"], errors="coerce").astype("Int64")
    cs = cs.dropna(subset=["gvkey"]).copy()
    cs["gvkey"] = cs["gvkey"].astype("int64")
    print(f"Compustat (standard view): {len(cs):,} rows")
    fund = build_fundamental_features(cs)
    fund["gvkey"] = fund["gvkey"].astype("int64")

    ccm = pd.read_parquet(config.CCM_LINKTABLE_PATH)
    ccm["linkdt"]    = pd.to_datetime(ccm["linkdt"])
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"]).fillna(FAR_FUTURE)
    print(f"CCM linktable: {len(ccm):,} links | "
          f"{ccm['gvkey'].nunique():,} unique gvkey | "
          f"{ccm['lpermno'].nunique():,} unique PERMNO")

    # attach the valid PERMNO to each (gvkey, datadate) using linkdt/linkenddt 
    linked = fund.merge(
        ccm[["gvkey", "lpermno", "linkdt", "linkenddt"]], on="gvkey", how="inner")
    mask = (linked["datadate"] >= linked["linkdt"]) \
         & (linked["datadate"] <= linked["linkenddt"])
    linked = (linked[mask]
              .drop(columns=["linkdt", "linkenddt", "cusip8"])
              .rename(columns={"lpermno": "PERMNO"})
              .drop_duplicates(["gvkey", "datadate", "PERMNO"], keep="first"))
    linked["available"] = linked["datadate"] + pd.DateOffset(months=REPORTING_LAG_MONTHS)
    print(f"Compustat after CCM link: {len(linked):,} rows | "
          f"{linked['PERMNO'].nunique():,} unique PERMNO")

    panel = panel.sort_values("date")
    linked = linked.sort_values("available")
    merged = pd.merge_asof(
        panel,
        linked[["PERMNO", "available", "size_proxy"] + FUNDAMENTAL_COLS],
        left_on="date", right_on="available", by="PERMNO", direction="backward")

    has_fund = merged[FUNDAMENTAL_COLS].notna().any(axis=1)
    usable = merged.dropna(subset=["target"] + FEATURE_COLS)
    rec = usable[usable["date"] >= "1980-01-01"]
    print(f"\nmerged: {len(merged):,} rows | fundamental match overall: {has_fund.mean():.1%}")
    print(f"usable rows (target + technical features) >= 1980: {len(rec):,} | "
          f"with fundamentals: {rec[FUNDAMENTAL_COLS].notna().any(axis=1).mean():.1%}")
    print("\nfundamental coverage (non-null %, all merged rows):")
    print((merged[FUNDAMENTAL_COLS].notna().mean() * 100).round(1).to_string())

    merged = merged.drop(columns=["available"])
    merged.to_parquet(config.MONTHLY_DATASET_PATH, index=False)
    print(f"\nsaved -> {config.MONTHLY_DATASET_PATH}  ({len(merged):,} rows | "
          f"{len(FEATURE_COLS)} technical + {len(FUNDAMENTAL_COLS)} fundamental features)")


if __name__ == "__main__":
    main()
