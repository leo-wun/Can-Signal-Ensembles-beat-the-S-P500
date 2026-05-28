"""
Step 1 of the monthly study: build the cross-sectional panel.

Loads the provided Monthly CRSP returns, cleans them, and adds technical
characteristics (short-term reversal, momentum, long-term reversal, volatility)
plus the next-month prediction target. Compustat fundamentals are merged in a
later step. Output: data/processed/monthly_panel.parquet.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.monthly_features import build_monthly_panel, FEATURE_COLS


def main(
) -> None:
    
    # Get the data and format
    df = pd.read_csv(config.MONTHLY_CRSP_PATH,
                     usecols=["PERMNO", "CUSIP", "SICCD", "MthCalDt", "MthRet", "sprtrn"],
                     dtype={"CUSIP": str})
    df["date"] = pd.to_datetime(df["MthCalDt"])
    df["ret"] = pd.to_numeric(df["MthRet"], errors="coerce")
    df = df.dropna(subset=["ret"])
    df = df.drop_duplicates(["date", "PERMNO"], keep="last")
    df = df.drop(columns=["MthCalDt", "MthRet"])
    
    # Display unique PERMNO and date range
    print(f"loaded {len(df):,} monthly obs | {df['PERMNO'].nunique():,} stocks | "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    # Build the monthly panel
    panel = build_monthly_panel(df) 
    
    # Drop NaN values
    usable = panel.dropna(subset=["target"] + FEATURE_COLS)
    
    # Display data
    print(f"usable rows (target + all {len(FEATURE_COLS)} features): {len(usable):,}")
    print(f"  since 1970: {len(usable[usable['date'] >= '1970-01-01']):,}"
          f"  | since 2000: {len(usable[usable['date'] >= '2000-01-01']):,}")
    print("\nfeature / target summary (usable rows):")
    print(usable[FEATURE_COLS + ["target"]].describe().T.to_string(
        float_format=lambda x: f"{x:.4f}"))

    # Save panel to .parquet 
    panel.to_parquet(config.MONTHLY_PANEL_PATH, index=False)
    print(f"\nsaved -> {config.MONTHLY_PANEL_PATH}  ({len(panel):,} rows)")


if __name__ == "__main__":
    main()
