"""
02 — Feature building and auxiliary data loading  (converted from 02_features.ipynb)

Assumes that data/processed/daily_crsp.parquet already exists (created by
01_preprocess_eda.py or by calling clean_crsp() manually).

Steps performed:
  1. Load cleaned CRSP daily parquet and rebuild cross-sectional features
     → FEATURES_PATH_CLEAN  (data/processed/features.parquet)
  2. Clean the Chen-Zimmermann factors via clean_cz_monthly() — identical output
     to 01_preprocess_eda.py (daily, decorrelated, shifted, date as index)
     → CZ_PATH_CLEAN         (data/processed/Chen_Zimmerman_monthly.parquet)
  3. [Optional — requires WRDS access] Fetch VIX from WRDS and clean it
     → VIX_PATH_CLEAN        (data/processed/vix.parquet)
     Skip step 3 if data/raw/vix_raw.parquet already exists.

Run from the project root:
    python notebooks/02_features.py [--no-vix]

Prerequisites:
    data/processed/daily_crsp.parquet  (run 01_preprocess_eda.py first)
    data/raw/Compustat_quarterly.csv   (course-provided)
    data/raw/Chen_Zimmerman_monthly.csv (course-provided)
    password.py with WRDS_USERNAME / WRDS_PASSWORD  (only needed for VIX fetch)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure UTF-8 console output on Windows (some src prints use → arrows)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.utils import shrink_polars
from src.feature_engineering import polars_features
from src.preprocessing import clean_cz_monthly
from src.data_loading import load_cz_monthly, clean_vix


# ── Step 1: Rebuild daily features ───────────────────────────────────────────

def build_features() -> None:
    """Load cleaned CRSP parquet, run polars_features, save to FEATURES_PATH_CLEAN."""
    print("=" * 60)
    print("Step 1: Build daily technical features")
    print("=" * 60)

    crsp_path = config.CRSP_PATH_CLEAN
    if not crsp_path.exists():
        raise FileNotFoundError(
            f"Cleaned CRSP parquet not found: {crsp_path}\n"
            "Run notebooks/01_preprocess_eda.py first."
        )

    data = pd.read_parquet(crsp_path)
    data = data.sort_values(["PERMNO", "date"]).reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])

    print(f"  Loaded CRSP: {data.shape}  |  Stocks: {data['PERMNO'].nunique():,}")

    df = polars_features(data, value_col="ret", date_col="date", target_col="ret",
                         crosssectional_rank=True)

    print(f"  RAM before shrink: {df.estimated_size() / 1024**3:.2f} GB")
    df = shrink_polars(df)
    print(f"  RAM after  shrink: {df.estimated_size() / 1024**3:.2f} GB")

    df.to_pandas().to_parquet(config.FEATURES_PATH_CLEAN, compression="zstd")
    print(f"  Saved → {config.FEATURES_PATH_CLEAN}")


# ── Step 2: Rebuild CZ processed parquet ─────────────────────────────────────

def build_cz() -> None:
    """
    Produce the cleaned daily CZ parquet via clean_cz_monthly().

    This is the same call 01_preprocess_eda.py makes, so both scripts now write
    an identical CZ_PATH_CLEAN (daily frequency, incomplete/correlated columns
    dropped, 1-day shift, /100 normalisation, date as the index).
    """
    print("\n" + "=" * 60)
    print("Step 2: Clean Chen-Zimmermann factors")
    print("=" * 60)

    # clean_cz_monthly reads CZ_PATH_RAW; bootstrap it from the CSV if absent.
    if not config.CZ_PATH_RAW.exists():
        print("  CZ raw parquet not found — loading from CSV...")
        load_cz_monthly()                 # reads Chen_Zimmerman_monthly.csv → CZ_PATH_RAW

    cz = clean_cz_monthly()               # reads CZ_PATH_RAW, saves CZ_PATH_CLEAN
    print(f"  Saved → {config.CZ_PATH_CLEAN}")
    print(f"  Shape: {cz.shape}  |  Factors: {cz.shape[1]}  |  date as index: "
          f"{cz.index.name == 'date'}")


# ── Step 3: VIX (optional — WRDS) ────────────────────────────────────────────

def build_vix() -> None:
    """
    Fetch VIX from WRDS (requires password.py) and produce the cleaned daily parquet.
    Skipped automatically if data/raw/vix_raw.parquet already exists.
    """
    print("\n" + "=" * 60)
    print("Step 3: VIX data")
    print("=" * 60)

    vix_raw_path = config.VIX_PATH_RAW

    if vix_raw_path.exists():
        print(f"  VIX raw parquet already present: {vix_raw_path}")
        print("  Skipping WRDS fetch — running clean_vix() on existing file")
    else:
        print("  Fetching VIX from WRDS (requires password.py) ...")
        try:
            from src.data_loading import load_VIX  # noqa: F401  (import triggers WRDS connection)
            load_VIX()                               # saves vix_raw_path
        except Exception as exc:
            print(f"  WARNING: VIX fetch failed ({exc}). Skipping VIX processing.")
            return

    df_vix = pd.read_parquet(vix_raw_path)
    clean_vix(df_vix)          # saves VIX_PATH_CLEAN
    print(f"  Saved → {config.VIX_PATH_CLEAN}")

    vix_clean = pd.read_parquet(config.VIX_PATH_CLEAN)
    print(f"  Shape: {vix_clean.shape}")
    print(vix_clean.head(3).to_string())


# ── Entry point ───────────────────────────────────────────────────────────────

def main(include_vix: bool = True) -> None:
    build_features()
    build_cz()
    if include_vix:
        build_vix()
    else:
        print("\nVIX step skipped (--no-vix flag).")

    print("\nDone.")
    print(f"  Features → {config.FEATURES_PATH_CLEAN}")
    print(f"  CZ       → {config.CZ_PATH_CLEAN}")
    if include_vix:
        print(f"  VIX      → {config.VIX_PATH_CLEAN}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features and auxiliary data parquets.")
    parser.add_argument(
        "--no-vix", action="store_true",
        help="Skip the VIX WRDS fetch (use when WRDS access is not available).",
    )
    args = parser.parse_args()
    main(include_vix=not args.no_vix)
