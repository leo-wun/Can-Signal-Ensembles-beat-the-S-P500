import pandas as pd
import numpy as np
import sys
from pathlib import Path
import wrds


sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_crsp(path=None) -> pd.DataFrame:
    """Load CRSP daily stock returns."""
    path = path or config.CRSP_PATH
    df = pd.read_csv(
        path,
        low_memory=False,
        dtype={"PERMNO": int, "PERMCO": "Int64", "SICCD": str, "NAICS": str},
        parse_dates=["DlyCalDt"],
    )
    df.rename(columns={"DlyCalDt": "date", "DlyRet": "ret", "sprtrn": "mkt_ret"}, inplace=True)
    df.sort_values(["PERMNO", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    db = wrds.Connection(wrds_username="leowunderli")
    # Fetch price and shares data for market cap calculations.
    permno_list = ",".join(str(int(p)) for p in df["PERMNO"].dropna().unique())
    query = f"""
    SELECT permno, date, prc AS dlyprc, shrout
    FROM crsp.dsf
    WHERE date BETWEEN '{config.START_DATE}' AND '{config.END_DATE}'
      AND permno IN ({permno_list})
    """
    wrds_df = db.raw_sql(query, date_cols=["date"])
    wrds_df.rename(columns={"permno": "PERMNO", "dlyprc": "DlyPrc", "shrout": "ShrOut"}, inplace=True)

    df = df.merge(wrds_df, on=["PERMNO", "date"], how="left")
    df["mktcap"] = df["DlyPrc"].abs() * df["ShrOut"] * 1_000
    df["log_mktcap"] = np.log(df["mktcap"].clip(lower=1))
    return df



def load_futures(path=None) -> pd.DataFrame:
    """Load daily futures prices (wide format, one column per instrument)."""
    path = path or config.FUTURES_PATH
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

