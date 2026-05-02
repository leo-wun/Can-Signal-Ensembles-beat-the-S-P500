import pandas as pd
import numpy as np
import sys
from pathlib import Path
import wrds
import polars as pl


sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_crsp(path=None) -> pd.DataFrame:
    """Load CRSP daily stock returns."""
    path = path or config.CRSP_PATH
    df = pd.read_csv(
        path,
        low_memory=False,
        dtype={"PERMNO": int, "PERMCO": "Int64", "SICCD": str, "NAICS": str, 'Ticker' : 'category'},
        parse_dates=["DlyCalDt"],
    )
    df.rename(columns={"DlyCalDt": "date", "DlyRet": "ret", "sprtrn": "mkt_ret"}, inplace=True)
    df.sort_values(["PERMNO", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    db = wrds.Connection()
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


def load_crsp_polars(path=None):
    """Load CRSP daily stock returns with polars"""
    df = pl.read_csv(config.CRSP_PATH,
                schema_overrides={'HdrCUSIP' : pl.String,
                                'Ticker' : pl.String,
                                'PERMNO' : pl.Int32,
                                'PERMCO' : pl.Int32,
                                'NAICS' : pl.Utf8,
                                'SICCD' : pl.Utf8,
                                'DlyCalDt' : pl.Date},
                                columns=['PERMNO', 'DlyCalDt', 'DlyRet', 'sprtrn']
                                )
    df = df.rename({'DlyCalDt' : 'date', 'sprtrn' : 'mkt_ret', 'DlyRet' : 'ret'})
    df = df.sort(['PERMNO', 'date'], descending=False)

    bad_permnos = (
        df.group_by('PERMNO')
        .agg(pl.col('ret').is_null().any().alias('has_null'))
        .filter(pl.col('has_null'))
        .select('PERMNO')
        .to_series()
        .to_list()
    )

    df = df.filter(~pl.col('PERMNO').is_in(bad_permnos))

    return df


def wrds_fetch(permno_list):

    # WRDS FETCH
    print('/!| PLEASE FILL CREDENTIALS /!|')
    db = wrds.Connection()

    # Fetch price and shares data for market cap calculations.
    permno_list_sql = ",".join(str(int(p)) for p in permno_list)
    query = f"""
    SELECT permno, date, prc AS dlyprc, shrout
    FROM crsp.dsf
    WHERE date BETWEEN '{config.START_DATE}' AND '{config.END_DATE}'
    AND permno IN ({permno_list_sql})
    """
    wrds_df = db.raw_sql(query, date_cols=["date"])
    wrds_df = pl.from_pandas(wrds_df)
    wrds_df = wrds_df.rename({"permno": "PERMNO", "dlyprc": "DlyPrc", "shrout": "ShrOut"})

    wrds_df = wrds_df.with_columns(
        pl.col('date').cast(pl.Date)
    )

    return wrds_df
    

def load_futures(path=None) -> pd.DataFrame:
    """Load daily futures prices (wide format, one column per instrument)."""
    path = path or config.FUTURES_PATH
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

