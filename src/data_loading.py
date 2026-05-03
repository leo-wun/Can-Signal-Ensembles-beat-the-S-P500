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

    start_date = df['date'].min()
    end_date = df['date'].max()

    db = wrds.Connection()
    # Fetch price and shares data for market cap calculations.
    permno_list = ",".join(str(int(p)) for p in df["PERMNO"].dropna().unique())
    query = f"""
    SELECT permno, date, prc AS dlyprc, shrout
    FROM crsp.dsf
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
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


def wrds_fetch(permno_list, start_date, end_date):

    # WRDS FETCH
    print('/!| PLEASE FILL CREDENTIALS /!|')
    db = wrds.Connection()

    # Fetch price and shares data for market cap calculations.
    permno_list_sql = ",".join(str(int(p)) for p in permno_list)
    query = f"""
    SELECT permno, date, prc AS dlyprc, shrout
    FROM crsp.dsf
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
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



def load_cz_monthly(path : str = None,
                    date_col : str = 'date',
                    completion_factor : float = 0.9,
                    corr_coef : float = 0.95
)-> pd.DataFrame:
    """
    Load and preprocess the Chen-Zimmerman dataset. Include a fill forward to daily frequency to match daily returns

    Input ->

    date_col:               Name of the date column in the dataset
    completion_factor:      Factor of completion of a column required (otherwise column is dropped)
    corr_coef:              Coefficient of correlation above which we drop one of the two columns (Not bringing information)
    
    Output ->

    df:                     Dataframe of the normalized, daily frequency data
    
    
    """

    PATH = path or config.CZ_PATH

    df_cz = pd.read_csv(PATH, parse_dates=[date_col])
    N, c = df_cz.shape

    print(f'Initial Size of the dataset: {N, c}')

    max_date = df_cz['date'].max()
    min_date = df_cz['date'].min()
    print(f'Total Date range : {min_date} -> {max_date}')


    # 1. Get rid of columns with too many NaN

    thresh = completion_factor * df_cz.shape[0]
    df_cz = df_cz.dropna(thresh=thresh, axis=1)
    print(f'Dataset shape after dropping column with less than {completion_factor * 100}% completion: {df_cz.shape}')
    print(f'Total column dropped so far : {c - df_cz.shape[1]}')


    # 2. Study intercolumn correlation (remove column with too high absolute correlation)

    # Create correlation matrix
    corr_matrix = df_cz.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than corr_coef
    to_drop = [column for column in upper.columns if any(upper[column] > corr_coef)]

    # Drop features 
    df_cz.drop(to_drop, axis=1, inplace=True)

    print(f'Dataset shape after dropping highly correlated (corr_coef > {corr_coef}) columns: {df_cz.shape}')
    print(f'Total column dropped so far : {c - df_cz.shape[1]}')

    # 3. Expansion en daily avec forward fill
    df_cz = df_cz.set_index('date').sort_index()

    daily_index = pd.date_range(
        start=df_cz.index.min(),
        end=df_cz.index.max() + pd.offsets.MonthEnd(1),  # étendre jusqu'à fin du dernier mois
        freq='D'
    )

    df_cz_daily = df_cz.reindex(daily_index).ffill()
    df_cz_daily.index.name = 'date'
    df_cz_daily = df_cz_daily.reset_index()
    df_cz_daily = df_cz_daily.dropna(axis=0)
    

    # 5. Scale % to decimal
    num_cols = df_cz_daily.select_dtypes('number').columns
    df_cz_daily[num_cols] = df_cz_daily[num_cols] / 100


    return df_cz_daily

