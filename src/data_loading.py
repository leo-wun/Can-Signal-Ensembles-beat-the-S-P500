import pandas as pd
import numpy as np
import sys, os
from pathlib import Path
import wrds
import polars as pl
from sklearn.preprocessing import RobustScaler
import password

sys.path.insert(0, str(Path(__file__).parent.parent))
import config




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

    PATH = path or config.CRSP_PATH_RAW

    df.to_pandas().to_parquet(PATH, index=False)
    print(f'Saved as .parquet file to {PATH}')

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

    PATH = path or config.CZ_PATH_RAW

    df_cz = pd.read_csv(config.CZ_PATH, parse_dates=[date_col])
    
    df_cz.to_parquet(PATH, index=False)
    print(f'Saved as .parquet file to {PATH}')

    return df_cz






def load_VIX(
    start_date = config.START_DATE,
    end_date = config.END_DATE,
    path : str = None
) -> pd.DataFrame:

    '''
    Query VIX data on wrds
    
    '''
    # 1. Set destination
    PATH = path or config.VIX_PATH_RAW

    # 2. Connect to wrds database
    db = wrds.Connection()

    # 3. Construction dynamique de la requête SQL
    if start_date and end_date:
        # Si les dates sont spécifiées, on filtre
        sql_cboe = f"""
            SELECT date, vix
            FROM cboe.cboe
            WHERE date >= '{start_date}'
            AND date <= '{end_date}'
        """
    else:
        # RANGE MAXIMAL : Si aucune date n'est fournie, on prend tout
        sql_cboe = """
            SELECT date, vix
            FROM cboe.cboe
        """

    vix_crsp = db.raw_sql(sql_cboe, date_cols=['date'])
    vix_crsp['date'] = pd.to_datetime(vix_crsp['date'])

    vix_crsp.to_parquet(PATH, index=False)
    print(f'Saved as .parquet file to {PATH}')

    return vix_crsp



def clean_vix(
        df : pd.DataFrame,
        path : str = None
) -> pd.DataFrame:


    # 1. Set path
    if path == None:
        path = config.VIX_PATH_RAW

    # 2. Set date as index and make it unique
   
    df = df.dropna()
    df = df.set_index('date')
    df = df.astype('float32')

    df = df[~df.index.duplicated(keep='last')]

    # 3. Reduce dataframe RAM size
    ma_range = config.MA_RANGE
    for elem in ma_range:
        df[f'vix_ma_{elem}'] = df['vix'].rolling(window=elem, min_periods=elem).mean().astype('float32')

    df = df.dropna()

    # 4. One day shift
    df = df.shift(1).dropna()

    # 5. Normalize
    df = df / 100

    df.to_parquet(config.VIX_PATH_CLEAN, index=True)
    print('Saved as .parquet file to {config.VIX_PATH_CLEAN}')

    return df


def fetch_sp500(
    start_date = None,
    end_date = None,
    path : str = None
) -> pd.DataFrame:
    
    '''
    Query S&P500 data on wrds
    
    
    '''
    
    # 1. Set destination

    if path == None:
        path = config.GSPC_PATH_RAW

    # 2. Connect to wrds database

    db = wrds.Connection()

    # 3. Construction dynamique de la requête SQL
    if start_date and end_date:
        # Si les dates sont spécifiées, on filtre
        sp500 = f"""
            SELECT date, vwretd, ewretd, sprtrn
            FROM crsp.dsi
            WHERE caldt >= '{start_date}',
            AND caldt <= '{end_date}'
        """
    else:
        # RANGE MAXIMAL : Si aucune date n'est fournie, on prend tout
        sp500 = """
            SELECT date, vwretd, ewretd, sprtrn
            FROM crsp.dsi
        """

    gspc_crsp = db.raw_sql(sp500, date_cols=['date'])
    gspc_crsp['date'] = pd.to_datetime(gspc_crsp['date'])

    gspc_crsp.to_parquet(path, index=False)
    print(f"Data saved to {path}")

    return gspc_crsp
