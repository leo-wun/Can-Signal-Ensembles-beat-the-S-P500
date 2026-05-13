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


