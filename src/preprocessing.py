import pandas as pd
import numpy as np
import sys
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def clean_crsp(path = config.CRSP_PATH_RAW) -> pd.DataFrame:
    """
    Clean CRSP daily returns.

    Expected columns: PERMNO, date, ret, mkt_ret
    - Drop rows where return is missing
    - Restrict to project date window
    - Cap extreme returns at ±300% (data errors / delistings)
    """
    df = pl.read_parquet(path)
    bad_permnos = (df.group_by("PERMNO").agg(pl.col('ret').is_null().any().alias('has_null')).filter(pl.col('has_null')).select('PERMNO').to_series().to_list())

    df = df.filter(~pl.col('PERMNO').is_in(bad_permnos))
    df = df.filter(
        pl.col('date').is_between(
            pl.lit(config.START_DATE).str.to_date(),
            pl.lit(config.END_DATE).str.to_date()
        )
    )
    df = df.with_columns(pl.col('ret').clip(-3.0, 3.0))

    df.to_pandas().to_parquet(config.CRSP_PATH_CLEAN, index=False)
    print(f'Saved as .parquet file to {config.CRSP_PATH_CLEAN}')

    return df.to_pandas()


def clean_cz_monthly(path=None, date_col='date', completion_factor=0.9, corr_coef=0.95, start_date=config.START_DATE, end_date=config.END_DATE):
    """
    Clean and preprocess the monthly CZ data.
    Steps:
    Load the parquet raw file.
    Drop columns with too many missing values (based on completion_factor).
    Drop columns with high correlation (based on corr_coef).
    """
    PATH = path or config.CZ_PATH_RAW

    df_cz = pd.read_parquet(PATH)
    
    thresh = completion_factor * df_cz.shape[0]
    df_cz = df_cz.dropna(axis=1, thresh=thresh)
    
    corr_matrix = df_cz.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_coef)]
    df_cz.drop(columns=to_drop, inplace=True)

    df_cz.set_index(date_col).sort_index()
    daily_index = pd.date_range(start=df_cz[date_col].min(), end=df_cz[date_col].max() + pd.offsets.MonthEnd(1), freq='D')
    df_cz_daily = df_cz.set_index(date_col).reindex(daily_index).ffill().reset_index().rename(columns={'index': date_col})
    df_cz_daily.dropna(axis=0, inplace=True)

    num_cols = df_cz_daily.select_dtypes('number').columns
    df_cz_daily[num_cols] = df_cz_daily[num_cols] / 100
    factor_cols = [c for c in df_cz_daily.columns if c != date_col]
    df_cz_daily[factor_cols] = df_cz_daily[factor_cols].shift(1)
    df_cz_daily = df_cz_daily.dropna()
    df_cz_daily = df_cz_daily[(df_cz_daily[date_col] >= start_date) & (df_cz_daily[date_col] <= end_date)]

    # Save with `date` as the (ms-precision) index. This is the canonical format:
    # regime_validation.py joins on the date index and expects only factor columns,
    # while utils.merge_and_batch normalises via _ensure_date_as_column.
    df_cz_daily = df_cz_daily.set_index(date_col).sort_index()
    df_cz_daily.index = pd.to_datetime(df_cz_daily.index).astype('datetime64[ms]')
    df_cz_daily.to_parquet(config.CZ_PATH_CLEAN)
    print(f'Saved as .parquet file to {config.CZ_PATH_CLEAN}')
    return df_cz_daily




def clean_vix(
        path = None,
        start_date = config.START_DATE,
        end_date = config.END_DATE,
        ma_range = config.MA_RANGE
) -> pd.DataFrame:

    # 1. Set path
    PATH = path or config.VIX_PATH_RAW

    # 2. Set date as index and make it unique   
    df = pd.read_parquet(PATH)
    df = df.dropna()
    df = df.set_index('date')
    df = df.astype('float32')
    df = df[~df.index.duplicated(keep='last')]

    # 3. Reduce dataframe RAM size
    for elem in ma_range:
        df[f'vix_ma_{elem}'] = df['vix'].rolling(window=elem, min_periods=elem).mean().astype('float32')
    df = df.dropna()

    # 4. One day shift
    df = df.shift(1).dropna()

    # 5. Normalize
    df = df / 100

    df.to_parquet(config.VIX_PATH_CLEAN, index=True)
    print(f'Saved as .parquet file to {config.VIX_PATH_CLEAN}')

    return df
