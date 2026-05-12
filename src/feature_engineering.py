"""
Feature engineering for next-day return prediction.

Pipeline:
  1. Momentum features  — cumulative log returns over multiple lookback windows
  2. Reversal feature   — previous-day return (short-term mean reversion)
  3. Volatility features — rolling realized std of daily returns
  4. Market cap         — log market cap (size), cross-sectionally z-scored per date
  5. Futures features   — lagged log returns for each futures instrument
  6. build_features()   — combines all of the above into a single panel
"""

import pandas as pd
import polars as pl
import polars.selectors as cs
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config



# ── Stock-level features ──────────────────────────────────────────────────────

def add_momentum(df: pd.DataFrame, value_col : str,  windows = None, min_period : float = 0.8, reversal : bool = True) -> pd.DataFrame:
    """
    Cumulative log return over each window (keep day t).

    Input ->

    df : Initial dataframe with at least a return column
    value_col : Name of df column corresponding to return
    windows : lookback window
    min_period : degree of completion on the lookback window for a value to be computed (otherwise -> NaN)
    reversal : Wether or not we want to consider last day return as reversal feature or not

    Output ->

    Dataframe with following columns added
    Column names: mom_5d, mom_21d, …

    """

    if not value_col:
        print('No parameter passed for value_col')
        return 

    windows = windows or config.MOMENTUM_WINDOWS
    df = df.copy().sort_values(["PERMNO", "date"])
    log_ret = np.log1p(df[value_col])

    # Attach as a named series so groupby can reference it
    df["_log_ret"] = log_ret

    # Set closed parameters
    closed = 'both' # closed = both set interval as [start, end]
    if reversal:
        closed = 'left' # closed = left set interval as [start, end)

    for w in windows:
        df[f"mom_{w}d"] = df.groupby("PERMNO")["_log_ret"].transform(
            lambda x: x.rolling(w, min_periods= int(w * min_period), closed=closed).sum() 
        )

    df.drop(columns=["_log_ret"], inplace=True)

    return df

def add_volatility_momentum(df: pd.DataFrame, value_col : str,  windows = None, min_period : float = 0.8, reversal : bool = True) -> pd.DataFrame:
    """
    Cumulative log return over each window, shifted by 2 days (keep day t).

    Input ->

    df : Initial dataframe with at least a return column
    value_col : Name of df column corresponding to return
    windows : lookback window
    min_period : degree of completion on the lookback window for a value to be computed (otherwise -> NaN)
    reversal : Wether or not we want to consider last day return as reversal feature or not

    Output ->

    Dataframe with following columns added
    Column names: mom_5d, mom_21d, …

    """

    if not value_col:
        print('No parameter passed for value_col')
        return 

    windows = windows or config.MOMENTUM_WINDOWS

    df = df.copy().sort_values(["PERMNO", "date"])
    log_ret = np.log1p(df[value_col])

    # Attach as a named series so groupby can reference it
    df["_log_ret"] = log_ret

    # Set closed parameters
    closed = 'left' if reversal else 'both' # closed = left set interval as [start, end) ; closed = both set interval as [start, end]

    for w in windows:
        min_p = int(w * min_period)
        
        # Calculate sum and std separately to avoid calculating rolling windows twice in a lambda
        roll_sum = df.groupby("PERMNO")["_log_ret"].transform(
            lambda x: x.rolling(w, min_periods=min_p, closed=closed).sum()
        )
        roll_std = df.groupby("PERMNO")["_log_ret"].transform(
            lambda x: x.rolling(w, min_periods=min_p, closed=closed).std()
        )
        
        # Scale and annualize
        df[f"mom_scaled_{w}d"] = (roll_sum / (roll_std * np.sqrt(252)))
        df[f"mom_{w}d"] = roll_sum 
        df[f'vol_{w}d'] = roll_std * np.sqrt(252)

    df.drop(columns=["_log_ret"], inplace=True)

    return df
 


def add_volatility(df: pd.DataFrame, value_col : str, windows = None, min_period = 0.8) -> pd.DataFrame:
    """
    Rolling realized volatility (annualised std of daily log returns).

    Uses returns up to and including t (we are aware of day t ).
    Column names: vol_21d, vol_63d, vol_252d

    Input ->
    df : Initial dataframe with at least a return column
    value_col : Name of df column corresponding to return
    min_period : Fraction of lookback windows necessary for computation (otherwise NaN)

    Output ->

    Dataframe with volatility column added

    """

    if not value_col:
        print('No parameter passed for value_col')
        return 

    windows = windows or config.VOLATILITY_WINDOWS

    df = df.copy().sort_values(["PERMNO", "date"])
    df["_log_ret"] = np.log1p(df[value_col])

    for w in windows:
        df[f"vol_{w}d"] = df.groupby("PERMNO")["_log_ret"].transform(
            lambda x: x.rolling(w, min_periods=int(w * min_period), closed='both').std() * np.sqrt(252)
        )

    df.drop(columns=["_log_ret"], inplace=True)

    return df


def add_reversal(df: pd.DataFrame, value_col : str = None, reversal : bool = True) -> pd.DataFrame:

    """
    Previous-day log return (short-term reversal signal).
    
    Input ->
    df : Initial dataframe with at least a return column
    value_col : Name of df column corresponding to return
    reversal : Wether or not we want to consider last day return as reversal feature or not

    Output -> 
    
    Dataframe with reversal column added
    
    """

    if not value_col:
        print('No parameter passed for value_col')
        return 

    if not reversal:
        print('1 day reversal is not considered. Change reversal to True if you want it to be considered.')
        return

    df = df.copy().sort_values(["PERMNO", "date"])

    df["_log_ret"] = np.log1p(df[value_col])

    df["reversal_1d"] = df.groupby("PERMNO")["_log_ret"].transform(lambda x: x.shift(1))

    df.drop(columns=["_log_ret"], inplace=True)

    return df


def add_mktcap_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score of log market cap within each date.

    Requires `log_mktcap` produced by preprocessing.clean_crsp (needs DlyPrc
    and ShrOut in the raw CRSP file — download these from WRDS if missing).
    """


    if "log_mktcap" not in df.columns:
        return df
    
    df = df.copy()
    df["size"] = df.groupby("date")["log_mktcap"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    return df


# ── Futures macro features ────────────────────────────────────────────────────

def add_futures_features(stock_df: pd.DataFrame, futures_df: pd.DataFrame,
                         lags=None) -> pd.DataFrame:
    """
    Merge lagged futures log returns into the stock panel as macro features.

    `futures_df` must have a `date` column and one column per instrument
    (log-returns, from preprocessing.clean_futures).

    For each lag l, appends columns named {instrument}__lag{l}.
    """
    lags = lags or config.FUTURES_LAGS
    fut = futures_df.copy().sort_values("date").set_index("date")

    lagged_frames = []
    for lag in lags:
        shifted = fut.shift(lag)
        shifted.columns = [f"{c}__lag{lag}" for c in fut.columns]
        lagged_frames.append(shifted)

    fut_lagged = pd.concat(lagged_frames, axis=1).reset_index()
    return pd.merge(stock_df, fut_lagged, on="date", how="left")


# ── Target ────────────────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame, value_col : str, trading_interval : bool = True) -> pd.DataFrame:

    """
    Next-day return as the prediction target (NaN for the last row per stock).
    
    Input -> 

    df : Initial dataframe with at least a return column
    value_col : Name of df column corresponding to return
    trading_interval : Wether we are forecasting t + 1 or t + 2. If true, we consider that data until time t (included) will be used on t + 1
    
    Output ->

    Dataframe with Target column appended

    """

    shift = -2
    if not  trading_interval:
        shift = -1

    df = df.copy().sort_values(["PERMNO", "date"])

    df["target"] = df.groupby("PERMNO")[value_col].shift(shift)


    return df


# ── Cross-sectional normalisation ─────────────────────────────────────────────

def crosssectional_rank(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Map each feature to a cross-sectional rank score in [-0.5, 0.5] per date.

    Rank normalisation makes the neural net input stable across time and
    comparable across stocks of very different sizes.
    """
    df = df.copy()
    for col in feature_cols:
        df[col] = df.groupby("date")[col].transform(
            lambda x: x.rank(pct=True, na_option="keep") - 0.5
        )
    return df

# ── Gather list of features columns ─────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:

    """
    Return all feature column names (excludes identifiers, raw return, and target).

    Input -> 
    df:     Pandas dataframe

    Output ->
    features_cols:      List of all features in df

    """
    non_feature = {
        "PERMNO", "HdrCUSIP", "CUSIP", "Ticker", "TradingSymbol",
        "PERMCO", "SICCD", "NAICS", "date", "ret", "mkt_ret",
        "DlyPrc", "ShrOut", "mktcap", "log_mktcap", "target",
    }
    return [c for c in df.columns if c not in non_feature]



# ── Master builder ────────────────────────────────────────────────────────────

def build_features(crsp: pd.DataFrame,
                   value_col : str,
                   rank_normalize: bool = True,
                   trading_interval : bool = True,
                   reversal : bool = True,
                   ) -> pd.DataFrame:
    """
    Assemble the full feature matrix for the neural net.

    Args:
        crsp:           Cleaned CRSP DataFrame (output of preprocessing.clean_crsp).
        futures:        Cleaned futures returns DataFrame (preprocessing.clean_futures).
        value_col:      Name of the column with returns
        rank_normalize: Apply cross-sectional rank normalization to stock-level features.
        trading_interval: Wether or not we consider an interval day between data (time t) and buying in the market (time t + 1)
        reversal:       Wether or not we consider the day before in momentum formula and if we add the reversal feature

    Returns:
        Panel DataFrame: one row per (PERMNO, date) with all features + `target`.
        Rows with NaN in features are kept — handle masking in the model pipeline.
    """
    df = crsp.copy()
    df = add_reversal(df=df, value_col=value_col, reversal=reversal)
    df = add_volatility_momentum(df=df, value_col=value_col, reversal=reversal)
    #df = add_mktcap_zscore(df)
    df = add_target(df=df, value_col=value_col, trading_interval=trading_interval)
    #df = add_futures_features(df, futures)

    df = df.dropna()

    FEATURES_COLUMNS = get_feature_cols(df)

    if rank_normalize:
        df = crosssectional_rank(df, FEATURES_COLUMNS)

    return df





# ── Features engineering using polars ────────────────────────────────────────────────────────────


def pl_add_reversal(df: pl.DataFrame, value_col : str = 'ret', date_col : str = 'date', reversal : bool = True) -> pl.DataFrame:

    """
    Previous-day log return (short-term reversal signal).
    
    Input ->

    df : Initial dataframe with at least a return column
    value_col : Name of df column corresponding to return
    reversal : Wether or not we want to consider last day return as reversal feature or not

    Output -> 
    
    Dataframe with reversal column added
    
    """

    if not value_col:
        print('No parameter passed for value_col')
        return 

    if not reversal:
        print('1 day reversal is not considered. Change reversal to True if you want it to be considered.')
        return

    df = df.sort(['PERMNO', date_col]).with_columns([
        pl.col(value_col)
            .log1p()
            .shift(1)
            .over('PERMNO')
            .alias('reversal_1d')
    ])

    return df





def polars_features(
        df,
        value_col : str = 'ret',
        date_col : str = 'date',
        target_col : str = 'ret',
        crosssectional_rank : bool = True
) -> pl.DataFrame:

    """
    Previous-day log return (short-term reversal signal).
    
    Input ->

    df :            Initial dataframe with at least a return column (can be polars or pandas)
    value_col :     Name of df column corresponding to return
    reversal :      Wether or not we want to consider last day return as reversal feature or not
    target_col:     Column we are regressing/classifying on

    Output -> 
    
    Dataframe with reversal column added
    
    """


    shift_days = 1
    
        
    # 1. Check if dataframe provided is polars or pandas. Transform to polars 
    
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    

    # 2. Clip to 1% and 99% quantile
    q_low = df[value_col].quantile(0.01)
    q_high = df[value_col].quantile(0.99)

    df = df.with_columns(
        pl.col(value_col).clip(lower_bound = q_low, upper_bound = q_high)
    )


    # 3. Structure for momentum features
    momentum_exprs = [
    (
        pl.col(value_col)
        .log1p()
        .rolling_sum(window_size = window)
        .exp() 
        - 1
    )
    .shift(shift_days)
    .over("PERMNO")
    .alias(f"mom_{window}d") 
    for window in config.MOMENTUM_WINDOWS
    ]
    
    # 4. Structure for volatility features
    
    vol_exprs = [
    (
        pl.col(value_col)
        .rolling_std(window_size = window)
    )
    .shift(shift_days)
    .over("PERMNO")
    .alias(f"vol_{window}d") 
    for window in config.VOLATILITY_WINDOWS
    ]
    
    
    # 5. Structure for volatility weighted momentum
    vol_w_mom_exprs = []
    
    # Creating a combination of all momentum and volatility windows. We weight momentum by rolling std computed on a similar window.
    for mom_window in config.MOMENTUM_WINDOWS:
        expr = (
            (
                # Use standard division `/` operator instead of .divide()
                (pl.col(value_col).log1p().rolling_sum(window_size=mom_window).exp() - 1) 
                / pl.col(value_col).rolling_std(window_size=mom_window)
            )
            .shift(shift_days)
            .over("PERMNO")
            .alias(f"vol_w_mom_{mom_window}d_std_{mom_window}d")
        )
        vol_w_mom_exprs.append(expr)



    df = df.sort(['PERMNO', date_col]).with_columns([
        
        # Reversal feature
        
        pl.col(value_col)
            .log1p()
            .shift(1)
            .over('PERMNO')
            .alias('log_reversal_1d'),

        # Market Return
        pl.col('mkt_ret')
            .log1p()
            .shift(1)
            .over('PERMNO')
            .alias('log_reversal_mkt_1d'),
            
        # Target 
        pl.col(target_col)

            .alias('target'),

        # Other features

        *momentum_exprs,
        *vol_exprs,
        *vol_w_mom_exprs


    ])

    # ── Cross-sectional rank normalization ────────────────────────────────
    # Applied AFTER computing features, per date, on cross-sectional cols only.
    # Macro features (e.g. mkt_ret) are excluded — they are identical across
    # stocks on a given date so ranking them is meaningless.

    if crosssectional_rank:
        cs_cols = (
            [f"mom_{w}d" for w in config.MOMENTUM_WINDOWS]
            + [f"vol_{w}d" for w in config.VOLATILITY_WINDOWS]
            + [f"vol_w_mom_{w}d_std_{w}d" for w in config.MOMENTUM_WINDOWS]
            + ["log_reversal_1d"]
            # Do NOT include: log_reversal_mkt_1d, VIX, or any macro feature
        )

        rank_exprs = [
            (
                pl.col(col)
                .rank(method="average")          # average handles ties
                .over(date_col)                  # cross-sectional: rank within each date
                / pl.col(col).count().over(date_col)  # normalize to [0, 1]
                - 0.5                            # center to [-0.5, 0.5]
            )
            .alias(col)
            for col in cs_cols
        ]

        df = df.with_columns(rank_exprs)

    return df

