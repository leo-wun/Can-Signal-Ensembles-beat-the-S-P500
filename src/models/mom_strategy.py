
import pandas as pd
import polars as pl
import polars.selectors as cs
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config




def mom(
        df,
        value_col : str = 'ret',
        date_col : str = 'date',
        mom_window : int = 7,
        skip_window : int = 1,
        vol_window : int = 7,
) -> pl.DataFrame:
    

    # IMPORTANT: Rolling calculations require sorted data. 
    # If the dataframe isn't sorted, rolling_sum will mix up dates.
    if date_col in df.columns and 'PERMNO' in df.columns:
        df = df.sort(['PERMNO', date_col])

    # 1. Define the core mathematical expressions (No shifts or groups yet)
    raw_mom = pl.col(value_col).log1p().rolling_sum(window_size=mom_window).exp() - 1
    raw_vol = pl.col(value_col).rolling_std(window_size=vol_window)

    # 2. Apply shift, grouping, and aliasing to the final logic
    df = df.with_columns([
        
        # Simple Momentum
        raw_mom
        .shift(1 + skip_window)
        .over('PERMNO')
        .alias(f'mom_{mom_window}d'),

        # Volatility-Weighted Momentum (Now correctly parenthesized!)
        (raw_mom / raw_vol)
        .shift(1 + skip_window)
        .over('PERMNO')
        .alias(f'vol_w_mom_{mom_window}d')
        
    ])

    return df




def build_alpha(
        
) -> pl.DataFrame:
    






    return