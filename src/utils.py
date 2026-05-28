
import pandas as pd
import polars as pl
import polars.selectors as cs
import numpy as np
import sys, os, gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config



# ── Fonction to reduce the RAM impact of dataframe ──────────────────────────────────────────────────────

def shrink(df : pd.DataFrame) -> pd.DataFrame:

    for col in df.select_dtypes('float64').columns:
        df[col] = df[col].astype('float32')

    for col in df.select_dtypes('int64').columns:
        df[col] = df[col].astype('int32')

    if 'PERMNO' in df.columns:
        df['PERMNO'] = df['PERMNO'].astype('int32')

    if 'date' in df.columns:
        df['date'] = df['date'].astype('datetime64[ms]')


    return df

def shrink_polars(df) -> pl.DataFrame:

    # 1. Convert to Polars if necessary
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
        
    # 2. Downcast generic float and int types globally
    df = df.with_columns([
        cs.by_dtype(pl.Float64).cast(pl.Float32),
        cs.by_dtype(pl.Int64).cast(pl.Int32),
    ])

    # 3. Handle specific columns safely (only if they exist)
    specific_casts = []
    
    if 'PERMNO' in df.columns:
        specific_casts.append(pl.col('PERMNO').cast(pl.Int32))
        
    # date is intentionally not downcast — changing datetime precision
    # (ms vs ns) causes silent merge failures in pandas 2.x
        
    # Apply specific casts if there are any
    if specific_casts:
        df = df.with_columns(specific_casts)

    return df


def generate_batches(df : pd.DataFrame, sample_col : str, batch_number : int, batch_size : int, overlap : bool = True) -> dict:
    """
    Generate a dictionnary of batches sampled on the the dataframe provided.
    
    Input ->
    
    df:             Dataframe containing at least the sample_col column
    sample_col:     Column on which we want to use to sample the batches
    batch_number:   Number of batch to create
    batch_size:     Number of unique element from sample_col contained in one batch  
    overlap:        Wether or not one element from sample_col can be found in multiple batch


    Output ->

    dict:           Dictionnary containing all batches following the naming format {batch_1, batch_2, ..., batch_N}

    """

    sample_list = df[sample_col].unique()
    batch_dict = {}
    total_unique = len(sample_list)

    # Base check: Do we even have enough data for a single batch?
    if total_unique < batch_size:
        raise ValueError(f"Not enough unique samples for a batch of size {batch_size}. Total unique samples available: {total_unique}")

    if not overlap:
        if len(sample_list) < batch_number * batch_size:
            raise ValueError(f"Not enough unique samples for {batch_number} batch of size {batch_size}. Total samples is {sample_list.shape}")
        samples = np.random.choice(sample_list, size=batch_size * batch_number, replace=False)
        for elem in range(batch_number):
            batch_dict[f'batch_{elem + 1}'] = df[df[sample_col].isin(samples[elem * batch_size : (elem + 1) * batch_size])]
    else :
        for elem in range(batch_number):
            batch_split = np.random.choice(sample_list, size=batch_size, replace=False)
            batch_dict[f'batch_{elem + 1}'] = df[df[sample_col].isin(batch_split)]

    return batch_dict


def generate_date_split(
        df : pd.DataFrame,
        df_to_join : list[pd.DataFrame] = None,
        date_col : str = 'date',
        train_split : float = 0.7,
        val_split : float = 0.15
) -> dict:

    """
    Creates a chronological train/val/test date split for a time-indexed dataframe.

    The split is performed on UNIQUE sorted dates to avoid leakage: all rows
    sharing the same date go to the same split. This matters for panel data
    where many PERMNO share the same date.

    Input ->

    df:             Dataframe containing at least a date_col columns or has date as index
    date_col:       Name of the column containing the dates
    train_split:    Fraction of the total date range used for training
    val_split:      Fraction of the total date range used for validation 

    
    Output ->

    dict:          Tuple containing the boundary of each split


    """

    # Check fraction input
    if not (0 < train_split < 1) or not (0 <= val_split < 1):
        raise ValueError("train_split and val_split must be in (0, 1).")
    if train_split + val_split >= 1:
        raise ValueError(
            f"train_split + val_split must be < 1, got {train_split + val_split}"
        )

    def get_dates(dataframe, col_name):
        if col_name in dataframe.columns:
            return dataframe[col_name]
        elif isinstance(dataframe.index, pd.MultiIndex) and col_name in dataframe.index.names:
            return dataframe.index.get_level_values(col_name)
        elif dataframe.index.name == col_name:
            return dataframe.index
        else:
            raise KeyError(
                f"Column or index level '{col_name}' not found. "
                f"Available columns: {list(dataframe.columns)}, "
                f"index names: {dataframe.index.names}"
            )


    # 1. Get range of main dataframe
    main_dates = get_dates(df, date_col)

    # 2. Get the max range common to all dataframe we want to merge
    common_min_date = main_dates.min()
    common_max_date = main_dates.max()

    for elem in df_to_join:
        elem_dates = get_dates(elem, date_col)
        # Narrow the window to the maximum of the minimums, and minimum of the maximums
        common_min_date = max(common_min_date, elem_dates.min())
        common_max_date = min(common_max_date, elem_dates.max())


    # 3. Convert main dates to datetime, get unique, and sort
    unique_dates = pd.to_datetime(pd.Series(main_dates.unique()))
    unique_dates = unique_dates.sort_values().reset_index(drop=True)

    # 4. Filter dates to only use the max common range computed above
    unique_dates = unique_dates[(unique_dates >= common_min_date) & (unique_dates <= common_max_date)]
    unique_dates = unique_dates.reset_index(drop=True) # Reset index after filtering

    n = len(unique_dates)
    if n < 3:
        raise ValueError(f"Need at least 3 unique dates to split, got {n}.")

    # --- Calcul des indices de coupure ---
    train_end_idx = int(np.floor(n * train_split))
    val_end_idx = int(np.floor(n * (train_split + val_split)))

    # Garde-fous : chaque split doit avoir au moins une date
    train_end_idx = max(train_end_idx, 1)
    val_end_idx = max(val_end_idx, train_end_idx + 1)
    val_end_idx = min(val_end_idx, n - 1)

    splits = {
        'train': (unique_dates.iloc[0], unique_dates.iloc[train_end_idx - 1]),
        'val':   (unique_dates.iloc[train_end_idx], unique_dates.iloc[val_end_idx - 1]),
        'test':  (unique_dates.iloc[val_end_idx], unique_dates.iloc[-1]),
    }

    # Summary
    print(f"Total unique dates: {n}")
    for name, (start, end) in splits.items():
        n_dates = ((unique_dates >= start) & (unique_dates <= end)).sum()
        print(f"  {name:5s}: {start.date()} --> {end.date()}  ({n_dates} dates, {n_dates/n:.1%})")

    return splits


def train_val_test_split(
        df : pd.DataFrame,
        date_col : str,
        df_to_join : list[pd.DataFrame] = None,
        train_split : float = 0.70,
        val_split : float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    """
    Splits a financial dataframe chronologically by unique dates to prevent data leakage.

    Input ->

    df:             Dataframe containing at least date_col
    date_col:       Name of the column/index containing the dates on which we do the split
    train_split:    Fraction of the total date range used for training
    val_split:      Fraction of the total date range used for validation 
    df_to_join:     Optional parameter containing dataframe we would like to join/merge with the baseline one
    
    Returns:
        tuple:      (train_df, val_df, test_df)
    """


    # Generate date split

    date_split = generate_date_split(df = df, df_to_join = df_to_join, date_col = date_col, train_split=train_split, val_split=val_split)

    TRAIN_START = date_split['train'][0]
    TRAIN_END = date_split['train'][1]
    VAL_START = date_split['val'][0]
    VAL_END = date_split['val'][1]
    TEST_START = date_split['test'][0]
    TEST_END = date_split['test'][1]

    df = df.copy().sort_values(date_col)

    train_df = df[df[date_col].between(TRAIN_START, TRAIN_END, inclusive='left')] # Inclusice left acts as [START_DATE : END_DATE)
    val_df   = df[df[date_col].between(VAL_START, VAL_END, inclusive='left')]
    test_df  = df[df[date_col].between(TEST_START, TEST_END, inclusive='both')] # Inclusice both acts as [START_DATE : END_DATE]

    train_df = train_df.set_index([date_col, 'PERMNO'])
    val_df = val_df.set_index([date_col, 'PERMNO'])
    test_df = test_df.set_index([date_col, 'PERMNO'])

    if df_to_join is not None:
        for elem in df_to_join:
            train_df = train_df.join(elem, on=date_col, how='left')
            val_df = val_df.join(elem, on=date_col, how='left')
            test_df = test_df.join(elem, on=date_col, how='left')


    # Étape 1 : drop if target is NaN 
    TARGET_COL = 'target' 
    train_df = train_df.dropna(subset=[TARGET_COL])
    val_df   = val_df.dropna(subset=[TARGET_COL])
    test_df  = test_df.dropna(subset=[TARGET_COL])
    
    # Étape 2 : Get columns features
    feature_cols = [c for c in train_df.columns if c not in [TARGET_COL, 'PERMNO']]
    
    # Option B (meilleure) : imputer avec la médiane du train
    medians = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(medians)
    val_df[feature_cols]   = val_df[feature_cols].fillna(medians)
    test_df[feature_cols]  = test_df[feature_cols].fillna(medians)
    
    print(f"After dropna - train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    return (train_df, val_df, test_df)


# ── Pipeline from .parquet to train/validation/test split batches ──────────────────────────────────────────────────────


def split_batch(df : pd.DataFrame,
                sample_col : str = 'PERMNO', 
                date_col : str = 'date',
                df_to_join : list[pd.DataFrame] = None,
                batch_number : int = 5,
                batch_size : int = 500,
                overlap : bool = True,
                train_split : float = 0.70,
                val_split : float = 0.15
) -> dict:
    """
    Take the dataframe, apply random sampling on the sample_col to create batch_number of batch each of size batch_size.
    For each batch, split into training, validation and test set following dates in the config file.


    Input ->

    df:             Dataframe containing at least sample_col and date_col as columns
    sample_col: 
    date_col:       Name of the column containing the dates on which we do the split
    sample_col:     Column on which we want to use to sample the batches
    batch_number:   Number of batch to create
    batch_size:     Number of unique element from sample_col contained in one batch  
    overlap:        Wether or not one element from sample_col can be found in multiple batch
    df_to_join:     Optional parameter containing dataframe we would like to join/merge with the baseline one


    Output ->

    dict:           Dictionnary of dictionnary of structure : {batch_1 : {train, val, test}, batch_2 : {train, val, test}}
        
    """

    batch_dict = generate_batches(df, sample_col, batch_number, batch_size, overlap)

    split_batch_dict = {}

    for batch_name, batch_df in batch_dict.items():
        # Call the split function
        train_df, val_df, test_df = train_val_test_split(batch_df, date_col, df_to_join=df_to_join, train_split=train_split, val_split=val_split)

        # Store the tuple
        split_batch_dict[batch_name] = {
            'train' : train_df,
            'validation' : val_df,
            'test' : test_df
        }


    return split_batch_dict


def _ensure_date_as_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Garantit que `date_col` est une colonne, pas un index."""
    if date_col in df.columns:
        return df
    if df.index.name == date_col or date_col in (df.index.names or []):
        return df.reset_index()
    raise KeyError(f"'{date_col}' n'est ni colonne ni index de ce DataFrame")


def merge_and_batch(
    train_split: float = 0.70,
    val_split: float = 0.15,
    verbose: bool = True,
) -> dict:
    """
    Read the features/cz/vix parquet files, deduplicate, and make the batch/split.
    """

    # 1. Load
    features = pd.read_parquet(config.FEATURES_PATH_CLEAN)
    cz       = pd.read_parquet(config.CZ_PATH_CLEAN)
    vix      = pd.read_parquet(config.VIX_PATH_CLEAN)

    # 2. Normaliser : date en colonne pour tous les df (homogénéité)
    features = _ensure_date_as_column(features, "date")
    cz       = _ensure_date_as_column(cz, "date")
    vix      = _ensure_date_as_column(vix, "date")

    # 3. Déduplication
    n_before = len(features)
    features = features.drop_duplicates(subset=["date", "PERMNO"], keep="last")
    if verbose:
        print(f"features : {n_before - len(features)} doublons (date, PERMNO) supprimés")

    n_before = len(cz)
    cz = cz.drop_duplicates(subset="date", keep="last")
    if verbose:
        print(f"cz       : {n_before - len(cz)} doublons (date) supprimés")

    n_before = len(vix)
    vix = vix.drop_duplicates(subset="date", keep="last")
    if verbose:
        print(f"vix      : {n_before - len(vix)} doublons (date) supprimés")

    # 4. Merge VIX avec CZ
    df_tmp = pd.merge(cz, vix, on="date", how="inner")

    n_dup = df_tmp.duplicated(subset="date").sum()
    if n_dup > 0:
        if verbose:
            print(f"⚠ df_tmp (cz+vix) : {n_dup} doublons après merge — dédup forcé")
        df_tmp = df_tmp.drop_duplicates(subset="date", keep="last")

    # 5. set_index pour le .join() en aval (date doit être l'index de df_tmp)
    df_tmp = df_tmp.set_index("date")

    if verbose:
        print(f"Number of unique PERMNO in dataset : {features['PERMNO'].nunique()}")
        print(f"Number of unique dates in df_tmp   : {df_tmp.index.nunique()}")

    # 6. Split
    split_batch_dict = split_batch(
        features,
        sample_col="PERMNO",
        date_col="date",
        df_to_join=[df_tmp],
        batch_number=config.BATCH_NUMBER,
        batch_size=config.BATCH_SIZE,
        overlap=False,
        train_split=train_split,
        val_split=val_split,
    )

    # 7. Sanity check post-split
    if verbose:
        for batch_name, splits in split_batch_dict.items():
            for split_name, split_df in splits.items():
                n_dup = split_df.index.duplicated().sum()
                if n_dup > 0:
                    print(f"⚠ {batch_name}/{split_name} : {n_dup} doublons restants")

    # 8. Cleanup
    del features, cz, vix, df_tmp
    gc.collect()

    return split_batch_dict