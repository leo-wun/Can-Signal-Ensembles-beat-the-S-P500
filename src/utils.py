
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


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

def train_val_test_split(
        df : pd.DataFrame,
        date_col : str,
        START_DATE = config.START_DATE,
        END_DATE = config.END_DATE,
        TRAIN_END = config.TRAIN_END,
        VAL_END = config.VAL_END
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    """
    Splits a financial dataframe chronologically by unique dates to prevent data leakage.

    Input ->

    df:         Dataframe containing at least date_col
    date_col:   Name of the column containing the dates on which we do the split
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """

    df = df.copy().sort_values(date_col)

    train_df = df[df[date_col].between(START_DATE, TRAIN_END, inclusive='left')] # Inclusice left acts as [START_DATE : END_DATE)
    val_df   = df[df[date_col].between(TRAIN_END, VAL_END, inclusive='left')]
    test_df  = df[df[date_col].between(VAL_END, END_DATE, inclusive='both')] # Inclusice both acts as [START_DATE : END_DATE]

    train_df = train_df.set_index([date_col, 'PERMNO'])
    val_df = val_df.set_index([date_col, 'PERMNO'])
    test_df = test_df.set_index([date_col, 'PERMNO'])

    return (train_df, val_df, test_df)



# ── Pipeline from .parquet to train/validation/test split batches ──────────────────────────────────────────────────────


def split_batch(df : pd.DataFrame,
                sample_col : str, 
                date_col : str,
                batch_number : int,
                batch_size : int,
                overlap : bool = True
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

    Output ->

    dict:           Dictionnary of dictionnary of structure : {batch_1 : {train, val, test}, batch_2 : {train, val, test}}
        
    """

    batch_dict = generate_batches(df, sample_col, batch_number, batch_size, overlap)

    split_batch_dict = {}

    for batch_name, batch_df in batch_dict.items():
        # Call the split function
        train_df, val_df, test_df = train_val_test_split(batch_df, date_col)

        # Store the tuple
        split_batch_dict[batch_name] = {
            'train' : train_df,
            'validation' : val_df,
            'test' : test_df
        }


    return split_batch_dict
