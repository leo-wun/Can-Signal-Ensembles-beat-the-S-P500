
import pandas as pd
import numpy as np
import sys
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_squared_error


sys.path.insert(0, str(Path(__file__).parent.parent))
import config




def train_lgb(
        X_train,
        y_train,
        X_val, 
        y_val,
        params : dict = None,
):
    """
    Training logic for the lightGBM model. 


    Input -> 

    X_train:        Features from training dataset
    y_train:        Target from training dataset
    X_val:          Features from evaluation dataset
    y_val:          Target from evaluation dataset
    params:         Model parameters


    Output ->

    lgb_model:      LightGBM model trained on (X_train, y_train)
    val_metric:     
    
    """

    # Format data
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        params,
        train_data, 
        valid_sets=[val_data],
        num_boost_round=100
    )

    

    return model
    


