import sys, os, gc
sys.path.insert(0, os.path.join('..'))   # project root on path

import random

import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV, Lasso, lasso_path
import lightgbm as lgb
import xgboost as  xgb
from sklearn.metrics import r2_score



def buy_and_hold(
    df: pd.DataFrame = None, 
    initial_capital: float = 1.0,
    start_date=None,
    end_date=None
) -> pd.DataFrame:
    
    """
    
    Get S&P 500 returns and create a benchmark to compare  with a strategy on a specific date range
    
    """
    
    if df is None:
        df = pd.read_parquet(config.GSPC_PATH_CLEAN)
    
        
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    filtered = df.loc[mask, ['date', 'sprtrn']].reset_index(drop=True)
    
    anchor_date = pd.to_datetime(start_date) - pd.tseries.offsets.BDay(1)
    initial_row = pd.DataFrame({'date': [anchor_date], 'sprtrn': [0.0]})
    result = pd.concat([initial_row, filtered], ignore_index=True)
    
    pf = (result['sprtrn'] + 1).cumprod() * initial_capital
    date = result['date']
    
    return pf, date



def get_train_val_test(
        batch_name: str,
        split_batch_dict: dict,
        target_col: str = 'target',
        drop_cols: list[str] | None = None,
        validation_key: str = 'validation',
        return_test: bool = True
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame | None, pd.Series | None]:
    """Return feature/target splits for a single batch.

    Args:
        batch_name: Name of the batch in split_batch_dict.
        split_batch_dict: Dictionary produced by split_batch(...).
        target_col: Target column name.
        drop_cols: Additional columns to drop from features.
        validation_key: Key for the validation split ('validation' or 'val').
        return_test: Whether to return the test split as well.
        signal_transform: If True, converts target values to signal form using np.sign.

    Returns:
        Tuple containing X_train, y_train, X_val, y_val, X_test, y_test.
        X_test and y_test are None when return_test is False.
    """

    if not isinstance(target_col, str):
        raise TypeError("target_col must be a string")

    if batch_name not in split_batch_dict:
        raise KeyError(f"Batch '{batch_name}' not found in split_batch_dict")

    batch = split_batch_dict[batch_name]
    if 'train' not in batch:
        raise KeyError(f"Batch '{batch_name}' does not contain a 'train' split")

    if validation_key not in batch:
        if validation_key == 'validation' and 'val' in batch:
            validation_key = 'val'
        elif validation_key == 'val' and 'validation' in batch:
            validation_key = 'validation'
        else:
            raise KeyError(
                f"Batch '{batch_name}' does not contain a '{validation_key}' split"
            )

    if return_test and 'test' not in batch:
        raise KeyError(f"Batch '{batch_name}' does not contain a 'test' split")

    if drop_cols is None:
        drop_cols = ['ret', 'mkt_ret']

    def _prepare_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in split")

        cols_to_drop = [col for col in [target_col] + drop_cols if col in df.columns]
        X = df.drop(cols_to_drop, axis=1)
        y = df[target_col]
        return X, y

    X_train, y_train = _prepare_split(batch['train'])
    X_val, y_val = _prepare_split(batch[validation_key])
    X_test = y_test = None

    if return_test:
        X_test, y_test = _prepare_split(batch['test'])

    return X_train, y_train, X_val, y_val, X_test, y_test



def deadband_signal_transform(
        df : pd.Series,
        upperband : float = 0.05,
        lowerband : float = -0.05
) -> pd.Series:
    
    df[(df < upperband) & (df > lowerband)] = 0.0

    return df

def random_sampling(
        df: pd.DataFrame,
        sample_number : int = 1,
        seed: int = 42
) -> pd.Series:
    """
    Pour chaque date, tire un PERMNO aléatoire parmi ceux disponibles.

    Args:
        df    : DataFrame avec MultiIndex (date, PERMNO)
        seed  : graine pour la reproductibilité (default=42)

    Returns:
        DataFrame avec MultiIndex (date, PERMNO), un seul PERMNO par date
    """
    sampled = (
        df
        .groupby(level="date")
        .apply(lambda g: g.sample(n=sample_number, random_state=seed))
    )
    return sampled


def basic_lasso(
    split_batch_dict : dict,
    signal_transform : bool,
    target_col : str = 'target'
):
    
    """"

    Unique batch test

    """
    # 1. Loop on each batch
    TARGET = target_col
    
    models = {}
    y_preds = {}
    
    for batch in split_batch_dict:

        X_train, y_train, X_val, y_val, _, _  = get_train_val_test(batch, split_batch_dict, target_col=TARGET)


        if signal_transform:
            y_train = deadband_signal_transform(y_train)
            y_val = deadband_signal_transform(y_val)

    
        # 1.1 Find the best alpha on a sample
        sample_idx = np.random.choice(len(X_train), size=50_000, replace=False)
        cv_model = LassoCV(cv=5, n_jobs=1)
        cv_model.fit(X_train.iloc[sample_idx], y_train.iloc[sample_idx])

        best_alpha = cv_model.alpha_
        print(f"Alpha optimal for {batch} : {best_alpha:.6f}")

        # 1.2. Train the model
        model = Lasso(alpha=best_alpha)
        model.fit(X_train, y_train)
        
        models[batch] = model
        
        y_pred = model.predict(X_val)
        y_preds[batch] = pd.DataFrame(y_pred, columns=['ret_pred'] ,index=y_val.index)

        print(f"R² train for {batch}: {r2_score(y_train, model.predict(X_train)):.4f}")
        print(f"R² val {batch}: {r2_score(y_val, y_pred):.4f}")

    return models, y_preds


def basic_lightGBM(
        split_batch_dict : dict,
        params : dict,
        target_col : str = 'target'
) -> tuple:
    

    TARGET = target_col

    models = {}
    y_preds = {}
    
    for batch in split_batch_dict:
        X_train, y_train, X_val, y_val, _, _  = get_train_val_test(batch, split_batch_dict, target_col=TARGET)
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val)

        model = lgb.LGBMRegressor(params, metric='rmse')
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        y_preds[batch] = pd.DataFrame(y_pred_val, columns=['ret_pred'], index =y_val.index)
        models[batch] = model

        print(f'Model R2 on train : {r2_score(y_train, y_pred_train)}')
        print(f'Model R2 on val : {r2_score(y_val, y_pred_val)}')

    return models, y_preds



def top_bottom(
    split_batch_dict : dict,
    y_preds : dict,
    threshold : float,
    y_true : str
):
    
    top_s = {}
    bottom_s = {}
    
    for batch, y_pred in y_preds.items():
        
        y_val = split_batch_dict[batch][y_true]['target']
        
        percentile_ranks = y_pred.groupby('date')['ret_pred'].transform(lambda x: x.rank(pct=True))

        is_top_ = percentile_ranks >= threshold
        is_bottom_ = percentile_ranks <= (1 - threshold)

        top_indexes = y_pred[is_top_].index
        bottom_indexes = y_pred[is_bottom_].index

        top_val_rows = pd.DataFrame(y_val.loc[top_indexes])
        bottom_val_rows = pd.DataFrame(y_val.loc[bottom_indexes])

        top_ = top_val_rows.groupby('date').mean()
        bottom_ = bottom_val_rows.groupby('date').mean()
        
        top_s[batch] = top_
        bottom_s[batch] = bottom_ 
    
    return top_s, bottom_s


def portfolio_value(
    top_20s: dict,
    bottom_20s: dict,
    N: float,
    LONG_WEIGHT: float,
    SHORT_WEIGHT: float,
    TRADING_COST: float,
    BORROW_COST_ANNUAL: float,
    TRADING_DAYS: int = 252,
) -> tuple[dict, pd.DataFrame]:

    daily_borrow_cost = SHORT_WEIGHT * (BORROW_COST_ANNUAL / TRADING_DAYS)
    daily_trading_cost = (LONG_WEIGHT + SHORT_WEIGHT) * 2 * TRADING_COST

    portfolios = {}
    metrics_rows = []

    for batch in top_20s:
        top_20 = top_20s[batch]
        bottom_20 = bottom_20s[batch]

        daily_returns_pct = (
            LONG_WEIGHT  * (top_20  - TRADING_COST).squeeze()
            - SHORT_WEIGHT * (bottom_20 - TRADING_COST).squeeze()
            - daily_borrow_cost
            - daily_trading_cost
        ).reset_index(drop=True)

        portfolio = pd.concat(
            [pd.Series([N]), N * (1 + daily_returns_pct).cumprod()],
            ignore_index=True
        )
        portfolios[batch] = portfolio

        total_return   = portfolio.iloc[-1] / N - 1
        annualized_ret = (1 + total_return) ** (TRADING_DAYS / len(daily_returns_pct)) - 1
        volatility     = daily_returns_pct.std() * np.sqrt(TRADING_DAYS)
        sharpe         = annualized_ret / volatility
        max_drawdown   = ((portfolio / portfolio.cummax()) - 1).min()

        metrics_rows.append({
            'batch':               batch,
            'total_return':        total_return,
            'annualized_return':   annualized_ret,
            'volatility':          volatility,
            'sharpe_ratio':        sharpe,
            'max_drawdown':        max_drawdown,
        })

    metrics = pd.DataFrame(metrics_rows).set_index('batch')
    return portfolios, metrics



def compute_portfolio_evolution(
    df: pd.DataFrame,
    return_col: str = "ret",
    N: float = 1000,
    long_weight: float = 0.50,
    short_weight: float = 0.10,
    trading_cost: float = 0.001,
    borrow_cost_annual: float = 0.01,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Calcule l'évolution d'un portefeuille long-short à partir d'une série de retours.

    Args:
        df               : DataFrame avec index 'date' et une colonne de retours
        return_col       : nom de la colonne de retours
        N                : capital initial (même pour tous les portfolios à comparer)
        long_weight      : fraction du capital allouée au leg long
        short_weight     : fraction du capital allouée au leg short
        trading_cost     : coût de transaction (en fraction, ex: 0.001 = 10bps)
        borrow_cost_annual: coût d'emprunt annuel sur le short leg
        trading_days     : nombre de jours de trading par an

    Returns:
        DataFrame avec colonnes:
            - portfolio_value   : valeur totale du portfolio
            - gross_return      : retour brut journalier
            - net_return        : retour net (après coûts)
            - cumulative_cost   : coûts cumulés depuis le début
    """
    df = df.copy().sort_index()
    returns = df[return_col].values
    n = len(returns)

    daily_borrow_cost = short_weight * (borrow_cost_annual / trading_days)
    daily_trading_cost = (long_weight + short_weight) * 2 * trading_cost

    portfolio_values = np.empty(n + 1)
    gross_returns    = np.empty(n)
    net_returns      = np.empty(n)
    cumulative_costs = np.empty(n)

    portfolio_values[0] = N
    total_cost = 0.0

    for t in range(n):
        pv = portfolio_values[t]
        r  = returns[t]

        # P&L brut : long gagne r, short perd r
        long_pnl  =  long_weight  * r * pv
        short_pnl = -short_weight * r * pv

        gross_pnl = long_pnl + short_pnl
        gross_ret = gross_pnl / pv

        # Coûts de transaction (appliqués chaque jour, proxy du turnover quotidien)
        tc_cost = daily_trading_cost * pv

        # Coût d'emprunt quotidien sur la jambe short
        borrow_cost = daily_borrow_cost * pv

        total_daily_cost = tc_cost + borrow_cost
        net_pnl          = gross_pnl - total_daily_cost
        net_ret          = net_pnl / pv

        total_cost += total_daily_cost

        portfolio_values[t + 1] = pv + net_pnl
        gross_returns[t]        = gross_ret
        net_returns[t]          = net_ret
        cumulative_costs[t]     = total_cost

    result = pd.DataFrame(
        {
            "portfolio_value" : portfolio_values[1:],
            "gross_return"    : gross_returns,
            "net_return"      : net_returns,
            "cumulative_cost" : cumulative_costs,
        },
        index=df.index,
    )

    return result