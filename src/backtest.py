"""
Minimal backtest engine for cross-sectional long-short strategies.

A strategy produces a per-stock master signal; this module turns it into
dollar-neutral weights, simulates the daily P&L net of trading and borrow
costs, and reports standard performance metrics. Deliberately simple: daily
rebalancing, flat transaction cost on turnover, no slippage model.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


class CostModel:
    """Flat transaction cost on turnover + borrow cost on the short leg.

    `periods_per_year` lets the same cost model be used at daily (252) or
    monthly (12) rebalancing frequency.
    """

    def __init__(self, trading_cost: float = 0.001, borrow_cost_annual: float = 0.01,
                 periods_per_year: int = TRADING_DAYS):
        self.trading_cost = trading_cost
        self.borrow_cost_period = borrow_cost_annual / periods_per_year


def long_short_weights(signal: pd.Series, quantile: float = 0.10,
                       leg_weight: float = 1.0) -> pd.Series:
    """Dollar-neutral long-short weights from a master signal.

    signal : Series indexed by (date, PERMNO). Long the top `quantile`, short
    the bottom `quantile`; each leg equal-weighted to +/- leg_weight.
    """
    def _weights(s: pd.Series) -> pd.Series:
        ranks = s.rank(pct=True)
        w = pd.Series(0.0, index=s.index)
        longs, shorts = ranks >= 1.0 - quantile, ranks <= quantile
        if longs.any():
            w[longs] = leg_weight / longs.sum()
        if shorts.any():
            w[shorts] = -leg_weight / shorts.sum()
        return w

    return signal.groupby(level="date", group_keys=False).apply(_weights)


def long_only_weights(signal: pd.Series, quantile: float = 0.20) -> pd.Series:
    """Long-only equal weights on the top `quantile` of `signal`, summing to 1
    on each date. quantile=1.0 gives an equal-weight portfolio of the whole
    universe."""
    def _weights(s: pd.Series) -> pd.Series:
        ranks = s.rank(pct=True)
        w = pd.Series(0.0, index=s.index)
        sel = ranks >= 1.0 - quantile
        if sel.any():
            w[sel] = 1.0 / sel.sum()
        return w

    return signal.groupby(level="date", group_keys=False).apply(_weights)


def held_weights(signal: pd.Series, rebal_dates, quantile: float = 0.10,
                 leg_weight: float = 1.0, weight_fn=None) -> pd.DataFrame:
    """Weights formed on `rebal_dates` and held constant until the next one.

    signal : Series indexed by (date, PERMNO), defined on every date.
    rebal_dates : the dates on which the portfolio is actually rebuilt.
    weight_fn : optional callable(signal) -> weights; defaults to
        long_short_weights with the given quantile / leg_weight.
    Returns a (date x PERMNO) weight panel covering every date in `signal`.
    """
    dates = signal.index.get_level_values("date").unique().sort_values()
    on_rebal = signal.index.get_level_values("date").isin(rebal_dates)
    sub = signal[on_rebal]
    if weight_fn is None:
        w = long_short_weights(sub, quantile=quantile, leg_weight=leg_weight)
    else:
        w = weight_fn(sub)
    return w.unstack(fill_value=0.0).reindex(dates).ffill().fillna(0.0)


def run_backtest(weights, returns: pd.Series, cost_model: CostModel,
                 exposure: pd.Series | None = None) -> pd.DataFrame:
    """Simulate daily P&L.

    weights : Series indexed by (date, PERMNO), or a pre-built date x PERMNO panel.
    returns : Series indexed by (date, PERMNO).
    exposure : optional per-date gross-exposure multiplier (e.g. a vol overlay).
    Returns a DataFrame indexed by date: gross_return, net_return, cost, equity.
    """
    if isinstance(weights, pd.Series):
        w_panel = weights.unstack(fill_value=0.0)
    else:
        w_panel = weights
    w_panel = w_panel.sort_index()
    r_panel = (returns.unstack(fill_value=0.0)
               .reindex(index=w_panel.index, columns=w_panel.columns, fill_value=0.0))
    dates = w_panel.index

    if exposure is None:
        exp = np.ones(len(dates))
    else:
        exp = exposure.reindex(dates).ffill().fillna(1.0).to_numpy()

    w_mat = w_panel.to_numpy() * exp[:, None]
    r_mat = r_panel.to_numpy()

    gross = (w_mat * r_mat).sum(axis=1)
    turnover = np.abs(np.diff(w_mat, axis=0,
                              prepend=np.zeros((1, w_mat.shape[1])))).sum(axis=1)
    short_exposure = np.abs(np.minimum(w_mat, 0.0)).sum(axis=1)
    cost = (cost_model.trading_cost * turnover
            + cost_model.borrow_cost_period * short_exposure)
    net = gross - cost

    out = pd.DataFrame({"gross_return": gross, "net_return": net, "cost": cost},
                       index=dates)
    out["equity"] = (1.0 + out["net_return"]).cumprod()
    return out


def performance_metrics(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Standard performance stats for a periodic return series (no risk-free
    rate). `periods_per_year` should be 252 for daily, 12 for monthly."""
    returns = returns.dropna()
    n = len(returns)
    total_return = (1.0 + returns).prod() - 1.0
    ann_return = (1.0 + total_return) ** (periods_per_year / n) - 1.0
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    equity = (1.0 + returns).cumprod()
    max_drawdown = (equity / equity.cummax() - 1.0).min()
    return {"total_return": total_return, "ann_return": ann_return,
            "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": max_drawdown}
