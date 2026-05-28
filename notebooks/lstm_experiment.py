"""
LSTM / recurrent-network meta-model.

Completes the ML model comparison with a sequence model. Instead of consuming a
single cross-section of engineered features, the LSTM reads the raw sequence of
each stock's last 20 daily returns and predicts the cross-sectional rank of the
next-day return ; it must learn reversal / momentum / volatility structure
itself from the return path. Backtested as a weekly decile long-short on the
full and the tradable top-1000 universe, exactly like the MLP / XGBoost run.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.backtest import CostModel, held_weights, run_backtest, performance_metrics
from src.models.lstm_model import train_lstm

SEQ_LEN = 20
TRAIN_SAMPLE = 200_000
VAL_SAMPLE = 200_000
TRAIN_END = pd.Timestamp(config.TRAIN_END)
VAL_END = pd.Timestamp(config.VAL_END)


def main(
    ) -> None:
    
    df = pd.read_parquet(config.FEATURES_PATH_CLEAN,
                         columns=["date", "PERMNO", "ret", "target"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "PERMNO"], keep="last")
    df = df.dropna(subset=["ret", "target"])
    df = df.sort_values(["PERMNO", "date"])

    # L-day return window via shifted columns (within PERMNO, chronological)
    g = df.groupby("PERMNO")["ret"]
    lag_cols = [f"r{k}" for k in range(1, SEQ_LEN + 1)]
    for k, c in enumerate(lag_cols, start=1):
        df[c] = g.shift(k).astype("float32")
    df = df.dropna(subset=lag_cols).copy()                # need a full window

    df["split"] = np.where(df["date"] <= TRAIN_END, "train",
                           np.where(df["date"] <= VAL_END, "val", "test"))
    df["yrank"] = df.groupby("date")["target"].rank(pct=True).astype("float32")

    seq_cols = [f"r{k}" for k in range(SEQ_LEN, 0, -1)]   # oldest -> newest

    def make_X(
        frame
        ):
        
        return frame[seq_cols].to_numpy("float32")[:, :, None]   # (N, L, 1)

    train = df[df.split == "train"]
    val = df[df.split == "val"]
    test = df[df.split == "test"].copy()
    tr = train.sample(n=min(TRAIN_SAMPLE, len(train)), random_state=0)
    va = val.sample(n=min(VAL_SAMPLE, len(val)), random_state=0)

    std = tr[seq_cols].to_numpy().std() + 1e-8            # scale by train return std
    Xtr, Xva, Xte = make_X(tr) / std, make_X(va) / std, make_X(test) / std
    ytr = tr["yrank"].to_numpy("float32")
    yva = va["yrank"].to_numpy("float32")
    print(f"sequences: train {len(Xtr):,} | val {len(Xva):,} | test {len(Xte):,}")

    model, val_mse = train_lstm(Xtr, ytr, Xva, yva, n_channels=1)
    print(f"LSTM best validation MSE: {val_mse:.5f}")

    model.eval()
    dev = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 32768): # 2^15 most effective batch size 
            preds.append(model(torch.tensor(Xte[i:i + 32768], device=dev)).cpu().numpy())
    test["pred_lstm"] = np.concatenate(preds)

    # liquidity universe (lagged market-cap rank)
    liq = pd.read_parquet(config.CRSP_LIQUIDITY_RAW, columns=["date", "PERMNO", "dlycap"])
    liq["date"] = pd.to_datetime(liq["date"])
    liq["dlycap"] = liq["dlycap"].astype("float64")
    liq = liq.sort_values(["PERMNO", "date"])
    liq["cap_lag"] = liq.groupby("PERMNO")["dlycap"].shift(1)
    liq = liq[liq["date"] > VAL_END]
    liq["cap_rank"] = liq.groupby("date")["cap_lag"].rank(ascending=False, method="first")
    test = test.merge(liq[["date", "PERMNO", "cap_rank"]], on=["date", "PERMNO"], how="left")
    test["cap_rank"] = test["cap_rank"].astype("float64")

    test_idx = test.set_index(["date", "PERMNO"]).sort_index()
    returns = test_idx["ret"]
    dates = returns.index.get_level_values("date").unique().sort_values()
    rebal = dates[::5]
    cost = CostModel(0.001, 0.01)

    def mean_ic(
        frame
        ):
        
        return frame.groupby("date").apply(
            lambda gg: spearmanr(gg["pred_lstm"], gg["target"])[0],
            include_groups=False).mean()

    # Get SP500 data as benchmark
    gspc = pd.read_parquet(config.GSPC_PATH_CLEAN)
    gspc["date"] = pd.to_datetime(gspc["date"])
    spx = gspc.set_index("date")["sprtrn"].reindex(dates).fillna(0.0)
    
    
    print(f"\n LSTM meta-model (weekly L/S decile, test 2019-2024) ")
    print(f"S&P500 reference Sharpe: {performance_metrics(spx)['sharpe']:+.2f}\n")
    
    
    for uni, N in [("full", None), ("top 1000", 1000)]:
        frame = test if N is None else test[test["cap_rank"] <= N]
        idx = test_idx if N is None else test_idx[test_idx["cap_rank"] <= N]
        w = held_weights(idx["pred_lstm"], rebal, quantile=0.10)
        bt = run_backtest(w, returns, cost)
        m = performance_metrics(bt["net_return"])
        gs = performance_metrics(bt["gross_return"])["sharpe"]
        
        print(f"  {uni:9s} | test IC {mean_ic(frame):+.4f} | gross Sharpe {gs:+.2f} "
              f"| net Sharpe {m['sharpe']:+.2f} | ann ret {m['ann_return']:+.2%}")


if __name__ == "__main__":
    main()
