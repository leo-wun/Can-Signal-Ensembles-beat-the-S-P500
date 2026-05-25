"""LSTM return-rank predictor (PyTorch) and its training loop."""

import numpy as np
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """LSTM over a sequence of past daily returns -> next-day return rank."""

    def __init__(self, n_channels: int = 1, hidden: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, num_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (batch, seq_len, n_channels)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)     # last-layer final hidden state


def train_lstm(X_tr, y_tr, X_va, y_va, n_channels=1, *, epochs=25, batch_size=8192,
               lr=1e-3, patience=4, seed=0, device=None):
    """Train the LSTM with Adam + early stopping on validation MSE.

    X_* are (N, seq_len, n_channels) arrays. Returns (best model, best val MSE).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_va, dtype=torch.float32, device=device)
    yva = torch.tensor(y_va, dtype=torch.float32, device=device)

    model = LSTMRegressor(n_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def forward_batched(X, chunk=32768):
        return torch.cat([model(X[i:i + chunk]) for i in range(0, len(X), chunk)])

    n = len(Xtr)
    best, best_state, since = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(forward_batched(Xva), yva).item()
        if val < best - 1e-7:
            best = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best
