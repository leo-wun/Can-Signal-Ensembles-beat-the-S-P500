"""Small MLP return-rank predictor (PyTorch) and its training loop."""

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Feed-forward net predicting the cross-sectional return rank from features."""

    def __init__(self, n_features: int, hidden=(64, 32), dropout: float = 0.10):
        super().__init__()
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_va, y_va, n_features, *, epochs=60, batch_size=65536,
              lr=1e-3, patience=6, seed=0, device=None):
    """Train an MLP with Adam + early stopping on validation MSE.

    X_*/y_* are numpy arrays. Returns (best model, best validation MSE).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_va, dtype=torch.float32, device=device)
    yva = torch.tensor(y_va, dtype=torch.float32, device=device)

    model = MLP(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = len(Xtr)
    best_val, best_state, since = float("inf"), None, 0
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
            val = loss_fn(model(Xva), yva).item()
        if val < best_val - 1e-7:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val
