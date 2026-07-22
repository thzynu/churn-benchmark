"""
Deep tabular-learning models: MLP, TabNet, and a simplified TabR.

Reviewer note on naming (Reviewer 3): the earlier manuscript draft used
"RealTabR" inconsistently with the cited reference (Gorishniy et al., 2023,
"TabR: Unlocking the Power of Retrieval-Augmented Tabular Deep Learning").
This module standardises the name to **TabR** and documents clearly that
the implementation here is a *lightweight, from-scratch* re-implementation
of TabR's core idea (an encoder + attention-based retrieval over a context
set of training examples), NOT the authors' original repository -- there is
no official pip package for TabR at the time of writing. This should be
stated explicitly in the revised Methods section.

All three models share a common sklearn-like interface:
    model.fit(X_train, y_train, X_val, y_val)
    model.predict_proba(X)  -> ndarray of shape (n, 2)
    model.predict(X)        -> ndarray of shape (n,)
    model.count_parameters() -> int
    model.name               -> str
so they can be evaluated with exactly the same code path as the classical
baselines in evaluate.py.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_tensor(X, y=None):
    X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
    if y is None:
        return X_t
    y_t = torch.tensor(np.asarray(y), dtype=torch.float32)
    return X_t, y_t


# ---------------------------------------------------------------------------
# 1. MLP  (mirrors the architecture described in the original manuscript:
#    fully-connected layers + Dropout + BatchNorm, trained with Adam and
#    binary cross-entropy, early stopping on validation AUC)
# ---------------------------------------------------------------------------
class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout, batch_norm):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPClassifier:
    name = "MLP"

    def __init__(self, input_dim, params=None, random_state=42, device="cpu"):
        params = params or config.MLP_PARAMS
        self.params = params
        self.device = device
        self.random_state = random_state
        set_seed(random_state)
        self.model = MLPNet(
            input_dim, params["hidden_layers"], params["dropout"], params["batch_norm"]
        ).to(device)

    def fit(self, X_train, y_train, X_val, y_val):
        p = self.params
        opt = torch.optim.Adam(self.model.parameters(), lr=p["learning_rate"])
        criterion = nn.BCEWithLogitsLoss()

        Xtr, ytr = _to_tensor(X_train, y_train)
        Xva, yva = _to_tensor(X_val, y_val)
        train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=p["batch_size"], shuffle=True)

        best_val_loss = np.inf
        best_state = None
        patience_left = p["patience"]
        history = []

        for epoch in range(p["max_epochs"]):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(Xva.to(self.device))
                val_loss = criterion(val_logits, yva.to(self.device)).item()
            history.append(val_loss)

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_left = p["patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.n_epochs_trained = epoch + 1
        return history

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = _to_tensor(X).to(self.device)
            logits = self.model(X_t)
            p1 = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - p1, p1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 2. TabNet  (thin wrapper around pytorch-tabnet so it exposes the same
#    interface as the other two models)
# ---------------------------------------------------------------------------
class TabNetClassifierWrapper:
    name = "TabNet"

    def __init__(self, input_dim=None, params=None, random_state=42, device="cpu"):
        from pytorch_tabnet.tab_model import TabNetClassifier
        params = params or config.TABNET_PARAMS
        self.params = params
        self.random_state = random_state
        self.model = TabNetClassifier(
            n_d=params["n_d"], n_a=params["n_a"], n_steps=params["n_steps"],
            gamma=params["gamma"], lambda_sparse=params["lambda_sparse"],
            seed=random_state, verbose=0,
            device_name=device,
        )

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            np.asarray(X_train), np.asarray(y_train),
            eval_set=[(np.asarray(X_val), np.asarray(y_val))],
            eval_metric=["auc"],
            max_epochs=self.params["max_epochs"],
            patience=self.params["patience"],
            batch_size=self.params["batch_size"],
        )

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.network.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 3. TabR (simplified, from-scratch re-implementation)
#
#    Core idea (Gorishniy et al. 2023): encode each sample to an embedding,
#    retrieve the k most similar samples from a fixed "context bank" drawn
#    from the training set, aggregate their (embedding, label) pairs with an
#    attention mechanism, and combine that retrieved signal with a direct
#    MLP prediction through a learned gate. This is a lightweight
#    approximation intended for benchmarking purposes, not a reproduction of
#    the authors' official implementation.
# ---------------------------------------------------------------------------
class TabRNet(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
        )
        # value embedding for retrieved labels (0/1 -> learned vector)
        self.label_embed = nn.Embedding(2, embed_dim)
        self.gate = nn.Linear(embed_dim * 2, 1)
        self.head = nn.Linear(embed_dim, 1)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x, context_emb, context_labels, k):
        z = self.encode(x)                                   # (batch, embed_dim)
        # similarity between query and every context embedding
        sims = z @ context_emb.T                              # (batch, n_context)
        topk_sims, topk_idx = torch.topk(sims, k=min(k, context_emb.shape[0]), dim=1)
        weights = F.softmax(topk_sims, dim=1)                 # (batch, k)

        retrieved_labels = context_labels[topk_idx]           # (batch, k)
        retrieved_value = self.label_embed(retrieved_labels)  # (batch, k, embed_dim)
        retrieved = (weights.unsqueeze(-1) * retrieved_value).sum(dim=1)  # (batch, embed_dim)

        gate_input = torch.cat([z, retrieved], dim=1)
        gate = torch.sigmoid(self.gate(gate_input))           # (batch, 1)
        combined = gate * retrieved + (1 - gate) * z

        logit = self.head(combined).squeeze(-1)
        return logit


class TabRClassifier:
    name = "TabR"

    def __init__(self, input_dim, params=None, random_state=42, device="cpu",
                 context_size=1024):
        params = params or config.TABR_PARAMS
        self.params = params
        self.device = device
        self.random_state = random_state
        self.context_size = context_size
        set_seed(random_state)
        self.model = TabRNet(input_dim, params["embed_dim"], params["dropout"]).to(device)
        self._context_X = None
        self._context_y = None

    def _refresh_context(self, X_train, y_train):
        """Fixed random subsample of the training set used as the retrieval
        context bank (kept constant during training for stability)."""
        n = len(X_train)
        size = min(self.context_size, n)
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(n, size=size, replace=False)
        self._context_X = torch.tensor(np.asarray(X_train)[idx], dtype=torch.float32).to(self.device)
        self._context_y = torch.tensor(np.asarray(y_train)[idx], dtype=torch.long).to(self.device)

    def fit(self, X_train, y_train, X_val, y_val):
        p = self.params
        self._refresh_context(X_train, y_train)
        opt = torch.optim.Adam(self.model.parameters(), lr=p["learning_rate"])
        criterion = nn.BCEWithLogitsLoss()

        Xtr, ytr = _to_tensor(X_train, y_train)
        Xva, yva = _to_tensor(X_val, y_val)
        train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=p["batch_size"], shuffle=True)

        best_val_loss = np.inf
        best_state = None
        patience_left = p["patience"]

        for epoch in range(p["max_epochs"]):
            self.model.train()
            context_emb = self.model.encode(self._context_X).detach()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb, context_emb, self._context_y, k=p["n_neighbors"])
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                context_emb = self.model.encode(self._context_X)
                val_logits = self.model(Xva.to(self.device), context_emb, self._context_y, k=p["n_neighbors"])
                val_loss = criterion(val_logits, yva.to(self.device)).item()

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_left = p["patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.n_epochs_trained = epoch + 1

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            context_emb = self.model.encode(self._context_X)
            X_t = _to_tensor(X).to(self.device)
            logits = self.model(X_t, context_emb, self._context_y, k=self.params["n_neighbors"])
            p1 = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - p1, p1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def get_deep_model(name, input_dim, random_state=42, device="cpu"):
    if name == "MLP":
        return MLPClassifier(input_dim, random_state=random_state, device=device)
    elif name == "TabNet":
        return TabNetClassifierWrapper(input_dim, random_state=random_state, device=device)
    elif name == "TabR":
        return TabRClassifier(input_dim, random_state=random_state, device=device)
    else:
        raise ValueError(f"Unknown deep model: {name}")
