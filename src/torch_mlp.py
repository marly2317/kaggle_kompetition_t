"""PyTorch MLP with entity embeddings for the tabular Titanic data.

Wrapped in an sklearn-style estimator (`fit` / `predict_proba`) so it plugs into
the same training path as the other models. Categorical columns are mapped to
integer codes and learned as embeddings; numeric columns are standardized. All
preprocessing state is fit on the training fold only.
"""
import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


def build_mlp(params):
    """Build the MLP classifier — entry point used by model.get_model."""
    return TabularMLPClassifier(**params)


def _embedding_dim(n_categories):
    """Embedding width for a categorical column — small, and capped so a
    high-cardinality column (Surname) cannot dominate the input."""
    return min(32, max(2, (n_categories + 1) // 2))


class _TabularMLP(torch.nn.Module):
    """The network itself — one embedding per categorical column, concatenated
    with the numeric columns, then a stack of Linear/ReLU/Dropout layers."""

    def __init__(self, embedding_sizes, n_numeric, hidden, dropout):
        super().__init__()
        self.embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(n_cat, dim) for n_cat, dim in embedding_sizes]
        )
        input_dim = sum(dim for _, dim in embedding_sizes) + n_numeric
        layers = []
        for width in hidden:
            layers += [
                torch.nn.Linear(input_dim, width),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]
            input_dim = width
        layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        parts = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        parts.append(x_num)
        return self.mlp(torch.cat(parts, dim=1)).squeeze(1)


class TabularMLPClassifier(BaseEstimator):
    """sklearn-style MLP classifier. Index 0 of each embedding is reserved for
    categories unseen at fit time."""

    def __init__(self, hidden=(64, 32), dropout=0.35, epochs=50, batch_size=64,
                 learning_rate=1e-3, weight_decay=1e-4, random_state=42):
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, x, y):
        torch.manual_seed(self.random_state)

        self.cat_columns_ = list(x.select_dtypes(include="object").columns)
        self.num_columns_ = [c for c in x.columns if c not in self.cat_columns_]

        self.cat_maps_ = {}
        for col in self.cat_columns_:
            categories = sorted(x[col].astype(str).unique())
            self.cat_maps_[col] = {value: i + 1 for i, value in enumerate(categories)}

        self.scaler_ = StandardScaler().fit(x[self.num_columns_])

        embedding_sizes = [
            (len(self.cat_maps_[col]) + 1, _embedding_dim(len(self.cat_maps_[col]) + 1))
            for col in self.cat_columns_
        ]
        self.model_ = _TabularMLP(
            embedding_sizes, len(self.num_columns_), list(self.hidden), self.dropout
        )

        x_cat, x_num = self._to_tensors(x)
        y_tensor = torch.tensor(np.asarray(y), dtype=torch.float32)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_cat, x_num, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()

        self.model_.train()
        for _ in range(self.epochs):
            for batch_cat, batch_num, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model_(batch_cat, batch_num), batch_y)
                loss.backward()
                optimizer.step()

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x):
        self.model_.eval()
        x_cat, x_num = self._to_tensors(x)
        with torch.no_grad():
            positive = torch.sigmoid(self.model_(x_cat, x_num)).numpy()
        return np.column_stack([1.0 - positive, positive])

    def _to_tensors(self, x):
        codes = np.zeros((len(x), len(self.cat_columns_)), dtype=np.int64)
        for j, col in enumerate(self.cat_columns_):
            mapped = x[col].astype(str).map(self.cat_maps_[col]).fillna(0)
            codes[:, j] = mapped.to_numpy()
        numeric = self.scaler_.transform(x[self.num_columns_])
        return (
            torch.tensor(codes, dtype=torch.long),
            torch.tensor(numeric, dtype=torch.float32),
        )
