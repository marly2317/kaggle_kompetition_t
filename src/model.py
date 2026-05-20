import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_model(config, params_override=None):
    """Build an unfitted model for ``config["model"]["family"]``. ``params_override``
    lets the tuner inject trial parameters over the config defaults."""
    family = config["model"]["family"]
    params = dict(config["model"].get("params", {}))
    if params_override:
        params.update(params_override)
    if family == "catboost":
        return CatBoostClassifier(**params)
    if family == "logreg":
        return _build_logreg(params)
    if family == "lightgbm":
        return _build_lightgbm(params)
    if family == "mlp":
        from .torch_mlp import build_mlp  # lazy: torch only loaded when needed

        return build_mlp(params)
    raise ValueError(f"unsupported model family: {family}")


def _build_logreg(params):
    """Logistic regression baseline. Numeric columns are standardized and
    object (categorical) columns are one-hot encoded; both transformers are
    fit per CV fold inside the pipeline, so no leakage across folds."""
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include=object)),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(**params)),
    ])


class _CategoryCaster(BaseEstimator, TransformerMixin):
    """LightGBM-specific transformer — used only by _build_lightgbm. Casts object
    (string) columns to pandas category dtype so LightGBM handles them natively,
    with no one-hot blow-up on high-cardinality columns. Categories are learned
    on the training fold and reused on valid/test, so category codes stay
    consistent across the split."""

    def fit(self, x, y=None):
        self.categories_ = {
            col: x[col].astype("category").cat.categories
            for col in x.select_dtypes(include="object").columns
        }
        return self

    def transform(self, x):
        x = x.copy()
        for col, categories in self.categories_.items():
            x[col] = pd.Categorical(x[col], categories=categories)
        return x


def _build_lightgbm(params):
    """LightGBM with native categorical handling (see _CategoryCaster)."""
    return Pipeline([
        ("cast", _CategoryCaster()),
        ("model", LGBMClassifier(**params)),
    ])


def get_fit_params(config, fit_params_override=None):
    """Collect fit-time kwargs (CatBoost only — e.g. early stopping)."""
    fit_params = dict(config["model"].get("fit_params", {}))
    if fit_params_override:
        fit_params.update(fit_params_override)
    return fit_params
