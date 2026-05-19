"""Detect distribution shift between train and test
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def check_distribution_shift(train_features, test_features, categorical, seed=42):
    train_marked = train_features.assign(_is_test=0)
    test_marked = test_features.assign(_is_test=1)
    combined = pd.concat([train_marked, test_marked], ignore_index=True)

    y = combined.pop("_is_test").to_numpy()
    x = combined.copy()
    if categorical:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        x[categorical] = encoder.fit_transform(x[categorical].astype(str))

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=seed),
    )
    aucs = cross_val_score(model, x, y, scoring="roc_auc", cv=5)
    model.fit(x, y)

    importance = pd.Series(np.abs(model[-1].coef_[0]), index=x.columns)
    mean_auc = float(aucs.mean())

    return {
        "mean_auc": mean_auc,
        "fold_aucs": [float(a) for a in aucs],
        "verdict": _verdict(mean_auc),
        "top_drifted": importance.sort_values(ascending=False).head(5).round(4).to_dict(),
    }


def _verdict(auc):
    if auc < 0.55:
        return "ok"
    if auc < 0.7:
        return "moderate shift"
    return "strong shift"
