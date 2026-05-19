from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


_METRICS = {
    "accuracy": accuracy_score,
    "roc_auc": roc_auc_score,
    "recall": recall_score,
    "precision": precision_score,
    "f1": f1_score,
}


def compute_all_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def get_metric(config, y_true, y_pred, y_score=None):
    name = config["metric"]["name"]
    if name not in _METRICS:
        raise ValueError(f"unknown metric: {name} (available: {sorted(_METRICS)})")
    if name in {"recall", "precision", "f1"}:
        return float(_METRICS[name](y_true, y_pred, zero_division=0))
    if name == "roc_auc":
        values = y_score if y_score is not None else y_pred
        return float(_METRICS[name](y_true, values))
    return float(_METRICS[name](y_true, y_pred))
