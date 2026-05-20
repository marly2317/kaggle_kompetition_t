import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .adversarial_validation import check_distribution_shift
from .data import load_raw_data
from .features import (
    apply_preprocessing_artifacts,
    build_base_features,
    fit_preprocessing_artifacts,
    get_categorical_features,
    get_features,
)
from .metrics import compute_all_metrics, get_metric
from .model import get_fit_params, get_model
from .utils import (
    collect_run_metadata,
    ensure_dir,
    predict_by_threshold,
    save_json,
    set_seed,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class FoldContext:
    """Shared, read-only context passed to each fold of a training run."""
    config: dict
    model_dir: Path
    report_dir: Path
    categorical: list[str]


def run(config, project_root):
    """Full training run — stratified K-fold fit, OOF predictions, threshold
    search, and saved models/reports. Returns the metrics dict."""
    set_seed(config["project"]["seed"])
    train, test, _ = load_raw_data(project_root, config)

    target = config["columns"]["target"]
    id_col = config["columns"]["id"]
    name = config["experiment"]["name"]
    seed = config["project"]["seed"]

    model_dir = project_root / "models" / name
    report_dir = project_root / config["paths"]["reports_dir"] / name
    ensure_dir(model_dir)
    ensure_dir(report_dir)

    features = get_features(config)
    categorical = get_categorical_features(config)

    _run_shift_check(train, test, target, categorical, report_dir, config, seed)

    ctx = FoldContext(
        config=config,
        model_dir=model_dir,
        report_dir=report_dir,
        categorical=categorical,
    )
    splits = list(_make_splits(train, train[target], config))
    fold_results = [
        _train_fold(train, idx_train, idx_valid, fold, ctx)
        for fold, (idx_train, idx_valid) in enumerate(splits)
    ]

    oof = pd.concat([r["preds"] for r in fold_results], ignore_index=True)
    best_threshold, best_score = _best_threshold(
        config, oof[target], oof["probability"]
    )
    oof["prediction"] = predict_by_threshold(oof["probability"], best_threshold)
    oof.sort_values(id_col).to_csv(report_dir / "oof_predictions.csv", index=False)

    fold_scores = [r["score"] for r in fold_results]
    fold_metrics = [r["metrics"] for r in fold_results]
    mean_metrics = {
        metric_name: float(np.mean([fold[metric_name] for fold in fold_metrics]))
        for metric_name in fold_metrics[0]
    }
    threshold_metrics = compute_all_metrics(
        oof[target], oof["prediction"], oof["probability"]
    )
    metrics = {
        "fold_scores": fold_scores,
        "mean_score": float(np.mean(fold_scores)),
        "std_score": float(np.std(fold_scores)),
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "best_threshold": best_threshold,
        "best_threshold_score": best_score,
        "metrics_at_best_threshold": threshold_metrics,
    }
    best_iters = [r["best_iteration"] for r in fold_results]
    if any(i is not None for i in best_iters):
        metrics["best_iterations"] = best_iters

    save_json({"threshold": best_threshold}, model_dir / "threshold.json")
    _save_importance(
        [r["importance"] for r in fold_results if r["importance"] is not None],
        report_dir,
    )
    save_json(
        {"config": config, "features": features, "categorical": categorical},
        report_dir / "experiment_info.json",
    )
    save_json(metrics, report_dir / "metrics.json")
    save_json(collect_run_metadata(config), report_dir / "run_meta.json")

    return metrics


def _run_shift_check(train, test, target, categorical, report_dir, config, seed):
    """Adversarial train/test shift report. Deliberately fits preprocessing on
    the full train — this is a diagnostic only and never feeds the model."""
    base_train = build_base_features(train, config)
    base_test = build_base_features(test, config)
    y = train[target]
    artifacts = fit_preprocessing_artifacts(base_train, y, config)
    full_train = apply_preprocessing_artifacts(base_train, artifacts, config, target=y)
    full_test = apply_preprocessing_artifacts(base_test, artifacts, config)

    report = check_distribution_shift(full_train, full_test, categorical, seed)
    save_json(report, report_dir / "adversarial_validation.json")
    LOGGER.info(
        "[adversarial] AUC=%.3f (%s)",
        report["mean_auc"],
        report["verdict"],
    )


def prepare_fold_data(train_part, valid_part, target_col, config):
    """Build features for one fold — preprocessing artifacts are fit on the
    train part only and applied to both parts."""
    base_train = build_base_features(train_part, config)
    base_valid = build_base_features(valid_part, config)
    y_train = train_part[target_col]
    y_valid = valid_part[target_col]
    artifacts = fit_preprocessing_artifacts(base_train, y_train, config)

    x_train = apply_preprocessing_artifacts(
        base_train, artifacts, config, target=y_train
    )
    x_valid = apply_preprocessing_artifacts(base_valid, artifacts, config)

    return x_train, y_train, x_valid, y_valid, artifacts


def fit_fold_model(
    config,
    x_train,
    y_train,
    x_valid,
    y_valid,
    categorical,
    model_params_override=None,
    fit_params_override=None,
):
    """Fit one model on a fold. CatBoost uses cat_features + eval_set; other
    families take the plain sklearn fit. Returns model, probs, preds, score, metrics."""
    model = get_model(config, params_override=model_params_override)

    if config["model"]["family"] == "catboost":
        model.fit(
            x_train,
            y_train,
            cat_features=categorical,
            eval_set=(x_valid, y_valid),
            **get_fit_params(config, fit_params_override),
        )
    else:
        model.fit(x_train, y_train)

    proba = model.predict_proba(x_valid)[:, 1]
    preds = predict_by_threshold(proba, 0.5)
    score = get_metric(config, y_valid, preds, y_score=proba)
    metrics = compute_all_metrics(y_valid, preds, proba)

    return model, proba, preds, float(score), metrics


def _make_splits(train, y, config):
    """Yield stratified K-fold (train_idx, valid_idx) index pairs."""
    strategy = config["validation"]["strategy"]
    if strategy != "kfold":
        raise ValueError(f"unsupported validation strategy: {strategy}")
    skf = StratifiedKFold(
        n_splits=config["validation"]["n_splits"],
        shuffle=True,
        random_state=config["project"]["seed"],
    )
    yield from skf.split(train, y)


def _train_fold(train, idx_train, idx_valid, fold, ctx):
    """Train one fold, persist the model and artifacts, and return its results dict."""
    config = ctx.config
    target = config["columns"]["target"]
    id_col = config["columns"]["id"]
    is_catboost = config["model"]["family"] == "catboost"

    train_part = train.iloc[idx_train]
    valid_part = train.iloc[idx_valid]
    x_train, y_train, x_valid, y_valid, artifacts = prepare_fold_data(
        train_part, valid_part, target, config
    )
    model, proba, preds, score, metrics = fit_fold_model(
        config, x_train, y_train, x_valid, y_valid, ctx.categorical
    )

    joblib.dump(model, ctx.model_dir / f"fold_{fold}.joblib")
    joblib.dump(artifacts, ctx.model_dir / f"fold_{fold}_artifacts.joblib")

    # eval history, best iteration and gain importance are CatBoost-only.
    best_iter = None
    importance = None
    if is_catboost:
        _save_eval_history(model.get_evals_result(), fold, ctx.report_dir)
        best_iter = model.get_best_iteration()
        importance = pd.DataFrame({
            "fold": fold,
            "feature": list(x_train.columns),
            "importance": model.get_feature_importance(),
        })

    return {
        "fold": fold,
        "score": float(score),
        "metrics": metrics,
        "best_iteration": int(best_iter) if best_iter is not None else None,
        "preds": pd.DataFrame({
            id_col: valid_part[id_col].to_numpy(),
            target: y_valid.to_numpy(),
            "fold": fold,
            "probability": proba,
            "prediction": preds,
        }),
        "importance": importance,
    }


def _save_eval_history(evals_result, fold, report_dir):
    """Write one fold's per-iteration CatBoost eval history to CSV."""
    rows = []
    for dataset, metrics in evals_result.items():
        for metric, values in metrics.items():
            for iteration, value in enumerate(values):
                rows.append({
                    "fold": fold,
                    "iteration": iteration,
                    "dataset": dataset,
                    "metric": metric,
                    "value": value,
                })

    if rows:
        pd.DataFrame(rows).to_csv(
            report_dir / f"eval_history_fold_{fold}.csv", index=False
        )


def _best_threshold(config, y_true, proba):
    """Search the probability threshold that maximizes the metric on OOF
    predictions. Ties break toward 0.5 to avoid an over-fit extreme threshold."""
    t = config["tuning"]["threshold"]
    grid = np.round(np.arange(t["min"], t["max"] + t["step"] / 2, t["step"]), 4)

    best_t, best_score = 0.5, -np.inf
    for threshold in grid:
        score = get_metric(
            config,
            y_true,
            predict_by_threshold(proba, threshold),
            y_score=proba,
        )
        if score > best_score or (
            np.isclose(score, best_score)
            and abs(float(threshold) - 0.5) < abs(best_t - 0.5)
        ):
            best_t, best_score = float(threshold), float(score)
    return best_t, best_score


def _save_importance(frames, report_dir):
    """Aggregate per-fold feature importance into per-fold and summary CSVs."""
    if not frames:
        return
    by_fold = pd.concat(frames, ignore_index=True)
    by_fold.to_csv(report_dir / "feature_importance_by_fold.csv", index=False)

    summary = (
        by_fold.groupby("feature")["importance"]
        .agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "mean_importance", "std": "std_importance"})
        .fillna({"std_importance": 0.0})
        .sort_values("mean_importance", ascending=False)
    )
    summary.to_csv(report_dir / "feature_importance.csv", index=False)
