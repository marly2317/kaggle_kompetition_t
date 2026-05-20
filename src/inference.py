import json

import joblib
import numpy as np

from .data import load_raw_data
from .features import apply_preprocessing_artifacts, build_base_features
from .utils import check_columns, ensure_dir, predict_by_threshold


def create_submission(config, project_root):
    """Build the submission CSV — average fold-model probabilities and apply the
    saved decision threshold."""
    name = config["experiment"]["name"]
    _validate_features_match_fit_time(project_root, config, name)

    _, test, sample = load_raw_data(project_root, config)
    id_col = config["columns"]["id"]
    target = config["columns"]["target"]

    proba = _average_fold_proba(project_root, name, test, config)
    threshold = _load_threshold(project_root / "models" / name)

    submission = sample.copy()
    check_columns(submission, [id_col, target], "sample_submission")
    submission[target] = predict_by_threshold(proba, threshold)

    out_dir = project_root / config["paths"]["submissions_dir"]
    ensure_dir(out_dir)
    out_path = out_dir / f"{name}.csv"
    submission.to_csv(out_path, index=False)
    return out_path


def _validate_features_match_fit_time(project_root, config, name):
    """Raise if the current feature config differs from the one used at fit
    time — guards against scoring with a drifted config."""
    reports_dir = config.get("paths", {}).get("reports_dir")
    if not reports_dir:
        return
    info_path = project_root / reports_dir / name / "experiment_info.json"
    if not info_path.exists():
        return

    info = json.loads(info_path.read_text(encoding="utf-8"))
    fit_features = info.get("config", {}).get("features", {})
    current_features = config.get("features", {})
    if fit_features != current_features:
        raise ValueError(
            f"feature config drift for experiment '{name}':\n"
            f"  fit-time: {fit_features}\n"
            f"  current:  {current_features}\n"
            f"refit the experiment or revert the config."
        )


def _average_fold_proba(project_root, name, test, config):
    """Average positive-class probabilities across all saved fold models."""
    model_dir = project_root / "models" / name
    fold_paths = sorted(
        p for p in model_dir.glob("fold_*.joblib")
        if not p.stem.endswith("_artifacts")
    )
    if not fold_paths:
        raise FileNotFoundError(f"no fold models in {model_dir}")

    base = build_base_features(test, config)
    probas = []
    for path in fold_paths:
        artifacts_path = model_dir / f"{path.stem}_artifacts.joblib"
        model = joblib.load(path)
        artifacts = joblib.load(artifacts_path)
        x = apply_preprocessing_artifacts(base, artifacts, config)
        probas.append(model.predict_proba(x)[:, 1])

    return np.mean(probas, axis=0)


def _load_threshold(model_dir):
    """Load the tuned decision threshold, falling back to 0.5 if absent."""
    path = model_dir / "threshold.json"
    if not path.exists():
        return 0.5
    with path.open(encoding="utf-8") as f:
        return float(json.load(f)["threshold"])
