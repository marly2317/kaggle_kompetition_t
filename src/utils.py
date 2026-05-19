import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def check_columns(df, required, name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"missing columns in {name}: {missing}")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_to_jsonable)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def collect_run_metadata(config):
    dependencies = {}
    for package in ("pandas", "numpy", "scikit-learn", "catboost", "optuna"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "experiment": config["experiment"]["name"],
        "seed": config["project"]["seed"],
        "dependencies": dependencies,
    }


def predict_by_threshold(probabilities, threshold):
    return (np.asarray(probabilities) >= threshold).astype(int)


def _to_jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"can't serialize {type(value).__name__}")
