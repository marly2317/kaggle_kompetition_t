import logging

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from .config import load_experiment_config_by_name
from .data import load_raw_data
from .features import get_categorical_features
from .train_functions import fit_fold_model, prepare_fold_data
from .utils import collect_run_metadata, save_json, set_seed


LOGGER = logging.getLogger(__name__)


def tune(project_root, experiment, n_trials=30, timeout=900):
    """Run an Optuna search over the config search space and save the best params."""
    config = load_experiment_config_by_name(project_root, experiment)
    set_seed(config["project"]["seed"])

    train, _, _ = load_raw_data(project_root, config)
    target = config["columns"]["target"]
    seed = config["project"]["seed"]
    n_splits = config["validation"]["n_splits"]
    categorical = get_categorical_features(config)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{experiment}_tune",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        lambda trial: _objective(
            trial, train, target, seed, n_splits, config, categorical
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    LOGGER.info("best mean score: %.5f", study.best_value)
    LOGGER.info("best params:")
    for key, value in study.best_params.items():
        LOGGER.info("  %s: %s", key, value)

    out_path = project_root / "reports" / experiment / "optuna.json"
    save_json(collect_run_metadata(config), project_root / "reports" / experiment / "optuna_run_meta.json")
    save_json(
        {
            "best_score": float(study.best_value),
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        },
        out_path,
    )
    LOGGER.info("saved to %s", out_path)
    return study


def _suggest(trial, name, spec):
    """Draw one hyperparameter from a config-defined search-space spec."""
    t = spec["type"]
    if t == "float":
        return trial.suggest_float(name, spec["min"], spec["max"], log=spec.get("log", False))
    if t == "int":
        return trial.suggest_int(name, spec["min"], spec["max"])
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"unknown search_space type for {name}: {t}")


def _objective(trial, train, target, seed, n_splits, config, categorical):
    """One Optuna trial — return the mean CV score for its hyperparameters."""
    trial_settings = config["tuning"]["trial"]
    search_space = config["tuning"]["search_space"]
    model_params = {name: _suggest(trial, name, spec) for name, spec in search_space.items()}
    model_params.update({
        "iterations": int(trial_settings["iterations"]),
        "loss_function": "Logloss",
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
    })

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for idx_train, idx_valid in skf.split(train, train[target]):
        train_part = train.iloc[idx_train]
        valid_part = train.iloc[idx_valid]
        x_train, y_train, x_valid, y_valid, _ = prepare_fold_data(
            train_part, valid_part, target, config
        )
        _, _, _, score, _ = fit_fold_model(
            config,
            x_train,
            y_train,
            x_valid,
            y_valid,
            categorical,
            model_params_override=model_params,
            fit_params_override={
                "early_stopping_rounds": int(trial_settings["early_stopping_rounds"]),
            },
        )
        scores.append(score)

    return float(np.mean(scores))
