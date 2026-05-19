import argparse
import logging
from pathlib import Path

from .config import load_experiment_config_by_name
from .inference import create_submission
from .train_functions import run
from .tuning import tune
from .utils import setup_logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = "001_best_solution"
LOGGER = logging.getLogger(__name__)


def fit(experiment=DEFAULT_EXPERIMENT):
    config = load_experiment_config_by_name(PROJECT_ROOT, experiment)
    LOGGER.info("[fit] experiment=%s", config["experiment"]["name"])
    metrics = run(config, PROJECT_ROOT)
    _print_metrics(metrics)
    return metrics


def submit(experiment=DEFAULT_EXPERIMENT):
    config = load_experiment_config_by_name(PROJECT_ROOT, experiment)
    path = create_submission(config, PROJECT_ROOT)
    LOGGER.info("[submit] saved to %s", path)
    return path


def _print_metrics(metrics):
    for fold, score in enumerate(metrics.get("fold_scores", []), start=1):
        LOGGER.info("  fold %s: %.5f", fold, score)
    for fold, values in enumerate(metrics.get("fold_metrics", []), start=1):
        LOGGER.info(
            "  fold %s all metrics: %s",
            fold,
            ", ".join(f"{k}={v:.5f}" for k, v in values.items()),
        )
    if "mean_score" in metrics:
        LOGGER.info(
            "  mean: %.5f (std: %.5f)",
            metrics["mean_score"],
            metrics["std_score"],
        )
    if "mean_metrics" in metrics:
        LOGGER.info(
            "  mean all metrics: %s",
            ", ".join(f"{k}={v:.5f}" for k, v in metrics["mean_metrics"].items()),
        )
    if "best_threshold" in metrics:
        LOGGER.info(
            "  best threshold: %.2f (score: %.5f)",
            metrics["best_threshold"],
            metrics["best_threshold_score"],
        )
    if "metrics_at_best_threshold" in metrics:
        LOGGER.info(
            "  metrics@best_threshold: %s",
            ", ".join(
                f"{k}={v:.5f}"
                for k, v in metrics["metrics_at_best_threshold"].items()
            ),
        )


def main():
    setup_logging()
    parser = argparse.ArgumentParser(prog="titanic")
    parser.add_argument("command", choices=["fit", "submit", "all", "tune"])
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    if args.command in ("fit", "all"):
        fit(args.experiment)
    if args.command in ("submit", "all"):
        submit(args.experiment)
    if args.command == "tune":
        tune(PROJECT_ROOT, args.experiment, args.n_trials, args.timeout)


if __name__ == "__main__":
    main()
