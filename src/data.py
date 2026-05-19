import pandas as pd

from .utils import check_columns


def load_raw_data(project_root, config):
    data_dir = project_root / config["paths"]["data_dir"]
    train = pd.read_csv(data_dir / config["paths"]["train_file"])
    test = pd.read_csv(data_dir / config["paths"]["test_file"])
    sample = pd.read_csv(data_dir / config["paths"]["sample_submission_file"])

    id_col = config["columns"]["id"]
    target_col = config["columns"]["target"]
    check_columns(train, [id_col, target_col], "train")
    check_columns(test, [id_col], "test")
    check_columns(sample, [id_col, target_col], "sample_submission")
    _check_unique_id(train, id_col, "train")
    _check_unique_id(test, id_col, "test")
    _check_unique_id(sample, id_col, "sample_submission")
    _check_binary_target(train[target_col], "train")

    return train, test, sample


def _check_unique_id(df, id_col, name):
    if df[id_col].isna().any():
        raise ValueError(f"{name}.{id_col} contains missing values")
    if not df[id_col].is_unique:
        raise ValueError(f"{name}.{id_col} must be unique")


def _check_binary_target(target, name):
    if target.isna().any():
        raise ValueError(f"{name} target contains missing values")
    values = set(target.unique())
    if not values.issubset({0, 1}):
        raise ValueError(f"{name} target must contain only 0/1, got: {sorted(values)}")
