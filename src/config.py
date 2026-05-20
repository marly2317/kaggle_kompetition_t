import yaml


def load_config(path):
    """Load a single YAML file into a dict."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(project_path, experiment_path):
    """Merge an experiment config (path or already-loaded dict) onto the project config."""
    experiment_config = (
        experiment_path
        if isinstance(experiment_path, dict)
        else load_config(experiment_path)
    )
    return _deep_merge(load_config(project_path), experiment_config)


def load_experiment_config_by_name(project_root, name):
    """Load a fully-merged config for an experiment by its file-stem name."""
    experiment_config = _load_experiment_config_tree(project_root, name)
    return load_experiment_config(
        project_root / "configs" / "project.yaml",
        experiment_config,
    )


def _load_experiment_config_tree(project_root, name, seen=None):
    """Resolve an experiment and its optional `parent` chain into one config.
    Raises on a cyclic parent reference."""
    seen = seen or set()
    if name in seen:
        raise ValueError(f"cyclic experiment parent chain: {name}")
    seen.add(name)

    config = load_config(project_root / "configs" / "experiments" / f"{name}.yaml")
    parent = config.get("experiment", {}).get("parent")
    if parent is None:
        return config
    return _deep_merge(_load_experiment_config_tree(project_root, parent, seen), config)


def _deep_merge(base, override):
    """Recursively merge `override` onto `base`: nested dicts are merged,
    scalars and lists are replaced."""
    out = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
