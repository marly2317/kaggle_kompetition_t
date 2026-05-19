from catboost import CatBoostClassifier


def get_model(config, params_override=None):
    if config["model"]["family"] != "catboost":
        raise ValueError(f"unsupported model family: {config['model']['family']}")
    params = dict(config["model"]["params"])
    if params_override:
        params.update(params_override)
    return CatBoostClassifier(**params)


def get_fit_params(config, fit_params_override=None):
    fit_params = dict(config["model"].get("fit_params", {}))
    if fit_params_override:
        fit_params.update(fit_params_override)
    return fit_params
