import numpy as np
import pandas as pd

from .utils import check_columns


SOURCE_COLUMNS = [
    "Pclass", "Name", "Sex", "Age", "SibSp", "Parch",
    "Cabin", "Ticket", "Fare", "Embarked",
]

CATEGORICAL = [
    "Sex", "Embarked", "Title", "Deck", "FamilySizeGroup", "PassengerType",
    "Surname", "TicketPrefix",
]

FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
    "FamilySize", "IsAlone", "FamilySizeGroup", "Title", "Deck",
    "HasCabin", "TicketFrequency", "FarePerPerson", "PassengerType",
    "Surname", "TicketPrefix", "NaNCount", "FamilySurvivalRate",
]


def get_features(config):
    """Return the model feature list, minus any features.drop from the config."""
    drop = set(_feature_settings(config).get("drop", []))
    unknown = sorted(drop - set(FEATURES))
    if unknown:
        raise ValueError(f"unknown features to drop: {unknown}")
    return [feature for feature in FEATURES if feature not in drop]


def get_categorical_features(config):
    """Return the categorical subset of the selected features."""
    selected = set(get_features(config))
    return [feature for feature in CATEGORICAL if feature in selected]


def build_base_features(df, config):
    """Derive label-independent features (Title, Surname, Deck, FamilySize, ...)
    from the raw columns. No fitting — safe to call on any split."""
    check_columns(df, SOURCE_COLUMNS, "features_source")
    out = df[SOURCE_COLUMNS].copy()

    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["FamilySizeGroup"] = out["FamilySize"].map(_family_group)
    out["Title"] = (
        out["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
        .fillna("Unknown").str.strip()
    )
    out["Surname"] = out["Name"].str.split(",", n=1).str[0].str.strip()
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    out["Deck"] = out["Cabin"].map(_deck)
    out["TicketPrefix"] = out["Ticket"].map(_ticket_prefix)
    out["Ticket"] = out["Ticket"].map(_clean_ticket)
    out["NaNCount"] = df[SOURCE_COLUMNS].isna().sum(axis=1).astype(int)

    return out.drop(columns=["Name", "Cabin"])


def fit_preprocessing_artifacts(df, target, config):
    """Fit artifacts on a training fold. Target is required because some artifacts
    (family survival rate) are computed from labels."""
    return {
        "age_by_title_sex_pclass": (
            df.groupby(["Title", "Sex", "Pclass"])["Age"].median().dropna()
        ),
        "age_by_sex_pclass": df.groupby(["Sex", "Pclass"])["Age"].median().dropna(),
        "age_global": float(df["Age"].median()),
        "fare_by_pclass": df.groupby("Pclass")["Fare"].median().dropna(),
        "fare_global": float(df["Fare"].median()),
        "ticket_counts": df["Ticket"].value_counts(),
        "embarked_mode": _mode_or_default(df["Embarked"], "Unknown"),
        "family_sum": target.groupby(df["Ticket"]).sum(),
        "family_count": target.groupby(df["Ticket"]).count(),
        "family_prior": float(target.mean()),
    }


def apply_preprocessing_artifacts(df, artifacts, config, target=None):
    """Apply artifacts. Pass `target` only when df is the training fold whose
    labels were used to fit `artifacts` — then leave-one-out is applied to
    family survival to avoid leakage. For valid/test pass target=None."""
    out = df.copy()
    smoothing = float(_feature_settings(config).get("family_smoothing", 10.0))

    out["Age"] = _impute_age(out, artifacts)
    out["Fare"] = (
        out["Fare"]
        .fillna(out["Pclass"].map(artifacts["fare_by_pclass"]))
        .fillna(artifacts["fare_global"])
    )
    out["Embarked"] = out["Embarked"].fillna(artifacts["embarked_mode"])

    out["TicketFrequency"] = (
        out["Ticket"].map(artifacts["ticket_counts"]).fillna(1).astype(int)
    )
    group_size = out[["FamilySize", "TicketFrequency"]].max(axis=1).clip(lower=1)
    out["FarePerPerson"] = out["Fare"] / group_size
    out["PassengerType"] = _passenger_type(out)
    out["FamilySurvivalRate"] = _family_survival(out, artifacts, target, smoothing)

    for col in CATEGORICAL:
        out[col] = out[col].fillna("Unknown").astype(str)

    out = out[get_features(config)]
    leftover_nans = out.columns[out.isna().any()].tolist()
    if leftover_nans:
        raise ValueError(f"NaN remain after preprocessing: {leftover_nans}")
    return out


def _family_survival(df, artifacts, target, smoothing):
    """Smoothed survival rate of a passenger's ticket group. When `target` is
    given (training fold), the passenger's own label is removed (leave-one-out)
    so the feature does not leak the target into the model."""
    sums = df["Ticket"].map(artifacts["family_sum"]).fillna(0).astype(float)
    counts = df["Ticket"].map(artifacts["family_count"]).fillna(0).astype(float)

    if target is not None:
        sums = sums - target.to_numpy()
        counts = counts - 1

    
    prior = artifacts["family_prior"]
    return (sums + smoothing * prior) / (counts + smoothing)


def _feature_settings(config):
    """Return the config's `features` section (empty dict if absent)."""
    return config.get("features", {})


def _impute_age(df, artifacts):
    """Fill missing Age from the most specific group median available —
    Title+Sex+Pclass, then Sex+Pclass, then the global median."""
    age = df["Age"]
    age = _fillna_lookup(age, df, ["Title", "Sex", "Pclass"], artifacts["age_by_title_sex_pclass"])
    age = _fillna_lookup(age, df, ["Sex", "Pclass"], artifacts["age_by_sex_pclass"])
    return age.fillna(artifacts["age_global"])


def _fillna_lookup(series, df, keys, lookup):
    """Fill NaNs in `series` from `lookup`, keyed by the `keys` columns of df."""
    if lookup.empty:
        return series
    idx = pd.MultiIndex.from_frame(df[keys])
    fill = pd.Series(lookup.reindex(idx).to_numpy(), index=df.index)
    return series.fillna(fill)


def _passenger_type(df):
    """Classify each passenger as woman / boy / adult_man — the "women and
    children first" signal as a single feature."""
    out = pd.Series("adult_man", index=df.index, dtype=object)
    out[(df["Sex"] == "male") & ((df["Title"] == "Master") | (df["Age"] < 16))] = "boy"
    out[df["Sex"] == "female"] = "woman"
    return out


def _family_group(size):
    """Bucket family size into Alone / Small / Medium / Large."""
    if size == 1:
        return "Alone"
    if size <= 4:
        return "Small"
    if size <= 6:
        return "Medium"
    return "Large"


def _deck(cabin):
    """Map a raw cabin string to a coarse deck bucket (ABC / DE / FG / Rare)."""
    if pd.isna(cabin):
        return "Unknown"
    letter = str(cabin).strip()[:1].upper()
    if letter in "ABC":
        return "ABC"
    if letter in "DE":
        return "DE"
    if letter in "FG":
        return "FG"
    return "Rare"


def _clean_ticket(ticket):
    """Normalize a ticket string (upper-case, strip punctuation) for grouping."""
    if pd.isna(ticket):
        return "Unknown"
    return str(ticket).upper().replace(".", "").replace("/", "").strip()


def _ticket_prefix(ticket):
    """Extract the non-numeric prefix of a ticket, or "None" if there is none."""
    if pd.isna(ticket):
        return "None"
    parts = str(ticket).split()
    if len(parts) > 1:
        return parts[0].upper().replace(".", "").replace("/", "")
    return "None"


def _mode_or_default(series, default):
    """Most frequent value of a series, or `default` if it is empty."""
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else default
