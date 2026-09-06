from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_ridge(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("standardize", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def make_hgb(params: dict, random_seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=float(params["learning_rate"]),
        max_iter=int(params["max_iter"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        early_stopping=False,
        random_state=random_seed,
    )


def monthly_climatology_predictions(train, evaluation):
    means = train.groupby(train["target_date"].dt.month)["y_true"].mean()
    requested_months = set(evaluation["target_date"].dt.month.unique())
    missing_months = requested_months - set(means.index)
    if missing_months:
        raise ValueError(f"Training data lack calendar months: {sorted(missing_months)}")
    return evaluation["target_date"].dt.month.map(means).to_numpy(dtype=float)
