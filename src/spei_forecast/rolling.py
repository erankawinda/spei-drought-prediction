from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .features import build_supervised, feature_columns
from .models import make_ridge
from .robustness import rolling_inner_folds, rolling_outer_folds


@dataclass(frozen=True)
class RidgeVariant:
    name: str
    feature_set: str


def run_nested_rolling_ridge(
    observations,
    benchmark_config,
    robustness_config,
    variants,
    targets=None,
    training_start_year=None,
):
    """Evaluate fixed ridge variants under one common nested rolling protocol."""
    variants = tuple(variants)
    if not variants:
        raise ValueError("At least one ridge variant is required")
    if len({variant.name for variant in variants}) != len(variants):
        raise ValueError("Ridge variant names must be unique")
    allowed_feature_sets = {"ar24", "met24", "ar24_met24"}
    if any(variant.feature_set not in allowed_feature_sets for variant in variants):
        raise ValueError("Unsupported ridge feature set")
    if training_start_year is not None:
        observations = observations.loc[
            observations["month"] >= pd.Timestamp(f"{training_start_year:04d}-01-01")
        ].copy()
    effective_start_year = int(observations["month"].dt.year.min())
    requested_targets = tuple(targets or benchmark_config.targets)
    predictions = []
    selections = []

    for target in requested_targets:
        station_frames = {
            station: _supervised(observations, benchmark_config, station, target)
            for station in sorted(benchmark_config.station_files)
        }
        for variant in variants:
            columns = feature_columns(
                variant.feature_set,
                target,
                benchmark_config.history_months,
                benchmark_config.meteorological_features,
            )
            for fold in rolling_outer_folds(robustness_config):
                inner_folds = rolling_inner_folds(
                    fold["evaluation_start"], robustness_config
                )
                candidates = []
                for alpha in benchmark_config.ridge_alphas:
                    scores = []
                    for inner in inner_folds:
                        for frame in station_frames.values():
                            train = frame.loc[
                                frame["target_date"] < inner["validation_start"]
                            ]
                            validation = frame.loc[
                                frame["target_date"].between(
                                    inner["validation_start"],
                                    inner["validation_end"],
                                )
                            ]
                            estimator = make_ridge(alpha).fit(
                                train[columns], train["y_true"]
                            )
                            scores.append(
                                mean_absolute_error(
                                    validation["y_true"],
                                    estimator.predict(validation[columns]),
                                )
                            )
                    candidates.append(
                        {
                            "alpha": alpha,
                            "inner_fold_station_mean_mae": float(np.mean(scores)),
                        }
                    )
                chosen = min(
                    candidates,
                    key=lambda row: (
                        row["inner_fold_station_mean_mae"],
                        -row["alpha"],
                    ),
                )
                selections.append(
                    {
                        "target": target,
                        "variant": variant.name,
                        "feature_set": variant.feature_set,
                        "feature_count": len(columns),
                        "feature_columns": json.dumps(columns),
                        "training_start_year": effective_start_year,
                        "outer_fold": fold["name"],
                        "alpha": chosen["alpha"],
                        "inner_fold_station_mean_mae": chosen[
                            "inner_fold_station_mean_mae"
                        ],
                        "candidate_scores": json.dumps(candidates, sort_keys=True),
                        "inner_folds": len(inner_folds),
                        "inner_validation_periods": json.dumps(
                            [
                                f"{item['validation_start'].strftime('%Y-%m')}/"
                                f"{item['validation_end'].strftime('%Y-%m')}"
                                for item in inner_folds
                            ]
                        ),
                        "outer_fit_end": (
                            fold["evaluation_start"] - pd.offsets.MonthBegin(1)
                        ).strftime("%Y-%m"),
                        "outer_evaluation_start": fold[
                            "evaluation_start"
                        ].strftime("%Y-%m"),
                        "outer_evaluation_end": fold["evaluation_end"].strftime(
                            "%Y-%m"
                        ),
                        "outer_outcomes_used_for_selection": False,
                    }
                )
                for station, frame in station_frames.items():
                    fit = frame.loc[frame["target_date"] < fold["evaluation_start"]]
                    evaluation = frame.loc[
                        frame["target_date"].between(
                            fold["evaluation_start"], fold["evaluation_end"]
                        )
                    ]
                    estimator = make_ridge(chosen["alpha"]).fit(
                        fit[columns], fit["y_true"]
                    )
                    predictions.append(
                        _prediction_frame(
                            evaluation,
                            target,
                            station,
                            variant.name,
                            estimator.predict(evaluation[columns]),
                            fold,
                            effective_start_year,
                        )
                    )

    prediction_frame = pd.concat(predictions, ignore_index=True).sort_values(
        ["target", "variant", "station", "target_date"]
    )
    duplicate_key = ["target", "station", "variant", "target_date"]
    if prediction_frame.duplicated(duplicate_key).any():
        raise AssertionError(
            "Nested rolling evaluation contains duplicate target months"
        )
    _validate_common_panel(prediction_frame)
    return prediction_frame.reset_index(drop=True), pd.DataFrame(selections)


def _supervised(observations, config, station, target):
    return build_supervised(
        observations.loc[observations["station"] == station].copy(),
        target,
        config.history_months,
        config.horizon_months,
        config.meteorological_features,
    )


def _prediction_frame(evaluation, target, station, variant, prediction, fold, start):
    return pd.DataFrame(
        {
            "target": target,
            "station": station,
            "variant": variant,
            "issue_date": evaluation["issue_date"].to_numpy(),
            "target_date": evaluation["target_date"].to_numpy(),
            "horizon_months": 1,
            "y_true": evaluation["y_true"].to_numpy(dtype=float),
            "y_pred": np.asarray(prediction, dtype=float),
            "outer_fold": fold["name"],
            "training_start_year": start,
            "fit_end": fold["evaluation_start"] - pd.offsets.MonthBegin(1),
        }
    )


def _validate_common_panel(predictions):
    key = ["target", "station", "target_date"]
    counts = predictions.groupby(key)["variant"].nunique()
    expected = predictions["variant"].nunique()
    if not counts.eq(expected).all():
        raise ValueError("Variants do not share a complete evaluation panel")
    truth = predictions.groupby(key)["y_true"].nunique()
    if not truth.eq(1).all():
        raise ValueError("Variants do not share identical truth values")
