from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import tomllib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .bootstrap import circular_block_indices
from .config import BenchmarkConfig
from .data import load_all
from .features import (
    FEATURE_SET_AR,
    FEATURE_SET_COMBINED,
    build_supervised,
    feature_columns,
)
from .metrics import alert_metrics, metrics_table
from .models import make_ridge
from .provenance import file_sha256, implementation_hashes, json_sha256


PRIMARY_MODELS = {
    "spei3": ("ridge_ar24", FEATURE_SET_AR),
    "spei6": ("ridge_ar24_met24", FEATURE_SET_COMBINED),
}


@dataclass(frozen=True)
class RobustnessConfig:
    config_path: Path
    training_start_years: tuple[int, ...]
    bootstrap_block_months: tuple[int, ...]
    bootstrap_resamples: int
    event_threshold_min: float
    event_threshold_max: float
    event_threshold_step: float
    rolling_outer_start: str
    rolling_outer_end: str
    rolling_outer_months: int
    rolling_inner_months: int
    rolling_inner_folds: int

    @property
    def config_sha256(self) -> str:
        return sha256(self.config_path.read_bytes()).hexdigest()


def load_robustness_config(path: str | Path) -> RobustnessConfig:
    config_path = Path(path).expanduser().resolve()
    settings = tomllib.loads(config_path.read_text(encoding="utf-8"))["robustness"]
    config = RobustnessConfig(
        config_path=config_path,
        training_start_years=tuple(
            int(value) for value in settings["training_start_years"]
        ),
        bootstrap_block_months=tuple(
            int(value) for value in settings["bootstrap_block_months"]
        ),
        bootstrap_resamples=int(settings["bootstrap_resamples"]),
        event_threshold_min=float(settings["event_threshold_min"]),
        event_threshold_max=float(settings["event_threshold_max"]),
        event_threshold_step=float(settings["event_threshold_step"]),
        rolling_outer_start=str(settings["rolling_outer_start"]),
        rolling_outer_end=str(settings["rolling_outer_end"]),
        rolling_outer_months=int(settings["rolling_outer_months"]),
        rolling_inner_months=int(settings["rolling_inner_months"]),
        rolling_inner_folds=int(settings["rolling_inner_folds"]),
    )
    _validate_robustness_config(config)
    return config


def _validate_robustness_config(config: RobustnessConfig) -> None:
    if len(set(config.training_start_years)) != len(config.training_start_years):
        raise ValueError("Training starts must be distinct")
    if len(set(config.bootstrap_block_months)) != len(config.bootstrap_block_months):
        raise ValueError("Bootstrap blocks must be distinct")
    if config.bootstrap_resamples < 1 or config.event_threshold_step <= 0:
        raise ValueError("Robustness resamples and threshold step must be positive")
    if any(value < 1 for value in config.bootstrap_block_months):
        raise ValueError("Bootstrap block lengths must be positive")
    if config.event_threshold_min > -1.0 or config.event_threshold_max < -1.0:
        raise ValueError("The threshold grid must include the physical -1.0 cutoff")
    threshold_steps = (-1.0 - config.event_threshold_min) / config.event_threshold_step
    if not np.isclose(threshold_steps, round(threshold_steps), atol=1e-12):
        raise ValueError("The threshold grid must land exactly on -1.0")
    grid_steps = (
        config.event_threshold_max - config.event_threshold_min
    ) / config.event_threshold_step
    if not np.isclose(grid_steps, round(grid_steps), atol=1e-12):
        raise ValueError("The threshold grid must land exactly on its maximum")
    if config.rolling_outer_months < 1 or config.rolling_inner_months < 1:
        raise ValueError("Rolling windows must be positive")
    if config.rolling_inner_folds < 1:
        raise ValueError("At least one inner fold is required")
    start = _month(config.rolling_outer_start)
    end = _month(config.rolling_outer_end)
    if start > end:
        raise ValueError("Rolling evaluation start must not follow its end")
    if any(year >= start.year for year in config.training_start_years):
        raise ValueError("Training-history starts must precede rolling evaluation")
    total_months = (end.year - start.year) * 12 + end.month - start.month + 1
    if total_months % config.rolling_outer_months:
        raise ValueError("Outer period must divide into non-overlapping folds")


def run_robustness(
    benchmark_config: BenchmarkConfig,
    robustness_config: RobustnessConfig,
    output_dir: str | Path,
) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _validate_benchmark_for_robustness(benchmark_config)
    observations, quality = load_all(benchmark_config)
    selection = _phase1_regression_selection(observations, benchmark_config)

    rolling_predictions, rolling_selection = rolling_origin_diagnostics(
        observations, benchmark_config, robustness_config
    )
    rolling_by_task, rolling_macro = metrics_table(rolling_predictions)
    rolling_fold_by_task, rolling_fold_macro = _scenario_metrics(
        rolling_predictions, "outer_fold"
    )
    start_predictions, start_selection = start_date_sensitivity(
        observations, benchmark_config, robustness_config
    )
    start_by_task, start_macro = _scenario_metrics(
        start_predictions, "training_start_year"
    )
    bootstrap_sensitivity = bootstrap_block_sensitivity(
        rolling_predictions, benchmark_config, robustness_config
    )
    threshold_sweep, threshold_selection, threshold_metrics = (
        event_threshold_diagnostics(
            observations, selection, benchmark_config, robustness_config
        )
    )

    _dated_csv(rolling_predictions, output / "rolling_predictions.csv")
    rolling_by_task.to_csv(output / "rolling_metrics_by_task.csv", index=False)
    rolling_macro.to_csv(output / "rolling_metrics_macro.csv", index=False)
    rolling_fold_by_task.to_csv(
        output / "rolling_metrics_by_fold_and_station.csv", index=False
    )
    rolling_fold_macro.to_csv(output / "rolling_metrics_by_fold.csv", index=False)
    rolling_selection.to_csv(output / "rolling_selection.csv", index=False)
    _dated_csv(start_predictions, output / "start_date_predictions.csv")
    start_by_task.to_csv(output / "start_date_metrics_by_task.csv", index=False)
    start_macro.to_csv(output / "start_date_metrics_macro.csv", index=False)
    start_selection.to_csv(output / "start_date_selection.csv", index=False)
    bootstrap_sensitivity.to_csv(
        output / "bootstrap_block_sensitivity.csv", index=False
    )
    threshold_sweep.to_csv(output / "event_threshold_sweep.csv", index=False)
    threshold_selection.to_csv(output / "event_threshold_selection.csv", index=False)
    threshold_metrics.to_csv(output / "event_threshold_metrics.csv", index=False)
    (output / "alert_regression_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    resolved_config = _resolved_config(benchmark_config, robustness_config)
    (output / "config_resolved.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    git_commit = _run_git(benchmark_config.repository_root, ["rev-parse", "HEAD"])
    git_status = _run_git(benchmark_config.repository_root, ["status", "--porcelain"])
    manifest = {
        "analysis": "robustness_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": _portable_path(output, benchmark_config.repository_root),
        "git_commit": git_commit,
        "git_worktree_dirty": bool(git_status),
        "git_status_porcelain": git_status.splitlines(),
        "evidence_status": "retrospective_robustness",
        "period_previously_inspected": True,
        "fresh_locked_test": False,
        "evaluation_period": (
            f"{_month(benchmark_config.validation_end) + pd.offsets.MonthBegin(1):%Y-%m}/"
            f"{_month(benchmark_config.test_end):%Y-%m}"
        ),
        "new_model_family_added": False,
        "primary_models": {target: spec[0] for target, spec in PRIMARY_MODELS.items()},
        "source_files": quality["source_files"],
        "rolling_update_policy": "refit at each five-year outer origin; frozen within block",
        "rolling_outer_fold_count": int(rolling_predictions["outer_fold"].nunique()),
        "rolling_predictions_sha256": file_sha256(
            output / "rolling_predictions.csv"
        ),
        "training_start_years": list(robustness_config.training_start_years),
        "bootstrap_block_months": list(robustness_config.bootstrap_block_months),
        "event_observed_threshold": -1.0,
        "event_decision_threshold_selected_on": (
            f"{_month(benchmark_config.train_end) + pd.offsets.MonthBegin(1):%Y-%m}/"
            f"{_month(benchmark_config.validation_end):%Y-%m} validation only"
        ),
        "test_labels_used_for_threshold_selection": False,
        "upstream_model_selection_uses_test": False,
        "upstream_model_selection_recomputed": True,
        "benchmark_config_sha256": benchmark_config.config_sha256,
        "robustness_config_sha256": robustness_config.config_sha256,
        "resolved_config_sha256": json_sha256(resolved_config),
        "implementation_files_sha256": implementation_hashes(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "target_construction_limitation": (
            "The recovered SPEI calibration/reference period is unknown; model fitting is "
            "temporally controlled, but target-construction leakage cannot be ruled out."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": str(output),
        "rolling_metrics_macro": rolling_macro,
        "start_date_metrics_macro": start_macro,
        "bootstrap_block_sensitivity": bootstrap_sensitivity,
        "event_threshold_metrics": threshold_metrics,
        "manifest": manifest,
    }


def rolling_origin_diagnostics(
    observations, benchmark_config, robustness_config, training_start_year=None
):
    prediction_rows = []
    selection_rows = []
    outer_folds = rolling_outer_folds(robustness_config)
    if training_start_year is not None:
        observations = observations.loc[
            observations["month"] >= pd.Timestamp(f"{training_start_year:04d}-01-01")
        ].copy()
    effective_start_year = int(observations["month"].dt.year.min())
    for target in benchmark_config.targets:
        model_name, feature_set = PRIMARY_MODELS[target]
        station_frames = {
            station: _supervised(observations, benchmark_config, station, target)
            for station in sorted(benchmark_config.station_files)
        }
        columns = feature_columns(
            feature_set,
            target,
            benchmark_config.history_months,
            benchmark_config.meteorological_features,
        )
        for fold in outer_folds:
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
                                inner["validation_start"], inner["validation_end"]
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
                key=lambda row: (row["inner_fold_station_mean_mae"], -row["alpha"]),
            )
            selection_rows.append(
                {
                    "target": target,
                    "model": model_name,
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
                    "outer_evaluation_start": fold["evaluation_start"].strftime(
                        "%Y-%m"
                    ),
                    "outer_evaluation_end": fold["evaluation_end"].strftime("%Y-%m"),
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
                estimator = make_ridge(chosen["alpha"]).fit(fit[columns], fit["y_true"])
                prediction_rows.append(
                    _prediction_frame(
                        evaluation,
                        target,
                        station,
                        model_name,
                        estimator.predict(evaluation[columns]),
                        outer_fold=fold["name"],
                        training_start_year=effective_start_year,
                        fit_end=(fold["evaluation_start"] - pd.offsets.MonthBegin(1)),
                        evaluation_start=fold["evaluation_start"],
                        evaluation_end=fold["evaluation_end"],
                    )
                )
                prediction_rows.append(
                    _prediction_frame(
                        evaluation,
                        target,
                        station,
                        "persistence",
                        evaluation[f"{target}_lag_1"].to_numpy(dtype=float),
                        outer_fold=fold["name"],
                        training_start_year=effective_start_year,
                        fit_end=(fold["evaluation_start"] - pd.offsets.MonthBegin(1)),
                        evaluation_start=fold["evaluation_start"],
                        evaluation_end=fold["evaluation_end"],
                    )
                )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    duplicate_key = ["target", "station", "model", "target_date"]
    if predictions.duplicated(duplicate_key).any():
        raise AssertionError("Rolling evaluation contains duplicate target months")
    return predictions, pd.DataFrame(selection_rows)


def rolling_outer_folds(config: RobustnessConfig) -> list[dict]:
    starts = pd.date_range(
        _month(config.rolling_outer_start),
        _month(config.rolling_outer_end),
        freq=pd.DateOffset(months=config.rolling_outer_months),
    )
    folds = []
    for start in starts:
        end = start + pd.DateOffset(months=config.rolling_outer_months - 1)
        folds.append(
            {
                "name": f"{start.strftime('%Y-%m')}_{end.strftime('%Y-%m')}",
                "evaluation_start": start,
                "evaluation_end": end,
            }
        )
    return folds


def rolling_inner_folds(
    outer_start: pd.Timestamp, config: RobustnessConfig
) -> list[dict]:
    first_start = outer_start - pd.DateOffset(
        months=config.rolling_inner_folds * config.rolling_inner_months
    )
    folds = []
    for index in range(config.rolling_inner_folds):
        start = first_start + pd.DateOffset(months=index * config.rolling_inner_months)
        end = start + pd.DateOffset(months=config.rolling_inner_months - 1)
        folds.append({"validation_start": start, "validation_end": end})
    return folds


def start_date_sensitivity(observations, benchmark_config, robustness_config):
    predictions = []
    selections = []
    for start_year in robustness_config.training_start_years:
        start_predictions, start_selection = rolling_origin_diagnostics(
            observations,
            benchmark_config,
            robustness_config,
            training_start_year=start_year,
        )
        predictions.append(start_predictions)
        selections.append(start_selection)
    return pd.concat(predictions, ignore_index=True), pd.concat(
        selections, ignore_index=True
    )


def bootstrap_block_sensitivity(predictions, benchmark_config, robustness_config):
    selected = predictions.loc[
        (predictions["model"] == "persistence")
        | predictions.apply(
            lambda row: row["model"] == PRIMARY_MODELS[row["target"]][0], axis=1
        )
    ].copy()
    rows = []
    for block_months in robustness_config.bootstrap_block_months:
        rows.append(
            _rolling_paired_mae_bootstrap(
                selected,
                resamples=robustness_config.bootstrap_resamples,
                block_months=block_months,
                seed=benchmark_config.random_seed,
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["target", "model", "block_months"]
    )


def _rolling_paired_mae_bootstrap(predictions, resamples, block_months, seed):
    """Paired bootstrap synchronized across stations and bounded within refit folds."""
    required = {
        "target",
        "station",
        "model",
        "target_date",
        "outer_fold",
        "y_true",
        "y_pred",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Rolling bootstrap is missing columns: {sorted(missing)}")
    rows = []
    for target, target_rows in predictions.groupby("target", sort=True):
        stations = sorted(target_rows["station"].unique())
        fold_errors = {}
        for fold, fold_rows in target_rows.groupby("outer_fold", sort=True):
            dates = np.array(sorted(fold_rows["target_date"].unique()))
            truth = _panel_matrix(fold_rows, "y_true", "persistence", dates, stations)
            persistence = _panel_matrix(
                fold_rows, "y_pred", "persistence", dates, stations
            )
            fold_errors[fold] = {
                "dates": dates,
                "persistence": np.abs(truth - persistence),
                "truth": truth,
            }
        persistence_point = float(
            np.concatenate(
                [values["persistence"] for values in fold_errors.values()], axis=0
            )
            .mean(axis=0)
            .mean()
        )
        if persistence_point == 0:
            raise ValueError("Relative skill is undefined for zero-error persistence")

        for model in sorted(target_rows["model"].unique()):
            sampled_persistence_parts = []
            sampled_model_parts = []
            model_point_parts = []
            for fold_index, (fold, values) in enumerate(fold_errors.items()):
                fold_rows = target_rows.loc[target_rows["outer_fold"] == fold]
                model_truth = _panel_matrix(
                    fold_rows, "y_true", model, values["dates"], stations
                )
                if not np.array_equal(model_truth, values["truth"]):
                    raise ValueError(
                        f"Truth panel for {model} differs from persistence"
                    )
                model_prediction = _panel_matrix(
                    fold_rows, "y_pred", model, values["dates"], stations
                )
                model_errors = np.abs(values["truth"] - model_prediction)
                model_point_parts.append(model_errors)
                indices = circular_block_indices(
                    len(values["dates"]),
                    block_months,
                    resamples,
                    seed + fold_index,
                )
                sampled_persistence_parts.append(values["persistence"][indices, :])
                sampled_model_parts.append(model_errors[indices, :])

            model_point = float(
                np.concatenate(model_point_parts, axis=0).mean(axis=0).mean()
            )
            point = persistence_point - model_point
            sampled_persistence = (
                np.concatenate(sampled_persistence_parts, axis=1)
                .mean(axis=1)
                .mean(axis=1)
            )
            sampled_model = (
                np.concatenate(sampled_model_parts, axis=1).mean(axis=1).mean(axis=1)
            )
            improvement = sampled_persistence - sampled_model
            lower, upper = np.percentile(improvement, [2.5, 97.5])
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "metric": "rolling_macro_mae_improvement_vs_persistence",
                    "point_estimate": float(point),
                    "ci_2_5": float(lower),
                    "ci_97_5": float(upper),
                    "relative_improvement": float(point / persistence_point),
                    "resamples": int(resamples),
                    "block_months": int(block_months),
                    "outer_fold_bounded": True,
                    "passes_5pct_positive_ci_gate": bool(
                        point / persistence_point >= 0.05 and lower > 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _panel_matrix(rows, value_column, model, dates, stations):
    selected = rows.loc[rows["model"] == model]
    pivot = selected.pivot(index="target_date", columns="station", values=value_column)
    pivot = pivot.reindex(index=dates, columns=stations)
    if pivot.isna().any().any():
        raise ValueError(f"Incomplete synchronized prediction panel for {model}")
    return pivot.to_numpy(dtype=float)


def _phase1_regression_selection(observations, benchmark_config):
    """Recompute only the phase-one ridge choices needed by alert diagnostics."""
    targets = {}
    train_end = _month(benchmark_config.train_end)
    validation_end = _month(benchmark_config.validation_end)
    for target in benchmark_config.targets:
        model_name, feature_set = PRIMARY_MODELS[target]
        columns = feature_columns(
            feature_set,
            target,
            benchmark_config.history_months,
            benchmark_config.meteorological_features,
        )
        station_frames = {
            station: _supervised(observations, benchmark_config, station, target)
            for station in sorted(benchmark_config.station_files)
        }
        candidates = []
        for alpha in benchmark_config.ridge_alphas:
            station_scores = []
            for frame in station_frames.values():
                train = frame.loc[frame["target_date"] <= train_end]
                validation = frame.loc[
                    (frame["target_date"] > train_end)
                    & (frame["target_date"] <= validation_end)
                ]
                estimator = make_ridge(alpha).fit(train[columns], train["y_true"])
                station_scores.append(
                    mean_absolute_error(
                        validation["y_true"], estimator.predict(validation[columns])
                    )
                )
            candidates.append(
                {"alpha": alpha, "macro_validation_mae": float(np.mean(station_scores))}
            )
        chosen = min(
            candidates, key=lambda row: (row["macro_validation_mae"], -row["alpha"])
        )
        targets[target] = {
            "candidate_scores_macro_validation_mae": {model_name: candidates},
            "chosen": {model_name: chosen},
        }
    return {
        "selection_partition": "validation",
        "test_partition_used_for_selection": False,
        "targets": targets,
    }


def event_threshold_diagnostics(
    observations, selection, benchmark_config, robustness_config
):
    sweep_rows = []
    selection_rows = []
    metric_rows = []
    thresholds = np.round(
        np.arange(
            robustness_config.event_threshold_min,
            robustness_config.event_threshold_max
            + robustness_config.event_threshold_step / 2,
            robustness_config.event_threshold_step,
        ),
        10,
    )
    train_end = _month(benchmark_config.train_end)
    validation_start = train_end + pd.offsets.MonthBegin(1)
    validation_end = _month(benchmark_config.validation_end)
    test_start = validation_end + pd.offsets.MonthBegin(1)
    test_end = _month(benchmark_config.test_end)
    for target in benchmark_config.targets:
        model_name, feature_set = PRIMARY_MODELS[target]
        columns = feature_columns(
            feature_set,
            target,
            benchmark_config.history_months,
            benchmark_config.meteorological_features,
        )
        alpha = selection["targets"][target]["chosen"][model_name]["alpha"]
        panels = {"model": [], "persistence": []}
        test_panels = {"model": [], "persistence": []}
        for station in sorted(benchmark_config.station_files):
            frame = _supervised(observations, benchmark_config, station, target)
            train = frame.loc[frame["target_date"] <= train_end]
            validation = frame.loc[
                frame["target_date"].between(validation_start, validation_end)
            ]
            development = frame.loc[frame["target_date"] <= validation_end]
            test = frame.loc[frame["target_date"].between(test_start, test_end)]
            validation_model = make_ridge(alpha).fit(train[columns], train["y_true"])
            final_model = make_ridge(alpha).fit(
                development[columns], development["y_true"]
            )
            panels["model"].append(
                _alert_panel(
                    station, validation, validation_model.predict(validation[columns])
                )
            )
            panels["persistence"].append(
                _alert_panel(station, validation, validation[f"{target}_lag_1"])
            )
            test_panels["model"].append(
                _alert_panel(station, test, final_model.predict(test[columns]))
            )
            test_panels["persistence"].append(
                _alert_panel(station, test, test[f"{target}_lag_1"])
            )

        for calibration_name in ("model", "persistence"):
            scored_predictor = (
                model_name if calibration_name == "model" else "persistence"
            )
            validation_panel = pd.concat(panels[calibration_name], ignore_index=True)
            eligible_rows = []
            for threshold in thresholds:
                station_metrics = []
                eligible = True
                for station, group in validation_panel.groupby("station", sort=True):
                    metrics = alert_metrics(
                        group["y_true"],
                        group["y_pred"],
                        observed_threshold=-1.0,
                        decision_threshold=float(threshold),
                        zero_division=0.0,
                    )
                    if metrics["predicted_support"] in (0, len(group)):
                        eligible = False
                    station_metrics.append(metrics)
                row = {
                    "target": target,
                    "model": scored_predictor,
                    "primary_model_context": model_name,
                    "calibration_for": calibration_name,
                    "decision_threshold": float(threshold),
                    "macro_validation_f1": float(
                        np.mean([item["f1"] for item in station_metrics])
                    ),
                    "worst_station_validation_f1": float(
                        np.min([item["f1"] for item in station_metrics])
                    ),
                    "macro_validation_balanced_accuracy": float(
                        np.mean([item["balanced_accuracy"] for item in station_metrics])
                    ),
                    "eligible": bool(eligible),
                }
                sweep_rows.append(row)
                if eligible:
                    eligible_rows.append(row)
            chosen = _select_alert_threshold(
                eligible_rows,
                [
                    row
                    for row in sweep_rows
                    if row["target"] == target
                    and row["model"] == scored_predictor
                    and row["calibration_for"] == calibration_name
                ],
            )
            selection_rows.append(
                {
                    **chosen,
                    "alpha_frozen_from_regression_selection": (
                        alpha if calibration_name == "model" else None
                    ),
                    "selected_on": (
                        f"validation_only_{validation_start.strftime('%Y-%m')}_"
                        f"{validation_end.strftime('%Y-%m')}"
                    ),
                    "test_labels_used_for_selection": False,
                }
            )
            metric_rows.extend(
                _alert_metric_rows(
                    pd.concat(panels[calibration_name], ignore_index=True),
                    target,
                    scored_predictor,
                    model_name,
                    calibration_name,
                    chosen["decision_threshold"],
                    period=(
                        f"{validation_start.strftime('%Y-%m')}/"
                        f"{validation_end.strftime('%Y-%m')}"
                    ),
                    interpretation_status="threshold_selection_apparent_performance",
                )
            )
            metric_rows.extend(
                _alert_metric_rows(
                    pd.concat(test_panels[calibration_name], ignore_index=True),
                    target,
                    scored_predictor,
                    model_name,
                    calibration_name,
                    chosen["decision_threshold"],
                    period=(
                        f"{test_start.strftime('%Y-%m')}/{test_end.strftime('%Y-%m')}"
                    ),
                    interpretation_status="post_hoc_retrospective_diagnostic",
                )
            )
    metric_frame = pd.DataFrame(metric_rows)
    persistence_f1 = metric_frame.loc[
        metric_frame["model"] == "persistence",
        ["target", "station", "aggregation", "period", "calibrated_f1"],
    ].rename(columns={"calibrated_f1": "calibrated_persistence_f1"})
    metric_frame = metric_frame.merge(
        persistence_f1,
        on=["target", "station", "aggregation", "period"],
        how="left",
        validate="many_to_one",
    )
    metric_frame["calibrated_f1_difference_vs_calibrated_persistence"] = (
        metric_frame["calibrated_f1"] - metric_frame["calibrated_persistence_f1"]
    )
    return pd.DataFrame(sweep_rows), pd.DataFrame(selection_rows), metric_frame


def _supervised(observations, config, station, target):
    return build_supervised(
        observations.loc[observations["station"] == station].copy(),
        target,
        config.history_months,
        config.horizon_months,
        config.meteorological_features,
    )


def _prediction_frame(frame, target, station, model, prediction, **metadata):
    output = pd.DataFrame(
        {
            "target": target,
            "station": station,
            "model": model,
            "issue_date": frame["issue_date"].to_numpy(),
            "target_date": frame["target_date"].to_numpy(),
            "horizon_months": 1,
            "y_true": frame["y_true"].to_numpy(dtype=float),
            "y_pred": np.asarray(prediction, dtype=float),
        }
    )
    for key, value in metadata.items():
        output[key] = value
    return output


def _alert_panel(station, frame, prediction):
    return pd.DataFrame(
        {
            "station": station,
            "y_true": frame["y_true"].to_numpy(dtype=float),
            "y_pred": np.asarray(prediction, dtype=float),
        }
    )


def _select_alert_threshold(eligible_rows, all_rows, tolerance=1e-12):
    """Apply deterministic, tolerance-aware validation-only tie-breaking."""
    if not eligible_rows:
        return next(
            row for row in all_rows if np.isclose(row["decision_threshold"], -1.0)
        )
    best = eligible_rows[0]
    descending = (
        "macro_validation_f1",
        "worst_station_validation_f1",
        "macro_validation_balanced_accuracy",
    )
    for candidate in eligible_rows[1:]:
        decision = 0
        for key in descending:
            difference = candidate[key] - best[key]
            if abs(difference) > tolerance:
                decision = 1 if difference > 0 else -1
                break
        if decision == 0:
            candidate_distance = abs(candidate["decision_threshold"] + 1.0)
            best_distance = abs(best["decision_threshold"] + 1.0)
            if abs(candidate_distance - best_distance) > tolerance:
                decision = 1 if candidate_distance < best_distance else -1
            elif candidate["decision_threshold"] < best["decision_threshold"]:
                decision = 1
        if decision > 0:
            best = candidate
    return best


def _alert_metric_rows(
    panel,
    target,
    model,
    primary_model_context,
    calibration_for,
    decision_threshold,
    period,
    interpretation_status,
):
    station_rows = []
    for station, group in panel.groupby("station", sort=True):
        station_rows.append(
            _alert_metric_row(
                group,
                target,
                model,
                primary_model_context,
                calibration_for,
                station,
                "station",
                decision_threshold,
                period,
                interpretation_status,
            )
        )
    macro = {
        "target": target,
        "model": model,
        "primary_model_context": primary_model_context,
        "calibration_for": calibration_for,
        "station": "__macro__",
        "aggregation": "equal_station_macro",
        "period": period,
        "observed_threshold": -1.0,
        "decision_threshold": decision_threshold,
        "interpretation_status": interpretation_status,
    }
    for key in (
        "default_f1",
        "calibrated_f1",
        "f1_change",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
    ):
        macro[key] = float(np.mean([row[key] for row in station_rows]))
    for key in (
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "observed_support",
        "predicted_support",
    ):
        macro[key] = int(np.sum([row[key] for row in station_rows]))
    macro["n"] = int(np.sum([row["n"] for row in station_rows]))
    macro["observed_rate"] = float(
        np.mean([row["observed_rate"] for row in station_rows])
    )
    macro["predicted_rate"] = float(
        np.mean([row["predicted_rate"] for row in station_rows])
    )
    pooled = _alert_metric_row(
        panel,
        target,
        model,
        primary_model_context,
        calibration_for,
        "__pooled__",
        "pooled",
        decision_threshold,
        period,
        interpretation_status,
    )
    return [*station_rows, macro, pooled]


def _alert_metric_row(
    group,
    target,
    model,
    primary_model_context,
    calibration_for,
    station,
    aggregation,
    decision_threshold,
    period,
    interpretation_status,
):
    default = alert_metrics(
        group["y_true"], group["y_pred"], -1.0, -1.0, zero_division=0.0
    )
    calibrated = alert_metrics(
        group["y_true"],
        group["y_pred"],
        -1.0,
        decision_threshold,
        zero_division=0.0,
    )
    return {
        "target": target,
        "model": model,
        "primary_model_context": primary_model_context,
        "calibration_for": calibration_for,
        "station": station,
        "aggregation": aggregation,
        "period": period,
        "observed_threshold": -1.0,
        "decision_threshold": decision_threshold,
        "default_f1": default["f1"],
        "calibrated_f1": calibrated["f1"],
        "f1_change": calibrated["f1"] - default["f1"],
        "true_positive": calibrated["true_positive"],
        "false_positive": calibrated["false_positive"],
        "false_negative": calibrated["false_negative"],
        "true_negative": calibrated["true_negative"],
        "precision": calibrated["precision"],
        "recall": calibrated["recall"],
        "specificity": calibrated["specificity"],
        "balanced_accuracy": calibrated["balanced_accuracy"],
        "observed_support": calibrated["observed_support"],
        "predicted_support": calibrated["predicted_support"],
        "n": int(len(group)),
        "observed_rate": float(calibrated["observed_support"] / len(group)),
        "predicted_rate": float(calibrated["predicted_support"] / len(group)),
        "interpretation_status": interpretation_status,
    }


def _scenario_metrics(predictions, scenario_column):
    by_task_frames = []
    macro_frames = []
    for scenario, group in predictions.groupby(scenario_column, sort=True):
        by_task, macro = metrics_table(group)
        by_task.insert(0, scenario_column, scenario)
        macro.insert(0, scenario_column, scenario)
        by_task_frames.append(by_task)
        macro_frames.append(macro)
    return pd.concat(by_task_frames, ignore_index=True), pd.concat(
        macro_frames, ignore_index=True
    )


def _dated_csv(frame, path):
    output = frame.copy()
    for column in (
        "issue_date",
        "target_date",
        "fit_end",
        "evaluation_start",
        "evaluation_end",
    ):
        if column in output:
            output[column] = output[column].dt.strftime("%Y-%m")
    output.to_csv(path, index=False)


def _month(value: str) -> pd.Timestamp:
    return pd.Period(value, freq="M").to_timestamp()


def _resolved_config(benchmark_config, robustness_config):
    benchmark = asdict(benchmark_config)
    benchmark["repository_root"] = "."
    benchmark["config_path"] = benchmark_config.config_path.relative_to(
        benchmark_config.repository_root
    ).as_posix()
    benchmark["source_dir"] = benchmark_config.source_dir.relative_to(
        benchmark_config.repository_root
    ).as_posix()
    benchmark["config_sha256"] = benchmark_config.config_sha256
    robustness = asdict(robustness_config)
    robustness["config_path"] = robustness_config.config_path.relative_to(
        benchmark_config.repository_root
    ).as_posix()
    return {
        "benchmark": benchmark,
        "primary_models": {target: spec[0] for target, spec in PRIMARY_MODELS.items()},
        "robustness": robustness,
    }


def _portable_path(path: Path, repository_root: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return path.name


def _run_git(root: Path, arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _validate_benchmark_for_robustness(config):
    if set(config.targets) != set(PRIMARY_MODELS):
        raise ValueError("Robustness v1 requires exactly SPEI-3 and SPEI-6 targets")
    if not (
        _month(config.train_end)
        < _month(config.validation_end)
        < _month(config.test_end)
    ):
        raise ValueError("Benchmark date boundaries must be strictly chronological")


def _package_versions():
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return packages
