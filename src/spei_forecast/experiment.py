from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from itertools import product
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .bootstrap import paired_mae_bootstrap
from .config import BenchmarkConfig
from .data import load_all
from .features import (
    FEATURE_SET_AR,
    FEATURE_SET_COMBINED,
    FEATURE_SET_MET,
    build_supervised,
    feature_columns,
)
from .metrics import acceptance_table, metrics_table
from .models import make_hgb, make_ridge, monthly_climatology_predictions
from .provenance import implementation_hashes, json_sha256
from .splits import split_frame


RIDGE_FEATURE_SETS = (FEATURE_SET_AR, FEATURE_SET_MET, FEATURE_SET_COMBINED)


def run_benchmark(config: BenchmarkConfig, output_dir: str | Path) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    observations, quality = load_all(config)
    prepared = _prepare_tasks(observations, config)

    selection = _select_models(prepared, config)
    predictions = _evaluate_fixed_period(prepared, selection, config)
    by_task, macro = metrics_table(predictions)
    intervals = paired_mae_bootstrap(
        predictions,
        resamples=config.bootstrap_resamples,
        block_months=config.bootstrap_block_months,
        seed=config.random_seed,
    )
    acceptance = acceptance_table(by_task, macro, intervals)

    predictions_to_write = predictions.copy()
    predictions_to_write["issue_date"] = predictions_to_write["issue_date"].dt.strftime(
        "%Y-%m"
    )
    predictions_to_write["target_date"] = predictions_to_write[
        "target_date"
    ].dt.strftime("%Y-%m")
    predictions_to_write.to_csv(output / "predictions.csv", index=False)
    by_task.to_csv(output / "metrics_by_task.csv", index=False)
    macro.to_csv(output / "metrics_macro.csv", index=False)
    intervals.to_csv(output / "bootstrap_intervals.csv", index=False)
    acceptance.to_csv(output / "acceptance.csv", index=False)
    _write_json(output / "selection.json", selection)
    _write_json(output / "data_quality.json", quality)
    _write_json(output / "config_resolved.json", config_as_dict(config))

    manifest = _manifest(config, quality, output)
    _write_json(output / "manifest.json", manifest)
    return {
        "output_dir": str(output),
        "metrics_macro": macro,
        "bootstrap_intervals": intervals,
        "acceptance": acceptance,
        "manifest": manifest,
    }


def _prepare_tasks(observations: pd.DataFrame, config: BenchmarkConfig) -> dict:
    prepared: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    for target in config.targets:
        prepared[target] = {}
        for station in sorted(config.station_files):
            station_frame = observations.loc[observations["station"] == station].copy()
            supervised = build_supervised(
                station_frame=station_frame,
                target=target,
                history_months=config.history_months,
                horizon_months=config.horizon_months,
                meteorological_features=config.meteorological_features,
            )
            prepared[target][station] = split_frame(supervised, config)
    return prepared


def _select_models(prepared: dict, config: BenchmarkConfig) -> dict:
    selection: dict[str, dict] = {
        "selection_partition": "validation",
        "test_partition_used_for_selection": False,
        "targets": {},
    }
    for target, stations in prepared.items():
        target_selection: dict[str, dict] = {
            "candidate_scores_macro_validation_mae": {},
            "chosen": {},
        }

        baseline_scores = {}
        for model_name in (
            "zero",
            "monthly_climatology",
            "persistence",
            "seasonal_persistence",
        ):
            station_scores = []
            for split in stations.values():
                train, validation = split["train"], split["validation"]
                if model_name == "zero":
                    prediction = np.zeros(len(validation))
                elif model_name == "monthly_climatology":
                    prediction = monthly_climatology_predictions(train, validation)
                elif model_name == "persistence":
                    prediction = validation[f"{target}_lag_1"].to_numpy()
                else:
                    prediction = validation[f"{target}_lag_12"].to_numpy()
                station_scores.append(
                    mean_absolute_error(validation["y_true"], prediction)
                )
            baseline_scores[model_name] = float(np.mean(station_scores))
        target_selection["baseline_validation_mae"] = baseline_scores

        for feature_set in RIDGE_FEATURE_SETS:
            scores = []
            columns = feature_columns(
                feature_set,
                target,
                config.history_months,
                config.meteorological_features,
            )
            for alpha in config.ridge_alphas:
                score = _candidate_score(
                    stations,
                    columns,
                    lambda: make_ridge(alpha),
                )
                scores.append({"alpha": alpha, "macro_validation_mae": score})
            chosen = min(
                scores, key=lambda row: (row["macro_validation_mae"], -row["alpha"])
            )
            model_name = f"ridge_{feature_set}"
            target_selection["candidate_scores_macro_validation_mae"][model_name] = (
                scores
            )
            target_selection["chosen"][model_name] = chosen

        hgb_scores = []
        hgb_columns = feature_columns(
            FEATURE_SET_COMBINED,
            target,
            config.history_months,
            config.meteorological_features,
        )
        for values in product(
            config.hgb_learning_rates,
            config.hgb_max_iters,
            config.hgb_max_leaf_nodes,
            config.hgb_min_samples_leaf,
            config.hgb_l2_regularization,
        ):
            params = dict(
                zip(
                    (
                        "learning_rate",
                        "max_iter",
                        "max_leaf_nodes",
                        "min_samples_leaf",
                        "l2_regularization",
                    ),
                    values,
                )
            )
            score = _candidate_score(
                stations,
                hgb_columns,
                lambda params=params: make_hgb(params, config.random_seed),
            )
            hgb_scores.append({**params, "macro_validation_mae": score})
        chosen_hgb = min(
            hgb_scores,
            key=lambda row: (
                row["macro_validation_mae"],
                row["max_leaf_nodes"],
                row["max_iter"],
                -row["min_samples_leaf"],
                -row["l2_regularization"],
                row["learning_rate"],
            ),
        )
        target_selection["candidate_scores_macro_validation_mae"]["hgb_ar24_met24"] = (
            hgb_scores
        )
        target_selection["chosen"]["hgb_ar24_met24"] = chosen_hgb
        selection["targets"][target] = target_selection
    return selection


def _candidate_score(stations: dict, columns: list[str], estimator_factory) -> float:
    station_scores = []
    for split in stations.values():
        train, validation = split["train"], split["validation"]
        estimator = estimator_factory()
        estimator.fit(train[columns], train["y_true"])
        prediction = estimator.predict(validation[columns])
        station_scores.append(mean_absolute_error(validation["y_true"], prediction))
    return float(np.mean(station_scores))


def _evaluate_fixed_period(
    prepared: dict, selection: dict, config: BenchmarkConfig
) -> pd.DataFrame:
    prediction_rows: list[pd.DataFrame] = []
    for target, stations in prepared.items():
        chosen = selection["targets"][target]["chosen"]
        for station, split in stations.items():
            development = pd.concat(
                [split["train"], split["validation"]], ignore_index=True
            )
            test = split["test"]
            baseline_predictions = {
                "zero": np.zeros(len(test)),
                "monthly_climatology": monthly_climatology_predictions(
                    development, test
                ),
                "persistence": test[f"{target}_lag_1"].to_numpy(dtype=float),
                "seasonal_persistence": test[f"{target}_lag_12"].to_numpy(dtype=float),
            }
            for model_name, prediction in baseline_predictions.items():
                prediction_rows.append(
                    _prediction_frame(test, target, station, model_name, prediction)
                )

            for feature_set in RIDGE_FEATURE_SETS:
                model_name = f"ridge_{feature_set}"
                columns = feature_columns(
                    feature_set,
                    target,
                    config.history_months,
                    config.meteorological_features,
                )
                estimator = make_ridge(chosen[model_name]["alpha"])
                estimator.fit(development[columns], development["y_true"])
                prediction_rows.append(
                    _prediction_frame(
                        test,
                        target,
                        station,
                        model_name,
                        estimator.predict(test[columns]),
                    )
                )

            hgb_name = "hgb_ar24_met24"
            params = {
                key: value
                for key, value in chosen[hgb_name].items()
                if key != "macro_validation_mae"
            }
            hgb_columns = feature_columns(
                FEATURE_SET_COMBINED,
                target,
                config.history_months,
                config.meteorological_features,
            )
            estimator = make_hgb(params, config.random_seed)
            estimator.fit(development[hgb_columns], development["y_true"])
            prediction_rows.append(
                _prediction_frame(
                    test,
                    target,
                    station,
                    hgb_name,
                    estimator.predict(test[hgb_columns]),
                )
            )

    return (
        pd.concat(prediction_rows, ignore_index=True)
        .sort_values(["target", "model", "station", "target_date"])
        .reset_index(drop=True)
    )


def _prediction_frame(test, target, station, model, prediction) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": target,
            "station": station,
            "model": model,
            "issue_date": test["issue_date"].to_numpy(),
            "target_date": test["target_date"].to_numpy(),
            "horizon_months": 1,
            "y_true": test["y_true"].to_numpy(dtype=float),
            "y_pred": np.asarray(prediction, dtype=float),
        }
    )


def config_as_dict(config: BenchmarkConfig) -> dict:
    resolved = asdict(config)
    resolved["repository_root"] = "."
    resolved["config_path"] = config.config_path.relative_to(
        config.repository_root
    ).as_posix()
    resolved["source_dir"] = config.source_dir.relative_to(
        config.repository_root
    ).as_posix()
    resolved["config_sha256"] = config.config_sha256
    resolved["forecast_contract"] = {
        "forecast_origin": "end of issue month",
        "target": "same-station SPEI in the following month",
        "available_information": "observations through issue_date only",
        "test_model_update": "none; models are frozen after 1999-12",
    }
    return resolved


def _manifest(config: BenchmarkConfig, quality: dict, output: Path) -> dict:
    git_commit = _run_git(config.repository_root, ["rev-parse", "HEAD"])
    git_status = _run_git(config.repository_root, ["status", "--porcelain"])
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "benchmark": "benchmark_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": _portable_path(output, config.repository_root),
        "git_commit": git_commit,
        "git_worktree_dirty": bool(git_status),
        "git_status_porcelain": git_status.splitlines(),
        "config_sha256": config.config_sha256,
        "resolved_config_sha256": json_sha256(config_as_dict(config)),
        "implementation_files_sha256": implementation_hashes(),
        "source_files": quality["source_files"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "random_seed": config.random_seed,
        "test_used_for_model_selection": False,
        "evaluation_period": "2000-01/2019-12",
        "period_previously_inspected": True,
        "fresh_locked_test": False,
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


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value
