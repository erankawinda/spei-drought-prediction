from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("Metric arrays must be aligned one-dimensional vectors")
    if not (np.isfinite(truth).all() and np.isfinite(prediction).all()):
        raise ValueError("Metrics reject missing or non-finite values")
    variance = float(np.var(truth))
    pearson = (
        float(np.corrcoef(truth, prediction)[0, 1])
        if variance > 0 and float(np.var(prediction)) > 0
        else float("nan")
    )
    return {
        "n": int(len(truth)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)) if variance > 0 else float("nan"),
        "pearson_r": pearson,
    }


def event_metrics(y_true, y_pred, threshold: float, prefix: str) -> dict[str, float]:
    observed = np.asarray(y_true, dtype=float) <= threshold
    predicted = np.asarray(y_pred, dtype=float) <= threshold
    true_positive = int(np.sum(observed & predicted))
    false_positive = int(np.sum(~observed & predicted))
    false_negative = int(np.sum(observed & ~predicted))
    true_negative = int(np.sum(~observed & ~predicted))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    balanced_accuracy = (
        float((recall + specificity) / 2)
        if np.isfinite(recall) and np.isfinite(specificity)
        else float("nan")
    )
    return {
        f"{prefix}_threshold": float(threshold),
        f"{prefix}_support": int(observed.sum()),
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_balanced_accuracy": balanced_accuracy,
    }


def alert_metrics(
    y_true,
    y_pred,
    observed_threshold: float,
    decision_threshold: float,
    zero_division: float = float("nan"),
) -> dict[str, float]:
    """Score an alert cutoff without changing the physical event definition."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("Alert arrays must be aligned one-dimensional vectors")
    if not (np.isfinite(truth).all() and np.isfinite(prediction).all()):
        raise ValueError("Alert metrics reject missing or non-finite values")
    observed = truth <= observed_threshold
    predicted = prediction <= decision_threshold
    true_positive = int(np.sum(observed & predicted))
    false_positive = int(np.sum(~observed & predicted))
    false_negative = int(np.sum(observed & ~predicted))
    true_negative = int(np.sum(~observed & ~predicted))

    def divide(numerator, denominator):
        return float(numerator / denominator) if denominator else float(zero_division)

    precision = divide(true_positive, true_positive + false_positive)
    recall = divide(true_positive, true_positive + false_negative)
    specificity = divide(true_negative, true_negative + false_positive)
    f1 = divide(2 * precision * recall, precision + recall)
    return {
        "observed_threshold": float(observed_threshold),
        "decision_threshold": float(decision_threshold),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "observed_support": int(observed.sum()),
        "predicted_support": int(predicted.sum()),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": float((recall + specificity) / 2),
    }


def metrics_table(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for (target, station, model), group in predictions.groupby(
        ["target", "station", "model"], sort=True
    ):
        row = {"target": target, "station": station, "model": model}
        row.update(regression_metrics(group["y_true"], group["y_pred"]))
        row.update(event_metrics(group["y_true"], group["y_pred"], -1.0, "drought"))
        row.update(event_metrics(group["y_true"], group["y_pred"], -1.5, "severe"))
        rows.append(row)
    by_task = pd.DataFrame(rows).sort_values(["target", "model", "station"])

    mean_columns = [
        "mae",
        "rmse",
        "r2",
        "pearson_r",
        "drought_precision",
        "drought_recall",
        "drought_f1",
        "drought_balanced_accuracy",
        "severe_precision",
        "severe_recall",
        "severe_f1",
        "severe_balanced_accuracy",
    ]
    macro = (
        by_task.groupby(["target", "model"], as_index=False)[mean_columns]
        .mean()
        .sort_values(["target", "mae", "model"])
    )
    support = by_task.groupby(["target", "model"], as_index=False)[
        ["n", "drought_support", "severe_support"]
    ].sum()
    macro = macro.merge(support, on=["target", "model"], how="left")
    valid_counts = (
        by_task.groupby(["target", "model"])[["drought_f1", "severe_f1"]]
        .count()
        .rename(
            columns={
                "drought_f1": "drought_f1_valid_station_count",
                "severe_f1": "severe_f1_valid_station_count",
            }
        )
        .reset_index()
    )
    macro = macro.merge(valid_counts, on=["target", "model"], how="left")

    persistence = macro.loc[
        macro["model"] == "persistence", ["target", "mae", "rmse", "drought_f1"]
    ].rename(
        columns={
            "mae": "persistence_mae",
            "rmse": "persistence_rmse",
            "drought_f1": "persistence_drought_f1",
        }
    )
    macro = macro.merge(persistence, on="target", how="left")
    macro["mae_skill_vs_persistence"] = 1 - macro["mae"] / macro["persistence_mae"]
    macro["rmse_skill_vs_persistence"] = 1 - macro["rmse"] / macro["persistence_rmse"]
    macro["drought_f1_change_vs_persistence"] = (
        macro["drought_f1"] - macro["persistence_drought_f1"]
    )
    return by_task.reset_index(drop=True), macro.reset_index(drop=True)


def acceptance_table(
    by_task: pd.DataFrame, macro: pd.DataFrame, intervals: pd.DataFrame
) -> pd.DataFrame:
    station_mae = by_task.pivot(
        index=["target", "station"], columns="model", values="mae"
    )
    station_skill_rows = []
    for model in station_mae.columns:
        skills = 1 - station_mae[model] / station_mae["persistence"]
        for target, values in skills.groupby(level="target"):
            station_skill_rows.append(
                {
                    "target": target,
                    "model": model,
                    "worst_station_mae_skill": float(values.min()),
                }
            )
    worst_station = pd.DataFrame(station_skill_rows)

    selected_columns = [
        "target",
        "model",
        "mae",
        "mae_skill_vs_persistence",
        "drought_f1",
        "drought_f1_change_vs_persistence",
    ]
    acceptance = macro[selected_columns].merge(
        intervals[
            [
                "target",
                "model",
                "ci_2_5",
                "ci_97_5",
                "passes_5pct_positive_ci_gate",
            ]
        ],
        on=["target", "model"],
        how="left",
    )
    acceptance = acceptance.merge(worst_station, on=["target", "model"], how="left")
    acceptance["passes_drought_f1_gate"] = (
        acceptance["drought_f1_change_vs_persistence"] >= -0.02
    )
    acceptance["passes_station_gate"] = acceptance["worst_station_mae_skill"] >= 0
    acceptance["passes_full_gate"] = (
        acceptance["passes_5pct_positive_ci_gate"]
        & acceptance["passes_drought_f1_gate"]
        & acceptance["passes_station_gate"]
    )
    return acceptance.sort_values(["target", "mae", "model"]).reset_index(drop=True)
