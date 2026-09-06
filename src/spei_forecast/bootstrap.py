from __future__ import annotations

import numpy as np
import pandas as pd


def circular_block_indices(
    n_months: int, block_months: int, resamples: int, seed: int
) -> np.ndarray:
    if n_months < 1 or block_months < 1 or resamples < 1:
        raise ValueError("Bootstrap dimensions must be positive")
    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(n_months / block_months))
    starts = rng.integers(0, n_months, size=(resamples, blocks_needed))
    offsets = np.arange(block_months)
    indices = (starts[..., None] + offsets) % n_months
    return indices.reshape(resamples, -1)[:, :n_months]


def paired_mae_bootstrap(
    predictions: pd.DataFrame,
    resamples: int,
    block_months: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for target, target_rows in predictions.groupby("target", sort=True):
        dates = np.array(sorted(target_rows["target_date"].unique()))
        stations = sorted(target_rows["station"].unique())
        index = circular_block_indices(len(dates), block_months, resamples, seed)

        truth = _matrix(target_rows, "y_true", "persistence", dates, stations)
        persistence = _matrix(target_rows, "y_pred", "persistence", dates, stations)
        persistence_errors = np.abs(truth - persistence)
        persistence_point = float(persistence_errors.mean(axis=0).mean())
        if persistence_point == 0:
            raise ValueError("Relative skill is undefined for zero-error persistence")

        for model in sorted(target_rows["model"].unique()):
            model_truth = _matrix(target_rows, "y_true", model, dates, stations)
            if not np.array_equal(model_truth, truth):
                raise ValueError(f"Truth panel for {model} differs from persistence")
            model_prediction = _matrix(target_rows, "y_pred", model, dates, stations)
            model_errors = np.abs(truth - model_prediction)
            point = float(persistence_point - model_errors.mean(axis=0).mean())

            sampled_persistence = persistence_errors[index, :].mean(axis=1).mean(axis=1)
            sampled_model = model_errors[index, :].mean(axis=1).mean(axis=1)
            improvement = sampled_persistence - sampled_model
            lower, upper = np.percentile(improvement, [2.5, 97.5])
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "metric": "macro_mae_improvement_vs_persistence",
                    "point_estimate": point,
                    "ci_2_5": float(lower),
                    "ci_97_5": float(upper),
                    "relative_improvement": float(point / persistence_point),
                    "resamples": int(resamples),
                    "block_months": int(block_months),
                    "passes_5pct_positive_ci_gate": bool(
                        point / persistence_point >= 0.05 and lower > 0
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "model"]).reset_index(drop=True)


def _matrix(rows, value_column, model, dates, stations) -> np.ndarray:
    selected = rows.loc[rows["model"] == model]
    pivot = selected.pivot(index="target_date", columns="station", values=value_column)
    pivot = pivot.reindex(index=dates, columns=stations)
    if pivot.isna().any().any():
        raise ValueError(f"Incomplete synchronized prediction panel for {model}")
    return pivot.to_numpy(dtype=float)
