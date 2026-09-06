import numpy as np
import pandas as pd
import pytest

from spei_forecast.bootstrap import circular_block_indices, paired_mae_bootstrap


def test_circular_blocks_are_reproducible_and_contiguous():
    first = circular_block_indices(24, 6, 10, 42)
    second = circular_block_indices(24, 6, 10, 42)
    np.testing.assert_array_equal(first, second)
    for row in first:
        for start in range(0, 24, 6):
            block = row[start : start + 6]
            np.testing.assert_array_equal((block[0] + np.arange(6)) % 24, block)


def test_paired_bootstrap_detects_uniform_improvement():
    dates = pd.date_range("2000-01-01", periods=24, freq="MS")
    rows = []
    for station in ("A", "B"):
        truth = np.linspace(-1, 1, len(dates))
        for model, error in (("persistence", 1.0), ("better", 0.5)):
            for date, value in zip(dates, truth):
                rows.append(
                    {
                        "target": "spei3",
                        "station": station,
                        "model": model,
                        "target_date": date,
                        "y_true": value,
                        "y_pred": value + error,
                    }
                )
    result = paired_mae_bootstrap(pd.DataFrame(rows), 200, 6, 42)
    better = result.loc[result["model"] == "better"].iloc[0]
    assert better["point_estimate"] == 0.5
    assert better["ci_2_5"] > 0


def test_paired_bootstrap_rejects_mismatched_truth_panels():
    dates = pd.date_range("2000-01-01", periods=12, freq="MS")
    rows = []
    for model in ("persistence", "candidate"):
        for date_index, date in enumerate(dates):
            rows.append(
                {
                    "target": "spei3",
                    "station": "A",
                    "model": model,
                    "target_date": date,
                    "y_true": float(
                        date_index + (model == "candidate" and date_index == 0)
                    ),
                    "y_pred": float(date_index + 1),
                }
            )
    with pytest.raises(ValueError, match="Truth panel"):
        paired_mae_bootstrap(pd.DataFrame(rows), 20, 3, 42)
