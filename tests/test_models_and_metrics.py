import numpy as np
import pandas as pd
import pytest

from spei_forecast.metrics import (
    alert_metrics,
    event_metrics,
    metrics_table,
    regression_metrics,
)
from spei_forecast.models import make_ridge, monthly_climatology_predictions


def test_monthly_climatology_uses_training_values_only():
    train = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2000-01-01", "2001-01-01", "2000-02-01"]),
            "y_true": [1.0, 3.0, -2.0],
        }
    )
    evaluation = pd.DataFrame(
        {"target_date": pd.to_datetime(["2010-01-01", "2010-02-01"])}
    )
    assert monthly_climatology_predictions(train, evaluation).tolist() == [2.0, -2.0]


def test_ridge_scaler_is_fit_only_to_supplied_training_rows():
    x_train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y_train = np.array([0.0, 1.0, 2.0])
    model = make_ridge(1.0).fit(x_train, y_train)
    assert model.named_steps["standardize"].mean_[0] == pytest.approx(1.0)


def test_metrics_reject_invalid_values_and_handle_constant_targets():
    result = regression_metrics([1.0, 1.0], [1.0, 2.0])
    assert np.isnan(result["r2"])
    assert np.isnan(result["pearson_r"])
    with pytest.raises(ValueError, match="non-finite"):
        regression_metrics([1.0, np.nan], [1.0, 2.0])


def test_drought_event_metric_counts_are_auditable():
    result = event_metrics([-2.0, -1.2, 0.0, 1.0], [-1.8, 0.0, -1.1, 1.0], -1.0, "d")
    assert result["d_support"] == 2
    assert result["d_precision"] == pytest.approx(0.5)
    assert result["d_recall"] == pytest.approx(0.5)
    assert result["d_f1"] == pytest.approx(0.5)


def test_alert_cutoff_does_not_redefine_observed_drought():
    result = alert_metrics(
        [-1.2, -0.8, 0.0],
        [-0.7, -0.6, 0.2],
        observed_threshold=-1.0,
        decision_threshold=-0.5,
        zero_division=0.0,
    )
    assert result["observed_support"] == 1
    assert result["predicted_support"] == 2
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1


@pytest.mark.parametrize(
    "truth,prediction",
    [
        ([-2.0, 0.0], [0.0, 0.0]),  # All observed events missed.
        ([0.0, 0.0], [-2.0, 0.0]),  # False alerts without observed events.
        ([-2.0, 0.0], [0.0, -2.0]),  # Misses and false alerts together.
    ],
)
def test_event_f1_is_zero_when_events_exist_but_none_are_correct(truth, prediction):
    assert event_metrics(truth, prediction, -1.0, "drought")["drought_f1"] == 0.0
    assert alert_metrics(truth, prediction, -1.0, -1.0)["f1"] == 0.0
    # The configurable fallback applies only to a zero F1 denominator.
    assert alert_metrics(truth, prediction, -1.0, -1.0, zero_division=1.0)["f1"] == 0.0


def test_event_f1_is_undefined_only_when_no_events_are_observed_or_predicted():
    truth, prediction = [0.0, 1.0], [0.5, 0.5]
    assert np.isnan(event_metrics(truth, prediction, -1.0, "drought")["drought_f1"])
    assert np.isnan(alert_metrics(truth, prediction, -1.0, -1.0)["f1"])
    assert alert_metrics(truth, prediction, -1.0, -1.0, zero_division=0.0)["f1"] == 0.0


def test_macro_event_f1_includes_failed_station_as_zero():
    rows = []
    for station, truth, prediction in [
        ("perfect", [-2.0, 0.0], [-2.0, 0.0]),
        ("missed", [-2.0, 0.0], [0.0, -2.0]),
        ("no_events", [0.0, 0.0], [0.0, 0.0]),
    ]:
        for model in ["persistence", "candidate"]:
            for observed, predicted in zip(truth, prediction):
                rows.append(
                    {
                        "target": "spei3",
                        "station": station,
                        "model": model,
                        "y_true": observed,
                        "y_pred": predicted,
                    }
                )
    by_task, macro = metrics_table(pd.DataFrame(rows))
    assert by_task.loc[by_task["station"] == "missed", "drought_f1"].eq(0.0).all()
    assert macro["drought_f1"].eq(0.5).all()
    assert macro["drought_f1_valid_station_count"].eq(2).all()
