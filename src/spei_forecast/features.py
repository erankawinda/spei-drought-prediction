from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_SET_AR = "ar24"
FEATURE_SET_MET = "met24"
FEATURE_SET_COMBINED = "ar24_met24"


def feature_columns(
    feature_set: str,
    target: str,
    history_months: int,
    meteorological_features: tuple[str, ...],
) -> list[str]:
    target_lags = [f"{target}_lag_{lag}" for lag in range(1, history_months + 1)]
    met_lags = [
        f"{feature}_lag_{lag}"
        for feature in meteorological_features
        for lag in range(1, history_months + 1)
    ]
    calendar = ["target_month_sin", "target_month_cos"]
    if feature_set == FEATURE_SET_AR:
        return [*target_lags, *calendar]
    if feature_set == FEATURE_SET_MET:
        return [*met_lags, *calendar]
    if feature_set == FEATURE_SET_COMBINED:
        return [*target_lags, *met_lags, *calendar]
    raise ValueError(f"Unknown feature set: {feature_set}")


def build_supervised(
    station_frame: pd.DataFrame,
    target: str,
    history_months: int,
    horizon_months: int,
    meteorological_features: tuple[str, ...],
) -> pd.DataFrame:
    if horizon_months != 1:
        raise ValueError("Benchmark v1 supports only horizon_months=1")
    frame = station_frame.sort_values("month").reset_index(drop=True)
    expected = pd.date_range(frame["month"].min(), frame["month"].max(), freq="MS")
    if not frame["month"].equals(pd.Series(expected)):
        raise ValueError("Feature construction cannot bridge missing months")

    identifiers = pd.DataFrame(
        {
            "station": frame["station"],
            "issue_date": frame["month"] - pd.offsets.MonthBegin(horizon_months),
            "target_date": frame["month"],
            "y_true": frame[target],
        }
    )
    lagged: dict[str, pd.Series] = {}
    for lag in range(1, history_months + 1):
        lagged[f"{target}_lag_{lag}"] = frame[target].shift(lag)
    for feature in meteorological_features:
        for lag in range(1, history_months + 1):
            lagged[f"{feature}_lag_{lag}"] = frame[feature].shift(lag)

    month = identifiers["target_date"].dt.month
    calendar = pd.DataFrame(
        {
            "target_month_sin": np.sin(2 * np.pi * month / 12),
            "target_month_cos": np.cos(2 * np.pi * month / 12),
        }
    )
    output = pd.concat(
        [identifiers, pd.DataFrame(lagged, index=frame.index), calendar], axis=1
    )

    required = feature_columns(
        FEATURE_SET_COMBINED, target, history_months, meteorological_features
    )
    output = output.dropna(subset=["y_true", *required]).reset_index(drop=True)

    if not (
        output["issue_date"] + pd.offsets.MonthBegin(horizon_months)
        == output["target_date"]
    ).all():
        raise AssertionError("Issue-date/target-date alignment failed")
    return output
