from pathlib import Path

import pandas as pd

from spei_forecast.config import load_config
from spei_forecast.data import load_all
from spei_forecast.features import (
    FEATURE_SET_COMBINED,
    build_supervised,
    feature_columns,
)
from spei_forecast.splits import split_frame


ROOT = Path(__file__).resolve().parents[1]


def _station_data():
    config = load_config(ROOT / "configs" / "benchmark_v1.toml")
    observations, _ = load_all(config)
    return config, observations.loc[observations["station"] == "Buttala"].copy()


def test_horizon_alignment_and_fixed_partition_counts():
    config, station = _station_data()
    expected_train = {"spei3": 982, "spei6": 979}
    for target in config.targets:
        supervised = build_supervised(
            station,
            target,
            config.history_months,
            config.horizon_months,
            config.meteorological_features,
        )
        assert (
            supervised["issue_date"] + pd.offsets.MonthBegin(1)
            == supervised["target_date"]
        ).all()
        partitions = split_frame(supervised, config)
        assert len(partitions["train"]) == expected_train[target]
        assert len(partitions["validation"]) == 180
        assert len(partitions["test"]) == 240
        assert partitions["train"]["target_date"].max() < partitions["validation"][
            "target_date"
        ].min()
        assert partitions["validation"]["target_date"].max() < partitions["test"][
            "target_date"
        ].min()


def test_future_meteorology_cannot_change_earlier_feature_rows():
    config, station = _station_data()
    original = build_supervised(
        station,
        "spei3",
        config.history_months,
        config.horizon_months,
        config.meteorological_features,
    )
    changed = station.copy()
    changed.loc[changed.index[-1], "mean_prep"] = 999999.0
    after = build_supervised(
        changed,
        "spei3",
        config.history_months,
        config.horizon_months,
        config.meteorological_features,
    )
    columns = feature_columns(
        FEATURE_SET_COMBINED,
        "spei3",
        config.history_months,
        config.meteorological_features,
    )
    pd.testing.assert_frame_equal(original[columns], after[columns])


def test_feature_allowlist_excludes_redundant_and_unverified_columns():
    config, _ = _station_data()
    columns = feature_columns(
        FEATURE_SET_COMBINED,
        "spei3",
        config.history_months,
        config.meteorological_features,
    )
    assert not any("max_tmp" in column for column in columns)
    assert not any("min_tmp" in column for column in columns)
    assert not any(column.startswith("pet_") for column in columns)
