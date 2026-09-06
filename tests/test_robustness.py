from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spei_forecast.config import load_config
from spei_forecast.data import load_all
from spei_forecast.features import build_supervised
from spei_forecast.robustness import (
    PRIMARY_MODELS,
    bootstrap_block_sensitivity,
    event_threshold_diagnostics,
    load_robustness_config,
    rolling_inner_folds,
    rolling_origin_diagnostics,
    rolling_outer_folds,
    run_robustness,
    _select_alert_threshold,
    _phase1_regression_selection,
    _validate_robustness_config,
    start_date_sensitivity,
)


ROOT = Path(__file__).resolve().parents[1]


def _configs():
    benchmark = load_config(ROOT / "configs" / "benchmark_v1.toml")
    benchmark = replace(benchmark, ridge_alphas=(1.0, 100.0))
    robustness = load_robustness_config(ROOT / "configs" / "robustness_v1.toml")
    robustness = replace(robustness, bootstrap_resamples=50)
    return benchmark, robustness


def test_nested_rolling_folds_are_nonoverlapping_and_trailing():
    _, robustness = _configs()
    outer = rolling_outer_folds(robustness)
    assert [fold["evaluation_start"].strftime("%Y-%m") for fold in outer] == [
        "2000-01",
        "2005-01",
        "2010-01",
        "2015-01",
    ]
    for fold in outer:
        inner = rolling_inner_folds(fold["evaluation_start"], robustness)
        assert len(inner) == 3
        assert all(item["validation_end"] < fold["evaluation_start"] for item in inner)
        assert (
            inner[-1]["validation_end"] + pd.offsets.MonthBegin(1)
            == fold["evaluation_start"]
        )


def test_history_cutoff_precedes_lag_construction():
    benchmark, _ = _configs()
    observations, _ = load_all(benchmark)
    cropped = observations.loc[observations["month"] >= pd.Timestamp("1971-01-01")]
    for station_name in benchmark.station_files:
        station = cropped.loc[cropped["station"] == station_name]
        for target in benchmark.targets:
            supervised = build_supervised(
                station,
                target,
                benchmark.history_months,
                benchmark.horizon_months,
                benchmark.meteorological_features,
            )
            assert supervised["target_date"].min() == pd.Timestamp("1973-01-01")
            assert (
                supervised["issue_date"] + pd.offsets.MonthBegin(1)
                == supervised["target_date"]
            ).all()
            first = supervised.iloc[0]
            source = station.set_index("month")
            for lag in (1, 12, 24):
                expected = source.loc[
                    first["target_date"] - pd.DateOffset(months=lag), target
                ]
                assert first[f"{target}_lag_{lag}"] == expected


def test_rolling_outputs_cover_each_retrospective_month_once():
    benchmark, robustness = _configs()
    observations, _ = load_all(benchmark)
    predictions, selection = rolling_origin_diagnostics(
        observations, benchmark, robustness
    )
    assert len(selection) == 2 * 4
    assert not selection["outer_outcomes_used_for_selection"].any()
    for target, (model, _) in PRIMARY_MODELS.items():
        subset = predictions.loc[
            (predictions["target"] == target) & (predictions["model"] == model)
        ]
        assert len(subset) == 3 * 240
        assert not subset.duplicated(["station", "target_date"]).any()
        assert subset["target_date"].min() == pd.Timestamp("2000-01-01")
        assert subset["target_date"].max() == pd.Timestamp("2019-12-01")
    fold = predictions.loc[predictions["outer_fold"] == "2005-01_2009-12"]
    from spei_forecast.metrics import metrics_table

    _, fold_macro = metrics_table(fold)
    severe = fold_macro.loc[
        (fold_macro["target"] == "spei6")
        & (fold_macro["model"] == PRIMARY_MODELS["spei6"][0])
    ].iloc[0]
    assert severe["severe_support"] == 0
    assert severe["severe_f1_valid_station_count"] == 0
    assert np.isnan(severe["severe_f1"])


def test_outer_selection_is_invariant_to_its_own_future_outcomes():
    benchmark, robustness = _configs()
    observations, _ = load_all(benchmark)
    _, baseline = rolling_origin_diagnostics(observations, benchmark, robustness)
    for fold in rolling_outer_folds(robustness):
        changed = observations.copy()
        future = changed["month"] >= fold["evaluation_start"]
        changed.loc[future, ["spei3", "spei6"]] += 50.0
        _, selected = rolling_origin_diagnostics(changed, benchmark, robustness)
        columns = ["target", "outer_fold", "alpha"]
        expected = baseline.loc[baseline["outer_fold"] == fold["name"], columns]
        actual = selected.loc[selected["outer_fold"] == fold["name"], columns]
        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True), actual.reset_index(drop=True)
        )


def test_start_sensitivity_reselects_and_preserves_persistence_panel():
    benchmark, robustness = _configs()
    observations, _ = load_all(benchmark)
    predictions, selection = start_date_sensitivity(observations, benchmark, robustness)
    assert len(selection) == 3 * 2 * 4
    assert not selection["outer_outcomes_used_for_selection"].any()
    for target in benchmark.targets:
        panels = []
        for start_year in robustness.training_start_years:
            panel = predictions.loc[
                (predictions["target"] == target)
                & (predictions["model"] == "persistence")
                & (predictions["training_start_year"] == start_year),
                ["station", "target_date", "y_true", "y_pred"],
            ].reset_index(drop=True)
            panels.append(panel)
        pd.testing.assert_frame_equal(panels[0], panels[1])
        pd.testing.assert_frame_equal(panels[0], panels[2])


def test_block_sensitivity_changes_intervals_not_point_estimates():
    benchmark, robustness = _configs()
    observations, _ = load_all(benchmark)
    predictions, _ = rolling_origin_diagnostics(observations, benchmark, robustness)
    result = bootstrap_block_sensitivity(predictions, benchmark, robustness)
    for _, group in result.groupby(["target", "model"]):
        assert group["point_estimate"].max() == group["point_estimate"].min()
        assert set(group["block_months"]) == {6, 12, 24}
        assert group["outer_fold_bounded"].all()


def test_threshold_selection_is_validation_only():
    benchmark, robustness = _configs()
    observations, _ = load_all(benchmark)
    selection = _phase1_regression_selection(observations, benchmark)
    _, chosen, metrics = event_threshold_diagnostics(
        observations, selection, benchmark, robustness
    )
    changed = observations.copy()
    changed.loc[changed["month"] >= pd.Timestamp("2000-01-01"), ["spei3", "spei6"]] += (
        50
    )
    _, changed_chosen, _ = event_threshold_diagnostics(
        changed, selection, benchmark, robustness
    )
    pd.testing.assert_frame_equal(chosen, changed_chosen)
    assert not chosen["test_labels_used_for_selection"].any()
    persistence = chosen.loc[chosen["model"] == "persistence"]
    assert set(persistence["calibration_for"]) == {"persistence"}
    assert persistence["alpha_frozen_from_regression_selection"].isna().all()
    assert "calibrated_f1_difference_vs_calibrated_persistence" in metrics


def test_threshold_tie_breaks_are_deterministic():
    def row(threshold, f1=0.5, worst=0.4, balanced=0.7):
        return {
            "decision_threshold": threshold,
            "macro_validation_f1": f1,
            "worst_station_validation_f1": worst,
            "macro_validation_balanced_accuracy": balanced,
        }

    assert (
        _select_alert_threshold([row(-1.0), row(-0.9, worst=0.5)], [])[
            "decision_threshold"
        ]
        == -0.9
    )
    assert (
        _select_alert_threshold([row(-1.0), row(-0.9, balanced=0.8)], [])[
            "decision_threshold"
        ]
        == -0.9
    )
    assert (
        _select_alert_threshold([row(-0.7), row(-1.2)], [])["decision_threshold"]
        == -1.2
    )
    assert (
        _select_alert_threshold([row(-0.8), row(-1.2)], [])["decision_threshold"]
        == -1.2
    )


def test_threshold_grid_must_land_on_physical_cutoff_and_maximum():
    _, robustness = _configs()
    with pytest.raises(ValueError, match="physical -1.0|land exactly on -1.0"):
        _validate_robustness_config(replace(robustness, event_threshold_step=0.3))


def test_threshold_fallback_uses_physical_cutoff_when_grid_is_degenerate():
    rows = [
        {
            "decision_threshold": -1.0,
            "macro_validation_f1": 0.0,
            "worst_station_validation_f1": 0.0,
            "macro_validation_balanced_accuracy": 0.5,
        }
    ]
    assert _select_alert_threshold([], rows)["decision_threshold"] == -1.0


def test_end_to_end_manifest_disclaims_fresh_test(tmp_path):
    benchmark, robustness = _configs()
    output = tmp_path / "robustness"
    run_robustness(
        benchmark,
        robustness,
        output,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["fresh_locked_test"] is False
    assert manifest["period_previously_inspected"] is True
    assert manifest["test_labels_used_for_threshold_selection"] is False
    assert manifest["upstream_model_selection_recomputed"] is True
    assert len(manifest["implementation_files_sha256"]) >= 8
    assert len(manifest["resolved_config_sha256"]) == 64
    assert (output / "config_resolved.json").exists()
    resolved = json.loads((output / "config_resolved.json").read_text())
    assert resolved["benchmark"]["ridge_alphas"] == [1.0, 100.0]
    assert len(list(output.glob("*.csv"))) >= 10
