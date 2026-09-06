from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from spei_forecast.config import load_config
from spei_forecast.experiment import run_benchmark


ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_writes_fixed_period_benchmark_evidence(tmp_path):
    config = load_config(ROOT / "configs" / "benchmark_v1.toml")
    config = replace(
        config,
        ridge_alphas=(1.0, 100.0),
        hgb_learning_rates=(0.05,),
        hgb_max_iters=(20,),
        hgb_max_leaf_nodes=(7,),
        hgb_min_samples_leaf=(20,),
        hgb_l2_regularization=(1.0,),
        bootstrap_resamples=50,
    )
    output = tmp_path / "run"
    run_benchmark(config, output)

    expected = {
        "predictions.csv",
        "metrics_by_task.csv",
        "metrics_macro.csv",
        "bootstrap_intervals.csv",
        "acceptance.csv",
        "selection.json",
        "data_quality.json",
        "config_resolved.json",
        "manifest.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    predictions = pd.read_csv(output / "predictions.csv")
    assert len(predictions) == 2 * 3 * 8 * 240
    assert predictions["target_date"].min() == "2000-01"
    assert predictions["target_date"].max() == "2019-12"
    selection = json.loads((output / "selection.json").read_text())
    assert selection["test_partition_used_for_selection"] is False
    acceptance = pd.read_csv(output / "acceptance.csv")
    assert set(acceptance["passes_full_gate"].unique()) <= {True, False}
