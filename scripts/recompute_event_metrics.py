"""Recompute reviewed F1 summaries without repeating model selection or bootstrap.

Run from the repository root after installing the package. Outputs go to an
ignored artifact directory; the reviewed results are never overwritten here.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from spei_forecast.config import load_config
from spei_forecast.data import load_all
from spei_forecast.experiment import _evaluate_fixed_period, _prepare_tasks
from spei_forecast.metrics import acceptance_table, metrics_table
from spei_forecast.provenance import file_sha256, implementation_hashes
from spei_forecast.robustness import _dated_csv, _scenario_metrics


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/event_f1_correction_v1")
    output = Path(parser.parse_args().output_dir).resolve()
    if output == ROOT / "results" or ROOT / "results" in output.parents:
        parser.error("Write to artifacts, not the reviewed results directory")
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    changes = []

    def read_input(relative):
        path = ROOT / relative
        inputs[relative] = file_sha256(path)
        return path

    def write_correction(relative, recomputed):
        original = read_input(relative)
        previous = pd.read_csv(original)
        editable = [c for c in previous if "f1" in c or c == "passes_full_gate"]
        pd.testing.assert_frame_equal(
            previous.drop(columns=editable), recomputed.drop(columns=editable),
            check_dtype=False, rtol=1e-12, atol=1e-12,
        )
        with original.open(newline="") as source:
            reader = csv.DictReader(source)
            fields, rows = reader.fieldnames, list(reader)
        for column in editable:
            old = previous[column].to_numpy(dtype=float)
            new = recomputed[column].to_numpy(dtype=float)
            for index in np.flatnonzero(~np.isclose(old, new, rtol=1e-12, atol=1e-12, equal_nan=True)):
                value = recomputed.iloc[index][column]
                before = rows[index][column]
                rows[index][column] = "" if pd.isna(value) else str(value)
                changes.append({
                    "file": relative, "csv_line": int(index) + 2,
                    "key": {k: rows[index][k] for k in
                            ("outer_fold", "target", "station", "model") if k in rows[index]},
                    "column": column, "before": before, "after": rows[index][column],
                })
        destination = output / relative.removeprefix("results/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        pd.testing.assert_frame_equal(
            pd.read_csv(destination), recomputed,
            check_dtype=False, rtol=1e-12, atol=1e-12,
        )

    for name in ("benchmark_v1", "robustness_v1"):
        read_input(f"results/{name}/manifest.json")
    config = load_config(read_input("configs/benchmark_v1.toml"))
    observations, quality = load_all(config)
    for source in quality["source_files"]:
        read_input(source["path"])

    # Refit the existing selected settings because this panel was not committed.
    selection = json.loads(read_input("results/benchmark_v1/selection.json").read_text())
    benchmark = _evaluate_fixed_period(_prepare_tasks(observations, config), selection, config)
    _dated_csv(benchmark, output / "benchmark_predictions.csv")
    by_task, macro = metrics_table(benchmark)
    intervals = pd.read_csv(read_input("results/benchmark_v1/bootstrap_intervals.csv"))
    write_correction("results/benchmark_v1/metrics_by_task.csv", by_task)
    write_correction("results/benchmark_v1/metrics_macro.csv", macro)
    write_correction("results/benchmark_v1/acceptance.csv", acceptance_table(by_task, macro, intervals))

    rolling_path = read_input("results/robustness_v1/rolling_predictions.csv")
    historical = json.loads((ROOT / "results/robustness_v1/manifest.json").read_text())
    if file_sha256(rolling_path) != historical["rolling_predictions_sha256"]:
        raise ValueError("Rolling predictions do not match their original run manifest")
    rolling = pd.read_csv(rolling_path)
    by_task, macro = metrics_table(rolling)
    for name, frame in (("rolling_metrics_by_task.csv", by_task), ("rolling_metrics_macro.csv", macro)):
        pd.testing.assert_frame_equal(
            pd.read_csv(read_input(f"results/robustness_v1/{name}")), frame,
            check_dtype=False, rtol=1e-12, atol=1e-12,
        )
    fold_task, fold_macro = _scenario_metrics(rolling, "outer_fold")
    write_correction("results/robustness_v1/rolling_metrics_by_fold_and_station.csv", fold_task)
    write_correction("results/robustness_v1/rolling_metrics_by_fold.csv", fold_macro)

    manifest = {
        "correction": "event_f1_zero_true_positive_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "formula": "2 * TP / (2 * TP + FP + FN); undefined only for zero denominator",
        "method": "Rescore saved rolling predictions; refit saved fixed-benchmark settings without model selection; reuse MAE bootstrap intervals after verifying every non-F1 metric matches.",
        "original_run_manifests_preserved": True,
        "fresh_locked_test": False,
        "input_files_sha256": inputs,
        "implementation_files_sha256": implementation_hashes(),
        "script_sha256": file_sha256(__file__),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_porcelain": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
        "output_files_sha256": {p.relative_to(output).as_posix(): file_sha256(p)
                                for p in sorted(output.rglob("*.csv"))},
        "changed_cells": changes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Verified and corrected {len(changes)} cells; artifacts: {output}")


if __name__ == "__main__":
    main()
