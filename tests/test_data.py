from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from spei_forecast.config import load_config
from spei_forecast.data import (
    DataValidationError,
    EXPECTED_COLUMNS,
    load_all,
    normalize_headers,
    write_clean_dataset,
)


ROOT = Path(__file__).resolve().parents[1]


def test_recovered_sources_pass_strict_audit():
    config = load_config(ROOT / "configs" / "benchmark_v1.toml")
    frame, quality = load_all(config)
    assert len(frame) == 3 * 1428
    assert set(frame["station"]) == {"Buttala", "Padaviya", "Tissamaharama"}
    for station in quality["stations"].values():
        assert station["first_month"] == "1901-01"
        assert station["last_month"] == "2019-12"
        assert station["missing_targets"] == {
            "spei1": 0,
            "spei3": 2,
            "spei6": 5,
            "spei9": 8,
            "spei12": 11,
        }
        assert all(station["physical_checks"].values())
    assert quality["rules"]["source_hashes_match_frozen_manifest"] is True
    assert {
        source["station"]: source["sha256"] for source in quality["source_files"]
    } == config.source_sha256
    assert all(not Path(source["path"]).is_absolute() for source in quality["source_files"])


def test_source_hash_manifest_rejects_changed_input():
    config = load_config(ROOT / "configs" / "benchmark_v1.toml")
    changed = dict(config.source_sha256)
    changed["Buttala"] = "0" * 64
    with pytest.raises(DataValidationError, match="SHA-256 mismatch"):
        load_all(replace(config, source_sha256=changed))


def test_header_variants_are_explicit_and_collisions_fail():
    headers = list(EXPECTED_COLUMNS)
    headers[headers.index("spei1")] = "spei_ 1"
    mapping = normalize_headers(headers)
    assert mapping["spei_ 1"] == "spei1"
    with pytest.raises(DataValidationError, match="collision"):
        normalize_headers([*EXPECTED_COLUMNS, "spei_ 1"])
    with pytest.raises(DataValidationError, match="Unexpected"):
        normalize_headers([*EXPECTED_COLUMNS, "future_target"])


def test_clean_derivative_preserves_structural_missing_values(tmp_path):
    config = load_config(ROOT / "configs" / "benchmark_v1.toml")
    frame, _ = load_all(config)
    output = write_clean_dataset(frame, tmp_path / "observations.csv")
    reloaded = pd.read_csv(output, na_values=["NA"])
    assert len(reloaded) == len(frame)
    assert int(reloaded["spei12"].isna().sum()) == 33
    assert int(reloaded["mean_prep"].isna().sum()) == 0
