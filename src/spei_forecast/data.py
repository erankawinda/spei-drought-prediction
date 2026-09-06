from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import BenchmarkConfig


PREDICTOR_COLUMNS = (
    "cloud_cover",
    "mean_prep",
    "pot_evap",
    "mean_tmp",
    "max_tmp",
    "min_tmp",
    "diurnal_tmp_range",
    "vapour_pressure",
    "wet_day_frq",
    "pet",
)
TARGET_COLUMNS = ("spei1", "spei3", "spei6", "spei9", "spei12")
EXPECTED_COLUMNS = ("date", *PREDICTOR_COLUMNS, *TARGET_COLUMNS)


class DataValidationError(ValueError):
    """Raised when a recovered source file violates the benchmark contract."""


def _header_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


_HEADER_ALLOWLIST = {_header_key(column): column for column in EXPECTED_COLUMNS}


def normalize_headers(columns: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    seen: set[str] = set()
    unknown: list[str] = []
    for original in columns:
        key = _header_key(original)
        canonical = _HEADER_ALLOWLIST.get(key)
        if canonical is None:
            unknown.append(str(original))
            continue
        if canonical in seen:
            raise DataValidationError(
                f"Header normalization collision for canonical column {canonical!r}"
            )
        mapping[str(original)] = canonical
        seen.add(canonical)
    if unknown:
        raise DataValidationError(f"Unexpected columns: {unknown}")
    missing = sorted(set(EXPECTED_COLUMNS) - seen)
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    return mapping


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_station(path: str | Path, station: str) -> tuple[pd.DataFrame, dict]:
    source_path = Path(path).resolve()
    frame = pd.read_csv(source_path, na_values=["NA"], keep_default_na=True)
    frame = frame.rename(columns=normalize_headers(frame.columns))
    frame = frame.loc[:, EXPECTED_COLUMNS].copy()

    source_dates = pd.to_datetime(frame["date"], errors="raise")
    if source_dates.isna().any():
        raise DataValidationError(f"{station}: invalid dates")
    for column in (*PREDICTOR_COLUMNS, *TARGET_COLUMNS):
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame.insert(0, "station", station)
    frame.insert(1, "source_date", source_dates)
    frame["month"] = source_dates.dt.to_period("M").dt.to_timestamp()
    frame = frame.drop(columns="date").sort_values("month").reset_index(drop=True)

    quality = validate_station(frame, station)
    source = {
        "station": station,
        "path": str(source_path),
        "sha256": file_sha256(source_path),
        "rows": int(len(frame)),
    }
    return frame, {"source": source, "quality": quality}


def validate_station(frame: pd.DataFrame, station: str) -> dict:
    if frame.empty:
        raise DataValidationError(f"{station}: empty source")
    if frame["month"].duplicated().any():
        raise DataValidationError(f"{station}: duplicate months")

    expected_months = pd.date_range(
        frame["month"].min(), frame["month"].max(), freq="MS"
    )
    if not frame["month"].reset_index(drop=True).equals(pd.Series(expected_months)):
        raise DataValidationError(f"{station}: missing, repeated, or unordered months")

    numeric = frame.loc[:, (*PREDICTOR_COLUMNS, *TARGET_COLUMNS)]
    finite_predictors = np.isfinite(frame.loc[:, PREDICTOR_COLUMNS].to_numpy(dtype=float))
    if not finite_predictors.all():
        raise DataValidationError(f"{station}: missing or non-finite predictor values")

    expected_gaps = {"spei1": 0, "spei3": 2, "spei6": 5, "spei9": 8, "spei12": 11}
    missing_targets: dict[str, int] = {}
    for target, expected_count in expected_gaps.items():
        missing = frame[target].isna().to_numpy()
        expected = np.zeros(len(frame), dtype=bool)
        expected[:expected_count] = True
        if not np.array_equal(missing, expected):
            raise DataValidationError(
                f"{station}: {target} missingness is not the expected leading "
                f"{expected_count}-month warm-up"
            )
        values = frame[target].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DataValidationError(f"{station}: {target} contains non-finite values")
        missing_targets[target] = int(missing.sum())

    days = frame["month"].dt.days_in_month
    checks = {
        "cloud_cover_0_100": frame["cloud_cover"].between(0, 100).all(),
        "nonnegative_precipitation": frame["mean_prep"].ge(0).all(),
        "nonnegative_potential_evaporation": frame["pot_evap"].ge(0).all(),
        "temperature_order": (
            frame["min_tmp"].le(frame["mean_tmp"])
            & frame["mean_tmp"].le(frame["max_tmp"])
        ).all(),
        "nonnegative_diurnal_range": frame["diurnal_tmp_range"].ge(0).all(),
        "positive_vapour_pressure": frame["vapour_pressure"].gt(0).all(),
        "wet_days_within_month": (
            frame["wet_day_frq"].ge(0) & frame["wet_day_frq"].le(days)
        ).all(),
        "nonnegative_pet": frame["pet"].ge(0).all(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise DataValidationError(f"{station}: physical checks failed: {failed}")

    derived_range = frame["max_tmp"] - frame["min_tmp"]
    if not np.allclose(derived_range, frame["diurnal_tmp_range"], atol=1e-10):
        raise DataValidationError(f"{station}: inconsistent diurnal temperature range")

    return {
        "rows": int(len(frame)),
        "first_month": frame["month"].min().strftime("%Y-%m"),
        "last_month": frame["month"].max().strftime("%Y-%m"),
        "duplicate_months": 0,
        "missing_targets": missing_targets,
        "physical_checks": {name: bool(value) for name, value in checks.items()},
        "temperature_redundancy": {
            "dtr_equals_max_minus_min": True,
            "mean_midpoint_max_abs_error": float(
                np.max(
                    np.abs(
                        frame["mean_tmp"]
                        - (frame["max_tmp"] + frame["min_tmp"]) / 2
                    )
                )
            ),
        },
        "ranges": {
            column: {
                "min": float(numeric[column].min(skipna=True)),
                "max": float(numeric[column].max(skipna=True)),
            }
            for column in numeric.columns
        },
    }


def load_all(config: BenchmarkConfig) -> tuple[pd.DataFrame, dict]:
    frames: list[pd.DataFrame] = []
    sources: list[dict] = []
    station_quality: dict[str, dict] = {}

    for station, filename in sorted(config.station_files.items()):
        path = config.source_dir / filename
        if not path.is_file():
            raise DataValidationError(f"Missing station source: {path}")
        frame, audit = load_station(path, station)
        expected_digest = config.source_sha256[station]
        actual_digest = audit["source"]["sha256"]
        if actual_digest != expected_digest:
            raise DataValidationError(
                f"{station}: source SHA-256 mismatch; expected {expected_digest}, "
                f"found {actual_digest}"
            )
        # Persist repository-relative paths so manifests remain portable and do
        # not disclose a contributor's local directory layout.
        audit["source"]["path"] = path.relative_to(config.repository_root).as_posix()
        frames.append(frame)
        sources.append(audit["source"])
        station_quality[station] = audit["quality"]

    combined = pd.concat(frames, ignore_index=True)
    quality = {
        "stations": station_quality,
        "source_files": sources,
        "total_rows": int(len(combined)),
        "rules": {
            "raw_sources_modified": False,
            "fill_or_interpolation_applied": False,
            "outlier_deletion_or_winsorization_applied": False,
            "model_temperature_columns": ["mean_tmp", "diurnal_tmp_range"],
            "excluded_redundant_columns": ["max_tmp", "min_tmp"],
            "excluded_unverified_evaporation_total": ["pet"],
            "source_hashes_match_frozen_manifest": True,
        },
    }
    return combined, quality


def write_clean_dataset(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exported = frame.copy()
    exported["source_date"] = exported["source_date"].dt.strftime("%Y-%m-%d")
    exported["month"] = exported["month"].dt.strftime("%Y-%m")
    exported.to_csv(path, index=False, na_rep="NA")
    return path
