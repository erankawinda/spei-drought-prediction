from __future__ import annotations

import pandas as pd

from .config import BenchmarkConfig


def split_frame(frame: pd.DataFrame, config: BenchmarkConfig) -> dict[str, pd.DataFrame]:
    train_end = pd.Period(config.train_end, freq="M").to_timestamp()
    validation_end = pd.Period(config.validation_end, freq="M").to_timestamp()
    test_end = pd.Period(config.test_end, freq="M").to_timestamp()
    if not train_end < validation_end < test_end:
        raise ValueError("Split boundaries must be strictly increasing")

    train = frame.loc[frame["target_date"] <= train_end].copy()
    validation = frame.loc[
        (frame["target_date"] > train_end)
        & (frame["target_date"] <= validation_end)
    ].copy()
    test = frame.loc[
        (frame["target_date"] > validation_end)
        & (frame["target_date"] <= test_end)
    ].copy()

    if any(part.empty for part in (train, validation, test)):
        raise ValueError("Every chronological partition must contain observations")
    if not (
        train["target_date"].max()
        < validation["target_date"].min()
        <= validation["target_date"].max()
        < test["target_date"].min()
    ):
        raise AssertionError("Chronological split ordering failed")
    return {"train": train, "validation": validation, "test": test}
