#!/usr/bin/env python3
"""Leak-aware one-month-ahead SPEI forecasting research prototype.

Each example has an explicit forecast origin at month t and predicts SPEI at
month t+1. Preprocessing and model selection use past data only, and the final
test period is evaluated once. The script intentionally makes no published
performance claim; rerun it after checking the provenance and units of the
bundled station data.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


SEED = 42
SPEI_COLUMNS = {"spei1", "spei3", "spei6", "spei9", "spei12"}


def load_station(path: Path) -> pd.DataFrame:
    """Load one station without backward-filling information from the future."""
    frame = pd.read_csv(path, na_values=["NA", "NaN", ""])
    frame.columns = (
        frame.columns.str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)

    if frame["date"].duplicated().any():
        raise ValueError(f"Duplicate dates found in {path}")
    month_number = frame["date"].dt.year * 12 + frame["date"].dt.month
    if not month_number.diff().dropna().eq(1).all():
        raise ValueError(
            f"Missing or repeated calendar months in {path}; row shifts would not "
            "represent a one-month forecast horizon"
        )
    return frame


def make_one_month_ahead_frame(
    frame: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, list[str]]:
    """Create predictors known at month t and a label for month t+1."""
    if target not in frame.columns:
        raise ValueError(f"Missing target column: {target}")

    work = frame.copy()
    work["month_sin"] = np.sin(2 * np.pi * work["date"].dt.month / 12)
    work["month_cos"] = np.cos(2 * np.pi * work["date"].dt.month / 12)

    # Values observed at the forecast origin or earlier.
    work[f"{target}_current"] = work[target]
    for lag in (1, 3, 6, 12):
        work[f"{target}_lag_{lag}"] = work[target].shift(lag)

    for window in (3, 6, 12):
        work[f"{target}_mean_{window}"] = work[target].rolling(window).mean()
        work[f"prep_mean_{window}"] = work["meanprep"].rolling(window).mean()
        work[f"temp_mean_{window}"] = work["meantmp"].rolling(window).mean()

    work["temp_range"] = work["maxtmp"] - work["mintmp"]

    # Explicit forecast target. The same-month target is never a feature for
    # the row being predicted.
    work["target_date"] = work["date"].shift(-1)
    work["target_value"] = work[target].shift(-1)
    work["persistence_naive"] = work[target]
    work["seasonal_naive"] = work[target].shift(11)

    excluded = {"date", "target_date", "target_value", *SPEI_COLUMNS}
    feature_columns = [
        column
        for column in work.columns
        if column not in excluded
        and column not in {"persistence_naive", "seasonal_naive"}
        and pd.api.types.is_numeric_dtype(work[column])
    ]

    required = feature_columns + [
        "date",
        "target_date",
        "target_value",
        "persistence_naive",
        "seasonal_naive",
    ]
    supervised = work[required].dropna().reset_index(drop=True)
    if supervised.empty:
        raise ValueError(f"No complete supervised rows for {target}")
    return supervised, feature_columns


def chronological_slices(n_rows: int) -> tuple[slice, slice, slice]:
    """Return non-overlapping 70/15/15 chronological partitions."""
    train_end = int(n_rows * 0.70)
    validation_end = int(n_rows * 0.85)
    if train_end < 50 or validation_end <= train_end or validation_end >= n_rows:
        raise ValueError("Not enough rows for chronological train/validation/test splits")
    return (
        slice(0, train_end),
        slice(train_end, validation_end),
        slice(validation_end, n_rows),
    )


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(observed, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(observed, predicted))),
        "R2": float(r2_score(observed, predicted)),
    }


def fit_random_forest(
    supervised: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Tune on training folds, check validation, then evaluate untouched test rows."""
    train_slice, validation_slice, test_slice = chronological_slices(len(supervised))
    x = supervised[feature_columns]
    y = supervised["target_value"]

    base_model = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    parameter_space = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 8, 16, 24],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.7, 1.0],
    }
    search = RandomizedSearchCV(
        base_model,
        parameter_space,
        n_iter=12,
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=5, gap=1),
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(x.iloc[train_slice], y.iloc[train_slice])

    validation_prediction = search.best_estimator_.predict(x.iloc[validation_slice])
    validation_metrics = metrics(
        y.iloc[validation_slice].to_numpy(), validation_prediction
    )

    # Hyperparameters are now fixed. Refit on train + validation and touch the
    # test period only once.
    final_model = RandomForestRegressor(
        **search.best_params_, random_state=SEED, n_jobs=-1
    )
    final_model.fit(x.iloc[: validation_slice.stop], y.iloc[: validation_slice.stop])
    test_prediction = final_model.predict(x.iloc[test_slice])
    test = supervised.iloc[test_slice][
        ["date", "target_date", "target_value", "persistence_naive", "seasonal_naive"]
    ].copy()
    test["random_forest"] = test_prediction

    test_metrics = {
        "persistence": metrics(
            test["target_value"].to_numpy(), test["persistence_naive"].to_numpy()
        ),
        "seasonal": metrics(
            test["target_value"].to_numpy(), test["seasonal_naive"].to_numpy()
        ),
        "random_forest": metrics(
            test["target_value"].to_numpy(), test["random_forest"].to_numpy()
        ),
    }
    return test, {"validation_random_forest": validation_metrics, **test_metrics}


def fit_lstm(
    supervised: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int = 24,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit an optional LSTM with scalers estimated from training rows only."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for --model lstm or --model both"
        ) from exc

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    train_slice, validation_slice, test_slice = chronological_slices(len(supervised))

    x_raw = supervised[feature_columns].to_numpy(dtype=float)
    y_raw = supervised[["target_value"]].to_numpy(dtype=float)
    x_scaler = StandardScaler().fit(x_raw[train_slice])
    y_scaler = StandardScaler().fit(y_raw[train_slice])
    x_scaled = x_scaler.transform(x_raw)
    y_scaled = y_scaler.transform(y_raw).ravel()

    sequences, labels, end_rows = [], [], []
    for end_row in range(sequence_length - 1, len(supervised)):
        start_row = end_row - sequence_length + 1
        sequences.append(x_scaled[start_row : end_row + 1])
        labels.append(y_scaled[end_row])
        end_rows.append(end_row)
    sequences = np.asarray(sequences)
    labels = np.asarray(labels)
    end_rows = np.asarray(end_rows)

    train_mask = end_rows < train_slice.stop
    validation_mask = (
        (end_rows >= validation_slice.start) & (end_rows < validation_slice.stop)
    )
    test_mask = end_rows >= test_slice.start

    model = keras.Sequential(
        [
            layers.Input((sequence_length, len(feature_columns))),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(
        sequences[train_mask],
        labels[train_mask],
        validation_data=(sequences[validation_mask], labels[validation_mask]),
        epochs=100,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
        ],
        verbose=0,
    )

    scaled_prediction = model.predict(sequences[test_mask], verbose=0)
    prediction = y_scaler.inverse_transform(scaled_prediction).ravel()
    observed = supervised.iloc[end_rows[test_mask]]["target_value"].to_numpy()
    return prediction, metrics(observed, prediction)


def run_station(
    station: str,
    path: Path,
    targets: list[str],
    model_choice: str,
    output_dir: Path,
) -> None:
    frame = load_station(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        supervised, feature_columns = make_one_month_ahead_frame(frame, target)
        test, scores = fit_random_forest(supervised, feature_columns)

        if model_choice == "both":
            lstm_prediction, lstm_metrics = fit_lstm(supervised, feature_columns)
            test["lstm"] = lstm_prediction
            scores["lstm"] = lstm_metrics

        print(f"\n{station} — {target} — one-month-ahead test")
        for name, values in scores.items():
            formatted = ", ".join(f"{key}={value:.4f}" for key, value in values.items())
            print(f"  {name}: {formatted}")

        output_path = output_dir / f"{station.lower()}_{target}_test_predictions.csv"
        test.to_csv(output_path, index=False)
        print(f"  wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs"
    )
    parser.add_argument("--targets", nargs="+", default=["spei3", "spei6"])
    parser.add_argument(
        "--model", choices=["rf", "both"], default="rf"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    station_files = {
        "Buttala": args.data_dir / "Buttala_speiall.csv",
        "Padaviya": args.data_dir / "Padaviya_speiall.csv",
        "Tissamaharama": args.data_dir / "Tissamaharama_speiall.csv",
    }
    for station, path in station_files.items():
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        run_station(station, path, args.targets, args.model, args.output_dir)


if __name__ == "__main__":
    main()
