# One-Month-Ahead SPEI Forecasting

Research code for forecasting the Standardized Precipitation Evapotranspiration
Index (SPEI) at three Sri Lankan locations. The project grew from undergraduate
work on recurrent neural networks for meteorological drought prediction.

## Current status

This repository is a research prototype under methodological revision. Earlier
versions fitted preprocessing on the full series, evaluated a Random Forest on
observations used for fitting, reused the LSTM holdout for model selection, and
combined predictions that were not aligned by date. Those earlier metrics are
not valid out-of-sample evidence and should not be cited.

The current script defines an explicit task: use information available at month
`t` to predict SPEI at month `t+1`. It uses chronological train, validation, and
test periods; fits preprocessing on training data only; tunes the Random Forest
within past-only folds; and compares the untouched test period with persistence
and same-month-last-year baselines. An optional LSTM uses the validation period
for early stopping. No ensemble is reported.

No performance result is claimed in this README. Results should be regenerated
from a documented data snapshot before being used in a report or application.

## Data

The repository contains monthly files for Buttala, Padaviya, and Tissamaharama.
They include meteorological variables and SPEI at several accumulation periods.
The original source files, retrieval date, variable units, SPEI implementation,
and redistribution terms were not recorded in this repository and must be
confirmed before the data are reused. In particular, precipitation and
evapotranspiration units must be checked before interpreting water-balance
features physically.

Structural warm-up gaps in longer SPEI windows are dropped. They are not filled
from future observations. The script also checks that consecutive rows represent
consecutive calendar months before it creates a one-row-ahead target.

## Run

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the leak-aware Random Forest benchmark:

```bash
python multi_station_spei_predictor.py --model rf
```

Run both the Random Forest and LSTM paths:

```bash
python multi_station_spei_predictor.py --model both
```

Predictions are written to `outputs/` with both forecast-origin and target dates
so that model comparisons can be aligned exactly.

## Evaluation protocol

- forecast horizon: one month;
- split: chronological 70% train, 15% validation, 15% test;
- preprocessing: fitted on training rows only;
- Random Forest selection: expanding time-series folds inside training data;
- LSTM selection: validation loss for early stopping;
- final test: evaluated once after choices are fixed; and
- baselines: persistence and the corresponding month from the previous year.

MAE and RMSE are the primary error summaries. R² is also printed for context; it
is not treated as an independent form of validation.

## Next checks

1. Recover and document the original data provenance, units, and licence.
2. Save an immutable input manifest and exact package versions.
3. Repeat the evaluation with rolling-origin uncertainty intervals.
4. Compare the models on the same target dates across all stations.
