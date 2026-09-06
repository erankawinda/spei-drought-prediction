# Benchmark v1 analysis protocol

## Evidence status

Benchmark v1 was the first fixed-period evaluation of the rebuilt pipeline. Its 2000–2019 outcomes were inspected on 13 August 2026, so that period is now consumed. The original acceptance rule is retained unchanged for reproducibility, but any rerun or follow-up analysis on these dates is retrospective and must not be described as a fresh locked test. See `robustness_protocol.md` for the phase-two protocol.

## Forecast contract

- Forecast origin: end of calendar month `t`.
- Target: same-station SPEI at calendar month `t + 1`.
- Horizon: one month.
- Predictor availability: observations dated `t` or earlier only.
- Primary target: SPEI-3.
- Secondary target: SPEI-6.
- Test updating: none. Final models are frozen after December 1999.

This is a forecast using observed meteorological history. It does not assume access to a numerical weather forecast for the target month.

## Data policy

The recovered CSVs are immutable inputs. Header normalization uses an explicit allow-list and fails on unknown or colliding fields. Dates are converted to calendar months only after preserving the original date. Every station must contain one uninterrupted, unique record per month.

No data value is filled or interpolated. Leading SPEI gaps created by the accumulation window remain missing in the canonical table and are removed only when a target-specific supervised sample requires them. Plausible climate extremes remain unchanged.

The model matrix retains `mean_tmp` and `diurnal_tmp_range` but excludes `max_tmp` and `min_tmp`, which are deterministically redundant in the recovered files. The first benchmark excludes the locally derived `PET` total because its exact historical implementation was not preserved. No precipitation-evaporation arithmetic mixes this field with the upstream daily `pot_evap` series.

## Features

All feature sets include sine/cosine encodings of the known target calendar month.

- `ar24`: 24 lags of the selected SPEI target.
- `met24`: 24 lags of the seven predeclared meteorological variables.
- `ar24_met24`: both histories.

Other SPEI scales are never used as predictors. The target month itself is never included in the inputs.

## Partitions and selection

Partitions are assigned from target dates:

- Development training: through 1984-12.
- Validation: 1985-01 through 1999-12.
- Fixed evaluation: 2000-01 through 2019-12 (now previously inspected).

One shared hyperparameter setting per target/model is selected by macro-average validation MAE across stations. After selection, a separate station model is fitted on training plus validation data and evaluated once on the test period.

## Comparators

- Zero (the standardized-index centre).
- Training-only monthly climatology.
- One-month persistence.
- Twelve-month seasonal persistence.
- Ridge on `ar24`, `met24` and `ar24_met24`.
- Histogram gradient boosting on `ar24_met24`.

The nonlinear model uses no random time split and disables internal early stopping. Neural models are outside benchmark v1.

## Evidence and acceptance rule

The primary score is macro-average station MAE. RMSE, R-squared, Pearson correlation, drought-event precision/recall/F1/balanced accuracy and event support are also retained. Skill is reported relative to persistence.

Event F1 is calculated directly from event counts as
`2*TP / (2*TP + FP + FN)`. It is zero when events are observed or predicted but
none are correct. It is undefined only when neither observed nor predicted
events exist. Macro F1 includes zero scores and omits only undefined scores;
the number of contributing stations is reported with each summary.

Uncertainty uses synchronized paired circular moving-block resampling of test months across all stations. The primary block length is 12 months. A model clears the full initial gate only when it:

- reduces macro-MAE by at least 5% relative to persistence;
- has a paired 95% interval for that improvement above zero;
- improves MAE at every recovered station; and
- loses no more than 0.02 absolute macro drought F1 relative to persistence.

The 0.02 event-skill tolerance is predeclared to prevent a regression-only gain from being presented as better drought identification.

Passing this gate justifies further analysis; it does not establish operational, causal or out-of-domain forecasting skill.
