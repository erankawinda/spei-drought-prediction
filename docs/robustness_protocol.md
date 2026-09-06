# Robustness v1 protocol

## Evidence status

This is a retrospective robustness analysis of the already-inspected 2000–2019 period. Every artifact records:

```text
evidence_status = retrospective_robustness
period_previously_inspected = true
fresh_locked_test = false
```

The analysis can test whether the benchmark conclusions are stable under reasonable design choices. It cannot create a new independent confirmation. Confirmation requires post-2019 observations or an external dataset with compatible predictors and a documented SPEI construction.

## Frozen forecast contract and primary models

The forecast remains same-station SPEI one month ahead. A row issued at the end of month `t` may use observations dated `t` or earlier and predicts month `t + 1`.

The model families were frozen for phase two after the phase-one 2000–2019 outcomes had already been inspected:

- SPEI-3: ridge regression using 24 SPEI-3 lags (`ridge_ar24`).
- SPEI-6: ridge regression using 24 SPEI-6 lags and 24 lags of the seven meteorological variables (`ridge_ar24_met24`).

Only the ridge regularization parameter is selected inside each scenario. No new model family is chosen from the retrospective results.

## Nested rolling-origin evaluation

Four non-overlapping five-year evaluation folds cover each month from January 2000 through December 2019 exactly once:

| Evaluation fold | Inner validation folds |
|---|---|
| 2000–2004 | 1985–1989, 1990–1994, 1995–1999 |
| 2005–2009 | 1990–1994, 1995–1999, 2000–2004 |
| 2010–2014 | 1995–1999, 2000–2004, 2005–2009 |
| 2015–2019 | 2000–2004, 2005–2009, 2010–2014 |

Each inner fold fits on the expanding history strictly before that validation fold. Alpha is selected by equal-weight mean MAE across inner folds and stations, with a deterministic tie-break. Separate station models are then refitted through the month before the outer fold and their coefficients remain frozen for that five-year block.

Later predictions within a block may use observed lags through their own issue month. This is sequential one-month-ahead forecasting, not a recursive five-year forecast made at the block origin.

## Training-history sensitivity

The complete nested rolling-origin policy is repeated after cropping source rows to start in 1901, 1951 or 1971. Cropping occurs before lag construction, so no model input row is dated before its declared start. The recovered SPEI values themselves are precomputed accumulation-scale indices and their calibration period is unknown, so this cutoff does not establish independence from earlier climate observations. Alpha is reselected inside each outer fold for every start, and all starts are reported without choosing a winner post hoc.

## Bootstrap block sensitivity

The primary 1901 rolling-origin predictions are resampled without refitting. Synchronized paired circular moving-block bootstrap intervals are recomputed with 6-, 12- and 24-month blocks and 5,000 resamples. Blocks are sampled separately within each five-year outer fold so they never cross a model-refit boundary, while the same month indices are used across stations and paired models. The point estimate must be identical across block lengths; conclusions are considered stable only when the interval result agrees at all three lengths.

## Alert-threshold diagnostic

Observed drought remains physically defined as `SPEI <= -1.0`. This label cutoff is never tuned. Because a regression model can be biased or compressed near -1, a separate prediction cutoff is selected on 1985–1999 validation predictions only.

This is deliberately a fixed-origin diagnostic, separate from the rolling update policy. The primary ridge alpha is recomputed from the same pre-2000 validation-only rule; no saved or ignored local artifact is required.

- Grid: -2.00 through 0.00 in increments of 0.05, including -1.00.
- One cutoff is shared across all three stations for each target/model.
- Cutoffs that predict every or no month as an alert at any station are ineligible.
- Objective: equal-station macro F1.
- Ties: worst-station F1, macro balanced accuracy, proximity to -1.0, then the more conservative cutoff.
- The MAE-selected model and alpha stay frozen; alert F1 cannot reopen model selection.
- Persistence receives the same calibration procedure.

Validation scores are apparent tuning performance. Scores on 2000–2019 are post-hoc retrospective diagnostics. Values between -1.0 and a selected warning cutoff are called alerts, not drought SPEI values.

## Outputs and fixed historical limitation

The run saves dated predictions, per-station and macro metrics, fold selections with candidate scores, start-date sensitivity, fold-bounded bootstrap intervals, threshold sweeps, selected cutoffs, the alert regression selection, and source/config/code hashes in a resolved manifest.

The recovered files do not document the SPEI fitting distribution, calibration reference period or whether it used the full record. The forecasting code is temporally controlled, but leakage in construction of the target series cannot be ruled out. This limitation must accompany every result.
