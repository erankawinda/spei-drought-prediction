# Robustness v1 results

First run: 13 August 2026

Repository reproduction: 6 September 2026

Evidence status: **retrospective robustness**

Evaluation period: 2000-01 through 2019-12, previously inspected

## Main finding

The regression improvement is internally stable in this retrospective three-station analysis for both targets, but the event-detection claim must differ by target.

- **SPEI-3:** `ridge_ar24` reduced rolling-origin macro MAE from 0.638 for persistence to 0.542, a 15.1% improvement. The improvement was positive in all four five-year folds and under every training-history and bootstrap-block sensitivity. However, physical-threshold drought F1 was 0.527 versus 0.561 for persistence and was especially weak in 2005–2009. This is a useful continuous-value forecast, not a demonstrated superior drought-alert classifier.
- **SPEI-6:** `ridge_ar24_met24` reduced rolling-origin macro MAE from 0.440 to 0.336, a 23.6% improvement. It beat persistence in all four folds and retained positive skill with all three training starts. Physical-threshold drought F1 improved from 0.688 to 0.754. This is the strongest result in the recovered dataset.

These are not independent confirmation results because model choice and the evaluation period were already viewed after benchmark v1.

## Nested rolling-origin results

| Target | Primary model | Model MAE | Persistence MAE | MAE skill | Model drought F1 | Persistence F1 | F1 change |
|---|---|---:|---:|---:|---:|---:|---:|
| SPEI-3 | ridge_ar24 | 0.542 | 0.638 | 15.1% | 0.527 | 0.561 | -0.035 |
| SPEI-6 | ridge_ar24_met24 | 0.336 | 0.440 | 23.6% | 0.754 | 0.688 | +0.066 |

The four evaluation blocks were 2000–2004, 2005–2009, 2010–2014 and 2015–2019. Ridge alpha was chosen from the three preceding five-year inner validation blocks at every outer origin. Model coefficients were refitted at the outer origin and frozen for five years, while each monthly forecast used observations available through its own issue date.

SPEI-3 MAE skill by fold was 16.1%, 11.5%, 14.4% and 18.0%. SPEI-6 skill was 30.3%, 29.3%, 14.8% and 17.7%. SPEI-3 drought F1 was unstable: its change relative to persistence was +0.109, -0.380, -0.013 and +0.047 across the same folds. SPEI-6 had no severe events in 2005–2009, so severe-event conclusions must not be inferred from that fold.

The [event-F1 correction](../event_f1_correction_v1/README.md) on 6 September
2026 changed SPEI-3 drought F1 in 2005–2009 from 0.244 to 0.162. A station with
no correctly predicted droughts had previously been omitted from the average
instead of receiving zero F1. The correction also updates severe-event scores
in the fold tables. Full-period headline scores and all MAE results are
unchanged.

## Training-history sensitivity

| Target | Start | MAE | MAE skill vs persistence | Drought-F1 change |
|---|---:|---:|---:|---:|
| SPEI-3 | 1901 | 0.542 | 15.1% | -0.035 |
| SPEI-3 | 1951 | 0.542 | 15.1% | -0.024 |
| SPEI-3 | 1971 | 0.556 | 12.8% | +0.035 |
| SPEI-6 | 1901 | 0.336 | 23.6% | +0.066 |
| SPEI-6 | 1951 | 0.354 | 19.6% | +0.050 |
| SPEI-6 | 1971 | 0.382 | 13.1% | +0.057 |

The cutoff was applied before lag construction and every start used the same four nested rolling folds. The combined SPEI-6 model is more sensitive to truncating the training history; this analysis cannot separate record length, sample size and climate-regime effects.

## Bootstrap-block sensitivity

The paired macro-MAE improvement remained positive with 5,000 synchronized circular block-bootstrap resamples at all predeclared block lengths.

| Target | Block months | Absolute MAE improvement | 95% interval |
|---|---:|---:|---:|
| SPEI-3 | 6 | 0.096 | [0.056, 0.137] |
| SPEI-3 | 12 | 0.096 | [0.054, 0.137] |
| SPEI-3 | 24 | 0.096 | [0.059, 0.130] |
| SPEI-6 | 6 | 0.104 | [0.074, 0.134] |
| SPEI-6 | 12 | 0.104 | [0.076, 0.131] |
| SPEI-6 | 24 | 0.104 | [0.079, 0.128] |

Every resampled block stayed within one five-year outer fold, so no bootstrap block crossed a refit boundary.

## Validation-only alert-cutoff diagnostic

Observed drought was kept fixed at `SPEI <= -1.0`. A separate station-shared prediction cutoff was selected on 1985–1999 only, with the regression model and alpha frozen. Persistence was calibrated identically.

| Target | Model cutoff | Retrospective model F1 | Calibrated persistence cutoff | Persistence F1 | Difference |
|---|---:|---:|---:|---:|---:|
| SPEI-3 | -0.40 | 0.619 | -0.80 | 0.617 | +0.001 |
| SPEI-6 | -0.70 | 0.754 | -0.80 | 0.693 | +0.060 |

For SPEI-3, threshold calibration largely removes the apparent event disadvantage, but it does not produce a meaningful advantage over calibrated persistence. For SPEI-6, the retrospective point estimate remains 0.060 higher than calibrated persistence. No confidence interval was computed for this event-F1 difference, so it is diagnostic rather than robust comparative evidence. An alert issued above -1.0 must not be called an observed drought month.

## What can and cannot be claimed

Supported within this recovered three-station dataset:

- one-month-ahead SPEI-3 and SPEI-6 regression forecasts improve on persistence;
- the MAE result is stable across rolling origins, training starts and bootstrap block lengths;
- SPEI-6 has the stronger and more consistent drought-event evidence.

Not supported:

- an untouched or external confirmation;
- superior SPEI-3 drought detection;
- recovery of the published four-region, nine-predictor experiment;
- operational forecasting skill, because input publication latency/revisions and prospective real-time performance were not evaluated;
- end-to-end leakage freedom in target construction, because the recovered SPEI calibration method and reference period are unknown.

The reviewed machine-readable summaries and portable manifest are committed in
this directory. A reproduction run writes the complete dated prediction panels
under the ignored `artifacts/` directory. The manifest binds the source data,
resolved configurations, exact implementation files, software versions, and
SHA-256 digest of that complete rolling prediction panel. The original run
manifest is retained; the separate correction manifest binds the later metric
code and corrected fold summaries.
