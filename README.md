# One-Month-Ahead SPEI Forecasting in Sri Lanka

A reproducible scientific machine-learning benchmark for predicting the next
month's Standardized Precipitation Evapotranspiration Index (SPEI) at three Sri
Lankan locations.

SPEI describes how wet or dry conditions are; lower values mean drier conditions.
SPEI-3 and SPEI-6 summarize three- and six-month periods. Both are predicted one
month ahead in this project.

This is a methodological rebuild of my 2021 undergraduate drought project. I do
not treat the earlier exploratory neural-network results as evidence here.
Instead, this repository asks one narrow question and tests it against simple
baselines with chronological model selection and time-series uncertainty.

> At the end of month *t*, how accurately can observations available through
> month *t* predict the same station's SPEI in month *t + 1*?

## Main result

The primary analysis is a nested rolling-origin evaluation on a common panel of
240 target months (2000-2019) at each of three stations. Models are selected on
earlier five-year blocks, refitted at each outer origin, and then held fixed for
the following five-year block. All station-level comparisons use the same target
dates.

**How to read the table:** MAE is mean absolute error; lower is better.
Persistence predicts the latest observed SPEI value again next month. Ridge is
linear regression with a penalty that limits the size of its coefficients.
Drought F1 balances missed droughts and false alerts; higher is better. Here,
a drought month has SPEI at or below -1.0. Scores are averaged equally across
the three stations.

| Target | Selected model | Model MAE | Persistence MAE | MAE improvement | Paired 95% interval* | Drought F1 (model / persistence) |
|---|---|---:|---:|---:|---:|---:|
| SPEI-3 | Ridge, 24 SPEI lags | 0.542 | 0.638 | 15.1% | [0.054, 0.137] | 0.527 / 0.561 |
| SPEI-6 | Ridge, 24 SPEI + meteorology lags | 0.336 | 0.440 | 23.6% | [0.076, 0.131] | 0.754 / 0.688 |

\*The interval is for the **absolute MAE reduction** relative to persistence.
It uses 5,000 synchronized paired circular block-bootstrap samples with 12-month
blocks kept within each outer fold.

The continuous-value MAE gain is positive in every outer fold and remains
positive when the training record starts in 1901, 1951, or 1971 and when the
bootstrap block is 6, 12, or 24 months. The event result is more limited:
SPEI-3 does not beat persistence on drought F1, while SPEI-6 does on this
retrospective panel. The complete, qualification-aware result is in
[`results/robustness_v1/REPORT.md`](results/robustness_v1/REPORT.md).

These results are internally reproduced, not an independent or operational
validation. The 2000-2019 period has already been inspected, and the recovered
files do not preserve the SPEI fitting distribution or calibration period.

## Scientific workflow

1. Verify every source file against a frozen SHA-256 digest.
2. Validate the allow-listed schema, monthly continuity, missingness pattern,
   and physical ranges without filling or deleting observations.
3. Build lagged features using only observations dated before the target month.
4. Select one shared ridge regularization value across stations using only the
   three inner validation folds preceding each outer fold.
5. Fit a separate model for each station and evaluate identical target months.
6. Compare against persistence and quantify paired uncertainty with block
   resampling that preserves temporal dependence and fold boundaries.

The fixed benchmark also compares zero, training-only monthly climatology,
12-month seasonal persistence, autoregressive ridge, meteorology-only ridge,
combined ridge, and histogram gradient boosting. No random train/test split is
used, and preprocessing is fitted inside each training-only pipeline.

## Reproduce the analysis

The exact tested environment is Python 3.13.0 with NumPy 2.2.6, pandas 2.2.3,
scikit-learn 1.6.1, and pytest 8.4.2. Python 3.11-3.13 is supported.

The three input CSV files are included. No separate data download or GPU is
required. Start with validation and tests before running the full analysis.

```bash
git clone https://github.com/erankawinda/spei-drought-prediction.git
cd spei-drought-prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
```

Validate the inputs and run the tests:

```bash
# macOS (use `sha256sum -c data/SHA256SUMS` on Linux)
shasum -a 256 -c data/SHA256SUMS
spei-forecast validate --config configs/benchmark_v1.toml
pytest
```

Reproduce the fixed-period benchmark and nested rolling analysis:

```bash
spei-forecast benchmark --config configs/benchmark_v1.toml
spei-forecast robustness \
  --config configs/benchmark_v1.toml \
  --robustness-config configs/robustness_v1.toml
```

Generated runs are written under `artifacts/` and intentionally ignored. The
reviewed summary tables and portable manifests are committed under `results/`.

## Repository map

```text
configs/                    Frozen benchmark and robustness settings
data/processed/legacy/      Three recovered, hash-bound source tables
docs/                       Forecast and statistical protocols
results/                    Reviewed reports, summary tables, and manifests
src/spei_forecast/          Validation, features, models, metrics, and runners
tests/                      Integrity, causality, split, bootstrap, and smoke tests
```

## Data provenance and limits

The 2021 project records identify the meteorological source as the Climatic
Research Unit (CRU), University of East Anglia, accessed through CEDA in 2020,
at 0.5-degree monthly resolution from January 1901 to December 2019. That period
and schema match [CRU TS v4.04](https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.04/),
but the recovered exports do not contain a version identifier, grid-cell
coordinates, or retrieval manifest. I therefore describe the version match as
a provenance reconstruction rather than a verified re-download.

The repository contains only Buttala, Padaviya, and Tissamaharama. The Ratnapura
data and IOD/ENSO predictors used in the wider 2021 study were not recovered, so
this repository does not reproduce that four-region experiment. The exact
Hargreaves calculation and SPEI calibration/reference period are also absent.
These fixed historical limitations prevent claims of operational performance or
end-to-end target-construction leakage freedom. See
[`data/README.md`](data/README.md) for the full field-level record.

## Academic context and licence

The original work was completed with Dr Siyath Gunewardene at the University of
Colombo and presented as *A deep learning approach to drought prediction in Sri
Lanka* at [ICMAS 2021](https://science.cmb.ac.lk/icmas2021/conference-program/workshops-special-sessions/international-symposium-on-applied-mathematics-modeling-analysis-and-simulations/).

The analysis code is released under the [MIT License](LICENSE). The bundled
climate-derived tables are not covered by the code licence. The version-specific
CRU page makes CRU TS v4.04 available under the Open Government Licence and asks
users to acknowledge the Climatic Research Unit (University of East Anglia) and
the Met Office. The appropriate dataset reference is Harris et al. (2020),
[*Scientific Data* 7, 109](https://doi.org/10.1038/s41597-020-0453-3).
