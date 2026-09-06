# Reviewed results

This directory contains the compact, reviewed evidence from the fixed benchmark
and its nested rolling-origin robustness analysis. The primary rolling
prediction panel is committed for direct inspection. Larger benchmark and
training-history prediction panels are reproducible from the repository commands
and are left out of Git to keep the review surface small.

- [`benchmark_v1/`](benchmark_v1/README.md) records the first fixed-period comparison and model selection.
- [`robustness_v1/`](robustness_v1/REPORT.md) is the main reported result: common target dates, four nested
  rolling folds, training-history sensitivity, and paired block-bootstrap
  uncertainty.

Start with the [main report](robustness_v1/REPORT.md), then inspect the
[summary metrics](robustness_v1/rolling_metrics_macro.csv) and
[dated predictions](robustness_v1/rolling_predictions.csv).

The original run manifests retain the data hashes, configurations,
implementation files, and Python package versions used for the reproduction
audit. A later [event-F1 correction](event_f1_correction_v1/README.md) records the
updated metric code and summaries separately; it leaves those original
manifests unchanged.
