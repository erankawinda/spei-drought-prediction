# Reviewed results

This directory contains the compact, reviewed evidence from the fixed benchmark
and its nested rolling-origin robustness analysis. The primary rolling
prediction panel is committed for direct inspection. Larger benchmark and
training-history prediction panels are reproducible from the repository commands
and are left out of Git to keep the review surface small.

- `benchmark_v1/` records the first fixed-period comparison and model selection.
- `robustness_v1/` is the main reported result: common target dates, four nested
  rolling folds, training-history sensitivity, and paired block-bootstrap
  uncertainty.

Both manifests bind the data hashes, configurations, implementation files, and
exact Python package versions used for the reproduction audit.
