# Fixed benchmark v1

This run used development training through 1984, validation from 1985-1999, and
a fixed 2000-2019 evaluation period. Hyperparameters were selected with
validation macro-MAE across the three stations; the evaluation labels were not
used for selection.

The results were inspected before the nested rolling-origin analysis, so the
period is now a consumed retrospective evaluation rather than an untouched
test. The rolling result reported in the repository README is under
`../robustness_v1/`.

Event F1 summaries were corrected on 6 September 2026 so a model that misses
every event receives zero F1, rather than a missing value. Model selection,
regression scores, bootstrap intervals, and acceptance decisions are unchanged.
The [correction record](../event_f1_correction_v1/README.md) preserves the
original run manifest and documents the updated summaries.

Files:

- `metrics_by_task.csv`: model metrics for each target and station;
- `metrics_macro.csv`: equal-station macro summaries;
- `bootstrap_intervals.csv`: paired 12-month block-bootstrap intervals;
- `acceptance.csv`: the predeclared multi-criterion checks;
- `selection.json`: validation candidates and selected settings;
- `data_quality.json`: schema, missingness, range, and source-hash audit;
- `config_resolved.json`: portable resolved run settings; and
- `manifest.json`: commit, source, code, config, and environment binding.
