# Event F1 correction

6 September 2026

F1 now uses the direct event-count formula `2*TP / (2*TP + FP + FN)`. A model
with no correct event predictions receives zero when it has misses or false
alerts. F1 is undefined only when no events are observed or predicted.

The earlier calculation divided through precision and recall. When these were
both zero or undefined, it produced a missing F1 value. The station average
then omitted that failed station, which could overstate the score.

## What changed

- The SPEI-3 2005–2009 rolling drought F1 changes from 0.243590 to 0.162393;
  its difference from persistence changes from -0.299267 to -0.380464.
- Fifteen station/fold F1 cells change from missing to zero: one drought cell
  and fourteen severe-event cells. The associated fold averages and valid
  station counts are updated.
- Fixed-benchmark F1 summaries for zero, monthly climatology, and seasonal
  persistence are corrected. Acceptance decisions are unchanged.

The headline full-period drought F1, regression scores, model settings, and MAE
bootstrap intervals remain unchanged. Training-history summaries have no
affected missing F1 cells. The alert-cutoff diagnostic already used an explicit
zero fallback, so its selected cutoffs and reported results are unchanged.

## How to trace the correction

The original `benchmark_v1/manifest.json` and `robustness_v1/manifest.json`
remain unchanged and describe the original runs. This directory's
[`manifest.json`](manifest.json) records the later correction: input and output
hashes, the corrected implementation and script hashes, the execution
environment, and all 96 changed summary cells across five CSV files. Its
`git_head` is the pre-correction checkout; the recorded working-tree changes
and file hashes identify the exact code used.

The rolling fold scores were recomputed directly from the committed prediction
panel. The fixed-benchmark panel was regenerated using the already selected
settings because that larger panel was not committed. Every non-F1 summary
column was checked against the original tables before retaining the existing
MAE bootstrap intervals. Model selection and bootstrap were not repeated.

To check the summaries again after installing the project:

```bash
python scripts/recompute_event_metrics.py
```

This writes tables, regenerated benchmark predictions and a new manifest under
`artifacts/event_f1_correction_v1/`. It does not overwrite reviewed results.
Against the corrected tables, no further changed cells should be reported.
The saved correction record here documents the original 96-cell correction.
