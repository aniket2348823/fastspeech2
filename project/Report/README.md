# Technical Report

`Technical_Report.pdf` — 15 pages, generated from the shipped artefacts (loss logs, inference
summaries and per-sentence CSVs), so every number in it can be traced back to a file in this
repository.

| Section | Covers |
|---|---|
| 1–2 | objective, reference implementation, architecture components actually instantiated |
| 3–4 | dataset selection, subsampling, splits, preprocessing, per-language modifications |
| 5 | training configuration, loss curves, convergence, overfitting decomposition |
| 6 | generated speech, per test sentence, both languages |
| 7 | variance-adaptor analysis (duration, F0, energy) with the 15 plots per language |
| 8–9 | English vs Telugu comparison, limitations and improvements |
| 10–11 | reproducibility, repository map |
| A | the ten test sentences verbatim, with the normalised strings the front-ends consumed |

`_stats.json` holds every aggregate quoted in the report (loss endpoints, ratios, per-sentence
duration/pitch/energy statistics), produced directly from
`project/<lang>/output/log/` and `project/Results/<lang>/variance_adaptor_plots/*.csv`.
