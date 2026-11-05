# Technical Report

Put `Technical_Report.pdf` here. Required sections (assignment brief §"Technical Report"):

1. Implementation approach — architecture components reused from the reference
   implementation, and what was modified for English and for Telugu.
2. Datasets — selection, subsampling strategy, preprocessing, train/val split,
   training configuration, per language. Quote `project/<lang>/data/split_stats.json`.
3. Loss analysis — training/validation curves, convergence, instability, overfitting.
   Figures: `Results/<lang>/training_loss.png`, `validation_loss.png`.
4. Speech observations — pronunciation and quality for each of E1–E4 and T1–T4.
   Compare against the input text as written (including `antidisestablishmentarianism`,
   `rhyth  m`, `andaroo`, `పట్టు`, `నిశ్శ బ్దంగా`).
5. Variance-adaptor analysis — duration, F0 and energy for every test sentence, with
   explanations. Numbers: `Results/<lang>/variance_adaptor_summary.json`.
6. English vs Telugu — limitations, failures, improvements.
7. Reproducibility — dependencies, config files, exact commands.
