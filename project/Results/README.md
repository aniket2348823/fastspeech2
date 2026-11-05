# Results

Produced by `python project/<lang>/inference.py --step 40000` and, for the loss curves, by
`python project/<lang>/train.py --plot-only`. Structure is fixed by the assignment brief:

```
Results/English/
├── training_loss.png            total + per-component, 400 logged points
├── validation_loss.png          total + per-component, 20 logged points
├── inference_summary.json       units, audio length, mean F0 and energy per sentence
├── test_generated_audios/       E1.wav E2a.wav E2b.wav E3.wav E4.wav   (22.05 kHz mono)
└── variance_adaptor_plots/
    ├── duration/  E1.png E2a.png E2b.png E3.png E4.png
    ├── pitch/     E1.png E2a.png E2b.png E3.png E4.png
    ├── energy/    E1.png E2a.png E2b.png E3.png E4.png
    └── E1.csv E2a.csv E2b.csv E3.csv E4.csv        values behind the plots
```

`Results/Telugu/` has the identical layout with `T1 T2a T2b T3 T4`.

Each CSV is one row per model input unit: `unit, roman, duration_frames, duration_sec, pitch_hz,
energy`. `duration_frames` are 11.61 ms frames; `pitch_hz` and `energy` are de-normalised with the
training statistics, so they are in physical units rather than z-scores.
